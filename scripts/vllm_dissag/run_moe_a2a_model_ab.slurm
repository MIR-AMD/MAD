#!/bin/bash
#SBATCH -A amd-rccl-guest
#SBATCH -p amd-arad-burst
#SBATCH -q low
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --job-name=moe-a2a-model-ab
#SBATCH --exclusive
#SBATCH --output=/shared_inference/%u/model_blog_logs/slurm-%j.out
#SBATCH --error=/shared_inference/%u/model_blog_logs/slurm-%j.err
# ---------------------------------------------------------------------------
# Is the MoRI decode regression MoRI-wide, or DSV4/FP4-only?
#
# One node, 8 GPUs, `-tp 1` + DP8 + EP8. No PD, no proxy, no router, no
# connector, no patchers except the MoRI combine fix (which is required for
# coherent output at wide EP, not optional).
#
# Measured so far, all `mori_low_latency` vs `allgather_reducescatter` at
# con=1 32/32 with async scheduling off (17-18 Sep):
#
#     EP2  1 node   MoRI 103.04 ms
#     EP4  1 node   MoRI 134.28 ms
#     EP8  1 node   MoRI 306.49 ms   allgather  22.9 ms (async on)
#     EP8  2 nodes  MoRI GPU-faults  allgather 41.86 ms
#     EP16 2 nodes  MoRI 592.20 ms   allgather 51.71 ms
#
# From EP4 up MoRI costs 36-43 ms per added expert-parallel rank; allgather
# costs 1.2 ms. At con=1 decode the payload is single-digit KB, so 306 ms is
# ~4 orders of magnitude above the transfer and the source has no sequential
# per-peer loop -- every peer loop in dispatch/combine is lane-strided.
#
# The one thing never tested is another model, and the shared filesystem has
# the matched pair this needs. Read out of the safetensors headers, not the
# config (DSV4's config.json says plain `fp8` and never mentions fp4):
#
#   DeepSeek-V4-Flash-FP8  experts w1.weight  I8      [2048,2048]
#                          experts w1.scale   F8_E8M0 [2048, 128]   -> MXFP4
#   GLM-5-MXFP4            experts gate_proj  U8      [2048,3072]
#                          weight_scale       U8      [2048, 192]   -> MXFP4
#   GLM-5-FP8              experts gate_proj  F8_E4M3 [2048,6144]
#                          weight_scale_inv   F32     [16,48]       -> block-FP8
#
# Two-per-byte packing with E8M0 scales at block 32 is OCP MXFP4 in both fp4
# rows. So DSV4-Flash and GLM-5-MXFP4 are in the SAME quantization class, and
# GLM-5-MXFP4 vs GLM-5-FP8 is one model family -- `glm_moe_dsa`, 78 layers,
# 256 experts, index_topk 2048, identical shapes -- varying ONLY the MoE
# quantization, which is exactly what picks the MoRI kernel
# (`EpCombineIntraNodeKernel_fp4_*` vs the fp8 variants).
#
#   mxfp4 slow + fp8 fast  -> the fp4 dispatch/combine path owns it. Narrow,
#                             reproducible on a non-DeepSeek model, and the
#                             MoRI bug report writes itself.
#   both slow              -> MoRI+DP is broken for every MoE model and the
#                             DSV4 framing was a red herring. Bigger blast
#                             radius, and the report says so.
#   both fast              -> not the quantization. Back to DSV4 specifics with
#                             the indexer cleared (see below).
#
# Either GLM cell being fast also kills the "lightning indexer x MoRI
# contention" idea, because `glm_moe_dsa` HAS a DSA indexer -- at index_topk
# 2048, four times Flash's 512.
#
# INSPECT=1 (default) additionally dumps, once, from inside the image:
# MoRI's standalone EP dispatch/combine benchmark (if it ships one) and the
# profiler readout surface. Both are needed for the next round and neither is
# worth its own allocation.
#
#   cd scripts/vllm_dissag
#   sbatch run_moe_a2a_model_ab.slurm
#   CELLS="glm5mxfp4_mori glm5mxfp4_allgather" sbatch run_moe_a2a_model_ab.slurm
#   CELLS="dsv4fls_mori" sbatch run_moe_a2a_model_ab.slurm       # re-measure the reference
#   EP=4 sbatch run_moe_a2a_model_ab.slurm                        # narrower width
#   HOLD=1 sbatch run_moe_a2a_model_ab.slurm                      # keep a live server
#   MORI_PROFILE=1 CELLS=dsv4fls_mori sbatch run_moe_a2a_model_ab.slurm
#                                     # per-phase kernel trace; ITL is void.
#                                     # NOTE 436312: the ENABLE_PROFILER=1
#                                     # build itself GPU-faults at
#                                     # 0x2000001000 on every rank.
#   CELLS=dsv4fls_mori MNBT_LIST="1024 512 256 128" sbatch run_moe_a2a_model_ab.slurm
#                                     # the MoRI buffer-size sweep -- see MNBT_LIST
#   CELLS=dsv4fls_mori MORI_CAP_LIST="0 128 64" sbatch run_moe_a2a_model_ab.slurm
#                                     # cap MoRI's buffer with mnbt HELD at 1024
#   CELLS=dsv4fls_mori TORCH_PROFILE=1 MORI_CAP_LIST="0 64" sbatch run_moe_a2a_model_ab.slurm
#                                     # both Phase 0 experiments in one job
#
# Pickup: ~/.cursor/rules/dsv4-flash-itl-pickup.mdc
# ---------------------------------------------------------------------------
set -uo pipefail

IMAGE="${DOCKER_IMAGE_NAME:-rocm/pytorch-private:vllm-recent-source-basem-v0290-aiter-10f8874-mori07bdace-tk}"
PORT="${PORT:-20005}"
REPO_DIR="$(pwd)"
COOKBOOK=/opt/nixl-vllm-cookbook

EP="${EP:-8}"

# 0.72, not the usual 0.80, and this is load-bearing. Both nodes tried on
# 18 Sep carried leaked VRAM that no running process owned and that rocm-smi
# under-reported: 097-028 stranded 42.7 GiB on GPU 4, 097-039 stranded
# 40.5 GiB on GPU 6 (identical across reads a minute apart, so leaked, not a
# live load). Both leave ~151 GiB free, and 0.80 asks for 153.59 -- just over,
# so exactly one DP rank dies and the other seven hang forever in the barrier
# inside MoRI's EpDispatchCombineOp.__init__. 0.72 asks for 138.2 GiB and
# clears it. Two nodes out of two means excluding nodes does not work.
#
# This does NOT change what is measured: --kv-cache-memory-bytes pins the KV
# cache at 20 GB regardless, so utilization only bounds the pool the cache is
# carved from. The DSV4 reference rows are unaffected by comparison.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.72}"
ISL="${BENCH_ISL:-32}"
OSL="${BENCH_OSL:-32}"
CONS="${BENCHMARK_CON:-1}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-900}"
INSPECT="${INSPECT:-1}"
HOLD="${HOLD:-0}"

# MoRI kernel profiler. Needs BOTH ENABLE_PROFILER=1 (compiles the spans into
# the JIT build) and apply_mori_profiler_dump.py (copies the trace buffer back
# -- vLLM never does). Off by default: the dump stalls one decode step for
# seconds, so a profiled cell's ITL is not a measurement.
MORI_PROFILE="${MORI_PROFILE:-0}"
MORI_PROFILE_AT="${MORI_PROFILE_AT:-1500}"
MORI_PROFILE_RANKS="${MORI_PROFILE_RANKS:-0}"

# mxfp4 first, and each model's own allgather baseline immediately after its
# mori cell: that way the job is interpretable at every prefix. After two
# cells one model has a verdict; after four the 2x2 is complete. The mxfp4
# model is also the smaller load (396 GB vs 705 GB), so the cell carrying the
# most signal is also the one reached soonest.
#
# The DSV4 cells are the known reference and are deliberately NOT in the
# default list -- re-measuring 306 ms costs 20 minutes and buys nothing.
CELLS="${CELLS:-glm5mxfp4_mori glm5mxfp4_allgather glm5fp8_mori glm5fp8_allgather}"

# --max-num-batched-tokens, swept. Every cell in CELLS runs once per value.
#
# This is not a prefill knob here, it is the MoRI buffer knob. The
# [mori-dispatch] line on 436312 showed the op is built with
# max_num_inp_token_per_rank=1024 -- vLLM's MoriAll2AllManager passes
# moe.max_num_tokens straight through, and that tracks mnbt. At con=1 decode
# we hand MoRI exactly ONE token while its buffers are sized for 1024 per
# rank across 8 ranks. If any dispatch/combine phase costs
# O(max_num_inp_token_per_rank) rather than O(actual tokens), the useful work
# is ~1/1024 of what the kernel walks, and the cost grows with world_size --
# which fits BOTH the magnitude (~300 ms for a kilobyte payload) AND the
# ~35 ms per EP rank ladder, neither of which bandwidth or a lane-parallel
# peer loop can explain.
#
# Clean single variable at con=1 with isl=32: the prompt is one chunk at any
# of these values, so decode ITL should not otherwise notice mnbt.
# ITL tracking mnbt ~linearly confirms it; flat ITL kills it outright.
MNBT_LIST="${MNBT_LIST:-1024}"

# ANSWERED by 436340. mnbt 1024/512/256/128 -> ITL 307.03/171.06/82.45/60.69,
# fit ITL = 20.1 + 0.2816*mnbt at R^2 = 0.996. The intercept IS the allgather
# baseline (22.9 measured), and the slope -- 35.2 us per token-slot per rank --
# predicts the EP ladder it was not fitted to: EP8 311 vs 306 measured, EP16
# 602 vs 592. MoRI's cost is buffer capacity x ranks, not tokens moved.
#
# But mnbt moves TWO things, because vLLM feeds one scheduler value into both
# the prefill chunk and MoRI's buffers (see apply_mori_capacity_clamp.py for
# the full chain). MORI_CAP_LIST cuts the link: it overrides ONLY MoRI's
# max_num_inp_token_per_rank and leaves mnbt alone. 0 = off (unpatched
# behaviour), so "0 128 64" carries its own control.
#
# Predicted from the fit, at mnbt 1024: cap 128 -> ~56 ms, cap 64 -> ~38 ms.
# Hitting those with mnbt held at 1024 proves capacity is causal and clears
# prefill chunking; flat ITL means the sweep was really about chunk size.
#
# DIAGNOSTIC. MoRI drops tokens handed to it beyond its capacity, so a cap is
# only valid while every forward stays under it. con=1 with isl=32 means 32
# tokens per rank, so 64 is safe here and would NOT be at a 1024-token prefill
# chunk. The product fix is a per-phase capacity, not a global clamp.
MORI_CAP_LIST="${MORI_CAP_LIST:-0}"

# Torch profiler. This is the attribution that MoRI's own profiler could not
# give us -- 436312 showed ENABLE_PROFILER=1 GPU-faults at 0x2000001000 on
# every rank, but torch/roctracer needs no instrumented MoRI build at all.
#
# The question it answers: at mnbt 1024 the dispatch() output is a view of
# (max_num_tokens_to_recv(), hidden_dim) = 8192 x 4096, and prepare() hands
# that padded buffer straight to rocm_aiter_fused_experts. So the capacity
# cost could be MoRI's comm kernels OR the expert GEMM walking padded rows.
# Both scale as mnbt x world_size, so only per-kernel times separate them:
#   EpDispatchIntraNodeKernel* / EpCombineIntraNodeKernel* dominant -> MoRI
#   moe_sorting / fused-MoE GEMM dominant                           -> vLLM
#
# v0.29 moved this off VLLM_TORCH_PROFILER_DIR onto --profiler-config; the env
# var is gone and setting it does nothing. Profiling runs AFTER the measured
# passes so it cannot contaminate the cell's ITL.
TORCH_PROFILE="${TORCH_PROFILE:-0}"
# Iterations are counted from /start_profile: 1 is the prefill, the rest are
# decode steps. Skip past prefill, then keep the trace small enough to move.
TORCH_PROFILE_DELAY="${TORCH_PROFILE_DELAY:-4}"
TORCH_PROFILE_ITERS="${TORCH_PROFILE_ITERS:-8}"

LOG_DIR="${LOG_DIR:-/shared_inference/${USER}/model_blog_logs/${SLURM_JOB_ID}}"
mkdir -p "$LOG_DIR" 2>/dev/null || true
SUMMARY="${LOG_DIR}/model_ab_summary.log"
JIT_CACHE_BASE="${JIT_CACHE_BASE:-/mnt/m2m_nobackup/${USER}/vllm_jit_cache/v0290-aiter-10f8874-mori07bdace-tk}"

NODE0="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)"

echo "=== MoE all2all model A/B (1 node, TP1 x DP${EP} x EP${EP}) ==="
echo "Job:    $SLURM_JOB_ID"
echo "Node:   $NODE0"
echo "Image:  $IMAGE"
echo "Cells:  $CELLS"
echo "Shape:  isl=$ISL osl=$OSL con=$CONS  async=off"
echo "Logs:   $LOG_DIR"
echo "==========================================================="

# ---------------------------------------------------------------------------
# Cell table. Adding a model means adding one line here and one case below.
# ---------------------------------------------------------------------------
model_for() {
    case "$1" in
        glm5mxfp4_*) echo "GLM-5-MXFP4" ;;         # quark mxfp4 experts
        glm5fp8_*)   echo "GLM-5-FP8" ;;           # block-fp8 experts, same arch
        glm51_*)     echo "GLM-5.1-FP8" ;;
        dsv4fls_*)   echo "DeepSeek-V4-Flash-FP8" ;;
        dsv4pro_*)   echo "DeepSeek-V4-Pro-FP8" ;;
        *) return 2 ;;
    esac
}

backend_for() {
    case "$1" in
        *_mori)      echo mori_low_latency ;;
        *_mori_ht)   echo mori_high_throughput ;;
        *_allgather) echo allgather_reducescatter ;;
        *) return 2 ;;
    esac
}

# DSV4 needs its own tokenizer mode and a 256 page (the CSA/HCA compression
# ratios force a multiple of lcm(m,m')=128). GLM has no such constraint, so it
# takes the vLLM default rather than a number borrowed from another model.
extra_args_for() {
    case "$1" in
        dsv4*) printf '%s\n' --tokenizer-mode deepseek_v4 --block-size 256 ;;
        *) : ;;
    esac
}

MODEL_PATH_BASE=/mnt/m2m_nobackup/models_blog
resolve_model() {
    local name="$1"
    if [[ -d "${MODEL_PATH_BASE}/${name}" ]]; then
        echo "${MODEL_PATH_BASE}/${name}"
    else
        echo "/shared_inference/models_blog/${name}"
    fi
}

on_node() { srun --overlap -N1 -n1 --nodelist="$NODE0" bash -c "$*"; }

CNAME=""
# scancel sends SIGTERM and a bare EXIT trap does not run on it. Without this
# the container outlives the job holding all 8 GPUs, and the next job on the
# node dies in ncclCommInitRank with "unhandled cuda error".
cleanup() {
    [ -n "$CNAME" ] && on_node "docker rm -f $CNAME >/dev/null 2>&1 || true" || true
}
trap cleanup EXIT INT TERM

: > "$SUMMARY"

docker_common() {
cat <<EOF
--device /dev/dri --device /dev/kfd --device /dev/infiniband \
--network host --ipc host --privileged \
--group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
--shm-size ${DOCKER_SHM_SIZE:-256G} \
--ulimit nofile=524288:524288 --ulimit memlock=-1:-1 \
-v /mnt:/mnt -v /shared_inference:/shared_inference \
-v ${LOG_DIR}:/run_logs -v ${REPO_DIR}:${COOKBOOK} \
-v ${JIT_CACHE_BASE}/\$(hostname):/opt/vllm_cache \
-e PYTHONUNBUFFERED=1 \
-e VLLM_ROCM_USE_AITER=1 -e VLLM_ROCM_USE_AITER_MOE=1 \
-e VLLM_ROCM_USE_AITER_MLA=1 -e VLLM_ROCM_USE_AITER_RMSNORM=1 \
-e VLLM_ENGINE_READY_TIMEOUT_S=10800 \
-e VLLM_RINGBUFFER_WARNING_INTERVAL=3600 \
-e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
-e VLLM_LOGGING_LEVEL=INFO ${PROFILE_ENV:-}
EOF
}

# ---------------------------------------------------------------------------
# Node 079 once reported idle with 0% VRAM and a FULL /var/lib, and every
# docker run died in 2 s with exit 125 "no space left on device" (436056).
# ---------------------------------------------------------------------------
preflight() {
    local free_gb
    free_gb="$(on_node "df -BG --output=avail /var/lib 2>/dev/null | tail -1 | tr -dc 0-9" 2>/dev/null | tail -1)"
    if [ -z "$free_gb" ] || [ "$free_gb" -lt "${MIN_DISK_GB:-20}" ]; then
        echo "[preflight] FAIL ${NODE0}: /var/lib has ${free_gb:-?} GB free (< ${MIN_DISK_GB:-20})"
        return 1
    fi
    echo "[preflight] ok ${NODE0}: /var/lib ${free_gb} GB free"
}

# ---------------------------------------------------------------------------
# The previous job's VRAM is not always released by the time slurm hands the
# node over. 436217 lost its first cell to exactly that: DP4 raised
#
#   ValueError: Free memory on device cuda:4 (147.52/191.98 GiB) on startup is
#   less than desired GPU memory utilization (0.8, 153.59 GiB)
#
# because something still held 44 GiB on GPU 4 -- and by the time we looked,
# eleven minutes later, that GPU was down to 6.9 GiB with no container
# running. Purely transient, and fatal: one dead DP rank leaves the other
# seven spinning forever in the torch.distributed.barrier() inside MoRI's
# EpDispatchCombineOp.__init__, so the cell burns its whole readiness window
# instead of failing fast. Wait for the node to drain rather than race it.
# ---------------------------------------------------------------------------
# Ask the HIP runtime, NOT rocm-smi. On the bad 097-028, rocm-smi reported
# GPU 4 at 6.9 GB used while torch.cuda.mem_get_info(4) reported 42.66 GiB
# used / 149.32 GiB free -- and 149.32 is just under the 153.59 GiB that
# gpu-memory-utilization 0.80 asks for, which is what actually killed DP4.
# An rocm-smi-based check waves that node straight through. mem_get_info is
# the same number vLLM's request_memory() tests, so this cannot disagree with
# the thing that fails.
gpu_min_free_gib() {
    on_node "docker run --rm --device /dev/dri --device /dev/kfd \
        --group-add video --security-opt seccomp=unconfined \
        --entrypoint /bin/bash $IMAGE -lc \
        \"python3 -c 'import torch; print(min(torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())) // 2**30)'\"" \
        2>/dev/null | tr -dc 0-9
}

wait_for_gpus() {
    # Derived from GPU_MEM_UTIL so the check and the thing it predicts cannot
    # drift apart: vLLM's request_memory() tests exactly util * total.
    local need i free
    need="${GPU_FREE_NEED_GIB:-$(awk -v u="$GPU_MEM_UTIL" 'BEGIN{printf "%d", u * 191.98 + 4}')}"
    for i in $(seq 1 "${GPU_FREE_WAIT_POLLS:-20}"); do
        free="$(gpu_min_free_gib)"
        if [ -n "$free" ] && [ "$free" -ge "$need" ]; then
            [ "$i" -gt 1 ] && echo "[gpu-wait] drained after ~$((i * 30))s"
            return 0
        fi
        [ "$i" = 1 ] && echo "[gpu-wait] emptiest GPU has ${free:-?} GiB free, need ${need}; waiting..."
        sleep 30
    done
    # Do not launch into it. One short DP rank does not fail fast: the other
    # seven spin forever in the barrier inside MoRI's EpDispatchCombineOp
    # __init__, so the cell eats its whole readiness window and the job walks
    # the same wall on every remaining cell (436217 lost all four this way).
    echo "[gpu-wait] FAIL: emptiest GPU still ${free:-?} GiB free, need ${need}."
    echo "[gpu-wait] This node has leaked VRAM that rocm-smi does not report."
    echo "[gpu-wait] Resubmit with --exclude=${NODE0}"
    return 1
}

# ---------------------------------------------------------------------------
# One-shot image inspection. Answers two questions we could not answer from
# the login node, both blockers for the next round:
#   - does MoRI ship a standalone EP dispatch/combine benchmark? If so it
#     isolates MoRI from vLLM with no model load at all.
#   - how is the kernel profiler read out? ENABLE_PROFILER=1 compiles the
#     instrumentation in (mori/jit/config.py:271) but nothing in the vLLM path
#     copies `debugTimeBuf` back, and analyze_ep_kernel_trace.py wants JSON.
# ---------------------------------------------------------------------------
inspect_image() {
    echo ""
    echo "############### IMAGE INSPECTION ###############"
    on_node "docker run --rm --entrypoint /bin/bash $IMAGE -lc '
M=/usr/local/lib/python3.12/dist-packages/mori
echo \"--- mori package tree (top level) ---\"; ls \$M
echo \"--- anything bench/example shaped ---\"
find \$M -maxdepth 3 \( -iname \"*bench*\" -o -iname \"*example*\" -o -iname \"*test*\" \) | head -30
echo \"--- console scripts ---\"; ls /usr/local/bin | grep -i mori
echo \"--- profiler readout surface ---\"
grep -rn \"debugTimeBuf\|profiler\" \$M/ops/dispatch_combine.py | head
python3 -c \"import mori, mori.ops as o; print([n for n in dir(o) if \\\"rof\\\" in n or \\\"race\\\" in n]); import mori.ops.dispatch_combine as dc; print([n for n in dir(dc.EpDispatchCombineOp) if not n.startswith(chr(95))])\"
echo \"--- trace tool entry points ---\"
sed -n \"1,40p\" \$M/_jit-sources/tools/profiler/aggregate_ep_kernel_trace.py | tail -18
'" 2>&1
    echo "############# END INSPECTION #############"
}

# ---------------------------------------------------------------------------
# The MoRI combine patch is REQUIRED: without it finalize() calls combine()
# with the DISPATCHED topk_ids, which describe other ranks' tokens, so it
# gathers from the wrong ranks and the model loses coherent English. The
# damage grows with EP width. Fail the boot rather than serve garbage.
# ---------------------------------------------------------------------------
write_serve_script() {
    local path="$1"; shift
    {
        echo '#!/bin/bash'
        echo 'set -e'
        echo "python3 ${COOKBOOK}/apply_mori_combine_original_topk_fix.py /usr/local/lib/python3.12/dist-packages/vllm"
        # Applied unconditionally. Without MORI_PROFILE_DIR the trace dump is
        # inert, but its `[mori-dispatch]` line still reports use_fp8_dispatch
        # and the MoRI op config -- the only place the dispatch dtype, and so
        # which MoRI kernel a model really drives, is visible. Order-independent
        # with the combine patcher; both orders were tested against the
        # pristine v0290 file.
        echo "python3 ${COOKBOOK}/apply_mori_profiler_dump.py /usr/local/lib/python3.12/dist-packages/vllm"
        # Applied unconditionally too: inert unless MORI_CAP_TOKENS is set to
        # something lower than what vLLM computed, and it logs which it chose
        # either way. Touches a different file from the other two patchers.
        echo "python3 ${COOKBOOK}/apply_mori_capacity_clamp.py /usr/local/lib/python3.12/dist-packages/vllm"
        echo 'n=$(grep -c _original_topk_ids /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/prepare_finalize/mori.py || true)'
        echo 'echo "[combine-patch] _original_topk_ids occurrences=$n"'
        echo '[ "$n" -ge 3 ] || { echo "[combine-patch] FATAL: patch did not land"; exit 3; }'
        printf 'exec vllm serve'
        printf ' %q' "$@"
        printf '\n'
    } > "$path"
    chmod +x "$path"
}

run_cell() {
    local cell="$1"
    local mnbt="${2:-1024}"
    local cap="${3:-0}"
    local a2a model_name model_path tag
    a2a="$(backend_for "$cell")"       || { echo "unknown backend in cell $cell" >&2; return 1; }
    model_name="$(model_for "$cell")"  || { echo "unknown model in cell $cell"   >&2; return 1; }
    model_path="$(resolve_model "$model_name")"
    # Keep the uncapped tag exactly as it was so 436340's log names still match.
    tag="${cell}_mnbt${mnbt}"
    [ "$cap" != 0 ] && tag="${tag}_cap${cap}"

    local bench_log="${LOG_DIR}/bench_${tag}.log"
    local boot="${LOG_DIR}/boot_${tag}.sh"
    CNAME="moeab-${SLURM_JOB_ID}-${tag}"

    PROFILE_ENV=""
    if [ "$cap" != 0 ]; then
        PROFILE_ENV="-e MORI_CAP_TOKENS=${cap}"
        echo "[main] MoRI capacity capped at ${cap} tokens/rank (mnbt stays ${mnbt})"
    fi
    if [ "$MORI_PROFILE" = 1 ]; then
        mkdir -p "${LOG_DIR}/traces_${tag}" 2>/dev/null || true
        # Append: MORI_CAP_TOKENS may already be in here.
        PROFILE_ENV="$PROFILE_ENV -e ENABLE_PROFILER=1 -e MORI_PROFILE_DIR=/run_logs/traces_${tag}"
        PROFILE_ENV="$PROFILE_ENV -e MORI_PROFILE_AT=${MORI_PROFILE_AT}"
        PROFILE_ENV="$PROFILE_ENV -e MORI_PROFILE_RANKS=${MORI_PROFILE_RANKS}"
        echo "[main] MoRI profiler ON -> ${LOG_DIR}/traces_${tag} (cold JIT build, ~10 min)"
        echo "[main] NOTE: a profiled cell's ITL is not a measurement -- the dump stalls a step."
    fi

    local row="${cell} | ${model_name} | ${a2a} | mnbt=${mnbt} | cap=${cap}"

    echo ""
    echo "############################################################"
    echo "## CELL ${cell}  model=${model_name}  a2a=${a2a}  ep=${EP}  mnbt=${mnbt}  cap=${cap}"
    echo "############################################################"

    if [ ! -d "$model_path" ]; then
        echo "[main] ${cell} model path missing: $model_path"
        echo "${row} | NO_MODEL | - | -" >> "$SUMMARY"
        return 0
    fi
    echo "[main] model path: $model_path"

    local args=(
        "$model_path"
        --trust-remote-code
        --port "$PORT"
        --tensor-parallel-size 1
        --data-parallel-size "$EP"
        --data-parallel-size-local "$EP"
        --enable-expert-parallel
        --all2all-backend "$a2a"
        --kv-cache-dtype fp8_e4m3
        --gpu-memory-utilization "$GPU_MEM_UTIL"
        --kv-cache-memory-bytes 20000000000
        --max-num-seqs "${MAX_NUM_SEQS:-4}"
        --max-model-len "${MAX_MODEL_LEN:-40960}"
        --max-num-batched-tokens "$mnbt"
        --no-enable-prefix-caching
        --distributed-timeout-seconds 7200
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["+quant_fp8"]}'
        --api-server-count=1
        # async scheduling off: it is what deadlocks allgather at EP16, and it
        # is worth 0.3 ms on MoRI (591.89 on vs 592.20 off), so switching it
        # off costs nothing here and removes a variable.
        --no-async-scheduling
    )
    if [ "$TORCH_PROFILE" = 1 ]; then
        mkdir -p "${LOG_DIR}/torchtrace_${tag}" 2>/dev/null || true
        args+=(
            --profiler-config.profiler=torch
            "--profiler-config.torch_profiler_dir=/run_logs/torchtrace_${tag}"
            # Stacks are the bulk of the trace and we only need kernel names
            # and durations, so drop them. ignore_frontend is required by the
            # docs whenever delay/limit are used: the AsyncLLM front end does
            # not count iterations and would trace the entire range.
            --profiler-config.torch_profiler_with_stack=false
            --profiler-config.ignore_frontend=true
            "--profiler-config.delay_iterations=${TORCH_PROFILE_DELAY}"
            "--profiler-config.max_iterations=${TORCH_PROFILE_ITERS}"
        )
    fi
    mapfile -t -O "${#args[@]}" args < <(extra_args_for "$cell")
    write_serve_script "$boot" "${args[@]}"
    echo "[launch] $(tail -1 "$boot" | cut -c1-240)"

    if ! wait_for_gpus; then
        echo "${row} | DIRTY_GPU | - | -" >> "$SUMMARY"
        return 1
    fi
    on_node "docker rm -f $CNAME >/dev/null 2>&1 || true; \
        docker run -d --name $CNAME $(docker_common) \
        --entrypoint /bin/bash $IMAGE /run_logs/$(basename "$boot") >/dev/null"

    # GLM-5.1-FP8 is 705 GB cold off NVMe and MoRI JITs its kernels per shape
    # set (a cold EP2 cell spent 590 s in JIT alone), so allow a long boot.
    echo "[main] waiting for /health on ${NODE0}:${PORT} (up to 45 min)..."
    local ready=0 i
    for i in $(seq 1 270); do
        if on_node "curl -s http://127.0.0.1:${PORT}/health >/dev/null 2>&1"; then
            echo "[main] ${cell} ready after $((i * 10))s"; ready=1; break
        fi
        if ! on_node "docker ps --format '{{.Names}}' | grep -qx $CNAME"; then
            echo "[main] ERROR: ${cell} container exited during startup"
            break
        fi
        # A dead DP rank does NOT exit the container: the surviving ranks spin
        # forever in MoRI's init barrier while the APIServer has already given
        # up, so the liveness check above stays true and the cell burns all 45
        # minutes. Watch for the engine's own verdict instead.
        if [ $((i % 6)) = 0 ] && on_node \
            "docker logs $CNAME 2>&1 | grep -q 'Engine core initialization failed\|EngineCore failed to start'"; then
            echo "[main] ERROR: ${cell} engine core failed to start -- not waiting out the window"
            break
        fi
        sleep 10
    done

    # Redirect on THIS side: /run_logs exists only inside the container, so
    # `docker logs ... > /run_logs/...` in the remote shell silently fails.
    on_node "docker logs $CNAME" > "${LOG_DIR}/serve_${tag}.log" 2>&1 || true

    if [ "$ready" != 1 ]; then
        echo "[main] ${cell} NEVER READY -- last 40 lines:"
        tail -40 "${LOG_DIR}/serve_${tag}.log"
        echo "${row} | NOT_READY | - | -" >> "$SUMMARY"
        on_node "docker rm -f $CNAME >/dev/null 2>&1 || true"; CNAME=""
        sleep 25
        return 0
    fi

    # Which MoRI kernels actually ran is the whole point of the cross-model
    # comparison, so record it rather than assuming fp8 from the model name.
    echo "[main] MoRI kernels compiled for ${cell}:"
    grep -o "EpCombineIntraNodeKernel_[a-z0-9_]*\|EpDispatchIntraNodeKernel_[a-z0-9_]*" \
        "${LOG_DIR}/serve_${tag}.log" | sort -u | head || true
    grep -m1 "Using MoriPrepareAndFinalize" "${LOG_DIR}/serve_${tag}.log" || true
    # The capacity the op was BUILT with, not the one we asked for. Both the
    # clamp's own line and the op config are printed, so a cap that silently
    # failed to take cannot be mistaken for one that did.
    grep -m1 "\[mori-cap\]" "${LOG_DIR}/serve_${tag}.log" || true
    grep -o "max_num_inp_token_per_rank=[0-9]*" "${LOG_DIR}/serve_${tag}.log" | sort -u || true
    if [ "$cap" != 0 ] && ! grep -q "max_num_inp_token_per_rank=${cap}\b" "${LOG_DIR}/serve_${tag}.log"; then
        echo "[main] WARNING: cap=${cap} requested but the op config does not show it."
    fi
    sleep 5

    {
        echo "===== ${cell} model=${model_name} a2a=${a2a} ep=${EP} mnbt=${mnbt} isl=${ISL} osl=${OSL} ====="
        for con in warmup $CONS; do
            local c=$con n=16
            [ "$con" = warmup ] && c=1
            [ "$c" -gt 8 ] && n=$((c * 4))
            echo "[RUNNING] ${cell} tag=${con} con=${c} prompts=${n}"
            # -v /mnt is required: without it the client cannot see the model
            # dir, transformers treats the path as a HF repo id and dies on
            # `Repo id must be in the form 'repo_name'...` (435757).
            on_node "timeout ${BENCH_TIMEOUT} docker run --rm --network host \
                -v /mnt:/mnt -v /shared_inference:/shared_inference \
                -v ${LOG_DIR}:/run_logs \
                $IMAGE \
                vllm bench serve \
                    --model $model_path --backend vllm \
                    --host 127.0.0.1 --port $PORT \
                    --dataset-name random \
                    --random-input-len $ISL --random-output-len $OSL \
                    --random-prefix-len 0 \
                    --num-prompts $n --request-rate inf --ignore-eos \
                    --percentile-metrics ttft,tpot,itl \
                    --max-concurrency $c"
            local rc=$?
            if [ "$rc" -ne 0 ]; then
                echo "[STALL] ${cell} tag=${con} con=${c} rc=${rc} (124 = hit ${BENCH_TIMEOUT}s)"
                [ "$con" = warmup ] && { echo "[STALL] ${cell} warmup failed -- skipping remaining passes"; break; }
            fi
        done
    } 2>&1 | tee "$bench_log"

    # A cell whose server came up but whose bench produced nothing is NOT OK,
    # and a stalled pass that printed a partial report is not OK either
    # (436083 recorded `OK | ITL=52.08 ms` off a warmup that timed out).
    local itl ttft status=OK
    itl=$(awk '/Median ITL/{v=$4} END{print (v==""?"-":v)}' "$bench_log")
    ttft=$(awk '/Median TTFT/{v=$4} END{print (v==""?"-":v)}' "$bench_log")
    [ "$itl" = "-" ] && status=NO_RESULT
    grep -q '^\[STALL\]' "$bench_log" && status=STALLED
    echo "${row} | ${status} | ITL=${itl} ms | TTFT=${ttft} ms" >> "$SUMMARY"

    # AFTER the measured passes, so the trace cannot move the number above.
    if [ "$TORCH_PROFILE" = 1 ] && [ "$status" != NO_RESULT ]; then
        profile_pass "$tag" "$model_path"
    fi

    on_node "docker rm -f $CNAME >/dev/null 2>&1 || true"; CNAME=""
    sleep 25
}

# ---------------------------------------------------------------------------
# One profiled request, bracketed by /start_profile and /stop_profile.
#
# The engine counts iterations from the start call, so delay_iterations skips
# the prefill and max_iterations bounds the trace. Output is one gzipped
# Chrome trace per DP rank under torchtrace_<tag>/.
#
# Read it with: python3 -c on torch.profiler's own parser, or drop the json
# into Perfetto. What matters is the top kernels by total CUDA time -- see
# TORCH_PROFILE above for how that assigns blame.
# ---------------------------------------------------------------------------
profile_pass() {
    local tag="$1" model_path="$2"
    local tdir="${LOG_DIR}/torchtrace_${tag}"
    echo "[profile] starting torch profiler (delay=${TORCH_PROFILE_DELAY} max=${TORCH_PROFILE_ITERS})"
    if ! on_node "curl -sf -X POST http://127.0.0.1:${PORT}/start_profile >/dev/null"; then
        echo "[profile] /start_profile failed -- skipping (is --profiler-config set?)"
        return 0
    fi
    # One request, long enough that delay+max iterations all land in decode.
    on_node "timeout 600 docker run --rm --network host \
        -v /mnt:/mnt -v /shared_inference:/shared_inference \
        -v ${LOG_DIR}:/run_logs \
        $IMAGE \
        vllm bench serve \
            --model $model_path --backend vllm \
            --host 127.0.0.1 --port $PORT \
            --dataset-name random \
            --random-input-len $ISL --random-output-len 32 \
            --random-prefix-len 0 \
            --num-prompts 1 --request-rate inf --ignore-eos \
            --max-concurrency 1" > "${LOG_DIR}/profile_${tag}.log" 2>&1 || true
    # Blocking: the workers flush their traces on stop, which takes a while.
    on_node "curl -sf -X POST --max-time 900 http://127.0.0.1:${PORT}/stop_profile >/dev/null" \
        || echo "[profile] /stop_profile returned non-zero (trace may still be complete)"
    sleep 20
    echo "[profile] traces in ${tdir}:"
    ls -la "$tdir" 2>/dev/null | tail -12 || echo "[profile] NO TRACES WRITTEN"
}

if ! preflight; then
    echo "DIRTY_NODE ${NODE0}" >> "$SUMMARY"
    cat "$SUMMARY"
    exit 2
fi

[ "$INSPECT" = 1 ] && inspect_image

for cell in $CELLS; do
    for mnbt in $MNBT_LIST; do
        for cap in $MORI_CAP_LIST; do
            # A node with leaked VRAM will fail every remaining cell
            # identically, so give the allocation back instead of walking the
            # same wall once per cell per mnbt per cap.
            run_cell "$cell" "$mnbt" "$cap" \
                || { echo "[main] unusable node -- abandoning remaining cells"; break 3; }
        done
    done
done

echo ""
echo "=================== SUMMARY ==================="
cat "$SUMMARY"
echo "==============================================="
echo ""
echo "Read (compare each mori cell against ITS OWN allgather cell, not across models):"
echo "  mxfp4 slow, fp8 fast -> the MoRI fp4 dispatch/combine path owns it."
echo "  both slow            -> MoRI+DP broken for every MoE model."
echo "  both fast            -> not the quantization; and the DSA indexer is cleared."
echo "Reference (DSV4-Flash MXFP4, EP8, 1 node, async off): mori 306.49 ms, allgather 22.9 ms."
echo ""
echo "Capacity cells (cap=N, mnbt held): fit ITL = 20.1 + 0.2816*capacity predicts"
echo "  cap=1024 -> ~307 ms   cap=128 -> ~56 ms   cap=64 -> ~38 ms"
echo "  Those landing -> capacity is causal, prefill chunk size is exonerated."
echo "  Flat vs cap=0 -> the mnbt sweep was chunk size after all; re-open it."

if [ "$HOLD" = 1 ]; then
    echo "[hold] HOLD=1 -- idling until scancel. Re-serve by hand from ${LOG_DIR}/boot_*.sh."
    while true; do sleep 300; done
fi
