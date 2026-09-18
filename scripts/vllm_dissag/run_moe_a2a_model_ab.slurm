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
#                                     # per-phase kernel trace; ITL is void
#
# Pickup: ~/.cursor/rules/dsv4-flash-itl-pickup.mdc
# ---------------------------------------------------------------------------
set -uo pipefail

IMAGE="${DOCKER_IMAGE_NAME:-rocm/pytorch-private:vllm-recent-source-basem-v0290-aiter-10f8874-mori07bdace-tk}"
PORT="${PORT:-20005}"
REPO_DIR="$(pwd)"
COOKBOOK=/opt/nixl-vllm-cookbook

EP="${EP:-8}"
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
wait_for_gpus() {
    local budget_mb="${GPU_FREE_WAIT_MB:-10000}" i busiest
    for i in $(seq 1 "${GPU_FREE_WAIT_POLLS:-30}"); do
        busiest="$(on_node "rocm-smi --showmeminfo vram 2>/dev/null \
            | awk '/VRAM Total Used Memory/{print \$NF}' | sort -n | tail -1" \
            2>/dev/null | tail -1)"
        if [ -n "$busiest" ] && [ "$((busiest / 1048576))" -lt "$budget_mb" ]; then
            [ "$i" -gt 1 ] && echo "[gpu-wait] drained after $((i * 10))s"
            return 0
        fi
        [ "$i" = 1 ] && echo "[gpu-wait] busiest GPU holds $((busiest / 1048576)) MB, waiting for < ${budget_mb} MB..."
        sleep 10
    done
    echo "[gpu-wait] WARNING: still $((busiest / 1048576)) MB on the busiest GPU; launching anyway"
    return 0
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
    local a2a model_name model_path
    a2a="$(backend_for "$cell")"       || { echo "unknown backend in cell $cell" >&2; return 1; }
    model_name="$(model_for "$cell")"  || { echo "unknown model in cell $cell"   >&2; return 1; }
    model_path="$(resolve_model "$model_name")"

    local bench_log="${LOG_DIR}/bench_${cell}.log"
    local boot="${LOG_DIR}/boot_${cell}.sh"
    CNAME="moeab-${SLURM_JOB_ID}-${cell}"

    PROFILE_ENV=""
    if [ "$MORI_PROFILE" = 1 ]; then
        mkdir -p "${LOG_DIR}/traces_${cell}" 2>/dev/null || true
        PROFILE_ENV="-e ENABLE_PROFILER=1 -e MORI_PROFILE_DIR=/run_logs/traces_${cell}"
        PROFILE_ENV="$PROFILE_ENV -e MORI_PROFILE_AT=${MORI_PROFILE_AT}"
        PROFILE_ENV="$PROFILE_ENV -e MORI_PROFILE_RANKS=${MORI_PROFILE_RANKS}"
        echo "[main] MoRI profiler ON -> ${LOG_DIR}/traces_${cell} (cold JIT build, ~10 min)"
        echo "[main] NOTE: a profiled cell's ITL is not a measurement -- the dump stalls a step."
    fi

    echo ""
    echo "############################################################"
    echo "## CELL ${cell}  model=${model_name}  a2a=${a2a}  ep=${EP}"
    echo "############################################################"

    if [ ! -d "$model_path" ]; then
        echo "[main] ${cell} model path missing: $model_path"
        echo "${cell} | ${model_name} | ${a2a} | NO_MODEL | - | -" >> "$SUMMARY"
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
        --gpu-memory-utilization 0.80
        --kv-cache-memory-bytes 20000000000
        --max-num-seqs "${MAX_NUM_SEQS:-4}"
        --max-model-len "${MAX_MODEL_LEN:-40960}"
        --max-num-batched-tokens 1024
        --no-enable-prefix-caching
        --distributed-timeout-seconds 7200
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["+quant_fp8"]}'
        --api-server-count=1
        # async scheduling off: it is what deadlocks allgather at EP16, and it
        # is worth 0.3 ms on MoRI (591.89 on vs 592.20 off), so switching it
        # off costs nothing here and removes a variable.
        --no-async-scheduling
    )
    mapfile -t -O "${#args[@]}" args < <(extra_args_for "$cell")
    write_serve_script "$boot" "${args[@]}"
    echo "[launch] $(tail -1 "$boot" | cut -c1-240)"

    wait_for_gpus
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
    on_node "docker logs $CNAME" > "${LOG_DIR}/serve_${cell}.log" 2>&1 || true

    if [ "$ready" != 1 ]; then
        echo "[main] ${cell} NEVER READY -- last 40 lines:"
        tail -40 "${LOG_DIR}/serve_${cell}.log"
        echo "${cell} | ${model_name} | ${a2a} | NOT_READY | - | -" >> "$SUMMARY"
        on_node "docker rm -f $CNAME >/dev/null 2>&1 || true"; CNAME=""
        sleep 25
        return 0
    fi

    # Which MoRI kernels actually ran is the whole point of the cross-model
    # comparison, so record it rather than assuming fp8 from the model name.
    echo "[main] MoRI kernels compiled for ${cell}:"
    grep -o "EpCombineIntraNodeKernel_[a-z0-9_]*\|EpDispatchIntraNodeKernel_[a-z0-9_]*" \
        "${LOG_DIR}/serve_${cell}.log" | sort -u | head || true
    grep -m1 "Using MoriPrepareAndFinalize" "${LOG_DIR}/serve_${cell}.log" || true
    sleep 5

    {
        echo "===== ${cell} model=${model_name} a2a=${a2a} ep=${EP} isl=${ISL} osl=${OSL} ====="
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
    echo "${cell} | ${model_name} | ${a2a} | ${status} | ITL=${itl} ms | TTFT=${ttft} ms" >> "$SUMMARY"

    on_node "docker rm -f $CNAME >/dev/null 2>&1 || true"; CNAME=""
    sleep 25
}

if ! preflight; then
    echo "DIRTY_NODE ${NODE0}" >> "$SUMMARY"
    cat "$SUMMARY"
    exit 2
fi

[ "$INSPECT" = 1 ] && inspect_image

for cell in $CELLS; do
    run_cell "$cell"
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

if [ "$HOLD" = 1 ]; then
    echo "[hold] HOLD=1 -- idling until scancel. Re-serve by hand from ${LOG_DIR}/boot_*.sh."
    while true; do sleep 300; done
fi
