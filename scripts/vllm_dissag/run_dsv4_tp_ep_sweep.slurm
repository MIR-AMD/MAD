#!/bin/bash
#SBATCH -A amd-rccl-guest
#SBATCH -p amd-arad-burst
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --job-name=dsv4fls-tp-ep-sweep
#SBATCH --exclusive
#SBATCH --output=/shared_inference/%u/model_blog_logs/slurm-%j.out
#SBATCH --error=/shared_inference/%u/model_blog_logs/slurm-%j.err
# ---------------------------------------------------------------------------
# Decode-ITL TP x EP sweep. One node, 8 GPUs, no PD, no proxy, no router.
#
# Question: TP8 decodes at ~18 ms/token; `-tp 1` + DP8 + EP decodes at ~300 ms.
# The step timer put 307 of those 300+ ms in `other` (MLA 2.6, dispatch 1.2,
# combine 1.5). The lightning indexer is ReplicatedLinear in BOTH topologies, so
# it cannot own the gap. What TP8 shards and `-tp 1` does not is the dense stack:
# q/kv/o projections, attention heads, shared expert. This sweep holds EP width
# at 8 and walks TP 8 -> 4 -> 2 -> 1, so any monotone ITL climb is attributable
# to TP width alone, with the all2all present in every EP cell.
#
# Two extra `-tp 1` cells probe token-dimension padding in the low-latency EP
# path (a grouped expert GEMM sized by max_num_batched_tokens rather than by the
# 1 real decode token would explain 7 ms/layer).
#
# After the last cell the job does NOT exit: it re-serves HOLD_CELL and idles so
# follow-up tests can run against a live server without another queue wait. The
# node is released by `scancel`, which still tears the container down via the
# TERM trap.
#
#   cd scripts/vllm_dissag
#   sbatch run_dsv4_tp_ep_sweep.slurm
#   CELLS="tp1dp8ep tp8" sbatch run_dsv4_tp_ep_sweep.slurm         # subset
#   HOLD_CELL=tp2dp4ep sbatch run_dsv4_tp_ep_sweep.slurm           # hold another cell
#   CELLS=tp1dp8ep HOLD_CELL=tp1dp8ep sbatch run_dsv4_tp_ep_sweep.slurm  # one cell + hold
#   HOLD=0 sbatch run_dsv4_tp_ep_sweep.slurm                       # old behaviour, tear down
# ---------------------------------------------------------------------------
set -uo pipefail

MODEL="${MODEL:-flash}"
case "$MODEL" in
    flash|dsv4fls) MODEL_NAME="DeepSeek-V4-Flash-FP8" ;;
    pro|dsv4pro)   MODEL_NAME="DeepSeek-V4-Pro-FP8" ;;
    *) echo "MODEL must be flash or pro (got $MODEL)" >&2; exit 2 ;;
esac

if [[ -d "/mnt/m2m_nobackup/models_blog/${MODEL_NAME}" ]]; then
    MODEL_PATH="/mnt/m2m_nobackup/models_blog/${MODEL_NAME}"
else
    MODEL_PATH="/shared_inference/models_blog/${MODEL_NAME}"
fi

IMAGE="${DOCKER_IMAGE_NAME:-rocm/pytorch-private:vllm-recent-source-basem-v0290-aiter-10f8874-mori07bdace-tk}"
PORT="${PORT:-20005}"
REPO_DIR="$(pwd)"
COOKBOOK=/opt/nixl-vllm-cookbook

ISL="${BENCH_ISL:-32}"
OSL="${BENCH_OSL:-32}"
CONS="${BENCHMARK_CON:-1}"

# TP8 first: cheapest cell, and it validates the harness against the known 18 ms
# before any EP cell is trusted. Then TP walks down at fixed EP width 8.
# Both anchors (TP8 ~18 ms, tp1+DP8+EP ~300 ms) run first so a job that dies
# early still yields the comparison the rest of the ladder is read against.
CELLS="${CELLS:-tp8 tp1dp8ep tp4dp2ep tp2dp4ep tp1dp8ep_mnbt64 tp1dp8ep_dpchunk tp1dp8ep_a2athru}"

LOG_DIR="${LOG_DIR:-/shared_inference/${USER}/model_blog_logs/${SLURM_JOB_ID}}"
mkdir -p "$LOG_DIR" 2>/dev/null || true
SUMMARY="${LOG_DIR}/tp_ep_sweep_summary.log"

# AITER/Triton JIT cache survives between cells so only the first boot per shape
# pays compile time. The image already points AITER_JIT_DIR / TRITON_CACHE_DIR /
# VLLM_CACHE_ROOT / COMGR_CACHE_DIR at /opt/vllm_cache.
JIT_CACHE_HOST="${JIT_CACHE_HOST:-/mnt/m2m_nobackup/${USER}/vllm_jit_cache/v0290-aiter-10f8874-mori07bdace-tk}"
mkdir -p "$JIT_CACHE_HOST"/{aiter_jit,triton,vllm,comgr} 2>/dev/null || true

NODE_IP="$(hostname -I | awk '{print $1}')"
if [ -z "$NODE_IP" ]; then
    echo "FATAL: could not resolve a node IP; --data-parallel-address would be empty" >&2
    exit 2
fi

echo "=== DSV4 decode-ITL TP x EP sweep (single node, no PD / proxy / router) ==="
echo "Job:     $SLURM_JOB_ID"
echo "Node:    $(hostname)  ip=$NODE_IP"
echo "Model:   $MODEL_NAME"
echo "Path:    $MODEL_PATH"
echo "Image:   $IMAGE"
echo "Cells:   $CELLS"
echo "Shape:   isl=$ISL osl=$OSL con=$CONS"
echo "JIT:     $JIT_CACHE_HOST"
echo "=========================================================================="

RDMA_MOUNTS=()
LIBDIR=/usr/lib/x86_64-linux-gnu
for f in "$LIBDIR"/libibverbs.so "$LIBDIR"/libibverbs.so.1* "$LIBDIR"/librdmacm.so \
         "$LIBDIR"/librdmacm.so.1* "$LIBDIR"/libmlx5.so*; do
    [ -f "$f" ] && RDMA_MOUNTS+=(-v "$f:$f:ro")
done
[ -d "$LIBDIR/libibverbs" ] && RDMA_MOUNTS+=(-v "$LIBDIR/libibverbs:$LIBDIR/libibverbs:ro")
[ -d /etc/libibverbs.d ] && RDMA_MOUNTS+=(-v /etc/libibverbs.d:/etc/libibverbs.d:ro)

DOCKER_COMMON=(
    --device /dev/dri --device /dev/kfd --device /dev/infiniband
    --network host --ipc host --privileged
    --group-add video --cap-add SYS_PTRACE --security-opt seccomp=unconfined
    --shm-size "${DOCKER_SHM_SIZE:-256G}"
    --ulimit nofile=524288:524288 --ulimit memlock=-1:-1
    -v /mnt:/mnt
    -v /shared_inference:/shared_inference
    -v "${LOG_DIR}:/run_logs"
    -v "${REPO_DIR}:${COOKBOOK}"
    -v "${JIT_CACHE_HOST}:/opt/vllm_cache"
    "${RDMA_MOUNTS[@]}"
    -e PYTHONUNBUFFERED=1
    -e VLLM_USE_V1=1
    -e VLLM_ROCM_USE_AITER=1
    -e VLLM_ROCM_USE_AITER_MOE=1
    -e VLLM_ROCM_USE_AITER_MLA=1
    -e VLLM_ROCM_USE_AITER_PAGED_ATTN=0
    -e VLLM_ROCM_USE_AITER_RMSNORM=1
    -e VLLM_USE_AITER_TRITON_SILU_MUL=0
    -e VLLM_ENGINE_READY_TIMEOUT_S=10800
    -e VLLM_RINGBUFFER_WARNING_INTERVAL=3600
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600
    -e VLLM_LOGGING_LEVEL=INFO
)

SERVE_NAME=""
HOLD_NAME=""
# scancel sends SIGTERM, and a bare EXIT trap does not run on it. Without this,
# dockerd keeps the container alive holding all 8 GPUs and the next job on the
# node dies in ncclCommInitRank with "unhandled cuda error" (224963/224964).
cleanup() {
    [ -n "$SERVE_NAME" ] && docker rm -f "$SERVE_NAME" >/dev/null 2>&1 || true
    [ -n "$HOLD_NAME" ] && docker rm -f "$HOLD_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

: > "$SUMMARY"

# ---------------------------------------------------------------------------
# cell_args <cell>  ->  prints serve args, one per line; extra `-e` env on fd 3
# ---------------------------------------------------------------------------
serve_args_for() {
    local cell="$1" tp dp mnbt=1024 seqs=4 a2a=mori_low_latency
    CELL_ENV=()
    case "$cell" in
        tp8)                 tp=8; dp=0 ;;
        tp4dp2ep)            tp=4; dp=2 ;;
        tp2dp4ep)            tp=2; dp=4 ;;
        tp1dp8ep)            tp=1; dp=8 ;;
        tp1dp8ep_mnbt64)     tp=1; dp=8; mnbt=64; seqs=1 ;;
        tp1dp8ep_dpchunk)    tp=1; dp=8; CELL_ENV=(-e VLLM_MOE_DP_CHUNK_SIZE=32) ;;
        tp1dp8ep_a2athru)    tp=1; dp=8; a2a=mori ;;
        *) echo "unknown cell $cell" >&2; return 2 ;;
    esac

    CELL_DESC="tp=${tp} dp=${dp:-0} ep=$([ "$dp" -gt 0 ] && echo $((tp * dp)) || echo off) a2a=$([ "$dp" -gt 0 ] && echo "$a2a" || echo none) mnbt=${mnbt} seqs=${seqs}"

    CELL_ARGS=(
        "${MODEL_PATH}"
        --trust-remote-code
        --tokenizer-mode deepseek_v4
        --port "${PORT}"
        --tensor-parallel-size "$tp"
        --block-size 256
        --kv-cache-dtype fp8_e4m3
        --gpu-memory-utilization 0.80
        --kv-cache-memory-bytes 20000000000
        --max-num-seqs "$seqs"
        --max-model-len 40960
        --max-num-batched-tokens "$mnbt"
        --no-enable-prefix-caching
        --distributed-timeout-seconds 7200
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["+quant_fp8"]}'
    )
    if [ "$dp" -gt 0 ]; then
        CELL_ARGS+=(
            --data-parallel-size "$dp"
            --data-parallel-size-local "$dp"
            --data-parallel-address "$NODE_IP"
            --data-parallel-rpc-port 13345
            --enable-expert-parallel
            --all2all-backend "$a2a"
            "--api-server-count=${dp}"
        )
    fi
}

run_cell() {
    local cell="$1"
    local serve_log="${LOG_DIR}/serve_${cell}.log"
    local bench_log="${LOG_DIR}/bench_${cell}.log"

    serve_args_for "$cell" || return 1
    SERVE_NAME="dsv4sweep-${SLURM_JOB_ID}-${cell}"

    echo ""
    echo "############################################################"
    echo "## CELL ${cell}  (${CELL_DESC})"
    echo "############################################################"
    echo "[serve] vllm serve ${CELL_ARGS[*]}"

    # The mori combine fix (Wei Sun / mori#475) is mandatory for any EP cell:
    # unpatched, finalize() gets dispatch()'s post-all2all indices and EP8 dies in
    # profile_run. Applied for every cell so the source tree is identical across
    # the sweep, and asserted so a silent miss cannot be read as a topology result.
    docker rm -f "$SERVE_NAME" >/dev/null 2>&1 || true
    docker run -d --name "$SERVE_NAME" "${DOCKER_COMMON[@]}" "${CELL_ENV[@]}" \
        --entrypoint /bin/bash "$IMAGE" -c "
set -e
python3 ${COOKBOOK}/apply_mori_combine_original_topk_fix.py \
    /usr/local/lib/python3.12/dist-packages/vllm
n=\$(grep -c _original_topk_ids \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/prepare_finalize/mori.py || true)
echo \"[combine-patch] _original_topk_ids occurrences=\$n\"
[ \"\$n\" -ge 3 ] || { echo '[combine-patch] FATAL: patch did not land'; exit 3; }
exec vllm serve $(printf '%q ' "${CELL_ARGS[@]}")
" > /dev/null

    echo "[main] waiting for /health (up to 60 min)..."
    local ready=0 i
    for i in $(seq 1 360); do
        if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo "[main] ${cell} ready after $((i * 10))s"
            ready=1
            break
        fi
        if ! docker ps --format '{{.Names}}' | grep -qx "$SERVE_NAME"; then
            echo "[main] ERROR: ${cell} serve container exited during startup"
            break
        fi
        sleep 10
    done
    docker logs "$SERVE_NAME" > "$serve_log" 2>&1 || true

    if [ "$ready" != 1 ]; then
        echo "[main] ${cell} NEVER READY -- last 60 lines:"
        tail -60 "$serve_log"
        echo "${cell} | ${CELL_DESC} | NOT_READY | - | -" >> "$SUMMARY"
        docker rm -f "$SERVE_NAME" >/dev/null 2>&1 || true
        SERVE_NAME=""
        return 0
    fi
    sleep 5

    {
        echo "===== ${cell} (${CELL_DESC}) isl=${ISL} osl=${OSL} ====="
        for con in warmup $CONS; do
            local c=$con n=16
            [ "$con" = warmup ] && c=1
            [ "$c" -gt 8 ] && n=$((c * 2))
            echo "[RUNNING] ${cell} tag=${con} con=${c} prompts=${n}"
            timeout 1800 docker run --rm --network host -e VLLM_USE_V1=1 "$IMAGE" \
                vllm bench serve \
                    --model "$MODEL_PATH" --backend vllm \
                    --host 127.0.0.1 --port "$PORT" \
                    --dataset-name random \
                    --random-input-len "$ISL" --random-output-len "$OSL" \
                    --random-prefix-len 0 \
                    --num-prompts "$n" --request-rate inf --ignore-eos \
                    --percentile-metrics ttft,tpot,itl \
                    --max-concurrency "$c"
            [ $? -eq 124 ] && echo "[STALL] ${cell} con=${c} timed out"
        done
    } 2>&1 | tee "$bench_log"

    # Skip the warmup block when harvesting; take the last Median ITL/TTFT seen.
    local itl ttft
    itl=$(awk '/Median ITL/{v=$4} END{print (v==""?"-":v)}' "$bench_log")
    ttft=$(awk '/Median TTFT/{v=$4} END{print (v==""?"-":v)}' "$bench_log")
    echo "${cell} | ${CELL_DESC} | OK | ITL=${itl} ms | TTFT=${ttft} ms" >> "$SUMMARY"
    echo "[harvest] ${cell} median ITL=${itl} ms TTFT=${ttft} ms"

    # Free evidence for the padding hypothesis: if the grouped expert GEMM is
    # sized by max_num_batched_tokens rather than the live token count, AITER
    # reports the padded M here on every step, not M=1.
    echo "--- ${cell} aiter GEMM shapes seen ---"
    grep -oE 'shape is M:[0-9]+, *N:[0-9]+, *K:[0-9]+' "$serve_log" 2>/dev/null \
        | sort | uniq -c | sort -rn | head -12 || echo "(none logged)"

    # Refresh the log after the bench so decode-time messages are captured too.
    docker logs "$SERVE_NAME" > "$serve_log" 2>&1 || true
    docker rm -f "$SERVE_NAME" >/dev/null 2>&1 || true
    SERVE_NAME=""
    sleep 20
    return 0
}

for cell in $CELLS; do
    run_cell "$cell"
done

echo ""
echo "================= SWEEP SUMMARY ================="
cat "$SUMMARY"
echo "================================================="
echo "Logs: ${LOG_DIR}"

# ---------------------------------------------------------------------------
# Hold a live server for interactive follow-ups, then idle until scancel or
# walltime. `vllm serve` runs in the BACKGROUND inside the container with
# `sleep infinity` in the foreground, so the container outlives the server:
# you can kill the server and relaunch it with different env/patchers without
# losing the allocation.
# ---------------------------------------------------------------------------
if [ "${HOLD:-1}" != 1 ]; then
    echo "[hold] HOLD=0 -- exiting and tearing down."
    exit 0
fi

HOLD_CELL="${HOLD_CELL:-tp1dp8ep}"
if ! serve_args_for "$HOLD_CELL"; then
    echo "[hold] bad HOLD_CELL=${HOLD_CELL}; nothing to hold." >&2
    exit 1
fi
HOLD_NAME="dsv4hold-${SLURM_JOB_ID}"
HOLD_LOG="${LOG_DIR}/serve_hold_${HOLD_CELL}.log"
HOLD_CMD="${LOG_DIR}/serve_cmd_hold.sh"

# Written out so a relaunch inside the held container is a copy-paste, not a
# retype of 20 flags from the log.
{
    echo "#!/bin/bash"
    echo "# ${HOLD_CELL}: ${CELL_DESC}"
    echo "# Run INSIDE ${HOLD_NAME}. Export any extra env first, e.g."
    echo "#   export DSV4_DECODE_TIMER=1 DSV4_DECODE_TIMER_EVERY=1 DSV4_EAGER=1"
    echo "exec vllm serve $(printf '%q ' "${CELL_ARGS[@]}")"
} > "$HOLD_CMD"
chmod +x "$HOLD_CMD" 2>/dev/null || true

echo ""
echo "############################################################"
echo "## HOLD ${HOLD_CELL}  (${CELL_DESC})"
echo "############################################################"

docker rm -f "$HOLD_NAME" >/dev/null 2>&1 || true
mkdir -p "${LOG_DIR}/prof" 2>/dev/null || true
# VLLM_TORCH_PROFILER_DIR only registers /start_profile and /stop_profile; it
# does not profile until asked. Set here so `vllm bench serve --profile` works
# on the held server with no restart.
docker run -d --name "$HOLD_NAME" "${DOCKER_COMMON[@]}" "${CELL_ENV[@]}" \
    -e VLLM_TORCH_PROFILER_DIR=/run_logs/prof \
    --entrypoint /bin/bash "$IMAGE" -c "
set -e
python3 ${COOKBOOK}/apply_mori_combine_original_topk_fix.py \
    /usr/local/lib/python3.12/dist-packages/vllm
n=\$(grep -c _original_topk_ids \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/prepare_finalize/mori.py || true)
echo \"[combine-patch] _original_topk_ids occurrences=\$n\"
[ \"\$n\" -ge 3 ] || { echo '[combine-patch] FATAL: patch did not land'; exit 3; }
vllm serve $(printf '%q ' "${CELL_ARGS[@]}") \
    > /run_logs/serve_hold_${HOLD_CELL}.log 2>&1 &
exec sleep infinity
" > /dev/null

echo "[hold] waiting for /health (up to 60 min)..."
HOLD_READY=0
for i in $(seq 1 360); do
    if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[hold] ready after $((i * 10))s"
        HOLD_READY=1
        break
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "$HOLD_NAME"; then
        echo "[hold] ERROR: hold container exited during startup"
        break
    fi
    sleep 10
done
[ "$HOLD_READY" != 1 ] && tail -60 "$HOLD_LOG" 2>/dev/null

cat <<EOF

##############################################################################
# JOB ${SLURM_JOB_ID} IS HOLDING -- server stays up until you scancel it.
#
#   node        $(hostname)   ip=${NODE_IP}
#   container   ${HOLD_NAME}
#   cell        ${HOLD_CELL}  (${CELL_DESC})
#   endpoint    http://127.0.0.1:${PORT}   (ready=${HOLD_READY})
#   logs        ${LOG_DIR}
#   serve log   ${HOLD_LOG}
#
# Shell in (repo is mounted at ${COOKBOOK}, logs at /run_logs):
#   ssh $(hostname)
#   docker exec -it ${HOLD_NAME} bash
#
# Re-bench the live server (from the node, host side):
#   docker run --rm --network host ${IMAGE} vllm bench serve \\
#     --model ${MODEL_PATH} --backend vllm --host 127.0.0.1 --port ${PORT} \\
#     --dataset-name random --random-input-len ${ISL} --random-output-len ${OSL} \\
#     --random-prefix-len 0 --num-prompts 16 --request-rate inf --ignore-eos \\
#     --percentile-metrics ttft,tpot,itl --max-concurrency 1
#
# Torch-profile a few decode steps (trace lands in ${LOG_DIR}/prof):
#   ... same bench command plus --profile --num-prompts 2
#
# Relaunch the server with different env (e.g. the decode step timer) without
# giving up the node -- the container survives the server:
#   docker exec -it ${HOLD_NAME} bash
#   pkill -f 'vllm serve'; sleep 20
#   export DSV4_DECODE_TIMER=1 DSV4_DECODE_TIMER_EVERY=1 DSV4_EAGER=1
#   bash /run_logs/serve_cmd_hold.sh        # host path: ${HOLD_CMD}
#
# Release the node when done:
#   scancel ${SLURM_JOB_ID}
##############################################################################

EOF

# `wait` on a background sleep so SIGTERM from scancel fires the trap at once
# instead of after the current sleep drains.
while true; do
    sleep 300 &
    wait $! || break
    left=$(squeue -h -j "${SLURM_JOB_ID}" -o %L 2>/dev/null)
    alive=$(docker ps --format '{{.Names}}' | grep -cx "$HOLD_NAME" || true)
    echo "[hold] $(date -u +%H:%M:%SZ) container_alive=${alive} time_left=${left:-unknown}"
done
