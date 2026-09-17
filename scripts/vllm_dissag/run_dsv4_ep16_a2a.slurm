#!/bin/bash
#SBATCH -A amd-rccl-guest
#SBATCH -p amd-arad-burst
#SBATCH -q low
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --job-name=dsv4fls-ep16-a2a
#SBATCH --exclusive
#SBATCH --output=/shared_inference/%u/model_blog_logs/slurm-%j.out
#SBATCH --error=/shared_inference/%u/model_blog_logs/slurm-%j.err
# ---------------------------------------------------------------------------
# EP16 all2all-backend A/B. Two nodes, 16 GPUs, `-tp 1` + DP16 + EP16.
# No PD, no proxy, no router, no connector.
#
# Why: on ONE node (17 Sep) the WideEP decode regression was localized to the
# MoRI all2all integration under data parallelism —
#
#     TP1 x DP8 x EP8  mori_low_latency      ~302 ms ITL
#     TP1 x DP8 x EP8  mori_high_throughput   302.6 ms
#     TP1 x DP8 x EP8  allgather_reducescatter  22.9 ms
#     TP8 x DP1 x EP8  mori_low_latency         24.0 ms   (DP1 -> fine)
#     TP4 x DP2        no EP                    23.7 ms
#
# Both questions this job answers are NEW, because every cell above was
# intra-node:
#
#   1. Does the MoRI+DP stall persist across nodes at EP16? Expect ~300 ms. If
#      it instead scales with EP width or node count, the "fixed wait" reading
#      is wrong and the fault is bandwidth, not a barrier.
#   2. Does `allgather_reducescatter` survive leaving the box? It replicates
#      every token to every expert rank, which is nearly free over xGMI inside
#      one node and is NOT free over the fabric. This is the run that decides
#      whether it is a shippable config or only an intra-node diagnostic.
#
# Read cell 1 (allgather) first: it is the unknown. The MoRI cell is the control
# and its expected value is already known, so it runs second on purpose —
# if the job dies early we still gain information.
#
#   cd scripts/vllm_dissag
#   sbatch run_dsv4_ep16_a2a.slurm
#   CELLS=ep16_allgather sbatch run_dsv4_ep16_a2a.slurm            # one cell
#   BENCHMARK_CON="1 4 16" sbatch run_dsv4_ep16_a2a.slurm          # concurrency ladder
#   HOLD=0 sbatch run_dsv4_ep16_a2a.slurm                          # tear down at the end
#
# Snapshot: scripts/vllm_dissag/reports/STANDUP_2026-09-17_mori_dp_a2a.md
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
DP_RPC_PORT="${DP_RPC_PORT:-13345}"
REPO_DIR="$(pwd)"
COOKBOOK=/opt/nixl-vllm-cookbook

ISL="${BENCH_ISL:-32}"
OSL="${BENCH_OSL:-32}"
CONS="${BENCHMARK_CON:-1}"

# allgather first: it is the unknown. mori is the control with a known value.
CELLS="${CELLS:-ep16_allgather ep16_mori ep16_mori_ht}"

LOG_DIR="${LOG_DIR:-/shared_inference/${USER}/model_blog_logs/${SLURM_JOB_ID}}"
mkdir -p "$LOG_DIR" 2>/dev/null || true
SUMMARY="${LOG_DIR}/ep16_a2a_summary.log"

JIT_CACHE_BASE="${JIT_CACHE_BASE:-/mnt/m2m_nobackup/${USER}/vllm_jit_cache/v0290-aiter-10f8874-mori07bdace-tk}"

# ---------------------------------------------------------------------------
# Topology: node 0 is the DP master + API servers + bench client.
#           node 1 is a --headless child at --data-parallel-start-rank 8.
# ---------------------------------------------------------------------------
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "${#NODES[@]}" -ne 2 ]; then
    echo "FATAL: need exactly 2 nodes, got ${#NODES[@]}: ${NODES[*]}" >&2
    exit 2
fi
NODE0="${NODES[0]}"
NODE1="${NODES[1]}"

HEAD_IP="$(srun --overlap -N1 -n1 --nodelist="$NODE0" bash -c 'hostname -I' | awk '{print $1}')"
if [ -z "$HEAD_IP" ]; then
    echo "FATAL: could not resolve head IP on $NODE0; --data-parallel-address would be empty" >&2
    exit 2
fi

DP_TOTAL=16
DP_LOCAL=8

echo "=== DSV4 EP16 all2all-backend A/B (2 nodes, no PD / proxy / router) ==="
echo "Job:      $SLURM_JOB_ID"
echo "Nodes:    master=$NODE0 ($HEAD_IP)  child=$NODE1"
echo "Model:    $MODEL_NAME"
echo "Path:     $MODEL_PATH"
echo "Image:    $IMAGE"
echo "Topology: -tp 1  dp=${DP_TOTAL} (local ${DP_LOCAL})  ep=${DP_TOTAL}"
echo "Cells:    $CELLS"
echo "Shape:    isl=$ISL osl=$OSL con=$CONS"
echo "Logs:     $LOG_DIR"
echo "======================================================================="

# ---------------------------------------------------------------------------
# Remote helper. Everything that touches a node goes through this so the
# --overlap / no-gres form is in exactly one place (a bare --gres=gpu:0 hangs
# on an exclusive allocation).
# ---------------------------------------------------------------------------
on_node() {
    local node="$1"; shift
    srun --overlap -N1 -n1 --nodelist="$node" bash -c "$*"
}

MASTER_NAME=""
CHILD_NAME=""
# scancel sends SIGTERM and a bare EXIT trap does not run on it. Without this,
# dockerd keeps containers alive holding all 16 GPUs and the next job on either
# node dies in ncclCommInitRank with "unhandled cuda error".
cleanup() {
    [ -n "$MASTER_NAME" ] && on_node "$NODE0" "docker rm -f $MASTER_NAME >/dev/null 2>&1 || true" || true
    [ -n "$CHILD_NAME" ]  && on_node "$NODE1" "docker rm -f $CHILD_NAME  >/dev/null 2>&1 || true" || true
}
trap cleanup EXIT INT TERM

: > "$SUMMARY"

# ---------------------------------------------------------------------------
# docker run preamble, rendered per node (JIT cache is per-node so two nodes
# cannot race on the same AITER/Triton cache files).
# ---------------------------------------------------------------------------
docker_common_for() {
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
-e VLLM_LOGGING_LEVEL=INFO
EOF
}

backend_for() {
    case "$1" in
        ep16_allgather) echo allgather_reducescatter ;;
        ep16_mori)      echo mori_low_latency ;;
        ep16_mori_ht)   echo mori_high_throughput ;;
        *) return 2 ;;
    esac
}

# Serve args shared by master and child. The ONLY difference between cells is
# --all2all-backend; everything else is byte-identical so the A/B is single
# variable.
common_serve_args() {
    local a2a="$1"
    printf '%s\n' \
        "${MODEL_PATH}" \
        --trust-remote-code \
        --tokenizer-mode deepseek_v4 \
        --port "${PORT}" \
        --tensor-parallel-size 1 \
        --data-parallel-size "${DP_TOTAL}" \
        --data-parallel-size-local "${DP_LOCAL}" \
        --data-parallel-address "${HEAD_IP}" \
        --data-parallel-rpc-port "${DP_RPC_PORT}" \
        --enable-expert-parallel \
        --all2all-backend "${a2a}" \
        --block-size 256 \
        --kv-cache-dtype fp8_e4m3 \
        --gpu-memory-utilization 0.80 \
        --kv-cache-memory-bytes 20000000000 \
        --max-num-seqs "${MAX_NUM_SEQS:-4}" \
        --max-model-len 40960 \
        --max-num-batched-tokens 1024 \
        --no-enable-prefix-caching \
        --distributed-timeout-seconds 7200 \
        --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["+quant_fp8"]}'
}

# Free-VRAM gate. A node that still holds a foreign container reports non-zero
# here, and booting into it wastes the whole load time (node 094, 17 Sep: a job
# cancelled 29 h earlier still held ~159 GiB/GPU and `sinfo` said idle).
preflight() {
    local node="$1" busy
    busy=$(on_node "$node" \
        "rocm-smi --showmemuse 2>/dev/null | grep -oP 'VRAM%\\): \\K[0-9]+' | sort -rn | head -1")
    busy="${busy:-99}"
    if [ "$busy" -gt 5 ]; then
        echo "[preflight] ABORT: $node has a GPU at ${busy}% VRAM before boot -- dirty node."
        on_node "$node" "docker ps --format '{{.Names}}\t{{.Status}}'" || true
        return 1
    fi
    echo "[preflight] $node clean (max ${busy}% VRAM)"
    return 0
}

launch_cell() {
    local cell="$1" a2a="$2"
    local args child_args
    mapfile -t args < <(common_serve_args "$a2a")

    MASTER_NAME="dsv4ep16-${SLURM_JOB_ID}-${cell}-m"
    CHILD_NAME="dsv4ep16-${SLURM_JOB_ID}-${cell}-c"

    # The mori combine fix (Wei Sun / mori#475) is applied on BOTH nodes for
    # every cell, including the allgather cell, so the source tree is identical
    # across the A/B and a patch difference cannot be mistaken for a backend
    # result. NOTE (17 Sep): EP8 booted unpatched on v0290/mori07bdace, so the
    # assert below may be the only thing proving the patch is present at all.
    local patch_block="
python3 ${COOKBOOK}/apply_mori_combine_original_topk_fix.py \
    /usr/local/lib/python3.12/dist-packages/vllm
n=\\\$(grep -c _original_topk_ids \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/prepare_finalize/mori.py || true)
echo \\\"[combine-patch] _original_topk_ids occurrences=\\\$n\\\"
[ \\\"\\\$n\\\" -ge 3 ] || { echo '[combine-patch] FATAL: patch did not land'; exit 3; }
"

    # Child first: the master blocks on the DP handshake, so starting it second
    # avoids a startup-timeout race on the child's much slower first JIT.
    echo "[launch] child  $NODE1  ranks 8-15  (--headless --data-parallel-start-rank 8)"
    child_args=$(printf '%q ' "${args[@]}" --headless --data-parallel-start-rank "${DP_LOCAL}")
    on_node "$NODE1" "docker rm -f $CHILD_NAME >/dev/null 2>&1 || true; \
        docker run -d --name $CHILD_NAME $(docker_common_for) \
        --entrypoint /bin/bash $IMAGE -c \"set -e${patch_block}exec vllm serve ${child_args}\" >/dev/null"

    sleep 10

    echo "[launch] master $NODE0  ranks 0-7   (--api-server-count=${DP_LOCAL})"
    local master_args
    master_args=$(printf '%q ' "${args[@]}" "--api-server-count=${DP_LOCAL}")
    on_node "$NODE0" "docker rm -f $MASTER_NAME >/dev/null 2>&1 || true; \
        docker run -d --name $MASTER_NAME $(docker_common_for) \
        --entrypoint /bin/bash $IMAGE -c \"set -e${patch_block}exec vllm serve ${master_args}\" >/dev/null"
}

teardown_cell() {
    local cell="$1"
    on_node "$NODE0" "docker logs $MASTER_NAME > /run_logs/serve_${cell}_master.log 2>&1 || true"
    on_node "$NODE1" "docker logs $CHILD_NAME  > /run_logs/serve_${cell}_child.log  2>&1 || true"
    on_node "$NODE0" "docker rm -f $MASTER_NAME >/dev/null 2>&1 || true"
    on_node "$NODE1" "docker rm -f $CHILD_NAME  >/dev/null 2>&1 || true"
    MASTER_NAME=""
    CHILD_NAME=""
    sleep 25
}

run_cell() {
    local cell="$1"
    local a2a; a2a="$(backend_for "$cell")" || { echo "unknown cell $cell" >&2; return 1; }
    local bench_log="${LOG_DIR}/bench_${cell}.log"

    echo ""
    echo "############################################################"
    echo "## CELL ${cell}   -tp 1  dp=${DP_TOTAL}  ep=${DP_TOTAL}  a2a=${a2a}"
    echo "############################################################"

    if ! preflight "$NODE0" || ! preflight "$NODE1"; then
        echo "${cell} | a2a=${a2a} | DIRTY_NODE | - | -" >> "$SUMMARY"
        return 0
    fi

    launch_cell "$cell" "$a2a"

    # 16 ranks across 2 nodes serialize on the AITER JIT lock on a cold cache,
    # so the first cell can take far longer than a single-node boot.
    echo "[main] waiting for /health on ${NODE0}:${PORT} (up to 60 min)..."
    local ready=0 i
    for i in $(seq 1 360); do
        if on_node "$NODE0" "curl -s http://127.0.0.1:${PORT}/health >/dev/null 2>&1"; then
            echo "[main] ${cell} ready after $((i * 10))s"
            ready=1
            break
        fi
        if ! on_node "$NODE0" "docker ps --format '{{.Names}}' | grep -qx $MASTER_NAME"; then
            echo "[main] ERROR: ${cell} master container exited during startup"
            break
        fi
        sleep 10
    done

    if [ "$ready" != 1 ]; then
        echo "[main] ${cell} NEVER READY -- last 60 lines of master:"
        on_node "$NODE0" "docker logs --tail 60 $MASTER_NAME" 2>&1 || true
        echo "[main] ${cell} last 40 lines of child:"
        on_node "$NODE1" "docker logs --tail 40 $CHILD_NAME" 2>&1 || true
        echo "${cell} | a2a=${a2a} | NOT_READY | - | -" >> "$SUMMARY"
        teardown_cell "$cell"
        return 0
    fi
    sleep 5

    # Bench from node 0, inside the image, so the client interpreter is not a
    # second variable. Warmup pass first: it pays cudagraph capture plus Triton
    # and AITER JIT on the decode shapes and is NOT the number.
    {
        echo "===== ${cell} a2a=${a2a} isl=${ISL} osl=${OSL} ====="
        for con in warmup $CONS; do
            local c=$con n=16
            [ "$con" = warmup ] && c=1
            [ "$c" -gt 8 ] && n=$((c * 4))
            echo "[RUNNING] ${cell} tag=${con} con=${c} prompts=${n}"
            on_node "$NODE0" "timeout 1800 docker run --rm --network host $IMAGE \
                vllm bench serve \
                    --model $MODEL_PATH --backend vllm \
                    --host 127.0.0.1 --port $PORT \
                    --dataset-name random \
                    --random-input-len $ISL --random-output-len $OSL \
                    --random-prefix-len 0 \
                    --num-prompts $n --request-rate inf --ignore-eos \
                    --percentile-metrics ttft,tpot,itl \
                    --max-concurrency $c"
            [ $? -eq 124 ] && echo "[STALL] ${cell} con=${c} timed out"
        done
    } 2>&1 | tee "$bench_log"

    local itl ttft
    itl=$(awk '/Median ITL/{v=$4} END{print (v==""?"-":v)}' "$bench_log")
    ttft=$(awk '/Median TTFT/{v=$4} END{print (v==""?"-":v)}' "$bench_log")
    echo "${cell} | a2a=${a2a} | OK | ITL=${itl} ms | TTFT=${ttft} ms" >> "$SUMMARY"
    echo "[harvest] ${cell} median ITL=${itl} ms TTFT=${ttft} ms"

    teardown_cell "$cell"
}

for cell in $CELLS; do
    run_cell "$cell"
done

echo ""
echo "================= EP16 all2all summary ================="
cat "$SUMMARY"
echo "========================================================"
echo "Single-node reference (17 Sep, EP8):"
echo "  TP1 x DP8 x EP8  allgather_reducescatter    22.9 ms"
echo "  TP1 x DP8 x EP8  mori_low_latency          ~302   ms"
echo "  TP1 x DP8 x EP8  mori_high_throughput       302.6 ms"
echo "  TP8 x DP1 x EP8  mori_low_latency            24.0 ms"
echo "========================================================"

# ---------------------------------------------------------------------------
# Hold the allocation so follow-ups do not requeue. scancel tears it down via
# the TERM trap.
# ---------------------------------------------------------------------------
if [ "${HOLD:-1}" != 1 ]; then
    echo "[hold] HOLD=0 -- exiting and tearing down."
    exit 0
fi

HOLD_CELL="${HOLD_CELL:-ep16_allgather}"
HOLD_A2A="$(backend_for "$HOLD_CELL")" || HOLD_A2A=allgather_reducescatter

echo ""
echo "[hold] re-serving ${HOLD_CELL} (a2a=${HOLD_A2A}) and idling until scancel."
if preflight "$NODE0" && preflight "$NODE1"; then
    launch_cell "$HOLD_CELL" "$HOLD_A2A"
    for i in $(seq 1 360); do
        on_node "$NODE0" "curl -s http://127.0.0.1:${PORT}/health >/dev/null 2>&1" && break
        sleep 10
    done
    echo "[hold] server should be live at ${HEAD_IP}:${PORT}"
    echo "[hold] bench from ${NODE0} with:"
    echo "[hold]   srun --overlap -N1 -n1 --nodelist=${NODE0} --pty bash"
    echo "[hold]   docker exec -it ${MASTER_NAME} bash"
else
    echo "[hold] preflight failed; not serving. Node is still held."
fi

while true; do
    sleep 300 &
    wait $! || break
    left=$(squeue -h -j "${SLURM_JOB_ID}" -o %L 2>/dev/null)
    echo "[hold] $(date -u +%H:%M:%SZ) time_left=${left:-unknown}"
done
