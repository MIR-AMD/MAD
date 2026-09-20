#!/bin/bash
#SBATCH -A amd-rccl-guest
#SBATCH -p amd-arad-burst
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:8
#SBATCH --time=02:00:00
#SBATCH --job-name=dsv4fls-tp8-decode
#SBATCH --exclusive
#SBATCH --exclude=useocpm2m-097-[008,014-015,017,019-020,023,025,027-028,030,033,039-043,045,049,069,077,080,082-084,089,094-095,100,114-115,121-125,132,135-136,139-140,142,144,154]
#SBATCH --output=/shared_inference/%u/model_blog_logs/slurm-%j.out
#SBATCH --error=/shared_inference/%u/model_blog_logs/slurm-%j.err
# ---------------------------------------------------------------------------
# Single-node Flash TP8 decode smoke. No PD, no proxy, no router, no DP, no EP,
# no MoRIIO, no patchers. Plain `vllm serve --tensor-parallel-size 8`, then:
#   1. three chat QA curls (Lisa / Paris / London) + one /v1/completions
#   2. vllm bench serve isl/osl=32/32 at con=1,4,8 (decode ITL; same shape as 434317)
#
#   cd scripts/vllm_dissag
#   sbatch run_dsv4_colocated_tp8.slurm
# ---------------------------------------------------------------------------
set -euo pipefail

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

LOG_DIR="/shared_inference/${USER}/model_blog_logs/${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR" 2>/dev/null || true
SERVE_NAME="dsv4coloc-${SLURM_JOB_ID}"

_job_tag="dsv4fls"
case "$MODEL" in pro|dsv4pro) _job_tag="dsv4pro" ;; esac
scontrol update "JobId=${SLURM_JOB_ID}" "JobName=${_job_tag}-tp8-decode" 2>/dev/null || true

echo "=== DSV4 co-located TP8 decode smoke (no disagg / no MoRIIO / no proxy) ==="
echo "Job:      $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Model:    $MODEL_NAME"
echo "Path:     $MODEL_PATH"
echo "Image:    $IMAGE"
echo "=============================================================="

DOCKER_ARGS=(
    --rm --privileged --network=host --ipc=host
    --device=/dev/kfd --device=/dev/dri
    --group-add video --cap-add=SYS_PTRACE
    -v /mnt:/mnt
    -v /shared_inference:/shared_inference
    -v "${LOG_DIR}:/run_logs"
    -e VLLM_USE_V1=1
    -e VLLM_ROCM_USE_AITER=1
    -e VLLM_ROCM_USE_AITER_PAGED_ATTN=0
    -e VLLM_ROCM_USE_AITER_RMSNORM=1
    -e VLLM_USE_AITER_TRITON_SILU_MUL=0
    -e VLLM_ROCM_USE_AITER_MLA=1
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600
    -e VLLM_LOGGING_LEVEL=INFO
)

# max-num-seqs 16 so con=8 actually batches (4 would queue the extra 4).
SERVE_ARGS=(
    "${MODEL_PATH}"
    --trust-remote-code
    --tokenizer-mode deepseek_v4
    --port "${PORT}"
    --tensor-parallel-size 8
    --block-size 256
    --kv-cache-dtype fp8_e4m3
    --gpu-memory-utilization 0.80
    --kv-cache-memory-bytes 20000000000
    --max-num-seqs 16
    --max-model-len 40960
    --max-num-batched-tokens 1024
    --no-enable-prefix-caching
    --distributed-timeout-seconds 7200
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","custom_ops":["+quant_fp8"]}'
)

echo "[serve] vllm serve ${SERVE_ARGS[*]}"
docker run --name "$SERVE_NAME" "${DOCKER_ARGS[@]}" "$IMAGE" \
    vllm serve "${SERVE_ARGS[@]}" \
    > "${LOG_DIR}/colocated_tp8_serve.log" 2>&1 &
SERVE_PID=$!

cleanup() {
    docker rm -f "$SERVE_NAME" >/dev/null 2>&1 || true
    kill "$SERVE_PID" 2>/dev/null || true
    wait "$SERVE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[main] waiting for /health (up to 60 min)..."
READY=0
for i in $(seq 1 360); do
    if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "[main] server ready after $((i*10))s"
        READY=1
        break
    fi
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
        echo "[main] ERROR: vllm serve exited during startup"
        tail -80 "${LOG_DIR}/colocated_tp8_serve.log"
        exit 1
    fi
    sleep 10
done
if [ "$READY" != "1" ]; then
    echo "[main] ERROR: server never became ready"
    tail -80 "${LOG_DIR}/colocated_tp8_serve.log"
    exit 1
fi
sleep 5

SMOKE_LOG="${LOG_DIR}/smoke_colocated_tp8.log"
BENCH_LOG="${LOG_DIR}/benchmark_colocated_tp8_CONCURRENCY.log"
echo "===== smoke curl: 3 chat QA + 1 completions =====" | tee "$SMOKE_LOG"

set +e
python3 - "$PORT" "$SMOKE_LOG" "$MODEL_PATH" <<'PY'
import json, sys, time, urllib.request
port, log_path, model = sys.argv[1], sys.argv[2], sys.argv[3]
url_chat = f"http://127.0.0.1:{port}/v1/chat/completions"
url_cmp = f"http://127.0.0.1:{port}/v1/completions"
probes = (
    ("amd", "Who is the CEO of AMD? Answer in one sentence.", "lisa"),
    ("france", "What is the capital of France? Answer in one sentence.", "paris"),
    ("uk", "What is the capital of the United Kingdom? Answer in one sentence.", "london"),
)

def post(url, payload, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())

n_ok = 0
lines = [f"===== colocated TP8 chat QA port={port} model={model} ====="]
for tag, question, expect in probes:
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 64,
        "top_k": 1,
        "messages": [{"role": "user", "content": question}],
    }
    lines.append(f"===== curl[{tag}] =====")
    t0 = time.time()
    try:
        body = post(url_chat, payload)
        dt = time.time() - t0
        text = body["choices"][0]["message"]["content"]
        hit = expect.lower() in text.lower()
        n_ok += int(hit)
        verdict = "COHERENT" if hit else "MISS"
        lines.append(f"dt={dt:.1f}s verdict={verdict} expect={expect!r}")
        lines.append(f"reply={text!r}")
    except Exception as e:
        dt = time.time() - t0
        lines.append(f"dt={dt:.1f}s FAIL {type(e).__name__}: {e}")

cmp_payload = {
    "model": model,
    "prompt": "Who is the CEO of AMD?",
    "temperature": 0,
    "max_tokens": 32,
}
lines.append("===== curl[completions] =====")
t0 = time.time()
rc = 0 if n_ok == 3 else 1
try:
    body = post(url_cmp, cmp_payload)
    dt = time.time() - t0
    text = body["choices"][0]["text"]
    lines.append(f"dt={dt:.1f}s chars={len(text)} text={text!r}")
    if not text.strip():
        lines.append("FAIL empty completions body")
        rc = 1
except Exception as e:
    dt = time.time() - t0
    lines.append(f"dt={dt:.1f}s FAIL {type(e).__name__}: {e}")
    rc = 1

lines.append(f"===== chat {n_ok}/3 =====")
report = "\n".join(lines)
print(report)
with open(log_path, "a") as f:
    f.write(report + "\n")
sys.exit(rc)
PY
CURL_RC=$?
set -e

ISL="${BENCH_ISL:-32}"
OSL="${BENCH_OSL:-32}"
CONS="${BENCHMARK_CON:-1 4 8}"
echo "===== vllm bench serve isl=${ISL} osl=${OSL} con=${CONS} =====" | tee -a "$SMOKE_LOG" "$BENCH_LOG"
echo "[bench] host=127.0.0.1 port=${PORT} (colocated TP8, no proxy)" | tee -a "$BENCH_LOG"

BENCH_RC=0
run_bench() {
    local con="$1"
    local n_prompts="$2"
    local tag="$3"
    echo "[RUNNING] ${tag} prompts=${n_prompts} isl=${ISL} osl=${OSL} con=${con}" | tee -a "$BENCH_LOG"
    set +e
    # Same mounts as the server: without them transformers treats MODEL_PATH
    # as a HF repo id (435757).
    timeout 1800 docker run --rm --network=host \
        -v /mnt:/mnt -v /shared_inference:/shared_inference \
        -e VLLM_USE_V1=1 \
        "$IMAGE" \
        vllm bench serve \
            --model "$MODEL_PATH" \
            --backend vllm \
            --host 127.0.0.1 \
            --port "$PORT" \
            --dataset-name random \
            --random-input-len "$ISL" \
            --random-output-len "$OSL" \
            --random-prefix-len 0 \
            --num-prompts "$n_prompts" \
            --request-rate inf \
            --ignore-eos \
            --max-concurrency "$con" \
        2>&1 | tee -a "$BENCH_LOG"
    local rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" -eq 124 ]; then
        echo "[STALL] ${tag} con=${con} timed out" | tee -a "$BENCH_LOG" "$SMOKE_LOG"
    elif [ "$rc" -ne 0 ]; then
        echo "[FAIL] ${tag} con=${con} rc=${rc}" | tee -a "$BENCH_LOG" "$SMOKE_LOG"
    fi
    if [ "$rc" -ne 0 ] && [ "$BENCH_RC" -eq 0 ]; then
        BENCH_RC=$rc
    fi
}

run_bench 1 16 warmup
for con in $CONS; do
    n_prompts=$((con * 2))
    if [ "$n_prompts" -lt 16 ]; then
        n_prompts=16
    fi
    run_bench "$con" "$n_prompts" "bench"
done

echo "===== harvest (Median ITL) =====" | tee -a "$SMOKE_LOG"
awk '/Maximum request concurrency:/{c=$4} /Successful requests:/{s=$3} /Failed requests:/{f=$3} /Output token throughput/{o=$5} /Median TTFT/{t=$4} /Median ITL/{print "  con="c" ok="s" fail="f" tok/s="o" TTFT="t" ITL="$4}' \
    "$BENCH_LOG" | tee -a "$SMOKE_LOG"

echo "Done. curl_rc=${CURL_RC} bench_rc=${BENCH_RC} Logs: ${LOG_DIR}"
if [ "$CURL_RC" -ne 0 ]; then
    exit "$CURL_RC"
fi
exit "$BENCH_RC"
