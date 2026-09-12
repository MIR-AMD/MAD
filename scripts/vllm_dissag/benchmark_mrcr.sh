#!/bin/bash
# Drop-in replacement for benchmark_xPyD.sh that runs OpenAI MRCR
# (https://huggingface.co/datasets/openai/mrcr) against the live disagg server.
# Selected via BENCHMARK_SCRIPT_FILE=benchmark_mrcr.sh.
#
# NOT product NIAH. Never set NIAH_LIST_PRIME. Flash-first 2-needle bins
# through 32k; raise MAX_MODEL_LEN before 64k–1M or CorpusQA.
set -u
timestamp=$(date "+%Y%m%d_%H%M%S")
BENCHMARK_PORT="${BENCHMARK_PORT:-30000}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="/run_logs/${SLURM_JOB_ID}/mrcr_${SLURM_JOB_ID}_${timestamp}_xP${xP}_yD${yD}_${MODEL_NAME}.log"

echo "==== OpenAI MRCR long-context ===="
echo "port=${BENCHMARK_PORT}  model=${MODEL_PATH}  needles=${MRCR_NEEDLES:-2}  bins=${MRCR_BINS:-8192,16384,32768}  per_bin=${MRCR_PER_BIN:-8}"

_ready=0
for _i in $(seq 1 60); do
    _code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
        "http://127.0.0.1:${BENCHMARK_PORT}/ready" 2>/dev/null || echo 000)
    if [ "$_code" = "200" ] || [ "$_code" = "405" ]; then
        _ready=1
        echo "[mrcr] proxy /ready HTTP ${_code} after ~$((_i*5))s"
        break
    fi
    sleep 5
done
if [ "$_ready" != 1 ]; then
    echo "[mrcr] Error: /ready not 200/405 in 300s (last HTTP ${_code:-none}). Abort." >&2
    curl -s -D - "http://127.0.0.1:${BENCHMARK_PORT}/ready" >&2 || true
    exit 1
fi

MRCR_URL="http://127.0.0.1:${BENCHMARK_PORT}" \
MRCR_MODEL="${MODEL_PATH}" \
MRCR_NEEDLES="${MRCR_NEEDLES:-2}" \
MRCR_BINS="${MRCR_BINS:-8192,16384,32768}" \
MRCR_PER_BIN="${MRCR_PER_BIN:-8}" \
MRCR_MAXTOK="${MRCR_MAXTOK:-1024}" \
MRCR_TIMEOUT="${MRCR_TIMEOUT:-1800}" \
MRCR_MAX_CTX="${MRCR_MAX_CTX:-40960}" \
MRCR_THINKING="${MRCR_THINKING:-chat}" \
MRCR_DATA_DIR="${MRCR_DATA_DIR:-}" \
  python3 "${DIR}/benchmark_mrcr.py" 2>&1 | tee -a "${LOG}"
rc=${PIPESTATUS[0]}
echo "MRCR results -> ${LOG}"
exit "$rc"
