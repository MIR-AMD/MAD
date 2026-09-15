#!/bin/bash
# Track A WideEP submit — derive every dependent knob from BENCH + MODEL + TOPO.
# =============================================================================
# Login node only (useocpslog-002). Do not run on WSL. Do not docker build here.
#
#   ./run_wideep_bench.sh niah glm 2p2d
#   ./run_wideep_bench.sh niah glm 4p4d
#   ./run_wideep_bench.sh niah hy3 3p3d
#   ./run_wideep_bench.sh niah glm52 4p4d
#   ./run_wideep_bench.sh niah dsv3 4p4d --image e03
#   ./run_wideep_bench.sh smoke dsv4fls 1p1d --image v0280
#   ./run_wideep_bench.sh smoke dsv4pro 1p1d --image v0280
#   ./run_wideep_bench.sh validate dsv4fls ep16 --image v0290
#   ./run_wideep_bench.sh validate dsv4pro ep32 --image v0290
#   ./run_wideep_bench.sh niah glm 4p4d --dry-run
  # OpenAI MRCR (no prime). Flash 2-needle through 32k, then Pro if MMR looks sane:
  #   MRCR_PER_BIN=8 MRCR_DATA_DIR=/shared_inference/bbarakat/datasets/openai_mrcr \\
  #   ./run_wideep_bench.sh mrcr dsv4fls 2p2d --image v0290 --time 04:00:00 --exclude ''
  # Flash HMA=0 2P/2D product+prime (colocated 225021 cell, on the product arm):
  #   DSV4_ENABLE_HMA=0 NIAH_METHOD=product NIAH_MAXTOK=512 \\
  #   NIAH_WORDS=2000,8000,16000 NIAH_SEEDS=0,0,0 NIAH_LIST_PRIME='1.' \\
  #   NIAH_STOP='11.|```|<｜end▁of▁file｜>|<｜begin▁of▁file▁name｜>' NIAH_LOGPROBS=5 \\
  #   ./run_wideep_bench.sh niah dsv4fls 2p2d --image v0280 --time 06:00:00
  # vllm-router (v0290 has no bake): git clone + cargo at NODE0 boot, not pip.
  #   QOS=low PROXY_TYPE=vllm_router ROUTER_BOOT_INSTALL=git \\
  #   NIAH_METHOD=product NIAH_MAXTOK=64 NIAH_SEEDS=0,0,0 NIAH_LIST_PRIME='1.' \\
  #   ./run_wideep_bench.sh niah dsv4fls 2p2d --image v0290 --time 04:00:00 --after JOBID
#
# Changing EP16 → EP24/EP32 is TOPO only. Do not set xP/yD/-N/PROXY_ROUTE_DP by hand.
#
# Not Track B (ROCm/MAD#176): never sets GLM_SKIP_PATCHERS. Persist-gate default
# is GATE=0 in moriio.sh; do not export GLM_PERSIST_GATE unless overriding.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMG_E03="rocm/pytorch-private:vllm-recent-source-basem-acb0f1dc-aiter-e03fa6040"
IMG_E03_TK="rocm/pytorch-private:vllm-recent-source-basem-acb0f1dc-aiter-e03fa6040-tk"
IMG_5A4C="rocm/pytorch-private:vllm-recent-source-basem-5a4c8d99-aiter-e03fa6040-tk"
IMG_D626="rocm/pytorch-private:vllm-recent-source-basem-d626108b-aiter-1d872fa-tk"
# 217850: that tag has AITER 1d872fa + flydsl 0.1.8. Use fd031.
IMG_D626FD="rocm/pytorch-private:vllm-recent-source-basem-d626108b-aiter-1d872fa-fd031-tk"
# fd031 + MoRI cfe7ed38 (2026-08-07) -> 624002c8 (2026-08-21), for the HMA=1
# arm. vllm#48989 only asks for MoRI v1.2.1; this pin is 103 commits past it.
IMG_MORI624="rocm/pytorch-private:vllm-recent-source-basem-d626108b-aiter-1d872fa-fd031-mori624002-tk"
# vLLM release v0.28.0 (2cf0a691) + MoRI main 6fcf6b3 (2026-08-26, +12 over
# 624002c8). AITER/flydsl/triton_kernels identical to mori624002, so a diff vs
# mori624002 is vLLM-or-MoRI and cannot be attributed further: the tie-breaker
# image (d626108b + 6fcf6b3) is written but SHELVED, build it only if this
# regresses. v0.28.0 is 13 ahead / 150 BEHIND d626108b -- reproducible release,
# not a newer vLLM.
IMG_V0280="rocm/pytorch-private:vllm-recent-source-basem-v0280-aiter-1d872fa-fd031-mori6fcf6b3-tk"
# vLLM release v0.29.0 (98dff2a) + AITER main 10f8874 + MoRI main 07bdace.
# Dockerfile: docker/vllm_disagg_inference.dsv4.v0290.ubuntu.amd.Dockerfile
# (v0280 is docker/vllm_disagg_inference.dsv4.v0280.ubuntu.amd.Dockerfile).
# Do not mix scores. Hub base is still ROCm 7.2.3 — AITER is source-built,
# not the UFB +rocm10.1.0a wheel.
IMG_V0290="rocm/pytorch-private:vllm-recent-source-basem-v0290-aiter-10f8874-mori07bdace-tk"
IMG_026="rocm/pytorch-private:vllm-recent-source-basem-20260815"

# Live exclude (override with EXCLUDE_NODES= or --exclude). Empty + --nodelist
# means "only that pool".
DEFAULT_EXCLUDE='useocpm2m-097-[008,015,019-020,025,033,040-042,045,049,069,077,080,082-084,089,094-095,100,114,115,121-125,132,135-136,139-140,142,144,154]'
# GLM-5.2 NVMe pool (/mnt/m2m_nobackup/models_blog/GLM-5.2-FP8). glm52 defaults here.
GLM52_STAGED='useocpm2m-097-[017,023,026,028,030,033,038,039,040,045,051,069,077,078,080,082,083,084,089,094,095,100,114,115,122,123,124,125,132,135,137,139,140,142,144,148,151,153,155]'

usage() {
    cat <<'EOF'
Usage: ./run_wideep_bench.sh BENCH MODEL TOPO [flags]

  BENCH   niah | smoke | mrcr | validate
  MODEL   glm | glm51 | glm52 | dsv3 | dsv4fls | dsv4pro | hy3 | hy3p | GLM-5.1-FP8 | ...
  TOPO    1p1d | 2p2d | 3p3d | 4p4d | ep8 | ep16 | ep24 | ep32
          asymmetric: 1p2d | 2p1d | 2p3d | 3p2d  (driver derives per-role DP)

Flags:
  --dry-run              print sbatch, do not submit
  --image e03|e03tk|5a4c|d626|d626fd|mori624002|v0280|v0290|026|<tag>
                         image (default: glm*→e03, dsv4fls/dsv4pro→mori624002, dsv3/hy3→026)
                         mori624002 = DSV4 vehicle. Hub d626108b + AITER 1d872fa + flydsl==0.3.1
                                  + MoRI 624002c8 + gRPC/UMBP. Pair with DSV4_HMA_UPSTREAM_GEOM=1.
                         v0280  = vLLM release v0.28.0 + MoRI 6fcf6b3. Same AITER/flydsl/tk as
                                  mori624002, so a delta is vLLM-or-MoRI, not attributable further.
                                  Unproven: smoke it, then HMA=0 2k+8k vs 218778 before any claim.
                         v0290  = vLLM v0.29.0 + AITER main 10f8874 + MoRI 07bdace. Own Dockerfile.
                                  Hub base still ROCm 7.2.3. flydsl==0.3.2. Do not mix with v0280.
                         d626fd = same vLLM/AITER/flydsl, older MoRI cfe7ed38, no UMBP. Closed.
                         d626   = alias of d626fd. Poison …-1d872fa-tk (flydsl 0.1.8) is 217850.
                         5a4c   = Hub 5a4c8d99 + AITER e03 + triton_kernels wheel. Not DSV4 HMA.
                         e03tk  = old acb0f1dc e03 + wheel only
  --exclude LIST         slurm --exclude (ignored if --nodelist is set)
  --nodelist LIST|glm52  pin to staged NVMe pool (no --exclude)
  --name NAME            slurm job name override
  --time HH:MM:SS        walltime override
  --after JOBID          sbatch --dependency=afterany:JOBID (queue behind a live job)

Env still wins for rare knobs (NIAH_SEEDS, PARTITION, DSV4_EAGER, ...).
  DSV4_EAGER=1  one-shot: decode cudagraph NONE +quant_fp8 (never --enforce-eager).
EOF
    exit 2
}

BENCH="${1:-}"
MODEL_IN="${2:-}"
TOPO_IN="${3:-}"
[[ -n "$BENCH" && -n "$MODEL_IN" && -n "$TOPO_IN" ]] || usage
shift 3

DRY_RUN=0
IMAGE_ARG=""
EXCLUDE_ARG="${EXCLUDE_NODES:-$DEFAULT_EXCLUDE}"
EXCLUDE_EXPLICIT=0
NODELIST_ARG="${NODELIST:-}"
JOB_NAME_ARG=""
TIME_ARG=""
AFTER_JOB=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --image) IMAGE_ARG="${2:-}"; shift 2 ;;
        --exclude) EXCLUDE_ARG="${2:-}"; EXCLUDE_EXPLICIT=1; shift 2 ;;
        --nodelist) NODELIST_ARG="${2:-}"; shift 2 ;;
        --name) JOB_NAME_ARG="${2:-}"; shift 2 ;;
        --time) TIME_ARG="${2:-}"; shift 2 ;;
        --after) AFTER_JOB="${2:-}"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Error: unknown flag '$1'" >&2; usage ;;
    esac
done

# --- MODEL ---
case "$MODEL_IN" in
    glm|glm51|GLM-5.1-FP8) MODEL_NAME="GLM-5.1-FP8"; SHORT=glm ;;
    glm52|GLM-5.2-FP8)     MODEL_NAME="GLM-5.2-FP8"; SHORT=glm52 ;;
    dsv3|DeepSeek-V3)      MODEL_NAME="DeepSeek-V3"; SHORT=dsv3 ;;
    dsv4fls|ds4fls|dsv4|dsv4flash|DeepSeek-V4-Flash-FP8) MODEL_NAME="DeepSeek-V4-Flash-FP8"; SHORT=dsv4fls ;;
    dsv4pro|dsv4-pro|DeepSeek-V4-Pro-FP8) MODEL_NAME="DeepSeek-V4-Pro-FP8"; SHORT=dsv4pro ;;
    hy3p|Hy3-preview)      MODEL_NAME="Hy3-preview"; SHORT=hy3p ;;
    hy3|Hy3)               MODEL_NAME="Hy3"; SHORT=hy3 ;;
    *) echo "Error: unknown MODEL '$MODEL_IN'" >&2; usage ;;
esac

# --- TOPO → xP yD N EP ---
# Asymmetric P/D is supported by the driver: vllm_disagg.sh computes
# PREFILL_DP_SIZE=xP*8 and DECODE_DP_SIZE=yD*8 independently, and the proxy
# routes on min(P,D) capped by PROXY_ROUTE_DP. Only this case list gated it.
case "$TOPO_IN" in
    1p1d|ep8)  xP=1; yD=1; TOPO=1p1d ;;
    2p2d|ep16) xP=2; yD=2; TOPO=2p2d ;;
    3p3d|ep24) xP=3; yD=3; TOPO=3p3d ;;
    4p4d|ep32) xP=4; yD=4; TOPO=4p4d ;;
    2p3d)      xP=2; yD=3; TOPO=2p3d ;;
    3p2d)      xP=3; yD=2; TOPO=3p2d ;;
    1p2d)      xP=1; yD=2; TOPO=1p2d ;;
    2p1d)      xP=2; yD=1; TOPO=2p1d ;;
    *) echo "Error: unknown TOPO '$TOPO_IN' (2p2d/3p3d/4p4d, asymmetric 1p2d/2p1d/2p3d/3p2d, or ep8/ep16/ep24/ep32)" >&2; usage ;;
esac
N=$((xP + yD))
# Per-role EP, display only — the real DP world is derived in vllm_disagg.sh.
# 2P/2D=16, 4P/4D=32. Not N*8. Asymmetric reports both roles.
EP_P=$((xP * 8))
EP_D=$((yD * 8))
if [[ "$EP_P" == "$EP_D" ]]; then EP="$EP_P"; else EP="P${EP_P}/D${EP_D}"; fi

# --- BENCH ---
case "$BENCH" in
    niah)
        BENCHMARK_SCRIPT=niah
        # Always the full ladder. Rungs are sequential: a 400/500 on 8k
        # is the failure point. Do not cap Pro to 2000 (that forced reruns).
        NIAH_WORDS="${NIAH_WORDS:-2000,8000,16000,20000,28000,35000}"
        NIAH_SEEDS="${NIAH_SEEDS:-0}"
        NIAH_HALT_ON_FAIL="${NIAH_HALT_ON_FAIL:-0}"
        NIAH_PAIR_SLEEP_S="${NIAH_PAIR_SLEEP_S:-30}"
        NIAH_WARMUP_SCORE_SLEEP_S="${NIAH_WARMUP_SCORE_SLEEP_S:-5}"
        BENCHMARK_COMBINATIONS=""
        ;;
    smoke)
        BENCHMARK_SCRIPT=sweep
        BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-1024/1024}"
        NIAH_WORDS=""
        # 231208/226360: con=64+ stalls with 32 decode slots (PROXY_ROUTE_DP=8 ×
        # max-num-seqs=4). Stop the DSV4 smoke ladder there until slots grow.
        # 231209 Pro 4P/4D: skip con=1 — eight serial 1024-token decodes at
        # 2.66 s/tok plus ~2.5 h load overrun the 8 h assoc wall.
        if [[ "$SHORT" == "dsv4pro" && "$TOPO" == "4p4d" ]]; then
            BENCHMARK_CON="${BENCHMARK_CON:-8 16 32}"
            STEP_SEC_PER_TOK="${STEP_SEC_PER_TOK:-3}"
            STEP_TIMEOUT="${STEP_TIMEOUT:-7200}"
        elif [[ "$SHORT" == "dsv4fls" || "$SHORT" == "dsv4pro" ]]; then
            BENCHMARK_CON="${BENCHMARK_CON:-1 8 16 32}"
        fi
        ;;
    validate)
        # One allocation: curl (moriio.sh) then NIAH then smoke. Do not halt
        # NIAH on a failed rung — smoke still has to run.
        BENCHMARK_SCRIPT=validate
        NIAH_WORDS="${NIAH_WORDS:-2000,8000,16000,20000,28000,35000}"
        NIAH_SEEDS="${NIAH_SEEDS:-0}"
        NIAH_HALT_ON_FAIL="${NIAH_HALT_ON_FAIL:-0}"
        NIAH_PAIR_SLEEP_S="${NIAH_PAIR_SLEEP_S:-30}"
        NIAH_WARMUP_SCORE_SLEEP_S="${NIAH_WARMUP_SCORE_SLEEP_S:-5}"
        BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS:-1024/1024}"
        if [[ "$SHORT" == "dsv4pro" && "$TOPO" == "4p4d" ]]; then
            BENCHMARK_CON="${BENCHMARK_CON:-8 16 32}"
            STEP_SEC_PER_TOK="${STEP_SEC_PER_TOK:-3}"
            STEP_TIMEOUT="${STEP_TIMEOUT:-7200}"
        elif [[ "$SHORT" == "dsv4fls" || "$SHORT" == "dsv4pro" ]]; then
            BENCHMARK_CON="${BENCHMARK_CON:-1 8 16 32}"
        fi
        ;;
    mrcr)
        # OpenAI MRCR. Flash first, 2-needle, bins that fit 40k. No prime.
        # Full 100/bin will not fit a 4h PD job — default PER_BIN=8.
        if [[ "$SHORT" != "dsv4fls" && "$SHORT" != "dsv4pro" ]]; then
            echo "Error: mrcr is DSV4-only (run Flash, then Pro if MMR looks sane)." >&2
            exit 1
        fi
        BENCHMARK_SCRIPT=mrcr
        MRCR_NEEDLES="${MRCR_NEEDLES:-2}"
        MRCR_BINS="${MRCR_BINS:-8192,16384,32768}"
        MRCR_PER_BIN="${MRCR_PER_BIN:-8}"
        MRCR_MAXTOK="${MRCR_MAXTOK:-1024}"
        MRCR_TIMEOUT="${MRCR_TIMEOUT:-1800}"
        MRCR_MAX_CTX="${MRCR_MAX_CTX:-40960}"
        MRCR_THINKING="${MRCR_THINKING:-chat}"
        MRCR_HALT_ON_FAIL="${MRCR_HALT_ON_FAIL:-0}"
        BENCHMARK_COMBINATIONS=""
        # A leftover NIAH_LIST_PRIME from a previous submit would zero MMR.
        unset NIAH_LIST_PRIME || true
        ;;
    *) echo "Error: unknown BENCH '$BENCH' (niah|smoke|mrcr|validate)" >&2; usage ;;
esac

# Walltime scales with node count (AITER JIT). validate uses niah's budget
# (load once, then niah + smoke).
ASSOC_MAX_WALL="${ASSOC_MAX_WALL:-08:00:00}"
WT_BENCH="$BENCH"
[[ "$BENCH" == "validate" ]] && WT_BENCH=niah
if [[ -z "$TIME_ARG" ]]; then
    case "$WT_BENCH-$TOPO" in
        niah-1p1d)  TIME_ARG=06:00:00 ;;
        niah-2p2d)  TIME_ARG=08:00:00 ;;
        niah-3p3d)  TIME_ARG=10:00:00 ;;
        niah-4p4d)  TIME_ARG=12:00:00 ;;
        mrcr-1p1d)  TIME_ARG=04:00:00 ;;
        mrcr-2p2d)  TIME_ARG=04:00:00 ;;
        mrcr-3p3d)  TIME_ARG=06:00:00 ;;
        mrcr-4p4d)  TIME_ARG=08:00:00 ;;
        smoke-1p1d) TIME_ARG=04:00:00 ;;
        smoke-2p2d) TIME_ARG=06:00:00 ;;
        smoke-3p3d) TIME_ARG=07:00:00 ;;
        smoke-4p4d) TIME_ARG=08:00:00 ;;
    esac
    # Flash ~69k shards; 4h is tight for first load + JIT. Pro is 865 GB.
    # 2P/2D is an EP16 MoRI decode probe (PROXY_ROUTE_DP=8). Not an ITL score.
    if [[ "$SHORT" == "dsv4fls" ]]; then
        case "$WT_BENCH-$TOPO" in
            smoke-1p1d) TIME_ARG=06:00:00 ;;
            smoke-2p2d) TIME_ARG=08:00:00 ;;
            niah-1p1d|niah-2p2d) TIME_ARG=12:00:00 ;;
            mrcr-1p1d|mrcr-2p2d) TIME_ARG=04:00:00 ;;
        esac
    fi
    if [[ "$SHORT" == "dsv4pro" ]]; then
        case "$WT_BENCH-$TOPO" in
            smoke-1p1d) TIME_ARG=08:00:00 ;;
            smoke-2p2d) TIME_ARG=10:00:00 ;;
            niah-1p1d|niah-2p2d|niah-4p4d) TIME_ARG=12:00:00 ;;
            mrcr-1p1d|mrcr-2p2d|mrcr-4p4d) TIME_ARG=04:00:00 ;;
        esac
    fi
    # Asymmetric topologies match no case arm above and would leave TIME_ARG
    # empty, which sbatch takes as no limit request. Fall back on node count.
    if [[ -z "$TIME_ARG" ]]; then
        case "$WT_BENCH" in
            niah)  TIME_ARG=$(printf '%02d:00:00' $(( N < 6 ? 6 : 8 )) ) ;;
            mrcr)  TIME_ARG=04:00:00 ;;
            smoke) TIME_ARG=$(printf '%02d:00:00' $(( N < 6 ? 4 : 6 )) ) ;;
        esac
        echo "note: $BENCH-$TOPO has no walltime default; using $TIME_ARG for N=$N" >&2
    fi
    # amd-rccl assoc MaxWall is 08:00:00 (sacctmgr show assoc user=$USER). The
    # Pro/Flash bumps above ask for 10-12h, and slurm does not reject that at
    # submit — it queues forever as AssocMaxWallDurationPerJobLimit. 224069 and
    # 223834 both died that way. Clamp the computed default; --time still wins.
    if [[ "$TIME_ARG" > "$ASSOC_MAX_WALL" ]]; then
        echo "note: default TIME=$TIME_ARG exceeds assoc MaxWall; clamping to $ASSOC_MAX_WALL" >&2
        TIME_ARG="$ASSOC_MAX_WALL"
    fi
elif [[ "$TIME_ARG" > "$ASSOC_MAX_WALL" ]]; then
    echo "WARN: --time $TIME_ARG exceeds assoc MaxWall $ASSOC_MAX_WALL; slurm will hold this job as AssocMaxWallDurationPerJobLimit" >&2
fi

# Rank pin: head local DP is always 8. Without this, 2P/2D after curl and all
# 4P/4D walk to rank 8+ (kv=null) while children are headless. 1P/1D world is
# 8 so this is a no-op. MORIIO_CHILD_HTTP=1 gives each child a real connector
# — leave PROXY_ROUTE_DP=0 so ranks 8+ POST to that pod.
if [[ "${MORIIO_CHILD_HTTP:-0}" == "1" ]]; then
    PROXY_ROUTE_DP="${PROXY_ROUTE_DP:-0}"
    # Child binds :8405; list that pod IP in remote_hosts or WRITE hangs
    # dialing the master's handshake from a rank that lives on the child.
    PROXY_HANDSHAKE_PER_POD="${PROXY_HANDSHAKE_PER_POD:-1}"
else
    PROXY_ROUTE_DP="${PROXY_ROUTE_DP:-8}"
fi
PROXY_MAX_CONCURRENCY="${PROXY_MAX_CONCURRENCY:-512}"
# 220693 Python PR-176: omit --moriio-dp-size, ping 36367, cap 512.
# vllm_router 2P2D KV-notify wants --moriio-dp-size when the binary has it.
# ROUTER_SKIP_MORIIO_DP_SIZE=1 omits the flag (toy-proxy default).
if [[ "${PROXY_TYPE:-moriio_toy}" == "vllm_router" ]]; then
    ROUTER_SKIP_MORIIO_DP_SIZE="${ROUTER_SKIP_MORIIO_DP_SIZE:-0}"
else
    ROUTER_SKIP_MORIIO_DP_SIZE="${ROUTER_SKIP_MORIIO_DP_SIZE:-1}"
fi
MORI_PROXY_PING_PORT="${MORI_PROXY_PING_PORT:-36367}"

# --- IMAGE ---
case "$MODEL_NAME" in
    DeepSeek-V4-Flash-FP8|DeepSeek-V4-Pro-FP8) DEFAULT_IMAGE="$IMG_V0280"; IMG_TAG=v0280 ;;
    GLM-*) DEFAULT_IMAGE="$IMG_E03"; IMG_TAG=e03 ;;
    *)     DEFAULT_IMAGE="$IMG_026"; IMG_TAG=026 ;;
esac
case "${IMAGE_ARG}" in
    "")    DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-$DEFAULT_IMAGE}" ;;
    e03)   DOCKER_IMAGE_NAME="$IMG_E03"; IMG_TAG=e03 ;;
    e03tk|e03-tk) DOCKER_IMAGE_NAME="$IMG_E03_TK"; IMG_TAG=e03tk ;;
    5a4c|5a4c8d99) DOCKER_IMAGE_NAME="$IMG_5A4C"; IMG_TAG=5a4c ;;
    mori624002|mori624|624002c8) DOCKER_IMAGE_NAME="$IMG_MORI624"; IMG_TAG=mori624 ;;
    v0280|v028|v0.28.0|mori6fcf|6fcf6b3) DOCKER_IMAGE_NAME="$IMG_V0280"; IMG_TAG=v0280 ;;
    v0290|v029|v0.29.0|10f8874|mori07bdace) DOCKER_IMAGE_NAME="$IMG_V0290"; IMG_TAG=v0290 ;;
    d626fd|fd031|d626|d626108b|1d872fa) DOCKER_IMAGE_NAME="$IMG_D626FD"; IMG_TAG=d626 ;;
    026|20260815) DOCKER_IMAGE_NAME="$IMG_026"; IMG_TAG=026 ;;
    *)     DOCKER_IMAGE_NAME="$IMAGE_ARG"; IMG_TAG=custom ;;
esac
# Classify from the resolved name when the alias did not already pin a tag
# (empty --image + DOCKER_IMAGE_NAME=..., or a full registry tag). Check
# d626 / 5a4c8d99 before e03fa6040 / *-tk so hybrid tags are not e03/e03tk.
if [[ -z "$IMAGE_ARG" || "$IMG_TAG" == "custom" ]]; then
    # v0280 / mori624002 first: both tags also contain fd031 / 1d872fa (and the
    # mori624002 one contains d626108b), so the d626 arm below would swallow
    # them and hide the vLLM / MoRI bump in JOB_NAME.
    if [[ "$DOCKER_IMAGE_NAME" == *v0290* || "$DOCKER_IMAGE_NAME" == *mori07bdace* || "$DOCKER_IMAGE_NAME" == *10f8874* ]]; then
        IMG_TAG=v0290
    elif [[ "$DOCKER_IMAGE_NAME" == *v0280* || "$DOCKER_IMAGE_NAME" == *mori6fcf6b3* ]]; then
        IMG_TAG=v0280
    elif [[ "$DOCKER_IMAGE_NAME" == *mori624002* ]]; then
        IMG_TAG=mori624
    elif [[ "$DOCKER_IMAGE_NAME" == *fd031* || "$DOCKER_IMAGE_NAME" == *d626108b* || "$DOCKER_IMAGE_NAME" == *1d872fa* ]]; then
        IMG_TAG=d626
    elif [[ "$DOCKER_IMAGE_NAME" == *5a4c8d99* ]]; then
        IMG_TAG=5a4c
    elif [[ "$DOCKER_IMAGE_NAME" == *-tk* || "$DOCKER_IMAGE_NAME" == *triton-kernels* ]]; then
        IMG_TAG=e03tk
    elif [[ "$DOCKER_IMAGE_NAME" == *e03fa6040* || "$DOCKER_IMAGE_NAME" == *acb0f1dc* ]]; then
        IMG_TAG=e03
    fi
fi

JOB_NAME="${JOB_NAME_ARG:-${SHORT}-${IMG_TAG}-${BENCH}-${TOPO}}"
# Site submit (2026-09-09): partition amd-arad-burst, account amd-rccl-guest.
# amd-rccl + amd-rccl-guest is rejected (Invalid account or account/partition).
PARTITION="${PARTITION:-amd-arad-burst}"
ACCOUNT="${ACCOUNT:-amd-rccl-guest}"
QOS="${QOS:-}"

# DSV3 on e03: fp8 KV + disable AITER fp8 BMM (216336 boot-die; 216636/217241).
# GLM persist-gate stays unset (moriio.sh default GATE=0).
# DSV3 EP32 217241/217303: 16 GiB static heap was full when dispatch_combine
# asked 7.5 GiB. 32 GiB bytes; docker -e only if set on the submit host.
# Flash: same heap + e03 BMM off. Do NOT set AITER_MLA=0 (V4 sparse needs it).
# 217457: NVIDIA --kv-cache-dtype fp8 remaps to fp8_ds_mla; ROCm MLA backends
# reject it. Use fp8_e4m3 (MI325X Flash recipe). models.yaml must match.
unset GLM_SKIP_PATCHERS || true
EXTRA_ENV=()
if [[ "$MODEL_NAME" == "DeepSeek-V3" ]]; then
    EXTRA_ENV+=(KV_CACHE_DTYPE=fp8 VLLM_ROCM_USE_AITER_MLA=0)
    EXTRA_ENV+=(MORI_SHMEM_HEAP_SIZE=34359738368)
    if [[ "$IMG_TAG" == "e03" || "$IMG_TAG" == "e03tk" || "$IMG_TAG" == "5a4c" || "$IMG_TAG" == "d626" || "$IMG_TAG" == "mori624" ]]; then
        EXTRA_ENV+=(VLLM_ROCM_USE_AITER_FP8BMM=false)
    fi
fi
if [[ "$MODEL_NAME" == "DeepSeek-V4-Flash-FP8" ]]; then
    EXTRA_ENV+=(KV_CACHE_DTYPE=fp8_e4m3 VLLM_ROCM_USE_AITER_MLA=1)
    EXTRA_ENV+=(MORI_SHMEM_HEAP_SIZE=34359738368)
    # Product default HMA=0 (yaml). Set DSV4_ENABLE_HMA=1 only for extra-cache A/B.
    EXTRA_ENV+=(DSV4_ENABLE_HMA="${DSV4_ENABLE_HMA:-0}")
    EXTRA_ENV+=(DSV4_SKIP_INDEXER_REGISTER="${DSV4_SKIP_INDEXER_REGISTER:-0}")
    # 218040: .attn (group-0 sliding_window=None) was never WRITTEN, leaving
    # decode a 128-token window. 1 = transfer it. Default 0 reproduces 218040.
    # HMA=1 owns wait_for_save, so this patcher is skipped at runtime.
    # HMA=0 control must WRITE .attn (218118). HMA=1 skips that patcher.
    if [[ "${DSV4_ENABLE_HMA:-0}" == "0" ]]; then
        EXTRA_ENV+=(DSV4_TRANSFER_ATTN="${DSV4_TRANSFER_ATTN:-1}")
    else
        EXTRA_ENV+=(DSV4_TRANSFER_ATTN="${DSV4_TRANSFER_ATTN:-0}")
    fi
    EXTRA_ENV+=(DSV4_HMA_UPSTREAM_GEOM="${DSV4_HMA_UPSTREAM_GEOM:-1}")
    # Per-page block_len + the region_len span floor. Set together: PAGE alone is
    # the 218257 garbage cell (region_len under-registers) and moriio.sh refuses
    # that combination. Together they aim at correct-and-minimal HMA=1.
    EXTRA_ENV+=(DSV4_HMA_PAGE_BLOCK_LEN="${DSV4_HMA_PAGE_BLOCK_LEN:-0}")
    # 223633: CLIP remainder did not make extra-cache visible. Native slice
    # for .attn only (HMA=0 TRANSFER_ATTN copy). Not PAGE_BLOCK_LEN (218257).
    # Forces REGION_LEN_SPAN so .attn offsets stay inside the MR.
    EXTRA_ENV+=(DSV4_HMA_NATIVE_ATTN="${DSV4_HMA_NATIVE_ATTN:-0}")
    if [[ "${DSV4_HMA_NATIVE_ATTN:-0}" == "1" ]]; then
        DSV4_REGION_LEN_SPAN=1
    fi
    EXTRA_ENV+=(DSV4_REGION_LEN_SPAN="${DSV4_REGION_LEN_SPAN:-0}")
    # 223559: whole-block copy from view origin S>0 stomps dest page P+1.
    # CLIP_PAGE subtracts S from block_len (not PAGE_BLOCK_LEN / 218257).
    EXTRA_ENV+=(DSV4_HMA_CLIP_PAGE="${DSV4_HMA_CLIP_PAGE:-0}")
    # vllm#48989 non-contiguous MR: N*stride[0]*es vs tight view bbox.
    # Default 0 keeps 217546/218443. HMA=1 GEOM=1 full-block copies want 1.
    EXTRA_ENV+=(DSV4_STORAGE_UPSTREAM_SPAN="${DSV4_STORAGE_UPSTREAM_SPAN:-0}")
    # 223392: yaml decode is FULL_DECODE_ONLY. DSV4_EAGER=1 splits CUDA-graph
    # capture vs compress_ratio: DECODE_CUDAGRAPH_MODE=NONE with +quant_fp8.
    # NEVER --enforce-eager (AITER aiter_tensor_t crash). Submit-time wins yaml.
    EXTRA_ENV+=(DSV4_EAGER="${DSV4_EAGER:-0}")
    if [[ "${DSV4_EAGER:-0}" == "1" ]]; then
        EXTRA_ENV+=(DECODE_CUDAGRAPH_MODE=NONE)
    elif [[ -n "${DECODE_CUDAGRAPH_MODE:-}" ]]; then
        # Reaches the container via --export=ALL + the slurm's -e either way, but
        # listing it makes the knob visible in the plan/EXTRA line instead of
        # silently absent (that invisibility is the 223837 NIAH_TERSE class).
        EXTRA_ENV+=(DECODE_CUDAGRAPH_MODE="${DECODE_CUDAGRAPH_MODE}")
    fi
    # 218687: HMA last-chunk is 5 groups × page 4 = 20 tokens. Curl (~17) writes;
    # 2k NIAH never schedules unless this is on. HMA=0 keeps the flat-id path.
    if [[ "${DSV4_ENABLE_HMA:-0}" == "0" ]]; then
        EXTRA_ENV+=(DSV4_CHUNK_HMA_FIX="${DSV4_CHUNK_HMA_FIX:-0}")
    else
        EXTRA_ENV+=(DSV4_CHUNK_HMA_FIX="${DSV4_CHUNK_HMA_FIX:-1}")
    fi
    EXTRA_ENV+=(DSV4_DP_PROBE="${DSV4_DP_PROBE:-0}")
    EXTRA_ENV+=(VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}")
    EXTRA_ENV+=(PROXY_LOG_LEVEL="${PROXY_LOG_LEVEL:-DEBUG}")
    EXTRA_ENV+=(CURL_SUITE=short)
    # 36367 is in ip_local_port_range; a DP worker can steal it (218629).
    EXTRA_ENV+=(MORI_PROXY_PING_PORT="${MORI_PROXY_PING_PORT:-36367}")
    if [[ "$IMG_TAG" == "e03" || "$IMG_TAG" == "e03tk" || "$IMG_TAG" == "5a4c" || "$IMG_TAG" == "d626" || "$IMG_TAG" == "mori624" ]]; then
        EXTRA_ENV+=(VLLM_ROCM_USE_AITER_FP8BMM=false)
    fi
    # Product NIAH: original 6-rung sizes, 64-token answers, bare prompt
    # (no <User>/<Assistant> wrap). Do not inherit NIAH_DS_WRAP from the
    # submit environment — that silently wrapped 218099/218112.
    if [[ "$BENCH" == "niah" || "$BENCH" == "validate" ]]; then
        # 27 Aug: hybrid is the standard method — PR-176's system+user through
        # the DeepSeek template PLUS a restatement after the haystack. Faithful
        # pr176 puts the ask before the haystack only, which collapses on DSV4
        # at 8k+ (224142 Pro: 0/10 x3, 512 tokens of filler); keep it available
        # for cross-track comparison, not for scoring. method=product restores
        # the bare 64-token stem, comparable ONLY to the old product rows
        # (GLM 217197 / DSV3 216165,217404 / Hy3 217411).
        # MAXTOK 512 rather than PR-176's 2048 because DSV4 runs ~1 s/tok and
        # 2048 x 18 requests overruns the 8h assoc walltime — a cap, not a target.
        NIAH_METHOD="${NIAH_METHOD:-hybrid}"
        if [[ "$NIAH_METHOD" == "product" ]]; then
            NIAH_MAXTOK="${NIAH_MAXTOK:-64}"
        else
            NIAH_MAXTOK="${NIAH_MAXTOK:-512}"
        fi
        NIAH_WARMUP=0
        NIAH_TIMEOUT="${NIAH_TIMEOUT:-1800}"
        # validate must finish the ladder so smoke still has a full NIAH log.
        if [[ "$BENCH" == "validate" ]]; then
            NIAH_HALT_ON_FAIL="${NIAH_HALT_ON_FAIL:-0}"
        else
            NIAH_HALT_ON_FAIL="${NIAH_HALT_ON_FAIL:-1}"
        fi
        EXTRA_ENV+=(NIAH_METHOD="$NIAH_METHOD")
        EXTRA_ENV+=(NIAH_DS_WRAP=0)
        EXTRA_ENV+=(NIAH_TERSE="${NIAH_TERSE:-0}")
        EXTRA_ENV+=(NIAH_STOP_BLANK="${NIAH_STOP_BLANK:-0}")
        EXTRA_ENV+=(NIAH_MIN_TOKENS="${NIAH_MIN_TOKENS:-0}")
        # Unset = byte-identical to 218778/224843. Empty still listed so the
        # plan/EXTRA line cannot hide them (223837 NIAH_TERSE class).
        EXTRA_ENV+=(NIAH_LIST_PRIME="${NIAH_LIST_PRIME:-}")
        EXTRA_ENV+=(NIAH_STOP="${NIAH_STOP:-}")
        EXTRA_ENV+=(NIAH_LOGPROBS="${NIAH_LOGPROBS:-0}")
    fi
fi
# 217723: Pro weights 125 GiB/GPU; 32 GiB heap left ~8 GiB for a 9 GiB
# profiling dummy (seqs*len*hidden). 16 GiB heap. yaml has the rest.
if [[ "$MODEL_NAME" == "DeepSeek-V4-Pro-FP8" ]]; then
    EXTRA_ENV+=(KV_CACHE_DTYPE=fp8_e4m3 VLLM_ROCM_USE_AITER_MLA=1)
    EXTRA_ENV+=(MORI_SHMEM_HEAP_SIZE=17179869184)
    EXTRA_ENV+=(DSV4_ENABLE_HMA="${DSV4_ENABLE_HMA:-0}")
    EXTRA_ENV+=(DSV4_SKIP_INDEXER_REGISTER="${DSV4_SKIP_INDEXER_REGISTER:-0}")
    if [[ "${DSV4_ENABLE_HMA:-0}" == "0" ]]; then
        EXTRA_ENV+=(DSV4_TRANSFER_ATTN="${DSV4_TRANSFER_ATTN:-1}")
    else
        EXTRA_ENV+=(DSV4_TRANSFER_ATTN="${DSV4_TRANSFER_ATTN:-0}")
    fi
    EXTRA_ENV+=(DSV4_HMA_UPSTREAM_GEOM="${DSV4_HMA_UPSTREAM_GEOM:-1}")
    EXTRA_ENV+=(DSV4_HMA_PAGE_BLOCK_LEN="${DSV4_HMA_PAGE_BLOCK_LEN:-0}")
    # 223633: CLIP remainder did not make extra-cache visible. Native slice
    # for .attn only (HMA=0 TRANSFER_ATTN copy). Not PAGE_BLOCK_LEN (218257).
    # Forces REGION_LEN_SPAN so .attn offsets stay inside the MR.
    EXTRA_ENV+=(DSV4_HMA_NATIVE_ATTN="${DSV4_HMA_NATIVE_ATTN:-0}")
    if [[ "${DSV4_HMA_NATIVE_ATTN:-0}" == "1" ]]; then
        DSV4_REGION_LEN_SPAN=1
    fi
    EXTRA_ENV+=(DSV4_REGION_LEN_SPAN="${DSV4_REGION_LEN_SPAN:-0}")
    # 223559: whole-block copy from view origin S>0 stomps dest page P+1.
    # CLIP_PAGE subtracts S from block_len (not PAGE_BLOCK_LEN / 218257).
    EXTRA_ENV+=(DSV4_HMA_CLIP_PAGE="${DSV4_HMA_CLIP_PAGE:-0}")
    # vllm#48989 non-contiguous MR: N*stride[0]*es vs tight view bbox.
    # Default 0 keeps 217546/218443. HMA=1 GEOM=1 full-block copies want 1.
    EXTRA_ENV+=(DSV4_STORAGE_UPSTREAM_SPAN="${DSV4_STORAGE_UPSTREAM_SPAN:-0}")
    # 223392: yaml decode is FULL_DECODE_ONLY. DSV4_EAGER=1 splits CUDA-graph
    # capture vs compress_ratio: DECODE_CUDAGRAPH_MODE=NONE with +quant_fp8.
    # NEVER --enforce-eager (AITER aiter_tensor_t crash). Submit-time wins yaml.
    EXTRA_ENV+=(DSV4_EAGER="${DSV4_EAGER:-0}")
    if [[ "${DSV4_EAGER:-0}" == "1" ]]; then
        EXTRA_ENV+=(DECODE_CUDAGRAPH_MODE=NONE)
    elif [[ -n "${DECODE_CUDAGRAPH_MODE:-}" ]]; then
        # Reaches the container via --export=ALL + the slurm's -e either way, but
        # listing it makes the knob visible in the plan/EXTRA line instead of
        # silently absent (that invisibility is the 223837 NIAH_TERSE class).
        EXTRA_ENV+=(DECODE_CUDAGRAPH_MODE="${DECODE_CUDAGRAPH_MODE}")
    fi
    if [[ "${DSV4_ENABLE_HMA:-0}" == "0" ]]; then
        EXTRA_ENV+=(DSV4_CHUNK_HMA_FIX="${DSV4_CHUNK_HMA_FIX:-0}")
    else
        EXTRA_ENV+=(DSV4_CHUNK_HMA_FIX="${DSV4_CHUNK_HMA_FIX:-1}")
    fi
    EXTRA_ENV+=(DSV4_DP_PROBE="${DSV4_DP_PROBE:-0}")
    EXTRA_ENV+=(VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}")
    EXTRA_ENV+=(PROXY_LOG_LEVEL="${PROXY_LOG_LEVEL:-DEBUG}")
    EXTRA_ENV+=(CURL_SUITE=short)
    # 36367 is in ip_local_port_range; a DP worker can steal it (218629).
    EXTRA_ENV+=(MORI_PROXY_PING_PORT="${MORI_PROXY_PING_PORT:-36367}")
    if [[ "$IMG_TAG" == "e03" || "$IMG_TAG" == "e03tk" || "$IMG_TAG" == "5a4c" || "$IMG_TAG" == "d626" || "$IMG_TAG" == "mori624" ]]; then
        EXTRA_ENV+=(VLLM_ROCM_USE_AITER_FP8BMM=false)
    fi
    if [[ "$BENCH" == "niah" || "$BENCH" == "validate" ]]; then
        # 27 Aug: hybrid is the standard method — PR-176's system+user through
        # the DeepSeek template PLUS a restatement after the haystack. Faithful
        # pr176 puts the ask before the haystack only, which collapses on DSV4
        # at 8k+ (224142 Pro: 0/10 x3, 512 tokens of filler); keep it available
        # for cross-track comparison, not for scoring. method=product restores
        # the bare 64-token stem, comparable ONLY to the old product rows
        # (GLM 217197 / DSV3 216165,217404 / Hy3 217411).
        # MAXTOK 512 rather than PR-176's 2048 because DSV4 runs ~1 s/tok and
        # 2048 x 18 requests overruns the 8h assoc walltime — a cap, not a target.
        NIAH_METHOD="${NIAH_METHOD:-hybrid}"
        if [[ "$NIAH_METHOD" == "product" ]]; then
            NIAH_MAXTOK="${NIAH_MAXTOK:-64}"
        else
            NIAH_MAXTOK="${NIAH_MAXTOK:-512}"
        fi
        NIAH_WARMUP=0
        NIAH_TIMEOUT="${NIAH_TIMEOUT:-1800}"
        # validate must finish the ladder so smoke still has a full NIAH log.
        if [[ "$BENCH" == "validate" ]]; then
            NIAH_HALT_ON_FAIL="${NIAH_HALT_ON_FAIL:-0}"
        else
            NIAH_HALT_ON_FAIL="${NIAH_HALT_ON_FAIL:-1}"
        fi
        EXTRA_ENV+=(NIAH_METHOD="$NIAH_METHOD")
        EXTRA_ENV+=(NIAH_DS_WRAP=0)
        EXTRA_ENV+=(NIAH_TERSE="${NIAH_TERSE:-0}")
        EXTRA_ENV+=(NIAH_STOP_BLANK="${NIAH_STOP_BLANK:-0}")
        EXTRA_ENV+=(NIAH_MIN_TOKENS="${NIAH_MIN_TOKENS:-0}")
        EXTRA_ENV+=(NIAH_LIST_PRIME="${NIAH_LIST_PRIME:-}")
        EXTRA_ENV+=(NIAH_STOP="${NIAH_STOP:-}")
        EXTRA_ENV+=(NIAH_LOGPROBS="${NIAH_LOGPROBS:-0}")
    fi
fi

# v0280/v0290 bake ENV SKIP_RUNTIME_PATCH=1 (image-build skip). That leaks into
# serve and skips Wei combine original-topk (339189: 8192 vs 1024). Force 0
# into docker -e; unset would keep the image ENV.
if [[ "$MODEL_NAME" == "DeepSeek-V4-Flash-FP8" || "$MODEL_NAME" == "DeepSeek-V4-Pro-FP8" ]]; then
    EXTRA_ENV+=(SKIP_RUNTIME_PATCH="${SKIP_RUNTIME_PATCH:-0}")
    # v0290/v0280 do not bake vllm-router. Git+cargo at NODE0 boot (GLM
    # Dockerfile path). pip is ROUTER_BOOT_INSTALL=pip — not "latest main".
    if [[ "${PROXY_TYPE:-moriio_toy}" == "vllm_router" ]]; then
        EXTRA_ENV+=(ROUTER_BOOT_INSTALL="${ROUTER_BOOT_INSTALL:-git}")
        EXTRA_ENV+=(ROUTER_REPO="${ROUTER_REPO:-https://github.com/vllm-project/router.git}")
        EXTRA_ENV+=(ROUTER_REF="${ROUTER_REF:-main}")
        EXTRA_ENV+=(RUST_TOOLCHAIN="${RUST_TOOLCHAIN:-1.88.0}")
    fi
fi

if [[ "$BENCH" == "mrcr" ]]; then
    EXTRA_ENV+=(MRCR_NEEDLES="${MRCR_NEEDLES:-2}")
    EXTRA_ENV+=(MRCR_BINS="${MRCR_BINS:-8192,16384,32768}")
    EXTRA_ENV+=(MRCR_PER_BIN="${MRCR_PER_BIN:-8}")
    EXTRA_ENV+=(MRCR_MAXTOK="${MRCR_MAXTOK:-1024}")
    EXTRA_ENV+=(MRCR_TIMEOUT="${MRCR_TIMEOUT:-1800}")
    EXTRA_ENV+=(MRCR_MAX_CTX="${MRCR_MAX_CTX:-40960}")
    EXTRA_ENV+=(MRCR_THINKING=chat)
    EXTRA_ENV+=(MRCR_HALT_ON_FAIL="${MRCR_HALT_ON_FAIL:-0}")
    if [[ -n "${MRCR_DATA_DIR:-}" ]]; then
        EXTRA_ENV+=(MRCR_DATA_DIR="$MRCR_DATA_DIR")
    fi
fi

# EP16 NFS+JIT boot exceeds the old 4000s gate (432977 timed out a dead
# decode; 216650 died while decode was still writing). --export=ALL can
# leak LOG_WAIT_TIMEOUT_SECONDS=4000 from the login shell; force unless
# the operator overrides.
LOG_WAIT_TIMEOUT_SECONDS="${LOG_WAIT_TIMEOUT_SECONDS:-10800}"
EXTRA_ENV+=(LOG_WAIT_TIMEOUT_SECONDS="$LOG_WAIT_TIMEOUT_SECONDS")

# glm52: default to NVMe staged pool unless operator passed --nodelist/--exclude.
if [[ "$SHORT" == "glm52" && -z "$NODELIST_ARG" && "$EXCLUDE_EXPLICIT" == "0" ]]; then
    NODELIST_ARG="$GLM52_STAGED"
fi
case "$NODELIST_ARG" in
    glm52|GLM52) NODELIST_ARG="$GLM52_STAGED" ;;
esac

# --nodelist (staged weights) replaces --exclude so the pool is not filtered out.
SBATCH_LOC=()
if [[ -n "$NODELIST_ARG" ]]; then
    SBATCH_LOC+=(--nodelist="$NODELIST_ARG")
elif [[ -n "$EXCLUDE_ARG" ]]; then
    SBATCH_LOC+=(--exclude="$EXCLUDE_ARG")
fi

if [[ ! -f run_xPyD_models.slurm ]]; then
    echo "Error: run_xPyD_models.slurm missing in $SCRIPT_DIR" >&2
    exit 1
fi

unset SLURM_NNODES SLURM_JOB_NUM_NODES SLURM_JOB_NODELIST || true

echo "=== WideEP bench plan ==="
echo "BENCH=$BENCH  MODEL=$MODEL_NAME  TOPO=$TOPO  EP=$EP  xP=$xP yD=$yD  N=$N"
echo "IMAGE=$DOCKER_IMAGE_NAME"
echo "PROXY_TYPE=${PROXY_TYPE:-moriio_toy}  PROXY_ROUTE_DP=$PROXY_ROUTE_DP  SKIP_MORIIO_DP=${ROUTER_SKIP_MORIIO_DP_SIZE}  PING=${MORI_PROXY_PING_PORT}  CONC=${PROXY_MAX_CONCURRENCY}  WIDE_EP=1"
[[ "${PROXY_TYPE:-moriio_toy}" == "vllm_router" ]] && echo "ROUTER_BOOT=${ROUTER_BOOT_INSTALL:-}  REPO=${ROUTER_REPO:-}  REF=${ROUTER_REF:-}"
echo "TIME=$TIME_ARG"
[[ "$BENCH" == "smoke" || "$BENCH" == "validate" ]] && echo "SMOKE CON=${BENCHMARK_CON:-default}  COMBOS=$BENCHMARK_COMBINATIONS  STEP_SEC_PER_TOK=${STEP_SEC_PER_TOK:-}  STEP_TIMEOUT=${STEP_TIMEOUT:-}"
[[ "$BENCH" == "niah" || "$BENCH" == "validate" ]] && echo "NIAH_METHOD=${NIAH_METHOD:-}  NIAH_WORDS=$NIAH_WORDS  NIAH_SEEDS=$NIAH_SEEDS  HALT=$NIAH_HALT_ON_FAIL  MAXTOK=${NIAH_MAXTOK:-}  WARMUP=${NIAH_WARMUP:-}  TIMEOUT=${NIAH_TIMEOUT:-}  WRAP=0  TERSE=${NIAH_TERSE:-0}  STOP_BLANK=${NIAH_STOP_BLANK:-0}  MINTOK=${NIAH_MIN_TOKENS:-0}  PRIME=${NIAH_LIST_PRIME:-}  STOP=${NIAH_STOP:-}  LOGPROBS=${NIAH_LOGPROBS:-0}"
[[ "$BENCH" == "mrcr" ]] && echo "MRCR needles=${MRCR_NEEDLES:-2}  bins=${MRCR_BINS:-}  per_bin=${MRCR_PER_BIN:-}  MAXTOK=${MRCR_MAXTOK:-}  TIMEOUT=${MRCR_TIMEOUT:-}  MAX_CTX=${MRCR_MAX_CTX:-}  THINKING=chat  PRIME=OFF  DATA_DIR=${MRCR_DATA_DIR:-hf}"
[[ ${#EXTRA_ENV[@]} -gt 0 ]] && echo "EXTRA ${EXTRA_ENV[*]}"
echo "JOB_NAME=$JOB_NAME  ACCOUNT=$ACCOUNT  PARTITION=$PARTITION  QOS=${QOS:-default}  ${SBATCH_LOC[*]:-any-node}"
[[ -n "$AFTER_JOB" ]] && echo "DEPENDENCY=afterany:${AFTER_JOB}"
echo "========================="

SBATCH_EXTRA=()
if [[ -n "$AFTER_JOB" ]]; then
    SBATCH_EXTRA+=(--dependency="afterany:${AFTER_JOB}")
fi

SBATCH_CMD=(sbatch -A "$ACCOUNT")
if [[ -n "$QOS" ]]; then
    SBATCH_CMD+=(--qos="$QOS")
fi
SBATCH_CMD+=(-p "$PARTITION" -N "$N" -n "$N" --ntasks-per-node=1 --gres=gpu:8 --time="$TIME_ARG"
    "${SBATCH_LOC[@]}" "${SBATCH_EXTRA[@]}"
    --job-name="$JOB_NAME" --export=ALL
    run_xPyD_models.slurm)
echo "SBATCH: ${SBATCH_CMD[*]}"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] not submitting"
    exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Error: SLURM_JOB_ID is set (${SLURM_JOB_ID}). Submit from a login node, not inside a job." >&2
    exit 1
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "Error: sbatch not found. Run this on useocpslog-002, not WSL." >&2
    exit 1
fi

export DOCKER_IMAGE_NAME PYTHONUNBUFFERED=1
export PROXY_TYPE="${PROXY_TYPE:-moriio_toy}"
export CONNECTOR=moriio WIDE_EP=1
export xP yD MODEL_NAME BENCHMARK_SCRIPT
export PROXY_ROUTE_DP
export PROXY_MAX_CONCURRENCY="${PROXY_MAX_CONCURRENCY:-512}"
export MORIIO_CHILD_HTTP="${MORIIO_CHILD_HTTP:-0}"
export PROXY_HANDSHAKE_PER_POD="${PROXY_HANDSHAKE_PER_POD:-}"
export ROUTER_SKIP_MORIIO_DP_SIZE
[[ -n "${ROUTER_BOOT_INSTALL:-}" ]] && export ROUTER_BOOT_INSTALL
[[ -n "${ROUTER_REPO:-}" ]] && export ROUTER_REPO
[[ -n "${ROUTER_REF:-}" ]] && export ROUTER_REF
[[ -n "${RUST_TOOLCHAIN:-}" ]] && export RUST_TOOLCHAIN
# 220693: 36367. 25000 dodges the ephemeral-port steal (218125/218144) if ping fails.
export MORI_PROXY_PING_PORT
if [[ "$BENCH" == "smoke" || "$BENCH" == "validate" ]]; then
    export BENCHMARK_COMBINATIONS
    [[ -n "${BENCHMARK_CON:-}" ]] && export BENCHMARK_CON
    [[ -n "${STEP_SEC_PER_TOK:-}" ]] && export STEP_SEC_PER_TOK
    [[ -n "${STEP_TIMEOUT:-}" ]] && export STEP_TIMEOUT
fi
if [[ "$BENCH" == "niah" || "$BENCH" == "validate" ]]; then
    export NIAH_WORDS NIAH_SEEDS NIAH_HALT_ON_FAIL
    export NIAH_PAIR_SLEEP_S NIAH_WARMUP_SCORE_SLEEP_S
    [[ -n "${NIAH_METHOD:-}" ]] && export NIAH_METHOD
    [[ -n "${NIAH_MAXTOK:-}" ]] && export NIAH_MAXTOK
    [[ -n "${NIAH_WARMUP:-}" ]] && export NIAH_WARMUP
    [[ -n "${NIAH_TIMEOUT:-}" ]] && export NIAH_TIMEOUT
    [[ -n "${NIAH_TERSE:-}" ]] && export NIAH_TERSE
    [[ -n "${NIAH_STOP_BLANK:-}" ]] && export NIAH_STOP_BLANK
    [[ -n "${NIAH_MIN_TOKENS:-}" ]] && export NIAH_MIN_TOKENS
    # Always export: empty is a real value (byte-identical product stem).
    # EXTRA_ENV export below also sets them; this covers niah on non-DSV4 models.
    export NIAH_LIST_PRIME="${NIAH_LIST_PRIME:-}"
    export NIAH_STOP="${NIAH_STOP:-}"
    export NIAH_LOGPROBS="${NIAH_LOGPROBS:-0}"
fi
if [[ "$BENCH" == "mrcr" ]]; then
    export MRCR_NEEDLES MRCR_BINS MRCR_PER_BIN MRCR_MAXTOK MRCR_TIMEOUT
    export MRCR_MAX_CTX MRCR_THINKING MRCR_HALT_ON_FAIL
    [[ -n "${MRCR_DATA_DIR:-}" ]] && export MRCR_DATA_DIR
fi
if [[ ${#EXTRA_ENV[@]} -gt 0 ]]; then
    export "${EXTRA_ENV[@]}"
fi

"${SBATCH_CMD[@]}"
