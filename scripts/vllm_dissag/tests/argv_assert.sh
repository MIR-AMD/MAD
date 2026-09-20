#!/bin/bash
# Offline argv + env assertions for the unified launcher: checks that each connector ×
# WIDE_EP × role cell emits the expected `vllm serve` flags/env (and omits the wrong ones).
# No cluster / no GPUs. Exits 0 if all assertions hold.
#
# Covers:
#   - moriio+TP (Llama): exactly ONE --compilation-config, has --disable-custom-all-reduce,
#     has --tensor-parallel-size, NO -tp 1 / --enable-expert-parallel / --all2all-backend
#   - moriio+wideEP (DSV3): -tp 1 + --data-parallel-size + --enable-expert-parallel +
#     --all2all-backend mori_high_throughput + --block-size 16, exactly ONE --compilation-config
#   - slurm docker -e forwards the RDMA-fix env (expandable_segments:False x2, IPC_MODE_LEGACY=0)
set -u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLURM="$DIR/run_xPyD_models.slurm"
pass=0; fail=0

# emit argv for a cell
_argv() { # connector wide_ep ep_backend model model_path
  env -i PATH="$PATH" HOME="$HOME" NIXL_COOKBOOK_PATH="$DIR" \
    DRY_RUN=1 NODE_RANK=0 xP=1 yD=1 CONNECTOR="$1" WIDE_EP="$2" EP_BACKEND="$3" \
    MODEL_NAME="$4" MODEL_PATH="$5" MASTER_ADDR=10.0.0.1 IPADDRS=10.0.0.1,10.0.0.2 \
    GPUS_PER_NODE=8 SLURM_JOB_ID=ASSERT PROXY_TYPE=vllm_router ROUTER_PORT=30000 \
    bash "$DIR/vllm_disagg.sh" 2>/dev/null | awk '/^===DRYRUN/{f=1;next} /^===END===/{f=0} f'
}

_has()   { grep -qF -- "$2" <<<"$1" && { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); } || { printf "  FAIL  %s (missing: %s)\n" "$3" "$2"; fail=$((fail+1)); }; }
_hasnot(){ grep -qF -- "$2" <<<"$1" && { printf "  FAIL  %s (unexpected: %s)\n" "$3" "$2"; fail=$((fail+1)); } || { printf "  PASS  %s\n" "$3"; pass=$((pass+1)); }; }
_count() { local n; n="$(grep -cF -- "$2" <<<"$1")"; [[ "$n" == "$3" ]] && { printf "  PASS  %s (=%s)\n" "$4" "$n"; pass=$((pass+1)); } || { printf "  FAIL  %s (got %s want %s)\n" "$4" "$n" "$3"; fail=$((fail+1)); }; }

echo "=== moriio + TP (Llama-70B) ==="
A="$(_argv moriio 0 '' amd-Llama-3.3-70B-Instruct-FP8-KV /m/Llama)"
_has    "$A" "--tensor-parallel-size" "has --tensor-parallel-size"
_has    "$A" "--disable-custom-all-reduce" "has --disable-custom-all-reduce"
_count  "$A" "--compilation-config" 1 "exactly one --compilation-config"
_hasnot "$A" "--enable-expert-parallel" "no --enable-expert-parallel"
_hasnot "$A" "--all2all-backend" "no --all2all-backend"
_hasnot "$A" "--data-parallel-size" "no --data-parallel-size"

echo ""
echo "=== moriio + wideEP (DeepSeek-V3, EP) ==="
B="$(_argv moriio 1 mori DeepSeek-V3 /m/DSV3)"
_has    "$B" "--enable-expert-parallel" "has --enable-expert-parallel"
_has    "$B" "--data-parallel-size" "has --data-parallel-size"
_has    "$B" "mori_high_throughput" "prefill all2all = mori_high_throughput"
_has    "$B" "--block-size" "has --block-size"
_has    "$B" "16" "block-size value 16 present"
_count  "$B" "--compilation-config" 1 "exactly one --compilation-config"
_hasnot "$B" "--tensor-parallel-size" "no --tensor-parallel-size (uses -tp 1)"

echo ""
echo "=== connector platform env files carry the RDMA-fix env ==="
# The ROCm-7.2.3 GPU-RDMA env now lives in per-connector .env files; the slurm
# sources connectors/<CONNECTOR>.env and forwards each var via docker -e.
S="$(cat "$SLURM")"
_has "$S" 'CONNECTOR_ENV_FILE="${SCRIPT_DIR}/connectors/${CONNECTOR}.env"' "slurm sources connector .env"
_has "$S" '${CONNECTOR_ENV_ARGS}' "slurm forwards CONNECTOR_ENV_ARGS in docker run"
for cf in moriio rixl; do
  F="$DIR/connectors/${cf}.env"
  if [[ -f "$F" ]]; then
    E="$(cat "$F")"
    _has "$E" "PYTORCH_ALLOC_CONF=expandable_segments:False" "${cf}.env: PYTORCH_ALLOC_CONF"
    _has "$E" "PYTORCH_HIP_ALLOC_CONF=expandable_segments:False" "${cf}.env: PYTORCH_HIP_ALLOC_CONF"
    _has "$E" "HSA_ENABLE_IPC_MODE_LEGACY=0" "${cf}.env: IPC_MODE_LEGACY=0"
    _has "$E" "MORI_GPU_ARCHS=gfx942" "${cf}.env: MORI_GPU_ARCHS"
  else
    printf "  FAIL  connectors/%s.env missing\n" "$cf"; fail=$((fail+1))
  fi
done
# parse check: the slurm's KEY=${KEY:-VAL} loop yields correct -e args (+ override wins)
_parse() { # $1=connector ; reads its .env with same logic as the slurm
  local A="" l k v
  while IFS= read -r l; do
    [[ "$l" =~ ^[[:space:]]*# || -z "${l// }" ]] && continue
    k="${l%%=*}"; v="${l#*=}"; A+=" -e ${k}=${!k:-$v}"
  done < "$DIR/connectors/$1.env"
  printf '%s' "$A"
}
_has "$(_parse moriio)" "-e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False" "parse yields HIP_ALLOC -e arg"
_has "$(PYTORCH_HIP_ALLOC_CONF=expandable_segments:True _parse moriio)" "-e PYTORCH_HIP_ALLOC_CONF=expandable_segments:True" "submit-time override wins"

# Per-shape warmup must stay opt-in: it is on the shared sweep path, so a default-on gate
# would change the measured TPOT of every already-validated recipe. GLM opts in via its
# models.yaml env:, and both GLM-only knobs need a docker -e line or they cannot be
# A/B-tested from the submit side (the recipe applies whenever the key is absent).
echo ""
echo "=== per-shape warmup is opt-in, not default-on ==="
B="$(cat "$DIR/benchmark_xPyD.sh")"
_has    "$B" '${SHAPE_WARMUP:-0}' "benchmark_xPyD.sh: warmup gate defaults OFF"
_hasnot "$B" '${SHAPE_WARMUP:-1}' "benchmark_xPyD.sh: gate is not default-on"
_OPTIN="$(python3 - "$DIR/models.yaml" <<'PY'
import sys, yaml
y = yaml.safe_load(open(sys.argv[1])) or {}
optin = [m for m, c in y.items()
         if isinstance(c, dict) and (c.get("env") or {}).get("SHAPE_WARMUP") == "1"]
print("[" + ",".join(sorted(optin)) + "]")
PY
)"
_has "$_OPTIN" "[GLM-5.1-FP8]" "models.yaml: GLM-5.1-FP8 is the ONLY warmup opt-in"
_has "$(cat "$SLURM")" '${SHAPE_WARMUP:+-e SHAPE_WARMUP=' "slurm forwards SHAPE_WARMUP override"
_has "$(cat "$SLURM")" '${USE_INDUCTOR_GRAPH_PARTITION:+-e USE_INDUCTOR_GRAPH_PARTITION=' "slurm forwards IGP override"

echo ""
echo "=== slurm forwards DSV4 MoRI-EP / NIAH knobs ==="
_has "$S" '${DSV4_EAGER:+-e DSV4_EAGER=' "slurm forwards DSV4_EAGER"
_has "$(cat "$DIR/connectors/moriio.sh")" "[dsv4-patch-roster]" "moriio dumps patch roster"
_has "$(cat "$DIR/connectors/moriio.sh")" "_dsv4_mixed_block_size_fix" "HMA=0 mixed-bs is called on product PD"
_has "$(cat "$DIR/connectors/moriio.sh")" "_dsv4_attn_transfer_fix" "HMA=0 attn-xfer is called on product PD"
_has "$S" '-e SKIP_RUNTIME_PATCH=${SKIP_RUNTIME_PATCH:-0}' "slurm forces SKIP_RUNTIME_PATCH=0 (overrides image ENV)"
_has "$S" '${DSV4_TRANSFER_ATTN:+-e DSV4_TRANSFER_ATTN=' "slurm forwards DSV4_TRANSFER_ATTN"

# The decode ITL fix (436486: 308.30 -> 26.80 ms, byte-identical output) is
# worth ~280 ms/token, so every link in its chain gets a static guard. It has
# four, and breaking any one of them fails SILENTLY as "the fix did nothing":
# the patcher must be applied, the knob must reach the container, the product
# default must live where precedence lets it win, and the wrapper must NOT
# default it (the yaml loader skips vars already in the environment, so a
# wrapper default would make the yaml entry dead config).
echo ""
echo "=== MoRI dispatch trim is wired end to end, ON by default, opt-OUT ==="
M="$(cat "$DIR/connectors/moriio.sh")"
_has    "$M" "_mori_trim_dispatch" "moriio calls the trim patcher"
_has    "$M" "apply_mori_trim_dispatch.py" "moriio names the trim patcher file"
_has    "$M" 'trim=${DSV4_PATCH_TRIM' "patch roster reports trim status"
_has    "$S" '${MORI_TRIM_DISPATCH:+-e MORI_TRIM_DISPATCH=' "slurm forwards MORI_TRIM_DISPATCH"
_has    "$S" '${MORI_TRIM_CHECK:+-e MORI_TRIM_CHECK=' "slurm forwards MORI_TRIM_CHECK"
W="$(cat "$DIR/run_wideep_bench.sh")"
_hasnot "$W" 'MORI_TRIM_DISPATCH="${MORI_TRIM_DISPATCH:-0}"' \
        "wrapper does NOT default trim (the connector owns the default)"
_has    "$W" 'MORI_TRIM=${MORI_TRIM_DISPATCH:-1 (default)}' "wrapper prints the effective trim source"
# The trim default is ON and lives in the connector. It must NOT be in
# models.yaml: a yaml value wins over an unset env, which is how 437704 served
# a stock cell while its plan line implied the default was applied.
_has    "$M" 'local _want="${MORI_TRIM_DISPATCH:-1}"' "connector defaults trim ON"
_TRIM="$(python3 - "$DIR/models.yaml" <<'PY'
import sys, yaml
y = yaml.safe_load(open(sys.argv[1])) or {}
hits = [m for m, c in sorted(y.items())
        if isinstance(c, dict) and "MORI_TRIM_DISPATCH" in (c.get("env") or {})]
print("models.yaml trim entries: " + (", ".join(hits) if hits else "NONE"))
PY
)"
_has "$_TRIM" "NONE" "models.yaml does not pin trim (would override the ON default)"
# The runtime gate inside the injected code must agree with the shell default.
_has "$(cat "$DIR/apply_mori_trim_dispatch.py")" \
     'os.environ.get("MORI_TRIM_DISPATCH", "1") != "1"' \
     "injected runtime gate defaults trim ON"

# The container-side init section. models.yaml is loaded INSIDE the container by
# vllm_disagg.sh with a bare `import yaml`, twice, and it resolves both the
# per-model env block and the per-role serve flags -- so PyYAML is a critical
# dependency of the serve path, not a reporting nicety. Images are a moving
# target, so the init step must keep covering it. pandas is the reporting half
# (benchmark_parser.py, the only ITL/TTFT/TPOT reporter).
echo ""
echo "=== container creation barrier is bounded (437656) ==="
B="$(cat "$DIR/socket_barrier.py")"
V="$(cat "$DIR/vllm_disagg.sh")"
_has "$B" '"--timeout"' "barrier accepts a deadline"
_has "$B" 'never opened their port' "barrier names the peers that never showed"
_has "$B" 'default=0' "barrier default stays wait-forever for the other callers"
_has "$V" 'CONTAINER_BARRIER_TIMEOUT_SECONDS:-1800' "launcher bounds the barrier"
_has "$V" 'container creation barrier failed' "launcher aborts when the barrier times out"

echo ""
echo "=== container init ensures the python deps the serve path imports ==="
_has "$V" "Initialization — container-side boot setup" "vllm_disagg has an init section"
_has "$V" "_ensure_py_deps" "init defines the dep check"
_has "$V" "_init_container_env" "init has an umbrella hook for future steps"
_has "$V" 'PY_BOOT_DEPS-yaml:PyYAML pandas' "init default covers BOTH PyYAML and pandas"
_has "$V" "import os, yaml, shlex" "models.yaml is still parsed with a bare import yaml"
_has "$V" 'PyYAML absent' "a failed PyYAML install says the serve path is affected"
_has "$V" 'pandas absent is harmless' "a failed pandas install says reporting only"
# Nothing in init may abort: a compute node can be offline from PyPI, and no
# reporting library is worth losing a multi-hour serving cell.
_INIT="$(python3 - "$DIR/vllm_disagg.sh" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
m = re.search(r'^_ensure_py_deps\(\) \{.*?^\}', s, re.S | re.M)
body = m.group(0) if m else ""
print("FOUND" if body else "MISSING",
      "ABORTS" if re.search(r'\b(exit 1|exit 2)\b', body) else "NONFATAL")
PY
)"
_has "$_INIT" "FOUND NONFATAL" "init dep step is non-fatal (no exit on failure)"

_has "$S" '${NIAH_HALT_ON_FAIL:+-e NIAH_HALT_ON_FAIL=' "slurm forwards NIAH_HALT_ON_FAIL"
_has "$S" '${NIAH_LIST_PRIME:+-e NIAH_LIST_PRIME=' "slurm forwards NIAH_LIST_PRIME"
_has "$S" '${MRCR_NEEDLES:+-e MRCR_NEEDLES=' "slurm forwards MRCR_NEEDLES"
_has "$S" 'mrcr)         BENCHMARK_SCRIPT_FILE="benchmark_mrcr.sh"' "slurm maps mrcr -> benchmark_mrcr.sh"
_has "$S" 'validate)     BENCHMARK_SCRIPT_FILE="benchmark_validate.sh"' "slurm maps validate -> benchmark_validate.sh"
_has "$S" '${NIAH_STOP:+-e NIAH_STOP=' "slurm forwards NIAH_STOP"
_has "$S" '${NIAH_LOGPROBS:+-e NIAH_LOGPROBS=' "slurm forwards NIAH_LOGPROBS"
_has "$S" '${PROXY_ROUTE_DP:+-e PROXY_ROUTE_DP=' "slurm forwards PROXY_ROUTE_DP"
_has "$S" '${ROUTER_BOOT_INSTALL:+-e ROUTER_BOOT_INSTALL=' "slurm forwards ROUTER_BOOT_INSTALL"
_has "$S" '${ROUTER_REPO:+-e ROUTER_REPO=' "slurm forwards ROUTER_REPO"
_has "$S" '${ROUTER_REF:+-e ROUTER_REF=' "slurm forwards ROUTER_REF"
W="$(cat "$DIR/run_wideep_bench.sh")"
_has "$W" 'DEFAULT_IMAGE="$IMG_V0290"' "wrapper DSV4 default image is v0290"
_has "$W" 'DSV4_ENABLE_HMA="${DSV4_ENABLE_HMA:-0}"' "wrapper DSV4 product default is HMA=0"
_has "$W" 'unset NIAH_LIST_PRIME' "wrapper mrcr unsets NIAH_LIST_PRIME"
_has "$W" 'NIAH_LIST_PRIME="1."' "wrapper Flash NIAH defaults list-prime 1."
_has "$W" 'Pro NIAH: never prime' "wrapper Pro NIAH forces the unprimed stem"
_has "$W" 'BENCHMARK_SCRIPT=mrcr' "wrapper mrcr sets BENCHMARK_SCRIPT"
_has "$W" 'BENCHMARK_SCRIPT=validate' "wrapper validate sets BENCHMARK_SCRIPT"
_hasnot "$W" 'ROUTER_BOOT_INSTALL="${ROUTER_BOOT_INSTALL:-git}"' "wrapper does not git-boot vllm-router (baked in v0290)"

echo ""
echo "=== moriio + wideEP (DeepSeek-V4-Flash-FP8) ==="
C="$(_argv moriio 1 mori DeepSeek-V4-Flash-FP8 /m/DSV4F)"
_has    "$C" "--enable-expert-parallel" "DSV4 has --enable-expert-parallel"
_has    "$C" "--data-parallel-size" "DSV4 has --data-parallel-size"
_has    "$C" "deepseek_v4" "DSV4 tokenizer-mode deepseek_v4"
_has    "$C" "--block-size" "DSV4 has --block-size"
_has    "$C" "256" "DSV4 block-size 256 present"
_count  "$C" "--compilation-config" 1 "DSV4 exactly one --compilation-config"
_hasnot "$C" "--tensor-parallel-size" "DSV4 no --tensor-parallel-size (uses -tp 1)"

echo ""
echo "=== shared NIAH path keeps develop's behaviour for non-DSV4 models ==="
# These four defaults are never read on a DSV4 run (the wrapper sets NIAH_METHOD
# and NIAH_MAXTOK explicitly in both model blocks), so they exist purely to leave
# GLM / DSV3 / Hy3 on the behaviour they had before this branch.
NSH="$(cat "${DIR}/benchmark_niah.sh")"
NPY="$(cat "${DIR}/benchmark_niah.py")"
_has "$NPY" 'os.environ.get("NIAH_METHOD", "product")' "niah.py defaults to the model-agnostic product stem"
_has "$NSH" 'NIAH_MAXTOK="${NIAH_MAXTOK:-2048}"' "niah.sh keeps develop's 2048 answer budget"
_has "$NSH" '/v1/models' "niah.sh falls back to the /v1/models readiness probe"
_has "$NSH" 'WARN: neither /ready' "niah.sh readiness failure is non-fatal"
_has "$NSH" 'parse_to_csv.py' "niah.sh still emits madengine perf.csv rows"
_has "$W" 'NIAH_METHOD="${NIAH_METHOD:-hybrid}"' "wrapper sets DSV4 method explicitly"
_has "$W" 'NIAH_MAXTOK="${NIAH_MAXTOK:-512}"' "wrapper sets DSV4 maxtok explicitly"

echo ""
echo "======================================================"
echo "  argv_assert: ${pass} passed, ${fail} failed"
echo "======================================================"
[[ "$fail" == "0" ]]
