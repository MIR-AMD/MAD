#!/bin/bash
# Connector profile: MoRIIO (MoRIIOConnector KV transfer).
# =============================================================================
# Sourced by vllm_disagg.sh. Connector hook contract:
#   connector_init, connector_setup_env, connector_runtime_patch,
#   connector_launch_worker, connector_wait_workers_ready, connector_start_proxy
#
# WIDE_EP=1 (MoriEP wideEP): byte-identical to legacy vllm_disagg_mori_ep.sh.
# WIDE_EP=0 (moriio + TP):   NEW cell, no legacy precedent (Stage B).
#
# Per-model FLAGS come from models.yaml via the driver ($MODEL_CONFIG_<ROLE>).
# Per-model ENV is exported by the driver (yaml env:) BEFORE connector_setup_env,
# so the ${VAR:-default} fallbacks below yield to any model/site override.
# =============================================================================

connector_init() {
    RPC_PORT="${MORI_RPC_PORT:-13345}"
    SERVE_PORT="${MORI_SERVE_PORT:-20005}"
    KV_PORT="${MORI_KV_PORT:-9711}"
    PROXY_PORT="${MORI_PROXY_PORT:-10001}"
    PROXY_PING_PORT="${MORI_PROXY_PING_PORT:-36367}"
    LOCAL_PING_PORT="${MORI_LOCAL_PING_PORT:-61555}"
    HANDSHAKE_PORT="${MORI_HANDSHAKE_PORT:-8405}"
    NOTIFY_PORT="${MORI_NOTIFY_PORT:-61005}"
    CONTAINER_BARRIER_PORT="${BARRIER_PORT_MORI:-2222}"

    # Proxy: vllm_router (default) or moriio_toy. vllm_router carries the DP-rank
    # KV-notify dpfix REQUIRED for wideEP DP — the toy proxy can't route the notify
    # and decode hangs ("remote blocks never arrived"). Default to vllm_router.
    PROXY_TYPE="${PROXY_TYPE:-vllm_router}"
    if [ "$PROXY_TYPE" != "vllm_router" ] && [ "$PROXY_TYPE" != "moriio_toy" ]; then
        echo "Error: invalid PROXY_TYPE='${PROXY_TYPE}' (expected 'vllm_router' or 'moriio_toy')." >&2
        exit 1
    fi
    ROUTER_PORT="${ROUTER_PORT:-${VLLM_ROUTER_HTTP_PORT:-30000}}"
    [ "$PROXY_TYPE" == "vllm_router" ] && PROXY_PORT="${ROUTER_PORT}"

    # Per-role MoRI all2all backend (wideEP only). Newer vLLM images (the v1.2.0
    # MoRI-EP image) split the kernel: prefill=high_throughput (InterNodeV1),
    # decode=low_latency (InterNodeV1LL) and REJECT the bare "mori" alias. Default
    # to the per-role names; override via PREFILL_MORI_BACKEND/DECODE_MORI_BACKEND
    # (or VLLM_ALL2ALL_BACKEND for the prefill side).
    PREFILL_MORI_BACKEND="${PREFILL_MORI_BACKEND:-${VLLM_ALL2ALL_BACKEND:-mori_high_throughput}}"
    DECODE_MORI_BACKEND="${DECODE_MORI_BACKEND:-mori_low_latency}"
}

# connector_setup_env [ep_backend]  (ep_backend only meaningful when WIDE_EP=1; mori path uses "mori")
connector_setup_env() {
    export VLLM_ROCM_USE_AITER=1
    export VLLM_ROCM_USE_AITER_MOE=1
    # MLA default on, but DeepSeek-V3 needs it OFF (block=16 + Triton MLA) to avoid
    # the fp8 decode-MLA kernel GPU-fault; respect an override from env/models.yaml.
    export VLLM_ROCM_USE_AITER_MLA="${VLLM_ROCM_USE_AITER_MLA:-1}"
    export VLLM_ROCM_USE_AITER_RMSNORM="${VLLM_ROCM_USE_AITER_RMSNORM:-1}"
    export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
    export VLLM_ROCM_USE_AITER_PAGED_ATTN=0
    export VLLM_USE_AITER_TRITON_SILU_MUL=0

    export VLLM_LOGGING_LEVEL=INFO
    export VLLM_USE_V1=1
    export VLLM_ALL2ALL_BACKEND=mori

    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MORI_SOCKET_IFNAME:-eth0}}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MORI_SOCKET_IFNAME:-eth0}}"

    export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-10800}"
    export VLLM_RINGBUFFER_WARNING_INTERVAL="${VLLM_RINGBUFFER_WARNING_INTERVAL:-3600}"
    export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600}"
    export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-300000}"

    export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
    export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
    export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-3}"
    export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
    export MORI_IB_GID_INDEX="${MORI_IB_GID_INDEX:-3}"
    export MORI_RDMA_DEVICES="${MORI_RDMA_DEVICES:-mlx5_0,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7,mlx5_8,mlx5_9}"
    # MoRI RDMA QoS — TC=41 (RoCE DSCP for lossless RDMA), SL=0. Override per-cluster.
    export MORI_RDMA_TC="${MORI_RDMA_TC:-41}"
    export MORI_RDMA_SL="${MORI_RDMA_SL:-0}"

    export MORI_NUM_QP_PER_PE="${MORI_NUM_QP_PER_PE:-4}"
    export VLLM_MORIIO_QP_PER_TRANSFER="${VLLM_MORIIO_QP_PER_TRANSFER:-4}"
    export VLLM_MORIIO_NUM_WORKERS="${VLLM_MORIIO_NUM_WORKERS:-4}"

    export VLLM_MORIIO_TRANSFER_TIMEOUT_S="${VLLM_MORIIO_TRANSFER_TIMEOUT_S:-600}"
    export VLLM_MORIIO_DEFERRED_TIMEOUT_S="${VLLM_MORIIO_DEFERRED_TIMEOUT_S:-1800}"
    export VLLM_HANDSHAKE_TIMEOUT_MINS="${VLLM_HANDSHAKE_TIMEOUT_MINS:-30}"

    export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/vllm_cache/triton}"
    export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/tmp/vllm_cache/vllm}"
    export COMGR_CACHE_DIR="${COMGR_CACHE_DIR:-/tmp/vllm_cache/comgr}"
    export AITER_JIT_DIR="${AITER_JIT_DIR:-/tmp/vllm_cache/aiter_jit}"
    mkdir -p "${TRITON_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${COMGR_CACHE_DIR}" "${AITER_JIT_DIR}" 2>/dev/null || true

    if [[ "${VLLM_ROCM_USE_AITER:-1}" == "1" ]]; then
        local _aiter_cfgs="/tmp/aiter_configs"
        local _aiter_src="/usr/local/lib/python3.12/dist-packages/aiter/configs"
        if [ -d "${_aiter_src}" ] && [ ! -f "${_aiter_cfgs}/a8w8_blockscale_tuned_gemm.csv" ]; then
            mkdir -p "${_aiter_cfgs}"
            cp "${_aiter_src}"/*.csv "${_aiter_cfgs}/" 2>/dev/null || true
        fi
    fi

    export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"
    export HIP_FORCE_DEV_KERNARG="${HIP_FORCE_DEV_KERNARG:-1}"
    export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"
    export HSA_NO_SCRATCH_RECLAIM="${HSA_NO_SCRATCH_RECLAIM:-1}"

    # NOTE: the ROCm-7.2.3 platform env that MUST reach the container at PID 1
    # (PYTORCH_ALLOC_CONF/PYTORCH_HIP_ALLOC_CONF=expandable_segments:False,
    # HSA_ENABLE_IPC_MODE_LEGACY=0, MORI_GPU_ARCHS) lives in connectors/moriio.env
    # and is forwarded by the slurm via `docker -e`. It is NOT exported here — a late
    # shell export is too late (PyTorch reads alloc-conf at import, before setup_env).

    export ROCSHMEM_HEAP_SIZE="${ROCSHMEM_HEAP_SIZE:-8589934592}"
    export ROCSHMEM_MAX_NUM_CONTEXTS="${ROCSHMEM_MAX_NUM_CONTEXTS:-256}"
    # MoRI shmem heap: 4 GiB default too small for EP>=32; 16 GiB (matches #324).
    export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-17179869184}"
    if [ -n "${NIXL_COOKBOOK_PATH:-}" ]; then
        export PYTHONPATH="${NIXL_COOKBOOK_PATH}/proxy${PYTHONPATH:+:$PYTHONPATH}"
    fi
}

_moriio_build_kv_transfer_config() {
    local kv_role="$1"
    # DSV4: 5a4c/v0280 ignore VLLM_MORIIO_QP_PER_TRANSFER env; extra_config is the path.
    # GLM / DSV3 stay on the original extra_config (their images still honor the env).
    local extra='"proxy_ip":"'"${MASTER_ADDR}"'","proxy_port":"'"${PROXY_PORT}"'","proxy_ping_port":"'"${PROXY_PING_PORT}"'","http_port":"'"${SERVE_PORT}"'","local_ping_port":"'"${LOCAL_PING_PORT}"'","handshake_port":"'"${HANDSHAKE_PORT}"'","notify_port":"'"${NOTIFY_PORT}"'"'
    if [ "${MODEL_NAME:-}" = "DeepSeek-V4-Flash-FP8" ] || [ "${MODEL_NAME:-}" = "DeepSeek-V4-Pro-FP8" ]; then
        extra="${extra},\"qp_per_transfer\":${VLLM_MORIIO_QP_PER_TRANSFER:-4},\"num_workers\":${VLLM_MORIIO_NUM_WORKERS:-4},\"post_batch_size\":${VLLM_MORIIO_POST_BATCH_SIZE:--1}"
    fi
    echo '{"kv_connector":"MoRIIOConnector","kv_role":"'"${kv_role}"'","kv_port":"'"${KV_PORT}"'","kv_connector_extra_config":{'"${extra}"'}}'
}

connector_runtime_patch() {
    # GLM / DSV3 / dense: no-op (fixes are in-source on develop images).
    # DSV4 Flash/Pro on the v0.28.0 image needs the runtime patchers below.
    if [ "${MODEL_NAME:-}" != "DeepSeek-V4-Flash-FP8" ] && [ "${MODEL_NAME:-}" != "DeepSeek-V4-Pro-FP8" ]; then
        return 0
    fi
    if [ "${SKIP_RUNTIME_PATCH:-0}" = "1" ]; then
        echo "[dsv4] SKIP_RUNTIME_PATCH=1"
        return 0
    fi
    if [ "${MORI_JIT_SCRUB:-1}" = "1" ]; then
        echo "[mori-scrub] removing stale MoRI JIT locks/state (/root/.mori, /tmp/mori_jit_*)"
        find /root/.mori -name '*.lock' -delete 2>/dev/null || true
        rm -rf /root/.mori /tmp/mori_jit_* 2>/dev/null || true
    fi
    if [ "${AITER_JIT_SCRUB:-1}" = "1" ]; then
        echo "[aiter-scrub] removing stale AITER JIT cache (${AITER_JIT_DIR:-/tmp/vllm_cache/aiter_jit})"
        rm -rf "${AITER_JIT_DIR:-/tmp/vllm_cache/aiter_jit}"/* 2>/dev/null || true
    fi
    _mori_combine_original_topk_fix
    if [ "${VLLM_PD_DEBUG:-1}" = "1" ]; then
        _moriio_pd_debug_patch
    fi
    _dsv4_moriio_attn_backend_fix
    _dsv4_skip_indexer_register
    _dsv4_skip_noncontiguous_register
    _dsv4_mixed_block_size_fix
    _dsv4_transfer_gate_fix
    _dsv4_supports_hma_fix
    _dsv4_hma_upstream_geom_fix
    _dsv4_region_len_span_fix
    _dsv4_backend_detect_fix
    _dsv4_defer_diag_fix
    _dsv4_rid_map_diag_fix
    _dsv4_write_diag_fix
    _dsv4_decode_diag_fix
    _dsv4_kv_hash_fix
    _dsv4_attn_transfer_fix
    _dsv4_chunked_prefill_hma_fix
    _dsv4_rdma_wait_fix
}

_mori_combine_original_topk_fix() {
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_mori_combine_original_topk_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [mori-combine] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [mori-combine] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[mori-combine] applying ${_py} against ${_vllm_dir}"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [mori-combine] patch failed — EP32 would emit garbage. Aborting." >&2
        exit 1
    }
}

_moriio_pd_debug_patch() {
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_pd_debug.py"
    if [ ! -f "${_py}" ]; then
        echo "[pd-debug] ${_py} missing; skipping connector logs."
        return 0
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "[pd-debug] cannot locate vLLM dir; skipping connector logs."
        return 0
    fi
    echo "[pd-debug] applying ${_py} against ${_vllm_dir}"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || echo "[pd-debug] connector patch failed (non-fatal)."
}

# DSV4 Flash: MoRIIO generic MLA selector rejects fp8_ds_mla (217457/217460).
# Flash-only. Abort on hard failure — boot would die the same way without it.
_dsv4_moriio_attn_backend_fix() {
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_dsv4_attn_backend_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-attn] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-attn] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-attn] applying ${_py} against ${_vllm_dir}"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-attn] patch failed — Flash PD would die in get_attn_backend. Aborting." >&2
        exit 1
    }
}

# DSV4 indexer k_cache is block 64, MLA is 256 (217463). Skip-register was
# the boot workaround; 217981/217952 then NIAH'd with a cold Lightning
# Indexer. Default off. =1 restores skip (do not mix with gate indexer WRITE).
# GLM indexer_transfer H2 is the wrong tool here (same MLA block_ids).
_dsv4_skip_indexer_register() {
    if [ "${DSV4_SKIP_INDEXER_REGISTER:-0}" != "1" ]; then
        echo "[dsv4-idx] DSV4_SKIP_INDEXER_REGISTER=${DSV4_SKIP_INDEXER_REGISTER:-0}: keeping .indexer. in RDMA register (217981 skipped 42 / NIAH 0/10)"
        return 0
    fi
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_glm_dsa_skip_indexer_register_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-idx-skip] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-idx-skip] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-idx-skip] DSV4_SKIP_INDEXER_REGISTER=1 applying ${_py} against ${_vllm_dir}"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-idx-skip] patch failed — Flash PD would die on indexer 64 != MLA 256. Aborting." >&2
        exit 1
    }
}

# DSV4 Flash: SupportsHMA + per-group block ids so hybrid KV stays on
# (217665 ~4s/tok with HMA off). Runs after mixed-bs + gate; rewrites
# wait_for_save to include block-64 .attn. Do not skip-swa in the same cell.
_dsv4_supports_hma_fix() {
    if [ "${DSV4_ENABLE_HMA:-1}" = "0" ]; then
        echo "[dsv4-hma] skipped DSV4_ENABLE_HMA=0 (217666 path; 217748 HMA-on ASCII)"
        return 0
    fi
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_dsv4_supports_hma_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-hma] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-hma] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-hma] applying ${_py} against ${_vllm_dir}"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-hma] patch failed — Flash would keep HMA off (~4s/tok). Aborting." >&2
        exit 1
    }
}

# DSV4: vllm-project/vllm#48989 relabels the compressed group-0 MLA page instead
# of folding it. 218042's fold divided dim[0] by kbpb (4x even, 128x odd), which
# shrank the page table until no remote id resolved — fold-ok on all 41 layers
# and n_remote=0 on every write. Runs AFTER the HMA patcher, whose fold it
# overrides. Runtime flag DSV4_HMA_UPSTREAM_GEOM (default 0 = 218042 fold).
_dsv4_hma_upstream_geom_fix() {
    # HMA-off has no fold to override, and DSV4_TRANSFER_ATTN already relabels.
    if [ "${DSV4_ENABLE_HMA:-1}" = "0" ]; then
        echo "[dsv4-hma-geom] skipped DSV4_ENABLE_HMA=0 (no fold; DSV4_TRANSFER_ATTN arm)"
        return 0
    fi
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_dsv4_hma_upstream_geom_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-hma-geom] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-hma-geom] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-hma-geom] applying ${_py} against ${_vllm_dir} (DSV4_HMA_UPSTREAM_GEOM=${DSV4_HMA_UPSTREAM_GEOM:-0} CLIP_PAGE=${DSV4_HMA_CLIP_PAGE:-0} NATIVE_ATTN=${DSV4_HMA_NATIVE_ATTN:-0})"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-hma-geom] patch failed — both HMA arms would be the 218042 fold. Aborting." >&2
        exit 1
    }
}

# DSV4: region_len = num_blocks * regions_per_block * block_len, but offsets step
# block_stride regardless, so a per-page block_len registers ~38x less than the
# addressing reaches — silently, since the connector's ValueError guards
# block_size, not block_len. That was 218257's garbage decode. Floor the extent at
# num_blocks * block_stride * element_size. DSV4_REGION_LEN_SPAN (default 0).
# Pairs with DSV4_HMA_PAGE_BLOCK_LEN=1; that flag alone reproduces 218257, so
# refuse the combination outright.
_dsv4_region_len_span_fix() {
    if [ "${DSV4_HMA_PAGE_BLOCK_LEN:-0}" = "1" ] && [ "${DSV4_REGION_LEN_SPAN:-0}" != "1" ]; then
        echo "Error: [dsv4-region] DSV4_HMA_PAGE_BLOCK_LEN=1 without DSV4_REGION_LEN_SPAN=1 is the 218257 garbage cell. Aborting." >&2
        exit 1
    fi
    if [ "${DSV4_HMA_NATIVE_ATTN:-0}" = "1" ] && [ "${DSV4_REGION_LEN_SPAN:-0}" != "1" ]; then
        echo "Error: [dsv4-region] DSV4_HMA_NATIVE_ATTN=1 without DSV4_REGION_LEN_SPAN=1 under-registers .attn (218257). Aborting." >&2
        exit 1
    fi
    if [ "${DSV4_REGION_LEN_SPAN:-0}" != "1" ]; then
        echo "[dsv4-region] DSV4_REGION_LEN_SPAN=${DSV4_REGION_LEN_SPAN:-0}: keeping the shipped region_len"
        return 0
    fi
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_dsv4_region_len_span_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-region] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-region] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-region] applying ${_py} against ${_vllm_dir} (DSV4_REGION_LEN_SPAN=1 PAGE_BLOCK_LEN=${DSV4_HMA_PAGE_BLOCK_LEN:-0})"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-region] patch failed — a per-page block_len would under-register. Aborting." >&2
        exit 1
    }
}

# DSV4: diagnostics for the 218282 hang. _is_remote_ready(task) is only
# "task.transfer_id in worker.moriio_wrapper.done_remote_allocate_req_dict", so a
# deferred write means D never told P its block allocation for that transfer id;
# at defer_timeout (60s default) P force-frees the blocks and D waits on KV that
# never comes. Log the awaited id beside the keys the dict does hold: empty is a
# lost notify, other-keys-present is a keying mismatch, late arrival shows as a
# large waited= on "ready". Diagnostic only. DSV4_DEFER_DIAG (default 0).
_dsv4_defer_diag_fix() {
    if [ "${DSV4_DEFER_DIAG:-0}" != "1" ]; then
        echo "[dsv4-defer] DSV4_DEFER_DIAG=${DSV4_DEFER_DIAG:-0}: keeping the shipped defer logging"
        return 0
    fi
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_dsv4_defer_diag_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-defer] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-defer] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-defer] applying ${_py} against ${_vllm_dir} (DSV4_DEFER_DIAG=1)"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-defer] patch failed — the hang would go unlogged. Aborting." >&2
        exit 1
    }
}

# 218318 HMA=1 logged 10 "unmap MISS ... table_size=0" — one per reap — while
# 218323 HMA=0 logged none, so the producer rid<->transfer_id table is empty
# only with HMA on and no ACK can be attributed. map_request_id sits after the
# "if not params: return" guard in update_state_after_alloc, so this traces map
# and unmap with the instance identity (dp/gdp/self) to separate an empty
# kv_transfer_params from a map and unmap landing on different schedulers.
# Diagnostic only. DSV4_RID_MAP_DIAG (default 0).
_dsv4_rid_map_diag_fix() {
    if [ "${DSV4_RID_MAP_DIAG:-0}" != "1" ]; then
        echo "[dsv4-ridmap] DSV4_RID_MAP_DIAG=${DSV4_RID_MAP_DIAG:-0}: keeping the shipped map/unmap logging"
        return 0
    fi
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_dsv4_rid_map_diag_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-ridmap] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-ridmap] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-ridmap] applying ${_py} against ${_vllm_dir} (DSV4_RID_MAP_DIAG=1)"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-ridmap] patch failed — the empty table would go unexplained. Aborting." >&2
        exit 1
    }
}

# 218417: offsets-cache group collision (now keyed) + pending-save merge +
# per-request write-pair logs. Always apply: the group-key and merge are
# behavioural. DSV4_WRITE_DIAG (default 1) only gates the extra logs.
_dsv4_write_diag_fix() {
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_dsv4_write_diag_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-write-diag] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-write-diag] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-write-diag] applying ${_py} against ${_vllm_dir} (DSV4_WRITE_DIAG=${DSV4_WRITE_DIAG:-1})"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-write-diag] patch failed — SWA twins would share dests. Aborting." >&2
        exit 1
    }
}

# 220985: WRITE sealed 243/243, decode LENGTH_CAPPED 63 tok, NIAH chars=0.
# Log consumer per-group dests on both update_state_after_alloc maps, and
# sampled token ids / detok skip=False vs True. Diagnostic only.
# DSV4_DECODE_DIAG (default 0) only gates the extra logs.
_dsv4_decode_diag_fix() {
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_dsv4_decode_diag_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-decode] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-decode] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-decode] applying ${_py} against ${_vllm_dir} (DSV4_DECODE_DIAG=${DSV4_DECODE_DIAG:-0})"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-decode] patch failed — 2k chars=0 would stay untyped. Aborting." >&2
        exit 1
    }
}

# 221697: WRITE sealed, decode sampled 63x BOS. Hash group-0 / SWA / indexer
# pages on prefill src vs decode dest, and log sparse-MLA metadata_key /
# paged_kv_indices zero_frac. 223342: also probe runner block_table each
# execute_model (DSV4-KV-RUNNER-BT). Diagnostic only. DSV4_KV_HASH (default 0).
# Graph-vs-compress is DSV4_EAGER=1 (decode cudagraph NONE), not this patcher.
# 223417: graph is out. runner-bt g0slot= logs group-0 dim[1]*ratio coverage.
_dsv4_kv_hash_fix() {
    if [ "${DSV4_KV_HASH:-0}" != "1" ]; then
        echo "[dsv4-kv] skipped DSV4_KV_HASH=${DSV4_KV_HASH:-0}"
        return 0
    fi
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_dsv4_kv_hash_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-kv] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-kv] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-kv] applying ${_py} against ${_vllm_dir} (DSV4_KV_HASH=${DSV4_KV_HASH:-0})"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-kv] patch failed — 221697 BOS vs empty group-0 would stay unsplit. Aborting." >&2
        exit 1
    }
}

# DSV4: vllm#48989's other code hunk. The shipped connector sets backend_name
# from one get_attn_backend() call in __init__, which cannot describe DSV4's
# hybrid set (group-0 MLAAttentionSpec + groups 1-4 SlidingWindowMLASpec +
# .indexer.k_cache). Upstream asks get_current_attn_backends() at register time
# instead. Additive: logs the list beside the shipped value so we can see
# whether the single query was ever wrong. DSV4_BACKEND_DETECT (default 0).
_dsv4_backend_detect_fix() {
    if [ "${DSV4_BACKEND_DETECT:-0}" != "1" ]; then
        echo "[dsv4-backend] DSV4_BACKEND_DETECT=${DSV4_BACKEND_DETECT:-0}: keeping the shipped __init__ get_attn_backend query"
        return 0
    fi
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_dsv4_backend_detect_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-backend] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-backend] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-backend] applying ${_py} against ${_vllm_dir} (DSV4_BACKEND_DETECT=1)"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-backend] patch failed — backend_name would stay the single __init__ query. Aborting." >&2
        exit 1
    }
}

# DSV4 Flash: KV views are non-contiguous (217514). 217532 skip-all left 0
# caches (StopIteration). Register a data_ptr-aligned storage span; no .contiguous().
# Dual-anchor: v0.28 per-layer register_local_tensor(kv_cache); v0.29 also
# rewrites _build_shared_kv_mr .view(uint8) and skips shared MR on strided KV.
_dsv4_skip_noncontiguous_register() {
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_dsv4_skip_noncontiguous_register_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-storage] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-storage] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-storage] applying ${_py} against ${_vllm_dir} (DSV4_STORAGE_UPSTREAM_SPAN=${DSV4_STORAGE_UPSTREAM_SPAN:-0})"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-storage] patch failed — Flash PD would die on non-contiguous register. Aborting." >&2
        exit 1
    }
}

# DSV4 Flash: SWA .attn is block 64, MLA is 256 (217473). Connector already
# has per-layer block_lens; only the uniform ValueError blocks register.
_dsv4_mixed_block_size_fix() {
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_dsv4_mixed_block_size_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-mixed-bs] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-mixed-bs] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-mixed-bs] applying ${_py} against ${_vllm_dir}"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-mixed-bs] patch failed — Flash PD would die on SWA 64 != MLA 256. Aborting." >&2
        exit 1
    }
}

# DSV4 Flash: 217546. wait_for_save must not dump 64-block .attn. Log
# writes_done / writes_expected / send_notify so DP1 is visible. 5a4c already
# has per-layer offsets + writes_expected; GLM gate/engine would no-op.
_dsv4_transfer_gate_fix() {
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_dsv4_transfer_gate_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-gate] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-gate] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-gate] applying ${_py} against ${_vllm_dir}"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-gate] patch failed — Flash DP1 WRITE would hang like 217546. Aborting." >&2
        exit 1
    }
}

# DSV4 218040: the gate skipped every model.layers.N.attn — the only cache
# with sliding_window=None (group 0 MLAAttentionSpec page 256). Decode was
# left with the 128-token swa_cache, so curl answered and NIAH never
# retrieved. dim[1] is compressed slots (256 / compress_ratios 4|128), so the
# shipped block_len / block_stride are already the right per-page span.
# Runtime flag DSV4_TRANSFER_ATTN (default 0 = 218040 behaviour).
_dsv4_attn_transfer_fix() {
    # HMA=1 rewrites the same wait_for_save with per-group block ids. Mutually
    # exclusive: this is the HMA-off arm.
    if [ "${DSV4_ENABLE_HMA:-1}" != "0" ]; then
        echo "[dsv4-attn-xfer] skipped: DSV4_ENABLE_HMA=1 owns wait_for_save"
        return 0
    fi
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_dsv4_attn_transfer_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-attn-xfer] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-attn-xfer] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-attn-xfer] applying ${_py} against ${_vllm_dir} (DSV4_TRANSFER_ATTN=${DSV4_TRANSFER_ATTN:-0})"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-attn-xfer] patch failed. Aborting." >&2
        exit 1
    }
}

# 218328 20-token cliff: last-chunk used len(groups)*smallest_page. 218687
# reproduced it with the patcher retired (curl 17 tok wrote, 2k never did).
# 218417 still hung after writes sealed — that is the write-diag group-key,
# not a reason to skip this patcher. Apply under HMA=1; env default 1.
_dsv4_chunked_prefill_hma_fix() {
    if [ "${DSV4_ENABLE_HMA:-1}" = "0" ]; then
        echo "[dsv4-chunk-hma] skipped: DSV4_ENABLE_HMA=0 (block ids are flat)"
        return 0
    fi
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_dsv4_chunked_prefill_hma_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-chunk-hma] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-chunk-hma] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-chunk-hma] applying ${_py} against ${_vllm_dir} (DSV4_CHUNK_HMA_FIX=${DSV4_CHUNK_HMA_FIX:-1})"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-chunk-hma] patch failed — 2k would stay on the 20-token cliff. Aborting." >&2
        exit 1
    }
}

# DSV4 Flash: 217557. writes_done=84 then ~391s CQE wait on WRITE 2+.
# 5a4c already waits outside the lock; log elapsed + time Succeeded().
# QP/worker knobs go in extra_config (env is ignored on 5a4c).
_dsv4_rdma_wait_fix() {
    local _patch_dir="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)}"
    local _py="${_patch_dir}/apply_moriio_dsv4_rdma_wait_fix.py"
    if [ ! -f "${_py}" ]; then
        echo "Error: [dsv4-rdma-wait] ${_py} not found. Aborting." >&2
        exit 1
    fi
    local _vllm_dir
    _vllm_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
    if [ -z "${_vllm_dir}" ] || [ ! -d "${_vllm_dir}" ]; then
        echo "Error: [dsv4-rdma-wait] cannot locate vLLM install dir. Aborting." >&2
        exit 1
    fi
    echo "[dsv4-rdma-wait] applying ${_py} against ${_vllm_dir}"
    python3 "${_py}" "${_vllm_dir}" 2>&1 || {
        echo "Error: [dsv4-rdma-wait] patch failed — Flash WRITE 2+ would stay ~391s. Aborting." >&2
        exit 1
    }
}

# connector_launch_worker <role> <dp_size> <dp_addr> <kv_role> <log_prefix> [dp_start_rank]
connector_launch_worker() {
    local role="$1" dp_size="$2" dp_addr="$3" kv_role="$4" log_prefix="$5" dp_start_rank="${6:-}"

    connector_setup_env "${EP_BACKEND:-mori}"

    # Patch PyTorch default_pg_timeout (DP Gloo groups) — wideEP only.
    if parallelism_is_wide_ep; then
        local _timeout_s="${DISTRIBUTED_TIMEOUT_SECONDS:-7200}"
        local _torch_const="/usr/local/lib/python3.12/dist-packages/torch/distributed/constants.py"
        if [ -f "$_torch_const" ]; then
            sed -i "s/default_pg_timeout: timedelta = _DEFAULT_PG_TIMEOUT/default_pg_timeout: timedelta = timedelta(seconds=${_timeout_s})/" "$_torch_const" 2>/dev/null || true
        fi
    fi

    # Per-role execution mode. Ported from #324: NEVER use bare --enforce-eager.
    # On these AITER images an enforce-eager worker (no +quant_fp8 custom op) routes
    # fp8 quant through an AITER op whose signature mismatches the build
    # (dynamic_per_token_scaled_quant: out aiter_tensor_t) -> engine-init crash.
    # So even "no cudagraph" is expressed as cudagraph_mode:NONE WITH +quant_fp8.
    # Per-role mode: DECODE_CUDAGRAPH_MODE / PREFILL_CUDAGRAPH_MODE, falling back to
    # the global VLLM_CUDAGRAPH_MODE (back-compat).
    local exec_args=()
    local _cudagraph_mode="${VLLM_CUDAGRAPH_MODE:-}"
    if [[ "$log_prefix" == "decode" ]]; then
        _cudagraph_mode="${DECODE_CUDAGRAPH_MODE:-$_cudagraph_mode}"
        if [[ "${DSV4_EAGER:-0}" == "1" ]]; then
            _cudagraph_mode=NONE
            echo "[dsv4-cg] DSV4_EAGER=1 -> decode cudagraph_mode=NONE +quant_fp8 (never --enforce-eager)"
        fi
    else
        _cudagraph_mode="${PREFILL_CUDAGRAPH_MODE:-$_cudagraph_mode}"
    fi
    # use_inductor_graph_partition=true moves graph partitioning from Dynamo/FX to
    # inductor codegen, splitting at cudagraph_unsafe ops (incl. the MLA KV-update) so
    # they run as eager boundaries. Default OFF: enabling it here would change
    # --compilation-config for EVERY model. GLM opts in via its models.yaml env:.
    local _igp_json=""
    [[ "${USE_INDUCTOR_GRAPH_PARTITION:-0}" == "1" ]] && _igp_json=',"use_inductor_graph_partition":true'
    if [[ -n "$_cudagraph_mode" && "$_cudagraph_mode" != "NONE" ]]; then
        local _capture_sizes="${CUDAGRAPH_CAPTURE_SIZES:-1 2 4 8 16 32 64 128 256}"
        exec_args+=(--compilation-config '{"cudagraph_mode":"'"${_cudagraph_mode}"'","custom_ops":["+quant_fp8"]'"${_igp_json}"'}')
        exec_args+=(--cudagraph-capture-sizes ${_capture_sizes})
    else
        exec_args+=(--compilation-config '{"cudagraph_mode":"NONE","custom_ops":["+quant_fp8"]'"${_igp_json}"'}')
    fi

    # Per-model flags from models.yaml (driver-exported; empty if none).
    local model_args=()
    local _mc; if [[ "$log_prefix" == "prefill" ]]; then _mc="${MODEL_CONFIG_PREFILL:-}"; else _mc="${MODEL_CONFIG_DECODE:-}"; fi
    [[ -n "$_mc" ]] && eval "model_args=(${_mc})"

    # Agentic gating: the default sweep keeps prefix caching OFF (clean, cache-free
    # throughput) via the hardcoded --no-enable-prefix-caching below. The agentic
    # trace-replay path (BENCHMARK_SCRIPT_FILE=benchmark_agentic.sh) — or an explicit
    # ENABLE_PREFIX_CACHE=1 — STRIPS that flag so prefix caching is ON. Gated so the
    # default (non-agentic) sweep argv is byte-for-byte unchanged.
    local _prefix_cache_flag="--no-enable-prefix-caching"
    if [[ "${BENCHMARK_SCRIPT:-}" == "agentic" || "${ENABLE_PREFIX_CACHE:-0}" == "1" ]]; then
        _prefix_cache_flag=""
    fi

    if parallelism_is_wide_ep; then
        # ---- WIDE_EP=1 (MoriEP) ----
        # Per-role all2all: prefill=high_throughput, decode=low_latency. The
        # v1.2.0 image rejects the bare "mori" alias; these names are required.
        local _all2all="${PREFILL_MORI_BACKEND}"
        [[ "$log_prefix" == "decode" ]] && _all2all="${DECODE_MORI_BACKEND}"

        local extra_args=() kv_args=()
        if [[ "$role" == "master" ]]; then
            extra_args+=(--api-server-count=${_GPUS_PER_NODE})
            local kv_config; kv_config=$(_moriio_build_kv_transfer_config "${kv_role}")
            kv_args+=(--kv-transfer-config "${kv_config}")
        else
            extra_args+=(--data-parallel-start-rank "${dp_start_rank}" --headless)
        fi

        # Recipe knobs (overridable via env / models.yaml). DeepSeek-V3 on AITER
        # needs block=16 + MLA off (the block=1 + AITER-MLA fp8 decode kernel
        # GPU-faults), and KV_CACHE_MEMORY_BYTES to SKIP the boot profiling forward
        # (that forward is the only path that calls AITER's eager per-token fp8
        # quant, which has the aiter_tensor_t torch_guard bug -> engine-init crash).
        local _block="${KV_BLOCK_SIZE:-1}"
        local _kvdtype="${KV_CACHE_DTYPE:-fp8}"
        local mem_args=()
        [[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]] && mem_args+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")

        if [[ "${DRY_RUN:-0}" == "1" ]]; then
            _dryrun_emit "moriio" "${log_prefix}" "${role}" \
                vllm serve "${MODEL_PATH}" \
                    -tp 1 \
                    --data-parallel-size "${dp_size}" \
                    --data-parallel-size-local "${DP_PARALLEL_SIZE_LOCAL}" \
                    --data-parallel-address "${dp_addr}" \
                    --data-parallel-rpc-port "${RPC_PORT}" \
                    --enable-expert-parallel \
                    --port "${SERVE_PORT}" \
                    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
                    "${mem_args[@]}" \
                    --kv-cache-dtype "${_kvdtype}" \
                    --block-size "${_block}" \
                    ${_prefix_cache_flag} \
                    --all2all-backend "${_all2all}" \
                    --trust-remote-code \
                    --distributed-timeout-seconds "${DISTRIBUTED_TIMEOUT_SECONDS:-7200}" \
                    "${exec_args[@]}" "${extra_args[@]}" "${kv_args[@]}" "${model_args[@]}"
            WORKER_PID=0; return 0
        fi

        vllm serve ${MODEL_PATH} \
            -tp 1 \
            --data-parallel-size "${dp_size}" \
            --data-parallel-size-local ${DP_PARALLEL_SIZE_LOCAL} \
            --data-parallel-address "${dp_addr}" \
            --data-parallel-rpc-port ${RPC_PORT} \
            --enable-expert-parallel \
            --port ${SERVE_PORT} \
            --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.8} \
            "${mem_args[@]}" \
            --kv-cache-dtype "${_kvdtype}" \
            --block-size "${_block}" \
            ${_prefix_cache_flag} \
            --all2all-backend "${_all2all}" \
            --trust-remote-code \
            --distributed-timeout-seconds ${DISTRIBUTED_TIMEOUT_SECONDS:-7200} \
            "${exec_args[@]}" \
            "${extra_args[@]}" \
            "${kv_args[@]}" \
            "${model_args[@]}" \
            2>&1 | tee /run_logs/${SLURM_JOB_ID}/${log_prefix}_NODE${NODE_RANK}.log >/dev/null &
        WORKER_PID=$!
        return 0
    fi

    # ---- WIDE_EP=0 (moriio + TP) — NEW cell (Stage B) ----
    # MoRIIO KV transfer over a plain tensor-parallel server (no EP). Every node
    # is a full server; the kv-transfer-config is attached on all nodes (no DP
    # master/child split). kv-cache-dtype/block-size come from models.yaml.
    local kv_config; kv_config=$(_moriio_build_kv_transfer_config "${kv_role}")
    local _tp_size="${IO_TP_SIZE:-${GENERIC_TP_SIZE:-8}}"

    # The connector owns the parallelism *degree* (--tensor-parallel-size) for the
    # moriio+TP path, so strip any --tensor-parallel-size N that a models.yaml entry
    # may carry (those entries are shared with the rixl path, which DOES want it in
    # yaml). Prevents a duplicate --tensor-parallel-size on the moriio+TP command.
    local _filtered=() _skip=0 _a
    for _a in "${model_args[@]}"; do
        if [[ "$_skip" == "1" ]]; then _skip=0; continue; fi
        if [[ "$_a" == "--tensor-parallel-size" ]]; then _skip=1; continue; fi
        if [[ "$_a" == "--tensor-parallel-size="* ]]; then continue; fi
        _filtered+=("$_a")
    done
    model_args=("${_filtered[@]}")

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        _dryrun_emit "moriio" "${log_prefix}" "${role}" \
            vllm serve "${MODEL_PATH}" \
                --tensor-parallel-size "${_tp_size}" \
                --port "${SERVE_PORT}" \
                --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
                --trust-remote-code \
                --distributed-timeout-seconds "${DISTRIBUTED_TIMEOUT_SECONDS:-7200}" \
                --kv-transfer-config "${kv_config}" \
                "${exec_args[@]}" "${model_args[@]}"
        WORKER_PID=0; return 0
    fi

    vllm serve ${MODEL_PATH} \
        --tensor-parallel-size "${_tp_size}" \
        --port ${SERVE_PORT} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION:-0.8} \
        --trust-remote-code \
        --distributed-timeout-seconds ${DISTRIBUTED_TIMEOUT_SECONDS:-7200} \
        --kv-transfer-config "${kv_config}" \
        "${exec_args[@]}" \
        "${model_args[@]}" \
        2>&1 | tee /run_logs/${SLURM_JOB_ID}/${log_prefix}_NODE${NODE_RANK}.log >/dev/null &
    WORKER_PID=$!
}

connector_wait_workers_ready() {
    echo "Waiting for prefill & decode servers to be ready..."
    sleep 20
    local TIMEOUT_SECONDS="${LOG_WAIT_TIMEOUT_SECONDS:-4000}"
    local SLEEP_SECONDS=10
    local SEARCH_SIGNAL="Application startup complete."
    local PREFILL_LOG=/run_logs/${SLURM_JOB_ID}/prefill_NODE0.log
    local DECODE_LOG=/run_logs/${SLURM_JOB_ID}/decode_NODE${xP}.log
    _wait_log_signal_or_fail "${PREFILL_LOG}" "prefill master" "${SEARCH_SIGNAL}" "${TIMEOUT_SECONDS}" "${SLEEP_SECONDS}"
    _wait_log_signal_or_fail "${DECODE_LOG}" "decode master" "${SEARCH_SIGNAL}" "${TIMEOUT_SECONDS}" "${SLEEP_SECONDS}"
}

connector_start_proxy() {
    # Ported faithfully from the validated MAD-private PR#324 mori launcher.
    # vllm_router: production router with --kv-connector moriio (needs the binary on
    #   PATH or ROUTER_BINARY set); includes a registration gate so the benchmark
    #   doesn't fire before prefill+decode register (else every request 503s).
    # moriio_toy: in-image toy proxy; resolves the script across the online_serving/
    #   -> disaggregated/ path move.
    # Sets BENCHMARK_PORT (router->ROUTER_PORT, toy->PROXY_PORT) for the driver.
    # Agentic replay: point aiperf at the backend vLLM servers' /metrics (SERVE_PORT)
    # for prefill+decode masters so it can scrape gpu cache-hit / throughput. Gated on
    # the agentic path so the default sweep is unaffected. Consumed by
    # scripts/common/agentic_lib.sh (build_replay_cmd -> aiperf --server-metrics).
    if [[ "${BENCHMARK_SCRIPT:-}" == "agentic" || "${ENABLE_SERVER_METRICS:-0}" == "1" ]]; then
        export AGENTIC_SERVER_METRICS="${AGENTIC_SERVER_METRICS:-${PREFILL_MASTER_ADDR}:${SERVE_PORT} ${DECODE_MASTER_ADDR}:${SERVE_PORT}}"
        echo "[metrics] AGENTIC_SERVER_METRICS=${AGENTIC_SERVER_METRICS}"
    fi
    sleep 10
    if [ "$PROXY_TYPE" == "vllm_router" ]; then
        local PREFILL_URL="http://${PREFILL_MASTER_ADDR}:${SERVE_PORT}"
        local DECODE_URL="http://${DECODE_MASTER_ADDR}:${SERVE_PORT}"
        # Router intra-node DP size = per-node DP rank count. wideEP has
        # DP_PARALLEL_SIZE_LOCAL(=GPUS_PER_NODE) ranks/node; TP (WIDE_EP=0) is a
        # single unit -> 1. Passing 8 on the TP path makes the router round-robin
        # to DP ranks 0..7 while the TP server only has rank 0 -> every non-rank-0
        # request fails "data_parallel_rank N out of range [0,1)" (7/8 -> 500).
        local _router_dp_local="${DP_PARALLEL_SIZE_LOCAL}"
        parallelism_is_wide_ep || _router_dp_local=1
        echo "Starting vllm-router (MoRIIO): HTTP ${ROUTER_PORT}"
        echo "  prefill=${PREFILL_URL}  decode=${DECODE_URL}  dp_local=${_router_dp_local}"
        [ -f /root/.cargo/env ] && source /root/.cargo/env

        local ROUTER_BIN="${ROUTER_BINARY:-$(command -v vllm-router 2>/dev/null || true)}"
        if [ -z "${ROUTER_BIN}" ] || [ ! -x "${ROUTER_BIN}" ]; then
            echo "Error: vllm-router not found. Set ROUTER_BINARY=<path>, or PROXY_TYPE=moriio_toy to use the in-image toy proxy." \
                | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log
            exit 1
        fi
        echo "Using vllm-router binary: ${ROUTER_BIN}"
        local _PROMETHEUS_PORT="${VLLM_ROUTER_PROMETHEUS_PORT:-29000}"
        "${ROUTER_BIN}" \
            --host 0.0.0.0 \
            --port "${ROUTER_PORT}" \
            --vllm-pd-disaggregation \
            --kv-connector moriio \
            --prefill "${PREFILL_URL}" \
            --decode "${DECODE_URL}" \
            --vllm-discovery-address "0.0.0.0:${PROXY_PING_PORT}" \
            --intra-node-data-parallel-size "${_router_dp_local}" \
            --policy round_robin \
            --prefill-policy round_robin \
            --decode-policy round_robin \
            --log-level "${VLLM_ROUTER_LOG_LEVEL:-info}" \
            --prometheus-port "${_PROMETHEUS_PORT}" \
            > >(tee /run_logs/${SLURM_JOB_ID}/vllm_router_NODE${NODE_RANK}.log >/dev/null) 2>&1 &
        proxy_pid=$!
        BENCHMARK_PORT=${ROUTER_PORT}
    else
        local _PROXY_SCRIPT=""
        for _candidate in \
            "${MORIIO_TOY_PROXY:-}" \
            "${NIXL_COOKBOOK_PATH:-}/proxy/moriio_pd_proxy.py" \
            "/app/vllm/examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py" \
            "/app/vllm/examples/online_serving/disaggregated_serving/moriio_toy_proxy_server.py" \
            "$(python3 -c 'import vllm, os; print(os.path.join(os.path.dirname(vllm.__file__), "..", "examples", "disaggregated", "disaggregated_serving", "moriio_toy_proxy_server.py"))' 2>/dev/null)"; do
            if [ -n "${_candidate}" ] && [ -f "${_candidate}" ]; then
                _PROXY_SCRIPT="${_candidate}"; break
            fi
        done
        if [ -z "${_PROXY_SCRIPT}" ]; then
            echo "Error: moriio_toy_proxy_server.py not found (upstream vLLM path changed?)" \
                | tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log
            exit 1
        fi
        if [[ "${_PROXY_SCRIPT}" == *moriio_pd_proxy.py ]]; then
            local _PROXY_LOG_LEVEL="${PROXY_LOG_LEVEL:-DEBUG}"
            local _PROXY_ARGS=(
                --port "${PROXY_PORT}"
                --discovery-port "${PROXY_PING_PORT}"
                --use-discovery
                --dp-size-local "${DP_PARALLEL_SIZE_LOCAL}"
                --expect-prefill "${xP}"
                --expect-decode "${yD}"
                --log-level "${_PROXY_LOG_LEVEL}"
                --max-concurrency "${PROXY_MAX_CONCURRENCY:-512}"
            )
            if [[ -n "${PROXY_ROUTE_DP:-}" && "${PROXY_ROUTE_DP}" != "0" ]]; then
                _PROXY_ARGS+=(--route-dp-size "${PROXY_ROUTE_DP}")
            fi
            local _i
            for ((_i=0; _i<xP && _i<${#IP_ARRAY[@]}; _i++)); do
                _PROXY_ARGS+=(--prefill "http://${IP_ARRAY[$_i]}:${SERVE_PORT}")
            done
            for ((_i=xP; _i<${#IP_ARRAY[@]}; _i++)); do
                _PROXY_ARGS+=(--decode "http://${IP_ARRAY[$_i]}:${SERVE_PORT}")
            done
            echo "Starting python PD proxy (DSV4 MoRI-EP): ${_PROXY_SCRIPT}"
            python "${_PROXY_SCRIPT}" "${_PROXY_ARGS[@]}" \
                > >(tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null) 2>&1 &
            proxy_pid=$!
            BENCHMARK_PORT=${PROXY_PORT}
            local _PROXY_LOG="/run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log"
            local _z=0
            while [ "${_z}" -lt 15 ]; do
                if grep -q "ZMQ discovery listening" "${_PROXY_LOG}" 2>/dev/null; then
                    break
                fi
                if grep -q "Address already in use" "${_PROXY_LOG}" 2>/dev/null; then
                    echo "Error: ZMQ ${PROXY_PING_PORT} bind failed. HTTP would 503 every request." >&2
                    grep -E "Address already in use|ZMQ" "${_PROXY_LOG}" | head -n 8 >&2 || true
                    exit 1
                fi
                sleep 1; _z=$((_z + 1))
            done
            if ! grep -q "ZMQ discovery listening" "${_PROXY_LOG}" 2>/dev/null; then
                echo "Error: ZMQ discovery did not listen on ${PROXY_PING_PORT} within 15s." >&2
                tail -n 30 "${_PROXY_LOG}" >&2 || true
                exit 1
            fi
        else
            echo "Starting MoRI toy proxy: ${_PROXY_SCRIPT}"
            python "${_PROXY_SCRIPT}" \
                > >(tee -a /run_logs/${SLURM_JOB_ID}/proxy_NODE${NODE_RANK}.log >/dev/null) 2>&1 &
            proxy_pid=$!
            BENCHMARK_PORT=${PROXY_PORT}
        fi
    fi
    export BENCHMARK_PORT

    echo "Proxy (${PROXY_TYPE}) ready for benchmarking on ${host_name}:${host_ip}:${BENCHMARK_PORT}"

    # Router-registration gate (vllm_router only): wait for prefill+decode to
    # register via discovery before benchmarking, else requests 503.
    if [ "$PROXY_TYPE" == "vllm_router" ]; then
        local _ROUTER_LOG="/run_logs/${SLURM_JOB_ID}/vllm_router_NODE${NODE_RANK}.log"
        local _REG_TIMEOUT="${ROUTER_REGISTER_TIMEOUT_S:-300}" _waited=0
        echo "Waiting up to ${_REG_TIMEOUT}s for prefill+decode to register with the router..."
        while [ "${_waited}" -lt "${_REG_TIMEOUT}" ]; do
            if grep -qa "Add Prefill" "${_ROUTER_LOG}" 2>/dev/null && \
               grep -qa "Add Decode"  "${_ROUTER_LOG}" 2>/dev/null; then
                echo "Router registration complete after ${_waited}s."; break
            fi
            sleep 5; _waited=$((_waited + 5))
        done
        [ "${_waited}" -ge "${_REG_TIMEOUT}" ] && echo "WARNING: router registration not confirmed in ${_REG_TIMEOUT}s; proceeding."
    else
        sleep 20
    fi

    curl -X POST http://127.0.0.1:${BENCHMARK_PORT}/v1/completions -H "Content-Type: application/json" -d '{
        "prompt": "Who is AMD CEO?",
        "temperature": 0,
        "max_tokens" : 10,
        "top_k": 1
    }'
    sleep 20
}
