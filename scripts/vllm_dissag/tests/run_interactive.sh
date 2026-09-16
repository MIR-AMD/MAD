#!/bin/bash
# Interactive multi-node launcher (mirrors run_xPyD_models.slurm's docker run, but
# NODE_RANK is passed per node instead of being derived from SLURM_PROCID). Used by
# tests/drive_cell.sh to drive an existing allocation via `srun --overlap`.
# Usage: run_interactive.sh <NODE_RANK>
# Env expected: DOCKER_IMAGE_NAME, MODEL_NAME, MODEL_PATH, IPADDRS, MASTER_ADDR,
#   MASTER_PORT, NNODES, xP, yD, CONNECTOR, WIDE_EP, BENCHMARK_CON,
#   BENCHMARK_COMBINATIONS, SLURM_JOB_ID, plus any recipe/-e overrides.
set -u
NODE_RANK="$1"

NIXL_COOKBOOK_PATH="/opt/nixl-vllm-cookbook"
# repo dir = the vllm_dissag dir containing this tests/ folder (override NIXL_REPO_DIR to relocate)
NIXL_REPO_DIR="${NIXL_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
LOG_PATH="${LOG_PATH:-/shared_inference/${USER}/model_blog_logs}"
mkdir -p "$LOG_PATH" 2>/dev/null || true
DOCKER_CONT_NAME="container_${MODEL_NAME}_${SLURM_JOB_ID}"
RUN_FILE_FULL="$NIXL_COOKBOOK_PATH/vllm_disagg.sh"

# cleanup any stale container/ports on this node
docker rm -f "$DOCKER_CONT_NAME" 2>/dev/null || true
fuser -k 5000/tcp 2>/dev/null || true
fuser -k 2222/tcp 2>/dev/null || true
fuser -k 15000/tcp 2>/dev/null || true
fuser -k 30000/tcp 2>/dev/null || true
fuser -k 36367/tcp 2>/dev/null || true   # router discovery / moriio proxy_ping
fuser -k 20005/tcp 2>/dev/null || true   # serve port
fuser -k 13345/tcp 2>/dev/null || true   # VLLM_DP_Coordinator (stale-run port clash)
fuser -k 39566/tcp 2>/dev/null || true   # data-parallel master port
sleep 2

mkdir -p /tmp/vllm_cache/{aiter_jit,triton,vllm,comgr} 2>/dev/null || true
# Persistent JIT cache: the image points AITER_JIT_DIR/TRITON_CACHE_DIR/VLLM_CACHE_ROOT/
# COMGR_CACHE_DIR at /opt/vllm_cache. Mount a host dir there so AITER CK kernels compile
# ONCE and are reused across runs (cold compile is ~15 min; warm boot is ~1 min). Host dir
# on local NVMe for speed. Keyed by the image ID so a new image (different kernels/ABI)
# starts a fresh cache instead of reusing stale .so's; set JIT_CACHE_HOST to override, or
# JIT_CACHE_PERSIST=0 to disable and fall back to an ephemeral in-container cache.
if [[ "${JIT_CACHE_PERSIST:-1}" == "1" ]]; then
    _IMG_KEY="$(docker image inspect --format '{{.Id}}' "$DOCKER_IMAGE_NAME" 2>/dev/null | sed 's/^sha256://; s/[^a-f0-9]//g' | cut -c1-12)"
    _IMG_KEY="${_IMG_KEY:-noimg}"
    _JIT_CACHE_HOST="${JIT_CACHE_HOST:-/mnt/m2m_nobackup/${USER}/vllm_jit_cache/${_IMG_KEY}}"
    mkdir -p "$_JIT_CACHE_HOST"/{aiter_jit,triton,vllm,comgr} 2>/dev/null || true
    _JIT_CACHE_MOUNT="-v ${_JIT_CACHE_HOST}:/opt/vllm_cache"
    echo "JIT cache (persistent, image ${_IMG_KEY}): ${_JIT_CACHE_HOST} -> /opt/vllm_cache"
else
    _JIT_CACHE_MOUNT=""
fi

# host RDMA library mounts
_RDMA_MOUNTS=""
_LIBDIR=/usr/lib/x86_64-linux-gnu
for _lib in libibverbs.so libibverbs.so.1 librdmacm.so librdmacm.so.1; do
    [ -f "$_LIBDIR/$_lib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_LIBDIR/$_lib:$_LIBDIR/$_lib:ro"
done
for _vlib in $_LIBDIR/libibverbs.so.1.* $_LIBDIR/librdmacm.so.1.*; do
    [ -f "$_vlib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_vlib:$_vlib:ro"
done
for _pattern in libmlx5.so* libionic*.so* libbnxt_re*.so* libefa.so* libhns.so*; do
    for _vlib in $_LIBDIR/${_pattern}; do
        [ -f "$_vlib" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_vlib:$_vlib:ro"
    done
done
[ -d "$_LIBDIR/libibverbs" ] && _RDMA_MOUNTS="$_RDMA_MOUNTS -v $_LIBDIR/libibverbs:$_LIBDIR/libibverbs:ro"
[ -d /etc/libibverbs.d ]     && _RDMA_MOUNTS="$_RDMA_MOUNTS -v /etc/libibverbs.d:/etc/libibverbs.d:ro"

# k3: optional instrumented MoRIIO connector overlay for the K3_MORIIO_TRACE probe.
# Bind-mounts a host copy of moriio_connector.py over the image baked-in path so we can
# localize the disagg KV-transfer/notify stall WITHOUT rebuilding the image. No-op unless
# K3_MORIIO_TRACE_SRC points at an existing (in-container-visible) file.
_MORIIO_TRACE_MOUNT=""
if [ -n "${K3_MORIIO_TRACE_SRC:-}" ] && [ -f "${K3_MORIIO_TRACE_SRC}" ]; then
    _MORIIO_TRACE_MOUNT="-v ${K3_MORIIO_TRACE_SRC}:/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py:ro"
    echo "MoRIIO trace overlay: ${K3_MORIIO_TRACE_SRC} -> image moriio_connector.py (K3_MORIIO_TRACE=${K3_MORIIO_TRACE:-0})"
fi

# k3 F40: optional kimi_k3 reasoning-parser overlay. Bind-mounts a host copy of
# kimi_k3_reasoning_parser.py over the image baked-in path so the chat-endpoint
# content-extraction fix applies at serve time WITHOUT rebuilding the image.
# No-op unless K3_PARSER_SRC points at an existing (in-container-visible) file.
_PARSER_OVERLAY_MOUNT=""
if [ -n "${K3_PARSER_SRC:-}" ] && [ -f "${K3_PARSER_SRC}" ]; then
    _PARSER_OVERLAY_MOUNT="-v ${K3_PARSER_SRC}:/usr/local/lib/python3.12/dist-packages/vllm/reasoning/kimi_k3_reasoning_parser.py:ro"
    echo "K3 parser overlay: ${K3_PARSER_SRC} -> image kimi_k3_reasoning_parser.py (K3F40_TRACE=${K3F40_TRACE:-0})"
fi

_MOE_OVERLAY_MOUNTS=""
if [ -n "${K3_MOE_SRC_DIR:-}" ]; then
    _MB=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe
    [ -f "${K3_MOE_SRC_DIR}/oracle_mxfp4.py" ] && _MOE_OVERLAY_MOUNTS="$_MOE_OVERLAY_MOUNTS -v ${K3_MOE_SRC_DIR}/oracle_mxfp4.py:$_MB/oracle/mxfp4.py:ro"
    [ -f "${K3_MOE_SRC_DIR}/aiter_mxfp4_w4a8_moe.py" ] && _MOE_OVERLAY_MOUNTS="$_MOE_OVERLAY_MOUNTS -v ${K3_MOE_SRC_DIR}/aiter_mxfp4_w4a8_moe.py:$_MB/experts/aiter_mxfp4_w4a8_moe.py:ro"
    [ -f "${K3_MOE_SRC_DIR}/rocm_aiter_moe.py" ] && _MOE_OVERLAY_MOUNTS="$_MOE_OVERLAY_MOUNTS -v ${K3_MOE_SRC_DIR}/rocm_aiter_moe.py:$_MB/experts/rocm_aiter_moe.py:ro"
    echo "K3 MoE overlay (F49): ${K3_MOE_SRC_DIR} -> fused_moe {oracle/mxfp4,experts/aiter_mxfp4_w4a8_moe,experts/rocm_aiter_moe}.py"
fi

# k3 B2fix: optional MoRIIO engine overlay. Bind-mounts a host copy of moriio_engine.py
# over the image baked-in path so the repaired K3_WRITE_READBACK read-after-write RDMA
# fence (batch_read with Sequence args) applies at serve time WITHOUT rebuilding the
# image. No-op unless K3_ENGINE_SRC points at an existing (in-container-visible) file.
_ENGINE_OVERLAY_MOUNT=""
if [ -n "${K3_ENGINE_SRC:-}" ] && [ -f "${K3_ENGINE_SRC}" ]; then
    _ENGINE_OVERLAY_MOUNT="-v ${K3_ENGINE_SRC}:/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py:ro"
    echo "MoRIIO engine overlay (B2fix): ${K3_ENGINE_SRC} -> image moriio_engine.py"
fi

docker run --rm \
    --device /dev/dri --device /dev/kfd --device /dev/infiniband \
    --network host --ipc host --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --privileged \
    -v $HOME:$HOME \
    -v /shared_inference:/shared_inference \
    -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
    -v $HOME/.ssh:/root/.ssh \
    --shm-size "${DOCKER_SHM_SIZE:-256G}" --ulimit nofile=524288:524288 --ulimit memlock=-1:-1 \
    -v ${LOG_PATH}:/run_logs \
    -v $NIXL_REPO_DIR:$NIXL_COOKBOOK_PATH \
    -v /tmp/vllm_cache:/tmp/vllm_cache \
    ${_JIT_CACHE_MOUNT} \
    ${_MORIIO_TRACE_MOUNT} \
    ${_PARSER_OVERLAY_MOUNT} \
    ${_MOE_OVERLAY_MOUNTS} \
    ${_ENGINE_OVERLAY_MOUNT} \
    $_RDMA_MOUNTS \
    --entrypoint /bin/bash \
    -e SLURM_JOB_ID=$SLURM_JOB_ID \
    -e NNODES=$NNODES \
    -e NODE_RANK=$NODE_RANK \
    -e MASTER_ADDR=$MASTER_ADDR \
    -e MASTER_PORT=$MASTER_PORT \
    -e MODEL_PATH=$MODEL_PATH \
    -e NIXL_COOKBOOK_PATH=$NIXL_COOKBOOK_PATH \
    -e xP=$xP -e yD=$yD \
    -e USER_NAME=$USER \
    -e MODEL_NAME=$MODEL_NAME \
    -e BENCHMARK_ITR=${BENCHMARK_ITR:-1} \
    -e BENCHMARK_CON="${BENCHMARK_CON}" \
    -e BENCHMARK_COMBINATIONS="${BENCHMARK_COMBINATIONS}" \
    -e IPADDRS=$IPADDRS \
    -e CONNECTOR=$CONNECTOR \
    -e WIDE_EP=$WIDE_EP \
    ${EP_BACKEND:+-e EP_BACKEND=$EP_BACKEND} \
    ${DECODE_MORI_BACKEND:+-e DECODE_MORI_BACKEND=$DECODE_MORI_BACKEND} \
    ${PREFILL_MORI_BACKEND:+-e PREFILL_MORI_BACKEND=$PREFILL_MORI_BACKEND} \
    ${KV_CACHE_MEMORY_BYTES:+-e KV_CACHE_MEMORY_BYTES=$KV_CACHE_MEMORY_BYTES} \
    ${MAX_NUM_BATCHED_TOKENS:+-e MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS} \
    ${MAX_MODEL_LEN:+-e MAX_MODEL_LEN=$MAX_MODEL_LEN} \
    ${KV_CACHE_DTYPE:+-e KV_CACHE_DTYPE=$KV_CACHE_DTYPE} \
    ${KV_BLOCK_SIZE:+-e KV_BLOCK_SIZE=$KV_BLOCK_SIZE} \
    -e VLLM_ROCM_USE_AITER=${VLLM_ROCM_USE_AITER:-1} \
    -e VLLM_ROCM_USE_AITER_MLA=${VLLM_ROCM_USE_AITER_MLA:-0} \
    -e VLLM_ROCM_USE_AITER_PAGED_ATTN=${VLLM_ROCM_USE_AITER_PAGED_ATTN:-0} \
    -e VLLM_ROCM_USE_AITER_RMSNORM=${VLLM_ROCM_USE_AITER_RMSNORM:-1} \
    -e VLLM_USE_AITER_TRITON_SILU_MUL=${VLLM_USE_AITER_TRITON_SILU_MUL:-0} \
    -e PROXY_TYPE=${PROXY_TYPE:-vllm_router} \
    -e ROUTER_PORT=${ROUTER_PORT:-30000} \
    ${ROUTER_BINARY:+-e ROUTER_BINARY=$ROUTER_BINARY} \
    ${GPU_MEMORY_UTILIZATION:+-e GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION} \
    -e GPUS_PER_NODE=${GPUS_PER_NODE:-8} \
    -e MORI_SOCKET_IFNAME=${MORI_SOCKET_IFNAME:-eth0} \
    -e DISTRIBUTED_TIMEOUT_SECONDS=${DISTRIBUTED_TIMEOUT_SECONDS:-7200} \
    -e VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-300000} \
    -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-3600} \
    -e PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:False} \
    -e PYTORCH_HIP_ALLOC_CONF=${PYTORCH_HIP_ALLOC_CONF:-expandable_segments:False} \
    -e HSA_ENABLE_IPC_MODE_LEGACY=${HSA_ENABLE_IPC_MODE_LEGACY:-0} \
    -e MORI_GPU_ARCHS=${MORI_GPU_ARCHS:-gfx942} \
    -e HSA_NO_SCRATCH_RECLAIM=${HSA_NO_SCRATCH_RECLAIM:-1} \
    ${DECODE_CUDAGRAPH_MODE:+-e DECODE_CUDAGRAPH_MODE=$DECODE_CUDAGRAPH_MODE} \
    ${CUDAGRAPH_CAPTURE_SIZES:+-e CUDAGRAPH_CAPTURE_SIZES="$CUDAGRAPH_CAPTURE_SIZES"} \
    ${K3_MORIIO_TRACE:+-e K3_MORIIO_TRACE=$K3_MORIIO_TRACE} \
    ${K3F40_TRACE:+-e K3F40_TRACE=$K3F40_TRACE} \
    ${K3F40_TRACE_FILE:+-e K3F40_TRACE_FILE=$K3F40_TRACE_FILE} \
     \
    ${K3_WRITE_READBACK:+-e K3_WRITE_READBACK=$K3_WRITE_READBACK} \
    ${K3_WRITE_READBACK_BYTES:+-e K3_WRITE_READBACK_BYTES=$K3_WRITE_READBACK_BYTES} \
    ${K3_WRITE_READBACK_MAX:+-e K3_WRITE_READBACK_MAX=$K3_WRITE_READBACK_MAX} \
    ${K3_WRITE_FENCE:+-e K3_WRITE_FENCE=$K3_WRITE_FENCE} \
    ${K3_WRITE_FENCE_MS:+-e K3_WRITE_FENCE_MS=$K3_WRITE_FENCE_MS} \
    ${K3_WRITE_DEVSYNC:+-e K3_WRITE_DEVSYNC=$K3_WRITE_DEVSYNC} \
     \
     \
     \
     \
     \
     \
     \
    ${BENCHMARK_SCRIPT_FILE:+-e BENCHMARK_SCRIPT_FILE=$BENCHMARK_SCRIPT_FILE} \
    ${VLLM_TORCH_PROFILER_DIR:+-e VLLM_TORCH_PROFILER_DIR=$VLLM_TORCH_PROFILER_DIR} \
    ${VLLM_LOGGING_LEVEL:+-e VLLM_LOGGING_LEVEL=$VLLM_LOGGING_LEVEL} \
    --name $DOCKER_CONT_NAME \
    $DOCKER_IMAGE_NAME -c "
        mkdir -p /run_logs/${SLURM_JOB_ID}
        $RUN_FILE_FULL 2>&1 | tee /run_logs/${SLURM_JOB_ID}/pd_vllm_bench_NODE${NODE_RANK}.log
    "
