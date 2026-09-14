# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
#################################################################################
# =============================================================================
# vllm_disagg_inference.dsv4.v0290.ubuntu.amd.Dockerfile
#   DeepSeek-V4 Flash/Pro MoRI-EP WideEP disagg image (v0.29.0 vehicle).
#   PER-MODEL image, isolated from the base vllm_disagg_inference Dockerfile
#   and from glmv5.1, so DSV4 can pin its own vLLM/AITER/MoRI.
#   Traced separately from v0280
#   (docker/vllm_disagg_inference.dsv4.v0280.ubuntu.amd.Dockerfile).
#   Do not retag over v0280. Do not mix scores with --image v0280 cells.
#
#   PINS (kept in this header, not the filename; queried 2026-09-09):
#   - BASE  -> vllm/vllm-openai-rocm:v0.29.0
#              digest sha256:e5e47f6aaab675c252c381f0dac237b31b10d87bb74d092b07fb4065efd7f5a1
#              Hub amd64, pushed 2026-09-09. Tag 98dff2a81d747d1dba01a47f939f48c3526d4206.
#   - vLLM  -> 98dff2a81d747d1dba01a47f939f48c3526d4206 (tag v0.29.0)
#   - AITER -> ROCm/aiter main 10f8874dc2cd69c07ed84b5f125c27d12baccb10 (2026-09-09)
#              +229 over 1d872fa. Built FROM SOURCE. Not the UFB TheRock wheel.
#   - flydsl -> ==0.3.2  (AITER main setup.py FLYDSL_VERSION)
#   - MoRI  -> ROCm/mori main 07bdace2ff7306928871f85afd92f1d2aae13ad0 (2026-09-09)
#              +24 over 6fcf6b3. Nearest tag v1.2.3 = 879983bdbd8c (+a few on main).
#   - triton_kernels -> ROCm/triton @ 0f380657 (v0.29.0 Dockerfile.rocm, unchanged)
#   Image tag (do not retag):
#     rocm/pytorch-private:vllm-recent-source-basem-v0290-aiter-10f8874-mori07bdace-tk
#   Wrapper: --image v0290
#
# Hub v0.29.0 is STILL ROCm 7.2.3 (docker/Dockerfile.rocm_base
# rocm/dev-ubuntu-22.04:7.2.3-complete). Hub AITER_BRANCH/MORI_BRANCH in
# /app/versions.txt are still v0.1.19 / v1.1.0 residue — trust AITER_REF /
# MORI_REF written below.
#
# "AITER 10.1" here means latest AITER *main* (the same tree UFB stamps as
# amd_aiter-0.1.22+rocm10.1.0a…). We do NOT pip-install that TheRock wheel onto
# this image: the wheel is built against TheRock ROCm 10.1, this Hub base is
# 7.2.3. Source-build AITER against the image torch instead.
#
# FlyDSL pin is STILL REQUIRED. AITER is installed --no-deps, so setup.py's
# flydsl==0.3.2 is not pulled. Hub leftover flydsl 0.1.8 is 217850
# (ImportError: expected >=0.2.4). AITER main _MIN_FLYDSL_VERSION is still
# 0.2.4; setup.py wants exactly 0.3.2. Do not drop the uninstall+assert.
# 0.3.1 is the v0280 pin — too old for this AITER.
#
# Build on a REMOTE host (not WSL, not login useocpslog-002). Repo root:
#   docker pull vllm/vllm-openai-rocm:v0.29.0
#   docker build -f docker/vllm_disagg_inference.dsv4.v0290.ubuntu.amd.Dockerfile \
#     -t rocm/pytorch-private:vllm-recent-source-basem-v0290-aiter-10f8874-mori07bdace-tk .
#   docker push rocm/pytorch-private:vllm-recent-source-basem-v0290-aiter-10f8874-mori07bdace-tk
#
# Submit: --image v0290  (v0280 stays the in-flight validation default)
# PROXY_TYPE=moriio_toy. First cell is curl smoke, then HMA=0 2k+8k vs v0280.
# Runtime patchers may WARN on v0.29.0 — check pd_vllm_bench_NODE*.log.
# =============================================================================

ARG BASE_IMAGE=vllm/vllm-openai-rocm:v0.29.0
FROM ${BASE_IMAGE}

ENTRYPOINT []
WORKDIR /app

ARG GFX_COMPILATION_ARCH="gfx942"
ARG PYTORCH_ROCM_ARCH="gfx942"
ARG MAX_JOBS=32

# Hub release bakes LEGACY=1; MoRI RDMA needs dmabuf (same as moriio.env).
ENV HSA_ENABLE_IPC_MODE_LEGACY=0

# -----------------------------------------------------------------------------
# 1. MoRI main 07bdace2 (2026-09-09), +24 over v0280's 6fcf6b3.
#    Do NOT disable ionic/bnxt (ep:0 init deadlock on RoCE/mlx5).
#    grpc/protobuf keep UMBP ON (same as mori624002 / v0280).
# -----------------------------------------------------------------------------
ARG MORI_REPO=https://github.com/ROCm/mori.git
ARG MORI_REF=07bdace2ff7306928871f85afd92f1d2aae13ad0
ARG MORI_NEAREST_TAG=v1.2.3
ENV MORI_GPU_ARCHS=gfx942
RUN sed -i 's|http://|https://|g' /etc/apt/sources.list 2>/dev/null || true && \
    sed -i 's|http://|https://|g' /etc/apt/sources.list.d/*.list 2>/dev/null || true && \
    apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build ccache libssl-dev pkg-config curl ca-certificates \
        libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc && \
    pip install meson==0.64.0 "pybind11[global]" tqdm prettytable && \
    pip uninstall -y mori amd-mori amd_mori 2>/dev/null || true && \
    rm -rf /tmp/mori-src && \
    git clone --recursive "${MORI_REPO}" /tmp/mori-src && \
    cd /tmp/mori-src && git checkout "${MORI_REF}" && git submodule update --init --recursive && \
    pip install -r requirements-build.txt && \
    pip install . --no-build-isolation && \
    python3 -c "import mori, mori.ops as o; print('mori OK; kernels:', [n for n in dir(o.EpDispatchCombineKernelType) if not n.startswith('_')])" && \
    mkdir -p /app && echo "MORI_REF=${MORI_REF}@$(git -C /tmp/mori-src rev-parse HEAD)" >> /app/versions.txt && \
    echo "MORI_NEAREST_TAG=${MORI_NEAREST_TAG} (+$(git -C /tmp/mori-src rev-list --count ${MORI_NEAREST_TAG}..${MORI_REF} 2>/dev/null || echo '?') commits; tag is provenance only, MORI_REF is the pin)" >> /app/versions.txt && \
    echo "MORI_HEAD_DATE=$(git -C /tmp/mori-src log -1 --format=%cI ${MORI_REF})" >> /app/versions.txt && \
    echo "MORI_UMBP_DEPS=grpc+protobuf installed (protoc=$(command -v protoc || echo none), grpc_cpp_plugin=$(command -v grpc_cpp_plugin || echo none))" >> /app/versions.txt && \
    rm -rf /tmp/mori-src

# -----------------------------------------------------------------------------
# 2. AITER main 10f8874 from source + flydsl==0.3.2.
#    Do not use Hub 0.1.19. Keep Hub ROCm triton (AITER_USE_SYSTEM_TRITON=1).
#    Do not install UFB +rocm10.1.0a wheels (TheRock ABI; this base is 7.2.3).
# -----------------------------------------------------------------------------
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_REF=10f8874dc2cd69c07ed84b5f125c27d12baccb10
ARG FLYDSL_PIN="flydsl==0.3.2"
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ENV AITER_USE_SYSTEM_TRITON=1
# Docker build has no GPU. get_gfx() defaults GPU_ARCHS=native and runs rocminfo.
# GPU_ARCHS is RUN-local so serve still uses the live GPU.
# Do not import aiter in the flydsl check (cwd /tmp/aiter-src + rocminfo).
RUN echo "Compiling AITER from source: ${AITER_REPO}@${AITER_REF}" && \
    rm -rf /tmp/aiter-src && \
    git clone --recursive "${AITER_REPO}" /tmp/aiter-src && \
    cd /tmp/aiter-src && git checkout "${AITER_REF}" && \
    git submodule update --init --recursive && \
    export GPU_ARCHS=gfx942 CU_NUM=304 && \
    (pip uninstall -y amd_aiter amd-aiter aiter 2>/dev/null || true) && \
    pip install --no-build-isolation --no-deps -v . && \
    echo "AITER_REF=${AITER_REF}@$(git rev-parse HEAD) (built from source)" >> /app/versions.txt && \
    rm -rf /tmp/aiter-src && \
    pip show amd-aiter 2>/dev/null | head -5 && echo "AITER OK (from source)"
RUN pip uninstall -y flydsl 2>/dev/null || true && \
    pip install --no-deps -U "${FLYDSL_PIN}" && \
    python3 -c "\
import importlib.metadata as m, importlib.util, pathlib, flydsl; \
v=m.version('flydsl'); \
assert v=='0.3.2', 'flydsl %s != 0.3.2 (217850 class: Hub leftover; AITER main wants 0.3.2)' % v; \
spec=importlib.util.find_spec('aiter'); \
assert spec and spec.origin, spec; \
p=pathlib.Path(spec.origin).parent/'ops'/'flydsl'/'kernels'/'mqa_logits'/'fp8_mqa_logits.py'; \
assert p.is_file(), p; \
assert 'def flydsl_fp8_mqa_logits' in p.read_text(), p; \
print('flydsl', v, getattr(flydsl,'__version__',None), flydsl.__file__); \
print('aiter flydsl_fp8_mqa_logits source OK', p)" && \
    echo "FLYDSL_PIN=${FLYDSL_PIN}" >> /app/versions.txt && \
    python3 -c 'import importlib.metadata as m; print("FLYDSL_VER="+m.version("flydsl"))' >> /app/versions.txt
RUN rm -rf /opt/vllm_cache/aiter_jit /root/.aiter && echo "cleared stale AITER JIT cache"

# -----------------------------------------------------------------------------
# 3. vLLM: same ref as BASE_IMAGE (tag v0.29.0). Full compile so the Hub
#    wheel is not the one that ends up imported.
# -----------------------------------------------------------------------------
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=98dff2a81d747d1dba01a47f939f48c3526d4206
ARG VLLM_TAG=v0.29.0
ENV VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} \
    MAX_JOBS=${MAX_JOBS}
RUN rm -rf /tmp/vllm-src && \
    git clone "${VLLM_REPO}" /tmp/vllm-src && \
    cd /tmp/vllm-src && git checkout "${VLLM_REF}" && \
    echo "VLLM_REF=${VLLM_REF}@$(git rev-parse HEAD) (tag ${VLLM_TAG})" >> /app/versions.txt && \
    test -f vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py && \
    pip uninstall -y vllm 2>/dev/null || true && \
    pip install --no-deps --no-build-isolation -v . && \
    python3 -c "import vllm; print('vLLM', vllm.__version__, 'from', vllm.__file__)" && \
    rm -rf /tmp/vllm-src

# -----------------------------------------------------------------------------
# 4. triton_kernels wheel into site-packages (Flash MXFP4 import).
#    0f380657 is v0.29.0's own ROCM_TRITON_KERNELS_COMMIT.
# -----------------------------------------------------------------------------
ARG TRITON_KERNELS_REPO=https://github.com/ROCm/triton.git
ARG TRITON_KERNELS_REF=0f380657dbf3ee86eb57558ff71df24f03b5d4e7
RUN pip install -U build && \
    rm -rf /tmp/triton-src && \
    git clone --filter=blob:none "${TRITON_KERNELS_REPO}" /tmp/triton-src && \
    cd /tmp/triton-src && git checkout "${TRITON_KERNELS_REF}" && \
    test -f python/triton_kernels/triton_kernels/matmul_ogs.py && \
    cd python/triton_kernels && python3 -m build --wheel && \
    pip uninstall -y triton_kernels triton-kernels 2>/dev/null || true && \
    pip install --no-deps dist/*.whl && \
    python3 -c "from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig; \
pc = PrecisionConfig.__dataclass_fields__; \
assert 'weight_scale' in pc, list(pc); \
print('triton_kernels.matmul_ogs OK')" && \
    echo "TRITON_KERNELS_REF=${TRITON_KERNELS_REF}@$(git -C /tmp/triton-src rev-parse HEAD) (wheel)" >> /app/versions.txt && \
    rm -rf /tmp/triton-src

RUN echo "=== AITER check ===" && pip show amd-aiter 2>/dev/null | head -5 && \
    echo "=== flydsl check ===" && pip show flydsl && \
    python3 -c "import importlib.metadata as m, flydsl; v=m.version('flydsl'); assert v=='0.3.2', v; print('flydsl', v, 'OK', flydsl.__file__)" && \
    echo "=== MoRI check ===" && python3 -c "import mori; print('mori', mori.__version__)" && \
    python3 -c "import mori.io; print('mori.io OK')" && \
    python3 -c "import mori.ops; print('mori.ops OK')" && \
    echo "=== MoRI-IO connector present ===" && \
    python3 -c "import vllm, pathlib; \
p = pathlib.Path(vllm.__file__).parent/'distributed'/'kv_transfer'/'kv_connector'/'v1'/'moriio'/'moriio_connector.py'; \
assert p.is_file(), p; print('moriio_connector.py OK', p)" && \
    echo "=== triton_kernels check ===" && \
    python3 -c "import triton_kernels, triton_kernels.matmul_ogs as m; print(triton_kernels.__file__); print('matmul_ogs', m.__file__)" && \
    echo "Post-vLLM cross-check OK"

# -----------------------------------------------------------------------------
# 5. No vllm-router bake. Bring-up uses PROXY_TYPE=moriio_toy (Python).
# -----------------------------------------------------------------------------
RUN echo "PROXY=moriio_toy (no vllm-router baked; Ravi fork is spec only)" >> /app/versions.txt

ENV SKIP_RUNTIME_PATCH=1
ENV AITER_JIT_DIR=/opt/vllm_cache/aiter_jit \
    VLLM_CACHE_ROOT=/opt/vllm_cache/vllm \
    TRITON_CACHE_DIR=/opt/vllm_cache/triton \
    COMGR_CACHE_DIR=/opt/vllm_cache/comgr

RUN rm -rf /root/.mori /tmp/mori_jit_* && mkdir -p /root/.mori && \
    echo "JIT_SCRUBBED: /root/.mori + /tmp/mori_jit_* cleared at build end" >> /app/versions.txt

RUN echo "=== Build versions ===" && cat /app/versions.txt
