# CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
###############################################################################
#
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
#
#################################################################################
# =============================================================================
# vllm_disagg_recent_source_v0280_aiter_1d872fa_mori6fcf6b3_triton_kernels.ubuntu.amd.Dockerfile
#
# TOP RUNG of a three-image ladder. Each rung moves exactly one variable, so a
# regression can be attributed instead of guessed:
#
#   rung 1  ..._d626108b_aiter_1d872fa_mori624002_...   (--image mori624002)
#           vLLM d626108b (nightly)  +  MoRI 624002c8
#           The current DSV4 vehicle: 218778 / 222577 / 223837 / 223903 / 223991.
#
#   rung 2  ..._d626108b_aiter_1d872fa_mori6fcf6b3_...  <-- MoRI is the only delta
#           vLLM d626108b (nightly)  +  MoRI 6fcf6b3
#
#   rung 3  THIS FILE                                    <-- vLLM is the only delta
#           vLLM v0.28.0 (release)   +  MoRI 6fcf6b3
#
# **rung 2 is SHELVED as of 2026-08-27 and is not being built.** This image is
# therefore scored directly against mori624002 (rung 1), which is a deliberate
# TWO-variable move: both vLLM and MoRI change at once. That is an accepted
# trade, not an oversight — if this image is clean, attribution was never
# needed. If it regresses, build rung 2 to split vLLM from MoRI before drawing
# any conclusion about which pin did it. Do not attribute a rung-3 regression to
# either pin on its own.
#
# AITER, flydsl and triton_kernels are byte-identical across all three rungs, so
# they are never a candidate explanation for a difference here.
#
# DELTA 1 — BASE + vLLM: nightly-d626108b (main, 2026-08-19) -> release v0.28.0
#   (tag = 2cf0a6915ce544dc493a0990f2ea38d81601128a, image pushed 2026-08-26).
#   READ THIS BEFORE ASSUMING IT IS AN UPGRADE: v0.28.0 is a RELEASE BRANCH, and
#   `git compare d626108b...v0.28.0` says **diverged: 13 ahead, 150 behind**. It
#   carries 13 release-only commits but is 150 commits of main BEHIND d626108b.
#   The trade is "unpinned nightly" -> "reproducible release", not "newer vLLM".
#
#   What does NOT move: v0.28.0's docker/Dockerfile.rocm_base is byte-identical
#   in every pin to d626108b's — rocm/dev-ubuntu-22.04:7.2.3-complete, Python
#   3.12, ROCm triton f0b55c0, pytorch 6bbd260 (release/2.12), FA 0e60e394, Hub
#   AITER v0.1.19, Hub MoRI v1.1.0. So the ROCm / torch / Python layer under us
#   is unchanged and the runtime patchers' site-packages paths still resolve.
#
#   What does NOT move, part 2 — the MoRI-IO connector stack is BYTE-IDENTICAL
#   on both refs (md5 checked, 2640 lines each):
#     distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py
#     distributed/kv_transfer/kv_connector/v1/moriio/moriio_common.py
#     distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py
#     distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py
#   as are model_executor/models/deepseek_v4.py,
#   model_executor/layers/fused_moe/prepare_finalize/mori.py and
#   v1/engine/detokenizer.py. Every DSV4 connector patcher (attn_transfer,
#   supports_hma, transfer_gate, storage_span, write_diag, rid_map, chunked
#   prefill, backend_detect, skip_noncontiguous, decode_diag) lands unchanged.
#
#   What DOES move, so re-read the patcher log on first boot: v1/core/sched/
#   scheduler.py, v1/worker/gpu_model_runner.py, v1/worker/gpu_worker.py,
#   v1/executor/multiproc_executor.py, v1/engine/core.py and the two MLA sparse
#   backends (v1/attention/backends/mla/flashmla_sparse.py,
#   .../rocm_aiter_mla_sparse.py) all differ. Those are the pd_debug anchors
#   (already down 4 on d626108b) plus apply_moriio_dsv4_kv_hash_fix.py and the
#   GLM dsa patchers. Check pd_vllm_bench_NODE*.log for WARN/MARKER lines before
#   trusting a score off this image.
#
# DELTA 2 — MoRI 624002c897a3 (2026-08-21) -> 6fcf6b386786fe75fdce5b8b502e22d7f344e1f0.
#   This delta is vs rung 1 only; rung 2 already carries it, which is the whole
#   point of building rung 2 first.
#   (main HEAD 2026-08-26 15:04 UTC, +12 commits, 0 behind). Recorded tag basis:
#   nearest release tag is **v1.2.2 = dafdcfcf1e27**, and this HEAD is +74 past
#   it, so pip will stamp 1.2.3.dev<N>+g6fcf6b386 — the tag string in
#   /app/versions.txt is NOT the pin, MORI_REF is.
#
#   The +12, two of which are why this bump is worth an image:
#     dfbe7e61cabe 08-24 (bugfix): intranode bw check in mixed-vendor environment (#590)
#     436af8042a92 08-24 docs(skills): known-issues skill for VMM/XGMI kernel-config trap (#595)
#     1951dc56d284 08-24 [JAX] drop local xla_ffi headers (#584)
#     5236e437e7fb 08-24 Perf/ep disp 1250 ship (#599)
#     4e801d4aeecf 08-24 feat(EPv2): forward a per-token scale row with dispatch (#593)
#   * 404255afed41 08-25 perf: io hotpath improvements (#583)      <-- our KV path
#     d335c033719d 08-25 feat(EPv2): extend support for rack-level wide-ep (#597)
#     901907da4b88 08-25 feat(shmem): MORI_ENABLE_RAIL_ONLY, restrict QPs to same-rail peers (#591)
#     916d2a46ae13 08-25 bench(cco): P2P latency probe, rename xgmi_bw -> ualoe_metric (#592)
#     fda767997242 08-26 fix(security): harden bootstrap and CI (#605)
#   * 799a046b1a98 08-26 [AMD][DSV4] fix: recognize fp4_blockwise in AUTO tuning config (#600)
#     6fcf6b386786 08-26 allocator: register mori as a torch SymmetricMemory backend (#544)
#
#   #583 is src/io — the exact path MoRIIOConnector writes through — and #600 is
#   a DSV4 quant-config fix. #544 touches the allocator and #591 adds a new QP
#   restriction env (MORI_ENABLE_RAIL_ONLY, unset here = old behaviour). Both are
#   opt-in-shaped but they are the two to suspect if RDMA regresses vs 624002c8.
#
# Do NOT "upgrade" MoRI to v1.2.1 because vllm-project/vllm#48989 validated
# DeepSeek-V4-Pro on that tag. v1.2.1 is e31d426a13e9 (2026-06-25), 115 commits
# behind this pin — adopting it drops the EPv2 kernel/tune work WideEP depends
# on. That PR is also -tp 8 with its WideEP CI commented out, so its MoRI tag
# says nothing about the EP path.
#
# Locked pins (only BASE/vLLM and MoRI moved vs mori624002):
#   - BASE  -> vllm/vllm-openai-rocm:v0.28.0                <-- CHANGED
#   - vLLM  -> 2cf0a6915ce544dc493a0990f2ea38d81601128a (tag v0.28.0)  <-- CHANGED
#   - MoRI  -> ROCm/mori @ 6fcf6b386786… (main 2026-08-26)  <-- CHANGED
#   - AITER -> ROCm/aiter @ 1d872fa07aadfcbb199851fe17e6822efaceca61  (same)
#   - flydsl -> ==0.3.1  (AITER 1d872fa setup.py FLYDSL_VERSION)      (same)
#   - triton_kernels wheel -> ROCm/triton @ 0f380657                  (same)
#
# 0f380657 is still correct on v0.28.0: its docker/Dockerfile.rocm sets
# ROCM_TRITON_KERNELS_COMMIT=0f380657dbf3ee86eb57558ff71df24f03b5d4e7 and greps
# cmake/external_projects/triton_kernels.cmake for the same TAG. It also ships
# the tree at /opt/rocm-triton-kernels with TRITON_KERNELS_SRC_DIR set; block 4
# still installs the wheel into site-packages, which wins the import.
#
# flydsl==0.3.1 is still baked and still asserted: AITER pip is --no-deps, so
# flydsl is NOT pulled from aiter's install_requires. 217850 (tag without
# fd031) died at 2k NIAH with
#   ImportError: Unsupported `flydsl` version: expected >=`0.2.4`, got `0.1.8`.
# Do not drop the assert.
#
# Build on a REMOTE host (not WSL, not login useocpslog-002). Repo root:
#   docker pull vllm/vllm-openai-rocm:v0.28.0
#   docker build -f docker/vllm_disagg_recent_source_v0280_aiter_1d872fa_mori6fcf6b3_triton_kernels.ubuntu.amd.Dockerfile \
#     -t rocm/pytorch-private:vllm-recent-source-basem-v0280-aiter-1d872fa-fd031-mori6fcf6b3-tk .
#   docker push rocm/pytorch-private:vllm-recent-source-basem-v0280-aiter-1d872fa-fd031-mori6fcf6b3-tk
#
# After push, on login (needs an --image alias in run_wideep_bench.sh):
#   DSV4_ENABLE_HMA=0 ./run_wideep_bench.sh niah dsv4pro 2p2d --image v0280
# PROXY_TYPE=moriio_toy. First cell off this image must be a curl smoke, then the
# HMA=0 2k+8k product stem, scored against 218778 — not a fresh 6-rung claim.
# GLM stays on e03: AITER 1d872fa is still newer than the 0.1.19/0.1.20 that
# GPU-faulted GLM on first WRITE in 216179 / 216286.
# =============================================================================

ARG BASE_IMAGE=vllm/vllm-openai-rocm:v0.28.0
FROM ${BASE_IMAGE}

ENTRYPOINT []
WORKDIR /app

ARG GFX_COMPILATION_ARCH="gfx942"
ARG PYTORCH_ROCM_ARCH="gfx942"
ARG MAX_JOBS=32

# Hub release bakes LEGACY=1; MoRI RDMA needs dmabuf (same as moriio.env).
ENV HSA_ENABLE_IPC_MODE_LEGACY=0

# -----------------------------------------------------------------------------
# 1. MoRI 6fcf6b386786 (main HEAD 2026-08-26), +12 commits over 624002c8.
#    Do NOT disable ionic/bnxt (ep:0 init deadlock on RoCE/mlx5).
#
#    libgrpc-dev / libgrpc++-dev / libprotobuf-dev / protobuf-compiler-grpc are
#    the UMBP distributed-control-plane deps, same as mori624002. MoRI setup.py
#    probes for the grpcpp + protobuf headers and for protoc / grpc_cpp_plugin
#    and gracefully disables UMBP when absent — keeping them installed holds
#    UMBP ON so MoRI is the only variable vs mori624002.
# -----------------------------------------------------------------------------
ARG MORI_REPO=https://github.com/ROCm/mori.git
ARG MORI_REF=6fcf6b386786fe75fdce5b8b502e22d7f344e1f0
ARG MORI_NEAREST_TAG=v1.2.2
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
# 2. AITER 1d872fa from source + flydsl==0.3.1. Unchanged vs mori624002.
#    Do not use Hub 0.1.19. Keep Hub ROCm triton (AITER_USE_SYSTEM_TRITON=1).
# -----------------------------------------------------------------------------
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_REF=1d872fa07aadfcbb199851fe17e6822efaceca61
ARG FLYDSL_PIN="flydsl==0.3.1"
ENV PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}
ENV AITER_USE_SYSTEM_TRITON=1
# Docker build has no GPU. AITER 1d872fa get_gfx() defaults GPU_ARCHS=native
# and runs rocminfo. GPU_ARCHS is RUN-local so serve still uses the live GPU.
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
assert v=='0.3.1', 'flydsl %s != 0.3.1 (217850: Hub 0.1.8 left in image)' % v; \
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
# 3. vLLM: same ref as BASE_IMAGE (tag v0.28.0). Full compile so the release
#    wheel baked into the Hub image is not the one that ends up imported.
# -----------------------------------------------------------------------------
ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=2cf0a6915ce544dc493a0990f2ea38d81601128a
ARG VLLM_TAG=v0.28.0
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
#    0f380657 is v0.28.0's own ROCM_TRITON_KERNELS_COMMIT, not Hub TRITON_BRANCH
#    f0b55c0 (that one is the ROCm triton runtime, kept via AITER_USE_SYSTEM_TRITON).
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
    python3 -c "import importlib.metadata as m, flydsl; v=m.version('flydsl'); assert v=='0.3.1', v; print('flydsl', v, 'OK', flydsl.__file__)" && \
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
