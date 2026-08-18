# GLM-5.* DSA runtime patchers (fallbacks for older images)

These are **fallbacks, not part of the validated serving path.**

Every fix here is already carried **in-source** by the image built from
`docker/vllm_disagg_inference.glmv5.1.ubuntu.amd.Dockerfile`, whose
`VLLM_REF=glm5.1-dsa-wideEP_on_vllm-v0.27` is the real contract for GLM-5.1-FP8.
They exist only so an image that **predates** that ref can still be driven —
typically an older lab image such as
`rocmshared/pytorch-private:vllm-wideep_06_29_2026_..._mori_v1.2.1_..._mori121`.

The validated model card `pyt_vllm_disagg_mori_glm-5.1-fp8` in
`scripts/vllm_dissag/models.json` sets **`GLM_SKIP_PATCHERS=1`**, so on the
supported path `connector_runtime_patch()` in `connectors/moriio.sh` returns
immediately and **none of these scripts run.** Prefer rebuilding the image over
enabling them.

All patchers are idempotent, anchor-based, and self-skip with rc 0 when their
anchor is absent, so they no-op cleanly on an image where the fix is already
native. A hard failure is fatal by design: a half-patched GLM emits garbage or
stalls the disagg KV transfer.

## Applied by default (when `GLM_SKIP_PATCHERS` is not 1)

| Script | What it fixes |
| --- | --- |
| `apply_glm_dsa_moriio_dualkv_fix.py` | DSA has two KV caches per layer (main MLA latent ~576 vs indexer 128). The connector derived ONE global geometry from `first_kv_cache`, so indexer blocks moved with the wrong byte size and the RDMA read never reconciled. Adds per-layer geometry. |
| `apply_glm_dsa_moriio_engine_fix.py` | `MoRIIOEngine._prepare_transfer_plan` cached the transfer offset once and reused it for all 156 caches, writing the 78 indexer layers with main-MLA geometry. Makes the offset per-layer. |
| `apply_glm_dsa_moriio_gate_fix.py` | vLLM never calls `save_kv_layer` for the 78 indexer caches, so `writes_done` caps at 78 of `num_layers=156` and the producer's completion notify never fires. Fixes the gate denominator. |
| `apply_glm_moriio_abort_guard_fix.py` | Aborting a request before its peer handshake completes raises `AttributeError` on a `None` `peer_zmq`; only `ValueError` was caught, so EngineCore died and cascaded across decode workers. |
| `apply_glm_aiter_sampling_oob_fix.py` | AITER TopP/TopK sampling kernel (ROCm/aiter #3658): uninitialized `last_valid_id` causes an OOB read and HSA page fault, plus a hang cap. Root cause of the 8k prefill/decode crash on aiter `post3`. |

## Opt-in, because they REGRESS a current image

Both target defects that the pinned image has already fixed **differently**, so
forcing them on that image actively breaks it. Enable one only when you have
confirmed your image genuinely has the underlying bug.

- **`GLM_PERSIST_GATE=1`** → `apply_glm_dsa_persistent_kernel_gate_fix.py`
  (vLLM #47567: gate off the AITER persistent sparse-MLA kernel for
  chunked-prefill continuations). vLLM #47766 superseded this by keeping
  persistent MLA on. Forcing the non-persistent path **aborts at
  `asm_mla.cu:945`** on the current image, because fp8/fp8 `gqa_ratio=64` has no
  non-persistent kernel. Use only on an image predating #47766.

- **`GLM_DSA_SENTINEL_FIX=1`** → `apply_glm_dsa_kernel_fix.py`
  (vLLM #45324: map invalid token slots to `-1` instead of `0` in
  `_convert_req_index_to_global_index_kernel`). The pinned image ships `0`
  **deliberately**, because aiter's `mla_decode_fwd` dereferences the index — so
  `-1` yields **`hipErrorIllegalAddress`** on short disagg decode. Use only on an
  older image that genuinely has the #45324 `!!!`-output bug.

## Opt-in diagnostics (non-fatal)

- **`GLM_INDEXER_WARMUP=1`** → `apply_glm_dsa_indexer_warmup_fix.py`
  Force-compiles the seq-length-specialized DSA indexer Triton kernels at boot so
  they never JIT mid-inference. Opt-in because it drives a large (>=8k) prefill at
  boot: where that forward faults, it makes the fault deterministic at boot
  instead of on the first real request.

- **`GLM_INSTRUMENT=1`** → `apply_glm_dsa_moriio_instrument.py`
  Logging only, no behavior change: dumps registered KV-cache layer names/shapes
  and the layers that actually trigger a write, to confirm dual-KV accounting.
  Diagnostic; do not leave enabled in production.
