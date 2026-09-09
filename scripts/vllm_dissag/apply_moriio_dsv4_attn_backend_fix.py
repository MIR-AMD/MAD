#!/usr/bin/env python3
"""Skip dense MLA selector in MoRIIO for DeepSeek-V4 Flash (jobs 217457/217460).

MoRIIOConnectorWorker.__init__ calls get_attn_backend(..., use_mla=True) with
use_sparse defaulting False. ROCm then only tries ROCM_AITER_MLA / TRITON_MLA /
ROCM_AITER_TRITON_MLA. Those reject kv_cache_dtype=fp8_ds_mla.

DSV4 remaps any --kv-cache-dtype fp8* (including fp8_e4m3) to fp8_ds_mla
(attention.py "Using DeepSeek's fp8_ds_mla KV cache format."). The model's
backend is ROCM_FLASHMLA_SPARSE_DSV4, which lists fp8_ds_mla. The connector
only needs backend.get_name() for handshake metadata.

Gated in connectors/moriio.sh to MODEL_NAME=DeepSeek-V4-Flash-FP8. GLM / DSV3 /
Hy3 never run this script. Idempotent. Missing file -> skip. Found-old that
fails to apply is a hard error (Flash would die the 217460 way).

Usage: apply_moriio_dsv4_attn_backend_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
MARKER = "DSV4-ATTN-FIX"

OLD = """        backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.cache_config.cache_dtype,
            use_mla=self.use_mla,
        )
"""

NEW = """        # DSV4-ATTN-FIX: 217457/217460. DSV4 remaps fp8* to fp8_ds_mla.
        # Generic get_attn_backend(use_mla=True) only tries dense ROCm MLA
        # (use_sparse defaults False) which reject fp8_ds_mla. Model backend
        # is ROCM_FLASHMLA_SPARSE_DSV4. Connector only needs get_name().
        _arch = tuple(getattr(self.model_config, "architectures", None) or ())
        if str(self.cache_config.cache_dtype) == "fp8_ds_mla" or any(
            "DeepseekV4" in a for a in _arch
        ):
            from vllm.models.deepseek_v4.amd.rocm import (
                DeepseekV4ROCMAiterMLASparseBackend,
            )
            backend = DeepseekV4ROCMAiterMLASparseBackend
            logger.info(
                "[dsv4-attn] using %s for cache_dtype=%s arch=%s "
                "(skip dense MLA selector)",
                backend.get_name(),
                self.cache_config.cache_dtype,
                _arch,
            )
        else:
            backend = get_attn_backend(
                self.model_config.get_head_size(),
                self.model_config.dtype,
                self.cache_config.cache_dtype,
                use_mla=self.use_mla,
            )
"""


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[dsv4-attn] {REL} not found under {sys.argv[1]} -- skipping.")
        return 0

    src = open(path).read()
    if MARKER in src:
        print(f"[dsv4-attn] already patched in {path} -- no-op.")
        return 0

    if OLD not in src:
        print(
            f"[dsv4-attn] ERROR: get_attn_backend() anchor missing in {path}.",
            file=sys.stderr,
        )
        for i, line in enumerate(src.splitlines(), 1):
            if "get_attn_backend" in line:
                print(f"[dsv4-attn]   line {i}: {line.rstrip()}", file=sys.stderr)
        return 1

    src = src.replace(OLD, NEW, 1)
    if MARKER not in src or "DeepseekV4ROCMAiterMLASparseBackend" not in src:
        print(f"[dsv4-attn] ERROR: post-write missing marker in {path}.", file=sys.stderr)
        return 1
    tmp = path + ".dsv4tmp"
    with open(tmp, "w") as f:
        f.write(src)
    os.replace(tmp, path)
    print(
        f"[dsv4-attn] patched: MoRIIO uses ROCM_FLASHMLA_SPARSE_DSV4 for "
        f"fp8_ds_mla in {path}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
