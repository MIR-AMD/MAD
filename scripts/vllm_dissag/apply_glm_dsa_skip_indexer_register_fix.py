#!/usr/bin/env python3
"""Do not RDMA-register GLM-5.1 DSA indexer k_caches with MoRIIO.

DIAGNOSTIC (jobs 216286 / 216287 / 216294):
  Recent-source first WRITE dies after Triton _indexer_k_quant_and_cache_kernel
  / _cp_gather_indexer_quant_cache_kernel. With persistent MLA on, that is GPU
  node-2 after LoadKernel mla_a8w8_*_ps. With persist forced off (216294), AITER
  prints "fp8/fp8 with gqa_ratio=64 only supports persistent mode" and
  VllmWorker-0 still dies (exit None) — so split-KV is not a fallback, and the
  indexer kernels still ran against MoRI-registered indexer k_cache.

  Hypothesis: those Triton kernels (or the persistent MLA .co that reads the
  same tensors) cannot write an RDMA-mapped `.indexer.` k_cache. Filter those
  caches out of register_kv_caches so they stay ordinary HIP allocations.
  Prefill still computes indexer locally. Decode does not receive indexer KV
  (smoke prompt is short enough that decode recomputes it). Long-ctx PD will
  be wrong until this is reverted.

Mutually exclusive with apply_glm_dsa_moriio_indexer_transfer_fix.py (that
patcher pairs and WRITEs the indexer caches). moriio.sh picks one.

Idempotent + anchor-based. Missing anchor is a hard error (would silently
keep indexer caches RDMA-mapped).

Usage: apply_glm_dsa_skip_indexer_register_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
OLD = "        self.kv_caches = kv_caches  # layer name to kv cache"
NEW = """        # Skip RDMA-register of DSA indexer k_caches (216286/216294). Rebind
        # before assignment so num_layers / RegisterRdmaMemoryRegion see only
        # main MLA caches. Decode recomputes indexer on the smoke prompt.
        _n_idx = sum(1 for _k in kv_caches if ".indexer." in _k)
        if _n_idx:
            kv_caches = {k: v for k, v in kv_caches.items() if ".indexer." not in k}
            logger.info(
                "[moriio] skipped RDMA-register of %d DSA indexer caches (%d remain)",
                _n_idx,
                len(kv_caches),
            )
        self.kv_caches = kv_caches  # layer name to kv cache"""


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[glm-idx-skip] {REL} not found -- skipping (connector layout differs).")
        return 0

    src = open(path).read()
    if "skipped RDMA-register of" in src:
        print("[glm-idx-skip] already patched -- no-op.")
        return 0
    if OLD not in src:
        print("[glm-idx-skip] ERROR: register_kv_caches assignment anchor not found "
              "-- refusing to boot with indexer caches still RDMA-mapped.",
              file=sys.stderr)
        return 1

    src = src.replace(OLD, NEW, 1)
    try:
        open(path, "w").write(src)
    except OSError as e:
        print(f"[glm-idx-skip] ERROR: write failed for {path}: {e}", file=sys.stderr)
        return 1

    chk = open(path).read()
    if "skipped RDMA-register of" not in chk or OLD not in chk:
        print("[glm-idx-skip] ERROR: post-write verification failed.", file=sys.stderr)
        return 1

    try:
        import py_compile
        py_compile.compile(path, doraise=True)
    except Exception as e:  # noqa: BLE001
        print(f"[glm-idx-skip] ERROR: patched file fails to compile: {e}", file=sys.stderr)
        return 1

    print(f"[glm-idx-skip] patched skip-register of .indexer. caches in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
