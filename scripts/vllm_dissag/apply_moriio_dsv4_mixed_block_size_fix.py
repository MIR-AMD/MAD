#!/usr/bin/env python3
"""Allow mixed MLA/SWA block sizes in MoRIIO register (Flash 217473).

217463: indexer k_cache was 64 vs MLA 256. Flash skip-register dropped
`.indexer.` caches.

217473: skip ran (`skipped RDMA-register of 42 DSA indexer caches (125 remain)`)
then died on `model.layers.2.attn: 64 != 256`. Flash is hybrid sparse MLA
(block 256) + SWA (block 64). MoRIIO already has per-layer `block_lens` and
`compute_block_transfer_offsets`; the leftover uniform ValueError is the
TODO(tms) hybrid-attn guard.

Gated in connectors/moriio.sh to MODEL_NAME=DeepSeek-V4-Flash-FP8. GLM / DSV3 /
Hy3 never run this script. Do not set KV_BLOCK_SIZE=64. Idempotent. Missing
file -> skip. Found-old that fails to apply is a hard error.

Usage: apply_moriio_dsv4_mixed_block_size_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
MARKER = "DSV4-MIXED-BS"

OLD = """            if geometry.block_size != self.block_size:
                raise ValueError(
                    "MoRIIO KV cache block size mismatch for layer "
                    f"{layer_name}: {geometry.block_size} != {self.block_size}"
                )
"""

NEW = """            if geometry.block_size != self.block_size:
                # DSV4-MIXED-BS: 217473. Flash SWA .attn is block 64 after
                # indexer skip; MLA is 256. Per-layer block_lens / transfer
                # offsets already exist. Do not fail register.
                logger.warning(
                    "[dsv4-mixed-bs] layer %s block_size=%d != global %d; "
                    "registering with per-layer geometry",
                    layer_name,
                    geometry.block_size,
                    self.block_size,
                )
"""


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[dsv4-mixed-bs] {REL} not found under {sys.argv[1]} -- skipping.")
        return 0

    src = open(path).read()
    if MARKER in src:
        print(f"[dsv4-mixed-bs] already patched in {path} -- no-op.")
        return 0

    if OLD not in src:
        print(
            f"[dsv4-mixed-bs] ERROR: uniform block_size ValueError anchor "
            f"missing in {path}.",
            file=sys.stderr,
        )
        for i, line in enumerate(src.splitlines(), 1):
            if "block size mismatch" in line or "geometry.block_size" in line:
                print(f"[dsv4-mixed-bs]   line {i}: {line.rstrip()}", file=sys.stderr)
        return 1

    src = src.replace(OLD, NEW, 1)
    if MARKER not in src:
        print(f"[dsv4-mixed-bs] ERROR: post-write missing marker in {path}.", file=sys.stderr)
        return 1
    if "MoRIIO KV cache block size mismatch" in src:
        print(
            f"[dsv4-mixed-bs] ERROR: post-write still raises mismatch in {path}.",
            file=sys.stderr,
        )
        return 1
    tmp = path + ".dsv4mbs"
    with open(tmp, "w") as f:
        f.write(src)
    os.replace(tmp, path)
    print(
        f"[dsv4-mixed-bs] patched: mixed MLA/SWA block sizes register in {path}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
