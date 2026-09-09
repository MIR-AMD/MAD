#!/usr/bin/env python3
"""WRITE ``model.layers.N.attn`` — the DSV4 full-attention MLA cache (218040).

218040 / 218041 / 218111: curl 3/3 but NIAH never retrieves. The transfer gate
sends ``swa_cache`` + ``compressor.state_cache`` + ``.indexer.`` and skips every
``model.layers.N.attn`` (Flash 41 of 43, Pro 61 of 61). 218042 group table:

  group=0 MLAAttentionSpec      block_size=256 sliding_window=None  <- .attn
  group=1/2 SlidingWindowMLASpec block_size=64  sliding_window=128  <- swa_cache
  group=3/4 SlidingWindowMLASpec block_size=4/8 sliding_window=8/128

``.attn`` is the only cache with ``sliding_window=None``. Skipping it leaves
decode with a 128-token window (config ``sliding_window: 128``), so a ~20-token
curl answers from the window and any needle past 128 tokens is invisible. The
Lightning Indexer (``index_topk`` 512) is transferred and still picks the right
tokens — it just points at KV that was never sent.

Why the gate skipped it: ``get_layer_transfer_geometry`` reads MLA 3D as
``(num_blocks, block_size, latent_dim)``. DSV4 compresses the group-0 page, so
dim[1] is *slots per page*, not tokens::

    config compress_ratios = [0,0,4,128,4,128,...,4,0]   # per layer
    even layers ratio   4 -> shape (19955,  64, 584)   64 * 4   = 256
    odd  layers ratio 128 -> shape (19955,   2, 584)    2 * 128 = 256

dim[0] is already the page count and is identical across group 0. The shipped
``block_len = dim[1] * slot_size_bytes`` and ``block_stride = stride[0]`` are
therefore already the correct per-page byte span — only the *reported*
``block_size`` is wrong, and the gate compares exactly that against 256.

``.attn.indexer.k_cache`` has the same compressed layout (ratio 4, dim[1]=64)
and is already force-transferred by the ``.indexer.`` branch, which is the
existing proof that the shipped byte math is right for these caches. This
patcher gives ``.attn`` the same treatment.

Do NOT reuse the 217546 skip-swa reasoning: that hang was the whole-storage MR
(fixed by the data_ptr span in apply_moriio_dsv4_skip_noncontiguous_register_fix).

Runtime-gated so one image serves both arms::

    DSV4_TRANSFER_ATTN=0   (default) 218040 behaviour, .attn skipped
    DSV4_TRANSFER_ATTN=1            transfer .attn

Must run AFTER apply_moriio_dsv4_transfer_gate_fix.py. Gated in
connectors/moriio.sh to DSV4 Flash/Pro. GLM / DSV3 / Hy3 never run this.
Idempotent. Missing gate anchors is a hard error.

Usage: apply_moriio_dsv4_attn_transfer_fix.py <vllm_install_dir>
       apply_moriio_dsv4_attn_transfer_fix.py --selftest
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
MARKER = "DSV4-ATTN-XFER"

FLAG = '''
# DSV4-ATTN-XFER: 218040. model.layers.N.attn is the group-0 MLAAttentionSpec
# (sliding_window=None) full-attention cache. Skipping it left decode with only
# the 128-token swa_cache, so NIAH could not retrieve at any length while a
# short curl still answered. Runtime flag so one image runs both arms.
import os as _dsv4_os

DSV4_TRANSFER_ATTN = _dsv4_os.environ.get("DSV4_TRANSFER_ATTN", "0") == "1"
'''

GATE_OLD = """        self.num_transfer_layers = sum(
            1
            for _ln in self.kv_caches
            if ".indexer." in _ln
            or (
                not _ln.endswith(".attn")
                and self._get_layer_transfer_geometry(_ln).block_size
                == self.block_size
            )
        ) or self.num_layers
"""

GATE_NEW = """        self.num_transfer_layers = sum(
            1
            for _ln in self.kv_caches
            if ".indexer." in _ln
            or (DSV4_TRANSFER_ATTN and _ln.endswith(".attn"))
            or (
                not _ln.endswith(".attn")
                and self._get_layer_transfer_geometry(_ln).block_size
                == self.block_size
            )
        ) or self.num_layers
"""

SAVE_OLD = """                if layer_name.endswith(".attn") or (
                    self._get_layer_transfer_geometry(layer_name).block_size
                    != self.block_size
                ):
                    _n_skip += 1
                    continue
"""

SAVE_NEW = """                if DSV4_TRANSFER_ATTN and layer_name.endswith(".attn"):
                    # DSV4-ATTN-XFER: group-0 full-attention page. dim[1] is
                    # compressed slots (256 / compress_ratio), so block_len and
                    # block_stride from the shipped MLA 3D path are already the
                    # correct per-page span; only block_size reads 64/2.
                    _n_attn += 1
                    self.save_kv_layer(metadata, layer_name, kv_layer, None)
                    continue
                if layer_name.endswith(".attn") or (
                    self._get_layer_transfer_geometry(layer_name).block_size
                    != self.block_size
                ):
                    _n_skip += 1
                    continue
"""

COUNT_OLD = """            _n_skip = 0
            _n_idx = 0
            _n_mla = 0
"""

COUNT_NEW = """            _n_skip = 0
            _n_idx = 0
            _n_mla = 0
            _n_attn = 0
"""

LOG_OLD = """            logger.info(
                "[dsv4-gate] wait_for_save mla=%d indexer=%d skipped=%d",
                _n_mla,
                _n_idx,
                _n_skip,
            )
"""

LOG_NEW = """            logger.info(
                "[dsv4-gate] wait_for_save mla=%d indexer=%d attn=%d skipped=%d "
                "(transfer_attn=%s)",
                _n_mla,
                _n_idx,
                _n_attn,
                _n_skip,
                DSV4_TRANSFER_ATTN,
            )
"""

# The gate patcher's agent-log region predates .attn transfer and reports only
# mla/indexer/skipped. Left alone it records skipped=0 with the 61 transferred
# .attn layers invisible, which reads like the pre-fix gate. Keep the region but
# give it the attn bucket.
DATA_OLD = (
    '                    "data": {"mla": _n_mla, "indexer": _n_idx, '
    '"skipped": _n_skip},\n'
)

DATA_NEW = (
    '                    "data": {"mla": _n_mla, "indexer": _n_idx, '
    '"attn": _n_attn, "skipped": _n_skip,\n'
    '                             "transferAttn": DSV4_TRANSFER_ATTN},\n'
)


def _apply(src: str) -> str:
    for old, new, what in (
        (GATE_OLD, GATE_NEW, "num_transfer_layers"),
        (COUNT_OLD, COUNT_NEW, "wait_for_save counters"),
        (SAVE_OLD, SAVE_NEW, "wait_for_save .attn branch"),
        (LOG_OLD, LOG_NEW, "wait_for_save log"),
    ):
        if old not in src:
            raise ValueError(f"anchor missing: {what}")
        src = src.replace(old, new, 1)

    # Soft: the agent-log region is debug telemetry and may be absent. Never let
    # it fail the transfer fix.
    if DATA_OLD in src:
        src = src.replace(DATA_OLD, DATA_NEW, 1)

    anchor = "\nlogger = init_logger(__name__)\n"
    if anchor in src:
        src = src.replace(anchor, anchor + FLAG, 1)
    else:
        lines = src.splitlines(keepends=True)
        cut = 0
        for i, line in enumerate(lines):
            if line.startswith(("import ", "from ")):
                cut = i + 1
        src = "".join(lines[:cut]) + FLAG + "".join(lines[cut:])
    return src


def _selftest() -> int:
    import ast

    stub = (
        "from vllm.logger import init_logger\n"
        "\nlogger = init_logger(__name__)\n"
        "\n\nclass C:\n"
        "    def register_kv_caches(self):\n"
        + GATE_OLD
        + "\n    def wait_for_save(self, metadata):\n"
        "        if True:\n"
        + COUNT_OLD
        + "            for layer_name, kv_layer in self.kv_caches.items():\n"
        "                if \".indexer.\" in layer_name:\n"
        "                    _n_idx += 1\n"
        "                    continue\n"
        + SAVE_OLD
        + "                _n_mla += 1\n"
        + LOG_OLD
        + "            _rec = {\n"
        + DATA_OLD
        + "            }\n"
    )
    out = _apply(stub)
    assert MARKER in out
    assert "DSV4_TRANSFER_ATTN and _ln.endswith" in out
    assert "_n_attn = 0" in out
    assert "attn=%d" in out
    assert '"attn": _n_attn' in out
    assert '"transferAttn": DSV4_TRANSFER_ATTN' in out
    ast.parse(out)
    try:
        _apply(out)
    except ValueError:
        pass
    else:
        raise AssertionError("second apply should not find anchors")
    print("[dsv4-attn-xfer] selftest OK")
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return _selftest()
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir> | --selftest", file=sys.stderr)
        return 2

    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[dsv4-attn-xfer] {REL} not found under {sys.argv[1]} -- skipping.")
        return 0

    src = open(path).read()
    if MARKER in src:
        print(f"[dsv4-attn-xfer] already patched in {path} -- no-op.")
        return 0

    try:
        out = _apply(src)
    except ValueError as e:
        print(
            f"[dsv4-attn-xfer] ERROR: {e}. This patcher must run after "
            "apply_moriio_dsv4_transfer_gate_fix.py.",
            file=sys.stderr,
        )
        return 1

    tmp = path + ".dsv4attn"
    with open(tmp, "w") as f:
        f.write(out)
    os.replace(tmp, path)

    try:
        import py_compile

        py_compile.compile(path, doraise=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dsv4-attn-xfer] ERROR: compile failed: {e}", file=sys.stderr)
        return 1

    print(
        f"[dsv4-attn-xfer] patched {path}: .attn transfer gated on "
        "DSV4_TRANSFER_ATTN (default 0)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
