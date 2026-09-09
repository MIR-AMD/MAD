#!/usr/bin/env python3
"""Relabel the DSV4 group-0 MLA page instead of folding it (vllm#48989).

218042 turned HMA on and logged ``fold-ok`` for all 41 group-0 ``.attn`` layers,
then wrote **nothing**: ``n_remote=0`` on every write. The fold is why.

``get_layer_transfer_geometry`` reads MLA 3D as
``(num_blocks, block_size, latent_dim)``, but DSV4 compresses the group-0 page,
so dim[1] is *slots per page*, not tokens::

    config compress_ratios = [0,0,4,128,4,128,...,4,0]   # per layer
    even layers ratio   4 -> shape (19955,  64, 584)   64 * 4   = 256
    odd  layers ratio 128 -> shape (19955,   2, 584)    2 * 128 = 256

dim[0] is already the page count and ``stride[0]`` already spans exactly one
page, so the *only* wrong value was the reported ``block_size`` (64 or 2 against
a group spec of 256). Our HMA fold treated it as a real page-size mismatch and
mutated four values to compensate::

    kbpb = spec_bs // block_size          # 4 on even layers, 128 on odd
    num_blocks  = num_blocks // kbpb      # 19955 -> 4988 / 155   <-- fatal
    block_stride = stride[0] * kbpb       # overshoots one page
    block_len    = block_len * kbpb       # overshoots one page
    block_size   = spec_bs                # the only correct line

Shrinking dim[0] by 4x (and 128x on the odd layers) shrinks the page table until
remote block ids fall outside it, which is exactly ``n_remote=0`` while the fold
still reports itself healthy.

vllm-project/vllm#48989 fixes it upstream as a pure relabel — keep dim[0], take
``block_size`` from the spec, and derive ``block_len`` from ``stride[0]``::

    num_blocks = shape[0]
    block_size = spec.block_size
    block_len = stride[0] * element_size
    slot_size_bytes = block_len // block_size

Keeping dim[0] is right, and 218220 proved it: ``num_blocks=13927`` against the
fold's ``108`` on the same layer, with curl 3/3 COHERENT where the fold arm
(218221) answered 0/3 in the wrong language.

``block_len`` looks wrong at first and is not. DSV4 KV caches are strided views
into one shared per-block buffer, so ``stride[0]`` spans the whole block, and
218220 duly logged ``block_len=1435968`` for all 243 layers at shapes from
``(N,2,584)`` to ``(N,4,2048)`` -- one page size for every shape. Each layer
therefore copies the entire block.

218257 tried the honest per-page value (``dim[1] * dim[2] * element_size``) and
**broke decode**: same model, same image, same flags, same ``block_size=256``,
same ``num_blocks=13927``, same 243 registered layers as 218220 -- ``block_len``
was the only difference, 1168 against 1435968, and curl went 3/3 COHERENT ->
0/3 GARBAGE (the 217779 SimpleGeo signature).

The reason is ``region_len``, established by grepping the installed package
(218285 / 218287). ``block_len`` has exactly two live consumers, both per-layer:

1. per-transfer size -- ``transfer_size_byte = geometry.block_len``, at offsets
   ``element_size * (block_id * block_stride)``;
2. region extent -- ``region_len = num_blocks * regions_per_block * block_len``,
   the registration length paired with ``cache.data_ptr()``.

The offsets step ``block_stride`` whatever ``block_len`` is, so the addresses
touched span ``num_blocks * block_stride * element_size``. Upstream's value makes
``region_len`` exactly that span; the per-page value shrinks it ~38x while the
offsets keep their stride, so addressing runs past the registered region. Nothing
catches it -- the ``ValueError`` at connector:1760 guards ``block_size``, not
``block_len`` -- so 218257 returned garbage instead of raising.

There is a third *write* of ``block_len``, and it is inert: ``self.block_len =
first_geometry.block_len`` feeds ``MoRIIOAgentMetadata(..., block_len=...)`` under
this TODO::

    # TODO(tms): self.block_len needs to be per-layer for sliding window,
    # hybrid attn, etc

But no ``.block_len`` read exists anywhere on the receiving side, and
``self.block_lens[layer_name]`` (connector:1765) is never read either. Both peers
derive geometry locally from their own tensors, so finishing that TODO would
change nothing. An earlier version of this docstring claimed the handshake scalar
was the constraint; that is retracted.

``merge_contiguous_offsets`` is the one real cost of a small ``block_len``: it
coalesces when ``offset == prev_offset + prev_size``, true only when ``block_len
== block_stride * element_size``. Upstream's value merges consecutive blocks into
one large RDMA op; a per-page value moves far fewer bytes in many more ops.

So the whole-block copy is what an *unfloored* ``region_len`` costs, not a law.
``DSV4_HMA_PAGE_BLOCK_LEN=1`` restores the per-page value and requires
``DSV4_REGION_LEN_SPAN=1`` (``apply_moriio_dsv4_region_len_span_fix.py``) to keep
registration covering the full span. Setting it alone reproduces 218257.

``DSV4_HMA_CLIP_PAGE=1`` is not that. It keeps ``block_len`` near the shared
block and only subtracts the view origin ``S`` so a WRITE of dest page P ends
at dest P+1's origin. ``S=0`` still copies ``stride[0]*es`` (curl). 223559
measured ``S>0`` overshooting into dest P+1; 2k dest ids include 7 and 8.
223633 applied CLIP; NIAH still NONE.

``DSV4_HMA_NATIVE_ATTN=1`` copies the HMA=0 TRANSFER_ATTN native slice, but
only for names ending in ``.attn``. PAGE_BLOCK_LEN is the same formula on
every 3D layer (SWA included) and broke curl. Needs ``DSV4_REGION_LEN_SPAN=1``.

This patcher appends the upstream relabel *after* the fold so it overrides
whichever of the three fold variants the HMA patcher installed, rather than
restructuring that battle-tested block. Runtime-gated so one image serves both
arms::

    DSV4_HMA_UPSTREAM_GEOM=0   (default) 218042 fold
    DSV4_HMA_UPSTREAM_GEOM=1            vllm#48989 relabel
    DSV4_HMA_PAGE_BLOCK_LEN=1           ... with a per-page block_len.
                                        Needs DSV4_REGION_LEN_SPAN=1.
    DSV4_HMA_CLIP_PAGE=1                GEOM=1 copy stops at the next
                                        page origin (223559). Not PAGE_BLOCK_LEN.
    DSV4_HMA_NATIVE_ATTN=1              .attn only: block_len = dim[1]*dim[2]*es
                                        (HMA=0 TRANSFER_ATTN copy). Needs SPAN.
                                        Not PAGE_BLOCK_LEN (all 3D / 218257).

Look for ``[dsv4-hma] upstream-geom layer=...`` once per layer to prove which
arm ran; ``fold-ok`` alone does not distinguish them (218042 logged it too).

Must run AFTER apply_moriio_dsv4_supports_hma_fix.py, which installs the fold
this overrides. HMA-on arm only — connectors/moriio.sh skips it when
DSV4_ENABLE_HMA=0, where there is no fold and the HMA-off ``.attn`` path
(DSV4_TRANSFER_ATTN) already relabels. Idempotent. Missing anchor is a hard
error.

Usage: apply_moriio_dsv4_hma_upstream_geom_fix.py <vllm_install_dir>
       apply_moriio_dsv4_hma_upstream_geom_fix.py --selftest
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py"
MARKER = "DSV4-HMA-UPSTREAM-GEOM"

FLAG = '''
# DSV4-HMA-UPSTREAM-GEOM: vllm-project/vllm#48989. DSV4 compresses the group-0
# MLA page, so tensor dim[1] is slots per page (256 / compress_ratio) and reads
# 64 or 2 against a group spec of 256. dim[0] is already the page count and
# stride[0] already spans one page, so the fix is a relabel. The HMA fold
# instead divided dim[0] by kbpb (4x even layers, 128x odd), shrinking the page
# table until no remote block id resolved -- 218042's n_remote=0 on every write
# with fold-ok on all 41 layers.
import os as _dsv4_os

DSV4_HMA_UPSTREAM_GEOM = (
    _dsv4_os.environ.get("DSV4_HMA_UPSTREAM_GEOM", "0") == "1"
)
# Per-page block_len instead of the shared-block stride. Requires
# DSV4_REGION_LEN_SPAN=1 or registration under-covers what the offsets reach
# (218257 garbage). See apply_moriio_dsv4_region_len_span_fix.py.
DSV4_HMA_PAGE_BLOCK_LEN = (
    _dsv4_os.environ.get("DSV4_HMA_PAGE_BLOCK_LEN", "0") == "1"
)
# 223559: whole-block copy from view origin S of dest page P is
# [S, S+stride[0]*es) = through dest P+1 offset S. Curl is one dest page
# so P+1 is unused; 2k dest ids include 7 and 8. Subtract S so the copy
# stops at the next page origin. Not PAGE_BLOCK_LEN (218257 / 1168).
# Default 0 = 223559.
DSV4_HMA_CLIP_PAGE = (
    _dsv4_os.environ.get("DSV4_HMA_CLIP_PAGE", "0") == "1"
)
# 223633: CLIP applied; NIAH still NONE. HMA=0 TRANSFER_ATTN copies this
# layer's native slice (dim[1]*dim[2]*es) and 2k passes. PAGE_BLOCK_LEN
# did that to every MLA 3D tensor including SWA and curl went GARBAGE
# (218257/218291/218305). Restrict the native copy to group-0 .attn.
# Default 0 = 223633 whole-block/.CLIP remainder. Needs SPAN.
DSV4_HMA_NATIVE_ATTN = (
    _dsv4_os.environ.get("DSV4_HMA_NATIVE_ATTN", "0") == "1"
)
_DSV4_UPSTREAM_GEOM_SEEN = set()


def _dsv4_upstream_geom_log(
    layer_name, shape, num_blocks, block_size, block_len, stor_off=-1,
    slice_b=-1, clip=-1,
):
    """INFO once per layer. fold-ok cannot tell the two arms apart (218042).

    223559: stor_off is view origin in the storage (bytes); slice_b is
    dim[1]*dim[2]*es; clip is bytes subtracted from stride[0]*es so the
    copy ends at the next page origin (0 when CLIP_PAGE is off or S=0).
    """
    if layer_name in _DSV4_UPSTREAM_GEOM_SEEN:
        return
    _DSV4_UPSTREAM_GEOM_SEEN.add(layer_name)
    if _dsv4_os.environ.get("DSV4_HMA_GEOM_SELFTEST") == "1":
        return
    try:
        from vllm.logger import init_logger as _dsv4_init_logger

        _dsv4_init_logger(__name__).info(
            "[dsv4-hma] upstream-geom layer=%s shape=%s num_blocks=%d "
            "block_size=%d block_len=%d stor_off=%s slice_b=%s clip=%s",
            layer_name,
            tuple(shape),
            num_blocks,
            block_size,
            block_len,
            stor_off,
            slice_b,
            clip,
        )
    except Exception:  # noqa: BLE001
        pass
'''

# Tail of every HMA fold variant (mla-3d-fold, mla-3d-floor, mla-3d-override)
# followed by the 3D return. Anchoring here means we do not care which variant
# is installed, and the anchor itself proves the HMA patcher ran first.
GEOM_OLD = """                block_size = spec_bs
        return LayerTransferGeometry(
"""

GEOM_NEW = """                block_size = spec_bs
        if DSV4_HMA_UPSTREAM_GEOM and spec_bs > 0:
            # DSV4-HMA-UPSTREAM-GEOM: vllm-project/vllm#48989, corrected by
            # 218220. Relabel the compressed group-0 page instead of folding it:
            # dim[0] is already the page count, so only the reported block_size
            # was ever wrong. Overrides the fold above.
            #
            # block_len from stride[0] is upstream verbatim and looks wrong but
            # is not: DSV4 KV caches are strided views into one shared per-block
            # buffer, so stride[0] spans the whole block. 218220 logged
            # block_len=1435968 for all 243 layers at shapes from (N,2,584) to
            # (N,4,2048) -- every layer copies the entire block, survivable for a
            # 1-page curl and a 1800s stall on a 2600-token prompt.
            #
            # DSV4_HMA_PAGE_BLOCK_LEN takes this layer's own page instead. That
            # is what 218257 did alone, and it decoded garbage: region_len is
            # num_blocks * regions_per_block * block_len while offsets still step
            # block_stride, so registration covered ~38x less than the addressing
            # reached. Pair it with DSV4_REGION_LEN_SPAN=1, which floors the
            # extent at the span. block_stride stays stride[0] either way, so the
            # copy still steps block-to-block. Do not set it (218257/218291/218305).
            #
            # DSV4_HMA_CLIP_PAGE keeps the whole-block map (block_size=256,
            # num_blocks=shape[0], block_stride=stride[0], S=0 still copies
            # stride[0]*es) and subtracts the intra-page origin so a WRITE of
            # dest P cannot enter dest P+1. 223559: S=12096 even stomps P+1
            # [0,12096); S=49536 odd stomps P+1 even extra-cache. Not the
            # native slice (1168/37376).
            #
            # DSV4_HMA_NATIVE_ATTN is the HMA=0 TRANSFER_ATTN copy, but only
            # for names that end with ".attn" (not swa/indexer/compressor).
            # 223633 CLIP (remainder of the shared block) did not make
            # extra-cache visible. PAGE_BLOCK_LEN is the same formula on
            # every 3D layer and broke curl; do not set it.
            #
            # All five are set explicitly because this runs AFTER the fold, which
            # has already multiplied block_len and block_stride by kbpb.
            num_blocks = shape[0]
            block_size = spec_bs
            block_len = stride[0] * element_size
            _native_attn = (
                DSV4_HMA_NATIVE_ATTN
                and str(layer_name).endswith(".attn")
                and len(shape) >= 3
            )
            if DSV4_HMA_PAGE_BLOCK_LEN and len(shape) >= 3:
                block_len = shape[1] * shape[2] * element_size
            elif _native_attn:
                block_len = shape[1] * shape[2] * element_size
            block_stride = stride[0]
            _stor_off = -1
            try:
                _so = kv_cache.storage_offset
                _so = _so() if callable(_so) else _so
                _stor_off = int(_so) * int(element_size)
            except Exception:
                _stor_off = -1
            _slice_b = -1
            try:
                if len(shape) >= 3:
                    _slice_b = int(shape[1]) * int(shape[2]) * int(element_size)
            except Exception:
                _slice_b = -1
            _clip = 0
            if (
                DSV4_HMA_CLIP_PAGE
                and not DSV4_HMA_PAGE_BLOCK_LEN
                and not _native_attn
                and block_len > 0
                and _stor_off > 0
            ):
                _stor_page = int(_stor_off) % int(block_len)
                if _stor_page > 0:
                    _clip = _stor_page
                    block_len = int(block_len) - _stor_page
            slot_size_bytes = block_len // block_size
            _dsv4_upstream_geom_log(
                layer_name, shape, num_blocks, block_size, block_len,
                _stor_off, _slice_b, _clip
            )
        return LayerTransferGeometry(
"""

# Self-contained stand-in for the post-HMA moriio_layout.py 3D MLA branch.
STUB = (
    "def get_layer_transfer_geometry(\n"
    "    layer_name,\n"
    "    kv_cache,\n"
    "    layer_to_spec,\n"
    "    remote_num_blocks=None,\n"
    "    spec_bs_override=None,\n"
    "):\n"
    "    shape = kv_cache.shape\n"
    "    stride = kv_cache.stride()\n"
    "    element_size = kv_cache.element_size()\n"
    "    if is_mla_cache and len(shape) == 3:\n"
    "        num_blocks, block_size, latent_dim = shape\n"
    "        slot_size_bytes = latent_dim * element_size\n"
    "        block_len = block_size * slot_size_bytes\n"
    "        block_stride = stride[0]\n"
    "        spec_bs = int(spec_bs_override or 0) or 0\n"
    "        if (\n"
    "            spec_bs > 0\n"
    "            and block_size > 0\n"
    "            and spec_bs != block_size\n"
    "            and spec_bs % block_size == 0\n"
    "        ):\n"
    "            kbpb = spec_bs // block_size\n"
    "            if num_blocks >= kbpb:\n"
    "                num_blocks = num_blocks // kbpb\n"
    "                block_stride = stride[0] * kbpb\n"
    "                block_len = block_len * kbpb\n"
    "                block_size = spec_bs\n"
    "        return LayerTransferGeometry(\n"
    "            num_blocks=num_blocks,\n"
    "            block_size=block_size,\n"
    "            block_len=block_len,\n"
    "            slot_size_bytes=slot_size_bytes,\n"
    "            block_stride=block_stride,\n"
    "        )\n"
)


def _insert_flag(src: str) -> str:
    """Place FLAG after the import block.

    moriio_layout.py has no ``logger = init_logger(__name__)``, and a
    line-prefix scan for import/from lands *inside* a multi-line
    ``from x import (...)`` -- which is what broke 218197/218198. Use the AST
    end_lineno so a parenthesised import is never split.
    """
    anchor = "\nlogger = init_logger(__name__)\n"
    if anchor in src:
        return src.replace(anchor, anchor + FLAG, 1)

    import ast

    cut = 0
    body = ast.parse(src).body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        cut = body[0].end_lineno or 0  # module docstring
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            cut = max(cut, node.end_lineno or node.lineno)
    lines = src.splitlines(keepends=True)
    return "".join(lines[:cut]) + FLAG + "".join(lines[cut:])


def _apply(src: str) -> str:
    if GEOM_OLD not in src:
        raise ValueError("anchor missing: MLA 3D fold tail + LayerTransferGeometry")
    return _insert_flag(src.replace(GEOM_OLD, GEOM_NEW, 1))


def _selftest() -> int:
    import ast
    import types

    os.environ["DSV4_HMA_GEOM_SELFTEST"] = "1"
    out = _apply(STUB)
    assert MARKER in out
    assert "DSV4_HMA_UPSTREAM_GEOM and spec_bs > 0" in out
    assert "num_blocks = shape[0]" in out
    assert "stor_off=" in out
    assert "slice_b=" in out
    assert "clip=" in out
    assert "DSV4_HMA_CLIP_PAGE" in out
    assert "DSV4_HMA_NATIVE_ATTN" in out
    # Upstream verbatim. 218257 replaced these two lines with the honest per-page
    # value and decode produced garbage, so they are load-bearing, not sloppy.
    assert "block_len = stride[0] * element_size" in out
    assert "slot_size_bytes = block_len // block_size" in out
    ast.parse(out)

    # 218197/218198 both aborted here: moriio_layout.py has no logger anchor and
    # opens with a parenthesised import, so a line-prefix scan dropped FLAG
    # *inside* the import list. Keep a stub with that exact shape.
    realistic = (
        '"""MoRIIO layout helpers.\n\nSecond docstring line.\n"""\n'
        "import torch\n"
        "from collections.abc import Mapping\n"
        "from vllm.v1.kv_cache_interface import (\n"
        "    KVCacheSpec,\n"
        "    MLAAttentionSpec,\n"
        ")\n"
        "\n\n" + STUB
    )
    assert "logger = init_logger(__name__)" not in realistic
    out_real = _apply(realistic)
    ast.parse(out_real)
    assert "    KVCacheSpec,\n    MLAAttentionSpec,\n)\n" in out_real, (
        "FLAG split the parenthesised import"
    )
    assert MARKER in out_real

    # 218220: DSV4 caches are strided views into ONE shared per-block buffer, so
    # stride[0] is the whole block and is the SAME for every layer whatever its
    # shape. A contiguous stub cannot catch a block_len taken from the stride,
    # because there stride[0] == shape[1]*shape[2]. Model the real thing.
    SHARED_BLOCK = 1435968

    class _Tensor:
        """(num_blocks, slots_per_page, latent) fp8 view of a shared block."""

        def __init__(self, shape):
            self.shape = shape

        def stride(self):
            return (SHARED_BLOCK, self.shape[2], 1)

        def element_size(self):
            return 1

        def storage_offset(self):
            return getattr(self, "_stor_off", 0)

    # DSV4 group 0: 19955 pages, compress_ratio 4 (even) and 128 (odd).
    even, odd = _Tensor((19955, 64, 584)), _Tensor((19955, 2, 584))
    assert even.stride()[0] == odd.stride()[0] != 0
    saved = os.environ.get("DSV4_HMA_UPSTREAM_GEOM")
    try:
        for arm in ("1", "0"):
            os.environ["DSV4_HMA_UPSTREAM_GEOM"] = arm
            ns = {
                "LayerTransferGeometry": lambda **kw: types.SimpleNamespace(**kw),
                "is_mla_cache": True,
            }
            exec(compile(out, "<patched>", "exec"), ns)  # noqa: S102
            geom = ns["get_layer_transfer_geometry"]
            g_even = geom("model.layers.0.attn", even, {}, None, 256)
            g_odd = geom("model.layers.1.attn", odd, {}, None, 256)

            if arm == "1":
                # Default (DSV4_HMA_PAGE_BLOCK_LEN unset): block_len ==
                # block_stride * element_size, which makes region_len cover the
                # span the offsets reach AND lets merge_contiguous_offsets
                # coalesce. Uniform across shapes because stride[0] is the shared
                # block. Assert the invariants, not a literal.
                for g in (g_even, g_odd):
                    assert g.num_blocks == 19955, g.num_blocks
                    assert g.block_size == 256, g.block_size
                    assert g.block_len == g.block_stride * 1, g.block_len
                assert g_even.block_len == g_odd.block_len, (
                    "stride[0] is the shared block; it cannot vary by shape"
                )
                assert g_even.block_size == g_odd.block_size, (
                    "connector raises ValueError on a non-uniform block_size"
                )
                # The per-page value is what differs per shape. Reaching it needs
                # DSV4_REGION_LEN_SPAN=1, checked below.
                assert (even.shape[1] * even.shape[2]) != (
                    odd.shape[1] * odd.shape[2]
                )
            else:
                # 218042 fold: dim[0] collapses 4x / 128x. This is the bug.
                assert g_even.num_blocks == 19955 // 4, g_even.num_blocks
                assert g_odd.num_blocks == 19955 // 128, g_odd.num_blocks
    finally:
        if saved is None:
            os.environ.pop("DSV4_HMA_UPSTREAM_GEOM", None)
        else:
            os.environ["DSV4_HMA_UPSTREAM_GEOM"] = saved

    # DSV4_HMA_PAGE_BLOCK_LEN: the honest per-page value, which is what 218257
    # ran and what DSV4_REGION_LEN_SPAN makes safe. It must vary by shape (that
    # is the point) while num_blocks, block_size and block_stride stay put -- if
    # block_stride moved, the offsets would no longer step block-to-block.
    saved_page = os.environ.get("DSV4_HMA_PAGE_BLOCK_LEN")
    try:
        os.environ["DSV4_HMA_UPSTREAM_GEOM"] = "1"
        os.environ["DSV4_HMA_PAGE_BLOCK_LEN"] = "1"
        ns = {
            "LayerTransferGeometry": lambda **kw: types.SimpleNamespace(**kw),
            "is_mla_cache": True,
        }
        exec(compile(out, "<patched>", "exec"), ns)  # noqa: S102
        geom = ns["get_layer_transfer_geometry"]
        p_even = geom("model.layers.0.attn", even, {}, None, 256)
        p_odd = geom("model.layers.1.attn", odd, {}, None, 256)
        assert p_even.block_len == 64 * 584, p_even.block_len
        assert p_odd.block_len == 2 * 584, p_odd.block_len
        assert p_even.block_len != p_odd.block_len
        for g in (p_even, p_odd):
            assert g.num_blocks == 19955, g.num_blocks
            assert g.block_size == 256, g.block_size
            assert g.block_stride == SHARED_BLOCK, g.block_stride
            assert g.block_len < g.block_stride, (
                "per-page must be smaller than the shared block, else "
                "DSV4_REGION_LEN_SPAN has nothing to floor"
            )
    finally:
        for key, val in (
            ("DSV4_HMA_UPSTREAM_GEOM", saved),
            ("DSV4_HMA_PAGE_BLOCK_LEN", saved_page),
        ):
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    # 223559: copy of stride[0]*es from view origin S of dest P writes
    # through dest P+1 offset S. CLIP_PAGE subtracts the intra-page origin.
    # S=0 (curl-coherent whole-block) is unchanged; S>0 must not become
    # PAGE_BLOCK_LEN's native slice.
    saved_clip = os.environ.get("DSV4_HMA_CLIP_PAGE")
    saved_native = os.environ.get("DSV4_HMA_NATIVE_ATTN")
    even_s, odd_s = 12096, 49536
    even._stor_off = even_s
    odd._stor_off = odd_s
    try:
        os.environ["DSV4_HMA_UPSTREAM_GEOM"] = "1"
        os.environ["DSV4_HMA_PAGE_BLOCK_LEN"] = "0"
        os.environ["DSV4_HMA_NATIVE_ATTN"] = "0"
        os.environ["DSV4_HMA_CLIP_PAGE"] = "0"
        ns = {
            "LayerTransferGeometry": lambda **kw: types.SimpleNamespace(**kw),
            "is_mla_cache": True,
        }
        exec(compile(out, "<patched>", "exec"), ns)  # noqa: S102
        geom = ns["get_layer_transfer_geometry"]
        z_even = geom("model.layers.0.attn", even, {}, None, 256)
        z_odd = geom("model.layers.1.attn", odd, {}, None, 256)
        # Flag off = 223559: whole-block even when S>0.
        assert z_even.block_len == SHARED_BLOCK, z_even.block_len
        assert z_odd.block_len == SHARED_BLOCK, z_odd.block_len

        os.environ["DSV4_HMA_CLIP_PAGE"] = "1"
        ns = {
            "LayerTransferGeometry": lambda **kw: types.SimpleNamespace(**kw),
            "is_mla_cache": True,
        }
        exec(compile(out, "<patched>", "exec"), ns)  # noqa: S102
        geom = ns["get_layer_transfer_geometry"]
        c_even = geom("model.layers.0.attn", even, {}, None, 256)
        c_odd = geom("model.layers.1.attn", odd, {}, None, 256)
        assert c_even.block_len == SHARED_BLOCK - even_s, c_even.block_len
        assert c_odd.block_len == SHARED_BLOCK - odd_s, c_odd.block_len
        assert c_even.block_len != 64 * 584, "CLIP_PAGE is not PAGE_BLOCK_LEN"
        assert c_odd.block_len != 2 * 584, "CLIP_PAGE is not PAGE_BLOCK_LEN"
        for g in (c_even, c_odd):
            assert g.num_blocks == 19955, g.num_blocks
            assert g.block_size == 256, g.block_size
            assert g.block_stride == SHARED_BLOCK, g.block_stride
        # Native slice still fits in the clipped remainder (223559 packing).
        assert c_even.block_len >= 64 * 584, c_even.block_len
        assert c_odd.block_len >= 2 * 584, c_odd.block_len

        even._stor_off = 0
        c0 = geom("model.layers.2.attn", even, {}, None, 256)
        assert c0.block_len == SHARED_BLOCK, c0.block_len
        even._stor_off = even_s

        # PAGE_BLOCK_LEN wins: native slice, not stride-S.
        os.environ["DSV4_HMA_PAGE_BLOCK_LEN"] = "1"
        ns = {
            "LayerTransferGeometry": lambda **kw: types.SimpleNamespace(**kw),
            "is_mla_cache": True,
        }
        exec(compile(out, "<patched>", "exec"), ns)  # noqa: S102
        geom = ns["get_layer_transfer_geometry"]
        both = geom("model.layers.0.attn", even, {}, None, 256)
        assert both.block_len == 64 * 584, both.block_len
    finally:
        even._stor_off = 0
        odd._stor_off = 0
        for key, val in (
            ("DSV4_HMA_UPSTREAM_GEOM", saved),
            ("DSV4_HMA_PAGE_BLOCK_LEN", saved_page),
            ("DSV4_HMA_CLIP_PAGE", saved_clip),
            ("DSV4_HMA_NATIVE_ATTN", saved_native),
        ):
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    # 223633: CLIP remainder did not make extra-cache visible. Native slice
    # on .attn only (HMA=0 TRANSFER_ATTN). SWA keeps CLIP/whole-block.
    even._stor_off = even_s
    odd._stor_off = 0
    try:
        os.environ["DSV4_HMA_UPSTREAM_GEOM"] = "1"
        os.environ["DSV4_HMA_PAGE_BLOCK_LEN"] = "0"
        os.environ["DSV4_HMA_CLIP_PAGE"] = "1"
        os.environ["DSV4_HMA_NATIVE_ATTN"] = "1"
        ns = {
            "LayerTransferGeometry": lambda **kw: types.SimpleNamespace(**kw),
            "is_mla_cache": True,
        }
        exec(compile(out, "<patched>", "exec"), ns)  # noqa: S102
        geom = ns["get_layer_transfer_geometry"]
        n_even = geom("model.layers.0.attn", even, {}, None, 256)
        n_odd = geom("model.layers.1.attn", odd, {}, None, 256)
        n_swa = geom("model.layers.0.attn.swa_cache", even, {}, None, 256)
        assert n_even.block_len == 64 * 584, n_even.block_len
        assert n_odd.block_len == 2 * 584, n_odd.block_len
        assert n_swa.block_len == SHARED_BLOCK - even_s, n_swa.block_len
        for g in (n_even, n_odd, n_swa):
            assert g.num_blocks == 19955, g.num_blocks
            assert g.block_size == 256, g.block_size
            assert g.block_stride == SHARED_BLOCK, g.block_stride
    finally:
        even._stor_off = 0
        odd._stor_off = 0
        for key, val in (
            ("DSV4_HMA_UPSTREAM_GEOM", saved),
            ("DSV4_HMA_PAGE_BLOCK_LEN", saved_page),
            ("DSV4_HMA_CLIP_PAGE", saved_clip),
            ("DSV4_HMA_NATIVE_ATTN", saved_native),
        ):
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    try:
        _apply(out)
    except ValueError:
        pass
    else:
        raise AssertionError("second apply should not find the anchor")
    print("[dsv4-hma-geom] selftest OK")
    os.environ.pop("DSV4_HMA_GEOM_SELFTEST", None)
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return _selftest()
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir> | --selftest", file=sys.stderr)
        return 2

    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[dsv4-hma-geom] {REL} not found under {sys.argv[1]} -- skipping.")
        return 0

    src = open(path).read()
    if MARKER in src:
        print(f"[dsv4-hma-geom] already patched in {path} -- no-op.")
        return 0

    try:
        out = _apply(src)
    except ValueError as e:
        print(
            f"[dsv4-hma-geom] ERROR: {e}. This patcher must run after "
            "apply_moriio_dsv4_supports_hma_fix.py, which installs the fold "
            "it overrides.",
            file=sys.stderr,
        )
        return 1

    tmp = path + ".dsv4geomup"
    with open(tmp, "w") as f:
        f.write(out)
    os.replace(tmp, path)

    try:
        import py_compile

        py_compile.compile(path, doraise=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dsv4-hma-geom] ERROR: compile failed: {e}", file=sys.stderr)
        return 1

    print(
        f"[dsv4-hma-geom] patched {path}: vllm#48989 relabel gated on "
        "DSV4_HMA_UPSTREAM_GEOM (default 0 = 218042 fold)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
