#!/usr/bin/env python3
"""Floor the MoRIIO registration extent at the real view span.

**218291 refuted the hypothesis this was built to test.** The floor worked --
every layer logged ``region_len=16266736 -> 19998726336`` and ``block_len`` came
back per-page (2912x 37376, 1952x 32768, 992x 1168, 960x 8448, 960x 8192, zero
1435968) -- yet curl still went GARBAGE against 218282's 3/3 COHERENT on the same
prompt and image. So registration extent was not what a per-page ``block_len``
breaks. ``transfer_size_byte`` is the only consumer left, which points at the page
being strided *inside* the shared block; the log now prints ``shape``, ``stride``
and ``page_contiguous`` to settle that. ``1435968 / 37376 = 38.4`` already hints
the block is not a whole number of pages.

Keep the patcher: it is a correct safety floor (``max`` never shrinks a working
region) and it now carries the stride diagnostic. Default stays 0.

218257 set ``block_len`` to the honest per-page value (dim[1] * dim[2] *
element_size) and decode produced garbage. The first explanation blamed the
handshake, and 218285/218287 disproved it: a package-wide grep for ``block_len``
finds the ``LayerTransferGeometry`` field, the six geometry branches that compute
it, ``region_len``, ``transfer_size_byte``, the ``MoRIIOAgentMetadata`` field
(``moriio_common.py:115``) and the two connector lines that fill it (1749, 1816)
-- and **no read of the received value anywhere**. ``self.block_lens`` is
likewise assigned (connector:1765) and never read. Both peers derive geometry
locally from their own tensors, so the handshake scalar cannot corrupt a decode.

What a per-page ``block_len`` actually breaks is one line of registration::

    region_len = geometry.num_blocks * geometry.regions_per_block * geometry.block_len

``compute_block_transfer_offsets`` steps offsets by ``element_size * (block_id *
block_stride)`` no matter what ``block_len`` is, so the addresses touched span
``num_blocks * block_stride * element_size``. Upstream's ``block_len =
block_stride * element_size`` makes ``region_len`` exactly that span. The
per-page value shrinks it ~38x on DSV4 (37376 against 1435968) while the offsets
keep their full stride, so addressing runs past the registered region. Nothing
validates it -- the ``ValueError`` at connector:1760 guards ``block_size``, not
``block_len`` -- which is why 218257 returned garbage rather than raising.

``apply_moriio_dsv4_skip_noncontiguous_register_fix.py`` does not already cover
this. That patcher rewrites the ``register_local_tensor(kv_cache)`` call to use a
``data_ptr``-aligned strided span; ``region_len`` feeds the separate
``caches_data.append((base_addr, region_len, ...))`` list. Different path.

So floor the extent instead of constraining ``block_len``::

    span = num_blocks * block_stride * element_size
    region_len = max(region_len, span)

``max`` never shrinks a region that works today, so every non-DSV4 geometry is
untouched: the 5D K/V branch has ``regions_per_block=2`` with ``2 * block_len ==
block_stride * element_size`` already, so the floor is a no-op there. Only the
compressed group-0 MLA page, where ``block_len`` is deliberately smaller than the
shared-block stride, sees a change. ``split_kv_regions`` layers are skipped --
there ``region_len`` describes one of several separate caches, so a total-span
floor would over-register.

Pair with ``DSV4_HMA_PAGE_BLOCK_LEN=1`` (in
``apply_moriio_dsv4_hma_upstream_geom_fix.py``), which supplies the per-page
``block_len`` this makes safe. Alone this patcher is a no-op, since upstream's
``block_len`` already equals the span.

The cost is merging: ``merge_contiguous_offsets`` coalesces when ``offset ==
prev_offset + prev_size``, true only when ``block_len == block_stride *
element_size``. A per-page ``block_len`` moves far fewer bytes in many more RDMA
operations. That trade is what the run measures.

Runtime-gated so one image serves both arms::

    DSV4_REGION_LEN_SPAN=0   (default) shipped region_len
    DSV4_REGION_LEN_SPAN=1            floor it at the view span

Look for ``[dsv4-region] span-floor layer=...`` once per layer to prove it ran.
Gated in connectors/moriio.sh to DSV4 Flash/Pro. Idempotent. Missing anchor is a
hard error; missing file is a skip.

Usage: apply_moriio_dsv4_region_len_span_fix.py <vllm_install_dir>
       apply_moriio_dsv4_region_len_span_fix.py --selftest
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_layout.py"
MARKER = "DSV4-REGION-LEN-SPAN"

FLAG = '''
# DSV4-REGION-LEN-SPAN: region_len = num_blocks * regions_per_block * block_len,
# but offsets step block_stride * element_size regardless. A per-page block_len
# (DSV4's compressed group-0 page) therefore registers ~38x less than the
# addressing touches, silently -- 218257's garbage decode. Floor the extent at
# the span the offsets actually reach.
import os as _dsv4_rl_os

DSV4_REGION_LEN_SPAN = (
    _dsv4_rl_os.environ.get("DSV4_REGION_LEN_SPAN", "0") == "1"
)
_DSV4_REGION_SPAN_SEEN = set()


def _dsv4_region_span_log(
    layer_name, shipped, floored, block_len, block_stride, shape=None, stride=None
):
    """INFO once per layer, and only when the floor actually raised the extent.

    shape/stride are the 218291 follow-up: the span floor worked and decode was
    still garbage, so the page may be strided *inside* the shared block. If
    stride[1] != shape[2] then a contiguous per-page copy cannot be right and the
    whole-block block_len is mandatory.
    """
    if layer_name in _DSV4_REGION_SPAN_SEEN:
        return
    _DSV4_REGION_SPAN_SEEN.add(layer_name)
    try:
        from vllm.logger import init_logger as _dsv4_rl_init_logger

        _dsv4_rl_init_logger(__name__).info(
            "[dsv4-region] span-floor layer=%s region_len=%d -> %d "
            "block_len=%d block_stride=%d shape=%s stride=%s page_contiguous=%s",
            layer_name,
            shipped,
            floored,
            block_len,
            block_stride,
            tuple(shape) if shape is not None else None,
            tuple(stride) if stride is not None else None,
            (
                None
                if shape is None or stride is None or len(shape) < 3
                else bool(stride[1] == shape[2])
            ),
        )
    except Exception:  # noqa: BLE001
        pass
'''

REGION_OLD = """    geometry = get_layer_transfer_geometry(layer_name, kv_cache, layer_to_spec)
    region_len = geometry.num_blocks * geometry.regions_per_block * geometry.block_len
"""

REGION_NEW = """    geometry = get_layer_transfer_geometry(layer_name, kv_cache, layer_to_spec)
    region_len = geometry.num_blocks * geometry.regions_per_block * geometry.block_len
    if DSV4_REGION_LEN_SPAN and not geometry.split_kv_regions:
        # DSV4-REGION-LEN-SPAN: compute_block_transfer_offsets steps offsets by
        # element_size * (block_id * block_stride), so the addresses reach
        # num_blocks * block_stride * element_size no matter how small block_len
        # is. max() never shrinks a working region, so the 5D K/V branch (where
        # 2 * block_len already equals block_stride * element_size) is a no-op
        # and only the compressed MLA page changes.
        _dsv4_span = (
            geometry.num_blocks * geometry.block_stride * kv_cache.element_size()
        )
        if _dsv4_span > region_len:
            _dsv4_region_span_log(
                layer_name,
                region_len,
                _dsv4_span,
                geometry.block_len,
                geometry.block_stride,
                getattr(kv_cache, "shape", None),
                kv_cache.stride() if hasattr(kv_cache, "stride") else None,
            )
            region_len = _dsv4_span
"""

# Stand-in for the shipped iter_layer_registration_regions.
STUB = (
    "def iter_layer_registration_regions(\n"
    "    layer_name,\n"
    "    kv_cache,\n"
    "    layer_to_spec,\n"
    "):\n"
    "    geometry = get_layer_transfer_geometry(layer_name, kv_cache, layer_to_spec)\n"
    "    region_len = geometry.num_blocks * geometry.regions_per_block * geometry.block_len\n"
    "    if geometry.split_kv_regions:\n"
    "        return [(cache, region_len) for cache in kv_cache]\n"
    "    return [(kv_cache, region_len)]\n"
)


def _insert_flag(src: str) -> str:
    """Place FLAG after the import block.

    moriio_layout.py has no ``logger = init_logger(__name__)`` and opens with a
    parenthesised ``from ... import (...)``. A line-prefix scan lands inside that
    import and breaks the module, which is what killed 218197/218198. Use the AST
    end_lineno so a multi-line construct is never split.
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
    if MARKER in src:
        raise ValueError("already patched")
    if REGION_OLD not in src:
        raise ValueError("anchor missing: iter_layer_registration_regions region_len")
    return _insert_flag(src.replace(REGION_OLD, REGION_NEW, 1))


def _selftest() -> int:
    import ast
    import types

    out = _apply(STUB)
    assert MARKER in out
    assert "DSV4_REGION_LEN_SPAN and not geometry.split_kv_regions" in out
    ast.parse(out)

    # 218197/218198 died on FLAG landing inside a parenthesised import. Keep a
    # stub with that exact shape and no logger anchor.
    realistic = (
        '"""MoRIIO layout helpers.\n\nSecond docstring line.\n"""\n'
        "import torch\n"
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

    # Behaviour. Two geometries: the DSV4 compressed MLA page (per-page block_len
    # far below the shared-block stride) and the 5D K/V branch (2 * block_len
    # already equals the stride span, so the floor must not fire).
    class _Geom:
        def __init__(self, num_blocks, regions, block_len, block_stride, split):
            self.num_blocks = num_blocks
            self.regions_per_block = regions
            self.block_len = block_len
            self.block_stride = block_stride
            self.split_kv_regions = split

    class _Tensor:
        """Carries shape/stride so the log's page_contiguous branch is exercised."""

        def __init__(self, esize=2):
            self._esize = esize
            self.shape = (13927, 64, 584)

        def element_size(self):
            return self._esize

        def stride(self):
            # stride[1] != shape[2]: a strided page inside the shared block.
            return (1435968 // 2, 1024, 1)

        def __iter__(self):
            return iter([self, self])

    def _run(geom, flag):
        mod = types.ModuleType("stub")
        env = os.environ.get("DSV4_REGION_LEN_SPAN")
        if flag:
            os.environ["DSV4_REGION_LEN_SPAN"] = "1"
        else:
            os.environ.pop("DSV4_REGION_LEN_SPAN", None)
        try:
            mod.__dict__["get_layer_transfer_geometry"] = lambda *a, **k: geom
            exec(compile(out, "stub", "exec"), mod.__dict__)
            return mod.iter_layer_registration_regions("L", _Tensor(), {})
        finally:
            if env is None:
                os.environ.pop("DSV4_REGION_LEN_SPAN", None)
            else:
                os.environ["DSV4_REGION_LEN_SPAN"] = env

    # DSV4 group-0 even layer: shape (13927, 64, 584) fp16 in a 1435968-byte
    # shared block. Per-page block_len is 74752; the stride span is 27.8x larger.
    page = _Geom(13927, 1, 64 * 584 * 2, 1435968 // 2, False)
    span = 13927 * (1435968 // 2) * 2
    off = _run(page, False)
    assert off[0][1] == 13927 * 64 * 584 * 2, off
    on = _run(page, True)
    assert on[0][1] == span, (on, span)
    assert on[0][1] > off[0][1]

    # 5D K/V: regions_per_block=2 and 2 * block_len == block_stride *
    # element_size, so the floor is inert even with the flag on.
    kv = _Geom(100, 2, 4096, 4096, False)
    assert _run(kv, False)[0][1] == _run(kv, True)[0][1] == 100 * 2 * 4096

    # split_kv_regions is skipped outright: region_len there describes one of
    # several caches, so a total-span floor would over-register each.
    sp = _Geom(100, 1, 8, 4096, True)
    sp_on, sp_off = _run(sp, True), _run(sp, False)
    assert [r for _, r in sp_on] == [r for _, r in sp_off] == [800, 800]

    # Idempotent: refuse a second apply rather than inserting FLAG twice.
    try:
        _apply(out)
    except ValueError as e:
        assert "already patched" in str(e)
    else:
        raise AssertionError("second apply should have refused")

    print("[dsv4-region] selftest OK")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir> | --selftest", file=sys.stderr)
        return 2
    if sys.argv[1] == "--selftest":
        return _selftest()

    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[dsv4-region] {REL} not found under {sys.argv[1]} -- skipping.")
        return 0

    src = open(path).read()
    if MARKER in src:
        print(f"[dsv4-region] already patched in {path} -- no-op.")
        return 0

    try:
        out = _apply(src)
    except ValueError as e:
        print(f"[dsv4-region] ERROR: {e}", file=sys.stderr)
        for i, line in enumerate(src.splitlines(), 1):
            if "region_len" in line:
                print(f"[dsv4-region]   line {i}: {line.rstrip()}", file=sys.stderr)
        return 1

    import ast

    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"[dsv4-region] ERROR: compile failed: {e}", file=sys.stderr)
        return 1

    tmp = path + ".dsv4rl"
    with open(tmp, "w") as f:
        f.write(out)
    os.replace(tmp, path)
    print(
        f"[dsv4-region] patched {path}: region_len floored at the view span, "
        "gated on DSV4_REGION_LEN_SPAN (default 0)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
