#!/usr/bin/env python3
"""Register Flash non-contiguous KV via a data_ptr-aligned storage span.

217514: register_torch_tensor raised 'input tensor must be contiguous'.
217532: skip-noncontiguous dropped all 125 remaining caches (MLA too) then
StopIteration on next(iter(kv_caches)). .contiguous() would RDMA a copy.

217546: first version registered a 1D uint8 view of the *entire*
untyped_storage from byte 0. Offsets are relative to the tensor view
(packed DSV4 stride(0) is the per-block span). Rank 0 Lisa Su; rank 1 hung
then WRONG. Align the MR to kv_cache.data_ptr() and the strided span.

vllm#48989's non-contiguous hunk is the same intent, different size::

    cache_u8 = kv_cache.view(torch.uint8)          # fails if non-contiguous
    total_bytes = shape[0] * stride(0) * element_size
    torch.as_strided(cache_u8, (total_bytes,), (1,), storage_offset)

``.view(uint8)`` is rejected on DSV4's strided caches, so we keep
``untyped_storage().set_(off, (span,))`` — that *is* a 1D uint8 as_strided
cover. Default span is the tight view bbox (218443 / 217546). Flag 1 uses
#48989's ``N * stride[0] * es`` so a full-block transfer
(``block_len = stride[0] * es``, GEOM=1) cannot overrun the last page.
Clamp to remaining storage so we never reproduce 217546's whole-allocation
MR. Keep original tensors in self.kv_caches (stride/offset math).

    DSV4_STORAGE_UPSTREAM_SPAN=0   (default) tight bbox
    DSV4_STORAGE_UPSTREAM_SPAN=1            vllm#48989 N*stride[0]*es

Gated to DeepSeek-V4-Flash-FP8 / Pro. GLM/DSV3/Hy3 never run this.
Idempotent. Missing file -> skip. Found-old that fails is a hard error.

Usage: apply_moriio_dsv4_skip_noncontiguous_register_fix.py <vllm_install_dir>
       apply_moriio_dsv4_skip_noncontiguous_register_fix.py --selftest
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
MARKER = "DSV4-STORAGE-SPAN"
UPSTREAM_MARKER = "DSV4-STORAGE-UPSTREAM-SPAN"

FLAG = '''
# DSV4-STORAGE-UPSTREAM-SPAN: vllm#48989 MR size N*stride[0]*es vs tight bbox.
import os as _dsv4_stor_os

DSV4_STORAGE_UPSTREAM_SPAN = (
    _dsv4_stor_os.environ.get("DSV4_STORAGE_UPSTREAM_SPAN", "0") == "1"
)
'''

OLD_BARE = """            moriio_mem_metadata = self.moriio_wrapper.register_local_tensor(kv_cache)
"""

OLD_V1 = """            # DSV4-STORAGE: 217514/217532. Flash KV views are non-contiguous;
            # MoRI requires contiguous. Do not .contiguous() (copy). Register
            # a 1D view of the live untyped_storage. Keep kv_cache for offsets.
            _reg = kv_cache
            if not kv_cache.is_contiguous():
                _reg = kv_cache.new_empty((0,), dtype=torch.uint8)
                _reg.set_(kv_cache.untyped_storage())
                logger.warning(
                    "[dsv4-storage] layer %s not contiguous; register storage "
                    "view nbytes=%d offset=%d",
                    layer_name,
                    int(_reg.untyped_storage().nbytes()),
                    int(kv_cache.storage_offset()),
                )
            moriio_mem_metadata = self.moriio_wrapper.register_local_tensor(_reg)
"""

OLD_SPAN = """            # DSV4-STORAGE-SPAN: 217514/217532/217546. Flash KV is a strided
            # packed view. MoRI requires contiguous. Do not .contiguous()
            # (copy). Register a 1D uint8 view of the live storage spanning
            # this tensor, starting at data_ptr — not the whole allocation
            # from byte 0. Keep kv_cache for offset math.
            _reg = kv_cache
            if not kv_cache.is_contiguous():
                _stor = kv_cache.untyped_storage()
                _span = 0
                if kv_cache.numel() > 0:
                    _end = 0
                    for _sz, _st in zip(kv_cache.size(), kv_cache.stride()):
                        if _sz:
                            _end += (_sz - 1) * _st
                    _span = int(kv_cache.element_size() * (_end + 1))
                _off = int(kv_cache.data_ptr()) - int(_stor.data_ptr())
                if _off < 0:
                    _off = 0
                _reg = kv_cache.new_empty((0,), dtype=torch.uint8)
                _reg.set_(_stor, _off, (_span,))
                logger.warning(
                    "[dsv4-storage] layer %s not contiguous; register storage "
                    "span nbytes=%d offset=%d span=%d",
                    layer_name,
                    int(_stor.nbytes()),
                    _off,
                    _span,
                )
            moriio_mem_metadata = self.moriio_wrapper.register_local_tensor(_reg)
"""

NEW = """            # DSV4-STORAGE-SPAN: 217514/217532/217546. Flash KV is a strided
            # packed view. MoRI requires contiguous. Do not .contiguous()
            # (copy). Register a 1D uint8 view of the live storage spanning
            # this tensor, starting at data_ptr — not the whole allocation
            # from byte 0. Keep kv_cache for offset math.
            # DSV4-STORAGE-UPSTREAM-SPAN: vllm#48989 is
            # as_strided(view(uint8), (shape[0]*stride(0)*es,), (1,), offset).
            # view(uint8) raises on a non-contiguous tensor, so set_() of
            # that byte count is the same 1D cover. Default 0 = tight bbox
            # (218443). Flag 1 = N*stride[0]*es so GEOM=1 full-block copies
            # cannot overrun the last page.
            _reg = kv_cache
            if not kv_cache.is_contiguous():
                _stor = kv_cache.untyped_storage()
                _es = int(kv_cache.element_size())
                _span = 0
                if kv_cache.numel() > 0:
                    if DSV4_STORAGE_UPSTREAM_SPAN and kv_cache.dim() >= 1:
                        _span = int(kv_cache.shape[0] * kv_cache.stride(0) * _es)
                    else:
                        _end = 0
                        for _sz, _st in zip(kv_cache.size(), kv_cache.stride()):
                            if _sz:
                                _end += (_sz - 1) * _st
                        _span = int(_es * (_end + 1))
                _off = int(kv_cache.data_ptr()) - int(_stor.data_ptr())
                if _off < 0:
                    _off = 0
                _remain = int(_stor.nbytes()) - _off
                if _span > _remain:
                    _span = max(_remain, 0)
                _reg = kv_cache.new_empty((0,), dtype=torch.uint8)
                _reg.set_(_stor, _off, (_span,))
                logger.warning(
                    "[dsv4-storage] layer %s not contiguous; register storage "
                    "span nbytes=%d offset=%d span=%d upstream=%s",
                    layer_name,
                    int(_stor.nbytes()),
                    _off,
                    _span,
                    DSV4_STORAGE_UPSTREAM_SPAN,
                )
            moriio_mem_metadata = self.moriio_wrapper.register_local_tensor(_reg)
"""


def _insert_flag(src: str) -> str:
    if "DSV4_STORAGE_UPSTREAM_SPAN = (" in src:
        return src
    anchor = "\nlogger = init_logger(__name__)\n"
    if anchor in src:
        return src.replace(anchor, anchor + FLAG, 1)
    raise ValueError("anchor missing: logger = init_logger")


def _apply(src: str) -> tuple[str, str]:
    """Return (patched_src, kind). kind is 'noop' when already current."""
    if "DSV4-CONTIG" in src:
        raise ValueError("DSV4-CONTIG skip-all still present; refusing")
    if "upstream=%s" in src and UPSTREAM_MARKER in src:
        return src, "noop"
    if OLD_V1 in src:
        return _insert_flag(src.replace(OLD_V1, NEW, 1)), "upgrade v1 full-storage -> SPAN+48989"
    if OLD_SPAN in src:
        return _insert_flag(src.replace(OLD_SPAN, NEW, 1)), "upgrade tight bbox -> SPAN+48989"
    if OLD_BARE in src:
        return _insert_flag(src.replace(OLD_BARE, NEW, 1)), "fresh SPAN+48989"
    if MARKER in src and UPSTREAM_MARKER not in src:
        raise ValueError(
            "SPAN present but neither OLD_SPAN nor upstream marker matched"
        )
    raise ValueError("register_local_tensor anchor missing")


def _selftest() -> int:
    import ast

    stub = (
        "from vllm.logger import init_logger\n"
        "\nlogger = init_logger(__name__)\n"
        "\nclass C:\n"
        "    def register(self, kv_cache, layer_name):\n"
        + OLD_BARE
    )
    out, kind = _apply(stub)
    assert kind == "fresh SPAN+48989", kind
    assert MARKER in out
    assert UPSTREAM_MARKER in out
    assert "shape[0] * kv_cache.stride(0)" in out
    assert "upstream=%s" in out
    ast.parse(out)
    out2, kind2 = _apply(out)
    assert kind2 == "noop", kind2
    assert out2 == out

    old = (
        "from vllm.logger import init_logger\n"
        "\nlogger = init_logger(__name__)\n"
        "\nclass C:\n"
        "    def register(self, kv_cache, layer_name):\n"
        + OLD_SPAN
    )
    up, kind3 = _apply(old)
    assert kind3.startswith("upgrade"), kind3
    assert "DSV4_STORAGE_UPSTREAM_SPAN and kv_cache.dim()" in up
    ast.parse(up)

    # Span arithmetic: shared-block DSV4 view (N, slots, latent), es=1.
    n, slots, latent, s0 = 10, 64, 584, 1435968
    tight = (n - 1) * s0 + (slots - 1) * latent + (latent - 1) * 1 + 1
    upstream = n * s0
    assert tight == 9 * s0 + slots * latent
    assert upstream - tight == s0 - slots * latent
    assert upstream > tight
    print("[dsv4-storage] selftest OK")
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return _selftest()
    if len(sys.argv) != 2:
        print(
            f"usage: {sys.argv[0]} <vllm_install_dir> | --selftest",
            file=sys.stderr,
        )
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[dsv4-storage] {REL} not found under {sys.argv[1]} -- skipping.")
        return 0

    src = open(path).read()
    try:
        out, kind = _apply(src)
    except ValueError as e:
        print(f"[dsv4-storage] ERROR: {e} in {path}.", file=sys.stderr)
        if "anchor missing" in str(e):
            for i, line in enumerate(src.splitlines(), 1):
                if "register_local_tensor" in line:
                    print(f"[dsv4-storage]   line {i}: {line.rstrip()}", file=sys.stderr)
        return 1

    if kind == "noop":
        print(f"[dsv4-storage] already patched (SPAN+48989) in {path} -- no-op.")
        return 0

    if MARKER not in out or UPSTREAM_MARKER not in out:
        print(
            f"[dsv4-storage] ERROR: post-write missing marker in {path}.",
            file=sys.stderr,
        )
        return 1

    tmp = path + ".dsv4stg"
    with open(tmp, "w") as f:
        f.write(out)
    os.replace(tmp, path)
    try:
        import py_compile

        py_compile.compile(path, doraise=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dsv4-storage] ERROR: compile failed: {e}", file=sys.stderr)
        return 1
    print(
        f"[dsv4-storage] patched ({kind}): data_ptr-aligned span; "
        "DSV4_STORAGE_UPSTREAM_SPAN=1 is vllm#48989 N*stride[0]*es."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
