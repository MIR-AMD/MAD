#!/usr/bin/env python3
"""Detect the attention backend per-cache at register time (vllm#48989).

The shipped connector asks once, in ``__init__``, with a single scalar query::

    backend = get_attn_backend(
        self.model_config.get_head_size(),
        self.model_config.dtype,
        self.cache_config.cache_dtype,
        use_mla=self.use_mla,
    )
    self.backend_name = backend.get_name()

One head size, one dtype, one ``use_mla`` -- so one answer. That is fine for
DSV3, whose caches are all MLA. DSV4 is hybrid: group 0 is ``MLAAttentionSpec``
and groups 1-4 are ``SlidingWindowMLASpec``, and the Lightning Indexer adds
``.indexer.k_cache`` on top. A single query cannot describe that set, and
``backend_name`` feeds layout decisions for every layer.

vllm-project/vllm#48989 moves the query into ``register_kv_caches`` and asks for
the *list*::

    backends = get_current_attn_backends(self.vllm_config)
    self.backend_name = backends[0].get_name()

Two changes in one: it runs late enough that the caches exist, and it reads the
backends actually in use rather than re-deriving one from config scalars.

This patcher is additive on purpose. Rather than deleting the ``__init__``
query (which boots fine today), it re-detects at ``register_kv_caches`` and
logs the whole list next to the old value, so the log answers whether the
single query was ever wrong for DSV4 before any behaviour depends on it. If
``get_current_attn_backends`` is absent from the image, it keeps the shipped
value and says so -- never a boot failure over telemetry.

Runtime-gated so one image serves both arms::

    DSV4_BACKEND_DETECT=0   (default) shipped single __init__ query
    DSV4_BACKEND_DETECT=1            re-detect at register_kv_caches

Gated in connectors/moriio.sh to DSV4 Flash/Pro. GLM / DSV3 / Hy3 never run
this. Idempotent. Missing anchor is a hard error; missing file is a skip.

Usage: apply_moriio_dsv4_backend_detect_fix.py <vllm_install_dir>
       apply_moriio_dsv4_backend_detect_fix.py --selftest
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
MARKER = "DSV4-BACKEND-DETECT"

FLAG = '''
# DSV4-BACKEND-DETECT: vllm#48989. The shipped connector derives backend_name
# from one get_attn_backend() call in __init__, which cannot describe DSV4's
# hybrid set (MLAAttentionSpec group 0 + SlidingWindowMLASpec groups 1-4 +
# .indexer.k_cache). Re-detect from the live backend list at register time.
import os as _dsv4_os

DSV4_BACKEND_DETECT = _dsv4_os.environ.get("DSV4_BACKEND_DETECT", "0") == "1"
'''

REG_OLD = '''    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in moriio."""
'''

REG_NEW = '''    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in moriio."""

        if DSV4_BACKEND_DETECT:
            # DSV4-BACKEND-DETECT: vllm#48989 asks for the backend list here,
            # after the caches exist, instead of re-deriving one backend from
            # config scalars in __init__. Log both so the single-query answer
            # can be compared against the set actually in use.
            try:
                from vllm.distributed.kv_transfer.kv_connector.utils import (
                    get_current_attn_backends as _dsv4_get_backends,
                )
            except ImportError:
                _dsv4_get_backends = None
            if _dsv4_get_backends is None:
                logger.warning(
                    "[dsv4-backend] get_current_attn_backends not in this image; "
                    "keeping shipped backend_name=%s",
                    getattr(self, "backend_name", None),
                )
            else:
                _dsv4_backends = _dsv4_get_backends(self.vllm_config) or []
                _dsv4_names = [_b.get_name() for _b in _dsv4_backends]
                _dsv4_shipped = getattr(self, "backend_name", None)
                if _dsv4_backends:
                    self.backend_name = _dsv4_names[0]
                logger.info(
                    "[dsv4-backend] register_kv_caches backends=%s n=%d "
                    "shipped=%s -> %s",
                    _dsv4_names,
                    len(_dsv4_names),
                    _dsv4_shipped,
                    getattr(self, "backend_name", None),
                )
'''


def _apply(src: str) -> str:
    # The anchor is the def line plus docstring, and the replacement keeps both,
    # so it survives the edit. Unlike patchers whose anchors are consumed, this
    # one has to refuse explicitly or a second pass would insert twice.
    if MARKER in src:
        raise ValueError("already patched")
    if REG_OLD not in src:
        raise ValueError("anchor missing: register_kv_caches def + docstring")
    src = src.replace(REG_OLD, REG_NEW, 1)

    anchor = "\nlogger = init_logger(__name__)\n"
    if anchor in src:
        return src.replace(anchor, anchor + FLAG, 1)

    import ast

    tree = ast.parse(src)
    cut = 0
    body = tree.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        cut = body[0].end_lineno or 0
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            cut = max(cut, node.end_lineno or node.lineno)
    lines = src.splitlines(keepends=True)
    return "".join(lines[:cut]) + FLAG + "".join(lines[cut:])


def _selftest() -> int:
    import ast
    import types

    stub = (
        "from vllm.logger import init_logger\n"
        "\nlogger = init_logger(__name__)\n"
        "\n\nclass C:\n"
        + REG_OLD
        + "        self.kv_caches = kv_caches\n"
    )
    out = _apply(stub)
    assert MARKER in out
    assert "get_current_attn_backends" in out
    assert "DSV4_BACKEND_DETECT" in out
    ast.parse(out)

    # Same shape that broke 218197/218198 on the geometry patcher: no logger
    # anchor, module docstring, and a parenthesised import to fall inside.
    realistic = (
        '"""MoRIIO connector.\n\nSecond line.\n"""\n'
        "import torch\n"
        "from vllm.config import (\n"
        "    CUDAGraphMode,\n"
        "    VllmConfig,\n"
        ")\n"
        "\n\nclass C:\n" + REG_OLD + "        self.kv_caches = kv_caches\n"
    )
    assert "logger = init_logger(__name__)" not in realistic
    out_real = _apply(realistic)
    ast.parse(out_real)
    assert "    CUDAGraphMode,\n    VllmConfig,\n)\n" in out_real, (
        "FLAG split the parenthesised import"
    )

    # Behaviour: absent helper must keep the shipped name, present helper must
    # take the first of the list, and both must log rather than raise.
    class _Backend:
        def __init__(self, name):
            self._name = name

        def get_name(self):
            return self._name

    logged = []

    class _Logger:
        def info(self, *a):
            logged.append(("info", a))

        def warning(self, *a):
            logged.append(("warning", a))

    saved = os.environ.get("DSV4_BACKEND_DETECT")
    try:
        os.environ["DSV4_BACKEND_DETECT"] = "1"
        body = REG_NEW.split('"""Register the KV Cache data in moriio."""\n', 1)[1]
        code = "def run(self):\n" + body
        for helper, expect in (
            (None, "fp8_ds_mla"),
            ([_Backend("TRITON_MLA"), _Backend("FLASH_ATTN")], "TRITON_MLA"),
        ):
            ns = {
                "DSV4_BACKEND_DETECT": True,
                "logger": _Logger(),
                "__dsv4_helper": helper,
            }
            # Stand in for the import: absent helper raises ImportError just as
            # an older image would.
            src = code.replace(
                "from vllm.distributed.kv_transfer.kv_connector.utils import (\n"
                "                    get_current_attn_backends as _dsv4_get_backends,\n"
                "                )",
                "_dsv4_get_backends = (\n"
                "                    (lambda _c: __dsv4_helper)\n"
                "                    if __dsv4_helper is not None else None\n"
                "                )",
            )
            exec(compile(src, "<t>", "exec"), ns)  # noqa: S102
            obj = types.SimpleNamespace(
                backend_name="fp8_ds_mla", vllm_config=object()
            )
            ns["run"](obj)
            assert obj.backend_name == expect, (obj.backend_name, expect)
        kinds = [k for k, _ in logged]
        assert "warning" in kinds, "absent helper must warn"
        assert "info" in kinds, "present helper must log the list"
    finally:
        if saved is None:
            os.environ.pop("DSV4_BACKEND_DETECT", None)
        else:
            os.environ["DSV4_BACKEND_DETECT"] = saved

    try:
        _apply(out)
    except ValueError:
        pass
    else:
        raise AssertionError("second apply should not find the anchor")

    print("[dsv4-backend] selftest OK")
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return _selftest()
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir> | --selftest", file=sys.stderr)
        return 2

    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[dsv4-backend] {REL} not found under {sys.argv[1]} -- skipping.")
        return 0

    src = open(path).read()
    if MARKER in src:
        print(f"[dsv4-backend] already patched in {path} -- no-op.")
        return 0

    try:
        out = _apply(src)
    except ValueError as e:
        print(f"[dsv4-backend] ERROR: {e}", file=sys.stderr)
        for i, line in enumerate(src.splitlines(), 1):
            if "def register_kv_caches" in line:
                print(f"[dsv4-backend]   line {i}: {line.rstrip()}", file=sys.stderr)
        return 1

    tmp = path + ".dsv4bk"
    with open(tmp, "w") as f:
        f.write(out)
    os.replace(tmp, path)

    try:
        import py_compile

        py_compile.compile(path, doraise=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dsv4-backend] ERROR: compile failed: {e}", file=sys.stderr)
        return 1

    print(
        f"[dsv4-backend] patched {path}: backend re-detect at register_kv_caches "
        "gated on DSV4_BACKEND_DETECT (default 0)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
