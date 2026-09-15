#!/usr/bin/env python3
"""Split DSV4 decode ITL: indexer vs sparse MLA vs MoE dispatch/combine.

434150 Flash 2P/2D smoke is ~600 ms ITL with MoRI InterNodeV1LL on mlx5.
DSV3 on the same all2all is 50-84 ms, so the extra is inside the DSV4 step.
This patcher times the Python-visible pieces with CUDA events:

  indexer       SparseAttnIndexer.forward_hip  (graph-breaking on ROCm)
  mla           DeepseekV4ROCMAiterMLAAttention._forward_decode
  moe_dispatch  MoriPrepareAndFinalize.prepare   (all2all send)
  moe_combine   MoriPrepareAndFinalize.finalize  (all2all recv)
  wall          GPUModelRunner.execute_model
  other         wall - (indexer+mla+moe_*)

Diagnostic only. OFF unless DSV4_DECODE_TIMER=1. Prefer DSV4_EAGER=1 on the
profiling cell so MLA/MoE wraps run (FULL_DECODE_ONLY replay skips Python
inside the graph; the indexer already breaks capture). Rank 0 only.
Logs at most DSV4_DECODE_TIMER_MAX decode steps (default 64).

Must run AFTER apply_mori_combine_original_topk_fix.py (wraps the class, not
the combine() string). Idempotent. Missing optional files warn; missing
helper write is a hard error.

Usage: apply_dsv4_decode_step_timer.py <vllm_install_dir>
       apply_dsv4_decode_step_timer.py --selftest
"""
from __future__ import annotations

import os
import sys
import tempfile
import textwrap

MARKER = "DSV4-STEP-TIMER"

HELPER = r'''# DSV4-STEP-TIMER
"""CUDA-event decode-step buckets. Importable from patched vLLM modules."""
from __future__ import annotations

import logging
import os

log = logging.getLogger("vllm")

ON = os.environ.get("DSV4_DECODE_TIMER", "0") == "1"
EVERY = int(os.environ.get("DSV4_DECODE_TIMER_EVERY", "8"))
MAX = int(os.environ.get("DSV4_DECODE_TIMER_MAX", "64"))
DECODE_ONLY = os.environ.get("DSV4_DECODE_TIMER_DECODE_ONLY", "1") == "1"
EAGER = os.environ.get("DSV4_EAGER", "0")

_pairs: dict[str, list] = {}
_wall0 = None
_in_step = False
_logged = 0
_step = 0
_n_tok = -1
_header_logged = False
_capture_warned = False


def _cuda():
    try:
        import torch

        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


def _capturing(torch):
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _rank0():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")) or 0) == 0


class span:
    def __init__(self, name: str):
        self.name = name
        self.skip = True
        self.s = None
        self.e = None

    def __enter__(self):
        torch = _cuda()
        if not ON or not _in_step or torch is None or _capturing(torch):
            return self
        try:
            self.s = torch.cuda.Event(enable_timing=True)
            self.e = torch.cuda.Event(enable_timing=True)
            self.s.record()
            self.skip = False
        except Exception:
            self.skip = True
        return self

    def __exit__(self, *exc):
        if self.skip:
            return
        try:
            self.e.record()
            _pairs.setdefault(self.name, []).append((self.s, self.e))
        except Exception:
            pass


def _log_patch_header() -> None:
    global _header_logged
    if _header_logged or not _rank0():
        return
    _header_logged = True
    log.info(
        "[dsv4-patch-roster] decode_hot combine=%s attn_backend=%s timer=%s "
        "eager=%s | write_boot storage=%s mixed_bs=%s gate=%s attn_xfer=%s "
        "rdma_wait=%s | TP8 applied none of these",
        os.environ.get("DSV4_PATCH_COMBINE", "NOT_CALLED"),
        os.environ.get("DSV4_PATCH_ATTN_BACKEND", "NOT_CALLED"),
        os.environ.get("DSV4_PATCH_TIMER", "NOT_CALLED"),
        EAGER,
        os.environ.get("DSV4_PATCH_STORAGE", "NOT_CALLED"),
        os.environ.get("DSV4_PATCH_MIXED_BS", "NOT_CALLED"),
        os.environ.get("DSV4_PATCH_GATE", "NOT_CALLED"),
        os.environ.get("DSV4_PATCH_ATTN_XFER", "NOT_CALLED"),
        os.environ.get("DSV4_PATCH_RDMA_WAIT", "NOT_CALLED"),
    )


def step_begin(n_tok: int = -1) -> None:
    global _in_step, _pairs, _wall0, _n_tok, _capture_warned
    if not ON:
        return
    torch = _cuda()
    if torch is None or _capturing(torch):
        if ON and torch is not None and _capturing(torch) and not _capture_warned:
            _capture_warned = True
            if _rank0():
                log.warning(
                    "[dsv4-timer] CUDA graph capture hides indexer/mla/moe "
                    "buckets; set DSV4_EAGER=1 (never --enforce-eager)"
                )
        _in_step = False
        return
    _in_step = True
    _pairs = {}
    _n_tok = n_tok
    try:
        _wall0 = torch.cuda.Event(enable_timing=True)
        _wall0.record()
    except Exception:
        _in_step = False


def step_end() -> None:
    global _in_step, _logged, _step, _wall0
    if not ON or not _in_step:
        _in_step = False
        return
    _in_step = False
    _step += 1
    torch = _cuda()
    if torch is None:
        return
    try:
        wall1 = torch.cuda.Event(enable_timing=True)
        wall1.record()
        torch.cuda.synchronize()
    except Exception:
        return
    if DECODE_ONLY and _n_tok > 4:
        return
    if not _rank0():
        return
    buckets = {}
    for name, pairs in _pairs.items():
        ms = 0.0
        for s, e in pairs:
            try:
                ms += float(s.elapsed_time(e))
            except Exception:
                pass
        buckets[name] = ms
    wall = 0.0
    try:
        wall = float(_wall0.elapsed_time(wall1)) if _wall0 is not None else 0.0
    except Exception:
        pass
    known = (
        buckets.get("indexer", 0.0)
        + buckets.get("mla", 0.0)
        + buckets.get("moe_dispatch", 0.0)
        + buckets.get("moe_combine", 0.0)
    )
    other = max(0.0, wall - known)
    if EVERY > 1 and (_step % EVERY) != 0 and _logged > 0:
        return
    if _logged >= MAX:
        return
    _logged += 1
    _log_patch_header()
    log.info(
        "[dsv4-timer] step=%d n_tok=%s wall=%.1f indexer=%.1f mla=%.1f "
        "moe_dispatch=%.1f moe_combine=%.1f other=%.1f n_idx=%d n_mla=%d "
        "n_disp=%d n_comb=%d eager=%s",
        _step,
        _n_tok,
        wall,
        buckets.get("indexer", 0.0),
        buckets.get("mla", 0.0),
        buckets.get("moe_dispatch", 0.0),
        buckets.get("moe_combine", 0.0),
        other,
        len(_pairs.get("indexer", [])),
        len(_pairs.get("mla", [])),
        len(_pairs.get("moe_dispatch", [])),
        len(_pairs.get("moe_combine", [])),
        EAGER,
    )


def wrap_methods(cls, **meth_to_bucket) -> None:
    if not ON:
        return
    for meth, bucket in meth_to_bucket.items():
        orig = getattr(cls, meth, None)
        if orig is None:
            log.warning("[dsv4-timer] %s.%s missing", cls.__name__, meth)
            continue

        def _make(o, b):
            def _wrapped(self, *a, **k):
                with span(b):
                    return o(self, *a, **k)

            _wrapped.__name__ = getattr(o, "__name__", b)
            _wrapped.__doc__ = getattr(o, "__doc__", None)
            return _wrapped

        setattr(cls, meth, _make(orig, bucket))
        log.info("[dsv4-timer] wrapped %s.%s -> %s", cls.__name__, meth, bucket)


def _scheduled_tokens(args, kwargs) -> int:
    so = args[0] if args else kwargs.get("scheduler_output")
    if so is None:
        return -1
    for attr in ("total_num_scheduled_tokens", "num_scheduled_tokens"):
        v = getattr(so, attr, None)
        if v is None:
            continue
        if isinstance(v, int):
            return v
        try:
            return int(sum(v.values()) if hasattr(v, "values") else v)
        except Exception:
            continue
    return -1


def patch_execute_model(cls) -> None:
    if not ON:
        return
    orig = getattr(cls, "execute_model", None)
    if orig is None:
        log.warning("[dsv4-timer] %s.execute_model missing", cls.__name__)
        return

    def _wrapped(self, *a, **k):
        step_begin(_scheduled_tokens(a, k))
        try:
            return orig(self, *a, **k)
        finally:
            step_end()

    cls.execute_model = _wrapped
    log.info("[dsv4-timer] wrapped %s.execute_model", cls.__name__)
'''

TAIL = {
    "model_executor/layers/sparse_attn_indexer.py": textwrap.dedent(
        """

        # DSV4-STEP-TIMER
        try:
            from vllm._dsv4_step_timer import wrap_methods
            wrap_methods(SparseAttnIndexer, forward_hip="indexer")
        except Exception as _dsv4_timer_err:
            import logging as _dsv4_timer_log
            _dsv4_timer_log.getLogger("vllm").warning(
                "[dsv4-timer] indexer wrap failed: %s", _dsv4_timer_err
            )
        """
    ),
    "model_executor/layers/fused_moe/prepare_finalize/mori.py": textwrap.dedent(
        """

        # DSV4-STEP-TIMER
        try:
            from vllm._dsv4_step_timer import wrap_methods
            wrap_methods(
                MoriPrepareAndFinalize,
                prepare="moe_dispatch",
                finalize="moe_combine",
            )
        except Exception as _dsv4_timer_err:
            import logging as _dsv4_timer_log
            _dsv4_timer_log.getLogger("vllm").warning(
                "[dsv4-timer] mori wrap failed: %s", _dsv4_timer_err
            )
        """
    ),
    "models/deepseek_v4/amd/rocm.py": textwrap.dedent(
        """

        # DSV4-STEP-TIMER
        try:
            from vllm._dsv4_step_timer import wrap_methods
            wrap_methods(
                DeepseekV4ROCMAiterMLAAttention,
                _forward_decode="mla",
            )
        except Exception as _dsv4_timer_err:
            import logging as _dsv4_timer_log
            _dsv4_timer_log.getLogger("vllm").warning(
                "[dsv4-timer] mla wrap failed: %s", _dsv4_timer_err
            )
        """
    ),
    "v1/worker/gpu_model_runner.py": textwrap.dedent(
        """

        # DSV4-STEP-TIMER
        try:
            from vllm._dsv4_step_timer import patch_execute_model
            patch_execute_model(GPUModelRunner)
        except Exception as _dsv4_timer_err:
            import logging as _dsv4_timer_log
            _dsv4_timer_log.getLogger("vllm").warning(
                "[dsv4-timer] execute_model wrap failed: %s", _dsv4_timer_err
            )
        """
    ),
}


def _append_once(path: str, tail: str) -> str:
    if not os.path.isfile(path):
        return "missing"
    src = open(path).read()
    if MARKER in src:
        return "already"
    with open(path, "a") as f:
        f.write(tail)
        if not tail.endswith("\n"):
            f.write("\n")
    return "ok"


def apply(vllm_dir: str) -> int:
    helper = os.path.join(vllm_dir, "_dsv4_step_timer.py")
    with open(helper, "w") as f:
        f.write(HELPER)
    print(f"[dsv4-timer] wrote {helper}")
    rc = 0
    landed = 0
    for rel, tail in TAIL.items():
        path = os.path.join(vllm_dir, rel)
        st = _append_once(path, tail)
        print(f"[dsv4-timer] {rel}: {st}")
        if st == "ok" or st == "already":
            landed += 1
        elif rel.endswith("mori.py") or rel.endswith("sparse_attn_indexer.py"):
            print(f"[dsv4-timer] ERROR: required {rel} {st}", file=sys.stderr)
            rc = 1
    if landed == 0:
        print("[dsv4-timer] ERROR: no wrap targets found", file=sys.stderr)
        return 1
    return rc


def selftest() -> int:
    tmp = tempfile.mkdtemp(prefix="dsv4timer_")
    os.makedirs(os.path.join(tmp, "model_executor/layers/fused_moe/prepare_finalize"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "models/deepseek_v4/amd"), exist_ok=True)
    os.makedirs(os.path.join(tmp, "v1/worker"), exist_ok=True)
    stubs = {
        "model_executor/layers/sparse_attn_indexer.py": "class SparseAttnIndexer:\n    def forward_hip(self):\n        return 1\n",
        "model_executor/layers/fused_moe/prepare_finalize/mori.py": (
            "class MoriPrepareAndFinalize:\n"
            "    def prepare(self):\n        return 2\n"
            "    def finalize(self):\n        return 3\n"
        ),
        "models/deepseek_v4/amd/rocm.py": (
            "class DeepseekV4ROCMAiterMLAAttention:\n"
            "    def _forward_decode(self):\n        return 4\n"
        ),
        "v1/worker/gpu_model_runner.py": (
            "class GPUModelRunner:\n"
            "    def execute_model(self, scheduler_output=None):\n        return 5\n"
        ),
    }
    for rel, body in stubs.items():
        open(os.path.join(tmp, rel), "w").write(body)
    rc = apply(tmp)
    if rc != 0:
        print("FAIL apply rc", rc)
        return 1
    for rel in stubs:
        src = open(os.path.join(tmp, rel)).read()
        if MARKER not in src:
            print("FAIL marker missing", rel)
            return 1
        # second apply is no-op
    rc2 = apply(tmp)
    if rc2 != 0:
        print("FAIL second apply", rc2)
        return 1
    for rel in stubs:
        n = open(os.path.join(tmp, rel)).read().count(MARKER)
        if n != 1:
            print("FAIL not idempotent", rel, n)
            return 1
    helper = open(os.path.join(tmp, "_dsv4_step_timer.py")).read()
    if "moe_dispatch" not in helper or "span" not in helper:
        print("FAIL helper content")
        return 1
    if "dsv4-patch-roster" not in helper or "CUDA graph capture hides" not in helper:
        print("FAIL helper missing patch roster / capture warn")
        return 1
    print("ok  selftest")
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return selftest()
    if len(sys.argv) != 2:
        print(
            "usage: apply_dsv4_decode_step_timer.py <vllm_install_dir>\n"
            "       apply_dsv4_decode_step_timer.py --selftest",
            file=sys.stderr,
        )
        return 2
    return apply(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
