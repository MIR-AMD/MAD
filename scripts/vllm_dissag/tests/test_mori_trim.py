#!/usr/bin/env python3
"""Offline test for apply_mori_trim_dispatch.py's row-bound decision.

The patcher hands the MoE expert stack `world_size * max_tokens_on_any_rank`
rows instead of MoRI's whole preallocated recv buffer (436457: 8192 rows for
one real token, 79.8% of decode GPU time in one GEMM). Getting that bound
wrong in either direction is expensive:

  - too large and the fix does nothing;
  - too small and tokens are silently DROPPED, which no benchmark number would
    reveal -- the answer just gets worse.

So the decision logic is asserted here rather than on a GPU. The methods are
lifted out of the patcher's replacement text and exec'd against stand-ins for
torch, the vLLM forward context and the logger, so this needs no GPU, no vLLM
and no MoRI.

Properties asserted:

  1. Unset MORI_TRIM_DISPATCH returns None (stock behaviour, byte-identical).
  2. Armed, the bound is world_size * max(num_tokens_across_dp_cpu) -- the MAX
     across ranks, not this rank's count, because dispatch is collective.
  3. Prefill, where the bound meets or exceeds the buffer, returns None so
     those forwards are untouched.
  4. Absent DP metadata returns None and warns once, rather than guessing a
     rank-local bound.
  5. MORI_TRIM_CHECK catches a violated bound and returns None (fail safe to
     the slow-but-correct path), and is skipped under cudagraph capture where
     a host sync is illegal.

Usage: python3 tests/test_mori_trim.py
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATCHER = os.path.join(HERE, os.pardir, "apply_mori_trim_dispatch.py")

_failures = []


def _check(label, ok):
    print(("  ok   " if ok else "  FAIL ") + label)
    if not ok:
        _failures.append(label)


# --- stand-ins -------------------------------------------------------------


class _Tensor:
    """Just enough tensor for the bound logic: a shape and a summable value."""

    def __init__(self, rows, cols=4096, total=0):
        self.shape = (rows, cols)
        self._total = total

    def sum(self):
        return self

    def item(self):
        return self._total

    def max(self):
        return self._total


class _Counts:
    """Stands in for DPMetadata.num_tokens_across_dp_cpu (a CPU tensor)."""

    def __init__(self, per_rank):
        self._per_rank = list(per_rank)

    def max(self):
        return max(self._per_rank)


class _Logger:
    def __init__(self):
        self.warnings = []
        self.errors = []

    def warning(self, msg, *a):
        self.warnings.append(msg % a if a else msg)

    def error(self, msg, *a):
        self.errors.append(msg % a if a else msg)

    def exception(self, msg, *a):
        self.errors.append(msg % a if a else msg)


def _load_methods():
    """exec the patcher's inserted methods into a throwaway class."""
    spec = importlib.util.spec_from_file_location("trim_patcher", PATCHER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    body = mod.NEW_PREPARE_DEF
    # Drop the trailing `def prepare(...)` anchor lines; keep only our methods.
    body = body.split("    def prepare(")[0]
    src = "class MoriPrepareAndFinalize:\n" + body

    logger = _Logger()
    ns = {"logger": logger}
    exec(compile(src, "<patched mori.py>", "exec"), ns)
    return ns["MoriPrepareAndFinalize"], logger


class _Op:
    def __init__(self, world_size):
        self.config = type("cfg", (), {"world_size": world_size})()


def _make(world_size=8, per_rank=(1,) * 8, capturing=False):
    """A patched instance wired to fake torch + forward context."""
    cls, logger = _load_methods()
    obj = cls()
    obj.mori_op = _Op(world_size)

    torch_stub = type(sys)("torch")
    torch_stub.cuda = type(sys)("torch.cuda")
    torch_stub.cuda.is_current_stream_capturing = lambda: capturing
    sys.modules["torch"] = torch_stub

    fc = type(sys)("vllm.forward_context")
    if per_rank is None:
        ctx = type("ctx", (), {"dp_metadata": None})()
    else:
        dp = type("dp", (), {"num_tokens_across_dp_cpu": _Counts(per_rank)})()
        ctx = type("ctx", (), {"dp_metadata": dp})()
    fc.get_forward_context = lambda: ctx
    vllm = type(sys)("vllm")
    vllm.forward_context = fc
    sys.modules["vllm"] = vllm
    sys.modules["vllm.forward_context"] = fc
    return obj, logger


def main():
    print("apply_mori_trim_dispatch: row-bound decision")

    # 1. Unset is stock. The real buffer at mnbt 1024 / EP8 is 8192 rows.
    os.environ.pop("MORI_TRIM_DISPATCH", None)
    os.environ.pop("MORI_TRIM_CHECK", None)
    obj, _ = _make()
    _check(
        "unset MORI_TRIM_DISPATCH -> None (no trim)",
        obj._mori_trim_rows(_Tensor(1), _Tensor(8192), _Tensor(1, total=8)) is None,
    )

    os.environ["MORI_TRIM_DISPATCH"] = "1"

    # 2. Decode: one token per rank on 8 ranks -> 8 rows, not 8192.
    obj, _ = _make(world_size=8, per_rank=(1,) * 8)
    _check(
        "decode con=1 EP8 -> bound 8 (was 8192)",
        obj._mori_trim_rows(_Tensor(1), _Tensor(8192), _Tensor(1, total=8)) == 8,
    )

    # The bound must use the MAX across ranks. A rank holding 1 token while
    # another holds 5 can still receive 5 from that peer.
    obj, _ = _make(world_size=8, per_rank=(1, 5, 1, 1, 1, 1, 1, 1))
    _check(
        "ragged DP batch -> bound uses max across ranks (8*5=40)",
        obj._mori_trim_rows(_Tensor(1), _Tensor(8192), _Tensor(1, total=40)) == 40,
    )

    # 3. Prefill/profile_run: bound meets the buffer, so leave it alone.
    obj, _ = _make(world_size=8, per_rank=(1024,) * 8)
    _check(
        "prefill 1024 tok/rank -> None (bound == buffer, untouched)",
        obj._mori_trim_rows(_Tensor(1024), _Tensor(8192), _Tensor(1, total=8192))
        is None,
    )

    # A chunk larger than capacity must not produce a bound past the buffer.
    obj, _ = _make(world_size=8, per_rank=(4096,) * 8)
    _check(
        "over-capacity batch -> None, never a bound past the buffer",
        obj._mori_trim_rows(_Tensor(4096), _Tensor(8192), _Tensor(1, total=8192))
        is None,
    )

    # 4. No DP metadata -> refuse to guess, and say so exactly once.
    obj, logger = _make(per_rank=None)
    r1 = obj._mori_trim_rows(_Tensor(1), _Tensor(8192), _Tensor(1, total=8))
    r2 = obj._mori_trim_rows(_Tensor(1), _Tensor(8192), _Tensor(1, total=8))
    _check("missing DP metadata -> None (no rank-local guess)",
           r1 is None and r2 is None)
    _check("missing DP metadata warns exactly once",
           len([w for w in logger.warnings if "rank-consistent" in w]) == 1)

    # 5. The tripwire.
    os.environ["MORI_TRIM_CHECK"] = "4"

    obj, logger = _make(world_size=8, per_rank=(1,) * 8)
    _check(
        "check armed, bound holds (recv 8 <= 8) -> trims",
        obj._mori_trim_rows(_Tensor(1), _Tensor(8192), _Tensor(1, total=8)) == 8,
    )
    _check("check logs the M reduction it verified",
           any("total_recv=8 <= bound=8" in w for w in logger.warnings))

    obj, logger = _make(world_size=8, per_rank=(1,) * 8)
    _check(
        "check armed, bound VIOLATED (recv 9 > 8) -> None, fails safe",
        obj._mori_trim_rows(_Tensor(1), _Tensor(8192), _Tensor(1, total=9)) is None,
    )
    _check("violation is logged as an error",
           any("BOUND VIOLATED" in e for e in logger.errors))

    # Exhausting the budget stops syncing but keeps trimming.
    obj, logger = _make(world_size=8, per_rank=(1,) * 8)
    for _ in range(4):
        obj._mori_trim_rows(_Tensor(1), _Tensor(8192), _Tensor(1, total=8))
    before = len(logger.warnings)
    # A violation past the budget is no longer detected -- by design, the
    # budget is a bring-up tripwire and not a permanent guard.
    obj._mori_trim_rows(_Tensor(1), _Tensor(8192), _Tensor(1, total=9))
    _check("check stops after MORI_TRIM_CHECK calls",
           len(logger.warnings) == before)

    # Under capture a sync is illegal, so the check passes without one.
    obj, logger = _make(world_size=8, per_rank=(1,) * 8, capturing=True)
    _check(
        "cudagraph capture -> trims without syncing",
        obj._mori_trim_rows(_Tensor(1), _Tensor(8192), _Tensor(1, total=8)) == 8,
    )
    _check("capture path logs nothing and counts nothing",
           not logger.warnings and not logger.errors)

    if _failures:
        print("\n%d check(s) FAILED" % len(_failures))
        return 1
    print("\nall mori-trim checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
