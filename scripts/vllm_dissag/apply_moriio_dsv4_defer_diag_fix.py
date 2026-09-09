#!/usr/bin/env python3
"""Log why a MoRIIO write task defers, and what the readiness dict holds.

Built for the 218282 hang. Four requests (proxy ranks 0-3) each completed RDMA
in ~1ms; the fifth (rank 4) entered ``wait_for_save`` and never emitted a
finalize, then at 60s the connector logged::

    Reaped 1 deferred sends with no finished_sending notification after 60s.
    This indicates lost async KV completion notifications from the KV connector.

and the proxy stalled that leg for 2100s until walltime. That is not slow
transport -- 60s is exactly ``MoRIIOConstants.DEFAULT_DEFER_TIMEOUT``.

The readiness test behind the defer is one line (``moriio_engine.py:279``)::

    return task.transfer_id in self.worker.moriio_wrapper.done_remote_allocate_req_dict

So a deferred write does not mean the transport is busy or the handshake is
missing: it means **D never told P its block allocation for that transfer id**.
``_write_worker_loop`` re-checks every 10ms until ``defer_timeout``, then
``_mark_request_done`` force-frees the blocks and P moves on having written
nothing, leaving D waiting on KV that will never arrive.

Three outcomes are indistinguishable in the shipped logs, and they have
different fixes:

* the dict is **empty** -- D's allocation notify never reached P (lost message,
  wrong notify port, D never scheduled the request);
* the dict holds **other keys** but not ours -- a transfer-id keying mismatch,
  where D notified under an id P is not waiting on;
* the key **arrives late** -- genuinely slow, and the timeout is the bug.

This patcher prints enough to tell them apart: the awaited id, the dict size, a
sample of the keys it does hold, and the remote ip/notify port the task targets.

Also note ``VLLM_MORIIO_TRANSFER_TIMEOUT_S`` / ``VLLM_MORIIO_DEFERRED_TIMEOUT_S``
are **not** read from the environment -- vLLM logs both as unknown env vars.
``MoRIIOConfig`` takes them from ``kv_connector_extra_config`` ("transfer_timeout",
"defer_timeout"), so every run so far used the 30s/60s defaults.

Four call sites in ``moriio_engine.py``:

``first-defer``  the write task was queued before D's allocation landed. Normal
                 once per transfer if D is merely a little behind.
``waiting``      still not ready; repeats at most once per 10s per transfer.
``ready``        the key arrived -- ``waited=`` is the true D-side latency, which
                 separates "slow" from "never".
``expired``      ``defer_timeout`` hit, blocks force-freed. This is the hang.

Diagnostic only: no transfer behaviour changes, and the helper swallows every
exception so it cannot be the reason a run fails. Default off::

    DSV4_DEFER_DIAG=0   (default) shipped logging
    DSV4_DEFER_DIAG=1            per-defer diagnostics

Idempotent. Missing anchor is a hard error; missing file is a skip.

Usage: apply_moriio_dsv4_defer_diag_fix.py <vllm_install_dir>
       apply_moriio_dsv4_defer_diag_fix.py --selftest
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py"
MARKER = "DSV4-DEFER-DIAG"

FLAG = '''
# DSV4-DEFER-DIAG: _is_remote_ready(task) is just
#     task.transfer_id in worker.moriio_wrapper.done_remote_allocate_req_dict
# so a deferred write means D never told P its block allocation for that
# transfer id. Print the awaited id next to the keys the dict actually holds:
# empty is a lost notify, other-keys-present is a keying mismatch, and a late
# arrival shows up as a large waited= on the "ready" line.
import os as _dsv4_dd_os
import time as _dsv4_dd_time

DSV4_DEFER_DIAG = _dsv4_dd_os.environ.get("DSV4_DEFER_DIAG", "0") == "1"
_DSV4_DEFER_SEEN: dict = {}


def _dsv4_defer_diag(engine, task, phase, waited=None):
    """WARNING once per (transfer, phase); "waiting" repeats every 10s.

    The write worker re-checks readiness every 10ms, so this MUST dedupe or it
    buries the log it is meant to expose. Never raises: a diagnostic must not
    be able to kill a transfer.
    """
    if not DSV4_DEFER_DIAG:
        return
    try:
        tx = getattr(task, "transfer_id", None)
        if waited is None:
            waited = _dsv4_dd_time.perf_counter() - task.enqueue_time
        bucket = int(waited // 10) if phase == "waiting" else 0
        key = (tx, phase, bucket)
        if key in _DSV4_DEFER_SEEN:
            return
        if len(_DSV4_DEFER_SEEN) > 4096:
            _DSV4_DEFER_SEEN.clear()
        _DSV4_DEFER_SEEN[key] = True
        ready = engine.worker.moriio_wrapper.done_remote_allocate_req_dict
        keys = list(ready.keys())
        from vllm.logger import init_logger as _dsv4_dd_init_logger

        _dsv4_dd_init_logger(__name__).warning(
            "[dsv4-defer] phase=%s tx=%s req=%s waited=%.1fs layer=%s "
            "dst_engine=%s remote=%s:%s local_blocks=%d ready_n=%d "
            "ready_has_tx=%s ready_sample=%s",
            phase,
            tx,
            getattr(task, "request_id", None),
            waited,
            getattr(task, "layer_name", None),
            getattr(task, "dst_engine_id", None),
            getattr(task, "remote_ip", None),
            getattr(task, "remote_notify_port", None),
            len(getattr(task, "local_block_ids", None) or ()),
            len(keys),
            tx in ready,
            keys[:3],
        )
    except Exception:  # noqa: BLE001
        pass
'''

DEFER_OLD = """            # Check if remote blocks are ready
            if not self._is_remote_ready(task):
                # task.retry_count += 1
                self._deferred_tasks.append(task)
"""

DEFER_NEW = """            # Check if remote blocks are ready
            if not self._is_remote_ready(task):
                # task.retry_count += 1
                _dsv4_defer_diag(self, task, "first-defer")
                self._deferred_tasks.append(task)
"""

EXPIRE_OLD = """            if now - task.enqueue_time > defer_timeout:
                logger.error(
                    "Deferred write task for request %s expired after %.1fs "
                    "(remote blocks never arrived), marking done",
"""

EXPIRE_NEW = """            if now - task.enqueue_time > defer_timeout:
                _dsv4_defer_diag(self, task, "expired", now - task.enqueue_time)
                logger.error(
                    "Deferred write task for request %s expired after %.1fs "
                    "(remote blocks never arrived), marking done",
"""

READY_OLD = """            if self._is_remote_ready(task):
                try:
                    self._execute_write_task(task)
"""

READY_NEW = """            if self._is_remote_ready(task):
                _dsv4_defer_diag(self, task, "ready", now - task.enqueue_time)
                try:
                    self._execute_write_task(task)
"""

WAIT_OLD = """            else:
                still_deferred.append(task)
"""

WAIT_NEW = """            else:
                _dsv4_defer_diag(self, task, "waiting", now - task.enqueue_time)
                still_deferred.append(task)
"""

PAIRS = (
    ("_write_worker_loop defer branch", DEFER_OLD, DEFER_NEW),
    ("_process_deferred_tasks expiry", EXPIRE_OLD, EXPIRE_NEW),
    ("_process_deferred_tasks ready branch", READY_OLD, READY_NEW),
    ("_process_deferred_tasks still-deferred branch", WAIT_OLD, WAIT_NEW),
)

# Stand-in carrying all four anchors with the shipped indentation.
STUB = '''"""MoRIIO engine."""
import threading
import time
from queue import Empty, Queue

from vllm.logger import init_logger

logger = init_logger(__name__)


class MoRIIOEngine:
    def _write_worker_loop(self) -> None:
        while True:
            self._process_deferred_tasks()

            try:
                task = self._write_task_q.get(timeout=0.01)
            except Empty:
                continue

            if self._is_transfer_terminal(task.transfer_id):
                continue

            # Check if remote blocks are ready
            if not self._is_remote_ready(task):
                # task.retry_count += 1
                self._deferred_tasks.append(task)
                continue

            self._execute_write_task(task)

    def _process_deferred_tasks(self) -> None:
        if not self._deferred_tasks:
            return

        defer_timeout = self._defer_timeout
        now = time.perf_counter()
        still_deferred = []

        for task in self._deferred_tasks:
            if self._is_transfer_terminal(task.transfer_id):
                continue
            if now - task.enqueue_time > defer_timeout:
                logger.error(
                    "Deferred write task for request %s expired after %.1fs "
                    "(remote blocks never arrived), marking done",
                    task.request_id,
                    now - task.enqueue_time,
                )
                self._mark_request_done(task.transfer_id)
                continue
            if self._is_remote_ready(task):
                try:
                    self._execute_write_task(task)
                except Exception:
                    logger.exception(
                        "Deferred write task failed for request %s, marking done",
                        task.request_id,
                    )
                    self._mark_request_done(task.transfer_id)
            else:
                still_deferred.append(task)

        self._deferred_tasks = still_deferred
'''


def _insert_flag(src: str) -> str:
    """Place FLAG after the import block.

    Prefer the logger anchor (moriio_engine.py has one). Fall back to the AST
    end_lineno so a parenthesised multi-line import is never split -- that
    split is what broke 218197/218198.
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
    out = src
    for label, old, new in PAIRS:
        if old not in out:
            raise ValueError(f"anchor missing: {label}")
        out = out.replace(old, new, 1)
    return _insert_flag(out)


def _selftest() -> int:
    import ast
    import types

    out = _apply(STUB)
    assert MARKER in out
    ast.parse(out)
    for phase in ("first-defer", "waiting", "ready", "expired"):
        assert f'"{phase}"' in out, phase

    # Every anchor must have fired exactly once.
    assert out.count("_dsv4_defer_diag(self, task,") == 4

    # Behaviour: install a fake vllm.logger so the helper's log is observable
    # instead of being swallowed by its own except.
    seen: list = []

    class _Log:
        def warning(self, fmt, *args):
            seen.append(fmt % args)

    vllm_mod = types.ModuleType("vllm")
    logger_mod = types.ModuleType("vllm.logger")
    logger_mod.init_logger = lambda _name: _Log()
    vllm_mod.logger = logger_mod
    saved = {k: sys.modules.get(k) for k in ("vllm", "vllm.logger")}
    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.logger"] = logger_mod

    class _Task:
        def __init__(self, tx):
            self.transfer_id = tx
            self.request_id = "req-" + tx
            self.layer_name = "model.layers.0.attn"
            self.dst_engine_id = "10.0.0.1:8405"
            self.remote_ip = "10.0.0.1"
            self.remote_notify_port = 61005
            self.local_block_ids = [1, 2, 3]
            self.enqueue_time = 0.0

    class _Engine:
        def __init__(self, ready):
            wrapper = types.SimpleNamespace(done_remote_allocate_req_dict=ready)
            self.worker = types.SimpleNamespace(moriio_wrapper=wrapper)

    def _load(flag):
        mod = types.ModuleType("stub")
        env = os.environ.get("DSV4_DEFER_DIAG")
        if flag:
            os.environ["DSV4_DEFER_DIAG"] = "1"
        else:
            os.environ.pop("DSV4_DEFER_DIAG", None)
        try:
            mod.__dict__["init_logger"] = lambda _n: _Log()
            src = out.replace("from vllm.logger import init_logger\n", "", 1)
            exec(compile(src, "stub", "exec"), mod.__dict__)
            return mod
        finally:
            if env is None:
                os.environ.pop("DSV4_DEFER_DIAG", None)
            else:
                os.environ["DSV4_DEFER_DIAG"] = env

    try:
        # Flag off: silent.
        off = _load(False)
        off._dsv4_defer_diag(_Engine({}), _Task("tx-a"), "first-defer", 0.0)
        assert not seen, seen

        on = _load(True)

        # Lost notify: dict empty.
        on._dsv4_defer_diag(_Engine({}), _Task("tx-a"), "first-defer", 0.0)
        assert len(seen) == 1, seen
        assert "ready_n=0" in seen[0] and "ready_has_tx=False" in seen[0]
        assert "tx=tx-a" in seen[0] and "remote=10.0.0.1:61005" in seen[0]

        # Deduped: the write loop calls this every 10ms.
        on._dsv4_defer_diag(_Engine({}), _Task("tx-a"), "first-defer", 0.0)
        assert len(seen) == 1, seen

        # Keying mismatch: other ids present, ours absent.
        seen.clear()
        on._dsv4_defer_diag(
            _Engine({"tx-other": 1, "tx-more": 2}), _Task("tx-b"), "expired", 60.0
        )
        assert len(seen) == 1 and "ready_n=2" in seen[0], seen
        assert "ready_has_tx=False" in seen[0] and "tx-other" in seen[0]

        # "waiting" is bucketed per 10s, so a long stall reports progress
        # without flooding.
        seen.clear()
        eng, task = _Engine({}), _Task("tx-c")
        for waited in (0.0, 3.0, 9.9, 10.1, 19.0, 20.5):
            on._dsv4_defer_diag(eng, task, "waiting", waited)
        assert len(seen) == 3, seen

        # A late arrival is visible as a large waited= on "ready".
        seen.clear()
        on._dsv4_defer_diag(_Engine({"tx-d": 1}), _Task("tx-d"), "ready", 41.5)
        assert "phase=ready" in seen[0] and "waited=41.5s" in seen[0], seen
        assert "ready_has_tx=True" in seen[0]

        # A broken engine must not raise into the write worker.
        seen.clear()
        on._dsv4_defer_diag(object(), _Task("tx-e"), "first-defer", 0.0)
        assert not seen
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    # Idempotent: refuse a second apply rather than inserting FLAG twice.
    try:
        _apply(out)
    except ValueError as e:
        assert "already patched" in str(e)
    else:
        raise AssertionError("second apply should have refused")

    print("[dsv4-defer] selftest OK")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir> | --selftest", file=sys.stderr)
        return 2
    if sys.argv[1] == "--selftest":
        return _selftest()

    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[dsv4-defer] {REL} not found under {sys.argv[1]} -- skipping.")
        return 0

    src = open(path).read()
    if MARKER in src:
        print(f"[dsv4-defer] already patched in {path} -- no-op.")
        return 0

    try:
        out = _apply(src)
    except ValueError as e:
        print(f"[dsv4-defer] ERROR: {e}", file=sys.stderr)
        for i, line in enumerate(src.splitlines(), 1):
            if "_deferred_tasks" in line or "_is_remote_ready" in line:
                print(f"[dsv4-defer]   line {i}: {line.rstrip()}", file=sys.stderr)
        return 1

    import ast

    try:
        ast.parse(out)
    except SyntaxError as e:
        print(f"[dsv4-defer] ERROR: compile failed: {e}", file=sys.stderr)
        return 1

    tmp = path + ".dsv4dd"
    with open(tmp, "w") as f:
        f.write(out)
    os.replace(tmp, path)
    print(
        f"[dsv4-defer] patched {path}: defer/ready/expiry diagnostics, "
        "gated on DSV4_DEFER_DIAG (default 0)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
