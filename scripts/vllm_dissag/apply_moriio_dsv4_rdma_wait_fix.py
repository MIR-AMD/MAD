#!/usr/bin/env python3
"""Flash: log CQE wait after WRITE seal, and time Succeeded() (217557).

217557: writes_done=84 (posts, not CQEs). Smoke rank 0 decode 38s Lisa Su.
Every later decode POST ~391s on ranks 0-7. No TransferError.

5a4c8d99 already snapshots transfer_statuses under _write_state_lock then
polls waiting_for_transfer_complete *outside* that lock. Un-nesting
_mark_write_done is also already upstream. A lock-release patcher would
miss its anchors on this image.

The 30s extra_config transfer_timeout cannot explain 391s unless
TransferStatus.Succeeded() blocks (MoRI Wait-like) so the Python deadline
never fires. This patcher:

  1. Wraps the existing outside-lock wait with elapsed / n_status logs.
  2. Times each Succeeded() inside waiting_for_transfer_complete.
  3. If an older image still waits *inside* the lock, un-nest that too.

Gated in connectors/moriio.sh to DeepSeek-V4-Flash-FP8. Idempotent.

Usage: apply_moriio_dsv4_rdma_wait_fix.py <vllm_install_dir>
"""
import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py"
MARKER = "DSV4-RDMA-WAIT"


def _replace_one(src: str, old: str, new: str, label: str, applied: list[str]) -> str:
    if old not in src:
        return src
    src = src.replace(old, new, 1)
    applied.append(label)
    return src


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[dsv4-rdma-wait] {REL} not found under {sys.argv[1]} -- skipping.")
        return 0

    src = open(path).read()
    if MARKER in src:
        print(f"[dsv4-rdma-wait] already patched in {path} -- no-op.")
        return 0

    orig = src
    applied: list[str] = []

    # 5a4c already finalizes after releasing the increment lock.
    h1_nested = """        with self._write_state_lock:
            request_info.writes_done += 1
            self._finalize_if_complete(transfer_id, request_info)
"""
    h1_new = """        with self._write_state_lock:
            request_info.writes_done += 1
        # DSV4-RDMA-WAIT: do not nest Lock acquire into _finalize.
        self._finalize_if_complete(transfer_id, request_info)
"""
    if h1_nested in src:
        src = _replace_one(src, h1_nested, h1_new, "mark_write_done_unnest", applied)
    else:
        print(
            "[dsv4-rdma-wait] _mark_write_done already finalizes outside lock -- ok."
        )

    # 5a4c: wait already at method indent (outside _write_state_lock).
    h2_outside_old = """        # Wait for this request's transfers to complete.
        self.worker.moriio_wrapper.waiting_for_transfer_complete(transfer_statuses)
"""
    h2_outside_new = """        # DSV4-RDMA-WAIT: 217557. CQE wait is already outside the write lock
        # on 5a4c. Log elapsed so WRITE 1 vs WRITE 2+ is visible.
        _t0 = time.monotonic()
        _nstat = len(transfer_statuses)
        try:
            self.worker.moriio_wrapper.waiting_for_transfer_complete(
                transfer_statuses
            )
        except Exception:
            logger.exception(
                "[dsv4-gate] rdma_wait failed transfer=%s n_status=%d "
                "elapsed=%.3fs decode_dp=%s writes_done=%s expected=%s",
                transfer_id,
                _nstat,
                time.monotonic() - _t0,
                int(request_info.decode_dp_rank),
                request_info.writes_done,
                expected,
            )
            raise
        logger.info(
            "[dsv4-gate] rdma_wait transfer=%s n_status=%d elapsed=%.3fs "
            "err=None decode_dp=%s writes_done=%s expected=%s",
            transfer_id,
            _nstat,
            time.monotonic() - _t0,
            int(request_info.decode_dp_rank),
            request_info.writes_done,
            expected,
        )
"""
    # Older overlay: wait still indented inside the lock (12 spaces).
    h2_inside_old = """            # Wait for this request's transfers to complete.
            self.worker.moriio_wrapper.waiting_for_transfer_complete(transfer_statuses)
"""
    h2_inside_new = """            _decode_dp_rank = int(request_info.decode_dp_rank)
            _writes_done = request_info.writes_done
            _expected = expected

        # DSV4-RDMA-WAIT: poll CQEs outside _write_state_lock.
        _t0 = time.monotonic()
        _nstat = len(transfer_statuses)
        try:
            self.worker.moriio_wrapper.waiting_for_transfer_complete(
                transfer_statuses
            )
        except Exception:
            logger.exception(
                "[dsv4-gate] rdma_wait failed transfer=%s n_status=%d "
                "elapsed=%.3fs decode_dp=%s writes_done=%s expected=%s",
                transfer_id,
                _nstat,
                time.monotonic() - _t0,
                _decode_dp_rank,
                _writes_done,
                _expected,
            )
            raise
        logger.info(
            "[dsv4-gate] rdma_wait transfer=%s n_status=%d elapsed=%.3fs "
            "err=None decode_dp=%s writes_done=%s expected=%s",
            transfer_id,
            _nstat,
            time.monotonic() - _t0,
            _decode_dp_rank,
            _writes_done,
            _expected,
        )
"""
    if h2_outside_old in src:
        src = _replace_one(
            src, h2_outside_old, h2_outside_new, "finalize_wait_log", applied
        )
    elif h2_inside_old in src:
        src = _replace_one(
            src, h2_inside_old, h2_inside_new, "finalize_wait_outside_lock", applied
        )
    else:
        print(
            "[dsv4-rdma-wait] ERROR: waiting_for_transfer_complete call site missing.",
            file=sys.stderr,
        )
        for i, line in enumerate(src.splitlines(), 1):
            if "waiting_for_transfer_complete" in line:
                print(f"[dsv4-rdma-wait]   line {i}: {line.rstrip()}", file=sys.stderr)
        return 1

    # Time Succeeded(): if this is ~391s, MoRI is blocking inside the probe
    # and extra_config transfer_timeout=30 cannot fire.
    h3_old = """        while remaining:
            timed_out = time.monotonic() > deadline
            still_waiting = []
            for status in remaining:
                if status.Succeeded():
                    continue
                if status.Failed():
"""
    h3_new = """        _max_succ = 0.0
        _n_succ_block = 0
        while remaining:
            timed_out = time.monotonic() > deadline
            still_waiting = []
            for status in remaining:
                # DSV4-RDMA-WAIT: Succeeded() must be a non-blocking probe.
                _s0 = time.monotonic()
                _ok = status.Succeeded()
                _dt = time.monotonic() - _s0
                if _dt > _max_succ:
                    _max_succ = _dt
                if _dt > 0.05:
                    _n_succ_block += 1
                if _ok:
                    continue
                if status.Failed():
"""
    h3_tail_old = """        if errors:
            raise TransferError(
                f"{len(errors)}/{len(transfers_to_wait)} transfers failed:\\n"
                + "\\n".join(errors)
            )
"""
    h3_tail_new = """        logger.info(
            "[dsv4-gate] cqe_poll n=%d elapsed=%.3fs remaining=%d errors=%d "
            "max_succeeded_s=%.3fs n_succeeded_gt_50ms=%d timeout=%.0fs",
            len(transfers_to_wait),
            time.monotonic() - (deadline - timeout),
            len(remaining),
            len(errors),
            _max_succ,
            _n_succ_block,
            timeout,
        )
        if errors:
            raise TransferError(
                f"{len(errors)}/{len(transfers_to_wait)} transfers failed:\\n"
                + "\\n".join(errors)
            )
"""
    if h3_old in src:
        src = _replace_one(src, h3_old, h3_new, "succeeded_probe_timer", applied)
        if h3_tail_old in src:
            src = _replace_one(src, h3_tail_old, h3_tail_new, "cqe_poll_log", applied)
        else:
            print(
                "[dsv4-rdma-wait] WARN: TransferError tail anchor missing -- "
                "Succeeded() timer applied without cqe_poll log."
            )
    else:
        print(
            "[dsv4-rdma-wait] WARN: waiting_for_transfer_complete loop anchor "
            "missing -- wait-site log only."
        )

    if MARKER not in src:
        print("[dsv4-rdma-wait] ERROR: post-write missing marker.", file=sys.stderr)
        return 1
    if src == orig:
        print(f"[dsv4-rdma-wait] no changes ({', '.join(applied)}) for {path}")
        return 0
    tmp = path + ".dsv4rdma"
    with open(tmp, "w") as f:
        f.write(src)
    os.replace(tmp, path)
    try:
        import py_compile

        py_compile.compile(path, doraise=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dsv4-rdma-wait] ERROR: compile failed: {e}", file=sys.stderr)
        return 1
    print(f"[dsv4-rdma-wait] patched hunks: {', '.join(applied)} in {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
