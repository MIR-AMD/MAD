#!/usr/bin/env python3
"""Diagnose the empty producer rid<->transfer_id table under HMA=1 (218318).

OBSERVATION (218318 Pro HMA=1 vs 218323 Pro HMA=0):
  218318 logged 10 "MoRI-IO unmap MISS: rid=... transfer_id=tx-... table_size=0"
  warnings across 10 distinct transfers -- exactly the 10 "Reaped 1 deferred
  sends with no finished_sending" reaps, one per hung request. 218323 (HMA=0,
  same image/model/nodes/proxy, 1h, sweep + 3/3 curl) logged ZERO. Since
  unmap_request_id also runs for successfully ACKed sends, an empty table at
  HMA=0 would have warned there too -- so the table is populated with HMA off
  and empty with HMA on.

  The writes and the notify are NOT at fault: cqe_poll reported n=243
  elapsed=0.000s remaining=0 errors=0 on every finalized transfer, across
  decode_dp 0-7, and send_notify is called right after. What never comes back
  is the scheduler-side finished_sending ACK, because with an empty table no
  ACK can be attributed to a request.

REFUTED (do not retry): SupportsHMA declaring an alloc-side "*_all_groups"
  hook we failed to override. Upstream base.py declares exactly ONE abstract
  method, request_finished_all_groups; update_state_after_alloc is the single
  alloc hook for HMA and non-HMA alike, so the patcher is not missing an
  override.

TWO SURVIVING EXPLANATIONS, which need different fixes:
  (a) params guard. update_state_after_alloc does
          params = request.kv_transfer_params
          if not params:
              return
      and map_request_id sits AFTER that guard. If kv_transfer_params is empty
      at alloc time on the producer under HMA, nothing is ever mapped. Our own
      fork already added the _req_kv_params snapshot cache because those params
      "can be mutated/cleared between" scheduler steps on chunked prefill, so
      this is not hypothetical.
  (b) instance routing. map and unmap execute on DIFFERENT connector-scheduler
      instances (DP rank / EngineCore), so the reaping rank legitimately holds
      an empty table while another rank holds the mapping.

This patcher distinguishes them. It logs:
  * every map_request_id: rid, transfer_id, resulting table sizes, dp_rank,
    is_producer, id(self)
  * every unmap_request_id ENTRY, before the lookup: rid, table sizes,
    dp_rank, is_producer, id(self)
  * every update_state_after_alloc that returns early on the params guard:
    rid, is_producer, dp_rank

Read the result as:
  no [dsv4-ridmap] map= lines at all      -> (a), the params guard
  map= lines whose self=/dp= differ from
    the unmap= line for the same rid      -> (b), instance routing
  map= then unmap= same self= and the
    rid still MISSes                      -> neither; the rid mutated between
                                             them (suffix), so re-read the
                                             suffix-strip fallback

Diagnostic only: pure logging, no behavior change. Runtime flag
DSV4_RID_MAP_DIAG (default 0 = shipped logging). Idempotent, anchor-based.
A missing anchor warns and skips -- this must never break a boot.

Usage: apply_moriio_dsv4_rid_map_diag_fix.py <vllm_install_dir>
       apply_moriio_dsv4_rid_map_diag_fix.py --selftest
"""

import os
import re
import sys
import tempfile

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
MARKER = "[dsv4-ridmap]"

# Emitted once per hunk; keep the helper self-contained so a missing import in
# the target module cannot break a boot.
_HELPER = '''
def _dsv4_ridmap_on() -> bool:
    """DSV4_RID_MAP_DIAG=1 enables the rid<->transfer_id map/unmap trace."""
    try:
        import os as _o

        return _o.environ.get("DSV4_RID_MAP_DIAG", "0") == "1"
    except Exception:
        return False


def _dsv4_ridmap_log(self, what: str, request_id, transfer_id=None) -> None:
    """Log a map/unmap event with the identity of the scheduler that saw it.

    dp/self identify the connector-scheduler instance: a map and an unmap that
    disagree on either one mean the ACK is being reaped by a rank that never
    allocated the request.
    """
    if not _dsv4_ridmap_on():
        return
    try:
        _fwd = getattr(self, "request_id_to_transfer_id", None)
        _rev = getattr(self, "transfer_id_to_request_id", None)
        logger.info(
            "[dsv4-ridmap] %s rid=%r tx=%r fwd=%d rev=%d dp=%s gdp=%s "
            "producer=%s self=%s",
            what,
            request_id,
            transfer_id,
            len(_fwd) if _fwd is not None else -1,
            len(_rev) if _rev is not None else -1,
            getattr(self, "dp_rank", None),
            getattr(self, "_global_dp_rank", None),
            getattr(self, "is_producer", None),
            hex(id(self)),
        )
    except Exception:
        pass
'''


def _insert_helper(src: str, applied: list) -> str:
    """Put the helper just above the class that owns the two dicts."""
    if "_dsv4_ridmap_log" in src:
        applied.append("helper (already)")
        return src
    anchor = "class MoRIIOConnectorScheduler"
    idx = src.find(anchor)
    if idx < 0:
        raise ValueError("MoRIIOConnectorScheduler class not found")
    src = src[:idx] + _HELPER.lstrip("\n") + "\n\n" + src[idx:]
    applied.append("helper")
    return src


def _patch_map(src: str, applied: list) -> str:
    old = """        self.transfer_id_to_request_id[transfer_id] = request_id
        self.request_id_to_transfer_id[request_id] = transfer_id
"""
    new = """        self.transfer_id_to_request_id[transfer_id] = request_id
        self.request_id_to_transfer_id[request_id] = transfer_id
        # DSV4-RIDMAP: 218318 producer table was empty (table_size=0) on every
        # unmap. Record who mapped, so an unmap on another instance is visible.
        _dsv4_ridmap_log(self, "map", request_id, transfer_id)
"""
    if '_dsv4_ridmap_log(self, "map"' in src:
        applied.append("map (already)")
        return src
    if old not in src:
        raise ValueError("map_request_id dict-assignment anchor missing")
    applied.append("map")
    return src.replace(old, new, 1)


def _patch_unmap(src: str, applied: list) -> str:
    """Log on entry to unmap_request_id, before the table lookup.

    218327 taught us the signature is wrapped across lines on this image and
    carries an optional transfer_id::

        def unmap_request_id(
            self, request_id: ReqId, transfer_id: TransferId | None = None
        ):
            \"\"\"docstring\"\"\"
            if request_id in self.request_id_to_transfer_id:

    so anchor on the first statement rather than the ``def`` line, and match
    both the wrapped and the single-line forms.
    """
    if '_dsv4_ridmap_log(self, "unmap-entry"' in src:
        applied.append("unmap (already)")
        return src

    m = re.search(
        r"    def unmap_request_id\((?:[^\n]*\n)*?"
        r"        if request_id in self\.request_id_to_transfer_id:\n",
        src,
    )
    if not m:
        raise ValueError("unmap_request_id first-statement anchor missing")

    # Pass the transfer_id through when this revision accepts one: the reap
    # path calls unmap_request_id(req_id, transfer_id=transfer_id), so it
    # names the transfer even when the rid lookup is about to miss.
    head = src[m.start() : m.end()]
    arg = ", transfer_id" if "transfer_id" in head.split("):")[0] else ""
    stmt = "        if request_id in self.request_id_to_transfer_id:\n"
    log = (
        "        # DSV4-RIDMAP: log BEFORE the lookup so a miss is attributable.\n"
        f'        _dsv4_ridmap_log(self, "unmap-entry", request_id{arg})\n'
    )
    applied.append("unmap")
    return src[: m.end() - len(stmt)] + log + stmt + src[m.end() :]


def _patch_params_guard(src: str, applied: list) -> str:
    """Log the early return that skips map_request_id entirely."""
    old = """        params = request.kv_transfer_params
        if not params:
            return
"""
    new = """        params = request.kv_transfer_params
        if not params:
            # DSV4-RIDMAP: map_request_id sits after this guard, so an empty
            # kv_transfer_params here means nothing is ever mapped for this
            # request and its ACK can never be attributed.
            _dsv4_ridmap_log(self, "alloc-no-params", request.request_id)
            return
"""
    if '_dsv4_ridmap_log(self, "alloc-no-params"' in src:
        applied.append("params-guard (already)")
        return src
    if old not in src:
        applied.append("params-guard (anchor missing, skipped)")
        return src
    applied.append("params-guard")
    return src.replace(old, new, 1)


def patch(path: str) -> int:
    src = open(path).read()
    orig = src
    applied: list = []

    # Each hunk stands alone. 218327 lost the whole trace because one missing
    # anchor (the wrapped unmap signature) aborted the map hunk too, so a hunk
    # that cannot apply must never discard the ones that can.
    try:
        src = _insert_helper(src, applied)
    except ValueError as e:
        print(f"[dsv4-ridmap] WARN: {e} -- no hunks possible.", file=sys.stderr)
        return 0
    for hunk in (_patch_map, _patch_unmap, _patch_params_guard):
        try:
            src = hunk(src, applied)
        except ValueError as e:
            # Never fatal: this is a diagnostic. A partial trace still answers
            # part of the question, and a boot without it is still valid.
            print(f"[dsv4-ridmap] WARN: {e} -- skipping that hunk.", file=sys.stderr)

    if src == orig:
        print(f"[dsv4-ridmap] no changes ({', '.join(applied)}) in {path}")
        return 0

    tmp = path + ".dsv4ridmap"
    with open(tmp, "w") as f:
        f.write(src)
    os.replace(tmp, path)
    try:
        import py_compile

        py_compile.compile(path, doraise=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dsv4-ridmap] ERROR: compile failed: {e}", file=sys.stderr)
        return 1
    print(f"[dsv4-ridmap] hunks: {', '.join(applied)} in {path}")
    return 0


# Stand-in carrying the anchors as they appear in the 218327 container:
# map_request_id single-line, unmap_request_id WRAPPED with an optional
# transfer_id and a docstring before the first statement, and the scheduler
# update_state_after_alloc params guard.
_FAKE = '''
import logging

logger = logging.getLogger(__name__)

ReqId = str
TransferId = str


class MoRIIOConnectorScheduler:
    def __init__(self):
        self.transfer_id_to_request_id = {}
        self.request_id_to_transfer_id = {}
        self.dp_rank = 0
        self._global_dp_rank = 0
        self.is_producer = True

    def map_request_id(self, request_id: ReqId, transfer_id: TransferId):
        self.transfer_id_to_request_id[transfer_id] = request_id
        self.request_id_to_transfer_id[request_id] = transfer_id

    def unmap_request_id(
        self, request_id: ReqId, transfer_id: TransferId | None = None
    ):
        """Unmap request_id/transfer_id. Uses transfer_id for lookup if
        exact request_id match fails (handles input_processor mutation)."""
        if request_id in self.request_id_to_transfer_id:
            tid = self.request_id_to_transfer_id[request_id]
            del self.request_id_to_transfer_id[request_id]
            self.transfer_id_to_request_id.pop(tid, None)
            return
        logger.warning(
            "MoRI-IO unmap MISS: rid=%r transfer_id=%r table_size=%d",
            request_id,
            transfer_id,
            len(self.request_id_to_transfer_id),
        )

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        params = request.kv_transfer_params
        if not params:
            return
        transfer_id = params.get("transfer_id")
        request_id = request.request_id
        self.map_request_id(request_id, transfer_id)
'''


def selftest() -> int:
    """Apply to a stand-in, then exercise both flag states."""
    d = tempfile.mkdtemp(prefix="dsv4ridmap-")
    mod = os.path.join(d, "moriio_connector.py")
    open(mod, "w").write(_FAKE)

    if patch(mod):
        print("[selftest] FAIL: patch returned non-zero")
        return 1
    out = open(mod).read()
    for needle in (
        'ridmap] %s rid=',
        '_dsv4_ridmap_log(self, "map"',
        '_dsv4_ridmap_log(self, "unmap-entry"',
        '_dsv4_ridmap_log(self, "alloc-no-params"',
    ):
        if needle not in out:
            print(f"[selftest] FAIL: missing {needle!r}")
            return 1

    if patch(mod):
        print("[selftest] FAIL: second apply returned non-zero")
        return 1
    if out != open(mod).read():
        print("[selftest] FAIL: not idempotent")
        return 1

    sys.path.insert(0, d)
    import logging

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    import moriio_connector as m  # noqa: PLC0415

    class _Req:
        request_id = "rid-1"
        kv_transfer_params = None

    for flag in ("0", "1"):
        os.environ["DSV4_RID_MAP_DIAG"] = flag
        s = m.MoRIIOConnectorScheduler()
        print(f"[selftest] --- DSV4_RID_MAP_DIAG={flag}")
        s.map_request_id("rid-1", "tx-aaa")
        s.unmap_request_id("rid-1")
        s.unmap_request_id("rid-missing")
        s.update_state_after_alloc(_Req(), None, 0)

    print("[selftest] PASS (expect trace lines only under =1)")
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return selftest()
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>|--selftest", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[dsv4-ridmap] {REL} not found -- skipping.")
        return 0
    return patch(path)


if __name__ == "__main__":
    sys.exit(main())
