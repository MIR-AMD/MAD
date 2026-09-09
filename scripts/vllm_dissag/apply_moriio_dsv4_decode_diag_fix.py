#!/usr/bin/env python3
"""Decode-side diagnostics for 220985 empty NIAH after a sealed WRITE.

220985 HMA=1 sealed 243/243 and decode FINISHED_LENGTH_CAPPED at 63 tokens
(~98 s, same as HMA=0 218778 2k RETRIEVAL) but NIAH scored chars=0 / FILLER.
pd-debug's decode update_state_after_alloc comment-anchor is gone on
d626108b, so dests at generate time were never logged. This patcher fills
that hole without changing WRITE or sampling:

1. Connector ``update_state_after_alloc`` (consumer only). After
   ``map_request_id``, log per-group block-id lens and a head/tail of each
   group. A second map for the same rid (WAITING_FOR_REMOTE re-admit) logs
   ``n=2 same_ids=False`` if dests changed after the producer already wrote.
2. ``v1/engine/detokenizer.py`` ``BaseIncrementalDetokenizer.update`` /
   ``get_next_output_text``. Log sampled ids plus tokenizer.decode with
   skip_special_tokens False and True *before* the skip-aware DecodeStream
   runs. ``sample-sum`` on finished:

     detok_off nonempty, detok_on empty  -> 63 specials, skip ate them
     both empty                          -> no printable tokens (empty SSE)
     detok_on FILLER                     -> skip is not the 2k chars=0

Diagnostic only. Runtime flag DSV4_DECODE_DIAG (default 0 = quiet; wrapper
sets 1 on DSV4). Idempotent. Missing connector map-anchor is a hard error;
missing detokenizer hunks warn and skip.

Usage: apply_moriio_dsv4_decode_diag_fix.py <vllm_install_dir>
       apply_moriio_dsv4_decode_diag_fix.py --selftest
"""
from __future__ import annotations

import os
import sys
import tempfile

CONN_REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
DETOK_REL = "v1/engine/detokenizer.py"
MARKER = "[dsv4-decode]"

CONN_HELPER = r'''
def _dsv4_decode_diag_on() -> bool:
    """DSV4_DECODE_DIAG=1 logs decode alloc dests and sampled detok."""
    try:
        import os as _o

        return _o.environ.get("DSV4_DECODE_DIAG", "0") == "1"
    except Exception:
        return False


def _dsv4_decode_ids_preview(ids, n=4):
    ids = list(ids)
    if len(ids) <= n * 2:
        return ids
    return ids[:n] + ["..."] + ids[-n:]


def _dsv4_decode_groups(blocks):
    """Return (lenses, groups) from HMA nested or flat get_block_ids()."""
    try:
        raw = blocks.get_block_ids() if blocks is not None else None
    except Exception:
        return None, None
    if not raw:
        return [], []
    if isinstance(raw[0], (list, tuple)):
        groups = [list(g) for g in raw]
    else:
        groups = [list(raw)]
    return [len(g) for g in groups], groups


def _dsv4_decode_alloc_log(self, request, blocks, num_external_tokens) -> None:
    """Consumer-only. n=2 same_ids=False is dest-replace after WAITING_FOR_REMOTE."""
    if not _dsv4_decode_diag_on():
        return
    try:
        if getattr(self, "is_producer", False):
            return
        params = getattr(request, "kv_transfer_params", None) or {}
        rid = getattr(request, "request_id", None)
        lenses, groups = _dsv4_decode_groups(blocks)
        snap = [list(g) for g in (groups or [])]
        store = getattr(self, "_dsv4_decode_alloc", None)
        if store is None:
            self._dsv4_decode_alloc = store = {}
        if len(store) > 256:
            store.clear()
        prev = store.get(rid)
        n = 1 if prev is None else int(prev[0]) + 1
        same = None if prev is None else prev[1] == snap
        store[rid] = (n, snap)
        heads = [_dsv4_decode_ids_preview(g) for g in snap]
        logger.info(
            "[dsv4-decode] alloc n=%d rid=%s tx=%s num_ext=%s "
            "do_remote_prefill=%s n_groups=%s lens=%s same_ids=%s "
            "groups=%s dp=%s gdp=%s self=%s",
            n,
            rid,
            params.get("transfer_id"),
            num_external_tokens,
            params.get("do_remote_prefill"),
            len(snap),
            lenses,
            same,
            heads,
            getattr(self, "dp_rank", None),
            getattr(self, "_global_dp_rank", None),
            hex(id(self)),
        )
    except Exception:
        pass
'''

DETOK_HELPER = r'''
def _dsv4_decode_diag_on() -> bool:
    """DSV4_DECODE_DIAG=1 logs sampled ids / detok before skip_special_tokens."""
    try:
        import os as _o

        return _o.environ.get("DSV4_DECODE_DIAG", "0") == "1"
    except Exception:
        return False


def _dsv4_decode_ids_preview(ids, n=8):
    ids = list(ids)
    if len(ids) <= n * 2:
        return ids
    return ids[:n] + ["..."] + ids[-n:]


def _dsv4_decode_tok_decode(tok, ids, skip):
    if tok is None or not ids:
        return None
    decode = getattr(tok, "decode", None)
    if decode is None:
        return None
    ids = list(ids)
    try:
        return decode(ids, skip_special_tokens=skip)
    except TypeError:
        try:
            return decode(ids)
        except Exception:
            return None
    except Exception:
        return None


def _dsv4_decode_clip(text, limit=96):
    if text is None:
        return None
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _dsv4_decode_sample_log(detok, new_token_ids, *, kind="tok") -> None:
    if not _dsv4_decode_diag_on():
        return
    try:
        if kind == "tok":
            n = int(getattr(detok, "_dsv4_sample_logs", 0) or 0)
            if n >= 24:
                return
            detok._dsv4_sample_logs = n + 1
        rid = getattr(detok, "request_id", None) or hex(id(detok))
        tok = getattr(detok, "tokenizer", None)
        skip = getattr(detok, "skip_special_tokens", None)
        off = _dsv4_decode_tok_decode(tok, new_token_ids, False)
        on = _dsv4_decode_tok_decode(tok, new_token_ids, True)
        logger.info(
            "[dsv4-decode] sample-%s rid=%s n_new=%d ids=%s skip_special=%s "
            "detok_off=%r detok_on=%r out_len=%s",
            kind,
            rid,
            len(new_token_ids) if new_token_ids is not None else 0,
            _dsv4_decode_ids_preview(new_token_ids or []),
            skip,
            _dsv4_decode_clip(off),
            _dsv4_decode_clip(on),
            len(getattr(detok, "output_text", "") or ""),
        )
    except Exception:
        pass


def _dsv4_decode_sample_sum(detok) -> None:
    if getattr(detok, "_dsv4_sample_sum_done", False):
        return
    detok._dsv4_sample_sum_done = True
    ids = getattr(detok, "output_token_ids", None)
    if ids is None:
        ids = getattr(detok, "token_ids", None) or []
    _dsv4_decode_sample_log(detok, list(ids), kind="sum")
'''


def _insert_before(src: str, anchor: str, blob: str) -> str:
    idx = src.find(anchor)
    if idx < 0:
        raise ValueError(f"anchor not found: {anchor!r}")
    return src[:idx] + blob.lstrip("\n") + "\n\n" + src[idx:]


_ALLOC_CALL = (
    "        _dsv4_decode_alloc_log(self, request, blocks, num_external_tokens)\n"
)


def _patch_connector(src: str, applied: list) -> str:
    if "def _dsv4_decode_alloc_log" not in src:
        src = _insert_before(src, "class MoRIIOConnectorScheduler", CONN_HELPER)
        applied.append("conn-helper")
    else:
        applied.append("conn-helper (already)")
    old = "        self.map_request_id(request_id, transfer_id)\n"
    new = (
        "        self.map_request_id(request_id, transfer_id)\n"
        "        # DSV4-DECODE-DIAG: 220985 consumer mapped twice after\n"
        "        # WAITING_FOR_REMOTE; log per-group dests on each map.\n"
        + _ALLOC_CALL
    )
    if _ALLOC_CALL in src:
        applied.append("conn-alloc (already)")
        return src
    n = src.count(old)
    if n != 1:
        raise ValueError(
            f"map_request_id assignment count={n} (want 1 unique alloc site)"
        )
    applied.append("conn-alloc")
    return src.replace(old, new, 1)


def _patch_detokenizer(src: str, applied: list) -> str:
    if "_dsv4_decode_sample_sum(" in src and "def _dsv4_decode_sample_sum" in src:
        applied.append("detok (already)")
        return src
    if "def _dsv4_decode_sample_sum" not in src:
        src = _insert_before(src, "class IncrementalDetokenizer", DETOK_HELPER)
        applied.append("detok-helper")

    tok_old = (
        "        if not new_token_ids:\n"
        "            # Skip detokenization if no new token ids.\n"
        "            return None\n"
    )
    tok_new = (
        "        if not new_token_ids:\n"
        "            # Skip detokenization if no new token ids.\n"
        "            return None\n"
        "        # DSV4-DECODE-DIAG: 220985 2k LENGTH_CAPPED chars=0. Log ids\n"
        "        # and decode(skip=False/True) before the skip-aware stream.\n"
        "        _dsv4_decode_sample_log(self, new_token_ids)\n"
    )
    if "_dsv4_decode_sample_log(self, new_token_ids)" in src:
        applied.append("detok-update (already)")
    elif tok_old not in src:
        applied.append("detok-update (anchor missing, skipped)")
    else:
        src = src.replace(tok_old, tok_new, 1)
        applied.append("detok-update")

    sum_old = (
        "        # We return the full output text if the sequence is finished.\n"
        "        buffer_length = 0 if finished else self.stop_buffer_length\n"
    )
    sum_new = (
        "        # We return the full output text if the sequence is finished.\n"
        "        if finished:\n"
        "            _dsv4_decode_sample_sum(self)\n"
        "        buffer_length = 0 if finished else self.stop_buffer_length\n"
    )
    if "_dsv4_decode_sample_sum(self)" in src:
        applied.append("detok-sum (already)")
    elif sum_old not in src:
        applied.append("detok-sum (anchor missing, skipped)")
    else:
        src = src.replace(sum_old, sum_new, 1)
        applied.append("detok-sum")
    return src


def _write(path: str, src: str, suffix: str) -> None:
    tmp = path + suffix
    with open(tmp, "w") as f:
        f.write(src)
    os.replace(tmp, path)


def _compile(path: str) -> None:
    import py_compile

    py_compile.compile(path, doraise=True)


def patch(vllm_dir: str) -> int:
    applied: list = []
    conn = os.path.join(vllm_dir, CONN_REL)
    detok = os.path.join(vllm_dir, DETOK_REL)

    if not os.path.isfile(conn):
        print(f"[dsv4-decode] {CONN_REL} not found -- skipping.")
        return 0

    src = open(conn).read()
    orig = src
    try:
        src = _patch_connector(src, applied)
    except ValueError as e:
        print(f"[dsv4-decode] ERROR: connector: {e}", file=sys.stderr)
        return 1
    if src != orig:
        _write(conn, src, ".dsv4ddiag")
        try:
            _compile(conn)
        except Exception as e:  # noqa: BLE001
            print(f"[dsv4-decode] ERROR: connector compile: {e}", file=sys.stderr)
            return 1

    if os.path.isfile(detok):
        dsrc = open(detok).read()
        dorig = dsrc
        try:
            dsrc = _patch_detokenizer(dsrc, applied)
        except ValueError as e:
            print(f"[dsv4-decode] WARN: detokenizer: {e} -- skipping.", file=sys.stderr)
            applied.append("detok (failed)")
        else:
            if dsrc != dorig:
                _write(detok, dsrc, ".dsv4ddiag")
                try:
                    _compile(detok)
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[dsv4-decode] ERROR: detokenizer compile: {e}",
                        file=sys.stderr,
                    )
                    return 1
    else:
        applied.append("detok (missing)")

    print(f"[dsv4-decode] hunks: {', '.join(applied)}")
    return 0


CONN_FAKE = '''
import logging

logger = logging.getLogger(__name__)


class MoRIIOConnectorScheduler:
    def __init__(self):
        self.transfer_id_to_request_id = {}
        self.request_id_to_transfer_id = {}
        self.dp_rank = 0
        self._global_dp_rank = 0
        self.is_producer = False

    def map_request_id(self, request_id, transfer_id):
        self.transfer_id_to_request_id[transfer_id] = request_id
        self.request_id_to_transfer_id[request_id] = transfer_id

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        params = request.kv_transfer_params
        if not params:
            return
        transfer_id = params.get("transfer_id")
        request_id = request.request_id
        self.map_request_id(request_id, transfer_id)
        if params.get("do_remote_decode"):
            return
'''

DETOK_FAKE = '''
import logging

logger = logging.getLogger(__name__)


class IncrementalDetokenizer:
    def __init__(self):
        self.token_ids = []


class BaseIncrementalDetokenizer(IncrementalDetokenizer):
    def __init__(self):
        super().__init__()
        self.output_text = ""
        self.stop_buffer_length = 0
        self._last_output_text_offset = 0
        self.include_stop_str_in_output = False
        self.min_tokens = 0
        self.stop = []

    def update(self, new_token_ids, stop_terminated):
        if not new_token_ids:
            # Skip detokenization if no new token ids.
            return None

        if stop_terminated and not self.include_stop_str_in_output:
            skipped_stop_token_id = new_token_ids[-1]
            new_token_ids = new_token_ids[:-1]
        else:
            skipped_stop_token_id = None
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            self.output_text += str(new_token_id)
        if skipped_stop_token_id is not None:
            self.token_ids.append(skipped_stop_token_id)
        return None

    def get_next_output_text(self, finished, delta):
        # We return the full output text if the sequence is finished.
        buffer_length = 0 if finished else self.stop_buffer_length
        if not delta:
            return self.output_text
        return self.output_text
'''


class _Req:
    def __init__(self, rid, tx, prefill=True):
        self.request_id = rid
        self.kv_transfer_params = {
            "transfer_id": tx,
            "do_remote_prefill": prefill,
            "do_remote_decode": False,
        }


class _Blocks:
    def __init__(self, groups):
        self._groups = groups

    def get_block_ids(self):
        return self._groups


class _Tok:
    def __init__(self, specials):
        self.specials = set(specials)

    def decode(self, ids, skip_special_tokens=True):
        out = []
        for i in ids:
            if skip_special_tokens and i in self.specials:
                continue
            out.append("<s>" if i in self.specials else chr(97 + (i % 26)))
        return "".join(out)


def selftest() -> int:
    d = tempfile.mkdtemp(prefix="dsv4decode-")
    conn_dir = os.path.join(
        d, "distributed/kv_transfer/kv_connector/v1/moriio"
    )
    detok_dir = os.path.join(d, "v1/engine")
    os.makedirs(conn_dir)
    os.makedirs(detok_dir)
    conn_path = os.path.join(conn_dir, "moriio_connector.py")
    detok_path = os.path.join(detok_dir, "detokenizer.py")
    open(conn_path, "w").write(CONN_FAKE)
    open(detok_path, "w").write(DETOK_FAKE)

    if patch(d):
        print("[selftest] FAIL: patch returned non-zero")
        return 1
    cout = open(conn_path).read()
    dout = open(detok_path).read()
    for needle in (
        "        _dsv4_decode_alloc_log(self, request, blocks, num_external_tokens)\n",
        "[dsv4-decode] alloc n=%d",
        "_dsv4_decode_sample_log(self, new_token_ids)",
        "_dsv4_decode_sample_sum(self)",
    ):
        blob = cout + dout
        if needle not in blob:
            print(f"[selftest] FAIL: missing {needle!r}")
            return 1

    if patch(d):
        print("[selftest] FAIL: second apply returned non-zero")
        return 1
    if cout != open(conn_path).read() or dout != open(detok_path).read():
        print("[selftest] FAIL: not idempotent")
        return 1

    sys.path.insert(0, os.path.dirname(conn_path))
    sys.path.insert(0, os.path.dirname(detok_path))
    import logging

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    import moriio_connector as mc  # noqa: PLC0415
    import detokenizer as dt  # noqa: PLC0415

    os.environ["DSV4_DECODE_DIAG"] = "1"
    s = mc.MoRIIOConnectorScheduler()
    req = _Req("rid-2k", "tx-aaa")
    g1 = ([1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12], [13, 14, 15], list(range(17)), [20, 21, 22])
    s.update_state_after_alloc(req, _Blocks(g1), 2193)
    s.update_state_after_alloc(req, _Blocks(g1), 2193)
    g2 = ([99] + list(g1[0][1:]),) + g1[1:]
    s.update_state_after_alloc(req, _Blocks(g2), 2193)
    s.is_producer = True
    s.update_state_after_alloc(_Req("rid-p", "tx-p"), _Blocks(g1), 17)

    os.environ["DSV4_DECODE_DIAG"] = "0"
    s2 = mc.MoRIIOConnectorScheduler()
    s2.update_state_after_alloc(req, _Blocks(g1), 2193)

    os.environ["DSV4_DECODE_DIAG"] = "1"
    det = dt.BaseIncrementalDetokenizer()
    det.tokenizer = _Tok({0, 1})
    det.skip_special_tokens = True
    det.request_id = "rid-2k"
    det.update([0, 1, 2], False)
    det.get_next_output_text(True, True)

    print("[selftest] OK (alloc n=1/2/3 + sample-tok/sum under =1; producer quiet)")
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return selftest()
    if len(sys.argv) != 2:
        print(
            f"usage: {sys.argv[0]} <vllm_install_dir>|--selftest",
            file=sys.stderr,
        )
        return 2
    return patch(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
