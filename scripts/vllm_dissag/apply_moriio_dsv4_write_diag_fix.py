#!/usr/bin/env python3
"""HMA write-offset cache key + per-request WRITE diagnostics.

218687 (HMA=1, chunk patcher retired): curl 3/3 then 2k NIAH client-timeout with
zero producer writes. That hang is the 20-token last-chunk cliff (chunk patcher).
This file is the *other* 218417 leftover:

1. ``_get_write_geometry_key`` was ``(shape, stride, dtype)``. HMA picks offsets
   per group, but groups 1 and 2 are SlidingWindowMLA twins (same swa_cache
   shape), so group 2 reused group 1's dests. **Fix: append the layer's group
   index to the key.** Collision logs stay as a tripwire (should be 0).
   220801: group-key held (geom-fill g1 and g2 both logged, collision=0) but
   NIAH is multi-chunk. Curl is one page; 2k's first chunk cached that
   geometry and later chunks HIT, so dests stayed the short-prompt set
   (``offsets local=9`` only on a miss). **Also append**
   ``len(task.local_block_ids)`` so a 1-page curl/chunk cannot supply
   offsets to a 9-page WRITE.
2. ``update_state_after_alloc`` / pending-save can fire twice and drop the
   first alloc's block ids. **Fix: merge per group** instead of overwrite.
   Still logs save-overwrite / pending-overwrite.

Also re-keys the HMA write-pair INFO to ``(transfer_id, group)`` so 2k/8k log
three lines each, not just curl.

Gated on DSV4_WRITE_DIAG (default 1). The group-key and merge are behavioural
and apply whenever this patcher runs. Idempotent.

218428/218435: the connector helper was inserted with a greedy ridmap
``return False`` regex that never matched (ridmap's except is 8-space).
Call sites in ``build_connector_meta`` then NameError'd on the first
chunked prompt. Curl is last-chunk immediately so it never hit that
branch. Insert the helper immediately above ``class MoRIIOConnectorScheduler``
(same as ridmap) and inline the env check at the call sites.

Usage: apply_moriio_dsv4_write_diag_fix.py <vllm_install_dir>
       apply_moriio_dsv4_write_diag_fix.py --selftest
"""

from __future__ import annotations

import os
import re
import sys
import tempfile

ENGINE_REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py"
CONN_REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
MARKER = "[dsv4-write-diag]"

ENGINE_HELPER = '''
def _dsv4_write_diag_on() -> bool:
    try:
        import os as _o

        return _o.environ.get("DSV4_WRITE_DIAG", "1") == "1"
    except Exception:
        return False


def _dsv4_geom_cache_note(request_info, geometry_key, layer_name, worker, hit: bool) -> None:
    """Log first fill of an offsets-cache key, and group collisions on hit."""
    if not _dsv4_write_diag_on():
        return
    l2g = getattr(worker, "_dsv4_layer_to_group", None) or {}
    gi = l2g.get(layer_name)
    tx = getattr(request_info, "transfer_id", None)
    owners = getattr(request_info, "_dsv4_geom_owner", None)
    if owners is None:
        request_info._dsv4_geom_owner = {}
        owners = request_info._dsv4_geom_owner
    shape = geometry_key[0] if geometry_key else None
    nloc = geometry_key[4] if geometry_key and len(geometry_key) > 4 else None
    if not hit:
        filled = getattr(request_info, "_dsv4_geom_fill_logged", None)
        if filled is None:
            request_info._dsv4_geom_fill_logged = set()
            filled = request_info._dsv4_geom_fill_logged
        owners[geometry_key] = (layer_name, gi)
        fill_id = (gi, nloc)
        if fill_id in filled:
            return
        filled.add(fill_id)
        logger.info(
            "[dsv4-write-diag] geom-fill tx=%s layer=%s group=%s "
            "key_shape=%s nloc=%s",
            tx,
            layer_name,
            gi,
            shape,
            nloc,
        )
        return
    olayer, ogi = owners.get(geometry_key, (None, None))
    if ogi == gi:
        return
    n = int(getattr(request_info, "_dsv4_geom_collisions", 0) or 0)
    if n >= 4:
        return
    request_info._dsv4_geom_collisions = n + 1
    logger.error(
        "[dsv4-write-diag] geom-collision tx=%s layer=%s group=%s "
        "owner_layer=%s owner_group=%s key_shape=%s",
        tx,
        layer_name,
        gi,
        olayer,
        ogi,
        shape,
    )

'''

CONN_HELPER = '''
def _dsv4_write_diag_on() -> bool:
    try:
        import os as _o

        return _o.environ.get("DSV4_WRITE_DIAG", "1") == "1"
    except Exception:
        return False


def _dsv4_merge_save_ids(existing, new_ids):
    """Keep earlier chunk/group ids when alloc or pending-save fires twice."""
    if not existing:
        return new_ids
    if not new_ids:
        return existing
    if isinstance(existing[0], (list, tuple)) and isinstance(new_ids[0], (list, tuple)):
        merged = [list(g) for g in existing]
        for _gi, _g in enumerate(new_ids):
            if _gi >= len(merged):
                merged.append(list(_g))
                continue
            _seen = set(merged[_gi])
            merged[_gi].extend(x for x in _g if x not in _seen)
        return merged
    if isinstance(existing[0], (list, tuple)):
        return existing
    _seen = set(existing)
    return list(existing) + [x for x in new_ids if x not in _seen]

'''


def _insert_after(src: str, anchor: str, insert: str, already: str, label: str, applied: list) -> str:
    if already in src:
        applied.append(f"{label} (already)")
        return src
    if anchor not in src:
        applied.append(f"{label} (anchor missing, skipped)")
        return src
    applied.append(label)
    return src.replace(anchor, anchor + insert, 1)


GEOM_KEY_LINE = "        geometry_key = _get_write_geometry_key(layer_cache)\n"
GEOM_KEY_GROUP = (
    "        _gi = (getattr(self.worker, \"_dsv4_layer_to_group\", None) or {}).get(\n"
    "            task.layer_name\n"
    "        )\n"
    "        geometry_key = (*_get_write_geometry_key(layer_cache), _gi)\n"
)
GEOM_KEY_GROUP_NLOC = (
    "        _gi = (getattr(self.worker, \"_dsv4_layer_to_group\", None) or {}).get(\n"
    "            task.layer_name\n"
    "        )\n"
    "        _nloc = len(task.local_block_ids or [])\n"
    "        geometry_key = (*_get_write_geometry_key(layer_cache), _gi, _nloc)\n"
)


def _patch_geom_group_key(src: str, applied: list) -> str:
    """HMA groups 1 and 2 share swa_cache shape; the key must include group.

    220801: also key by n_local so chunked NIAH cannot reuse curl/first-chunk
    offsets (1-page HIT on a 9-page WRITE).
    """
    if ", _gi, _nloc)" in src:
        applied.append("geom-group-nloc (already)")
        return src
    if GEOM_KEY_GROUP in src:
        applied.append("geom-group-nloc")
        return src.replace(GEOM_KEY_GROUP, GEOM_KEY_GROUP_NLOC, 1)
    if GEOM_KEY_LINE not in src:
        applied.append("geom-group-key (anchor missing, skipped)")
        return src
    applied.append("geom-group-nloc")
    return src.replace(GEOM_KEY_LINE, GEOM_KEY_GROUP_NLOC, 1)


def _patch_engine(src: str, applied: list) -> str:
    src = _insert_after(
        src,
        "def _get_write_geometry_key(kv_cache: torch.Tensor) -> WriteGeometryKey:\n"
        "    return (tuple(kv_cache.shape), tuple(kv_cache.stride()), kv_cache.dtype)\n",
        ENGINE_HELPER,
        "def _dsv4_geom_cache_note(",
        "engine-helper",
        applied,
    )
    old = (
        "        geometry_key = _get_write_geometry_key(layer_cache)\n"
        "        offsets = request_info.transfer_offsets.get(geometry_key)\n"
        "        if offsets is None:\n"
        "            offsets = self.worker._compute_block_transfer_offsets(\n"
        "                task.layer_name,\n"
        "                task.local_block_ids,\n"
        "                request_info.block_ids,\n"
        "                remote_moriio_meta,\n"
        "            )\n"
        "            request_info.transfer_offsets[geometry_key] = offsets\n"
    )
    new = (
        "        geometry_key = _get_write_geometry_key(layer_cache)\n"
        "        offsets = request_info.transfer_offsets.get(geometry_key)\n"
        "        if offsets is None:\n"
        "            offsets = self.worker._compute_block_transfer_offsets(\n"
        "                task.layer_name,\n"
        "                task.local_block_ids,\n"
        "                request_info.block_ids,\n"
        "                remote_moriio_meta,\n"
        "            )\n"
        "            request_info.transfer_offsets[geometry_key] = offsets\n"
        "            _dsv4_geom_cache_note(\n"
        "                request_info, geometry_key, task.layer_name,\n"
        "                self.worker, hit=False,\n"
        "            )\n"
        "        else:\n"
        "            _dsv4_geom_cache_note(\n"
        "                request_info, geometry_key, task.layer_name,\n"
        "                self.worker, hit=True,\n"
        "            )\n"
    )
    if "_dsv4_geom_cache_note(" in src and "hit=False" in src:
        applied.append("geom-cache (already)")
    elif old not in src:
        applied.append("geom-cache (anchor missing, skipped)")
    else:
        applied.append("geom-cache")
        src = src.replace(old, new, 1)
    return _patch_geom_group_key(src, applied)


def _patch_conn_helper(src: str, applied: list) -> str:
    """Put the helper just above the scheduler class, same as ridmap.

    Do not regex-append onto ``_dsv4_ridmap_on``: that function's ``return False``
    is 8-space (except body), so a 4-space match either misses or swallows
    later helpers. 218428/218435 then NameError'd ``_dsv4_write_diag_on``.
    """
    if re.search(r"^def _dsv4_merge_save_ids\(", src, re.M):
        applied.append("conn-helper (already)")
        return src
    if re.search(r"^def _dsv4_write_diag_on\(\)", src, re.M):
        # Older logging-only helper: splice the merge in before the class.
        anchor = "class MoRIIOConnectorScheduler"
        idx = src.find(anchor)
        if idx < 0:
            applied.append("conn-helper (anchor missing, skipped)")
            return src
        merge_only = (
            "def _dsv4_merge_save_ids(existing, new_ids):\n"
            "    \"\"\"Keep earlier chunk/group ids when alloc or pending-save fires twice.\"\"\"\n"
            "    if not existing:\n"
            "        return new_ids\n"
            "    if not new_ids:\n"
            "        return existing\n"
            "    if isinstance(existing[0], (list, tuple)) and isinstance(new_ids[0], (list, tuple)):\n"
            "        merged = [list(g) for g in existing]\n"
            "        for _gi, _g in enumerate(new_ids):\n"
            "            if _gi >= len(merged):\n"
            "                merged.append(list(_g))\n"
            "                continue\n"
            "            _seen = set(merged[_gi])\n"
            "            merged[_gi].extend(x for x in _g if x not in _seen)\n"
            "        return merged\n"
            "    if isinstance(existing[0], (list, tuple)):\n"
            "        return existing\n"
            "    _seen = set(existing)\n"
            "    return list(existing) + [x for x in new_ids if x not in _seen]\n\n\n"
        )
        applied.append("conn-helper-merge")
        return src[:idx] + merge_only + src[idx:]
    anchor = "class MoRIIOConnectorScheduler"
    idx = src.find(anchor)
    if idx < 0:
        applied.append("conn-helper (anchor missing, skipped)")
        return src
    applied.append("conn-helper")
    return src[:idx] + CONN_HELPER.lstrip("\n") + "\n\n" + src[idx:]


def _patch_need_save_overwrite(src: str, applied: list) -> str:
    if "_dsv4_merge_save_ids(_old, local_block_ids)" in src:
        applied.append("save-overwrite (already)")
        return src
    picks = (
        "            local_block_ids = _dsv4_all_group_block_ids(blocks)\n",
        "            local_block_ids = blocks.get_block_ids()[0]\n",
    )
    for pick in picks:
        old = (
            "        if params.get(\"do_remote_decode\"):\n"
            + pick
            + "            self._reqs_need_save[request.request_id] = (request, local_block_ids)\n"
        )
        new = (
            "        if params.get(\"do_remote_decode\"):\n"
            + pick
            + "            if request.request_id in self._reqs_need_save:\n"
            "                _old = self._reqs_need_save[request.request_id][1]\n"
            "                if _dsv4_write_diag_on():\n"
            "                    logger.warning(\n"
            "                        \"[dsv4-write-diag] save-overwrite rid=%s tx=%s \"\n"
            "                        \"old_n=%s new_n=%s n_groups=%s\",\n"
            "                        request.request_id,\n"
            "                        params.get(\"transfer_id\"),\n"
            "                        len(_old) if _old is not None else None,\n"
            "                        len(local_block_ids) if local_block_ids is not None else None,\n"
            "                        len(blocks.get_block_ids()) if blocks is not None else None,\n"
            "                    )\n"
            "                local_block_ids = _dsv4_merge_save_ids(_old, local_block_ids)\n"
            "            self._reqs_need_save[request.request_id] = (request, local_block_ids)\n"
        )
        if old in src:
            applied.append("save-overwrite")
            return src.replace(old, new, 1)
    applied.append("save-overwrite (anchor missing, skipped)")
    return src


def _patch_pending_overwrite(src: str, applied: list) -> str:
    if "_dsv4_merge_save_ids(_old, block_ids)" in src:
        applied.append("pending-overwrite (already)")
        return src
    old = (
        "                # not last chunk prefill\n"
        "                self._reqs_need_pending_save[req_id] = (req, block_ids)\n"
        "                continue\n"
    )
    new = (
        "                # not last chunk prefill\n"
        "                if req_id in self._reqs_need_pending_save:\n"
        "                    _old = self._reqs_need_pending_save[req_id][1]\n"
        "                    if _dsv4_write_diag_on():\n"
        "                        logger.warning(\n"
        "                            \"[dsv4-write-diag] pending-overwrite rid=%s \"\n"
        "                            \"old_n=%s new_n=%s prompt=%s\",\n"
        "                            req_id,\n"
        "                            len(_old) if _old is not None else None,\n"
        "                            len(block_ids) if block_ids is not None else None,\n"
        "                            getattr(req, \"num_prompt_tokens\", None),\n"
        "                        )\n"
        "                    block_ids = _dsv4_merge_save_ids(_old, block_ids)\n"
        "                self._reqs_need_pending_save[req_id] = (req, block_ids)\n"
        "                continue\n"
    )
    if old not in src:
        applied.append("pending-overwrite (anchor missing, skipped)")
        return src
    applied.append("pending-overwrite")
    return src.replace(old, new, 1)


def _patch_ungate_pair(src: str, applied: list) -> str:
    if "_pkey = (_xid, _gi)" in src:
        applied.append("ungate-pair (already)")
        return src
    old = (
        "        if layer_name not in self._dsv4_logged_pair:\n"
        "            self._dsv4_logged_pair.add(layer_name)\n"
    )
    new = (
        "        _xid = getattr(meta, \"transfer_id\", None)\n"
        "        _pkey = (_xid, _gi)\n"
        "        if _pkey not in self._dsv4_logged_pair:\n"
        "            self._dsv4_logged_pair.add(_pkey)\n"
    )
    n = src.count(old)
    if n == 0:
        applied.append("ungate-pair (anchor missing, skipped)")
        return src
    applied.append(f"ungate-pair x{n}")
    return src.replace(old, new)


def _patch_ungate_fold(src: str, applied: list) -> str:
    """Fold/geom-skip were once-per-layer, so 2k never logged (218417)."""
    if "_dsv4_fold_logs" in src:
        applied.append("ungate-fold (already)")
        return src
    old = (
        "            if layer_name not in self._dsv4_logged_fold:\n"
        "                self._dsv4_logged_fold.add(layer_name)\n"
    )
    new = (
        "            _fold_n = int(getattr(self, \"_dsv4_fold_logs\", 0) or 0)\n"
        "            if _fold_n < 64 or geom.block_size != _gbs:\n"
        "                self._dsv4_fold_logs = _fold_n + 1\n"
        "                self._dsv4_logged_fold.add(layer_name)\n"
    )
    if old not in src:
        applied.append("ungate-fold (anchor missing, skipped)")
        return src
    applied.append("ungate-fold")
    src = src.replace(old, new, 1)
    skip_old = (
        "        if layer_name not in self._dsv4_logged_geom_skip:\n"
        "            self._dsv4_logged_geom_skip.add(layer_name)\n"
    )
    skip_new = (
        "        _skip_n = int(getattr(self, \"_dsv4_geom_skip_logs\", 0) or 0)\n"
        "        if _skip_n < 64:\n"
        "            self._dsv4_geom_skip_logs = _skip_n + 1\n"
        "            self._dsv4_logged_geom_skip.add(layer_name)\n"
    )
    if skip_old in src:
        src = src.replace(skip_old, skip_new, 1)
        applied.append("ungate-geom-skip")
    return src


def _write(path: str, src: str) -> None:
    tmp = path + ".dsv4wdiag"
    with open(tmp, "w") as f:
        f.write(src)
    os.replace(tmp, path)


def patch_engine(path: str) -> int:
    src = open(path).read()
    orig = src
    applied: list = []
    src = _patch_engine(src, applied)
    if src == orig:
        print(f"[dsv4-write-diag] engine no changes ({', '.join(applied)}) in {path}")
        return 0
    _write(path, src)
    print(f"[dsv4-write-diag] engine patched ({', '.join(applied)}) in {path}")
    return 0


def patch_connector(path: str) -> int:
    src = open(path).read()
    orig = src
    applied: list = []
    src = _patch_conn_helper(src, applied)
    src = _patch_need_save_overwrite(src, applied)
    src = _patch_pending_overwrite(src, applied)
    src = _patch_ungate_pair(src, applied)
    src = _patch_ungate_fold(src, applied)
    if src == orig:
        print(f"[dsv4-write-diag] connector no changes ({', '.join(applied)}) in {path}")
        return 0
    _write(path, src)
    print(f"[dsv4-write-diag] connector patched ({', '.join(applied)}) in {path}")
    return 0


_FAKE_ENGINE = '''
import torch
import logging
logger = logging.getLogger(__name__)
WriteGeometryKey = tuple

def _get_write_geometry_key(kv_cache: torch.Tensor) -> WriteGeometryKey:
    return (tuple(kv_cache.shape), tuple(kv_cache.stride()), kv_cache.dtype)

class MoRIIOWriter:
    def _prepare_transfer_plan(self, task, request_info, remote_moriio_meta):
        layer_cache = self.worker.kv_caches[task.layer_name]
        geometry_key = _get_write_geometry_key(layer_cache)
        offsets = request_info.transfer_offsets.get(geometry_key)
        if offsets is None:
            offsets = self.worker._compute_block_transfer_offsets(
                task.layer_name,
                task.local_block_ids,
                request_info.block_ids,
                remote_moriio_meta,
            )
            request_info.transfer_offsets[geometry_key] = offsets
        local_off, remote_off, sizes = offsets
        return offsets
'''

# Production-shaped ridmap helper (8-space except return). The old write-diag
# insert regex looked for 4-space ``return False`` after this def and missed,
# then NameError'd at the pending-save call site (218428/218435).
_FAKE_CONN = '''
import logging
logger = logging.getLogger(__name__)

def _dsv4_ridmap_on() -> bool:
    """DSV4_RID_MAP_DIAG=1 enables the rid<->transfer_id map/unmap trace."""
    try:
        import os as _o

        return _o.environ.get("DSV4_RID_MAP_DIAG", "0") == "1"
    except Exception:
        return False


def _dsv4_ridmap_log(self, what: str, request_id, transfer_id=None) -> None:
    if not _dsv4_ridmap_on():
        return
    try:
        logger.info("[dsv4-ridmap] %s rid=%r", what, request_id)
    except Exception:
        pass


class MoRIIOConnectorScheduler:
    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        params = request.kv_transfer_params
        if not params:
            return
        if params.get("do_remote_decode"):
            local_block_ids = blocks.get_block_ids()[0]
            self._reqs_need_save[request.request_id] = (request, local_block_ids)
            self._req_kv_params[request.request_id] = dict(params)

    def build_connector_meta(self, scheduler_output):
        for req_id, (req, block_ids) in self._reqs_need_save.items():
            kv_params = self._req_kv_params.get(req_id, req.kv_transfer_params or {})
            if req.num_prompt_tokens > len(block_ids) * self.block_size:
                # not last chunk prefill
                self._reqs_need_pending_save[req_id] = (req, block_ids)
                continue
            meta = kv_params
        _gi = 1
        layer_name = "x"
        meta = type("M", (), {"transfer_id": "tx-1"})()
        if not getattr(self, "_dsv4_logged_pair", None):
            self._dsv4_logged_pair = set()
        if layer_name not in self._dsv4_logged_pair:
            self._dsv4_logged_pair.add(layer_name)
'''


def selftest() -> int:
    d = tempfile.mkdtemp(prefix="dsv4wdiag-")
    eng = os.path.join(d, "moriio_engine.py")
    con = os.path.join(d, "moriio_connector.py")
    open(eng, "w").write(_FAKE_ENGINE)
    open(con, "w").write(_FAKE_CONN)
    applied_e: list = []
    applied_c: list = []
    e2 = _patch_engine(open(eng).read(), applied_e)
    c2 = _patch_conn_helper(open(con).read(), applied_c)
    c2 = _patch_need_save_overwrite(c2, applied_c)
    c2 = _patch_pending_overwrite(c2, applied_c)
    c2 = _patch_ungate_pair(c2, applied_c)
    c2 = _patch_ungate_fold(c2, applied_c)
    for needle, blob in (
        ("_dsv4_geom_cache_note", e2),
        ("geom-collision", e2),
        ("_nloc = len(task.local_block_ids or [])", e2),
        (", _gi, _nloc)", e2),
        ("nloc=%s", e2),
        ("save-overwrite rid=", c2),
        ("pending-overwrite rid=", c2),
        ("_dsv4_merge_save_ids(_old, local_block_ids)", c2),
        ("_dsv4_merge_save_ids(_old, block_ids)", c2),
        ("_pkey = (_xid, _gi)", c2),
    ):
        if needle not in blob:
            print(f"[selftest] FAIL: missing {needle!r} applied={applied_e + applied_c}")
            return 1
    if "_dsv4_write_diag_on() and req_id" in c2:
        print("[selftest] FAIL: pending-overwrite still calls the helper")
        return 1
    helper_line = None
    class_line = None
    for i, ln in enumerate(c2.splitlines()):
        if ln.startswith("def _dsv4_write_diag_on()"):
            helper_line = i
        if ln.startswith("class MoRIIOConnectorScheduler"):
            class_line = i
            break
    if helper_line is None or class_line is None or helper_line > class_line:
        print(
            f"[selftest] FAIL: helper not module-level before class "
            f"(helper={helper_line} class={class_line})"
        )
        return 1
    try:
        compile(e2, "moriio_engine.py", "exec")
        ns: dict = {}
        exec(compile(c2, "moriio_connector.py", "exec"), ns)
    except Exception as exc:
        print(f"[selftest] FAIL: compile/exec: {exc}")
        return 1
    if not callable(ns.get("_dsv4_write_diag_on")):
        print("[selftest] FAIL: _dsv4_write_diag_on not visible at module scope")
        return 1
    if not callable(ns.get("_dsv4_merge_save_ids")):
        print("[selftest] FAIL: _dsv4_merge_save_ids not visible at module scope")
        return 1
    merged = ns["_dsv4_merge_save_ids"]([[1, 2], [3]], [[2, 4], [5]])
    if merged[0] != [1, 2, 4] or merged[1] != [3, 5]:
        print(f"[selftest] FAIL: merge {merged}")
        return 1
    sched = ns["MoRIIOConnectorScheduler"]()
    sched.block_size = 16
    sched._reqs_need_save = {}
    sched._reqs_need_pending_save = {}
    sched._req_kv_params = {}

    class _Req:
        request_id = "r1"
        num_prompt_tokens = 10_000
        kv_transfer_params = {"do_remote_decode": True, "transfer_id": "tx-1"}

    class _Blocks:
        def get_block_ids(self):
            return [[1, 2, 3]]

    sched.update_state_after_alloc(_Req(), _Blocks(), 0)

    class _Blocks2:
        def get_block_ids(self):
            return [[3, 4, 5]]

    sched.update_state_after_alloc(_Req(), _Blocks2(), 0)
    if sched._reqs_need_save["r1"][1] != [1, 2, 3, 4, 5]:
        print(f"[selftest] FAIL: save merge {sched._reqs_need_save['r1'][1]}")
        return 1
    era_applied: list = []
    era_src = _FAKE_ENGINE.replace(GEOM_KEY_LINE, GEOM_KEY_GROUP)
    era_out = _patch_engine(era_src, era_applied)
    if GEOM_KEY_GROUP_NLOC not in era_out or GEOM_KEY_GROUP in era_out:
        print(f"[selftest] FAIL: 220801 group-key did not upgrade nloc {era_applied}")
        return 1
    sched._reqs_need_pending_save["r1"] = (_Req(), [1])
    try:
        sched.build_connector_meta(None)
    except NameError as exc:
        print(f"[selftest] FAIL: NameError in build_connector_meta: {exc}")
        return 1
    e3 = _patch_engine(e2, [])
    if e3 != e2:
        print("[selftest] FAIL: engine not idempotent")
        return 1
    c3 = _patch_conn_helper(c2, [])
    c3 = _patch_need_save_overwrite(c3, [])
    c3 = _patch_pending_overwrite(c3, [])
    c3 = _patch_ungate_pair(c3, [])
    c3 = _patch_ungate_fold(c3, [])
    if c3 != c2:
        print("[selftest] FAIL: connector not idempotent")
        return 1
    print("[selftest] OK", applied_e + applied_c)
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return selftest()
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    root = sys.argv[1]
    eng = os.path.join(root, ENGINE_REL)
    con = os.path.join(root, CONN_REL)
    rc = 0
    if os.path.isfile(eng):
        rc |= patch_engine(eng)
    else:
        print(f"[dsv4-write-diag] {ENGINE_REL} not found under {root} -- skipping.")
    if os.path.isfile(con):
        rc |= patch_connector(con)
    else:
        print(f"[dsv4-write-diag] {CONN_REL} not found under {root} -- skipping.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
