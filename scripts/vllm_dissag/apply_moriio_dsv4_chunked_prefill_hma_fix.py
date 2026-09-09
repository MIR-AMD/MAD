#!/usr/bin/env python3
"""Chunked prefill: the last-chunk test is block arithmetic that cannot work (218328).

Symptom (218328 / 218342, HMA=1, DSV4 Pro and Flash): short curl prompts return
COHERENT, every longer prompt hangs until the 60s reap
(``no finished_sending notification after 60s``) and decode streams zero bytes.
The producer logs ``[dsv4-hma] wait_for_save transferring 243 registered
layers`` but there is **no** ``[dsv4-gate] seal transfer=`` and **no**
``[dsv4-gate] finalize transfer=`` for the hanging transfer, while short
transfers count a clean 243/243. Nothing was ever scheduled for the long
request: ``build_connector_meta`` never called ``add_new_req``,
``seal_pending_transfers`` found nothing pending, and decode waited forever.

Root cause. ``update_state_after_alloc`` stores the save entry as::

    local_block_ids = _dsv4_all_group_block_ids(blocks)   # [[...], [...], ...]
    self._reqs_need_save[request.request_id] = (request, local_block_ids)

and ``build_connector_meta`` decides "is this the last chunk?" with::

    if req.num_prompt_tokens > len(block_ids) * self.block_size:
        # not last chunk prefill

Both factors are wrong under HMA, and they are wrong in ways that multiply:

* ``_dsv4_all_group_block_ids`` keeps **one sublist per KV-cache group** (that is
  the point of the HMA patcher), so ``len(block_ids)`` is the **group count** —
  ``kv_groups=5`` on DSV4 — not a block count.
* ``self.block_size`` is the **smallest** group page, not the MLA page. DSV4
  groups are 256 / 64 / 64 / 8 / **4**, so ``self.block_size`` is **4**.

The product is the constant **5 × 4 = 20 tokens**, with no relation to the
prompt or to how much of it has been computed. Measured on hardware in 218364:

    [dsv4-chunk] defer rid=… prompt=17 covered=4 nested=True outer=5

Everything over ~20 tokens is classified "not last chunk" forever, parked in
``_reqs_need_pending_save``, and never emitted. That is the whole bug, and it
also explains the "stochastic" short-prompt bisect (14/17/18 tokens passed,
24/32 hung): a deterministic ~20-token cliff, not a race.

Do not try to repair this with a better block count. The first attempt at this
patcher swapped ``len(block_ids)`` for the group-0 block count and kept
``* self.block_size``; on a 17-token prompt that gave ``covered = 1 * 4 = 4`` and
deferred requests that used to work (218364, every transfer dead). Any variant
is guessing at a token count from a per-group block layout whose page sizes the
connector does not have in the scheduler process.

Fix: count tokens, which the scheduler already knows exactly::

    done = req.num_computed_tokens + scheduler_output.num_scheduled_tokens[req_id]
    incomplete = done < req.num_prompt_tokens

Verified against the installed scheduler: ``build_connector_meta`` is called at
``scheduler.py:1303`` and ``_update_after_schedule`` — which does
``request.num_computed_tokens += num_scheduled_token`` — only at ``:1318``. So
at decision time ``num_computed_tokens`` does **not** yet include the current
step, and adding ``num_scheduled_tokens`` is exact rather than double counting.
This is geometry-free and correct for HMA and non-HMA alike.

218687 (patcher retired): curl 17 tokens wrote, 2k NIAH never called add_new_req.
Re-applied 23 Aug with DSV4_CHUNK_HMA_FIX default 1. 218417 empty-decode after
243/243 is the write-diag group-key, not a reason to keep this off.

Also fixed: the drain path did
``updated_blocks = list(existing_blocks) + (block_ids)``, concatenating the
nested-by-group list with a flat group-0 list and producing a mixed list of
lists and ints — both the old ``len()`` and the ``local_block_ids`` handed to
``add_new_req`` were meaningless. Chunks now merge per group, preserving the
nesting the write-pick needs (the 217685 ``int() … not 'list'`` failure mode).

Runtime-gated so one image serves both arms::

    DSV4_CHUNK_HMA_FIX=0   218328 20-token cliff (len(groups)*page)
    DSV4_CHUNK_HMA_FIX=1   (default) token-based last-chunk test

The defer/final logs are emitted in **both** arms, so the off arm still reports
which branch a hanging request took.

Idempotent. Missing anchors is a hard error: a silent no-op here looks exactly
like the bug it fixes. Anchors verified byte-exact (indent 20/20/24/12, no tabs)
against the installed module.

Usage: apply_moriio_dsv4_chunked_prefill_hma_fix.py <vllm_install_dir>
       apply_moriio_dsv4_chunked_prefill_hma_fix.py --selftest
"""

import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
MARKER = "DSV4-CHUNK-HMA"

# Anchor the helpers to _dsv4_block_id_count so they land beside the counter
# they must not be confused with.
HELPER_ANCHOR = """def _dsv4_block_id_count(block_ids) -> int:
    if not block_ids:
        return 0
    if isinstance(block_ids[0], (list, tuple)):
        return sum(len(g) for g in block_ids)
    return len(block_ids)
"""

HELPERS = '''

# DSV4-CHUNK-HMA: 218328. The chunked-prefill last-chunk test multiplied a block
# count by self.block_size. Under HMA the stored list is nested per KV-cache
# group, so len() is the group count (5), and self.block_size is the SMALLEST
# group page (4 on DSV4: 256/64/64/8/4) rather than the MLA page. The product is
# a constant 20 tokens, unrelated to the prompt, so everything longer deferred
# forever and never reached add_new_req. Count tokens instead.
import os as _dsv4_chunk_os

DSV4_CHUNK_HMA_FIX = _dsv4_chunk_os.environ.get("DSV4_CHUNK_HMA_FIX", "1") == "1"


def _dsv4_chunk_progress(req, req_id, scheduler_output):
    """(computed, scheduled_this_step) for a prefill.

    build_connector_meta runs before _update_after_schedule, so
    num_computed_tokens excludes the current step and the two must be added.
    """
    sched = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
    return getattr(req, "num_computed_tokens", 0) or 0, sched.get(req_id, 0)


def _dsv4_chunk_incomplete(req, req_id, block_ids, block_size, scheduler_output):
    """True while a prefill still has chunks left to compute.

    Deliberately not block arithmetic. No block count can be turned into a token
    count here: the list is per-group and the connector does not carry the
    per-group page sizes in the scheduler process.
    """
    if not DSV4_CHUNK_HMA_FIX:
        return req.num_prompt_tokens > len(block_ids) * block_size
    _done, _sched = _dsv4_chunk_progress(req, req_id, scheduler_output)
    return (_done + _sched) < req.num_prompt_tokens


def _dsv4_merge_group_blocks(existing, new_ids):
    """Append a chunk's new block ids per group, keeping the nested shape.

    The unpatched line concatenates a nested-by-group list with a flat group-0
    list, producing a mix of lists and ints that no consumer can read.
    """
    if not DSV4_CHUNK_HMA_FIX:
        return list(existing) + (new_ids)
    if not existing or not isinstance(existing[0], (list, tuple)):
        return list(existing) + list(new_ids or [])
    merged = [list(g) for g in existing]
    if new_ids and isinstance(new_ids[0], (list, tuple)):
        for _gi, _g in enumerate(new_ids):
            if _gi < len(merged):
                merged[_gi].extend(_g)
    elif new_ids:
        merged[0].extend(new_ids)
    return merged
'''

# Keep every group's new blocks when the fix is on; the merge needs them.
PICK_OLD = """                    block_ids = new_block_ids[0]
"""

PICK_NEW = """                    block_ids = (
                        list(new_block_ids)
                        if DSV4_CHUNK_HMA_FIX
                        else new_block_ids[0]
                    )
"""

MERGE_OLD = """                    updated_blocks = list(existing_blocks) + (block_ids)
"""

MERGE_NEW = """                    updated_blocks = _dsv4_merge_group_blocks(
                        existing_blocks, block_ids
                    )
"""

DRAIN_OLD = """                    if (
                        len(self._reqs_need_pending_save[req_id][1]) * self.block_size
                        >= req.num_prompt_tokens
                    ):
"""

DRAIN_NEW = """                    if not _dsv4_chunk_incomplete(
                        req,
                        req_id,
                        self._reqs_need_pending_save[req_id][1],
                        self.block_size,
                        scheduler_output,
                    ):
                        _dsv4_done, _dsv4_step = _dsv4_chunk_progress(
                            req, req_id, scheduler_output
                        )
                        logger.info(
                            "[dsv4-chunk] final rid=%s prompt=%d computed=%d "
                            "sched=%d",
                            str(req_id)[-24:],
                            req.num_prompt_tokens,
                            _dsv4_done,
                            _dsv4_step,
                        )
"""

DEFER_OLD = """            if req.num_prompt_tokens > len(block_ids) * self.block_size:
"""

DEFER_NEW = """            if _dsv4_chunk_incomplete(
                req, req_id, block_ids, self.block_size, scheduler_output
            ):
                _dsv4_done, _dsv4_step = _dsv4_chunk_progress(
                    req, req_id, scheduler_output
                )
                logger.info(
                    "[dsv4-chunk] defer rid=%s prompt=%d computed=%d sched=%d "
                    "nested=%s outer=%d fix=%s",
                    str(req_id)[-24:],
                    req.num_prompt_tokens,
                    _dsv4_done,
                    _dsv4_step,
                    bool(block_ids) and isinstance(block_ids[0], (list, tuple)),
                    len(block_ids),
                    DSV4_CHUNK_HMA_FIX,
                )
"""


def _apply(src: str) -> str:
    if HELPER_ANCHOR not in src:
        raise ValueError("_dsv4_block_id_count anchor missing")
    src = src.replace(HELPER_ANCHOR, HELPER_ANCHOR + HELPERS, 1)

    for old, new, what in (
        (PICK_OLD, PICK_NEW, "drain group pick"),
        (MERGE_OLD, MERGE_NEW, "drain per-group merge"),
        (DRAIN_OLD, DRAIN_NEW, "drain last-chunk test"),
        (DEFER_OLD, DEFER_NEW, "defer last-chunk test"),
    ):
        if old not in src:
            raise ValueError(f"anchor missing: {what}")
        src = src.replace(old, new, 1)
    return src


def _stub() -> str:
    """Mirror the installed build_connector_meta shape, indents included."""
    return (
        "from vllm.logger import init_logger\n"
        "\nlogger = init_logger(__name__)\n"
        "\n\n" + HELPER_ANCHOR + "\n\nclass C:\n"
        "    def build_connector_meta(self, scheduler_output):\n"
        "        if True:\n"
        "            for i, req_id in enumerate(sched.req_ids):\n"
        "                new_block_ids = sched.new_block_ids[i]\n"
        "                if new_block_ids is not None:\n"
        + PICK_OLD
        + "                    if req_id not in self._reqs_need_pending_save:\n"
        "                        continue\n"
        "                    req, existing_blocks = self._pending[req_id]\n"
        + MERGE_OLD
        + "                    self._pending[req_id] = (req, updated_blocks)\n"
        + DRAIN_OLD
        + "                        kv_params = self._req_kv_params.pop(req_id, {})\n"
        "                        del self._reqs_need_pending_save[req_id]\n"
        "        for req_id, (req, block_ids) in self._reqs_need_save.items():\n"
        + DEFER_OLD
        + "                # not last chunk prefill\n"
        "                self._reqs_need_pending_save[req_id] = (req, block_ids)\n"
        "                continue\n"
    )


def _selftest() -> int:
    import ast

    out = _apply(_stub())
    ast.parse(out)
    assert MARKER in out
    assert "_dsv4_chunk_incomplete(" in out
    assert "_dsv4_merge_group_blocks(" in out
    assert "[dsv4-chunk] defer" in out
    assert "[dsv4-chunk] final" in out
    # Block arithmetic must be gone from the fix path.
    assert "* self.block_size\n                        >=" not in out
    # The counter it must not be confused with survives untouched.
    assert "sum(len(g) for g in block_ids)" in out

    try:
        _apply(out)
    except ValueError:
        pass
    else:
        raise AssertionError("second apply should not find anchors")

    ns: dict = {}
    exec(HELPERS, ns)  # noqa: S102
    inc = ns["_dsv4_chunk_incomplete"]
    merge = ns["_dsv4_merge_group_blocks"]

    class _Req:
        def __init__(self, prompt, computed):
            self.num_prompt_tokens = prompt
            self.num_computed_tokens = computed

    class _SO:
        def __init__(self, sched):
            self.num_scheduled_tokens = sched

    nested = [[1, 2, 3, 4], [9, 9], [8], [7], [6]]

    # --- off arm reproduces the bug exactly: len(nested)=5 groups x page 4 = 20
    ns["DSV4_CHUNK_HMA_FIX"] = False
    assert inc(_Req(17, 0), "r", nested, 4, _SO({"r": 17})) is False, "17 <= 20 passes"
    assert inc(_Req(21, 0), "r", nested, 4, _SO({"r": 21})) is True, "21 > 20 defers"
    assert inc(_Req(2193, 2193), "r", nested, 4, _SO({"r": 0})) is True, "long defers"
    assert merge(nested, [5, 6]) == nested + [5, 6], "off arm unchanged"

    # --- on arm: token accounting, geometry irrelevant
    ns["DSV4_CHUNK_HMA_FIX"] = True
    # 17-token prompt, whole thing scheduled this step -> final immediately.
    assert inc(_Req(17, 0), "r", nested, 4, _SO({"r": 17})) is False
    # The 218364 regression: must NOT defer a short prompt any more.
    assert inc(_Req(17, 0), "r", [[1], [1], [1], [1], [1]], 4, _SO({"r": 17})) is False
    # 2193 tokens at mnbt 1024: defer, defer, then final on the third chunk.
    r = _Req(2193, 0)
    assert inc(r, "r", nested, 4, _SO({"r": 1024})) is True
    r.num_computed_tokens = 1024
    assert inc(r, "r", nested, 4, _SO({"r": 1024})) is True
    r.num_computed_tokens = 2048
    assert inc(r, "r", nested, 4, _SO({"r": 145})) is False, "final chunk"
    # Missing entries must not crash or falsely finalize.
    assert inc(_Req(2193, 0), "r", nested, 4, _SO({})) is True
    assert inc(_Req(2193, 0), "r", nested, 4, None) is True

    # --- merge keeps the nesting the write-pick needs
    merged = merge(nested, [[10, 11], [12], [], [], []])
    assert merged[0] == [1, 2, 3, 4, 10, 11], merged[0]
    assert merged[1] == [9, 9, 12], merged[1]
    assert all(isinstance(g, list) for g in merged), "nesting must survive"
    flat = merge(nested, [10, 11])
    assert flat[0] == [1, 2, 3, 4, 10, 11]
    assert all(isinstance(g, list) for g in flat)

    print("[dsv4-chunk-hma] selftest OK")
    return 0


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        return _selftest()
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir> | --selftest", file=sys.stderr)
        return 2

    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[dsv4-chunk-hma] {REL} not found under {sys.argv[1]} -- skipping.")
        return 0

    src = open(path).read()
    if MARKER in src:
        print(f"[dsv4-chunk-hma] already patched in {path} -- no-op.")
        return 0

    try:
        out = _apply(src)
    except ValueError as e:
        print(
            f"[dsv4-chunk-hma] ERROR: {e}. Refusing a partial patch: a silent "
            "no-op here is indistinguishable from the 218328 hang.",
            file=sys.stderr,
        )
        return 1

    tmp = path + ".dsv4chunk"
    with open(tmp, "w") as f:
        f.write(out)
    os.replace(tmp, path)

    try:
        import py_compile

        py_compile.compile(path, doraise=True)
    except Exception as e:  # noqa: BLE001
        print(f"[dsv4-chunk-hma] ERROR: compile failed: {e}", file=sys.stderr)
        return 1

    print(
        f"[dsv4-chunk-hma] patched {path}: last-chunk test counts tokens "
        "(num_computed + num_scheduled vs num_prompt_tokens), chunks merge per "
        "group. Gated on DSV4_CHUNK_HMA_FIX (default 1)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
