#!/usr/bin/env python3
"""HMA=1: prove group-0 bytes landed, then log sparse-MLA metadata (221697).

221697 sealed WRITE 243/243 and decode sampled 63x BOS (token 0) at 2k/8k
while 16k emitted FILLER. Dest-replace is out. Curl Lisa Su never proved
group-0. This patcher splits the remaining tree without changing WRITE or
sampling:

1. Connector ``_compute_block_transfer_offsets`` (producer). Hash one
   representative layer each of ``.attn`` (not swa), ``.attn.swa_cache``,
   ``.indexer.k_cache``: n_pages, abs_sum, 16-byte prefix, id head/tail.
2. Consumer dest ``.attn[dest_ids]`` **per HMA group**. 223113/223141:
   dest ids live on EngineCore alloc (2k group-0 ``[7,9,6,5,…,1,8]``,
   16k goes to page 68) but Worker ``store=0`` — WRITE-mode
   ``start_load_kv`` is ``ForwardContext``, ``_get_connector_metadata()``
   has empty requests, and a 4-gen dest-scan cap burned on curl so 2k/16k
   never hashed. Ship dest ids: scheduler stash → ``kv_transfer_params``
   + dest-dir JSON + ``meta.dsv4_dest``; Worker also reads ForwardContext
   ``block_tables``. Hash ``.attn[g0]`` / ``.attn[g1]`` / … (scatter) and
   role-matched swa/indexer. Dest key is ``(side, role, extra, bucket,
   z|nz)`` so curl cannot hide 2k (bucket ``m``) or 16k (``l``, pages
   10–68). Scan 1–16 is fallback only. Hooks: ``load`` / ``post-wait`` /
   ``finished`` / ``recving`` / consumer ``offsets`` / ``build_connector_meta``.
3. Producer zip dest ids (``extra=zip-remote``). 223147 hashed dest g0
   live but src-attn prefix ``7d5a71f3`` landed on dest ``.attn[10,11,12]``
   (SWA slots), not on g0 ``[7,9,…,8]``. Offsets only logged counts
   (``local=9 remote=9``). Log picked ``remote_block_ids`` after
   ``_dsv4_pick_group_ids`` / ``_dsv4_align_write_ids`` (kv_hash runs
   last). Nested remote after pick is flatten-miss. Do not hash dest ids
   against the src tensor.
4. DSV4 sparse backend ``build()``: ``metadata_key``, seq_lens, query_len,
   ``paged_kv_indices`` min/max/zero_frac. Missing backend file or anchors
   warn and skip (image-local path). The helper is appended at EOF — never
   inserted before the first ``class`` (222135: ``@dataclass`` on ``rocm.py``
   wrapped the helper and vLLM died at import). Missing connector offsets
   call is a hard error. 223147/223210 logged 0 ``[dsv4-mla-meta]`` lines
   — ``build()`` never ran. Hook ``_forward_decode`` and dump
   ``block_table`` (the AITER path; not ``paged_kv_indices``).
5. Post-seal pair-hash (``extra=pair-*``). 223210 zip remote was g0
   ``[7,9,…,8]`` but dest ``.attn[10,11,12]`` got src-attn prefix
   ``7d5a71f3``. ``load-g0`` at 2k was **before** ``rdma_wait``. Hash dest
   g0[0] (page 7) vs dest SWA0 (page 10) on 2nd ``start_load`` /
   ``alloc num_ext=0`` / ``get_finished`` / decode forward. Same prefix on
   10 not 7 => offset/geom; page 7 matches src => sparse ``block_table``.
   223327 pair-seal was still pre-WRITE (14:14:58 vs zip 14:14:59). Hash
   dest 7 vs 10 **inside** ``_forward_decode`` (decode is post-seal).
   6. 223327 ``fwd-decode`` keyed ``(where, sb)`` from ``qlen=16`` at CUDA-graph
   capture (``zero_frac=1``, ``abs7=0``, ``bt=[0,…]``) and hid 2k. Pair keyed
   ``(extra, bucket, id0)`` and hid the post-RDMA snapshot of the same pages.
   Key on **content** (live/prefix/``n_nz``/``imax``), log the full g0+SWA dest
   census (not only 7 vs 10), ``sb`` from ``seq_lens`` / ``swa_metadata`` never
   qlen. Cap 96 lines/process so 41-layer decode cannot flood. Marker
   ``DSV4-MLA-FWD-LIVE`` / ``DSV4-KV-PAIR-LIVE``.
   7. 223342 dest g0 pages are NZ after WRITE; ``fwd-decode`` still only
   logged graph capture (Python does not re-enter). Probe ``block_table``
   from ``GPUModelRunner._build_attention_metadata`` (Python every
   execute_model / dummy capture). Log per-HMA-group runner tables vs the
   built layer metadata. ``g0_nnz=0`` at 2k → scheduler never put dests in
   the table; ``g0_nnz>0 hit7=1`` but ``meta_attn`` zeros → builder/graph
   stale; both live → compress/layout. ``dec_cg=`` is the decode graph
   mode (``DSV4_EAGER=1`` → ``NONE``). ``g0slot=`` is group-0 indexing:
   dim[1] is slots/page (64 even / 2 odd), ``cov_s`` is ``pages*dim1``,
   ``cov_t`` is ``pages*256``. 2k ``need=2193`` with ``cov_s=576,18``
   means a kernel that treats dim[1] as tokens cannot see the haystack;
   ``cov_t=2304`` is the spec-token cover. Marker ``DSV4-KV-RUNNER-BT``.
   Key omits seq0 so 64 decode steps do not burn the cap.
   8. 223449: ``_forward_decode`` ran (curl ``bt=[1] n_nz=1 block_size=256``)
   but 2k at 22:37 logged **zero** ``[dsv4-mla-meta]`` lines. Per-layer
   ``prefix7`` in the key + cap 96 burned on curl's 41 layers. ``seq_lens=None``
   so ``seq0`` was ``max_seqlen_k`` (17) and hid 2k as ``sb=s``. Re-key
   ``(where, n_nz, imax, bt[:8], kvd1)``; bucket ``sb`` from real ``seq_lens``
   or ``n_nz`` never max_seqlen; log ``kvd1=`` (cache ``shape[1]``). Marker
   ``kvd1=`` upgrades the 223449 helper.
   9. 223514 2k ``block_size=256 kvd1=64 n_nz=9 bt=[7,9,…]`` dest NZ. d626
   already does ``block_size // compress_ratio`` then
   ``local_idx // block_size`` into that table — do not assign
   ``block_size=kvd1``. Log ``cr=`` ``kbs=`` ``swa_only=`` ``ntopk=``
   ``tmax=`` (``DSV4-MLA-CR``). ``tmax~seq0`` vs ``tmax~seq0/cr`` splits
   uncompressed vs compressed indexer space. Curl/16k FILLER still fit
   the SWA 128 window.

222578 hashed curl only: ``(side, role)`` once per process, so 2k BOS
never got ``side=src``. Buckets: ``s`` (n_pages<=2, curl), ``m`` (3–16,
2k attn=9), ``l`` (>16, 16k). 223141 dest scan always n=16 (bucket ``m``)
plus a 4-gen cap — curl took every slot. Src still keys ``(side, role,
bucket, extra)``.

Compare ``[dsv4-kv] side=src`` vs ``side=dst extra=…-g0`` on the same
rung (bucket ``m`` at 2k, ``l`` at 16k):

  dst abs_sum ~ 0, src not     -> dest/geom (stay in MoRIIO)
  dst extra=-g1 live in .attn  -> HMA groups scattered into group-0 pages
  sums match, zero_frac ~ 1    -> metadata_key / paged_kv_indices
                                  (GLM commit 2 analog on the DSV4 backend)
  both healthy                 -> compress_ratio slot mapping

Do not flip the GLM indexer sentinel. Do not apply GLM indexer H2.

Diagnostic only. Runtime flag DSV4_KV_HASH (default 0 = quiet; wrapper sets
1 on DSV4). Idempotent.

Usage: apply_moriio_dsv4_kv_hash_fix.py <vllm_install_dir>
       apply_moriio_dsv4_kv_hash_fix.py --selftest
"""
from __future__ import annotations

import os
import re
import sys
import tempfile

CONN_REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
RUNNER_REL = "v1/worker/gpu_model_runner.py"
MARKER = "[dsv4-kv]"
MLA_MARKER = "[dsv4-mla-meta]"

MLA_CANDIDATES = (
    "models/deepseek_v4/amd/rocm.py",
    "models/deepseek_v4/amd/rocm/__init__.py",
    "models/deepseek_v4/amd/rocm/mla.py",
    "models/deepseek_v4/amd/rocm_aiter_mla_sparse.py",
    "models/deepseek_v4/amd/flashmla.py",
    "v1/attention/backends/mla/rocm_aiter_mla_sparse.py",
    "v1/attention/backends/mla/rocm_flashmla_sparse.py",
    "v1/attention/backends/mla/flashmla_sparse.py",
)

# Inserted as a GPUModelRunner method (4-space indent) before capture_model.
RUNNER_BT_METHOD = r'''    def _dsv4_kv_runner_bt_log(
        self, attn_metadata, num_reqs, num_tokens, for_cudagraph_capture=False
    ) -> None:
        """DSV4-KV-RUNNER-BT: live block_table each execute_model (not graph replay).

        223342 dest g0 was NZ after WRITE; fwd-decode only logged CUDA-graph
        capture. This path runs in Python every step. Do not key on seq0 —
        64 decode tokens would burn the cap before 16k. ``dec_cg=`` is the
        decode graph mode (``DSV4_EAGER=1`` → ``NONE``). ``g0slot=`` is
        compress_ratio coverage (223417): even dim[1]=64 ratio 4, odd
        dim[1]=2 ratio 128, spec page 256.
        """
        try:
            import os as _o

            if _o.environ.get("DSV4_KV_HASH", "0") != "1":
                return
        except Exception:
            return
        try:
            nlog = int(getattr(self, "_dsv4_kv_bt_n", 0) or 0)
            if nlog >= 64:
                return

            def _ints(raw, cap=64):
                if raw is None:
                    return []
                try:
                    if hasattr(raw, "detach"):
                        raw = raw.detach()
                    if hasattr(raw, "cpu"):
                        raw = raw.cpu()
                    if hasattr(raw, "flatten"):
                        raw = raw.flatten()
                    if hasattr(raw, "tolist"):
                        raw = raw.tolist()
                except Exception:
                    return []
                out = []
                try:
                    for x in list(raw)[:cap]:
                        try:
                            out.append(int(x))
                        except Exception:
                            break
                except Exception:
                    return []
                return out

            def _row0(tbl):
                arr = None
                get = getattr(tbl, "get_numpy_array", None)
                if get is not None:
                    try:
                        arr = get()
                    except Exception:
                        arr = None
                if arr is None:
                    arr = getattr(tbl, "np", None)
                if arr is None:
                    return _ints(tbl)
                try:
                    if hasattr(arr, "shape") and getattr(arr, "ndim", 1) >= 2:
                        return _ints(arr[0], cap=128)
                    if hasattr(arr, "__getitem__") and not isinstance(
                        arr, (bytes, str)
                    ):
                        row = arr[0]
                        return _ints(row, cap=128)
                except Exception:
                    pass
                return _ints(arr)

            def _preview(ids, n=8):
                ids = [int(x) for x in ids if int(x) != 0]
                if len(ids) <= n:
                    return ids
                return ids[:n] + ["..."]

            def _meta_ids(meta):
                if meta is None:
                    return []
                for name in (
                    "block_table_tensor",
                    "block_table",
                    "block_tables",
                    "paged_kv_indices",
                ):
                    ids = _ints(getattr(meta, name, None), cap=128)
                    nz = [x for x in ids if x != 0]
                    if nz:
                        return nz
                return []

            def _meta_map(md):
                if md is None:
                    return {}
                if isinstance(md, dict):
                    return md
                if isinstance(md, (list, tuple)) and md and isinstance(md[0], dict):
                    return md[0]
                return {}

            groups_s = []
            g0_nz = []
            hit7 = 0
            hit10 = 0
            ng = 0
            tables = None
            bt = getattr(getattr(self, "input_batch", None), "block_table", None)
            if bt is not None:
                tables = getattr(bt, "block_tables", None)
                if not tables:
                    try:
                        n_g = len(getattr(
                            getattr(self, "kv_cache_config", None),
                            "kv_cache_groups",
                            [],
                        ) or [])
                    except Exception:
                        n_g = 0
                    if n_g:
                        tables = []
                        for i in range(min(n_g, 5)):
                            try:
                                tables.append(bt[i])
                            except Exception:
                                break
                    else:
                        tables = [bt]
            for gi, tbl in enumerate(list(tables or [])[:5]):
                ng += 1
                nums = _row0(tbl)
                nz = [x for x in nums if x != 0]
                if gi == 0:
                    g0_nz = nz
                    if 7 in nz:
                        hit7 = 1
                if 10 in nz:
                    hit10 = 1
                bs = getattr(tbl, "block_size", None)
                groups_s.append(
                    "%d:%s:%d:%s"
                    % (gi, bs if bs is not None else "-", len(nz), _preview(nz))
                )
            mmap = _meta_map(attn_metadata)
            attn_m = []
            swa_m = []
            upd = "-"
            try:
                ag = self.attn_groups[0][0]
                bld = ag.get_metadata_builder(0)
                upd = str(bool(getattr(bld, "supports_update_block_table", False)))
            except Exception:
                upd = "-"
            for lname, meta in list(mmap.items())[:48]:
                ln = str(lname)
                ids = _meta_ids(meta)
                if ".swa" in ln:
                    if not swa_m:
                        swa_m = ids
                elif ln.endswith(".attn") and "indexer" not in ln and "compressor" not in ln:
                    if not attn_m:
                        attn_m = ids
                if attn_m and swa_m:
                    break
            seq0 = None
            try:
                sl = getattr(self, "optimistic_seq_lens_cpu", None)
                if sl is not None:
                    seq0 = int(sl[0])
            except Exception:
                seq0 = None
            if _o.environ.get("DSV4_EAGER", "0") == "1":
                dec_cg = "NONE"
            else:
                dec_cg = (
                    _o.environ.get("DECODE_CUDAGRAPH_MODE")
                    or _o.environ.get("VLLM_CUDAGRAPH_MODE")
                    or "-"
                )
            spec_bs = 256
            try:
                if tables:
                    spec_bs = int(getattr(tables[0], "block_size", None) or 256)
            except Exception:
                spec_bs = 256
            npages = len(g0_nz)
            cov_t = npages * spec_bs if spec_bs else 0
            even_d1 = odd_d1 = None
            meta_bs_e = meta_bs_o = None
            try:
                pairs = []
                caches = getattr(self, "kv_caches", None)
                if isinstance(caches, dict):
                    pairs = list(caches.items())
                else:
                    names = []
                    try:
                        for _g in (
                            getattr(
                                getattr(self, "kv_cache_config", None),
                                "kv_cache_groups",
                                None,
                            )
                            or []
                        ):
                            ln = getattr(_g, "layer_names", None)
                            if isinstance(ln, str):
                                names.append(ln)
                            elif ln:
                                names.extend(list(ln))
                    except Exception:
                        names = []
                    if caches is not None and names:
                        try:
                            pairs = list(zip(names, list(caches)))
                        except Exception:
                            pairs = []
                for _ln, _ten in pairs:
                    _s = str(_ln)
                    if not _s.endswith(".attn"):
                        continue
                    if (
                        "indexer" in _s
                        or "compressor" in _s
                        or ".swa" in _s
                    ):
                        continue
                    shp = getattr(_ten, "shape", None)
                    if shp is None or len(shp) < 2:
                        continue
                    d1 = int(shp[1])
                    if d1 == 64 and even_d1 is None:
                        even_d1 = d1
                    elif d1 == 2 and odd_d1 is None:
                        odd_d1 = d1
                    elif d1 == 256 and even_d1 is None:
                        even_d1 = d1
            except Exception:
                pass
            for lname, meta in list(mmap.items())[:48]:
                ln = str(lname)
                if not ln.endswith(".attn"):
                    continue
                if "indexer" in ln or "compressor" in ln or ".swa" in ln:
                    continue
                mb = getattr(meta, "block_size", None)
                if mb is None:
                    continue
                try:
                    mb = int(mb)
                except Exception:
                    continue
                try:
                    li = int(ln.split("layers.")[1].split(".")[0])
                except Exception:
                    li = -1
                if li >= 0 and li % 2 == 0 and meta_bs_e is None:
                    meta_bs_e = mb
                elif li >= 0 and li % 2 == 1 and meta_bs_o is None:
                    meta_bs_o = mb
            def _ratio(d1):
                if not d1 or not spec_bs or spec_bs % int(d1):
                    return 0
                return spec_bs // int(d1)

            re_ = _ratio(even_d1) if even_d1 else 0
            ro_ = _ratio(odd_d1) if odd_d1 else 0
            cov_se = (npages * even_d1) if even_d1 else 0
            cov_so = (npages * odd_d1) if odd_d1 else 0
            need = int(seq0) if seq0 else 0
            last = (need - 1) if need else 0
            pg = (last // spec_bs) if spec_bs else 0
            sl = (last % spec_bs) if spec_bs else 0
            sl4 = (sl // re_) if re_ else 0
            sl128 = (sl // ro_) if ro_ else 0
            g0slot = (
                "e=%sr%s/o=%sr%s cov_s=%s,%s cov_t=%d need=%d "
                "tok=%d pg=%d sl=%d sl4=%d sl128=%d meta_bs=%s,%s"
                % (
                    even_d1 if even_d1 is not None else "?",
                    re_ or "?",
                    odd_d1 if odd_d1 is not None else "?",
                    ro_ or "?",
                    cov_se,
                    cov_so,
                    cov_t,
                    need,
                    last,
                    pg,
                    sl,
                    sl4,
                    sl128,
                    meta_bs_e if meta_bs_e is not None else "-",
                    meta_bs_o if meta_bs_o is not None else "-",
                )
            )
            key = (
                int(bool(for_cudagraph_capture)),
                int(num_reqs or 0),
                int(num_tokens or 0),
                len(g0_nz),
                tuple(g0_nz[:8]),
                len(attn_m),
                tuple(attn_m[:8]),
                hit7,
                hit10,
            )
            seen = getattr(self, "_dsv4_kv_bt_seen", None)
            if seen is None:
                self._dsv4_kv_bt_seen = seen = set()
            if key in seen:
                return
            seen.add(key)
            self._dsv4_kv_bt_n = nlog + 1
            logger.info(
                "[dsv4-kv] extra=runner-bt cap=%s dec_cg=%s nreq=%s ntok=%s "
                "seq0=%s ng=%d upd=%s g0_nnz=%d g0_ids=%s hit7=%d hit10=%d "
                "groups=%s meta_attn=%s meta_swa=%s g0slot=%s",
                int(bool(for_cudagraph_capture)),
                dec_cg,
                num_reqs,
                num_tokens,
                seq0,
                ng,
                upd,
                len(g0_nz),
                _preview(g0_nz),
                hit7,
                hit10,
                ";".join(groups_s) or "-",
                _preview(attn_m) if attn_m else "none",
                _preview(swa_m) if swa_m else "none",
                g0slot,
            )
        except Exception:
            pass

'''


def _patch_runner(src: str, applied: list) -> str:
    if "def _dsv4_kv_runner_bt_log" in src:
        if "cov_t=" in src and "g0slot=" in src:
            applied.append("runner-bt-fn (already)")
        else:
            rx = re.compile(
                r"    def _dsv4_kv_runner_bt_log\([\s\S]*?"
                r"(?=\n    def capture_model)",
            )
            new_src, n = rx.subn(RUNNER_BT_METHOD.rstrip("\n") + "\n", src, count=1)
            if n:
                src = new_src
                applied.append("runner-bt-fn (upgraded g0slot)")
            else:
                applied.append("runner-bt-fn (already)")
    else:
        inserted = False
        for cap in (
            "    def capture_model(self) -> int:\n",
            "    def capture_model(self) -> None:\n",
            "    def capture_model(self):\n",
        ):
            if cap in src:
                src = src.replace(cap, RUNNER_BT_METHOD + cap, 1)
                applied.append("runner-bt-fn")
                inserted = True
                break
        if not inserted:
            m = re.search(r"^    def capture_model\(self[^\n]*:\n", src, re.M)
            if m:
                src = src[: m.start()] + RUNNER_BT_METHOD + src[m.start() :]
                applied.append("runner-bt-fn")
            else:
                applied.append("runner-bt-fn (anchor missing, skipped)")

    if "# DSV4-KV-RUNNER-BT" in src:
        applied.append("runner-bt-call (already)")
        return src
    if "def _dsv4_kv_runner_bt_log" not in src:
        applied.append("runner-bt-call (no fn)")
        return src
    ret = "        return attn_metadata, spec_decode_common_attn_metadata\n"
    blob = (
        "        try:\n"
        "            self._dsv4_kv_runner_bt_log("
        "attn_metadata, num_reqs, num_tokens, "
        "for_cudagraph_capture)  # DSV4-KV-RUNNER-BT\n"
        "        except Exception:\n"
        "            pass\n"
        + ret
    )
    if ret not in src:
        applied.append("runner-bt-call (anchor missing, skipped)")
        return src
    src = src.replace(ret, blob, 1)
    applied.append("runner-bt-call")
    return src


CONN_HELPER = r'''
def _dsv4_kv_hash_on() -> bool:
    """DSV4_KV_HASH=1 logs prefill/decode KV page abs_sum + MLA metadata."""
    try:
        import os as _o

        return _o.environ.get("DSV4_KV_HASH", "0") == "1"
    except Exception:
        return False


def _dsv4_kv_role(layer_name):
    ln = layer_name or ""
    if ".indexer." in ln and ln.rstrip("/").endswith("k_cache"):
        return "indexer"
    if ".swa_cache" in ln:
        return "swa"
    if (
        ln.endswith(".attn")
        and "compressor" not in ln
        and ".indexer." not in ln
        and "swa_cache" not in ln
    ):
        return "attn"
    return None


def _dsv4_kv_ids_preview(ids, n=4):
    ids = [int(x) for x in list(ids or [])]
    if len(ids) <= n * 2:
        return ids
    return ids[:n] + ["..."] + ids[-n:]


def _dsv4_kv_sample_ids(ids, head=8, tail=4):
    ids = [int(x) for x in list(ids or [])]
    if not ids:
        return []
    if len(ids) <= head + tail:
        return ids
    out = ids[:head] + ids[-tail:]
    seen = set()
    uniq = []
    for i in out:
        if i in seen:
            continue
        seen.add(i)
        uniq.append(i)
    return uniq


def _dsv4_kv_bucket(n_ids):
    """s=curl (1 page), m=2k attn (~9), l=16k. 223113 16k src skipped as l."""
    n = int(n_ids or 0)
    if n <= 2:
        return "s"
    if n <= 16:
        return "m"
    return "l"


def _dsv4_kv_zip_nested(ids):
    try:
        return bool(ids) and isinstance(ids[0], (list, tuple))
    except Exception:
        return False


def _dsv4_kv_zip_flat(ids):
    if _dsv4_kv_zip_nested(ids):
        try:
            return [int(x) for x in list(ids[0] or [])]
        except Exception:
            return []
    try:
        return [int(x) for x in list(ids or [])]
    except Exception:
        return []


def _dsv4_kv_zip_log(self, layer_name, local_block_ids, remote_block_ids) -> None:
    """DSV4-KV-ZIP: dest ids actually handed to compute_block_transfer_offsets.

    After supports_hma pick/align. extra=zip-remote. 2k .attn remote must
    be g0 [7,9,…,8]; [10,11,12] is SWA dests (theory 1). Nested=1 means
    pick never flattened. Keyed (side, role, bucket) so curl cannot hide
    2k/16k. Does not hash dest ids against the src cache.
    """
    if not _dsv4_kv_hash_on():
        return
    role = _dsv4_kv_role(layer_name)
    if role is None:
        return
    try:
        loc_nested = _dsv4_kv_zip_nested(local_block_ids)
        rem_nested = _dsv4_kv_zip_nested(remote_block_ids)
        loc = _dsv4_kv_zip_flat(local_block_ids)
        rem = _dsv4_kv_zip_flat(remote_block_ids)
        n_groups = len(remote_block_ids) if rem_nested else 0
        bucket = _dsv4_kv_bucket(len(loc) or len(rem))
        side = "src" if getattr(self, "is_producer", True) else "dst"
        seen = getattr(self, "_dsv4_kv_zipped", None)
        if seen is None:
            self._dsv4_kv_zipped = seen = set()
        key = (side, role, bucket)
        if key in seen:
            return
        if len(seen) > 48:
            seen.clear()
        seen.add(key)
        self._dsv4_kv_last_zip = {
            "side": side,
            "role": role,
            "bucket": bucket,
            "local": loc,
            "remote": rem,
            "nested": rem_nested,
            "loc_nested": loc_nested,
            "n_groups": n_groups,
        }
        logger.info(
            "[dsv4-kv] extra=zip-remote side=%s role=%s bucket=%s layer=%s "
            "n_local=%d n_remote=%d nested=%d loc_nested=%d n_groups=%d "
            "local_ids=%s remote_ids=%s",
            side,
            role,
            bucket,
            layer_name,
            len(loc),
            len(rem),
            int(rem_nested),
            int(loc_nested),
            n_groups,
            _dsv4_kv_ids_preview(loc),
            _dsv4_kv_ids_preview(rem),
        )
    except Exception:
        pass


def _dsv4_kv_nblocks(cache):
    if cache is None:
        return 0
    try:
        return int(cache.shape[0])
    except Exception:
        pass
    try:
        return int(len(cache))
    except Exception:
        return 0


def _dsv4_kv_scan_ids(cache, lo=1, hi=16):
    """Worker dest ids never arrived (223113 store=0). 2k group-0 dests
    were [7,9,6,5,…,1,8] — all in 1..16 of the dest .attn tensor."""
    n = _dsv4_kv_nblocks(cache)
    if n <= 0:
        return []
    last = n - 1
    if last < lo:
        return [0] if n else []
    return list(range(lo, min(hi, last) + 1))


def _dsv4_kv_live_stats(cache, max_n=80):
    """(n_scan, n_live, max_abs) over pages 1..max_n. Caps GPU reads.
    80 covers 16k dest page 68 (223141 scan 1-16 / census 64 missed it)."""
    n = _dsv4_kv_nblocks(cache)
    if n <= 1:
        return 0, 0, 0.0
    hi = min(int(max_n), n - 1)
    n_scan = 0
    n_live = 0
    max_abs = 0.0
    for bid in range(1, hi + 1):
        n_scan += 1
        try:
            _n, hashed, abs_sum, _p = _dsv4_kv_page_stats(cache, [bid])
        except Exception:
            continue
        if hashed and abs_sum > 0:
            n_live += 1
            if abs_sum > max_abs:
                max_abs = abs_sum
    return n_scan, n_live, max_abs


def _dsv4_kv_pick_group(dests, role, layer_name, l2g):
    if not isinstance(dests, (list, tuple)) or not dests:
        return dests
    if not isinstance(dests[0], (list, tuple)):
        return dests
    gi = None
    if l2g:
        gi = l2g.get(layer_name)
    if gi is None:
        if role == "attn":
            gi = 0
        elif role == "swa":
            gi = 1 if len(dests) > 1 else 0
        elif role == "indexer":
            gi = 3 if len(dests) > 3 else 0
        else:
            gi = 0
    if gi < len(dests):
        return dests[gi]
    return dests[0]


def _dsv4_kv_page_stats(cache, ids):
    """Return (n_ids, n_hashed, abs_sum, prefix_hex). Never raises."""
    ids = [int(x) for x in list(ids or [])]
    n = len(ids)
    if cache is None or not ids:
        return n, 0, 0.0, ""
    sample = _dsv4_kv_sample_ids(ids)
    total = 0.0
    prefix = ""
    hashed = 0
    for i, bid in enumerate(sample):
        try:
            page = cache[bid]
        except Exception:
            continue
        try:
            if hasattr(page, "detach"):
                page = page.detach()
            if hasattr(page, "float"):
                val = page.float().abs().sum()
                total += float(val.item() if hasattr(val, "item") else val)
            elif hasattr(page, "__iter__") and not isinstance(page, (str, bytes)):
                total += sum(abs(float(x)) for x in page)
            else:
                total += abs(float(page))
            hashed += 1
        except Exception:
            continue
        if i == 0 and not prefix:
            try:
                raw = page
                if hasattr(raw, "detach"):
                    raw = raw.detach()
                if hasattr(raw, "view"):
                    _mod = type(raw).__module__ or ""
                    if _mod == "torch" or _mod.startswith("torch."):
                        import torch as _t

                        b = raw.view(_t.uint8).flatten()[:16]
                        if hasattr(b, "cpu"):
                            b = b.cpu()
                        prefix = bytes(int(x) for x in b.tolist()).hex()
                elif isinstance(raw, (bytes, bytearray)):
                    prefix = bytes(raw[:16]).hex()
            except Exception:
                prefix = ""
    return n, hashed, total, prefix


def _dsv4_kv_liveness(abs_sum):
    """z = empty dest (pre-RDMA); nz = bytes present. 223141 first load was z."""
    try:
        return "z" if float(abs_sum or 0) == 0 else "nz"
    except Exception:
        return "z"


def _dsv4_kv_as_groups(dests):
    if not dests:
        return None
    try:
        if isinstance(dests, dict):
            return None
        if isinstance(dests[0], (list, tuple)):
            return [list(g) for g in dests]
        return [list(dests)]
    except Exception:
        return None


def _dsv4_kv_as_int_list(obj):
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)):
        if not obj:
            return []
        if isinstance(obj[0], (list, tuple)):
            return [list(g) for g in obj]
        try:
            nums = [int(x) for x in obj]
        except Exception:
            return None
        while nums and nums[0] == 0:
            nums = nums[1:]
        while nums and nums[-1] == 0:
            nums = nums[:-1]
        return nums
    try:
        raw = obj
        if hasattr(raw, "detach"):
            raw = raw.detach()
        if hasattr(raw, "cpu"):
            raw = raw.cpu()
        if hasattr(raw, "flatten"):
            raw = raw.flatten()
        if hasattr(raw, "tolist"):
            nums = [int(x) for x in raw.tolist()]
        else:
            nums = [int(x) for x in list(raw)]
        while nums and nums[0] == 0:
            nums = nums[1:]
        while nums and nums[-1] == 0:
            nums = nums[:-1]
        return nums
    except Exception:
        return None


def _dsv4_kv_ids_from_forward(metadata):
    """ForwardContext / attn metadata dest pages. Skip slot_mapping (token slots)."""
    if metadata is None:
        return None
    for name in (
        "block_tables",
        "paged_kv_indices",
        "block_ids",
        "local_block_ids",
        "new_block_ids",
    ):
        raw = getattr(metadata, name, None)
        if raw is None and isinstance(metadata, dict):
            raw = metadata.get(name)
        ids = _dsv4_kv_as_int_list(raw)
        if ids:
            return ids
    layers = getattr(metadata, "no_compile_layers", None)
    if isinstance(layers, dict):
        for _lname, layer in layers.items():
            meta = getattr(layer, "attn_metadata", None) or layer
            ids = _dsv4_kv_ids_from_forward(meta)
            if ids:
                return ids
    attn = getattr(metadata, "attn_metadata", None)
    if attn is not None and attn is not metadata:
        return _dsv4_kv_ids_from_forward(attn)
    return None


def _dsv4_kv_dest_dir():
    try:
        import os as _o

        return _o.environ.get("DSV4_KV_DEST_DIR", "/tmp/dsv4-kv-dest")
    except Exception:
        return "/tmp/dsv4-kv-dest"


def _dsv4_kv_dest_write(rid, groups) -> None:
    """EngineCore → Worker: dest ids never ride WRITE-mode connector metadata."""
    try:
        import json
        import os as _o

        d = _dsv4_kv_dest_dir()
        _o.makedirs(d, exist_ok=True)
        safe = "".join(
            c if (c.isalnum() or c in "-_.") else "_" for c in str(rid or "unknown")
        )[:120]
        payload = {"rid": str(rid), "groups": groups}
        open(_o.path.join(d, safe + ".json"), "w").write(json.dumps(payload))
        names = [n for n in _o.listdir(d) if n.endswith(".json")]
        if len(names) > 48:
            paths = [_o.path.join(d, n) for n in names]
            paths.sort(key=lambda p: _o.path.getmtime(p))
            for p in paths[:-24]:
                try:
                    _o.remove(p)
                except Exception:
                    pass
    except Exception:
        pass


def _dsv4_kv_dest_read():
    out = []
    try:
        import json
        import os as _o

        d = _dsv4_kv_dest_dir()
        if not _o.path.isdir(d):
            return out
        for name in sorted(_o.listdir(d)):
            if not name.endswith(".json"):
                continue
            try:
                payload = json.loads(open(_o.path.join(d, name)).read())
            except Exception:
                continue
            if isinstance(payload, dict):
                groups = payload.get("groups")
                rid = payload.get("rid")
            else:
                groups, rid = payload, None
            if groups:
                out.append((rid, groups))
    except Exception:
        pass
    return out


def _dsv4_kv_attach_dest_meta(self, meta) -> None:
    """Copy EngineCore dest groups onto scheduler metadata for the Worker."""
    if not _dsv4_kv_hash_on() or meta is None:
        return
    try:
        store = getattr(self, "_dsv4_kv_dest", None) or {}
        payload = {}
        for k, v in list(store.items()):
            if str(k).startswith("_") or not v:
                continue
            payload[str(k)] = v
        if not payload:
            return
        try:
            setattr(meta, "dsv4_dest", payload)
        except Exception:
            pass
        reqs = (
            getattr(meta, "requests", None)
            or getattr(meta, "req_meta", None)
            or getattr(meta, "reqs", None)
        )
        if isinstance(reqs, dict):
            items = list(reqs.values())
        elif reqs:
            items = list(reqs)
        else:
            items = []
        for r in items:
            params = (
                r.get("kv_transfer_params")
                if isinstance(r, dict)
                else getattr(r, "kv_transfer_params", None)
            )
            if isinstance(params, dict) and "dsv4_dest_groups" not in params:
                rid = (
                    r.get("request_id")
                    if isinstance(r, dict)
                    else getattr(r, "request_id", None)
                )
                params["dsv4_dest_groups"] = payload.get(str(rid)) or next(
                    iter(payload.values())
                )
    except Exception:
        pass


def _dsv4_kv_hash_dest_groups(self, dests, extra="ids") -> None:
    """Hash each HMA group's dest ids against Worker .attn (223141 scatter)."""
    caches = getattr(self, "kv_caches", None) or {}
    attn_layer = None
    for layer_name in caches:
        if _dsv4_kv_role(layer_name) == "attn":
            attn_layer = layer_name
            break
    if attn_layer is None:
        return
    for gi, ids in enumerate(list(dests)[:5]):
        ids = list(ids or [])
        _dsv4_kv_hash_log(
            self, "dst", attn_layer, ids, extra="%s-g%d" % (extra, gi)
        )
        nz = [int(x) for x in ids if int(x) != 0]
        if nz and nz != [int(x) for x in ids]:
            _dsv4_kv_hash_log(
                self, "dst", attn_layer, nz, extra="%s-g%dnz" % (extra, gi)
            )


def _dsv4_kv_hash_log(self, side, layer_name, block_ids, extra=None) -> None:
    if not _dsv4_kv_hash_on():
        return
    role = _dsv4_kv_role(layer_name)
    if role is None:
        return
    try:
        n_ids = len(list(block_ids or []))
        # 222578: (side, role) once per process burned on curl n_pages=1.
        # 223113: 2k and 16k both used bucket=l; 16k src never logged.
        # DSV4-KV-DEST-IDS: dest key includes extra + z|nz so curl scan
        # (always n=16 / bucket=m) cannot hide 2k (m, ids-g0) or 16k (l).
        bucket = _dsv4_kv_bucket(n_ids)
        seen = getattr(self, "_dsv4_kv_logged", None)
        if seen is None:
            self._dsv4_kv_logged = seen = set()
        if side != "dst":
            key = (side, role, bucket, extra)
            if key in seen:
                return
        elif (side, role, extra, bucket, "nz") in seen:
            return
        caches = getattr(self, "kv_caches", None) or {}
        cache = caches.get(layer_name) if hasattr(caches, "get") else None
        n, hashed, abs_sum, prefix = _dsv4_kv_page_stats(cache, block_ids)
        live = _dsv4_kv_liveness(abs_sum)
        if side == "dst":
            key = (side, role, extra, bucket, live)
            if key in seen:
                return
        if len(seen) > 96:
            seen.clear()
        seen.add(key)
        logger.info(
            "[dsv4-kv] side=%s role=%s bucket=%s live=%s layer=%s n_pages=%d "
            "hashed=%d abs_sum=%.4g prefix=%s ids=%s extra=%s",
            side,
            role,
            bucket,
            live,
            layer_name,
            n,
            hashed,
            abs_sum,
            prefix or "-",
            _dsv4_kv_ids_preview(block_ids),
            extra,
        )
    except Exception:
        pass


def _dsv4_kv_stash_dests(self, request, blocks) -> None:
    """Consumer: remember per-group dest ids so recving-done can hash them."""
    if not _dsv4_kv_hash_on():
        return
    try:
        if getattr(self, "is_producer", False):
            return
        rid = getattr(request, "request_id", None)
        if rid is None:
            return
        try:
            raw = blocks.get_block_ids() if blocks is not None else None
        except Exception:
            raw = None
        if not raw:
            return
        if isinstance(raw[0], (list, tuple)):
            groups = [list(g) for g in raw]
        else:
            groups = [list(raw)]
        store = getattr(self, "_dsv4_kv_dest", None)
        if store is None:
            self._dsv4_kv_dest = store = {}
        if len(store) > 256:
            store.clear()
        store[rid] = groups
        params = getattr(request, "kv_transfer_params", None)
        if not isinstance(params, dict):
            params = {}
        params["dsv4_dest_groups"] = groups
        try:
            request.kv_transfer_params = params
        except Exception:
            pass
        tx = params.get("transfer_id")
        if tx:
            store[tx] = groups
        _dsv4_kv_dest_write(rid, groups)
    except Exception:
        pass


def _dsv4_kv_ids_from(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        for name in (
            "block_ids",
            "remote_block_ids",
            "local_block_ids",
            "new_block_ids",
            "block_ids_to_load",
            "dsv4_dest_groups",
        ):
            ids = obj.get(name)
            if ids:
                return ids
        params = obj.get("kv_transfer_params")
        if isinstance(params, dict) and params.get("dsv4_dest_groups"):
            return params.get("dsv4_dest_groups")
        inner = obj.get("load_spec") or obj.get("spec")
        if inner is not None and inner is not obj:
            return _dsv4_kv_ids_from(inner)
        return None
    params = getattr(obj, "kv_transfer_params", None)
    if isinstance(params, dict) and params.get("dsv4_dest_groups"):
        return params.get("dsv4_dest_groups")
    for name in (
        "block_ids",
        "remote_block_ids",
        "local_block_ids",
        "new_block_ids",
        "block_ids_to_load",
        "dsv4_dest_groups",
    ):
        ids = getattr(obj, name, None)
        if ids:
            return ids
    spec = getattr(obj, "load_spec", None) or getattr(obj, "spec", None)
    if spec is not None and spec is not obj:
        return _dsv4_kv_ids_from(spec)
    return None


def _dsv4_kv_stash_load_meta(self, metadata) -> None:
    """Worker start_load_kv: dest ids live on connector metadata.

    223113: the bound arg is ForwardContext (no requests). Dest ids are on
    ``_get_connector_metadata()`` in the Worker, not the scheduler stash.
    223141 WRITE: unwrap is empty; keep ForwardContext, dest-dir JSON, and
    ``meta.dsv4_dest`` so 2k/16k dest pages still hash.
    """
    if not _dsv4_kv_hash_on():
        return
    try:
        fwd = metadata
        try:
            got = self._get_connector_metadata()
            if got is not None:
                metadata = got
        except Exception:
            pass
        store = getattr(self, "_dsv4_kv_dest", None)
        if store is None:
            self._dsv4_kv_dest = store = {}
        sched = getattr(self, "connector_scheduler", None)
        if sched is not None:
            other = getattr(sched, "_dsv4_kv_dest", None)
            if other:
                store.update(other)
        dsv4 = getattr(metadata, "dsv4_dest", None)
        if isinstance(dsv4, dict) and dsv4:
            for k, v in dsv4.items():
                if v:
                    store[k] = v
            store["_last"] = next(iter(dsv4.values()))
        fwd_ids = _dsv4_kv_ids_from_forward(fwd)
        if fwd_ids:
            groups = _dsv4_kv_as_groups(fwd_ids)
            if groups:
                store["_fwd"] = groups
                if "_last" not in store:
                    store["_last"] = groups
        for rid, groups in _dsv4_kv_dest_read():
            if groups:
                if rid:
                    store[rid] = groups
                store["_last"] = groups
        reqs = None
        if metadata is not None:
            reqs = (
                getattr(metadata, "requests", None)
                or getattr(metadata, "req_meta", None)
                or getattr(metadata, "reqs", None)
                or getattr(metadata, "request_info", None)
            )
        if not reqs and isinstance(metadata, dict):
            reqs = metadata
        if reqs:
            if isinstance(reqs, dict):
                items = list(reqs.items())
            else:
                items = []
                for r in reqs:
                    rid = (
                        getattr(r, "request_id", None)
                        or getattr(r, "req_id", None)
                        or (r.get("request_id") if isinstance(r, dict) else None)
                    )
                    items.append((rid, r))
            for rid, r in items:
                ids = _dsv4_kv_ids_from(r)
                if not ids:
                    continue
                if rid is not None:
                    store[rid] = ids
                store["_last"] = ids
        if len(store) > 256:
            keep = {k: store[k] for k in list(store)[-64:]}
            store.clear()
            store.update(keep)
    except Exception:
        pass


def _dsv4_kv_pair_seal(self, extra="pair") -> None:
    """DSV4-KV-PAIR-LIVE: dest g0 census vs dest SWA census on the same tensor.

    223210 zip remote was g0 [7,9,…,8] (correct ids) but .attn[10,11,12]
    got src-attn prefix 7d5a71f3. load-g0 was pre-rdma_wait leftover.
    223327 keyed (extra, bucket, id0) so the empty/leftover snapshot ate
    the only slot; post-RDMA dest[7] vs dest[10] never logged. Key on
    content (prefix/live of g0[0], g0[-1], every SWA dest). Log all g0
    pages (cap 12) and all SWA dests (cap 4), not only 7 vs 10.
    """
    if not _dsv4_kv_hash_on():
        return
    if getattr(self, "is_producer", False):
        return
    try:
        caches = getattr(self, "kv_caches", None) or {}
        attn_layer = None
        for layer_name in caches:
            if _dsv4_kv_role(layer_name) == "attn":
                attn_layer = layer_name
                break
        if attn_layer is None:
            return
        cache = caches.get(attn_layer) if hasattr(caches, "get") else None
        store = getattr(self, "_dsv4_kv_dest", None) or {}
        groups = None
        for _k, v in list(store.items()):
            g = _dsv4_kv_as_groups(v)
            if not g or not g[0]:
                continue
            groups = g
            if len(g[0]) >= 9:
                break
        if not groups:
            return
        g0 = [int(x) for x in (groups[0] or []) if int(x) != 0]
        if not g0:
            return
        g1nz = []
        if len(groups) > 1:
            g1nz = [int(x) for x in (groups[1] or []) if int(x) != 0]
        if not g1nz:
            g1nz = [10]

        def _one(bid):
            _n, _h, ab, pr = _dsv4_kv_page_stats(cache, [int(bid)])
            live = "nz" if ab > 0 else "z"
            return int(bid), ab, pr or "", live

        g0_stats = [_one(i) for i in g0[:12]]
        swa_stats = [_one(i) for i in g1nz[:4]]
        id0, abs0, p0, live0 = g0_stats[0]
        idt, abs_t, pt, live_t = g0_stats[-1]
        swa_id, abs_s, ps, live_s = swa_stats[0]
        bucket = _dsv4_kv_bucket(len(g0))
        seen = getattr(self, "_dsv4_kv_paired", None)
        if seen is None:
            self._dsv4_kv_paired = seen = set()
        key = (
            str(extra),
            bucket,
            id0,
            live0,
            live_t,
            live_s,
            p0[:16],
            pt[:16],
            ps[:16],
            tuple((b, lv) for b, _a, _p, lv in swa_stats),
        )
        nlog = int(getattr(self, "_dsv4_kv_pair_n", 0) or 0)
        if key in seen or nlog >= 96:
            return
        seen.add(key)
        self._dsv4_kv_pair_n = nlog + 1
        same = bool(p0) and p0 == ps

        def _fmt(stats):
            out = []
            for bid, ab, pr, live in stats:
                tag = (pr[:8] if pr else "-")
                out.append("%d:%s:%s" % (bid, live, tag))
            return ",".join(out)

        g0_s = _fmt(g0_stats)
        swa_s = _fmt(swa_stats)
        self._dsv4_kv_last_pair = {
            "extra": extra,
            "bucket": bucket,
            "id0": id0,
            "abs0": abs0,
            "prefix0": p0,
            "idt": idt,
            "absT": abs_t,
            "prefixT": pt,
            "swa0": swa_id,
            "absS": abs_s,
            "prefixS": ps,
            "same_prefix": same,
            "g0": g0_s,
            "swa": swa_s,
            "n_g0": len(g0_stats),
            "n_swa": len(swa_stats),
        }
        logger.info(
            "[dsv4-kv] extra=pair-%s bucket=%s layer=%s "
            "id0=%d abs0=%.4g prefix0=%s "
            "idT=%d absT=%.4g prefixT=%s "
            "swa0=%d absS=%.4g prefixS=%s same_prefix=%s "
            "n_g0=%d g0=%s n_swa=%d swa=%s",
            extra,
            bucket,
            attn_layer,
            id0,
            abs0,
            p0 or "-",
            idt,
            abs_t,
            pt or "-",
            swa_id,
            abs_s,
            ps or "-",
            int(same),
            len(g0_stats),
            g0_s,
            len(swa_stats),
            swa_s,
        )
    except Exception:
        pass


def _dsv4_kv_hash_dests(self, done_req_ids=None, extra="dst") -> None:
    """Hash dest pages. extra=load is start_load_kv (may be pre-RDMA).
    extra=post-wait is wait_for_layer_load after WAITING_FOR_REMOTE.
    WRITE-mode never hits recving/_pop_done_transfers (222968).
    223141: dest ids via dest-dir / dsv4_dest / ForwardContext; scan is
    fallback only (curl must not cap 2k/16k).
    223210: 2nd start_load of the same g0 fingerprint is extra=post-seal
    so curl leftover cannot hide dest[7] vs dest[10] after rdma_wait."""
    if not _dsv4_kv_hash_on():
        return
    try:
        if getattr(self, "is_producer", False):
            return
        caches = getattr(self, "kv_caches", None) or {}
        if not caches:
            return
        store = getattr(self, "_dsv4_kv_dest", None) or {}
        l2g = getattr(self, "_dsv4_layer_to_group", None) or {}
        ids_iter = list(done_req_ids or [])[:4]
        candidates = []
        fps = set()

        def _add(dests):
            groups = _dsv4_kv_as_groups(dests)
            if not groups:
                return
            g0 = [int(x) for x in (groups[0] or [])]
            fp = (len(g0), tuple(g0[:4]), tuple(g0[-2:]))
            if fp in fps:
                return
            fps.add(fp)
            candidates.append(groups)

        for rid in ids_iter:
            _add(store.get(rid))
        for k, v in list(store.items()):
            _add(v)
        if not candidates:
            nmiss = int(getattr(self, "_dsv4_kv_miss", 0) or 0)
            if nmiss < 8:
                logger.info(
                    "[dsv4-kv] side=dst dests-missing extra=%s recving=%s "
                    "store=%d — scanning pages 1-16",
                    extra,
                    ids_iter,
                    len(store),
                )
            self._dsv4_kv_miss = nmiss + 1
            _dsv4_kv_hash_dest_scan(self, caches, extra=extra)
            return
        extra_use = extra
        extra_by = []
        if extra == "load":
            fps_seen = getattr(self, "_dsv4_kv_load_fps", None)
            if fps_seen is None:
                self._dsv4_kv_load_fps = fps_seen = set()
        else:
            fps_seen = None
        for groups in candidates:
            extra_i = extra
            if fps_seen is not None:
                g0 = [int(x) for x in (groups[0] or [])]
                fp = (len(g0), tuple(g0[:4]), tuple(g0[-2:]))
                if fp in fps_seen:
                    extra_i = "post-seal"
                fps_seen.add(fp)
            extra_by.append((groups, extra_i))
            extra_use = extra_i
        for groups, extra_i in extra_by:
            dests = groups
            extra_use = extra_i
            if len(groups) > 1:
                _dsv4_kv_hash_dest_groups(self, groups, extra=extra_use)
            seen_role = set()
            for layer_name in caches:
                role = _dsv4_kv_role(layer_name)
                if role is None or role in seen_role:
                    continue
                if len(groups) > 1 and role == "attn":
                    continue
                seen_role.add(role)
                ids = _dsv4_kv_pick_group(dests, role, layer_name, l2g)
                _dsv4_kv_hash_log(self, "dst", layer_name, ids, extra=extra_use)
                if role == "attn":
                    _dsv4_kv_hash_log(
                        self, "dst", layer_name, [0], extra="%s+p0" % extra_use
                    )
            _dsv4_kv_pair_seal(self, extra=extra_use)
    except Exception:
        pass


def _dsv4_kv_hash_dest_scan(self, caches, extra="scan") -> None:
    """Hash dest .attn / swa / indexer without dest ids (Worker store=0)."""
    extra_scan = "%s+scan" % extra
    extra_live = "%s+live" % extra
    extra_p0 = "%s+p0" % extra
    seen_role = set()
    live_done = getattr(self, "_dsv4_kv_live_logged", None)
    if live_done is None:
        self._dsv4_kv_live_logged = live_done = set()
    for layer_name in caches:
        role = _dsv4_kv_role(layer_name)
        if role is None or role in seen_role:
            continue
        seen_role.add(role)
        cache = caches.get(layer_name) if hasattr(caches, "get") else None
        ids = _dsv4_kv_scan_ids(cache)
        if ids:
            _dsv4_kv_hash_log(
                self, "dst", layer_name, ids, extra=extra_scan
            )
        _dsv4_kv_hash_log(self, "dst", layer_name, [0], extra=extra_p0)
        if role == "attn" and extra_live not in live_done:
            live_done.add(extra_live)
            n_scan, n_live, max_abs = _dsv4_kv_live_stats(cache)
            logger.info(
                "[dsv4-kv] side=dst role=%s extra=%s layer=%s n_scan=%d "
                "n_live=%d max_abs=%.4g nblocks=%d",
                role,
                extra_live,
                layer_name,
                n_scan,
                n_live,
                max_abs,
                _dsv4_kv_nblocks(cache),
            )


def _dsv4_kv_hash_on_recving(self, done_req_ids) -> None:
    """PULL-mode: hash dest once KV recving reports complete."""
    _dsv4_kv_hash_dests(self, done_req_ids, extra="recving")
'''

MLA_KV_HASH_ON = r'''
def _dsv4_kv_hash_on() -> bool:
    try:
        import os as _o

        return _o.environ.get("DSV4_KV_HASH", "0") == "1"
    except Exception:
        return False
'''

MLA_META_FN = r'''
def _dsv4_mla_meta_log(
    metadata_key=None,
    attn_meta=None,
    paged_kv_indices=None,
    kv_cache=None,
    where="",
    swa_meta=None,
    cr=None,
    swa_only=None,
    topk_buf=None,
) -> None:
    """DSV4-MLA-FWD-LIVE / DSV4-MLA-CR: dump block_table + dest census.

    223514 2k: block_size=256 kvd1=64 n_nz=9 bt=[7,9,…] dest NZ. Do not
    set block_size=kvd1 (d626 already does 256//compress_ratio). Log cr=
    kbs=block_size//cr swa_only= ntopk= tmax= (max >=0 indexer id).
    Key (where, n_nz, imax, bt[:8], kvd1). Cap 96.
    """
    if not _dsv4_kv_hash_on():
        return
    try:
        import logging as _lg

        _log = _lg.getLogger("vllm")

        def _as_list(v):
            if v is None:
                return None
            try:
                if hasattr(v, "detach"):
                    v = v.detach()
                if hasattr(v, "flatten") and hasattr(v, "shape") and getattr(v, "ndim", 1) > 1:
                    v = v.flatten()
                if hasattr(v, "cpu"):
                    v = v.cpu()
                if hasattr(v, "tolist"):
                    v = v.tolist()
            except Exception:
                return None
            if isinstance(v, (list, tuple)):
                return list(v)
            try:
                return [int(v)]
            except Exception:
                return None

        def _seq_from(obj):
            if obj is None:
                return None, 0
            seq = None
            for name in (
                "seq_lens_cpu",
                "seq_lens",
                "context_lens_cpu",
                "context_lens",
                "prefill_seq_lens",
            ):
                seq = _as_list(getattr(obj, name, None))
                if seq:
                    break
            seq0 = 0
            if seq:
                try:
                    seq0 = int(seq[0])
                except Exception:
                    seq0 = 0
            if seq0 <= 0:
                for name in ("max_seq_len", "max_seqlen_k", "max_context_len"):
                    try:
                        mv = getattr(obj, name, None)
                        if mv is None:
                            continue
                        seq0 = int(mv.item() if hasattr(mv, "item") else mv)
                        if seq0 > 0:
                            break
                    except Exception:
                        pass
            if seq is not None and len(seq) > 8:
                seq = seq[:8] + ["..."]
            return seq, seq0

        def _scalar(obj, names):
            if obj is None:
                return None
            for name in names:
                v = getattr(obj, name, None)
                if v is None:
                    continue
                try:
                    return int(v.item() if hasattr(v, "item") else v)
                except Exception:
                    return v
            return None

        qlen = None
        bsz = None
        if attn_meta is not None:
            qlen = getattr(attn_meta, "num_actual_tokens", None)
            if qlen is None:
                qsl = getattr(attn_meta, "query_start_loc", None)
                if qsl is not None:
                    try:
                        qlen = int(qsl[-1]) if len(qsl) else None
                    except Exception:
                        qlen = None
            bsz = getattr(attn_meta, "block_size", None)
        seq, seq0 = _seq_from(attn_meta)
        if seq0 <= 0:
            seq_s, seq0s = _seq_from(swa_meta)
            if seq0s > 0:
                seq, seq0 = seq_s, seq0s
        ndec = _scalar(
            swa_meta,
            ("num_decode_tokens", "num_decodes", "num_decode_seqs"),
        )
        if ndec is None:
            ndec = _scalar(
                attn_meta,
                ("num_decode_tokens", "num_decodes", "num_decode_seqs"),
            )
        zf = None
        imin = None
        imax = None
        nidx = 0
        n_nz = 0
        nums = []
        idx = paged_kv_indices
        if idx is None and attn_meta is not None:
            idx = getattr(attn_meta, "paged_kv_indices", None)
            if idx is None:
                idx = getattr(attn_meta, "block_table", None)
            if idx is None:
                idx = getattr(attn_meta, "block_tables", None)
        if idx is not None:
            try:
                flat = idx
                if hasattr(flat, "detach"):
                    flat = flat.detach()
                if hasattr(flat, "flatten"):
                    flat = flat.flatten()
                if hasattr(flat, "cpu"):
                    flat = flat.cpu()
                if hasattr(flat, "tolist"):
                    nums = [int(x) for x in flat.tolist()]
                else:
                    nums = [int(x) for x in list(flat)]
                nidx = len(nums)
                if nums:
                    imin = min(nums)
                    imax = max(nums)
                    n_nz = sum(1 for x in nums if x != 0)
                    zf = (nidx - n_nz) / float(nidx)
            except Exception:
                nums = []

        def _page(bid):
            if kv_cache is None:
                return 0.0, ""
            try:
                page = kv_cache[int(bid)]
                tot = 0.0
                if hasattr(page, "detach"):
                    page = page.detach()
                if hasattr(page, "float"):
                    val = page.float().abs().sum()
                    tot = float(val.item() if hasattr(val, "item") else val)
                prefix = ""
                try:
                    raw = page
                    mod = type(raw).__module__ or ""
                    if hasattr(raw, "view") and (
                        mod == "torch" or mod.startswith("torch.")
                    ):
                        import torch as _t

                        b = raw.view(_t.uint8).flatten()[:16]
                        if hasattr(b, "cpu"):
                            b = b.cpu()
                        prefix = bytes(int(x) for x in b.tolist()).hex()
                    elif isinstance(raw, (bytes, bytearray)):
                        prefix = bytes(raw[:16]).hex()
                except Exception:
                    prefix = ""
                return tot, prefix
            except Exception:
                return 0.0, ""

        nz = []
        for x in nums:
            if x and x not in nz:
                nz.append(x)
            if len(nz) >= 8:
                break
        want = []
        for bid in (7, 8, 10, 11, 12):
            if bid not in want:
                want.append(bid)
        for bid in nz:
            if bid not in want:
                want.append(bid)
            if len(want) >= 12:
                break
        page_stats = []
        for bid in want:
            ab, pr = _page(bid)
            live = "nz" if ab > 0 else "z"
            page_stats.append((bid, ab, pr or "", live))
        by_id = {b: (ab, pr, lv) for b, ab, pr, lv in page_stats}
        a7, p7, live7s = by_id.get(7, (0.0, "", "z"))
        a10, p10, live10s = by_id.get(10, (0.0, "", "z"))
        live7 = int(live7s == "nz")
        live10 = int(live10s == "nz")
        live_i = int(imax is not None and int(imax) > 0)
        pages_s = ",".join(
            "%d:%s:%s" % (b, lv, (pr[:8] if pr else "-"))
            for b, _a, pr, lv in page_stats
        )
        kvd1 = None
        try:
            shp = getattr(kv_cache, "shape", None)
            if shp is not None and len(shp) >= 2:
                kvd1 = int(shp[1])
        except Exception:
            kvd1 = None
        spec_bs = 256
        try:
            if bsz:
                spec_bs = int(bsz)
        except Exception:
            spec_bs = 256
        cov_t = (int(n_nz) * spec_bs) if n_nz else 0
        kbs = None
        ntopk = None
        tmax = None
        try:
            if cr is not None and bsz:
                kbs = int(bsz) // int(cr)
        except Exception:
            kbs = None
        try:
            cr_i = int(cr) if cr is not None else None
        except Exception:
            cr_i = None

        def _tmax_row(row):
            if row is None:
                return None, None
            if hasattr(row, "ndim") and getattr(row, "ndim", 1) >= 2:
                row = row[0]
            lst = _as_list(row)
            if not lst:
                return None, None
            pos = []
            for x in lst:
                try:
                    xi = int(x)
                except Exception:
                    continue
                if xi >= 0:
                    pos.append(xi)
            if not pos:
                return 0, None
            return len(pos), max(pos)

        try:
            if cr_i == 128 and attn_meta is not None:
                tlens = getattr(attn_meta, "c128a_decode_topk_lens", None)
                lst = _as_list(tlens) if tlens is not None else None
                if lst:
                    ntopk = int(lst[0])
                tmax = _tmax_row(
                    getattr(attn_meta, "c128a_global_decode_topk_indices", None)
                )[1]
            else:
                ntopk, tmax = _tmax_row(topk_buf)
                if ntopk is None and attn_meta is not None:
                    tlens = getattr(attn_meta, "c128a_decode_topk_lens", None)
                    lst = _as_list(tlens) if tlens is not None else None
                    if lst:
                        ntopk = int(lst[0])
                    if tmax is None:
                        tmax = _tmax_row(
                            getattr(
                                attn_meta, "c128a_global_decode_topk_indices", None
                            )
                        )[1]
        except Exception:
            ntopk = None
            tmax = None
        # 223449: seq_lens=None, seq0=max_seqlen_k (17) hid 2k as sb=s.
        if seq:
            sb = "s" if seq0 <= 64 else ("m" if seq0 <= 4096 else "l")
        elif n_nz <= 0:
            sb = "u"
        elif n_nz <= 2:
            sb = "s"
        elif n_nz <= 16:
            sb = "m"
        else:
            sb = "l"
        bt_head = nums[:8] if nums else []
        bt_tail = nums[-8:] if nums else []
        seen = getattr(_dsv4_mla_meta_log, "_seen", None)
        if seen is None:
            _dsv4_mla_meta_log._seen = seen = set()
        key = (
            str(where or ""),
            int(n_nz),
            int(imax or 0),
            tuple(bt_head),
            kvd1 if kvd1 is not None else -1,
        )
        n = int(getattr(_dsv4_mla_meta_log, "_n", 0) or 0)
        if key in seen or n >= 96:
            return
        seen.add(key)
        _dsv4_mla_meta_log._n = n + 1
        _log.info(
            "[dsv4-mla-meta] where=%s sb=%s key=%r seq_lens=%s seq0=%d "
            "qlen=%s ndec=%s block_size=%s kvd1=%s cov_t=%d n_idx=%d "
            "n_nz=%d min=%s max=%s cr=%s kbs=%s swa_only=%s ntopk=%s tmax=%s "
            "zero_frac=%s bt=%s bt_tail=%s nz=%s pages=%s "
            "live7=%d abs7=%.4g prefix7=%s live10=%d abs10=%.4g prefix10=%s",
            where,
            sb,
            metadata_key,
            seq,
            seq0,
            qlen,
            ndec,
            bsz,
            kvd1 if kvd1 is not None else "-",
            cov_t,
            nidx,
            n_nz,
            imin,
            imax,
            cr if cr is not None else "-",
            kbs if kbs is not None else "-",
            swa_only if swa_only is not None else "-",
            ntopk if ntopk is not None else "-",
            tmax if tmax is not None else "-",
            ("%.3f" % zf) if zf is not None else None,
            bt_head,
            bt_tail,
            nz,
            pages_s,
            live7,
            a7,
            p7 or "-",
            live10,
            a10,
            p10 or "-",
        )
    except Exception:
        pass
'''

MLA_HELPER = MLA_KV_HASH_ON + "\n" + MLA_META_FN



def _insert_before(src: str, anchor: str, blob: str) -> str:
    idx = src.find(anchor)
    if idx < 0:
        raise ValueError(f"anchor not found: {anchor!r}")
    return src[:idx] + blob.lstrip("\n") + "\n\n" + src[idx:]


OFFSETS_OLD = """        return compute_block_transfer_offsets(
            layer_name=layer_name,
            kv_cache=self.kv_caches[layer_name],
"""

OFFSETS_NEW = """        _dsv4_kv_hash_log(self, "src", layer_name, local_block_ids)
        _dsv4_kv_zip_log(
            self, layer_name, local_block_ids, remote_block_ids
        )  # DSV4-KV-ZIP
        if not getattr(self, "is_producer", True):
            _dsv4_kv_hash_log(
                self, "dst", layer_name, local_block_ids, extra="offsets"
            )
            try:
                _st = getattr(self, "_dsv4_kv_dest", None)
                if _st is None:
                    self._dsv4_kv_dest = _st = {}
                _st["_offsets"] = list(local_block_ids or [])
            except Exception:
                pass
        return compute_block_transfer_offsets(
            layer_name=layer_name,
            kv_cache=self.kv_caches[layer_name],
"""

_ALLOC_STASH = (
    "        _dsv4_kv_stash_dests(self, request, blocks)\n"
)


def _ensure_conn_helper(src: str, applied: list) -> str:
    """Insert or upgrade connector helpers. Fresh image each job; also
    upgrade a tree that already has the 222968 stash-only helper."""
    cls = src.find("class MoRIIOConnectorScheduler")
    if cls < 0:
        raise ValueError("class MoRIIOConnectorScheduler missing")
    have = "def _dsv4_kv_hash_log" in src[:cls]
    need_upgrade = have and (
        "def _dsv4_kv_hash_dests" not in src[:cls]
        or "def _dsv4_kv_scan_ids" not in src[:cls]
        or "DSV4-KV-DEST-IDS" not in src[:cls]
        or "def _dsv4_kv_attach_dest_meta" not in src[:cls]
        or "def _dsv4_kv_ids_from_forward" not in src[:cls]
        or "def _dsv4_kv_zip_log" not in src[:cls]
        or "def _dsv4_kv_pair_seal" not in src[:cls]
        or "DSV4-KV-PAIR-LIVE" not in src[:cls]
    )
    blob = CONN_HELPER.strip() + "\n\n"
    if not have:
        src = src[:cls] + blob + src[cls:]
        applied.append("conn-helper")
        return src
    if need_upgrade:
        h = src.rfind("def _dsv4_kv_hash_on()", 0, cls)
        if h < 0:
            applied.append("conn-helper (already)")
            return src
        src = src[:h] + blob + src[cls:]
        applied.append("conn-helper (upgraded)")
        return src
    applied.append("conn-helper (already)")
    return src


def _patch_connector(src: str, applied: list) -> str:
    src = _ensure_conn_helper(src, applied)

    if "_dsv4_kv_hash_log(self, \"src\", layer_name, local_block_ids)" in src:
        applied.append("conn-src (already)")
    elif OFFSETS_OLD not in src:
        raise ValueError(
            "compute_block_transfer_offsets kv_cache= self.kv_caches "
            "anchor missing"
        )
    else:
        src = src.replace(OFFSETS_OLD, OFFSETS_NEW, 1)
        applied.append("conn-src")

    zip_call = (
        "_dsv4_kv_zip_log(\n            self, layer_name, "
        "local_block_ids, remote_block_ids\n        )"
    )
    zip_call_one = (
        "_dsv4_kv_zip_log(self, layer_name, local_block_ids, "
        "remote_block_ids)"
    )
    if "DSV4-KV-ZIP" in src or zip_call in src or zip_call_one in src:
        applied.append("conn-zip (already)")
    elif '_dsv4_kv_hash_log(self, "src", layer_name, local_block_ids)' in src:
        src = src.replace(
            '_dsv4_kv_hash_log(self, "src", layer_name, local_block_ids)',
            '_dsv4_kv_hash_log(self, "src", layer_name, local_block_ids)\n'
            "        _dsv4_kv_zip_log(\n"
            "            self, layer_name, local_block_ids, "
            "remote_block_ids\n"
            "        )  # DSV4-KV-ZIP",
            1,
        )
        applied.append("conn-zip")
    else:
        applied.append("conn-zip (no src hook)")

    if _ALLOC_STASH in src:
        applied.append("conn-stash (already)")
    else:
        old = "        self.map_request_id(request_id, transfer_id)\n"
        if src.count(old) >= 1:
            src = src.replace(
                old,
                old
                + "        # DSV4-KV-HASH: stash consumer dest ids for recving hash.\n"
                + _ALLOC_STASH,
                1,
            )
            applied.append("conn-stash")
        else:
            applied.append("conn-stash (anchor missing, skipped)")

    if "# DSV4-KV-PAIR-ALLOC0" in src:
        applied.append("conn-pair-alloc (already)")
    elif _ALLOC_STASH in src:
        src = src.replace(
            _ALLOC_STASH,
            _ALLOC_STASH
            + "        try:\n"
            + "            if not getattr(self, \"is_producer\", True) and "
            + "int(num_external_tokens or 0) == 0:\n"
            + "                _dsv4_kv_pair_seal(self, extra=\"alloc0\")  "
            + "# DSV4-KV-PAIR-ALLOC0\n"
            + "        except Exception:\n"
            + "            pass\n",
            1,
        )
        applied.append("conn-pair-alloc")
    else:
        applied.append("conn-pair-alloc (no stash)")

    if "# DSV4-KV-HASH-META" in src:
        applied.append("conn-meta (already)")
    else:
        old_meta = "        meta.reqs_to_send = self._reqs_need_send\n"
        if old_meta in src:
            src = src.replace(
                old_meta,
                old_meta
                + "        _dsv4_kv_attach_dest_meta(self, meta)  "
                + "# DSV4-KV-HASH-META\n",
                1,
            )
            applied.append("conn-meta")
        else:
            applied.append("conn-meta (anchor missing, skipped)")

    if "# DSV4-KV-HASH-LOAD" in src:
        applied.append("conn-load (already)")
    else:
        # 222403: d626 is ``start_load_kv(self, forward_context)``; the mixin
        # calls ``start_load_kv(get_forward_context())``. Inserting a bare
        # ``metadata`` identifier is a NameError that kills the worker on
        # first curl (PARSE_ERROR / NIAH 502). Parse the real arg; wrap so a
        # future rename cannot take the process down.
        rx = re.compile(
            r"^([ \t]+)def start_load_kv\("
            r"[\s\S]*?\bself\s*,\s*([A-Za-z_][A-Za-z0-9_]*)"
            r"[\s\S]*?\)(?:\s*->\s*[^:\n]+)?\s*:\s*\n",
            re.M,
        )
        matches = list(rx.finditer(src))
        if not matches:
            applied.append("conn-load (anchor missing, skipped)")
        else:
            for m in reversed(matches):
                indent = m.group(1) + "    "
                arg = m.group(2)
                blob = (
                    f"{indent}try:\n"
                    f"{indent}    _dsv4_kv_stash_load_meta(self, {arg})  "
                    f"# DSV4-KV-HASH-LOAD\n"
                    f"{indent}    _dsv4_kv_hash_dests(self, extra=\"load\")  "
                    f"# DSV4-KV-HASH-LOAD-HASH\n"
                    f"{indent}except Exception:\n"
                    f"{indent}    pass\n"
                )
                src = src[: m.end()] + blob + src[m.end() :]
            applied.append("conn-load")

    if "# DSV4-KV-HASH-LOAD-HASH" in src:
        applied.append("conn-load-hash (already)")
    elif "# DSV4-KV-HASH-LOAD" in src:
        def _add_load_hash(m):
            line = m.group(1)
            indent = re.match(r"[ \t]*", line).group(0)
            return (
                line
                + f'{indent}_dsv4_kv_hash_dests(self, extra="load")  '
                f"# DSV4-KV-HASH-LOAD-HASH\n"
            )

        src, nsub = re.subn(
            r"([ \t]*_dsv4_kv_stash_load_meta\(self, [A-Za-z_][A-Za-z0-9_]*\)  "
            r"# DSV4-KV-HASH-LOAD\n)"
            r"(?![ \t]*_dsv4_kv_hash_dests)",
            _add_load_hash,
            src,
        )
        applied.append("conn-load-hash" if nsub else "conn-load-hash (skipped)")
    else:
        applied.append("conn-load-hash (no load hook)")

    if "# DSV4-KV-HASH-POSTWAIT" in src:
        applied.append("conn-postwait (already)")
    else:
        rx = re.compile(
            r"^([ \t]+)def wait_for_layer_load\("
            r"[\s\S]*?\)(?:\s*->\s*[^:\n]+)?\s*:\s*\n",
            re.M,
        )
        matches = list(rx.finditer(src))
        if not matches:
            applied.append("conn-postwait (anchor missing, skipped)")
        else:
            for m in reversed(matches):
                indent = m.group(1) + "    "
                blob = (
                    f"{indent}try:\n"
                    f"{indent}    _dsv4_kv_hash_dests("
                    f"self, extra=\"post-wait\")  "
                    f"# DSV4-KV-HASH-POSTWAIT\n"
                    f"{indent}except Exception:\n"
                    f"{indent}    pass\n"
                )
                src = src[: m.end()] + blob + src[m.end() :]
            applied.append("conn-postwait")

    if "# DSV4-KV-HASH-FINISHED" in src:
        applied.append("conn-finished (already)")
    else:
        rx = re.compile(
            r"^([ \t]+)def get_finished\("
            r"[\s\S]*?\bself\s*,\s*([A-Za-z_][A-Za-z0-9_]*)"
            r"[\s\S]*?\)(?:\s*->\s*[^:\n]+)?\s*:\s*\n",
            re.M,
        )
        matches = list(rx.finditer(src))
        if not matches:
            applied.append("conn-finished (anchor missing, skipped)")
        else:
            for m in reversed(matches):
                indent = m.group(1) + "    "
                arg = m.group(2)
                blob = (
                    f"{indent}try:\n"
                    f"{indent}    _dsv4_kv_hash_dests("
                    f"self, {arg}, extra=\"finished\")  "
                    f"# DSV4-KV-HASH-FINISHED\n"
                    f"{indent}except Exception:\n"
                    f"{indent}    pass\n"
                )
                src = src[: m.end()] + blob + src[m.end() :]
            applied.append("conn-finished")

    if "# DSV4-KV-HASH-RECV" in src:
        applied.append("conn-recv (already)")
        return src
    m = re.search(
        r"def _pop_done_transfers\(self\)[^\n]*:\n(?:.*?\n)*?"
        r"([ \t]+)return done_req_ids\n",
        src,
    )
    if not m:
        applied.append("conn-recv (anchor missing, skipped)")
        return src
    indent = m.group(1)
    src = (
        src[: m.start(1)]
        + f"{indent}_dsv4_kv_hash_on_recving(self, done_req_ids)  # DSV4-KV-HASH-RECV\n"
        + src[m.start(1) :]
    )
    applied.append("conn-recv")
    return src


def _find_mla_paths(vllm_dir: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for rel in MLA_CANDIDATES:
        path = os.path.join(vllm_dir, rel)
        if os.path.isfile(path) and path not in seen:
            seen.add(path)
            found.append(path)
    for base in (
        os.path.join(vllm_dir, "models/deepseek_v4"),
        os.path.join(vllm_dir, "v1/attention/backends"),
        os.path.join(vllm_dir, "v1/attention/ops"),
    ):
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(root, fn)
                if path in seen:
                    continue
                try:
                    txt = open(path).read(12000)
                except OSError:
                    continue
                if "DeepseekV4ROCMAiterMLASparse" in txt or (
                    "metadata_key" in txt and "DeepseekV4" in txt
                ) or "def _forward_decode" in txt and "DeepseekV4" in txt:
                    seen.add(path)
                    found.append(path)
    return found


def _find_mla_path(vllm_dir: str) -> str | None:
    paths = _find_mla_paths(vllm_dir)
    return paths[0] if paths else None


def _append_mla_helper(src: str) -> str:
    """Module-level helper at EOF. Never insert before the first class:

    ``rocm.py`` starts its first type with ``@dataclass`` (222135). Putting the
    helper between the decorator and ``class`` makes dataclass wrap a function
    (``'function' object has no attribute '__mro__'``) and vLLM dies at import.
    """
    if "def _dsv4_mla_meta_log" in src:
        return src
    return src.rstrip() + "\n\n" + MLA_HELPER.lstrip("\n") + "\n"


def _mla_meta_span(src: str) -> tuple[int, int] | None:
    start = src.find("def _dsv4_mla_meta_log")
    if start < 0:
        return None
    # Module-level helper. Nested ``def _as_list`` / ``def _page`` must not
    # end the span — ``kvd1=`` sits after those (223449 upgrade).
    nl = src.find("\n", start)
    i = (nl + 1) if nl >= 0 else len(src)
    while i < len(src):
        line_end = src.find("\n", i)
        if line_end < 0:
            line_end = len(src)
        line = src[i:line_end]
        if line.strip() and not line.startswith((" ", "\t")):
            return start, i
        i = line_end + 1
    return start, len(src)


def _mla_helper_is_live(src: str) -> bool:
    span = _mla_meta_span(src)
    if span is None:
        return False
    start, end = span
    return "kbs=" in src[start:end] and "kvd1=" in src[start:end] and "tmax=" in src[start:end]


def _replace_mla_meta_fn(src: str) -> str:
    span = _mla_meta_span(src)
    if span is None:
        return src
    start, end = span
    return src[:start] + MLA_META_FN.strip() + "\n" + src[end:].lstrip("\n")


def _fwd_decode_insert(indent: str) -> str:
    return (
        f"{indent}try:\n"
        f"{indent}    _dsv4_mla_meta_log("
        f"attn_meta=attn_metadata, kv_cache=kv_cache, "
        f'swa_meta=locals().get("swa_metadata"), '
        f'where="fwd-decode", '
        f'cr=getattr(self, "compress_ratio", None), '
        f'swa_only=locals().get("swa_only"), '
        f'topk_buf=getattr(self, "topk_indices_buffer", None))  '
        f"# DSV4-MLA-CR\n"
        f"{indent}except Exception:\n"
        f"{indent}    pass\n"
    )


def _ensure_mla_helper(src: str, applied: list, has_call: bool) -> str:
    if "def _dsv4_mla_meta_log" in src:
        if _mla_helper_is_live(src):
            applied.append("mla-helper (already)")
            return src
        src = _replace_mla_meta_fn(src)
        applied.append("mla-helper (upgraded)")
        return src
    if has_call:
        src = _append_mla_helper(src)
        applied.append("mla-helper")
        return src
    applied.append("mla-helper (no call site, skipped)")
    return src


def _patch_mla(src: str, applied: list) -> str:
    persist = "        if metadata_key != self._prev_metadata_key:"
    persist_log = (
        "        _dsv4_mla_meta_log("
        "metadata_key=metadata_key, attn_meta=common_attn_metadata, "
        "where=\"persist-guard\")\n"
        "        if metadata_key != self._prev_metadata_key:"
    )
    if "_dsv4_mla_meta_log(" in src and "where=\"persist-guard\"" in src:
        applied.append("mla-key (already)")
    elif persist in src:
        src = src.replace(persist, persist_log, 1)
        applied.append("mla-key")
    else:
        m = re.search(r"^([ \t]+)metadata_key = .+$", src, re.M)
        if not m:
            applied.append("mla-key (anchor missing, skipped)")
        else:
            indent = m.group(1)
            insert = (
                f"\n{indent}_dsv4_mla_meta_log("
                f"metadata_key=metadata_key, "
                f"attn_meta=locals().get('common_attn_metadata'), "
                f'where="key")'
            )
            src = src[: m.end()] + insert + src[m.end() :]
            applied.append("mla-key")

    if 'where="indices"' in src:
        applied.append("mla-idx (already)")
    else:
        m = re.search(
            r"^([ \t]+)paged_kv_indices = .+$",
            src,
            re.M,
        )
        if m:
            indent = m.group(1)
            insert = (
                f"\n{indent}_dsv4_mla_meta_log("
                f"metadata_key=locals().get('metadata_key'), "
                f"attn_meta=locals().get('common_attn_metadata'), "
                f"paged_kv_indices=paged_kv_indices, where=\"indices\")"
            )
            src = src[: m.end()] + insert + src[m.end() :]
            applied.append("mla-idx")
        elif "paged_kv_indices" in src:
            applied.append("mla-idx (assign missing, skipped)")
        else:
            applied.append("mla-idx (paged_kv_indices missing, skipped)")

    if "_dsv4_mla_meta_log(" not in src:
        m = re.search(r"^([ \t]+)def build\(self[^\n]*:\n", src, re.M)
        if m:
            indent = m.group(1) + "    "
            insert = (
                f"{indent}_dsv4_mla_meta_log("
                f"attn_meta=locals().get('common_attn_metadata') or "
                f"locals().get('attn_metadata'), where=\"build\")\n"
            )
            src = src[: m.end()] + insert + src[m.end() :]
            applied.append("mla-build")
        else:
            applied.append("mla-build (anchor missing, skipped)")

    old_fwd = (
        '_dsv4_mla_meta_log(attn_meta=attn_metadata, kv_cache=kv_cache, '
        'where="fwd-decode")'
    )
    live_fwd = (
        'swa_meta=locals().get("swa_metadata"), where="fwd-decode")  '
        "# DSV4-MLA-FWD-LIVE"
    )
    cr_fwd = (
        'swa_meta=locals().get("swa_metadata"), where="fwd-decode", '
        'cr=getattr(self, "compress_ratio", None), '
        'swa_only=locals().get("swa_only"), '
        'topk_buf=getattr(self, "topk_indices_buffer", None))  '
        "# DSV4-MLA-CR"
    )
    new_fwd = (
        '_dsv4_mla_meta_log(attn_meta=attn_metadata, kv_cache=kv_cache, '
        + cr_fwd
    )
    if "# DSV4-MLA-CR" in src and 'where="fwd-decode"' in src:
        applied.append("mla-fwd (already)")
    elif live_fwd in src:
        src = src.replace(live_fwd, cr_fwd, 1)
        applied.append("mla-fwd (upgraded cr)")
    elif old_fwd in src:
        src = src.replace(old_fwd, new_fwd, 1)
        applied.append("mla-fwd (upgraded)")
    elif 'where="fwd-decode"' in src:
        applied.append("mla-fwd (already)")
    else:
        rx = re.compile(
            r"^([ \t]+)def _forward_decode\("
            r"[\s\S]*?\)(?:\s*->\s*[^:\n]+)?\s*:\s*\n",
            re.M,
        )
        matches = list(rx.finditer(src))
        if not matches:
            applied.append("mla-fwd (anchor missing, skipped)")
        else:
            m = matches[0]
            indent = m.group(1) + "    "
            src = src[: m.end()] + _fwd_decode_insert(indent) + src[m.end() :]
            applied.append("mla-fwd")

    src = _ensure_mla_helper(src, applied, "_dsv4_mla_meta_log(" in src)
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
    if not os.path.isfile(conn):
        print(f"[dsv4-kv] {CONN_REL} not found -- skipping.")
        return 0

    src = open(conn).read()
    orig = src
    try:
        src = _patch_connector(src, applied)
    except ValueError as e:
        print(f"[dsv4-kv] ERROR: connector: {e}", file=sys.stderr)
        return 1
    if src != orig:
        _write(conn, src, ".dsv4kvhash")
        try:
            _compile(conn)
        except Exception as e:  # noqa: BLE001
            print(f"[dsv4-kv] ERROR: connector compile: {e}", file=sys.stderr)
            return 1

    mlas = _find_mla_paths(vllm_dir)
    if not mlas:
        applied.append("mla (file missing, skipped)")
        print(
            "[dsv4-kv] WARN: DSV4 sparse backend not found -- MLA metadata "
            "log skipped (connector hash still on).",
            file=sys.stderr,
        )
    else:
        for mla in mlas:
            msrc = open(mla).read()
            morig = msrc
            msrc = _patch_mla(msrc, applied)
            if msrc == morig:
                continue
            _write(mla, msrc, ".dsv4kvhash")
            try:
                _compile(mla)
            except Exception as e:  # noqa: BLE001
                print(
                    f"[dsv4-kv] WARN: MLA compile failed ({e}); "
                    "leaving connector hash. Backend log skipped.",
                    file=sys.stderr,
                )
                open(mla, "w").write(morig)
                applied.append("mla (compile failed, reverted)")

    runner = os.path.join(vllm_dir, RUNNER_REL)
    if not os.path.isfile(runner):
        applied.append("runner-bt (file missing, skipped)")
        print(
            "[dsv4-kv] WARN: gpu_model_runner.py not found -- "
            "block_table probe skipped.",
            file=sys.stderr,
        )
    else:
        rsrc = open(runner).read()
        rorig = rsrc
        rsrc = _patch_runner(rsrc, applied)
        if rsrc != rorig:
            _write(runner, rsrc, ".dsv4kvhash")
            try:
                _compile(runner)
            except Exception as e:  # noqa: BLE001
                print(
                    f"[dsv4-kv] ERROR: runner compile failed ({e}); "
                    "reverted. block_table probe is required.",
                    file=sys.stderr,
                )
                open(runner, "w").write(rorig)
                return 1
        if "# DSV4-KV-RUNNER-BT" not in open(runner).read():
            print(
                "[dsv4-kv] ERROR: gpu_model_runner.py patched but "
                "DSV4-KV-RUNNER-BT call missing.",
                file=sys.stderr,
            )
            return 1

    print(f"[dsv4-kv] hunks: {', '.join(applied)}")
    return 0


CONN_FAKE = '''
import logging

logger = logging.getLogger(__name__)


def compute_block_transfer_offsets(**kwargs):
    return kwargs


class MoRIIOConnectorScheduler:
    def __init__(self):
        self.kv_caches = {}
        self.is_producer = False
        self.transfer_id_to_request_id = {}
        self.request_id_to_transfer_id = {}
        self._reqs_need_send = {}

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

    def build_connector_meta(self, scheduler_output):
        meta = type("Meta", (), {"requests": {}})()
        meta.reqs_to_send = self._reqs_need_send
        return meta

    def _compute_block_transfer_offsets(
        self,
        layer_name,
        local_block_ids,
        remote_block_ids,
        remote_moriio_meta,
    ):
        return compute_block_transfer_offsets(
            layer_name=layer_name,
            kv_cache=self.kv_caches[layer_name],
            layer_to_spec=getattr(self, "layer_to_spec", None),
            local_block_ids=local_block_ids,
            remote_block_ids=remote_block_ids,
        )

    def start_load_kv(self, forward_context):
        return None

    def _get_connector_metadata(self):
        return None

    def wait_for_layer_load(self, layer_name):
        return None

    def get_finished(self, finished_req_ids):
        return set(), set()

    def _pop_done_transfers(self):
        done_req_ids = {"rid-2k"}
        return done_req_ids
'''

MLA_FAKE = '''
from dataclasses import dataclass

@dataclass
class DeepseekV4ROCMAiterMLAMetadata:
    seq_lens: int = 0


class DeepseekV4ROCMAiterMLASparseMetadataBuilder:
    def __init__(self):
        self._prev_metadata_key = None

    def build(self, common_attn_metadata):
        metadata_key = (1, 2, 3)
        if metadata_key != self._prev_metadata_key:
            self._prev_metadata_key = metadata_key
        paged_kv_indices = [0, 0, 5, 6, 7]
        return metadata_key, paged_kv_indices

    def _forward_decode(
        self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output
    ):
        return None
'''

RUNNER_FAKE = '''
import logging

logger = logging.getLogger("vllm")


class GPUModelRunner:
    def _build_attention_metadata(
        self, num_tokens, num_reqs, max_query_len, for_cudagraph_capture=False
    ):
        attn_metadata = {}
        spec_decode_common_attn_metadata = None
        return attn_metadata, spec_decode_common_attn_metadata

    def capture_model(self) -> int:
        return 0
'''

MLA_DECORATED_NO_CALL = '''
from dataclasses import dataclass

@dataclass
class DeepseekV4ROCMAiterMLAMetadata:
    seq_lens: int = 0
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


class _Page:
    def __init__(self, xs):
        self.xs = [float(x) for x in xs]

    def detach(self):
        return self

    def float(self):
        return self

    def abs(self):
        return _Page([abs(x) for x in self.xs])

    def sum(self):
        return _Scalar(sum(self.xs))

    def view(self, *_a, **_k):
        return self

    def flatten(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return [int(x) % 256 for x in self.xs[:16]]

    def __getitem__(self, sl):
        xs = self.xs[sl] if not isinstance(sl, int) else [self.xs[sl]]
        return _Page(xs)


class _Scalar:
    def __init__(self, v):
        self.v = float(v)

    def item(self):
        return self.v


class _Cache:
    def __init__(self, fill, nblocks=32):
        self.fill = fill
        self.shape = (nblocks, 4, 8)

    def __getitem__(self, i):
        return _Page([self.fill] * 8)


def selftest() -> int:
    d = tempfile.mkdtemp(prefix="dsv4kvhash-")
    conn_dir = os.path.join(
        d, "distributed/kv_transfer/kv_connector/v1/moriio"
    )
    mla_dir = os.path.join(d, "models/deepseek_v4/amd")
    runner_dir = os.path.join(d, "v1/worker")
    os.makedirs(conn_dir)
    os.makedirs(mla_dir)
    os.makedirs(runner_dir)
    conn_path = os.path.join(conn_dir, "moriio_connector.py")
    mla_path = os.path.join(mla_dir, "rocm.py")
    runner_path = os.path.join(runner_dir, "gpu_model_runner.py")
    open(conn_path, "w").write(CONN_FAKE)
    open(mla_path, "w").write(MLA_FAKE)
    open(runner_path, "w").write(RUNNER_FAKE)

    if patch(d):
        print("[selftest] FAIL: patch returned non-zero")
        return 1
    cout = open(conn_path).read()
    mout = open(mla_path).read()
    rout = open(runner_path).read()
    if "DSV4-KV-RUNNER-BT" not in rout or "def _dsv4_kv_runner_bt_log" not in rout:
        print("[selftest] FAIL: runner block_table probe not inserted")
        return 1
    if "g0slot=" not in rout or "cov_t=" not in rout:
        print("[selftest] FAIL: runner-bt g0slot coverage not inserted")
        return 1
    if "self._dsv4_kv_runner_bt_log(" not in rout:
        print("[selftest] FAIL: runner-bt call missing")
        return 1
    try:
        compile(rout, runner_path, "exec")
    except SyntaxError as e:
        print(f"[selftest] FAIL: patched gpu_model_runner SyntaxError: {e}")
        return 1
    for needle in (
        '_dsv4_kv_hash_log(self, "src", layer_name, local_block_ids)',
        "_dsv4_kv_stash_dests(self, request, blocks)",
        "_dsv4_kv_stash_load_meta(self, forward_context)  # DSV4-KV-HASH-LOAD",
        '_dsv4_kv_hash_dests(self, extra="load")  # DSV4-KV-HASH-LOAD-HASH',
        '_dsv4_kv_hash_dests(self, extra="post-wait")  # DSV4-KV-HASH-POSTWAIT',
        '_dsv4_kv_hash_dests(self, finished_req_ids, extra="finished")  # DSV4-KV-HASH-FINISHED',
        "_dsv4_kv_hash_on_recving(self, done_req_ids)  # DSV4-KV-HASH-RECV",
        'extra="offsets"',
        "def _dsv4_kv_scan_ids",
        "_dsv4_kv_attach_dest_meta(self, meta)  # DSV4-KV-HASH-META",
        "DSV4-KV-DEST-IDS",
        "def _dsv4_kv_ids_from_forward",
        "def _dsv4_kv_hash_dest_groups",
        "def _dsv4_kv_zip_log",
        "DSV4-KV-ZIP",
        "def _dsv4_kv_pair_seal",
        "DSV4-KV-PAIR",
        "DSV4-KV-PAIR-LIVE",
        "_dsv4_mla_meta_log(",
        'where="persist-guard"',
        'where="indices"',
        'where="fwd-decode"',
        "DSV4-MLA-FWD-LIVE",
        "DSV4-MLA-CR",
        'swa_meta=locals().get("swa_metadata")',
        "kvd1=",
        "kbs=",
        "tmax=",
        'cr=getattr(self, "compress_ratio", None)',
        "cov_t=%d",
    ):
        if needle not in cout and needle not in mout:
            print(f"[selftest] FAIL: missing {needle!r}")
            return 1
    if "_dsv4_kv_stash_load_meta(self, metadata)  # DSV4-KV-HASH-LOAD" in cout:
        print("[selftest] FAIL: LOAD hook still hardcodes metadata (222403)")
        return 1
    if re.search(
        r"def start_load_kv\([^\n]*forward_context\n\s+_dsv4_kv_stash_load_meta",
        cout,
    ):
        print("[selftest] FAIL: LOAD hook split the start_load_kv signature")
        return 1
    try:
        compile(cout, conn_path, "exec")
    except SyntaxError as e:
        print(f"[selftest] FAIL: patched connector SyntaxError: {e}")
        return 1

    typed_src = CONN_FAKE.replace(
        "    def start_load_kv(self, forward_context):\n        return None\n",
        '    def start_load_kv(\n        self,\n        forward_context: "ForwardContext",\n    ) -> None:\n        return None\n',
    )
    typed_applied: list = []
    typed_out = _patch_connector(typed_src, typed_applied)
    if "conn-load" not in typed_applied:
        print(f"[selftest] FAIL: typed start_load_kv not patched {typed_applied}")
        return 1
    if "_dsv4_kv_stash_load_meta(self, forward_context)" not in typed_out:
        print("[selftest] FAIL: typed LOAD hook missing forward_context")
        return 1
    try:
        compile(typed_out, "<typed-conn>", "exec")
    except SyntaxError as e:
        print(f"[selftest] FAIL: typed start_load_kv SyntaxError: {e}")
        return 1

    if patch(d):
        print("[selftest] FAIL: second apply returned non-zero")
        return 1
    if (
        cout != open(conn_path).read()
        or mout != open(mla_path).read()
        or rout != open(runner_path).read()
    ):
        print("[selftest] FAIL: not idempotent")
        return 1

    i_meta = mout.find("class DeepseekV4ROCMAiterMLAMetadata")
    i_helper = mout.find("def _dsv4_mla_meta_log")
    if i_helper < 0 or i_meta < 0 or i_helper < i_meta:
        print("[selftest] FAIL: MLA helper sandwiched before @dataclass class (222135)")
        return 1
    try:
        compile(mout, mla_path, "exec")
        exec(compile(mout, mla_path, "exec"), {})
    except Exception as e:  # noqa: BLE001
        print(f"[selftest] FAIL: patched rocm.py not importable: {e}")
        return 1

    dec_applied: list = []
    dec_out = _patch_mla(MLA_DECORATED_NO_CALL, dec_applied)
    if "mla-helper (no call site, skipped)" not in dec_applied:
        print(f"[selftest] FAIL: no-call MLA applied={dec_applied}")
        return 1
    if "def _dsv4_mla_meta_log" in dec_out:
        print("[selftest] FAIL: helper inserted with no call site")
        return 1
    try:
        exec(compile(dec_out, "<dec>", "exec"), {})
    except Exception as e:  # noqa: BLE001
        print(f"[selftest] FAIL: decorated metadata class not importable: {e}")
        return 1

    old_mla = MLA_FAKE.replace(
        "        return None\n",
        "        try:\n"
        "            _dsv4_mla_meta_log(attn_meta=attn_metadata, "
        "kv_cache=kv_cache, where=\"fwd-decode\")\n"
        "        except Exception:\n"
        "            pass\n"
        "        return None\n",
        1,
    )
    old_mla = (
        old_mla.rstrip()
        + "\n\n"
        + MLA_KV_HASH_ON.strip()
        + "\n\n"
        + "def _dsv4_mla_meta_log(metadata_key=None, attn_meta=None, "
        "paged_kv_indices=None, kv_cache=None, where=\"\"):\n"
        '    """Keyed (where, sb) from qlen. 223327 capture burn."""\n'
        "    return None\n"
    )
    old_applied: list = []
    old_out = _patch_mla(old_mla, old_applied)
    if "mla-fwd (upgraded)" not in old_applied:
        print(f"[selftest] FAIL: old fwd call not upgraded {old_applied}")
        return 1
    if "mla-helper (upgraded)" not in old_applied:
        print(f"[selftest] FAIL: old mla helper not upgraded {old_applied}")
        return 1
    if "DSV4-MLA-CR" not in old_out or (
        'cr=getattr(self, "compress_ratio", None)' not in old_out
    ):
        print("[selftest] FAIL: upgraded mla missing CR marker")
        return 1
    if "DSV4-MLA-FWD-LIVE" not in old_out or (
        'swa_meta=locals().get("swa_metadata")' not in old_out
    ):
        print("[selftest] FAIL: upgraded mla missing live marker")
        return 1
    if "kvd1=" not in old_out:
        print("[selftest] FAIL: upgraded mla missing kvd1")
        return 1
    if "kbs=" not in old_out:
        print("[selftest] FAIL: upgraded mla missing kbs")
        return 1
    if "tmax=" not in old_out:
        print("[selftest] FAIL: upgraded mla missing tmax")
        return 1
    if "Keyed (where, sb) from qlen" in old_out:
        print("[selftest] FAIL: old qlen-keyed helper still present")
        return 1
    try:
        compile(old_out, "<old-mla>", "exec")
    except SyntaxError as e:
        print(f"[selftest] FAIL: upgraded old mla SyntaxError: {e}")
        return 1

    live449 = (
        MLA_FAKE.rstrip()
        + "\n\n"
        + MLA_KV_HASH_ON.strip()
        + "\n\n"
        + "def _dsv4_mla_meta_log(metadata_key=None, attn_meta=None, "
        "paged_kv_indices=None, kv_cache=None, where=\"\", swa_meta=None):\n"
        '    """DSV4-MLA-FWD-LIVE 223449 prefix key, no kvd1."""\n'
        "    return None\n"
    )
    a449: list = []
    out449 = _patch_mla(live449, a449)
    if "mla-helper (upgraded)" not in a449:
        print(f"[selftest] FAIL: 223449 helper not upgraded {a449}")
        return 1
    if "kvd1=" not in out449:
        print("[selftest] FAIL: 223449 helper missing kvd1 after upgrade")
        return 1
    if "223449 prefix key, no kvd1" in out449:
        print("[selftest] FAIL: old 223449 helper still present")
        return 1
    if "kbs=" not in out449:
        print("[selftest] FAIL: 223449 helper missing kbs after upgrade")
        return 1
    if "tmax=" not in out449:
        print("[selftest] FAIL: 223449 helper missing tmax after upgrade")
        return 1

    fwd_live_src = MLA_FAKE.replace(
        "        return None\n",
        "        try:\n"
        "            _dsv4_mla_meta_log(attn_meta=attn_metadata, "
        "kv_cache=kv_cache, "
        'swa_meta=locals().get("swa_metadata"), where="fwd-decode")  '
        "# DSV4-MLA-FWD-LIVE\n"
        "        except Exception:\n"
        "            pass\n"
        "        return None\n",
        1,
    )
    a_cr: list = []
    out_cr = _patch_mla(fwd_live_src, a_cr)
    if "mla-fwd (upgraded cr)" not in a_cr:
        print(f"[selftest] FAIL: FWD-LIVE not upgraded to CR {a_cr}")
        return 1
    if "# DSV4-MLA-CR" not in out_cr:
        print("[selftest] FAIL: CR marker missing after FWD-LIVE upgrade")
        return 1

    sys.path.insert(0, os.path.dirname(conn_path))
    import logging

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    import moriio_connector as mc  # noqa: PLC0415

    dest_dir = os.path.join(d, "kv-dest")
    os.makedirs(dest_dir)
    os.environ["DSV4_KV_DEST_DIR"] = dest_dir

    def _dst(seen, extra, bucket, live="nz", role="attn"):
        return any(
            len(k) >= 5
            and k[0] == "dst"
            and k[1] == role
            and k[2] == extra
            and k[3] == bucket
            and k[4] == live
            for k in seen
        )

    os.environ["DSV4_KV_HASH"] = "1"
    s = mc.MoRIIOConnectorScheduler()
    s.kv_caches = {
        "model.layers.2.attn": _Cache(3.0),
        "model.layers.0.attn.swa_cache": _Cache(1.0),
        "model.layers.2.attn.indexer.k_cache": _Cache(2.0),
        "model.layers.2.attn.compressor.state_cache": _Cache(9.0),
    }
    s._compute_block_transfer_offsets(
        "model.layers.2.attn", [1, 2, 3], [10, 11, 12], {}
    )
    s._compute_block_transfer_offsets(
        "model.layers.0.attn.swa_cache", [4, 5], [14, 15], {}
    )
    s._compute_block_transfer_offsets(
        "model.layers.2.attn.indexer.k_cache", [6], [16], {}
    )
    s._compute_block_transfer_offsets(
        "model.layers.2.attn.compressor.state_cache", [7], [17], {}
    )
    seen = getattr(s, "_dsv4_kv_logged", set())
    if ("src", "attn", "m", None) not in seen:
        print(f"[selftest] FAIL: 2k-sized attn bucket missing {seen}")
        return 1
    s._compute_block_transfer_offsets(
        "model.layers.2.attn", list(range(20)), list(range(20, 40)), {}
    )
    seen = getattr(s, "_dsv4_kv_logged", set())
    if ("src", "attn", "l", None) not in seen:
        print(f"[selftest] FAIL: 16k-sized attn bucket missing {seen}")
        return 1
    s._compute_block_transfer_offsets(
        "model.layers.2.attn", [1], [10], {}
    )
    seen = getattr(s, "_dsv4_kv_logged", set())
    if ("src", "attn", "s", None) not in seen:
        print(f"[selftest] FAIL: short curl-sized attn bucket missing {seen}")
        return 1

    s.is_producer = True
    s._compute_block_transfer_offsets(
        "model.layers.2.attn",
        [11, 10, 9, 6, 5, 4, 3, 2, 1],
        [7, 9, 6, 5, 4, 3, 2, 1, 8],
        {},
    )
    z = getattr(s, "_dsv4_kv_last_zip", None) or {}
    if (
        z.get("side") != "src"
        or z.get("role") != "attn"
        or z.get("bucket") != "m"
        or z.get("nested")
        or z.get("remote") != [7, 9, 6, 5, 4, 3, 2, 1, 8]
    ):
        print(f"[selftest] FAIL: zip 2k g0 dests {z}")
        return 1
    zipped = getattr(s, "_dsv4_kv_zipped", set())
    if ("src", "attn", "m") not in zipped:
        print(f"[selftest] FAIL: zip-remote 2k key {zipped}")
        return 1
    s._compute_block_transfer_offsets("model.layers.2.attn", [1], [1], {})
    s._compute_block_transfer_offsets(
        "model.layers.2.attn",
        list(range(68)),
        list(range(1, 69)),
        {},
    )
    zipped = getattr(s, "_dsv4_kv_zipped", set())
    if ("src", "attn", "s") not in zipped or ("src", "attn", "l") not in zipped:
        print(f"[selftest] FAIL: zip s/l {zipped}")
        return 1
    z = getattr(s, "_dsv4_kv_last_zip", None) or {}
    if not z.get("remote") or z["remote"][-1] != 68:
        print(f"[selftest] FAIL: 16k zip tail {z}")
        return 1
    s._compute_block_transfer_offsets(
        "model.layers.2.attn.compressor.state_cache",
        [7, 9, 6],
        [10, 11, 12],
        {},
    )
    z = getattr(s, "_dsv4_kv_last_zip", None) or {}
    if z.get("role") == "attn" and z.get("remote") == [10, 11, 12]:
        print(f"[selftest] FAIL: compressor zip overwrote attn {z}")
        return 1

    nested = mc.MoRIIOConnectorScheduler()
    nested.is_producer = True
    nested.kv_caches = s.kv_caches
    nested._compute_block_transfer_offsets(
        "model.layers.2.attn",
        [11, 10, 9, 6, 5, 4, 3, 2, 1],
        [[10, 11, 12], [13, 14, 15]],
        {},
    )
    z = getattr(nested, "_dsv4_kv_last_zip", None) or {}
    if not z.get("nested") or z.get("remote") != [10, 11, 12]:
        print(f"[selftest] FAIL: nested zip {z}")
        return 1
    s.is_producer = False

    req = _Req("rid-2k", "tx-aaa")
    g1 = (
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
        list(range(17)),
        [20, 21, 22],
    )
    s.update_state_after_alloc(req, _Blocks(g1), 2193)
    if "rid-2k" not in getattr(s, "_dsv4_kv_dest", {}):
        print("[selftest] FAIL: dest stash missing")
        return 1
    if req.kv_transfer_params.get("dsv4_dest_groups") != [list(g) for g in g1]:
        print("[selftest] FAIL: dest groups not copied onto kv_transfer_params")
        return 1
    meta = s.build_connector_meta(None)
    if not getattr(meta, "dsv4_dest", None):
        print("[selftest] FAIL: build_connector_meta missing dsv4_dest")
        return 1
    s.start_load_kv(type("M", (), {"requests": []})())
    seen = getattr(s, "_dsv4_kv_logged", set())
    if not _dst(seen, "load-g0", "m"):
        print(f"[selftest] FAIL: dest hash at start_load_kv g0 missing {seen}")
        return 1
    if not _dst(seen, "load-g1", "m"):
        print(f"[selftest] FAIL: per-group dest hash g1 missing {seen}")
        return 1
    pair = getattr(s, "_dsv4_kv_last_pair", None) or {}
    if (
        pair.get("extra") != "load"
        or pair.get("id0") != 1
        or pair.get("swa0") != 10
        or pair.get("bucket") != "m"
        or pair.get("n_g0") != 9
        or pair.get("n_swa") != 3
        or "10:" not in (pair.get("swa") or "")
        or "12:" not in (pair.get("swa") or "")
    ):
        print(f"[selftest] FAIL: pair-load g0 vs swa0 {pair}")
        return 1
    class _Flip:
        def __init__(self):
            self.fills = {i: 3.0 for i in range(32)}
            self.fills[10] = 0.0
            self.shape = (32, 4, 8)

        def __getitem__(self, i):
            return _Page([self.fills.get(int(i), 0.0)] * 8)

    s_flip = mc.MoRIIOConnectorScheduler()
    s_flip.kv_caches = {"model.layers.2.attn": _Flip()}
    s_flip._dsv4_kv_dest = {
        "rid": [[7, 9, 6, 5, 4, 3, 2, 1, 8], [10, 11, 12]]
    }
    s_flip.start_load_kv(type("M", (), {"requests": []})())
    p1 = dict(getattr(s_flip, "_dsv4_kv_last_pair", None) or {})
    if "10:z:" not in (p1.get("swa") or ""):
        print(f"[selftest] FAIL: pair census missed empty dest10 {p1}")
        return 1
    s_flip.kv_caches["model.layers.2.attn"].fills[10] = 3.0
    mc._dsv4_kv_pair_seal(s_flip, extra="load")
    p2 = dict(getattr(s_flip, "_dsv4_kv_last_pair", None) or {})
    if "10:nz:" not in (p2.get("swa") or ""):
        print(f"[selftest] FAIL: identity key hid dest10 going live {p2}")
        return 1
    s.start_load_kv(type("M", (), {"requests": []})())
    pair = getattr(s, "_dsv4_kv_last_pair", None) or {}
    if pair.get("extra") != "post-seal":
        print(f"[selftest] FAIL: 2nd start_load not post-seal {pair}")
        return 1
    seen = getattr(s, "_dsv4_kv_logged", set())
    if not _dst(seen, "post-seal-g0", "m"):
        print(f"[selftest] FAIL: post-seal-g0 missing {seen}")
        return 1
    s.update_state_after_alloc(req, _Blocks(g1), 0)
    pair = getattr(s, "_dsv4_kv_last_pair", None) or {}
    if pair.get("extra") != "alloc0":
        print(f"[selftest] FAIL: alloc0 pair missing {pair}")
        return 1
    s.wait_for_layer_load("model.layers.2.attn")
    seen = getattr(s, "_dsv4_kv_logged", set())
    if not _dst(seen, "post-wait-g0", "m"):
        print(f"[selftest] FAIL: dest hash at wait_for_layer_load missing {seen}")
        return 1
    s.get_finished({"rid-2k"})
    seen = getattr(s, "_dsv4_kv_logged", set())
    if not _dst(seen, "finished-g0", "m"):
        print(f"[selftest] FAIL: dest hash at get_finished missing {seen}")
        return 1
    s._pop_done_transfers()
    seen = getattr(s, "_dsv4_kv_logged", set())
    if not _dst(seen, "recving-g0", "m"):
        print(f"[selftest] FAIL: dest hash at recving missing {seen}")
        return 1

    scan_dir = os.path.join(d, "kv-dest-scan")
    os.makedirs(scan_dir)
    os.environ["DSV4_KV_DEST_DIR"] = scan_dir
    # 223113: Worker store=0. Scan pages 1-16 instead of dests-missing return.
    s_scan = mc.MoRIIOConnectorScheduler()
    s_scan.kv_caches = {
        "model.layers.2.attn": _Cache(3.0),
        "model.layers.0.attn.swa_cache": _Cache(1.0),
        "model.layers.2.attn.indexer.k_cache": _Cache(2.0),
    }
    s_scan.start_load_kv(type("M", (), {"requests": []})())
    seen = getattr(s_scan, "_dsv4_kv_logged", set())
    if not _dst(seen, "load+scan", "m"):
        print(f"[selftest] FAIL: dest scan-1-16 missing {seen}")
        return 1
    if not _dst(seen, "load+p0", "s"):
        print(f"[selftest] FAIL: dest page-0 hash missing {seen}")
        return 1

    empty = mc.MoRIIOConnectorScheduler()
    empty.kv_caches = {"model.layers.2.attn": _Cache(0.0)}
    empty.start_load_kv(type("M", (), {"requests": []})())
    seen = getattr(empty, "_dsv4_kv_logged", set())
    if not _dst(seen, "load+scan", "m", live="z"):
        print(f"[selftest] FAIL: empty dest still must scan {seen}")
        return 1

    os.environ["DSV4_KV_DEST_DIR"] = os.path.join(d, "kv-dest-meta")
    os.makedirs(os.environ["DSV4_KV_DEST_DIR"])

    class _WorkerMeta(mc.MoRIIOConnectorScheduler):
        def _get_connector_metadata(self):
            req = type(
                "R",
                (),
                {
                    "request_id": "rid-meta",
                    "block_ids": [[7, 9, 6, 5, 4, 3, 2, 1, 8]],
                },
            )()
            return type("M", (), {"requests": {"rid-meta": req}})()

    w = _WorkerMeta()
    w.kv_caches = {"model.layers.2.attn": _Cache(3.0)}
    w.start_load_kv(type("Fwd", (), {})())
    if "rid-meta" not in getattr(w, "_dsv4_kv_dest", {}):
        print("[selftest] FAIL: dest ids not unwrapped from _get_connector_metadata")
        return 1
    seen = getattr(w, "_dsv4_kv_logged", set())
    if not _dst(seen, "load", "m"):
        print(f"[selftest] FAIL: dest-id hash after metadata unwrap missing {seen}")
        return 1

    # 223141: curl dest hashes must not hide 2k (bucket m) or 16k (bucket l).
    cap_dir = os.path.join(d, "kv-dest-cap")
    os.makedirs(cap_dir)
    os.environ["DSV4_KV_DEST_DIR"] = cap_dir
    s_cap = mc.MoRIIOConnectorScheduler()
    s_cap.kv_caches = {"model.layers.2.attn": _Cache(3.0, nblocks=80)}
    s_cap._dsv4_kv_dest = {"curl": [[1]]}
    for _ in range(6):
        s_cap.start_load_kv(type("M", (), {"requests": []})())
    s_cap._dsv4_kv_dest["rid-2k"] = [[7, 9, 6, 5, 4, 3, 2, 1, 8]]
    s_cap.start_load_kv(type("M", (), {"requests": []})())
    s_cap._dsv4_kv_dest["rid-16k"] = [
        list(range(1, 69)),
        [2],
        [3],
        list(range(5)),
        [8],
    ]
    s_cap.start_load_kv(type("M", (), {"requests": []})())
    seen = getattr(s_cap, "_dsv4_kv_logged", set())
    if not _dst(seen, "load", "s"):
        print(f"[selftest] FAIL: curl dest after cap missing {seen}")
        return 1
    if not _dst(seen, "load", "m"):
        print(f"[selftest] FAIL: 2k dest after curl cap missing {seen}")
        return 1
    if not _dst(seen, "load-g0", "l"):
        print(f"[selftest] FAIL: 16k dest pages 1-68 missing {seen}")
        return 1

    swa_pad = os.path.join(d, "kv-dest-swa")
    os.makedirs(swa_pad)
    os.environ["DSV4_KV_DEST_DIR"] = swa_pad
    s_pad = mc.MoRIIOConnectorScheduler()
    s_pad.kv_caches = {"model.layers.2.attn": _Cache(3.0)}
    s_pad._dsv4_kv_dest = {
        "pad": [[7, 9, 6], [0, 0, 10, 11, 12], [3], [4], [5]]
    }
    s_pad.start_load_kv(type("M", (), {"requests": []})())
    seen = getattr(s_pad, "_dsv4_kv_logged", set())
    if not _dst(seen, "load-g1nz", "m"):
        print(f"[selftest] FAIL: zero-stripped SWA dest hash missing {seen}")
        return 1
    if not _dst(seen, "load-g1", "m"):
        print(f"[selftest] FAIL: padded SWA dest hash missing {seen}")
        return 1

    fwd_dir = os.path.join(d, "kv-dest-fwd")
    os.makedirs(fwd_dir)
    os.environ["DSV4_KV_DEST_DIR"] = fwd_dir
    w_fwd = mc.MoRIIOConnectorScheduler()
    w_fwd.kv_caches = {"model.layers.2.attn": _Cache(3.0)}
    w_fwd.start_load_kv(
        type("Fwd", (), {"block_tables": [7, 9, 6, 5, 4, 3, 2, 1, 8]})()
    )
    if "_fwd" not in getattr(w_fwd, "_dsv4_kv_dest", {}):
        print("[selftest] FAIL: ForwardContext block_tables not stashed")
        return 1
    seen = getattr(w_fwd, "_dsv4_kv_logged", set())
    if not _dst(seen, "load", "m"):
        print(f"[selftest] FAIL: ForwardContext dest hash missing {seen}")
        return 1

    os.environ["DSV4_KV_HASH"] = "0"
    s2 = mc.MoRIIOConnectorScheduler()
    s2.kv_caches = s.kv_caches
    s2._compute_block_transfer_offsets(
        "model.layers.2.attn", [1], [10], {}
    )

    ns: dict = {}
    exec(MLA_HELPER, ns)
    os.environ["DSV4_KV_HASH"] = "1"
    meta = type(
        "A",
        (),
        {"seq_lens_cpu": [2193], "num_actual_tokens": 1},
    )()
    ns["_dsv4_mla_meta_log"](
        metadata_key=(8, 2193),
        attn_meta=meta,
        paged_kv_indices=[0, 0, 0, 5],
        where="selftest",
    )

    class _SplitCache:
        def __init__(self, fills, nblocks=16, d1=4):
            self.fills = fills
            self.shape = (nblocks, d1, 8)

        def __getitem__(self, i):
            return _Page([float(self.fills.get(int(i), 0.0))] * 8)

    class _Mem(logging.Handler):
        def __init__(self):
            super().__init__()
            self.lines = []

        def emit(self, record):
            self.lines.append(record.getMessage())

    live_ns: dict = {}
    exec(MLA_HELPER, live_ns)
    mem = _Mem()
    lg = logging.getLogger("vllm")
    lg.setLevel(logging.INFO)
    lg.addHandler(mem)
    prev_prop = lg.propagate
    lg.propagate = False
    fn = live_ns["_dsv4_mla_meta_log"]
    cap_meta = type(
        "A",
        (),
        {
            "num_actual_tokens": 16,
            "block_size": 256,
            "block_table": [0] * 8,
        },
    )()
    fn(attn_meta=cap_meta, kv_cache=_SplitCache({}), where="fwd-decode")
    if not mem.lines or "sb=u" not in mem.lines[0] or "live7=0" not in mem.lines[0]:
        print(f"[selftest] FAIL: capture should be sb=u live7=0 got {mem.lines}")
        return 1
    if " sb=s " in mem.lines[0]:
        print(f"[selftest] FAIL: capture used qlen as sb {mem.lines[0]}")
        return 1
    if "kvd1=4" not in mem.lines[0]:
        print(f"[selftest] FAIL: capture kvd1 {mem.lines[0]}")
        return 1
    curl_meta = type(
        "A",
        (),
        {
            "num_actual_tokens": 1,
            "block_size": 256,
            "block_table": [1] + [0] * 7,
        },
    )()
    n_cap = len(mem.lines)
    for i in range(50):
        fn(
            attn_meta=curl_meta,
            kv_cache=_SplitCache({1: float(i + 1), 7: float(i + 2)}),
            where="fwd-decode",
        )
    if len(mem.lines) != n_cap + 1:
        print(
            f"[selftest] FAIL: curl layers burned cap "
            f"{len(mem.lines) - n_cap} {mem.lines}"
        )
        return 1
    if "kvd1=4" not in mem.lines[-1] or "n_nz=1" not in mem.lines[-1]:
        print(f"[selftest] FAIL: curl kvd1 {mem.lines[-1]}")
        return 1
    fn(
        attn_meta=curl_meta,
        kv_cache=_SplitCache({1: 1.0}, d1=2),
        where="fwd-decode",
    )
    if len(mem.lines) != n_cap + 2 or "kvd1=2" not in mem.lines[-1]:
        print(f"[selftest] FAIL: odd kvd1 {mem.lines}")
        return 1
    swa = type("S", (), {"seq_lens": [2193], "num_decode_tokens": 1})()
    bt_meta = type(
        "A",
        (),
        {
            "num_actual_tokens": 1,
            "block_size": 256,
            "block_table": [0, 0, 7, 9, 6, 5, 0, 0],
        },
    )()
    fn(
        attn_meta=bt_meta,
        kv_cache=_SplitCache({7: 3.0, 10: 1.5}, d1=64),
        swa_meta=swa,
        where="fwd-decode",
        cr=4,
        swa_only=False,
        topk_buf=[3, 4, -1],
    )
    last = mem.lines[-1]
    if "sb=m" not in last or "seq0=2193" not in last or "live10=1" not in last:
        print(f"[selftest] FAIL: 2k swa seq/dest10 {last}")
        return 1
    if "n_nz=4" not in last or "pages=" not in last:
        print(f"[selftest] FAIL: n_nz/pages {last}")
        return 1
    if "kvd1=64" not in last or "cov_t=1024" not in last:
        print(f"[selftest] FAIL: 2k kvd1/cov_t {last}")
        return 1
    if "cr=4" not in last or "kbs=64" not in last:
        print(f"[selftest] FAIL: 2k cr/kbs {last}")
        return 1
    if "swa_only=False" not in last or "ntopk=2" not in last:
        print(f"[selftest] FAIL: 2k swa_only/ntopk {last}")
        return 1
    if "tmax=4" not in last:
        print(f"[selftest] FAIL: 2k tmax {last}")
        return 1
    n_2k = len(mem.lines)
    fn(
        attn_meta=bt_meta,
        kv_cache=_SplitCache({7: 9.0, 10: 2.0}, d1=64),
        swa_meta=swa,
        where="fwd-decode",
    )
    if len(mem.lines) != n_2k:
        print(f"[selftest] FAIL: 2k prefix re-logged {mem.lines}")
        return 1
    lg.removeHandler(mem)
    lg.propagate = prev_prop

    cns: dict = {"logger": logging.getLogger("t")}
    exec(CONN_HELPER, cns)
    n, hashed, abs_sum, prefix = cns["_dsv4_kv_page_stats"](_Cache(3.0), [1, 2, 3])
    if hashed != 3 or abs_sum <= 0:
        print(f"[selftest] FAIL: page stats n={n} hashed={hashed} sum={abs_sum}")
        return 1
    n0, h0, zsum, _p0 = cns["_dsv4_kv_page_stats"](_Cache(0.0), list(range(1, 10)))
    if h0 == 0 or zsum != 0:
        print(f"[selftest] FAIL: empty dest pages should hash to 0 got h={h0} sum={zsum}")
        return 1
    if cns["_dsv4_kv_bucket"](1) != "s" or cns["_dsv4_kv_bucket"](9) != "m":
        print("[selftest] FAIL: bucket s/m")
        return 1
    if cns["_dsv4_kv_bucket"](20) != "l":
        print("[selftest] FAIL: bucket l")
        return 1
    scan = cns["_dsv4_kv_scan_ids"](_Cache(3.0))
    if scan != list(range(1, 17)):
        print(f"[selftest] FAIL: scan ids {scan}")
        return 1
    picked = cns["_dsv4_kv_pick_group"](
        ([7, 9, 6], [10], [11], list(range(5)), [20]),
        "indexer",
        "model.layers.2.attn.indexer.k_cache",
        {},
    )
    if picked != list(range(5)):
        print(f"[selftest] FAIL: indexer group pick {picked}")
        return 1
    n_scan, n_live, max_abs = cns["_dsv4_kv_live_stats"](_Cache(3.0), 8)
    if n_live < 1 or max_abs <= 0:
        print(f"[selftest] FAIL: live census n_live={n_live} max={max_abs}")
        return 1
    if cns["_dsv4_kv_role"]("model.layers.2.attn") != "attn":
        print("[selftest] FAIL: attn role")
        return 1
    if cns["_dsv4_kv_role"]("model.layers.0.attn.swa_cache") != "swa":
        print("[selftest] FAIL: swa role")
        return 1
    if cns["_dsv4_kv_role"]("model.layers.2.attn.indexer.k_cache") != "indexer":
        print("[selftest] FAIL: indexer role")
        return 1
    if cns["_dsv4_kv_role"]("model.layers.2.attn.compressor.state_cache") is not None:
        print("[selftest] FAIL: compressor should be skipped")
        return 1
    if cns["_dsv4_kv_liveness"](0) != "z" or cns["_dsv4_kv_liveness"](1.5) != "nz":
        print("[selftest] FAIL: liveness z/nz")
        return 1
    groups = cns["_dsv4_kv_as_groups"]([7, 9, 6])
    if groups != [[7, 9, 6]]:
        print(f"[selftest] FAIL: as_groups flat {groups}")
        return 1
    fwd_ids = cns["_dsv4_kv_ids_from_forward"](
        type("F", (), {"block_tables": [0, 0, 7, 9, 6, 0]})()
    )
    if fwd_ids != [7, 9, 6]:
        print(f"[selftest] FAIL: forward block_tables trim {fwd_ids}")
        return 1

    os.environ["DSV4_KV_HASH"] = "1"
    rns: dict = {}
    exec(compile(open(runner_path).read(), runner_path, "exec"), rns)
    runner_cls = rns["GPUModelRunner"]
    inst = runner_cls()

    class _Tbl:
        def __init__(self, row, block_size=256):
            self.block_size = block_size
            self._row = list(row)

        def get_numpy_array(self):
            return [self._row]

    class _Multi:
        def __init__(self, tables):
            self.block_tables = tables

        def __getitem__(self, i):
            return self.block_tables[i]

    inst.input_batch = type("B", (), {})()
    inst.input_batch.block_table = _Multi(
        [
            _Tbl([7, 9, 6, 5, 4, 3, 2, 1, 8, 0, 0], 256),
            _Tbl([10, 11, 12, 0], 64),
        ]
    )
    inst.kv_cache_config = type("K", (), {"kv_cache_groups": [0, 1]})()
    inst.attn_groups = [
        [
            type(
                "G",
                (),
                {
                    "get_metadata_builder": lambda self, i: type(
                        "Bld", (), {"supports_update_block_table": False}
                    )()
                },
            )()
        ]
    ]
    inst.optimistic_seq_lens_cpu = [2193]
    inst.kv_caches = {
        "model.layers.2.attn": type("T", (), {"shape": (80, 64, 584)})(),
        "model.layers.3.attn": type("T", (), {"shape": (80, 2, 584)})(),
    }
    mem2 = _Mem()
    lg2 = logging.getLogger("vllm")
    lg2.setLevel(logging.INFO)
    lg2.addHandler(mem2)
    meta_live = {
        "model.layers.0.attn": type(
            "M", (), {"block_table": [7, 9, 6, 0], "block_size": 256}
        )(),
        "model.layers.3.attn": type(
            "M", (), {"block_table": [7, 9, 6, 0], "block_size": 256}
        )(),
        "model.layers.0.attn.swa_cache": type(
            "M", (), {"block_table": [10, 11, 12]}
        )(),
    }
    inst._dsv4_kv_runner_bt_log(meta_live, 1, 1, False)
    if not mem2.lines or "extra=runner-bt" not in mem2.lines[-1]:
        print(f"[selftest] FAIL: runner-bt 2k not logged {mem2.lines}")
        return 1
    live = mem2.lines[-1]
    if "hit7=1" not in live or "g0_nnz=9" not in live or "cap=0" not in live:
        print(f"[selftest] FAIL: runner-bt 2k dests {live}")
        return 1
    if "dec_cg=-" not in live:
        print(f"[selftest] FAIL: runner-bt missing dec_cg {live}")
        return 1
    if "g0slot=" not in live:
        print(f"[selftest] FAIL: runner-bt missing g0slot {live}")
        return 1
    if "e=64r4/o=2r128" not in live:
        print(f"[selftest] FAIL: runner-bt even/odd dim {live}")
        return 1
    if "cov_s=576,18" not in live or "cov_t=2304" not in live:
        print(f"[selftest] FAIL: runner-bt coverage {live}")
        return 1
    if "need=2193" not in live or "pg=8" not in live or "sl=144" not in live:
        print(f"[selftest] FAIL: runner-bt last-token slot {live}")
        return 1
    if "sl4=36" not in live or "sl128=1" not in live:
        print(f"[selftest] FAIL: runner-bt compressed slot {live}")
        return 1
    if "meta_bs=256,256" not in live:
        print(f"[selftest] FAIL: runner-bt meta_bs {live}")
        return 1
    os.environ["DSV4_EAGER"] = "1"
    inst._dsv4_kv_bt_seen = set()
    inst._dsv4_kv_runner_bt_log(meta_live, 1, 1, False)
    eager_line = mem2.lines[-1]
    if "dec_cg=NONE" not in eager_line:
        print(f"[selftest] FAIL: DSV4_EAGER not NONE {eager_line}")
        return 1
    os.environ.pop("DSV4_EAGER", None)
    if "meta_attn=[7, 9, 6]" not in live and "meta_attn=[7, 9, 6" not in live:
        if "7, 9, 6" not in live:
            print(f"[selftest] FAIL: runner-bt meta_attn {live}")
            return 1
    inst.optimistic_seq_lens_cpu = [2194]
    n_bt = len(mem2.lines)
    inst._dsv4_kv_runner_bt_log(meta_live, 1, 1, False)
    if len(mem2.lines) != n_bt:
        print(f"[selftest] FAIL: seq0 increment re-logged {mem2.lines}")
        return 1
    cap_meta = {
        "model.layers.0.attn": type("M", (), {"block_table": [0, 0, 0, 0]})(),
    }
    inst.input_batch.block_table = _Multi([_Tbl([0, 0, 0, 0], 256)])
    inst._dsv4_kv_runner_bt_log(cap_meta, 8, 16, True)
    if len(mem2.lines) < 2 or "cap=1" not in mem2.lines[-1]:
        print(f"[selftest] FAIL: capture runner-bt hidden {mem2.lines}")
        return 1
    if "g0_nnz=0" not in mem2.lines[-1]:
        print(f"[selftest] FAIL: capture should be empty table {mem2.lines[-1]}")
        return 1

    print(
        "[selftest] OK (src s/m/l + zip-remote g0/nested + pair-seal "
        "load/post-seal/alloc0 + dest census g0/swa + curl/2k/16k + "
        "unwrap/forward/scan z-nz + mla-fwd-live + runner-bt + dec_cg + g0slot + kvd1 + cr + tmax)"
    )
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
