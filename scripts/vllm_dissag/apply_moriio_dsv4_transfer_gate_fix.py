#!/usr/bin/env python3
"""Flash WRITE completion: only transfer layers that match global block size.

5a4c already seals on scheduled writes (not num_layers). Two leftovers from
217546 still break DP1:

1. wait_for_save dumps every registered cache, including any 64-block .attn
   that skip-swa missed. Rank 0 finished ~38s; rank 1 hung ~406s then WRONG.
2. No INFO on writes_done / writes_expected / notify port, so DP0 vs DP1
   was invisible. GLM gate/engine patchers no-op on 5a4c (geometry_key +
   writes_expected already native).

Fix (Flash-only):
  - Connector: num_transfer_layers = caches whose geometry.block_size matches
    the global block_size (256 after first MLA/swa_cache). Log it.
  - wait_for_save: save only those layers, then seal.
  - Engine: INFO on seal + finalize (writes_done, expected, decode_dp, port).

Per-layer offsets already exist on 5a4c (_compute_block_transfer_offsets +
geometry_key). Do not apply GLM dualkv/engine/gate here.

Gated in connectors/moriio.sh to DeepSeek-V4-Flash-FP8. Idempotent.
Missing wait_for_save dump loop is a hard error (would keep the 217546 hang).

Usage: apply_moriio_dsv4_transfer_gate_fix.py <vllm_install_dir>
"""
import os
import sys

CONN_REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"
ENG_REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_engine.py"


def patch_connector(path: str) -> int:
    src = open(path).read()
    orig = src
    applied = []

    h1_vanilla = "        self.num_layers = len(self.kv_caches.keys())\n"
    h1_gate_no_idx = """        self.num_layers = len(self.kv_caches.keys())
        # DSV4-GATE: 217546. Count caches that share the global (MLA/swa)
        # block_size. 64-block .attn is not a transfer layer.
        self.num_transfer_layers = sum(
            1
            for _ln in self.kv_caches
            if self._get_layer_transfer_geometry(_ln).block_size == self.block_size
        ) or self.num_layers
        logger.info(
            "[dsv4-gate] completion gate: num_transfer_layers=%d (num_layers=%d)",
            self.num_transfer_layers,
            self.num_layers,
        )
"""
    h1_new = """        self.num_layers = len(self.kv_caches.keys())
        # DSV4-GATE: 217546 skip .attn / block!=256 (DP1 hang). 217981 NIAH
        # 0/10: skip-register dropped .indexer. (block 64). Count MLA 256
        # plus Lightning Indexer. Still exclude SWA .attn and block-2.
        self.num_transfer_layers = sum(
            1
            for _ln in self.kv_caches
            if ".indexer." in _ln
            or (
                not _ln.endswith(".attn")
                and self._get_layer_transfer_geometry(_ln).block_size
                == self.block_size
            )
        ) or self.num_layers
        logger.info(
            "[dsv4-gate] completion gate: num_transfer_layers=%d "
            "(num_layers=%d indexer=%d)",
            self.num_transfer_layers,
            self.num_layers,
            sum(1 for _ln in self.kv_caches if ".indexer." in _ln),
        )
"""
    if "indexer=%d" in src and "[dsv4-gate] completion gate" in src:
        applied.append("num_transfer_layers (already)")
    elif h1_gate_no_idx in src:
        src = src.replace(h1_gate_no_idx, h1_new, 1)
        applied.append("num_transfer_layers-from-mla-only")
    elif h1_vanilla in src:
        src = src.replace(h1_vanilla, h1_new, 1)
        applied.append("num_transfer_layers")
    else:
        print("[dsv4-gate] ERROR: connector num_layers= anchor missing.", file=sys.stderr)
        return 1

    h2_old = """    def wait_for_save(self, metadata: MoRIIOConnectorMetadata):
        if self.mode == MoRIIOMode.WRITE and self.is_producer:
            for layer_name, kv_layer in self.kv_caches.items():
                self.save_kv_layer(metadata, layer_name, kv_layer, None)
            self._writer.seal_pending_transfers()
"""
    h2_mla_only = """    def wait_for_save(self, metadata: MoRIIOConnectorMetadata):
        if self.mode == MoRIIOMode.WRITE and self.is_producer:
            # DSV4-GATE: 217546. Do not dump every registered cache. Hybrid
            # Flash still had 64-block .attn after indexer skip; those RDMA
            # writes hung DP1 ~406s. Only layers matching global block_size.
            _n_skip = 0
            for layer_name, kv_layer in self.kv_caches.items():
                if layer_name.endswith(".attn") or (
                    self._get_layer_transfer_geometry(layer_name).block_size
                    != self.block_size
                ):
                    _n_skip += 1
                    continue
                self.save_kv_layer(metadata, layer_name, kv_layer, None)
            if _n_skip:
                logger.info(
                    "[dsv4-gate] wait_for_save skipped %d non-transfer layers",
                    _n_skip,
                )
            self._writer.seal_pending_transfers()
"""
    h2_new = """    def wait_for_save(self, metadata: MoRIIOConnectorMetadata):
        if self.mode == MoRIIOMode.WRITE and self.is_producer:
            # DSV4-GATE: 217546. Do not dump every registered cache. Hybrid
            # Flash still had 64-block .attn after indexer skip; those RDMA
            # writes hung DP1 ~406s. WRITE MLA 256 + .indexer. (217981 cold
            # Lightning Indexer). Still skip SWA .attn and block-2.
            _n_skip = 0
            _n_idx = 0
            _n_mla = 0
            for layer_name, kv_layer in self.kv_caches.items():
                if ".indexer." in layer_name:
                    _n_idx += 1
                    self.save_kv_layer(metadata, layer_name, kv_layer, None)
                    continue
                if layer_name.endswith(".attn") or (
                    self._get_layer_transfer_geometry(layer_name).block_size
                    != self.block_size
                ):
                    _n_skip += 1
                    continue
                _n_mla += 1
                self.save_kv_layer(metadata, layer_name, kv_layer, None)
            logger.info(
                "[dsv4-gate] wait_for_save mla=%d indexer=%d skipped=%d",
                _n_mla,
                _n_idx,
                _n_skip,
            )
            # #region agent log
            try:
                import json as _dj, os as _do, time as _dt
                _rec = {
                    "sessionId": "8b20ac",
                    "hypothesisId": "A",
                    "location": "moriio_connector.wait_for_save",
                    "message": "dsv4 wait_for_save layer split",
                    "data": {"mla": _n_mla, "indexer": _n_idx, "skipped": _n_skip},
                    "timestamp": int(_dt.time() * 1000),
                    "runId": _do.environ.get("SLURM_JOB_ID", ""),
                }
                _jid = _do.environ.get("SLURM_JOB_ID", "")
                for _p in (
                    "/home/basem/.cursor/debug-8b20ac.log",
                    (("/run_logs/%s/debug-8b20ac.ndjson" % _jid) if _jid else ""),
                ):
                    if not _p:
                        continue
                    try:
                        _d = _do.path.dirname(_p)
                        if _d:
                            _do.makedirs(_d, exist_ok=True)
                        open(_p, "a").write(_dj.dumps(_rec) + "\\n")
                    except Exception:
                        pass
            except Exception:
                pass
            # #endregion
            if _n_skip:
                logger.info(
                    "[dsv4-gate] wait_for_save skipped %d non-transfer layers",
                    _n_skip,
                )
            self._writer.seal_pending_transfers()
"""
    if "wait_for_save mla=%d indexer=%d skipped=%d" in src:
        applied.append("wait_for_save (already)")
    elif h2_mla_only in src:
        src = src.replace(h2_mla_only, h2_new, 1)
        applied.append("wait_for_save-from-mla-only")
    elif h2_old in src:
        src = src.replace(h2_old, h2_new, 1)
        applied.append("wait_for_save")
    else:
        print(
            "[dsv4-gate] ERROR: wait_for_save all-cache dump anchor missing "
            "(would keep the 217546 DP1 hang).",
            file=sys.stderr,
        )
        for i, line in enumerate(src.splitlines(), 1):
            if "wait_for_save" in line or "seal_pending_transfers" in line:
                print(f"[dsv4-gate]   line {i}: {line.rstrip()}", file=sys.stderr)
        return 1

    if src == orig:
        print(f"[dsv4-gate] connector no changes ({', '.join(applied)}) for {path}")
        return 0
    tmp = path + ".dsv4gate"
    with open(tmp, "w") as f:
        f.write(src)
    os.replace(tmp, path)
    print(f"[dsv4-gate] connector hunks: {', '.join(applied)} in {path}")
    return 0


def patch_engine(path: str) -> int:
    src = open(path).read()
    orig = src

    h_seal_old = """                if request_info is not None:
                    request_info.writes_expected = write_count
                    pending.append((transfer_id, request_info))
"""
    h_seal_new = """                if request_info is not None:
                    request_info.writes_expected = write_count
                    pending.append((transfer_id, request_info))
                    logger.info(
                        "[dsv4-gate] seal transfer=%s writes_expected=%s "
                        "decode_dp=%s",
                        transfer_id,
                        write_count,
                        getattr(request_info, "decode_dp_rank", None),
                    )
"""
    if "[dsv4-gate] seal transfer=" in src:
        print("[dsv4-gate] engine seal log already patched -- no-op.")
    elif h_seal_old in src:
        src = src.replace(h_seal_old, h_seal_new, 1)
        print("[dsv4-gate] patched engine seal log")
    else:
        print("[dsv4-gate] WARN: engine seal log anchor missing -- skipping.")

    h_fin_old = """            expected = request_info.writes_expected
            if expected is None or request_info.writes_done < expected:
                return
"""
    h_fin_new = """            expected = request_info.writes_expected
            logger.info(
                "[dsv4-gate] finalize transfer=%s writes_done=%s expected=%s "
                "decode_dp=%s notified=%s",
                transfer_id,
                request_info.writes_done,
                expected,
                getattr(request_info, "decode_dp_rank", None),
                request_info.completion_notified,
            )
            if expected is None or request_info.writes_done < expected:
                return
"""
    if "[dsv4-gate] finalize transfer=" in src:
        print("[dsv4-gate] engine finalize log already patched -- no-op.")
    elif h_fin_old in src:
        src = src.replace(h_fin_old, h_fin_new, 1)
        print("[dsv4-gate] patched engine finalize log")
    else:
        print("[dsv4-gate] WARN: engine finalize log anchor missing -- skipping.")

    if src == orig:
        print(f"[dsv4-gate] engine no changes for {path}")
        return 0
    if "[dsv4-gate]" not in src:
        print("[dsv4-gate] ERROR: engine post-write missing logs.", file=sys.stderr)
        return 1
    tmp = path + ".dsv4gate"
    with open(tmp, "w") as f:
        f.write(src)
    os.replace(tmp, path)
    print(f"[dsv4-gate] engine logs written to {path}")
    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    base = sys.argv[1]
    conn = os.path.join(base, CONN_REL)
    eng = os.path.join(base, ENG_REL)
    if not os.path.isfile(conn):
        print(f"[dsv4-gate] {CONN_REL} not found -- skipping.")
        return 0
    rc = patch_connector(conn)
    if rc:
        return rc
    if os.path.isfile(eng):
        rc = patch_engine(eng)
        if rc:
            return rc
    else:
        print(f"[dsv4-gate] WARN: {ENG_REL} not found -- connector only.")

    try:
        import py_compile

        py_compile.compile(conn, doraise=True)
        if os.path.isfile(eng):
            py_compile.compile(eng, doraise=True)
        print("[dsv4-gate] py_compile OK")
    except Exception as e:  # noqa: BLE001
        print(f"[dsv4-gate] ERROR: compile failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
