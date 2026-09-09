#!/usr/bin/env python3
"""Runtime INFO logs in MoRIIO connector for PD WRITE hangs (job 216066/216106).

No behavior change. Idempotent. Lenient if an anchor is missing (newer nightly).

What we need to see in prefill_NODE0.log / decode_NODE*.log:

  [pd-debug] handshake dial host:port expected=... remote_dp=...
      all 16 decode DP ranks, including child-pod IPs. Missing lines after
      the first 8 = child handshake hung (write_ready never sets).
  [pd-debug] decode update_state_after_alloc do_remote_prefill ...
      decode HTTP reached the scheduler. Silence = POST never entered engine.
  [pd-debug] decode notify should=...
      WRITE notify decision (rank 0 should be true).
  [pd-debug] save_kv_layer waiting write_ready_flags=...
      prefill entered the wait that times out today.
  [pd-debug] handshake ALL-DP done ... write_ready=True
      all-to-all handshake finished; absence means wait_all_dp never returned.

Usage: apply_moriio_pd_debug.py <vllm_install_dir>
"""
from __future__ import annotations

import os
import sys

REL = "distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py"


def _patch(src: str, old: str, new: str, already: str, label: str) -> str:
    if already in src:
        print(f"[pd-debug] skip {label}: already present")
        return src
    if old not in src:
        print(f"[pd-debug] WARN: {label} anchor not found (ok on a newer nightly)")
        return src
    print(f"[pd-debug] patch {label}")
    return src.replace(old, new, 1)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <vllm_install_dir>", file=sys.stderr)
        return 2
    path = os.path.join(sys.argv[1], REL)
    if not os.path.isfile(path):
        print(f"[pd-debug] {REL} not found -- skipping.")
        return 0

    src = open(path).read()
    orig = src

    src = _patch(
        src,
        '        logger.debug("handshake Querying metadata on path: %s", path)\n',
        "        logger.info(\n"
        '            "[pd-debug] handshake dial %s expected=%s remote_dp=%s",\n'
        "            path,\n"
        "            expected_engine_id,\n"
        "            remote_dp_rank,\n"
        "        )\n",
        "[pd-debug] handshake dial",
        "handshake dial",
    )

    src = _patch(
        src,
        "                else:\n"
        "                    # WRITE mode, decode side: notify P that blocks are ready\n",
        "                else:\n"
        "                    # WRITE mode, decode side: notify P that blocks are ready\n"
        "                    logger.info(\n"
        '                        "[pd-debug] decode update_state_after_alloc "\n'
        '                        "do_remote_prefill req=%s num_ext=%s params=%s",\n'
        "                        request.request_id,\n"
        "                        num_external_tokens,\n"
        "                        dict(params),\n"
        "                    )\n",
        "[pd-debug] decode update_state_after_alloc",
        "decode update_state_after_alloc",
    )

    src = _patch(
        src,
        "                    if _should_notify:\n"
        "                        peer_zmq = get_peer_zmq_from_request_id(\n",
        "                    logger.info(\n"
        '                        "[pd-debug] decode notify should=%s global_dp=%s "\n'
        '                        "remote_dp_rank=%s override=%s req=%s",\n'
        "                        _should_notify,\n"
        "                        self._global_dp_rank,\n"
        "                        remote_dp_rank,\n"
        "                        request.kv_transfer_params.get(\n"
        '                            "remote_dp_rank_override"\n'
        "                        ),\n"
        "                        request.request_id,\n"
        "                    )\n"
        "                    if _should_notify:\n"
        "                        peer_zmq = get_peer_zmq_from_request_id(\n",
        "[pd-debug] decode notify should",
        "decode notify should",
    )

    src = _patch(
        src,
        "            _deadline = time.monotonic() + self.moriio_config.transfer_timeout\n"
        "            while True:\n",
        "            _deadline = time.monotonic() + self.moriio_config.transfer_timeout\n"
        "            logger.info(\n"
        '                "[pd-debug] save_kv_layer waiting write_ready_flags[%s] "\n'
        '                "timeout=%ss ready=%s",\n'
        "                remote_engine_id,\n"
        "                self.moriio_config.transfer_timeout,\n"
        "                remote_engine_id in self.write_ready_flags,\n"
        "            )\n"
        "            while True:\n",
        "[pd-debug] save_kv_layer waiting",
        "save_kv_layer wait",
    )

    src = _patch(
        src,
        "            def request_ready(_f: Future[Any], entry=(req_id, meta)):\n"
        '                logger.info("MoRIIO handshake done for request %s", req_id)\n',
        "            def request_ready(_f: Future[Any], entry=(req_id, meta)):\n"
        "                logger.info(\n"
        '                    "[pd-debug] handshake ALL-DP done req=%s "\n'
        '                    "remote_engine=%s write_ready=True",\n'
        "                    req_id,\n"
        "                    remote_engine_id,\n"
        "                )\n"
        '                logger.info("MoRIIO handshake done for request %s", req_id)\n',
        "[pd-debug] handshake ALL-DP done",
        "handshake all-dp done",
    )

    if src == orig:
        print(f"[pd-debug] already instrumented / nothing to do for {path}")
        return 0
    open(path, "w").write(src)
    print(f"[pd-debug] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
