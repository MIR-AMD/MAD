#!/usr/bin/env python3
"""Stdlib tests for PD proxy routing + logging. No Quart/GPU required."""
import io
import logging
import os
import unittest
from unittest.mock import patch

from moriio_pd_proxy import (
    TRANSFER_PREFIX,
    _handshake_hosts,
    build_kv_transfer_params,
    configure_logging,
    http_backend,
    http_master,
    instance_for_rank,
    next_global_rank,
    parse_args,
    pod_index,
    require_discovery,
    routing_dp_world,
    seed_instance,
    synthesize_zmq,
    upsert_registration,
    zmq_ready,
)


class RouteTests(unittest.TestCase):
    def test_2p2d_rank_to_pod(self):
        self.assertEqual(pod_index(0, 8), 0)
        self.assertEqual(pod_index(7, 8), 0)
        self.assertEqual(pod_index(8, 8), 1)
        self.assertEqual(pod_index(15, 8), 1)

    def test_4p4d_rank_to_pod(self):
        self.assertEqual(pod_index(24, 8), 3)
        self.assertEqual(pod_index(31, 8), 3)

    def test_round_robin_world(self):
        ranks = []
        counter = 0
        for _ in range(16):
            rank, counter = next_global_rank(counter, 16)
            ranks.append(rank)
        self.assertEqual(ranks, list(range(16)))
        rank, _ = next_global_rank(counter, 16)
        self.assertEqual(rank, 0)

    def test_4p2d_route_world_is_min(self):
        self.assertEqual(routing_dp_world(32, 16), 16)
        self.assertEqual(routing_dp_world(16, 32), 16)
        self.assertEqual(routing_dp_world(32, 32), 32)
        self.assertEqual(routing_dp_world(32, 32, 8), 8)
        self.assertEqual(routing_dp_world(32, 16, 8), 8)
        self.assertEqual(routing_dp_world(32, 16, 0), 16)

    def test_4p2d_round_robin_stays_in_decode_ranks(self):
        world = routing_dp_world(32, 16)
        ranks = []
        counter = 0
        for _ in range(20):
            rank, counter = next_global_rank(counter, world)
            ranks.append(rank)
        self.assertTrue(all(r < 16 for r in ranks))
        self.assertEqual(ranks[:16], list(range(16)))

    def test_parse_4p2d_per_leg_dp(self):
        with patch.dict(os.environ, {"PREFILL_DP_SIZE": "0", "DECODE_DP_SIZE": "0"}):
            cfg = parse_args(
                [
                    "--moriio-dp-size",
                    "32",
                    "--decode-dp-size",
                    "16",
                    "--prefill",
                    "http://10.0.0.0:20005",
                    "--decode",
                    "http://10.0.0.4:20005",
                ]
            )
        self.assertEqual(cfg.prefill_dp, 32)
        self.assertEqual(cfg.decode_dp, 16)
        self.assertEqual(cfg.dp_world, 16)

    def test_parse_4p4d_decode_defaults_to_prefill(self):
        with patch.dict(os.environ, {"PREFILL_DP_SIZE": "0", "DECODE_DP_SIZE": "0"}):
            cfg = parse_args(["--moriio-dp-size", "32"])
        self.assertEqual(cfg.prefill_dp, 32)
        self.assertEqual(cfg.decode_dp, 32)
        self.assertEqual(cfg.dp_world, 32)

    def test_parse_route_dp_cap_pins_head_ranks(self):
        with patch.dict(os.environ, {"PREFILL_DP_SIZE": "0", "DECODE_DP_SIZE": "0", "PROXY_ROUTE_DP": "8"}):
            cfg = parse_args(["--moriio-dp-size", "32", "--decode-dp-size", "32"])
        self.assertEqual(cfg.prefill_dp, 32)
        self.assertEqual(cfg.decode_dp, 32)
        self.assertEqual(cfg.dp_world, 8)

    def test_use_discovery_default_on(self):
        with patch.dict(
            os.environ,
            {"PREFILL_DP_SIZE": "0", "DECODE_DP_SIZE": "0"},
            clear=False,
        ):
            os.environ.pop("PROXY_USE_DISCOVERY", None)
            cfg = parse_args(["--moriio-dp-size", "16"])
        self.assertTrue(cfg.use_discovery)
        require_discovery(cfg)

    def test_no_use_discovery_is_rejected(self):
        with patch.dict(os.environ, {"PREFILL_DP_SIZE": "0", "DECODE_DP_SIZE": "0"}):
            cfg = parse_args(["--moriio-dp-size", "16", "--no-use-discovery"])
        self.assertFalse(cfg.use_discovery)
        with self.assertRaises(SystemExit):
            require_discovery(cfg)

    def test_4p2d_decode_pod_overflow_without_min(self):
        """216534: rank 16 → decode pod 2, but 2D only seeded 2 pods."""
        decode_pods = [{"http_address": f"10.0.0.{i}:20005"} for i in range(2)]
        with self.assertRaises(LookupError):
            instance_for_rank(decode_pods, 16, 8)
        self.assertEqual(
            instance_for_rank(decode_pods, 15, 8)["http_address"],
            "10.0.0.1:20005",
        )

    def test_instance_for_rank_uses_pod_index(self):
        pods = [{"http_address": f"10.0.0.{i}:20005"} for i in range(2)]
        self.assertEqual(instance_for_rank(pods, 0, 8)["http_address"], "10.0.0.0:20005")
        self.assertEqual(instance_for_rank(pods, 8, 8)["http_address"], "10.0.0.1:20005")

    def test_http_goes_to_zmq_master_not_headless_child(self):
        pods = [
            {
                "http_address": "10.0.0.0:20005",
                "zmq_address": "host:10.0.0.0,handshake:8405,notify:61005",
            },
            {"http_address": "10.0.0.1:20005", "zmq_address": None},
        ]
        self.assertEqual(http_master(pods)["http_address"], "10.0.0.0:20005")
        self.assertEqual(instance_for_rank(pods, 8, 8)["http_address"], "10.0.0.1:20005")
        # Headless child has no ZMQ → still POST to the master (216966).
        self.assertEqual(http_backend(pods, 8, 8)["http_address"], "10.0.0.0:20005")

    def test_http_backend_posts_to_child_when_child_has_zmq(self):
        pods = [
            {
                "http_address": "10.0.0.0:20005",
                "zmq_address": "host:10.0.0.0,handshake:8405,notify:61005",
            },
            {
                "http_address": "10.0.0.1:20005",
                "zmq_address": "host:10.0.0.1,handshake:8405,notify:61005",
            },
        ]
        self.assertEqual(http_backend(pods, 0, 8)["http_address"], "10.0.0.0:20005")
        self.assertEqual(http_backend(pods, 8, 8)["http_address"], "10.0.0.1:20005")

    def test_handshake_hosts_repeat_master_by_default(self):
        pods = [
            {
                "http_address": "10.0.0.0:20005",
                "zmq_address": "host:10.0.0.0,handshake:8405,notify:61005",
            },
            {"http_address": "10.0.0.1:20005", "zmq_address": None},
        ]
        self.assertEqual(_handshake_hosts(pods), ["10.0.0.0", "10.0.0.0"])

    def test_handshake_hosts_per_pod_when_all_zmq(self):
        pods = [
            {
                "http_address": "10.0.0.0:20005",
                "zmq_address": "host:10.0.0.0,handshake:8405,notify:61005",
            },
            {
                "http_address": "10.0.0.1:20005",
                "zmq_address": "host:10.0.0.1,handshake:8405,notify:61005",
            },
        ]
        with patch.dict(os.environ, {"PROXY_HANDSHAKE_PER_POD": "1"}):
            self.assertEqual(_handshake_hosts(pods), ["10.0.0.0", "10.0.0.1"])

    def test_parse_max_concurrency_default_512(self):
        with patch.dict(os.environ, {"PREFILL_DP_SIZE": "0", "DECODE_DP_SIZE": "0"}):
            cfg = parse_args(["--moriio-dp-size", "16"])
        self.assertEqual(cfg.max_concurrency, 512)

    def test_route_cap_zero_is_full_world(self):
        self.assertEqual(routing_dp_world(16, 16, 0), 16)
        self.assertEqual(routing_dp_world(16, 16, 8), 8)

    def test_synthesize_zmq_keeps_master_host_for_headless_child(self):
        child = {"http_address": "10.0.0.1:20005", "zmq_address": None}
        master_zmq = "host:10.0.0.0,handshake:8405,notify:61005"
        self.assertEqual(synthesize_zmq(child, master_zmq), master_zmq)
        self.assertEqual(synthesize_zmq(http_master([
            {"http_address": "10.0.0.0:20005", "zmq_address": master_zmq},
        ]), master_zmq), master_zmq)

    def test_zmq_ready_headless_children(self):
        p0 = seed_instance("http://10.0.0.0:20005", "P", 16)
        p1 = seed_instance("http://10.0.0.1:20005", "P", 16)
        self.assertFalse(zmq_ready([p0, p1], 2))
        p0["zmq_address"] = "host:10.0.0.0,handshake:8405,notify:61005"
        self.assertTrue(zmq_ready([p0, p1], 2))
        self.assertFalse(zmq_ready([p0], 2))

    def test_zmq_ready_child_http_waits_for_every_pod(self):
        p0 = seed_instance("http://10.0.0.0:20005", "P", 16)
        p1 = seed_instance("http://10.0.0.1:20005", "P", 16)
        p0["zmq_address"] = "host:10.0.0.0,handshake:8405,notify:61005"
        with patch.dict(os.environ, {"MORIIO_CHILD_HTTP": "1"}):
            self.assertFalse(zmq_ready([p0, p1], 2))
            p1["zmq_address"] = "host:10.0.0.1,handshake:8405,notify:61005"
            self.assertTrue(zmq_ready([p0, p1], 2))

    def test_upsert_heartbeat_is_unchanged(self):
        target = [seed_instance("http://10.0.0.0:20005", "P", 16)]
        inst = {
            "role": "P",
            "request_address": "http://10.0.0.0:20005/v1",
            "http_address": "10.0.0.0:20005",
            "zmq_address": "host:10.0.0.0,handshake:8405,notify:61005",
            "dp_size": 16,
            "tp_size": 1,
            "transfer_mode": "WRITE",
        }
        self.assertEqual(upsert_registration(target, inst), "registered")
        self.assertEqual(upsert_registration(target, inst), "unchanged")

    def test_kv_params_have_override(self):
        remotes = [
            {
                "http_address": "10.0.0.2:20005",
                "zmq_address": "host:10.0.0.2,handshake:8405,notify:61005",
            },
            {"http_address": "10.0.0.3:20005", "zmq_address": None},
        ]
        params = build_kv_transfer_params(
            role="prefill",
            transfer_id="tx-1",
            remote_instances=remotes,
            remote_dp_size=16,
            remote_dp_size_local=8,
            remote_dp_rank=9,
            remote_tp_size=1,
        )
        self.assertTrue(params["remote_dp_rank_override"])
        self.assertEqual(params["remote_dp_rank"], 9)
        self.assertEqual(params["remote_dp_size"], 16)
        self.assertEqual(params["remote_dp_size_local"], 8)
        # Handshake :8405 is on the master; repeat that host per pod slot.
        self.assertEqual(params["remote_hosts"], ["10.0.0.2", "10.0.0.2"])


class LoggingTests(unittest.TestCase):
    def test_info_hides_debug(self):
        buf = io.StringIO()
        configure_logging("INFO")
        handler = logging.StreamHandler(buf)
        handler.setLevel(logging.DEBUG)
        log = logging.getLogger("moriio_pd_proxy")
        log.addHandler(handler)
        try:
            log.debug("hidden-debug")
            log.info("visible-info")
        finally:
            log.removeHandler(handler)
        self.assertNotIn("hidden-debug", buf.getvalue())
        # root is INFO so the extra handler at DEBUG still won't emit debug
        # unless the logger is DEBUG. configure_logging(INFO) sets logger INFO.
        self.assertEqual(log.level, logging.INFO)

    def test_debug_enables_debug(self):
        configure_logging("INFO", debug=True)
        self.assertEqual(logging.getLogger("moriio_pd_proxy").level, logging.DEBUG)
        self.assertEqual(logging.getLogger().level, logging.DEBUG)

    def test_http_debug_middleware_exists(self):
        from moriio_http_debug import log_http

        self.assertTrue(callable(log_http))

    def test_headers_redact_bearer(self):
        from moriio_pd_proxy import _headers_for_log, _json_preview, _instance_dump

        self.assertEqual(
            _headers_for_log({"Authorization": "Bearer secret", "X-Request-Id": "abc"})[
                "Authorization"
            ],
            "Bearer <redacted>",
        )
        self.assertIn("...", _json_preview({"a": "x" * 50}, limit=20))
        dump = _instance_dump(
            [{"http_address": "10.0.0.1:20005", "zmq_address": "host:10.0.0.1"}]
        )
        self.assertIn("10.0.0.1:20005", dump)


class SseFrameTests(unittest.TestCase):
    def test_client_wants_stream(self):
        from moriio_pd_proxy import _client_wants_stream

        self.assertFalse(_client_wants_stream({}))
        self.assertFalse(_client_wants_stream({"stream": False}))
        self.assertTrue(_client_wants_stream({"stream": True}))
        self.assertTrue(_client_wants_stream({"stream": "true"}))

    def test_looks_like_sse(self):
        from moriio_pd_proxy import _looks_like_sse

        self.assertTrue(_looks_like_sse(b"data: {\"a\":1}\n\n"))
        self.assertTrue(_looks_like_sse(b"  data: [DONE]\n\n"))
        self.assertFalse(_looks_like_sse(b'{"id":"cmpl-1","choices":[]}'))
        self.assertFalse(_looks_like_sse(b""))

    def test_json_body_to_sse_wraps_completion(self):
        from moriio_pd_proxy import _json_body_to_sse

        body = b'{"id":"cmpl-1","object":"text_completion","choices":[{"text":"elephant","index":0}]}'
        out = _json_body_to_sse(body)
        self.assertTrue(out.startswith(b"data: {"))
        self.assertIn(b'"text":"elephant"', out)
        self.assertTrue(out.endswith(b"data: [DONE]\n\n"))
        # NIAH scorer: only data: lines
        texts = []
        for line in out.splitlines():
            if not line.startswith(b"data:"):
                continue
            chunk = line[5:].strip()
            if chunk == b"[DONE]":
                break
            texts.append(__import__("json").loads(chunk)["choices"][0]["text"])
        self.assertEqual("".join(texts), "elephant")

    def test_json_body_to_sse_passthrough_already_sse(self):
        from moriio_pd_proxy import _json_body_to_sse

        sse = b'data: {"choices":[{"text":"giraffe"}]}\n\ndata: [DONE]\n\n'
        self.assertEqual(_json_body_to_sse(sse), sse)

    def test_empty_body_is_done_only(self):
        from moriio_pd_proxy import _json_body_to_sse

        self.assertEqual(_json_body_to_sse(b"  "), b"data: [DONE]\n\n")


class TransferIdTests(unittest.TestCase):
    def test_transfer_prefix_defined(self):
        self.assertEqual(TRANSFER_PREFIX, "tx")
        self.assertTrue(f"{TRANSFER_PREFIX}-deadbeef".startswith("tx-"))


if __name__ == "__main__":
    unittest.main()
