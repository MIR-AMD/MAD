#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""MAD Python PD proxy — Ravi MoRIIO semantics on Quart + Hypercorn.

Replaces the in-image ``moriio_toy_proxy_server.py`` for Wide-EP 2P/2D and
4P/4D. HTTP + KV-ready notify only; KV bytes stay on MoRI RDMA.

Ravi contract (see UPSTREAM_ROUTER_ANALYSIS.md):
  * round-robin over ranks that exist on BOTH legs: min(prefill DP, decode DP).
    ``--moriio-dp-size`` is prefill DP; ``--decode-dp-size`` is decode DP.
    4P/2D (32/16) must not pin rank 16–31 (216534/216576 PARSE_ERROR flood).
  * map global rank → pod via ``--dp-size-local`` (not ``n_instances * dp_size``)
  * pin both legs with ``X-data-parallel-rank`` + ``remote_dp_rank``
  * set ``remote_dp_rank_override=true`` so the connector does not re-hash
  * each leg's ``remote_dp_size`` is the *remote* role's DP (not always prefill)
  * advertise ``remote_hosts`` / ``remote_dp_size_local`` for multi-pod WRITE
  * POST HTTP to the (non-headless) P/D masters; rank 8+ is X-data-parallel-rank
  * handshake/notify ZMQ and remote_hosts stay on the HTTP master (children do
    not bind :8405 — 213116). Child CLI URLs are topology only.
  * /ready when each role's HTTP master has ZMQ'd (headless children never register)
  * client ``stream=true``: OpenAI SSE to the client (``data:`` + ``[DONE]``).
    This image's PD decode often returns one JSON object even when stream is
    set; wrap that JSON as a single SSE event. Already-SSE decode is forwarded.
    ``stream=false`` (curl / ITL) stays a JSON body.

Concurrency: Hypercorn backlog 4096, one shared aiohttp session, asyncio
overlap across requests (Ravi concurrent-prefill equivalent). Default cap is
512 in-flight client POSTs (``--max-concurrency`` / ``PROXY_MAX_CONCURRENCY``).
Toy ``app.run()`` dropped ~16% at c=512; Hypercorn must queue, not RST.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("moriio_pd_proxy")

# Prefixed onto uuid4 so proxy logs / KV handshake can grep a transfer.
TRANSFER_PREFIX = "tx"

def _env_int(name: str, default: int = 0) -> int:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = True) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return default
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return default


_LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def configure_logging(level_name: str, *, debug: bool = False) -> int:
    """INFO for job logs; DEBUG for rank/KV/ZMQ detail.

    Quart ``app.debug`` stays False either way — that path is what RST'd
    ~16% of connections at c=512. ``--debug`` / ``--log-level DEBUG`` only
    raises log verbosity.
    """
    if debug:
        level_name = "DEBUG"
    level = _LOG_LEVELS.get(str(level_name).strip().lower(), logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(level)
    stream.setFormatter(fmt)
    root.addHandler(stream)
    log_file = os.environ.get("PROXY_LOG_FILE", "").strip()
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
    for name in (
        "moriio_pd_proxy",
        "quart",
        "quart.app",
        "hypercorn",
        "hypercorn.error",
        "hypercorn.access",
        "aiohttp",
        "aiohttp.access",
        "aiohttp.client",
        "werkzeug",
    ):
        logging.getLogger(name).setLevel(level)
    logging.getLogger("asyncio").setLevel(max(level, logging.INFO))
    logger.info(
        "logging configured level=%s handlers=stdout%s "
        "(Quart app.debug=False; use --log-level DEBUG or --debug / PROXY_LOG_LEVEL)",
        logging.getLevelName(level),
        f"+file:{log_file}" if log_file else "",
    )
    if level > logging.DEBUG:
        # DEBUG is the standing default because this proxy exists for EP
        # debugging. Anything quieter drops per-request kv_transfer_params, the
        # rank map and ZMQ HELLO -- i.e. exactly what a wedged rank is diagnosed
        # from. Loud, not fatal: INFO is the right choice once the PD path is
        # resolved and a 4P/4D sweep would otherwise write tens of MB.
        logger.warning(
            "level=%s is QUIETER than DEBUG: per-request kv_transfer_params, "
            "the rank map and ZMQ HELLO are suppressed. EP debugging wants "
            "DEBUG (unset PROXY_LOG_LEVEL, or pass --debug).",
            logging.getLevelName(level),
        )
    return level


def _headers_for_log(headers: dict[str, str]) -> dict[str, str]:
    out = dict(headers)
    if out.get("Authorization"):
        tok = out["Authorization"]
        out["Authorization"] = (
            "Bearer <redacted>" if tok.startswith("Bearer ") else "<redacted>"
        )
    return out


def _json_preview(obj: Any, limit: int = 4000) -> str:
    try:
        text = json.dumps(obj, default=str)
    except Exception as exc:
        return f"<unserializable {type(obj).__name__}: {exc}>"
    if len(text) > limit:
        return text[:limit] + f"... ({len(text)} bytes)"
    return text


def _client_wants_stream(req: dict[str, Any] | None) -> bool:
    """True when the OpenAI client asked for SSE (NIAH / chat UIs)."""
    if not req:
        return False
    v = req.get("stream", False)
    return v is True or v == 1 or (isinstance(v, str) and v.strip().lower() in ("1", "true", "yes"))


def _looks_like_sse(buf: bytes) -> bool:
    """Decode body is already Server-Sent Events (pass through)."""
    s = buf.lstrip()
    return s.startswith((b"data:", b"event:", b"id:", b"retry:", b":"))


def _json_body_to_sse(body: bytes) -> bytes:
    """One JSON completion object → one OpenAI SSE event + [DONE].

    NIAH (vllm #47042) only concatenates lines that start with ``data:``.
    """
    payload = body.strip()
    if not payload:
        return b"data: [DONE]\n\n"
    if _looks_like_sse(payload):
        if not payload.endswith(b"\n\n"):
            payload += b"\n\n"
        if b"data: [DONE]" not in payload:
            payload += b"data: [DONE]\n\n"
        return payload
    try:
        payload = json.dumps(json.loads(payload), separators=(",", ":")).encode()
    except Exception:
        pass
    return b"data: " + payload + b"\n\n" + b"data: [DONE]\n\n"


async def _stall_watch(
    label: str,
    stop: asyncio.Event,
    interval: float = 10.0,
    last_progress: list[float] | None = None,
) -> None:
    """WARN after ``interval`` s with no progress, not wall time since start.

    Prefill POST leaves ``last_progress`` None — headers should land in <1s,
    so a 10s wait is a real hang. Decode STREAM must pass a one-slot list
    updated on every byte: Flash 1024-token decode is ~0.5 s/tok (~10 min)
    with first_chunk at ~3s (434017). The old wall-clock watch logged 801
    STALL lines on a healthy SSE stream.
    """
    started = time.monotonic()
    if last_progress is not None and not last_progress:
        last_progress.append(started)
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
            return
        except asyncio.TimeoutError:
            now = time.monotonic()
            idle = now - (last_progress[0] if last_progress is not None else started)
            if idle + 1e-6 < interval:
                continue
            logger.warning(
                "STALL %s still waiting after %.1fs idle", label, idle
            )


def _fail_source(side: str, tx: str, detail: str, exc: BaseException | None = None) -> None:
    """One-line locator: grep FAIL_SOURCE=prefill|decode|proxy in proxy_NODE0.log.

    Pass ``exc`` when the failure is about to be re-raised. The outer handler
    reads the tag back instead of re-deriving the side from the message text,
    which used to emit a second, contradicting FAIL_SOURCE: 218328 logged
    FAIL_SOURCE=decode and then FAIL_SOURCE=proxy for one refused connect,
    because "Cannot connect to host <ip>:20005" names neither leg.
    """
    logger.error("FAIL_SOURCE=%s tx=%s %s", side, tx, detail)
    if exc is not None and getattr(exc, "_pd_side", None) is None:
        exc._pd_side = side  # type: ignore[attr-defined]


def _error_payload(status: int, side: str, etype: str, message: str, tx: str) -> tuple:
    """Machine-readable error naming the responsible leg.

    Every non-2xx must say which leg to blame, because the harness reads it to
    decide whether a run is a model result or a deployment failure. 218328
    returned ``text/html`` "Service Unavailable: prefill_http=0 ..." for a
    *proxy* readiness failure, and the harness blamed prefill purely because
    the word appeared in the sentence.
    """
    return (
        json.dumps(
            {
                "error": {
                    "message": message,
                    "type": etype,
                    "fail_side": side,
                    "transfer_id": tx,
                }
            }
        ),
        status,
        {"Content-Type": "application/json"},
    )


def _upstream_unreachable(exc: BaseException) -> bool:
    """True when we never got a response from an upstream leg.

    aiohttp raises ClientConnectorError/ServerDisconnectedError (both
    ClientError, and OSError for the connector case) before any headers
    arrive. That is a dead or unreachable upstream -- a gateway condition,
    not a bug in this proxy -- so it must not be reported as 500.
    """
    if isinstance(exc, (ConnectionError, asyncio.TimeoutError)):
        return True
    # aiohttp is a hard dependency of this proxy, but keep the import local so
    # a missing wheel degrades to "call it a 500" rather than breaking routing.
    try:
        import aiohttp
    except Exception:  # noqa: BLE001
        return False
    return isinstance(exc, (aiohttp.ClientConnectorError, aiohttp.ServerDisconnectedError))


def _instance_dump(instances: list[dict[str, Any]]) -> str:
    rows = []
    for i, inst in enumerate(instances):
        rows.append(
            f"{i}:http={inst.get('http_address')} zmq={inst.get('zmq_address')} "
            f"req={inst.get('request_address')} dp={inst.get('dp_size')} "
            f"tp={inst.get('tp_size')} mode={inst.get('transfer_mode')}"
        )
    return "[" + "; ".join(rows) + "]" if rows else "[]"


_REQUIRED_IMPORTS = (
    ("quart", "quart"),
    ("hypercorn", "hypercorn"),
    ("aiohttp", "aiohttp"),
    ("zmq", "pyzmq"),
    ("msgpack", "msgpack"),
)


def check_deps() -> None:
    """Fail fast with a pip line if Quart/Hypercorn/aiohttp/pyzmq/msgpack are missing."""
    missing: list[str] = []
    found: list[str] = []
    for mod, pip_name in _REQUIRED_IMPORTS:
        try:
            m = __import__(mod)
            found.append(f"{pip_name}={getattr(m, '__version__', 'unknown')}")
        except ImportError:
            missing.append(pip_name)
    if missing:
        req = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
        raise SystemExit(
            "missing Python deps: "
            + ", ".join(missing)
            + f"\nInstall with: pip install -r {req}"
        )
    logger.info("proxy deps: %s", " ".join(found))


# ---------------------------------------------------------------------------
# Routing (stdlib only — unit-tested without Quart)
# ---------------------------------------------------------------------------

def hostport_from_url(url: str) -> str:
    """Normalize ``http://10.0.0.1:20005/v1`` or ``10.0.0.1:20005`` to host:port."""
    raw = (url or "").strip()
    if not raw:
        raise ValueError("empty URL")
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urlparse(raw)
    host = parsed.hostname
    port = parsed.port
    if not host:
        raise ValueError(f"cannot parse host from {url!r}")
    if port is None:
        port = 80 if parsed.scheme != "https" else 443
    return f"{host}:{port}"


def request_address(http_address: str) -> str:
    hp = hostport_from_url(http_address)
    return f"http://{hp}/v1"


def pod_index(global_dp_rank: int, dp_size_local: int) -> int:
    """Pod index a global DP rank lives on. ``dp_size_local==0`` → single pod."""
    if dp_size_local <= 0:
        return 0
    return global_dp_rank // dp_size_local


def instance_for_rank(
    instances: list[dict[str, Any]], global_rank: int, dp_size_local: int
) -> dict[str, Any]:
    """Pod row for a global rank (CLI order). HTTP still goes to http_master()."""
    if not instances:
        raise LookupError("no instances registered")
    idx = pod_index(global_rank, dp_size_local)
    if idx >= len(instances):
        raise LookupError(
            f"global rank {global_rank} maps to pod {idx} but only "
            f"{len(instances)} instance(s) registered (dp_size_local={dp_size_local})"
        )
    return instances[idx]


def http_master(instances: list[dict[str, Any]]) -> dict[str, Any]:
    """HTTP endpoint for a DP group: the first ZMQ-registered (non-headless) pod.

    WideEP children are ``--headless`` — they never bind :20005 and never
    ZMQ-register. X-data-parallel-rank on the master API reaches those ranks.
    """
    if not instances:
        raise LookupError("no instances registered")
    for inst in instances:
        if inst.get("zmq_address"):
            return inst
    raise LookupError("no HTTP master has ZMQ-registered")


def synthesize_zmq(instance: dict[str, Any], template_zmq: str | None) -> str | None:
    """Handshake/notify ZMQ for a pod. Headless children never bind :8405.

    Format: ``host:<ip>,handshake:<port>,notify:<port>``. Use the HTTP master's
    payload as-is (Ravi / job 213116). Do not swap ``host`` to the child IP —
    recent-source then dials ``tcp://<child>:8405`` and WRITE hangs (216121).
    """
    if instance.get("zmq_address"):
        return instance["zmq_address"]
    return template_zmq


def http_backend(
    instances: list[dict[str, Any]], rank: int, dp_local: int
) -> dict[str, Any]:
    """HTTP POST target for ``rank``.

    Headless children never bind :20005 (216121). Rank 8+ then stays on the
    ZMQ master via ``X-data-parallel-rank`` and the child's connector is
    missing → ``kv=null`` (216966). If that pod has ZMQ-registered it is a
    real HTTP server (``MORIIO_CHILD_HTTP=1``): POST there so WRITE hits a
    GPU that owns MoRIIO. Otherwise fall back to the master.
    """
    pod = instance_for_rank(instances, rank, dp_local)
    if pod.get("zmq_address"):
        return pod
    return http_master(instances)


def routing_dp_world(prefill_dp: int, decode_dp: int, route_cap: int = 0) -> int:
    """Ranks that exist on both PD legs. 4P/2D → 16, not 32 (216534).

    ``route_cap`` (``--route-dp-size`` / ``PROXY_ROUTE_DP``) further limits
    round-robin. 8 pins HTTP-head GPUs so NIAH does not land on rank 8+
    ``kv=null`` (216966) while children are headless. 0 = no extra cap
    (use with ``MORIIO_CHILD_HTTP=1`` so ranks 8+ POST to a real connector).
    """
    p = max(int(prefill_dp), 1)
    d = max(int(decode_dp), 1)
    world = min(p, d)
    cap = int(route_cap or 0)
    if cap > 0:
        world = min(world, cap)
    return world


def next_global_rank(counter: int, dp_world: int) -> tuple[int, int]:
    """Return (rank, next_counter). Rank is 0 when dp_world <= 1."""
    world = max(int(dp_world), 1)
    rank = 0 if world <= 1 else (counter % world)
    return rank, counter + 1


def build_request_id(prefill_zmq: str, decode_zmq: str) -> str:
    uid = uuid.uuid4().hex
    return f"___prefill_addr_{prefill_zmq}___decode_addr_{decode_zmq}_{uid}"


def _pod_hosts(instances: list[dict[str, Any]]) -> list[str]:
    hosts: list[str] = []
    for inst in instances:
        hp = inst.get("http_address") or ""
        host = hp.split(":")[0] if hp else ""
        if host:
            hosts.append(host)
    return hosts


def _handshake_hosts(instances: list[dict[str, Any]]) -> list[str]:
    """``remote_hosts`` length stays one-per-pod so rank//dp_local indexes it.

    Default: repeat the HTTP master's IP. Headless children do not listen
    on :8405; listing their IPs makes all-to-all WRITE dial a closed port
    (216121). ``PROXY_HANDSHAKE_PER_POD=1`` (with ``MORIIO_CHILD_HTTP``)
    uses each ZMQ-registered pod's IP so rank 8+ WRITEs to that node's
    connector instead of DP-RPC into a child with ``kv=null``.
    """
    per_pod = str(os.environ.get("PROXY_HANDSHAKE_PER_POD", "")).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if per_pod:
        hosts: list[str] = []
        for inst in instances:
            hp = inst.get("http_address") or ""
            host = hp.split(":")[0] if hp else ""
            if inst.get("zmq_address") and host:
                hosts.append(host)
            else:
                hosts.append("")
        if all(hosts):
            return hosts
        master = next((h for h in hosts if h), "")
        if master:
            return [h or master for h in hosts]
        return _pod_hosts(instances)
    master = ""
    for inst in instances:
        if not inst.get("zmq_address"):
            continue
        hp = inst.get("http_address") or ""
        master = hp.split(":")[0] if hp else ""
        if master:
            break
    if not master:
        return _pod_hosts(instances)
    return [master] * len(instances)


def build_kv_transfer_params(
    *,
    role: str,
    transfer_id: str,
    remote_instances: list[dict[str, Any]],
    remote_dp_size: int,
    remote_dp_size_local: int,
    remote_dp_rank: int,
    remote_tp_size: int,
) -> dict[str, Any]:
    """WRITE-mode params for one leg. ``role`` is ``prefill`` or ``decode``."""
    if role == "prefill":
        params: dict[str, Any] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
        }
    elif role == "decode":
        params = {
            "do_remote_decode": False,
            "do_remote_prefill": True,
        }
    else:
        raise ValueError(f"role must be prefill|decode, got {role!r}")
    params.update(
        {
            "remote_engine_id": None,
            "remote_block_ids": None,
            "transfer_id": transfer_id,
            "remote_dp_size": int(remote_dp_size),
            "remote_dp_size_local": int(remote_dp_size_local),
            "remote_tp_size": int(remote_tp_size),
            "remote_dp_rank": int(remote_dp_rank),
            "remote_dp_rank_override": True,
            "remote_hosts": _handshake_hosts(remote_instances),
        }
    )
    return params


def seed_instance(url: str, role: str, dp_world: int) -> dict[str, Any]:
    hp = hostport_from_url(url)
    return {
        "role": role,
        "request_address": request_address(hp),
        "http_address": hp,
        "zmq_address": None,
        "dp_size": dp_world,
        "tp_size": 1,
        "transfer_mode": None,
    }


def upsert_registration(
    target: list[dict[str, Any]], instance: dict[str, Any]
) -> str:
    """Insert or replace by http_address.

    Returns ``registered`` (first ZMQ for this HTTP addr), ``updated`` (fields
    changed), or ``unchanged`` (API-server heartbeat; 8 masters × ~1.5s).
    """
    http_address = instance["http_address"]
    for idx, existing in enumerate(target):
        if existing.get("http_address") == http_address:
            had_zmq = bool(existing.get("zmq_address"))
            merged = dict(existing)
            merged.update(instance)
            if merged == existing:
                return "unchanged"
            target[idx] = merged
            if not had_zmq and merged.get("zmq_address"):
                return "registered"
            return "updated"
    target.append(instance)
    return "registered"


def zmq_ready(instances: list[dict[str, Any]], expected: int) -> bool:
    """HTTP master has ZMQ'd; CLI-seeded pod count is present for remote_hosts.

    Headless children never ZMQ-register. Requiring zmq on every seeded URL
    (expect P=2 D=2) blocks /ready forever on 2P/2D and 4P/4D.

    ``MORIIO_CHILD_HTTP=1`` children *do* register (they own MoRIIO). Ready
    then waits for every seeded pod so rank 8+ is not POSTed to a child
    whose handshake :8405 is still down (216121 hang).
    """
    if not instances:
        return False
    if expected > 0 and len(instances) < expected:
        return False
    n_zmq = sum(1 for i in instances if i.get("zmq_address"))
    if _env_bool("MORIIO_CHILD_HTTP", False):
        need = expected if expected > 0 else len(instances)
        return n_zmq >= need
    return n_zmq > 0


# ---------------------------------------------------------------------------
# Server (Quart / Hypercorn / aiohttp / pyzmq) — imported lazily for tests
# ---------------------------------------------------------------------------

@dataclass
class ProxyConfig:
    port: int = 10001
    discovery_port: int = 36367
    use_discovery: bool = True
    dp_world: int = 16
    prefill_dp: int = 16
    decode_dp: int = 16
    dp_size_local: int = 8
    expect_prefill: int = 0
    expect_decode: int = 0
    prefill_urls: list[str] = field(default_factory=list)
    decode_urls: list[str] = field(default_factory=list)
    listen_backlog: int = 4096
    log_level: str = "DEBUG"
    max_concurrency: int = 512


class ProxyState:
    def __init__(self, cfg: ProxyConfig):
        self.cfg = cfg
        self.lock = threading.RLock()
        self.prefill: list[dict[str, Any]] = [
            seed_instance(u, "P", cfg.prefill_dp) for u in cfg.prefill_urls
        ]
        self.decode: list[dict[str, Any]] = [
            seed_instance(u, "D", cfg.decode_dp) for u in cfg.decode_urls
        ]
        self.req_n = 0
        self.transfer_type: str | None = None
        self.session = None  # aiohttp.ClientSession
        self.limit: asyncio.Semaphore | None = None
        self.in_flight = 0

    def expect_prefill(self) -> int:
        return self.cfg.expect_prefill or len(self.cfg.prefill_urls) or 1

    def expect_decode(self) -> int:
        return self.cfg.expect_decode or len(self.cfg.decode_urls) or 1

    def is_ready(self) -> tuple[bool, str]:
        with self.lock:
            p_ok = zmq_ready(self.prefill, self.expect_prefill())
            d_ok = zmq_ready(self.decode, self.expect_decode())
            p_zmq = sum(1 for i in self.prefill if i.get("zmq_address"))
            d_zmq = sum(1 for i in self.decode if i.get("zmq_address"))
            msg = (
                f"prefill_http={p_zmq} pods={len(self.prefill)}/{self.expect_prefill()} "
                f"decode_http={d_zmq} pods={len(self.decode)}/{self.expect_decode()} "
                f"transfer={self.transfer_type}"
            )
            return p_ok and d_ok, msg

    def next_rank(self) -> int:
        with self.lock:
            rank, self.req_n = next_global_rank(self.req_n, self.cfg.dp_world)
            return rank

    def pick_rank(self, headers) -> int:
        """Choose DP rank for this HTTP request.

        Curl/ITL: round-robin. The NIAH client only talks to :10001 and must
        not know ranks — it already sends x-request-id=niah-*. Pin those to
        PROXY_NIAH_RANK (default 0) so warmup N and score N stay on one GPU.
        Explicit X-data-parallel-rank still wins (load-test pin).
        """
        raw = headers.get("X-data-parallel-rank") or headers.get(
            "x-data-parallel-rank"
        )
        if raw is not None and str(raw).strip() != "":
            try:
                pinned = int(str(raw).strip())
            except ValueError:
                logger.warning("ignore bad X-data-parallel-rank=%r; round-robin", raw)
                return self.next_rank()
            if self.cfg.dp_world <= 0:
                return 0
            rank = pinned % self.cfg.dp_world
            if rank != pinned:
                logger.warning(
                    "X-data-parallel-rank=%s wrapped to %s (dp_world=%s)",
                    pinned,
                    rank,
                    self.cfg.dp_world,
                )
            return rank
        rid = headers.get("X-Request-Id") or headers.get("x-request-id") or ""
        niah_rank = os.environ.get("PROXY_NIAH_RANK", "0")
        if str(rid).startswith("niah-") and niah_rank.strip() != "":
            try:
                rank = int(niah_rank) % max(self.cfg.dp_world, 1)
            except ValueError:
                rank = 0
            logger.info("niah pin rank=%s x-request-id=%s", rank, rid)
            return rank
        return self.next_rank()


def _listen_for_register(state: ProxyState, hostname: str, port: int) -> None:
    import msgpack
    import zmq

    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    # 217951: bind("tcp://0.0.0.0:36367") EADDRINUSE with empty /proc LISTEN.
    # Dual-stack ZMQ also binds [::] on the same port. IPv4-only.
    try:
        router_socket.setsockopt(zmq.IPV6, 0)
    except Exception:
        pass
    router_socket.bind(f"tcp://{hostname}:{port}")
    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)
    logger.info(
        "ZMQ discovery listening on tcp://%s:%s use_discovery=true "
        "(P/D register WRITE zmq + transfer_mode; overlapping P/D POSTs)",
        hostname,
        port,
    )
    hb_every = float(os.environ.get("PROXY_ZMQ_HEARTBEAT_LOG_S", "60"))
    hb_all = str(os.environ.get("PROXY_ZMQ_HEARTBEAT_LOG", "")).lower() in (
        "1",
        "all",
        "true",
        "yes",
    )
    hb_n = {"P": 0, "D": 0, "HELLO": 0}
    hb_last = {"P": 0.0, "D": 0.0, "HELLO": 0.0}

    def _hb_tick(kind: str, detail: str) -> None:
        hb_n[kind] += 1
        now = time.monotonic()
        if hb_all or now - hb_last[kind] >= hb_every:
            logger.debug(
                "ZMQ heartbeat %s n=%s in %.0fs %s",
                kind,
                hb_n[kind],
                now - hb_last[kind] if hb_last[kind] else 0.0,
                detail,
            )
            hb_n[kind] = 0
            hb_last[kind] = now

    while True:
        socks = dict(poller.poll())
        if router_socket not in socks:
            continue
        _remote_addr, msg = router_socket.recv_multipart()
        data = msgpack.loads(msg)
        kind = data.get("type")
        if kind == "HELLO":
            _hb_tick("HELLO", f"ident={_remote_addr!r}")
            continue
        if kind not in ("P", "D"):
            logger.warning("unrecognized ZMQ type %r; ignoring", kind)
            continue
        required = {
            "http_address",
            "zmq_address",
            "dp_size",
            "tp_size",
            "transfer_mode",
        }
        missing = required - data.keys()
        if missing:
            logger.error("registration missing keys %s; skip", missing)
            continue
        instance = {
            "role": kind,
            "request_address": request_address(data["http_address"]),
            "http_address": hostport_from_url(data["http_address"]),
            "zmq_address": data["zmq_address"],
            "dp_size": data["dp_size"],
            "tp_size": data["tp_size"],
            "transfer_mode": data["transfer_mode"],
        }
        with state.lock:
            mode = str(instance["transfer_mode"] or "").lower()
            if state.transfer_type is None:
                state.transfer_type = mode
                logger.info("SET TRANSFER TYPE TO %s", mode)
            elif mode and mode != state.transfer_type:
                logger.error(
                    "mismatched transfer mode: expected %s got %s; skip %s",
                    state.transfer_type,
                    mode,
                    instance["http_address"],
                )
                continue
            target = state.prefill if kind == "P" else state.decode
            action = upsert_registration(target, instance)
        role_name = "Prefill" if kind == "P" else "Decode"
        if action == "unchanged":
            _hb_tick(
                kind,
                f"{instance['http_address']} zmq={instance['zmq_address']}",
            )
            continue
        logger.debug(
            "ZMQ message ident=%r type=%s payload=%s",
            _remote_addr,
            kind,
            _json_preview(data),
        )
        logger.info(
            "%s %s instance %s zmq=%s dp_size=%s tp_size=%s mode=%s",
            action,
            role_name,
            instance["http_address"],
            instance["zmq_address"],
            instance["dp_size"],
            instance["tp_size"],
            instance["transfer_mode"],
        )
        ok, ready_msg = state.is_ready()
        logger.info("registration snapshot ready=%s %s", ok, ready_msg)


def create_app(state: ProxyState):
    import aiohttp
    from quart import Quart, make_response, request as qreq

    app = Quart(__name__)
    app.config["BODY_TIMEOUT"] = 360000
    app.config["RESPONSE_TIMEOUT"] = 360000
    app.debug = False

    @app.before_serving
    async def _startup() -> None:
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=float(os.environ.get("PROXY_CONNECT_TIMEOUT_S", "30")),
            sock_connect=float(os.environ.get("PROXY_CONNECT_TIMEOUT_S", "30")),
            sock_read=float(os.environ.get("PROXY_SOCK_READ_TIMEOUT_S", "360000")),
        )
        # 512 client POSTs × (prefill + decode) = 1024 upstream. Default
        # per-host 512 queued the second leg and looked like a hang.
        _conn_kwargs = dict(
            limit=int(os.environ.get("PROXY_HTTP_LIMIT", "2048")),
            limit_per_host=int(os.environ.get("PROXY_HTTP_LIMIT_PER_HOST", "1024")),
            ttl_dns_cache=300,
        )
        try:
            connector = aiohttp.TCPConnector(
                **_conn_kwargs, enable_cleanup_closed=True
            )
        except TypeError:
            connector = aiohttp.TCPConnector(**_conn_kwargs)
        cap = max(int(state.cfg.max_concurrency or 0), 0)
        state.limit = asyncio.Semaphore(cap) if cap > 0 else None
        state.in_flight = 0
        trace = aiohttp.TraceConfig()

        async def _on_request_start(session, ctx, params):
            ctx.start = asyncio.get_running_loop().time()
            ctx.url = str(params.url)
            ctx.first_chunk = False
            logger.debug("aiohttp request_start %s %s", params.method, params.url)

        async def _on_request_end(session, ctx, params):
            dt = asyncio.get_running_loop().time() - getattr(ctx, "start", 0.0)
            status = getattr(params.response, "status", None)
            logger.debug(
                "aiohttp request_end %s %s status=%s dt=%.3fs",
                params.method,
                params.url,
                status,
                dt,
            )

        async def _on_connection_queued_start(session, ctx, params):
            logger.debug("aiohttp connection queued url=%s", getattr(ctx, "url", "?"))

        async def _on_connection_create_start(session, ctx, params):
            logger.debug("aiohttp TCP connect start url=%s", getattr(ctx, "url", "?"))

        async def _on_connection_create_end(session, ctx, params):
            logger.debug("aiohttp TCP connect ok url=%s", getattr(ctx, "url", "?"))

        async def _on_connection_reuseconn(session, ctx, params):
            logger.debug("aiohttp TCP reuse url=%s", getattr(ctx, "url", "?"))

        async def _on_dns_end(session, ctx, params):
            logger.debug("aiohttp DNS resolved host=%s", params.host)

        async def _on_headers_sent(session, ctx, params):
            logger.debug("aiohttp request headers sent %s", params.url)

        async def _on_chunk_recv(session, ctx, params):
            if getattr(ctx, "first_chunk", False):
                return
            ctx.first_chunk = True
            n = len(params.chunk) if getattr(params, "chunk", None) is not None else 0
            logger.debug(
                "aiohttp first response chunk %s bytes url=%s",
                n,
                getattr(ctx, "url", "?"),
            )

        async def _on_request_exception(session, ctx, params):
            logger.error(
                "aiohttp request_exception %s %s: %s",
                params.method,
                params.url,
                params.exception,
            )

        for hook_name, cb in (
            ("on_request_start", _on_request_start),
            ("on_request_end", _on_request_end),
            ("on_connection_queued_start", _on_connection_queued_start),
            ("on_connection_create_start", _on_connection_create_start),
            ("on_connection_create_end", _on_connection_create_end),
            ("on_connection_reuseconn", _on_connection_reuseconn),
            ("on_dns_resolvehost_end", _on_dns_end),
            ("on_request_headers_sent", _on_headers_sent),
            ("on_response_chunk_received", _on_chunk_recv),
            ("on_request_exception", _on_request_exception),
        ):
            hook = getattr(trace, hook_name, None)
            if hook is None:
                logger.warning("aiohttp TraceConfig missing %s", hook_name)
                continue
            hook.append(cb)
        state.session = aiohttp.ClientSession(
            timeout=timeout, connector=connector, trace_configs=[trace]
        )
        logger.info(
            "aiohttp pool limit=%s per_host=%s connect_timeout=%ss sock_read=%ss",
            connector.limit,
            connector.limit_per_host,
            timeout.sock_connect,
            timeout.sock_read,
        )

    @app.after_serving
    async def _shutdown() -> None:
        if state.session is not None:
            await state.session.close()
            state.session = None

    @app.before_request
    async def _log_inbound():
            qs = qreq.query_string
            if isinstance(qs, (bytes, bytearray)):
                qs = qs.decode()
            logger.debug(
                "IN %s %s remote=%s query=%s headers=%s",
                qreq.method,
                qreq.path,
                qreq.remote_addr,
                qs,
                _headers_for_log({k: v for k, v in qreq.headers.items()}),
            )

    @app.after_request
    async def _log_outbound(response):
        logger.debug(
            "OUT %s %s -> %s content_type=%s",
            qreq.method,
            qreq.path,
            response.status_code,
            response.headers.get("Content-Type"),
        )
        return response

    @app.route("/health", methods=["GET"])
    async def health():
        logger.debug("GET /health")
        return ("ok", 200)

    @app.route("/ready", methods=["GET"])
    async def ready():
        ok, msg = state.is_ready()
        logger.debug("GET /ready -> %s %s", 200 if ok else 503, msg)
        if ok:
            return (msg, 200)
        return (f"not ready: {msg}", 503)

    @app.route("/v1/completions", methods=["POST"])
    async def completions():
        return await _gated("/completions", state)

    @app.route("/v1/chat/completions", methods=["POST"])
    async def chat_completions():
        return await _gated("/chat/completions", state)

    async def _gated(api: str, st: ProxyState):
        sem = st.limit
        if sem is None:
            return await handle_request(api, st)
        async with sem:
            st.in_flight += 1
            try:
                if st.in_flight in (1, 8, 32, 64, 128, 256, 512) or st.in_flight % 64 == 0:
                    logger.info("in_flight=%s cap=%s", st.in_flight, st.cfg.max_concurrency)
                return await handle_request(api, st)
            finally:
                st.in_flight -= 1

    async def handle_request(api: str, st: ProxyState):
        try:
            ready_ok, ready_msg = st.is_ready()
            if not ready_ok:
                logger.warning("reject %s: not ready (%s)", api, ready_msg)
                _fail_source("proxy", "-", f"not ready: {ready_msg}")
                return await make_response(
                    _error_payload(
                        503,
                        "proxy",
                        "not_ready",
                        f"proxy not ready (no registered pods): {ready_msg}",
                        "-",
                    )
                )

            req_data = await qreq.get_json()
            if not isinstance(req_data, dict):
                logger.error("client %s body is not JSON object: %s", api, type(req_data))
                return await make_response(("Bad Request: JSON object required", 400))
            prompt = req_data.get("prompt", req_data.get("messages"))
            logger.info(
                "client %s remote=%s keys=%s stream=%s max_tokens=%s prompt=%s",
                api,
                qreq.remote_addr,
                sorted(req_data.keys()),
                req_data.get("stream"),
                req_data.get("max_tokens", req_data.get("max_completion_tokens")),
                _json_preview(prompt, 300),
            )
            logger.debug("client %s full body=%s", api, _json_preview(req_data))
            rank = st.pick_rank(qreq.headers)
            logger.info(
                "route_completion called, use_discovery=true rank=%s "
                "concurrent_prefill=true in_flight=%s cap=%s rank_pin=0..%s",
                rank,
                st.in_flight,
                st.cfg.max_concurrency,
                max(st.cfg.dp_world - 1, 0),
            )
            dp_local = st.cfg.dp_size_local
            with st.lock:
                prefill_instances = list(st.prefill)
                decode_instances = list(st.decode)
                transfer_type = st.transfer_type

            logger.debug("prefill instances %s", _instance_dump(prefill_instances))
            logger.debug("decode instances %s", _instance_dump(decode_instances))

            prefill_http = http_backend(prefill_instances, rank, dp_local)
            decode_http = http_backend(decode_instances, rank, dp_local)
            prefill_pod = instance_for_rank(prefill_instances, rank, dp_local)
            decode_pod = instance_for_rank(decode_instances, rank, dp_local)
            prefill_zmq = synthesize_zmq(prefill_pod, prefill_http.get("zmq_address"))
            decode_zmq = synthesize_zmq(decode_pod, decode_http.get("zmq_address"))
            if not prefill_zmq or not decode_zmq:
                logger.error(
                    "rank=%s missing zmq prefill=%s decode=%s P=%s D=%s",
                    rank,
                    prefill_pod.get("http_address"),
                    decode_pod.get("http_address"),
                    _instance_dump(prefill_instances),
                    _instance_dump(decode_instances),
                )
                _fail_source("proxy", "-", f"selected pod missing zmq_address rank={rank}")
                return await make_response(
                    _error_payload(
                        503,
                        "proxy",
                        "pod_missing_zmq",
                        f"selected pod missing zmq_address rank={rank}",
                        "-",
                    )
                )

            request_id = build_request_id(prefill_zmq, decode_zmq)
            transfer_id = f"{TRANSFER_PREFIX}-{uuid.uuid4()}"
            header_rank = None if st.cfg.dp_world <= 1 else rank
            pod_p = pod_index(rank, dp_local)
            pod_d = pod_index(rank, dp_local)

            logger.info(
                "route api=%s rank=%s podP=%s podD=%s httpP=%s httpD=%s "
                "zmqP=%s zmqD=%s transfer=%s tx=%s req_id=%s",
                api,
                rank,
                pod_p,
                pod_d,
                prefill_http["http_address"],
                decode_http["http_address"],
                prefill_zmq,
                decode_zmq,
                transfer_type,
                transfer_id,
                request_id,
            )
            logger.debug(
                "rank map dp_world=%s prefill_dp=%s decode_dp=%s dp_local=%s "
                "header_rank=%s prefill_pod=%s decode_pod=%s "
                "prefill_master=%s decode_master=%s",
                st.cfg.dp_world,
                st.cfg.prefill_dp,
                st.cfg.decode_dp,
                dp_local,
                header_rank,
                prefill_pod.get("http_address"),
                decode_pod.get("http_address"),
                prefill_http.get("http_address"),
                decode_http.get("http_address"),
            )

            req_prefill = copy.deepcopy(req_data)
            req_prefill["kv_transfer_params"] = build_kv_transfer_params(
                role="prefill",
                transfer_id=transfer_id,
                remote_instances=decode_instances,
                remote_dp_size=st.cfg.decode_dp,
                remote_dp_size_local=dp_local,
                remote_dp_rank=rank,
                remote_tp_size=int(decode_http.get("tp_size") or 1),
            )
            req_prefill["stream"] = False
            req_prefill["max_tokens"] = 1
            if "max_completion_tokens" in req_prefill:
                req_prefill["max_completion_tokens"] = 1
            req_prefill.pop("stream_options", None)
            # Both legs rewrite max_tokens, so a client min_tokens has to follow
            # or vLLM 400s: "min_tokens must be less than or equal to
            # max_tokens=1, got 32" (224395, every request, before any compute).
            # Prefill's single token is discarded, so clamping it there costs
            # nothing; the answer comes from the decode leg, which is where the
            # ban needs to survive.
            if req_prefill.get("min_tokens") is not None:
                req_prefill["min_tokens"] = min(int(req_prefill["min_tokens"]), 1)

            req_decode = copy.deepcopy(req_data)
            if "max_completion_tokens" in req_decode:
                req_decode["max_completion_tokens"] = max(
                    0, int(req_decode["max_completion_tokens"]) - 1
                )
            elif "max_tokens" in req_decode:
                req_decode["max_tokens"] = max(0, int(req_decode["max_tokens"]) - 1)
            if req_decode.get("min_tokens") is not None:
                _dcap = req_decode.get("max_completion_tokens")
                if _dcap is None:
                    _dcap = req_decode.get("max_tokens")
                if _dcap is not None:
                    req_decode["min_tokens"] = min(
                        int(req_decode["min_tokens"]), int(_dcap)
                    )
            req_decode["kv_transfer_params"] = build_kv_transfer_params(
                role="decode",
                transfer_id=transfer_id,
                remote_instances=prefill_instances,
                remote_dp_size=st.cfg.prefill_dp,
                remote_dp_size_local=dp_local,
                remote_dp_rank=rank,
                remote_tp_size=int(prefill_http.get("tp_size") or 1),
            )

            headers = {
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
                "X-Request-Id": request_id,
            }
            if header_rank is not None:
                headers["X-data-parallel-rank"] = str(header_rank)

            logger.debug(
                "prefill kv_transfer_params=%s",
                _json_preview(req_prefill.get("kv_transfer_params")),
            )
            logger.debug(
                "decode kv_transfer_params=%s",
                _json_preview(req_decode.get("kv_transfer_params")),
            )
            logger.debug("prefill POST body=%s", _json_preview(req_prefill))
            logger.debug("decode POST body=%s", _json_preview(req_decode))
            logger.debug("upstream headers=%s", _headers_for_log(headers))

            prefill_url = prefill_http["request_address"] + api
            decode_url = decode_http["request_address"] + api
            stall_s = float(os.environ.get("PROXY_STALL_INTERVAL_S", "10"))
            logger.info(
                "WRITE=%s firing prefill=%s decode=%s X-data-parallel-rank=%s "
                "X-Request-Id=%s stall_warn=%ss",
                (transfer_type or "write") != "read",
                prefill_url,
                decode_url,
                header_rank,
                request_id,
                stall_s,
            )
            session = st.session
            assert session is not None

            async def _post_prefill():
                stop = asyncio.Event()
                watch = asyncio.create_task(
                    _stall_watch(
                        f"prefill POST {prefill_url} rank={rank} tx={transfer_id}",
                        stop,
                        stall_s,
                    )
                )
                t0 = asyncio.get_running_loop().time()
                logger.info("prefill POST START %s rank=%s tx=%s", prefill_url, rank, transfer_id)
                try:
                    async with session.post(
                        url=prefill_url, json=req_prefill, headers=headers
                    ) as resp:
                        dt = asyncio.get_running_loop().time() - t0
                        logger.info(
                            "prefill POST HEADERS status=%s dt=%.3fs %s resp_headers=%s",
                            resp.status,
                            dt,
                            prefill_url,
                            dict(resp.headers),
                        )
                        if resp.status != 200:
                            body = await resp.text()
                            _fail_source(
                                "prefill",
                                transfer_id,
                                f"HTTP {resp.status} {prefill_url} body={body[:500]!r}",
                            )
                            logger.error(
                                "prefill POST status=%s %s: %s",
                                resp.status,
                                prefill_url,
                                body[:2000],
                            )
                            err = RuntimeError(
                                f"prefill {resp.status} {resp.reason} {prefill_url}: {body[:500]}"
                            )
                            err._pd_side = "prefill"  # type: ignore[attr-defined]
                            raise err
                        payload = await resp.json()
                        logger.info(
                            "prefill POST BODY done dt=%.3fs %s keys=%s kv=%s",
                            asyncio.get_running_loop().time() - t0,
                            prefill_url,
                            sorted(payload.keys()) if isinstance(payload, dict) else type(payload),
                            _json_preview(
                                payload.get("kv_transfer_params")
                                if isinstance(payload, dict)
                                else None,
                                1000,
                            ),
                        )
                        logger.debug("prefill POST full response=%s", _json_preview(payload))
                        return payload
                finally:
                    stop.set()
                    watch.cancel()
                    try:
                        await watch
                    except asyncio.CancelledError:
                        pass

            is_read = (transfer_type or "write") == "read"
            prefill_task = None
            if is_read:
                logger.info("READ path: prefill then decode (sequential)")
                prefill_response = await _post_prefill()
                prefill_kv = prefill_response.get("kv_transfer_params") or {}
                req_decode["kv_transfer_params"]["remote_engine_id"] = prefill_kv.get(
                    "remote_engine_id"
                )
                req_decode["kv_transfer_params"]["remote_block_ids"] = prefill_kv.get(
                    "remote_block_ids"
                )
                req_decode["kv_transfer_params"]["transfer_id"] = prefill_kv.get(
                    "transfer_id", transfer_id
                )
                logger.debug(
                    "READ patched decode kv_transfer_params=%s",
                    _json_preview(req_decode["kv_transfer_params"]),
                )
            else:
                logger.info("WRITE path: prefill+decode in parallel; decode POST next")
                prefill_task = asyncio.create_task(_post_prefill())

                def _log_prefill_fail(task: asyncio.Task) -> None:
                    if task.cancelled():
                        logger.warning("prefill POST cancelled tx=%s", transfer_id)
                        return
                    err = task.exception()
                    if err is not None:
                        _fail_source("prefill", transfer_id, f"POST failed: {err}")
                        logger.error("prefill POST failed tx=%s: %s", transfer_id, err)

                prefill_task.add_done_callback(_log_prefill_fail)
                await asyncio.sleep(0)

            stop_d = asyncio.Event()
            watch_d = asyncio.create_task(
                _stall_watch(
                    f"decode POST {decode_url} rank={rank} tx={transfer_id}",
                    stop_d,
                    stall_s,
                )
            )
            t_decode = asyncio.get_running_loop().time()
            logger.info(
                "decode POST START %s rank=%s tx=%s headers=%s",
                decode_url,
                rank,
                transfer_id,
                _headers_for_log(headers),
            )
            try:
                resp = await session.post(
                    url=decode_url, json=req_decode, headers=headers
                )
            except Exception as exc:
                _fail_source(
                    "decode",
                    transfer_id,
                    f"POST raised before headers dt={asyncio.get_running_loop().time() - t_decode:.3f}s {decode_url}: {exc}",
                    exc,
                )
                logger.exception(
                    "decode POST raised before headers dt=%.3fs %s",
                    asyncio.get_running_loop().time() - t_decode,
                    decode_url,
                )
                raise
            finally:
                stop_d.set()
                watch_d.cancel()
                try:
                    await watch_d
                except asyncio.CancelledError:
                    pass
            logger.info(
                "decode POST HEADERS status=%s dt=%.3fs %s resp_headers=%s",
                resp.status,
                asyncio.get_running_loop().time() - t_decode,
                decode_url,
                dict(resp.headers),
            )

            client_stream = _client_wants_stream(req_data)

            async def _stream():
                stop_s = asyncio.Event()
                last_chunk = [time.monotonic()]
                watch_s = asyncio.create_task(
                    _stall_watch(
                        f"decode STREAM {decode_url} rank={rank} tx={transfer_id}",
                        stop_s,
                        stall_s,
                        last_chunk,
                    )
                )
                try:
                    if resp.status != 200:
                        body = await resp.text()
                        _fail_source(
                            "decode",
                            transfer_id,
                            f"HTTP {resp.status} {decode_url} body={body[:500]!r}",
                        )
                        logger.error(
                            "decode %s %s %s: %s",
                            resp.status,
                            resp.reason,
                            decode_url,
                            body[:2000],
                        )
                        err = RuntimeError(
                            f"decode {resp.status} {resp.reason} {decode_url}: {body[:500]}"
                        )
                        err._pd_side = "decode"  # type: ignore[attr-defined]
                        raise err
                    first = True
                    nbytes = 0
                    nchunks = 0
                    sse_mode = None  # None | "sse" | "json"
                    json_buf: list[bytes] = []
                    lead = b""
                    chunk_iter = resp.content.iter_chunked(1024).__aiter__()
                    prefill_pending = (not is_read) and prefill_task is not None
                    while True:
                        next_chunk = asyncio.create_task(anext(chunk_iter))
                        waiters: set[asyncio.Task] = {next_chunk}
                        if prefill_pending:
                            waiters.add(prefill_task)  # type: ignore[arg-type]
                        done, _pending = await asyncio.wait(
                            waiters, return_when=asyncio.FIRST_COMPLETED
                        )
                        if prefill_pending and prefill_task in done:
                            pref_exc = prefill_task.exception()
                            if pref_exc is not None:
                                next_chunk.cancel()
                                try:
                                    await next_chunk
                                except (asyncio.CancelledError, StopAsyncIteration, Exception):
                                    pass
                                msg = "prefill failed: %s" % pref_exc
                                _fail_source("prefill", transfer_id, msg)
                                logger.error(
                                    "aborting decode STREAM because prefill failed tx=%s: %s",
                                    transfer_id,
                                    pref_exc,
                                )
                                if client_stream:
                                    yield _json_body_to_sse(
                                        json.dumps(
                                            {
                                                "error": {
                                                    "message": msg,
                                                    "type": "InternalServerError",
                                                }
                                            }
                                        ).encode()
                                    )
                                return
                            prefill_pending = False
                            if next_chunk not in done:
                                await next_chunk
                        try:
                            chunk = next_chunk.result()
                        except StopAsyncIteration:
                            break
                        nchunks += 1
                        nbytes += len(chunk)
                        last_chunk[0] = time.monotonic()
                        if first and chunk:
                            logger.info(
                                "decode STREAM first_chunk=%s bytes preview=%r tx=%s",
                                len(chunk),
                                chunk[:200],
                                transfer_id,
                            )
                            if b"EngineCore" in chunk or b"InternalServerError" in chunk:
                                _fail_source(
                                    "decode",
                                    transfer_id,
                                    f"EngineCore/500 in first SSE/JSON chunk preview={chunk[:200]!r}",
                                )
                            first = False
                        if not client_stream:
                            yield chunk
                            continue
                        if sse_mode is None:
                            lead += chunk
                            if not lead.lstrip():
                                continue
                            if _looks_like_sse(lead):
                                sse_mode = "sse"
                                yield lead
                                lead = b""
                            else:
                                sse_mode = "json"
                                json_buf.append(lead)
                                lead = b""
                            continue
                        if sse_mode == "sse":
                            yield chunk
                        else:
                            json_buf.append(chunk)
                    if client_stream and sse_mode == "json":
                        framed = _json_body_to_sse(b"".join(json_buf))
                        logger.info(
                            "decode STREAM framed JSON as SSE in=%s out=%s tx=%s",
                            nbytes,
                            len(framed),
                            transfer_id,
                        )
                        yield framed
                    elif client_stream and sse_mode is None:
                        framed = _json_body_to_sse(lead)
                        yield framed
                    logger.info(
                        "decode STREAM done chunks=%s bytes=%s sse_mode=%s tx=%s",
                        nchunks,
                        nbytes,
                        sse_mode if client_stream else "passthrough",
                        transfer_id,
                    )
                finally:
                    stop_s.set()
                    watch_s.cancel()
                    try:
                        await watch_s
                    except asyncio.CancelledError:
                        pass
                    resp.release()
                    logger.debug("decode POST connection released tx=%s", transfer_id)

            response = await make_response(_stream())
            if client_stream:
                response.headers["Content-Type"] = "text/event-stream"
                response.headers["Cache-Control"] = "no-cache"
            else:
                response.headers["Content-Type"] = resp.headers.get(
                    "Content-Type", "application/json"
                )
            return response
        except LookupError as e:
            tx = transfer_id if "transfer_id" in locals() else "-"
            _fail_source("proxy", tx, f"routing: {e}")
            logger.exception("routing error: %s", e)
            return await make_response(
                _error_payload(503, "proxy", "routing_error", f"routing: {e}", tx)
            )
        except Exception as e:
            msg = str(e)
            tx = transfer_id if "transfer_id" in locals() else "-"
            # Trust an inner attribution; only guess when nobody claimed it.
            side = getattr(e, "_pd_side", None)
            if side is None:
                side = "prefill" if "prefill" in msg.lower() else (
                    "decode" if "decode" in msg.lower() else "proxy"
                )
                _fail_source(side, tx, msg[:500])
            logger.exception("request failed: %s", e)
            if _upstream_unreachable(e):
                # Never a 500: an unreachable upstream is a gateway condition,
                # and text/html 500 is what left 218328 reporting
                # "fail_side=proxy-or-unknown".
                return await make_response(
                    _error_payload(
                        502,
                        side,
                        "upstream_unreachable",
                        f"{side} upstream unreachable: {msg}",
                        tx,
                    )
                )
            return await make_response(
                _error_payload(
                    500,
                    side,
                    "proxy_internal_error",
                    f"Internal Server Error: {msg}",
                    tx,
                )
            )

    return app


def parse_args(argv: list[str] | None = None) -> ProxyConfig:
    p = argparse.ArgumentParser(description="MAD MoRIIO PD proxy (Ravi semantics)")
    p.add_argument("--port", type=int, default=int(os.environ.get("MORI_PROXY_PORT", "10001")))
    p.add_argument(
        "--discovery-port",
        type=int,
        default=int(os.environ.get("MORI_PROXY_PING_PORT", "36367")),
    )
    p.add_argument(
        "--use-discovery",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("PROXY_USE_DISCOVERY", True),
        help="ZMQ P/D registration (WRITE zmq_address + transfer_mode). "
        "Default on. --no-use-discovery is rejected. Rank pin stays "
        "PROXY_ROUTE_DP. Overlapping P/D POSTs are the default.",
    )
    p.add_argument(
        "--moriio-dp-size",
        type=int,
        default=int(os.environ.get("PREFILL_DP_SIZE", "0")),
        help="Prefill DP world (16 for 2P, 32 for 4P). Routing uses "
        "min(this, --decode-dp-size).",
    )
    p.add_argument(
        "--decode-dp-size",
        type=int,
        default=int(os.environ.get("DECODE_DP_SIZE", "0")),
        help="Decode DP world (16 for 2D, 32 for 4D). 0 = same as prefill. "
        "4P/2D must pass 16 so ranks 16–31 are not pinned (216534).",
    )
    p.add_argument(
        "--dp-size-local",
        type=int,
        default=int(os.environ.get("DP_PARALLEL_SIZE_LOCAL", "8")),
        help="Per-pod DP size (GPUs/node). Used to map global rank → pod.",
    )
    p.add_argument(
        "--route-dp-size",
        type=int,
        default=_env_int("PROXY_ROUTE_DP", 0),
        help="Cap round-robin to ranks 0..N-1. 0 = min(prefill,decode). "
        "8 pins HTTP-head GPUs (216966 rank 8+ kv=null). With "
        "MORIIO_CHILD_HTTP=1 leave this 0 so ranks 8+ POST to the child.",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=_env_int("PROXY_MAX_CONCURRENCY", 512),
        help="In-flight client POSTs (default 512). Extra connections wait "
        "on the listen backlog; they are not RST'd.",
    )
    p.add_argument("--expect-prefill", type=int, default=0)
    p.add_argument("--expect-decode", type=int, default=0)
    p.add_argument(
        "--prefill",
        action="append",
        default=[],
        help="Prefill pod HTTP URL. Repeat once per pod, CLI order = pod index.",
    )
    p.add_argument(
        "--decode",
        action="append",
        default=[],
        help="Decode pod HTTP URL. Repeat once per pod, CLI order = pod index.",
    )
    p.add_argument(
        "--log-level",
        default=os.environ.get("PROXY_LOG_LEVEL", "DEBUG"),
        help="DEBUG|INFO|WARNING|ERROR. Also PROXY_LOG_LEVEL. "
        "Default DEBUG until PD WRITE is resolved; set INFO to quiet.",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Same as --log-level DEBUG. Does not enable Quart app.debug. "
        "Also honoured via PROXY_DEBUG=1.",
    )
    args = p.parse_args(argv)
    env_debug = str(os.environ.get("PROXY_DEBUG", "")).lower() in (
        "1",
        "true",
        "yes",
        "debug",
    )
    want_debug = bool(args.debug or env_debug)
    log_level_name = "DEBUG" if want_debug else str(args.log_level).upper()
    configure_logging(log_level_name, debug=want_debug)
    dp_prefill = args.moriio_dp_size
    if dp_prefill <= 0:
        dp_prefill = max(args.dp_size_local, 1)
        logger.warning("--moriio-dp-size unset; falling back to %s", dp_prefill)
    dp_decode = args.decode_dp_size
    if dp_decode <= 0:
        dp_decode = dp_prefill
    dp_world = routing_dp_world(dp_prefill, dp_decode, args.route_dp_size)
    if args.route_dp_size > 0 and args.route_dp_size < min(dp_prefill, dp_decode):
        logger.info(
            "route_dp capped to %s (PROXY_ROUTE_DP/--route-dp-size); ranks %s+ not pinned",
            dp_world,
            dp_world,
        )
    if dp_prefill != dp_decode:
        logger.info(
            "asymmetric PD: prefill_dp=%s decode_dp=%s route_ranks=0..%s",
            dp_prefill,
            dp_decode,
            dp_world - 1,
        )
    return ProxyConfig(
        port=args.port,
        discovery_port=args.discovery_port,
        use_discovery=bool(args.use_discovery),
        dp_world=dp_world,
        prefill_dp=dp_prefill,
        decode_dp=dp_decode,
        dp_size_local=args.dp_size_local,
        expect_prefill=args.expect_prefill,
        expect_decode=args.expect_decode,
        prefill_urls=args.prefill,
        decode_urls=args.decode,
        listen_backlog=int(os.environ.get("PROXY_LISTEN_BACKLOG", "4096")),
        log_level=log_level_name,
        max_concurrency=max(int(args.max_concurrency or 0), 0),
    )


def require_discovery(cfg: ProxyConfig) -> None:
    """PD WRITE needs ZMQ P/D register. Rank pin is a separate knob."""
    if not cfg.use_discovery:
        raise SystemExit(
            "PD WRITE requires ZMQ discovery (--use-discovery). "
            "--no-use-discovery is not supported."
        )
    if cfg.discovery_port == 0:
        raise SystemExit("discovery port cannot be 0")


def main(argv: list[str] | None = None) -> None:
    cfg = parse_args(argv)
    require_discovery(cfg)
    check_deps()
    state = ProxyState(cfg)
    # PROXY_TYPE=moriio_toy names the launch mode, not this program, and the
    # in-image examples/.../moriio_toy_proxy_server.py is a different one that
    # ships in the ROCm vLLM build. The whole point of running this fork is EP
    # debugging, so say which implementation is serving before anything else.
    logger.info(
        "python proxy impl=%s (MAD-private fork -- NOT the in-image "
        "examples/disaggregated/disaggregated_serving/moriio_toy_proxy_server.py)",
        os.path.abspath(__file__),
    )
    logger.info(
        "python proxy port=%s discovery=%s use_discovery=%s "
        "concurrent_prefill=true max_concurrency=%s route_dp=%s "
        "prefill_dp=%s decode_dp=%s "
        "dp_local=%s expect P=%s D=%s log_level=%s prefill_urls=%s decode_urls=%s",
        cfg.port,
        cfg.discovery_port,
        cfg.use_discovery,
        cfg.max_concurrency,
        cfg.dp_world,
        cfg.prefill_dp,
        cfg.decode_dp,
        cfg.dp_size_local,
        state.expect_prefill(),
        state.expect_decode(),
        cfg.log_level,
        cfg.prefill_urls,
        cfg.decode_urls,
    )
    t = threading.Thread(
        target=_listen_for_register,
        args=(state, "0.0.0.0", cfg.discovery_port),
        daemon=True,
    )
    t.start()

    from hypercorn.asyncio import serve
    from hypercorn.config import Config as HypercornConfig

    app = create_app(state)
    hcfg = HypercornConfig()
    hcfg.bind = [f"0.0.0.0:{cfg.port}"]
    hcfg.backlog = cfg.listen_backlog
    hcfg.keep_alive_timeout = 360000.0
    hcfg.graceful_timeout = 30.0
    hcfg.workers = 1
    if hasattr(hcfg, "max_app_queue"):
        hcfg.max_app_queue = max(int(cfg.max_concurrency or 512) * 2, 1024)
    hcfg.errorlog = "-"
    hcfg.loglevel = "debug" if cfg.log_level.upper() == "DEBUG" else "info"
    hcfg.accesslog = "-" if cfg.log_level.upper() == "DEBUG" else None
    logger.info(
        "Hypercorn bind=0.0.0.0:%s backlog=%s workers=1 max_concurrency=%s "
        "loglevel=%s accesslog=%s (c=1..%s: queue, do not RST)",
        cfg.port,
        cfg.listen_backlog,
        cfg.max_concurrency,
        hcfg.loglevel,
        "stdout" if hcfg.accesslog else "off",
        cfg.max_concurrency or 512,
    )
    try:
        asyncio.run(serve(app, hcfg))
    finally:
        t.join(timeout=1)


if __name__ == "__main__":
    main()
