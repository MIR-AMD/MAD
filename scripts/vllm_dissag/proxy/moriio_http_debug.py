# SPDX-License-Identifier: Apache-2.0
"""ASGI middleware: log HTTP arrival on vLLM API servers.

Ours, not vLLM's. Wired via ``vllm serve --middleware
moriio_http_debug.log_http``; PYTHONPATH must include this directory
(connectors/moriio.sh sets it).

This is the proof that ``POST /v1/completions`` reached ``:20005`` *before* the
engine or the MoRIIO connector. Silence here means our python proxy never landed
TCP on this process (or landed on a different API-server pid).

It runs on **both** the prefill and decode masters, so every line carries
``role=`` / ``kv=`` / ``host=``. In a merged EP log an ``HTTP IN`` with no role
is unattributable, and prefill vs decode is usually the first thing you need to
know. ``role=?`` means moriio.sh did not export ``MORIIO_DEBUG_ROLE`` --
the line is still valid, just unattributed.

One line is emitted at import per API-server process (8 per node with
``--api-server-count=8``) naming this file, for the same reason the python proxy
announces its own path: so a debug log never leaves you guessing which
implementation produced it.
"""
from __future__ import annotations

import logging
import os
import socket
import time

logger = logging.getLogger("moriio_http_debug")

_SKIP = frozenset({"/health", "/metrics", "/ping", "/load", "/ready"})

# Computed once: os.uname()/getpid() per request would show up at c=512.
_ROLE = os.environ.get("MORIIO_DEBUG_ROLE") or "?"
_KV_ROLE = os.environ.get("MORIIO_DEBUG_KV_ROLE") or "?"
_HOST = socket.gethostname()
_PID = os.getpid()
_TAG = f"role={_ROLE} kv={_KV_ROLE} host={_HOST} pid={_PID}"

logger.info(
    "moriio_http_debug impl=%s %s (ours, on the vLLM server side -- not the "
    "python proxy and not vLLM's own request log)",
    os.path.abspath(__file__),
    _TAG,
)


async def log_http(request, call_next):
    path = request.url.path
    if path in _SKIP:
        return await call_next(request)
    t0 = time.monotonic()
    hdr = request.headers
    client = getattr(request, "client", None)
    logger.info(
        "HTTP IN %s %s %s client=%s x_dp_rank=%s x_request_id=%s",
        _TAG,
        request.method,
        path,
        client,
        hdr.get("x-data-parallel-rank") or hdr.get("X-data-parallel-rank"),
        hdr.get("x-request-id") or hdr.get("X-Request-Id"),
    )
    try:
        resp = await call_next(request)
    except Exception:
        logger.exception(
            "HTTP FAIL %s %s %s dt=%.3fs",
            _TAG,
            request.method,
            path,
            time.monotonic() - t0,
        )
        raise
    logger.info(
        "HTTP OUT %s %s %s status=%s dt=%.3fs",
        _TAG,
        request.method,
        path,
        getattr(resp, "status_code", "?"),
        time.monotonic() - t0,
    )
    return resp
