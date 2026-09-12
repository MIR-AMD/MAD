#!/usr/bin/env python3
"""OpenAI MRCR against a live disagg /v1/completions server.

Official dataset: huggingface.co/datasets/openai/mrcr
Official grade: hash prefix required, then difflib.SequenceMatcher ratio (MMR).

This is NOT product NIAH. Do not set NIAH_LIST_PRIME. Flash-first slice is
2-needle bins through 32k (fits MAX_MODEL_LEN=40960). CorpusQA / 1M wait
until the window is raised.

Usage (inside the container, via benchmark_mrcr.sh):
  MRCR_URL=http://127.0.0.1:10001 MRCR_MODEL=... python3 benchmark_mrcr.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from mrcr_lib import (  # noqa: E402
    encode_dsv4_chat,
    grade,
    resolve_data_paths,
    select_rows,
    iter_parquet_rows,
)

URL = os.environ.get("MRCR_URL") or os.environ.get("NIAH_URL") or "http://127.0.0.1:10001"
MODEL = os.environ.get("MRCR_MODEL") or os.environ.get("NIAH_MODEL") or os.environ.get(
    "MODEL_PATH", "unknown"
)
NEEDLES = int(os.environ.get("MRCR_NEEDLES", "2"))
BINS = [int(x) for x in os.environ.get("MRCR_BINS", "8192,16384,32768").split(",") if x.strip()]
PER_BIN = int(os.environ.get("MRCR_PER_BIN", "8"))
MAXTOK = int(os.environ.get("MRCR_MAXTOK", "1024"))
TIMEOUT = float(os.environ.get("MRCR_TIMEOUT", "1800"))
MAX_CTX = int(os.environ.get("MRCR_MAX_CTX", "40960"))
DATA_DIR = os.environ.get("MRCR_DATA_DIR", "")
THINKING = os.environ.get("MRCR_THINKING", "chat").strip().lower()
HALT = os.environ.get("MRCR_HALT_ON_FAIL", "0").strip() == "1"

_REQ_N = [0]


def _utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _request(prompt, rid):
    body = {
        "model": MODEL,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": MAXTOK,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        URL.rstrip("/") + "/v1/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-request-id": rid,
        },
    )
    t0 = time.time()
    try:
        text = ""
        http_status = None
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            http_status = getattr(r, "status", None)
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("error"):
                    err = obj["error"]
                    msg = err.get("message", err) if isinstance(err, dict) else err
                    return None, "sse error: %s" % msg, time.time() - t0, http_status
                try:
                    text += obj["choices"][0]["text"]
                except Exception:
                    pass
        return text, None, time.time() - t0, http_status
    except Exception as exc:
        return None, str(exc), time.time() - t0, None


def main():
    if THINKING != "chat":
        print(
            "Error: MRCR_THINKING=%r — must be 'chat'. thinking mode burns "
            "the hash prefix into a <think> block and zeros MMR."
            % THINKING,
            file=sys.stderr,
        )
        sys.exit(2)
    if os.environ.get("NIAH_LIST_PRIME", "").strip():
        print(
            "Error: NIAH_LIST_PRIME is set. MRCR already specifies the last "
            "user turn (hash + Nth writing). Do not prime.",
            file=sys.stderr,
        )
        sys.exit(2)

    max_prompt = max(1024, MAX_CTX - MAXTOK)
    print("=== OpenAI MRCR (2-needle slice unless MRCR_NEEDLES overrides) ===", flush=True)
    print(
        "url=%s  model=%s  needles=%d  bins=%s  per_bin=%d  maxtok=%d  "
        "timeout=%.0fs  max_prompt_tok=%d  thinking=chat  prime=OFF"
        % (URL, MODEL, NEEDLES, BINS, PER_BIN, MAXTOK, TIMEOUT, max_prompt),
        flush=True,
    )
    paths = resolve_data_paths(NEEDLES, DATA_DIR or None)
    print("[mrcr] data=%s" % paths, flush=True)
    rows = list(iter_parquet_rows(paths))
    print("[mrcr] loaded %d rows" % len(rows), flush=True)
    buckets, skipped_bin, skipped_ctx = select_rows(rows, BINS, PER_BIN, max_prompt)
    print(
        "[mrcr] skipped_other_bins=%d skipped_over_ctx=%d"
        % (skipped_bin, skipped_ctx),
        flush=True,
    )
    for upper in sorted(buckets):
        print("[mrcr] bin<=%d n=%d" % (upper, len(buckets[upper])), flush=True)

    summary = []
    for upper in sorted(buckets):
        items = buckets[upper]
        if not items:
            print("[mrcr] empty bin<=%d" % upper, flush=True)
            summary.append((upper, []))
            continue
        scores = []
        for ntok, messages, row in items:
            _REQ_N[0] += 1
            rid = "mrcr-%06d" % _REQ_N[0]
            prompt = encode_dsv4_chat(messages)
            prefix = row.get("random_string_to_prepend") or ""
            answer = row.get("answer") or ""
            print(
                "[mrcr] start %s bin<=%d ntok=%d max_tokens=%d x-request-id=%s"
                % (_utc(), upper, ntok, MAXTOK, rid),
                flush=True,
            )
            text, err, dt, http_status = _request(prompt, rid)
            if err or text is None:
                print(
                    "[mrcr] fail %s bin<=%d dt=%.1fs http=%s x-request-id=%s err=%s"
                    % (_utc(), upper, dt, http_status, rid, err),
                    flush=True,
                )
                scores.append(0.0)
                if HALT:
                    print("[mrcr] halt_on_fail", flush=True)
                    _print_summary(summary + [(upper, scores)])
                    sys.exit(1)
                continue
            mmr = grade(text, answer, prefix)
            hit = 1 if text.startswith(str(prefix)) else 0
            preview = (text[:160] or "").replace("\n", "\\n")
            print(
                "[mrcr] done %s bin<=%d dt=%.1fs chars=%d hash_ok=%d mmr=%.4f "
                "x-request-id=%s text=%r"
                % (_utc(), upper, dt, len(text), hit, mmr, rid, preview),
                flush=True,
            )
            scores.append(mmr)
        summary.append((upper, scores))

    _print_summary(summary)
    # Nonzero exit only if every scored item is 0 and we actually ran something.
    ran = [s for _, sc in summary for s in sc]
    if ran and all(x == 0.0 for x in ran):
        sys.exit(1)


def _print_summary(summary):
    print("=== MRCR summary (MMR = SequenceMatcher after hash) ===", flush=True)
    for upper, scores in summary:
        if not scores:
            print("  bin<=%5d  n=0" % upper, flush=True)
            continue
        mean = sum(scores) / len(scores)
        print(
            "  bin<=%5d  n=%d  mean_mmr=%.4f  min=%.4f  max=%.4f  scores=%s"
            % (
                upper,
                len(scores),
                mean,
                min(scores),
                max(scores),
                ",".join("%.3f" % x for x in scores),
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
