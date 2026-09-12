"""OpenAI MRCR helpers: official grade, DSV4 chat encoding, binning.

No NIAH_LIST_PRIME. The last MRCR user turn already says
"Prepend <hash> to the Nth … Do not include any other text".
Appending `1.` would fail the hash prefix and score 0.

Chat encoding follows DeepSeek-V4 encoding/README.md thinking_mode=chat:
historical assistant turns are closed with </think> so generation is the
answer, not a reasoning block. Do not use thinking_mode=thinking here.
"""
from __future__ import annotations

import json
import os
from difflib import SequenceMatcher

# Official OpenAI MRCR bin edges (prompt + answer tokens, o200k_base).
# First bin is closed [4096, 8192]; later bins are (prev, next].
BIN_EDGES = (4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)

BOS = "<\uff5cbegin\u2581of\u2581sentence\uff5c>"
EOS = "<\uff5cend\u2581of\u2581sentence\uff5c>"
USER = "<\uff5cUser\uff5c>"
ASSISTANT = "<\uff5cAssistant\uff5c>"
THINK_END = "</think>"


def grade(response, answer, random_string_to_prepend):
    """OpenAI MRCR SequenceMatcher grade. Missing hash => 0.0."""
    if response is None:
        return 0.0
    text = str(response)
    prefix = str(random_string_to_prepend or "")
    if not prefix or not text.startswith(prefix):
        return 0.0
    got = text[len(prefix) :]
    want = str(answer or "")
    if want.startswith(prefix):
        want = want[len(prefix) :]
    return float(SequenceMatcher(None, got, want).ratio())


def bin_upper(n_tokens, edges=BIN_EDGES):
    """Return the bin's upper edge, or None if outside the official set."""
    n = int(n_tokens)
    if n < edges[0]:
        return None
    if n <= edges[1]:
        return int(edges[1])
    for i in range(2, len(edges)):
        if n <= edges[i]:
            return int(edges[i])
    return None


def encode_dsv4_chat(messages):
    """Render OpenAI-format messages to a DeepSeek-V4 chat completions prompt.

    thinking_mode=chat: after each user turn that is followed by an assistant
    (or is last), emit Assistant + </think>. Historical assistant content then
    follows and is closed with EOS. The last user turn therefore ends with
    Assistant + </think> so the model generates the hash+needle directly.
    """
    if not messages:
        raise ValueError("empty messages")
    parts = [BOS]
    n = len(messages)
    for i, msg in enumerate(messages):
        role = (msg.get("role") or "").strip().lower()
        content = msg.get("content")
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if role == "system":
            parts.append(content)
            continue
        if role == "user":
            parts.append(USER)
            parts.append(content)
            nxt = messages[i + 1] if i + 1 < n else None
            nxt_role = (nxt.get("role") or "").strip().lower() if nxt else ""
            if nxt is None or nxt_role == "assistant":
                parts.append(ASSISTANT)
                parts.append(THINK_END)
            continue
        if role == "assistant":
            parts.append(content)
            parts.append(EOS)
            continue
        raise ValueError("unsupported role %r (MRCR is user/assistant only)" % role)
    return "".join(parts)


def parse_prompt_field(raw):
    """parquet `prompt` is a JSON string of messages."""
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    raise TypeError("prompt must be JSON str or list, got %s" % type(raw).__name__)


def row_n_tokens(row, messages=None):
    """Prefer a dataset column; else tiktoken o200k; else a warned char heuristic."""
    for key in ("n_tokens", "total_tokens", "tokens", "n_token"):
        if key in row and row[key] is not None and str(row[key]).strip() != "":
            try:
                return int(row[key])
            except (TypeError, ValueError):
                pass
    msgs = messages if messages is not None else parse_prompt_field(row["prompt"])
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        return sum(len(enc.encode(m.get("content") or "")) for m in msgs)
    except Exception:
        text = " ".join((m.get("content") or "") for m in msgs)
        # ~4 chars/tok; only used when tiktoken and columns are both missing.
        return max(1, len(text) // 4)


def select_rows(rows, bin_uppers, per_bin, max_prompt_tokens):
    """Keep official bins, first `per_bin` items per bin, skip over-context."""
    wanted = {int(x) for x in bin_uppers}
    buckets = {b: [] for b in sorted(wanted)}
    skipped_bin = skipped_ctx = 0
    for row in rows:
        messages = parse_prompt_field(row["prompt"])
        ntok = row_n_tokens(row, messages)
        upper = bin_upper(ntok)
        if upper not in buckets:
            skipped_bin += 1
            continue
        # Leave room for the completion; MRCR answers are long writings.
        if ntok > max_prompt_tokens:
            skipped_ctx += 1
            continue
        if len(buckets[upper]) >= per_bin:
            continue
        buckets[upper].append((ntok, messages, row))
    return buckets, skipped_bin, skipped_ctx


def iter_parquet_rows(paths):
    """Yield dict rows from openai/mrcr parquet files. Needs pandas."""
    import pandas as pd

    frames = [pd.read_parquet(p) for p in paths]
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    for rec in df.to_dict(orient="records"):
        yield rec


def default_parquet_names(needles):
    return [
        "%dneedle/%dneedle_0.parquet" % (needles, needles),
        "%dneedle/%dneedle_1.parquet" % (needles, needles),
    ]


def resolve_data_paths(needles, data_dir=None):
    """Local dir first (NFS stage), then huggingface_hub download."""
    needles = int(needles)
    names = default_parquet_names(needles)
    data_dir = (data_dir or os.environ.get("MRCR_DATA_DIR") or "").strip()
    candidates = []
    if data_dir:
        candidates.append(data_dir)
    candidates.append("/shared_inference/bbarakat/datasets/openai_mrcr")
    for root in candidates:
        paths = [os.path.join(root, n) for n in names]
        if all(os.path.isfile(p) for p in paths):
            return paths
        # also accept files dumped flat
        flat = [os.path.join(root, os.path.basename(n)) for n in names]
        if all(os.path.isfile(p) for p in flat):
            return flat
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise FileNotFoundError(
            "openai/mrcr parquet not on disk (tried %s) and huggingface_hub "
            "is missing. Stage 2needle_*.parquet under MRCR_DATA_DIR."
            % candidates
        ) from exc
    out = []
    for name in names:
        out.append(
            hf_hub_download(
                repo_id="openai/mrcr", filename=name, repo_type="dataset"
            )
        )
    return out
