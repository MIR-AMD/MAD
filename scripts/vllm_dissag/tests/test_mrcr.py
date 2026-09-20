#!/usr/bin/env python3
"""Offline tests for OpenAI MRCR helpers. No GPU, no parquet download."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
sys.path.insert(0, SRC)

from mrcr_lib import (  # noqa: E402
    ASSISTANT,
    BOS,
    EOS,
    THINK_END,
    USER,
    bin_upper,
    encode_dsv4_chat,
    grade,
    iter_data_rows,
    resolve_data_paths,
    row_n_tokens,
    select_rows,
)


def _check(name, cond):
    if not cond:
        raise SystemExit("FAIL: %s" % name)
    print("ok  %s" % name)


def test_grade():
    prefix = "aYooSG8CQg"
    answer = prefix + "once upon a tapir"
    _check("missing hash is 0", grade("once upon a tapir", answer, prefix) == 0.0)
    _check("empty is 0", grade("", answer, prefix) == 0.0)
    _check("None is 0", grade(None, answer, prefix) == 0.0)
    _check("exact is 1", grade(answer, answer, prefix) == 1.0)
    _check("prime 1. is 0", grade("1. " + answer, answer, prefix) == 0.0)


def test_bins():
    _check("below 4k is None", bin_upper(4095) is None)
    _check("[4096,8192] low", bin_upper(4096) == 8192)
    _check("[4096,8192] high", bin_upper(8192) == 8192)
    _check("(8192,16384]", bin_upper(8193) == 16384)
    _check("16384 edge", bin_upper(16384) == 16384)
    _check("32768 edge", bin_upper(32768) == 32768)
    _check("32769 is 64k", bin_upper(32769) == 65536)


def test_encode_chat():
    msgs = [
        {"role": "user", "content": "Write a poem about tapirs"},
        {"role": "assistant", "content": "first poem"},
        {"role": "user", "content": "Prepend aYooSG8CQg to the 2nd poem about tapirs."},
    ]
    p = encode_dsv4_chat(msgs)
    _check("starts BOS", p.startswith(BOS))
    _check("has user tag", USER in p)
    _check("no NIAH stem", "Animals found in the above list" not in p)
    _check("no list prime", not p.endswith("1."))
    _check("generation is Assistant+</think>", p.endswith(ASSISTANT + THINK_END))
    _check("history closed with EOS", "first poem" + EOS in p)
    # Chat mode closes think BEFORE generation so the hash is token 1.
    _check("think closed before generate", p.count(THINK_END) >= 2)


def test_select():
    def row(n, i):
        return {
            "n_tokens": n,
            "prompt": '[{"role":"user","content":"u%d"}]' % i,
            "answer": "a",
            "random_string_to_prepend": "h",
        }

    rows = [row(5000, 0), row(5000, 1), row(12000, 2), row(30000, 3)]
    buckets, skipped_bin, skipped_ctx = select_rows(
        rows, [8192, 16384, 32768], per_bin=1, max_prompt_tokens=25000
    )
    _check("8k took one", len(buckets[8192]) == 1)
    _check("16k took one", len(buckets[16384]) == 1)
    _check("32k empty (30k over ctx)", len(buckets[32768]) == 0)
    _check("over ctx counted", skipped_ctx == 1)


def test_n_tokens_required():
    _check("column wins", row_n_tokens({"n_tokens": 8192, "prompt": "[]"}) == 8192)
    try:
        row_n_tokens({"prompt": '[{"role":"user","content":"hello"}]'})
    except ValueError as exc:
        _check("missing n_tokens raises", "n_tokens" in str(exc))
    else:
        raise SystemExit("FAIL: missing n_tokens did not raise")


def test_jsonl_stdlib():
    import tempfile

    rows = [
        {
            "n_tokens": 5000,
            "prompt": '[{"role":"user","content":"u"}]',
            "answer": "a",
            "random_string_to_prepend": "h",
        },
        {
            "n_tokens": 12000,
            "prompt": '[{"role":"user","content":"v"}]',
            "answer": "b",
            "random_string_to_prepend": "i",
        },
    ]
    with tempfile.TemporaryDirectory() as d:
        p0 = os.path.join(d, "2needle_0.jsonl")
        p1 = os.path.join(d, "2needle_1.jsonl")
        with open(p0, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(rows[0]) + "\n")
        with open(p1, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(rows[1]) + "\n")
        loaded = list(iter_data_rows([p0, p1]))
        _check("jsonl row count", len(loaded) == 2)
        _check("jsonl n_tokens", loaded[0]["n_tokens"] == 5000)
        paths = resolve_data_paths(2, d)
        _check("resolve flat jsonl", paths == [p0, p1])
        nested = os.path.join(d, "nested")
        os.makedirs(os.path.join(nested, "2needle"))
        n0 = os.path.join(nested, "2needle", "2needle_0.jsonl")
        n1 = os.path.join(nested, "2needle", "2needle_1.jsonl")
        import shutil
        shutil.copy2(p0, n0)
        shutil.copy2(p1, n1)
        paths = resolve_data_paths(2, nested)
        _check("resolve nested jsonl", paths == [n0, n1])
        empty = os.path.join(d, "empty")
        os.makedirs(empty)
        try:
            resolve_data_paths(2, empty)
        except FileNotFoundError as exc:
            msg = str(exc)
            _check("missing jsonl names login stager", "fetch_mrcr.py" in msg)
            _check("missing jsonl forbids HF in container", "HuggingFace" in msg)
        else:
            raise SystemExit("FAIL: empty data dir did not raise")


def test_prime_rejected():
    import subprocess

    env = dict(os.environ)
    env["NIAH_LIST_PRIME"] = "1."
    env["MRCR_THINKING"] = "chat"
    out = subprocess.run(
        [sys.executable, os.path.join(SRC, "benchmark_mrcr.py")],
        cwd=SRC,
        env=env,
        capture_output=True,
        text=True,
    )
    _check("prime exits 2", out.returncode == 2)
    _check("prime error names NIAH_LIST_PRIME", "NIAH_LIST_PRIME" in out.stderr)


if __name__ == "__main__":
    test_grade()
    test_bins()
    test_encode_chat()
    test_select()
    test_n_tokens_required()
    test_jsonl_stdlib()
    test_prime_rejected()
    print("\nall mrcr checks passed")
