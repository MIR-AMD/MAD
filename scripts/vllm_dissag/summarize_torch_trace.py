#!/usr/bin/env python3
"""Aggregate GPU kernel time out of vLLM's torch profiler traces.

Written for one question: at MoRI + DP8 + EP8 decode, where does the
capacity-proportional cost live? 436340 established that ITL scales as
`max_num_inp_token_per_rank x world_size` (ITL = 20.1 + 0.2816*mnbt, R^2
0.996), but two consumers scale that way and latency alone cannot separate
them:

  MoRI's comm kernels        -- EpDispatch*/EpCombine*, capacity-bounded loops
                                exist at internode_v1.cpp:500 and
                                intranode_1250x.hpp:579
  the expert GEMM            -- dispatch() returns a view of
                                (max_num_tokens_to_recv(), hidden_dim) =
                                8192 x 4096 at mnbt 1024 / EP8, and
                                MoriPrepareAndFinalize.prepare() hands that
                                padded buffer straight to
                                rocm_aiter_fused_experts

So this bins kernels into those two families plus the rest, and reports each
family's share. Run it on the cap=0 and capped cells and compare: whichever
family's share tracks the capacity is the one to fix.

Reads what vLLM writes -- gzipped Chrome traces, one per rank. GPU kernels are
the `cat == "kernel"` events; `dur` is microseconds.

    summarize_torch_trace.py <trace.json.gz | dir> [...]   # any mix
    summarize_torch_trace.py --top 40 traces/

Ranks are summarized separately AND pooled, because a straggler rank is a
live hypothesis at wide EP and an average would hide it.
"""
import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict

# Ordered: first match wins, so the specific patterns precede the generic.
FAMILIES = (
    ("mori-comm", ("epdispatch", "epcombine", "mori", "crossdevicebarrier")),
    ("moe-expert", ("moe_sorting", "moe_sort", "fmoe", "fused_moe", "moe_stage",
                    "ck_moe", "asm_moe", "moe_align", "topk_softmax", "silu")),
    ("gemm", ("gemm", "cijk", "kernel_func", "hgemm", "f8gemm", "wvsplitk")),
    ("attention", ("attn", "mla", "flash", "paged", "indexer")),
    ("norm-elem", ("rmsnorm", "layernorm", "elementwise", "vectorized",
                   "copy", "cast", "quant", "transpose", "reduce")),
    ("comm-other", ("nccl", "rccl", "allreduce", "allgather", "reducescatter")),
)


def classify(name: str) -> str:
    low = name.lower()
    for fam, pats in FAMILIES:
        if any(p in low for p in pats):
            return fam
    return "other"


def load(path: str) -> list:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        blob = json.load(fh)
    if isinstance(blob, dict):
        return blob.get("traceEvents", [])
    return blob


def kernels(events) -> list:
    out = []
    for ev in events:
        if ev.get("ph") != "X" or ev.get("cat") != "kernel":
            continue
        dur = ev.get("dur")
        if dur:
            out.append((ev.get("name", "?"), float(dur)))
    return out


def expand(paths) -> list:
    files = []
    for p in paths:
        if os.path.isdir(p):
            for pat in ("*.json.gz", "*.json", "**/*.json.gz", "**/*.json"):
                files += glob.glob(os.path.join(p, pat), recursive=True)
        else:
            files += glob.glob(p)
    return sorted(set(files))


def report(label: str, ks, top: int) -> None:
    if not ks:
        print(f"\n### {label}: no GPU kernel events")
        return
    by_name = defaultdict(lambda: [0.0, 0])
    by_fam = defaultdict(float)
    total = 0.0
    for name, dur in ks:
        rec = by_name[name]
        rec[0] += dur
        rec[1] += 1
        by_fam[classify(name)] += dur
        total += dur

    print(f"\n### {label}")
    print(f"total GPU kernel time {total / 1000:.1f} ms across {len(ks)} launches")
    print("\n  by family:")
    for fam, ms in sorted(by_fam.items(), key=lambda kv: -kv[1]):
        print(f"    {fam:12} {ms / 1000:9.2f} ms  {100 * ms / total:5.1f}%")
    print(f"\n  top {top} kernels:")
    print(f"    {'total ms':>9}  {'%':>5}  {'n':>6}  {'mean us':>9}  family       name")
    rows = sorted(by_name.items(), key=lambda kv: -kv[1][0])[:top]
    for name, (ms, n) in rows:
        print(
            f"    {ms / 1000:9.2f}  {100 * ms / total:5.1f}  {n:6}  "
            f"{ms / n:9.1f}  {classify(name):12} {name[:90]}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="trace files and/or directories")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--per-rank", action="store_true",
                    help="also break down each trace file separately")
    args = ap.parse_args()

    files = expand(args.paths)
    if not files:
        print(f"no trace files under {args.paths}", file=sys.stderr)
        return 1

    pooled = []
    for path in files:
        try:
            ks = kernels(load(path))
        except Exception as exc:  # a truncated trace should not kill the rest
            print(f"[warn] {path}: {exc}", file=sys.stderr)
            continue
        pooled += ks
        if args.per_rank:
            report(os.path.basename(path), ks, args.top)

    report(f"POOLED over {len(files)} trace file(s)", pooled, args.top)
    print(
        "\nRead: mori-comm dominant -> the bug is in MoRI's dispatch/combine "
        "kernels.\n"
        "      moe-expert/gemm dominant -> the bug is vLLM handing the experts "
        "a\n"
        "      buffer padded to max_num_tokens_to_recv(). See the module "
        "docstring."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
