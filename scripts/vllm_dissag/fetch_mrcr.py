#!/usr/bin/env python3
"""Stage openai/mrcr onto NFS as JSONL so serving jobs stay stdlib-only.

Login node only (needs outbound HTTPS + pandas/pyarrow). Do not run on WSL
and do not install those packages in the vLLM image — the job reads *.jsonl.

  python3 fetch_mrcr.py --needles 2 --out /shared_inference/bbarakat/datasets/openai_mrcr

Then submit with:
  MRCR_DATA_DIR=/shared_inference/bbarakat/datasets/openai_mrcr \\
    ./run_wideep_bench.sh mrcr dsv4fls 2p2d --image v0290 --time 04:00:00 --exclude ''
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys


NEEDED = ("prompt", "answer", "random_string_to_prepend", "n_tokens")


def _parquet_to_jsonl(parquet_path):
    jsonl_path = (
        parquet_path[:-8] + ".jsonl"
        if parquet_path.endswith(".parquet")
        else parquet_path + ".jsonl"
    )
    try:
        import pandas as pd
    except ImportError:
        print(
            "fetch_mrcr.py needs pandas+pyarrow on the LOGIN node to emit JSONL. "
            "Activate madeng (or pip install pandas pyarrow) and retry. "
            "Do not install these in the serving image.",
            file=sys.stderr,
        )
        sys.exit(2)
    df = pd.read_parquet(parquet_path)
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        print("parquet missing columns %s: %s" % (missing, parquet_path), file=sys.stderr)
        sys.exit(2)
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for rec in df.loc[:, list(NEEDED)].to_dict(orient="records"):
            rec["n_tokens"] = int(rec["n_tokens"])
            if rec["prompt"] is None:
                rec["prompt"] = "[]"
            elif not isinstance(rec["prompt"], (str, list)):
                rec["prompt"] = json.dumps(rec["prompt"], ensure_ascii=False)
            json.dump(rec, fh, ensure_ascii=False)
            fh.write("\n")
    return jsonl_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--needles", type=int, default=2, choices=(2, 4, 8))
    p.add_argument(
        "--out",
        default="/shared_inference/bbarakat/datasets/openai_mrcr",
    )
    args = p.parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("pip install huggingface_hub  (login node only)", file=sys.stderr)
        sys.exit(2)
    os.makedirs(args.out, exist_ok=True)
    names = [
        "%dneedle/%dneedle_0.parquet" % (args.needles, args.needles),
        "%dneedle/%dneedle_1.parquet" % (args.needles, args.needles),
    ]
    for name in names:
        dest = os.path.join(args.out, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.isfile(dest):
            src = hf_hub_download(
                repo_id="openai/mrcr", filename=name, repo_type="dataset"
            )
            if os.path.abspath(src) != os.path.abspath(dest):
                shutil.copy2(src, dest)
            print("ok parquet", dest, os.path.getsize(dest))
        else:
            print("have parquet", dest, os.path.getsize(dest))
        jsonl = _parquet_to_jsonl(dest)
        print("ok jsonl", jsonl, os.path.getsize(jsonl))


if __name__ == "__main__":
    main()
