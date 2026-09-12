#!/usr/bin/env python3
"""Stage openai/mrcr parquet onto NFS so jobs do not hit HuggingFace at NIAH time.

Login node only (needs outbound HTTPS). Do not run on WSL.

  python3 fetch_mrcr.py --needles 2 --out /shared_inference/bbarakat/datasets/openai_mrcr

Then submit with:
  MRCR_DATA_DIR=/shared_inference/bbarakat/datasets/openai_mrcr \\
    ./run_wideep_bench.sh mrcr dsv4fls 2p2d --image v0290 --time 04:00:00 --exclude ''
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys


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
        print("pip install huggingface_hub", file=sys.stderr)
        sys.exit(2)
    os.makedirs(args.out, exist_ok=True)
    names = [
        "%dneedle/%dneedle_0.parquet" % (args.needles, args.needles),
        "%dneedle/%dneedle_1.parquet" % (args.needles, args.needles),
    ]
    for name in names:
        src = hf_hub_download(
            repo_id="openai/mrcr", filename=name, repo_type="dataset"
        )
        dest = os.path.join(args.out, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.abspath(src) != os.path.abspath(dest):
            shutil.copy2(src, dest)
        print("ok", dest, os.path.getsize(dest))


if __name__ == "__main__":
    main()
