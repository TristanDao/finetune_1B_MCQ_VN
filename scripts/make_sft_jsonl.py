"""Build SFT/eval JSONL from raw JSON dir (train set).

Usage:
  python scripts/make_sft_jsonl.py --in data/raw/train --out data/processed
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from temprun.data import build_rows, label_distribution, read_jsonl, stratified_split, write_jsonl
from temprun.utils import ensure_dir, repo_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default=str(repo_root() / "data" / "raw" / "train"))
    parser.add_argument("--out", default=str(repo_root() / "data" / "processed"))
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    random.seed(args.seed)
    inp = Path(args.inp)
    out = Path(args.out)
    ensure_dir(out)

    rows, dropped = build_rows(inp)
    print(f"[sft] built {len(rows)} rows (dropped {dropped} docs without content)")

    train, evald = stratified_split(rows, test_size=args.test_size, seed=args.seed)
    print(f"[sft] train={len(train)} eval={len(evald)}")
    print(f"[sft] train label dist: {label_distribution(train)}")
    print(f"[sft] eval  label dist: {label_distribution(evald)}")

    write_jsonl(train, out / "train.jsonl")
    write_jsonl(evald, out / "eval.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
