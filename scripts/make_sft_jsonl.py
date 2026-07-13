"""Build SFT train/eval JSONL from raw JSON directory.

Usage:
  python scripts/make_sft_jsonl.py --in data/raw/train --out data/processed --mode conversation
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from temprun.data import (
    balance_via_reorder,
    build_rows,
    label_distribution,
    stratified_split,
    write_jsonl,
)
from temprun.prompts import Mode
from temprun.utils import ensure_dir, repo_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default=str(repo_root() / "data" / "raw" / "train"))
    parser.add_argument("--out", default=str(repo_root() / "data" / "processed"))
    parser.add_argument("--mode", choices=["conversation", "cot"], default="conversation")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no-balance", action="store_true", help="Skip label balancing")
    args = parser.parse_args()

    random.seed(args.seed)
    mode: Mode = args.mode  # type: ignore[assignment]
    inp = Path(args.inp)
    out = Path(args.out)
    final_dir = out / "final"
    ensure_dir(out)
    ensure_dir(final_dir)

    rows, dropped = build_rows(inp, mode=mode)
    print(f"[sft] built {len(rows)} rows (dropped {dropped} docs without content)")

    train, evald = stratified_split(rows, test_size=args.test_size, seed=args.seed)
    print(f"[sft] train={len(train)} eval={len(evald)}")
    print(f"[sft] train dist: {label_distribution(train)}")
    print(f"[sft] eval  dist: {label_distribution(evald)}")

    if not args.no_balance:
        train = balance_via_reorder(train, seed=args.seed, mode=mode)
        post_dist = label_distribution(train)
        print(f"[sft] train after balance: {post_dist}")

    print(f"[sft] final train={len(train)} eval={len(evald)}")
    write_jsonl(train, final_dir / "train.jsonl")
    write_jsonl(evald, final_dir / "eval.jsonl")
    print(f"[sft] wrote {final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
