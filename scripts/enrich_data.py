"""Enrich a SFT jsonl via Alibaba DashScope (Qwen3-max-preview).

Usage:
  python scripts/enrich_data.py --in data/processed/train.jsonl --out data/processed/enriched.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from temprun.enrich import enrich_dataset
from temprun.utils import repo_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default=str(repo_root() / "data" / "processed" / "train.jsonl"))
    parser.add_argument("--out", default=str(repo_root() / "data" / "processed" / "enriched.jsonl"))
    parser.add_argument("--no-paraphrase", action="store_true")
    parser.add_argument("--no-explain", action="store_true")
    parser.add_argument("--synth", action="store_true", help="Also synthesize new questions")
    parser.add_argument("--synth-max", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407)
    balance = parser.add_mutually_exclusive_group()
    balance.add_argument(
        "--balance",
        dest="balance",
        action="store_true",
        default=None,
        help="Force class-balanced enrichment (paraphrase + targeted synth).",
    )
    balance.add_argument(
        "--no-balance",
        dest="balance",
        action="store_false",
        help="Disable class balancing even when the data is skewed.",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=None,
        help="Target count per label when balancing. Defaults to the max current count.",
    )
    parser.add_argument(
        "--synth-retry",
        type=int,
        default=3,
        help="Retries per article when synth is targeted at a minority class.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max in-flight API calls (>=2 enables async path; 1 forces sequential).",
    )
    parser.add_argument(
        "--no-async",
        dest="async_enabled",
        action="store_false",
        default=True,
        help="Force sequential path (equivalent to --concurrency 1).",
    )
    args = parser.parse_args()

    concurrency = args.concurrency if args.async_enabled else 1

    try:
        counters = enrich_dataset(
            input_jsonl=args.inp,
            output_jsonl=args.out,
            paraphrase=not args.no_paraphrase,
            explain=not args.no_explain,
            synth=args.synth,
            synth_max=args.synth_max,
            seed=args.seed,
            balance=args.balance,
            target_per_class=args.target_per_class,
            synth_retry=args.synth_retry,
            concurrency=concurrency,
        )
    except Exception as e:
        print(f"[enrich] failed: {e}", file=sys.stderr)
        return 1
    print(f"[enrich] summary: {counters}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
