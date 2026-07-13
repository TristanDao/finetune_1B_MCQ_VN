"""CLI for inference / submission generation.

Usage:
  python scripts/infer.py --adapter artifacts/unsloth_qwen3_0_6b/adapter --test-dir data/raw/test --out submissions/sub.csv
"""
from __future__ import annotations

import argparse

from temprun.infer import _load_model, iter_test_items, predict, write_submission
from temprun.prompts import SYSTEM_PROMPT_CONVERSATION, SYSTEM_PROMPT_COT, Mode
from temprun.utils import load_env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to Unsloth LoRA adapter dir")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["conversation", "cot"], default="conversation")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()

    load_env()
    mode: Mode = args.mode  # type: ignore[assignment]
    system_prompt = SYSTEM_PROMPT_COT if mode == "cot" else SYSTEM_PROMPT_CONVERSATION

    model, tokenizer = _load_model(args.adapter, max_seq_length=args.max_length)
    items = iter_test_items(args.test_dir)
    print(f"[infer] mode={mode} items={len(items)}")

    results = predict(
        model, tokenizer, items,
        system_prompt=system_prompt,
        batch_size=args.batch_size,
        max_length=args.max_length,
        mode=mode,
    )
    out = write_submission(results, args.out)
    print(f"[infer] done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
