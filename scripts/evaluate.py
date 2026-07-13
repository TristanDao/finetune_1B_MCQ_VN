"""CLI for evaluation.

Usage:
  python scripts/evaluate.py --adapter artifacts/unsloth_qwen3_0_6b/adapter --eval-jsonl data/processed/final/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from temprun.data import read_jsonl
from temprun.evaluate import run_eval
from temprun.infer import _load_model
from temprun.prompts import SYSTEM_PROMPT_CONVERSATION, SYSTEM_PROMPT_COT, Mode
from temprun.utils import load_env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to Unsloth LoRA adapter dir")
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--mode", choices=["conversation", "cot"], default="conversation")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    load_env()
    mode: Mode = args.mode  # type: ignore[assignment]
    system_prompt = SYSTEM_PROMPT_COT if mode == "cot" else SYSTEM_PROMPT_CONVERSATION

    model, tokenizer = _load_model(args.adapter, max_seq_length=args.max_length)
    rows = read_jsonl(args.eval_jsonl)
    print(f"[eval] mode={mode} rows={len(rows)}")

    if "instruction" not in rows[0]:
        for r in rows:
            r["instruction"] = r["messages"][1]["content"]

    out_jsonl = args.out or str(Path(args.adapter).parent / "eval_details.jsonl")
    summary = run_eval(
        model, tokenizer, rows,
        system_prompt=system_prompt,
        batch_size=args.batch_size,
        max_length=args.max_length,
        out_jsonl=out_jsonl,
    )
    summary_path = Path(out_jsonl).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
