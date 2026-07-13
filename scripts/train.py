"""CLI for training.

Usage:
  python scripts/train.py --train-jsonl data/processed/final/train.jsonl --eval-jsonl data/processed/final/eval.jsonl --mode conversation
"""
from __future__ import annotations

import argparse

from temprun.train import train
from temprun.utils import load_env, repo_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", default=str(repo_root() / "data" / "processed" / "final" / "train.jsonl"))
    parser.add_argument("--eval-jsonl", default=str(repo_root() / "data" / "processed" / "final" / "eval.jsonl"))
    parser.add_argument("--output-dir", default=str(repo_root() / "artifacts" / "unsloth_qwen3_0_6b"))
    parser.add_argument("--mode", choices=["conversation", "cot"], default="conversation")
    parser.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    load_env()
    adapter_dir = train(
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        output_dir=args.output_dir,
        mode=args.mode,
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        per_device_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
    )
    print(f"[cli] done. adapter at: {adapter_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
