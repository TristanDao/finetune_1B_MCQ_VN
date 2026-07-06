"""CLI for inference: predict labels for every test JSON, write submission.csv."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from temprun.infer import iter_test_items, load_model_for_inference, predict, write_submission
from temprun.utils import load_config, load_env, repo_root


def _load_default_system_prompt() -> str:
    cfg_path = repo_root() / "configs" / "base.yaml"
    if cfg_path.exists():
        try:
            cfg = load_config(cfg_path)
            return cfg.get("system_prompt", "Bạn là hệ thống trả lời trắc nghiệm. Chỉ xuất đúng 1 ký tự A/B/C/D.")
        except Exception:  # noqa: BLE001
            pass
    return "Bạn là hệ thống trả lời trắc nghiệm. Chỉ xuất đúng 1 ký tự A/B/C/D."


def main(args=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=["logits", "generate"], default="logits")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", dest="use_4bit", action="store_false")
    parser.add_argument("--system-prompt", default=None)
    parsed = parser.parse_args(args)

    load_env()
    system_prompt = parsed.system_prompt or _load_default_system_prompt()

    base_model = parsed.base_model
    train_cfg_dict: dict = {}
    if not base_model:
        train_cfg_path = Path(parsed.checkpoint) / "train_config.json"
        if train_cfg_path.exists():
            train_cfg_dict = json.loads(train_cfg_path.read_text(encoding="utf-8"))
            base_model = train_cfg_dict.get("model_name")

    # If checkpoint doesn't have train_config.json, try parent (Unsloth saves under adapter/)
    if not base_model and not train_cfg_dict:
        parent_cfg = Path(parsed.checkpoint).parent / "train_config.json"
        if parent_cfg.exists():
            train_cfg_dict = json.loads(parent_cfg.read_text(encoding="utf-8"))
            base_model = train_cfg_dict.get("model_name")

    chat_kwargs = train_cfg_dict.get("chat_template_kwargs") or {}
    if not chat_kwargs:
        try:
            base_cfg = load_config(repo_root() / "configs" / "base.yaml")
            chat_kwargs = base_cfg.get("chat_template_kwargs") or {}
        except Exception:  # noqa: BLE001
            chat_kwargs = {}

    adapter = None
    if (Path(parsed.checkpoint) / "adapter_config.json").exists():
        adapter = parsed.checkpoint
        if not base_model:
            raise RuntimeError("LoRA adapter found but no base model. Pass --base-model.")
    else:
        base_model = parsed.checkpoint

    print(f"[infer] base={base_model} adapter={adapter or '(none - standalone model)'} test_dir={parsed.test_dir} chat_kwargs={chat_kwargs}")
    model, tokenizer = load_model_for_inference(base_model, adapter, use_4bit=parsed.use_4bit)

    items = iter_test_items(parsed.test_dir)
    print(f"[infer] {len(items)} questions from {parsed.test_dir}")

    results = predict(
        model,
        tokenizer,
        items,
        system_prompt=system_prompt,
        batch_size=parsed.batch_size,
        max_length=parsed.max_length,
        mode=parsed.mode,
        chat_kwargs=chat_kwargs,
    )
    out = write_submission(results, parsed.out)
    print(f"[infer] submission written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
