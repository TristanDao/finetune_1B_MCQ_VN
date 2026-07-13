"""Evaluation: batch inference via logits-mode, accuracy, confusion matrix."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .utils import ensure_dir, get_letter_token_ids, render_chat_for_inference


def to_chat_prompt(tokenizer, instruction: str, system_prompt: str) -> str:
    """Apply chat template to (system, user) pair for inference."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]
    return render_chat_for_inference(
        tokenizer, messages,
        kwargs={"enable_thinking": False},
    )


def eval_logits_mode(
    model,
    tokenizer,
    prompts: list[str],
    labels: list[str],
    max_length: int,
) -> tuple[list[dict], int]:
    """One forward pass; pick argmax of next-token logits over {A,B,C,D}."""
    letter_map = get_letter_token_ids(tokenizer)
    enc = tokenizer(
        prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_length,
    ).to(model.device)

    with torch.no_grad():
        logits = model(**enc).logits
        last_pos = enc.attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        next_logits = logits[batch_idx, last_pos, :]
        cand_ids = torch.tensor(list(letter_map.values()), device=logits.device)
        cand_logits = next_logits[:, cand_ids]
        pred_idx = cand_logits.argmax(dim=1)

    idx2letter = list(letter_map.keys())
    preds = [idx2letter[i.item()] for i in pred_idx]

    details: list[dict] = []
    correct = 0
    for p, g in zip(preds, labels, strict=False):
        ok = p == g
        correct += int(ok)
        details.append({"pred": p, "label": g, "is_correct": ok})
    return details, correct


def run_eval(
    model,
    tokenizer,
    rows: list[dict],
    *,
    system_prompt: str,
    batch_size: int = 16,
    max_length: int = 2048,
    out_jsonl: str | Path | None = None,
) -> dict[str, Any]:
    """Run evaluation on eval rows (must have 'messages' and 'label' keys)."""
    prompts = []
    labels = []
    for r in rows:
        prompts.append(to_chat_prompt(tokenizer, r["messages"][1]["content"], system_prompt))
        labels.append(r["label"])

    total = len(rows)
    all_details: list[dict] = []
    correct = 0
    start = time.time()

    for i in tqdm(range(0, total, batch_size), desc="Evaluating (logits)"):
        ps = prompts[i : i + batch_size]
        ls = labels[i : i + batch_size]
        details, corr = eval_logits_mode(model, tokenizer, ps, ls, max_length)
        all_details.extend(details)
        correct += corr

    elapsed = time.time() - start
    acc = 100.0 * correct / total if total else 0.0

    cm = defaultdict(lambda: Counter())
    dist = Counter()
    for d in all_details:
        dist[d["pred"]] += 1
        cm[d["label"]][d["pred"]] += 1

    if out_jsonl is not None:
        out_path = ensure_dir(Path(out_jsonl).parent) / Path(out_jsonl).name
        with open(out_path, "w", encoding="utf-8") as f:
            for d in all_details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    letters = sorted(set(cm.keys()) | set(dist.keys()))
    print("\n=== EVAL SUMMARY ===")
    print(f"Total:      {total}")
    print(f"Correct:    {correct}")
    print(f"Accuracy:   {acc:.2f}%")
    print(f"Time:       {elapsed:.2f}s")
    print(f"Pred dist:  {dict(dist)}")
    print("Confusion (rows=label, cols=pred):")
    header = "     " + "  ".join(f"{c:>5}" for c in letters)
    print(header)
    for lab in letters:
        if lab not in cm:
            continue
        row = [cm[lab][p] for p in letters]
        print(f"{lab:>3}: " + "  ".join(f"{n:>5}" for n in row))

    return {
        "total": total, "correct": correct, "accuracy": acc,
        "time_sec": elapsed,
        "items_per_sec": total / elapsed if elapsed else 0.0,
        "pred_dist": dict(dist),
        "confusion": {lab: dict(cm[lab]) for lab in cm},
    }
