"""Evaluation: batch inference, accuracy, confusion matrix, JSONL details.

Refactored from the original Colab notebook BaselineTempoRun.ipynb.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from tqdm import tqdm

from .utils import ensure_dir, get_letter_token_ids, parse_generated, render_chat_for_inference


def to_chat_prompt(
    tokenizer,
    instruction: str,
    system_prompt: str,
    chat_kwargs: dict | None = None,
) -> str:
    """Apply chat template to a (system, user) pair, return string for tokenization.

    Renders with `add_generation_prompt=True` + `enable_thinking=False` (qua
    chat_kwargs) để model xuất A/B/C/D trực tiếp, khớp với train.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]
    return render_chat_for_inference(tokenizer, messages, kwargs=chat_kwargs)


def eval_logits_mode(
    model,
    tokenizer,
    prompts: list[str],
    labels: list[str],
    max_length: int,
) -> tuple[list[dict], int]:
    """One forward pass; pick argmax of next-token logits over {A,B,C,D}."""
    letter_map = get_letter_token_ids(tokenizer)
    if letter_map is None:
        return None  # type: ignore[return-value]

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
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
    for p, g in zip(preds, labels):
        ok = p == g
        correct += int(ok)
        details.append({"pred": p, "label": g, "is_correct": ok})
    return details, correct


def eval_generate_mode(
    model,
    tokenizer,
    prompts: list[str],
    labels: list[str],
    max_length: int,
    max_new_tokens: int = 4,
) -> tuple[list[dict], int]:
    """Greedy short generation; parse first A-D from raw text."""
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    gens = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
    details: list[dict] = []
    correct = 0
    for g, lab in zip(gens, labels):
        pred = parse_generated(g) or "E"  # 'E' = parse failed
        ok = pred == lab
        correct += int(ok)
        details.append({"pred": pred, "label": lab, "raw": g.strip(), "is_correct": ok})
    return details, correct


def run_eval(
    model,
    tokenizer,
    rows: list[dict],
    *,
    system_prompt: str,
    batch_size: int = 16,
    max_length: int = 2048,
    mode: str = "logits",
    out_jsonl: str | Path | None = None,
    instruction_fn: Callable[[dict], str] | None = None,
    chat_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Run full evaluation. `rows` must have keys: 'instruction' (str) and 'label' (str).

    If `instruction_fn` is given, called as `instruction_fn(row)` to get the prompt
    text (allows custom row shapes). Otherwise expects row['instruction'].

    `chat_kwargs` được truyền vào `apply_chat_template` (vd `enable_thinking=False`
    cho Qwen3) để khớp với prefix dùng lúc train.
    """
    if instruction_fn is None:
        instruction_fn = lambda r: r["instruction"]  # noqa: E731

    prompts = [to_chat_prompt(tokenizer, instruction_fn(r), system_prompt, chat_kwargs=chat_kwargs) for r in rows]
    labels = [r["label"] for r in rows]
    total = len(rows)
    all_details: list[dict] = []
    correct = 0
    current_mode = mode
    start = time.time()

    for i in tqdm(range(0, total, batch_size), desc=f"Evaluating ({current_mode})"):
        ps = prompts[i : i + batch_size]
        ls = labels[i : i + batch_size]
        res: tuple[list[dict], int] | None = None

        if current_mode == "logits":
            r = eval_logits_mode(model, tokenizer, ps, ls, max_length)
            if r is None:
                print("[info] letters not single-token; falling back to generate mode")
                current_mode = "generate"

        if current_mode == "generate":
            r = eval_generate_mode(model, tokenizer, ps, ls, max_length)

        assert r is not None
        details, corr = r
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
    print(f"Items/sec:  {total / elapsed:.2f}" if elapsed else "n/a")
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
        "total": total,
        "correct": correct,
        "accuracy": acc,
        "time_sec": elapsed,
        "items_per_sec": total / elapsed if elapsed else 0.0,
        "pred_dist": dict(dist),
        "confusion": {lab: dict(cm[lab]) for lab in cm},
        "mode": current_mode,
    }
