"""Inference: predict A/B/C/D for every test item, build submission.csv."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel

from .data import _get_content, _get_title
from .evaluate import to_chat_prompt
from .prompts import Mode, build_user_instruction
from .utils import ensure_dir, get_letter_token_ids


def _load_model(adapter_path: str, max_seq_length: int = 2048) -> tuple:
    """Load trained model + tokenizer via Unsloth from adapter directory."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def iter_test_items(test_dir: str) -> list[dict]:
    """Walk test JSON files; return items with row_id = {file_stem}__q{n}."""
    items: list[dict] = []
    for path in sorted(glob.glob(str(Path(test_dir) / "*.json"))):
        article_id = Path(path).stem
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        content = _get_content(d)
        title = _get_title(d)
        questions = d.get("questions") or []
        if not content or not questions:
            continue
        for idx, q in enumerate(questions, start=1):
            question = (q.get("question") or "").strip()
            choices = q.get("choices") or {}
            if not question or not choices:
                continue
            items.append({
                "article_id": article_id,
                "qid": idx,
                "row_id": f"{article_id}__q{idx}",
                "title": title,
                "content": content,
                "question": question,
                "choices": choices,
            })
    return items


def predict(
    model,
    tokenizer,
    items: list[dict],
    *,
    system_prompt: str = "",
    batch_size: int = 4,
    max_length: int = 2048,
    mode: Mode = "conversation",
) -> list[dict]:
    """Run inference using logits-mode (argmax over A/B/C/D tokens)."""
    letter_map = get_letter_token_ids(tokenizer)

    prompts = []
    for it in items:
        instr = build_user_instruction(
            it["title"], it["content"], it["question"], it["choices"],
            mode=mode,
        )
        prompts.append(to_chat_prompt(tokenizer, instr, system_prompt))

    results: list[dict] = []
    idx2letter = list(letter_map.keys())
    cand_ids = torch.tensor(list(letter_map.values()))

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Inferring"):
            ps = prompts[i : i + batch_size]
            enc = tokenizer(
                ps, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            ).to(model.device)

            outputs = model(**enc, num_logits_to_keep=1)
            next_logits = outputs.logits[:, 0, :]

            cand_ids_dev = cand_ids.to(outputs.logits.device)
            cand_logits = next_logits[:, cand_ids_dev]
            pred_idx = cand_logits.argmax(dim=1)

            for j, p in enumerate(pred_idx):
                results.append({
                    "row_id": items[i + j]["row_id"],
                    "answer": idx2letter[p.item()],
                })

            del enc, outputs
            torch.cuda.empty_cache()

    return results


def write_submission(results: list[dict], out_csv: str) -> Path:
    """Write row_id,answer CSV."""
    p = ensure_dir(Path(out_csv).parent) / Path(out_csv).name
    df = pd.DataFrame(results)
    df = df[["row_id", "answer"]]
    df.to_csv(p, index=False)
    print(f"[infer] wrote {len(df)} rows to {p}")
    return p
