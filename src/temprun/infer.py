"""Inference: predict A/B/C/D for every test article, build submission.csv."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from .data import _get_content, _get_title, load_json_files
from .utils import ensure_dir, get_letter_token_ids, parse_generated, set_seed

# Heavy ML deps are imported lazily inside functions.


def load_model_for_inference(base_model: str, adapter_path: str | None, *, use_4bit: bool = True, attn_impl: str = "flash_attention_2"):
    """Load model + tokenizer. If adapter_path is given, attach LoRA on top.

    `attn_impl` defaults to "flash_attention_2" (nếu flash-attn chưa cài, HF
    tự fallback sang sdpa/eager). Đối với Unsloth adapter, dùng
    `FastLanguageModel.for_inference` nếu Unsloth có sẵn — nhưng peft path
    chuẩn vẫn hoạt động.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bf16 = torch.cuda.is_bf16_supported()
    bnb = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        if use_4bit
        else None
    )
    # Try flash_attention_2 first; fall back silently if flash-attn not installed.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16 if bf16 else torch.float16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
            quantization_config=bnb,
        )
    except Exception as e:  # noqa: BLE001
        if "flash" in str(e).lower() or "FlashAttention" in str(e):
            print(f"[infer] flash_attention_2 unavailable ({e}); falling back to sdpa")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                torch_dtype=torch.bfloat16 if bf16 else torch.float16,
                attn_implementation="sdpa",
                trust_remote_code=True,
                quantization_config=bnb,
            )
        else:
            raise

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        # Merge for faster inference
        model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_questions_for_inference(test_dir: str | os.PathLike) -> list[dict]:
    """Walk test JSON files; return list of {article_id, qid, row_id, title, content, question, choices}."""
    items: list[dict] = []
    docs = load_json_files(test_dir)
    for d in docs:
        content = _get_content(d)
        title = _get_title(d)
        questions = d.get("questions") or []
        if not content or not questions:
            continue
        article_id = Path(d.get("url", "x")).stem or "x"
        # NOTE: original submission uses the file stem (e.g. "0164aa98"), not the URL stem.
        # We'll override below with the file stem by re-globbing. Simpler: re-walk.
    return items


def iter_test_items(test_dir: str | os.PathLike) -> list[dict]:
    """Walk test JSON files; return items with row_id = {file_stem}__q{n}."""
    import glob
    items: list[dict] = []
    for path in sorted(glob.glob(os.path.join(str(test_dir), "*.json"))):
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
            items.append(
                {
                    "article_id": article_id,
                    "qid": idx,
                    "row_id": f"{article_id}__q{idx}",
                    "title": title,
                    "content": content,
                    "question": question,
                    "choices": choices,
                }
            )
    return items


def predict(
    model,
    tokenizer,
    items: list[dict],
    *,
    system_prompt: str,
    batch_size: int = 16,
    max_length: int = 2048,
    mode: str = "logits",
    chat_kwargs: dict | None = None,
) -> list[dict]:
    """Run inference; return [{row_id, pred}] list (in input order)."""
    from .evaluate import eval_generate_mode, eval_logits_mode, to_chat_prompt
    from .prompts import build_user_instruction

    prompts_text = [
        build_user_instruction(it["title"], it["content"], it["question"], it["choices"]) for it in items
    ]
    prompts = [to_chat_prompt(tokenizer, t, system_prompt, chat_kwargs=chat_kwargs) for t in prompts_text]

    results: list[dict] = []
    current_mode = mode
    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Inferring ({current_mode})"):
        ps = prompts[i : i + batch_size]
        sub_items = items[i : i + batch_size]

        if current_mode == "logits":
            letter_map = get_letter_token_ids(tokenizer)
            if letter_map is None:
                current_mode = "generate"

        if current_mode == "logits":
            details, _ = eval_logits_mode(model, tokenizer, ps, [it["row_id"] for it in sub_items], max_length)
            # `details` only has pred/label/is_correct; we need pred
            for it, d in zip(sub_items, details):
                results.append({"row_id": it["row_id"], "pred": d["pred"]})
        else:
            details, _ = eval_generate_mode(model, tokenizer, ps, [it["row_id"] for it in sub_items], max_length)
            for it, d in zip(sub_items, details):
                results.append({"row_id": it["row_id"], "pred": d["pred"]})

    return results


def write_submission(results: list[dict], out_csv: str | os.PathLike) -> Path:
    """Write row_id,answer CSV."""
    p = ensure_dir(Path(out_csv).parent) / Path(out_csv).name
    df = pd.DataFrame(results).rename(columns={"pred": "answer"})
    df = df[["row_id", "answer"]]
    df.to_csv(p, index=False)
    print(f"[infer] wrote {len(df)} rows to {p}")
    return p
