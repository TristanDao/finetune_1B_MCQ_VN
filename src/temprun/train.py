"""Training: Unsloth + QLoRA 4-bit + SFT with `train_on_responses_only`.

Supports two modes:
- ``conversation``: short answer (single letter A/B/C/D).
- ``cot``: Chain-of-Thought (explanation + "Do đó, đáp án đúng là [X]").
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

from .data import read_jsonl
from .prompts import Mode
from .utils import ensure_dir, render_chat_for_training, set_seed

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"


def _load_model(
    model_name: str,
    max_seq_length: int,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float = 0.0,
    target_modules: list[str] | None = None,
) -> tuple:
    """Load model + tokenizer via Unsloth, attach LoRA."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
    )
    model.print_trainable_parameters()
    return model, tokenizer


def _render_conversation(tokenizer, messages: list[dict]) -> str:
    """Render for conversation mode: split assistant, add_generation_prompt=True."""
    return render_chat_for_training(
        tokenizer, messages,
        kwargs={"enable_thinking": False},
    )


def _render_cot(tokenizer, messages: list[dict]) -> str:
    """Render for CoT mode: full messages with add_generation_prompt=False."""
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    eos = getattr(tokenizer, "eos_token", None)
    if eos and not text.endswith(eos):
        text += eos
    return text


def train(
    train_jsonl: str,
    eval_jsonl: str,
    output_dir: str,
    *,
    mode: Mode = "conversation",
    model_name: str = DEFAULT_MODEL_NAME,
    max_seq_length: int | None = None,
    lora_r: int = 32,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    per_device_batch_size: int = 4,
    per_device_eval_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 10,
    eval_steps: int = 200,
    save_steps: int = 200,
    save_total_limit: int = 2,
    max_grad_norm: float = 0.3,
    optim: str = "adamw_8bit",
    packing: bool = False,
    seed: int = 3407,
    report_to: list[str] | None = None,
) -> str:
    """Run SFT training with Unsloth + QLoRA.

    Returns path to the saved adapter directory.
    """
    set_seed(seed)

    if report_to is None:
        report_to = []

    max_seq = max_seq_length or (2048 if mode == "conversation" else 4096)
    output_root = Path(output_dir)

    train_rows = read_jsonl(train_jsonl)
    eval_rows = read_jsonl(eval_jsonl)

    if not train_rows:
        raise FileNotFoundError(f"Train JSONL rỗng hoặc không tồn tại: {train_jsonl}")
    if not eval_rows:
        raise FileNotFoundError(f"Eval JSONL rỗng hoặc không tồn tại: {eval_jsonl}")

    print(f"[train] mode={mode} train={len(train_rows)} eval={len(eval_rows)}")
    print(f"[train] max_seq={max_seq} lora_r={lora_r} lora_alpha={lora_alpha}")

    model, tokenizer = _load_model(
        model_name, max_seq,
        lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
    )

    render_fn = _render_cot if mode == "cot" else _render_conversation

    def to_text(row):
        return {"text": render_fn(tokenizer, row["messages"])}

    train_ds = Dataset.from_list(train_rows).map(
        to_text, remove_columns=list(train_rows[0].keys()),
    )
    eval_ds = Dataset.from_list(eval_rows).map(
        to_text, remove_columns=list(eval_rows[0].keys()),
    )

    bf16 = torch.cuda.is_bf16_supported()

    sft_args = SFTConfig(
        output_dir=str(output_root),
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_accumulation_steps=4,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        logging_steps=20,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        max_grad_norm=max_grad_norm,
        optim=optim,
        fp16=not bf16,
        bf16=bf16,
        gradient_checkpointing=True,
        dataset_text_field="text",
        packing=packing,
        max_seq_length=max_seq,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=report_to,
        seed=seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()

    adapter_dir = output_root / "adapter"
    ensure_dir(adapter_dir)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    try:
        merged_dir = output_root / "merged_16bit"
        ensure_dir(merged_dir)
        model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
    except Exception as e:
        print(f"[train] save_pretrained_merged failed: {e}")

    cfg_path = output_root / "train_config.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name, "mode": mode,
            "max_seq_length": max_seq, "lora_r": lora_r, "lora_alpha": lora_alpha,
        }, f, ensure_ascii=False, indent=2)

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    print(f"[train] done. adapter: {adapter_dir}")
    return str(adapter_dir)
