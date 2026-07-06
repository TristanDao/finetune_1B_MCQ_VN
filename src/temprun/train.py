"""Training entry: QLoRA 4-bit + SFT, or full FT bf16 + SFT.

Supports two backends via `cfg["backend"]`:
- "trl" (default): HF transformers + TRL SFTTrainer.
- "unsloth": Unsloth FastLanguageModel (Triton kernels, ~2x faster, FA2 built-in).

Uses `completion_only_loss=True` so loss is computed only on the assistant's
single-letter response (not the system/user prompt). Chat template is rendered
with `enable_thinking=False` (Qwen3) to match between train and infer.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from .data import read_jsonl
from .utils import ensure_dir, render_chat_for_training, set_seed


def _make_bnb_config(quant: dict | None) -> BitsAndBytesConfig | None:
    if not quant:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=bool(quant.get("load_in_4bit", True)),
        bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=getattr(torch, quant.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_use_double_quant=bool(quant.get("bnb_4bit_use_double_quant", True)),
    )


def _make_lora_config(lora: dict | None) -> LoraConfig | None:
    if not lora:
        return None
    return LoraConfig(
        r=int(lora["r"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora.get("dropout", 0.05)),
        bias=lora.get("bias", "none"),
        target_modules=list(lora["target_modules"]),
        task_type=TaskType.CAUSAL_LM,
    )


def load_tokenizer(model_name: str, tok_cfg: dict):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=bool(tok_cfg.get("trust_remote_code", True))
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, quant: dict | None, bf16: bool, attn_impl: str = "eager"):
    bnb_config = _make_bnb_config(quant)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        try:
            model.enable_input_require_grads()
        except Exception:  # noqa: BLE001
            pass
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def attach_lora(model, lora_cfg: dict | None):
    if not lora_cfg:
        return model
    peft_cfg = _make_lora_config(lora_cfg)
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


def to_chat_text(tokenizer, messages: list[dict], chat_kwargs: dict | None = None) -> str:
    """Render messages as one string with chat template, ensuring EOS.

    Uses `render_chat_for_training` so the assistant turn is rendered with
    `add_generation_prompt=True` + `enable_thinking=False` (when chat_kwargs
    has it), matching the inference-time prefix exactly.
    """
    return render_chat_for_training(tokenizer, messages, kwargs=chat_kwargs)


def _load_unsloth_model(model_name: str, max_seq_length: int, load_in_4bit: bool, dtype: str = "bf16"):
    """Load model + tokenizer via Unsloth FastLanguageModel.

    Returns (model, tokenizer). Unsloth handles FA2 + Triton kernels internally.
    """
    from unsloth import FastLanguageModel  # type: ignore[import-not-found]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None if dtype == "auto" else dtype,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


def _attach_unsloth_lora(model, lora_cfg: dict | None, max_seq_length: int = 2048):
    """Attach LoRA via Unsloth's get_peft_model (uses Triton-backed LoRA)."""
    if not lora_cfg:
        return model
    from unsloth import FastLanguageModel  # type: ignore[import-not-found]

    model = FastLanguageModel.get_peft_model(
        model,
        r=int(lora_cfg["r"]),
        target_modules=list(lora_cfg["target_modules"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none" if lora_cfg.get("bias", "none") == "none" else lora_cfg["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=max_seq_length,
        loftq_config=None,
    )
    model.print_trainable_parameters()
    return model


def train(
    cfg: dict[str, Any],
    *,
    output_dir: str | os.PathLike | None = None,
) -> tuple[str, str]:
    """Run training. Returns (output_dir, tokenizer_path)."""
    seed = int(cfg.get("seed", 3407))
    set_seed(seed)

    model_name = cfg["model_name"]
    run_name = cfg.get("run_name", "run")
    output_root = Path(cfg.get("output_root", "artifacts"))
    output_dir = Path(output_dir) if output_dir else (output_root / run_name)
    ensure_dir(output_dir)

    data_cfg = cfg["data"]
    tok_cfg = cfg.get("tokenizer", {})
    train_cfg = cfg["training"]
    quant_cfg = cfg.get("quant")
    lora_cfg = cfg.get("lora")
    bf16 = bool(train_cfg.get("bf16", torch.cuda.is_bf16_supported()))
    fp16 = bool(train_cfg.get("fp16", not bf16))
    attn_impl = cfg.get("attn_implementation", "eager")
    backend = cfg.get("backend", "trl")
    chat_kwargs = cfg.get("chat_template_kwargs") or {}
    max_seq = int(data_cfg.get("max_seq_length", 2048))

    print(
        f"[train] model={model_name} run={run_name} backend={backend} "
        f"bf16={bf16} quant={'4bit' if quant_cfg else 'none'} "
        f"lora={'yes' if lora_cfg else 'no'} attn={attn_impl} "
        f"chat_kwargs={chat_kwargs}"
    )

    if backend == "unsloth":
        load_in_4bit = bool(quant_cfg and quant_cfg.get("load_in_4bit", True))
        dtype = "bf16" if bf16 else ("fp16" if fp16 else "auto")
        model, tokenizer = _load_unsloth_model(model_name, max_seq, load_in_4bit, dtype=dtype)
        model = _attach_unsloth_lora(model, lora_cfg, max_seq_length=max_seq)
    else:
        tokenizer = load_tokenizer(model_name, tok_cfg)
        model = load_model(model_name, quant_cfg, bf16=bf16, attn_impl=attn_impl)
        model = attach_lora(model, lora_cfg)

    # Load JSONL → HF Dataset with `text` field (chat-rendered, EOS-terminated)
    train_rows = read_jsonl(data_cfg["train_jsonl"])
    eval_rows = read_jsonl(data_cfg["eval_jsonl"])

    if not train_rows:
        raise FileNotFoundError(
            f"Train JSONL rỗng hoặc không tồn tại: {data_cfg['train_jsonl']}\n"
            "Hãy chạy BƯỚC 1 (download_data + make_sft_jsonl) và BƯỚC 2.3 (merge_enriched) trước."
        )
    if not eval_rows:
        raise FileNotFoundError(
            f"Eval JSONL rỗng hoặc không tồn tại: {data_cfg['eval_jsonl']}\n"
            "Hãy chạy BƯỚC 1 (download_data + make_sft_jsonl) và BƯỚC 2.3 (merge_enriched) trước."
        )

    def _render(row):
        return {"text": to_chat_text(tokenizer, row["messages"], chat_kwargs=chat_kwargs)}

    train_ds = Dataset.from_list(train_rows).map(_render, remove_columns=list(train_rows[0].keys()))
    eval_ds = Dataset.from_list(eval_rows).map(_render, remove_columns=list(eval_rows[0].keys()))
    print(f"[train] rows: train={len(train_ds)} eval={len(eval_ds)}")

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", train_cfg["per_device_train_batch_size"])),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        num_train_epochs=int(train_cfg.get("num_train_epochs", 1)),
        max_steps=int(train_cfg.get("max_steps", -1)),
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.06)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=int(train_cfg.get("logging_steps", 20)),
        eval_steps=int(train_cfg.get("eval_steps", 200)),
        save_steps=int(train_cfg.get("save_steps", 200)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.3)),
        optim=train_cfg.get("optim", "adamw_bnb_8bit"),
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        torch_compile=bool(train_cfg.get("torch_compile", False)),
        report_to=[],  # disable W&B by default
        seed=seed,
        dataset_text_field="text",
        packing=bool(train_cfg.get("packing", True)),
        max_length=int(data_cfg.get("max_seq_length", 2048)),
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        completion_only_loss=bool(train_cfg.get("assistant_only_loss", True)),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    if backend == "unsloth":
        # Unsloth: save merged 16-bit + LoRA adapter. Adapter để infer load bằng peft.
        adapter_dir = output_dir / "adapter"
        ensure_dir(adapter_dir)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        # Also save a merged 16-bit checkpoint for standalone inference
        try:
            merged_dir = output_dir / "merged_16bit"
            ensure_dir(merged_dir)
            model.save_pretrained_merged(str(merged_dir), tokenizer, save_method="merged_16bit")
        except Exception as e:  # noqa: BLE001
            print(f"[train] warn: save_pretrained_merged failed: {e}")
        # Point output_dir to adapter for downstream eval/infer (LoRA path)
        trainer_save_dir = str(adapter_dir)
    else:
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        trainer_save_dir = str(output_dir)

    # Save config snapshot for reproducibility
    import json as _json
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        _json.dump(cfg, f, ensure_ascii=False, indent=2, default=str)

    return str(output_dir), trainer_save_dir
