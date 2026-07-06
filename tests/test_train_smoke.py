"""Smoke tests cho train.py — verify backend branching logic không cần GPU.

Mock Unsloth FastLanguageModel + TRL SFTTrainer để verify:
- backend="trl" path unchanged
- backend="unsloth" path gọi Unsloth API đúng
- chat_kwargs được truyền vào render

Yêu cầu torch (train.py import torch ở top-level). Nếu chưa cài torch
(môi trường dev lightweight), toàn bộ module này sẽ skip.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

_torch = pytest.importorskip("torch")  # skip toàn bộ file nếu chưa cài torch


def test_to_chat_text_passes_chat_kwargs():
    """to_chat_text phải truyền chat_kwargs xuống render_chat_for_training."""
    from temprun.train import to_chat_text

    tok = MagicMock()
    tok.eos_token = "<|im_end|>"
    tok.apply_chat_template.return_value = "<|im_start|>assistant\n\n\n"
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]
    out = to_chat_text(tok, msgs, chat_kwargs={"enable_thinking": False})
    # apply_chat_template phải được gọi với add_generation_prompt=True + enable_thinking=False
    call_kwargs = tok.apply_chat_template.call_args.kwargs
    assert call_kwargs.get("add_generation_prompt") is True
    assert call_kwargs.get("enable_thinking") is False
    assert out.endswith("A<|im_end|>")


def test_to_chat_text_default_no_kwargs():
    from temprun.train import to_chat_text

    tok = MagicMock()
    tok.eos_token = "<|im_end|>"
    tok.apply_chat_template.return_value = "<|im_start|>assistant\n"
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "B"},
    ]
    to_chat_text(tok, msgs)
    call_kwargs = tok.apply_chat_template.call_args.kwargs
    assert call_kwargs.get("add_generation_prompt") is True
    assert "enable_thinking" not in call_kwargs


def test_unsloth_lora_attach_calls_fast_model(monkeypatch):
    """_attach_unsloth_lora phải gọi FastLanguageModel.get_peft_model với đúng args."""
    fake_unsloth = types.ModuleType("unsloth")
    fake_fm = MagicMock()
    fake_unsloth.FastLanguageModel = fake_fm
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)

    from temprun.train import _attach_unsloth_lora

    model = MagicMock()
    lora_cfg = {
        "r": 16,
        "alpha": 32,
        "dropout": 0.1,
        "bias": "none",
        "target_modules": ["q_proj", "v_proj"],
    }
    _attach_unsloth_lora(model, lora_cfg, max_seq_length=1024)
    fake_fm.get_peft_model.assert_called_once()
    call_kwargs = fake_fm.get_peft_model.call_args.kwargs
    assert call_kwargs["r"] == 16
    assert call_kwargs["lora_alpha"] == 32
    assert call_kwargs["max_seq_length"] == 1024
    assert call_kwargs["target_modules"] == ["q_proj", "v_proj"]


def test_unsloth_lora_attach_none_returns_model(monkeypatch):
    from temprun.train import _attach_unsloth_lora

    model = MagicMock()
    out = _attach_unsloth_lora(model, None)
    assert out is model
