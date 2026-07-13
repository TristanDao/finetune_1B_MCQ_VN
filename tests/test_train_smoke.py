"""Smoke tests for train.py — verify render and model-load branching logic without GPU.

Requires torch (train.py imports torch).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

_torch = pytest.importorskip("torch")


def test_render_conversation_adds_generation_prompt():
    from temprun.train import _render_conversation

    tok = MagicMock()
    tok.eos_token = "<|im_end|>"
    tok.apply_chat_template.return_value = "<|im_start|>assistant\n\n\n"
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "A"},
    ]
    out = _render_conversation(tok, msgs)
    call_kwargs = tok.apply_chat_template.call_args.kwargs
    assert call_kwargs.get("add_generation_prompt") is True
    assert call_kwargs.get("enable_thinking") is False
    assert out.endswith("A<|im_end|>")


def test_render_cot_no_generation_prompt():
    from temprun.train import _render_cot

    tok = MagicMock()
    tok.eos_token = "<|im_end|>"
    tok.apply_chat_template.return_value = "<|im_start|>assistant\nexplanation...<|im_end|>"
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "Do đó, đáp án đúng là A."},
    ]
    out = _render_cot(tok, msgs)
    call_kwargs = tok.apply_chat_template.call_args.kwargs
    assert call_kwargs.get("add_generation_prompt") is False
    assert call_kwargs.get("enable_thinking") is False
    assert "Do đó, đáp án đúng là A." in out
