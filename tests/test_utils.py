"""Tests for utils: seeding, letter-token-ids, chat rendering."""

from __future__ import annotations

from pathlib import Path

import pytest

from temprun.utils import (
    get_letter_token_ids,
    load_env,
    render_chat_for_inference,
    render_chat_for_training,
    set_seed,
)


def test_load_env_missing_does_not_raise(tmp_path: Path):
    load_env(tmp_path / "nonexistent.env")


def test_set_seed_is_idempotent():
    import random

    import numpy as np
    set_seed(123)
    a = random.random()
    b = np.random.random()
    set_seed(123)
    c = random.random()
    d = np.random.random()
    assert a == c
    assert b == d


class _MockTok:
    def __init__(self, mapping: dict[str, list[int]]):
        self._m = mapping

    def __call__(self, text, add_special_tokens=True):
        ids = self._m.get(text, [99, 99])
        return type("E", (), {"input_ids": ids})()


def test_get_letter_token_ids_ok():
    tok = _MockTok({"A": [10], "B": [11], "C": [12], "D": [13]})
    out = get_letter_token_ids(tok)
    assert out == {"A": 10, "B": 11, "C": 12, "D": 13}


def test_get_letter_token_ids_multi_token_raises():
    tok = _MockTok({"A": [10, 20], "B": [11], "C": [12], "D": [13]})
    with pytest.raises(RuntimeError, match="not a single token"):
        get_letter_token_ids(tok)


class _MockQwenTok:
    eos_token = "<|im_end|>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        out = ""
        for m in messages:
            out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n"
            if kwargs.get("enable_thinking") is False:
                out += "\n\n"
        return out


def test_render_chat_for_training_includes_assistant():
    tok = _MockQwenTok()
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
        {"role": "assistant", "content": "A"},
    ]
    text = render_chat_for_training(tok, msgs, kwargs={"enable_thinking": False})
    assert "<|im_start|>assistant\n\n\nA<|im_end|>" in text
    assert text.endswith("<|im_end|>")


def test_render_chat_for_inference_matches_training_prefix():
    tok = _MockQwenTok()
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
        {"role": "assistant", "content": "A"},
    ]
    infer_text = render_chat_for_inference(tok, msgs, kwargs={"enable_thinking": False})
    train_msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
        {"role": "assistant", "content": "A"},
    ]
    train_text = render_chat_for_training(tok, train_msgs, kwargs={"enable_thinking": False})
    assert infer_text in train_text
    assert infer_text.endswith("<|im_start|>assistant\n\n\n")


def test_render_chat_no_kwargs():
    tok = _MockQwenTok()
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
        {"role": "assistant", "content": "B"},
    ]
    text = render_chat_for_training(tok, msgs, kwargs=None)
    assert "<|im_start|>assistant\nB<|im_end|>" in text
