"""Tests for utils (env, seeding, config, letter-token-id check uses a tiny mock)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from temprun.utils import (
    _deep_merge,
    get_letter_token_ids,
    load_config,
    load_env,
    parse_generated,
    render_chat_for_inference,
    render_chat_for_training,
    set_seed,
)


def test_deep_merge_basic():
    a = {"x": {"y": 1, "z": 2}, "w": 0}
    b = {"x": {"y": 10}, "q": 9}
    out = _deep_merge(a, b)
    assert out == {"x": {"y": 10, "z": 2}, "w": 0, "q": 9}


def test_load_config_with_extends(tmp_path: Path):
    base = tmp_path / "base.yaml"
    base.write_text("a: 1\nb: {c: 2}\n", encoding="utf-8")
    child = tmp_path / "child.yaml"
    child.write_text("extends: base.yaml\nb: {c: 99}\nd: 4\n", encoding="utf-8")
    # Need to set CWD to tmp_path for the extends resolution
    import os
    old = os.getcwd()
    try:
        os.chdir(tmp_path)
        cfg = load_config(child)
    finally:
        os.chdir(old)
    assert cfg["a"] == 1
    assert cfg["b"]["c"] == 99
    assert cfg["d"] == 4


def test_load_env_missing_does_not_raise(tmp_path: Path):
    load_env(tmp_path / "nonexistent.env")  # should silently no-op
    # No assertion: just ensure no exception


def test_set_seed_is_idempotent():
    import random
    import numpy as np
    set_seed(123)
    a = random.random()
    set_seed(123)
    b = random.random()
    assert a == b


class _MockTok:
    """Minimal stand-in for HF AutoTokenizer to test get_letter_token_ids."""

    def __init__(self, mapping: dict[str, list[int]]):
        self._m = mapping

    def __call__(self, text, add_special_tokens=True):
        ids = self._m.get(text, [99, 99])
        return type("E", (), {"input_ids": ids})()


def test_get_letter_token_ids_ok():
    tok = _MockTok({"A": [10], "B": [11], "C": [12], "D": [13]})
    out = get_letter_token_ids(tok)
    assert out == {"A": 10, "B": 11, "C": 12, "D": 13}


def test_get_letter_token_ids_multi_token_returns_none():
    tok = _MockTok({"A": [10, 20], "B": [11], "C": [12], "D": [13]})
    assert get_letter_token_ids(tok) is None


class _MockQwenTok:
    """Mô phỏng chat template Qwen3 với enable_thinking.

    Khi add_generation_prompt=True + enable_thinking=False →
    `<|im_start|>assistant\\n` + `\\n\\n` (empty thinking block).
    """

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


def test_render_chat_for_training_includes_assistant_after_thinking_off():
    tok = _MockQwenTok()
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
        {"role": "assistant", "content": "A"},
    ]
    text = render_chat_for_training(tok, msgs, kwargs={"enable_thinking": False})
    # Prefix phải có assistant header + empty thinking block, rồi mới "A" + eos
    assert "<|im_start|>assistant\n\n\nA<|im_end|>" in text
    assert text.endswith("<|im_end|>")


def test_render_chat_for_inference_matches_training_prefix():
    tok = _MockQwenTok()
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
        {"role": "assistant", "content": "A"},  # should be dropped
    ]
    infer_text = render_chat_for_inference(tok, msgs, kwargs={"enable_thinking": False})
    train_msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
        {"role": "assistant", "content": "A"},
    ]
    train_text = render_chat_for_training(tok, train_msgs, kwargs={"enable_thinking": False})
    # Infer prefix phải là prefix của train text (khớp đến hết assistant header + empty thinking)
    assert infer_text in train_text
    assert infer_text.endswith("<|im_start|>assistant\n\n\n")


def test_render_chat_no_kwargs_still_works():
    tok = _MockQwenTok()
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "USR"},
        {"role": "assistant", "content": "B"},
    ]
    text = render_chat_for_training(tok, msgs, kwargs=None)
    # Không truyền enable_thinking → không có empty thinking block,但仍 có assistant header
    assert "<|im_start|>assistant\nB<|im_end|>" in text
