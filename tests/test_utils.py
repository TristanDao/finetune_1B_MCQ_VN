"""Tests for utils (env, seeding, config, letter-token-id check uses a tiny mock)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from temprun.utils import _deep_merge, get_letter_token_ids, load_config, load_env, parse_generated, set_seed


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
