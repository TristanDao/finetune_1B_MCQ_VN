"""Utility helpers: env loading, seeding, paths, chat rendering, generation parsing."""

from __future__ import annotations

import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_torch():
    import torch
    return torch


def _import_numpy():
    import numpy as np
    return np


def load_env(env_path: str | os.PathLike | None = None) -> None:
    if env_path is None:
        env_path = REPO_ROOT / ".env"
    load_dotenv(dotenv_path=env_path, override=False)


def repo_root() -> Path:
    return REPO_ROOT


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        np = _import_numpy()
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        torch = _import_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_generated(text: str) -> str | None:
    """Extract a single letter A-D from model generation."""
    if not text:
        return None
    m = re.search(r"(?:Đáp án|Trả lời|Câu\s*trả\s*lời)\s*:?\s*([A-D])", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])[:.]?\b", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1).upper()
    return None


def get_letter_token_ids(tokenizer) -> dict[str, int]:
    """Map A/B/C/D to single-token ids.

    Raises RuntimeError if any letter is multi-token (logits mode won't work).
    """
    letter_ids: dict[str, int] = {}
    for ch in ["A", "B", "C", "D"]:
        ids = tokenizer(ch, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise RuntimeError(
                f"Letter '{ch}' is not a single token (got {ids}). Logits mode won't work."
            )
        letter_ids[ch] = ids[0]
    return letter_ids


def render_chat_for_training(tokenizer, messages: list[dict], kwargs: dict | None = None) -> str:
    """Render messages to a string for SFT training.

    Splits out assistant message, renders prefix with `add_generation_prompt=True`,
    then appends `assistant_content + eos`. This ensures train and infer use the
    same prefix.
    """
    assistant_content = None
    prefix_msgs: list[dict] = []
    for m in messages:
        if m.get("role") == "assistant":
            assistant_content = m.get("content", "")
        else:
            prefix_msgs.append(m)

    render_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if kwargs:
        render_kwargs.update(kwargs)
    text = tokenizer.apply_chat_template(prefix_msgs, **render_kwargs)

    if assistant_content is not None:
        text += assistant_content

    eos = getattr(tokenizer, "eos_token", None)
    if eos and not text.endswith(eos):
        text += eos
    return text


def render_chat_for_inference(tokenizer, messages: list[dict], kwargs: dict | None = None) -> str:
    """Render prefix (system+user) for inference with `add_generation_prompt=True`.

    Drops any assistant message. Use `enable_thinking=False` in kwargs.
    """
    prefix_msgs = [m for m in messages if m.get("role") != "assistant"]
    render_kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    if kwargs:
        render_kwargs.update(kwargs)
    return tokenizer.apply_chat_template(prefix_msgs, **render_kwargs)
