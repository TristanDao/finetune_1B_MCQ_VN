"""Utility helpers: env loading, seeding, paths, generation parsing."""
from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Heavy ML deps are imported lazily so this module stays usable in lightweight
# contexts (e.g. unit tests, prompt building without GPU/torch installed).
def _import_torch():
    import torch  # type: ignore
    return torch


def _import_numpy():
    import numpy as np  # type: ignore
    return np

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env(env_path: str | os.PathLike | None = None) -> None:
    """Load .env from repo root (or given path). Safe to call multiple times."""
    if env_path is None:
        env_path = REPO_ROOT / ".env"
    load_dotenv(dotenv_path=env_path, override=False)


def repo_root() -> Path:
    return REPO_ROOT


def set_seed(seed: int) -> None:
    """Seed python, numpy, torch (cpu+cuda). Skips torch if not installed."""
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


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load a YAML config. If it has 'extends:', merge parent first.

    `extends:` is resolved relative to the child config's directory.
    """
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    cfg_path = cfg_path.resolve()
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    parent = cfg.pop("extends", None)
    if parent:
        parent_path = (cfg_path.parent / parent).resolve()
        parent_cfg = load_config(parent_path)
        parent_cfg = _deep_merge(parent_cfg, cfg)
        return parent_cfg
    return cfg


def _deep_merge(a: dict, b: dict) -> dict:
    """Deep-merge b into a (b wins). Returns new dict."""
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_generated(text: str) -> str | None:
    """Extract a single letter A-D from a model generation.

    Tries (in order):
      1. "Đáp án/Trả lời/Câu trả lời: X"
      2. "X" followed by ":" or "."
      3. Any standalone "X" token
    Returns uppercase letter or None.
    """
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


def get_letter_token_ids(tokenizer) -> dict[str, int] | None:
    """Map A/B/C/D to single-token ids. Returns None if any letter is multi-token."""
    letter_ids: dict[str, int] = {}
    for ch in ["A", "B", "C", "D"]:
        ids = tokenizer(ch, add_special_tokens=False).input_ids
        if len(ids) != 1:
            print(
                f"[warn] Letter '{ch}' is not a single token "
                f"(got {ids}). Logits mode disabled; falling back to generate.",
                file=sys.stderr,
            )
            return None
        letter_ids[ch] = ids[0]
    return letter_ids


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
