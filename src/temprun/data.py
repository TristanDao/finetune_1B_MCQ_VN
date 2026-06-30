"""Data loading: read raw JSON, build SFT/eval JSONL, stratified split."""
from __future__ import annotations

import glob
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from .prompts import make_row
from .utils import ensure_dir


def load_json_files(dataset_path: str | os.PathLike) -> list[dict]:
    """Load all *.json in a directory into a list of dicts."""
    files = sorted(glob.glob(os.path.join(str(dataset_path), "*.json")))
    out: list[dict] = []
    for p in files:
        try:
            with open(p, encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] Failed to read {p}: {e}")
    return out


def _get_content(d: dict) -> str:
    return (d.get("content:") or d.get("content") or "").strip()


def _get_title(d: dict) -> str:
    return (d.get("title:") or d.get("title") or "").strip()


def iter_qa(dataset_path: str | os.PathLike) -> Iterable[dict]:
    """Yield QA items: {article_id, title, content, question, choices, label_or_None}."""
    for path in sorted(glob.glob(os.path.join(str(dataset_path), "*.json"))):
        article_id = Path(path).stem
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        content = _get_content(data)
        title = _get_title(data)
        questions = data.get("questions") or []
        if not content or not questions:
            continue
        for idx, q in enumerate(questions, start=1):
            question = (q.get("question") or "").strip()
            choices = q.get("choices") or {}
            raw_label = q.get("correct_answer")
            label = raw_label.strip().upper()[:1] if raw_label else None
            if not question or not choices:
                continue
            if label is not None and label not in {"A", "B", "C", "D"}:
                continue
            yield {
                "article_id": article_id,
                "qid": idx,
                "title": title,
                "content": content,
                "question": question,
                "choices": choices,
                "label": label,
            }


def build_rows(dataset_path: str | os.PathLike) -> tuple[list[dict], int]:
    """Convert raw JSON dir to SFT rows. Returns (rows, num_dropped_docs)."""
    docs = load_json_files(dataset_path)
    rows: list[dict] = []
    dropped_docs = 0
    for d in docs:
        content = _get_content(d)
        title = _get_title(d)
        questions = d.get("questions") or []
        if not content or not questions:
            dropped_docs += 1
            continue
        for idx, q in enumerate(questions, start=1):
            question = (q.get("question") or "").strip()
            choices = q.get("choices") or {}
            raw_label = q.get("correct_answer")
            label = raw_label.strip().upper()[:1] if raw_label else None
            if not question or not choices:
                continue
            if label is not None and label not in {"A", "B", "C", "D"}:
                continue
            row = make_row(title=title, content=content, question=question, choices=choices, label=label)
            row["row_id"] = f"{Path(d.get('url', 'x')).stem or 'x'}__{idx}"
            # article_id from the JSON file path is more reliable than url
            rows.append(row)
    return rows, dropped_docs


def write_jsonl(rows: list[dict], path: str | os.PathLike) -> None:
    p = ensure_dir(Path(path).parent) / Path(path).name
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_split(
    rows: list[dict],
    test_size: float = 0.1,
    seed: int = 3407,
) -> tuple[list[dict], list[dict]]:
    """Split rows by 'label' (A/B/C/D) so each split keeps the same label distribution."""
    if not rows:
        return [], []
    labels = [r.get("label", "UNK") for r in rows]
    train, eval_ = train_test_split(
        rows,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    return list(train), list(eval_)


def label_distribution(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(r.get("label", "UNK") for r in rows))
