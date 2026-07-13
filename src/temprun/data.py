"""Data pipeline: load raw JSON, build SFT rows, balance labels, stratified split."""

from __future__ import annotations

import glob
import json
import random
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

from .prompts import Mode, build_messages, make_row
from .utils import ensure_dir


def _get_content(d: dict) -> str:
    return (d.get("content:") or d.get("content") or "").strip()


def _get_title(d: dict) -> str:
    return (d.get("title:") or d.get("title") or "").strip()


def load_json_files(dataset_path: str) -> list[dict]:
    """Load all *.json files in a directory into a list of dicts."""
    files = sorted(glob.glob(str(Path(dataset_path) / "*.json")))
    items: list[dict] = []
    for p in files:
        try:
            with open(p, encoding="utf-8") as f:
                items.append(json.load(f))
        except Exception as e:
            print(f"[warn] Failed to read {p}: {e}")
    return items


def build_rows(dataset_path: str, *, mode: Mode = "conversation") -> tuple[list[dict], int]:
    """Convert raw JSON directory to SFT rows.

    Returns (rows, num_dropped_docs). Each row has messages, label, title,
    content, question, choices, explanation (CoT mode).
    """
    files = sorted(glob.glob(str(Path(dataset_path) / "*.json")))
    rows: list[dict] = []
    dropped = 0

    for path in files:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        content = _get_content(d)
        title = _get_title(d)
        questions = d.get("questions") or []

        if not content or not questions:
            dropped += 1
            continue

        for q in questions:
            question = (q.get("question") or "").strip()
            choices = q.get("choices") or {}
            raw_label = q.get("correct_answer")
            label = raw_label.strip().upper()[:1] if raw_label else None

            if not question or not choices:
                continue
            if label is None or label not in {"A", "B", "C", "D"}:
                continue

            explanation = (q.get("explanation") or "").strip()
            row = make_row(
                title=title, content=content, question=question,
                choices=choices, label=label, explanation=explanation,
                mode=mode,
            )
            rows.append(row)

    return rows, dropped


def write_jsonl(rows: list[dict], path: str) -> None:
    """Write list of dicts as JSONL file."""
    p = ensure_dir(Path(path).parent) / Path(path).name
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    """Read JSONL file into list of dicts."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def stratified_split(
    rows: list[dict],
    test_size: float = 0.1,
    seed: int = 3407,
) -> tuple[list[dict], list[dict]]:
    """Stratified split by label, preserving distribution."""
    if not rows:
        return [], []
    labels = [r.get("label", "UNK") for r in rows]
    train, eval_ = train_test_split(
        rows, test_size=test_size, random_state=seed, shuffle=True, stratify=labels,
    )
    return list(train), list(eval_)


def label_distribution(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(r.get("label", "UNK") for r in rows))


# ---------------------------------------------------------------------------
# Label balancing: rotate choices so correct answer lands A→B→C→D cyclically.
# ---------------------------------------------------------------------------

def _reorder_row_for_label(row: dict, target_label: str, *, mode: Mode = "conversation") -> dict:
    """Rotate choices so the correct answer lands on `target_label`.

    Regenerates messages to reflect the new choice order and label.
    """
    orig_label = row.get("label", "")
    if orig_label not in {"A", "B", "C", "D"} or target_label not in {"A", "B", "C", "D"}:
        return row

    choices = row.get("choices") or {}
    if not all(k in choices for k in ("A", "B", "C", "D")):
        return row

    shift = (ord(target_label) - ord(orig_label)) % 4
    if shift == 0:
        return row

    new_choices: dict[str, str] = {}
    for k, v in choices.items():
        if k in {"A", "B", "C", "D"}:
            new_choices[chr(ord("A") + (ord(k) - ord("A") + shift) % 4)] = v
        else:
            new_choices[k] = v

    explanation = row.get("explanation", "")
    new_messages = build_messages(
        title=row.get("title", ""),
        content=row.get("content", ""),
        question=row.get("question", ""),
        choices=new_choices,
        label=target_label,
        explanation=explanation if explanation else None,
        mode=mode,
    )

    new_row = dict(row)
    new_row["messages"] = new_messages
    new_row["choices"] = new_choices
    new_row["label"] = target_label
    return new_row


def balance_via_reorder(rows: list[dict], *, seed: int = 3407, mode: Mode = "conversation") -> list[dict]:
    """Rebalance labels by shuffling and rotating each row's answer to cycle A→B→C→D.

    Result: ~ceil(N/4) per label (off by at most 1), ratio ≈ 1.00.
    """
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    target_labels = ["A", "B", "C", "D"]
    return [
        _reorder_row_for_label(rows[i], target_labels[new_idx % 4], mode=mode)
        for new_idx, i in enumerate(indices)
    ]
