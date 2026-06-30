"""Tests for data loading: requires the kaggle data zip at repo root
(tempo-run-2025-run-with-ai-break-limits.zip) OR a pre-extracted tree.

Skipped automatically if neither is present.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from temprun.data import _get_content, _get_title, build_rows, label_distribution, stratified_split
from temprun.utils import repo_root


@pytest.fixture(scope="module")
def raw_train_dir(tmp_path_factory) -> Path:
    """Extract a small slice of train/ from the kaggle zip to a temp dir."""
    zip_path = repo_root() / "tempo-run-2025-run-with-ai-break-limits.zip"
    if not zip_path.exists():
        pytest.skip("kaggle zip not present at repo root")
    out = Path(tmp_path_factory.mktemp("raw_train"))
    with zipfile.ZipFile(zip_path) as z:
        members = [n for n in z.namelist() if n.startswith("Dataset/Dataset/train/") and n.endswith(".json")]
        # Sample first 30 files to keep test fast
        for m in members[:30]:
            z.extract(m, out)
    train_dir = out / "Dataset" / "Dataset" / "train"
    if not train_dir.exists() or not any(train_dir.iterdir()):
        pytest.skip("could not extract train files")
    return train_dir


def test_get_content_handles_both_keys():
    assert _get_content({"content": "x"}) == "x"
    assert _get_content({"content:": "y"}) == "y"
    assert _get_content({"content": ""}) == ""
    assert _get_content({}) == ""


def test_get_title_handles_both_keys():
    assert _get_title({"title": "t"}) == "t"
    assert _get_title({"title:": "t2"}) == "t2"
    assert _get_title({}) == ""


def test_build_rows_smoke(raw_train_dir):
    rows, dropped = build_rows(raw_train_dir)
    assert len(rows) > 0
    for r in rows:
        assert r["label"] in {"A", "B", "C", "D"}
        assert len(r["messages"]) == 3
    # Dist is roughly balanced
    dist = label_distribution(rows)
    assert all(k in dist for k in ["A", "B", "C", "D"])


def test_stratified_split_preserves_distribution():
    rows = [{"label": c} for c in (["A"] * 50 + ["B"] * 30 + ["C"] * 15 + ["D"] * 5)]
    train, evald = stratified_split(rows, test_size=0.2, seed=42)
    assert len(train) + len(evald) == len(rows)
    train_dist = label_distribution(train)
    evald_dist = label_distribution(evald)
    for k in ["A", "B", "C", "D"]:
        assert k in train_dist and k in evald_dist
