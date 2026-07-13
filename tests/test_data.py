"""Tests for data pipeline: build_rows, balance, stratified split."""

from __future__ import annotations

from temprun.data import (
    _get_content,
    _get_title,
    balance_via_reorder,
    label_distribution,
    stratified_split,
)


def test_get_content_handles_both_keys():
    assert _get_content({"content": "x"}) == "x"
    assert _get_content({"content:": "y"}) == "y"
    assert _get_content({"content": ""}) == ""
    assert _get_content({}) == ""


def test_get_title_handles_both_keys():
    assert _get_title({"title": "t"}) == "t"
    assert _get_title({"title:": "t2"}) == "t2"
    assert _get_title({}) == ""


def test_stratified_split_preserves_distribution():
    rows = [{"label": c} for c in (["A"] * 50 + ["B"] * 30 + ["C"] * 15 + ["D"] * 5)]
    train, evald = stratified_split(rows, test_size=0.2, seed=42)
    assert len(train) + len(evald) == len(rows)
    train_dist = label_distribution(train)
    evald_dist = label_distribution(evald)
    for k in ["A", "B", "C", "D"]:
        assert k in train_dist and k in evald_dist


def _reorder_row(label: str, qid: int) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"q{qid}?"},
            {"role": "assistant", "content": label},
        ],
        "title": f"t-{qid}",
        "content": "body",
        "question": f"q{qid}?",
        "choices": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "label": label,
    }


def test_reorder_row_rotates_choices():
    from temprun.data import _reorder_row_for_label

    row = _reorder_row("A", 1)
    out = _reorder_row_for_label(row, "C", mode="conversation")
    assert out["choices"] == {"A": "3", "B": "4", "C": "1", "D": "2"}
    assert out["label"] == "C"
    assert out["messages"][2]["content"] == "C"
    assert "C: 1" in out["messages"][1]["content"]


def test_reorder_row_noop_when_already_correct():
    from temprun.data import _reorder_row_for_label

    row = _reorder_row("B", 1)
    out = _reorder_row_for_label(row, "B", mode="conversation")
    assert out is row


def test_balance_via_reorder_distributes_evenly():
    rows = [_reorder_row("A", i) for i in range(40)]
    out = balance_via_reorder(rows, seed=3407)
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for r in out:
        counts[r["label"]] += 1
    assert counts == {"A": 10, "B": 10, "C": 10, "D": 10}


def test_balance_via_reorder_is_deterministic():
    rows = [_reorder_row("A", i) for i in range(20)] + [_reorder_row("B", i) for i in range(20)]
    out1 = balance_via_reorder(rows, seed=42)
    out2 = balance_via_reorder(rows, seed=42)
    pairs1 = [(r["label"], r.get("row_id", r["question"])) for r in out1]
    pairs2 = [(r["label"], r.get("row_id", r["question"])) for r in out2]
    assert pairs1 == pairs2
    out3 = balance_via_reorder(rows, seed=99)
    pairs3 = [(r["label"], r.get("row_id", r["question"])) for r in out3]
    assert pairs1 != pairs3


def test_balance_via_reorder_preserves_correct_text():
    rows = [_reorder_row("B", i) for i in range(8)]
    out = balance_via_reorder(rows, seed=3407)
    by_question = {r["question"]: r for r in rows}
    for r in out:
        orig = by_question[r["question"]]
        orig_text = orig["choices"][orig["label"]]
        assert r["choices"][r["label"]] == orig_text
