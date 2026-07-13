"""Smoke tests for prompts + generation parsing."""

from __future__ import annotations

import pytest

from temprun.prompts import (
    SYSTEM_PROMPT_CONVERSATION,
    SYSTEM_PROMPT_COT,
    build_assistant,
    build_messages,
    build_user_instruction,
    format_choices,
    make_row,
)
from temprun.utils import parse_generated


def test_format_choices_deterministic():
    choices = {"A": "x", "B": "y", "C": "z", "D": "t"}
    out = format_choices(choices)
    assert out == "A: x\nB: y\nC: z\nD: t"


def test_format_choices_skips_missing():
    choices = {"B": "y", "D": "t"}
    out = format_choices(choices)
    assert out == "B: y\nD: t"


def test_build_user_instruction_conversation():
    out = build_user_instruction("Tieu de", "Noi dung", "Cau hoi?", {"A": "1", "B": "2", "C": "3", "D": "4"}, mode="conversation")
    assert "Tiêu đề: Tieu de" in out
    assert "Nội dung: Noi dung" in out
    assert "Câu hỏi:" in out and "Cau hoi?" in out
    assert "A: 1" in out and "D: 4" in out
    assert "A, B, C hoặc D" in out


def test_build_user_instruction_cot():
    out = build_user_instruction("Tieu de", "Noi dung", "Cau hoi?", {"A": "1", "B": "2", "C": "3", "D": "4"}, mode="cot")
    assert "### VĂN BẢN" in out
    assert "**Tiêu đề:** Tieu de" in out
    assert "### CÂU HỎI" in out
    assert "### LỰA CHỌN" in out
    assert "A: 1" in out


def test_build_assistant_conversation():
    assert build_assistant("A", mode="conversation") == "A"
    assert build_assistant("B", mode="conversation") == "B"


def test_build_assistant_cot():
    out = build_assistant("A", explanation="Vi vay A dung.", mode="cot")
    assert "Vi vay A dung." in out
    assert "Do đó, đáp án đúng là A." in out

    out2 = build_assistant("B", mode="cot")
    assert out2 == "Do đó, đáp án đúng là B."


def test_build_messages_conversation():
    msgs = build_messages(
        "t", "c", "q", {"A": "1", "B": "2", "C": "3", "D": "4"}, "A",
        mode="conversation",
    )
    assert msgs[0]["content"] == SYSTEM_PROMPT_CONVERSATION
    assert msgs[2]["content"] == "A"


def test_build_messages_cot():
    msgs = build_messages(
        "t", "c", "q", {"A": "1", "B": "2", "C": "3", "D": "4"}, "A",
        explanation="Vi vay A dung.", mode="cot",
    )
    assert msgs[0]["content"] == SYSTEM_PROMPT_COT
    assert "Vi vay A dung." in msgs[2]["content"]
    assert "Do đó, đáp án đúng là A." in msgs[2]["content"]


def test_make_row_schema():
    row = make_row(
        title="t", content="c", question="q",
        choices={"A": "1", "B": "2", "C": "3", "D": "4"},
        label="A", mode="conversation",
    )
    assert row["label"] == "A"
    assert len(row["messages"]) == 3
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][1]["role"] == "user"
    assert row["messages"][2]["role"] == "assistant"
    assert row["messages"][2]["content"] == "A"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A", "A"),
        ("B.", "B"),
        ("Dap an: C", "C"),
        ("Cau tra loi: D", "D"),
        ("Tra loi: A", "A"),
        ("Toi chon dap an A.", "A"),
        ("", None),
        ("XYZ", None),
        ("E", None),
    ],
)
def test_parse_generated(text, expected):
    assert parse_generated(text) == expected
