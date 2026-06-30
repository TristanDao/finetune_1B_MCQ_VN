"""Smoke tests for prompts + generation parsing (no model/GPU required)."""
from __future__ import annotations

import pytest

from temprun.prompts import build_chat_messages, build_user_instruction, format_choices, make_row
from temprun.utils import parse_generated


def test_format_choices_deterministic():
    choices = {"A": "x", "B": "y", "C": "z", "D": "t"}
    out = format_choices(choices)
    assert out == "A: x\nB: y\nC: z\nD: t"


def test_format_choices_skips_missing():
    choices = {"B": "y", "D": "t"}
    out = format_choices(choices)
    assert out == "B: y\nD: t"


def test_build_user_instruction_contains_required_sections():
    out = build_user_instruction("Tiêu đề", "Nội dung", "Câu hỏi?", {"A": "1", "B": "2", "C": "3", "D": "4"})
    assert "Tiêu đề: Tiêu đề" in out
    assert "Nội dung: Nội dung" in out
    # Section header is "Câu hỏi:" on its own line, followed by the question
    assert "Câu hỏi:" in out and "Câu hỏi?" in out
    assert "A: 1" in out and "D: 4" in out
    assert "A, B, C hoặc D" in out


def test_make_row_schema():
    row = make_row(
        title="t", content="c", question="q", choices={"A": "1", "B": "2", "C": "3", "D": "4"}, label="A"
    )
    assert row["label"] == "A"
    assert len(row["messages"]) == 3
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][1]["role"] == "user"
    assert row["messages"][2]["role"] == "assistant"
    assert row["messages"][2]["content"] == "A"


def test_build_chat_messages_optional_assistant():
    msgs = build_chat_messages("sys", "user")
    assert len(msgs) == 2
    msgs2 = build_chat_messages("sys", "user", "A")
    assert len(msgs2) == 3
    assert msgs2[2]["content"] == "A"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A", "A"),
        ("B.", "B"),
        ("Đáp án: C", "C"),
        ("Câu trả lời: D", "D"),
        ("Tra loi: A", "A"),
        ("\n\n**B**\n", "B"),
        ("Tôi chọn đáp án A.", "A"),
        ("", None),
        ("XYZ", None),
        ("E", None),
    ],
)
def test_parse_generated(text, expected):
    assert parse_generated(text) == expected
