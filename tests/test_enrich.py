"""Smoke tests for the enrich module's prompt builders and JSON extractor.

Does NOT call the real DashScope API; only verifies the prompt structure and
the JSON parsing logic.
"""
from __future__ import annotations

from temprun.enrich import (
    _extract_json,
    _user_text_from_parsed,
    explain_prompt,
    paraphrase_prompt,
    synth_q_prompt,
)


def test_paraphrase_prompt_has_system_and_user():
    msgs = paraphrase_prompt("Câu hỏi?", {"A": "1", "B": "2", "C": "3", "D": "4"}, "A")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "Câu hỏi gốc: Câu hỏi?" in msgs[1]["content"]
    assert "Đáp án đúng: A" in msgs[1]["content"]


def test_explain_prompt_includes_content():
    msgs = explain_prompt("t", "Nội dung dài", "q?", {"A": "1"}, "A")
    assert "Nội dung dài" in msgs[1]["content"]
    assert "Đáp án đúng: A" in msgs[1]["content"]


def test_synth_q_prompt_includes_title():
    msgs = synth_q_prompt("Tiêu đề bài", "Nội dung bài")
    assert "Tiêu đề bài" in msgs[1]["content"]
    assert "Nội dung bài" in msgs[1]["content"]


def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_fence():
    text = 'Đây là JSON:\n```json\n{"a": 2}\n```\nHết.'
    assert _extract_json(text) == {"a": 2}


def test_extract_json_embedded():
    text = 'Some text {"a": 3, "b": "x"} more text'
    assert _extract_json(text) == {"a": 3, "b": "x"}


def test_extract_json_invalid():
    assert _extract_json("not json at all") is None
    assert _extract_json("") is None


def test_user_text_from_parsed_uses_prompt_builder():
    original = {"title": "t", "content": "c", "question": "q?", "choices": {"A": "1"}, "label": "A"}
    parsed = {"question": "q paraphrase?", "choices": {"A": "1 paraphrase"}}
    out = _user_text_from_parsed(parsed, original)
    assert "q paraphrase?" in out
    assert "1 paraphrase" in out
