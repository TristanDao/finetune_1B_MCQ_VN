"""Smoke tests for the enrich module's prompt builders and JSON extractor.

Does NOT call the real DashScope API; only verifies the prompt structure and
the JSON parsing logic.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from temprun.enrich import (
    _balance_plan,
    _call_chat_async,
    _explain_once_async,
    _extract_json,
    _paraphrase_key,
    _paraphrase_once_async,
    _synth_candidates,
    _synth_once_async,
    _user_text_from_parsed,
    enrich_dataset,
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


# -----------------------------------------------------------------------------
# Balancing plan
# -----------------------------------------------------------------------------

def _rows(spec: dict[str, int]) -> list[dict]:
    """Build a row list with the given per-label counts."""
    out: list[dict] = []
    for lbl, n in spec.items():
        out.extend({"label": lbl} for _ in range(n))
    return out


def test_balance_plan_balanced_input():
    plan = _balance_plan(_rows({"A": 100, "B": 100, "C": 100, "D": 100}))
    assert plan["is_imbalanced"] is False
    assert plan["ratio"] == 1.0
    assert plan["target"] == 100
    assert plan["repeat_per_row"] == {"A": 0, "B": 0, "C": 0, "D": 0}
    assert plan["synth_needed"] == {"A": 0, "B": 0, "C": 0, "D": 0}


def test_balance_plan_auto_target_is_max():
    # Mimics the current dataset shape: A=1257, B=1638, C=1002, D=144
    plan = _balance_plan(_rows({"A": 1257, "B": 1638, "C": 1002, "D": 144}))
    assert plan["is_imbalanced"] is True
    assert plan["target"] == 1638
    # ratio = 1638 / 144 ≈ 11.375
    assert plan["ratio"] == 1638 / 144
    # D needs ceil((1638-144)/144) = 11 paraphrases
    assert plan["repeat_per_row"]["D"] == 11
    # C: ceil((1638-1002)/1002) = 1
    assert plan["repeat_per_row"]["C"] == 1
    # A: ceil((1638-1257)/1257) = 1
    assert plan["repeat_per_row"]["A"] == 1
    # B is already at max → no repeat
    assert plan["repeat_per_row"]["B"] == 0
    # After paraphrase, every class is at or above target
    assert plan["synth_needed"] == {"A": 0, "B": 0, "C": 0, "D": 0}


def test_balance_plan_explicit_target():
    plan = _balance_plan(_rows({"A": 10, "B": 5, "C": 2, "D": 1}), target_per_class=20)
    assert plan["target"] == 20
    # D: ceil((20-1)/1) = 19
    assert plan["repeat_per_row"]["D"] == 19
    # C: ceil((20-2)/2) = 9
    assert plan["repeat_per_row"]["C"] == 9
    # A: ceil((20-10)/10) = 1
    assert plan["repeat_per_row"]["A"] == 1
    # B: ceil((20-5)/5) = 3
    assert plan["repeat_per_row"]["B"] == 3


def test_balance_plan_min_ratio_threshold():
    # Ratio = 3.0, with min_ratio=2.0 this is imbalanced.
    plan = _balance_plan(_rows({"A": 30, "B": 10, "C": 10, "D": 10}), min_ratio=2.0)
    assert plan["is_imbalanced"] is True
    # With min_ratio=5.0 it is no longer imbalanced.
    plan2 = _balance_plan(_rows({"A": 30, "B": 10, "C": 10, "D": 10}), min_ratio=5.0)
    assert plan2["is_imbalanced"] is False


def test_balance_plan_empty_input():
    plan = _balance_plan([])
    assert plan["current"] == {"A": 0, "B": 0, "C": 0, "D": 0}
    assert plan["target"] == 0
    assert plan["ratio"] == 0.0
    assert plan["is_imbalanced"] is False
    assert plan["repeat_per_row"] == {"A": 0, "B": 0, "C": 0, "D": 0}


def test_paraphrase_key_n_zero_uses_bare_key():
    """n=0 must always use the legacy key so existing caches and the explain
    cache stay valid."""
    r = {"question": "Q?", "label": "A"}
    assert _paraphrase_key(r, 0, balanced=False) == _paraphrase_key(r, 0, balanced=True)
    # n>0 with balanced=True adds a suffix
    assert _paraphrase_key(r, 0, balanced=True) != _paraphrase_key(r, 1, balanced=True)
    # n>0 with balanced=False still uses the bare key (backward compat)
    assert _paraphrase_key(r, 0, balanced=False) == _paraphrase_key(r, 2, balanced=False)


# -----------------------------------------------------------------------------
# Async helpers + concurrency control
# -----------------------------------------------------------------------------

VALID_PARA_JSON = json.dumps(
    {
        "question": "Paraphrased Q?",
        "choices": {"A": "1p", "B": "2p", "C": "3p", "D": "4p"},
        "correct_answer": "B",
    },
    ensure_ascii=False,
)

VALID_SYNTH_JSON = json.dumps(
    {
        "question": "Synth Q?",
        "choices": {"A": "1s", "B": "2s", "C": "3s", "D": "4s"},
        "correct_answer": "C",
    },
    ensure_ascii=False,
)


def _fake_response(content: str) -> MagicMock:
    """Build a fake OpenAI chat completion response with the given content."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _row(label: str = "A", question: str = "Q?", title: str = "T", content: str = "C " * 200) -> dict:
    return {
        "title": title,
        "content": content,
        "question": question,
        "choices": {"A": "1", "B": "2", "C": "3", "D": "4"},
        "label": label,
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"user: {question}"},
            {"role": "assistant", "content": label},
        ],
    }


def test_synth_candidates_dedup_by_title_and_filters_short_content():
    rows = [
        {"title": "T1", "content": "x" * 400, "label": "A"},
        {"title": "T1", "content": "x" * 400, "label": "B"},  # duplicate title
        {"title": "T2", "content": "x" * 100, "label": "C"},  # too short
        {"title": "", "content": "x" * 400, "label": "D"},    # empty title
        {"title": "T3", "content": "x" * 400, "label": "A"},
    ]
    out = _synth_candidates(rows)
    assert len(out) == 2
    assert {r["title"] for r in out} == {"T1", "T3"}


def test_call_chat_async_retries_then_succeeds():
    """Two failures then a success: should return content, not raise."""
    client = MagicMock()
    completions = MagicMock()
    completions.create = AsyncMock(
        side_effect=[RuntimeError("boom"), RuntimeError("boom"), _fake_response("hello")]
    )
    client.chat.completions = completions

    out = asyncio.run(_call_chat_async(client, [{"role": "user", "content": "x"}], retries=3, backoff=1.0))
    assert out == "hello"
    assert completions.create.await_count == 3


def test_call_chat_async_raises_after_max_retries():
    client = MagicMock()
    completions = MagicMock()
    completions.create = AsyncMock(side_effect=RuntimeError("nope"))
    client.chat.completions = completions

    with pytest.raises(RuntimeError, match="async call failed"):
        asyncio.run(_call_chat_async(client, [{"role": "user", "content": "x"}], retries=2, backoff=1.0))
    assert completions.create.await_count == 2


def test_paraphrase_once_async_returns_row_on_valid_json():
    client = MagicMock()
    completions = MagicMock()
    completions.create = AsyncMock(return_value=_fake_response(VALID_PARA_JSON))
    client.chat.completions = completions

    r = _row(label="B")
    out = asyncio.run(_paraphrase_once_async(client, "m", r))
    assert out is not None
    assert out["label"] == "B"
    assert out["_source"] == "paraphrase"
    assert "Paraphrased Q?" in out["messages"][1]["content"]


def test_paraphrase_once_async_returns_none_on_invalid_label():
    """If the LLM says correct_answer='Z', we drop the row (label not in A-D)."""
    bad = json.dumps(
        {"question": "Q?", "choices": {"A": "1", "B": "2", "C": "3", "D": "4"}, "correct_answer": "Z"}
    )
    client = MagicMock()
    completions = MagicMock()
    completions.create = AsyncMock(return_value=_fake_response(bad))
    client.chat.completions = completions

    assert asyncio.run(_paraphrase_once_async(client, "m", _row())) is None


def test_synth_once_async_returns_row():
    client = MagicMock()
    completions = MagicMock()
    completions.create = AsyncMock(return_value=_fake_response(VALID_SYNTH_JSON))
    client.chat.completions = completions

    out = asyncio.run(_synth_once_async(client, "m", _row()))
    assert out is not None
    assert out["label"] == "C"
    assert out["_source"] == "synth"


def test_explain_once_async_returns_text():
    expl = json.dumps({"explanation": "vì sao đúng"}, ensure_ascii=False)
    client = MagicMock()
    completions = MagicMock()
    completions.create = AsyncMock(return_value=_fake_response(expl))
    client.chat.completions = completions

    out = asyncio.run(_explain_once_async(client, "m", _row()))
    assert out == "vì sao đúng"


# -----------------------------------------------------------------------------
# enrich_dataset dispatcher behaviour
# -----------------------------------------------------------------------------

def test_enrich_dataset_concurrency_1_uses_sync_path(monkeypatch, tmp_path):
    """When concurrency=1 we must call the sync implementation, not asyncio.run."""
    from temprun import enrich as enrich_mod

    called = {"sync": 0, "async": 0}

    def fake_sync(rows, cache, plan, effective_balance, **kwargs):
        called["sync"] += 1
        return [], {"paraphrase": 0, "explain": 0, "synth": 0, "skipped": 0,
                    "errors": 0, "paraphrase_attempts": 0, "synth_attempts": 0}, cache

    def fake_async(*args, **kwargs):
        called["async"] += 1
        return [], {}, {}

    monkeypatch.setattr(enrich_mod, "_run_passes_sync", fake_sync)
    monkeypatch.setattr(enrich_mod, "_run_passes_async", fake_async)

    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text("", encoding="utf-8")
    # Empty input → plan will have all zeros, loop won't hit API.
    enrich_dataset(inp, out, concurrency=1)
    assert called["sync"] == 1
    assert called["async"] == 0


def test_enrich_dataset_async_failure_falls_back_to_sync(monkeypatch, tmp_path, capsys):
    """If `_run_passes_async` raises, we log a warning and call the sync path."""
    from temprun import enrich as enrich_mod

    called = {"sync": 0, "async": 0}

    def boom_async(*args, **kwargs):
        called["async"] += 1
        raise RuntimeError("event loop broken")

    def fake_sync(rows, cache, plan, effective_balance, **kwargs):
        called["sync"] += 1
        return [], {"paraphrase": 0, "explain": 0, "synth": 0, "skipped": 0,
                    "errors": 0, "paraphrase_attempts": 0, "synth_attempts": 0}, cache

    monkeypatch.setattr(enrich_mod, "_run_passes_async", boom_async)
    monkeypatch.setattr(enrich_mod, "_run_passes_sync", fake_sync)

    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    inp.write_text("", encoding="utf-8")
    enrich_dataset(inp, out, concurrency=10)
    assert called["async"] == 1
    assert called["sync"] == 1
    captured = capsys.readouterr()
    assert "falling back to sync" in captured.out
