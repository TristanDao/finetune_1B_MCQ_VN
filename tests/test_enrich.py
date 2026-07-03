"""Smoke tests for the enrich module's prompt builders and JSON extractor.

Does NOT call the real DashScope API; only verifies the prompt structure and
the JSON parsing logic.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
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


# -----------------------------------------------------------------------------
# Choice-reorder balancing (cheap, no API)
# -----------------------------------------------------------------------------

def _reorder_row(label: str, qid: int) -> dict:
    """Build a row with deterministic choices so reorders are easy to verify."""
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
        "row_id": f"art{qid}__1",
    }


def test_reorder_row_for_label_rotates_choices():
    """`A` should rotate to `C` after a +2 shift: {A,B,C,D} -> {C,D,A,B}."""
    from temprun.enrich import _reorder_row_for_label
    row = _reorder_row("A", 1)
    out = _reorder_row_for_label(row, "C")
    # A→C means shift=2: old A goes to C, old B to D, old C to A, old D to B
    assert out["choices"] == {"A": "3", "B": "4", "C": "1", "D": "2"}
    assert out["label"] == "C"
    # messages must be regenerated: assistant letter must match new label
    assert out["messages"][2]["content"] == "C"
    # The original correct text ("1") must now appear at position C in the user content
    assert "C: 1" in out["messages"][1]["content"]


def test_reorder_row_for_label_noop_when_already_correct():
    from temprun.enrich import _reorder_row_for_label
    row = _reorder_row("B", 1)
    out = _reorder_row_for_label(row, "B")
    # Same object identity (no rotation done)
    assert out is row
    assert out["choices"] == {"A": "1", "B": "2", "C": "3", "D": "4"}


def test_reorder_row_for_label_skips_invalid_rows():
    from temprun.enrich import _reorder_row_for_label
    # Missing choices
    assert _reorder_row_for_label({"label": "A", "question": "q"}, "B") == {
        "label": "A", "question": "q",
    }
    # Bad label
    bad = _reorder_row("Z", 1)
    assert _reorder_row_for_label(bad, "A") is bad
    # Incomplete choices
    incomplete = {"label": "A", "choices": {"A": "1", "B": "2"}, "question": "q"}
    assert _reorder_row_for_label(incomplete, "B") is incomplete


def test_balance_via_reorder_distributes_labels_evenly():
    """Highly imbalanced input (all label='A') should map to ~equal per label."""
    from temprun.enrich import balance_via_reorder
    rows = [_reorder_row("A", i) for i in range(40)]
    out = balance_via_reorder(rows, seed=3407)
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for r in out:
        counts[r["label"]] += 1
    assert counts == {"A": 10, "B": 10, "C": 10, "D": 10}


def test_balance_via_reorder_is_deterministic():
    """Same seed → identical (label, row_id) sequence. Different seed →
    different row ordering (even though the label sequence itself is the
    fixed A,B,C,D cycle)."""
    from temprun.enrich import balance_via_reorder
    rows = [_reorder_row("A", i) for i in range(20)] + [_reorder_row("B", i) for i in range(20)]
    out1 = balance_via_reorder(rows, seed=42)
    out2 = balance_via_reorder(rows, seed=42)
    pairs1 = [(r["label"], r["row_id"]) for r in out1]
    pairs2 = [(r["label"], r["row_id"]) for r in out2]
    assert pairs1 == pairs2, "same seed should give identical (label, row_id) order"
    out3 = balance_via_reorder(rows, seed=99)
    pairs3 = [(r["label"], r["row_id"]) for r in out3]
    assert pairs1 != pairs3, "different seed should give a different row ordering"


def test_balance_via_reorder_preserves_correct_text():
    """For every row, the text at the new label position must equal the text
    that was at the original label position in the input row."""
    from temprun.enrich import balance_via_reorder
    rows = [_reorder_row("B", i) for i in range(8)]
    out = balance_via_reorder(rows, seed=3407)
    # Build a map: row_id -> (input_correct_text, new_label)
    by_id = {r["row_id"]: r for r in rows}
    for r in out:
        orig = by_id[r["row_id"]]
        # Original correct text (at orig["label"]) must equal new choices[r["label"]]
        orig_text = orig["choices"][orig["label"]]
        assert r["choices"][r["label"]] == orig_text


def test_balance_via_reorder_keeps_total_row_count():
    from temprun.enrich import balance_via_reorder
    rows = [_reorder_row("A", i) if i % 2 == 0 else _reorder_row("B", i) for i in range(33)]
    out = balance_via_reorder(rows, seed=1)
    assert len(out) == len(rows)


def test_balance_via_reorder_with_4041_imbalanced_rows_yields_ratio_1():
    """Mirrors the real dataset shape (4041 rows, A=1257, B=1638, C=1002, D=144)
    and checks the resulting ratio is essentially 1.0."""
    from temprun.enrich import balance_via_reorder
    rows = (
        [_reorder_row("A", i) for i in range(1257)]
        + [_reorder_row("B", i) for i in range(1638)]
        + [_reorder_row("C", i) for i in range(1002)]
        + [_reorder_row("D", i) for i in range(144)]
    )
    out = balance_via_reorder(rows, seed=3407)
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for r in out:
        counts[r["label"]] += 1
    # ~1010 per label (off by at most 1)
    assert counts == {"A": 1011, "B": 1010, "C": 1010, "D": 1010}
    ratio = max(counts.values()) / min(counts.values())
    assert ratio < 1.01, f"ratio {ratio} should be ~1.0"


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


# -----------------------------------------------------------------------------
# Incremental save (cache + enriched.jsonl flushed periodically)
# -----------------------------------------------------------------------------

def _make_input_rows(n: int, path: Path) -> None:
    """Write a synthetic input JSONL with `n` eligible rows."""
    import json
    rows = []
    for i in range(n):
        rows.append({
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"q{i}?"},
                {"role": "assistant", "content": "A"},
            ],
            "title": f"title-{i}",
            "content": "x" * 500,
            "question": f"q{i}?",
            "choices": {"A": "1", "B": "2", "C": "3", "D": "4"},
            "label": "A",
            "row_id": f"art{i}__1",
        })
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_enrich_dataset_flushes_cache_incrementally(monkeypatch, tmp_path):
    """Cache file must be written periodically during the run, not only at the end."""
    from temprun import enrich as enrich_mod

    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    _make_input_rows(20, inp)

    def fake_sync(rows, cache, plan, effective_balance, **kwargs):
        on_progress = kwargs.get("on_progress")
        # Simulate 20 completions.
        for i in range(1, 21):
            cache[f"key-{i}"] = {"paraphrase": True}
            if on_progress is not None:
                on_progress("paraphrase", i, 20, {"messages": [], "label": "A"})
        return [{"messages": [], "label": "A"}] * 20, {
            "paraphrase": 20, "explain": 0, "synth": 0, "skipped": 0,
            "errors": 0, "paraphrase_attempts": 20, "synth_attempts": 0,
        }, cache

    monkeypatch.setattr(enrich_mod, "_run_passes_sync", fake_sync)

    enrich_dataset(inp, out, concurrency=1, progress_every=5)

    cache_path = out.with_suffix(out.suffix + ".cache.json")
    assert cache_path.exists(), "cache file must exist after the run"
    final_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(final_cache) == 20, f"expected 20 cache entries, got {len(final_cache)}"


def test_enrich_dataset_appends_output_incrementally(monkeypatch, tmp_path):
    """Output file must contain originals + new rows when the run finishes."""
    from temprun import enrich as enrich_mod

    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    _make_input_rows(15, inp)

    def fake_sync(rows, cache, plan, effective_balance, **kwargs):
        on_progress = kwargs.get("on_progress")
        for i in range(1, 11):
            if on_progress is not None:
                on_progress("paraphrase", i, 10, {"messages": [{"role": "system", "content": f"new-{i}"}], "label": "A", "_source": "paraphrase"})
        return (
            [{"messages": [{"role": "system", "content": f"new-{i}"}], "label": "A", "_source": "paraphrase"} for i in range(1, 11)],
            {"paraphrase": 10, "explain": 0, "synth": 0, "skipped": 0,
             "errors": 0, "paraphrase_attempts": 10, "synth_attempts": 0},
            cache,
        )

    monkeypatch.setattr(enrich_mod, "_run_passes_sync", fake_sync)

    enrich_dataset(inp, out, concurrency=1, progress_every=2)

    out_lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines() if line]
    # 15 originals + 10 new = 25 total
    assert len(out_lines) == 25, f"expected 25 lines, got {len(out_lines)}"
    new_rows = [r for r in out_lines if r.get("_source") == "paraphrase"]
    assert len(new_rows) == 10, f"expected 10 new rows, got {len(new_rows)}"


def test_enrich_dataset_preserves_cache_on_exception(monkeypatch, tmp_path):
    """If the pass function raises mid-run, the cache + output must still be flushed via the finally block."""
    from temprun import enrich as enrich_mod

    inp = tmp_path / "in.jsonl"
    out = tmp_path / "out.jsonl"
    _make_input_rows(5, inp)

    def fake_sync(rows, cache, plan, effective_balance, **kwargs):
        on_progress = kwargs.get("on_progress")
        for i in range(1, 4):
            cache[f"k-{i}"] = {"paraphrase": True}
            if on_progress is not None:
                on_progress("paraphrase", i, 10, {"messages": [], "label": "A"})
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(enrich_mod, "_run_passes_sync", fake_sync)

    with pytest.raises(RuntimeError, match="simulated failure"):
        enrich_dataset(inp, out, concurrency=1, progress_every=2)

    cache_path = out.with_suffix(out.suffix + ".cache.json")
    assert cache_path.exists(), "cache must be flushed even when the pass function raises"
    final_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert len(final_cache) == 3, f"expected 3 cache entries from mid-run, got {len(final_cache)}"

    out_lines = [line for line in out.read_text(encoding="utf-8").splitlines() if line]
    assert len(out_lines) == 5 + 3, f"expected 5 originals + 3 new = 8 lines, got {len(out_lines)}"
