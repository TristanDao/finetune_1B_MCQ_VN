"""Smoke tests for `temprun.hf_utils.push_enriched_to_hf`.

No real network calls — we monkeypatch `huggingface_hub.HfApi` and
`huggingface_hub.create_repo` and assert the right repo, prefix, and
`path_in_repo` values are used.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from temprun.hf_utils import PROCESSED_PREFIX, push_enriched_to_hf


def _setup_env(monkeypatch, *, repo: str = "ThinhDao/TempoRun2025_UIT",
               token: str = "hf_test_token"):
    monkeypatch.setenv("HF_DATASET_REPO", repo)
    monkeypatch.setenv("HF_TOKEN", token)


def test_push_enriched_uploads_both_files(monkeypatch, tmp_path, capsys):
    _setup_env(monkeypatch)
    enriched = tmp_path / "enriched.jsonl"
    enriched.write_text('{"messages":[],"label":"A"}\n', encoding="utf-8")
    cache = tmp_path / "enriched.jsonl.cache.json"
    cache.write_text(json.dumps({"k": "v"}), encoding="utf-8")

    api = MagicMock()
    create_repo = MagicMock()
    monkeypatch.setattr("temprun.hf_utils.HfApi", lambda token: api)
    monkeypatch.setattr("temprun.hf_utils.create_repo", create_repo)

    uploaded = push_enriched_to_hf(enriched)

    assert uploaded == [
        f"{PROCESSED_PREFIX}enriched.jsonl",
        f"{PROCESSED_PREFIX}enriched.jsonl.cache.json",
    ]
    # Repo creation: private=True, exist_ok=True, dataset type
    create_repo.assert_called_once()
    # `repo_id` is passed positionally; check args[0] for it.
    assert create_repo.call_args.args[0] == "ThinhDao/TempoRun2025_UIT"
    assert create_repo.call_args.kwargs == {
        "repo_type": "dataset",
        "private": True,
        "exist_ok": True,
        "token": "hf_test_token",
    }

    # upload_file called twice with the right in-repo paths
    calls = api.upload_file.call_args_list
    assert [c.kwargs["path_in_repo"] for c in calls] == uploaded
    for c in calls:
        assert c.kwargs["repo_id"] == "ThinhDao/TempoRun2025_UIT"
        assert c.kwargs["repo_type"] == "dataset"
        assert c.kwargs["token"] == "hf_test_token"

    # Both files are in the captured output
    out = capsys.readouterr().out
    assert "no cache file" not in out
    assert "processed/enriched.jsonl" in out
    assert "processed/enriched.jsonl.cache.json" in out


def test_push_enriched_skips_missing_cache(monkeypatch, tmp_path, capsys):
    _setup_env(monkeypatch)
    enriched = tmp_path / "enriched.jsonl"
    enriched.write_text('{"messages":[],"label":"A"}\n', encoding="utf-8")
    # No cache file
    api = MagicMock()
    create_repo = MagicMock()
    monkeypatch.setattr("temprun.hf_utils.HfApi", lambda token: api)
    monkeypatch.setattr("temprun.hf_utils.create_repo", create_repo)

    uploaded = push_enriched_to_hf(enriched)
    assert uploaded == [f"{PROCESSED_PREFIX}enriched.jsonl"]
    assert api.upload_file.call_count == 1
    out = capsys.readouterr().out
    assert "no cache file" in out


def test_push_enriched_missing_repo(monkeypatch, tmp_path):
    # Patch load_env first so .env re-loading does not restore HF_DATASET_REPO
    monkeypatch.setattr("temprun.hf_utils.load_env", lambda: None)
    monkeypatch.delenv("HF_DATASET_REPO", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    enriched = tmp_path / "enriched.jsonl"
    enriched.write_text('{"messages":[]}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="HF_DATASET_REPO"):
        push_enriched_to_hf(enriched)


def test_push_enriched_missing_token(monkeypatch, tmp_path):
    # Patch load_env first so .env re-loading does not restore HF_TOKEN
    monkeypatch.setattr("temprun.hf_utils.load_env", lambda: None)
    monkeypatch.setenv("HF_DATASET_REPO", "ThinhDao/TempoRun2025_UIT")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    enriched = tmp_path / "enriched.jsonl"
    enriched.write_text('{"messages":[]}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        push_enriched_to_hf(enriched)


def test_push_enriched_missing_local_file(monkeypatch, tmp_path):
    _setup_env(monkeypatch)
    missing = tmp_path / "does_not_exist.jsonl"

    with pytest.raises(FileNotFoundError):
        push_enriched_to_hf(missing)


def test_push_enriched_does_not_recreate_existing_repo(monkeypatch, tmp_path):
    """Verifies `create_repo(exist_ok=True)` is used (so an existing repo
    is left alone — visibility, files, etc.)."""
    _setup_env(monkeypatch)
    enriched = tmp_path / "enriched.jsonl"
    enriched.write_text('{"messages":[]}', encoding="utf-8")

    create_repo = MagicMock()
    monkeypatch.setattr("temprun.hf_utils.HfApi", lambda token: MagicMock())
    monkeypatch.setattr("temprun.hf_utils.create_repo", create_repo)

    push_enriched_to_hf(enriched)
    assert create_repo.call_args.kwargs["exist_ok"] is True
