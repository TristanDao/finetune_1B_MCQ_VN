"""Helpers to push processed SFT data to the same HF dataset repo as the raw zip.

Used by `--push-after` in `scripts/enrich_data.py` to back up `enriched.jsonl`
and its API-response cache, so a Colab session reset does not cost another
round of DashScope calls. Pushed files land under
`${HF_DATASET_REPO}/processed/...` to keep them clearly separate from the
raw zip at the repo root.
"""
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

from .utils import load_env

PROCESSED_PREFIX = "processed/"


def push_enriched_to_hf(enriched_jsonl: str | Path) -> list[str]:
    """Upload `enriched_jsonl` and (if present) its `.cache.json` to
    `${HF_DATASET_REPO}/processed/...`. The repo is created as PRIVATE if
    it does not yet exist; an existing repo is left untouched (visibility
    is not changed).

    Args:
        enriched_jsonl: local path to the enriched JSONL (e.g.
            `data/processed/enriched.jsonl`). Its sibling
            `enriched.jsonl.cache.json` is uploaded too if it exists.

    Returns:
        List of `path_in_repo` strings that were uploaded, in upload order.
        Useful for logging and for tests.

    Raises:
        RuntimeError: if `HF_DATASET_REPO` or `HF_TOKEN` is missing.
        FileNotFoundError: if `enriched_jsonl` does not exist locally.
    """
    load_env()
    repo_id = os.environ.get("HF_DATASET_REPO", "").strip()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not repo_id:
        raise RuntimeError("HF_DATASET_REPO missing. Fill it in .env first.")
    if not token:
        raise RuntimeError("HF_TOKEN missing. Fill it in .env first.")

    enriched_path = Path(enriched_jsonl)
    cache_path = enriched_path.with_suffix(enriched_path.suffix + ".cache.json")
    if not enriched_path.is_file():
        raise FileNotFoundError(f"enriched file not found: {enriched_path}")

    api = HfApi(token=token)
    print(f"[push] create_repo({repo_id!r}, private=True, exist_ok=True) ...")
    create_repo(
        repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=token,
    )

    files: list[tuple[Path, str]] = [
        (enriched_path, f"{PROCESSED_PREFIX}{enriched_path.name}")
    ]
    if cache_path.is_file():
        files.append((cache_path, f"{PROCESSED_PREFIX}{cache_path.name}"))
    else:
        print(f"[push] no cache file at {cache_path}; skipping")

    for local, in_repo in files:
        size_mb = local.stat().st_size / 1e6
        print(f"[push] -> {repo_id}/{in_repo}  ({size_mb:.1f} MB)")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
    return [in_repo for _, in_repo in files]
