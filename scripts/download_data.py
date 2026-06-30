"""Download dataset từ Hugging Face private dataset repo, sau đó unzip về data/raw/.

Repo HF phải là PRIVATE và chứa 1 file zip (`HF_DATASET_FILE`, mặc định
`tempo-run-2025-run-with-ai-break-limits.zip`) với cấu trúc bên trong:

    <zip>/Dataset/Dataset/<split>/*.json    (split = train, test, public, private)
    <zip>/sample_submission.csv             (optional, copied to out_dir root)

Fallback: nếu HF download fail, thử dùng zip ở repo root
(`tempo-run-2025-run-with-ai-break-limits.zip`) nếu có.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

from temprun.utils import ensure_dir, load_env, repo_root

SPLITS = ("train", "test", "public", "private")
DEFAULT_HF_FILE = "tempo-run-2025-run-with-ai-break-limits.zip"


def _detect_splits(root: Path) -> dict[str, Path]:
    """Return {split_name: src_dir} for every split found under `root`."""
    found: dict[str, Path] = {}
    for split in SPLITS:
        candidate = root / "Dataset" / "Dataset" / split
        if candidate.is_dir():
            found[split] = candidate
            continue
        candidate = root / split
        if candidate.is_dir():
            found[split] = candidate
    return found


def _flatten_split(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _extract_zip_to_flat(z: Path, out_dir: Path) -> dict[str, Path]:
    """Extract zip vào scratch, flatten Dataset/Dataset/<split> → out_dir/<split>.

    Preserves top-level helper files (.csv/.md/.txt) into out_dir root.
    """
    scratch = out_dir.parent / f".{out_dir.name}.scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    print(f"[download] unzip {z.name} -> {scratch}")
    with zipfile.ZipFile(z) as zf:
        zf.extractall(scratch)
    splits = _detect_splits(scratch)
    extracted: dict[str, Path] = {}
    for split, src in splits.items():
        dst = out_dir / split
        _flatten_split(src, dst)
        extracted[split] = dst
    for helper in scratch.glob("*"):
        if helper.is_file() and helper.suffix in {".csv", ".md", ".txt"}:
            shutil.copy2(helper, out_dir / helper.name)
            print(f"[download] preserved helper file: {out_dir / helper.name}")
    shutil.rmtree(scratch, ignore_errors=True)
    return extracted


def download_from_hf(repo_id: str, filename: str, out_dir: Path, *, token: str) -> dict[str, Path]:
    """Download a single file from a private HF dataset repo, then flatten."""
    from huggingface_hub import hf_hub_download

    if not repo_id:
        raise RuntimeError("HF_DATASET_REPO is empty. Set it in .env")
    if not token:
        raise RuntimeError("HF_TOKEN is empty. Set it in .env")

    cache_dir = out_dir.parent / ".hf_cache"
    ensure_dir(cache_dir)
    print(f"[download] hf_hub_download({repo_id!r}, {filename!r}, repo_type='dataset') ...")
    cached = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
            local_dir=cache_dir,
        )
    )
    print(f"[download] hf downloaded: {cached}")
    return _extract_zip_to_flat(cached, out_dir)


def extract_local_zip(local_zip: Path, out_dir: Path) -> dict[str, Path]:
    """Offline fallback: dùng zip có sẵn ở repo root (gitignored)."""
    print(f"[download] using local {local_zip}")
    return _extract_zip_to_flat(local_zip, out_dir)


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="Default: <repo>/data/raw")
    parser.add_argument(
        "--source",
        choices=("hf", "local"),
        default="hf",
        help="Primary download source. 'hf' = Hugging Face private dataset, 'local' = local zip at repo root.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else repo_root() / "data" / "raw"
    nested = out_dir / "Dataset"
    if nested.exists():
        print(f"[download] removing stale nested dir: {nested}")
        shutil.rmtree(nested)
    ensure_dir(out_dir)

    extracted: dict[str, Path] = {}

    if args.source == "hf":
        hf_repo = os.environ.get("HF_DATASET_REPO", "").strip()
        hf_file = os.environ.get("HF_DATASET_FILE", DEFAULT_HF_FILE).strip() or DEFAULT_HF_FILE
        hf_token = os.environ.get("HF_TOKEN", "").strip()
        try:
            extracted = download_from_hf(hf_repo, hf_file, out_dir, token=hf_token)
        except Exception as e:
            print(f"[download] HF download failed: {type(e).__name__}: {e}")
            print("[download] fallback: trying local zip at repo root")
            local = repo_root() / DEFAULT_HF_FILE
            if not local.exists():
                print("[download] no local zip; abort.", file=sys.stderr)
                return 1
            extracted = extract_local_zip(local, out_dir)
    else:  # --source local
        local = repo_root() / DEFAULT_HF_FILE
        if not local.exists():
            print(f"[download] local zip not found: {local}", file=sys.stderr)
            return 1
        extracted = extract_local_zip(local, out_dir)

    if not extracted:
        print("[download] no train/test/public/private split found", file=sys.stderr)
        return 1

    for split, p in extracted.items():
        n_files = sum(1 for _ in p.glob("*.json"))
        print(f"[download] ready: {p}  ({n_files} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
