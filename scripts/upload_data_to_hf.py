"""One-time helper: upload tempo-run-2025-run-with-ai-break-limits.zip lên
Hugging Face private dataset repo.

Chạy LOCAL (1 lần duy nhất) trước khi sang Colab. Sau khi upload xong,
Colab chỉ cần `python scripts/download_data.py` để tải về.

Cần:
- HF_TOKEN trong .env (scope: write)
- HF_DATASET_REPO trong .env (vd: ThinhDao/TempoRun2025_UIT), repo sẽ được
  tạo PRIVATE nếu chưa tồn tại
- File zip ở repo root (mặc định) hoặc truyền --zip
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

from temprun.utils import ensure_dir, load_env, repo_root

DEFAULT_ZIP = "tempo-run-2025-run-with-ai-break-limits.zip"


def validate_zip(zip_path: Path) -> None:
    """Make sure the zip looks like the expected dataset layout."""
    if not zip_path.is_file():
        raise FileNotFoundError(f"zip not found: {zip_path}")
    if zip_path.stat().st_size == 0:
        raise ValueError(f"zip is empty: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    # Must contain at least one train/ file in the expected nested layout
    has_train = any(n.startswith("Dataset/Dataset/train/") and n.endswith(".json") for n in names)
    if not has_train:
        raise ValueError(
            f"zip {zip_path} does not look like a TempoRun2025 dataset "
            f"(missing Dataset/Dataset/train/*.json). First 5 entries:\n"
            + "\n".join(names[:5])
        )
    print(f"[upload] zip OK ({zip_path.stat().st_size / 1e6:.1f} MB, {len(names)} entries)")


def main() -> int:
    load_env()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        default=str(repo_root() / DEFAULT_ZIP),
        help="Path to local zip (default: <repo>/tempo-run-2025-run-with-ai-break-limits.zip)",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="HF dataset repo (default: env HF_DATASET_REPO)",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="File name inside repo (default: basename of --zip)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make repo PUBLIC instead of private. OFF by default — DO NOT use for this dataset.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Skip the safety prompt before uploading",
    )
    args = parser.parse_args()

    repo_id = (args.repo or os.environ.get("HF_DATASET_REPO", "")).strip()
    if not repo_id:
        print("[upload] HF_DATASET_REPO is empty. Set it in .env or pass --repo.", file=sys.stderr)
        return 1

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("[upload] HF_TOKEN is empty. Fill it in .env.", file=sys.stderr)
        return 1

    zip_path = Path(args.zip)
    try:
        validate_zip(zip_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[upload] validation failed: {e}", file=sys.stderr)
        return 1

    filename = args.filename or zip_path.name

    visibility = "PUBLIC" if args.public else "PRIVATE"
    if not args.yes:
        print()
        print("=" * 60)
        print(f"  HF repo:  {repo_id}  (type: dataset, {visibility})")
        print(f"  File:     {filename}")
        print(f"  Source:   {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
        print("=" * 60)
        if not args.public:
            print("  ⚠️  Repo sẽ là PRIVATE (mặc định). Chỉ bạn truy cập qua HF_TOKEN.")
        else:
            print("  ⚠️  Repo sẽ PUBLIC! Ai có link đều tải được. Cân nhắc kỹ.")
        print()
        ans = input("Continue? [y/N] ").strip().lower()
        if ans != "y":
            print("[upload] aborted.")
            return 1

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    print(f"[upload] create_repo({repo_id!r}, private={not args.public}, exist_ok=True) ...")
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
        token=token,
    )
    print(f"[upload] upload_file -> {repo_id}/{filename}")
    api.upload_file(
        path_or_fileobj=str(zip_path),
        path_in_repo=filename,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    print(f"[upload] DONE. Repo URL: {url}")
    print()
    print("Trên Colab, đảm bảo .env có:")
    print(f"  HF_DATASET_REPO={repo_id}")
    print(f"  HF_DATASET_FILE={filename}")
    print(f"  HF_TOKEN=<token write scope, cùng token vừa dùng>")
    print()
    print("Sau đó chạy:  python scripts/download_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
