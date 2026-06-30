"""CLI for training. Loads a YAML config (with optional `extends:`) and runs SFT."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from temprun.train import train
from temprun.utils import load_config, load_env, repo_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config (relative to repo root)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    load_env()
    cfg = load_config(args.config)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root() / cfg_path
    print(f"[cli] loaded config: {cfg_path}")

    out_dir, _ = train(cfg, output_dir=args.output_dir)
    print(f"[cli] done. artifacts at: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
