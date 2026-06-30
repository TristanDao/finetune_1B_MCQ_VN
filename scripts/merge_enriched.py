"""Merge enriched.jsonl into the base train.jsonl, then re-split 90/10.

Input layout expected:
  data/processed/train.jsonl   # base, from make_sft_jsonl.py
  data/processed/eval.jsonl
  data/processed/enriched.jsonl # contains base + new rows (with _source: paraphrase|synth)

Output:
  data/processed/final/train.jsonl
  data/processed/final/eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from temprun.data import read_jsonl, stratified_split, write_jsonl
from temprun.utils import ensure_dir, repo_root


def _flatten_title_content_choices(row: dict) -> dict:
    """Reconstruct {title, content, question, choices, label} from a row that
    was produced by make_sft_jsonl.py (so it has full fields) OR by enrich.py
    (which has only `messages` and `label`)."""
    if all(k in row for k in ("title", "content", "question", "choices", "label")):
        return row
    # Otherwise, the original fields are missing; we can't recover title/content
    # for a paraphrased/synth row that was emitted by enrich.py. We just
    # normalize so it can still be written. Title/content will be blank.
    out = dict(row)
    out.setdefault("title", "")
    out.setdefault("content", "")
    out.setdefault("question", "")
    out.setdefault("choices", {})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", default=str(repo_root() / "data" / "processed"))
    parser.add_argument("--out", default=str(repo_root() / "data" / "processed" / "final"))
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    random.seed(args.seed)
    proc = Path(args.processed)
    out = Path(args.out)
    ensure_dir(out)

    base_train = read_jsonl(proc / "train.jsonl")
    base_eval = read_jsonl(proc / "eval.jsonl")
    enriched_path = proc / "enriched.jsonl"
    if not enriched_path.exists():
        print(f"[merge] no {enriched_path}; using base only")
        merged = base_train + base_eval
    else:
        enriched = read_jsonl(enriched_path)
        print(f"[merge] base_train={len(base_train)} base_eval={len(base_eval)} enriched={len(enriched)}")
        # enriched already contains base rows + new ones; just dedup on text/messages
        seen: set[str] = set()
        merged: list[dict] = []
        for r in enriched:
            key = json.dumps(r.get("messages", []), ensure_ascii=False, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            merged.append(_flatten_title_content_choices(r))

    print(f"[merge] merged={len(merged)} unique rows")

    train, evald = stratified_split(merged, test_size=args.test_size, seed=args.seed)
    print(f"[merge] final train={len(train)} eval={len(evald)}")
    write_jsonl(train, out / "train.jsonl")
    write_jsonl(evald, out / "eval.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
