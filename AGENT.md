# AGENT.md — Tempo Run 2025 (MCQ Tiếng Việt, <1B params)

Contract for AI agents (and humans) working on this repo. Read this before
editing anything.

## Goal
Finetune a sub-1B LLM on Vietnamese MCQ reading comprehension (4,491 train
questions, 1,488 test questions). Metric: **accuracy** on the public/private
test sets. Constraint: total model params < 1B; no test-set leakage.

## Locked decisions
| | Choice | Why |
|---|---|---|
| Base model | `Qwen/Qwen3-0.6B` | <1B cap, decent Vietnamese, sub-1s inference |
| Methods | QLoRA 4-bit + Full FT bf16, side by side | Compare on identical data, A100 budget |
| Sequence length | 2048 | Covers ~80% of articles (median 1.2k tokens) |
| Prompt | Vietnamese, deterministic A–D order, single-token target | Fast argmax inference |
| Eval mode | `logits` (primary) → `generate` (fallback) | 5–10× faster than greedy decoding |
| Data enrichment | DashScope `qwen3-max-preview` paraphrase + explanation | Free 1M-token tier |
| Data source | HF private dataset `ThinhDao/TempoRun2025_UIT` | Data redistribution is forbidden; private HF repo is reproducible-by-token |

## Code conventions
- Python ≥ 3.10, full type hints
- `src/temprun/` is the package — install with `pip install -e .`
- Hyperparameters live in YAML (`configs/`), never hardcoded in `.py`
- CLI scripts in `scripts/`, one per pipeline stage; notebooks are glue only
- Test with `PYTHONPATH=src pytest -q` (33 smoke tests)

## Pipeline
```
LOCAL ONE-TIME:
  upload_data_to_hf.py  →  Hugging Face private dataset

EACH COLAB SESSION:
  download_data.py   →  data/raw/{train,test,sample_submission.csv}
  make_sft_jsonl.py  →  data/processed/{train,eval}.jsonl
  enrich_data.py     →  data/processed/enriched.jsonl     (optional)
  merge_enriched.py  →  data/processed/final/{train,eval}.jsonl
  train.py           →  artifacts/<run>/
  evaluate.py        →  artifacts/<run>/eval_details.jsonl.summary.json
  infer.py           →  submissions/sub_<run>.csv
```

## Repo state
- **Commit**: source, configs, notebooks, tests, docs
- **Never commit**: data, artifacts, submissions, reports/*.json, .env, *.zip
- **Push to HF** (when results are good): LoRA adapter (~50–100MB) or merged model (~1.2GB)

## Known quirks
- A few JSONs have empty `content` (3 train + 4 test) — code drops them, no crash
- Some test files have no `questions` — skipped, no `row_id` emitted for them
- `A`/`B`/`C`/`D` must be single-token (verified for Qwen3 tokenizer; if a
  different base is used, the evaluator auto-falls back to `generate` mode)
- W&B is disabled by default; enable with `export WANDB_MODE=online` if needed
- **Never push data, checkpoints, or submission CSVs to GitHub.** All large
  artifacts go through HF (private) or Drive (personal).
