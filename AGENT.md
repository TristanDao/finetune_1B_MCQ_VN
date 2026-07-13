# AGENT.md — Tempo Run 2025 (MCQ Tiếng Việt, <1B params)

Contract for AI agents (and humans) working on this repo.

## Goal
Fine-tune a sub-1B LLM on Vietnamese MCQ reading comprehension (4,491 train
questions, 1,500 articles). Two training modes: **conversation** (direct answer)
and **Chain-of-Thought** (step-by-step reasoning). Metric: **accuracy** on
held-out eval split and public/private test sets.

## Locked Decisions

| | Choice | Why |
|---|---|---|
| Base model | `Qwen/Qwen3-0.6B` | <1B cap, good Vietnamese, single-token A/B/C/D |
| Backend | Unsloth only | Triton kernels + FA2 built-in, ~2× faster |
| Quantization | QLoRA 4-bit (NF4) | Fits T4 (15GB), no accuracy loss vs full FT |
| Prompt | Vietnamese, deterministic A–D order | No English leakage, reproducible outputs |
| Chat template | `enable_thinking=False` | Qwen3 defaults to `<think>` mode — blocks it |
| Loss masking | `train_on_responses_only` | Only the assistant turn contributes to loss |
| Label balancing | Choice rotation (no API) | A→B→C→D cycle per row, ~25% per label |
| Eval | Logits-mode (argmax over 4 tokens) | Single forward pass, 5–10× faster than generation |
| Data source | HF private dataset | Competition forbids redistribution |

## Project Structure

```
src/temprun/
├── prompts.py    # Dual-mode: conversation / Chain-of-Thought
├── data.py       # JSON → JSONL, label balancing, stratified split
├── train.py      # Unsloth + QLoRA + SFTTrainer
├── evaluate.py   # Logits-mode eval + confusion matrix
├── infer.py      # Test inference → submission CSV
└── utils.py      # Chat rendering, seeding, letter-token extraction

scripts/          # One script per pipeline stage
configs/          # Reference hyperparameters
tests/            # 33 pytest smoke tests
```

## Pipeline

```
download_data.py       → data/raw/{train,test}/
make_sft_jsonl.py      → data/processed/final/{train,eval}.jsonl
  --mode {conversation|cot}
train.py               → artifacts/<run>/adapter/
  --mode {conversation|cot}
evaluate.py            → eval_details.jsonl + summary.json
  --adapter artifacts/<run>/adapter
infer.py               → submissions/sub.csv
  --adapter artifacts/<run>/adapter
```

## Code Conventions

- Python ≥ 3.10
- `src/temprun/` is the installable package (`pip install -e .`)
- Hyperparameters as CLI args (not YAML configs); configs/ are reference only
- Test: `python -m pytest tests/ -v`
- Lint: `python -m ruff check src/ scripts/ tests/`

## Never Commit

- `data/`, `artifacts/`, `submissions/`, `reports/`
- `.env`, `*.zip`, `*.bin`, `*.safetensors`
- Notebook output cells (clear before commit)
