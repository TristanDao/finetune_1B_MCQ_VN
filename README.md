# finetune_1B_MCQ_VN

End-to-end fine-tuning of a sub-1B LLM on Vietnamese multiple-choice reading
comprehension. Built around the Tempo Run 2025 competition
([UIT](https://www.uit.edu.vn/), Vietnam) and packaged as a reusable MLOps template.

> **Task**: given a Vietnamese news article and a 4-way MCQ, predict the correct
> option (A/B/C/D). 4,491 training questions over 1,500 articles, ~1,500 test
> questions. Model parameter cap: 1B.

## Highlights

- **QLoRA + Full FT, side by side** on the same data with identical configs
- **Sub-1B base** (`Qwen3-0.6B`) so the whole thing fits a single A100
- **Vietnamese prompt, single-token target** → fast argmax-on-logits inference,
  5–10× faster than greedy decoding
- **Reproducible from a fresh VM** in <15 min: clone → `pip install -e .` →
  data download → train → eval → submission CSV
- **Data discipline**: dataset hosted on a private HF repo, never committed
  to git; raw data, artifacts, and submissions all gitignored

## Results

| Method                          | Trainable params | Hold-out acc | Public test |
|---------------------------------|------------------|--------------|-------------|
| Qwen3-0.6B (zero-shot) | 0                | _TBD_        | _TBD_       |
| Qwen3-0.6B + QLoRA (r=32)       | ~3M              | _TBD_        | _TBD_       |
| Qwen3-0.6B + Full FT            | ~600M            | _TBD_        | _TBD_       |
| Qwen3-0.6B + Unsloth QLoRA      | ~3M              | _TBD_        | _TBD_       |

> Eval JSONs land in `artifacts/<run>/eval_details.jsonl.summary.json` after
> each training run. Update this table when numbers come in.

## Stack

- **Model**: [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) (post-trained instruct, text-only, dense)
- **Training**: `transformers` · `peft` (LoRA) · `trl` (SFTTrainer) · `bitsandbytes` (4-bit)
- **Training (optional)**: [Unsloth](https://github.com/unslothai/unsloth) `FastLanguageModel` — Triton kernels + FA2 tích hợp, nhanh ~2x. Cài qua `pip install -e ".[unsloth]"`.
- **Attention**: FlashAttention-2 (qua Unsloth hoặc `pip install -e ".[flash]"`); fallback `sdpa` nếu chưa cài.
- **Data enrichment** (optional): Alibaba DashScope `qwen3-max-preview`
- **Distribution**: HF Hub — private dataset for raw data, public/private repo for the trained adapter
- **Compute**: Google Colab Pro+ (A100 40/80GB) / Kaggle / Runpod

## Architecture

```
raw data  →  HF private dataset  ──┐
                                   ├──> download_data.py
                                   ▼
                          data/raw/{train,test}/
                                   │
                                   ▼
                         make_sft_jsonl.py
                                   │
                                   ▼
                data/processed/{train,eval}.jsonl
                                   │
                ┌──────────────────┴──────────────────┐
                ▼                                     ▼
         enrich_data.py (optional)            train.py
              (Qwen3 paraphrase)            (QLoRA or Full FT)
                │                                   │
                └─────────────┬─────────────────────┘
                              ▼
                       evaluate.py  →  accuracy, confusion, JSONL
                              │
                              ▼
                         infer.py  →  submissions/sub_<run>.csv
```

## Layout

```
src/temprun/        # Reusable Python package
  prompts.py        # Vietnamese prompt construction
  data.py           # JSON → SFT JSONL with stratified split
  enrich.py         # Qwen3 API client for synthetic data
  train.py          # SFT trainer (QLoRA + Full FT)
  evaluate.py       # Batch inference, accuracy, confusion
  infer.py          # Submission CSV generation
scripts/            # CLI entrypoints — one per pipeline stage
configs/            # YAML hyperparameter files
notebooks/          # Colab glue — one notebook per stage
tests/              # pytest smoke tests
```

## Quick start

```bash
git clone https://github.com/TristanDao/finetune_1B_MCQ_VN
cd finetune_1B_MCQ_VN
pip install -e .
cp .env.example .env                 # fill HF_TOKEN, HF_DATASET_REPO

python scripts/download_data.py
python scripts/make_sft_jsonl.py
python scripts/train.py --config configs/qlora_qwen3_0_6b.yaml
# Hoặc dùng Unsloth (nhanh ~2x, cần `pip install -e ".[unsloth]"`):
# python scripts/train.py --config configs/unsloth_qwen3_0_6b.yaml
python scripts/evaluate.py --checkpoint artifacts/qlora_qwen3_0_6b
python scripts/infer.py    --checkpoint artifacts/qlora_qwen3_0_6b \
                            --test-dir  data/raw/test \
                            --out       submissions/sub_qlora_public.csv
PYTHONPATH=src pytest -q              # 33 tests
```

For the Colab walkthrough see [`COLAB_GUIDE.md`](./COLAB_GUIDE.md). For the
agent contract / project conventions see [`AGENT.md`](./AGENT.md).

## Dependency management

`pyproject.toml` is the **single source of truth** for dependencies. The two
`requirements*.txt` files and `uv.lock` are auto-generated from it — never edit
them by hand.

| File | Generated by | Used by |
|---|---|---|
| `pyproject.toml` | hand-edited | dev workflow, `pip install -e .` |
| `requirements.txt` | `uv pip compile pyproject.toml` | `pip install -r` on Colab/Docker |
| `requirements-dev.txt` | `uv pip compile pyproject.toml --extra dev` | local dev / CI |
| `uv.lock` | `uv lock` | `uv sync` (reproducible installs) |

Regenerate everything after editing `pyproject.toml`:

```bash
uv lock                                # refresh uv.lock
uv pip compile pyproject.toml -o requirements.txt
uv pip compile pyproject.toml --extra dev -o requirements-dev.txt
```

Or install directly with `uv` (skips the .txt files entirely):

```bash
uv sync                                # prod + dev
uv sync --no-dev                       # prod only
```

## Engineering choices

- **Sub-1B base** (`Qwen3-0.6B`): respects the hard parameter cap and
  keeps the QLoRA-vs-FT comparison honest (both fit comfortably on an A100).
  Note: `Qwen/Qwen3-0.6B` on HF is already the post-trained (instruct) variant;
  the base pre-trained model is `Qwen/Qwen3-0.6B-Base`.
- **`enable_thinking=False` in chat template** (Qwen3-specific): Qwen3 defaults
  to thinking mode and would emit `<think>…</think>` before the answer. We render
  with `enable_thinking=False` (via `chat_template_kwargs` in config) so the
  assistant turn starts right after the empty-thinking block — train and infer
  share the exact same prefix. Without this, logits-mode eval (argmax over
  A/B/C/D at the first assistant token) returns near-random.
- **`assistant_only_loss=True`** in TRL: loss is computed only on the
  single-character response, not on the system/user prompt — prevents the
  model from memorising the question template.
- **Two training backends**: `trl` (HF + TRL SFTTrainer, default) and `unsloth`
  (FastLanguageModel, Triton kernels + FA2, ~2x faster). Selected via
  `backend:` in YAML. Unsloth adapter exports are peft-compatible →
  `infer.py` loads them unchanged.
- **Deterministic A–D ordering** in the prompt, single-token output →
  evaluator uses argmax-over-logits inference (no generation loop).
- **Stratified 90/10 split on the answer label**: `D` is only ~3% of the data,
  so naïve splits risk an eval set with no D examples.
- **Private HF dataset for the corpus**: the competition forbids
  redistributing the data; private HF repos give us reproducible-by-token
  downloads without leaking data into git history.

## License

MIT
