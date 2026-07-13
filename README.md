# Fine-tuning Sub-1B LLM for Vietnamese MCQ

Fine-tuning [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) on Vietnamese
reading comprehension MCQs from the **Tempo Run 2025** competition.

> 1,500 news articles → 4,491 training questions → predict A/B/C/D. Model cap: 1B params.

## Why This Project Is Interesting

- **Sub-1B + QLoRA 4-bit**: fits a single T4 GPU (15 GB VRAM), trains in ~30 min
- **Two training modes**: direct answer (fast) and Chain-of-Thought (interpretable)
- **Smart label balancing**: skewed labels (D=3%) → choice-rotation → ~25% each, zero API cost
- **Logits-mode inference**: argmax over 4 tokens in a single forward pass, 5–10× faster than generation
- **Vietnamese NLP**: prompts, system instructions, and model output all in Vietnamese

## Results

| Experiment | Trainable | Accuracy |
|---|---|---|
| Qwen3-0.6B (zero-shot) | 0 | *TBD* |
| Qwen3-0.6B + QLoRA (conversation) | ~3M | *TBD* |
| Qwen3-0.6B + QLoRA (CoT) | ~3M | *TBD* |

## Quick Start

```bash
git clone https://github.com/TristanDao/finetune_1B_MCQ_VN
cd finetune_1B_MCQ_VN
pip install unsloth datasets scikit-learn pandas python-dotenv huggingface-hub tqdm
cp .env.example .env      # fill HF_TOKEN, HF_DATASET_REPO

python scripts/download_data.py
python scripts/make_sft_jsonl.py --mode conversation
python scripts/train.py --mode conversation --epochs 3
python scripts/evaluate.py --adapter artifacts/unsloth_qwen3_0_6b/adapter --eval-jsonl data/processed/final/eval.jsonl
python scripts/infer.py --adapter artifacts/unsloth_qwen3_0_6b/adapter --test-dir data/raw/test --out submissions/sub.csv
```

## Two Training Modes

| | **Conversation** | **Chain-of-Thought** |
|---|---|---|
| **Answer** | `"A"` | `"Vì lý do...\n\nDo đó, đáp án đúng là A."` |
| **Max seq** | 2048 | 4096 |
| **LoRA** | r=32, α=16 | r=32, α=64 |
| **Epochs** | 3 | 4 |
| **Use case** | Speed, deployment | Interpretability |

## Pipeline

```
data/raw/{train,test}.json
        │  download_data.py
        ▼
data/processed/final/{train,eval}.jsonl
        │  make_sft_jsonl.py (stratified split + label balancing)
        ▼
train.py ── Unsloth + QLoRA + SFTTrainer
        │
        ├── evaluate.py ── logits-mode accuracy + confusion matrix
        └── infer.py    ── submission.csv
```

## Key Engineering Decisions

| Decision | Why |
|---|---|
| **Unsloth only** | Triton kernels + FA2 → 2× faster, LoRA stays PEFT-compatible |
| **`enable_thinking=False`** | Qwen3 would emit `<think>` tokens, breaking logits-mode eval |
| **Loss on responses only** | Only assistant turn contributes to loss, no template memorization |
| **Choice-rotation balancing** | D: 3% → ~25% per label with deterministic rotation, no API cost |
| **Logits-mode inference** | `num_logits_to_keep=1`, argmax over {A,B,C,D} in one pass |

## Project Structure

```
src/temprun/           # Core library
  prompts.py           # Dual-mode prompts (conversation / CoT)
  data.py              # JSON → JSONL, label balancing, stratified split
  train.py             # Unsloth + QLoRA + SFTTrainer
  evaluate.py          # Logits-mode batch eval
  infer.py             # Test inference → submission CSV
  utils.py             # Chat rendering, seeding
scripts/               # One script per pipeline stage
configs/               # Reference hyperparameters
tests/                 # 33 pytest tests
```

## License

MIT
