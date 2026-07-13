# Colab / Kaggle Guide

Step-by-step to train and submit on a GPU notebook.

## Prerequisites

- GPU with ≥15GB VRAM (T4, L4, A100)
- Hugging Face token with read access to a private dataset repo

## Full Pipeline (single session)

### 0. Setup

```bash
!pip install -q unsloth datasets scikit-learn pandas python-dotenv pyyaml huggingface-hub protobuf tqdm sentencepiece
```

```python
import os, sys
!git clone https://github.com/TristanDao/finetune_1B_MCQ_VN /content/finetune_1B_MCQ_VN
%cd /content/finetune_1B_MCQ_VN
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
```

```python
# Write .env
%%writefile .env
HF_TOKEN=hf_your_token
HF_DATASET_REPO=your_username/dataset_repo
```

```python
# Verify GPU
import torch
assert torch.cuda.is_available(), "Need GPU!"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 1. Download data

```bash
!python scripts/download_data.py
!echo "train: $(ls data/raw/train | wc -l) files"
!echo "test:  $(ls data/raw/test | wc -l) files"
```

### 2. Build SFT JSONL

```bash
# Conversation mode (direct answer)
!python scripts/make_sft_jsonl.py --mode conversation

# Or Chain-of-Thought mode (step-by-step reasoning)
# !python scripts/make_sft_jsonl.py --mode cot
```

Output: `data/processed/final/{train,eval}.jsonl` with balanced labels.

### 3. Train

```bash
# Conversation mode — fast, small LoRA
!python scripts/train.py \
    --mode conversation \
    --epochs 3 \
    --batch-size 4 \
    --grad-accum-steps 8 \
    --lr 2e-4 \
    --output-dir artifacts/conversation

# Chain-of-Thought mode — deeper reasoning, needs longer context
# !python scripts/train.py \
#     --mode cot \
#     --epochs 4 \
#     --max-seq-length 4096 \
#     --lora-alpha 64 \
#     --lr 3e-4 \
#     --output-dir artifacts/cot
```

Artifacts saved to `artifacts/<run>/adapter/` (LoRA weights) and `artifacts/<run>/merged_16bit/` (standalone model).

### 4. Evaluate

```bash
!python scripts/evaluate.py \
    --adapter artifacts/conversation/adapter \
    --eval-jsonl data/processed/final/eval.jsonl \
    --mode conversation
```

Outputs accuracy, confusion matrix, and per-question details to `artifacts/conversation/eval_details.jsonl`.

### 5. Generate submission

```bash
!python scripts/infer.py \
    --adapter artifacts/conversation/adapter \
    --test-dir data/raw/test \
    --out submissions/sub_conversation.csv \
    --mode conversation
```

```python
# Quick check
import pandas as pd
df = pd.read_csv("submissions/sub_conversation.csv")
print(f"Rows: {len(df)}, pred dist: {df['answer'].value_counts().to_dict()}")
```

## Choosing Between Modes

| | Conversation | Chain-of-Thought |
|---|---|---|
| **Answer format** | `"A"` | `"Vì lý do X...\n\nDo đó, đáp án đúng là A."` |
| **Training time** | ~30 min (T4) | ~45 min (T4) |
| **Max sequence** | 2048 | 4096 |
| **LoRA alpha** | 16 | 64 |
| **Epochs** | 3 | 4 |
| **Best for** | Speed, deployment | Interpretability, debugging |

## Troubleshooting

- **OOM during training**: reduce `--batch-size` to 2 and increase `--grad-accum-steps`
- **OOM during inference**: reduce `--batch-size` to 2
- **Letter tokens not single**: Qwen3 tokenizer guarantees A/B/C/D are single tokens — this should never happen
- **Random predictions**: ensure `enable_thinking=False` is passed to chat template (handled automatically)
