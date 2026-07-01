# COLAB_GUIDE.md

Step-by-step blocks to run this project on a fresh Colab Pro+ A100 session.

| # | Stage | Notebook | Time |
|---|---|---|---|
| 0a | One-time: upload data to HF private dataset (run **local**) | — | 1 min |
| 0 | Setup Colab | `00_setup_colab.ipynb` | 2 min |
| 1 | Download data + build SFT JSONL | `01_data_prep.ipynb` | 1 min |
| 2 | Enrich data (optional, DashScope) | `02_enrich.ipynb` | 10–30 min |
| 3 | Train QLoRA | `03_train_qlora.ipynb` | 30–60 min |
| 4 | Train Full FT | `04_train_fullft.ipynb` | 1–2 h |
| 5 | Evaluate + compare | `05_evaluate.ipynb` | 5 min |
| 6 | Generate submission | `06_infer_submission.ipynb` | 5 min |
| 7 | Push model to HF Hub (optional) | `07_push_hf.ipynb` | 5–15 min |

## BƯỚC 0a — Upload data to HF (local, one-time)

```bash
cd finetune_1B_MCQ_VN
cp .env.example .env
# edit .env: set HF_TOKEN (write scope), HF_DATASET_REPO=ThinhDao/TempoRun2025_UIT

pip install -e .
python scripts/upload_data_to_hf.py
```

Script validates the zip, asks for confirmation, creates the repo as PRIVATE
(if it doesn't exist), and uploads. It prints the HF URL when done.

## BƯỚC 0 — Setup Colab

**Notebook**: `notebooks/00_setup_colab.ipynb`

**Block 0.1 — GPU check**
```python
import torch
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM (GB):', torch.cuda.get_device_properties(0).total_memory / 1e9)
```

**Block 0.2 — Clone repo**
```python
import os
REPO_URL = "https://github.com/TristanDao/finetune_1B_MCQ_VN.git"
REPO_DIR = "/content/finetune_1B_MCQ_VN"
if os.path.isdir(REPO_DIR):
    %cd {REPO_DIR} && !git pull --rebase --autostash
else:
    !git clone {REPO_URL} {REPO_DIR} && %cd {REPO_DIR}
```

**Block 0.3 — Install deps**
```python
!pip install -q -e .
!pip install -q -r requirements.txt
```

**Block 0.4 — Write `.env`** (fill the values, then run)
```python
%%writefile .env
HF_TOKEN=hf_your_hf_token
HF_DATASET_REPO=ThinhDao/TempoRun2025_UIT
HF_DATASET_FILE=tempo-run-2025-run-with-ai-break-limits.zip
HF_REPO=your_hf_username/temprun-qwen3-0_6b
DASHSCOPE_API_KEY=sk-your-dashscope-key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3-max-preview
DRIVE_ROOT=/content/drive/MyDrive/temprun_runs
```

**Block 0.5 — Verify `.env`**
```python
from temprun.utils import load_env, repo_root
load_env(repo_root() / ".env")
import os
required = ["HF_TOKEN", "HF_DATASET_REPO", "HF_DATASET_FILE", "HF_REPO"]
for k in required:
    v = os.environ.get(k, "")
    print(f"{k:20s} {'OK' if v and 'your' not in v else 'MISSING'}")
```

**Block 0.6 — Mount Drive (optional, recommended)**
```python
from google.colab import drive
drive.mount('/content/drive')
import os
DRIVE_ROOT = os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive/temprun_runs")
os.makedirs(DRIVE_ROOT, exist_ok=True)
print(f"DRIVE_ROOT = {DRIVE_ROOT}")
```

**Block 0.7 — Smoke test**
```python
from temprun.data import build_rows, stratified_split
from temprun.prompts import build_user_instruction
from temprun.utils import parse_generated, set_seed
print("All imports OK")
```

## BƯỚC 1 — Download data + build SFT JSONL

**Notebook**: `notebooks/01_data_prep.ipynb`

```python
%cd /content/finetune_1B_MCQ_VN
from temprun.utils import load_env, repo_root
load_env(repo_root() / ".env")
```

```bash
!python scripts/download_data.py
```

```bash
!echo "train files: $(ls data/raw/train | wc -l)"
!echo "test  files: $(ls data/raw/test | wc -l)"
```

```bash
!python scripts/make_sft_jsonl.py --in data/raw/train --out data/processed
```

## BƯỚC 2 — Enrich (optional)

Skip this if you don't want to use the Qwen3 API. Falls back to the 4,491 raw
training examples, which is enough for a first pass.

```python
%cd /content/finetune_1B_MCQ_VN
from temprun.utils import load_env, repo_root
load_env(repo_root() / ".env")
import os
assert os.environ.get("DASHSCOPE_API_KEY"), "DASHSCOPE_API_KEY missing"
```

```python
from temprun.enrich import call_chat, get_client, get_model
out = call_chat(get_client(),
    [{"role": "system", "content": "Bạn là trợ lý."},
     {"role": "user", "content": "2+2=?"}],
    max_tokens=20, temperature=0.0)
print(out)
```

```bash
!python scripts/enrich_data.py \
    --in  data/processed/train.jsonl \
    --out data/processed/enriched.jsonl
!python scripts/merge_enriched.py
```

## BƯỚC 3 — Train QLoRA

**Notebook**: `notebooks/03_train_qlora.ipynb`

```python
%cd /content/finetune_1B_MCQ_VN
from temprun.utils import load_env, repo_root
load_env(repo_root() / ".env")
import torch
assert torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory >= 15e9
```

```bash
!cat configs/qlora_qwen3_0_6b.yaml
!python scripts/train.py --config configs/qlora_qwen3_0_6b.yaml
```

```python
# Backup to Drive
import os, shutil
DRIVE_ROOT = os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive/temprun_runs")
if os.path.isdir("/content/drive/MyDrive"):
    src, dst = "artifacts/qlora_qwen3_0_6b", os.path.join(DRIVE_ROOT, "qlora_qwen3_0_6b")
    os.makedirs(DRIVE_ROOT, exist_ok=True)
    if os.path.exists(dst): shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"Copied to: {dst}")
```

## BƯỚC 4 — Train Full FT

**Notebook**: `notebooks/04_train_fullft.ipynb`

```python
%cd /content/finetune_1B_MCQ_VN
from temprun.utils import load_env, repo_root
load_env(repo_root() / ".env")
import torch
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
assert vram_gb >= 30, f"Full FT cần ≥30GB VRAM (hiện có {vram_gb:.1f}GB)"
```

```bash
!python scripts/train.py --config configs/fullft_qwen3_0_6b.yaml
```

Backup to Drive using the same pattern as Bước 3.

## BƯỚC 5 — Evaluate

**Notebook**: `notebooks/05_evaluate.ipynb`

```bash
!python scripts/evaluate.py \
    --checkpoint artifacts/qlora_qwen3_0_6b \
    --eval-jsonl data/processed/final/eval.jsonl \
    --mode logits --batch-size 16

!python scripts/evaluate.py \
    --checkpoint artifacts/fullft_qwen3_0_6b \
    --eval-jsonl data/processed/final/eval.jsonl \
    --mode logits --batch-size 16
```

```python
import json
from pathlib import Path
rows = []
for run in ["qlora_qwen3_0_6b", "fullft_qwen3_0_6b"]:
    p = Path(f"artifacts/{run}/eval_details.jsonl.summary.json")
    if p.exists():
        s = json.loads(p.read_text())
        rows.append({"run": run, **s})
print(f"{'run':25s} {'acc':>8s} {'correct':>8s} {'time_s':>8s}")
for r in rows:
    print(f"{r['run']:25s} {r['accuracy']:7.2f}% {r['correct']:8d} {r['time_sec']:8.2f}")
```

## BƯỚC 6 — Submission

**Notebook**: `notebooks/06_infer_submission.ipynb`

```python
RUN = "qlora_qwen3_0_6b"  # or "fullft_qwen3_0_6b" if it's better
```

```bash
!python scripts/infer.py \
    --checkpoint artifacts/{RUN} \
    --test-dir  data/raw/test \
    --out       submissions/sub_{RUN}_public.csv \
    --mode logits --batch-size 16
```

```python
import pandas as pd
df = pd.read_csv(f"submissions/sub_{RUN}_public.csv")
print(f"Rows: {len(df)}  |  pred dist: {df['answer'].value_counts().to_dict()}")
print(f"Duplicates: {df['row_id'].duplicated().sum()}")
print(df.head())
```

## BƯỚC 7 — Push model to HF (optional)

**Notebook**: `notebooks/07_push_hf.ipynb`

```python
MODE = "lora"  # "lora" (50–100MB) | "merged" (1.2GB)
RUN  = "qlora_qwen3_0_6b"
```

```python
import os
from huggingface_hub import HfApi, create_repo
from temprun.utils import load_env, repo_root
load_env(repo_root() / ".env")

REPO = os.environ["HF_REPO"]
api = HfApi(token=os.environ["HF_TOKEN"])
create_repo(REPO, token=os.environ["HF_TOKEN"], exist_ok=True, private=False)
print(f"HF repo ready: https://huggingface.co/{REPO}")
```

```python
if MODE == "lora":
    api.upload_folder(
        folder_path=f"artifacts/{RUN}",
        repo_id=REPO,
        commit_message=f"Upload LoRA adapter from {RUN}",
        ignore_patterns=["*.bin", "*.safetensors", "global_step/*", "runs/*", "eval_details*"],
    )
elif MODE == "merged":
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Instruct", torch_dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(base, f"artifacts/{RUN}").merge_and_unload()
    tok = AutoTokenizer.from_pretrained(f"artifacts/{RUN}")
    out = f"artifacts/{RUN}_merged"
    import os; os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True, max_shard_size="500MB")
    tok.save_pretrained(out)
    api.upload_folder(folder_path=out, repo_id=REPO, commit_message=f"Upload merged {RUN}")
print(f"Pushed to https://huggingface.co/{REPO}")
```
