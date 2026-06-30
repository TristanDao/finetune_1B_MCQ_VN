"""Data enrichment: gọi Alibaba DashScope (OpenAI-compatible) để paraphrase, sinh explanation,
và (tuỳ chọn) sinh câu hỏi mới.

Cần DASHSCOPE_API_KEY trong .env.
Model mặc định: qwen3-max-preview (override bằng DASHSCOPE_MODEL).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI

from .utils import ensure_dir, load_env


def get_client() -> OpenAI:
    load_env()
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY missing. Fill .env first.")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_model() -> str:
    return os.environ.get("DASHSCOPE_MODEL", "qwen3-max-preview")


def call_chat(
    client: OpenAI,
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    """Single chat completion with exponential backoff."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model or get_model(),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            sleep = backoff ** attempt
            print(f"[enrich] attempt {attempt + 1} failed: {e}; sleeping {sleep:.1f}s")
            time.sleep(sleep)
    raise RuntimeError(f"DashScope call failed after {retries} retries: {last_err}")


# -----------------------------------------------------------------------------
# Prompt builders
# -----------------------------------------------------------------------------

PARAPHRASE_SYSTEM = (
    "Bạn là trợ lý tạo dữ liệu huấn luyện. Nhiệm vụ: diễn đạt lại câu hỏi và các lựa chọn "
    "bằng tiếng Việt tự nhiên, giữ nguyên ý nghĩa và đáp án đúng. Chỉ trả về JSON hợp lệ."
)


def paraphrase_prompt(question: str, choices: dict[str, str], correct: str) -> list[dict]:
    user = (
        "Cho câu hỏi trắc nghiệm sau (đáp án đúng đã biết), hãy paraphrase để tạo phiên bản mới.\n"
        f"Câu hỏi gốc: {question}\n"
        f"Các lựa chọn gốc: {json.dumps(choices, ensure_ascii=False)}\n"
        f"Đáp án đúng: {correct}\n\n"
        "Yêu cầu:\n"
        "1. Diễn đạt lại câu hỏi và các lựa chọn bằng tiếng Việt, KHÔNG đổi nghĩa.\n"
        "2. Có thể đổi thứ tự lựa chọn nhưng PHẢI ghi rõ mapping đáp án mới.\n"
        "3. Trả về DUY NHẤT một JSON theo schema:\n"
        '{"question": "...", "choices": {"A": "...", "B": "...", "C": "...", "D": "..."}, '
        '"correct_answer": "A|B|C|D"}\n'
    )
    return [
        {"role": "system", "content": PARAPHRASE_SYSTEM},
        {"role": "user", "content": user},
    ]


EXPLAIN_SYSTEM = (
    "Bạn là trợ lý giải thích. Đọc văn bản, câu hỏi, các lựa chọn và đáp án đúng, "
    "rồi viết một đoạn giải thích ngắn (1-3 câu) bằng tiếng Việt vì sao đáp án đó đúng. "
    "Chỉ trả về JSON."
)


def explain_prompt(title: str, content: str, question: str, choices: dict[str, str], correct: str) -> list[dict]:
    user = (
        f"Tiêu đề: {title}\n\nNội dung: {content}\n\n"
        f"Câu hỏi: {question}\n"
        f"Các lựa chọn: {json.dumps(choices, ensure_ascii=False)}\n"
        f"Đáp án đúng: {correct}\n\n"
        'Trả về JSON: {"explanation": "..."}'
    )
    return [
        {"role": "system", "content": EXPLAIN_SYSTEM},
        {"role": "user", "content": user},
    ]


SYNTH_Q_SYSTEM = (
    "Bạn là trợ lý tạo câu hỏi trắc nghiệm chất lượng cao. Sinh 1 câu hỏi mới dựa trên văn bản, "
    "kèm 4 lựa chọn A/B/C/D (chỉ 1 đúng) và chỉ rõ đáp án. Trả về JSON hợp lệ."
)


def synth_q_prompt(title: str, content: str) -> list[dict]:
    user = (
        f"Tiêu đề: {title}\n\nNội dung: {content}\n\n"
        "Sinh 1 câu hỏi trắc nghiệm mới, 4 lựa chọn, 1 đáp án đúng. "
        'JSON: {"question": "...", "choices": {"A":"...", "B":"...", "C":"...", "D":"..."}, "correct_answer": "A|B|C|D"}'
    )
    return [
        {"role": "system", "content": SYNTH_Q_SYSTEM},
        {"role": "user", "content": user},
    ]


# -----------------------------------------------------------------------------
# Extraction helpers
# -----------------------------------------------------------------------------

def _extract_json(text: str) -> dict | None:
    """Best-effort JSON extract. Handles ```json ...``` fences."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    # Find outermost { ... }
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start : end + 1])
    except json.JSONDecodeError:
        return None


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def enrich_dataset(
    input_jsonl: str | os.PathLike,
    output_jsonl: str | os.PathLike,
    *,
    paraphrase: bool = True,
    explain: bool = True,
    synth: bool = False,
    synth_max: int = 200,
    seed: int = 3407,
) -> dict[str, int]:
    """Enrich a SFT jsonl by adding new rows. Idempotent via content-hash cache."""
    import hashlib
    from .data import read_jsonl

    random.seed(seed)
    client = get_client()
    model_name = get_model()

    in_path = Path(input_jsonl)
    out_path = Path(output_jsonl)
    ensure_dir(out_path.parent)

    rows = read_jsonl(in_path)
    print(f"[enrich] loaded {len(rows)} rows from {in_path}")

    cache_path = out_path.with_suffix(out_path.suffix + ".cache.json")
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    written: list[dict] = []
    counter = {"paraphrase": 0, "explain": 0, "synth": 0, "skipped": 0, "errors": 0}

    for r in rows:
        # The original row already has label; we don't need to re-derive here.
        # For paraphrasing we need the actual question/choices/label.
        # Re-derive from messages[1].content? Easier: assume input jsonl has
        # `question` and `choices` and `label` and `title` and `content` fields.
        if not all(k in r for k in ("question", "choices", "label", "title", "content")):
            counter["skipped"] += 1
            continue
        key = _hash(r["question"] + "|" + r["label"])
        if key in cache:
            continue

        try:
            if paraphrase:
                msgs = paraphrase_prompt(r["question"], r["choices"], r["label"])
                content = call_chat(client, msgs, model=model_name, temperature=0.8, max_tokens=800)
                parsed = _extract_json(content)
                if parsed and "question" in parsed and "choices" in parsed and "correct_answer" in parsed:
                    new_row = {
                        "messages": [
                            r["messages"][0],
                            {"role": "user", "content": _user_text_from_parsed(parsed, r)},
                            {"role": "assistant", "content": str(parsed["correct_answer"]).strip().upper()[:1]},
                        ],
                        "label": str(parsed["correct_answer"]).strip().upper()[:1],
                        "_source": "paraphrase",
                    }
                    written.append(new_row)
                    counter["paraphrase"] += 1
                    cache[key] = {"paraphrase": True}

            if explain:
                msgs = explain_prompt(r["title"], r["content"], r["question"], r["choices"], r["label"])
                content = call_chat(client, msgs, model=model_name, temperature=0.3, max_tokens=400)
                parsed = _extract_json(content)
                if parsed and "explanation" in parsed:
                    counter["explain"] += 1
                    cache[key] = {**cache.get(key, {}), "explanation": parsed["explanation"]}
        except Exception as e:  # noqa: BLE001
            counter["errors"] += 1
            print(f"[enrich] row {key[:8]} failed: {e}")

    # Synth questions: pick N random articles (use first question as title signal)
    if synth:
        seen_titles: set[str] = set()
        candidates = []
        for r in rows:
            t = r.get("title", "").strip()
            if t and t not in seen_titles and len(r.get("content", "")) > 300:
                candidates.append(r)
                seen_titles.add(t)
        random.shuffle(candidates)
        for r in candidates[:synth_max]:
            try:
                msgs = synth_q_prompt(r["title"], r["content"])
                content = call_chat(client, msgs, model=model_name, temperature=0.8, max_tokens=800)
                parsed = _extract_json(content)
                if parsed and "question" in parsed and "choices" in parsed and "correct_answer" in parsed:
                    label = str(parsed["correct_answer"]).strip().upper()[:1]
                    if label not in {"A", "B", "C", "D"}:
                        continue
                    new_row = {
                        "messages": [
                            r["messages"][0],
                            {"role": "user", "content": _user_text_from_parsed(parsed, r)},
                            {"role": "assistant", "content": label},
                        ],
                        "label": label,
                        "_source": "synth",
                    }
                    written.append(new_row)
                    counter["synth"] += 1
            except Exception as e:  # noqa: BLE001
                counter["errors"] += 1
                print(f"[enrich] synth failed: {e}")

    # Persist cache
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    # Append to original
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in written:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[enrich] done. counters={counter} out={out_path}")
    return counter


def _user_text_from_parsed(parsed: dict, original: dict) -> str:
    """Build a user-prompt string from a parsed paraphrase/synth dict, in the
    same format used by `prompts.build_user_instruction`."""
    from .prompts import build_user_instruction

    return build_user_instruction(
        title=original.get("title", ""),
        content=original.get("content", ""),
        question=parsed["question"],
        choices=parsed["choices"],
    )
