"""Data enrichment: gọi Alibaba DashScope (OpenAI-compatible) để paraphrase, sinh explanation,
và (tuỳ chọn) sinh câu hỏi mới.

Cần DASHSCOPE_API_KEY trong .env.
Model mặc định: qwen3-max-preview (override bằng DASHSCOPE_MODEL).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

from openai import AsyncOpenAI, OpenAI

from .utils import ensure_dir, load_env

VALID_LABELS = frozenset({"A", "B", "C", "D"})


def get_client() -> OpenAI:
    load_env()
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY missing. Fill .env first.")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_model() -> str:
    return (
        os.environ.get("DASHSCOPE_MODEL")
        or os.environ.get("DASHSCOPE_MODEL_BACKUP")
        or ""
    )


def get_model_source() -> str:
    """Which env var supplied the model name (for logging)."""
    if os.environ.get("DASHSCOPE_MODEL"):
        return "DASHSCOPE_MODEL"
    if os.environ.get("DASHSCOPE_MODEL_BACKUP"):
        return "DASHSCOPE_MODEL_BACKUP"
    return "unset"


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


async def _call_chat_async(
    client: AsyncOpenAI,
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    """Async version of `call_chat`. Same retry policy, awaits the API call."""
    last_err: Exception | None = None
    actual_model = model or get_model()
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(
                model=actual_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            sleep = backoff ** attempt
            print(f"[enrich] async attempt {attempt + 1} failed: {e}; sleeping {sleep:.1f}s")
            await asyncio.sleep(sleep)
    raise RuntimeError(f"DashScope async call failed after {retries} retries: {last_err}")


def _get_async_client() -> AsyncOpenAI:
    """Build an AsyncOpenAI client using the same env vars as the sync client."""
    load_env()
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    base_url = os.environ.get(
        "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY missing. Fill .env first.")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


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
# Class-balancing plan
# -----------------------------------------------------------------------------

def _balance_plan(
    rows: list[dict],
    target_per_class: int | None = None,
    min_ratio: float = 2.0,
    labels: frozenset[str] = VALID_LABELS,
) -> dict[str, Any]:
    """Compute a balancing plan for a SFT row set.

    Args:
        rows: input rows; each must have a `label` in {A,B,C,D}.
        target_per_class: desired final count per label. Defaults to the max
            current count.
        min_ratio: max/min ratio above which we consider the set imbalanced
            and recommend the balanced enrichment mode.

    Returns:
        {
          "current":       {label: count},      # per-label totals in `rows`
          "target":        int,                 # target per class
          "is_imbalanced": bool,                # current ratio > min_ratio
          "ratio":         float,               # max / min (0 if a class empty)
          "repeat_per_row":{label: int},        # paraphrases per existing row
          "synth_needed":  {label: int},        # leftover after paraphrase
        }

    `repeat_per_row[lbl]` is the number of *additional* paraphrases to run
    per row of that label, so the final total for the label is
    `(1 + repeat_per_row[lbl]) * current[lbl]`. It is at least 1 whenever the
    projected total is still below the target.
    """
    current: dict[str, int] = {k: 0 for k in labels}
    for r in rows:
        lbl = r.get("label")
        if lbl in current:
            current[lbl] += 1

    observed = [v for v in current.values() if v > 0]
    if observed:
        max_c = max(observed)
        min_c = min(observed)
        ratio = max_c / min_c if min_c > 0 else float("inf")
    else:
        max_c = 0
        min_c = 0
        ratio = 0.0

    target = target_per_class if target_per_class is not None else max_c
    is_imbalanced = ratio > min_ratio

    repeat_per_row: dict[str, int] = {}
    for lbl in labels:
        cur = current.get(lbl, 0)
        if cur <= 0 or target <= cur:
            repeat_per_row[lbl] = 0
        else:
            # ceil((target - cur) / cur)  →  enough paraphrases to reach target
            repeat_per_row[lbl] = -(-(target - cur) // cur)

    synth_needed: dict[str, int] = {}
    for lbl in labels:
        cur = current.get(lbl, 0)
        rep = repeat_per_row.get(lbl, 0)
        projected = cur * (1 + rep)
        synth_needed[lbl] = max(0, target - projected)

    return {
        "current": current,
        "target": target,
        "is_imbalanced": is_imbalanced,
        "ratio": ratio,
        "repeat_per_row": repeat_per_row,
        "synth_needed": synth_needed,
    }


# -----------------------------------------------------------------------------
# Per-call helpers (sync + async mirrors)
# -----------------------------------------------------------------------------

def _paraphrase_once(client: OpenAI, model_name: str, r: dict) -> dict | None:
    """One paraphrase call. Returns the new row or None on any failure."""
    msgs = paraphrase_prompt(r["question"], r["choices"], r["label"])
    content = call_chat(client, msgs, model=model_name, temperature=0.8, max_tokens=800)
    parsed = _extract_json(content)
    if not (parsed and "question" in parsed and "choices" in parsed and "correct_answer" in parsed):
        return None
    label = str(parsed["correct_answer"]).strip().upper()[:1]
    if label not in VALID_LABELS:
        return None
    return {
        "messages": [
            r["messages"][0],
            {"role": "user", "content": _user_text_from_parsed(parsed, r)},
            {"role": "assistant", "content": label},
        ],
        "label": label,
        "_source": "paraphrase",
    }


def _synth_once(client: OpenAI, model_name: str, r: dict) -> dict | None:
    """One synth call. Returns the new row or None on any failure."""
    msgs = synth_q_prompt(r["title"], r["content"])
    content = call_chat(client, msgs, model=model_name, temperature=0.8, max_tokens=800)
    parsed = _extract_json(content)
    if not (parsed and "question" in parsed and "choices" in parsed and "correct_answer" in parsed):
        return None
    label = str(parsed["correct_answer"]).strip().upper()[:1]
    if label not in VALID_LABELS:
        return None
    return {
        "messages": [
            r["messages"][0],
            {"role": "user", "content": _user_text_from_parsed(parsed, r)},
            {"role": "assistant", "content": label},
        ],
        "label": label,
        "_source": "synth",
    }


async def _paraphrase_once_async(
    client: AsyncOpenAI, model_name: str, r: dict
) -> dict | None:
    """Async version of `_paraphrase_once`."""
    msgs = paraphrase_prompt(r["question"], r["choices"], r["label"])
    content = await _call_chat_async(
        client, msgs, model=model_name, temperature=0.8, max_tokens=800
    )
    parsed = _extract_json(content)
    if not (parsed and "question" in parsed and "choices" in parsed and "correct_answer" in parsed):
        return None
    label = str(parsed["correct_answer"]).strip().upper()[:1]
    if label not in VALID_LABELS:
        return None
    return {
        "messages": [
            r["messages"][0],
            {"role": "user", "content": _user_text_from_parsed(parsed, r)},
            {"role": "assistant", "content": label},
        ],
        "label": label,
        "_source": "paraphrase",
    }


async def _synth_once_async(
    client: AsyncOpenAI, model_name: str, r: dict
) -> dict | None:
    """Async version of `_synth_once`."""
    msgs = synth_q_prompt(r["title"], r["content"])
    content = await _call_chat_async(
        client, msgs, model=model_name, temperature=0.8, max_tokens=800
    )
    parsed = _extract_json(content)
    if not (parsed and "question" in parsed and "choices" in parsed and "correct_answer" in parsed):
        return None
    label = str(parsed["correct_answer"]).strip().upper()[:1]
    if label not in VALID_LABELS:
        return None
    return {
        "messages": [
            r["messages"][0],
            {"role": "user", "content": _user_text_from_parsed(parsed, r)},
            {"role": "assistant", "content": label},
        ],
        "label": label,
        "_source": "synth",
    }


async def _explain_once_async(
    client: AsyncOpenAI, model_name: str, r: dict
) -> str | None:
    """Async version of the explain call. Returns the explanation string or None."""
    msgs = explain_prompt(
        r["title"], r["content"], r["question"], r["choices"], r["label"]
    )
    content = await _call_chat_async(
        client, msgs, model=model_name, temperature=0.3, max_tokens=400
    )
    parsed = _extract_json(content)
    if parsed and "explanation" in parsed:
        return parsed["explanation"]
    return None


def _paraphrase_key(r: dict, n: int, balanced: bool) -> str:
    """Cache key for the n-th paraphrase attempt of a row.

    n=0 always uses the bare `(q, label)` key so legacy caches stay valid
    and so that n=0 shares the slot with the explain cache.
    """
    if not balanced or n == 0:
        return _hash(r["question"] + "|" + r["label"])
    return _hash(r["question"] + "|" + r["label"] + f"|n={n}")


def _synth_candidates(rows: Iterable[dict]) -> list[dict]:
    """Pick unique-title articles with non-empty content for synth candidates."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        t = r.get("title", "").strip()
        if t and t not in seen and len(r.get("content", "")) > 300:
            out.append(r)
            seen.add(t)
    return out


# -----------------------------------------------------------------------------
# Main entry — dispatcher (sync or async) and per-mode implementations
# -----------------------------------------------------------------------------

def _new_counter() -> dict[str, int]:
    return {
        "paraphrase": 0,
        "explain": 0,
        "synth": 0,
        "skipped": 0,
        "errors": 0,
        "paraphrase_attempts": 0,
        "synth_attempts": 0,
    }


def _log_progress(
    phase: str,
    done: int,
    total: int,
    counter: dict[str, int],
    *,
    every: int = 25,
) -> None:
    """Print a one-line heartbeat when `done` hits the heartbeat boundary
    or when the phase finishes. Cheap when `every > total` (e.g. tiny inputs)."""
    if done == 0:
        return
    if done == total or done % every == 0:
        print(
            f"[enrich] {phase}: {done}/{total} | "
            f"paraphrase={counter['paraphrase']} "
            f"explain={counter['explain']} "
            f"synth={counter['synth']} "
            f"errors={counter['errors']}"
        )


def _iter_eligible_rows(rows: Iterable[dict], counter: dict[str, int]) -> Iterable[dict]:
    """Yield rows that have all required fields and a valid label, counting
    the rest in `counter['skipped']`."""
    for r in rows:
        if not all(k in r for k in ("question", "choices", "label", "title", "content")):
            counter["skipped"] += 1
            continue
        if r["label"] not in VALID_LABELS:
            counter["skipped"] += 1
            continue
        yield r


def _run_passes_sync(
    rows: list[dict],
    cache: dict[str, dict],
    plan: dict[str, Any],
    effective_balance: bool,
    *,
    paraphrase: bool,
    explain: bool,
    synth: bool,
    synth_retry: int,
    synth_max: int,
    model_name: str,
    progress_every: int = 25,
) -> tuple[list[dict], dict[str, int], dict[str, dict]]:
    """Sequential (concurrency=1) implementation of the two passes."""
    client = get_client()
    counter = _new_counter()
    written: list[dict] = []

    # Pre-compute plan totals so we can show N/M progress without iterating twice.
    eligible = list(_iter_eligible_rows(rows, counter))
    paraphrase_total = sum(
        1 + (plan["repeat_per_row"].get(r["label"], 0) if effective_balance else 0)
        for r in eligible
    )
    explain_total = len(eligible) if explain else 0

    if paraphrase and paraphrase_total:
        print(f"[enrich] paraphrase: starting {paraphrase_total} jobs (sequential)")
    if explain:
        print(f"[enrich] explain: starting {explain_total} jobs (sequential)")

    # ---- Pass 1: paraphrase + explain ----
    paraphrase_done = 0
    explain_done = 0
    for r in eligible:
        repeat = plan["repeat_per_row"].get(r["label"], 0) if effective_balance else 0
        n_total = 1 + repeat
        for n in range(n_total):
            p_key = _paraphrase_key(r, n, effective_balance)
            entry = cache.get(p_key, {})
            if entry.get("paraphrase"):
                paraphrase_done += 1
                continue
            if paraphrase:
                counter["paraphrase_attempts"] += 1
                try:
                    new_row = _paraphrase_once(client, model_name, r)
                except Exception as e:  # noqa: BLE001
                    counter["errors"] += 1
                    print(f"[enrich] paraphrase {p_key[:8]}/n={n} failed: {e}")
                    continue
                if new_row is None:
                    continue
                written.append(new_row)
                counter["paraphrase"] += 1
            paraphrase_done += 1
            cache[p_key] = {**entry, "paraphrase": True}
        if paraphrase_total:
            _log_progress(
                "paraphrase", paraphrase_done, paraphrase_total, counter, every=progress_every
            )

        if explain:
            e_key = _hash(r["question"] + "|" + r["label"])
            entry = cache.get(e_key, {})
            if entry.get("explanation"):
                explain_done += 1
                _log_progress(
                    "explain", explain_done, explain_total, counter, every=progress_every
                )
                continue
            try:
                msgs = explain_prompt(
                    r["title"], r["content"], r["question"], r["choices"], r["label"]
                )
                content = call_chat(
                    client, msgs, model=model_name, temperature=0.3, max_tokens=400
                )
                parsed = _extract_json(content)
                if parsed and "explanation" in parsed:
                    counter["explain"] += 1
                    cache[e_key] = {**entry, "explanation": parsed["explanation"]}
            except Exception as e:  # noqa: BLE001
                counter["errors"] += 1
                print(f"[enrich] explain {e_key[:8]} failed: {e}")
            explain_done += 1
            _log_progress(
                "explain", explain_done, explain_total, counter, every=progress_every
            )

    # ---- Pass 2: synth ----
    if synth:
        candidates = _synth_candidates(rows)
        random.shuffle(candidates)
        if effective_balance:
            # Targeted: keep rolling until each minority class reaches its
            # remaining target. Re-roll a single synth call up to
            # `synth_retry` times per article before moving on; stop entirely
            # once every `needs` entry has been satisfied.
            needs: dict[str, int] = {
                lbl: n for lbl, n in plan["synth_needed"].items() if n > 0
            }
            synth_total = sum(needs.values())
            if synth_total:
                print(f"[enrich] synth: starting up to {synth_total} targeted jobs (sequential)")
            synth_done = 0
            for r in candidates:
                if not needs:
                    break
                for lbl, remaining in list(needs.items()):
                    if remaining <= 0:
                        continue
                    for attempt in range(synth_retry):
                        s_key = _hash(
                            r["title"] + "|synth|target=" + lbl + f"|n={attempt}"
                        )
                        entry = cache.get(s_key, {})
                        if f"synth:{lbl}:{attempt}" in entry:
                            cached_lbl = entry[f"synth:{lbl}:{attempt}"]
                            if cached_lbl == lbl:
                                needs[lbl] -= 1
                            break  # slot is decided
                        counter["synth_attempts"] += 1
                        try:
                            new_row = _synth_once(client, model_name, r)
                        except Exception as e:  # noqa: BLE001
                            counter["errors"] += 1
                            print(f"[enrich] synth->{lbl} failed: {e}")
                            continue
                        if new_row is None:
                            continue
                        actual = new_row["label"]
                        cache[s_key] = {**entry, f"synth:{lbl}:{attempt}": actual}
                        if actual == lbl:
                            written.append(new_row)
                            counter["synth"] += 1
                            needs[lbl] -= 1
                        break  # one attempt per slot, even on miss
                    synth_done += 1
                    _log_progress(
                        "synth", synth_done, synth_total, counter, every=progress_every
                    )
        else:
            synth_total = min(synth_max, len(candidates))
            if synth_total:
                print(f"[enrich] synth: starting {synth_total} jobs (sequential)")
            for i, r in enumerate(candidates[:synth_max], start=1):
                s_key = _hash(r["title"] + "|synth|generic")
                entry = cache.get(s_key, {})
                if entry.get("synth"):
                    _log_progress(
                        "synth", i, synth_total, counter, every=progress_every
                    )
                    continue
                counter["synth_attempts"] += 1
                try:
                    new_row = _synth_once(client, model_name, r)
                except Exception as e:  # noqa: BLE001
                    counter["errors"] += 1
                    print(f"[enrich] synth failed: {e}")
                    continue
                if new_row is None:
                    continue
                written.append(new_row)
                counter["synth"] += 1
                cache[s_key] = {**entry, "synth": True}
                _log_progress(
                    "synth", i, synth_total, counter, every=progress_every
                )

    return written, counter, cache


async def _run_passes_async(
    rows: list[dict],
    cache: dict[str, dict],
    plan: dict[str, Any],
    effective_balance: bool,
    *,
    paraphrase: bool,
    explain: bool,
    synth: bool,
    synth_retry: int,
    synth_max: int,
    model_name: str,
    concurrency: int,
    progress_every: int = 25,
) -> tuple[list[dict], dict[str, int], dict[str, dict]]:
    """Concurrent implementation: collects jobs in the main coroutine, then
    fires them through an `asyncio.Semaphore(concurrency)` so at most
    `concurrency` requests are in flight. Cache is only mutated by the
    caller (after gather returns), so no locks are needed.
    """
    counter = _new_counter()
    written: list[dict] = []
    sem = asyncio.Semaphore(concurrency)
    client = _get_async_client()

    async with client:

        # ---- Pass 1a: collect paraphrase jobs ----
        paraphrase_jobs: list[tuple[dict, int, str]] = []  # (row, n, p_key)
        for r in _iter_eligible_rows(rows, counter):
            repeat = plan["repeat_per_row"].get(r["label"], 0) if effective_balance else 0
            n_total = 1 + repeat
            for n in range(n_total):
                p_key = _paraphrase_key(r, n, effective_balance)
                if cache.get(p_key, {}).get("paraphrase"):
                    continue
                paraphrase_jobs.append((r, n, p_key))

        async def _run_paraphrase(r: dict, n: int, p_key: str) -> tuple[str, int, str, Any]:
            async with sem:
                counter["paraphrase_attempts"] += 1
                try:
                    new_row = await _paraphrase_once_async(client, model_name, r)
                    return ("ok", n, p_key, new_row)
                except Exception as e:  # noqa: BLE001
                    return ("err", n, p_key, str(e))

        if paraphrase and paraphrase_jobs:
            print(
                f"[enrich] paraphrase: starting {len(paraphrase_jobs)} jobs "
                f"(concurrency={concurrency})"
            )
            t0 = time.monotonic()
            results = await asyncio.gather(
                *(_run_paraphrase(r, n, k) for r, n, k in paraphrase_jobs)
            )
            print(
                f"[enrich] paraphrase: done in {time.monotonic() - t0:.1f}s | "
                f"counters={counter}"
            )
            for status, n, p_key, payload in results:
                if status == "err":
                    counter["errors"] += 1
                    print(f"[enrich] paraphrase {p_key[:8]}/n={n} failed: {payload}")
                    continue
                cache[p_key] = {**cache.get(p_key, {}), "paraphrase": True}
                new_row = payload
                if new_row is not None:
                    written.append(new_row)
                    counter["paraphrase"] += 1
        else:
            for _, _, p_key in paraphrase_jobs:
                cache[p_key] = {**cache.get(p_key, {}), "paraphrase": True}

        # ---- Pass 1b: collect + run explain jobs ----
        explain_jobs: list[tuple[dict, str]] = []
        if explain:
            for r in _iter_eligible_rows(rows, counter):
                e_key = _hash(r["question"] + "|" + r["label"])
                if cache.get(e_key, {}).get("explanation"):
                    continue
                explain_jobs.append((r, e_key))

        async def _run_explain(r: dict, e_key: str) -> tuple[str, str, Any]:
            async with sem:
                try:
                    explanation = await _explain_once_async(client, model_name, r)
                    return ("ok", e_key, explanation)
                except Exception as e:  # noqa: BLE001
                    return ("err", e_key, str(e))

        if explain and explain_jobs:
            print(
                f"[enrich] explain: starting {len(explain_jobs)} jobs "
                f"(concurrency={concurrency})"
            )
            t0 = time.monotonic()
            results = await asyncio.gather(*(_run_explain(r, k) for r, k in explain_jobs))
            print(
                f"[enrich] explain: done in {time.monotonic() - t0:.1f}s | "
                f"counters={counter}"
            )
            for status, e_key, payload in results:
                if status == "err":
                    counter["errors"] += 1
                    print(f"[enrich] explain {e_key[:8]} failed: {payload}")
                    continue
                explanation = payload
                if explanation is not None:
                    counter["explain"] += 1
                    cache[e_key] = {**cache.get(e_key, {}), "explanation": explanation}

        # ---- Pass 2: synth ----
        if synth:
            candidates = _synth_candidates(rows)
            random.shuffle(candidates)

            if effective_balance:
                # Targeted synth: keep sequential — early-stop on `needs`
                # depends on the label returned by the previous call.
                needs: dict[str, int] = {
                    lbl: n for lbl, n in plan["synth_needed"].items() if n > 0
                }
                synth_total = sum(needs.values())
                if synth_total:
                    print(
                        f"[enrich] synth: starting up to {synth_total} targeted jobs (sequential)"
                    )
                synth_done = 0
                t0 = time.monotonic()
                for r in candidates:
                    if not needs:
                        break
                    for lbl, remaining in list(needs.items()):
                        if remaining <= 0:
                            continue
                        for attempt in range(synth_retry):
                            s_key = _hash(
                                r["title"] + "|synth|target=" + lbl + f"|n={attempt}"
                            )
                            entry = cache.get(s_key, {})
                            if f"synth:{lbl}:{attempt}" in entry:
                                cached_lbl = entry[f"synth:{lbl}:{attempt}"]
                                if cached_lbl == lbl:
                                    needs[lbl] -= 1
                                break
                            counter["synth_attempts"] += 1
                            try:
                                new_row = await _synth_once_async(client, model_name, r)
                            except Exception as e:  # noqa: BLE001
                                counter["errors"] += 1
                                print(f"[enrich] synth->{lbl} failed: {e}")
                                continue
                            if new_row is None:
                                continue
                            actual = new_row["label"]
                            cache[s_key] = {**entry, f"synth:{lbl}:{attempt}": actual}
                            if actual == lbl:
                                written.append(new_row)
                                counter["synth"] += 1
                                needs[lbl] -= 1
                            break
                        synth_done += 1
                        _log_progress(
                            "synth", synth_done, synth_total, counter, every=progress_every
                        )
                if synth_total:
                    print(
                        f"[enrich] synth: done in {time.monotonic() - t0:.1f}s | "
                        f"counters={counter}"
                    )
            else:
                synth_jobs: list[tuple[dict, str]] = []
                for r in candidates[:synth_max]:
                    s_key = _hash(r["title"] + "|synth|generic")
                    if cache.get(s_key, {}).get("synth"):
                        continue
                    synth_jobs.append((r, s_key))

                async def _run_synth(r: dict, s_key: str) -> tuple[str, str, Any]:
                    async with sem:
                        counter["synth_attempts"] += 1
                        try:
                            new_row = await _synth_once_async(client, model_name, r)
                            return ("ok", s_key, new_row)
                        except Exception as e:  # noqa: BLE001
                            return ("err", s_key, str(e))

                if synth_jobs:
                    print(
                        f"[enrich] synth: starting {len(synth_jobs)} jobs "
                        f"(concurrency={concurrency})"
                    )
                    t0 = time.monotonic()
                    results = await asyncio.gather(
                        *(_run_synth(r, k) for r, k in synth_jobs)
                    )
                    print(
                        f"[enrich] synth: done in {time.monotonic() - t0:.1f}s | "
                        f"counters={counter}"
                    )
                    for status, s_key, payload in results:
                        if status == "err":
                            counter["errors"] += 1
                            print(f"[enrich] synth failed: {payload}")
                            continue
                        cache[s_key] = {**cache.get(s_key, {}), "synth": True}
                        new_row = payload
                        if new_row is not None:
                            written.append(new_row)
                            counter["synth"] += 1

    return written, counter, cache


def enrich_dataset(
    input_jsonl: str | os.PathLike,
    output_jsonl: str | os.PathLike,
    *,
    paraphrase: bool = True,
    explain: bool = True,
    synth: bool = False,
    synth_max: int = 200,
    seed: int = 3407,
    balance: bool | None = None,
    target_per_class: int | None = None,
    synth_retry: int = 3,
    concurrency: int = 10,
    progress_every: int = 25,
) -> dict[str, int]:
    """Enrich a SFT jsonl by adding new rows. Idempotent via content-hash cache.

    Output schema (slim): each row has only `messages` (the chat-formatted
    training text, embedding title/content/question/choices in the user
    prompt), `label` (top-level mirror of the assistant letter), and
    optional `_source` (`"paraphrase"` | `"synth"`). The full top-level
    `title`/`content`/`question`/`choices`/`row_id` from the input rows
    are dropped on write — `train.py` and `scripts/evaluate.py` only read
    `messages` and `label`, and `infer.py` reads the test set from
    `data/raw/test/` (never from enriched data).

    Args:
        balance: None → auto-enable when max/min label ratio > 2.0 (see
            `_balance_plan`); True → force on; False → off. When on, paraphrase
            counts and synth targets are class-conditional so the output
            distribution is balanced.
        target_per_class: target count per label. Defaults to the max current
            count (i.e. "match max"). Ignored when balance is off.
        synth_retry: when synth is targeted at a minority class, retry up to
            this many times per article before giving up on that label.
        concurrency: max in-flight API calls when running async. `1` runs the
            sequential path; `>1` uses `AsyncOpenAI` + `asyncio.Semaphore`.
            If the async path raises, we automatically fall back to the
            sequential path. Defaults to 10.
        progress_every: in sync mode, print a heartbeat every N items processed
            per phase (paraphrase, explain, synth). Async mode logs a single
            start/done line per phase instead.
    """
    from .data import read_jsonl

    random.seed(seed)
    in_path = Path(input_jsonl)
    out_path = Path(output_jsonl)
    ensure_dir(out_path.parent)

    rows = read_jsonl(in_path)
    print(f"[enrich] loaded {len(rows)} rows from {in_path}")

    model_name = get_model()
    print(f"[enrich] model: {model_name!r}  (source={get_model_source()})")

    plan = _balance_plan(rows, target_per_class=target_per_class)
    effective_balance = plan["is_imbalanced"] if balance is None else balance
    print(
        f"[enrich] balance plan: current={plan['current']} target={plan['target']} "
        f"ratio={plan['ratio']:.2f} imbalanced={plan['is_imbalanced']} "
        f"effective_balance={effective_balance} concurrency={concurrency} "
        f"repeat={plan['repeat_per_row']} synth_needed={plan['synth_needed']}"
    )

    cache_path = out_path.with_suffix(out_path.suffix + ".cache.json")
    cache: dict[str, dict] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    use_async = concurrency and concurrency > 1
    if use_async:
        try:
            written, counter, cache = asyncio.run(
                _run_passes_async(
                    rows=rows,
                    cache=cache,
                    plan=plan,
                    effective_balance=effective_balance,
                    paraphrase=paraphrase,
                    explain=explain,
                    synth=synth,
                    synth_retry=synth_retry,
                    synth_max=synth_max,
                    model_name=model_name,
                    concurrency=concurrency,
                    progress_every=progress_every,
                )
            )
        except Exception as e:  # noqa: BLE001
            print(f"[enrich] async path failed ({e}); falling back to sync")
            written, counter, cache = _run_passes_sync(
                rows=rows,
                cache=cache,
                plan=plan,
                effective_balance=effective_balance,
                paraphrase=paraphrase,
                explain=explain,
                synth=synth,
                synth_retry=synth_retry,
                synth_max=synth_max,
                model_name=model_name,
                progress_every=progress_every,
            )
    else:
        written, counter, cache = _run_passes_sync(
            rows=rows,
            cache=cache,
            plan=plan,
            effective_balance=effective_balance,
            paraphrase=paraphrase,
            explain=explain,
            synth=synth,
            synth_retry=synth_retry,
            synth_max=synth_max,
            model_name=model_name,
            progress_every=progress_every,
        )

    # Persist cache
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    # Append to original (slim schema: messages + label + optional _source).
    # `train.py` and `scripts/evaluate.py` only read `messages` (+ `label`),
    # so we drop the redundant top-level title/content/question/choices/row_id
    # that came from `train.jsonl`. See `_slim_row` for the schema.
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(_slim_row(r), ensure_ascii=False) + "\n")
        for r in written:
            f.write(json.dumps(_slim_row(r), ensure_ascii=False) + "\n")

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


def _slim_row(r: dict) -> dict:
    """Reduce a row to the minimal schema used downstream by train/eval:
    `messages` (already embeds title/content/question/choices in the user
    prompt), `label` (top-level mirror of `messages[-1]['content']`), and
    optional `_source` (debug metadata: "paraphrase" | "synth").

    All other fields (`title`, `content`, `question`, `choices`, `row_id`)
    are dropped — they are not read by `train.py` or `scripts/evaluate.py`,
    and the test set is loaded from `data/raw/test/` by `infer.py`, never
    from enriched data.
    """
    slim: dict = {
        "messages": r["messages"],
        "label": r.get("label", ""),
    }
    if "_source" in r:
        slim["_source"] = r["_source"]
    return slim
