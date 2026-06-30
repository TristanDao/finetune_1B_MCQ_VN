"""Prompt construction (training + evaluation)."""
from __future__ import annotations

from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "Bạn là hệ thống trả lời trắc nghiệm. Chỉ xuất duy nhất 1 ký tự A/B/C/D."
)


def format_choices(choices: dict[str, str]) -> str:
    """Render choices deterministically A-D. Skip missing keys, append extras."""
    ordered: list[str] = []
    for k in ["A", "B", "C", "D"]:
        if k in choices and choices[k] is not None:
            ordered.append(f"{k}: {choices[k]}")
    if len(ordered) < len(choices):
        for k, v in choices.items():
            if k not in ("A", "B", "C", "D") and v is not None:
                ordered.append(f"{k}: {v}")
    return "\n".join(ordered)


def build_user_instruction(
    title: str,
    content: str,
    question: str,
    choices: dict[str, str],
) -> str:
    """Build the user-message text for a single MCQ item.

    Vietnamese prompt; deterministic A-D ordering; instructs the model to output
    exactly one letter, no explanation.
    """
    choices_text = format_choices(choices)
    return (
        "Bạn là hệ thống trả lời trắc nghiệm. Hãy đọc văn bản và câu hỏi, "
        "chỉ chọn **một đáp án duy nhất** từ A/B/C/D, không giải thích, không thêm nội dung khác.\n\n"
        "Văn bản:\n"
        f"Tiêu đề: {title}\n\n"
        f"Nội dung: {content}\n\n"
        "Câu hỏi:\n"
        f"{question}\n\n"
        "Các lựa chọn:\n"
        f"{choices_text}\n\n"
        "Chỉ trả lời đúng 1 ký tự: A, B, C hoặc D."
    )


def build_chat_messages(
    system_prompt: str,
    user_instruction: str,
    assistant: str | None = None,
) -> list[dict[str, str]]:
    """Return OpenAI-style messages list."""
    msgs: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_instruction},
    ]
    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
    return msgs


def make_row(
    *,
    title: str,
    content: str,
    question: str,
    choices: dict[str, str],
    label: str | None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict[str, Any]:
    """Build a single training/eval row in our standard schema.

    Schema (JSONL):
        {
          "messages": [system, user, assistant?],
          "label": "A"|"B"|"C"|"D",   # for eval only
          "row_id": "..."              # optional, for submission
        }
    """
    user = build_user_instruction(title, content, question, choices)
    assistant = label.strip().upper()[:1] if label is not None else None
    msgs = build_chat_messages(system_prompt, user, assistant=assistant)
    row: dict[str, Any] = {"messages": msgs}
    if label is not None:
        row["label"] = label.strip().upper()[:1]
    return row
