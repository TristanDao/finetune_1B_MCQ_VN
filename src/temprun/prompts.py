"""Prompt construction: two training modes — conversation (short answer) and
Chain-of-Thought (with explanation step-by-step)."""

from __future__ import annotations

from typing import Any, Literal

Mode = Literal["conversation", "cot"]

SYSTEM_PROMPT_CONVERSATION = (
    "Bạn là hệ thống trả lời trắc nghiệm. Chỉ xuất duy nhất 1 ký tự A/B/C/D."
)

SYSTEM_PROMPT_COT = (
    "Bạn là một chuyên gia giải quyết vấn đề xuất sắc. Nhiệm vụ của bạn là đọc kỹ văn bản, "
    "câu hỏi và các lựa chọn, suy luận từng bước dựa trên thông tin trong bài. "
    "Cuối cùng, bắt buộc chốt lại bằng định dạng chính xác: 'Do đó, đáp án đúng là [A/B/C/D]'."
)

ASSISTANT_TEMPLATE_COT = "{explanation}\n\nDo đó, đáp án đúng là {label}."


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
    *,
    mode: Mode = "conversation",
) -> str:
    """Build the user-message text for a single MCQ item.

    ``conversation`` mode: plain text, short prompt, direct answer.
    ``cot`` mode: markdown-structured sections, Chain-of-Thought prompt.
    """
    choices_text = format_choices(choices)

    if mode == "cot":
        return (
            f"### VĂN BẢN\n"
            f"**Tiêu đề:** {title}\n"
            f"**Nội dung:**\n{content}\n\n"
            f"### CÂU HỎI\n{question}\n\n"
            f"### LỰA CHỌN\n{choices_text}\n\n"
            f"Vui lòng cung cấp đáp án của bạn (A, B, C, hoặc D):"
        )
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


def build_assistant(label: str, *, explanation: str | None = None, mode: Mode = "conversation") -> str:
    """Build assistant response string.

    ``conversation`` mode: just the letter (e.g. "A").
    ``cot`` mode: explanation + "Do đó, đáp án đúng là {label}."
    """
    if mode == "cot":
        if explanation:
            return ASSISTANT_TEMPLATE_COT.format(explanation=explanation, label=label)
        return f"Do đó, đáp án đúng là {label}."
    return label


def build_messages(
    title: str,
    content: str,
    question: str,
    choices: dict[str, str],
    label: str,
    *,
    explanation: str | None = None,
    mode: Mode = "conversation",
) -> list[dict[str, str]]:
    """Build OpenAI-style [system, user, assistant] messages for one MCQ row."""
    system_prompt = SYSTEM_PROMPT_COT if mode == "cot" else SYSTEM_PROMPT_CONVERSATION
    user = build_user_instruction(title, content, question, choices, mode=mode)
    assistant = build_assistant(label, explanation=explanation, mode=mode)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def make_row(
    *,
    title: str,
    content: str,
    question: str,
    choices: dict[str, str],
    label: str,
    explanation: str | None = None,
    mode: Mode = "conversation",
) -> dict[str, Any]:
    """Build a single training/eval row with messages, label, and metadata."""
    msgs = build_messages(title, content, question, choices, label,
                          explanation=explanation, mode=mode)
    row: dict[str, Any] = {
        "messages": msgs,
        "title": title,
        "content": content,
        "question": question,
        "choices": choices,
        "label": label,
    }
    if explanation:
        row["explanation"] = explanation
    return row
