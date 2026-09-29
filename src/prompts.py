"""Load prompt templates from prompts/*.txt and fill in their placeholders.

Placeholders look like `{question}`. We substitute them with plain string
replacement (not str.format) so that braces inside retrieved Wikipedia text
or inside the JSON examples of the judge prompts cannot break formatting.
"""
from __future__ import annotations

import re
from typing import Any

from src.config import resolve_path

PLACEHOLDER_RE = re.compile(r"\{([a-z_]+)\}")
REQUIRED_PLACEHOLDERS = {
    "system": set(),
    "no_rag": {"question"},
    "rag": {"question", "context"},
    "judge_answer": {"question", "best_answer", "correct_answers", "incorrect_answers", "answer"},
    "judge_context": {"question", "best_answer", "correct_answers", "context"},
}


class PromptError(ValueError):
    pass


def load_template(cfg: dict[str, Any], name: str) -> str:
    path = resolve_path(cfg["prompts"][name])
    text = path.read_text(encoding="utf-8").strip()
    missing = {p for p in REQUIRED_PLACEHOLDERS.get(name, set()) if "{" + p + "}" not in text}
    if missing:
        raise PromptError(f"Prompt template {path} is missing placeholders: {sorted(missing)}")
    return text


def fill(template: str, **values: str) -> str:
    """Replace `{name}` placeholders; every known placeholder must be supplied."""
    needed = {m for m in PLACEHOLDER_RE.findall(template) if m in _all_placeholders()}
    missing = needed - values.keys()
    if missing:
        raise PromptError(f"No value supplied for placeholders: {sorted(missing)}")
    out = template
    for key, value in values.items():
        out = out.replace("{" + key + "}", str(value))
    return out


def _all_placeholders() -> set[str]:
    return set().union(*REQUIRED_PLACEHOLDERS.values())


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Numbered context block shown to the LLM in the RAG condition."""
    if not chunks:
        raise PromptError("Cannot build a RAG prompt from an empty list of retrieved chunks")
    parts = []
    for c in chunks:
        header = f"[{c['rank']}] {c['title']}"
        if c.get("section") and c["section"] != "Introduction":
            header += f" — {c['section']}"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n".join(parts)


def format_answer_list(answers: list[str]) -> str:
    return "\n".join(f"- {a}" for a in answers) if answers else "- (none)"


def build_no_rag_prompt(template: str, question: str) -> str:
    return fill(template, question=question)


def build_rag_prompt(template: str, question: str, chunks: list[dict[str, Any]]) -> str:
    return fill(template, question=question, context=format_context(chunks))
