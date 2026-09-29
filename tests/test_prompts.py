import pytest

from src.config import load_config
from src.prompts import (PromptError, build_no_rag_prompt, build_rag_prompt, fill,
                         format_context, load_template)

CHUNKS = [
    {"rank": 1, "title": "Canberra", "section": "Introduction", "text": "Canberra is the capital {of} Australia."},
    {"rank": 2, "title": "Sydney", "section": "History", "text": "Sydney is the largest city."},
]


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_templates_have_placeholders(cfg):
    for name in ("system", "no_rag", "rag", "judge_answer", "judge_context"):
        assert load_template(cfg, name)


def test_no_rag_prompt(cfg):
    p = build_no_rag_prompt(load_template(cfg, "no_rag"), "What is the capital of Australia?")
    assert "What is the capital of Australia?" in p
    assert "{question}" not in p and "Context" not in p


def test_rag_prompt_contains_context_and_question(cfg):
    p = build_rag_prompt(load_template(cfg, "rag"), "What is the capital of Australia?", CHUNKS)
    assert "[1] Canberra\nCanberra is the capital {of} Australia." in p  # braces survive
    assert "[2] Sydney — History" in p
    assert p.rstrip().endswith("Answer:")
    assert "{context}" not in p


def test_conditions_differ_only_by_context(cfg):
    """Both prompts start with the same instruction and end with the same question block."""
    q = "Why do veins appear blue?"
    no_rag = build_no_rag_prompt(load_template(cfg, "no_rag"), q)
    rag = build_rag_prompt(load_template(cfg, "rag"), q, CHUNKS)
    assert rag.startswith("Answer the following factual question.")
    assert no_rag.startswith("Answer the following factual question.")
    assert rag.endswith(f"Question: {q}\n\nAnswer:") and no_rag.endswith(f"Question: {q}\n\nAnswer:")


def test_empty_context_rejected():
    with pytest.raises(PromptError):
        format_context([])


def test_fill_requires_all_values():
    with pytest.raises(PromptError):
        fill("Q: {question} C: {context}", question="x")
    assert fill("Q: {question}", question="x {y}") == "Q: x {y}"
