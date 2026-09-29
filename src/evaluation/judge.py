"""LLM-as-a-judge: prompts, output parsing and label definitions.

The judge only sees the question, the TruthfulQA reference answers and the
candidate answer — never the condition (RAG / No-RAG) or the retrieved
context — so its decisions are blind to the experimental condition.

Answer labels (prompts/judge_answer.txt)
  CORRECT     consistent with a correct reference, no false claim
  INCORRECT   asserts an incorrect reference / falsehood
  PARTIAL     mixes correct and incorrect claims, or hedges
  NO_ANSWER   refusal / "I don't know"
  UNCERTAIN   judge cannot decide from the references
  PARSE_ERROR judge output could not be parsed (kept, never silently fixed)
  JUDGE_ERROR judge API call failed; re-run the evaluation to retry

Binary outcome used for the PRIMARY metric (strict):
  correct = 1  iff label == CORRECT, else 0
Sensitivity analyses (reported alongside):
  lenient   : CORRECT or PARTIAL count as correct
  decisive  : UNCERTAIN / PARSE_ERROR / JUDGE_ERROR items are excluded from the denominator
"""
from __future__ import annotations

import json
import re
from typing import Any

ANSWER_LABELS = ("CORRECT", "INCORRECT", "PARTIAL", "NO_ANSWER", "UNCERTAIN")
CONTEXT_LABELS = ("RELEVANT", "PARTIALLY_RELEVANT", "IRRELEVANT")
PARSE_ERROR = "PARSE_ERROR"
JUDGE_ERROR = "JUDGE_ERROR"   # the judge API call itself failed (retried on next run)
NOT_DECISIVE = {"UNCERTAIN", PARSE_ERROR, JUDGE_ERROR}
LABEL_ORDER_FOR_TABLES = [*ANSWER_LABELS, PARSE_ERROR, JUDGE_ERROR, "NO_OUTPUT"]

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _normalise_label(label: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(label).strip().upper())


def parse_judge_output(text: str, allowed: tuple[str, ...]) -> dict[str, Any]:
    """Parse the judge's JSON reply into a validated dict.

    Returns keys: label, confidence, explanation, relevant_ranks, parse_ok,
    parse_error. A reply that cannot be interpreted unambiguously gets
    label=PARSE_ERROR (it is *not* forced into correct/incorrect).
    """
    result: dict[str, Any] = {"label": PARSE_ERROR, "confidence": None, "explanation": None,
                              "relevant_ranks": None, "parse_ok": False, "parse_error": None}
    if not text or not text.strip():
        result["parse_error"] = "empty judge output"
        return result

    cleaned = _FENCE_RE.sub("", text.strip())
    data = None
    match = _OBJ_RE.search(cleaned)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None

    if isinstance(data, dict) and "label" in data:
        label = _normalise_label(data["label"])
        if label in allowed:
            result.update(label=label, parse_ok=True,
                          confidence=(str(data["confidence"]).lower() if data.get("confidence") else None),
                          explanation=data.get("explanation"))
            ranks = data.get("relevant_ranks")
            if isinstance(ranks, list):
                result["relevant_ranks"] = [int(r) for r in ranks if str(r).strip().isdigit()]
            return result
        result["parse_error"] = f"label {data['label']!r} not in {allowed}"
        return result

    # Fallback: no valid JSON. Accept only if exactly one allowed label appears
    # as a whole word (longest labels first so PARTIALLY_RELEVANT != RELEVANT).
    found = set()
    upper = cleaned.upper().replace("-", "_")
    for lab in sorted(allowed, key=len, reverse=True):
        if re.search(rf"(?<![A-Z_]){lab}(?![A-Z_])", upper):
            found.add(lab)
            upper = re.sub(rf"(?<![A-Z_]){lab}(?![A-Z_])", " ", upper)
    if len(found) == 1:
        result.update(label=found.pop(), parse_ok=True, explanation=cleaned[:500],
                      parse_error="recovered label from non-JSON output")
    else:
        result["parse_error"] = f"no unambiguous label in output (found {sorted(found)})"
    return result


def strict_correct(label: str) -> int:
    return int(label == "CORRECT")


def lenient_correct(label: str) -> int:
    return int(label in ("CORRECT", "PARTIAL"))


def is_decisive(label: str) -> bool:
    return label not in NOT_DECISIVE
