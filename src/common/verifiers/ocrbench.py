"""OCRBench verifier (official substring matching logic).

Mirrors VLMEvalKit's OCRBench_eval:
    - Handwritten Mathematical Expression Recognition (HMER):
      strip spaces then substring match
    - Other categories: lowercase then substring match
    - Multiple gold answers: any match counts as correct
"""

from __future__ import annotations

import ast
from typing import Any, Dict


def _parse_ocrbench_answers(answer_str: Any) -> list[str]:
    """Parse answers stored as "['a', 'b']" / list / bare string into list[str]."""
    if not answer_str:
        return []

    if isinstance(answer_str, list):
        return [str(a) for a in answer_str]

    try:
        parsed = ast.literal_eval(answer_str)
        if isinstance(parsed, list):
            return [str(a) for a in parsed]
        return [str(parsed)]
    except (ValueError, SyntaxError):
        pass

    return [answer_str]


def ocrbench_verify(
    prediction: str,
    gold_answer: str,
    category: str | None = None,
) -> tuple[bool, float, Dict[str, Any]]:
    """Verify a single OCRBench prediction. Returns (verified, score, details)."""
    answers = _parse_ocrbench_answers(gold_answer)
    if not answers:
        return False, 0.0, {
            "method": "ocrbench",
            "verified": False,
            "score": 0.0,
            "error": "no_valid_answers",
            "gold_answer": gold_answer,
        }

    prediction_text = prediction.strip()
    is_hmer = category == "Handwritten Mathematical Expression Recognition"

    matched_answer = None
    for answer in answers:
        if is_hmer:
            answer_norm = answer.strip().replace("\n", " ").replace(" ", "")
            predict_norm = prediction_text.replace("\n", " ").replace(" ", "")
        else:
            answer_norm = answer.lower().strip().replace("\n", " ")
            predict_norm = prediction_text.lower().strip().replace("\n", " ")

        if answer_norm in predict_norm:
            matched_answer = answer
            break

    verified = matched_answer is not None
    score = 1.0 if verified else 0.0
    details: Dict[str, Any] = {
        "method": "ocrbench",
        "verified": verified,
        "score": score,
        "category": category,
        "is_hmer": is_hmer,
        "gold_answers": answers,
        "matched_answer": matched_answer,
        "extracted_answer": matched_answer,
    }

    if not verified:
        details["error"] = "substring_not_found"
    return verified, score, details
