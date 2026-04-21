"""ANLS (Average Normalized Levenshtein Similarity) verifier.

Used by DocVQA / InfoVQA. Mirrors VLMEvalKit's anls_compute:
    - Normalize both strings (lowercase, whitespace collapse)
    - Compute normalized Levenshtein distance
    - Clamp scores below `threshold` to 0.0
"""

from __future__ import annotations

from typing import Any, Dict

from .ocrbench import _parse_ocrbench_answers


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Iterative Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def anls_compute(groundtruth: str, prediction: str, threshold: float = 0.5) -> float:
    """Compute a single ANLS score between `groundtruth` and `prediction`."""
    gt_answer = " ".join(groundtruth.strip().lower().split())
    det_answer = " ".join(prediction.strip().lower().split())

    dist = _levenshtein_distance(gt_answer, det_answer)
    length = max(len(gt_answer), len(det_answer))

    if length == 0:
        return 1.0

    normalized_distance = float(dist) / float(length)
    anls = 1.0 - normalized_distance

    if anls <= threshold:
        anls = 0.0
    return anls


def anls_verify(
    prediction: str,
    gold_answer: str,
    threshold: float = 0.5,
) -> tuple[bool, float, Dict[str, Any]]:
    """Verify a prediction against a (possibly list-encoded) gold answer via ANLS."""
    answers = _parse_ocrbench_answers(gold_answer)
    if not answers:
        return False, 0.0, {
            "method": "anls",
            "verified": False,
            "score": 0.0,
            "error": "no_valid_answers",
            "gold_answer": gold_answer,
        }

    scores = [anls_compute(ans, prediction, threshold) for ans in answers]
    best_score = max(scores)
    best_idx = scores.index(best_score)
    matched_answer = answers[best_idx] if best_score > 0 else None

    verified = best_score > threshold
    details: Dict[str, Any] = {
        "method": "anls",
        "verified": verified,
        "score": best_score,
        "threshold": threshold,
        "gold_answers": answers,
        "all_scores": scores,
        "matched_answer": matched_answer,
        "extracted_answer": matched_answer,
    }

    if not verified:
        details["error"] = "anls_below_threshold"
    return verified, best_score, details
