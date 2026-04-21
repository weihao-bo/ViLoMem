"""MathGlance verifier: bbox IoU, MCQ, numeric, or text matching by answer_type."""

from __future__ import annotations

import re
from typing import Any, Dict


def _parse_bbox_from_string(s: str) -> list[int] | None:
    """Extract the first 4 numbers from `s` as an [x1, y1, x2, y2] bbox."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", s)
    if len(numbers) >= 4:
        try:
            return [int(float(n)) for n in numbers[:4]]
        except (ValueError, TypeError):
            return None
    return None


def _compute_iou(box1: list[int], box2: list[int]) -> float:
    """IoU for two [x1, y1, x2, y2] axis-aligned boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def mathglance_verify(
    prediction: str,
    gold_answer: str,
    answer_type: str | None = None,
    problem_version: str | None = None,
) -> tuple[bool, float, Dict[str, Any]]:
    """Dispatch on `answer_type` (or infer from `problem_version`) and verify."""
    if not answer_type:
        if problem_version == "task_grd":
            answer_type = "bbox"
        elif problem_version == "task_cls":
            answer_type = "mcq"
        elif problem_version == "task_cnt":
            answer_type = "number"
        else:
            answer_type = "text"

    details: Dict[str, Any] = {
        "method": "mathglance",
        "answer_type": answer_type,
        "problem_version": problem_version,
    }

    if answer_type == "mcq":
        pred_match = re.search(r"\b([A-E])\b", prediction.upper())
        pred_letter = pred_match.group(1) if pred_match else None
        gold_letter = gold_answer.strip().upper()

        verified = pred_letter == gold_letter
        details.update({
            "extracted_answer": pred_letter,
            "gold_answer": gold_letter,
            "verified": verified,
            "score": 1.0 if verified else 0.0,
        })
        return verified, 1.0 if verified else 0.0, details

    if answer_type == "bbox":
        pred_bbox = _parse_bbox_from_string(prediction)
        gold_bbox = _parse_bbox_from_string(gold_answer)

        if pred_bbox is None or gold_bbox is None:
            details.update({
                "extracted_answer": pred_bbox,
                "gold_bbox": gold_bbox,
                "verified": False,
                "score": 0.0,
                "error": "bbox_parse_failed",
            })
            return False, 0.0, details

        iou = _compute_iou(pred_bbox, gold_bbox)
        verified = iou >= 0.5
        details.update({
            "extracted_answer": pred_bbox,
            "gold_bbox": gold_bbox,
            "iou": iou,
            "verified": verified,
            "score": iou,
        })
        return verified, iou, details

    if answer_type == "number":
        pred_numbers = re.findall(r"-?\d+(?:\.\d+)?", prediction)
        gold_numbers = re.findall(r"-?\d+(?:\.\d+)?", gold_answer)

        if not pred_numbers or not gold_numbers:
            details.update({
                "extracted_answer": pred_numbers[0] if pred_numbers else None,
                "gold_answer": gold_numbers[0] if gold_numbers else gold_answer,
                "verified": False,
                "score": 0.0,
                "error": "number_parse_failed",
            })
            return False, 0.0, details

        try:
            pred_num = float(pred_numbers[-1])
            gold_num = float(gold_numbers[0])
            verified = abs(pred_num - gold_num) < 0.01 or pred_num == gold_num
            details.update({
                "extracted_answer": pred_num,
                "gold_answer": gold_num,
                "verified": verified,
                "score": 1.0 if verified else 0.0,
            })
            return verified, 1.0 if verified else 0.0, details
        except (ValueError, TypeError):
            details.update({
                "extracted_answer": pred_numbers[-1] if pred_numbers else None,
                "gold_answer": gold_answer,
                "verified": False,
                "score": 0.0,
                "error": "number_conversion_failed",
            })
            return False, 0.0, details

    pred_text = prediction.strip().lower()
    gold_text = gold_answer.strip().lower()

    verified = gold_text == pred_text or gold_text in pred_text
    details.update({
        "extracted_answer": prediction.strip(),
        "gold_answer": gold_answer.strip(),
        "verified": verified,
        "score": 1.0 if verified else 0.0,
    })
    return verified, 1.0 if verified else 0.0, details
