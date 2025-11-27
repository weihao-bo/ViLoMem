"""VLMEvalKit Native Verification Module.

This module provides VLMEvalKit's native verification logic for MathVista and other benchmarks.
Directly copied from: https://github.com/open-compass/VLMEvalKit

Key Functions:
- post_check: VLMEvalKit's native answer verification (MCQ + numeric + text)
- list_to_dict: Convert choice list to dict mapping (A->choice[0], B->choice[1], ...)
- verify_answer_native: Wrapper for easy integration

Based on: https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/dataset/utils/mathvista.py
"""

from __future__ import annotations

import ast
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Try to import VLMEvalKit's can_infer, fallback to our implementation
try:
    from vlmeval.utils import can_infer

    logger.info("✓ Using VLMEvalKit's native can_infer function")
    VLMEVAL_AVAILABLE = True
except Exception as exc:  # pylint: disable=broad-except
    from tools.vlmeval_matching import can_infer

    logger.warning(
        "VLMEvalKit not available: using built-in can_infer (%s). "
        "For full VLMEvalKit compatibility, install: pip install vlmeval",
        exc,
    )
    VLMEVAL_AVAILABLE = False


def list_to_dict(lst: list[str]) -> dict[str, str]:
    """Convert choice list to dict mapping.

    Converts ['Yes', 'No'] to {'A': 'Yes', 'B': 'No'}.

    Args:
        lst: List of choice strings

    Returns:
        Dict mapping option letters (A, B, C, ...) to choices

    Examples:
        >>> list_to_dict(['Yes', 'No'])
        {'A': 'Yes', 'B': 'No'}
        >>> list_to_dict(['cat', 'dog', 'bird'])
        {'A': 'cat', 'B': 'dog', 'C': 'bird'}
    """
    return {chr(65 + i): val for i, val in enumerate(lst)}


def _ensure_choices_dict(
    choices: list[str] | dict[str, str] | str | None,
) -> dict[str, str]:
    """Normalize different choice formats to a dict."""
    if choices is None:
        return {}
    if isinstance(choices, dict):
        return {
            str(key).strip().upper(): str(value)
            for key, value in choices.items()
            if str(value).strip()
        }
    if isinstance(choices, list):
        return list_to_dict([str(val) for val in choices])
    if isinstance(choices, str):
        try:
            parsed = ast.literal_eval(choices)
        except (ValueError, SyntaxError):
            return {}
        return _ensure_choices_dict(parsed)
    return {}


def should_use_vlmeval_matcher(
    benchmark_config: Any | None,
    sample_metadata: dict[str, Any] | None,
) -> tuple[bool, dict[str, str]]:
    """Determine if VLMEvalKit MCQ matcher should be used."""
    metadata = sample_metadata or {}
    question_type = (metadata.get("question_type") or "").lower()
    choices_raw = metadata.get("choices")
    choices = _ensure_choices_dict(choices_raw)

    is_mcq = "choice" in question_type or bool(choices)
    if not is_mcq and benchmark_config is not None:
        task_type = getattr(benchmark_config, "task_type", None)
        if isinstance(task_type, str) and task_type.lower() == "mcq":
            is_mcq = True

    use_vlmeval = bool(choices) and is_mcq
    return use_vlmeval, choices


def verify_with_vlmeval(
    prediction: str,
    gold_answer: str,
    choices: list[str] | dict[str, str] | str,
    question_type: str | None = "multi_choice",
    answer_option: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Verify MCQ prediction using VLMEvalKit logic."""
    normalized_choices = _ensure_choices_dict(choices)
    if not normalized_choices:
        return False, {
            "method": "vlmeval_matcher",
            "verified": False,
            "error": "choices_missing",
        }

    extracted = can_infer(prediction, normalized_choices)
    if not extracted or extracted == "Z":
        from tools.vlmeval_matching import parse_multi_choice_response

        all_choices = list(normalized_choices.keys())
        if not all_choices:
            return False, {
                "method": "vlmeval_matcher",
                "verified": False,
                "error": "choices_missing",
            }
        extracted = parse_multi_choice_response(
            prediction, all_choices, normalized_choices
        )

    mapped_answer = normalized_choices.get(extracted) if extracted else None
    gold_norm = str(gold_answer).strip().lower()
    mapped_norm = str(mapped_answer).strip().lower() if mapped_answer else None

    verified = False
    if answer_option and extracted:
        verified = extracted.upper() == answer_option.upper()
    if not verified and mapped_norm is not None:
        verified = mapped_norm == gold_norm

    details: dict[str, Any] = {
        "method": "vlmeval_matcher",
        "extracted_option": extracted,
        "mapped_answer": mapped_answer,
        "extracted_answer": mapped_answer,
        "choices_available": bool(normalized_choices),
        "question_type": question_type,
    }

    if not verified:
        details["error"] = "answer_mismatch"

    return verified, details


def post_check(
    line: dict[str, Any], prefetch: bool = False
) -> bool | str | int | float:
    """VLMEvalKit's native answer verification logic.

    This is the core verification function from VLMEvalKit's MathVista implementation.
    Supports MCQ, integer, float, and text answer types.

    Args:
        line: Dict containing question metadata and model response
            Required keys:
            - 'question_type': 'multi_choice' or other
            - 'prediction' (if prefetch=True) or 'res' (if prefetch=False)
            - 'answer': gold answer
            For MCQ:
            - 'answer_option': correct option letter (e.g., 'A', 'B')
            - 'choices': string representation of choice list (e.g., "['Yes', 'No']")
            For numeric:
            - 'answer_type': 'integer' or 'float'
        prefetch: If True, return extracted answer; if False, return True/False for correctness

    Returns:
        If prefetch=True: extracted answer (str/int/float) or False if extraction failed
        If prefetch=False: True if correct, False if incorrect

    Examples:
        >>> line = {
        ...     'question_type': 'multi_choice',
        ...     'prediction': 'The answer is A',
        ...     'answer_option': 'A',
        ...     'choices': "['Yes', 'No']"
        ... }
        >>> post_check(line, prefetch=True)
        'A'
        >>> post_check(line, prefetch=False)
        True
    """
    res = None
    ans = line["answer"]
    response = (
        line["prediction"] if prefetch else line.get("res", line.get("prediction", ""))
    )

    try:
        if line.get("question_type") == "multi_choice":
            ans = line.get("answer_option", ans)

            # Parse choices from string representation
            choices_raw = line.get("choices", "[]")
            if isinstance(choices_raw, str):
                choices_list = eval(choices_raw)  # "['Yes', 'No']" -> ['Yes', 'No']
            else:
                choices_list = choices_raw

            # Convert list to dict: ['Yes', 'No'] -> {'A': 'Yes', 'B': 'No'}
            if isinstance(choices_list, list):
                choices = list_to_dict(choices_list)
            else:
                # Already a dict
                choices = choices_list

            # Try can_infer first (VLMEvalKit's preferred method)
            res = can_infer(response, choices)

            # If can_infer fails, fallback to parse_multi_choice_response
            if not res or res == False:
                from tools.vlmeval_matching import parse_multi_choice_response

                all_choices = list(choices.keys())
                if not all_choices:
                    # No choices available - cannot proceed with MCQ verification
                    logger.warning(f"MCQ verification failed: choices is empty")
                    if prefetch:
                        return False
                else:
                    res = parse_multi_choice_response(response, all_choices, choices)
                    logger.debug(
                        f"can_infer failed, using parse_multi_choice_response: {res}"
                    )

            if prefetch:
                return res
        else:
            # Numeric or text answer
            answer_type = line.get("answer_type", "text")
            if answer_type == "integer":
                res = int(response)
                ans = int(line["answer"])
            elif answer_type == "float":
                res = float(response)
                ans = float(line["answer"])
            else:
                res = str(response).strip()
                ans = str(ans).strip()
    except (ValueError, SyntaxError, NameError) as e:
        logger.debug(f"post_check parsing error: {e}")
        if prefetch:
            return False
        return False

    # Compare extracted answer with gold answer
    if res == ans:
        return res if prefetch else True
    else:
        return False


def verify_answer_native(
    prediction: str,
    gold_answer: str,
    question_type: str | None = None,
    answer_type: str | None = None,
    choices: list[str] | dict[str, str] | str | None = None,
    answer_option: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Wrapper for VLMEvalKit's native verification.

    This function provides a simplified interface to post_check for easy integration.

    Args:
        prediction: Model's prediction text
        gold_answer: Gold answer
        question_type: 'multi_choice' or None (for numeric/text)
        answer_type: 'integer', 'float', or 'text' (for non-MCQ)
        choices: Choice list/dict/string (for MCQ)
        answer_option: Correct option letter (for MCQ)

    Returns:
        Tuple of (verified: bool, details: dict) with verification metadata

    Examples:
        >>> # MCQ example
        >>> verify_answer_native(
        ...     prediction="The answer is A",
        ...     gold_answer="Yes",
        ...     question_type="multi_choice",
        ...     choices=['Yes', 'No'],
        ...     answer_option='A'
        ... )
        (True, {'extracted_answer': 'A', 'method': 'vlmeval_post_check'})

        >>> # Numeric example
        >>> verify_answer_native(
        ...     prediction="42",
        ...     gold_answer="42",
        ...     answer_type="integer"
        ... )
        (True, {'extracted_answer': 42, 'method': 'vlmeval_post_check'})
    """
    # Prepare line dict for post_check
    line: dict[str, Any] = {
        "prediction": prediction,
        "answer": gold_answer,
        "question_type": question_type or "free_form",
        "answer_type": answer_type or "text",
    }

    # Add MCQ-specific fields
    if question_type == "multi_choice":
        normalized_choices = _ensure_choices_dict(choices)
        if not normalized_choices:
            logger.warning(
                "MCQ question but choices is None - cannot perform MCQ verification"
            )
            return False, {
                "extracted_answer": None,
                "method": "vlmeval_post_check_failed",
                "error": "choices_missing",
                "vlmeval_available": VLMEVAL_AVAILABLE,
            }

        ordered_values = [
            normalized_choices[key] for key in sorted(normalized_choices.keys())
        ]
        line["choices"] = str(ordered_values)

        if answer_option:
            line["answer_option"] = answer_option

    # Try prefetch (direct extraction without GPT-4)
    extracted = post_check(line, prefetch=True)

    details: dict[str, Any] = {
        "extracted_answer": extracted,
        "method": "vlmeval_post_check",
        "vlmeval_available": VLMEVAL_AVAILABLE,
    }

    # Verify correctness
    if extracted is False:
        # Extraction failed
        return False, details

    # For MCQ, post_check already compared with answer_option
    # For numeric/text, post_check compared the values
    line["res"] = str(extracted) if not isinstance(extracted, bool) else prediction
    verified = post_check(line, prefetch=False)

    return bool(verified), details
