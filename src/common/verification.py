"""Shared verification utilities for LangGraph agents."""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from tools.answer_parser import (
    AnswerParserConfig,
    compare_answers,
    parse_prediction,
)
from tools.vlmeval_verifier import (
    should_use_vlmeval_matcher,
    verify_with_vlmeval,
)

from .llm_judge import llm_verify_answer
from .vlmeval_bridge import vlmeval_verify_sample

logger = logging.getLogger(__name__)


def _to_serializable(value: Any) -> Any:
    """Convert values (including SymPy/math objects) to JSON-safe types."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_serializable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}

    # SymPy numbers (e.g., Integer, Rational)
    try:
        import sympy  # type: ignore

        if isinstance(value, sympy.Basic):
            if value.is_Integer:
                return int(value)
            if value.is_Rational or value.is_Float:
                try:
                    return float(value)
                except TypeError:
                    return str(value)
            return str(value)
    except Exception:
        pass

    for caster in (float, int):
        try:
            return caster(value)  # type: ignore[arg-type]
        except Exception:
            continue

    return str(value)


def _normalize_choice_key(choice: Any) -> str:
    return str(choice).strip().upper()


def _resolve_option_from_answer(
    answer: Any, choices: dict[str, Any]
) -> str | None:
    """Map a parsed answer (letter or text) back to a choice option."""

    if not isinstance(answer, str):
        return None

    normalized = _normalize_choice_key(answer)
    for option in choices.keys():
        if normalized == _normalize_choice_key(option):
            return str(option)

    normalized_text = answer.strip().lower()
    for option, value in choices.items():
        if isinstance(value, str) and value.strip().lower() == normalized_text:
            return str(option)

    return None


def run_answer_parser_check(
    prediction: str,
    gold_answer: str,
    sample_metadata: dict[str, Any] | None = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Attempt verification via tools.answer_parser."""
    metadata = sample_metadata or {}
    try:
        config = AnswerParserConfig.from_sample_metadata(metadata)
        parsed_prediction = parse_prediction(prediction, config)
        parsed_gold = parse_prediction(gold_answer, config)

        if parsed_prediction is None or parsed_gold is None:
            return False, {
                "method": "answer_parser",
                "verified": False,
                "parsed_prediction": parsed_prediction,
                "parsed_gold": parsed_gold,
                "answer_type": config.answer_type,
                "extracted_answer": parsed_prediction,
                "error": "parser_failed",
            }

        choices = metadata.get("choices") or {}
        answer_option = metadata.get("answer_option")

        use_choice_alignment = (
            isinstance(choices, dict) and choices and answer_option is not None
        )

        if use_choice_alignment:
            expected_option = _normalize_choice_key(answer_option)
            predicted_option = _resolve_option_from_answer(parsed_prediction, choices)
            verified = (
                predicted_option is not None
                and _normalize_choice_key(predicted_option) == expected_option
            )
            parsed_gold = expected_option
        else:
            predicted_option = None
            expected_option = None
            verified = compare_answers(gold_answer, prediction, config)

        details: Dict[str, Any] = {
            "method": "answer_parser",
            "verified": bool(verified),
            "parsed_prediction": parsed_prediction,
            "parsed_gold": parsed_gold,
            "answer_type": config.answer_type,
            "extracted_answer": parsed_prediction,
            "error": None,
        }

        if use_choice_alignment:
            details["predicted_option"] = predicted_option
            details["expected_option"] = expected_option
            if predicted_option is None:
                details["error"] = "choice_not_mapped"
            elif not verified:
                details["error"] = "choice_mismatch"

        return bool(verified), details
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Answer parser check failed: %s", exc)
        return False, {
            "method": "answer_parser",
            "verified": False,
            "error": str(exc),
        }


def run_math_verify_check(
    prediction: str,
    gold_answer: str,
) -> Tuple[bool, Dict[str, Any]]:
    """Fallback verification using math-verify."""
    try:
        from math_verify import (  # type: ignore
            ExprExtractionConfig,
            LatexExtractionConfig,
            StringExtractionConfig,
            parse,
            verify,
        )
    except Exception as exc:  # pragma: no cover - import guard
        return False, {
            "method": "math_verify",
            "verified": False,
            "error": f"math_verify_unavailable: {exc}",
        }

    try:
        extraction_configs = [
            LatexExtractionConfig(boxed_match_priority=0),
            ExprExtractionConfig(),
            StringExtractionConfig(
                strings=("Yes", "No", "True", "False"), lowercase=True
            ),
        ]

        gold_parsed = parse(
            gold_answer.strip(),
            extraction_config=extraction_configs,
        )
        pred_parsed = parse(
            prediction.strip(),
            extraction_config=extraction_configs,
        )

        verified = verify(
            gold_parsed,
            pred_parsed,
            float_rounding=6,
            numeric_precision=15,
            strict=True,
            timeout_seconds=5,
            raise_on_error=False,
        )

        return bool(verified), {
            "method": "math_verify",
            "verified": bool(verified),
            "parsed_prediction": pred_parsed,
            "parsed_gold": gold_parsed,
            "extracted_answer": pred_parsed
            if isinstance(pred_parsed, (int, float, str))
            else None,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - runtime guard
        logger.debug("math_verify check failed: %s", exc)
        return False, {
            "method": "math_verify",
            "verified": False,
            "error": str(exc),
        }


def verify_prediction(
    prediction: str,
    gold_answer: str,
    sample_metadata: dict[str, Any] | None = None,
    benchmark_config: Any | None = None,
    question: str | None = None,
) -> Dict[str, Any]:
    """Run the full verification pipeline (VLMEval → parser → math-verify)."""
    attempts: list[Dict[str, Any]] = []
    metadata = sample_metadata or {}

    # Stage 0: MathVista-specific VLMEvalKit verification (full pipeline)
    vlmeval_result = vlmeval_verify_sample(
        prediction=prediction,
        gold_answer=gold_answer,
        question=question,
        sample_metadata=metadata,
    )
    if vlmeval_result is not None:
        vl_verified, vl_details = vlmeval_result
        vl_details.setdefault("stage", "vlmeval")
        attempts.append(vl_details)
        if vl_verified:
            attempts_serializable = _to_serializable(attempts)
            return {
                "verified": True,
                "verification_method": vl_details.get("method"),
                "verification_error": None,
                "extracted_answer": _to_serializable(
                    vl_details.get("extracted_answer")
                ),
                "verification_attempts": attempts_serializable,
            }

    # Stage 1: VLMEvalKit matcher for MCQ
    use_vlmeval, choices = should_use_vlmeval_matcher(
        benchmark_config=benchmark_config,
        sample_metadata=metadata,
    )
    if use_vlmeval:
        answer_option = metadata.get("answer_option")
        question_type = metadata.get("question_type", "multi_choice")
        verified, vl_details = verify_with_vlmeval(
            prediction=prediction,
            gold_answer=gold_answer,
            choices=choices,
            question_type=question_type,
            answer_option=answer_option,
        )
        vl_details.setdefault("method", "vlmeval_matcher")
        if "extracted_answer" not in vl_details:
            vl_details["extracted_answer"] = vl_details.get("mapped_answer")
        vl_details.update({"stage": "vlmeval", "verified": bool(verified)})
        attempts.append(vl_details)
        if verified:
            attempts_serializable = _to_serializable(attempts)
            return {
                "verified": True,
                "verification_method": vl_details["method"],
                "verification_error": None,
                "extracted_option": vl_details.get("extracted_option"),
                "extracted_answer": _to_serializable(
                    vl_details.get("extracted_answer")
                ),
                "verification_attempts": attempts_serializable,
            }

    # Stage 2: Answer parser
    parser_verified, parser_details = run_answer_parser_check(
        prediction=prediction,
        gold_answer=gold_answer,
        sample_metadata=metadata,
    )
    if "extracted_answer" not in parser_details:
        parser_details["extracted_answer"] = parser_details.get("parsed_prediction")
    parser_details.update({"stage": "answer_parser"})
    attempts.append(parser_details)
    if parser_details.get("error") == "choice_mismatch":
        attempts_serializable = _to_serializable(attempts)
        return {
            "verified": False,
            "verification_method": parser_details.get("method"),
            "verification_error": parser_details.get("error"),
            "extracted_answer": _to_serializable(
                parser_details.get("extracted_answer")
            ),
            "extracted_option": parser_details.get("predicted_option"),
            "expected_option": parser_details.get("expected_option"),
            "verification_attempts": attempts_serializable,
        }
    if parser_verified:
        attempts_serializable = _to_serializable(attempts)
        return {
            "verified": True,
            "verification_method": parser_details["method"],
            "verification_error": None,
            "extracted_answer": _to_serializable(
                parser_details.get("extracted_answer")
            ),
            "verification_attempts": attempts_serializable,
        }

    # Stage 3: math-verify fallback
    math_verified, math_details = run_math_verify_check(
        prediction=prediction,
        gold_answer=gold_answer,
    )
    if "extracted_answer" not in math_details:
        math_details["extracted_answer"] = math_details.get("parsed_prediction")
    math_details.update({"stage": "math_verify"})
    attempts.append(math_details)
    if math_verified:
        attempts_serializable = _to_serializable(attempts)
        return {
            "verified": True,
            "verification_method": math_details["method"],
            "verification_error": None,
            "extracted_answer": _to_serializable(math_details.get("extracted_answer")),
            "verification_attempts": attempts_serializable,
        }

    # Stage 4: LLM judge fallback (when all rule-based methods fail)
    llm_verified, llm_details = llm_verify_answer(
        question=question or "",
        prediction=prediction,
        gold_answer=gold_answer,
        choices=metadata.get("choices"),
        judge_model_name=metadata.get("judge_model"),
    )
    if "extracted_answer" not in llm_details:
        llm_details["extracted_answer"] = (
            None  # LLM judge doesn't extract structured answer
        )
    llm_details.update({"stage": "llm_judge"})
    attempts.append(llm_details)
    if llm_verified:
        attempts_serializable = _to_serializable(attempts)
        return {
            "verified": True,
            "verification_method": llm_details["method"],
            "verification_error": None,
            "extracted_answer": _to_serializable(llm_details.get("extracted_answer")),
            "llm_judge_reasoning": llm_details.get("reasoning"),
            "verification_attempts": attempts_serializable,
        }

    # All stages failed
    last_attempt = attempts[-1] if attempts else {}
    attempts_serializable = _to_serializable(attempts)
    return {
        "verified": False,
        "verification_method": last_attempt.get("method"),
        "verification_error": last_attempt.get("error", "verification_failed"),
        "extracted_answer": _to_serializable(last_attempt.get("extracted_answer")),
        "extracted_option": last_attempt.get("extracted_option"),
        "verification_attempts": attempts_serializable,
    }
