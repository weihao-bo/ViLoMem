"""Answer parser for VLMEvalKit benchmarks.

This module provides robust answer parsing and verification:
- Tiered parsing strategy (exact match → regex → content match → fallback)
- Type-specific comparison (float with tolerance, MCQ with content matching, etc.)
- Inspired by VLMEvalKit's parse_multi_choice_response
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

NUMERIC_TOKEN_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
FRACTION_PATTERN = re.compile(r"-?\d+\s*/\s*-?\d+")
ALLOWED_EXPR_PATTERN = re.compile(r"^[0-9+\-*/().\s]+$")


def _normalize_answer_type(value: str | None) -> Literal["float", "mcq", "text"]:
    """Normalize answer_type metadata to supported parser values."""
    if not value:
        return "text"

    value_lower = value.lower()
    if value_lower in {"mcq", "multi_choice", "multiple_choice"}:
        return "mcq"
    if value_lower in {"float", "integer", "number", "numeric"}:
        return "float"
    return "text"


@dataclass
class AnswerParserConfig:
    """Configuration for answer parsing and verification."""

    # Answer type determines parsing strategy
    answer_type: Literal["float", "mcq", "text"] = "text"

    # Float settings
    float_tolerance: float = 0.001

    # MCQ settings
    mcq_choices: list[str] = field(default_factory=lambda: ["A", "B", "C", "D"])
    mcq_index2ans: dict[str, str] = field(
        default_factory=dict
    )  # e.g., {"A": "Circle", "B": "Square"}

    # Text normalization
    text_lowercase: bool = True
    text_strip_punctuation: bool = True

    @classmethod
    def from_sample_metadata(cls, metadata: dict) -> "AnswerParserConfig":
        """Create config from sample metadata.

        Args:
            metadata: Sample metadata dict

        Returns:
            AnswerParserConfig instance
        """
        raw_answer_type = metadata.get("answer_type")
        if not raw_answer_type:
            question_type = metadata.get("question_type")
            if isinstance(question_type, str) and question_type.lower() in {
                "mcq",
                "multi_choice",
                "multiple_choice",
            }:
                raw_answer_type = "mcq"
            elif metadata.get("choices"):
                raw_answer_type = "mcq"

        answer_type = _normalize_answer_type(raw_answer_type)

        # Extract choices for MCQ
        choices = metadata.get("choices", {})
        mcq_choices = list(choices.keys()) if choices else ["A", "B", "C", "D", "E"]
        mcq_index2ans = choices if choices else {}

        return cls(
            answer_type=answer_type,
            mcq_choices=mcq_choices,
            mcq_index2ans=mcq_index2ans,
        )


def parse_prediction(prediction: str, config: AnswerParserConfig) -> str | float | None:
    """Parse prediction based on answer type.

    Args:
        prediction: Raw prediction string from model
        config: Parser configuration

    Returns:
        Parsed answer in canonical form, or None if parsing fails
    """
    if config.answer_type == "float":
        return _parse_numeric(prediction)
    elif config.answer_type == "mcq":
        return _parse_mcq(prediction, config)
    elif config.answer_type == "text":
        return _normalize_text(prediction, config)
    else:
        # Fallback: return as-is
        return prediction.strip()


def _parse_numeric(text: str) -> float | None:
    """Parse numeric answer with support for units, scientific notation, etc.

    Args:
        text: Text to parse

    Returns:
        Parsed float value, or None if parsing fails
    """
    text = str(text).strip()
    if not text:
        return None

    # Remove common units/symbols but preserve math syntax
    text = text.replace(",", "")
    text = re.sub(r"[°%$€£¥]", "", text)
    text = text.replace("π", "pi")

    # Helper to safely evaluate simple numeric expressions
    def _safe_eval(expr: str) -> float | None:
        expr = expr.strip()
        if not expr:
            return None
        if not ALLOWED_EXPR_PATTERN.fullmatch(expr):
            return None
        try:
            return float(eval(expr, {"__builtins__": {}}, {}))
        except Exception:
            return None

    # Try \boxed{} content first
    boxed_match = re.search(r"\\boxed\{(.+?)\}", text)
    if boxed_match:
        boxed_value = boxed_match.group(1).strip()
        evaluated = _safe_eval(boxed_value)
        if evaluated is not None:
            return evaluated

    # If expression contains '=', prefer the segment after the last '='
    if "=" in text:
        trailing = text.split("=")[-1].strip()
        evaluated = _safe_eval(trailing)
        if evaluated is not None:
            return evaluated

    # Handle explicit fractions (e.g., "3/4")
    frac_matches = FRACTION_PATTERN.findall(text)
    if frac_matches:
        num_str, denom_str = frac_matches[-1].split("/")
        try:
            return float(Fraction(int(num_str), int(denom_str)))
        except Exception:
            pass

    # As a final fallback, grab the last numeric token in the string
    tokens = NUMERIC_TOKEN_PATTERN.findall(text)
    if tokens:
        try:
            return float(tokens[-1])
        except ValueError:
            return None

    # Attempt to directly evaluate if the entire text is a numeric expression
    direct_eval = _safe_eval(text)
    if direct_eval is not None:
        return direct_eval

    return None


def _parse_mcq(response: str, config: AnswerParserConfig) -> str | None:
    """Parse MCQ response with tiered strategy (from VLMEvalKit).

    Parsing order:
    1. Structured patterns: (A), A., A), \\boxed{A}
    2. Isolated letter: " A " (with spaces)
    3. Content matching: if response mentions option text
    4. Random fallback (for fair evaluation)

    Args:
        response: Model response text
        config: Parser configuration with choices

    Returns:
        Extracted choice letter, or None
    """
    response = str(response).strip()

    # Clean punctuation at start/end
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)

    # Add spaces for clean matching (avoid partial matches)
    response = " " + response + " "

    candidates = []

    # Step 1: Structured patterns
    for choice in config.mcq_choices:
        patterns = [
            f"({choice})",  # (A)
            f"{choice}. ",  # A.
            f"{choice})",  # A)
            f"\\boxed{{{choice}}}",  # \boxed{A}
        ]
        if any(pattern in response for pattern in patterns):
            candidates.append(choice)

    # Step 2: Isolated letter with spaces (avoid partial matches)
    if not candidates:
        for choice in config.mcq_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    # Step 3: Content matching (if response is long and we have option texts)
    if not candidates and len(response.split()) > 5 and config.mcq_index2ans:
        for idx, ans_text in config.mcq_index2ans.items():
            # Check if option text appears in response
            if ans_text.lower() in response.lower():
                candidates.append(idx)

    # Step 4: Fallback strategies
    if not candidates:
        # Try extracting any uppercase letter
        for choice in config.mcq_choices:
            if choice in response:
                candidates.append(choice)

    if not candidates:
        # Random fallback for fair evaluation
        return random.choice(config.mcq_choices)
    elif len(candidates) > 1:
        # Return last occurrence (model's final answer usually at end)
        positions = [response.rfind(c) for c in candidates]
        return candidates[positions.index(max(positions))]
    else:
        return candidates[0]


def _normalize_text(text: str, config: AnswerParserConfig) -> str:
    """Normalize text answer.

    Args:
        text: Text to normalize
        config: Normalization settings

    Returns:
        Normalized text
    """
    text = text.strip()

    # Extract from \boxed{} if present
    boxed_match = re.search(r"\\boxed\{(.+?)\}", text)
    if boxed_match:
        text = boxed_match.group(1)

    if config.text_strip_punctuation:
        text = re.sub(r"[^\w\s]", "", text)
    if config.text_lowercase:
        text = text.lower()

    return text.strip()


def compare_answers(
    gold: str | float,
    prediction: str | float,
    config: AnswerParserConfig,
) -> bool:
    """Compare gold and prediction based on answer type.

    Args:
        gold: Gold answer (can be parsed or unparsed)
        prediction: Predicted answer (can be parsed or unparsed)
        config: Parser configuration

    Returns:
        True if answers match, False otherwise
    """
    # Parse both if they are strings
    if isinstance(gold, str):
        gold_parsed = parse_prediction(gold, config)
    else:
        gold_parsed = gold

    if isinstance(prediction, str):
        pred_parsed = parse_prediction(prediction, config)
    else:
        pred_parsed = prediction

    # Handle None cases
    if gold_parsed is None or pred_parsed is None:
        return False

    # Type-specific comparison
    if config.answer_type == "float":
        # Tolerance-based comparison
        if not isinstance(gold_parsed, (int, float)) or not isinstance(
            pred_parsed, (int, float)
        ):
            return False
        return abs(float(gold_parsed) - float(pred_parsed)) <= config.float_tolerance

    elif config.answer_type == "mcq":
        # Two-stage matching: letter OR option content
        if gold_parsed == pred_parsed:
            return True

        # If both are letters, check if option contents match
        if (
            isinstance(gold_parsed, str)
            and isinstance(pred_parsed, str)
            and config.mcq_index2ans
        ):
            # Handle case where gold is letter but pred is letter
            if gold_parsed in config.mcq_choices and pred_parsed in config.mcq_choices:
                gold_content = config.mcq_index2ans.get(gold_parsed, "")
                pred_content = config.mcq_index2ans.get(pred_parsed, "")
                if gold_content and pred_content:
                    return gold_content.lower().strip() == pred_content.lower().strip()

            # Handle case where gold is content but pred is letter (or vice versa)
            if gold_parsed in config.mcq_index2ans.values():
                # Gold is content, check if pred letter maps to it
                return (
                    config.mcq_index2ans.get(pred_parsed, "").lower().strip()
                    == gold_parsed.lower().strip()
                )
            elif pred_parsed in config.mcq_index2ans.values():
                # Pred is content, check if gold letter maps to it
                return (
                    config.mcq_index2ans.get(gold_parsed, "").lower().strip()
                    == pred_parsed.lower().strip()
                )

        return False

    elif config.answer_type == "text":
        # Normalized string comparison
        if not isinstance(gold_parsed, str) or not isinstance(pred_parsed, str):
            return False
        return gold_parsed == pred_parsed

    else:
        # Fallback: simple equality
        return gold_parsed == pred_parsed
