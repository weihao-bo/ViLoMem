"""VLMEvalKit MCQ Answer Matching Utilities.

This module implements VLMEvalKit's answer matching logic for MCQ verification.
It provides robust extraction and matching of predicted answers against gold answers.

Key Functions:
- can_infer: Main matching function (combines option and text matching)
- can_infer_option: Extract option letter from prediction (e.g., 'A', 'B', 'C')
- can_infer_text: Match option content in prediction
- parse_multi_choice_response: Parse prediction to extract chosen option

Based on: https://github.com/open-compass/VLMEvalKit
"""

from __future__ import annotations

import random
import re
import string
from typing import Any

import numpy as np


def can_infer_option(answer: str, choices: dict[str, str]) -> str | bool:
    """Infer option letter from answer by pattern matching.

    Args:
        answer: Model's raw prediction text
        choices: Dict mapping option letters to their content (e.g., {'A': 'Yes', 'B': 'No'})

    Returns:
        Option letter if exactly one is found, 'Z' for rejection, False otherwise

    Examples:
        >>> can_infer_option("The answer is A", {"A": "Yes", "B": "No"})
        'A'
        >>> can_infer_option("I choose B. because...", {"A": "Yes", "B": "No"})
        'B'
    """
    # Handle rejection phrases
    reject_phrases = [
        "Sorry",
        "I can't",
        "I cannot",
        "I'm sorry",
        "I am sorry",
        "I'm not able",
        "I am not able",
        "I don't have enough information",
        "without additional information",
        "not possible to determine",
    ]
    for phrase in reject_phrases:
        if phrase.lower() in answer.lower():
            return "Z"  # Rejection indicator

    # Clean answer: replace punctuation with spaces
    def get_location(string: str, opts: list[str]) -> dict[str, int]:
        """Find first occurrence position of each option in string."""
        locations = {}
        for opt in opts:
            locations[opt] = string.find(opt) if opt in string else -1
        return locations

    # Replace punctuation to isolate option letters
    for punc in [",", ".", "!", "?", ";", ":", "'", '"']:
        answer = answer.replace(punc, " ")

    # Count occurrences of each option letter
    cands = list(choices.keys())
    cnt_dict = {cand: answer.count(f" {cand} ") for cand in cands}

    # If exactly one option found, return it
    cnt_dict = {k: v for k, v in cnt_dict.items() if v > 0}
    if len(cnt_dict) == 1:
        return list(cnt_dict.keys())[0]

    # Check for 'Z' (rejection)
    if "Z" in answer:
        return "Z"

    return False


def can_infer_text(answer: str, choices: dict[str, str]) -> str | bool:
    """Infer option by matching content text in answer.

    Args:
        answer: Model's raw prediction text
        choices: Dict mapping option letters to their content

    Returns:
        Option letter if exactly one option's content is found, False otherwise

    Examples:
        >>> can_infer_text("The food is half eaten", {"A": "Yes", "B": "No"})
        False  # "half" might match partial text
        >>> can_infer_text("The answer is Yes", {"A": "Yes", "B": "No"})
        'A'
    """
    answer_lower = answer.lower()
    matches = []

    for key, value in choices.items():
        if str(value).lower() in answer_lower:
            matches.append(key)

    # Return key only if exactly one match found
    return matches[0] if len(matches) == 1 else False


def can_infer(answer: str, choices: dict[str, str]) -> str | bool:
    """Infer answer from prediction using both option and text matching.

    This is the main matching function that combines can_infer_option and can_infer_text.

    Args:
        answer: Model's raw prediction text
        choices: Dict mapping option letters to their content

    Returns:
        Option letter if inferred, 'Z' for rejection, False otherwise

    Examples:
        >>> can_infer("I choose A", {"A": "Yes", "B": "No"})
        'A'
        >>> can_infer("The answer is Yes", {"A": "Yes", "B": "No"})
        'A'
        >>> can_infer("I don't know", {"A": "Yes", "B": "No"})
        False
    """
    # First try option matching
    result = can_infer_option(answer, choices)
    if result:
        return result

    # Fallback to text matching
    return can_infer_text(answer, choices)


def parse_multi_choice_response(
    response: str, all_choices: list[str], index2ans: dict[str, str]
) -> str:
    """Parse multi-choice response to extract predicted option.

    This function handles various response formats:
    - Bracketed options: "(A)" or "A."
    - Standalone letters: " A " (with spaces)
    - Content matching: matches option text if response is long

    Args:
        response: Model's raw response text
        all_choices: List of valid option letters (e.g., ['A', 'B', 'C', 'D'])
        index2ans: Dict mapping option letters to their content

    Returns:
        Predicted option letter (or random choice if no match found)

    Examples:
        >>> parse_multi_choice_response("(A)", ["A", "B"], {"A": "Yes", "B": "No"})
        'A'
        >>> parse_multi_choice_response("The answer is Yes", ["A", "B"], {"A": "Yes", "B": "No"})
        'A'
    """
    response = str(response)

    # Clean response: strip punctuation
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)

    # Add spaces to avoid partial matches
    response = " " + response + " "

    # Track whether we're matching option letters vs content
    index_ans = True
    ans_with_brack = False
    candidates = []

    # Method 1: Look for bracketed options like (A) or A.
    for choice in all_choices:
        if f"({choice})" in response or f"{choice}. " in response:
            candidates.append(choice)
            ans_with_brack = True

    # Method 2: Look for standalone letter options like " A "
    if len(candidates) == 0:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)

    # Method 3: If response is long (>5 tokens), try content matching
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # This is content matching, not letter matching

    # Method 4: No match found - return random choice
    if len(candidates) == 0:
        return random.choice(all_choices)

    # Method 5: Multiple candidates - select the last occurrence
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:  # Letter matching
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:  # Content matching
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)

        # Return candidate with largest index (last occurrence)
        pred_index = candidates[np.argmax(start_indexes)]
    else:
        # Single candidate
        pred_index = candidates[0]

    return pred_index


def build_choices_from_list(choices_list: list[str]) -> dict[str, str]:
    """Build choices dict from list format.

    Converts a list like ['Yes', 'No'] to {'A': 'Yes', 'B': 'No'}.

    Args:
        choices_list: List of choice content strings

    Returns:
        Dict mapping option letters (A, B, C, ...) to choice content

    Examples:
        >>> build_choices_from_list(['Yes', 'No'])
        {'A': 'Yes', 'B': 'No'}
        >>> build_choices_from_list(['cat', 'dog', 'bird'])
        {'A': 'cat', 'B': 'dog', 'C': 'bird'}
    """
    option_letters = list(string.ascii_uppercase[:10])  # A-J
    return {
        option_letters[i]: str(choice)
        for i, choice in enumerate(choices_list)
        if i < len(option_letters)
    }


def verify_mcq_answer(
    prediction: str,
    gold_answer: str,
    choices: dict[str, str] | None = None,
    answer_option: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Verify MCQ answer using VLMEvalKit matching logic.

    Args:
        prediction: Model's prediction text
        gold_answer: Gold answer (can be option letter or content)
        choices: Optional dict mapping option letters to content
        answer_option: Optional gold answer as option letter

    Returns:
        Tuple of (verified: bool, details: dict) with matching metadata

    Examples:
        >>> verify_mcq_answer("A", "Yes", {"A": "Yes", "B": "No"}, "A")
        (True, {'extracted_option': 'A', 'gold_option': 'A', 'method': 'option_match'})
        >>> verify_mcq_answer("The answer is Yes", "Yes", {"A": "Yes", "B": "No"}, "A")
        (True, {'extracted_option': 'A', 'gold_option': 'A', 'method': 'can_infer'})
    """
    details: dict[str, Any] = {
        "extracted_option": None,
        "gold_option": answer_option,
        "method": None,
    }

    # Case 1: No choices - direct text comparison
    if not choices:
        verified = prediction.strip().lower() == gold_answer.strip().lower()
        details["method"] = "direct_text"
        return verified, details

    # Case 2: Try to extract option from prediction using can_infer
    extracted_option = can_infer(prediction, choices)

    if extracted_option and extracted_option != "Z":
        details["extracted_option"] = extracted_option
        details["method"] = "can_infer"

        # Compare with gold option letter if available
        if answer_option:
            verified = extracted_option == answer_option
        else:
            # Fallback: compare with gold answer content
            extracted_content = choices.get(extracted_option, "")
            verified = extracted_content.strip().lower() == gold_answer.strip().lower()

        return verified, details

    # Case 3: Fallback to parse_multi_choice_response
    all_choices = list(choices.keys())
    extracted_option = parse_multi_choice_response(prediction, all_choices, choices)
    details["extracted_option"] = extracted_option
    details["method"] = "parse_multi_choice"

    if answer_option:
        verified = extracted_option == answer_option
    else:
        extracted_content = choices.get(extracted_option, "")
        verified = extracted_content.strip().lower() == gold_answer.strip().lower()

    return verified, details
