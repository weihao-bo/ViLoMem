"""LLM-based answer verification as fallback verifier."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from common.utils import load_chat_model, strip_reasoning_tags

logger = logging.getLogger(__name__)


LLM_JUDGE_PROMPT = """You are an expert answer verification judge. Your task is to determine if a prediction matches the gold answer.

**CRITICAL RULE: Everything is based on the Gold Answer - IGNORE reasoning quality!**

**Core Principle**:
- If the extracted answer from prediction EXACTLY MATCHES the gold answer → verified=true
- If they don't match (even by 1 character/number) → verified=false
- DO NOT consider whether the prediction's reasoning is correct or makes sense
- ONLY compare the final extracted answers

**Verification Steps:**

1. **Identify the Gold Answer format**:
   - Single letter (A/B/C/D/E)? → Multiple choice (compare letters only)
   - Number? → Numerical answer (compare numeric values, ignore formatting like 7.0 vs 7)
   - Text? → Text answer (compare semantic meaning)

2. **Extract the final answer from Prediction**:

   **For Multiple Choice (Gold = letter)**:
   - Look for the FINAL answer letter in prediction (often after "Final Answer:", "Answer:", or at the end)
   - ONLY extract the letter (A/B/C/D/E), ignore the full text
   - Example: "Therefore C. Text text... **Final Answer: C**" → extract "C"
   - Compare: prediction_letter == gold_letter

   **For Numbers (Gold = number)**:
   - Look for the FINAL numerical answer in prediction (often after "Final Answer:", or in **bold**)
   - Extract ONLY the number, ignore units/text
   - Example: "Therefore the string will be cut into 2 pieces. **Final Answer: 2**" → extract 2
   - Example: "Gold = 9" → They don't match → verified=false
   - Compare: prediction_number == gold_number (ignore .0 differences)

   **For Text (Gold = text/phrase)**:
   - Extract the final answer text (may be a sentence fragment)
   - Compare semantic meaning (allow minor wording differences)
   - Example: "Yes, the baby is crawling right" matches "Yes"

3. **Critical Rules**:
   - ✓ ONLY compare the FINAL extracted answers
   - ✗ DO NOT judge if the prediction's reasoning is correct
   - ✗ DO NOT use your knowledge to decide if an answer is "reasonable"
   - ✗ DO NOT give credit for "close" answers (2 ≠ 9, C ≠ A)
   - ✓ STRICTLY follow: match → true, no match → false

**Examples**:

Example 1 (Number mismatch):
- Prediction: "The string forms a loop... Therefore 2 pieces. **Final Answer: 2**"
- Gold: "9"
- Reasoning: "Prediction says 2, Gold says 9. Numbers don't match."
- Verified: false

Example 2 (Letter mismatch):
- Prediction: "C. The catcher is behind the plate. **Final Answer: C**"
- Gold: "A"
- Reasoning: "Prediction chose C, Gold is A. Letters don't match."
- Verified: false

Example 3 (Semantic match):
- Prediction: "Yes, the baby is crawling to the right."
- Gold: "Yes"
- Reasoning: "Prediction contains 'Yes', Gold is 'Yes'. Match."
- Verified: true

Now verify this case:

**Question:** {question}

**Gold Answer:** {gold_answer}

{choices_text}

**Prediction:** {prediction}

Respond with ONLY a JSON object in this exact format:
{{
  "reasoning": "Step 1: Extract answer from prediction: [extracted_value]. Step 2: Compare with gold: [gold_value]. Step 3: Match result: [yes/no].",
  "verified": true or false
}}"""


def create_llm_judge_prompt(
    question: str,
    prediction: str,
    gold_answer: str,
    choices: dict[str, str] | None = None,
) -> str:
    """Create a prompt for LLM-based verification.

    Args:
        question: The question being asked
        prediction: The model's prediction
        gold_answer: The correct answer
        choices: Optional dictionary of answer choices (e.g., {"A": "text", "B": "text"})

    Returns:
        Formatted prompt string
    """
    choices_text = ""
    if choices:
        choices_text = "**Answer Choices:**\n"
        for key, value in choices.items():
            choices_text += f"{key}. {value}\n"
        choices_text += "\n"

    return LLM_JUDGE_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        choices_text=choices_text,
        prediction=prediction,
    )


def llm_verify_answer(
    question: str,
    prediction: str,
    gold_answer: str,
    choices: dict[str, str] | None = None,
    judge_model_name: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Use LLM to verify if prediction matches gold answer.

    This is a fallback verification method when rule-based methods fail.

    Args:
        question: The question being asked
        prediction: The model's prediction
        gold_answer: The correct answer
        choices: Optional dictionary of answer choices
        judge_model_name: Optional judge model name (provider:model format)
                         If not provided, uses VLMEVAL_JUDGE_MODEL env var or defaults

    Returns:
        Tuple of (verified: bool, details: dict)
    """
    # Determine judge model
    if judge_model_name is None:
        judge_model_name = os.getenv("VLMEVAL_JUDGE_MODEL", "qwen:qwen-plus")

    try:
        # Load judge model
        judge_model = load_chat_model(judge_model_name)

        # Create prompt
        prompt = create_llm_judge_prompt(question, prediction, gold_answer, choices)

        # Invoke judge model
        messages = [
            SystemMessage(content="You are an expert answer verification judge."),
            HumanMessage(content=prompt),
        ]

        response = judge_model.invoke(messages)

        # Handle response content (may be str or list)
        if isinstance(response.content, str):
            response_text = strip_reasoning_tags(response.content.strip())
        else:
            # If content is a list, join text parts
            response_text = " ".join(
                item if isinstance(item, str) else str(item)
                for item in response.content
            ).strip()
            response_text = strip_reasoning_tags(response_text)

        # Parse JSON response
        # Handle markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)
        verified = result.get("verified", False)
        reasoning = result.get("reasoning", "No reasoning provided")

        return bool(verified), {
            "method": "llm_judge",
            "verified": bool(verified),
            "reasoning": reasoning,
            "judge_model": judge_model_name,
            "extracted_answer": None,  # LLM judge doesn't extract structured answer
            "error": None,
        }

    except Exception as e:
        logger.debug("LLM judge verification failed: %s", e)
        return False, {
            "method": "llm_judge",
            "verified": False,
            "reasoning": None,
            "judge_model": judge_model_name,
            "extracted_answer": None,
            "error": f"llm_judge_failed: {str(e)}",
        }
