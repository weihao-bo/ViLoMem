"""Node functions for Qwen VL baseline agent (no memory).

This module contains simplified node logic without memory functionality.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from common.prompt_utils import build_combined_prompt
from common.retry import async_retry_with_backoff
from common.utils import get_message_text, load_chat_model
from common.verification import verify_prediction
from vl_agent_baseline.context import Context
from vl_agent_baseline.state import EvaluationState, State

# Get logger
logger = logging.getLogger("qwen_vl_eval.nodes")

# ========== Retry-wrapped Model Invocation Helper ==========


@async_retry_with_backoff(max_retries=3, initial_delay=1.0)
async def _invoke_model_with_retry(model: Any, messages: list) -> AIMessage:
    """Invoke model with automatic retry on transient failures.

    Args:
        model: Configured LangChain chat model
        messages: List of messages to send to the model

    Returns:
        AIMessage response from the model
    """
    response = await model.ainvoke(messages)
    if not isinstance(response, AIMessage):
        raise ValueError(f"Expected AIMessage, got {type(response)}")
    return response


# ========== Node Functions ==========


async def call_model(
    state: State | EvaluationState, runtime: Runtime[Context]
) -> Dict[str, AIMessage | str | None]:
    """Invoke the vision-language model for math reasoning.

    This node:
    1. Loads the configured VL model
    2. Formats the question using the fixed reasoning prompt (without memory)
    3. Invokes the model with the question and any attached images
    4. Returns the model's response

    Args:
        state: Current state containing messages and question
        runtime: Runtime context with model configuration

    Returns:
        Dictionary with the model's response message and optional prediction
    """
    # Load the configured vision-language model
    base_model = load_chat_model(runtime.context.model)

    # Configure model parameters from context
    configured_model = base_model
    if runtime.context.temperature is not None:
        configured_model = configured_model.bind(
            temperature=runtime.context.temperature
        )  # type: ignore
    if runtime.context.max_tokens is not None:
        configured_model = configured_model.bind(max_tokens=runtime.context.max_tokens)  # type: ignore

    # Get the last message which should be a HumanMessage with the question
    last_message = state.messages[-1] if state.messages else None

    if not last_message or not isinstance(last_message, HumanMessage):
        raise ValueError(
            "Expected a HumanMessage with the question in the last message position"
        )

    # ========== [PRIORITY 2] Reconstruct prompt if benchmark config available ==========
    # Fallback when VLMEvalKit prompt is not available
    if (
        isinstance(state, EvaluationState)
        and state.benchmark_config
        and state.sample_metadata
    ):
        from tools.prompt_builder import reconstruct_question_with_options

        # Reconstruct question text with options, hints, CoT, etc.
        reconstructed_question = reconstruct_question_with_options(
            {
                "inputs": {"question": state.question or ""},
                "metadata": state.sample_metadata,
            },
            state.benchmark_config,
        )

        # Extract images from original message content
        original_content = last_message.content
        image_blocks = []
        if isinstance(original_content, list):
            for item in original_content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_blocks.append(item)

        # Rebuild message with images + reconstructed question
        new_content_list: list[dict[Any, Any]] = []
        # Add images first (following VLMEvalKit pattern)
        new_content_list.extend(image_blocks)
        # Add reconstructed question text
        new_content_list.append({"type": "text", "text": reconstructed_question})

        # Create new message with reconstructed content
        last_message = HumanMessage(content=new_content_list)

    vlmeval_prompt_data = None
    if isinstance(state, EvaluationState) and state.sample_metadata:
        vlmeval_prompt_data = state.sample_metadata.get("vlmeval_prompt")

    question_text = getattr(state, "question", "") or ""
    system_prompt_text = runtime.context.system_prompt or ""
    if system_prompt_text:
        system_prompt_text = system_prompt_text.replace("{question}", question_text)
    system_section = system_prompt_text if system_prompt_text.strip() else None

    message_to_send, prompt_meta, combined_text = build_combined_prompt(
        original_content=last_message.content,
        vlmeval_prompt=vlmeval_prompt_data,
        system_section=system_section,
    )

    response = await _invoke_model_with_retry(configured_model, [message_to_send])

    result: Dict[str, AIMessage | str | None] = {"messages": [response]}  # type: ignore
    if isinstance(state, EvaluationState):
        prediction = (
            get_message_text(response) if isinstance(response, AIMessage) else ""
        )
        result["prediction"] = prediction
        result["model_user_prompt"] = combined_text or get_message_text(message_to_send)
        result["model_system_prompt"] = system_prompt_text or None
        result["model_prompt_metadata"] = prompt_meta

    return result


async def verify_answer(
    state: State | EvaluationState, runtime: Runtime[Context]
) -> Dict[str, Any]:
    """Verify the model's answer against the gold answer.

    This node:
    1. Extracts the prediction from the last AI message
    2. Verifies using VLMEvalKit logic for MCQ or math-verify for other types
    3. Updates the state with verification results and matching metadata

    Args:
        state: Current state containing messages and gold_answer
        runtime: Runtime context with verification settings

    Returns:
        Dictionary with verification results (verified, extracted_option, method, etc.)
    """
    # Skip verification if disabled or no gold answer provided
    if not runtime.context.enable_verification or state.gold_answer is None:
        return {}

    # Extract prediction - prefer state.prediction if available (EvaluationState)
    if isinstance(state, EvaluationState) and state.prediction:
        prediction = state.prediction
    else:
        # Extract from messages (State)
        ai_messages = [msg for msg in state.messages if isinstance(msg, AIMessage)]
        if not ai_messages:
            return {
                "verified": False,
                "verification_error": "No AI message found in conversation",
            }
        prediction = get_message_text(ai_messages[-1])

    sample_metadata = (
        state.sample_metadata if isinstance(state, EvaluationState) else None
    )
    benchmark_config = (
        state.benchmark_config if isinstance(state, EvaluationState) else None
    )

    verification = verify_prediction(
        prediction=prediction,
        gold_answer=state.gold_answer,
        sample_metadata=sample_metadata,
        benchmark_config=benchmark_config,
        question=state.question if isinstance(state, EvaluationState) else None,
    )

    result: Dict[str, Any] = {
        "verified": verification["verified"],
        "verification_method": verification.get("verification_method"),
        "verification_error": verification.get("verification_error"),
        "verification_attempts": verification.get("verification_attempts", []),
        "extracted_answer": verification.get("extracted_answer"),
    }

    if verification.get("extracted_option"):
        result["extracted_option"] = verification["extracted_option"]

    if not isinstance(state, EvaluationState):
        result["prediction"] = prediction

    return result


async def save_output(
    state: EvaluationState, runtime: Runtime[Context]
) -> Dict[str, None]:
    """Append evaluation result to the results JSON file.

    This node:
    1. Prepares a result entry with all evaluation metadata
    2. Reads existing results file or creates new structure
    3. Appends the new result and updates summary statistics
    4. Writes back to file with streaming updates

    Args:
        state: Evaluation state with all metadata and results
        runtime: Runtime context (unused but required by signature)

    Returns:
        Empty dictionary (no state updates needed)
    """
    if not state.output_dir:
        return {}

    output_path = Path(state.output_dir)
    results_file = output_path / "results.json"

    # Extract model input components from messages (for reproducibility)
    # This shows how the prompt was constructed for the model
    model_input_text = None
    user_prompt = None  # The actual user message sent to the model

    if state.messages:
        for msg in state.messages:
            if isinstance(msg, HumanMessage):
                # Extract text content from the message
                if isinstance(msg.content, list):
                    # Multimodal message: extract text parts
                    text_parts = [
                        item.get("text", "")
                        for item in msg.content
                        if isinstance(item, dict) and item.get("type") == "text"
                    ]
                    user_prompt = "\n".join(text_parts)
                elif isinstance(msg.content, str):
                    user_prompt = msg.content
                break  # Use first HumanMessage

    actual_user_prompt = state.model_user_prompt or user_prompt or ""
    actual_system_prompt = state.model_system_prompt or ""

    model_input_info = {
        "system_prompt": actual_system_prompt,
        "user_prompt": actual_user_prompt,
        "original_question": state.question,
        "prompt_metadata": state.model_prompt_metadata,
    }

    result_entry = {
        "example_id": state.example_id,
        "question": state.question,
        "image_path": state.image_path,
        "task": state.task,
        "prediction": state.prediction,
        "gold_answer": state.gold_answer,
        "verified": state.verified,
        "verification_error": state.verification_error,
        "timestamp": datetime.now().isoformat(),
    }

    # Add prompt composition for reproducibility
    # This shows exactly what was sent to the model and how it was constructed
    result_entry["model_input"] = model_input_info

    # Add MCQ-specific metadata if available
    if state.sample_metadata and state.sample_metadata.get("choices"):
        result_entry["choices"] = state.sample_metadata["choices"]
        result_entry["answer_option"] = state.sample_metadata.get("answer_option")

    # Add verification metadata
    if state.extracted_answer is not None:
        result_entry["extracted_answer"] = state.extracted_answer
    if state.extracted_option:
        result_entry["extracted_option"] = state.extracted_option
    if state.verification_method:
        result_entry["verification_method"] = state.verification_method
    if state.verification_attempts:
        result_entry["verification_attempts"] = state.verification_attempts

    result_entry["model_call"] = {
        "input": model_input_info,
        "output": state.prediction,
    }

    # Read existing data or create new structure
    if results_file.exists():
        with open(results_file, encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(
                    "Corrupted results file at %s detected; reinitializing.",
                    results_file,
                )
                data = {"summary": {}, "results": []}
    else:
        data = {"summary": {}, "results": []}

    # Append new result
    data["results"].append(result_entry)

    # Update summary statistics
    total = len(data["results"])
    verified = sum(1 for r in data["results"] if r.get("verified"))

    data["summary"].update(
        {
            "total_examples": total,
            "verified_count": verified,
            "accuracy": verified / total if total > 0 else 0,
            "memory_enabled": False,  # Baseline agent has no memory
        }
    )

    # Write back to file
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {}
