"""Node functions for Qwen VL agent graphs.

This module contains all the node logic for both the main agent graph and evaluation graph.
Each node is a standalone async function that can be composed into different workflows.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from common.prompt_utils import build_combined_prompt
from common.utils import get_message_text, load_chat_model
from common.verification import verify_prediction
from vl_agent.context import (
    ERROR_ANALYSIS_PROMPT,
    PROBLEM_ANALYSIS_PROMPT,
    VISUAL_KEYWORD_EXTRACTION_PROMPT,
    Context,
)
from vl_agent.memory import (
    load_memories,
    parse_error_analysis,
    parse_problem_analysis,
    save_memory,
    update_memory_usage,
)
from vl_agent.state import EvaluationState, State

# Get logger
logger = logging.getLogger("qwen_vl_eval.nodes")

# Global lock for thread-safe file operations (concurrent execution)
_results_file_lock = asyncio.Lock()


def _compose_memory_section(state: EvaluationState) -> str | None:
    sections: list[str] = []

    if state.retrieved_visual_memories:
        visual_memories_text = "\n".join(
            f"- {m}" for m in state.retrieved_visual_memories
        )
        sections.append(f"Visual Guidelines:\n{visual_memories_text}")

    if state.retrieved_memories:
        logic_memories_text = "\n".join(f"- {m}" for m in state.retrieved_memories)
        sections.append(f"Logic Guidelines:\n{logic_memories_text}")

    if not sections:
        return None

    return (
        "IMPORTANT: Guidelines from Past Experience (consider these carefully when solving the problem):\n"
        + "\n\n".join(sections)
    )


async def call_model(
    state: State | EvaluationState, runtime: Runtime[Context]
) -> Dict[str, AIMessage | str | None]:
    """Invoke the vision-language model for math reasoning.

    Prompt Construction Strategy:
    =============================
    1. [Priority] If vlmeval_prompt exists in sample_metadata, use it
       - VLMEvalKit official format (highest priority for benchmark compatibility)
       - Replaces the entire message content

    2. [Default] Otherwise, use the message constructed by build_message_content
       - If benchmark_config provided: uses benchmark-specific prompt (options, hints, CoT)
       - If no benchmark_config: uses system_prompt with {question} placeholder

    Memory Injection:
    =================
    - Retrieved memories (logic + visual) are injected AFTER the prompt
    - Format: "IMPORTANT: Guidelines from Past Experience..."
    - Appended to the last text block in the message

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

    vlmeval_prompt_data = None
    if isinstance(state, EvaluationState) and state.sample_metadata:
        vlmeval_prompt_data = state.sample_metadata.get("vlmeval_prompt")

    memory_section = None
    if isinstance(state, EvaluationState):
        memory_section = _compose_memory_section(state)

    message_to_send: HumanMessage
    prompt_meta: dict[str, Any]
    combined_text: str

    question_text = getattr(state, "question", "") or ""
    system_prompt_text = runtime.context.system_prompt
    if system_prompt_text:
        system_prompt_text = system_prompt_text.replace("{question}", question_text)

    system_section = (
        system_prompt_text
        if system_prompt_text and system_prompt_text.strip()
        else None
    )

    message_to_send, prompt_meta, combined_text = build_combined_prompt(
        original_content=last_message.content,
        vlmeval_prompt=vlmeval_prompt_data,
        system_section=system_section,
        memory_section=memory_section,
    )

    # Configure model with retry
    model_with_retry = configured_model.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )

    try:
        response = await model_with_retry.ainvoke([message_to_send])
    except Exception as exc:
        logger.error(
            f"[call_model] Model invocation failed after 3 retries "
            f"(model: {runtime.context.model}): {exc}"
        )
        raise  # Re-raise to stop the workflow

    # Validate response type
    if not isinstance(response, AIMessage):
        raise ValueError(f"Expected AIMessage, got {type(response)}")

    # Extract prediction for EvaluationState
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
    2. For MCQ tasks: Maps option letters to actual values using choices metadata
    3. Compares prediction with gold answer using math-verify
    4. Updates the state with verification results

    Args:
        state: Current state containing messages and gold_answer
        runtime: Runtime context with verification settings

    Returns:
        Dictionary with verification results (prediction, verified, verification_error)
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

    # Prepare result entry
    actual_user_prompt = state.model_user_prompt or ""
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
        # Logic Memory fields
        "error_type": state.error_type if state.error_type else "",
        "retrieved_logic_memories": "\n".join(state.retrieved_memories)
        if state.retrieved_memories
        else "",
        "generated_logic_memory": state.new_memory_guideline
        if state.new_memory_guideline
        else "",
        # Visual Memory fields
        "is_visual_error": state.is_visual_error
        if state.is_visual_error is not None
        else None,
        "retrieved_visual_memories": "\n".join(state.retrieved_visual_memories)
        if state.retrieved_visual_memories
        else "",
        "generated_visual_memory": state.new_visual_memory_guideline
        if state.new_visual_memory_guideline
        else "",
        # Heatmap fields
        "heatmap_text": state.heatmap_text if state.heatmap_text else "",
        "heatmap_method": state.heatmap_method if state.heatmap_method else "",
    }

    result_entry["model_input"] = model_input_info
    result_entry["model_call"] = {
        "input": model_input_info,
        "output": state.prediction,
    }

    # Add verification metadata
    if state.extracted_answer is not None:
        result_entry["extracted_answer"] = state.extracted_answer
    if state.extracted_option:
        result_entry["extracted_option"] = state.extracted_option
    if state.verification_method:
        result_entry["verification_method"] = state.verification_method
    if state.verification_attempts:
        result_entry["verification_attempts"] = state.verification_attempts

    # Thread-safe file operations for concurrent execution
    async with _results_file_lock:
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

        # Logic memory statistics
        logic_memories_generated = sum(
            1 for r in data["results"] if r.get("generated_logic_memory", "").strip()
        )
        logic_memories_retrieved_count = sum(
            1 for r in data["results"] if r.get("retrieved_logic_memories", "").strip()
        )

        # Visual memory statistics
        visual_memories_generated = sum(
            1 for r in data["results"] if r.get("generated_visual_memory", "").strip()
        )
        visual_memories_retrieved_count = sum(
            1 for r in data["results"] if r.get("retrieved_visual_memories", "").strip()
        )

        data["summary"].update(
            {
                "total_examples": total,
                "verified_count": verified,
                "accuracy": verified / total if total > 0 else 0,
                "logic_memory_retrieval_enabled": runtime.context.logic_memory_enable_retrieval,
                "logic_memory_generation_enabled": runtime.context.logic_memory_enable_generation,
                "visual_memory_retrieval_enabled": runtime.context.visual_memory_enable_retrieval,
                "visual_memory_generation_enabled": runtime.context.visual_memory_enable_generation,
                # Logic memory stats
                "total_logic_memories_generated": logic_memories_generated,
                "total_logic_memories_retrieved": logic_memories_retrieved_count,
                # Visual memory stats
                "total_visual_memories_generated": visual_memories_generated,
                "total_visual_memories_retrieved": visual_memories_retrieved_count,
            }
        )

        # Write back to file
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return {}


# ========== Memory-Related Nodes ==========


async def retrieve_logic_memories(
    state: EvaluationState, runtime: Runtime[Context]
) -> Dict[str, str | list[str]]:
    """Retrieve relevant memories using DashScope Rerank API.

    This node:
    1. Uses the problem_analysis from state (generated by initialize_case node)
    2. Constructs a search query combining question and analysis
    3. Loads all memories from JSON file
    4. Uses DashScope Rerank API to rank memories by relevance
    5. Filters results by similarity threshold
    6. Updates usage counts for retrieved memories

    Args:
        state: Evaluation state with question and problem_analysis
        runtime: Runtime context with retrieval configuration

    Returns:
        Dictionary with retrieved_memories and retrieved_memory_ids
    """
    # Log configuration status
    logger.info("[Logic Memory Retrieval] Configuration:")
    logger.info(
        f"  - Enable retrieval: {runtime.context.logic_memory_enable_retrieval}"
    )
    logger.info(f"  - Embedding model: {runtime.context.logic_memory_embedding_model}")
    logger.info(
        f"  - Similarity threshold: {runtime.context.logic_memory_similarity_threshold}"
    )
    logger.info(f"  - Retrieval limit: {runtime.context.logic_memory_retrieval_limit}")

    # Skip if logic memory retrieval is disabled
    if not runtime.context.logic_memory_enable_retrieval:
        return {
            "retrieved_memories": [],
            "retrieved_memory_ids": [],
        }

    # Get question text and problem analysis (from initialize_case node)
    question = state.question or ""
    problem_analysis = state.problem_analysis or ""

    if not question or not problem_analysis:
        return {
            "retrieved_memories": [],
            "retrieved_memory_ids": [],
        }

    # ========== Retrieve Memories ==========
    # Skip retrieval if no output directory
    if not state.output_dir:
        return {
            "retrieved_memories": [],
            "retrieved_memory_ids": [],
        }

    # Resolve memory file path
    memory_file = Path(state.output_dir) / runtime.context.logic_memory_file_path
    logger.info(f"Loading logic memories from: {memory_file}")
    logger.info(f"Memory file exists: {memory_file.exists()}")

    # Load all memories from JSON
    memories = load_memories(memory_file)
    logger.info(f"Loaded {len(memories)} logic memories from storage")

    if not memories:
        if runtime.context.memory_list:
            logger.warning(
                "No logic memories found at %s even though memory_list=%s was provided. "
                "Check that the earlier runs produced logic memories.",
                memory_file,
                runtime.context.memory_list,
            )
        else:
            logger.info(
                "No logic memories found in %s (fresh run will build new memories).",
                memory_file,
            )
        return {
            "retrieved_memories": [],
            "retrieved_memory_ids": [],
        }

    # Parse problem analysis
    subject, key_concepts = parse_problem_analysis(problem_analysis)
    logger.info(f"Problem analysis - Subject: {subject}, Key Concepts: {key_concepts}")

    # Construct search query
    concepts_str = ", ".join(key_concepts) if key_concepts else "N/A"
    query = f"{state.question}\n\nSubject: {subject}\nKey Concepts: {concepts_str}"
    logger.info(f"Embedding query: {query[:200]}...")  # Log first 200 chars

    # Retrieve memories using text embedding similarity
    try:
        logger.info(
            f"Retrieving from {len(memories)} logic memories using {runtime.context.logic_memory_embedding_model}"
        )
        logger.info(f"Top N: {runtime.context.logic_memory_retrieval_limit}")
        logger.info(
            f"Similarity threshold: {runtime.context.logic_memory_similarity_threshold}"
        )

        from vl_agent.memory import retrieve_memories_by_text_embedding

        filtered_results = await retrieve_memories_by_text_embedding(
            query=query,
            memories=memories,
            memory_type="logic",
            output_dir=Path(state.output_dir),
            model=runtime.context.logic_memory_embedding_model,
            top_n=runtime.context.logic_memory_retrieval_limit,
            similarity_threshold=runtime.context.logic_memory_similarity_threshold,
        )
        logger.info(f"Retrieved {len(filtered_results)} logic memories after filtering")
    except Exception as exc:
        # Log error but continue with empty results
        logger.warning(
            f"[retrieve_logic_memories] Embedding retrieval failed after retries "
            f"(model: {runtime.context.logic_memory_embedding_model}): {exc}"
        )
        return {
            "retrieved_memories": [],
            "retrieved_memory_ids": [],
        }

    # Log all returned scores for debugging
    if filtered_results:
        all_scores = [score for _, score in filtered_results]
        logger.info(f"All similarity scores: {all_scores}")
        logger.info(
            f"Max score: {max(all_scores):.4f}, Min score: {min(all_scores):.4f}"
        )
    else:
        logger.debug(
            "No memories passed similarity threshold (%.2f) from %d candidates",
            runtime.context.logic_memory_similarity_threshold,
            len(memories),
        )

    # Extract guidelines and IDs, update usage counts
    guidelines = []
    memory_ids = []
    timestamp = datetime.now().isoformat()
    for memory, score in filtered_results:
        if "guideline" in memory:
            guidelines.append(memory["guideline"])
            memory_id = memory.get("memory_id", "")
            if memory_id:
                memory_ids.append(memory_id)
                # Update usage count in JSON file
                update_memory_usage(memory_file, memory_id, timestamp)
            logger.debug(
                f"Retrieved logic memory (score={score:.4f}): {memory['guideline'][:100]}..."
            )

    logger.info(f"Retrieved {len(guidelines)} logic memories for this question")

    return {
        "retrieved_memories": guidelines,
        "retrieved_memory_ids": memory_ids,
    }


async def generate_logic_memory(
    state: EvaluationState, runtime: Runtime[Context]
) -> Dict[str, str]:
    """Analyze the error to generate a guideline for future reference.

    This node is only called when verification fails (verified=False).

    This node:
    1. Extracts the incorrect reasoning from the prediction
    2. Calls the memory generation model with ERROR_ANALYSIS_PROMPT
    3. Parses the output to extract error type, summary, and guideline
    4. Returns the analysis results

    Args:
        state: Evaluation state with prediction, question, and gold_answer
        runtime: Runtime context with memory generation model configuration

    Returns:
        Dictionary with error_analysis, error_type, and new_memory_guideline
    """
    # Skip if logic memory generation is disabled
    if not runtime.context.logic_memory_enable_generation:
        return {
            "error_analysis": "",
            "error_type": "Non-Logical",
            "new_memory_guideline": "",
        }

    # Skip if answer is correct
    if state.verified:
        return {
            "error_analysis": "",
            "error_type": "Non-Logical",
            "new_memory_guideline": "",
        }

    # Extract reasoning steps
    reasoning_steps = state.prediction or ""
    question = state.question or ""
    gold_answer = state.gold_answer or ""

    # Load memory generation model
    model = load_chat_model(runtime.context.logic_memory_generation_model)
    if runtime.context.logic_memory_generation_temperature is not None:
        model = model.bind(
            temperature=runtime.context.logic_memory_generation_temperature
        )  # type: ignore
    if runtime.context.logic_memory_generation_max_tokens is not None:
        model = model.bind(
            max_tokens=runtime.context.logic_memory_generation_max_tokens
        )  # type: ignore

    # Create prompt
    prompt = ERROR_ANALYSIS_PROMPT.format(
        question=question,
        reasoning_steps=reasoning_steps,
        gold_answer=gold_answer,
    )

    # Configure model with retry and invoke
    model_with_retry = model.with_retry(stop_after_attempt=3)
    try:
        response = await model_with_retry.ainvoke([HumanMessage(content=prompt)])
        analysis_text = get_message_text(response)
    except Exception as exc:
        logger.warning(
            f"[generate_logic_memory] Error analysis failed after 3 retries "
            f"(model: {runtime.context.logic_memory_generation_model}): {exc}"
        )
        # Return empty results on error
        return {
            "error_analysis": "",
            "error_type": "Non-Logical",
            "new_memory_guideline": "",
        }

    # Parse error analysis
    error_type, _analysis_summary, guideline = parse_error_analysis(analysis_text)

    # ========== Store Memory (if logical error and guideline exists) ==========
    generated_memory_id = ""

    # Only store if it's a logical error with a guideline and we have output_dir
    if error_type == "Logical" and guideline and state.output_dir:
        # Parse problem analysis
        subject, key_concepts = parse_problem_analysis(state.problem_analysis or "")

        # Resolve memory file path
        memory_file = Path(state.output_dir) / runtime.context.logic_memory_file_path

        # Load existing memories for similarity check
        existing_memories = load_memories(memory_file)

        # Create new memory
        memory_id = str(uuid.uuid4())

        # Create memory entry
        memory_data = {
            # Core identifier
            "memory_id": memory_id,
            # Core content
            "guideline": guideline,
            "error_type": error_type,
            # Problem analysis info
            "subject": subject,
            "key_concepts": key_concepts,
            # Metadata
            "created_at": datetime.now().isoformat(),
            "usage_count": 0,
            "last_used_at": None,
            # Source tracking
            "source_question": question,
            "source_example_id": state.example_id,
            "source_image_path": state.image_path,
        }

        # Save new memory
        save_memory(memory_file, memory_data)
        generated_memory_id = memory_id

        # Compute and cache text embedding for the guideline
        try:
            from vl_agent.memory import get_or_compute_text_embedding

            embedding = await get_or_compute_text_embedding(
                text=guideline,
                memory_id=memory_id,
                memory_type="logic",
                output_dir=Path(state.output_dir),
                model=runtime.context.logic_memory_embedding_model,
            )
            if embedding:
                logger.info(f"✓ Cached text embedding for logic memory {memory_id}")
            else:
                logger.error(
                    f"❌ Failed to cache text embedding for logic memory {memory_id}. "
                    f"This memory will not be retrievable until embedding is computed manually."
                )
        except Exception as e:
            logger.error(
                f"❌ Error caching text embedding for logic memory {memory_id}: {e}. "
                f"Check if embedding model is running at {runtime.context.logic_memory_embedding_model}"
            )

    return {
        "error_analysis": analysis_text,
        "error_type": error_type,
        "new_memory_guideline": guideline,
        "generated_memory_id": generated_memory_id,
    }


# ========== Visual Memory Nodes ==========


async def initialize_case(
    state: EvaluationState, runtime: Runtime[Context]
) -> Dict[str, list[str] | str]:
    """Extract image URLs and analyze problem for memory retrieval.

    This node:
    1. Iterates through all messages in state.messages
    2. Extracts image URLs or base64-encoded images from multimodal content
    3. Analyzes the problem to identify subject and key concepts (for memory retrieval)
    4. Stores image_urls and problem_analysis in state

    Args:
        state: Evaluation state with messages and question
        runtime: Runtime context with analysis model configuration

    Returns:
        Dictionary with image_urls and problem_analysis fields
    """
    image_urls: list[str] = []

    # Iterate through messages to find images
    for message in state.messages:
        # Check if message has multimodal content (list format)
        if isinstance(message.content, list):
            for item in message.content:
                if isinstance(item, dict):
                    # Look for image_url type
                    if item.get("type") == "image_url":
                        image_data = item.get("image_url", {})
                        url = image_data.get("url", "")
                        if url:
                            image_urls.append(url)
                    # Also check for "image" type (some formats use this)
                    elif item.get("type") == "image":
                        url = item.get("source", {}).get("url", "")
                        if url:
                            image_urls.append(url)

    # ========== Analyze Problem ==========
    # Only analyze if we have a question (this analysis is used by both logic and visual memory retrieval)
    problem_analysis = ""
    question = state.question or ""

    if question:
        # Load analysis model
        model = load_chat_model(runtime.context.analysis_model)
        if runtime.context.analysis_temperature is not None:
            model = model.bind(temperature=runtime.context.analysis_temperature)  # type: ignore
        if runtime.context.analysis_max_tokens is not None:
            model = model.bind(max_tokens=runtime.context.analysis_max_tokens)  # type: ignore

        # Create analysis prompt
        analysis_prompt = PROBLEM_ANALYSIS_PROMPT.format(question=question)

        # Configure model with retry
        model_with_retry = model.with_retry(stop_after_attempt=3)

        try:
            # Invoke model
            response = await model_with_retry.ainvoke(
                [HumanMessage(content=analysis_prompt)]
            )
            problem_analysis = get_message_text(response)
            logger.info(f"[Problem Analysis] {problem_analysis}")
        except Exception as exc:
            logger.warning(
                f"[initialize_case] Problem analysis failed after 3 retries "
                f"(model: {runtime.context.analysis_model}): {exc}"
            )
            # Continue with empty analysis on error
            problem_analysis = ""

    return {"image_urls": image_urls, "problem_analysis": problem_analysis}


async def retrieve_visual_memories(
    state: EvaluationState, runtime: Runtime[Context]
) -> Dict[str, list[str] | list[Any]]:
    """Retrieve relevant visual memories using two-stage retrieval (optional).

    This node supports two retrieval modes:

    Mode 1: Single-stage (visual_memory_enable_text_rerank=False):
    1. Uses multimodal embeddings to retrieve top-N candidates
    2. Filters candidates by similarity threshold
    3. Returns filtered results

    Mode 2: Two-stage (visual_memory_enable_text_rerank=True):
    1. Stage 1: Uses multimodal embeddings to retrieve top-N candidates (visual_embedding_top_n)
    2. Stage 2: Uses text embedding similarity with Question to select top-M from candidates (visual_memory_retrieval_limit)
    3. Filters results by similarity threshold
    4. Returns filtered results

    Both modes update usage counts for retrieved memories.

    Args:
        state: Evaluation state with image_path, question, and problem_analysis
        runtime: Runtime context with visual memory configuration

    Returns:
        Dictionary with retrieved_visual_memories and retrieved_visual_memory_ids
    """
    # Skip if visual memory retrieval is disabled
    if not runtime.context.visual_memory_enable_retrieval:
        return {"retrieved_visual_memories": [], "retrieved_visual_memory_ids": []}

    # Skip if no images available
    if not state.image_urls:
        return {"retrieved_visual_memories": [], "retrieved_visual_memory_ids": []}

    # Skip if no output directory
    if not state.output_dir:
        return {"retrieved_visual_memories": [], "retrieved_visual_memory_ids": []}

    # Skip if no image_path (required for benchmark-scoped path construction)
    if not state.image_path:
        logger.warning("Visual memory retrieval skipped: state.image_path is empty")
        return {"retrieved_visual_memories": [], "retrieved_visual_memory_ids": []}

    # Resolve visual memory file path
    visual_memory_file = (
        Path(state.output_dir) / runtime.context.visual_memory_file_path
    )

    # Load all visual memories from JSON
    from vl_agent.memory import (
        load_memories,
        make_benchmark_scoped_path,
        retrieve_visual_memories_by_similarity,
        update_memory_usage,
    )

    visual_memories = load_memories(visual_memory_file)
    if not visual_memories:
        return {"retrieved_visual_memories": [], "retrieved_visual_memory_ids": []}

    # Determine benchmark configuration
    current_benchmark = runtime.context.current_benchmark
    if not current_benchmark:
        logger.warning(
            "Visual memory retrieval skipped: current_benchmark is empty. "
            "Set benchmark_root_dir and current_benchmark in Context."
        )
        return {"retrieved_visual_memories": [], "retrieved_visual_memory_ids": []}

    benchmark_root_dir = runtime.context.benchmark_root_dir
    if not benchmark_root_dir:
        logger.warning(
            "Visual memory retrieval skipped: benchmark_root_dir is empty. "
            "Set benchmark_root_dir in Context."
        )
        return {"retrieved_visual_memories": [], "retrieved_visual_memory_ids": []}

    # ========== Multi-Image Support: Retrieve for each image and merge results ==========
    # For multi-image problems, retrieve memories for each image separately
    # This ensures we find relevant memories associated with any of the images

    # Parse image_path - may contain multiple images separated by semicolon
    # Format: "15.jpg" or "15.jpg;16.jpg;17.jpg"
    image_filenames = [f.strip() for f in state.image_path.split(";") if f.strip()]

    if not image_filenames:
        logger.warning(
            "Visual memory retrieval skipped: no valid image filenames in state.image_path"
        )
        return {"retrieved_visual_memories": [], "retrieved_visual_memory_ids": []}

    logger.debug(
        f"Visual memory retrieval for {len(image_filenames)} image(s): {image_filenames}"
    )

    # Collect results from all images
    all_ranked_results: list[tuple[dict[str, Any], float]] = []

    for image_filename in image_filenames:
        # Construct benchmark-scoped path for this image
        query_image_path = make_benchmark_scoped_path(image_filename, current_benchmark)
        logger.debug(f"Querying visual memories for: {query_image_path}")

        # Retrieve memories using precomputed multimodal embedding similarity
        try:
            ranked_results = await retrieve_visual_memories_by_similarity(
                query_image_path=query_image_path,
                memories=visual_memories,
                root_dir=benchmark_root_dir,
                model=runtime.context.visual_embedding_model,
                top_n=runtime.context.visual_embedding_top_n,
            )
            all_ranked_results.extend(ranked_results)
        except Exception as exc:
            # Log error but continue with other images
            logger.warning(
                f"Visual memory retrieval failed for {query_image_path}: {exc}"
            )
            continue

    if not all_ranked_results:
        logger.debug("No visual memories retrieved for any image")
        return {"retrieved_visual_memories": [], "retrieved_visual_memory_ids": []}

    # Merge results: deduplicate by memory_id and keep highest score
    memory_scores: dict[str, tuple[dict[str, Any], float]] = {}
    for memory, score in all_ranked_results:
        memory_id = memory.get("memory_id", "")
        if not memory_id:
            continue

        # Keep highest score for each memory_id
        if memory_id not in memory_scores or score > memory_scores[memory_id][1]:
            memory_scores[memory_id] = (memory, score)

    # Sort by score (highest first) and limit to top_n
    embedding_candidates = sorted(
        memory_scores.values(), key=lambda x: x[1], reverse=True
    )[: runtime.context.visual_embedding_top_n]

    logger.debug(
        f"Stage 1 (Embedding): Retrieved {len(embedding_candidates)} unique visual memories "
        f"(from {len(all_ranked_results)} total matches across {len(image_filenames)} images)"
    )

    # ========== Stage 2: Text Embedding with Question (Optional) ==========
    if runtime.context.visual_memory_enable_text_rerank and embedding_candidates:
        logger.info("[Visual Memory Stage 2] Two-stage retrieval enabled")
        logger.info(f"  - Stage 1 candidates: {len(embedding_candidates)}")
        logger.info(
            f"  - Text embedding model: {runtime.context.visual_memory_text_embedding_model}"
        )
        logger.info(
            f"  - Target final count: {runtime.context.visual_memory_retrieval_limit}"
        )

        # Prepare question and analysis for text embedding query
        question = state.question or ""
        problem_analysis = state.problem_analysis or ""

        if question:
            # Construct text query (similar to logic memory)
            from vl_agent.memory import (
                parse_problem_analysis,
                retrieve_memories_by_text_embedding,
            )

            subject, key_concepts = parse_problem_analysis(problem_analysis)
            concepts_str = ", ".join(key_concepts) if key_concepts else "N/A"
            text_query = (
                f"{question}\n\nSubject: {subject}\nKey Concepts: {concepts_str}"
            )

            logger.debug(f"Stage 2 text query: {text_query[:200]}...")

            # Extract memories for Stage 2 (without scores)
            candidate_memories = [memory for memory, _score in embedding_candidates]

            # Call text embedding retrieval
            try:
                text_ranked_results = await retrieve_memories_by_text_embedding(
                    query=text_query,
                    memories=candidate_memories,
                    memory_type="visual",
                    output_dir=Path(state.output_dir),
                    model=runtime.context.visual_memory_text_embedding_model,
                    top_n=runtime.context.visual_memory_retrieval_limit,
                    similarity_threshold=runtime.context.visual_memory_similarity_threshold,
                )
                logger.info(
                    f"Stage 2 (Text Embedding): Returned {len(text_ranked_results)} results"
                )

                # Use text embedding results as final candidates
                ranked_results = text_ranked_results
            except Exception as exc:
                logger.warning(
                    f"[retrieve_visual_memories] Stage 2 text embedding failed "
                    f"(model: {runtime.context.visual_memory_text_embedding_model}): {exc}"
                )
                logger.warning("Falling back to Stage 1 embedding-only results")
                # Fall back to embedding results on error
                ranked_results = embedding_candidates
        else:
            logger.warning(
                "No question available for Stage 2, using Stage 1 results only"
            )
            ranked_results = embedding_candidates
    else:
        # Single-stage retrieval: use image embedding results directly
        ranked_results = embedding_candidates
        if not runtime.context.visual_memory_enable_text_rerank:
            logger.info(
                "[Visual Memory Retrieval] Single-stage (image embedding only) mode"
            )

    # Filter by similarity threshold
    filtered_results = [
        (memory, score)
        for memory, score in ranked_results
        if score >= runtime.context.visual_memory_similarity_threshold
    ]

    logger.info(
        f"After similarity threshold filtering ({runtime.context.visual_memory_similarity_threshold}): "
        f"{len(filtered_results)} visual memories"
    )

    # Extract guidelines and IDs, update usage counts
    guidelines = []
    memory_ids = []
    timestamp = datetime.now().isoformat()
    for memory, _score in filtered_results:
        if "guideline" in memory:
            guidelines.append(memory["guideline"])
            memory_id = memory.get("memory_id", "")
            if memory_id:
                memory_ids.append(memory_id)
                # Update usage count in JSON file
                update_memory_usage(visual_memory_file, memory_id, timestamp)

    # ========== Heatmap Generation (Optional) ==========
    heatmap_text = None
    heatmap_method = None
    updated_image_urls = None

    if guidelines and runtime.context.enable_heatmap_generation:
        logger.info("Heatmap generation enabled - preparing to generate attention map")

        # Step 1: Prepare text for heatmap generation
        if runtime.context.enable_keyword_extraction:
            logger.info("Extracting visual keywords from retrieved memory guidelines")
            heatmap_text = await extract_visual_keywords(
                guidelines=guidelines,
                runtime=runtime,
            )
            logger.info(f"Using extracted keywords for heatmap: {heatmap_text}")
        else:
            # Use raw guidelines (joined by newline)
            heatmap_text = "\n".join(guidelines)
            logger.debug(
                "Using raw guidelines for heatmap (keyword extraction disabled)"
            )

        # Step 1.5: Optionally prepend question to heatmap text
        if runtime.context.include_question_in_heatmap and state.question:
            heatmap_text = f"Question: {state.question}\n\n{heatmap_text}"
            logger.info("Prepended question to heatmap text")
            logger.debug(f"Final heatmap text: {heatmap_text[:200]}...")

        # Step 2: Generate heatmap using selected method
        heatmap_method = runtime.context.heatmap_method.lower()
        # Use the first image from state.image_urls (base64 URL)
        original_image_url = state.image_urls[0] if state.image_urls else None

        if not original_image_url:
            logger.warning("Cannot generate heatmap: no image URLs available")
        else:
            try:
                if heatmap_method == "grad-eclip":
                    from vl_agent.heatmap_grad_eclip import (
                        generate_grad_eclip_heatmap,
                    )

                    logger.info("Generating Grad-Eclip heatmap")
                    heatmap_overlaid_image_url = await generate_grad_eclip_heatmap(
                        image_url=original_image_url,
                        text=heatmap_text,
                        model_name=runtime.context.gradcam_model,
                        output_dir=state.output_dir,
                        example_id=state.example_id,
                        benchmark=runtime.context.current_benchmark,
                        debug=runtime.context.debug_heatmap,
                        device=runtime.context.gradcam_device,
                        withksim=True,  # Enable Q-K similarity weighting
                    )
                elif heatmap_method == "gradcam":
                    from vl_agent.heatmap_gradcam import generate_gradcam_heatmap

                    logger.info("Generating GradCAM heatmap")
                    heatmap_overlaid_image_url = await generate_gradcam_heatmap(
                        image_url=original_image_url,
                        text=heatmap_text,
                        model_name=runtime.context.gradcam_model,
                        output_dir=state.output_dir,
                        example_id=state.example_id,
                        benchmark=runtime.context.current_benchmark,
                        debug=runtime.context.debug_heatmap,
                        device=runtime.context.gradcam_device,
                    )
                elif heatmap_method == "qwen25vl-attention":
                    from vl_agent.heatmap_qwen25vl import (
                        generate_qwen25vl_attention_heatmap,
                    )

                    logger.info("Generating Qwen2.5-VL cross-attention heatmap")
                    device_spec = (
                        runtime.context.qwen25vl_devices
                        or runtime.context.qwen25vl_device
                    )
                    heatmap_overlaid_image_url = await generate_qwen25vl_attention_heatmap(
                        image_url=original_image_url,
                        text=heatmap_text,  # Supports long text (no 77 token limit!)
                        model_name=runtime.context.qwen25vl_model,
                        general_prompt=runtime.context.qwen25vl_general_prompt,
                        attention_layer=runtime.context.qwen25vl_attention_layer,
                        output_dir=state.output_dir,
                        example_id=state.example_id,
                        benchmark=runtime.context.current_benchmark,
                        debug=runtime.context.debug_heatmap,
                        device=device_spec,
                        per_device_max_parallel=runtime.context.qwen25vl_per_device_limit,
                    )
                else:
                    raise ValueError(
                        f"Invalid heatmap_method: '{heatmap_method}'. "
                        f"Supported methods: 'gradcam', 'grad-eclip', 'qwen25vl-attention'"
                    )

                logger.info(
                    f"Heatmap generated successfully using {heatmap_method} method"
                )

                # Step 3: Add heatmap to image_urls (will be second image)
                updated_image_urls = list(state.image_urls)
                if heatmap_overlaid_image_url not in updated_image_urls:
                    updated_image_urls.append(heatmap_overlaid_image_url)
                    logger.debug(
                        f"Added heatmap image to image_urls: {heatmap_overlaid_image_url[:100]}..."
                    )

            except Exception as exc:
                logger.warning(f"Failed to generate heatmap: {exc}")
                # Continue without heatmap - only use text injection
                heatmap_text = None
                heatmap_method = None

    # Return results
    result: Dict[str, list[str] | list[Any] | str | None] = {
        "retrieved_visual_memories": guidelines,
        "retrieved_visual_memory_ids": memory_ids,
    }

    if heatmap_text is not None:
        result["heatmap_text"] = heatmap_text
    if heatmap_method is not None:
        result["heatmap_method"] = heatmap_method
    if updated_image_urls is not None:
        result["image_urls"] = updated_image_urls

    return result


async def extract_visual_keywords(
    guidelines: list[str],
    runtime: Runtime[Context],
) -> str:
    """Extract error-prone visual keywords from multiple visual memory guidelines.

    This function summarizes visual memory guidelines into concise keywords/phrases:
    1. Takes multiple visual memory guidelines (abstract, long text)
    2. Uses LLM to extract concrete visual elements that are error-prone
    3. Returns short, comma-separated keywords suitable for CLIP (under 77 tokens)

    The extraction focuses on:
    - Confusing elements (e.g., "digit 6 vs 9", "striped vs spotted patterns")
    - Subtle features (e.g., "ear color variations", "negative space shapes")
    - Spatial relations (e.g., "overlapping objects", "hidden vertices")
    - Key objects and attributes (e.g., "white kittens with dark ears")

    Output format: Removes conjunctions and prepositions, keeps nouns and adjectives
    as concise phrases that preserve semantic meaning.

    Args:
        guidelines: List of visual memory guideline strings
        runtime: Runtime context with model configuration

    Returns:
        Comma-separated keywords/phrases string (e.g., "digits in cartoon style, ear color contrast, overlapping shapes")
    """
    if not guidelines:
        return ""

    # Determine which model to use for keyword extraction
    model_name = runtime.context.keyword_extraction_model
    if not model_name:
        # Use analysis model as fallback
        model_name = runtime.context.analysis_model
        logger.debug(
            f"No keyword_extraction_model specified, using analysis_model: {model_name}"
        )

    # Load model
    model = load_chat_model(model_name)

    # Apply temperature and max_tokens if specified
    if runtime.context.keyword_extraction_temperature is not None:
        model = model.bind(temperature=runtime.context.keyword_extraction_temperature)  # type: ignore
    if runtime.context.keyword_extraction_max_tokens is not None:
        model = model.bind(max_tokens=runtime.context.keyword_extraction_max_tokens)  # type: ignore

    try:
        logger.debug(
            f"Extracting keywords from {len(guidelines)} visual memory guidelines"
        )

        # Prepare prompt
        combined_guidelines = "\n\n".join(
            f"Guideline {i + 1}: {g}" for i, g in enumerate(guidelines)
        )
        prompt = VISUAL_KEYWORD_EXTRACTION_PROMPT.format(guideline=combined_guidelines)

        # Configure model with retry and invoke
        model_with_retry = model.with_retry(stop_after_attempt=3)
        response = await model_with_retry.ainvoke([HumanMessage(content=prompt)])
        keywords = get_message_text(response).strip()

        # Clean up and validate keywords
        keywords = keywords.strip('"').strip("'")

        # Truncate to ~200 chars (conservative estimate for 77 CLIP tokens)
        if len(keywords) > 200:
            keywords = keywords[:200].rsplit(",", 1)[0]  # Truncate at last comma

        logger.debug(f"Extracted keywords ({len(keywords)} chars): {keywords}")

        return keywords

    except Exception as exc:
        logger.warning(
            f"[extract_visual_keywords] Keyword extraction failed after 3 retries "
            f"(model: {runtime.context.keyword_extraction_model}): {exc}"
        )
        # Fallback: use first 100 chars from all guidelines
        fallback = "; ".join(g[:50] for g in guidelines[:2])
        return fallback[:100]


async def generate_visual_memory(
    state: EvaluationState, runtime: Runtime[Context]
) -> Dict[str, str | bool]:
    """Analyze the error to determine if it's a visual understanding error.

    This node is only called when verification fails (verified=False).

    This node:
    1. Takes the incorrect reasoning and the first image
    2. Calls the visual memory generation model with VISUAL_ERROR_ANALYSIS_PROMPT
    3. Parses the JSON output to extract is_visual_error, summary, and guideline
    4. Returns the analysis results

    Args:
        state: Evaluation state with prediction, question, gold_answer, and images
        runtime: Runtime context with visual memory generation model configuration

    Returns:
        Dictionary with visual_error_analysis, is_visual_error, and new_visual_memory_guideline
    """
    # Skip if visual memory generation is disabled
    if not runtime.context.visual_memory_enable_generation:
        return {
            "visual_error_analysis": "",
            "is_visual_error": False,
            "new_visual_memory_guideline": "",
        }

    # Skip if answer is correct
    if state.verified:
        return {
            "visual_error_analysis": "",
            "is_visual_error": False,
            "new_visual_memory_guideline": "",
        }

    # Skip if no images available
    if not state.image_urls:
        return {
            "visual_error_analysis": "",
            "is_visual_error": False,
            "new_visual_memory_guideline": "",
        }

    # Extract reasoning steps and prepare data
    reasoning_steps = state.prediction or ""
    question = state.question or ""
    gold_answer = state.gold_answer or ""

    # Load visual memory generation model
    model = load_chat_model(runtime.context.visual_memory_generation_model)
    if runtime.context.visual_memory_generation_temperature is not None:
        model = model.bind(
            temperature=runtime.context.visual_memory_generation_temperature
        )  # type: ignore
    if runtime.context.visual_memory_generation_max_tokens is not None:
        model = model.bind(
            max_tokens=runtime.context.visual_memory_generation_max_tokens
        )  # type: ignore

    # Create prompt with image
    from vl_agent.context import VISUAL_ERROR_ANALYSIS_PROMPT

    prompt_text = VISUAL_ERROR_ANALYSIS_PROMPT.format(
        question=question,
        reasoning_steps=reasoning_steps,
        gold_answer=gold_answer,
    )

    # Create multimodal message with image
    # Use the first image
    image_url = state.image_urls[0]
    message_content: list[str | dict[str, Any]] = [
        {"type": "text", "text": prompt_text},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]

    # Configure model with retry and invoke
    model_with_retry = model.with_retry(stop_after_attempt=3)
    try:
        response = await model_with_retry.ainvoke(
            [HumanMessage(content=message_content)]
        )
        analysis_text = get_message_text(response)
    except Exception as exc:
        logger.warning(
            f"[generate_visual_memory] Visual error analysis failed after 3 retries "
            f"(model: {runtime.context.visual_memory_generation_model}): {exc}"
        )
        # Return empty results on error
        return {
            "visual_error_analysis": "",
            "is_visual_error": False,
            "new_visual_memory_guideline": "",
        }

    # Parse visual error analysis
    from vl_agent.memory import parse_visual_error_analysis

    is_visual_error, _analysis_summary, guideline = parse_visual_error_analysis(
        analysis_text
    )

    # ========== Store Visual Memory (if guideline exists) ==========
    generated_visual_memory_id = ""

    # Only store if guideline exists, we have output_dir, and images
    if guideline and state.output_dir and state.image_urls:
        # Parse problem analysis
        subject, key_concepts = parse_problem_analysis(state.problem_analysis or "")

        # Resolve visual memory file path
        visual_memory_file = (
            Path(state.output_dir) / runtime.context.visual_memory_file_path
        )

        # Load existing visual memories for similarity check
        from vl_agent.memory import (
            load_memories,
            retrieve_visual_memories_by_similarity,
        )

        # Create new visual memory
        # Note: Image embeddings will be automatically computed and cached
        # by retrieve_visual_memories_by_similarity during next retrieval
        memory_id = str(uuid.uuid4())

        # Construct benchmark-scoped path for source_image_path
        from vl_agent.memory import make_benchmark_scoped_path

        current_benchmark = runtime.context.current_benchmark

        # Support multi-image problems by splitting on semicolons
        image_filenames = (
            [f.strip() for f in state.image_path.split(";") if f.strip()]
            if state.image_path
            else []
        )

        # Construct benchmark-scoped paths for each image (if configured)
        source_image_paths_scoped: list[str] = []
        if current_benchmark and image_filenames:
            for filename in image_filenames:
                try:
                    scoped_path = make_benchmark_scoped_path(
                        filename, current_benchmark
                    )
                    source_image_paths_scoped.append(scoped_path)
                except ValueError:
                    logger.warning(
                        "Failed to build benchmark-scoped path for %s in %s",
                        filename,
                        current_benchmark,
                    )

        if source_image_paths_scoped:
            source_image_path_scoped = source_image_paths_scoped[0]
        else:
            # Fallback to plain filename if benchmark not configured
            source_image_path_scoped = state.image_path or ""
            # Also keep best-effort list for downstream consumers
            if source_image_path_scoped:
                source_image_paths_scoped = [source_image_path_scoped]

        # Construct embedding paths for this visual memory (one per image)
        # Format: {benchmark}/embeddings/{model_name}/{filename}.npy
        embedding_paths: list[str] = []
        if current_benchmark and image_filenames:
            model_name_normalized = (
                runtime.context.visual_embedding_model.split(":", 1)[1]
                if ":" in runtime.context.visual_embedding_model
                else runtime.context.visual_embedding_model
            )
            model_name_normalized = model_name_normalized.replace("/", "_")
            embedding_paths = [
                f"{current_benchmark}/embeddings/{model_name_normalized}/{filename}.npy"
                for filename in image_filenames
            ]

        embedding_path = embedding_paths[0] if embedding_paths else ""

        # Create visual memory entry (similar to logic memory structure)
        memory_data = {
            # Core identifier
            "memory_id": memory_id,
            # Core content
            "guideline": guideline,
            "error_type": "Visual",
            "is_visual_error": is_visual_error
            if is_visual_error is not None
            else False,
            # Problem analysis info
            "subject": subject,
            "key_concepts": key_concepts,
            # Metadata
            "created_at": datetime.now().isoformat(),
            "usage_count": 0,
            "last_used_at": None,
            # Source tracking (benchmark-scoped paths for cross-benchmark reuse)
            "source_question": question,
            "source_example_id": state.example_id,
            "source_image_path": source_image_path_scoped,  # ✅ Benchmark-scoped path (primary image)
            "source_image_paths": source_image_paths_scoped,  # ✅ All benchmark-scoped paths
            "embedding_path": embedding_path,  # ✅ Primary .npy path (backward compatibility)
            "embedding_paths": embedding_paths,  # ✅ All candidate .npy paths
        }

        # Save new visual memory
        from vl_agent.memory import save_memory

        save_memory(visual_memory_file, memory_data)
        generated_visual_memory_id = memory_id

        # Compute and cache text embedding for the guideline
        try:
            from vl_agent.memory import get_or_compute_text_embedding

            embedding = await get_or_compute_text_embedding(
                text=guideline,
                memory_id=memory_id,
                memory_type="visual",
                output_dir=Path(state.output_dir),
                model=runtime.context.visual_memory_text_embedding_model,
            )
            if embedding:
                logger.info(f"✓ Cached text embedding for visual memory {memory_id}")
            else:
                logger.error(
                    f"❌ Failed to cache text embedding for visual memory {memory_id}. "
                    f"This memory will not be retrievable until embedding is computed manually."
                )
        except Exception as e:
            logger.error(
                f"❌ Error caching text embedding for visual memory {memory_id}: {e}. "
                f"Check if embedding model is running at {runtime.context.visual_memory_text_embedding_model}"
            )

    return {
        "visual_error_analysis": analysis_text,
        "is_visual_error": is_visual_error,
        "new_visual_memory_guideline": guideline,
        "generated_visual_memory_id": generated_visual_memory_id,
    }
