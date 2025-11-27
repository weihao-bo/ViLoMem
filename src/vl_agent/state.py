"""Define the state structures for the Qwen VL math reasoning agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


@dataclass
class InputState:
    """Input state for the Qwen VL agent.

    This defines the interface for incoming requests to the agent.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages] = field(
        default_factory=list
    )
    """
    Messages tracking the conversation with the agent.
    Uses add_messages annotation to merge new messages with existing ones.
    """

    gold_answer: Optional[str] = field(default=None)
    """
    The expected correct answer for verification (optional).
    Only provided during evaluation.
    """


@dataclass
class State(InputState):
    """Complete state for the Qwen VL agent.

    Extends InputState with additional fields for tracking reasoning and verification.
    """

    prediction: Optional[str] = field(default=None)
    """
    The model's predicted answer extracted from the final response.
    Populated by the reasoning node after model inference.
    """

    verified: Optional[bool] = field(default=None)
    """
    Whether the prediction matches the gold answer.
    Populated by the verification node if gold_answer is provided.
    """

    verification_error: Optional[str] = field(default=None)
    """
    Error message if verification fails.
    Stores exception details from the math-verify grader.
    """

    verification_method: Optional[str] = field(default=None)
    """
    Method used for verification (e.g., 'vlmeval_matcher', 'answer_parser', 'math_verify').
    """

    extracted_option: Optional[str] = field(default=None)
    """
    Extracted MCQ option letter (if applicable).
    """

    extracted_answer: Optional[str] = field(default=None)
    """
    Parsed/normalized answer extracted during verification (numeric/text).
    """

    verification_attempts: list[dict[str, Any]] = field(default_factory=list)
    """
    Detailed logs of each verification stage (VLMEval → parser → math_verify).
    """

    model_user_prompt: Optional[str] = field(default=None)
    """Actual human-side prompt text sent to the model."""

    model_system_prompt: Optional[str] = field(default=None)
    """System prompt string used for the current model invocation."""

    model_prompt_metadata: dict[str, Any] = field(default_factory=dict)
    """Metadata describing how the final prompt was constructed (VLMEval usage, etc.)."""


@dataclass
class EvaluationState(State):
    """Extended state for evaluation workflow with output persistence.

    Adds fields for tracking evaluation metadata and output paths.
    """

    question: Optional[str] = field(default=None)
    """The original question text for logging purposes."""

    example_id: Optional[str] = field(default=None)
    """Unique identifier for this example (from LangSmith dataset)."""

    output_dir: Optional[str] = field(default=None)
    """Directory path where evaluation results should be saved."""

    image_path: Optional[str] = field(default=None)
    """Image filename from attachments (e.g., '981.jpg')."""

    task: Optional[str] = field(default=None)
    """Task type from dataset metadata (e.g., 'figure question answering')."""

    # ========== Logic Memory Fields ==========
    problem_analysis: Optional[str] = field(default=None)
    """Problem analysis output identifying subject and key concepts."""

    retrieved_memories: list[str] = field(default_factory=list)
    """List of guideline texts retrieved from logic memory store."""

    retrieved_memory_ids: list[str] = field(default_factory=list)
    """List of logic memory UUIDs that were retrieved and used."""

    error_analysis: Optional[str] = field(default=None)
    """Full error analysis output (only generated when answer is wrong)."""

    error_type: Optional[str] = field(default=None)
    """Error type: 'Logical' / 'Non-Logical'."""

    new_memory_guideline: Optional[str] = field(default=None)
    """New guideline generated from error analysis (only for logical errors)."""

    generated_memory_id: Optional[str] = field(default=None)
    """UUID of newly stored logic memory (if any)."""

    # ========== Visual Memory Fields ==========
    image_urls: list[str] = field(default_factory=list)
    """List of image URLs or base64-encoded images extracted from messages."""

    retrieved_visual_memories: list[str] = field(default_factory=list)
    """List of visual guideline texts retrieved from visual memory store."""

    retrieved_visual_memory_ids: list[str] = field(default_factory=list)
    """List of visual memory UUIDs that were retrieved and used."""

    visual_error_analysis: Optional[str] = field(default=None)
    """Full visual error analysis output (only generated when answer is wrong)."""

    is_visual_error: Optional[bool] = field(default=None)
    """Whether the error is a visual understanding error."""

    new_visual_memory_guideline: Optional[str] = field(default=None)
    """New visual guideline generated from error analysis (only for visual errors)."""

    generated_visual_memory_id: Optional[str] = field(default=None)
    """UUID of newly stored visual memory (if any)."""

    heatmap_text: Optional[str] = field(default=None)
    """Text used for heatmap generation (extracted keywords or raw guidelines)."""

    heatmap_method: Optional[str] = field(default=None)
    """Heatmap generation method used: 'gradcam' or 'grad-eclip'."""

    # ========== Benchmark-specific fields (for VLMEvalKit integration) ==========
    benchmark_config: Optional[Any] = field(default=None)
    """Benchmark-specific prompt configuration (BenchmarkPromptConfig instance)."""

    sample_metadata: dict[str, Any] = field(default_factory=dict)
    """Sample metadata from converted dataset (includes answer_type, choices, hint, etc.)."""
