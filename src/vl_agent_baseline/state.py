"""Define the state structures for the Qwen VL baseline agent (no memory).

This is a simplified version without memory-related fields, used as a baseline
for accuracy comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated


@dataclass
class InputState:
    """Input state for the Qwen VL baseline agent.

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
    """Complete state for the Qwen VL baseline agent.

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

    extracted_option: Optional[str] = field(default=None)
    """
    Extracted option letter for MCQ questions (e.g., 'A', 'B', 'C').
    Populated by VLMEvalKit matching logic.
    """

    extracted_answer: Optional[str] = field(default=None)
    """
    Parsed/normalized answer extracted during verification (e.g., numeric value).
    """

    verification_method: Optional[str] = field(default=None)
    """
    Method used for verification (e.g., 'can_infer', 'parse_multi_choice', 'math_verify').
    Helps track how the answer was verified.
    """

    verification_attempts: list[dict[str, Any]] = field(default_factory=list)
    """
    Detailed logs of each verification stage (VLMEval → parser → math_verify).
    """

    model_user_prompt: Optional[str] = field(default=None)
    """Actual user-side prompt text sent to the model."""

    model_system_prompt: Optional[str] = field(default=None)
    """System prompt content used for the model call."""

    model_prompt_metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the constructed model prompt (VLMEval usage, etc.)."""


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

    # ========== Benchmark-specific fields (for VLMEvalKit integration) ==========
    benchmark_config: Optional[Any] = field(default=None)
    """Benchmark-specific prompt configuration (BenchmarkPromptConfig instance)."""

    sample_metadata: dict[str, Any] = field(default_factory=dict)
    """Sample metadata from converted dataset (includes answer_type, choices, hint, etc.)."""


