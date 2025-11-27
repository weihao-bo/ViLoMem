"""Prompt builder for VLMEvalKit benchmarks.

This module provides benchmark-specific prompt construction logic:
- Detects task type (MCQ, Math, VQA, etc.)
- Formats questions with options for MCQ tasks
- Includes hints when available
- Adds CoT instructions for reasoning tasks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class BenchmarkPromptConfig:
    """Configuration for benchmark-specific prompt construction."""

    # Task type determines prompt structure
    task_type: Literal["mcq", "math", "vqa", "captioning"] = "vqa"

    # MCQ-specific settings
    mcq_format_template: str = "Options:\n{options}\n"
    mcq_option_template: str = "{key}. {value}\n"
    mcq_suffix: str = "Please select the correct answer from the options above."

    # CoT settings
    enable_cot: bool = False
    cot_template: str = (
        "Think step by step before answering. "
        "The last line of your response should be 'Answer: \\boxed{$ANSWER}'."
    )

    # Hint support
    include_hint: bool = True
    hint_template: str = "Hint: {hint}\n"

    # Question wrapper
    question_template: str = "Question: {question}\n"

    @classmethod
    def from_benchmark_meta(
        cls, benchmark_meta: dict[str, Any]
    ) -> "BenchmarkPromptConfig":
        """Create config from benchmark metadata.

        Args:
            benchmark_meta: Metadata dict from benchmark_meta.json

        Returns:
            BenchmarkPromptConfig instance
        """
        task_type = benchmark_meta.get("task_type", "vqa")
        enable_cot = benchmark_meta.get("enable_cot", False)

        # Customize CoT template based on task type
        if task_type == "math":
            cot_template = (
                "Solve this step-by-step. End with 'Final Answer: \\boxed{$ANSWER}'."
            )
        elif task_type == "mcq":
            cot_template = (
                "Think step by step before selecting. "
                "End with 'Answer: \\boxed{$LETTER}' where LETTER is one of the options."
            )
        else:
            cot_template = "Think step by step. End with 'Answer: \\boxed{$ANSWER}'."

        return cls(
            task_type=task_type,
            enable_cot=enable_cot,
            cot_template=cot_template,
        )


def build_prompt(
    sample: dict[str, Any],
    config: BenchmarkPromptConfig,
) -> list[dict[str, str]]:
    """Build multimodal prompt from sample and config.

    Args:
        sample: Sample dict with inputs/outputs/metadata
        config: Benchmark-specific prompt configuration

    Returns:
        List of message dicts in format:
        [
            {"type": "image", "value": "path/to/img1.jpg"},
            {"type": "image", "value": "path/to/img2.jpg"},
            {"type": "text", "value": "Question: ...\\nOptions:\\n..."},
        ]
    """
    # Extract data
    inputs = sample.get("inputs", {})
    metadata = sample.get("metadata", {})
    question = inputs.get("question", "")

    # Build text prompt
    text_parts = []

    # 1. Hint (if present and enabled)
    hint = metadata.get("hint")
    if hint and config.include_hint:
        text_parts.append(config.hint_template.format(hint=hint))

    # 2. Question
    text_parts.append(config.question_template.format(question=question))

    # 3. Options (for MCQ)
    if config.task_type == "mcq":
        choices = metadata.get("choices", {})
        if choices:
            options_text = ""
            # Sort keys to ensure A, B, C, D, ... order
            for key in sorted(choices.keys()):
                options_text += config.mcq_option_template.format(
                    key=key, value=choices[key]
                )
            text_parts.append(
                config.mcq_format_template.format(options=options_text.strip())
            )
            text_parts.append(config.mcq_suffix)

    # 4. CoT instruction (if enabled)
    if config.enable_cot:
        text_parts.append("\n" + config.cot_template)

    # Combine text parts
    full_text = "\n".join(text_parts)

    # Build message list: [images..., text]
    messages = []

    # Add images (support both single and multiple)
    image_data = inputs.get("image")
    if image_data:
        if isinstance(image_data, list):
            for img_filename in image_data:
                messages.append({"type": "image", "value": img_filename})
        else:
            # Fallback for legacy single-image format
            messages.append({"type": "image", "value": image_data})

    # Add text
    messages.append({"type": "text", "value": full_text})

    return messages


def reconstruct_question_with_options(
    sample: dict[str, Any],
    config: BenchmarkPromptConfig,
) -> str:
    """Reconstruct question text with options for MCQ tasks.

    Args:
        sample: Sample dict with inputs/metadata
        config: Benchmark-specific prompt configuration

    Returns:
        Reconstructed question text
    """
    # Extract data
    inputs = sample.get("inputs", {})
    metadata = sample.get("metadata", {})
    question = inputs.get("question", "")

    # Build text
    text_parts = []

    # Add hint if present
    hint = metadata.get("hint")
    if hint and config.include_hint:
        text_parts.append(f"Hint: {hint}")

    # Add question
    text_parts.append(f"Question: {question}")

    # Add options for MCQ
    if config.task_type == "mcq":
        choices = metadata.get("choices", {})
        if choices:
            text_parts.append("Options:")
            for key in sorted(choices.keys()):
                text_parts.append(f"{key}. {choices[key]}")
            text_parts.append(
                "Please select the correct answer from the options above."
            )

    # Add CoT instruction
    if config.enable_cot:
        text_parts.append("")  # Empty line
        text_parts.append(config.cot_template)

    return "\n".join(text_parts)
