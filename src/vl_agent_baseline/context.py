"""Define the configurable parameters for the Qwen VL baseline agent (no memory)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

# VLMEvalKit-compatible default: no additional prompt (use question directly)
# For VLMEvalKit compatibility, most models use the question directly without modification.
# Reference: https://github.com/open-compass/VLMEvalKit
# Models like llama4, InternVLChat, etc. use line['question'] directly for MathVista
DEFAULT_VLMEVAL_PROMPT = ""  # Empty = use question directly (VLMEvalKit pattern)

# Alternative prompt for custom reasoning guidance (not VLMEvalKit standard)
DEFAULT_MATH_REASONING_PROMPT = """
Objective:
    Solve the given problem using a step by step process.

Final Answer Format:
    - Single final boxed result, no text after it.

    - Multiple-choice: only the option letter inside the box (e.g., \\boxed{A}).

    - Non-multiple-choice: only the number/word/expression inside the box.

Expected Output Structure:
Step 1:
Step 2:
...
Step n: Final Answer: \\boxed{answer}

Question: <image>{question}

"""


@dataclass(kw_only=True)
class Context:
    """The context for the Qwen VL baseline agent (no memory functionality).

    This is a simplified version of the full agent context, containing only
    the essential parameters for model inference and verification.

    Following LangGraph best practices:
    - Uses dataclass with kw_only for clear parameter passing
    - Supports environment variable overrides
    - Includes metadata for LangGraph Studio integration
    """

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="qwen:qwen3-vl-8b-instruct",
        metadata={
            "description": "The vision-language model to use for math reasoning. "
            "Should be a multimodal model that supports image inputs. "
            "Format: provider:model-name.",
            "json_schema_extra": {"langgraph_nodes": ["call_model"]},
        },
    )

    system_prompt: str = field(
        default=DEFAULT_VLMEVAL_PROMPT,
        metadata={
            "description": "The system prompt template (for VLMEvalKit compatibility, default is empty). "
            "Empty string = use question directly (VLMEvalKit pattern). "
            "Non-empty = inject question using {question} placeholder. "
            "For custom reasoning guidance, use DEFAULT_MATH_REASONING_PROMPT.",
        },
    )

    enable_verification: bool = field(
        default=True,
        metadata={
            "description": "Whether to enable answer verification using math-verify. "
            "Only works when gold_answer is provided in the state.",
            "json_schema_extra": {"langgraph_nodes": ["verify_answer"]},
        },
    )

    temperature: float | None = field(
        default=0.7,
        metadata={
            "description": "Temperature parameter for model inference. "
            "Lower values (e.g., 0.1) make output more focused and deterministic. "
            "Higher values (e.g., 0.9) make output more creative. "
            "Set to None to use model default.",
            "json_schema_extra": {"langgraph_nodes": ["call_model"]},
        },
    )

    max_tokens: int | None = field(
        default=4096,
        metadata={
            "description": "Maximum number of tokens to generate in the response. "
            "Set to None to use model default.",
            "json_schema_extra": {"langgraph_nodes": ["call_model"]},
        },
    )

    def __post_init__(self) -> None:
        """Fetch environment variables for attributes that were not passed as args.

        This follows the same pattern as the full agent context:
        - Only override with environment variable if current value equals default
        - This preserves explicit configuration from LangGraph configurable
        """
        import os
        from dataclasses import fields

        for f in fields(self):
            if not f.init:
                continue

            current_value = getattr(self, f.name)
            default_value = f.default
            env_var_name = f.name.upper()
            env_value = os.environ.get(env_var_name)

            # Only override with environment variable if current value equals default
            if current_value == default_value and env_value is not None:
                if isinstance(default_value, bool):
                    # Handle boolean environment variables
                    env_bool_value = env_value.lower() in ("true", "1", "yes", "on")
                    setattr(self, f.name, env_bool_value)
                elif isinstance(default_value, int):
                    setattr(self, f.name, int(env_value))
                elif isinstance(default_value, float):
                    setattr(self, f.name, float(env_value))
                else:
                    setattr(self, f.name, env_value)
