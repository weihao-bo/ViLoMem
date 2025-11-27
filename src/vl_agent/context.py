"""Define the configurable parameters for the Qwen VL math reasoning agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

# ========== Prompts for Memory System ==========

PROBLEM_ANALYSIS_PROMPT = """Objective:
    Analyze the following problem to identify its subject area and the key concepts, principles, formulas, or laws required for its solution. This analysis will be used to retrieve relevant guiding principles from a knowledge base.

Instructions:
    - Do not solve the problem.
    - First, identify the primary subject (e.g., Physics, Chemistry, Biology, Mathematics).
    - Then, list the core concepts or principles involved (e.g., Newton's Second Law, Conservation of Energy, Stoichiometry, Pythagorean theorem).
    - Keep the analysis concise and focused.

Problem:
{question}

Output Format:
Subject: <The primary subject>
Key Concepts: <A brief list of key concepts>"""

ERROR_ANALYSIS_PROMPT = """Objective:

    Analyze the provided incorrect reasoning process for a scientific or mathematical problem. Your goal is to classify the error and, if it is a logical error, generate a high-quality, actionable guideline (a "memory") to prevent similar mistakes in the future.

Context:

    - Problem: {question}

    - Incorrect Reasoning Steps: {reasoning_steps}

    - Correct Answer (for reference): {gold_answer}

Instructions:

1.  **Analyze the Mistake:** Carefully review the `Incorrect Reasoning Steps` against the `Problem` and `Correct Answer` to pinpoint the primary mistake.

2.  **Categorize the Error:** Classify the error into one of two types:

    *   `Logical`: Any error in the reasoning process itself. This includes calculation mistakes, misapplication of a formula or theorem, logical fallacies, or conceptual misunderstandings. These errors can be identified from the text of the reasoning alone.

    *   `Non-Logical`: An error that stems purely from misinterpreting the visual information in an image. This kind of error can **only** be confirmed by looking at the image (e.g., misidentifying a shape, reading a value from a graph incorrectly).

3.  **Generate the Guideline (Memory):**

    *   **Only if the `error_type` is `Logical`**, you must generate a guideline.

    *   If the `error_type` is `Non-Logical`, the guideline must be left empty.

    **Guideline Quality Requirements:**

    *   **Be Specific and Concrete:** The guideline must target the specific principle, theorem, formula, or reasoning pattern that was misused. Name the concept directly.

    *   **Be Actionable:** Frame it as a clear instruction, a warning, or a rule of thumb (e.g., "Always check...", "Remember to differentiate between...", "When X occurs, apply Y...").

    *   **Be Generalizable:** The advice should be abstracted from the specific numbers and context of this single problem so it can apply to a whole class of similar problems.

    *   **Keep it Concise:** The guideline should be one to two sentences long.

    ---

    **Guideline Examples (Good, Specific Examples):**

    *   **(Physics):** "When applying the conservation of energy to rolling objects, always include both translational and rotational kinetic energy in the equation."

    *   **(Math):** "In geometry problems involving tangents to a circle, remember that the radius to the point of tangency is perpendicular to the tangent line."

    *   **(Chemistry):** "For stoichiometry calculations, always ensure the chemical equation is correctly balanced before determining the molar ratios."

    **Avoid these types of guidelines (Bad, Vague Examples):**

    *   "You need to pay attention to the geometric relationships."

    *   "Avoid making calculation errors."

    *   "Analyze the problem carefully."

    ---

Output Format (use this exact structure):

error_type: <"Logical" or "Non-Logical">

analysis_summary: <A brief, one-sentence summary of what went wrong.>

guideline: <Your 1-2 sentence guideline if the error is "Logical", otherwise leave this field empty.>"""

VISUAL_KEYWORD_EXTRACTION_PROMPT = """Objective:
    Extract the key error-prone visual elements from a visual memory guideline. Focus on concrete, visually observable elements that the model needs to pay special attention to avoid making mistakes.

Guideline:
{guideline}

Instructions:
1. Identify concrete visual elements that are:
   - **Error-prone**: Easy to overlook, misidentify, or confuse
   - **Observable**: Can be directly seen in an image (not abstract concepts)
   - **Specific**: Concrete objects, attributes, spatial relations, or patterns

2. Categories to extract (in order of priority):
   - **Confusing elements**: Things that look similar but are different (e.g., "digit 6 vs 9", "striped vs spotted")
   - **Subtle features**: Small details that are easy to miss (e.g., "ear color", "negative space", "embedded shapes")
   - **Spatial relations**: Position or arrangement (e.g., "on the left", "overlapping", "hidden behind")
   - **Key objects**: Main entities mentioned (e.g., "digits", "kittens", "body parts")
   - **Distinctive attributes**: Colors, patterns, textures (e.g., "white fur", "cartoon style", "curved lines")

3. Output format:
   - Return ONLY a comma-separated list of keywords/short phrases (3-10 words total)
   - Use concrete nouns, adjectives, and spatial terms
   - NO verbs, NO abstract instructions, NO complete sentences
   - Keep it under 77 tokens (CLIP's limit)

Examples:

Guideline: "When counting digits in stylized or cartoon images, examine all components—including eyes, limbs, body contours, and negative space—for shapes that resemble numerals."
Keywords: digits, eyes as numbers, limbs shaped like digits, body contours, negative space, cartoon numerals

Guideline: "When checking for 'different color ears,' carefully compare each kitten's ear color to its body's base color, paying special attention to high-contrast cases like white kittens with dark ears."
Keywords: ear color, body color, white kittens, dark ears, high contrast, color mismatch

Guideline: "For geometry problems involving overlapping shapes, identify all visible vertices even when partially occluded by other shapes."
Keywords: overlapping shapes, hidden vertices, partially occluded, geometry intersection points

Now extract keywords from the given guideline above. Output ONLY the comma-separated keyword list."""

VISUAL_ERROR_ANALYSIS_PROMPT = """Objective:
    You are an expert in visual reasoning and error analysis. Your task is to first describe the provided image objectively, then analyze an incorrect reasoning process to determine if the error stems from misinterpreting that image. If a visual error is found, you must generate a concise, actionable guideline (a "visual memory") to prevent this mistake in the future.

Context:
    - Problem: {question}
    - Incorrect Reasoning Steps: {reasoning_steps}
    - Correct Answer (for reference): {gold_answer}

**Attached Image:** <image>

**Thinking Process and Final Output:**
Your response must follow a strict two-stage process. The first stage is your internal "thought process" which you will write out. The second stage is the final JSON output.

**Stage 1: Internal Thought Process (Write this out first)**

1.  **Describe the Image:** Begin by providing an objective, detailed description of the attached image. List all key elements, labels, values, geometric shapes, and their relationships. This description will serve as the "ground truth" for your analysis.
2.  **Analyze for Discrepancies:** Compare your image description and the image itself against the text in `Incorrect Reasoning Steps`. Identify any contradictions, misinterpretations, or omissions.

**Stage 2: Final JSON Output (Provide ONLY this JSON block as the final answer)**

After completing your thought process, generate a JSON object based on your analysis. The JSON should adhere to the following structure and guidelines.

---
**Guidelines for `guideline` Generation:**

*   The guideline MUST be about how to correctly interpret a specific visual pattern or element.
*   It must be a rule that can be applied to other, similar-looking problems.
*   It should be concise (one to two sentences).

**Guideline Examples (Good, Specific Visual Memories):**
*   **(Physics/Diagrams):** "In a free-body diagram, always verify that all forces, including friction and normal force, are accounted for before applying Newton's laws."
*   **(Geometry):** "When an angle appears to be a right angle in a diagram, do not assume it is 90 degrees unless it is explicitly marked with a square symbol."
*   **(Chemistry/Molecules):** "For complex organic molecules, double-check the placement of double bonds and functional groups as they dictate the molecule's reactivity."
*   **(Biology/Graphs):** "When reading a bar chart, pay close attention to the Y-axis scale and units to avoid misinterpreting the magnitude of the results."

**Avoid these types of guidelines (Bad, Non-Visual or Too Vague):**
*   "The model made a calculation error." (This is a logical error, not visual)
*   "You need to look at the image more carefully." (Not actionable)
*   "The reasoning about the physics was wrong." (Too general)
---

**Final Output Format (use this exact JSON structure):**
{{
    "is_visual_error": true/false,
    "analysis_summary": "A brief, one-sentence summary of the visual misinterpretation.",
    "guideline": "Your 1-2 sentence visual guideline. Provide this only if is_visual_error is true, otherwise it should be null."
}}"""

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

If the Question specifies special output requirements, follow the Question's instructions exactly.
"""


@dataclass(kw_only=True)
class Context:
    """The context for the Qwen VL math reasoning agent.

    This class manages all configurable parameters for the agent, including
    the model selection, prompt template, and verification settings.

    Following LangGraph best practices:
    - Uses dataclass with kw_only for clear parameter passing
    - Supports environment variable overrides
    - Includes metadata for LangGraph Studio integration
    """

    # ========== Benchmark Configuration (for cross-benchmark memory reuse) ==========
    benchmark_root_dir: str = field(
        default="",
        metadata={
            "description": "Root directory for all benchmarks (e.g., '/Users/bwh/Documents/Datasets'). "
            "Used to resolve benchmark-scoped image paths for cross-benchmark memory reuse. "
            "Should match the 'root_dir' in dataset configuration.",
        },
    )

    current_benchmark: str = field(
        default="",
        metadata={
            "description": "Current benchmark name (e.g., 'MathVision_MINI'). "
            "Used to create benchmark-scoped paths for visual memories. "
            "Should match the 'benchmark' field in dataset configuration.",
        },
    )

    # ========== Model Configuration ==========
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

    temperature: float = field(
        default=0.7,
        metadata={
            "description": "Temperature parameter for model inference. "
            "Lower values (e.g., 0.1) make output more focused and deterministic. "
            "Higher values (e.g., 0.9) make output more creative.",
            "json_schema_extra": {"langgraph_nodes": ["call_model"]},
        },
    )

    max_tokens: int = field(
        default=4096,
        metadata={
            "description": "Maximum number of tokens to generate in the response.",
            "json_schema_extra": {"langgraph_nodes": ["call_model"]},
        },
    )

    # ========== Logic Memory Feature Switches ==========
    logic_memory_enable_retrieval: bool = field(
        default=True,
        metadata={
            "description": "Whether to enable logic memory retrieval. "
            "When disabled, no logic memories will be retrieved from storage.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_logic_memories"]},
        },
    )

    logic_memory_enable_generation: bool = field(
        default=True,
        metadata={
            "description": "Whether to enable logic memory generation. "
            "When disabled, no new logic memories will be generated from errors.",
            "json_schema_extra": {"langgraph_nodes": ["generate_logic_memory"]},
        },
    )

    # ========== Logic Memory Retrieval Configuration ==========
    logic_memory_retrieval_limit: int = field(
        default=3,
        metadata={
            "description": "Maximum number of logic memories to retrieve for each question.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_logic_memories"]},
        },
    )

    logic_memory_similarity_threshold: float = field(
        default=0.5,
        metadata={
            "description": "Minimum similarity threshold for logic memory retrieval (0.0 to 1.0). "
            "Logic memories with similarity below this threshold will be filtered out.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_logic_memories"]},
        },
    )

    # ========== Logic Memory Store Configuration ==========
    logic_memory_file_path: str = field(
        default="memories.json",
        metadata={
            "description": "Path to the JSON file for storing logic memories. "
            "Relative paths are resolved from the output directory.",
        },
    )

    logic_memory_embedding_model: str = field(
        default="local:qwen3-embedding",
        metadata={
            "description": "Text embedding model for logic memory retrieval in 'provider:model' format. "
            "Supported providers: "
            "- local: Local vLLM embedding API (e.g., 'local:qwen3-embedding') "
            "- qwen/dashscope: DashScope text embedding API (e.g., 'qwen:text-embedding-v2') "
            "- openai: OpenAI text embedding API (e.g., 'openai:text-embedding-3-small'). "
            "Uses cosine similarity for semantic search.",
        },
    )

    # ========== Problem Analysis Configuration (for retrieve_logic_memories node) ==========
    analysis_model: str = field(
        default="qwen:qwen-flash",
        metadata={
            "description": "Model for problem analysis (identifying subject and key concepts). "
            "Can use a faster/cheaper model than the main reasoning model.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_logic_memories"]},
        },
    )

    analysis_temperature: float = field(
        default=0.3,
        metadata={
            "description": "Temperature for problem analysis. Lower values for more focused analysis.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_logic_memories"]},
        },
    )

    analysis_max_tokens: int = field(
        default=512,
        metadata={
            "description": "Maximum tokens for problem analysis output.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_logic_memories"]},
        },
    )

    # ========== Logic Memory Generation Configuration (for generate_logic_memory node) ==========
    logic_memory_generation_model: str = field(
        default="qwen:qwen3-next-80b-a3b-instruct",
        metadata={
            "description": "Model for logic error analysis and memory generation. "
            "Uses qwen3-next-80b-a3b-instruct for deeper reasoning capability.",
            "json_schema_extra": {"langgraph_nodes": ["generate_logic_memory"]},
        },
    )

    logic_memory_generation_temperature: float = field(
        default=0.3,
        metadata={
            "description": "Temperature for logic memory generation. Lower values for more consistent guidelines.",
            "json_schema_extra": {"langgraph_nodes": ["generate_logic_memory"]},
        },
    )

    logic_memory_generation_max_tokens: int = field(
        default=256,
        metadata={
            "description": "Maximum tokens for logic error analysis and guideline generation.",
            "json_schema_extra": {"langgraph_nodes": ["generate_logic_memory"]},
        },
    )

    # ========== Visual Memory Feature Switches ==========
    visual_memory_enable_retrieval: bool = field(
        default=True,
        metadata={
            "description": "Whether to enable visual memory retrieval. "
            "When disabled, no visual memories will be retrieved from storage.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_visual_memories"]},
        },
    )

    visual_memory_enable_generation: bool = field(
        default=True,
        metadata={
            "description": "Whether to enable visual memory generation. "
            "When disabled, no new visual memories will be generated from errors.",
            "json_schema_extra": {"langgraph_nodes": ["generate_visual_memory"]},
        },
    )

    # ========== Visual Memory Retrieval Configuration ==========
    visual_memory_retrieval_limit: int = field(
        default=3,
        metadata={
            "description": "Maximum number of visual memories to retrieve for each question.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_visual_memories"]},
        },
    )

    visual_memory_similarity_threshold: float = field(
        default=0.5,
        metadata={
            "description": "Minimum similarity threshold for visual memory retrieval (0.0 to 1.0). "
            "Visual memories with similarity below this threshold will be filtered out.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_visual_memories"]},
        },
    )

    # ========== Visual Memory Store Configuration ==========
    visual_memory_file_path: str = field(
        default="visual_memories.json",
        metadata={
            "description": "Path to the JSON file for storing visual memories. "
            "Relative paths are resolved from the output directory.",
        },
    )

    visual_embedding_model: str = field(
        default="qwen:qwen2.5-vl-embedding",
        metadata={
            "description": "Visual embedding model for visual memory retrieval. "
            "Uses Qwen online multimodal embedding API for high-quality embeddings. "
            "Format: provider:model-name (e.g., 'qwen:qwen2.5-vl-embedding')",
        },
    )

    visual_embedding_top_n: int = field(
        default=10,
        metadata={
            "description": "First-stage candidate size for visual memory retrieval. "
            "When visual_memory_enable_text_rerank=False, these candidates are filtered by similarity threshold. "
            "When visual_memory_enable_text_rerank=True, these candidates are passed to text embedding similarity as the second stage.",
        },
    )

    visual_memory_enable_text_rerank: bool = field(
        default=True,
        metadata={
            "description": "Whether to enable two-stage retrieval for visual memories. "
            "Stage 1: Use multimodal embeddings to retrieve top-N candidates (visual_embedding_top_n). "
            "Stage 2: Use text embedding (like logic memory) to select top-M from candidates (visual_memory_retrieval_limit). "
            "When False, only uses image embedding-based similarity filtering.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_visual_memories"]},
        },
    )

    visual_memory_text_embedding_model: str = field(
        default="local:qwen3-embedding",
        metadata={
            "description": "Text embedding model for visual memory second-stage retrieval in 'provider:model' format. "
            "Only used when visual_memory_enable_text_rerank=True. "
            "Supported providers: "
            "- local: Local vLLM embedding API (e.g., 'local:qwen3-embedding') "
            "- qwen/dashscope: DashScope text embedding API (e.g., 'qwen:text-embedding-v2') "
            "- openai: OpenAI text embedding API (e.g., 'openai:text-embedding-3-small'). "
            "Retrieves candidates using Question + visual guideline text similarity.",
        },
    )

    # ========== Visual Memory Generation Configuration (for generate_visual_memory node) ==========
    visual_memory_generation_model: str = field(
        default="qwen:qwen3-vl-30b-a3b-instruct",
        metadata={
            "description": "Vision-language model for visual error analysis and memory generation. "
            "Uses qwen3-vl-30b-a3b-instruct for analyzing visual understanding errors.",
            "json_schema_extra": {"langgraph_nodes": ["generate_visual_memory"]},
        },
    )

    visual_memory_generation_temperature: float = field(
        default=0.3,
        metadata={
            "description": "Temperature for visual memory generation. Lower values for more consistent guidelines.",
            "json_schema_extra": {"langgraph_nodes": ["generate_visual_memory"]},
        },
    )

    visual_memory_generation_max_tokens: int = field(
        default=512,
        metadata={
            "description": "Maximum tokens for visual error analysis and guideline generation.",
            "json_schema_extra": {"langgraph_nodes": ["generate_visual_memory"]},
        },
    )

    # ========== Heatmap Generation Configuration (Qwen2.5-VL Attention) ==========
    enable_heatmap_generation: bool = field(
        default=True,
        metadata={
            "description": "Whether to enable heatmap generation using Qwen2.5-VL attention when visual memories are retrieved.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_visual_memories"]},
        },
    )

    include_question_in_heatmap: bool = field(
        default=False,
        metadata={
            "description": "Whether to include the question text in the heatmap generation prompt. "
            "When True, prepends 'Question: {question}\\n\\n' to the heatmap text. "
            "This helps the heatmap model better understand the context of visual memories.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_visual_memories"]},
        },
    )

    debug_heatmap: bool = field(
        default=False,
        metadata={
            "description": "Whether to save heatmap images for debugging. "
            "When True, saves overlaid heatmap images to output directory's qwen25vl-attention/ subdirectory.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_visual_memories"]},
        },
    )

    # ========== Qwen2.5-VL Attention Heatmap Model Configuration ==========
    qwen25vl_model: str = field(
        default="Qwen/Qwen2.5-VL-2B-Instruct",
        metadata={
            "description": "Qwen2.5-VL model for cross-attention based heatmap generation. "
            "Recommended: 'Qwen/Qwen2.5-VL-2B-Instruct' (4GB VRAM) or 'Qwen/Qwen2.5-VL-7B-Instruct' (14GB VRAM). "
            "Note: Requires 'attn_implementation=eager' mode for attention extraction (slower than flash attention).",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_visual_memories"]},
        },
    )

    qwen25vl_general_prompt: str = field(
        default="Describe this image.",
        metadata={
            "description": "General/baseline prompt for relative attention computation. "
            "Used to normalize attention scores: A_rel(x,q) = A(x,q) / A(x,general_prompt). "
            "This helps filter semantic noise and focus on query-specific regions.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_visual_memories"]},
        },
    )

    qwen25vl_attention_layer: int = field(
        default=22,
        metadata={
            "description": "Which transformer layer to extract attention from (0-indexed). "
            "Default: 22 (recommended by 'MLLMs Know Where to Look' paper). "
            "Higher layers capture more semantic information, lower layers capture more low-level features.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_visual_memories"]},
        },
    )

    qwen25vl_device: str = field(
        default="cuda:0",
        metadata={
            "description": "Device to use for Qwen2.5-VL attention extraction. "
            "Options: 'cpu', 'cuda:0', 'cuda:1', etc. for CUDA GPUs, or 'mps' for Apple Silicon. "
            "Note: Qwen2.5-VL 2B requires ~4GB VRAM, 7B requires ~14GB VRAM (bfloat16).",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_visual_memories"]},
        },
    )

    qwen25vl_devices: list[str] = field(
        default_factory=list,
        metadata={
            "description": "Optional list of devices that the Qwen2.5-VL heatmap generator can round-robin. "
            "Accepts either ['cuda:0', 'cuda:1'] style entries or comma-separated strings. "
            "When empty, falls back to qwen25vl_device.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_visual_memories"]},
        },
    )

    qwen25vl_per_device_limit: int = field(
        default=1,
        metadata={
            "description": "Maximum number of concurrent heatmap jobs allowed per configured device. "
            "Use 1 for strict serialization per GPU to avoid VRAM spikes.",
            "json_schema_extra": {"langgraph_nodes": ["retrieve_visual_memories"]},
        },
    )

    # ========== Memory Reuse Configuration ==========
    memory_list: list[str] = field(
        default_factory=list,
        metadata={
            "description": "List of previous output directories to merge memories from. "
            "Supports empty list, single directory, or multiple directories. "
            "All memories from these directories will be merged and usage counts reset to 0. "
            "Examples: [], ['output/run1/20241107-120000'], "
            "['output/run1/20241107-120000', 'output/run2/20241107-130000']",
        },
    )

    def __post_init__(self) -> None:
        """Fetch environment variables for attributes that were not passed as args.

        This follows the same pattern as common.context.Context:
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
