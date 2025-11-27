"""Shared utilities for loading and processing local datasets."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any


def build_message_content(
    question: str,
    image_paths: list[Path] | Path | None,
    system_prompt: str,
    sample_metadata: dict[str, Any] | None = None,
    benchmark_config: Any | None = None,
) -> list:
    """Build message content with question and images from local files.

    Args:
        question: The question text (may be reconstructed with options if benchmark_config provided)
        image_paths: Path(s) to image file(s) (single Path, list of Paths, or None)
        system_prompt: System prompt template with {question} placeholder
        sample_metadata: Optional sample metadata (for prompt reconstruction)
        benchmark_config: Optional BenchmarkPromptConfig (for task-specific formatting)

    Returns:
        List of content blocks (text + images)
    """
    # Normalize image_paths to list
    if image_paths is None:
        image_paths_list: list[Path] = []
    elif isinstance(image_paths, Path):
        image_paths_list = [image_paths]
    else:
        image_paths_list = image_paths

    # If benchmark config provided, use prompt_builder to reconstruct
    # VLMEvalKit compatibility: Use original question format without reconstruction
    # The question field in VLMEvalKit datasets already contains the complete formatted prompt:
    # - "Hint: ...\nQuestion: ...\nChoices:\n(A) ...\n(B) ..." for MCQ
    # - Most VLMs use line['question'] directly without modification
    # Reference: https://github.com/open-compass/VLMEvalKit (llama4, InternVLChat, etc.)

    # Build content: [images..., text with original question]
    content = []

    # Add images first (VLMEvalKit pattern: images before text)
    for image_path in image_paths_list:
        if image_path.exists():
            content.append(_encode_image_to_content(image_path))

    # Add question text
    # VLMEvalKit compatibility: Use question directly if system_prompt is empty
    # Most VLMEvalKit models (llama4, InternVLChat, etc.) use line['question'] directly
    if not system_prompt or system_prompt.strip() == "":
        # Empty system_prompt = use question directly (VLMEvalKit pattern)
        prompt_text = question
    elif system_prompt.strip() == "{question}":
        # Simple placeholder - use question directly
        prompt_text = question
    elif "{question}" in system_prompt:
        # Complex system prompt - inject question
        prompt_text = system_prompt.replace("{question}", question)
    else:
        # System prompt without placeholder - append question
        prompt_text = f"{system_prompt}\n\n{question}"

    content.append({"type": "text", "text": prompt_text})

    return content


def _encode_image_to_content(image_path: Path) -> dict:
    """Encode image file to base64 content block.

    Args:
        image_path: Path to image file

    Returns:
        Content block dict with base64-encoded image
    """
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        # Default to jpeg if can't determine
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
        }


def load_local_dataset(
    dataset_path: str, show_progress: bool = True
) -> list[dict[str, Any]]:
    """Load examples from local JSON or JSONL file.

    Args:
        dataset_path: Path to the dataset file or directory
        show_progress: Whether to show progress bar (default: True)

    Returns:
        List of example dictionaries

    Raises:
        ValueError: If file format is unsupported or path doesn't exist
    """
    import logging

    logger = logging.getLogger("qwen_vl_eval.dataset")

    try:
        from tqdm import tqdm

        use_tqdm = show_progress
    except ImportError:
        use_tqdm = False
        if show_progress:
            logger.warning("tqdm not installed, progress bar will not be shown")

    dataset_path_obj = Path(dataset_path)

    if dataset_path_obj.is_file():
        logger.info(f"Loading dataset from file: {dataset_path_obj}")
        # Single file: JSON or JSONL
        if dataset_path_obj.suffix == ".jsonl":
            examples = []
            logger.debug("Reading JSONL file...")
            with open(dataset_path_obj, encoding="utf-8") as f:
                lines = f.readlines()

            logger.info(f"Parsing {len(lines)} lines from JSONL file")
            line_iterator = (
                tqdm(lines, desc="Loading examples", unit="line") if use_tqdm else lines
            )

            for line in line_iterator:
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line: {e}")

            logger.info(f"Successfully loaded {len(examples)} examples from JSONL")
            return examples
        elif dataset_path_obj.suffix == ".json":
            logger.debug("Reading JSON file...")
            with open(dataset_path_obj, encoding="utf-8") as f:
                data = json.load(f)
                # Handle both list and dict formats
                if isinstance(data, list):
                    logger.info(
                        f"Successfully loaded {len(data)} examples from JSON (list format)"
                    )
                    return data
                elif isinstance(data, dict) and "examples" in data:
                    examples = data["examples"]
                    logger.info(
                        f"Successfully loaded {len(examples)} examples from JSON (dict format)"
                    )
                    return examples
                else:
                    logger.info(
                        "Successfully loaded 1 example from JSON (single object)"
                    )
                    return [data]
        else:
            raise ValueError(f"Unsupported file format: {dataset_path_obj.suffix}")
    elif dataset_path_obj.is_dir():
        # Directory: look for JSONL files
        logger.info(f"Searching for JSONL files in directory: {dataset_path_obj}")
        jsonl_files = list(dataset_path_obj.glob("*.jsonl"))
        if jsonl_files:
            logger.info(
                f"Found {len(jsonl_files)} JSONL file(s), using: {jsonl_files[0].name}"
            )
            # Use first JSONL file found
            return load_local_dataset(str(jsonl_files[0]), show_progress=show_progress)
        else:
            raise ValueError(f"No JSONL files found in directory: {dataset_path}")
    else:
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
