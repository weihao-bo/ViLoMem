"""Utilities for constructing multimodal prompts."""

from __future__ import annotations

from typing import Any, Iterable, Tuple

from langchain_core.messages import HumanMessage


def _split_content(content: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Split a multimodal message content into image and text blocks."""

    image_blocks: list[dict[str, Any]] = []
    text_blocks: list[str] = []

    if isinstance(content, str):
        text_blocks.append(content)
        return image_blocks, text_blocks

    if not isinstance(content, list):
        return image_blocks, text_blocks

    for item in content:
        if isinstance(item, dict):
            block_type = item.get("type")
            if block_type == "image_url":
                image_blocks.append(item)
            elif block_type == "text":
                text_blocks.append(item.get("text", ""))
        elif isinstance(item, str):
            text_blocks.append(item)

    return image_blocks, text_blocks


def _extract_vlmeval_text(vlmeval_prompt: Any) -> list[str]:
    """Extract text segments from a VLMEvalKit prompt structure."""

    if not isinstance(vlmeval_prompt, list):
        return []

    text_segments: list[str] = []
    for item in vlmeval_prompt:
        if isinstance(item, dict) and item.get("type") == "text":
            text_segments.append(item.get("value", ""))
    return text_segments


def _sanitize_section(section: str | None) -> str | None:
    if section is None:
        return None
    section = section.strip()
    return section or None


def build_combined_prompt(
    *,
    original_content: Any,
    vlmeval_prompt: Any = None,
    system_section: str | None = None,
    memory_section: str | None = None,
) -> tuple[HumanMessage, dict[str, Any], str]:
    """Combine system/memory/VLMEval sections into a single human message."""

    image_blocks, fallback_text_blocks = _split_content(original_content)
    vl_text_blocks = _extract_vlmeval_text(vlmeval_prompt)
    vlmeval_used = bool(vl_text_blocks)

    core_text_blocks = vl_text_blocks if vlmeval_used else fallback_text_blocks
    core_text_blocks = [block for block in core_text_blocks if isinstance(block, str)]
    if not core_text_blocks:
        core_text_blocks = [""]

    system_section = _sanitize_section(system_section)
    memory_section = _sanitize_section(memory_section)

    text_blocks: list[str] = []
    if system_section:
        text_blocks.append(system_section)
    if memory_section:
        text_blocks.append(memory_section)

    text_blocks.extend(block for block in core_text_blocks if block.strip())

    if not text_blocks:
        text_blocks = [core_text_blocks[-1]]

    combined_content: list[Any] = list(image_blocks)
    for block in text_blocks:
        combined_content.append({"type": "text", "text": block})

    combined_text = "\n\n".join(text_blocks).strip()

    metadata = {
        "vlmeval_prompt_used": vlmeval_used,
        "memory_inserted": bool(memory_section),
        "system_prompt_injected": bool(system_section),
        "image_count": len(image_blocks),
        "text_block_count": len(text_blocks),
    }

    return HumanMessage(content=combined_content), metadata, combined_text
