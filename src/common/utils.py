"""Utility & helper functions."""

import re

from langchain_core.messages import BaseMessage


def normalize_region(region: str) -> str | None:
    """Normalize region aliases to standard values.

    Args:
        region: Region string to normalize

    Returns:
        Normalized region ('prc' or 'international') or None if invalid
    """
    if not region:
        return None

    region_lower = region.lower()
    if region_lower in ("prc", "cn"):
        return "prc"
    elif region_lower in ("international", "en"):
        return "international"
    return None


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)
ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
ANSWER_OPEN_RE = re.compile(r"<answer>(.*)", re.DOTALL | re.IGNORECASE)


def strip_reasoning_tags(text: str) -> str:
    """Remove <think>/<answer> wrappers, keeping only the final answer text."""

    if not text:
        return ""

    # Prefer explicit <answer> blocks if present
    answer_match = ANSWER_BLOCK_RE.search(text)
    if answer_match:
        return answer_match.group(1).strip()

    answer_open = ANSWER_OPEN_RE.search(text)
    if answer_open:
        return answer_open.group(1).strip()

    cleaned = THINK_BLOCK_RE.sub("", text)
    cleaned = OPEN_THINK_RE.sub("", cleaned)
    cleaned = re.sub(r"</answer>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message with reasoning tags removed."""

    content = msg.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, dict):
        text = content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        text = "".join(txts)

    return strip_reasoning_tags(text)
