"""Utility & helper functions."""

import re
from typing import Optional, Union

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_qwq import ChatQwen, ChatQwQ


def normalize_region(region: str) -> Optional[str]:
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


def load_chat_model(
    fully_specified_name: str,
) -> Union[BaseChatModel, ChatQwQ, ChatQwen]:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider:model'.
                                   Supported providers:
                                   - 'local': Locally deployed models via vLLM
                                   - 'qwen': Qwen models via DashScope
                                   - 'siliconflow': SiliconFlow models
                                   - Other providers via LangChain init_chat_model
    """
    provider, model = fully_specified_name.split(":", maxsplit=1)
    provider_lower = provider.lower()

    # Handle OpenAI-compatible providers with explicit env prefixes
    if provider_lower in {"openai", "openai2"}:
        from .models import create_openai_model

        env_prefix = "OPENAI" if provider_lower == "openai" else "OPENAI2"
        return create_openai_model(model, env_prefix=env_prefix)

    # Handle locally deployed models via vLLM
    if provider_lower == "local":
        from .models import create_local_model

        return create_local_model(model)

    # Handle Qwen models specially with dashscope integration
    if provider_lower == "qwen":
        from .models import create_qwen_model

        return create_qwen_model(model)

    # Handle SiliconFlow models
    if provider_lower == "siliconflow":
        from .models import create_siliconflow_model

        return create_siliconflow_model(model)

    # Use standard langchain initialization for other providers
    return init_chat_model(model, model_provider=provider)
