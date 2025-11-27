"""OpenAI-compatible model integrations."""

from __future__ import annotations

import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI


def create_openai_model(
    model_name: str,
    *,
    env_prefix: str = "OPENAI",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Create an OpenAI-compatible chat model.

    Args:
        model_name: The target model name (e.g., 'gpt-4o-mini').
        env_prefix: Prefix for env vars (e.g., 'OPENAI', 'OPENAI2').
        api_key: Optional API key override.
        base_url: Optional base URL override.
        **kwargs: Additional parameters forwarded to ChatOpenAI.

    Returns:
        Configured ChatOpenAI instance with the requested settings.
    """

    api_key = api_key or os.getenv(f"{env_prefix}_API_KEY")
    base_url = base_url or os.getenv(f"{env_prefix}_API_BASE")

    config: dict[str, Any] = {"model": model_name, **kwargs}

    if api_key:
        config["api_key"] = api_key
    if base_url:
        config["base_url"] = base_url

    return ChatOpenAI(**config)
