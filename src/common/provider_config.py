"""Shared helpers for resolving model provider API credentials."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .utils import normalize_region


@dataclass(frozen=True)
class ModelAPIConfig:
    """Resolved API configuration for a provider:model string."""

    provider: str
    model: str
    api_key: str
    api_base: Optional[str]


def _ensure_chat_completions_endpoint(base_url: str | None) -> str | None:
    if not base_url:
        return base_url

    trimmed = base_url.rstrip("/")
    if trimmed.upper() == "OFFICIAL":
        return trimmed

    if "/chat/completions" in trimmed.split("?")[0]:
        return trimmed

    return f"{trimmed}/chat/completions"


def _resolve_dashscope_base_url() -> str:
    env_base = os.getenv("DASHSCOPE_API_BASE")
    if env_base:
        return env_base

    region = normalize_region(os.getenv("REGION") or "")
    if region == "prc":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def _resolve_siliconflow_base_url() -> str:
    env_base = os.getenv("SILICONFLOW_API_BASE")
    if env_base:
        return env_base

    region = normalize_region(os.getenv("REGION") or "")
    if region == "prc":
        return "https://api.siliconflow.cn/v1"
    if region == "international":
        return "https://api.siliconflow.com/v1"
    return "https://api.siliconflow.com/v1"


def get_api_config(model_name: str) -> ModelAPIConfig:
    """Resolve API credentials for a provider:model string."""

    if ":" in model_name:
        provider, actual_model = model_name.split(":", 1)
    else:
        provider, actual_model = "openai", model_name

    provider_lower = provider.lower()

    if provider_lower in {"openai", "openai2"}:
        env_prefix = "OPENAI" if provider_lower == "openai" else "OPENAI2"
        api_key = os.getenv(f"{env_prefix}_API_KEY")
        if not api_key:
            raise ValueError(
                f"API key not found for provider '{provider}'. Set {env_prefix}_API_KEY in the environment."
            )
        api_base = os.getenv(f"{env_prefix}_API_BASE")
        return ModelAPIConfig(
            provider_lower,
            actual_model,
            api_key,
            _ensure_chat_completions_endpoint(api_base),
        )

    if provider_lower in {"qwen", "dashscope"}:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY is required for Qwen/DashScope providers"
            )
        base_url = _resolve_dashscope_base_url()
        return ModelAPIConfig(
            provider_lower,
            actual_model,
            api_key,
            _ensure_chat_completions_endpoint(base_url),
        )

    if provider_lower == "siliconflow":
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise ValueError(
                "SILICONFLOW_API_KEY is required for siliconflow providers"
            )
        base_url = _ensure_chat_completions_endpoint(_resolve_siliconflow_base_url())
        return ModelAPIConfig(provider_lower, actual_model, api_key, base_url)

    if provider_lower == "local":
        base_url = os.getenv("LOCAL_MODEL_BASE_URL")
        if not base_url:
            raise ValueError("LOCAL_MODEL_BASE_URL is required for local providers")
        api_key = os.getenv("LOCAL_MODEL_API_KEY", "sk-local-vllm")
        return ModelAPIConfig(
            provider_lower,
            actual_model,
            api_key,
            _ensure_chat_completions_endpoint(base_url),
        )

    raise ValueError(
        f"Unknown model provider '{provider}'. Supported providers: openai, openai2, qwen/dashscope, siliconflow, local."
    )
