"""Provider registration for load_chat_model.

All providers are registered as OpenAI-compatible endpoints. DashScope's
`/compatible-mode/v1` endpoint is fully compatible with the OpenAI Chat
Completions API for the Qwen model family (including qwen-flash, qwen3-vl-*,
qwen3-next-*), which is what this project uses — so we do not need the
`langchain-qwq` wrapper.
"""

import os

from langchain_dev_utils.chat_models import register_model_provider

_providers_registered = False


def _get_dashscope_base_url() -> str:
    """Return the DashScope OpenAI-compatible endpoint for the current region."""
    region = os.getenv("REGION", "").lower()
    if region in ("prc", "cn"):
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def register_all_providers() -> None:
    """Register every configured provider (runs lazily on first load_chat_model)."""
    global _providers_registered
    if _providers_registered:
        return

    if os.getenv("OPENAI_API_KEY"):
        register_model_provider(
            provider_name="openai",
            chat_model="openai-compatible",
            base_url=os.getenv("OPENAI_API_BASE"),
        )

    if os.getenv("OPENAI2_API_KEY"):
        register_model_provider(
            provider_name="openai2",
            chat_model="openai-compatible",
            base_url=os.getenv("OPENAI2_API_BASE"),
        )

    if os.getenv("LOCAL_MODEL_BASE_URL") or os.getenv("LOCAL_VLLM_API_BASE"):
        register_model_provider(
            provider_name="local",
            chat_model="openai-compatible",
            base_url=os.getenv("LOCAL_MODEL_BASE_URL") or os.getenv("LOCAL_VLLM_API_BASE"),
        )

    if os.getenv("VLLM_API_BASE"):
        register_model_provider(
            provider_name="vllm",
            chat_model="openai-compatible",
            base_url=os.getenv("VLLM_API_BASE"),
        )

    # Qwen/DashScope via the OpenAI-compatible endpoint.
    # langchain-dev-utils reads QWEN_API_KEY by convention; mirror the
    # DASHSCOPE_API_KEY into it so users only need to set one env var.
    if os.getenv("DASHSCOPE_API_KEY"):
        os.environ.setdefault("QWEN_API_KEY", os.environ["DASHSCOPE_API_KEY"])
        register_model_provider(
            provider_name="qwen",
            chat_model="openai-compatible",
            base_url=_get_dashscope_base_url(),
        )

    _providers_registered = True
