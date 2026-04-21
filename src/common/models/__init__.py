"""Unified chat model loader."""

from langchain_core.language_models import BaseChatModel
from langchain_dev_utils.chat_models import load_chat_model as _ldu_load_chat_model

from .registry import register_all_providers

__all__ = ["load_chat_model"]


def load_chat_model(fully_specified_name: str, **kwargs) -> BaseChatModel:
    """Load a chat model by `provider:model_name` identifier.

    Supported providers: openai, qwen, local, vllm.
    """
    register_all_providers()
    return _ldu_load_chat_model(fully_specified_name, **kwargs)
