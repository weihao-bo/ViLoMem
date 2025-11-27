"""Local model integrations via vLLM OpenAI-compatible API.

This module provides integration with locally deployed models via vLLM server.
vLLM provides an OpenAI-compatible API, allowing seamless integration with
LangChain's ChatOpenAI interface.

Usage:
    1. Deploy VL model:
       # Single GPU
       bash scripts/manage_single_vllm.sh --gpu-id 0 --port 18000

       # Multi-GPU (tensor parallel)
       bash scripts/manage_single_vllm.sh --gpu-id "0,1,2,3" --port 18000

    2. Deploy rerank model:
       bash scripts/manage_rerank.sh --gpu-id 0 --port 19000

    3. Configure environment variables:
       # VL Model
       export LOCAL_MODEL_BASE_URL="http://localhost:18000/v1"
       export LOCAL_MODEL_API_KEY="sk-local-vllm"

       # Rerank Model
       export LOCAL_RERANK_BASE_URL="http://localhost:19000/v1"
       export LOCAL_RERANK_API_KEY="sk-local-vllm"

    4. Use in YAML config:
       model:
         name: "local:qwen3-vl-8b-instruct"

       rerank_model: "local:qwen3-rerank"

    5. Use in code:
       from common.models import create_local_model
       vl_model = create_local_model("qwen3-vl-8b-instruct")
       rerank_model = create_local_model("qwen3-rerank")
"""

import os
from typing import Any, Optional

from langchain_openai import ChatOpenAI


def get_model_config(model_name: str) -> tuple[str, str]:
    """Get base URL and API key for a specific model.

    Args:
        model_name: Model name (e.g., 'qwen3-vl-8b-instruct', 'qwen3-rerank')

    Returns:
        Tuple of (base_url, api_key)
    """
    # Model-specific configuration
    if "rerank" in model_name.lower():
        base_url = os.getenv("LOCAL_RERANK_BASE_URL", "http://localhost:19000/v1")
        api_key = os.getenv("LOCAL_RERANK_API_KEY", "sk-local-vllm")
    else:
        # Default to VL model configuration
        base_url = os.getenv("LOCAL_MODEL_BASE_URL", "http://localhost:18000/v1")
        api_key = os.getenv("LOCAL_MODEL_API_KEY", "sk-local-vllm")

    return base_url, api_key


def create_local_model(
    model_name: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """Create a locally deployed model client using vLLM OpenAI-compatible API.

    This function creates a ChatOpenAI instance configured to communicate with
    a local vLLM server. It automatically determines the correct endpoint based
    on the model name.

    Args:
        model_name: The served model name (e.g., 'qwen3-vl-8b-instruct', 'qwen3-rerank')
                   This should match the --served-model-name parameter used
                   when starting the vLLM server.
        base_url: Base URL for the vLLM server API (e.g., 'http://localhost:18000/v1')
                 If not provided, automatically determined based on model name.
        api_key: API key for authentication
                If not provided, automatically determined based on model name.
        **kwargs: Additional model parameters (temperature, max_tokens, etc.)

    Returns:
        Configured ChatOpenAI instance connected to local vLLM server

    Environment Variables:
        LOCAL_MODEL_BASE_URL: Base URL for VL model (default: http://localhost:18000/v1)
        LOCAL_MODEL_API_KEY: API key for VL model (default: sk-local-vllm)
        LOCAL_RERANK_BASE_URL: Base URL for rerank model (default: http://localhost:19000/v1)
        LOCAL_RERANK_API_KEY: API key for rerank model (default: sk-local-vllm)

    Example:
        >>> # Deploy VL model
        >>> # bash scripts/manage_single_vllm.sh --gpu-id "0,1,2,3" --port 18000
        >>>
        >>> # Deploy rerank model
        >>> # bash scripts/manage_rerank.sh --gpu-id 0 --port 19000
        >>>
        >>> # Use VL model (routes to LOCAL_MODEL_BASE_URL)
        >>> vl_model = create_local_model("qwen3-vl-8b-instruct")
        >>>
        >>> # Use rerank model (routes to LOCAL_RERANK_BASE_URL)
        >>> rerank_model = create_local_model("qwen3-rerank")

    Notes:
        - The vLLM server must be started before using this function
        - The model_name must exactly match the --served-model-name parameter
        - For multimodal models, use content with image_url format:
          [
              {"type": "text", "text": "What is in this image?"},
              {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
          ]
    """
    # Get configuration based on model name
    if base_url is None or api_key is None:
        default_base_url, default_api_key = get_model_config(model_name)
        base_url = base_url or default_base_url
        api_key = api_key or default_api_key

    # Create ChatOpenAI instance with vLLM server configuration
    config = {
        "model": model_name,
        "base_url": base_url,
        "api_key": api_key,
        **kwargs,
    }

    return ChatOpenAI(**config)
