"""Memory storage and retrieval utilities for Qwen VL agent.

This module provides functions for:
1. JSON-based memory storage (append-only)
2. Text Embedding API for semantic search (text-based logic and visual memory)
3. DashScope Multimodal Embedding API for visual memory (image-based)
4. Memory data structure utilities
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

import httpx
import numpy as np

from common.retry import async_retry_with_backoff

# Get logger - will inherit configuration from parent qwen_vl_eval logger
logger = logging.getLogger("qwen_vl_eval.memory")


def _determine_dashscope_min_interval() -> float:
    """Determine minimum interval between DashScope requests.

    Priority:
    1. DASHSCOPE_EMBEDDING_MIN_INTERVAL (seconds)
    2. DASHSCOPE_EMBEDDING_QPS (requests per second)
    3. Default: 0.25s (~4 QPS)
    """

    interval_env = os.environ.get("DASHSCOPE_EMBEDDING_MIN_INTERVAL")
    if interval_env:
        try:
            interval = float(interval_env)
            if interval >= 0:
                return interval
        except ValueError:
            logger.warning(
                "Invalid DASHSCOPE_EMBEDDING_MIN_INTERVAL=%s; falling back to default",
                interval_env,
            )

    qps_env = os.environ.get("DASHSCOPE_EMBEDDING_QPS")
    if qps_env:
        try:
            qps = float(qps_env)
            if qps > 0:
                return 1.0 / qps
        except ValueError:
            logger.warning(
                "Invalid DASHSCOPE_EMBEDDING_QPS=%s; falling back to default",
                qps_env,
            )

    return 0.25


_DASHSCOPE_MIN_INTERVAL = _determine_dashscope_min_interval()
_dashscope_rate_limiter_lock: asyncio.Lock | None = None
_dashscope_next_request_time: float = 0.0


async def _throttle_dashscope_embeddings() -> None:
    """Enforce a minimum delay between DashScope embedding requests."""

    if _DASHSCOPE_MIN_INTERVAL <= 0:
        return

    global _dashscope_rate_limiter_lock, _dashscope_next_request_time

    if _dashscope_rate_limiter_lock is None:
        _dashscope_rate_limiter_lock = asyncio.Lock()

    loop = asyncio.get_running_loop()

    while True:
        async with _dashscope_rate_limiter_lock:
            now = loop.time()
            wait = _dashscope_next_request_time - now
            if wait <= 0:
                _dashscope_next_request_time = now + _DASHSCOPE_MIN_INTERVAL
                return

        await asyncio.sleep(min(wait, _DASHSCOPE_MIN_INTERVAL))

# ========== JSON Storage Functions ==========


def load_memories(memory_file: str | Path) -> list[dict[str, Any]]:
    """Load all memories from JSON file.

    Args:
        memory_file: Path to the memories JSON file

    Returns:
        List of memory dictionaries (empty list if file doesn't exist)
    """
    memory_path = Path(memory_file)
    if not memory_path.exists():
        return []

    try:
        with open(memory_path, encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_memory(memory_file: str | Path, memory: dict[str, Any]) -> None:
    """Append a new memory to the JSON file.

    Args:
        memory_file: Path to the memories JSON file
        memory: Memory dictionary to append
    """
    memory_path = Path(memory_file)
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing memories
    memories = load_memories(memory_path)

    # Append new memory
    memories.append(memory)

    # Write back to file
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)


def update_memory_usage(
    memory_file: str | Path, memory_id: str, timestamp: str
) -> None:
    """Update usage count and last_used_at for a memory.

    Args:
        memory_file: Path to the memories JSON file
        memory_id: Memory ID to update
        timestamp: ISO format timestamp for last_used_at
    """
    memory_path = Path(memory_file)
    if not memory_path.exists():
        return

    # Load memories
    memories = load_memories(memory_path)

    # Find and update the memory
    for memory in memories:
        if memory.get("memory_id") == memory_id:
            memory["usage_count"] = memory.get("usage_count", 0) + 1
            memory["last_used_at"] = timestamp
            break

    # Write back
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)


def update_memory(
    memory_file: str | Path, memory_id: str, updated_memory: dict[str, Any]
) -> None:
    """Update an existing memory in the JSON file.

    Args:
        memory_file: Path to the memories JSON file
        memory_id: Memory ID to update
        updated_memory: Updated memory dictionary (will replace the existing one)
    """
    memory_path = Path(memory_file)
    if not memory_path.exists():
        return

    # Load memories
    memories = load_memories(memory_path)

    # Find and replace the memory
    for i, memory in enumerate(memories):
        if memory.get("memory_id") == memory_id:
            # Preserve memory_id and created_at
            updated_memory["memory_id"] = memory_id
            if "created_at" in memory:
                updated_memory["created_at"] = memory["created_at"]
            memories[i] = updated_memory
            break

    # Write back
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)


def merge_memories_from_directories(
    output_dirs: list[str | Path],
    memory_filename: str,
    reset_usage_count: bool = True,
) -> list[dict[str, Any]]:
    """Merge memories from multiple output directories.

    Args:
        output_dirs: List of output directory paths to load memories from
        memory_filename: Memory filename (e.g., "logic_memories.json" or "visual_memories.json")
        reset_usage_count: Whether to reset usage_count to 0 for all memories (default: True)

    Returns:
        Merged list of memories with duplicates removed (based on memory_id)
    """
    merged_memories = []
    seen_ids = set()

    for output_dir in output_dirs:
        memory_file = Path(output_dir) / memory_filename
        if not memory_file.exists():
            logger.warning(f"Memory file not found: {memory_file}, skipping")
            continue

        memories = load_memories(memory_file)
        logger.info(f"Loaded {len(memories)} memories from {memory_file}")

        for memory in memories:
            memory_id = memory.get("memory_id")
            if memory_id and memory_id in seen_ids:
                # Skip duplicate memory_id
                logger.debug(f"Skipping duplicate memory_id: {memory_id}")
                continue

            # Reset usage count if requested
            if reset_usage_count:
                memory["usage_count"] = 0
                memory["last_used_at"] = None

            merged_memories.append(memory)
            if memory_id:
                seen_ids.add(memory_id)

    logger.info(
        f"Merged {len(merged_memories)} unique memories from {len(output_dirs)} directories"
    )
    return merged_memories


# ========== Text Embedding API (Local vLLM) ==========


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    import math

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


@async_retry_with_backoff(max_retries=3, initial_delay=1.0)
async def compute_text_embeddings(
    texts: list[str],
    model: str = "local:qwen3-embedding",
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[list[float]]:
    """Compute text embeddings using multiple providers (local vLLM, DashScope, OpenAI).

    Args:
        texts: List of texts to embed
        model: Model name in 'provider:model' format (default: 'local:qwen3-embedding')
               Supported providers:
               - local: Local vLLM embedding API (e.g., 'local:qwen3-embedding')
               - qwen/dashscope: DashScope text embedding API (e.g., 'qwen:text-embedding-v2')
               - openai: OpenAI text embedding API (e.g., 'openai:text-embedding-3-small')
        api_key: API key (defaults to provider-specific env vars)
        base_url: API base URL (for local provider only, defaults to LOCAL_EMBEDDING_BASE_URL)

    Returns:
        List of embedding vectors

    Raises:
        ValueError: If API key is not provided or provider is unsupported
        httpx.HTTPError: If API request fails
    """
    if not texts:
        return []

    # Parse provider and model
    if ":" in model:
        provider, model_name = model.split(":", maxsplit=1)
    else:
        provider, model_name = "local", model

    provider_lower = provider.lower()

    # Route to provider-specific implementation
    if provider_lower == "local":
        return await _compute_local_text_embeddings(texts, model_name, api_key, base_url)
    elif provider_lower in {"qwen", "dashscope"}:
        return await _compute_dashscope_text_embeddings(texts, model_name, api_key)
    elif provider_lower == "openai":
        return await _compute_openai_text_embeddings(texts, model_name, api_key, base_url)
    else:
        raise ValueError(
            f"Unsupported text embedding provider: {provider}. "
            f"Supported providers: local, qwen/dashscope, openai"
        )


async def _compute_local_text_embeddings(
    texts: list[str],
    model_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[list[float]]:
    """Compute text embeddings using local vLLM embedding API."""
    # Get API credentials
    if not api_key:
        api_key = os.environ.get(
            "LOCAL_EMBEDDING_API_KEY", os.environ.get("LOCAL_VLLM_API_KEY")
        )
    if not api_key:
        raise ValueError(
            "LOCAL_EMBEDDING_API_KEY (or LOCAL_VLLM_API_KEY) environment variable must be set "
            "to use local embedding API"
        )

    if not base_url:
        base_url = os.environ.get(
            "LOCAL_EMBEDDING_BASE_URL", "http://localhost:19001/v1"
        )

    normalized_base = base_url.rstrip("/")
    if not normalized_base.endswith("/v1"):
        normalized_base = f"{normalized_base}/v1"

    embeddings_url = f"{normalized_base}/embeddings"

    # Prepare request
    payload = {
        "model": model_name,
        "input": texts,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Call API
    async with httpx.AsyncClient() as client:
        response = await client.post(
            embeddings_url,
            headers=headers,
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()

    # Extract embeddings from response
    # Expected format: {"data": [{"embedding": [...], "index": 0}, ...]}
    data = result.get("data", [])
    if not data:
        logger.warning(f"Embeddings API returned empty data. Response: {result}")
        return [[] for _ in texts]

    # Sort by index and extract embeddings
    data_sorted = sorted(data, key=lambda x: x.get("index", 0))
    embeddings = [item.get("embedding", []) for item in data_sorted]

    return embeddings


async def _compute_dashscope_text_embeddings(
    texts: list[str],
    model_name: str,
    api_key: str | None = None,
) -> list[list[float]]:
    """Compute text embeddings using DashScope text embedding API.
    
    DashScope text embedding API format:
    - Endpoint: /api/v1/services/embeddings/text-embedding/text-embedding
    - Input: {"model": "...", "input": {"texts": ["text1", "text2", ...]}}
    - Output: {"output": {"embeddings": [{"embedding": [...], "text_index": 0}, ...]}}
    """
    if not api_key:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "DASHSCOPE_API_KEY environment variable must be set "
            "to use DashScope text embedding API"
        )

    # Determine region endpoint (support both PRC and international)
    region = os.environ.get("REGION", "cn").lower()
    if region in {"prc", "cn"}:
        base_url = "https://dashscope.aliyuncs.com"
    else:
        base_url = "https://dashscope-intl.aliyuncs.com"

    # DashScope text embedding API endpoint
    url = f"{base_url}/api/v1/services/embeddings/text-embedding/text-embedding"

    # DashScope limits concurrent texts per request (10 for v4/v3, 25 for v2/v1, default 10)
    model_lower = model_name.lower()
    if model_lower.startswith("text-embedding-v4") or model_lower.startswith(
        "text-embedding-v3"
    ):
        max_batch_size = 10
    elif model_lower.startswith("text-embedding-v2") or model_lower.startswith(
        "text-embedding-v1"
    ):
        max_batch_size = 25
    else:
        max_batch_size = 10

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    embeddings: list[list[float]] = []

    max_batch_retries = 5

    async with httpx.AsyncClient() as client:
        for start in range(0, len(texts), max_batch_size):
            batch_texts = texts[start : start + max_batch_size]

            # Prepare request payload
            payload = {
                "model": model_name,
                "input": {"texts": batch_texts},
            }

            await _throttle_dashscope_embeddings()
            attempt = 0
            while True:
                attempt += 1
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=120.0,
                )

                try:
                    response.raise_for_status()
                    break
                except httpx.HTTPStatusError as exc:
                    status_code = exc.response.status_code
                    if status_code == 429 and attempt < max_batch_retries:
                        retry_after_header = exc.response.headers.get("Retry-After")
                        retry_after = None
                        if retry_after_header:
                            try:
                                retry_after = float(retry_after_header)
                            except ValueError:
                                retry_after = None
                        backoff_delay = retry_after or min(10.0, 1.5 ** attempt)
                        request_id = exc.response.headers.get(
                            "X-DashScope-Request-Id"
                        ) or exc.response.headers.get("x-ca-request-id")
                        logger.warning(
                            "DashScope rate limit hit (attempt %s/%s, request %s). "
                            "Retrying in %.2fs",
                            attempt,
                            max_batch_retries,
                            request_id,
                            backoff_delay,
                        )
                        await asyncio.sleep(backoff_delay)
                        continue

                    body_snippet = exc.response.text[:200]
                    logger.error(
                        "DashScope embedding request failed (status %s): %s",
                        status_code,
                        body_snippet,
                    )
                    raise

            result = response.json()

            # Extract embeddings from DashScope response format
            output = result.get("output", {})
            embeddings_data = output.get("embeddings", [])

            if not embeddings_data:
                logger.warning(
                    "DashScope API returned empty embeddings for batch %s-%s. Response: %s",
                    start,
                    start + len(batch_texts) - 1,
                    result,
                )
                embeddings.extend([[] for _ in batch_texts])
                continue

            embeddings_data_sorted = sorted(
                embeddings_data, key=lambda x: x.get("text_index", 0)
            )
            embeddings.extend([item.get("embedding", []) for item in embeddings_data_sorted])

    return embeddings


async def _compute_openai_text_embeddings(
    texts: list[str],
    model_name: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[list[float]]:
    """Compute text embeddings using OpenAI-compatible text embedding API."""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set "
            "to use OpenAI text embedding API"
        )

    if not base_url:
        base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

    normalized_base = base_url.rstrip("/")
    if not normalized_base.endswith("/v1"):
        normalized_base = f"{normalized_base}/v1"

    embeddings_url = f"{normalized_base}/embeddings"

    # Prepare request
    payload = {
        "model": model_name,
        "input": texts,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Call API
    async with httpx.AsyncClient() as client:
        response = await client.post(
            embeddings_url,
            headers=headers,
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()

    # Extract embeddings from response
    # Expected format: {"data": [{"embedding": [...], "index": 0}, ...]}
    data = result.get("data", [])
    if not data:
        logger.warning(f"OpenAI API returned empty data. Response: {result}")
        return [[] for _ in texts]

    # Sort by index and extract embeddings
    data_sorted = sorted(data, key=lambda x: x.get("index", 0))
    embeddings = [item.get("embedding", []) for item in data_sorted]

    return embeddings


def get_text_embedding_path(
    memory_id: str,
    memory_type: str,
    output_dir: Path,
    model: str = "local:qwen3-embedding",
) -> Path:
    """Get the path for storing text embedding cache.

    Args:
        memory_id: Memory ID
        memory_type: "logic" or "visual"
        output_dir: Output directory
        model: Embedding model name

    Returns:
        Path to .npy file for storing embedding
    """
    # Normalize model name for directory naming
    model_name = model.split(":", 1)[1] if ":" in model else model
    model_name_normalized = model_name.replace("/", "_")

    # Create directory structure: {output_dir}/text_embeddings/{model_name}/{memory_type}/
    embedding_dir = output_dir / "text_embeddings" / model_name_normalized / memory_type
    embedding_dir.mkdir(parents=True, exist_ok=True)

    return embedding_dir / f"{memory_id}.npy"


def copy_cached_text_embeddings(
    memories: list[dict[str, Any]],
    memory_type: str,
    output_dir: Path,
    model: str,
    source_dirs: Sequence[str | Path],
) -> int:
    """Copy cached embeddings from previous output directories.

    This avoids redundant embedding API calls when reusing memory outputs.

    Returns:
        Number of embeddings successfully copied into ``output_dir``.
    """

    if not memories or not source_dirs:
        return 0

    model_name = model.split(":", 1)[1] if ":" in model else model
    model_name_normalized = model_name.replace("/", "_")

    copied = 0
    target_dir = output_dir / "text_embeddings" / model_name_normalized / memory_type
    target_dir.mkdir(parents=True, exist_ok=True)

    for memory in memories:
        memory_id = memory.get("memory_id")
        if not memory_id:
            continue

        target_path = target_dir / f"{memory_id}.npy"
        if target_path.exists():
            continue

        for source in source_dirs:
            source_path = (
                Path(source)
                / "text_embeddings"
                / model_name_normalized
                / memory_type
                / f"{memory_id}.npy"
            )
            if source_path.exists():
                try:
                    shutil.copy2(source_path, target_path)
                    copied += 1
                except OSError as exc:
                    logger.warning(
                        "Failed to copy cached %s embedding %s from %s: %s",
                        memory_type,
                        memory_id,
                        source_path,
                        exc,
                    )
                break

    return copied


async def get_or_compute_text_embedding(
    text: str,
    memory_id: str,
    memory_type: str,
    output_dir: Path,
    model: str = "local:qwen3-embedding",
    force_recompute: bool = False,
) -> list[float] | None:
    """Get or compute text embedding with caching.

    Args:
        text: Text to embed
        memory_id: Memory ID
        memory_type: "logic" or "visual"
        output_dir: Output directory
        model: Embedding model name
        force_recompute: Force recomputation even if cached

    Returns:
        Embedding vector, or None if computation fails
    """
    # 1. Check cache
    npy_path = get_text_embedding_path(memory_id, memory_type, output_dir, model)

    if not force_recompute and npy_path.exists():
        embedding_np = load_embedding_from_npy(npy_path)
        if embedding_np is not None:
            return embedding_np.tolist()

    # 2. Compute embedding
    try:
        embeddings = await compute_text_embeddings([text], model=model)
        if not embeddings or not embeddings[0]:
            logger.error(f"Failed to compute embedding for memory {memory_id}")
            return None

        # 3. Save to cache
        embedding_np = np.array(embeddings[0], dtype=np.float32)
        save_embedding_to_npy(embedding_np, npy_path)
        logger.debug(f"Cached text embedding for memory {memory_id} at {npy_path}")

        return embeddings[0]
    except Exception as e:
        logger.error(f"Error computing text embedding for memory {memory_id}: {e}")
        return None


async def batch_precompute_text_embeddings(
    memories: list[dict[str, Any]],
    memory_type: str,
    output_dir: Path,
    model: str = "local:qwen3-embedding",
    batch_size: int = 32,
    text_field: str = "guideline",
) -> dict[str, Path]:
    """Batch precompute text embeddings for memories.

    Args:
        memories: List of memory dictionaries
        memory_type: "logic" or "visual"
        output_dir: Output directory
        model: Embedding model name
        batch_size: Batch size for API calls
        text_field: Field name containing text to embed (default: "guideline")

    Returns:
        Dictionary mapping memory_id to embedding file path
    """
    from tqdm import tqdm

    embedding_paths = {}
    memories_to_compute = []

    # Filter memories that need computation
    for memory in memories:
        memory_id = memory.get("memory_id", "")
        if not memory_id:
            continue

        npy_path = get_text_embedding_path(memory_id, memory_type, output_dir, model)
        if npy_path.exists():
            embedding_paths[memory_id] = npy_path
        else:
            text = memory.get(text_field, "")
            if text:
                memories_to_compute.append((memory_id, text, npy_path))

    if not memories_to_compute:
        logger.info(f"All {memory_type} text embeddings already cached")
        return embedding_paths

    logger.info(
        f"Computing {len(memories_to_compute)} {memory_type} text embeddings "
        f"(batch_size={batch_size})..."
    )

    # Process in batches
    failed_count = 0
    with tqdm(
        total=len(memories_to_compute),
        desc=f"Computing {memory_type} embeddings",
        unit="memory",
    ) as pbar:
        for i in range(0, len(memories_to_compute), batch_size):
            batch = memories_to_compute[i : i + batch_size]
            texts = [text for _, text, _ in batch]

            try:
                # Compute embeddings for batch
                embeddings = await compute_text_embeddings(texts, model=model)

                # Save each embedding
                for (memory_id, _, npy_path), embedding in zip(batch, embeddings):
                    if embedding:
                        embedding_np = np.array(embedding, dtype=np.float32)
                        save_embedding_to_npy(embedding_np, npy_path)
                        embedding_paths[memory_id] = npy_path
                    else:
                        logger.warning(
                            f"Empty embedding returned for memory {memory_id}"
                        )
                        failed_count += 1

                pbar.update(len(batch))

            except Exception as e:
                logger.error(f"Failed to compute embeddings for batch: {e}")
                failed_count += len(batch)
                pbar.update(len(batch))

    logger.info(
        f"Text embedding computation complete: {len(embedding_paths)} succeeded, {failed_count} failed"
    )

    return embedding_paths


async def retrieve_memories_by_text_embedding(
    query: str,
    memories: list[dict[str, Any]],
    memory_type: str,
    output_dir: Path,
    model: str = "local:qwen3-embedding",
    top_n: int = 5,
    similarity_threshold: float = 0.7,
) -> list[tuple[dict[str, Any], float]]:
    """Retrieve memories using text embedding similarity.

    Workflow:
    1. Compute query embedding
    2. Load all memory embeddings from cache
    3. Calculate cosine similarity
    4. Sort and filter by threshold

    Args:
        query: Query text
        memories: List of memory dictionaries
        memory_type: "logic" or "visual"
        output_dir: Output directory
        model: Embedding model name
        top_n: Return top N results
        similarity_threshold: Minimum similarity score

    Returns:
        List of (memory, similarity_score) tuples sorted by similarity (descending)
    """
    if not memories:
        return []

    # 1. Compute query embedding
    try:
        query_embeddings = await compute_text_embeddings([query], model=model)
        if not query_embeddings or not query_embeddings[0]:
            logger.error("Failed to compute query embedding")
            return []
        query_embedding = query_embeddings[0]
    except Exception as e:
        logger.error(f"Error computing query embedding: {e}")
        return []

    # 2. Load memory embeddings and compute similarities
    similarities = []
    missing_embeddings = 0

    for memory in memories:
        memory_id = memory.get("memory_id", "")
        if not memory_id:
            continue

        # Try to load cached embedding
        npy_path = get_text_embedding_path(memory_id, memory_type, output_dir, model)
        if not npy_path.exists():
            logger.debug(f"Embedding cache not found for memory {memory_id}")
            missing_embeddings += 1
            continue

        memory_embedding_np = load_embedding_from_npy(npy_path)
        if memory_embedding_np is None:
            logger.warning(f"Failed to load embedding for memory {memory_id}")
            continue

        memory_embedding = memory_embedding_np.tolist()

        # Calculate similarity
        similarity = _cosine_similarity(query_embedding, memory_embedding)
        similarities.append((memory, similarity))

    if missing_embeddings > 0:
        logger.warning(
            f"{missing_embeddings} memories have missing embeddings. "
            f"Run batch_precompute_text_embeddings() to cache them."
        )

    # 3. Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 4. Filter by threshold and limit to top_n
    filtered_results = [
        (memory, score)
        for memory, score in similarities
        if score >= similarity_threshold
    ][:top_n]

    logger.debug(
        f"Retrieved {len(filtered_results)} memories (threshold={similarity_threshold:.2f}, "
        f"top_n={top_n}) from {len(similarities)} candidates"
    )

    return filtered_results


# ========== Helper Functions ==========


def parse_problem_analysis(analysis_text: str) -> tuple[str, list[str]]:
    """Parse problem analysis output to extract subject and key concepts.

    Expected format:
        Subject: <subject>
        Key Concepts: <concept1, concept2, ...>

    Args:
        analysis_text: Raw analysis output from the model

    Returns:
        Tuple of (subject, key_concepts_list)
    """
    import re

    subject_match = re.search(r"Subject:\s*(.+?)(?:\n|$)", analysis_text, re.IGNORECASE)
    concepts_match = re.search(
        r"Key Concepts:\s*(.+?)(?:\n|$)", analysis_text, re.IGNORECASE
    )

    subject = subject_match.group(1).strip() if subject_match else "Unknown"

    # Parse concepts - handle comma-separated or newline-separated
    concepts_text = concepts_match.group(1).strip() if concepts_match else ""
    if "," in concepts_text:
        key_concepts = [c.strip() for c in concepts_text.split(",") if c.strip()]
    else:
        # Fallback: treat as single concept
        key_concepts = [concepts_text] if concepts_text else []

    return subject, key_concepts


def parse_error_analysis(analysis_text: str) -> tuple[str, str, str]:
    """Parse error analysis output to extract error type, summary, and guideline.

    Expected format:
        error_type: <Logical|Non-Logical>
        analysis_summary: <summary text>
        guideline: <guideline text or empty>

    Args:
        analysis_text: Raw error analysis output from the model

    Returns:
        Tuple of (error_type, analysis_summary, guideline)
    """
    import re

    error_type_match = re.search(
        r"error_type:\s*(.+?)(?:\n|$)", analysis_text, re.IGNORECASE
    )
    summary_match = re.search(
        r"analysis_summary:\s*(.+?)(?:\n|$)", analysis_text, re.IGNORECASE | re.DOTALL
    )
    guideline_match = re.search(
        r"guideline:\s*(.+?)(?:\n|$)", analysis_text, re.IGNORECASE | re.DOTALL
    )

    error_type = (
        error_type_match.group(1).strip() if error_type_match else "Non-Logical"
    )
    analysis_summary = summary_match.group(1).strip() if summary_match else ""
    guideline = guideline_match.group(1).strip() if guideline_match else ""

    # Clean up error type - ensure it's one of the valid values
    error_type = error_type.strip('"').strip("'")
    if error_type not in ("Logical", "Non-Logical"):
        error_type = "Non-Logical"

    return error_type, analysis_summary, guideline


# ========== Visual Memory Functions ==========


# ========== Benchmark-Scoped Path Utilities ==========


def make_benchmark_scoped_path(image_filename: str, benchmark: str) -> str:
    """Create benchmark-scoped image path for cross-benchmark memory reuse.

    Args:
        image_filename: Image filename (e.g., "15.jpg")
        benchmark: Benchmark name (e.g., "MathVision_MINI")

    Returns:
        Benchmark-scoped path (e.g., "MathVision_MINI/15.jpg")

    Examples:
        >>> make_benchmark_scoped_path("15.jpg", "MathVision_MINI")
        'MathVision_MINI/15.jpg'
    """
    if not image_filename or not benchmark:
        raise ValueError("Both image_filename and benchmark must be non-empty")
    return f"{benchmark}/{image_filename}"


def split_semicolon_values(value: str | None) -> list[str]:
    """Split a semicolon-separated string into trimmed components."""

    if not value:
        return []
    return [part.strip() for part in value.split(";") if part.strip()]


def normalize_semicolon_scoped_paths(scoped_value: str | None) -> list[str]:
    """Normalize semicolon-separated benchmark-scoped paths.

    Handles legacy storage format where only the first entry contains the
    benchmark prefix and subsequent entries only include filenames.
    """

    parts = split_semicolon_values(scoped_value)
    if not parts:
        return []

    normalized: list[str] = []
    benchmark_prefix = ""

    for part in parts:
        if "/" in part:
            benchmark_prefix = part.split("/", 1)[0]
            normalized.append(part)
        elif benchmark_prefix:
            normalized.append(f"{benchmark_prefix}/{part}")
        else:
            normalized.append(part)

    return normalized


def expand_embedding_path_value(embedding_path: str | None) -> list[str]:
    """Expand an embedding_path field that may encode multiple images.

    Example legacy format:
        MMMU/embeddings/model/1.jpg;2.jpg.npy ->
        [MMMU/embeddings/model/1.jpg.npy, MMMU/embeddings/model/2.jpg.npy]
    """

    if not embedding_path or not embedding_path.strip():
        return []

    value = embedding_path.strip()
    if ";" not in value:
        return [value]

    base_path = Path(value)
    parent_dir = str(base_path.parent)
    filename = base_path.name

    if filename.endswith(".npy"):
        filename = filename[: -len(".npy")]

    parts = split_semicolon_values(filename)
    if not parts:
        return [value]

    expanded_paths: list[str] = []
    for part in parts:
        if parent_dir and parent_dir != ".":
            expanded_paths.append(f"{parent_dir}/{part}.npy")
        else:
            expanded_paths.append(f"{part}.npy")

    return expanded_paths


def scoped_paths_to_embedding_paths(
    scoped_paths: list[str], model_name_normalized: str
) -> list[str]:
    """Convert benchmark-scoped image paths to embedding file paths."""

    embedding_paths: list[str] = []
    for scoped_path in scoped_paths:
        if not scoped_path:
            continue
        try:
            benchmark, filename = extract_benchmark_from_path(scoped_path)
        except ValueError:
            logger.debug(
                "Invalid benchmark-scoped path '%s' in memory entry", scoped_path
            )
            continue

        embedding_paths.append(
            f"{benchmark}/embeddings/{model_name_normalized}/{filename}.npy"
        )

    return embedding_paths


def _collect_memory_embedding_paths(
    memory: dict[str, Any], model_name_normalized: str
) -> list[str]:
    """Aggregate all possible embedding paths for a memory entry."""

    candidate_paths: list[str] = []

    # Prefer explicitly stored embedding_paths list
    embedding_paths_field = memory.get("embedding_paths")
    if isinstance(embedding_paths_field, (list, tuple)):
        candidate_paths.extend(
            [
                path.strip()
                for path in embedding_paths_field
                if isinstance(path, str) and path.strip()
            ]
        )

    # Backward compatibility: single embedding_path string (possibly multi-image)
    embedding_path_str = memory.get("embedding_path")
    if isinstance(embedding_path_str, str) and embedding_path_str.strip():
        candidate_paths.extend(expand_embedding_path_value(embedding_path_str))

    # Legacy fallback: derive paths from stored image info
    if not candidate_paths:
        source_paths_field = memory.get("source_image_paths")
        if isinstance(source_paths_field, (list, tuple)):
            scoped_paths = [
                path.strip()
                for path in source_paths_field
                if isinstance(path, str) and path.strip()
            ]
            candidate_paths.extend(
                scoped_paths_to_embedding_paths(scoped_paths, model_name_normalized)
            )
        else:
            benchmark_scoped_path = memory.get("source_image_path")
            if isinstance(benchmark_scoped_path, str) and benchmark_scoped_path.strip():
                normalized_scoped_paths = normalize_semicolon_scoped_paths(
                    benchmark_scoped_path
                )
                candidate_paths.extend(
                    scoped_paths_to_embedding_paths(
                        normalized_scoped_paths, model_name_normalized
                    )
                )

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_paths: list[str] = []
    for path in candidate_paths:
        if not path or path in seen:
            continue
        unique_paths.append(path)
        seen.add(path)

    return unique_paths


def resolve_image_full_path(benchmark_scoped_path: str, root_dir: str | Path) -> Path:
    """Resolve benchmark-scoped path to full filesystem path.

    Args:
        benchmark_scoped_path: Path like "MathVision_MINI/15.jpg"
        root_dir: Dataset root directory (e.g., "/Users/bwh/Documents/Datasets")

    Returns:
        Full path: /Users/bwh/Documents/Datasets/MathVision_MINI/images/15.jpg

    Raises:
        ValueError: If benchmark_scoped_path format is invalid

    Examples:
        >>> resolve_image_full_path("MathVision_MINI/15.jpg", "/root")
        Path('/root/MathVision_MINI/images/15.jpg')
    """
    parts = benchmark_scoped_path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid benchmark-scoped path: {benchmark_scoped_path}. "
            f"Expected format: 'benchmark/filename'"
        )

    benchmark, filename = parts
    return Path(root_dir) / benchmark / "images" / filename


def extract_benchmark_from_path(benchmark_scoped_path: str) -> tuple[str, str]:
    """Extract benchmark and filename from scoped path.

    Args:
        benchmark_scoped_path: Path like "MathVision_MINI/15.jpg"

    Returns:
        Tuple of (benchmark, filename)

    Raises:
        ValueError: If path format is invalid

    Examples:
        >>> extract_benchmark_from_path("MathVision_MINI/15.jpg")
        ('MathVision_MINI', '15.jpg')
    """
    parts = benchmark_scoped_path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid benchmark-scoped path: {benchmark_scoped_path}. "
            f"Expected format: 'benchmark/filename'"
        )
    return parts[0], parts[1]


# ========== Visual Embedding Functions (Batch Precompute) ==========


async def precompute_embeddings_batch(
    images_dir: Path,
    benchmark_dir: Path,
    model: str = "qwen:qwen2.5-vl-embedding",
    api_key: str | None = None,
    batch_size: int = 10,
) -> dict[str, Path]:
    """Batch precompute embeddings for all images using online API and save to model-specific directory.

    Directory structure:
        {benchmark_dir}/embeddings/{model_name}/{image_filename}.npy

    Example:
        /root/Datasets/MathVision_MINI/embeddings/qwen2.5-vl-embedding/15.jpg.npy

    Args:
        images_dir: Directory containing images (e.g., {benchmark_dir}/images/)
        benchmark_dir: Benchmark root directory (e.g., /root/Datasets/MathVision_MINI/)
        model: Embedding model name (provider:model format, e.g., 'qwen:qwen2.5-vl-embedding')
        api_key: API key (defaults to environment variable)
        batch_size: Number of images to process per API call

    Returns:
        Dictionary mapping image filenames to their embedding file paths
    """
    import base64
    import mimetypes

    from tqdm import tqdm

    # Normalize model name for directory naming (remove provider prefix)
    model_name_normalized = model.split(":", 1)[1] if ":" in model else model
    model_name_normalized = model_name_normalized.replace(
        "/", "_"
    )  # Replace slashes for filesystem

    # Create model-specific embeddings directory
    embeddings_dir = benchmark_dir / "embeddings" / model_name_normalized
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_files = [
        f
        for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        logger.warning(f"No image files found in {images_dir}")
        return {}

    benchmark_name = benchmark_dir.name
    logger.info(
        f"Precomputing embeddings for {len(image_files)} images in {benchmark_name}"
    )
    logger.info(f"Using model: {model}")
    logger.info(f"Saving to: {embeddings_dir}")

    embedding_paths = {}
    skipped_count = 0

    # Separate images that need computation from those already computed
    images_to_compute = []
    for image_file in image_files:
        embedding_file = embeddings_dir / f"{image_file.name}.npy"
        if embedding_file.exists():
            embedding_paths[image_file.name] = embedding_file
            skipped_count += 1
        else:
            images_to_compute.append(image_file)

    logger.info(
        f"Found {skipped_count} existing embeddings, need to compute {len(images_to_compute)} new embeddings"
    )

    if not images_to_compute:
        logger.info("All embeddings already cached")
        return embedding_paths

    # Process images in batches with progress bar
    failed_count = 0
    total_batches = (len(images_to_compute) + batch_size - 1) // batch_size

    with tqdm(
        total=len(images_to_compute),
        desc="Computing embeddings",
        unit="image",
        ncols=100,
    ) as pbar:
        for i in range(0, len(images_to_compute), batch_size):
            batch = images_to_compute[i : i + batch_size]
            batch_num = i // batch_size + 1

            # Prepare base64-encoded images for API
            image_urls = []
            valid_files = []
            for img_file in batch:
                try:
                    mime_type, _ = mimetypes.guess_type(str(img_file))
                    if not mime_type or not mime_type.startswith("image/"):
                        mime_type = "image/jpeg"

                    with open(img_file, "rb") as f:
                        image_data = f.read()
                        image_base64 = base64.b64encode(image_data).decode("utf-8")
                        image_url = f"data:{mime_type};base64,{image_base64}"
                        image_urls.append(image_url)
                        valid_files.append(img_file)
                except Exception as e:
                    tqdm.write(f"⚠️  Failed to load image {img_file.name}: {e}")
                    failed_count += 1
                    pbar.update(1)

            if not image_urls:
                continue

            # Call API to get embeddings with rate limiting (max 3 concurrent within batch)
            try:
                embeddings = await compute_multimodal_embeddings(
                    images=image_urls,
                    model=model,
                    api_key=api_key,
                    max_concurrent=3,  # Conservative rate limiting to avoid API overload
                )

                # Save embeddings to .npy files
                for img_file, embedding in zip(valid_files, embeddings):
                    embedding_file = embeddings_dir / f"{img_file.name}.npy"
                    if not embedding:
                        # Empty embedding indicates failure
                        tqdm.write(f"⚠️  Image {img_file.name}: No embedding returned")
                        failed_count += 1
                        pbar.update(1)
                        continue

                    embedding_np = np.array(embedding, dtype=np.float32)
                    save_embedding_to_npy(embedding_np, embedding_file)
                    embedding_paths[img_file.name] = embedding_file
                    pbar.update(1)

                tqdm.write(
                    f"✓ Batch {batch_num}/{total_batches}: {len([e for e in embeddings if e])} embeddings saved"
                )

            except Exception as e:
                tqdm.write(f"✗ Failed to compute embeddings for batch {batch_num}: {e}")
                failed_count += len(valid_files)
                pbar.update(len(valid_files))

    logger.info(
        f"Embedding computation complete: {len(embedding_paths) - skipped_count} new, "
        f"{skipped_count} skipped, {failed_count} failed"
    )
    logger.info(f"Total embeddings available: {len(embedding_paths)}")
    logger.info(f"Embeddings saved to: {embeddings_dir}")

    return embedding_paths


def save_embedding_to_npy(embedding: np.ndarray, output_path: Path) -> None:
    """Save embedding vector to .npy file.

    Args:
        embedding: Embedding vector as numpy array
        output_path: Path to save the .npy file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embedding)


def load_embedding_from_npy(embedding_path: Path) -> np.ndarray | None:
    """Load embedding vector from .npy file.

    Args:
        embedding_path: Path to the .npy file

    Returns:
        Embedding vector as numpy array, or None if file doesn't exist or loading fails
    """
    if not embedding_path.exists():
        return None

    try:
        return np.load(embedding_path)
    except Exception as e:
        logger.error(f"Failed to load embedding from {embedding_path}: {e}")
        return None


async def get_or_compute_image_embedding(
    benchmark_scoped_path: str,
    full_image_path: Path,
    root_dir: str | Path,
    model: str = "qwen:qwen2.5-vl-embedding",
    model_path: str | None = None,
    device: str = "cuda:0",
    api_key: str | None = None,
) -> list[float] | None:
    """Get image embedding from precomputed .npy file in benchmark directory.

    This function ONLY loads precomputed embeddings. It does NOT compute embeddings on-the-fly.
    All embeddings must be precomputed using precompute_embeddings_batch() before calling this function.

    Args:
        benchmark_scoped_path: Benchmark-scoped path (e.g., "MathVision_MINI/15.jpg")
        full_image_path: Full filesystem path to the image file (unused, kept for compatibility)
        root_dir: Benchmark root directory (e.g., "/root/Datasets")
        model: Embedding model name (provider:model format, e.g., 'qwen:qwen2.5-vl-embedding')
        model_path: Unused (kept for compatibility)
        device: Unused (kept for compatibility)
        api_key: Unused (kept for compatibility)

    Returns:
        Embedding vector as list, or None if embedding file not found
    """
    # Extract benchmark and filename from scoped path
    try:
        benchmark, filename = extract_benchmark_from_path(benchmark_scoped_path)
    except ValueError:
        logger.error(f"Invalid benchmark-scoped path: {benchmark_scoped_path}")
        return None

    # Normalize model name for directory lookup
    model_name_normalized = model.split(":", 1)[1] if ":" in model else model
    model_name_normalized = model_name_normalized.replace("/", "_")

    # Construct path to model-specific embeddings directory
    benchmark_dir = Path(root_dir) / benchmark
    embeddings_dir = benchmark_dir / "embeddings" / model_name_normalized
    npy_file = embeddings_dir / f"{filename}.npy"

    if npy_file.exists():
        embedding_np = load_embedding_from_npy(npy_file)
        if embedding_np is not None:
            return embedding_np.tolist()

    # Embedding not found - this should not happen if precomputation was done correctly
    logger.error(f"Embedding not found: {npy_file}")
    logger.error(f"Please run precompute_embeddings_batch() with model '{model}' first")
    return None


async def compute_multimodal_embeddings(
    images: list[str],
    model: str = "qwen:tongyi-embedding-vision-plus",
    api_key: str | None = None,
    max_concurrent: int = 5,
) -> list[list[float]]:
    """Compute embeddings for images using multimodal embedding API with multi-provider support.

    Args:
        images: List of image URLs or base64-encoded images
        model: Model name in 'provider:model' format (e.g., 'qwen:tongyi-embedding-vision-plus')
               For backward compatibility, also accepts plain model name (defaults to qwen provider)
        api_key: API key (provider-specific, defaults to environment variable)
        max_concurrent: Maximum number of concurrent API requests (default: 5)

    Returns:
        List of embedding vectors (one per image)

    Raises:
        ValueError: If API key is not provided or unknown provider
        httpx.HTTPError: If API request fails
    """
    # If no images, return empty list
    if not images:
        return []

    # Parse provider and model
    if ":" in model:
        provider, model_name = model.split(":", maxsplit=1)
    else:
        # Backward compatibility: default to qwen provider
        provider, model_name = "qwen", model

    provider_lower = provider.lower()

    # Route to provider-specific implementation
    if provider_lower == "qwen":
        return await _compute_qwen_multimodal_embeddings(
            images, model_name, api_key, max_concurrent
        )
    elif provider_lower == "openai":
        # OpenAI has no native multimodal embedding API
        raise NotImplementedError("OpenAI multimodal embeddings not available")
    elif provider_lower == "qianfan":
        # Baidu Qianfan focuses on generation, not embeddings
        raise NotImplementedError("Baidu Qianfan multimodal embeddings not available")
    else:
        raise ValueError(f"Unknown multimodal embedding provider: {provider}")


async def _compute_single_image_embedding(
    client: httpx.AsyncClient,
    image: str,
    model: str,
    api_key: str,
    url: str,
    idx: int,
    semaphore: asyncio.Semaphore,
) -> tuple[int, list[float]]:
    """Compute embedding for a single image with rate limiting and retry logic.

    This function includes built-in retry logic with exponential backoff for transient errors.
    It handles specific HTTP status codes (429, 500, 503) and timeout errors with up to 3 retries.

    Args:
        client: HTTP client
        image: Base64-encoded image
        model: Model name
        api_key: API key
        url: API endpoint
        idx: Image index (for ordering)
        semaphore: Semaphore for rate limiting

    Returns:
        Tuple of (index, embedding_vector) to maintain order
    """
    # Use semaphore to limit concurrent requests
    async with semaphore:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # DashScope API format: input is an object with 'contents' array
        payload = {
            "model": model,
            "input": {"contents": [{"image": image}]},
        }

        # Retry logic for transient errors
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    url, headers=headers, json=payload, timeout=60.0
                )

                if response.status_code == 200:
                    result = response.json()
                    output = result.get("output", {})
                    embeddings_data = output.get("embeddings", [])

                    if embeddings_data and len(embeddings_data) > 0:
                        embedding = embeddings_data[0].get("embedding", [])
                        return (idx, embedding)
                    else:
                        logger.error(f"No embedding returned for image {idx}")
                        return (idx, [])

                # Handle retryable errors (500, 429, 503)
                elif response.status_code in (429, 500, 503):
                    wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Image {idx}: API error {response.status_code}, "
                            f"retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(
                            f"Image {idx}: API error {response.status_code} after {max_retries} retries"
                        )
                        logger.error(f"Response: {response.text[:500]}")
                        return (idx, [])
                else:
                    # Non-retryable error
                    logger.error(f"Image {idx}: API error {response.status_code}")
                    logger.error(f"Response: {response.text[:500]}")
                    return (idx, [])

            except httpx.TimeoutException:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Image {idx}: Timeout, retrying (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(2**attempt)
                    continue
                else:
                    logger.error(f"Image {idx}: Timeout after {max_retries} retries")
                    return (idx, [])
            except Exception as e:
                logger.error(f"Image {idx}: Unexpected error: {e}")
                return (idx, [])

        return (idx, [])


async def _compute_qwen_multimodal_embeddings(
    images: list[str],
    model: str,
    api_key: str | None = None,
    max_concurrent: int = 5,
) -> list[list[float]]:
    """DashScope-specific multimodal embedding implementation with rate-limited parallel processing.

    Note: DashScope API does not support batch processing, but we use rate-limited concurrent requests
    to speed up processing while avoiding API overload.

    Args:
        images: List of image URLs or base64-encoded images
        model: DashScope model name (e.g., 'qwen2.5-vl-embedding' or 'qwen:qwen2.5-vl-embedding')
        api_key: DashScope API key (defaults to DASHSCOPE_API_KEY env var)
        max_concurrent: Maximum number of concurrent requests (default: 5)

    Returns:
        List of embedding vectors (one per image)
    """
    if not api_key:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "DASHSCOPE_API_KEY environment variable must be set "
            "to use DashScope Multimodal Embedding API"
        )

    # Strip provider prefix if present (e.g., 'qwen:qwen2.5-vl-embedding' -> 'qwen2.5-vl-embedding')
    model_name = model.split(":", 1)[1] if ":" in model else model

    url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    # Process images concurrently with rate limiting
    async with httpx.AsyncClient() as client:
        tasks = [
            _compute_single_image_embedding(
                client, img, model_name, api_key, url, idx, semaphore
            )
            for idx, img in enumerate(images)
        ]

        # Execute all requests with rate limiting
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Sort by index to maintain original order and extract embeddings
    embeddings = [None] * len(images)
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Exception in concurrent embedding computation: {result}")
            continue
        idx, embedding = result
        embeddings[idx] = embedding if embedding else []

    # Fill any None values with empty lists (shouldn't happen but for safety)
    embeddings = [emb if emb is not None else [] for emb in embeddings]

    return embeddings


async def retrieve_visual_memories_by_similarity(
    query_image_path: str,
    memories: list[dict[str, Any]],
    root_dir: str | Path,
    model: str = "qwen:qwen2.5-vl-embedding",
    model_path: str | None = None,
    device: str = "cuda:0",
    top_n: int = 5,
    api_key: str | None = None,
) -> list[tuple[dict[str, Any], float]]:
    """Retrieve visual memories by loading precomputed embeddings from model-specific cache.

    This function ONLY loads precomputed .npy embeddings. It does NOT compute embeddings on-the-fly.

    Directory structure:
        {benchmark}/embeddings/{model_name}/{filename}.npy

    Args:
        query_image_path: Query image benchmark-scoped path (e.g., "MathVision_MINI/981.jpg")
        memories: List of visual memory dictionaries with 'embedding_path' field
        root_dir: Benchmark root directory (e.g., "/root/Datasets")
        model: Embedding model name (provider:model format, e.g., 'qwen:qwen2.5-vl-embedding')
        model_path: Unused (for compatibility)
        device: Unused (for compatibility)
        top_n: Maximum number of results to return
        api_key: Unused (for compatibility)

    Returns:
        List of (memory, similarity_score) tuples sorted by similarity (highest first)
    """
    if not memories:
        return []

    # Normalize model name for directory lookup
    model_name_normalized = model.split(":", 1)[1] if ":" in model else model
    model_name_normalized = model_name_normalized.replace("/", "_")

    # Extract benchmark and filename from query image path
    try:
        query_benchmark, query_filename = extract_benchmark_from_path(query_image_path)
    except ValueError:
        logger.error(f"Invalid query image path format: {query_image_path}")
        return []

    # Load query image embedding from model-specific directory
    query_embeddings_dir = (
        Path(root_dir) / query_benchmark / "embeddings" / model_name_normalized
    )
    query_npy_file = query_embeddings_dir / f"{query_filename}.npy"

    if not query_npy_file.exists():
        logger.error(f"Query image embedding not found: {query_npy_file}")
        logger.error(
            f"Please run precompute_embeddings_batch() with model '{model}' first"
        )
        return []

    query_embedding_np = load_embedding_from_npy(query_npy_file)
    if query_embedding_np is None:
        logger.error(f"Failed to load query embedding from {query_npy_file}")
        return []

    query_embedding = query_embedding_np.tolist()

    # Compute cosine similarity with each stored memory
    import math

    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    # Calculate similarities using embedding_path candidates from each memory
    similarities = []
    for memory in memories:
        candidate_paths = _collect_memory_embedding_paths(memory, model_name_normalized)
        if not candidate_paths:
            logger.debug(
                "Memory %s has no embedding paths; skipping", memory.get("memory_id")
            )
            continue

        best_similarity: float | None = None
        for candidate in candidate_paths:
            memory_npy_file = Path(candidate)
            if not memory_npy_file.is_absolute():
                memory_npy_file = Path(root_dir) / memory_npy_file

            if not memory_npy_file.exists():
                logger.warning(
                    f"Memory embedding not found: {memory_npy_file}, skipping"
                )
                continue

            memory_embedding_np = load_embedding_from_npy(memory_npy_file)
            if memory_embedding_np is None:
                logger.warning(
                    f"Failed to load memory embedding from {memory_npy_file}, skipping"
                )
                continue

            stored_embedding = memory_embedding_np.tolist()
            similarity = cosine_similarity(query_embedding, stored_embedding)

            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity

        if best_similarity is None:
            continue

        similarities.append((memory, best_similarity))

    # Sort by similarity (highest first) and return top_n
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def parse_visual_error_analysis(analysis_text: str) -> tuple[bool, str, str]:
    """Parse visual error analysis output to extract JSON fields.

    Expected format (JSON):
        {
            "is_visual_error": true/false,
            "analysis_summary": "summary text",
            "guideline": "guideline text or null"
        }

    Args:
        analysis_text: Raw visual error analysis output from the model (may contain JSON)

    Returns:
        Tuple of (is_visual_error, analysis_summary, guideline)
    """
    import re

    # Try to extract JSON from the response
    # Look for JSON block (may be wrapped in markdown code block)
    json_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", analysis_text, re.DOTALL | re.IGNORECASE
    )
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON object
        json_match = re.search(r"\{.*?\}", analysis_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # No JSON found, return defaults
            return False, "", ""

    # Parse JSON
    try:
        data = json.loads(json_str)
        is_visual_error = data.get("is_visual_error", False)
        analysis_summary = data.get("analysis_summary", "")
        guideline = data.get("guideline") or ""  # Handle null as empty string

        return is_visual_error, analysis_summary, guideline
    except json.JSONDecodeError:
        # If JSON parsing fails, return defaults
        return False, "", ""


# ========== Memory Merging Functions ==========

MEMORY_MERGE_PROMPT = """Objective:
    You are tasked with merging two similar memory guidelines into a single, more comprehensive guideline.

Context:
    - Existing Memory Guideline: {existing_guideline}
    - Existing Memory Context: Subject: {existing_subject}, Key Concepts: {existing_concepts}
    - New Memory Guideline: {new_guideline}
    - New Memory Context: Subject: {new_subject}, Key Concepts: {new_concepts}

Instructions:
    1. Analyze both guidelines to identify their core messages and overlapping concepts.
    2. Create a unified guideline that:
       - Preserves the essential information from both guidelines
       - Eliminates redundancy
       - Maintains clarity and actionability
       - Is generalizable to both contexts
    3. The merged guideline should be concise (1-2 sentences) and follow the same quality standards as the individual guidelines.

Output Format:
    Provide only the merged guideline text, without any additional explanation or formatting.
"""


async def merge_memories(
    existing_memory: dict[str, Any],
    new_memory: dict[str, Any],
    model: str = "qwen:qwen3-next-80b-a3b-instruct",
    temperature: float = 0.3,
    max_tokens: int = 256,
    api_key: str | None = None,
) -> str:
    """Merge two similar memories into a single comprehensive guideline with retry on transient failures.

    Args:
        existing_memory: Existing memory dictionary
        new_memory: New memory dictionary to merge with
        model: Model to use for merging (default: qwen:qwen3-next-80b-a3b-instruct)
        temperature: Temperature for model inference
        max_tokens: Maximum tokens for output
        api_key: API key (defaults to environment variable)

    Returns:
        Merged guideline text
    """
    from langchain_core.messages import AIMessage, HumanMessage

    from common.utils import get_message_text, load_chat_model

    # Prepare prompt
    existing_guideline = existing_memory.get("guideline", "")
    existing_subject = existing_memory.get("subject", "Unknown")
    existing_concepts = ", ".join(existing_memory.get("key_concepts", []))

    new_guideline = new_memory.get("guideline", "")
    new_subject = new_memory.get("subject", "Unknown")
    new_concepts = ", ".join(new_memory.get("key_concepts", []))

    prompt = MEMORY_MERGE_PROMPT.format(
        existing_guideline=existing_guideline,
        existing_subject=existing_subject,
        existing_concepts=existing_concepts,
        new_guideline=new_guideline,
        new_subject=new_subject,
        new_concepts=new_concepts,
    )

    # Load model
    chat_model = load_chat_model(model)
    if temperature is not None:
        chat_model = chat_model.bind(temperature=temperature)  # type: ignore
    if max_tokens is not None:
        chat_model = chat_model.bind(max_tokens=max_tokens)  # type: ignore

    # Configure model with retry and invoke
    model_with_retry = chat_model.with_retry(stop_after_attempt=3)
    response = await model_with_retry.ainvoke([HumanMessage(content=prompt)])
    merged_guideline = (
        get_message_text(response) if isinstance(response, AIMessage) else ""
    )

    return merged_guideline.strip()
