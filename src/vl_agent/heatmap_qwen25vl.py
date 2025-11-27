"""Qwen2.5-VL Cross-Attention based heatmap generation.

This module implements attention-based visual explanation for Qwen2.5-VL models.
Unlike gradient-based methods (GradCAM, Grad-Eclip), this approach directly extracts
the model's internal cross-attention weights to identify which image regions the model
focuses on when processing text queries.

Key advantages over CLIP-based methods:
1. Supports long text inputs (no 77 token limit)
2. Better understanding of abstract/reasoning language
3. Direct cross-attention between text and image tokens
4. Training-free (uses model's internal attention)

Implementation based on:
- "MLLMs Know Where to Look" (arXiv 2502.17422, Feb 2025)
- Code: https://github.com/saccharomycetes/mllms_know

References:
- Qwen2.5-VL: https://github.com/QwenLM/Qwen2.5-VL
- Transformers: https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

# Get logger - will inherit configuration from parent qwen_vl_eval logger
logger = logging.getLogger("qwen_vl_eval.heatmap")


# Global model cache for efficient model reuse across multiple heatmap generations
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}  # key: f"{model_name}_{device}"

# Locks to ensure only one load per (model, device) even under concurrency
_MODEL_CACHE_LOCK = Lock()
_MODEL_SINGLEFLIGHT_LOCKS: dict[str, Lock] = {}


@dataclass
class DevicePool:
    """Simple device pool that hands out device IDs with bounded concurrency."""

    devices: tuple[str, ...]
    per_device_limit: int
    queue: asyncio.Queue[str]

    @classmethod
    def create(cls, devices: Sequence[str], per_device_limit: int) -> "DevicePool":
        per_device_limit = max(1, per_device_limit)
        queue: asyncio.Queue[str] = asyncio.Queue()
        for device in devices:
            for _ in range(per_device_limit):
                queue.put_nowait(device)
        return cls(tuple(devices), per_device_limit, queue)

    async def acquire(self) -> str:
        return await self.queue.get()

    def release(self, device: str) -> None:
        self.queue.put_nowait(device)


_DEVICE_POOL_REGISTRY: dict[tuple[tuple[str, ...], int], DevicePool] = {}
_DEVICE_POOL_REGISTRY_LOCK = Lock()


def _get_or_create_model_lock(cache_key: str) -> Lock:
    """Return a dedicated lock for the given cache key."""

    with _MODEL_CACHE_LOCK:
        if cache_key not in _MODEL_SINGLEFLIGHT_LOCKS:
            _MODEL_SINGLEFLIGHT_LOCKS[cache_key] = Lock()
        return _MODEL_SINGLEFLIGHT_LOCKS[cache_key]


def encode_base64_image(image: Image.Image) -> str:
    """Encode PIL image to base64 string.

    Args:
        image: PIL Image object

    Returns:
        Base64-encoded string (without data URI prefix)
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def prepare_qwen25vl_input(messages: list[dict], processor: Any) -> dict:
    """Prepare input for Qwen2.5-VL model.

    Args:
        messages: Chat messages with vision content
        processor: Qwen2.5-VL processor

    Returns:
        Processed inputs ready for model forward pass
    """
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        raise ImportError(
            "qwen_vl_utils is required for Qwen2.5-VL heatmap generation. "
            "Install with: pip install qwen-vl-utils"
        )

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return inputs


def get_or_create_qwen25vl_model(
    model_name: str,
    device: str,
) -> tuple[Any, Any]:
    """Get cached Qwen2.5-VL model or create new one if not exists.

    This function implements a simple cache to avoid loading the same model multiple times.
    The cache key is f"{model_name}_{device}" to support multiple devices.

    Args:
        model_name: Qwen2.5-VL model name (e.g., "Qwen/Qwen2.5-VL-2B-Instruct")
        device: Device to use (e.g., "cuda:0", "cpu")

    Returns:
        Tuple of (model, processor)
    """
    cache_key = f"{model_name}_{device}"

    # Fast path without locking
    if cache_key in _MODEL_CACHE:
        logger.debug(f"Reusing cached Qwen2.5-VL model: {model_name} on {device}")
        return _MODEL_CACHE[cache_key]

    model_lock = _get_or_create_model_lock(cache_key)

    with model_lock:
        # Double-check cache after acquiring the lock
        if cache_key in _MODEL_CACHE:
            logger.debug(f"Reusing cached Qwen2.5-VL model: {model_name} on {device}")
            return _MODEL_CACHE[cache_key]

        logger.info(f"Loading Qwen2.5-VL model: {model_name} on device: {device}")

        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError:
            raise ImportError(
                "transformers>=4.50.0 is required for Qwen2.5-VL heatmap generation. "
                "Install with: pip install -U transformers>=4.50.0"
            )

        # Validate and use the specified device
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning(
                    f"CUDA device '{device}' requested but CUDA is not available. Falling back to CPU."
                )
                device = "cpu"
            else:
                # Validate CUDA device index
                try:
                    device_idx = int(device.split(":")[1]) if ":" in device else 0
                    if device_idx >= torch.cuda.device_count():
                        logger.warning(
                            f"CUDA device index {device_idx} out of range "
                            f"(available: 0-{torch.cuda.device_count() - 1}). Falling back to cuda:0."
                        )
                        device = "cuda:0"
                except (ValueError, IndexError):
                    logger.warning(f"Invalid device format '{device}'. Using cuda:0.")
                    device = "cuda:0"
        elif device == "mps":
            if not torch.backends.mps.is_available():
                logger.warning(
                    "MPS device requested but not available. Falling back to CPU."
                )
                device = "cpu"

        # Load model with eager attention (required for attention extraction)
        load_kwargs: dict[str, Any] = {
            "dtype": torch.bfloat16,
            "attn_implementation": "eager",  # Required to access attention weights
        }

        move_after_load: str | None = None
        supports_device_map = False

        try:
            from transformers.utils import is_accelerate_available

            supports_device_map = bool(is_accelerate_available())
        except Exception:
            supports_device_map = False

        if device != "cpu" and supports_device_map:
            load_kwargs["device_map"] = device
        elif device != "cpu":
            move_after_load = device
            logger.warning(
                "Accelerate is not available; loading model on CPU and moving to %s. "
                "Install accelerate>=1.0 for optimal multi-device support.",
                device,
            )

        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                **load_kwargs,
            )
            if move_after_load:
                model = model.to(move_after_load, dtype=torch.bfloat16)
            logger.info("Successfully loaded Qwen2.5-VL model")
        except Exception as e:
            logger.error(f"Failed to load Qwen2.5-VL model: {e}")
            raise

        model.eval()

        processor = AutoProcessor.from_pretrained(model_name)

        # Cache the model
        _MODEL_CACHE[cache_key] = (model, processor)
        logger.info(f"Cached Qwen2.5-VL model: {model_name} on {device}")

        return model, processor

    # This line is only reached if another thread populated the cache while we waited
    return _MODEL_CACHE[cache_key]


def extract_relative_attention_qwen25vl(
    image: Image.Image,
    prompt: str,
    model: Any,
    processor: Any,
    general_prompt: str = "Describe this image.",
    attention_layer: int = 22,
) -> np.ndarray:
    """Extract relative attention map from Qwen2.5-VL model.

    This implements the "relative attention" method from "MLLMs Know Where to Look".
    The relative attention is computed as:
        A_rel(x, q) = A_si(x, q) / A_si(x, general_q)

    This normalizes the attention by a baseline prompt to filter semantic noise.

    Args:
        image: PIL Image object
        prompt: Text query (can be long, e.g., visual memory guideline)
        model: Qwen2.5-VL model
        processor: Qwen2.5-VL processor
        general_prompt: Baseline prompt for normalization
        attention_layer: Which layer to extract attention from (default: 22)

    Returns:
        Attention map as numpy array with shape inferred from vision tokens
    """
    # Encode image to base64
    image_str = encode_base64_image(image)

    # Prepare messages for both prompts
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{image_str}"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    general_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image;base64,{image_str}"},
                {"type": "text", "text": general_prompt},
            ],
        }
    ]

    # Prepare inputs
    inputs = prepare_qwen25vl_input(messages, processor).to(
        model.device, torch.bfloat16
    )
    general_inputs = prepare_qwen25vl_input(general_messages, processor).to(
        model.device, torch.bfloat16
    )

    # Get attention shape from image grid
    # image_grid_thw: [num_images, temporal, height, width]
    # For images, temporal=1, and we need to divide spatial dims by 2 due to pooling
    att_shape = (inputs["image_grid_thw"][0, 1:] / 2).cpu().numpy().astype(int).tolist()

    # Find vision token positions
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        "<|vision_start|>"
    )
    vision_end_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")

    input_ids_list = inputs["input_ids"].tolist()[0]
    pos = input_ids_list.index(vision_start_token_id) + 1
    pos_end = input_ids_list.index(vision_end_token_id)

    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        general_outputs = model(**general_inputs, output_attentions=True)

    # Extract attention from specified layer
    # attentions[layer]: [batch, heads, seq_len, seq_len]
    # We want the last token's attention to vision tokens
    att = (
        outputs["attentions"][attention_layer][0, :, -1, pos:pos_end]
        .mean(dim=0)  # Average across heads
        .to(torch.float32)
        .detach()
        .cpu()
        .numpy()
    )

    general_att = (
        general_outputs["attentions"][attention_layer][0, :, -1, pos:pos_end]
        .mean(dim=0)
        .to(torch.float32)
        .detach()
        .cpu()
        .numpy()
    )

    # Compute relative attention
    att_map = att / (general_att + 1e-8)  # Avoid division by zero

    # Reshape to spatial dimensions
    att_map = att_map.reshape(att_shape)

    return att_map


def cleanup_qwen25vl_models() -> None:
    """Clean up all cached Qwen2.5-VL models and free GPU memory.

    This should be called at the end of evaluation to properly release resources.
    """
    global _MODEL_CACHE

    if not _MODEL_CACHE:
        return

    logger.info(f"Cleaning up {len(_MODEL_CACHE)} cached Qwen2.5-VL model(s)")

    for cache_key, (model, _) in _MODEL_CACHE.items():
        try:
            # Move model to CPU and delete
            model.cpu()
            del model
            logger.debug(f"Cleaned up model: {cache_key}")
        except Exception as e:
            logger.warning(f"Error cleaning up model {cache_key}: {e}")

    # Clear cache
    _MODEL_CACHE.clear()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA cache after cleanup")


async def generate_qwen25vl_attention_heatmap(
    image_url: str,
    text: str,
    model_name: str = "Qwen/Qwen2.5-VL-2B-Instruct",
    general_prompt: str = "Describe this image.",
    attention_layer: int = 22,
    output_dir: str | None = None,
    example_id: str | None = None,
    benchmark: str | None = None,
    debug: bool = False,
    device: str | Sequence[str] | None = "cuda:0",
    per_device_max_parallel: int = 1,
) -> str:
    """Generate heatmap using Qwen2.5-VL cross-attention and overlay on original image.

    This function uses relative attention extraction to visualize which image regions
    the model focuses on when processing the given text query. It supports long text
    inputs (e.g., complete visual memory guidelines) unlike CLIP-based methods.

    Args:
        image_url: Base64-encoded image URL (data:image/...)
        text: Text query for attention visualization (supports long text!)
        model_name: Qwen2.5-VL model name (2B or 7B recommended)
        general_prompt: Baseline prompt for relative attention normalization
        attention_layer: Which layer to extract attention from (default: 22)
        output_dir: Optional output directory for saving debug images
        example_id: Optional example ID for debug file naming
        benchmark: Optional benchmark name for file naming (e.g., 'MathVista_MINI')
        debug: Whether to save heatmap images for debugging
        device: Device (or devices) to use for computation. Accepts a single string,
            a comma-separated string, a sequence of device strings, or None to auto-select.
        per_device_max_parallel: Maximum concurrent heatmap tasks allowed per device.

    Returns:
        Base64-encoded image URL with heatmap overlay
    """
    # Initialize variables for cleanup
    attention_map = None
    acquired_device: str | None = None

    normalized_devices = _normalize_device_spec(device)
    device_pool = _get_device_pool(normalized_devices, per_device_max_parallel)

    try:
        # Parse base64 image
        if image_url.startswith("data:image/"):
            header, encoded = image_url.split(",", 1)
            image_data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            raise ValueError("Only base64-encoded images are supported")

        # Acquire device slot (round-robin across configured GPUs)
        acquired_device = await device_pool.acquire()
        logger.debug(
            "Using device '%s' for Qwen2.5-VL attention heatmap (pool=%s, per_device_limit=%d)",
            acquired_device,
            ",".join(normalized_devices),
            device_pool.per_device_limit,
        )

        model, processor = get_or_create_qwen25vl_model(model_name, acquired_device)

        # Extract relative attention map (GPU-heavy)
        attention_map = extract_relative_attention_qwen25vl(
            image=image,
            prompt=text,
            model=model,
            processor=processor,
            general_prompt=general_prompt,
            attention_layer=attention_layer,
        )

        logger.debug(
            f"Qwen2.5-VL attention map stats - min: {attention_map.min():.4f}, "
            f"max: {attention_map.max():.4f}, mean: {attention_map.mean():.4f}, "
            f"shape: {attention_map.shape}"
        )

        # Normalize attention map to [0, 1]
        if attention_map.max() > attention_map.min():
            attention_map = (attention_map - attention_map.min()) / (
                attention_map.max() - attention_map.min()
            )
        else:
            attention_map = np.zeros_like(attention_map)

        # Resize heatmap to original image size
        original_size = image.size  # (width, height)
        heatmap_pil = Image.fromarray((attention_map * 255).astype(np.uint8))
        heatmap_resized = heatmap_pil.resize(original_size, Image.Resampling.BILINEAR)

        # Convert to numpy for processing
        heatmap_normalized_resized = (
            np.array(heatmap_resized).astype(np.float32) / 255.0
        )
        original_array = np.array(image).astype(np.float32) / 255.0

        # Apply colormap (jet: blue=low, red=high)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colormap = plt.cm.jet(heatmap_normalized_resized)[:, :, :3]  # RGBA -> RGB

        # Blend heatmap with original image
        alpha = 0.5  # Transparency factor
        heatmap_mask = heatmap_normalized_resized[:, :, np.newaxis]
        blended_array = (
            original_array * (1 - alpha * heatmap_mask)
            + colormap * alpha * heatmap_mask
        )
        blended_array = np.clip(blended_array, 0, 1)

        # Convert back to PIL Image
        blended_image = Image.fromarray((blended_array * 255).astype(np.uint8))

        # Save debug images if requested
        if debug and output_dir:
            # Create qwen25vl-attention subdirectory
            output_path = Path(output_dir) / "qwen25vl-attention"
            output_path.mkdir(parents=True, exist_ok=True)

            # Generate filename: benchmark_case_id.png
            if benchmark and example_id:
                filename = f"{benchmark}_{example_id}.png"
            elif example_id:
                filename = f"{example_id}.png"
            else:
                filename = "qwen25vl_attention_overlaid.png"

            # Save only the overlaid image
            overlaid_file = output_path / filename
            blended_image.save(overlaid_file)
            logger.info(f"Saved Qwen2.5-VL attention heatmap to {overlaid_file}")

        # Convert blended image to base64
        buffer = io.BytesIO()
        blended_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"data:image/png;base64,{image_base64}"

    except Exception as e:
        logger.error(f"Error generating Qwen2.5-VL attention heatmap: {e}")
        raise

    finally:
        # Clean up temporary resources (but keep model cached for reuse)
        if attention_map is not None:
            del attention_map

        # Clear CUDA cache to prevent memory fragmentation
        # This is safe to call even if CUDA is not available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return device slot to the pool
        if acquired_device is not None:
            device_pool.release(acquired_device)

        # Note: We do NOT clean up the model here - it remains in cache for reuse
        # The model will be cleaned up at the end of evaluation via cleanup_qwen25vl_models()


def _default_device_list() -> list[str]:
    """Return a sensible default device list based on runtime availability."""

    if torch.cuda.is_available():
        return ["cuda:0"]
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return ["mps"]
    return ["cpu"]


def _normalize_device_spec(device: str | Sequence[str] | None) -> list[str]:
    """Normalize device specification into a unique ordered list."""

    if device is None:
        return _default_device_list()

    raw_items: list[str] = []
    if isinstance(device, str):
        raw_items = [item.strip() for item in device.split(",") if item.strip()]
    else:
        for item in device:
            if isinstance(item, str):
                normalized = item.strip()
            else:
                normalized = str(item).strip()
            if normalized:
                raw_items.append(normalized)

    if not raw_items:
        return _default_device_list()

    devices: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if item.lower() == "auto":
            for fallback in _default_device_list():
                if fallback not in seen:
                    devices.append(fallback)
                    seen.add(fallback)
            continue

        if item not in seen:
            devices.append(item)
            seen.add(item)

    return devices or _default_device_list()


def _get_device_pool(devices: Sequence[str], per_device_limit: int) -> DevicePool:
    """Return (or create) a shared device pool for the provided devices."""

    if not devices:
        raise ValueError(
            "At least one device must be configured for Qwen2.5-VL heatmaps"
        )

    per_device_limit = max(1, per_device_limit)
    key = (tuple(devices), per_device_limit)

    with _DEVICE_POOL_REGISTRY_LOCK:
        pool = _DEVICE_POOL_REGISTRY.get(key)
        if pool is None:
            pool = DevicePool.create(devices, per_device_limit)
            _DEVICE_POOL_REGISTRY[key] = pool

    return pool
