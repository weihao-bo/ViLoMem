"""Bridge LangGraph verification with VLMEvalKit evaluators.

Benchmark-specific verifiers (ANLS, OCRBench, MathGlance) live in
`common.verifiers.*`; this module dispatches to them and falls back to
VLMEvalKit's generic post_check/auxeval pipeline for every other dataset.
"""

from __future__ import annotations

import ast
import importlib
import logging
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

from .utils import normalize_region
from .verifiers import anls_verify, mathglance_verify, ocrbench_verify

logger = logging.getLogger(__name__)


# ========== API Config for VLMEvalKit Wrapper ==========


@dataclass(frozen=True)
class _ModelAPIConfig:
    """Resolved API configuration for a provider:model string."""

    provider: str
    model: str
    api_key: str
    api_base: str | None


def _resolve_dashscope_base_url() -> str:
    env_base = os.getenv("DASHSCOPE_API_BASE")
    if env_base:
        return env_base
    region = normalize_region(os.getenv("REGION") or "")
    if region == "prc":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def get_api_config(model_name: str) -> _ModelAPIConfig:
    """Resolve API credentials for a provider:model string (for VLMEvalKit)."""
    if ":" in model_name:
        provider, actual_model = model_name.split(":", 1)
    else:
        provider, actual_model = "openai", model_name

    provider_lower = provider.lower()

    if provider_lower in {"openai", "openai2"}:
        env_prefix = "OPENAI" if provider_lower == "openai" else "OPENAI2"
        api_key = os.getenv(f"{env_prefix}_API_KEY")
        if not api_key:
            raise ValueError(f"API key not found. Set {env_prefix}_API_KEY.")
        api_base = os.getenv(f"{env_prefix}_API_BASE")
        return _ModelAPIConfig(provider_lower, actual_model, api_key, api_base)

    if provider_lower == "local":
        api_key = os.getenv("LOCAL_VLLM_API_KEY", "EMPTY")
        api_base = os.getenv("LOCAL_MODEL_BASE_URL") or os.getenv("LOCAL_VLLM_API_BASE")
        return _ModelAPIConfig(provider_lower, actual_model, api_key, api_base)

    if provider_lower in {"qwen", "dashscope"}:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("API key not found. Set DASHSCOPE_API_KEY.")
        api_base = _resolve_dashscope_base_url()
        return _ModelAPIConfig("qwen", actual_model, api_key, api_base)

    raise ValueError(f"Unknown provider: {provider}")

try:  # pragma: no cover - optional dependency
    from vlmeval.api import OpenAIWrapper

    _VLMEVAL_AVAILABLE = True
except Exception:  # pragma: no cover
    OpenAIWrapper = None  # type: ignore
    _VLMEVAL_AVAILABLE = False


POST_CHECK_FN = "post_check"


@dataclass
class DatasetHandler:
    name: str
    module_name: str
    post_check: Callable[[dict[str, Any], bool], Any]
    auxeval: Callable[[Any, dict[str, Any]], dict[str, Any]] | None = None


_MODULE_HINTS = {
    "mathvista": "vlmeval.dataset.utils.mathvista",
    "mathvision": "vlmeval.dataset.utils.mathv",
    "mathvision_mini": "vlmeval.dataset.utils.mathv",
    "mmvet": "vlmeval.dataset.utils.mmvet",
    "llavabench": "vlmeval.dataset.utils.llavabench",
    "mmbench_video": "vlmeval.dataset.utils.mmbench_video",
    "mmdu": "vlmeval.dataset.utils.mmdu",
    "mmlongbench": "vlmeval.dataset.utils.mmdu",
    "multiple_choice": "vlmeval.dataset.utils.multiple_choice",
    "ocrbench": "vlmeval.dataset.utils.ocrbench",
    "videomme": "vlmeval.dataset.utils.videomme",
    "yorn": "vlmeval.dataset.utils.yorn",
}


_HANDLER_CACHE: dict[str, DatasetHandler] = {}
_WRAPPER_CACHE: dict[Tuple[str, str | None, int], OpenAIWrapper] = {}
_LOCK = threading.Lock()


@contextmanager
def suppress_vlmeval_logging() -> None:
    """Temporarily silence VLMEvalKit loggers that leak API keys."""

    logger_names = ["GPT4V API", "VLMEvalKit", "VLMEvalKit Tool"]
    prev_levels: dict[str, int] = {}
    for name in logger_names:
        logger = logging.getLogger(name)
        prev_levels[name] = logger.level
        logger.setLevel(max(logging.WARNING, logger.level or logging.WARNING))
    try:
        yield
    finally:
        for name, level in prev_levels.items():
            logging.getLogger(name).setLevel(level)


def _normalize_benchmark(name: str | None) -> str | None:
    if not name:
        return None
    base = name.lower()
    base = re.sub(r"[- ]", "_", base)
    base = re.sub(r"(_mini|_test.*|_val.*|_dev.*|_en|_cn)$", "", base)
    return base


def _load_handler(benchmark: str | None) -> DatasetHandler | None:
    normalized = _normalize_benchmark(benchmark)
    if not normalized:
        return None

    module_name = _MODULE_HINTS.get(normalized, f"vlmeval.dataset.utils.{normalized}")

    with _LOCK:
        handler = _HANDLER_CACHE.get(module_name)
        if handler:
            return handler

        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return None

        post_check = getattr(module, POST_CHECK_FN, None)
        if not callable(post_check):
            return None

        auxeval = None
        for attr in dir(module):
            if attr.lower().endswith("auxeval"):
                candidate = getattr(module, attr)
                if callable(candidate):
                    auxeval = candidate
                    break

        handler = DatasetHandler(normalized, module_name, post_check, auxeval)
        _HANDLER_CACHE[module_name] = handler
        return handler


def _get_wrapper(model_name: str, max_tokens: int) -> OpenAIWrapper:
    cfg = get_api_config(model_name)
    key = (cfg.model, cfg.api_base, max_tokens)
    with _LOCK:
        wrapper = _WRAPPER_CACHE.get(key)
        if wrapper is None:
            if OpenAIWrapper is None:
                raise RuntimeError("VLMEvalKit is not installed")
            with suppress_vlmeval_logging():
                wrapper = OpenAIWrapper(
                    model=cfg.model,
                    key=cfg.api_key,
                    api_base=cfg.api_base,
                    max_tokens=max_tokens,
                    retry=5,
                    wait=3,
                    temperature=0,
                )
            _WRAPPER_CACHE[key] = wrapper
        return wrapper


def _serialize_choices(raw_choices: Any) -> str:
    """Normalize metadata choice formats to the string list expected by VLMEvalKit."""

    def _as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, dict):
            return [str(value[key]) for key in sorted(value.keys()) if value[key] is not None]
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value if item is not None]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return []
            return _as_list(parsed)
        return []

    normalized = _as_list(raw_choices)
    return str(normalized) if normalized else "[]"


def _build_line(
    prediction: str,
    gold_answer: str,
    question: str | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    line = {**metadata}
    line["question"] = (
        question or metadata.get("question") or metadata.get("original_question", "")
    )
    line["prediction"] = prediction
    line["answer"] = gold_answer
    line["choices"] = _serialize_choices(metadata.get("choices"))
    return line


def _judge_model_from_env(metadata: dict[str, Any]) -> str | None:
    from os import getenv

    env_model = getenv("VLMEVAL_JUDGE_MODEL")
    if env_model:
        return env_model

    for key in ("judge_model", "model"):
        if metadata.get(key):
            return metadata[key]

    return getenv("MODEL")


# ========== Benchmark-specific verification ==========
# These benchmarks have custom evaluation logic not covered by VLMEvalKit's post_check

# Benchmarks that should use OCRBench verification (substring matching)
_OCRBENCH_BENCHMARKS = {"ocrbench", "ocrbench_mini", "ocrbench_v2"}

# Benchmarks that should use ANLS verification (edit distance)
_ANLS_BENCHMARKS = {"docvqa", "docvqa_val", "docvqa_test", "infovqa", "infovqa_val", "infovqa_test"}

# Benchmarks that should use MathGlance verification (bounding box IoU, multi-choice, counting)
_MATHGLANCE_BENCHMARKS = {"mathglance"}


def _is_ocrbench(benchmark: str | None) -> bool:
    """Check if the benchmark is OCRBench or its variants."""
    if not benchmark:
        return False
    normalized = _normalize_benchmark(benchmark)
    return normalized in _OCRBENCH_BENCHMARKS


def _is_anls_benchmark(benchmark: str | None) -> bool:
    """Check if the benchmark uses ANLS scoring (DocVQA, InfoVQA, etc.)."""
    if not benchmark:
        return False
    normalized = _normalize_benchmark(benchmark)
    return normalized in _ANLS_BENCHMARKS


def _is_mathglance(benchmark: str | None) -> bool:
    """Check if the benchmark is MathGlance."""
    if not benchmark:
        return False
    normalized = _normalize_benchmark(benchmark)
    return normalized in _MATHGLANCE_BENCHMARKS


def vlmeval_verify_sample(
    prediction: str,
    gold_answer: str,
    question: str | None,
    sample_metadata: dict[str, Any] | None,
) -> tuple[bool, Dict[str, Any]] | None:
    """Verify a sample using benchmark-specific logic.

    Returns:
        Tuple of (verified: bool, details: dict)
        details always includes 'score' field (0.0-1.0) for continuous scoring
    """
    if not sample_metadata or not gold_answer:
        return None

    benchmark = sample_metadata.get("benchmark") or sample_metadata.get(
        "benchmark_name"
    )

    # OCRBench: use custom verification (VLMEvalKit doesn't have post_check for OCRBench)
    if _is_ocrbench(benchmark):
        category = sample_metadata.get("category")
        verified, score, details = ocrbench_verify(prediction, gold_answer, category)
        details["stage"] = "vlmeval"
        details["dataset"] = benchmark
        return verified, details

    # DocVQA/InfoVQA: use ANLS verification
    if _is_anls_benchmark(benchmark):
        verified, score, details = anls_verify(prediction, gold_answer)
        details["stage"] = "vlmeval"
        details["dataset"] = benchmark
        return verified, details

    # MathGlance: custom verification for bbox, mcq, number, text
    if _is_mathglance(benchmark):
        answer_type = sample_metadata.get("answer_type")
        problem_version = sample_metadata.get("problem_version")
        verified, score, details = mathglance_verify(
            prediction, gold_answer, answer_type, problem_version
        )
        details["stage"] = "vlmeval"
        details["dataset"] = benchmark
        return verified, details

    # For other benchmarks, require VLMEvalKit
    if not _VLMEVAL_AVAILABLE:
        return None

    handler = _load_handler(benchmark)
    if handler is None:
        return None

    line = _build_line(prediction, gold_answer, question, dict(sample_metadata))

    extracted = handler.post_check(line, prefetch=True)
    details: Dict[str, Any] = {
        "method": f"vlmeval_{handler.name}_prefetch",
        "stage": "vlmeval",
        "dataset": benchmark,
        "extracted_answer": extracted,
        "score": 1.0 if extracted not in (False, None) else 0.0,  # Binary score for other benchmarks
    }
    if extracted not in (False, None):
        line["res"] = extracted
        verified = bool(handler.post_check(line, prefetch=False))
        details["score"] = 1.0 if verified else 0.0
        return verified, details

    if handler.auxeval is None:
        return None

    judge_model = _judge_model_from_env(sample_metadata)
    if not judge_model:
        return None

    max_tokens = int(sample_metadata.get("judge_max_tokens") or 128)
    wrapper = _get_wrapper(judge_model, max_tokens)
    aux = handler.auxeval(wrapper, line)
    line["res"] = aux.get("res") or aux.get("score")
    line["log"] = aux.get("log")
    verified = bool(handler.post_check(line, prefetch=False))
    details.update(
        {
            "method": f"vlmeval_{handler.name}_judge",
            "judge_log": aux.get("log"),
            "extracted_answer": aux.get("res") or aux.get("score"),
            "score": 1.0 if verified else 0.0,
        }
    )
    details["verified"] = verified
    return verified, details
