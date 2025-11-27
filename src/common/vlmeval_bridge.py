"""Bridge LangGraph verification with VLMEvalKit evaluators."""

from __future__ import annotations

import ast
import importlib
import logging
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from .provider_config import get_api_config

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
    auxeval: Optional[Callable[[Any, dict[str, Any]], dict[str, Any]]] = None


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
_WRAPPER_CACHE: dict[Tuple[str, Optional[str], int], OpenAIWrapper] = {}
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


def _normalize_benchmark(name: str | None) -> Optional[str]:
    if not name:
        return None
    base = name.lower()
    base = re.sub(r"[- ]", "_", base)
    base = re.sub(r"(_mini|_test.*|_val.*|_dev.*|_en|_cn)$", "", base)
    return base


def _load_handler(benchmark: str | None) -> Optional[DatasetHandler]:
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
    question: Optional[str],
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


def _judge_model_from_env(metadata: dict[str, Any]) -> Optional[str]:
    from os import getenv

    env_model = getenv("VLMEVAL_JUDGE_MODEL")
    if env_model:
        return env_model

    for key in ("judge_model", "model"):
        if metadata.get(key):
            return metadata[key]

    return getenv("MODEL")


def vlmeval_verify_sample(
    prediction: str,
    gold_answer: str,
    question: Optional[str],
    sample_metadata: Optional[dict[str, Any]],
) -> Optional[tuple[bool, Dict[str, Any]]]:
    if not _VLMEVAL_AVAILABLE or not sample_metadata or not gold_answer:
        return None

    benchmark = sample_metadata.get("benchmark") or sample_metadata.get(
        "benchmark_name"
    )
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
    }
    if extracted not in (False, None):
        line["res"] = extracted
        verified = bool(handler.post_check(line, prefetch=False))
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
        }
    )
    details["verified"] = verified
    return verified, details
