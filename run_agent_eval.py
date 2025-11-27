"""Unified evaluation runner for ViLoMem and baseline agents.

This script auto-detects whether the provided configuration enables logic/visual
memory (ViLoMem mode) or runs baseline inference. The workflow: load a local
dataset, prepare LangGraph inputs with local images, invoke the appropriate
graph, and persist results.
"""

from __future__ import annotations

import os

# ========== Disable Progress Bars for Cleaner Logs ==========
# These environment variables must be set BEFORE importing other libraries
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import asyncio
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root and src to Python path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ========== Security: Mask API Keys in Logs ==========
class APIKeyMaskingFilter(logging.Filter):
    """Filter to mask API keys in log messages for security."""

    # Pattern to match common API key formats
    API_KEY_PATTERNS = [
        (re.compile(r"(API[_ ]?Key[:\s]*)['\"]?([a-zA-Z0-9_-]{20,})['\"]?", re.I), r"\1***MASKED***"),
        (re.compile(r"(sk-[a-zA-Z0-9_-]{20,})"), r"sk-***MASKED***"),
        (re.compile(r"(api_key['\"]?\s*[:=]\s*['\"]?)([a-zA-Z0-9_-]{20,})(['\"]?)", re.I), r"\1***MASKED***\3"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if record.msg:
            msg = str(record.msg)
            for pattern, replacement in self.API_KEY_PATTERNS:
                msg = pattern.sub(replacement, msg)
            record.msg = msg
        return True


# Apply the filter to root logger to catch all logs
logging.getLogger().addFilter(APIKeyMaskingFilter())
# Also apply to common external loggers that may log API keys
for logger_name in ["ChatAPI", "vlmeval", "httpx", "openai"]:
    logging.getLogger(logger_name).addFilter(APIKeyMaskingFilter())

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from langchain_core.messages import HumanMessage  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common.logging_utils import (  # noqa: E402
    log_dataset_info,
    log_example_error,
    log_example_result,
    log_example_start,
    log_run_config,
    log_run_summary,
    setup_run_logging,
)
from tools.dataset_utils import build_message_content, load_local_dataset  # noqa: E402
from tools.resume_utils import (  # noqa: E402
    PreparedExample,
    ResumeState,
    load_resume_state,
    prepare_examples,
    resolve_resume_dir,
)

load_dotenv()

# Default configuration file paths
DEFAULT_MEMORY_CONFIG_PATH = "config/ViLoMem/MathVista_MINI.yaml"
DEFAULT_BASELINE_CONFIG_PATH = "config/baseline/MathVista_MINI.yaml"
DEFAULT_CONFIG_PATH = DEFAULT_BASELINE_CONFIG_PATH


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Apply environment variable substitution for dataset root_dir
    dataset_config = config.get("dataset", {})
    root_dir = dataset_config.get("root_dir")

    # If root_dir is not set or is a placeholder, use environment variable
    if not root_dir or root_dir in [
        "${DATASET_ROOT_DIR}",
        "$DATASET_ROOT_DIR",
        "env:DATASET_ROOT_DIR",
    ]:
        env_root_dir = os.getenv("DATASET_ROOT_DIR")
        if env_root_dir:
            dataset_config["root_dir"] = env_root_dir
            config["dataset"] = dataset_config
        elif not root_dir:
            # If neither config nor env variable is set, raise an error
            raise ValueError(
                "dataset.root_dir not specified in config and DATASET_ROOT_DIR environment variable not set. "
                "Please set DATASET_ROOT_DIR in .env file or specify root_dir in config."
            )

    return config


def detect_evaluation_mode(config: dict[str, Any]) -> str:
    """Return ``"memory"`` if logic/visual memory is enabled, else ``"baseline"``."""

    logic_memory_cfg = config.get("logic_memory", {})
    visual_memory_cfg = config.get("visual_memory", {})

    retrieval_enabled = any(
        cfg.get("retrieval", {}).get("enable", False)
        for cfg in (logic_memory_cfg, visual_memory_cfg)
    )
    generation_enabled = any(
        cfg.get("generation", {}).get("enable", False)
        for cfg in (logic_memory_cfg, visual_memory_cfg)
    )

    return "memory" if retrieval_enabled or generation_enabled else "baseline"


def _normalize_device_list(value: Any) -> list[str]:
    """Convert YAML device field (str/list) into a clean list of strings."""

    if value is None:
        return []

    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]

    if isinstance(value, (list, tuple, set)):
        normalized: list[str] = []
        for item in value:
            if item is None:
                continue
            normalized.append(str(item).strip())
        return [item for item in normalized if item]

    return []


def generate_output_dir(output_dir_prefix: str | Path, config_file_path: str) -> Path:
    """Create an output directory using the config file name with de-duplication.

    The directory name is the stem of ``config_file_path`` (e.g., ``HallusionBench``)
    under ``output_dir_prefix``. If the directory already exists, numeric suffixes
    (``name2``, ``name3`` ...) are appended until an unused path is found.
    """

    prefix_path = Path(output_dir_prefix)
    prefix_path.mkdir(parents=True, exist_ok=True)

    base_name = Path(config_file_path).stem
    candidate = prefix_path / base_name

    if candidate.exists() and not candidate.is_dir():
        raise FileExistsError(
            f"Output path already exists as a file: {candidate}. Please remove or rename it."
        )

    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    suffix = 2
    while True:
        candidate = prefix_path / f"{base_name}{suffix}"
        if candidate.exists() and not candidate.is_dir():
            raise FileExistsError(
                f"Output path already exists as a file: {candidate}. Please remove or rename it."
            )
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        suffix += 1


def ensure_dataset_ready(
    dataset_config: dict[str, Any], logger: logging.Logger | None = None
) -> str:
    """Ensure dataset is ready for evaluation with automatic validation and fixing.

    This function implements an enhanced workflow:
    1. Check if dataset.path is provided (backward compatibility - no validation)
    2. If not, use dataset.root_dir + dataset.benchmark
    3. Validate dataset integrity (version, structure, images)
    4. Auto-reconvert if outdated or corrupted
    5. Support force_reconvert option to rebuild dataset

    Args:
        dataset_config: Dataset configuration from YAML
        logger: Logger instance (optional, uses tqdm.write if not provided)

    Returns:
        Path to the validated dataset file (JSONL)
    """

    def log_info(msg: str) -> None:
        """Log info message to logger or tqdm.write."""
        if logger:
            logger.info(msg)
        else:
            tqdm.write(msg)

    # Backward compatibility: prioritize dataset.path (skip validation for manual paths)
    dataset_path = dataset_config.get("path")
    if dataset_path:
        log_info(f"Using configured dataset path (validation skipped): {dataset_path}")
        return dataset_path

    # New workflow: use root_dir + benchmark with validation
    root_dir = dataset_config.get("root_dir")
    benchmark = dataset_config.get("benchmark")
    split = dataset_config.get("split")  # Optional
    force_reconvert = dataset_config.get("force_reconvert", False)

    if not root_dir or not benchmark:
        error_msg = "Either dataset.path or both dataset.root_dir and dataset.benchmark must be specified"
        if logger:
            logger.error(error_msg)
        raise ValueError(error_msg)

    log_info(f"Benchmark: {benchmark}")
    log_info(f"Root directory: {root_dir}")
    if split:
        log_info(f"Split: {split}")
    if force_reconvert:
        log_info("Force reconversion: enabled")

    try:
        from tools.vlmevalkit_exporter import validate_and_fix_dataset

        # Validate and fix if needed
        log_info("Validating dataset integrity...")
        output_file, was_reconverted = validate_and_fix_dataset(
            benchmark=benchmark,
            root_dir=root_dir,
            force_reconvert=force_reconvert,
        )

        if was_reconverted:
            log_info(f"Dataset converted/updated: {output_file}")
        else:
            log_info(f"Dataset validated: {output_file}")

        return str(output_file)

    except Exception as e:
        error_msg = f"Failed to validate/convert dataset: {e}"
        if logger:
            logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def create_memory_context(config: dict[str, Any]):
    """Create memory-enabled Context object from configuration dictionary."""
    from vl_agent import Context

    # Extract configurations
    model_config = config.get("model", {})
    dataset_config = config.get("dataset", {})
    logic_memory_config = config.get("logic_memory", {})
    visual_memory_config = config.get("visual_memory", {})
    analysis_config = config.get("analysis", {})
    logic_memory_gen_config = config.get("logic_memory_generation", {})
    visual_memory_gen_config = config.get("visual_memory_generation", {})
    heatmap_gen_config = config.get("heatmap_generation", {})
    keyword_extraction_config = config.get("keyword_extraction", {})
    output_config = config.get("output", {})

    # Extract benchmark config (for cross-benchmark memory reuse)
    benchmark_root_dir = dataset_config.get("root_dir", "")
    current_benchmark = dataset_config.get("benchmark", "")

    qwen25vl_config = heatmap_gen_config.get("qwen25vl", {})
    qwen25vl_devices = _normalize_device_list(qwen25vl_config.get("devices"))
    per_device_limit_raw = qwen25vl_config.get("per_device_max_parallel", 1)
    try:
        qwen25vl_per_device_limit = int(per_device_limit_raw)
    except (TypeError, ValueError):
        qwen25vl_per_device_limit = 1
    qwen25vl_per_device_limit = max(1, qwen25vl_per_device_limit)

    qwen25vl_device = qwen25vl_config.get("device")
    if not qwen25vl_device:
        qwen25vl_device = qwen25vl_devices[0] if qwen25vl_devices else "cuda:0"

    # Prepare Context parameters
    context_params = {
        # Main model configuration
        "model": model_config.get("name", "qwen:qwen3-vl-8b-instruct"),
        "temperature": model_config.get("temperature", 0.7),
        "max_tokens": model_config.get("max_tokens", 4096),
        # Verification configuration
        "enable_verification": config.get("verification", {}).get("enable", True),
        # Benchmark configuration (for cross-benchmark memory reuse)
        "benchmark_root_dir": benchmark_root_dir,
        "current_benchmark": current_benchmark,
        # Logic Memory configuration
        "logic_memory_enable_retrieval": logic_memory_config.get("retrieval", {}).get(
            "enable", True
        ),
        "logic_memory_enable_generation": logic_memory_config.get("generation", {}).get(
            "enable", True
        ),
        "logic_memory_retrieval_limit": logic_memory_config.get("retrieval", {}).get(
            "limit", 3
        ),
        "logic_memory_similarity_threshold": logic_memory_config.get(
            "retrieval", {}
        ).get("similarity_threshold", 0.5),
        "logic_memory_file_path": logic_memory_config.get("store", {}).get(
            "file_path", "memories.json"
        ),
        "logic_memory_embedding_model": logic_memory_config.get("store", {}).get(
            "embedding_model", "local:qwen3-embedding"
        ),
        # Problem analysis configuration
        "analysis_model": analysis_config.get("model", "qwen:qwen-flash"),
        "analysis_temperature": analysis_config.get("temperature", 0.3),
        "analysis_max_tokens": analysis_config.get("max_tokens", 512),
        # Logic Memory generation configuration
        "logic_memory_generation_model": logic_memory_gen_config.get(
            "model", "qwen:qwen3-next-80b-a3b-instruct"
        ),
        "logic_memory_generation_temperature": logic_memory_gen_config.get(
            "temperature", 0.3
        ),
        "logic_memory_generation_max_tokens": logic_memory_gen_config.get(
            "max_tokens", 256
        ),
        # Visual Memory configuration
        "visual_memory_enable_retrieval": visual_memory_config.get("retrieval", {}).get(
            "enable", True
        ),
        "visual_memory_enable_generation": visual_memory_config.get(
            "generation", {}
        ).get("enable", True),
        "visual_memory_retrieval_limit": visual_memory_config.get("retrieval", {}).get(
            "limit", 3
        ),
        "visual_memory_similarity_threshold": visual_memory_config.get(
            "retrieval", {}
        ).get("similarity_threshold", 0.5),
        "visual_memory_enable_text_rerank": visual_memory_config.get(
            "retrieval", {}
        ).get("enable_text_rerank", True),
        "visual_memory_file_path": visual_memory_config.get("store", {}).get(
            "file_path", "visual_memories.json"
        ),
        "visual_embedding_model": visual_memory_config.get("store", {}).get(
            "embedding_model", "qwen:qwen2.5-vl-embedding"
        ),
        "visual_embedding_top_n": visual_memory_config.get("store", {}).get(
            "embedding_top_n", 10
        ),
        "visual_memory_text_embedding_model": visual_memory_config.get("store", {}).get(
            "text_embedding_model", "local:qwen3-embedding"
        ),
        # Visual Memory generation configuration
        "visual_memory_generation_model": visual_memory_gen_config.get(
            "model", "qwen:qwen3-vl-30b-a3b-instruct"
        ),
        "visual_memory_generation_temperature": visual_memory_gen_config.get(
            "temperature", 0.3
        ),
        "visual_memory_generation_max_tokens": visual_memory_gen_config.get(
            "max_tokens", 512
        ),
        # Heatmap generation configuration (Qwen2.5-VL Attention only)
        "enable_heatmap_generation": heatmap_gen_config.get("enable", False),
        "debug_heatmap": heatmap_gen_config.get("debug", False),
        "include_question_in_heatmap": heatmap_gen_config.get(
            "include_question_in_heatmap", False
        ),
        # Qwen2.5-VL Attention model configuration
        "qwen25vl_model": qwen25vl_config.get("model", "Qwen/Qwen2.5-VL-2B-Instruct"),
        "qwen25vl_general_prompt": qwen25vl_config.get(
            "general_prompt", "Describe this image."
        ),
        "qwen25vl_attention_layer": qwen25vl_config.get("attention_layer", 22),
        "qwen25vl_device": qwen25vl_device,
        "qwen25vl_devices": qwen25vl_devices,
        "qwen25vl_per_device_limit": qwen25vl_per_device_limit,
        # Memory reuse configuration
        "memory_list": output_config.get("memory_list", []),
    }

    # system_prompt: can be overridden from config file (aligned with baseline)
    if "system_prompt" in model_config:
        context_params["system_prompt"] = model_config["system_prompt"]

    context = Context(**context_params)

    return context


def create_baseline_context(config: dict[str, Any]):
    """Create baseline Context object from configuration dictionary."""

    from vl_agent_baseline import Context

    model_config = config.get("model", {})

    context_params = {
        "model": model_config.get("name", "qwen:qwen3-vl-8b-instruct"),
        "temperature": model_config.get("temperature"),
        "max_tokens": model_config.get("max_tokens"),
        "enable_verification": config.get("verification", {}).get("enable", True),
    }

    if "system_prompt" in model_config:
        context_params["system_prompt"] = model_config["system_prompt"]

    return Context(**context_params)


async def run_evaluation(
    config: dict[str, Any],
    config_file_path: str,
    resume_path: str | None = None,
) -> dict:
    """Run evaluation on local dataset using LangGraph.

    Args:
        config: Configuration dictionary loaded from YAML file
        config_file_path: Path to the configuration file

    Returns:
        Summary statistics
    """
    eval_mode = detect_evaluation_mode(config)
    is_memory_mode = eval_mode == "memory"

    if is_memory_mode:
        from vl_agent import graph as eval_graph

        context_builder = create_memory_context
        default_output_prefix = "output/vl_agent"
    else:
        from vl_agent_baseline import graph as eval_graph

        context_builder = create_baseline_context
        default_output_prefix = "output/vl_agent_baseline"

    # Extract configurations
    dataset_config = config.get("dataset", {})
    model_config = config.get("model", {})
    output_config = config.get("output", {})

    start_index = dataset_config.get("start_index", 1)  # Start from 1
    limit = dataset_config.get("limit", 2)
    task_filter = dataset_config.get("task_filter")
    output_dir_prefix = Path(output_config.get("dir_prefix", default_output_prefix))
    enable_tracing = output_config.get("enable_tracing", False)
    resume_dir = resolve_resume_dir(resume_path, output_config)

    resume_state: ResumeState | None = None
    if resume_dir:
        resume_state = load_resume_state(resume_dir)
        output_dir = resume_state.output_dir
    else:
        output_dir = generate_output_dir(output_dir_prefix, config_file_path)

    # Initialize logging system (immediately after creating output directory)
    logger = setup_run_logging(output_dir)
    mode_label = "Qwen VL Agent" if is_memory_mode else "Qwen VL Baseline Agent"
    if resume_state:
        logger.info(
            "Resuming %s Evaluation from %s (completed=%d, invalid=%d)",
            mode_label,
            output_dir,
            resume_state.completed_count,
            resume_state.invalid_count,
        )
    else:
        logger.info("Starting %s Evaluation", mode_label)

    logger.info("Evaluation mode detected: %s", eval_mode.upper())

    # Log complete configuration info
    log_run_config(logger, config, config_file_path)

    # Ensure dataset is ready (download and convert if needed)
    dataset_path = ensure_dataset_ready(dataset_config, logger)
    if resume_state:
        previous_dataset_path = resume_state.summary.get("dataset_path")
        if previous_dataset_path and previous_dataset_path != dataset_path:
            logger.warning(
                "Resume directory dataset (%s) does not match current config dataset (%s)",
                previous_dataset_path,
                dataset_path,
            )

    # ========== [NEW] Load benchmark metadata for prompt construction ==========
    dataset_dir = Path(dataset_path).parent
    benchmark_meta_file = dataset_dir / "benchmark_meta.json"

    benchmark_config = None
    if benchmark_meta_file.exists():
        try:
            with open(benchmark_meta_file, encoding="utf-8") as f:
                benchmark_meta = json.load(f)

            from tools.prompt_builder import BenchmarkPromptConfig

            # Create prompt config from metadata
            benchmark_config = BenchmarkPromptConfig.from_benchmark_meta(benchmark_meta)
            logger.info(
                f"✓ Loaded benchmark config: task_type={benchmark_config.task_type}, enable_cot={benchmark_config.enable_cot}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to load benchmark_meta.json: {e}, using default prompt construction"
            )
    else:
        logger.info("No benchmark_meta.json found, using default prompt construction")

    # Create Context object
    context = context_builder(config)
    if resume_state:
        previous_model = resume_state.summary.get("model")
        if previous_model and previous_model != context.model:
            logger.warning(
                "Resume directory model (%s) differs from current config model (%s)",
                previous_model,
                context.model,
            )

    logger.info("=" * 60)
    logger.info("System Prompt Configuration")
    logger.info("=" * 60)
    logger.info(f"System Prompt:\n{context.system_prompt}")
    logger.info("=" * 60)
    logger.info("")

    if is_memory_mode:
        memory_list = context.memory_list
        if resume_state:
            logger.info(
                "Resume mode: keeping existing memory files in %s (memory_list merge skipped)",
                output_dir,
            )
        elif memory_list:
            from vl_agent.memory import (
                batch_precompute_text_embeddings,
                copy_cached_text_embeddings,
                merge_memories_from_directories,
            )

            logger.info(
                f"Merging memories from {len(memory_list)} previous output directories:"
            )
            for memory_dir in memory_list:
                logger.info(f"  - {memory_dir}")

            logic_memories = merge_memories_from_directories(
                output_dirs=memory_list,
                memory_filename=context.logic_memory_file_path,
                reset_usage_count=True,
            )
            if logic_memories:
                logic_memory_file = output_dir / context.logic_memory_file_path
                logic_memory_file.parent.mkdir(parents=True, exist_ok=True)
                with open(logic_memory_file, "w", encoding="utf-8") as f:
                    json.dump(logic_memories, f, indent=2, ensure_ascii=False)
                logger.info(
                    f"  Merged {len(logic_memories)} logic memories -> {context.logic_memory_file_path}"
                )

                copied = copy_cached_text_embeddings(
                    memories=logic_memories,
                    memory_type="logic",
                    output_dir=output_dir,
                    model=context.logic_memory_embedding_model,
                    source_dirs=memory_list,
                )
                if copied:
                    logger.info(
                        "  ✓ Copied %d cached logic memory text embeddings", copied
                    )

                logger.info("Precomputing text embeddings for logic memories...")
                embedding_paths = await batch_precompute_text_embeddings(
                    memories=logic_memories,
                    memory_type="logic",
                    output_dir=output_dir,
                    model=context.logic_memory_embedding_model,
                    batch_size=32,
                )
                logger.info(
                    f"  ✓ Precomputed {len(embedding_paths)} logic memory text embeddings"
                )

            visual_memories = merge_memories_from_directories(
                output_dirs=memory_list,
                memory_filename=context.visual_memory_file_path,
                reset_usage_count=True,
            )
            if visual_memories:
                visual_memory_file = output_dir / context.visual_memory_file_path
                visual_memory_file.parent.mkdir(parents=True, exist_ok=True)
                with open(visual_memory_file, "w", encoding="utf-8") as f:
                    json.dump(visual_memories, f, indent=2, ensure_ascii=False)
                logger.info(
                    f"  Merged {len(visual_memories)} visual memories -> {context.visual_memory_file_path}"
                )

                copied = copy_cached_text_embeddings(
                    memories=visual_memories,
                    memory_type="visual",
                    output_dir=output_dir,
                    model=context.visual_memory_text_embedding_model,
                    source_dirs=memory_list,
                )
                if copied:
                    logger.info(
                        "  ✓ Copied %d cached visual memory text embeddings", copied
                    )

                logger.info("Precomputing text embeddings for visual memories...")
                embedding_paths = await batch_precompute_text_embeddings(
                    memories=visual_memories,
                    memory_type="visual",
                    output_dir=output_dir,
                    model=context.visual_memory_text_embedding_model,
                    batch_size=32,
                )
                logger.info(
                    f"  ✓ Precomputed {len(embedding_paths)} visual memory text embeddings"
                )
        else:
            logger.info("Starting with fresh memories (no memory_list specified)")

    # Configure LangSmith tracing
    if not enable_tracing:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith tracing disabled")
    else:
        logger.info("LangSmith tracing enabled")

    # Load dataset from local file
    logger.info(f"Loading dataset from: {dataset_path}")
    all_examples = load_local_dataset(dataset_path)
    logger.info(f"Loaded {len(all_examples)} examples from local dataset")

    # Determine images directory
    dataset_dir = Path(dataset_path)
    if dataset_dir.is_file():
        # Check if this is a converted dataset
        if dataset_dir.parent.name == "converted":
            # Images are in benchmark root directory (parent of converted/)
            images_dir = dataset_dir.parent.parent / "images"
        else:
            # Images are in same directory as dataset file
            images_dir = dataset_dir.parent / "images"
    else:
        images_dir = dataset_dir / "images"

    if not images_dir.exists():
        logger.warning(f"Images directory not found at {images_dir}")

    # Filter by task type if specified
    if task_filter:
        examples = [
            ex
            for ex in all_examples
            if ex.get("metadata") and ex["metadata"].get("task") == task_filter
        ]
        logger.info(
            f"Filtered {len(examples)} examples with task='{task_filter}' from {len(all_examples)} total"
        )
    else:
        examples = all_examples

    # Apply start_index (from 1-based to 0-based)
    if start_index > 1:
        if start_index > len(examples):
            logger.warning(
                f"start_index ({start_index}) exceeds total examples ({len(examples)}), no examples to process"
            )
            examples = []
        else:
            examples = examples[start_index - 1 :]
            logger.info(
                f"Starting from index {start_index} (skipping first {start_index - 1} examples)"
            )

    # Apply limit after start_index
    if examples:
        if limit and limit > 0:
            examples = examples[:limit]
            logger.info(f"Limited to {limit} examples")
        else:
            logger.info(f"Processing all remaining examples from index {start_index}")

    # Log dataset information summary
    log_dataset_info(
        logger,
        dataset_path,
        len(all_examples),
        len(examples),
        start_index,
        limit,
        task_filter,
    )
    logger.info(f"Output directory: {output_dir}")

    # Attach deterministic IDs to each example so resume filtering is stable
    examples_with_ids: list[PreparedExample] = prepare_examples(examples, start_index)
    examples = examples_with_ids
    current_example_ids = {prepared.example_id for prepared in examples}

    if resume_state:
        before_resume = len(examples)
        unmatched_completed = resume_state.valid_example_ids - current_example_ids
        if unmatched_completed:
            logger.warning(
                "Resume mode: %d completed examples are not part of the current selection (likely config/dataset mismatch)",
                len(unmatched_completed),
            )
        if resume_state.valid_example_ids:
            examples = [
                ex
                for ex in examples
                if ex.example_id not in resume_state.valid_example_ids
            ]
        skipped_examples = before_resume - len(examples)
        if skipped_examples:
            logger.info(
                "Resume mode: skipped %d completed examples based on prior results",
                skipped_examples,
            )
        else:
            logger.info("Resume mode: no overlapping completed examples found")

        if resume_state.invalid_example_ids:
            logger.warning(
                "Resume mode detected %d incomplete results (missing predictions); these will be re-run",
                resume_state.invalid_count,
            )
            logger.debug(
                "Incomplete example IDs: %s",
                ", ".join(sorted(resume_state.invalid_example_ids)),
            )

    pending_count = len(examples)
    if resume_state:
        logger.info(
            "Resume mode summary: %d pending / %d previously completed",
            pending_count,
            resume_state.completed_count,
        )
    else:
        logger.info(f"Pending examples: {pending_count}")

    # ========== Preload Heatmap Model (if heatmap generation is enabled) ==========
    # Preload Qwen2.5-VL model for heatmap generation (if enabled)
    if is_memory_mode and context.enable_heatmap_generation and examples:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Preloading Qwen2.5-VL Heatmap Model")
        logger.info("=" * 60)

        try:
            from vl_agent.heatmap_qwen25vl import get_or_create_qwen25vl_model

            logger.info(f"Model: {context.qwen25vl_model}")
            device_pool = context.qwen25vl_devices or [context.qwen25vl_device]
            logger.info(
                "Devices: %s (per_device_limit=%d)",
                ", ".join(device_pool),
                context.qwen25vl_per_device_limit,
            )
            logger.info(f"Attention layer: {context.qwen25vl_attention_layer}")

            # Preload model to cache (first call will load, subsequent calls will reuse)
            seen_devices: list[str] = []
            for dev in device_pool:
                if dev not in seen_devices:
                    seen_devices.append(dev)

            for dev in seen_devices:
                get_or_create_qwen25vl_model(
                    model_name=context.qwen25vl_model,
                    device=dev,
                )
            logger.info("✓ Qwen2.5-VL model preloaded successfully")
        except Exception as e:
            logger.error(f"Failed to preload heatmap model: {e}", exc_info=True)
            logger.warning("Heatmap generation may be slower on first use")

        logger.info("=" * 60)
        logger.info("")

    # Precompute visual embeddings for all images using online API (if visual memory retrieval is enabled)
    if is_memory_mode and context.visual_memory_enable_retrieval and examples:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Precomputing Visual Embeddings")
        logger.info("=" * 60)

        # Import precompute function
        from vl_agent.memory import precompute_embeddings_batch

        # Determine benchmark directory
        benchmark_dir = Path(context.benchmark_root_dir) / context.current_benchmark

        # Normalize model name for directory naming
        model_name_normalized = (
            context.visual_embedding_model.split(":", 1)[1]
            if ":" in context.visual_embedding_model
            else context.visual_embedding_model
        )
        model_name_normalized = model_name_normalized.replace("/", "_")
        embeddings_dir = benchmark_dir / "embeddings" / model_name_normalized

        # Check if embeddings already exist
        if embeddings_dir.exists():
            existing_count = len(list(embeddings_dir.glob("*.npy")))
            logger.info(
                f"Found {existing_count} existing embeddings in {embeddings_dir}"
            )
        else:
            logger.info(f"No existing embeddings found, will create {embeddings_dir}")

        # Collect all unique images referenced in the dataset (support both single and multiple images)
        dataset_images = set()
        for prepared_example in examples:
            inputs = prepared_example.data.get("inputs", {})
            image_data = inputs.get("image")
            if image_data:
                if isinstance(image_data, list):
                    # Multiple images
                    dataset_images.update(image_data)
                else:
                    # Single image
                    dataset_images.add(image_data)

        logger.info(
            f"Dataset references {len(dataset_images)} unique images (including multi-image examples)"
        )

        # Precompute embeddings using online API (will skip existing ones)
        # Note: Uses concurrent API calls within each batch for speed
        embedding_paths = await precompute_embeddings_batch(
            images_dir=images_dir,
            benchmark_dir=benchmark_dir,
            model=context.visual_embedding_model,
            batch_size=5,
        )

        # Validate embeddings completeness based on dataset-referenced images
        missing_embeddings = []
        for img_filename in dataset_images:
            if img_filename not in embedding_paths:
                missing_embeddings.append(img_filename)

        logger.info(
            f"Embeddings available for dataset images: {len(dataset_images) - len(missing_embeddings)}/{len(dataset_images)}"
        )
        logger.info(f"Embeddings stored in: {embeddings_dir}")

        if missing_embeddings:
            logger.warning(
                f"⚠️  {len(missing_embeddings)} dataset images are missing embeddings!"
            )
            logger.warning(
                f"Missing images: {', '.join(list(missing_embeddings)[:10])}"
            )
            if len(missing_embeddings) > 10:
                logger.warning(f"... and {len(missing_embeddings) - 10} more")
            logger.warning("Visual memory may not work correctly for these examples")
            # Don't fail - just warn and continue
        else:
            logger.info("✓ All dataset images have embeddings generated")

        logger.info("=" * 60)
        logger.info("")

    # Initialize results.json with metadata
    results_file = output_dir / "results.json"

    # Use the pre-compiled graph (memory or baseline)
    graph_name = "vl_agent" if is_memory_mode else "vl_agent_baseline"
    logger.info("Using pre-compiled graph: %s", graph_name)

    # Get the actual model being used (for logging)
    actual_model = context.model

    if resume_state:
        if not results_file.exists():
            raise FileNotFoundError(
                f"results.json not found in resume directory: {results_file}"
            )
        logger.info(
            "Resume mode: appending to existing results file at %s", results_file
        )
    else:
        initial_data = {
            "summary": {
                "dataset_path": dataset_path,
                "model": actual_model,
                "start_time": datetime.now().isoformat(),
                "total_examples": 0,
                "verified_count": 0,
                "accuracy": 0.0,
                "evaluation_mode": eval_mode,
                "memory_enabled": is_memory_mode,
            },
            "results": [],
        }
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Starting evaluation of {len(examples)} examples")
    logger.info("")

    # Extract concurrent execution configuration
    concurrent_config = config.get("concurrent_execution", {})
    max_workers = concurrent_config.get("max_workers", 1)
    logger.info(
        f"Concurrent execution: max_workers={max_workers} ({'parallel' if max_workers > 1 else 'serial'})"
    )
    logger.info("")

    # Helper function to process a single example
    async def process_single_example(
        prepared_example: PreparedExample,
        index: int,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        """Process a single evaluation example.

        Args:
            prepared_example: Example wrapper with deterministic metadata
            index: 1-based index in the batch
            semaphore: Optional semaphore for concurrency control
        """
        # Acquire semaphore if provided (for concurrent execution)
        if semaphore:
            await semaphore.acquire()

        try:
            actual_index = prepared_example.absolute_index
            example = prepared_example.data
            example_id = prepared_example.example_id

            # Log example start
            log_example_start(logger, example_id, index, len(examples), actual_index)

            # Extract question
            inputs = example.get("inputs", {})
            question = None
            for key in ("question", "prompt", "input", "problem"):
                if key in inputs:
                    question = inputs[key]
                    break

            if not question:
                logger.warning(f"Example {example_id}: No question found, skipping")
                return

            # Extract image filename(s) and build full path(s)
            image_data = inputs.get("image")
            image_paths = []
            image_filenames = []
            if image_data:
                if isinstance(image_data, list):
                    # Multiple images
                    for img_filename in image_data:
                        img_path = images_dir / img_filename
                        if img_path.exists():
                            image_paths.append(img_path)
                            image_filenames.append(img_filename)
                        else:
                            logger.warning(
                                f"Example {example_id}: Image not found: {img_path}"
                            )
                else:
                    # Single image (legacy format)
                    img_path = images_dir / image_data
                    if img_path.exists():
                        image_paths.append(img_path)
                        image_filenames.append(image_data)
                    else:
                        logger.warning(
                            f"Example {example_id}: Image not found: {img_path}"
                        )

            # Combine image filenames with semicolon separator
            image_path_str = ";".join(image_filenames) if image_filenames else None
            primary_image = image_filenames[0] if image_filenames else None

            # Extract sample metadata
            sample_metadata = dict(example.get("metadata", {}))
            sample_metadata.setdefault("model", model_config.get("name"))
            sample_metadata.setdefault("benchmark", dataset_config.get("benchmark"))
            sample_metadata.setdefault("question", question)
            judge_model_env = os.getenv("VLMEVAL_JUDGE_MODEL")
            if judge_model_env:
                sample_metadata.setdefault("judge_model", judge_model_env)
            judge_cfg_model = config.get("evaluation", {}).get("judge_model")
            if judge_cfg_model:
                sample_metadata.setdefault("judge_model", judge_cfg_model)
            judge_tokens_env = os.getenv("VLMEVAL_JUDGE_MAX_TOKENS")
            judge_tokens_cfg = config.get("evaluation", {}).get("judge_max_tokens")
            sample_metadata.setdefault(
                "judge_max_tokens",
                int(judge_tokens_cfg or judge_tokens_env or 128),
            )
            evaluation_cfg = config.get("evaluation", {})
            if "judge_model" in evaluation_cfg:
                sample_metadata.setdefault("judge_model", evaluation_cfg["judge_model"])
            else:
                sample_metadata.setdefault("judge_model", model_config.get("name"))
            if "judge_max_tokens" in evaluation_cfg:
                sample_metadata.setdefault(
                    "judge_max_tokens", evaluation_cfg["judge_max_tokens"]
                )

            # Parse VLMEvalKit metadata stored under "extra" (baseline compatibility)
            if "extra" in sample_metadata:
                try:
                    import ast

                    extra = sample_metadata["extra"]
                    if isinstance(extra, str):
                        extra = ast.literal_eval(extra)

                    if isinstance(extra, dict):
                        if "choices" in extra:
                            sample_metadata["choices"] = extra["choices"]
                        if "answer_option" in extra:
                            sample_metadata["answer_option"] = str(
                                extra["answer_option"]
                            )
                        for key in ("question_type", "answer_type"):
                            if key in extra:
                                sample_metadata[key] = extra[key]
                except Exception as parse_error:
                    logger.warning(
                        f"Example {example_id}: Failed to parse extra metadata: {parse_error}"
                    )

            # Add VLMEvalKit prompt if available
            vlmeval_prompt = inputs.get("vlmeval_prompt")
            if vlmeval_prompt:
                sample_metadata["vlmeval_prompt"] = vlmeval_prompt

            # Build message with local images
            message_content = build_message_content(
                question,
                image_paths,
                context.system_prompt,
                sample_metadata=sample_metadata,
                benchmark_config=benchmark_config,
            )
            message = HumanMessage(content=message_content)

            # Get gold answer
            outputs = example.get("outputs", {})
            gold_answer = None
            for key in ("answer", "gold", "expected", "label"):
                if key in outputs:
                    gold_answer = outputs[key]
                    break

            # Extract task from metadata
            task = sample_metadata.get("task")

            # Invoke evaluation graph
            try:
                config_dict = {
                    "configurable": {
                        "thread_id": example_id,
                    }
                }
                payload = {
                    "messages": [message],
                    "question": question,
                    "gold_answer": gold_answer,
                    "example_id": example_id,
                    "output_dir": str(output_dir),
                    "task": task,
                    "benchmark_config": benchmark_config,
                    "sample_metadata": sample_metadata,
                }
                if is_memory_mode:
                    payload["image_path"] = image_path_str
                else:
                    payload["image_path"] = primary_image

                state = await eval_graph.ainvoke(
                    payload,
                    context=context,
                    config=config_dict,
                )

                # Log results
                prediction = state.get("prediction", "")
                verified = state.get("verified", False)
                verification_error = state.get("verification_error")

                log_example_result(
                    logger,
                    example_id,
                    prediction,
                    gold_answer,
                    verified,
                    verification_error,
                )

            except Exception as exc:
                # Log exception as warning (non-blocking)
                log_example_error(logger, example_id, exc)

        finally:
            # Release semaphore if acquired
            if semaphore:
                semaphore.release()

    # Execute evaluations based on max_workers configuration
    if examples:
        if max_workers > 1:
            # Concurrent execution
            logger.info(
                f"Running {len(examples)} examples with {max_workers} concurrent workers"
            )
            semaphore = asyncio.Semaphore(max_workers)

            # Create tasks for all examples
            tasks = [
                process_single_example(prepared_example, i, semaphore)
                for i, prepared_example in enumerate(examples, 1)
            ]

            # Execute with progress bar
            with tqdm(
                total=len(examples),
                desc="Evaluating",
                unit="example",
                ncols=80,
                file=sys.stderr,
            ) as pbar:
                # Use as_completed to update progress bar as tasks finish
                for coro in asyncio.as_completed(tasks):
                    await coro
                    pbar.update(1)
        else:
            # Serial execution (original behavior)
            logger.info(f"Running {len(examples)} examples in serial mode")
            with tqdm(
                total=len(examples),
                desc="Evaluating",
                unit="example",
                ncols=80,
                file=sys.stderr,
            ) as pbar:
                for i, prepared_example in enumerate(examples, 1):
                    await process_single_example(prepared_example, i, semaphore=None)
                    pbar.update(1)
    else:
        logger.info("No pending examples to evaluate after resume filtering")

    # Read final statistics from results.json
    with open(results_file, encoding="utf-8") as f:
        final_data = json.load(f)

    final_summary = final_data.setdefault("summary", {})
    final_summary.setdefault("evaluation_mode", eval_mode)
    final_summary.setdefault("memory_enabled", is_memory_mode)

    # Add end time and output dir
    final_summary["end_time"] = datetime.now().isoformat()
    final_summary["output_dir"] = str(output_dir)

    # Write final version
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    # Log final summary
    log_run_summary(logger, final_summary)

    # Clean up Qwen2.5-VL heatmap models if heatmap generation was enabled
    if is_memory_mode and context.enable_heatmap_generation:
        logger.info("Cleaning up Qwen2.5-VL heatmap models...")
        from vl_agent.heatmap_qwen25vl import cleanup_qwen25vl_models

        cleanup_qwen25vl_models()
        logger.info("Qwen2.5-VL models cleaned up successfully")

    return final_data["summary"]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to configuration YAML file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from an existing output directory (skips completed cases)",
    )
    return parser.parse_args()


async def main() -> None:
    """Run the evaluation workflow."""
    args = parse_args()

    # Load configuration from YAML file (use tqdm.write before logger is initialized)
    tqdm.write(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    tqdm.write("Configuration loaded")

    # Extract configuration for display
    dataset_config = config.get("dataset", {})
    output_config = config.get("output", {})
    model_config = config.get("model", {})
    resume_target = resolve_resume_dir(args.resume, output_config)

    eval_mode = detect_evaluation_mode(config)
    mode_label = "Memory" if eval_mode == "memory" else "Baseline"

    # Print startup summary to terminal
    tqdm.write("")
    tqdm.write("=" * 60)
    tqdm.write(f"Qwen VL Agent Evaluation ({mode_label})")
    tqdm.write("=" * 60)
    tqdm.write(f"Configuration file: {args.config}")
    tqdm.write(f"Model: {model_config.get('name', 'N/A')}")
    start_index = dataset_config.get("start_index", 1)
    limit = dataset_config.get("limit", 2)
    tqdm.write(f"Start index: {start_index}")
    tqdm.write(f"Limit: {limit if limit > 0 else 'All examples'}")
    task_filter = dataset_config.get("task_filter")
    if task_filter:
        tqdm.write(f"Task filter: {task_filter}")
    tqdm.write(
        f"LangSmith tracing: {'Enabled' if output_config.get('enable_tracing') else 'Disabled'}"
    )
    if resume_target:
        tqdm.write(f"Resume from: {resume_target}")
    tqdm.write("=" * 60)
    tqdm.write("")

    # Run evaluation (logger will be initialized inside)
    summary = await run_evaluation(config, args.config, resume_path=args.resume)

    # Print final summary to terminal
    tqdm.write("")
    tqdm.write("=" * 60)
    tqdm.write("Evaluation Summary")
    tqdm.write("=" * 60)
    for key, value in summary.items():
        tqdm.write(f"{key}: {value}")
    tqdm.write("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
