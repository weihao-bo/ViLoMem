"""Logging utilities for evaluation scripts.

This module provides a unified logging system with:
- Dual handlers: file (INFO level) + console (WARNING level)
- Configuration snapshot logging
- tqdm-compatible output
- Proper separation between detailed logs and progress display
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Verification errors that indicate expected mismatches (e.g., model chose wrong option)
EXPECTED_VERIFICATION_WARNINGS = {
    "choice_mismatch",
    "choice_not_mapped",
}


class RelativePathFilter(logging.Filter):
    """Filter that converts absolute paths to project-relative paths.

    This filter modifies the pathname attribute of log records to show
    paths relative to the project root directory instead of absolute paths.
    """

    def __init__(self, project_root: Path | None = None) -> None:
        """Initialize the filter.

        Args:
            project_root: Project root directory. If None, auto-detect from git root or cwd.
        """
        super().__init__()
        if project_root is None:
            # Try to find git root
            try:
                import subprocess

                result = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                self.project_root = Path(result.stdout.strip())
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to current working directory
                self.project_root = Path.cwd()
        else:
            self.project_root = Path(project_root)

    def filter(self, record: logging.LogRecord) -> bool:
        """Convert pathname to relative path."""
        try:
            abs_path = Path(record.pathname)
            record.pathname = str(abs_path.relative_to(self.project_root))
        except (ValueError, AttributeError):
            # If relative path conversion fails, keep original
            pass
        return True


class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that works with tqdm progress bars.

    This handler uses tqdm.write() to output log messages, ensuring
    they don't interfere with progress bar display.
    """

    def __init__(self, level: int = logging.NOTSET) -> None:
        """Initialize the handler."""
        super().__init__(level)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record using tqdm.write()."""
        try:
            from tqdm import tqdm

            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)


def setup_run_logging(
    output_dir: Path | str, project_root: Path | None = None
) -> logging.Logger:
    """Set up logging system for evaluation run.

    Creates two handlers:
    1. File handler: Logs DEBUG and above to {output_dir}/run.log
    2. Console handler: Logs WARNING and above to stderr (tqdm-compatible)

    Args:
        output_dir: Directory to store log files
        project_root: Project root directory for relative paths. If None, auto-detect.

    Returns:
        Configured logger instance
    """
    output_dir = Path(output_dir)
    log_file = output_dir / "run.log"

    # Create logger
    logger = logging.getLogger("qwen_vl_eval")
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Add relative path filter to logger
    relative_path_filter = RelativePathFilter(project_root=project_root)
    logger.addFilter(relative_path_filter)

    # Create formatters with file location information
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        fmt="%(levelname)s [%(module)s:%(lineno)d]: %(message)s"
    )

    # File handler (DEBUG level, detailed format) - captures everything
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Console handler (WARNING level, simple format, tqdm-compatible)
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def log_run_config(
    logger: logging.Logger, config: dict[str, Any], config_file_path: str
) -> None:
    """Log complete run configuration to file.

    This function logs:
    1. Configuration file path
    2. Complete configuration dictionary (as formatted JSON)
    3. Timestamp

    Args:
        logger: Logger instance
        config: Configuration dictionary from YAML
        config_file_path: Path to the configuration file
    """
    logger.info("=" * 80)
    logger.info("EVALUATION RUN CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Configuration file: {config_file_path}")
    logger.info("")
    logger.info("Configuration parameters:")
    logger.info(json.dumps(config, indent=2, ensure_ascii=False))
    logger.info("=" * 80)
    logger.info("")


def log_dataset_info(
    logger: logging.Logger,
    dataset_path: str,
    total_examples: int,
    filtered_examples: int,
    start_index: int,
    limit: int | None,
    task_filter: str | None,
) -> None:
    """Log dataset loading information.

    Args:
        logger: Logger instance
        dataset_path: Path to dataset file
        total_examples: Total number of examples in dataset
        filtered_examples: Number of examples after filtering
        start_index: Starting index (1-based)
        limit: Maximum number of examples to process
        task_filter: Task type filter (if any)
    """
    logger.info("Dataset Information:")
    logger.info(f"  Path: {dataset_path}")
    logger.info(f"  Total examples: {total_examples}")
    if task_filter:
        logger.info(f"  Task filter: {task_filter}")
        logger.info(f"  Filtered examples: {filtered_examples}")
    logger.info(f"  Start index: {start_index}")
    if limit and limit > 0:
        logger.info(f"  Limit: {limit}")
    else:
        logger.info("  Limit: All remaining examples")
    logger.info(
        f"  Examples to process: {min(limit or float('inf'), filtered_examples)}"
    )
    logger.info("")


def log_example_start(
    logger: logging.Logger,
    example_id: str,
    current_index: int,
    total_count: int,
    global_index: int,
) -> None:
    """Log the start of example processing.

    Args:
        logger: Logger instance
        example_id: Example identifier
        current_index: Current position in batch (1-based)
        total_count: Total examples in batch
        global_index: Global index in dataset (1-based)
    """
    logger.info(
        f"[{current_index}/{total_count}] Processing example {example_id} "
        f"(global index: {global_index})"
    )


def log_example_result(
    logger: logging.Logger,
    example_id: str,
    prediction: str,
    gold_answer: str | None,
    verified: bool,
    verification_error: str | None = None,
) -> None:
    """Log the result of example processing.

    Args:
        logger: Logger instance
        example_id: Example identifier
        prediction: Model prediction
        gold_answer: Gold standard answer
        verified: Whether verification succeeded
        verification_error: Verification error message (if any)
    """
    logger.info(f"Example {example_id} - Results:")
    logger.info(
        f"  Prediction: {prediction[:200]}{'...' if len(prediction) > 200 else ''}"
    )
    if gold_answer:
        logger.info(f"  Gold answer: {gold_answer}")
    logger.info(f"  Verified: {verified}")
    if verification_error:
        log_fn = logger.warning
        if verification_error in EXPECTED_VERIFICATION_WARNINGS:
            log_fn = logger.info
        log_fn(f"  Verification error: {verification_error}")
    logger.info("")


def log_example_error(
    logger: logging.Logger, example_id: str, error: Exception
) -> None:
    """Log an error during example processing.

    Args:
        logger: Logger instance
        example_id: Example identifier
        error: Exception that occurred
    """
    logger.warning(f"Example {example_id} failed with error: {error}", exc_info=True)
    logger.info("")


def log_run_summary(logger: logging.Logger, summary: dict[str, Any]) -> None:
    """Log final evaluation summary.

    Args:
        logger: Logger instance
        summary: Summary statistics dictionary
    """
    logger.info("=" * 80)
    logger.info("EVALUATION RUN SUMMARY")
    logger.info("=" * 80)
    for key, value in summary.items():
        logger.info(f"{key}: {value}")
    logger.info("=" * 80)
