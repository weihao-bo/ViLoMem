"""Helpers for resuming interrupted evaluation runs.

This module centralizes the logic required by both evaluation scripts to:

1. Load prior run metadata (results.json, summary stats, etc.).
2. Determine which examples already produced valid predictions.
3. Prepare dataset entries with stable identifiers so they can be filtered
   safely even when concurrent execution reorders results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class PreparedExample:
    """Dataset example annotated with a deterministic ID and index."""

    data: dict[str, Any]
    example_id: str
    absolute_index: int  # 1-based index after start_index filtering


@dataclass
class ResumeState:
    """Represents the cached state from a previous evaluation run."""

    output_dir: Path
    results_file: Path
    summary: dict[str, Any]
    valid_example_ids: set[str]
    invalid_example_ids: set[str]
    total_results: int

    @property
    def completed_count(self) -> int:
        """Return the number of valid, fully-processed examples."""

        return len(self.valid_example_ids)

    @property
    def invalid_count(self) -> int:
        """Return the number of results without usable predictions."""

        return len(self.invalid_example_ids)


def resolve_resume_dir(
    cli_value: str | None, output_config: dict[str, Any]
) -> str | None:
    """Determine which output directory (if any) should be resumed."""

    if cli_value:
        return cli_value

    config_value = (output_config or {}).get("resume_dir")
    if config_value:
        return str(config_value)

    return None


def prepare_examples(
    examples: Sequence[dict[str, Any]], start_index: int
) -> list[PreparedExample]:
    """Attach deterministic IDs and indices to dataset examples."""

    prepared: list[PreparedExample] = []
    for absolute_index, example in enumerate(examples, start=start_index):
        example_id = _extract_example_id(example, absolute_index)
        prepared.append(
            PreparedExample(
                data=example,
                example_id=example_id,
                absolute_index=absolute_index,
            )
        )
    return prepared


def load_resume_state(output_dir: str | Path) -> ResumeState:
    """Load prior run metadata from ``output_dir``.

    Raises:
        FileNotFoundError: If ``results.json`` is missing.
        json.JSONDecodeError: If ``results.json`` cannot be parsed.
        ValueError: If no examples contain IDs.
    """

    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    results_file = output_path / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(
            f"results.json not found in output directory: {output_path}"
        )

    with open(results_file, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    summary = data.get("summary", {})

    valid_example_ids: set[str] = set()
    invalid_example_ids: set[str] = set()

    for entry in results:
        example_id = entry.get("example_id")
        if not example_id:
            continue

        if _is_valid_result(entry):
            valid_example_ids.add(str(example_id))
        else:
            invalid_example_ids.add(str(example_id))

    if results and not (valid_example_ids or invalid_example_ids):
        raise ValueError(
            "Existing results.json does not contain recognizable example IDs"
        )

    return ResumeState(
        output_dir=output_path,
        results_file=results_file,
        summary=summary,
        valid_example_ids=valid_example_ids,
        invalid_example_ids=invalid_example_ids,
        total_results=len(results),
    )


def _extract_example_id(example: dict[str, Any], fallback_index: int) -> str:
    """Derive a stable example ID from metadata or fall back to index."""

    metadata = example.get("metadata") or {}

    for key in ("question_id", "example_id", "id"):
        value = metadata.get(key)
        if value is not None and value != "":
            return str(value)

    return f"example_{fallback_index}"


def _is_valid_result(entry: dict[str, Any]) -> bool:
    """Return True if the stored result contains a non-empty prediction."""

    prediction = entry.get("prediction")
    if prediction is None:
        return False

    if isinstance(prediction, str):
        normalized = prediction.strip()
    else:
        normalized = str(prediction).strip()

    return bool(normalized)
