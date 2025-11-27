"""VLMEvalKit Dataset Exporter.

This module provides functionality to export VLMEvalKit benchmarks to a standard format:
- Downloads benchmark data via VLMEvalKit's build_dataset()
- Converts TSV format to standardized JSONL + images/
- Supports multi-image samples
- Preserves metadata (answer_type, choices, hint, etc.)
- Generates benchmark_meta.json for prompt/verification configuration

Target structure:
    {root_dir}/{benchmark}/
    ├── converted/
    │   ├── dataset.jsonl    # Standard format with metadata
    │   └── benchmark_meta.json  # Benchmark-level configuration
    └── images/
        ├── 1.jpg
        ├── 1--2.jpg  # Multi-image samples use --N suffix
        └── ...
"""

from __future__ import annotations

import ast
import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Conversion version - increment when format changes
CONVERSION_VERSION = "1.3.0"  # v1.3.0: Preserve native choices/answer_option columns for VLMEvalKit verification
OPTION_LETTERS = [chr(65 + i) for i in range(26)]


def _safe_literal_eval(value: Any) -> Any:
    """Safely parse literal strings; return original value on failure."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _normalize_answer_option(option: Any) -> str | None:
    """Normalize answer option letters (A-J)."""
    if option is None or (isinstance(option, float) and pd.isna(option)):
        return None

    option_str = str(option).strip()
    if not option_str:
        return None

    if len(option_str) == 1:
        letter = option_str.upper()
        if letter in OPTION_LETTERS:
            return letter
    return option_str


def _normalize_choices(raw: Any) -> dict[str, str]:
    """Normalize raw choices (list/dict/tuple) into {'A': 'option', ...}."""
    normalized: dict[str, str] = {}
    parsed = _safe_literal_eval(raw)

    if isinstance(parsed, dict):
        for key, value in parsed.items():
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            letter = str(key).strip().upper()
            if not letter:
                continue
            normalized[letter] = str(value)
    elif isinstance(parsed, (list, tuple)):
        for idx, value in enumerate(parsed):
            if idx >= len(OPTION_LETTERS):
                break
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            option_letter = OPTION_LETTERS[idx]
            normalized[option_letter] = str(value)

    # Ensure values are non-empty strings
    return {letter: val for letter, val in normalized.items() if str(val).strip()}


def _extract_mcq_metadata(row: pd.Series) -> tuple[dict[str, str], str | None]:
    """Extract choices and answer_option metadata from a dataset row."""
    choices: dict[str, str] = {}
    answer_option: str | None = _normalize_answer_option(row.get("answer_option"))

    # Method 1: Native `choices` column (list/dict/string)
    if "choices" in row and pd.notna(row["choices"]):
        normalized = _normalize_choices(row["choices"])
        if normalized:
            choices = normalized

    # Method 2: Individual columns (A, B, C, ...) if no native choices
    if not choices:
        column_choices = {}
        for letter in OPTION_LETTERS:
            if letter not in row:
                continue
            value = row.get(letter)
            if pd.notna(value) and str(value).strip():
                column_choices[letter] = str(value)
        if column_choices:
            choices = column_choices

    # Method 3: `extra` dict storing choices/answer_option
    if not choices or not answer_option:
        extra_raw = row.get("extra")
        if pd.notna(extra_raw):
            extra = _safe_literal_eval(extra_raw)
            if isinstance(extra, dict):
                if not choices and "choices" in extra:
                    normalized = _normalize_choices(extra["choices"])
                    if normalized:
                        choices = normalized
                if not answer_option and "answer_option" in extra:
                    answer_option = _normalize_answer_option(extra["answer_option"])

    # Infer answer option from gold answer if still missing
    if choices and not answer_option:
        answer_value = str(row["answer"]).strip()
        for letter, text in choices.items():
            if str(text).strip() == answer_value:
                answer_option = letter
                break

    return choices, answer_option


def validate_and_fix_dataset(
    benchmark: str,
    root_dir: str | Path,
    force_reconvert: bool = False,
) -> tuple[Path, bool]:
    """Validate dataset integrity and re-convert if needed.

    This function should be called before running evaluations to ensure:
    - Dataset exists and is up-to-date (conversion version matches)
    - All required files are present (JSONL, images, metadata)
    - Data structure is correct

    Args:
        benchmark: VLMEvalKit benchmark name
        root_dir: Root directory for all datasets
        force_reconvert: If True, always re-convert even if valid

    Returns:
        Tuple of (dataset_path, was_reconverted)
            - dataset_path: Path to validated dataset.jsonl
            - was_reconverted: True if dataset was re-converted

    Raises:
        RuntimeError: If validation or conversion fails
    """
    root_dir = Path(root_dir)
    benchmark_dir = root_dir / benchmark
    converted_dir = benchmark_dir / "converted"
    images_dir = benchmark_dir / "images"
    output_file = converted_dir / "dataset.jsonl"
    meta_file = converted_dir / "benchmark_meta.json"

    print(f"🔍 Validating dataset: {benchmark}")

    # Check if files exist
    if not output_file.exists() or not meta_file.exists():
        print("  ⚠️  Dataset not found, converting...")
        dataset_path = export_dataset(benchmark, root_dir)
        return dataset_path, True

    # Force reconversion if requested
    if force_reconvert:
        print("  🔄 Force reconversion requested")
        # Remove old files
        if output_file.exists():
            output_file.unlink()
        if meta_file.exists():
            meta_file.unlink()
        dataset_path = export_dataset(benchmark, root_dir)
        return dataset_path, True

    # Check conversion version
    try:
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)

        current_version = meta.get("conversion_version", "0.0.0")
        if current_version != CONVERSION_VERSION:
            print(f"  ⚠️  Outdated version: {current_version} → {CONVERSION_VERSION}")
            print("  🔄 Re-converting...")
            # Remove old files
            output_file.unlink()
            meta_file.unlink()
            dataset_path = export_dataset(benchmark, root_dir)
            return dataset_path, True

        print(f"  ✓ Version: {current_version}")
    except Exception as e:
        print(f"  ⚠️  Failed to read metadata: {e}")
        print("  🔄 Re-converting...")
        dataset_path = export_dataset(benchmark, root_dir)
        return dataset_path, True

    # Validate data integrity
    validation_result = _quick_validate_dataset(output_file, images_dir)
    if not validation_result["valid"]:
        print(f"  ⚠️  Validation failed: {validation_result['error']}")
        print("  🔄 Re-converting...")
        # Remove old files
        output_file.unlink()
        meta_file.unlink()
        dataset_path = export_dataset(benchmark, root_dir)
        return dataset_path, True

    print("  ✓ Validation passed")
    print(f"  ✓ Dataset ready: {output_file}")

    return output_file, False


def export_dataset(
    benchmark: str,
    root_dir: str | Path,
    split: str | None = None,
) -> Path:
    """Export VLMEvalKit benchmark to standard JSONL format.

    Args:
        benchmark: VLMEvalKit benchmark name (e.g., "MathVista_MINI")
        root_dir: Root directory for all datasets
        split: Optional split name (if None, uses dataset default)

    Returns:
        Path to the converted dataset.jsonl file

    Raises:
        ValueError: If benchmark is invalid or split doesn't exist
        RuntimeError: If download or conversion fails
    """
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # Set LMUData environment variable to control VLMEvalKit's data location
    os.environ["LMUData"] = str(root_dir)
    print(f"📁 Data root: {root_dir}")

    # Import VLMEvalKit after setting environment variable
    try:
        from vlmeval.dataset import build_dataset
    except ImportError as e:
        raise RuntimeError(f"Failed to import VLMEvalKit: {e}") from e

    # Setup directory structure
    benchmark_dir = root_dir / benchmark
    converted_dir = benchmark_dir / "converted"
    images_dir = benchmark_dir / "images"

    converted_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    output_file = converted_dir / "dataset.jsonl"
    meta_file = converted_dir / "benchmark_meta.json"

    # Check if already converted and up-to-date
    if output_file.exists() and meta_file.exists():
        # Validate conversion version
        try:
            with open(meta_file, encoding="utf-8") as f:
                existing_meta = json.load(f)

            existing_version = existing_meta.get("conversion_version", "0.0.0")
            if existing_version == CONVERSION_VERSION:
                print(f"✓ Using cached dataset (v{existing_version}): {output_file}")

                # Quick validation
                validation_result = _quick_validate_dataset(output_file, images_dir)
                if validation_result["valid"]:
                    print("  ✓ Quick validation passed")
                    return output_file
                else:
                    print(f"  ⚠️  Validation failed: {validation_result['error']}")
                    print("  🔄 Re-converting dataset...")
            else:
                print(
                    f"  ⚠️  Outdated conversion (v{existing_version} → v{CONVERSION_VERSION})"
                )
                print("  🔄 Re-converting dataset...")
        except Exception as e:
            print(f"  ⚠️  Failed to validate existing dataset: {e}")
            print("  🔄 Re-converting dataset...")

    # Load dataset via VLMEvalKit
    print(f"📦 Loading benchmark: {benchmark}")
    try:
        dataset = build_dataset(benchmark)
        print(f"✓ Loaded dataset: {type(dataset).__name__}")
    except Exception as e:
        raise RuntimeError(f"Failed to load benchmark '{benchmark}': {e}") from e

    # Get data from dataset
    if not hasattr(dataset, "data"):
        raise RuntimeError(f"Dataset '{benchmark}' does not have .data attribute")

    data: pd.DataFrame = dataset.data

    if not isinstance(data, pd.DataFrame):
        raise RuntimeError(f"Dataset.data is not a pandas DataFrame, got {type(data)}")

    print(
        f"📊 Dataset shape: {data.shape} (rows={len(data)}, columns={len(data.columns)})"
    )
    print(f"   Columns: {list(data.columns)}")

    # Convert TSV data to standard format
    print(f"\n🔄 Converting {len(data)} samples...")

    converted_count = 0
    failed_count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for position, (df_index, row) in enumerate(data.iterrows()):
            try:
                # Build VLMEvalKit prompt (official format)
                vlmeval_prompt = None
                if hasattr(dataset, "build_prompt"):
                    try:
                        # build_prompt expects positional indices (0..N-1)
                        vlmeval_prompt = dataset.build_prompt(position)
                    except Exception as e:
                        print(
                            f"  ⚠️  Warning: Failed to build prompt for sample {df_index} (pos {position}): {e}"
                        )

                # Convert row to standard format
                converted = _convert_sample(row, images_dir, vlmeval_prompt)

                # Write to JSONL
                f.write(json.dumps(converted, ensure_ascii=False) + "\n")
                converted_count += 1

                # Progress indicator
                if (position + 1) % 100 == 0:
                    print(f"  Progress: {position + 1}/{len(data)} samples converted")

            except Exception as e:
                print(
                    f"  ⚠️  Warning: Failed to convert sample {df_index} (pos {position}): {e}"
                )
                failed_count += 1

    print("\n✓ Conversion complete!")
    print(f"  - Successfully converted: {converted_count} samples")
    if failed_count > 0:
        print(f"  - Failed: {failed_count} samples")
    print(f"  - Output: {output_file}")

    # Count images (including multi-image files with -- suffix)
    image_files = list(images_dir.glob("*.*"))
    print(f"  - Images: {images_dir}/ ({len(image_files)} files)")

    # Generate benchmark metadata
    print("\n📋 Generating benchmark metadata...")
    benchmark_meta = _generate_benchmark_meta(benchmark, dataset, data)
    benchmark_meta["conversion_version"] = CONVERSION_VERSION  # Add version tracking
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_meta, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Metadata saved: {meta_file}")
    print(f"  - Conversion version: {CONVERSION_VERSION}")
    print(f"  - Task type: {benchmark_meta['task_type']}")
    print(f"  - Answer type: {benchmark_meta['default_answer_type']}")

    # Validate converted data
    _validate_converted_data(output_file, images_dir, converted_count)

    return output_file


def _convert_sample(
    row: pd.Series, images_dir: Path, vlmeval_prompt: list[dict] | None = None
) -> dict[str, Any]:
    """Convert a single TSV row to standard format.

    Args:
        row: Pandas Series representing one sample
        images_dir: Directory to save decoded images
        vlmeval_prompt: Optional VLMEvalKit prompt from dataset.build_prompt()

    Returns:
        Dictionary in standard format with complete metadata
    """
    # Extract question_id (use index field)
    question_id = str(row["index"])

    # Handle both single and multiple images
    image_data = row["image"]
    image_filenames = []

    if isinstance(image_data, list):
        # Multiple images: save each with --N suffix
        for i, img_b64 in enumerate(image_data, 1):
            filename = _save_base64_image(
                img_b64, images_dir, question_id, suffix=f"--{i}"
            )
            image_filenames.append(filename)
    elif isinstance(image_data, str):
        # Single image
        filename = _save_base64_image(image_data, images_dir, question_id)
        image_filenames.append(filename)
    else:
        raise ValueError(f"Image data must be string or list, got {type(image_data)}")

    # Build standard format
    standard = {
        "inputs": {
            "question": str(row["question"]),
            "image": image_filenames,  # Always a list for consistency
        },
        "outputs": {
            "answer": str(row["answer"]),
        },
        "metadata": {
            "question_id": question_id,
        },
    }

    # Add VLMEvalKit prompt if available (official benchmark format)
    if vlmeval_prompt is not None:
        standard["inputs"]["vlmeval_prompt"] = vlmeval_prompt

    # Infer answer type
    answer_type = _infer_answer_type(row)
    if answer_type:
        standard["metadata"]["answer_type"] = answer_type

    # Extract MCQ metadata (choices + answer_option) for verification
    choices, answer_option = _extract_mcq_metadata(row)
    if choices:
        standard["metadata"]["choices"] = choices
    if answer_option:
        standard["metadata"]["answer_option"] = answer_option

    # Extract hint (if present)
    hint = row.get("hint")
    if pd.notna(hint) and str(hint).strip():
        standard["metadata"]["hint"] = str(hint)

    # Add other optional metadata fields
    optional_fields = [
        "task",
        "category",
        "context",
        "source",
        "question_type",
        "skills",
        "l2-category",
        "l2_category",
        "split",
        "subfield",
        "topic_difficulty",
        "image_type",
    ]

    for field in optional_fields:
        value = row.get(field)
        if pd.notna(value):
            # Use safe field name (replace '-' with '_')
            safe_field = field.replace("-", "_")
            standard["metadata"][safe_field] = str(value)

    return standard


def _save_base64_image(
    base64_string: str, images_dir: Path, question_id: str, suffix: str = ""
) -> str:
    """Save base64-encoded image to file.

    Args:
        base64_string: Base64-encoded image data
        images_dir: Directory to save image
        question_id: Question ID for filename
        suffix: Optional suffix (e.g., '--2' for second image)

    Returns:
        Filename (relative to images_dir)
    """
    # Decode base64
    try:
        decoded_image = base64.b64decode(base64_string)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {e}") from e

    # Determine image format
    if decoded_image[:4] == b"\x89PNG":
        ext = "png"
    elif decoded_image[:2] == b"\xff\xd8":
        ext = "jpg"
    elif decoded_image[:3] == b"GIF":
        ext = "gif"
    else:
        ext = "jpg"  # Default

    # Build filename
    filename = f"{question_id}{suffix}.{ext}"
    image_path = images_dir / filename

    # Save image
    with open(image_path, "wb") as f:
        f.write(decoded_image)

    return filename


def _infer_answer_type(row: pd.Series) -> str | None:
    """Infer answer type from row data.

    Args:
        row: Row from dataset

    Returns:
        Answer type: 'float', 'mcq', 'text', or None if cannot infer
    """
    answer = str(row["answer"]).strip()

    # Check if explicit answer_type field exists
    if "answer_type" in row and pd.notna(row["answer_type"]):
        return str(row["answer_type"])

    # Check if MCQ (answer is single letter A-J)
    if re.match(r"^[A-J]$", answer):
        return "mcq"

    # Check if numeric
    try:
        float(answer.replace(",", ""))
        return "float"
    except ValueError:
        pass

    # Check if yes/no
    if answer.lower() in ("yes", "no", "true", "false"):
        return "text"

    # Default to text
    return "text"


def _generate_benchmark_meta(
    benchmark: str, dataset: Any, data: pd.DataFrame
) -> dict[str, Any]:
    """Generate benchmark-level metadata for prompt construction.

    Args:
        benchmark: Benchmark name
        dataset: VLMEvalKit dataset object
        data: DataFrame with all samples

    Returns:
        Benchmark metadata dictionary
    """
    dataset_name = type(dataset).__name__.lower()
    benchmark_lower = benchmark.lower()

    # Determine task type
    task_type = "vqa"  # Default
    if "mcq" in dataset_name or any(
        x in benchmark_lower
        for x in ["mmbench", "seed", "mmmu", "scienceqa", "mmstar", "blink"]
    ):
        task_type = "mcq"
    elif any(
        x in benchmark_lower
        for x in ["mathvista", "mathverse", "olympiad", "math", "geometryqa"]
    ):
        task_type = "math"
    elif any(x in benchmark_lower for x in ["chartqa", "docvqa", "infovqa"]):
        task_type = "vqa"

    # Determine default answer type
    default_answer_type = "text"
    if task_type == "mcq":
        default_answer_type = "mcq"
    elif task_type == "math":
        # Check first few samples
        sample_answers = data["answer"].head(10)
        numeric_count = sum(1 for ans in sample_answers if _is_numeric(str(ans)))
        if numeric_count > 5:
            default_answer_type = "float"
        else:
            default_answer_type = "text"

    # Check if dataset has choices
    has_choices = any(col in data.columns for col in ["A", "B", "C", "D"])

    # Check if dataset has hints
    has_hint = "hint" in data.columns and data["hint"].notna().any()

    # Determine if CoT should be enabled
    enable_cot = task_type in ["math", "vqa"]  # Enable CoT for reasoning tasks

    return {
        "benchmark": benchmark,
        "dataset_class": type(dataset).__name__,
        "task_type": task_type,
        "default_answer_type": default_answer_type,
        "has_choices": bool(has_choices),  # Convert numpy.bool_ to Python bool
        "has_hint": bool(has_hint),  # Convert numpy.bool_ to Python bool
        "enable_cot": bool(enable_cot),  # Ensure Python bool
        "num_samples": int(len(data)),  # Ensure Python int
    }


def _is_numeric(s: str) -> bool:
    """Check if string represents a numeric value."""
    try:
        float(s.replace(",", "").replace("$", "").replace("%", ""))
        return True
    except ValueError:
        return False


def _quick_validate_dataset(
    output_file: Path,
    images_dir: Path,
) -> dict[str, Any]:
    """Quick validation of converted dataset.

    Args:
        output_file: Path to converted JSONL file
        images_dir: Path to images directory

    Returns:
        Dictionary with 'valid' (bool) and 'error' (str or None)
    """
    try:
        # Check JSONL file exists
        if not output_file.exists():
            return {"valid": False, "error": "Output file not found"}

        # Check at least one sample exists
        with open(output_file, encoding="utf-8") as f:
            first_line = f.readline()
            if not first_line:
                return {"valid": False, "error": "Empty dataset"}

            # Validate first sample structure
            sample = json.loads(first_line)

            # Check required fields
            required_fields = [
                ("inputs", "question"),
                ("inputs", "image"),
                ("outputs", "answer"),
                ("metadata", "question_id"),
            ]

            for *path, field in required_fields:
                obj = sample
                for key in path:
                    obj = obj.get(key, {})
                if field not in obj:
                    return {
                        "valid": False,
                        "error": f"Missing field: {'.'.join([*path, field])}",
                    }

            # Check that image is a list
            if not isinstance(sample["inputs"]["image"], list):
                return {"valid": False, "error": "inputs.image must be a list"}

            # Check that at least one image file exists
            image_files = sample["inputs"]["image"]
            if image_files:
                first_image = images_dir / image_files[0]
                if not first_image.exists():
                    return {
                        "valid": False,
                        "error": f"Image file not found: {image_files[0]}",
                    }

        return {"valid": True, "error": None}

    except Exception as e:
        return {"valid": False, "error": str(e)}


def _validate_converted_data(
    output_file: Path,
    images_dir: Path,
    expected_count: int,
) -> None:
    """Comprehensive validation of converted dataset.

    Args:
        output_file: Path to converted JSONL file
        images_dir: Path to images directory
        expected_count: Expected number of samples
    """
    print("\n🔍 Validating converted data...")

    # Check JSONL file
    if not output_file.exists():
        raise RuntimeError(f"Output file not found: {output_file}")

    # Count lines in JSONL
    with open(output_file, encoding="utf-8") as f:
        actual_count = sum(1 for _ in f)

    if actual_count != expected_count:
        print(f"  ⚠️  Warning: Expected {expected_count} samples, found {actual_count}")
    else:
        print(f"  ✓ Sample count: {actual_count}")

    # Validate sample structure (check first 10 samples)
    missing_images = []
    samples_checked = 0

    with open(output_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 10:  # Check first 10 samples
                break

            sample = json.loads(line)

            # Check required fields
            required_fields = [
                ("inputs", "question"),
                ("inputs", "image"),
                ("outputs", "answer"),
                ("metadata", "question_id"),
            ]

            for *path, field in required_fields:
                obj = sample
                for key in path:
                    obj = obj.get(key, {})
                if field not in obj:
                    raise ValueError(
                        f"Sample {i}: Missing field: {'.'.join([*path, field])}"
                    )

            # Check that image is a list
            if not isinstance(sample["inputs"]["image"], list):
                raise ValueError(f"Sample {i}: inputs.image must be a list")

            # Check image files exist
            for img_filename in sample["inputs"]["image"]:
                img_path = images_dir / img_filename
                if not img_path.exists():
                    missing_images.append(img_filename)

            samples_checked += 1

    print(f"  ✓ Checked {samples_checked} samples")
    print("  ✓ Required fields present")
    print("  ✓ Image field format correct (list)")

    if missing_images:
        print(f"  ⚠️  Warning: {len(missing_images)} image files not found")
        print(f"     First missing: {missing_images[0]}")
    else:
        print("  ✓ Image files exist")

    print("✓ Validation passed!")
