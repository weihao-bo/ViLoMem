#!/bin/bash
# Trigger first-time benchmark download (and visual-embedding precomputation)
# by running the eval pipeline once. VLMEvalKit will fetch the dataset under
# DATASET_ROOT_DIR on its own.
#
# Usage:
#   scripts/download_benchmarks.sh [path/to/config.yaml]
# Default: config/ViLoMem/MathVista_MINI.yaml
#
# To download another benchmark, duplicate the reference config, change
# `dataset.benchmark`, and pass the new path to this script.

set -e
cd "$(dirname "$0")/.."

CONFIG="${1:-config/ViLoMem/MathVista_MINI.yaml}"
uv run python run_agent_eval.py --config "$CONFIG"
