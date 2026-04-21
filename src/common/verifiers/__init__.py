"""Benchmark-specific verifiers extracted from vlmeval_bridge.

Each submodule implements verification for a benchmark whose evaluation
is not covered by VLMEvalKit's generic post_check:
    - anls: DocVQA / InfoVQA continuous scoring
    - ocrbench: OCRBench substring matching
    - mathglance: MathGlance bbox IoU / MCQ / number / text
"""

from .anls import anls_compute, anls_verify
from .mathglance import mathglance_verify
from .ocrbench import ocrbench_verify

__all__ = [
    "anls_compute",
    "anls_verify",
    "mathglance_verify",
    "ocrbench_verify",
]
