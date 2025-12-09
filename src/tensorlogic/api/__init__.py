"""PatternAPI: Einops-style pattern notation for tensor logic operations.

High-level API providing string-based pattern notation for self-documenting
logical operations over tensors.
"""

from __future__ import annotations

from tensorlogic.api.errors import (
    PatternSyntaxError,
    PatternValidationError,
    TensorLogicError,
)

__all__ = [
    "TensorLogicError",
    "PatternSyntaxError",
    "PatternValidationError",
]
