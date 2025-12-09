"""Core logical operations as tensor primitives.

This module implements the fundamental tensor-to-logic operations based on
Pedro Domingos' Tensor Logic paper (arXiv:2510.12269). Logical operations
are mapped to tensor operations:
- AND: Hadamard product (element-wise multiply)
- OR: Element-wise maximum
- NOT: Complement (1 - a)
- IMPLIES: max(1-a, b)

All operations use the TensorBackend protocol for backend abstraction,
supporting MLX (primary) and NumPy (fallback) implementations.
"""

from __future__ import annotations

from tensorlogic.core.operations import (
    logical_and,
    logical_not,
    logical_or,
)

__all__ = [
    "logical_and",
    "logical_or",
    "logical_not",
]
