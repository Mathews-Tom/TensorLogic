"""Core logical operations as tensor primitives.

This module implements the fundamental tensor-to-logic operations based on
Pedro Domingos' Tensor Logic paper (arXiv:2510.12269). Logical operations
are mapped to tensor operations:
- AND: Hadamard product (element-wise multiply)
- OR: Element-wise maximum
- NOT: Complement (1 - a)
- IMPLIES: max(1-a, b)
- EXISTS: Summation + step (hard) or maximum (soft)
- FORALL: Product + step (hard) or minimum (soft)

All operations use the TensorBackend protocol for backend abstraction,
supporting MLX (primary) and NumPy (fallback) implementations.
"""

from __future__ import annotations

from tensorlogic.core.operations import (
    logical_and,
    logical_implies,
    logical_not,
    logical_or,
    step,
)
from tensorlogic.core.quantifiers import (
    exists,
    forall,
    soft_exists,
    soft_forall,
)
from tensorlogic.core.composition import (
    compose_rules,
    compose_and,
    compose_or,
)
from tensorlogic.core.temperature import (
    temperature_scaled_operation,
    deductive_operation,
    analogical_operation,
)

__all__ = [
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_implies",
    "step",
    "exists",
    "forall",
    "soft_exists",
    "soft_forall",
    "compose_rules",
    "compose_and",
    "compose_or",
    "temperature_scaled_operation",
    "deductive_operation",
    "analogical_operation",
]
