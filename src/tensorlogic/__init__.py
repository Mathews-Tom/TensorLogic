"""TensorLogic: Neural-symbolic AI framework unifying logical reasoning and tensor computation.

Based on Pedro Domingos' Tensor Logic paper (arXiv:2510.12269), implements the
mathematical equivalence between logical rules and Einstein summation.

Core Concept:
    Logical operations map to tensor operations:
    - AND -> Hadamard product (element-wise multiply)
    - OR -> Element-wise maximum
    - NOT -> Complement (1 - x)
    - EXISTS -> Summation over axes + step
    - FORALL -> Product over axes + step
    - Temperature control: T=0 (deductive), T>0 (analogical)

Quick Start:
    >>> from tensorlogic import quantify, reason, create_backend
    >>> backend = create_backend()  # MLX if available, else NumPy
    >>> result = reason(
    ...     'exists y: Parent(x, y) and Parent(y, z)',
    ...     relations={'Parent': parent_tensor},
    ...     temperature=0.0  # Deductive mode
    ... )

Components:
    - api: High-level einops-style pattern API (quantify, reason)
    - backends: Backend abstraction (MLX, NumPy)
    - compilation: Compilation strategies (soft, hard, fuzzy)
    - core: Fundamental tensor-logic operations
"""

from __future__ import annotations

# High-level API functions
from tensorlogic.api import quantify, reason

# Error types
from tensorlogic.api import (
    TensorLogicError,
    PatternSyntaxError,
    PatternValidationError,
)

# Backend abstraction
from tensorlogic.backends import (
    TensorBackend,
    NumpyBackend,
    create_backend,
    validate_backend,
)

# Compilation strategies
from tensorlogic.compilation import (
    CompilationStrategy,
    create_strategy,
    get_available_strategies,
)

# Core operations
from tensorlogic.core import (
    # Logical operators
    logical_and,
    logical_or,
    logical_not,
    logical_implies,
    step,
    # Quantifiers
    exists,
    forall,
    soft_exists,
    soft_forall,
    # Composition
    compose_rules,
    compose_and,
    compose_or,
    # Temperature control
    temperature_scaled_operation,
    deductive_operation,
    analogical_operation,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # High-level API
    "quantify",
    "reason",
    # Errors
    "TensorLogicError",
    "PatternSyntaxError",
    "PatternValidationError",
    # Backend
    "TensorBackend",
    "NumpyBackend",
    "create_backend",
    "validate_backend",
    # Compilation
    "CompilationStrategy",
    "create_strategy",
    "get_available_strategies",
    # Core operations
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_implies",
    "step",
    # Quantifiers
    "exists",
    "forall",
    "soft_exists",
    "soft_forall",
    # Composition
    "compose_rules",
    "compose_and",
    "compose_or",
    # Temperature control
    "temperature_scaled_operation",
    "deductive_operation",
    "analogical_operation",
]
