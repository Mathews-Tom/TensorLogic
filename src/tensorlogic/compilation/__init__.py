"""Compilation strategies for logical operations.

This module provides protocol-based abstraction for multiple semantic interpretations
of logical operations: soft differentiable (product), hard boolean (step), and fuzzy
variants (Gödel, product, Łukasiewicz).
"""

from __future__ import annotations

from tensorlogic.compilation.factory import (
    create_strategy,
    get_available_strategies,
    register_strategy,
    unregister_strategy,
)
from tensorlogic.compilation.protocol import CompilationStrategy
from tensorlogic.compilation.strategies import (
    GodelStrategy,
    HardBooleanStrategy,
    LukasiewiczStrategy,
    ProductStrategy,
    SoftDifferentiableStrategy,
)

# Register implemented strategies
register_strategy("soft_differentiable", SoftDifferentiableStrategy)
register_strategy("hard_boolean", HardBooleanStrategy)
register_strategy("godel", GodelStrategy)
register_strategy("product", ProductStrategy)
register_strategy("lukasiewicz", LukasiewiczStrategy)

__all__ = [
    "CompilationStrategy",
    "SoftDifferentiableStrategy",
    "HardBooleanStrategy",
    "GodelStrategy",
    "ProductStrategy",
    "LukasiewiczStrategy",
    "create_strategy",
    "register_strategy",
    "unregister_strategy",
    "get_available_strategies",
]
