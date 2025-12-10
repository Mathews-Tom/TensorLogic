"""Compilation strategy implementations.

This module provides concrete implementations of the CompilationStrategy protocol
for different semantic interpretations of logical operations.
"""

from __future__ import annotations

from tensorlogic.compilation.strategies.godel import GodelStrategy
from tensorlogic.compilation.strategies.hard import HardBooleanStrategy
from tensorlogic.compilation.strategies.lukasiewicz import LukasiewiczStrategy
from tensorlogic.compilation.strategies.product import ProductStrategy
from tensorlogic.compilation.strategies.soft import SoftDifferentiableStrategy

__all__ = [
    "SoftDifferentiableStrategy",
    "HardBooleanStrategy",
    "GodelStrategy",
    "ProductStrategy",
    "LukasiewiczStrategy",
]
