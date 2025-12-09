"""Temperature-controlled reasoning for TensorLogic.

This module implements temperature parameter for controlling reasoning mode
interpolation between deductive (T=0) and analogical (T>0) inference.

Temperature Modes:
    T=0.0: Purely deductive (hard boolean, no hallucinations)
    T>0.0: Analogical reasoning (soft probabilities, generalization)
    T→∞: Maximum entropy (uniform distribution)

Mathematical Formulation:
    result = (1-α)·hard_result + α·soft_result
    where α = 1 - exp(-T) for smooth interpolation

Key Insight:
    At T=0, α=0, giving 100% hard boolean operations (step function).
    As T increases, α→1, giving continuous soft operations.
    This enables controlled neural-symbolic reasoning.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

from tensorlogic.backends import TensorBackend
from tensorlogic.core.operations import step


def temperature_scaled_operation(
    operation: Callable[..., Any],
    temperature: float,
    *,
    backend: TensorBackend,
) -> Callable[..., Any]:
    """Scale logical operation by temperature.

    Wraps a logical operation to support temperature-controlled reasoning.
    At T=0, applies hard boolean logic (step function). At T>0, uses
    soft continuous operations for analogical reasoning.

    Temperature Modes:
        T=0.0: Purely deductive (hard boolean, no hallucinations)
        T>0.0: Analogical reasoning (soft probabilities, generalization)
        T→∞: Maximum entropy (uniform distribution)

    Mathematical Formulation:
        result = (1-α)·step(op(...)) + α·op(...)
        where α = 1 - exp(-T)

    Args:
        operation: Logical operation function to wrap
        temperature: Temperature parameter (≥0.0)
        backend: Tensor backend for computation

    Returns:
        Temperature-scaled operation function

    Raises:
        ValueError: If temperature is negative

    Examples:
        >>> # Deductive reasoning (T=0)
        >>> op = temperature_scaled_operation(logical_and, temperature=0.0, backend=backend)
        >>> result = op(a, b, backend=backend)  # Exact boolean AND

        >>> # Analogical reasoning (T=1.0)
        >>> op = temperature_scaled_operation(logical_and, temperature=1.0, backend=backend)
        >>> result = op(a, b, backend=backend)  # Soft probabilistic AND

        >>> # High temperature (T=5.0) - maximum entropy
        >>> op = temperature_scaled_operation(logical_or, temperature=5.0, backend=backend)
        >>> result = op(a, b, backend=backend)  # Very soft OR

    Notes:
        - The interpolation parameter α uses exponential decay: α = 1 - exp(-T)
        - At T=0, α=0 (100% hard boolean)
        - At T=1, α≈0.632 (63% soft, 37% hard)
        - At T≥5, α≈0.993 (99% soft, 1% hard)
        - This provides smooth transition from deductive to analogical reasoning
    """
    # Validate temperature
    if temperature < 0.0:
        raise ValueError(f"Temperature must be non-negative, got {temperature}")

    # Compute interpolation weight: α = 1 - exp(-T)
    # At T=0: α=0 (fully hard/boolean)
    # At T→∞: α→1 (fully soft/continuous)
    import math

    alpha = 1.0 - math.exp(-temperature)

    # Capture backend from closure to avoid parameter shadowing
    closure_backend = backend

    def wrapped_operation(*args: Any, backend: TensorBackend | None = None, **kwargs: Any) -> Any:
        """Temperature-scaled wrapper around the operation.

        Args:
            *args: Positional arguments for the operation
            backend: Tensor backend (optional, uses closure backend if not provided)
            **kwargs: Keyword arguments for the operation

        Returns:
            Temperature-scaled result tensor
        """
        # Use provided backend or fall back to closure backend
        active_backend = backend if backend is not None else closure_backend

        # Compute soft (continuous) result by calling the underlying operation
        # with the active backend
        soft_result = operation(*args, backend=active_backend, **kwargs)

        # If temperature is exactly 0, return hard boolean result
        if temperature == 0.0:
            hard_result = step(soft_result, backend=active_backend)
            return hard_result

        # Otherwise, interpolate between hard and soft
        # result = (1-α)·hard + α·soft
        hard_result = step(soft_result, backend=active_backend)

        # Interpolate: (1-α)·hard + α·soft
        result = active_backend.add(
            active_backend.multiply(hard_result, (1.0 - alpha)),
            active_backend.multiply(soft_result, alpha),
        )

        return result

    return wrapped_operation


def deductive_operation(
    operation: Callable[..., Any],
    *,
    backend: TensorBackend,
) -> Callable[..., Any]:
    """Create deductive (hard boolean) version of operation.

    Convenience function equivalent to:
        temperature_scaled_operation(operation, temperature=0.0, backend=backend)

    Args:
        operation: Logical operation function to wrap
        backend: Tensor backend for computation

    Returns:
        Deductive (hard boolean) operation function

    Examples:
        >>> op = deductive_operation(logical_and, backend=backend)
        >>> result = op(a, b, backend=backend)  # Hard boolean AND
    """
    return temperature_scaled_operation(operation, temperature=0.0, backend=backend)


def analogical_operation(
    operation: Callable[..., Any],
    temperature: float = 1.0,
    *,
    backend: TensorBackend,
) -> Callable[..., Any]:
    """Create analogical (soft probabilistic) version of operation.

    Convenience function for T>0 operations with default T=1.0.

    Args:
        operation: Logical operation function to wrap
        temperature: Temperature parameter (default: 1.0)
        backend: Tensor backend for computation

    Returns:
        Analogical (soft probabilistic) operation function

    Raises:
        ValueError: If temperature ≤ 0

    Examples:
        >>> op = analogical_operation(logical_or, temperature=1.0, backend=backend)
        >>> result = op(a, b, backend=backend)  # Soft OR with T=1.0
    """
    if temperature <= 0.0:
        raise ValueError(f"Analogical operation requires temperature > 0, got {temperature}")

    return temperature_scaled_operation(operation, temperature=temperature, backend=backend)


__all__ = [
    "temperature_scaled_operation",
    "deductive_operation",
    "analogical_operation",
]
