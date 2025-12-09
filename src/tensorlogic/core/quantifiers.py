"""Quantifier operations for TensorLogic.

Implements existential (∃) and universal (∀) quantification via tensor aggregation.
Provides both hard (boolean) and soft (differentiable) variants for flexible
reasoning modes.

Mathematical Foundations:
- ∃x.P(x) = step(Σ_x P(x))  [hard]
- ∀x.P(x) = step(Π_x P(x) - ε)  [hard, ε = numerical threshold]
- Soft ∃: max_x P(x)  [differentiable]
- Soft ∀: min_x P(x)  [differentiable]
"""

from __future__ import annotations

from typing import Any

from tensorlogic.backends import TensorBackend
from tensorlogic.core.operations import step


def exists(
    predicate: Any,
    axis: int | tuple[int, ...] | None = None,
    *,
    backend: TensorBackend,
) -> Any:
    """Existential quantification via summation + step.

    Mathematical formulation: ∃x.P(x) = step(Σ_x P(x))

    Returns 1.0 if at least one element is true (>0) along the specified axis/axes.
    This is the "hard" variant that produces boolean outputs.

    Args:
        predicate: Boolean tensor over domain (values in [0, 1])
        axis: Axis or axes to quantify over. If None, quantifies over all axes.
        backend: Tensor backend for computation

    Returns:
        Boolean tensor: 1.0 if at least one true along axis, 0.0 otherwise

    Examples:
        >>> # Single axis: ∃x.P(x) where P = [0, 0, 1, 0]
        >>> exists([0.0, 0.0, 1.0, 0.0], axis=0, backend=backend)
        1.0  # At least one true

        >>> # Multi-axis: ∃x,y.P(x,y)
        >>> exists([[0, 1], [0, 0]], axis=(0, 1), backend=backend)
        1.0  # At least one true in entire matrix

        >>> # All false
        >>> exists([0.0, 0.0, 0.0], axis=0, backend=backend)
        0.0  # No true values
    """
    # Sum over the specified axes
    summation = backend.sum(predicate, axis=axis)

    # Apply step function: step(sum) = 1.0 if sum > 0, else 0.0
    return step(summation, backend=backend)


def forall(
    predicate: Any,
    axis: int | tuple[int, ...] | None = None,
    *,
    backend: TensorBackend,
) -> Any:
    """Universal quantification via product + step.

    Mathematical formulation: ∀x.P(x) = step(Π_x P(x) - ε)

    Returns 1.0 if all elements are true (>0) along the specified axis/axes.
    This is the "hard" variant that produces boolean outputs.

    For numerical stability with products, we use the threshold ε = 0.5.
    This ensures: product ≥ 1.0 only when all inputs are 1.0 (since inputs ∈ [0,1]).

    Args:
        predicate: Boolean tensor over domain (values in [0, 1])
        axis: Axis or axes to quantify over. If None, quantifies over all axes.
        backend: Tensor backend for computation

    Returns:
        Boolean tensor: 1.0 if all true along axis, 0.0 otherwise

    Examples:
        >>> # All true: ∀x.P(x) where P = [1, 1, 1, 1]
        >>> forall([1.0, 1.0, 1.0, 1.0], axis=0, backend=backend)
        1.0

        >>> # One false: ∀x.P(x) where P = [1, 1, 0, 1]
        >>> forall([1.0, 1.0, 0.0, 1.0], axis=0, backend=backend)
        0.0  # Not all true

        >>> # Multi-axis: ∀x,y.P(x,y)
        >>> forall([[1, 1], [1, 1]], axis=(0, 1), backend=backend)
        1.0  # All true
    """
    # Product over the specified axes
    product = backend.prod(predicate, axis=axis)

    # Subtract threshold (0.5) for numerical stability
    # For boolean inputs: product = 1.0 iff all inputs are 1.0
    # product - 0.5 > 0 ⟺ product ≥ 0.5 (approximately ≥ 1.0 for booleans)
    product_minus_threshold = backend.subtract(product, 0.5)

    # Apply step function: step(product - 0.5)
    return step(product_minus_threshold, backend=backend)


def soft_exists(
    predicate: Any,
    axis: int | tuple[int, ...] | None = None,
    *,
    backend: TensorBackend,
) -> Any:
    """Soft existential quantification via maximum.

    Mathematical formulation: soft-∃x.P(x) = max_x P(x)

    Returns the maximum value along the specified axis/axes. This is the "soft"
    (differentiable) variant that preserves gradient information.

    Unlike `exists()`, this does not apply a step function, so outputs are
    continuous in [0, 1] rather than binary {0, 1}.

    Args:
        predicate: Boolean tensor over domain (values in [0, 1])
        axis: Axis or axes to take maximum over. If None, takes maximum over all axes.
        backend: Tensor backend for computation

    Returns:
        Soft boolean tensor: max value ∈ [0, 1] along axis

    Examples:
        >>> # Returns maximum confidence
        >>> soft_exists([0.0, 0.3, 0.7, 0.2], axis=0, backend=backend)
        0.7  # Maximum value (differentiable)

        >>> # Multi-axis maximum
        >>> soft_exists([[0.1, 0.5], [0.3, 0.2]], axis=(0, 1), backend=backend)
        0.5  # Global maximum
    """
    return backend.max(predicate, axis=axis)


def soft_forall(
    predicate: Any,
    axis: int | tuple[int, ...] | None = None,
    *,
    backend: TensorBackend,
) -> Any:
    """Soft universal quantification via minimum.

    Mathematical formulation: soft-∀x.P(x) = min_x P(x)

    Returns the minimum value along the specified axis/axes. This is the "soft"
    (differentiable) variant that preserves gradient information.

    Unlike `forall()`, this does not apply a step function, so outputs are
    continuous in [0, 1] rather than binary {0, 1}.

    Args:
        predicate: Boolean tensor over domain (values in [0, 1])
        axis: Axis or axes to take minimum over. If None, takes minimum over all axes.
        backend: Tensor backend for computation

    Returns:
        Soft boolean tensor: min value ∈ [0, 1] along axis

    Examples:
        >>> # Returns minimum confidence (weakest link)
        >>> soft_forall([1.0, 0.9, 0.7, 1.0], axis=0, backend=backend)
        0.7  # Minimum value (differentiable)

        >>> # Multi-axis minimum
        >>> soft_forall([[0.9, 0.8], [1.0, 0.7]], axis=(0, 1), backend=backend)
        0.7  # Global minimum
    """
    return backend.min(predicate, axis=axis)
