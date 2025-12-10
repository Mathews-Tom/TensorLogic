"""Logical operations as tensor primitives.

This module implements fundamental logical operations using tensor operations
following the mathematical equivalences from Domingos' Tensor Logic paper:
- Logical AND: a ∧ b = a ⊙ b (Hadamard product)
- Logical OR: a ∨ b = max(a, b)
- Logical NOT: ¬a = 1 - a
- Logical IMPLIES: a → b = max(1-a, b)
- Step function: step(x) = 1 if x > 0 else 0 (Heaviside)

All operations use the TensorBackend protocol abstraction for backend-agnostic
implementation across MLX, NumPy, and future backends.
"""

from __future__ import annotations

from typing import Any

from tensorlogic.backends import TensorBackend


def logical_and(a: Any, b: Any, *, backend: TensorBackend) -> Any:
    """Logical AND via Hadamard product.

    Implements logical conjunction using element-wise multiplication.
    For boolean tensors (values in {0.0, 1.0}), this correctly implements
    the AND truth table: 1.0 ∧ 1.0 = 1.0, all other combinations = 0.0.

    Mathematical formulation:
        a ∧ b = a ⊙ b (element-wise multiply)

    Truth table:
        | a   | b   | a ∧ b |
        |-----|-----|-------|
        | 0.0 | 0.0 | 0.0   |
        | 0.0 | 1.0 | 0.0   |
        | 1.0 | 0.0 | 0.0   |
        | 1.0 | 1.0 | 1.0   |

    Properties:
        - Commutative: a ∧ b = b ∧ a
        - Associative: (a ∧ b) ∧ c = a ∧ (b ∧ c)
        - Idempotent: a ∧ a = a
        - Identity: a ∧ 1 = a
        - Annihilator: a ∧ 0 = 0

    Args:
        a: Boolean tensor (values in {0.0, 1.0})
        b: Boolean tensor (same shape as a, or broadcastable)
        backend: Tensor backend for operations

    Returns:
        Boolean tensor: 1.0 where both inputs are 1.0, 0.0 otherwise

    Raises:
        ValueError: If shapes are not broadcastable

    Examples:
        >>> import numpy as np
        >>> from tensorlogic.backends import create_backend
        >>> backend = create_backend("numpy")
        >>> a = np.array([1.0, 1.0, 0.0, 0.0])
        >>> b = np.array([1.0, 0.0, 1.0, 0.0])
        >>> result = logical_and(a, b, backend=backend)
        >>> result
        array([1., 0., 0., 0.])
    """
    return backend.multiply(a, b)


def logical_or(a: Any, b: Any, *, backend: TensorBackend) -> Any:
    """Logical OR via element-wise maximum.

    Implements logical disjunction using element-wise maximum operation.
    For boolean tensors (values in {0.0, 1.0}), this correctly implements
    the OR truth table: max(a, b) = 1.0 if either is 1.0, else 0.0.

    Mathematical formulation:
        a ∨ b = max(a, b)

    Truth table:
        | a   | b   | a ∨ b |
        |-----|-----|-------|
        | 0.0 | 0.0 | 0.0   |
        | 0.0 | 1.0 | 1.0   |
        | 1.0 | 0.0 | 1.0   |
        | 1.0 | 1.0 | 1.0   |

    Properties:
        - Commutative: a ∨ b = b ∨ a
        - Associative: (a ∨ b) ∨ c = a ∨ (b ∨ c)
        - Idempotent: a ∨ a = a
        - Identity: a ∨ 0 = a
        - Annihilator: a ∨ 1 = 1

    Args:
        a: Boolean tensor (values in {0.0, 1.0})
        b: Boolean tensor (same shape as a, or broadcastable)
        backend: Tensor backend for operations

    Returns:
        Boolean tensor: 1.0 where at least one input is 1.0, 0.0 otherwise

    Raises:
        ValueError: If shapes are not broadcastable

    Examples:
        >>> import numpy as np
        >>> from tensorlogic.backends import create_backend
        >>> backend = create_backend("numpy")
        >>> a = np.array([1.0, 1.0, 0.0, 0.0])
        >>> b = np.array([1.0, 0.0, 1.0, 0.0])
        >>> result = logical_or(a, b, backend=backend)
        >>> result
        array([1., 1., 1., 0.])
    """
    return backend.maximum(a, b)


def logical_not(a: Any, *, backend: TensorBackend) -> Any:
    """Logical NOT via complement.

    Implements logical negation using complement operation (1 - a).
    For boolean tensors (values in {0.0, 1.0}), this correctly implements
    the NOT truth table: ¬1.0 = 0.0, ¬0.0 = 1.0.

    Mathematical formulation:
        ¬a = 1 - a

    Truth table:
        | a   | ¬a  |
        |-----|-----|
        | 0.0 | 1.0 |
        | 1.0 | 0.0 |

    Properties:
        - Involution (double negation): ¬¬a = a
        - Self-dual: ¬(¬a) = a
        - De Morgan's laws (with AND/OR):
            ¬(a ∧ b) = ¬a ∨ ¬b
            ¬(a ∨ b) = ¬a ∧ ¬b

    Args:
        a: Boolean tensor (values in {0.0, 1.0})
        backend: Tensor backend for operations

    Returns:
        Boolean tensor: 1.0 where input is 0.0, 0.0 where input is 1.0

    Examples:
        >>> import numpy as np
        >>> from tensorlogic.backends import create_backend
        >>> backend = create_backend("numpy")
        >>> a = np.array([1.0, 0.0, 1.0, 0.0])
        >>> result = logical_not(a, backend=backend)
        >>> result
        array([0., 1., 0., 1.])
    """
    # Use Python scalar 1 directly - backends handle scalar broadcasting efficiently
    return backend.subtract(1, a)


def logical_implies(a: Any, b: Any, *, backend: TensorBackend) -> Any:
    """Logical implication via max(1-a, b).

    Implements logical implication using the mathematical equivalence:
    a → b = ¬a ∨ b = max(1-a, b)

    For boolean tensors (values in {0.0, 1.0}), this correctly implements
    the implication truth table: returns 0.0 only when a is 1.0 and b is 0.0.

    Mathematical formulation:
        a → b = ¬a ∨ b = max(1-a, b)

    Truth table:
        | a   | b   | a → b |
        |-----|-----|-------|
        | 0.0 | 0.0 | 1.0   |
        | 0.0 | 1.0 | 1.0   |
        | 1.0 | 0.0 | 0.0   |
        | 1.0 | 1.0 | 1.0   |

    Properties:
        - Contrapositive: a → b ≡ ¬b → ¬a
        - Modus ponens: (a ∧ (a → b)) → b (tautology)
        - Material implication: a → b ≡ ¬a ∨ b
        - Not commutative: a → b ≠ b → a (in general)
        - Chain rule: ((a → b) ∧ (b → c)) → (a → c) (tautology)

    Args:
        a: Boolean tensor (antecedent, values in {0.0, 1.0})
        b: Boolean tensor (consequent, same shape as a, or broadcastable)
        backend: Tensor backend for operations

    Returns:
        Boolean tensor: 0.0 only where a is 1.0 and b is 0.0, 1.0 otherwise

    Raises:
        ValueError: If shapes are not broadcastable

    Examples:
        >>> import numpy as np
        >>> from tensorlogic.backends import create_backend
        >>> backend = create_backend("numpy")
        >>> a = np.array([1.0, 1.0, 0.0, 0.0])
        >>> b = np.array([1.0, 0.0, 1.0, 0.0])
        >>> result = logical_implies(a, b, backend=backend)
        >>> result
        array([1., 0., 1., 1.])
    """
    # Implementation: a → b = ¬a ∨ b = max(1-a, b)
    not_a = logical_not(a, backend=backend)
    return logical_or(not_a, b, backend=backend)


def step(x: Any, *, backend: TensorBackend) -> Any:
    """Heaviside step function for boolean conversion.

    Converts continuous values to discrete boolean {0.0, 1.0} values using
    the Heaviside step function. Critical for quantifiers and hard logical
    reasoning (deductive mode with temperature T=0).

    Mathematical formulation:
        step(x) = 1.0 if x > 0
                  0.0 otherwise

    Edge case handling:
        - x = 0      → 0.0 (boundary convention)
        - x = NaN    → 0.0 (treat invalid as false)
        - x = +inf   → 1.0 (positive infinity is positive)
        - x = -inf   → 0.0 (negative infinity is negative)

    Properties:
        - Discontinuous at x = 0
        - Non-differentiable (use sigmoid approximation for gradient-based learning)
        - Idempotent: step(step(x)) = step(x)

    Args:
        x: Input tensor (any real values)
        backend: Tensor backend for operations

    Returns:
        Boolean tensor with values in {0.0, 1.0}

    Examples:
        >>> import numpy as np
        >>> from tensorlogic.backends import create_backend
        >>> backend = create_backend("numpy")
        >>> x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        >>> result = step(x, backend=backend)
        >>> result
        array([0., 0., 0., 1., 1.])

    Note:
        Critical for converting continuous values to discrete boolean.
        Used in quantifiers (EXISTS, FORALL) and rule application.
    """
    return backend.step(x)


__all__ = [
    "logical_and",
    "logical_or",
    "logical_not",
    "logical_implies",
    "step",
]
