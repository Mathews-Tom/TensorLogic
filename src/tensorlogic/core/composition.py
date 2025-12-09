"""Rule composition utilities for TensorLogic.

Provides high-level utilities for composing multiple logical rules via
logical operations. Enables multi-predicate rule encoding like:
    Aunt(x,z) ← Sister(x,y) ∧ Parent(y,z)

Composition supports both 'and' and 'or' operations over variable numbers
of rule inputs.
"""

from __future__ import annotations

from typing import Any

from tensorlogic.backends import TensorBackend
from tensorlogic.core.operations import logical_and, logical_or


def compose_rules(
    *rules: Any,
    operation: str = "and",
    backend: TensorBackend,
) -> Any:
    """Compose multiple logical rules via AND or OR operations.

    Takes a variable number of boolean tensors (rules) and composes them
    using the specified logical operation. All rules must have compatible
    shapes for broadcasting.

    This is a convenience utility for chaining multiple logical operations.
    For complex multi-predicate rules with index manipulation, combine this
    with backend.einsum().

    Args:
        *rules: Variable number of boolean tensors to compose (at least 1)
        operation: Composition operator - 'and' or 'or' (default: 'and')
        backend: Tensor backend for computation

    Returns:
        Composed boolean tensor

    Raises:
        ValueError: If operation not in ('and', 'or') or fewer than 1 rule
        ValueError: If rule shapes are not broadcastable

    Examples:
        >>> # Simple AND composition of 3 rules
        >>> result = compose_rules(rule1, rule2, rule3, operation="and", backend=backend)
        >>> # Equivalent to: rule1 ∧ rule2 ∧ rule3

        >>> # OR composition
        >>> result = compose_rules(rule1, rule2, operation="or", backend=backend)
        >>> # Equivalent to: rule1 ∨ rule2

        >>> # Multi-predicate rule with einsum:
        >>> # Aunt(x,z) ← Sister(x,y) ∧ Parent(y,z)
        >>> sister = backend.zeros((10, 10))  # Sister[x, y]
        >>> parent = backend.zeros((10, 10))  # Parent[y, z]
        >>> # First use einsum to align axes: xy,yz->xz
        >>> combined = backend.einsum("xy,yz->xz", sister, parent)
        >>> aunt = step(combined, backend=backend)

    Notes:
        - For complex multi-predicate rules with different arities, use
          backend.einsum() to align tensor indices before composition
        - Use step() after composition to convert continuous values to boolean
        - All rules must be broadcastable with each other
    """
    # Validate inputs
    if len(rules) == 0:
        raise ValueError("At least one rule required for composition")

    if operation not in ("and", "or"):
        raise ValueError(f"Invalid operation '{operation}'. Must be 'and' or 'or'")

    # Single rule - return as is
    if len(rules) == 1:
        return rules[0]

    # Select composition operation
    if operation == "and":
        compose_fn = logical_and
    else:  # operation == "or"
        compose_fn = logical_or

    # Iteratively compose all rules
    result = rules[0]
    for rule in rules[1:]:
        result = compose_fn(result, rule, backend=backend)

    return result


def compose_and(*rules: Any, backend: TensorBackend) -> Any:
    """Compose multiple rules with AND operation.

    Convenience function equivalent to:
        compose_rules(*rules, operation="and", backend=backend)

    Args:
        *rules: Variable number of boolean tensors to AND together
        backend: Tensor backend for computation

    Returns:
        Composed boolean tensor (r1 ∧ r2 ∧ ... ∧ rn)

    Examples:
        >>> result = compose_and(rule1, rule2, rule3, backend=backend)
        >>> # Equivalent to: rule1 ∧ rule2 ∧ rule3
    """
    return compose_rules(*rules, operation="and", backend=backend)


def compose_or(*rules: Any, backend: TensorBackend) -> Any:
    """Compose multiple rules with OR operation.

    Convenience function equivalent to:
        compose_rules(*rules, operation="or", backend=backend)

    Args:
        *rules: Variable number of boolean tensors to OR together
        backend: Tensor backend for computation

    Returns:
        Composed boolean tensor (r1 ∨ r2 ∨ ... ∨ rn)

    Examples:
        >>> result = compose_or(rule1, rule2, rule3, backend=backend)
        >>> # Equivalent to: rule1 ∨ rule2 ∨ rule3
    """
    return compose_rules(*rules, operation="or", backend=backend)
