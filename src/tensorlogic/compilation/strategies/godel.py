"""Gödel fuzzy logic compilation strategy.

Implements Gödel fuzzy semantics using min/max t-norms. This strategy provides
a middle ground between soft probabilistic and hard boolean logic, maintaining
differentiability (via subgradients) while using simpler operations than soft logic.

Mathematical Semantics:
    - AND: min(a, b) (minimum t-norm)
    - OR: max(a, b) (maximum t-conorm)
    - NOT: 1 - a (standard fuzzy complement)
    - IMPLIES: where(a ≤ b, 1, b) (Gödel implication)
    - EXISTS: max(predicate, axis) (supremum)
    - FORALL: min(predicate, axis) (infimum)

Key Properties:
    - Differentiable: Subgradients exist for min/max operations
    - Idempotent: AND(a, a) = a, OR(a, a) = a
    - Boundary preserving: min/max preserve [0, 1] range
    - Computationally efficient: No multiplications needed for AND/OR
"""

from __future__ import annotations

from typing import Any

from tensorlogic.backends import TensorBackend


class GodelStrategy:
    """Gödel fuzzy semantics (differentiable min/max).

    This strategy implements Gödel fuzzy logic using minimum and maximum
    operations as t-norm and t-conorm. It maintains differentiability through
    subgradients while providing cleaner semantics than probabilistic approaches.

    Design Philosophy:
        - Differentiable: Subgradients enable gradient-based training
        - Idempotent: AND(a, a) = a preserves idempotency
        - Efficient: Min/max cheaper than multiplication
        - Fuzzy semantics: Continuous truth values in [0, 1]

    Args:
        backend: TensorBackend instance for tensor operations

    Example:
        >>> from tensorlogic.backends import create_backend
        >>> backend = create_backend("mlx")
        >>> strategy = GodelStrategy(backend)
        >>>
        >>> # Gödel AND (minimum)
        >>> a = backend.array([0.8, 0.6, 0.3])
        >>> b = backend.array([0.9, 0.4, 0.7])
        >>> result = strategy.compile_and(a, b)  # [0.8, 0.4, 0.3]
    """

    def __init__(self, backend: TensorBackend | None = None) -> None:
        """Initialize Gödel fuzzy strategy.

        Args:
            backend: TensorBackend instance for tensor operations.
                    If None, creates a default NumPy backend for compatibility.
        """
        if backend is None:
            from tensorlogic.backends import create_backend

            backend = create_backend("numpy")
        self._backend = backend

    def compile_and(self, a: Any, b: Any) -> Any:
        """Compile logical AND using min(a, b).

        Gödel AND uses the minimum t-norm, taking the smaller truth value.
        This implements fuzzy conjunction where the result is limited by the
        weakest input.

        Mathematical Properties:
            - Commutative: AND(a, b) = AND(b, a)
            - Associative: AND(AND(a, b), c) = AND(a, AND(b, c))
            - Idempotent: AND(a, a) = a
            - Identity: AND(a, 1) = a
            - Annihilator: AND(a, 0) = 0
            - Monotonic: If a ≤ a' and b ≤ b', then AND(a,b) ≤ AND(a',b')
            - Differentiable: Subgradients exist (gradient flows to minimum element)

        Args:
            a: First input tensor (values in [0, 1])
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor representing Gödel AND(a, b) = min(a, b)

        Example:
            >>> result = strategy.compile_and(0.8, 0.9)  # 0.8
            >>> result = strategy.compile_and([0.5, 0.7], [0.6, 0.3])  # [0.5, 0.3]
        """
        return self._backend.minimum(a, b)

    def compile_or(self, a: Any, b: Any) -> Any:
        """Compile logical OR using max(a, b).

        Gödel OR uses the maximum t-conorm, taking the larger truth value.
        This implements fuzzy disjunction where the result is determined by the
        strongest input.

        Mathematical Properties:
            - Commutative: OR(a, b) = OR(b, a)
            - Associative: OR(OR(a, b), c) = OR(a, OR(b, c))
            - Idempotent: OR(a, a) = a
            - Identity: OR(a, 0) = a
            - Annihilator: OR(a, 1) = 1
            - Monotonic: If a ≤ a' and b ≤ b', then OR(a,b) ≤ OR(a',b')
            - Differentiable: Subgradients exist (gradient flows to maximum element)

        Args:
            a: First input tensor (values in [0, 1])
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor representing Gödel OR(a, b) = max(a, b)

        Example:
            >>> result = strategy.compile_or(0.3, 0.4)  # 0.4
            >>> result = strategy.compile_or([0.2, 0.5], [0.3, 0.4])  # [0.3, 0.5]
        """
        return self._backend.maximum(a, b)

    def compile_not(self, a: Any) -> Any:
        """Compile logical NOT using 1 - a.

        Gödel NOT uses the standard fuzzy complement. This is identical to
        the soft differentiable strategy's NOT operation.

        Mathematical Properties:
            - Involution: NOT(NOT(a)) = a
            - Fixed points: NOT(0.5) = 0.5
            - Boundary: NOT(0) = 1, NOT(1) = 0
            - Differentiable: ∂(1-a)/∂a = -1

        Args:
            a: Input tensor (values in [0, 1])

        Returns:
            Tensor representing Gödel NOT(a) = 1 - a

        Example:
            >>> result = strategy.compile_not(0.8)  # 0.2
            >>> result = strategy.compile_not([0.3, 0.7, 0.5])  # [0.7, 0.3, 0.5]
        """
        return self._backend.subtract(1, a)

    def compile_implies(self, a: Any, b: Any) -> Any:
        """Compile logical implication using Gödel implication.

        Gödel implication: (a → b) = 1 if a ≤ b, else b.
        This implements residuum of the minimum t-norm.

        Mathematical Properties:
            - Residuum property: a ⊗ c ≤ b iff c ≤ (a → b)
            - Boundary: IMPLIES(0, b) = 1, IMPLIES(a, 1) = 1
            - Contraposition fails: a → b ≠ ¬b → ¬a in general
            - Differentiable: Subgradients exist

        Implementation:
            Uses where(a <= b, 1, b). Python's <= operator works with NumPy
            and MLX arrays to produce boolean masks.

        Args:
            a: Premise tensor (values in [0, 1])
            b: Consequent tensor (must be broadcastable with a)

        Returns:
            Tensor representing Gödel (a → b)

        Example:
            >>> result = strategy.compile_implies(0.9, 0.3)  # 0.3 (0.9 > 0.3)
            >>> result = strategy.compile_implies(0.2, 0.7)  # 1.0 (0.2 ≤ 0.7)
        """
        # Gödel implication: if a ≤ b then 1 else b
        # Use Python's <= which produces boolean array for NumPy/MLX
        return self._backend.where(a <= b, 1.0, b)

    def compile_exists(self, predicate: Any, axis: int) -> Any:
        """Compile existential quantifier (∃) using max over axis.

        Gödel EXISTS uses supremum (maximum) over the quantified dimension.
        This implements fuzzy existential quantification - the result is the
        maximum truth value encountered.

        Mathematical Properties:
            - Monotonic: If predicate increases, EXISTS increases
            - Idempotent: EXISTS(EXISTS(P)) = EXISTS(P) for same axis
            - Differentiable: Subgradients route to maximal element

        Args:
            predicate: Input tensor (values in [0, 1])
            axis: Axis to quantify over (reduced dimension)

        Returns:
            Tensor with specified axis reduced via Gödel EXISTS = max

        Example:
            >>> # Batch of 3 examples, check if any satisfies predicate
            >>> pred = backend.array([[0.2, 0.8, 0.3],
            ...                        [0.1, 0.4, 0.2],
            ...                        [0.9, 0.7, 0.6]])
            >>> result = strategy.compile_exists(pred, axis=1)  # [0.8, 0.4, 0.9]
        """
        return self._backend.max(predicate, axis=axis)

    def compile_forall(self, predicate: Any, axis: int) -> Any:
        """Compile universal quantifier (∀) using min over axis.

        Gödel FORALL uses infimum (minimum) over the quantified dimension.
        This implements fuzzy universal quantification - the result is the
        minimum truth value encountered (limited by the weakest element).

        Mathematical Properties:
            - Monotonic: If predicate increases, FORALL increases
            - Idempotent: FORALL(FORALL(P)) = FORALL(P) for same axis
            - Differentiable: Subgradients route to minimal element

        Args:
            predicate: Input tensor (values in [0, 1])
            axis: Axis to quantify over (reduced dimension)

        Returns:
            Tensor with specified axis reduced via Gödel FORALL = min

        Example:
            >>> # Batch of 3 examples, check if all satisfy predicate
            >>> pred = backend.array([[0.9, 0.8, 0.7],
            ...                        [0.6, 0.5, 0.9],
            ...                        [0.3, 0.8, 0.4]])
            >>> result = strategy.compile_forall(pred, axis=1)  # [0.7, 0.5, 0.3]
        """
        return self._backend.min(predicate, axis=axis)

    @property
    def is_differentiable(self) -> bool:
        """Whether strategy supports gradient computation.

        Gödel fuzzy strategy is differentiable via subgradients. While min/max
        operations have discontinuous derivatives, they provide valid subgradients
        that enable gradient-based training.

        Returns:
            True (strategy supports gradient computation via subgradients)

        Example:
            >>> assert strategy.is_differentiable
            >>> # Safe to use with gradient-based training
            >>> loss = compute_loss(strategy.compile_and(a, b))
            >>> gradients = backend.grad(loss)
        """
        return True

    @property
    def name(self) -> str:
        """Strategy identifier.

        Returns:
            "godel"

        Example:
            >>> assert strategy.name == "godel"
        """
        return "godel"


__all__ = ["GodelStrategy"]
