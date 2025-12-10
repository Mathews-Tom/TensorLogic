"""Łukasiewicz fuzzy logic compilation strategy.

Implements Łukasiewicz (strict) fuzzy semantics using bounded sum and difference
operations. This strategy provides the "strictest" fuzzy logic system with strong
boundary conditions and additive structure.

Mathematical Semantics:
    - AND: max(0, a + b - 1) (bounded difference, Łukasiewicz t-norm)
    - OR: min(1, a + b) (bounded sum, Łukasiewicz t-conorm)
    - NOT: 1 - a (standard fuzzy complement)
    - IMPLIES: min(1, 1 - a + b) (Łukasiewicz implication)
    - EXISTS: max(predicate, axis) (supremum)
    - FORALL: min(predicate, axis) (infimum)

Key Properties:
    - Fully differentiable: All operations support gradient computation
    - Strict boundaries: Strong enforcement of [0, 1] range
    - Additive structure: AND/OR based on addition with bounds
    - MV-algebra: Forms a many-valued logic algebra
"""

from __future__ import annotations

from typing import Any

from tensorlogic.backends import TensorBackend


class LukasiewiczStrategy:
    """Łukasiewicz fuzzy semantics (strict, differentiable).

    This strategy implements Łukasiewicz fuzzy logic, the strictest of the
    standard fuzzy logic systems. It uses bounded sum and difference operations
    that maintain strong boundary conditions.

    Design Philosophy:
        - Strict boundaries: Strong [0, 1] range enforcement
        - Additive algebra: AND/OR based on addition with clamping
        - Full differentiability: All operations support gradient computation
        - MV-logic: Many-valued logic with involutive negation

    Args:
        backend: TensorBackend instance for tensor operations

    Example:
        >>> from tensorlogic.backends import create_backend
        >>> backend = create_backend("mlx")
        >>> strategy = LukasiewiczStrategy(backend)
        >>>
        >>> # Łukasiewicz AND (bounded difference)
        >>> a = backend.array([0.8, 0.6, 0.3])
        >>> b = backend.array([0.9, 0.4, 0.7])
        >>> result = strategy.compile_and(a, b)  # [0.7, 0.0, 0.0]
    """

    def __init__(self, backend: TensorBackend | None = None) -> None:
        """Initialize Łukasiewicz fuzzy strategy.

        Args:
            backend: TensorBackend instance for tensor operations.
                    If None, creates a default NumPy backend for compatibility.
        """
        if backend is None:
            from tensorlogic.backends import create_backend

            backend = create_backend("numpy")
        self._backend = backend

    def compile_and(self, a: Any, b: Any) -> Any:
        """Compile logical AND using Łukasiewicz t-norm: max(0, a + b - 1).

        Łukasiewicz AND uses bounded difference. This implements strict
        conjunction where both inputs must significantly support the result.
        Result is 0 unless a + b > 1.

        Mathematical Properties:
            - Commutative: AND(a, b) = AND(b, a)
            - Associative: AND(AND(a, b), c) = AND(a, AND(b, c))
            - Identity: AND(a, 1) = a
            - Annihilator: AND(a, 0) = 0
            - Strict: AND(a, b) = 0 if a + b ≤ 1
            - Differentiable: ∂max(0, a+b-1)/∂a = 1 if a+b>1, else 0
            - Monotonic: If a ≤ a' and b ≤ b', then AND(a,b) ≤ AND(a',b')

        Args:
            a: First input tensor (values in [0, 1])
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor representing Łukasiewicz AND(a, b) = max(0, a + b - 1)

        Example:
            >>> result = strategy.compile_and(0.8, 0.9)  # 0.7 (0.8+0.9-1)
            >>> result = strategy.compile_and(0.5, 0.4)  # 0.0 (0.5+0.4<1)
        """
        sum_result = self._backend.add(a, b)
        diff = self._backend.subtract(sum_result, 1.0)
        return self._backend.maximum(0.0, diff)

    def compile_or(self, a: Any, b: Any) -> Any:
        """Compile logical OR using Łukasiewicz t-conorm: min(1, a + b).

        Łukasiewicz OR uses bounded sum. This implements disjunction where
        the result saturates at 1 (unlike probabilistic sum which never
        exceeds 1 but uses different formula).

        Mathematical Properties:
            - Commutative: OR(a, b) = OR(b, a)
            - Associative: OR(OR(a, b), c) = OR(a, OR(b, c))
            - Identity: OR(a, 0) = a
            - Annihilator: OR(a, 1) = 1
            - Bounded: OR(a, b) = 1 if a + b ≥ 1
            - Differentiable: ∂min(1, a+b)/∂a = 1 if a+b<1, else 0
            - Monotonic: If a ≤ a' and b ≤ b', then OR(a,b) ≤ OR(a',b')

        Args:
            a: First input tensor (values in [0, 1])
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor representing Łukasiewicz OR(a, b) = min(1, a + b)

        Example:
            >>> result = strategy.compile_or(0.3, 0.4)  # 0.7 (0.3+0.4)
            >>> result = strategy.compile_or(0.6, 0.8)  # 1.0 (0.6+0.8>1)
        """
        sum_result = self._backend.add(a, b)
        return self._backend.minimum(1.0, sum_result)

    def compile_not(self, a: Any) -> Any:
        """Compile logical NOT using standard fuzzy complement (1 - a).

        Łukasiewicz NOT uses the standard fuzzy negation, providing an
        involutive complement.

        Mathematical Properties:
            - Involution: NOT(NOT(a)) = a
            - Fixed points: NOT(0.5) = 0.5
            - Boundary: NOT(0) = 1, NOT(1) = 0
            - Differentiable: ∂(1-a)/∂a = -1

        Args:
            a: Input tensor (values in [0, 1])

        Returns:
            Tensor representing Łukasiewicz NOT(a) = 1 - a

        Example:
            >>> result = strategy.compile_not(0.8)  # 0.2
            >>> result = strategy.compile_not([0.3, 0.7, 0.5])  # [0.7, 0.3, 0.5]
        """
        return self._backend.subtract(1, a)

    def compile_implies(self, a: Any, b: Any) -> Any:
        """Compile logical implication using Łukasiewicz implication: min(1, 1 - a + b).

        Łukasiewicz implication is the residuum of the Łukasiewicz t-norm.
        It has the property that a ⊗ c ≤ b iff c ≤ (a → b) for Łukasiewicz ⊗.

        Mathematical Properties:
            - Residuum: For Łukasiewicz AND: AND(a, c) ≤ b iff c ≤ IMPLIES(a, b)
            - Boundary: IMPLIES(0, b) = 1, IMPLIES(a, 1) = 1
            - IMPLIES(1, 0) = 0 (falsity preservation)
            - Differentiable: Subgradients exist

        Args:
            a: Premise tensor (values in [0, 1])
            b: Consequent tensor (must be broadcastable with a)

        Returns:
            Tensor representing Łukasiewicz (a → b) = min(1, 1 - a + b)

        Example:
            >>> result = strategy.compile_implies(0.9, 0.3)  # 0.4 (min(1, 0.1+0.3))
            >>> result = strategy.compile_implies(0.2, 0.7)  # 1.0 (min(1, 0.8+0.7))
        """
        not_a = self.compile_not(a)  # 1 - a
        sum_result = self._backend.add(not_a, b)  # 1 - a + b
        return self._backend.minimum(1.0, sum_result)  # min(1, 1 - a + b)

    def compile_exists(self, predicate: Any, axis: int) -> Any:
        """Compile existential quantifier (∃) using max over axis.

        Łukasiewicz EXISTS uses supremum (maximum), identical to other
        fuzzy strategies.

        Mathematical Properties:
            - Monotonic: If predicate increases, EXISTS increases
            - Idempotent: EXISTS(EXISTS(P)) = EXISTS(P) for same axis
            - Differentiable: Subgradients route to maximal element

        Args:
            predicate: Input tensor (values in [0, 1])
            axis: Axis to quantify over (reduced dimension)

        Returns:
            Tensor with specified axis reduced via Łukasiewicz EXISTS = max

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

        Łukasiewicz FORALL uses infimum (minimum), identical to other
        fuzzy strategies.

        Mathematical Properties:
            - Monotonic: If predicate increases, FORALL increases
            - Idempotent: FORALL(FORALL(P)) = FORALL(P) for same axis
            - Differentiable: Subgradients route to minimal element

        Args:
            predicate: Input tensor (values in [0, 1])
            axis: Axis to quantify over (reduced dimension)

        Returns:
            Tensor with specified axis reduced via Łukasiewicz FORALL = min

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

        Łukasiewicz fuzzy strategy is fully differentiable. All operations
        maintain gradient flow via subgradients at boundary conditions.

        Returns:
            True (all operations are differentiable)

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
            "lukasiewicz"

        Example:
            >>> assert strategy.name == "lukasiewicz"
        """
        return "lukasiewicz"


__all__ = ["LukasiewiczStrategy"]
