"""Soft differentiable compilation strategy.

Implements continuous, gradient-friendly logical operations using probabilistic
semantics. This is the default strategy for neural-symbolic training as it maintains
full differentiability for gradient-based optimization.

Mathematical Semantics:
    - AND: a * b (product)
    - OR: a + b - a*b (probabilistic sum)
    - NOT: 1 - a (complement)
    - IMPLIES: max(1 - a, b) (equivalent to OR(NOT(a), b))
    - EXISTS: max(predicate, axis) (soft maximum)
    - FORALL: min(predicate, axis) (soft minimum)
"""

from __future__ import annotations

from typing import Any

from tensorlogic.backends import TensorBackend


class SoftDifferentiableStrategy:
    """Soft probabilistic semantics (differentiable).

    This strategy implements logical operations using continuous functions that
    preserve gradients for backpropagation. All operations assume inputs are in
    the range [0, 1] representing truth probabilities.

    Design Philosophy:
        - Full differentiability: All operations support gradient computation
        - Probabilistic interpretation: Values represent truth probabilities
        - Soft extrema: Use max/min instead of hard thresholds
        - Default strategy: Recommended for neural predicate training

    Args:
        backend: TensorBackend instance for tensor operations

    Example:
        >>> from tensorlogic.backends import create_backend
        >>> backend = create_backend("mlx")
        >>> strategy = SoftDifferentiableStrategy(backend)
        >>>
        >>> # Soft AND (product)
        >>> a = backend.array([0.8, 0.6, 0.3])
        >>> b = backend.array([0.9, 0.4, 0.7])
        >>> result = strategy.compile_and(a, b)  # [0.72, 0.24, 0.21]
    """

    def __init__(self, backend: TensorBackend | None = None) -> None:
        """Initialize soft differentiable strategy.

        Args:
            backend: TensorBackend instance for tensor operations.
                    If None, creates a default NumPy backend for compatibility.
        """
        if backend is None:
            from tensorlogic.backends import create_backend

            backend = create_backend("numpy")
        self._backend = backend

    def compile_and(self, a: Any, b: Any) -> Any:
        """Compile logical AND using product (a * b).

        Soft AND interprets conjunction as independent event multiplication.
        For probabilities P(A) and P(B), P(A AND B) = P(A) * P(B) under independence.

        Mathematical Properties:
            - Commutative: AND(a, b) = AND(b, a)
            - Associative: AND(AND(a, b), c) = AND(a, AND(b, c))
            - Identity: AND(a, 1) = a
            - Annihilator: AND(a, 0) = 0
            - Differentiable: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a

        Args:
            a: First input tensor (values in [0, 1])
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor representing soft AND(a, b) = a * b

        Example:
            >>> result = strategy.compile_and(0.8, 0.9)  # 0.72
            >>> result = strategy.compile_and([0.5, 0.7], [0.6, 0.8])  # [0.30, 0.56]
        """
        return self._backend.multiply(a, b)

    def compile_or(self, a: Any, b: Any) -> Any:
        """Compile logical OR using probabilistic sum (a + b - a*b).

        Soft OR uses the inclusion-exclusion principle from probability theory.
        P(A OR B) = P(A) + P(B) - P(A AND B) = a + b - a*b.

        Mathematical Properties:
            - Commutative: OR(a, b) = OR(b, a)
            - Associative: OR(OR(a, b), c) = OR(a, OR(b, c))
            - Identity: OR(a, 0) = a
            - Annihilator: OR(a, 1) = 1
            - Differentiable: ∂(a+b-a*b)/∂a = 1-b, ∂(a+b-a*b)/∂b = 1-a

        Args:
            a: First input tensor (values in [0, 1])
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor representing soft OR(a, b) = a + b - a*b

        Example:
            >>> result = strategy.compile_or(0.3, 0.4)  # 0.58
            >>> result = strategy.compile_or([0.2, 0.5], [0.3, 0.6])  # [0.44, 0.80]
        """
        return self._backend.subtract(
            self._backend.add(a, b), self._backend.multiply(a, b)
        )

    def compile_not(self, a: Any) -> Any:
        """Compile logical NOT using complement (1 - a).

        Soft NOT interprets negation as probability complement.
        P(NOT A) = 1 - P(A).

        Mathematical Properties:
            - Involution: NOT(NOT(a)) = a
            - Fixed points: NOT(0.5) = 0.5
            - Differentiable: ∂(1-a)/∂a = -1

        Args:
            a: Input tensor (values in [0, 1])

        Returns:
            Tensor representing soft NOT(a) = 1 - a

        Example:
            >>> result = strategy.compile_not(0.8)  # 0.2
            >>> result = strategy.compile_not([0.3, 0.7, 0.5])  # [0.7, 0.3, 0.5]
        """
        # Subtract from scalar 1, relying on backend broadcasting
        return self._backend.subtract(1, a)

    def compile_implies(self, a: Any, b: Any) -> Any:
        """Compile logical implication (a → b) using max(1 - a, b).

        Soft IMPLIES uses the equivalence (a → b) ≡ (¬a ∨ b).
        Implemented as OR(NOT(a), b) = max(1 - a, b).

        Mathematical Properties:
            - NOT transitive: a→b, b→c does NOT imply a→c in soft logic
            - Differentiable: Subgradients at max discontinuity
            - Boundary: IMPLIES(1, b) = b, IMPLIES(0, b) = 1

        Args:
            a: Premise tensor (values in [0, 1])
            b: Consequent tensor (must be broadcastable with a)

        Returns:
            Tensor representing soft (a → b) = max(1 - a, b)

        Example:
            >>> result = strategy.compile_implies(0.9, 0.3)  # 0.3
            >>> result = strategy.compile_implies(0.2, 0.7)  # 0.8
        """
        not_a = self.compile_not(a)
        return self._backend.maximum(not_a, b)

    def compile_exists(self, predicate: Any, axis: int) -> Any:
        """Compile existential quantifier (∃) using max over axis.

        Soft EXISTS interprets existential quantification as taking the maximum
        truth value along the quantified dimension. This maintains differentiability
        via subgradients at the maximum.

        Mathematical Properties:
            - Monotonic: If predicate increases, EXISTS increases
            - Idempotent: EXISTS(EXISTS(P)) = EXISTS(P) for same axis
            - Differentiable: Subgradients route to maximal element

        Args:
            predicate: Input tensor (values in [0, 1])
            axis: Axis to quantify over (reduced dimension)

        Returns:
            Tensor with specified axis reduced via soft EXISTS = max

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

        Soft FORALL interprets universal quantification as taking the minimum
        truth value along the quantified dimension. This maintains differentiability
        via subgradients at the minimum.

        Mathematical Properties:
            - Monotonic: If predicate increases, FORALL increases
            - Idempotent: FORALL(FORALL(P)) = FORALL(P) for same axis
            - Differentiable: Subgradients route to minimal element

        Args:
            predicate: Input tensor (values in [0, 1])
            axis: Axis to quantify over (reduced dimension)

        Returns:
            Tensor with specified axis reduced via soft FORALL = min

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

        Soft differentiable strategy maintains full differentiability for all
        operations. Even max/min operations provide subgradients for backpropagation.

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
            "soft_differentiable"

        Example:
            >>> assert strategy.name == "soft_differentiable"
        """
        return "soft_differentiable"


__all__ = ["SoftDifferentiableStrategy"]
