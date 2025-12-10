"""Product fuzzy logic compilation strategy.

Implements product fuzzy semantics using product t-norm and probabilistic
t-conorm. This strategy is mathematically equivalent to SoftDifferentiableStrategy
and provides an alternative naming convention emphasizing fuzzy logic terminology.

Mathematical Semantics:
    - AND: a * b (product t-norm)
    - OR: a + b - a*b (probabilistic t-conorm)
    - NOT: 1 - a (standard fuzzy complement)
    - IMPLIES: max(1 - a, b) (residual implication)
    - EXISTS: max(predicate, axis) (supremum)
    - FORALL: min(predicate, axis) (infimum)

Key Properties:
    - Fully differentiable: All operations support gradient computation
    - Probabilistic interpretation: Values represent truth probabilities
    - Product algebra: AND implemented as multiplication
    - Continuous: No discontinuities in operations
"""

from __future__ import annotations

from typing import Any

from tensorlogic.backends import TensorBackend


class ProductStrategy:
    """Product fuzzy semantics (fully differentiable).

    This strategy implements product fuzzy logic using multiplication for AND
    and probabilistic sum for OR. It is mathematically equivalent to the soft
    differentiable strategy but uses fuzzy logic terminology.

    Design Philosophy:
        - Full differentiability: All operations support gradient computation
        - Product algebra: AND as independent probability multiplication
        - Probabilistic t-conorm: OR follows inclusion-exclusion principle
        - Fuzzy semantics: Continuous truth values in [0, 1]

    Args:
        backend: TensorBackend instance for tensor operations

    Example:
        >>> from tensorlogic.backends import create_backend
        >>> backend = create_backend("mlx")
        >>> strategy = ProductStrategy(backend)
        >>>
        >>> # Product AND (multiplication)
        >>> a = backend.array([0.8, 0.6, 0.3])
        >>> b = backend.array([0.9, 0.4, 0.7])
        >>> result = strategy.compile_and(a, b)  # [0.72, 0.24, 0.21]
    """

    def __init__(self, backend: TensorBackend | None = None) -> None:
        """Initialize product fuzzy strategy.

        Args:
            backend: TensorBackend instance for tensor operations.
                    If None, creates a default NumPy backend for compatibility.
        """
        if backend is None:
            from tensorlogic.backends import create_backend

            backend = create_backend("numpy")
        self._backend = backend

    def compile_and(self, a: Any, b: Any) -> Any:
        """Compile logical AND using product t-norm (a * b).

        Product AND interprets conjunction as independent probability
        multiplication. This is the standard product t-norm in fuzzy logic.

        Mathematical Properties:
            - Commutative: AND(a, b) = AND(b, a)
            - Associative: AND(AND(a, b), c) = AND(a, AND(b, c))
            - Identity: AND(a, 1) = a
            - Annihilator: AND(a, 0) = 0
            - Differentiable: ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
            - Monotonic: If a ≤ a' and b ≤ b', then AND(a,b) ≤ AND(a',b')

        Args:
            a: First input tensor (values in [0, 1])
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor representing product AND(a, b) = a * b

        Example:
            >>> result = strategy.compile_and(0.8, 0.9)  # 0.72
            >>> result = strategy.compile_and([0.5, 0.7], [0.6, 0.8])  # [0.30, 0.56]
        """
        return self._backend.multiply(a, b)

    def compile_or(self, a: Any, b: Any) -> Any:
        """Compile logical OR using probabilistic sum (a + b - a*b).

        Product OR uses the probabilistic t-conorm following the
        inclusion-exclusion principle: P(A ∪ B) = P(A) + P(B) - P(A ∩ B).

        Mathematical Properties:
            - Commutative: OR(a, b) = OR(b, a)
            - Associative: OR(OR(a, b), c) = OR(a, OR(b, c))
            - Identity: OR(a, 0) = a
            - Annihilator: OR(a, 1) = 1
            - Differentiable: ∂(a+b-a*b)/∂a = 1-b, ∂(a+b-a*b)/∂b = 1-a
            - Monotonic: If a ≤ a' and b ≤ b', then OR(a,b) ≤ OR(a',b')

        Args:
            a: First input tensor (values in [0, 1])
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor representing product OR(a, b) = a + b - a*b

        Example:
            >>> result = strategy.compile_or(0.3, 0.4)  # 0.58
            >>> result = strategy.compile_or([0.2, 0.5], [0.3, 0.6])  # [0.44, 0.80]
        """
        return self._backend.subtract(
            self._backend.add(a, b), self._backend.multiply(a, b)
        )

    def compile_not(self, a: Any) -> Any:
        """Compile logical NOT using standard fuzzy complement (1 - a).

        Product NOT uses the standard fuzzy negation. This is identical across
        most fuzzy logic systems.

        Mathematical Properties:
            - Involution: NOT(NOT(a)) = a
            - Fixed points: NOT(0.5) = 0.5
            - Boundary: NOT(0) = 1, NOT(1) = 0
            - Differentiable: ∂(1-a)/∂a = -1

        Args:
            a: Input tensor (values in [0, 1])

        Returns:
            Tensor representing product NOT(a) = 1 - a

        Example:
            >>> result = strategy.compile_not(0.8)  # 0.2
            >>> result = strategy.compile_not([0.3, 0.7, 0.5])  # [0.7, 0.3, 0.5]
        """
        return self._backend.subtract(1, a)

    def compile_implies(self, a: Any, b: Any) -> Any:
        """Compile logical implication using max(1 - a, b).

        Product IMPLIES uses the residual implication derived from the
        product t-norm. This is implemented as max(1 - a, b) which
        approximates (a → b) ≡ (¬a ∨ b).

        Mathematical Properties:
            - Boundary: IMPLIES(0, b) = 1, IMPLIES(a, 1) = 1
            - Not transitive: a→b, b→c does NOT imply a→c in general
            - Differentiable: Subgradients exist at max discontinuity

        Args:
            a: Premise tensor (values in [0, 1])
            b: Consequent tensor (must be broadcastable with a)

        Returns:
            Tensor representing product (a → b) = max(1 - a, b)

        Example:
            >>> result = strategy.compile_implies(0.9, 0.3)  # 0.3
            >>> result = strategy.compile_implies(0.2, 0.7)  # 0.8
        """
        not_a = self.compile_not(a)
        return self._backend.maximum(not_a, b)

    def compile_exists(self, predicate: Any, axis: int) -> Any:
        """Compile existential quantifier (∃) using max over axis.

        Product EXISTS uses supremum (maximum) for existential quantification,
        identical to Gödel strategy.

        Mathematical Properties:
            - Monotonic: If predicate increases, EXISTS increases
            - Idempotent: EXISTS(EXISTS(P)) = EXISTS(P) for same axis
            - Differentiable: Subgradients route to maximal element

        Args:
            predicate: Input tensor (values in [0, 1])
            axis: Axis to quantify over (reduced dimension)

        Returns:
            Tensor with specified axis reduced via product EXISTS = max

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

        Product FORALL uses infimum (minimum) for universal quantification,
        identical to Gödel strategy.

        Mathematical Properties:
            - Monotonic: If predicate increases, FORALL increases
            - Idempotent: FORALL(FORALL(P)) = FORALL(P) for same axis
            - Differentiable: Subgradients route to minimal element

        Args:
            predicate: Input tensor (values in [0, 1])
            axis: Axis to quantify over (reduced dimension)

        Returns:
            Tensor with specified axis reduced via product FORALL = min

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

        Product fuzzy strategy is fully differentiable. All operations
        maintain gradient flow for backpropagation-based training.

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
            "product"

        Example:
            >>> assert strategy.name == "product"
        """
        return "product"


__all__ = ["ProductStrategy"]
