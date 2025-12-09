"""Hard boolean compilation strategy.

Implements exact discrete boolean semantics using step functions. This is the
non-differentiable strategy designed for production inference where exact boolean
logic is required with zero hallucinations.

Mathematical Semantics:
    - AND: step(a * b) (product then threshold)
    - OR: step(a + b) (sum then threshold)
    - NOT: 1 - step(a) (complement after threshold)
    - IMPLIES: step((1 - step(a)) + step(b)) (boolean implication)
    - EXISTS: step(sum(predicate, axis)) (any element true)
    - FORALL: step(prod(predicate, axis) - 0.99) (all elements near 1.0)

Key Properties:
    - Non-differentiable: Gradients are zero almost everywhere
    - Exact logic: Produces only {0, 1} outputs
    - Zero hallucinations: No intermediate probability values
    - Production-ready: Suitable for deployment where exact logic required
"""

from __future__ import annotations

from typing import Any

from tensorlogic.backends import TensorBackend


class HardBooleanStrategy:
    """Hard boolean semantics (non-differentiable).

    This strategy implements exact discrete boolean logic using Heaviside step
    functions. All operations produce binary {0, 1} outputs, making it suitable
    for production inference where probabilistic outputs are unacceptable.

    Design Philosophy:
        - Exact boolean semantics: No soft probabilities
        - Zero hallucinations: Crisp binary decisions
        - Non-differentiable: Not suitable for gradient-based training
        - Production inference: Deploy where correctness > learning

    Args:
        backend: TensorBackend instance for tensor operations

    Example:
        >>> from tensorlogic.backends import create_backend
        >>> backend = create_backend("mlx")
        >>> strategy = HardBooleanStrategy(backend)
        >>>
        >>> # Hard AND (exact boolean)
        >>> a = backend.array([0.8, 0.6, 0.3])
        >>> b = backend.array([0.9, 0.4, 0.7])
        >>> result = strategy.compile_and(a, b)  # [1.0, 0.0, 0.0]

    Warning:
        This strategy is NOT differentiable. Attempting to compute gradients
        will result in zero gradients almost everywhere due to step function
        discontinuities. Use SoftDifferentiableStrategy for training.
    """

    def __init__(self, backend: TensorBackend | None = None) -> None:
        """Initialize hard boolean strategy.

        Args:
            backend: TensorBackend instance for tensor operations.
                    If None, creates a default NumPy backend for compatibility.
        """
        if backend is None:
            from tensorlogic.backends import create_backend

            backend = create_backend("numpy")
        self._backend = backend

    def compile_and(self, a: Any, b: Any) -> Any:
        """Compile logical AND using step(a * b).

        Hard AND first multiplies inputs, then applies step function to produce
        binary output. This implements exact boolean conjunction where both
        inputs must be "truthy" (> 0 after multiplication).

        Mathematical Properties:
            - Commutative: AND(a, b) = AND(b, a)
            - Associative: AND(AND(a, b), c) = AND(a, AND(b, c))
            - Identity: AND(a, 1) = step(a)
            - Annihilator: AND(a, 0) = 0
            - Non-differentiable: Gradient zero almost everywhere

        Args:
            a: First input tensor (values interpreted as boolean)
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Binary tensor {0, 1} representing hard AND(a, b) = step(a * b)

        Example:
            >>> result = strategy.compile_and(0.8, 0.9)  # 1.0 (both > 0)
            >>> result = strategy.compile_and([0.5, 0.0], [0.6, 0.8])  # [1.0, 0.0]
        """
        product = self._backend.multiply(a, b)
        return self._backend.step(product)

    def compile_or(self, a: Any, b: Any) -> Any:
        """Compile logical OR using step(a + b).

        Hard OR sums inputs, then applies step function to produce binary output.
        This implements exact boolean disjunction where at least one input must
        be "truthy" (> 0).

        Mathematical Properties:
            - Commutative: OR(a, b) = OR(b, a)
            - Associative: OR(OR(a, b), c) = OR(a, OR(b, c))
            - Identity: OR(a, 0) = step(a)
            - Annihilator: OR(a, 1) = 1
            - Non-differentiable: Gradient zero almost everywhere

        Args:
            a: First input tensor (values interpreted as boolean)
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Binary tensor {0, 1} representing hard OR(a, b) = step(a + b)

        Example:
            >>> result = strategy.compile_or(0.3, 0.4)  # 1.0 (sum > 0)
            >>> result = strategy.compile_or([0.0, 0.5], [0.0, 0.6])  # [0.0, 1.0]
        """
        sum_result = self._backend.add(a, b)
        return self._backend.step(sum_result)

    def compile_not(self, a: Any) -> Any:
        """Compile logical NOT using 1 - step(a).

        Hard NOT first applies step function to binarize input, then computes
        complement. This implements exact boolean negation.

        Mathematical Properties:
            - Involution: NOT(NOT(a)) = step(a)
            - Fixed point: NOT(0) = 1, NOT(positive) = 0
            - Non-differentiable: Gradient zero almost everywhere

        Args:
            a: Input tensor (values interpreted as boolean)

        Returns:
            Binary tensor {0, 1} representing hard NOT(a) = 1 - step(a)

        Example:
            >>> result = strategy.compile_not(0.8)  # 0.0
            >>> result = strategy.compile_not([0.0, 0.3, -0.5])  # [1.0, 0.0, 1.0]
        """
        stepped = self._backend.step(a)
        return self._backend.subtract(1, stepped)

    def compile_implies(self, a: Any, b: Any) -> Any:
        """Compile logical implication (a → b) using step((1 - step(a)) + step(b)).

        Hard IMPLIES uses the boolean equivalence (a → b) ≡ (¬a ∨ b).
        First binarizes both inputs, then implements exact boolean implication.

        Mathematical Properties:
            - NOT transitive in general boolean logic
            - Boundary: IMPLIES(1, 0) = 0, IMPLIES(0, anything) = 1
            - Non-differentiable: Gradient zero almost everywhere

        Args:
            a: Premise tensor (values interpreted as boolean)
            b: Consequent tensor (must be broadcastable with a)

        Returns:
            Binary tensor {0, 1} representing hard (a → b)

        Example:
            >>> result = strategy.compile_implies(0.9, 0.3)  # 0.0 (1 → 0 = 0)
            >>> result = strategy.compile_implies(0.2, 0.7)  # 1.0 (0 → 1 = 1)
        """
        not_a = self.compile_not(a)
        stepped_b = self._backend.step(b)
        return self._backend.step(self._backend.add(not_a, stepped_b))

    def compile_exists(self, predicate: Any, axis: int) -> Any:
        """Compile existential quantifier (∃) using step(sum(predicate, axis)).

        Hard EXISTS sums over the specified axis, then applies step function.
        This implements exact boolean "any" semantics - true if at least one
        element is truthy.

        Mathematical Properties:
            - Monotonic: If any element > 0, result = 1
            - Exact semantics: Matches boolean "any" operation
            - Non-differentiable: Gradient zero almost everywhere

        Args:
            predicate: Input tensor (values interpreted as boolean)
            axis: Axis to quantify over (reduced dimension)

        Returns:
            Binary tensor {0, 1} with axis reduced via hard EXISTS

        Example:
            >>> # Batch of 3 examples, check if any satisfies predicate
            >>> pred = backend.array([[0.0, 0.0, 0.0],
            ...                        [0.0, 0.4, 0.0],
            ...                        [0.9, 0.7, 0.6]])
            >>> result = strategy.compile_exists(pred, axis=1)  # [0.0, 1.0, 1.0]
        """
        sum_result = self._backend.sum(predicate, axis=axis)
        return self._backend.step(sum_result)

    def compile_forall(self, predicate: Any, axis: int) -> Any:
        """Compile universal quantifier (∀) using step(prod(predicate, axis) - 0.99).

        Hard FORALL computes product over the specified axis, then checks if
        the product is near 1.0 (> 0.99). This implements approximate boolean
        "all" semantics - true if all elements are close to 1.0.

        Mathematical Properties:
            - Threshold-based: Product must exceed 0.99 to be true
            - Approximate: Due to floating-point precision
            - Non-differentiable: Gradient zero almost everywhere

        Design Note:
            The 0.99 threshold accounts for floating-point precision issues.
            For exact boolean inputs {0, 1}, product will be exactly 1.0 when
            all elements are 1.0, safely exceeding the 0.99 threshold.

        Args:
            predicate: Input tensor (values interpreted as boolean)
            axis: Axis to quantify over (reduced dimension)

        Returns:
            Binary tensor {0, 1} with axis reduced via hard FORALL

        Example:
            >>> # Batch of 3 examples, check if all satisfy predicate
            >>> pred = backend.array([[0.9, 0.8, 0.7],
            ...                        [1.0, 1.0, 1.0],
            ...                        [0.3, 0.8, 0.4]])
            >>> result = strategy.compile_forall(pred, axis=1)  # [0.0, 1.0, 0.0]
        """
        prod_result = self._backend.prod(predicate, axis=axis)
        threshold = self._backend.subtract(prod_result, 0.99)
        return self._backend.step(threshold)

    @property
    def is_differentiable(self) -> bool:
        """Whether strategy supports gradient computation.

        Hard boolean strategy uses step functions which have zero gradients
        almost everywhere. This makes it unsuitable for gradient-based training
        but perfect for production inference where exact logic is required.

        Returns:
            False (strategy is non-differentiable)

        Example:
            >>> assert not strategy.is_differentiable
            >>> # Do NOT use for training - gradients will be zero
        """
        return False

    @property
    def name(self) -> str:
        """Strategy identifier.

        Returns:
            "hard_boolean"

        Example:
            >>> assert strategy.name == "hard_boolean"
        """
        return "hard_boolean"


__all__ = ["HardBooleanStrategy"]
