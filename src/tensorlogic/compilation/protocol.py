"""CompilationStrategy Protocol defining logical operation compilation interface.

This module defines a Protocol-based abstraction for compilation strategies that
transform logical operations into tensor operations. Multiple semantic interpretations
are supported: soft differentiable (product), hard boolean (step), and fuzzy variants
(Gödel, product, Łukasiewicz).

Design Philosophy:
    - Zero vendor lock-in: Switch strategies without code changes
    - Strategy pattern: Different semantic interpretations of same logical operations
    - Differentiability tracking: Explicit support for gradient-based training
    - Minimal abstraction: 6 core operations + 2 properties
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CompilationStrategy(Protocol):
    """Protocol defining logical operation compilation interface for strategy abstraction.

    This Protocol defines 6 core logical operations that can be compiled with different
    semantic interpretations (soft probabilistic, hard boolean, fuzzy logic variants).
    Strategies implement these methods using appropriate mathematical formulations.

    Design Philosophy:
        - Abstract at operation level, not model level
        - Support multiple semantic interpretations (boolean, fuzzy, differentiable)
        - Explicit differentiability tracking for gradient-based training
        - Structural typing via Protocol (no inheritance required)

    Implementation Requirements:
        - All 6 logical operations must be implemented by concrete strategies
        - Type hints must use Array from backends (backend-agnostic)
        - Properties must indicate differentiability and strategy name
        - Mathematical semantics documented in each method

    Example:
        >>> strategy = create_strategy("soft_differentiable")
        >>> result = strategy.compile_and(a, b)  # Soft AND via product
    """

    def compile_and(self, a: Any, b: Any) -> Any:
        """Compile logical AND operation.

        Semantic interpretation varies by strategy:
            - Soft differentiable: Product (a * b)
            - Hard boolean: Step function applied to product
            - Gödel fuzzy: Minimum (min(a, b))
            - Product fuzzy: Product (a * b)
            - Łukasiewicz fuzzy: max(0, a + b - 1)

        Args:
            a: First input tensor (values in [0, 1] for fuzzy/soft semantics)
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor representing AND(a, b) according to strategy semantics

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> # Soft differentiable: AND as product
            >>> result = strategy.compile_and(a, b)  # a * b
        """
        ...

    def compile_or(self, a: Any, b: Any) -> Any:
        """Compile logical OR operation.

        Semantic interpretation varies by strategy:
            - Soft differentiable: Probabilistic sum (a + b - a*b)
            - Hard boolean: Step function applied to max
            - Gödel fuzzy: Maximum (max(a, b))
            - Product fuzzy: Probabilistic sum (a + b - a*b)
            - Łukasiewicz fuzzy: min(1, a + b)

        Args:
            a: First input tensor (values in [0, 1] for fuzzy/soft semantics)
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor representing OR(a, b) according to strategy semantics

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> # Soft differentiable: OR as probabilistic sum
            >>> result = strategy.compile_or(a, b)  # a + b - a*b
        """
        ...

    def compile_not(self, a: Any) -> Any:
        """Compile logical NOT operation.

        Semantic interpretation consistent across strategies:
            - All strategies: Complement (1 - a)

        Args:
            a: Input tensor (values in [0, 1] for fuzzy/soft semantics)

        Returns:
            Tensor representing NOT(a) = 1 - a

        Example:
            >>> # NOT as complement (universal across strategies)
            >>> result = strategy.compile_not(a)  # 1 - a
        """
        ...

    def compile_implies(self, a: Any, b: Any) -> Any:
        """Compile logical implication (a → b).

        Semantic interpretation varies by strategy:
            - Soft differentiable: max(1 - a, b)
            - Hard boolean: Step function applied to max(1 - a, b)
            - Gödel fuzzy: If a ≤ b then 1 else b (Gödel implication)
            - Product fuzzy: min(1, b / a) for a > 0 else 1
            - Łukasiewicz fuzzy: min(1, 1 - a + b)

        Mathematical Note:
            Implication is equivalent to OR(NOT(a), b) but some strategies
            use specialized formulations for numerical stability.

        Args:
            a: Premise tensor (values in [0, 1] for fuzzy/soft semantics)
            b: Consequent tensor (must be broadcastable with a)

        Returns:
            Tensor representing (a → b) according to strategy semantics

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> # Soft differentiable: implication via max
            >>> result = strategy.compile_implies(a, b)  # max(1 - a, b)
        """
        ...

    def compile_exists(self, predicate: Any, axis: int) -> Any:
        """Compile existential quantifier (∃).

        Semantic interpretation varies by strategy:
            - Soft differentiable: Maximum over axis (soft max)
            - Hard boolean: Any (boolean OR reduction)
            - Gödel fuzzy: Maximum over axis
            - Product fuzzy: Maximum over axis
            - Łukasiewicz fuzzy: Maximum over axis

        Mathematical Note:
            Existential quantification reduces over specified axis.
            Soft strategies preserve differentiability via max/sum operations.

        Args:
            predicate: Input tensor (values in [0, 1] for fuzzy/soft semantics)
            axis: Axis to quantify over (reduced dimension)

        Returns:
            Tensor with specified axis reduced via existential quantification

        Example:
            >>> # Soft differentiable: EXISTS as max
            >>> result = strategy.compile_exists(predicate, axis=1)  # max over axis 1
        """
        ...

    def compile_forall(self, predicate: Any, axis: int) -> Any:
        """Compile universal quantifier (∀).

        Semantic interpretation varies by strategy:
            - Soft differentiable: Minimum over axis (soft min)
            - Hard boolean: All (boolean AND reduction)
            - Gödel fuzzy: Minimum over axis
            - Product fuzzy: Minimum over axis
            - Łukasiewicz fuzzy: Minimum over axis

        Mathematical Note:
            Universal quantification reduces over specified axis.
            Soft strategies preserve differentiability via min/product operations.

        Args:
            predicate: Input tensor (values in [0, 1] for fuzzy/soft semantics)
            axis: Axis to quantify over (reduced dimension)

        Returns:
            Tensor with specified axis reduced via universal quantification

        Example:
            >>> # Soft differentiable: FORALL as min
            >>> result = strategy.compile_forall(predicate, axis=0)  # min over axis 0
        """
        ...

    @property
    def is_differentiable(self) -> bool:
        """Whether strategy supports gradient computation.

        Differentiability is critical for neural-symbolic training. Soft and fuzzy
        strategies typically support gradients, while hard boolean strategies do not
        (due to discontinuous step functions).

        Returns:
            True if strategy supports automatic differentiation, False otherwise

        Example:
            >>> if strategy.is_differentiable:
            ...     loss = compute_loss(strategy.compile_and(a, b))
            ...     gradients = backend.grad(loss)
        """
        ...

    @property
    def name(self) -> str:
        """Strategy identifier for debugging and logging.

        Returns:
            Human-readable strategy name (e.g., "soft_differentiable", "hard_boolean")

        Example:
            >>> print(f"Using strategy: {strategy.name}")
            Using strategy: soft_differentiable
        """
        ...


__all__ = ["CompilationStrategy"]
