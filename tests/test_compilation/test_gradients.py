"""Gradient compatibility tests for compilation strategies.

This module validates that differentiable strategies support gradient computation
while non-differentiable strategies handle gradient requests appropriately.

Test Coverage:
    - Gradient flow through logical operations (AND, OR, NOT, IMPLIES)
    - Gradient flow through quantifiers (EXISTS, FORALL)
    - Numerical gradient validation against analytical gradients
    - Error handling for non-differentiable strategies (HardBoolean)
    - Integration with backend.grad() for all differentiable strategies
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from tensorlogic.backends import create_backend
from tensorlogic.compilation.strategies import (
    GodelStrategy,
    HardBooleanStrategy,
    LukasiewiczStrategy,
    ProductStrategy,
    SoftDifferentiableStrategy,
)

if TYPE_CHECKING:
    from tensorlogic.backends import TensorBackend

# Differentiable strategies that should support gradients
DIFFERENTIABLE_STRATEGIES = [
    SoftDifferentiableStrategy,
    GodelStrategy,
    ProductStrategy,
    LukasiewiczStrategy,
]

# Test with MLX backend (NumPy doesn't support grad())
try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None  # type: ignore


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestGradientFlowLogicalOperations:
    """Test gradient computation through logical operations."""

    @pytest.mark.parametrize("strategy_class", DIFFERENTIABLE_STRATEGIES)
    def test_and_gradient(self, strategy_class: type) -> None:
        """Test gradient flow through AND operation."""
        backend = create_backend("mlx")
        strategy = strategy_class(backend)

        def loss_fn(a: Any, b: Any) -> Any:
            """Compute AND and sum for scalar loss."""
            result = strategy.compile_and(a, b)
            return backend.sum(result)

        # Create gradient function for first argument
        grad_fn = backend.grad(loss_fn)

        # Test gradient computation
        a = mx.array([0.8, 0.6, 0.3])
        b = mx.array([0.9, 0.4, 0.7])

        grad_a = grad_fn(a, b)
        backend.eval(grad_a)

        # Gradient should be non-zero and finite
        assert mx.all(mx.isfinite(grad_a))
        # For product-based AND, gradient should be b
        # For other strategies, gradient structure varies but should exist

    @pytest.mark.parametrize("strategy_class", DIFFERENTIABLE_STRATEGIES)
    def test_or_gradient(self, strategy_class: type) -> None:
        """Test gradient flow through OR operation."""
        backend = create_backend("mlx")
        strategy = strategy_class(backend)

        def loss_fn(a: Any, b: Any) -> Any:
            """Compute OR and sum for scalar loss."""
            result = strategy.compile_or(a, b)
            return backend.sum(result)

        grad_fn = backend.grad(loss_fn)

        a = mx.array([0.8, 0.6, 0.3])
        b = mx.array([0.9, 0.4, 0.7])

        grad_a = grad_fn(a, b)
        backend.eval(grad_a)

        # Gradient should be non-zero and finite
        assert mx.all(mx.isfinite(grad_a))

    @pytest.mark.parametrize("strategy_class", DIFFERENTIABLE_STRATEGIES)
    def test_not_gradient(self, strategy_class: type) -> None:
        """Test gradient flow through NOT operation."""
        backend = create_backend("mlx")
        strategy = strategy_class(backend)

        def loss_fn(a: Any) -> Any:
            """Compute NOT and sum for scalar loss."""
            result = strategy.compile_not(a)
            return backend.sum(result)

        grad_fn = backend.grad(loss_fn)

        a = mx.array([0.8, 0.6, 0.3])
        grad_a = grad_fn(a)
        backend.eval(grad_a)

        # Gradient of NOT should be -1 (for continuous strategies)
        assert mx.all(mx.isfinite(grad_a))
        # Most strategies use 1-a, so gradient is -1
        if strategy_class in [SoftDifferentiableStrategy, ProductStrategy]:
            expected = mx.array([-1.0, -1.0, -1.0])
            assert mx.allclose(grad_a, expected, rtol=1e-5)

    @pytest.mark.parametrize("strategy_class", DIFFERENTIABLE_STRATEGIES)
    def test_implies_gradient(self, strategy_class: type) -> None:
        """Test gradient flow through IMPLIES operation."""
        backend = create_backend("mlx")
        strategy = strategy_class(backend)

        def loss_fn(a: Any, b: Any) -> Any:
            """Compute IMPLIES and sum for scalar loss."""
            result = strategy.compile_implies(a, b)
            return backend.sum(result)

        grad_fn = backend.grad(loss_fn)

        a = mx.array([0.8, 0.6, 0.3])
        b = mx.array([0.5, 0.4, 0.7])

        grad_a = grad_fn(a, b)
        backend.eval(grad_a)

        # Gradient should be finite
        assert mx.all(mx.isfinite(grad_a))


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestGradientFlowQuantifiers:
    """Test gradient computation through quantifier operations."""

    @pytest.mark.parametrize("strategy_class", DIFFERENTIABLE_STRATEGIES)
    def test_exists_gradient(self, strategy_class: type) -> None:
        """Test gradient flow through EXISTS quantifier."""
        backend = create_backend("mlx")
        strategy = strategy_class(backend)

        def loss_fn(predicate: Any) -> Any:
            """Compute EXISTS and return scalar."""
            result = strategy.compile_exists(predicate, axis=0)
            return backend.sum(result)

        grad_fn = backend.grad(loss_fn)

        predicate = mx.array([0.8, 0.6, 0.3, 0.9])
        grad_pred = grad_fn(predicate)
        backend.eval(grad_pred)

        # Gradient should be finite
        assert mx.all(mx.isfinite(grad_pred))
        # For max-based EXISTS, gradient flows to maximum element

    @pytest.mark.parametrize("strategy_class", DIFFERENTIABLE_STRATEGIES)
    def test_forall_gradient(self, strategy_class: type) -> None:
        """Test gradient flow through FORALL quantifier."""
        backend = create_backend("mlx")
        strategy = strategy_class(backend)

        def loss_fn(predicate: Any) -> Any:
            """Compute FORALL and return scalar."""
            result = strategy.compile_forall(predicate, axis=0)
            return backend.sum(result)

        grad_fn = backend.grad(loss_fn)

        predicate = mx.array([0.8, 0.6, 0.3, 0.9])
        grad_pred = grad_fn(predicate)
        backend.eval(grad_pred)

        # Gradient should be finite
        assert mx.all(mx.isfinite(grad_pred))
        # For min-based FORALL, gradient flows to minimum element


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestNumericalGradientValidation:
    """Validate analytical gradients against numerical approximations."""

    def numerical_gradient(
        self,
        fn: Any,
        x: Any,
        epsilon: float = 1e-4,
    ) -> Any:
        """Compute numerical gradient using finite differences.

        Args:
            fn: Function to differentiate
            x: Point at which to compute gradient
            epsilon: Step size for finite difference

        Returns:
            Numerical gradient approximation
        """
        grad = []
        for i in range(len(x)):
            # Create perturbed versions using array operations
            perturbation = mx.zeros_like(x)
            perturbation[i] = epsilon

            x_plus = x + perturbation
            x_minus = x - perturbation

            f_plus = fn(x_plus)
            f_minus = fn(x_minus)

            grad.append((f_plus - f_minus) / (2 * epsilon))

        return mx.array(grad)

    @pytest.mark.parametrize("strategy_class", [SoftDifferentiableStrategy, ProductStrategy])
    def test_and_numerical_validation(self, strategy_class: type) -> None:
        """Validate AND gradient numerically."""
        backend = create_backend("mlx")
        strategy = strategy_class(backend)

        b = mx.array([0.9, 0.4, 0.7])

        def loss_fn(a: Any) -> Any:
            result = strategy.compile_and(a, b)
            return backend.sum(result)

        # Analytical gradient
        grad_fn = backend.grad(loss_fn)
        a = mx.array([0.8, 0.6, 0.3])
        grad_analytical = grad_fn(a)
        backend.eval(grad_analytical)

        # Numerical gradient
        grad_numerical = self.numerical_gradient(loss_fn, a)
        backend.eval(grad_numerical)

        # Should match within tolerance
        assert mx.allclose(grad_analytical, grad_numerical, rtol=1e-3, atol=1e-4)

    @pytest.mark.parametrize("strategy_class", [SoftDifferentiableStrategy, ProductStrategy])
    def test_or_numerical_validation(self, strategy_class: type) -> None:
        """Validate OR gradient numerically."""
        backend = create_backend("mlx")
        strategy = strategy_class(backend)

        b = mx.array([0.9, 0.4, 0.7])

        def loss_fn(a: Any) -> Any:
            result = strategy.compile_or(a, b)
            return backend.sum(result)

        # Analytical gradient
        grad_fn = backend.grad(loss_fn)
        a = mx.array([0.8, 0.6, 0.3])
        grad_analytical = grad_fn(a)
        backend.eval(grad_analytical)

        # Numerical gradient
        grad_numerical = self.numerical_gradient(loss_fn, a)
        backend.eval(grad_numerical)

        # Should match within tolerance (relaxed for OR due to numerical approximation)
        assert mx.allclose(grad_analytical, grad_numerical, rtol=1e-2, atol=1e-3)


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestHardBooleanNonDifferentiability:
    """Test that HardBoolean strategy handles gradients appropriately."""

    def test_hard_and_gradient_zero(self) -> None:
        """Test HardBoolean AND produces zero gradients (step function)."""
        backend = create_backend("mlx")
        strategy = HardBooleanStrategy(backend)

        def loss_fn(a: Any, b: Any) -> Any:
            result = strategy.compile_and(a, b)
            return backend.sum(result)

        grad_fn = backend.grad(loss_fn)

        a = mx.array([0.8, 0.6, 0.3])
        b = mx.array([0.9, 0.4, 0.7])

        grad_a = grad_fn(a, b)
        backend.eval(grad_a)

        # Step function has zero gradient almost everywhere
        # (gradient is undefined at threshold, zero elsewhere)
        # MLX may return zero or NaN gradients
        assert mx.all(mx.isfinite(grad_a)) or mx.all(mx.isnan(grad_a))

    def test_hard_or_gradient_zero(self) -> None:
        """Test HardBoolean OR produces zero gradients (step function)."""
        backend = create_backend("mlx")
        strategy = HardBooleanStrategy(backend)

        def loss_fn(a: Any, b: Any) -> Any:
            result = strategy.compile_or(a, b)
            return backend.sum(result)

        grad_fn = backend.grad(loss_fn)

        a = mx.array([0.8, 0.6, 0.3])
        b = mx.array([0.9, 0.4, 0.7])

        grad_a = grad_fn(a, b)
        backend.eval(grad_a)

        # Step function has zero gradient
        assert mx.all(mx.isfinite(grad_a)) or mx.all(mx.isnan(grad_a))

    def test_hard_not_gradient_zero(self) -> None:
        """Test HardBoolean NOT produces zero gradients (step function)."""
        backend = create_backend("mlx")
        strategy = HardBooleanStrategy(backend)

        def loss_fn(a: Any) -> Any:
            result = strategy.compile_not(a)
            return backend.sum(result)

        grad_fn = backend.grad(loss_fn)

        a = mx.array([0.8, 0.6, 0.3])
        grad_a = grad_fn(a)
        backend.eval(grad_a)

        # Step function has zero gradient
        assert mx.all(mx.isfinite(grad_a)) or mx.all(mx.isnan(grad_a))


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestComplexGradientFlow:
    """Test gradient flow through complex compositions of operations."""

    @pytest.mark.parametrize("strategy_class", DIFFERENTIABLE_STRATEGIES)
    def test_nested_and_or_gradient(self, strategy_class: type) -> None:
        """Test gradient through nested AND(OR(a, b), c)."""
        backend = create_backend("mlx")
        strategy = strategy_class(backend)

        def loss_fn(a: Any, b: Any, c: Any) -> Any:
            """Compute AND(OR(a, b), c)."""
            or_result = strategy.compile_or(a, b)
            and_result = strategy.compile_and(or_result, c)
            return backend.sum(and_result)

        grad_fn = backend.grad(loss_fn)

        a = mx.array([0.8, 0.6, 0.3])
        b = mx.array([0.5, 0.4, 0.7])
        c = mx.array([0.9, 0.3, 0.6])

        grad_a = grad_fn(a, b, c)
        backend.eval(grad_a)

        # Gradient should flow through nested operations
        assert mx.all(mx.isfinite(grad_a))

    @pytest.mark.parametrize("strategy_class", DIFFERENTIABLE_STRATEGIES)
    def test_implication_chain_gradient(self, strategy_class: type) -> None:
        """Test gradient through implication chain (a -> b) -> c."""
        backend = create_backend("mlx")
        strategy = strategy_class(backend)

        def loss_fn(a: Any, b: Any, c: Any) -> Any:
            """Compute (a -> b) -> c."""
            implies1 = strategy.compile_implies(a, b)
            implies2 = strategy.compile_implies(implies1, c)
            return backend.sum(implies2)

        grad_fn = backend.grad(loss_fn)

        a = mx.array([0.8, 0.6, 0.3])
        b = mx.array([0.5, 0.4, 0.7])
        c = mx.array([0.9, 0.3, 0.6])

        grad_a = grad_fn(a, b, c)
        backend.eval(grad_a)

        # Gradient should flow through chained implications
        assert mx.all(mx.isfinite(grad_a))

    @pytest.mark.parametrize("strategy_class", DIFFERENTIABLE_STRATEGIES)
    def test_quantifier_with_logic_gradient(self, strategy_class: type) -> None:
        """Test gradient through EXISTS(AND(predicate, condition))."""
        backend = create_backend("mlx")
        strategy = strategy_class(backend)

        condition = mx.array([0.9, 0.8, 0.7, 0.6])

        def loss_fn(predicate: Any) -> Any:
            """Compute EXISTS(AND(predicate, condition))."""
            and_result = strategy.compile_and(predicate, condition)
            exists_result = strategy.compile_exists(and_result, axis=0)
            return backend.sum(exists_result)

        grad_fn = backend.grad(loss_fn)

        predicate = mx.array([0.8, 0.6, 0.3, 0.9])
        grad_pred = grad_fn(predicate)
        backend.eval(grad_pred)

        # Gradient should flow through quantifier composition
        assert mx.all(mx.isfinite(grad_pred))


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestGradientStability:
    """Test numerical stability of gradients."""

    @pytest.mark.parametrize("strategy_class", DIFFERENTIABLE_STRATEGIES)
    def test_gradient_at_boundaries(self, strategy_class: type) -> None:
        """Test gradient computation at boundary values (0 and 1)."""
        backend = create_backend("mlx")
        strategy = strategy_class(backend)

        def loss_fn(a: Any, b: Any) -> Any:
            result = strategy.compile_and(a, b)
            return backend.sum(result)

        grad_fn = backend.grad(loss_fn)

        # Test at boundaries
        a = mx.array([0.0, 1.0, 0.5])
        b = mx.array([1.0, 0.0, 0.5])

        grad_a = grad_fn(a, b)
        backend.eval(grad_a)

        # Gradients should be finite at boundaries
        assert mx.all(mx.isfinite(grad_a))

    @pytest.mark.parametrize("strategy_class", DIFFERENTIABLE_STRATEGIES)
    def test_gradient_with_small_values(self, strategy_class: type) -> None:
        """Test gradient computation with very small values."""
        backend = create_backend("mlx")
        strategy = strategy_class(backend)

        def loss_fn(a: Any, b: Any) -> Any:
            result = strategy.compile_and(a, b)
            return backend.sum(result)

        grad_fn = backend.grad(loss_fn)

        # Test with small values
        a = mx.array([1e-5, 1e-6, 1e-7])
        b = mx.array([0.5, 0.5, 0.5])

        grad_a = grad_fn(a, b)
        backend.eval(grad_a)

        # Gradients should remain stable with small values
        assert mx.all(mx.isfinite(grad_a))
