"""Tests for compilation strategy integration with quantify() API.

Tests that quantify() correctly accepts and uses compilation strategies,
including both string names and direct strategy instances.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlogic.api import quantify
from tensorlogic.backends import create_backend
from tensorlogic.compilation import (
    CompilationStrategy,
    GodelStrategy,
    HardBooleanStrategy,
    LukasiewiczStrategy,
    ProductStrategy,
    SoftDifferentiableStrategy,
    create_strategy,
)


@pytest.fixture
def numpy_backend():
    """Create NumPy backend for testing."""
    return create_backend("numpy")


@pytest.fixture
def sample_predicates():
    """Sample predicates for testing."""
    return {
        "P": np.array([0.8, 0.6, 0.3]),
        "Q": np.array([0.9, 0.4, 0.7]),
        "R": np.array([0.2, 0.8, 0.5]),
    }


class TestStrategyParameterStringNames:
    """Test quantify() with strategy names as strings."""

    def test_default_strategy_soft_differentiable(
        self, numpy_backend, sample_predicates
    ) -> None:
        """Test quantify() uses soft_differentiable by default."""
        result = quantify(
            "P(x) and Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            backend=numpy_backend,
        )

        # Should use soft differentiable (product) semantics: a * b
        expected = np.array([0.8 * 0.9, 0.6 * 0.4, 0.3 * 0.7])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_explicit_soft_differentiable_strategy(
        self, numpy_backend, sample_predicates
    ) -> None:
        """Test quantify() with explicit 'soft_differentiable' strategy."""
        result = quantify(
            "P(x) and Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy="soft_differentiable",
            backend=numpy_backend,
        )

        expected = np.array([0.8 * 0.9, 0.6 * 0.4, 0.3 * 0.7])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_hard_boolean_strategy_string(self, numpy_backend, sample_predicates) -> None:
        """Test quantify() with 'hard_boolean' strategy name."""
        result = quantify(
            "P(x) and Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy="hard_boolean",
            backend=numpy_backend,
        )

        # Should use hard boolean (step) semantics
        # step(0.8 * 0.9) = 1, step(0.6 * 0.4) = 1, step(0.3 * 0.7) = 1
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_godel_strategy_string(self, numpy_backend, sample_predicates) -> None:
        """Test quantify() with 'godel' strategy name."""
        result = quantify(
            "P(x) and Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy="godel",
            backend=numpy_backend,
        )

        # Should use Gödel (min) semantics
        expected = np.array([min(0.8, 0.9), min(0.6, 0.4), min(0.3, 0.7)])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_product_strategy_string(self, numpy_backend, sample_predicates) -> None:
        """Test quantify() with 'product' strategy name."""
        result = quantify(
            "P(x) and Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy="product",
            backend=numpy_backend,
        )

        # Should use product semantics: a * b
        expected = np.array([0.8 * 0.9, 0.6 * 0.4, 0.3 * 0.7])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_lukasiewicz_strategy_string(
        self, numpy_backend, sample_predicates
    ) -> None:
        """Test quantify() with 'lukasiewicz' strategy name."""
        result = quantify(
            "P(x) and Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy="lukasiewicz",
            backend=numpy_backend,
        )

        # Should use Łukasiewicz (max(0, a+b-1)) semantics
        expected = np.array(
            [
                max(0, 0.8 + 0.9 - 1),
                max(0, 0.6 + 0.4 - 1),
                max(0, 0.3 + 0.7 - 1),
            ]
        )
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_invalid_strategy_name_raises(
        self, numpy_backend, sample_predicates
    ) -> None:
        """Test quantify() raises ValueError for invalid strategy name."""
        with pytest.raises(ValueError, match="Unknown compilation strategy"):
            quantify(
                "P(x) and Q(x)",
                predicates=sample_predicates,
                bindings={"x": np.array([0, 1, 2])},
                strategy="invalid_strategy",
                backend=numpy_backend,
            )


class TestStrategyParameterInstances:
    """Test quantify() with strategy instances."""

    def test_soft_differentiable_instance(
        self, numpy_backend, sample_predicates
    ) -> None:
        """Test quantify() with SoftDifferentiableStrategy instance."""
        strategy = SoftDifferentiableStrategy(numpy_backend)

        result = quantify(
            "P(x) and Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy=strategy,
            backend=numpy_backend,
        )

        expected = np.array([0.8 * 0.9, 0.6 * 0.4, 0.3 * 0.7])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_hard_boolean_instance(self, numpy_backend, sample_predicates) -> None:
        """Test quantify() with HardBooleanStrategy instance."""
        strategy = HardBooleanStrategy(numpy_backend)

        result = quantify(
            "P(x) and Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy=strategy,
            backend=numpy_backend,
        )

        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_godel_instance(self, numpy_backend, sample_predicates) -> None:
        """Test quantify() with GodelStrategy instance."""
        strategy = GodelStrategy(numpy_backend)

        result = quantify(
            "P(x) and Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy=strategy,
            backend=numpy_backend,
        )

        expected = np.array([min(0.8, 0.9), min(0.6, 0.4), min(0.3, 0.7)])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_product_instance(self, numpy_backend, sample_predicates) -> None:
        """Test quantify() with ProductStrategy instance."""
        strategy = ProductStrategy(numpy_backend)

        result = quantify(
            "P(x) and Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy=strategy,
            backend=numpy_backend,
        )

        expected = np.array([0.8 * 0.9, 0.6 * 0.4, 0.3 * 0.7])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_lukasiewicz_instance(self, numpy_backend, sample_predicates) -> None:
        """Test quantify() with LukasiewiczStrategy instance."""
        strategy = LukasiewiczStrategy(numpy_backend)

        result = quantify(
            "P(x) and Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy=strategy,
            backend=numpy_backend,
        )

        expected = np.array(
            [
                max(0, 0.8 + 0.9 - 1),
                max(0, 0.6 + 0.4 - 1),
                max(0, 0.3 + 0.7 - 1),
            ]
        )
        np.testing.assert_allclose(result, expected, atol=1e-6)


class TestStrategyWithLogicalOperators:
    """Test strategies with different logical operators."""

    def test_or_operator_with_strategies(self, numpy_backend, sample_predicates) -> None:
        """Test OR operation with different strategies."""
        # Gödel OR: max(a, b)
        result_godel = quantify(
            "P(x) or Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy="godel",
            backend=numpy_backend,
        )
        expected_godel = np.array([max(0.8, 0.9), max(0.6, 0.4), max(0.3, 0.7)])
        np.testing.assert_allclose(result_godel, expected_godel, atol=1e-6)

        # Łukasiewicz OR: min(1, a+b)
        result_luk = quantify(
            "P(x) or Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy="lukasiewicz",
            backend=numpy_backend,
        )
        expected_luk = np.array(
            [min(1, 0.8 + 0.9), min(1, 0.6 + 0.4), min(1, 0.3 + 0.7)]
        )
        np.testing.assert_allclose(result_luk, expected_luk, atol=1e-6)

    def test_not_operator_with_strategies(
        self, numpy_backend, sample_predicates
    ) -> None:
        """Test NOT operation with different strategies."""
        # All strategies use 1 - a for NOT
        result_soft = quantify(
            "not P(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy="soft_differentiable",
            backend=numpy_backend,
        )
        expected = np.array([1 - 0.8, 1 - 0.6, 1 - 0.3])
        np.testing.assert_allclose(result_soft, expected, atol=1e-6)

        # Verify same result with Gödel
        result_godel = quantify(
            "not P(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy="godel",
            backend=numpy_backend,
        )
        np.testing.assert_allclose(result_godel, expected, atol=1e-6)

    def test_implies_operator_with_strategies(
        self, numpy_backend, sample_predicates
    ) -> None:
        """Test IMPLIES operation with different strategies."""
        # Gödel implication: if a <= b then 1 else b
        result_godel = quantify(
            "P(x) -> Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy="godel",
            backend=numpy_backend,
        )
        # P[0]=0.8 <= Q[0]=0.9: 1, P[1]=0.6 > Q[1]=0.4: 0.4, P[2]=0.3 <= Q[2]=0.7: 1
        expected_godel = np.array([1.0, 0.4, 1.0])
        np.testing.assert_allclose(result_godel, expected_godel, atol=1e-6)

        # Łukasiewicz implication: min(1, 1-a+b)
        result_luk = quantify(
            "P(x) -> Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy="lukasiewicz",
            backend=numpy_backend,
        )
        expected_luk = np.array(
            [
                min(1, 1 - 0.8 + 0.9),
                min(1, 1 - 0.6 + 0.4),
                min(1, 1 - 0.3 + 0.7),
            ]
        )
        np.testing.assert_allclose(result_luk, expected_luk, atol=1e-6)


class TestStrategyWithQuantifiers:
    """Test strategies with quantifiers."""

    def test_exists_quantifier_with_strategies(
        self, numpy_backend, sample_predicates
    ) -> None:
        """Test EXISTS quantifier with different strategies."""
        # All strategies use max for EXISTS
        result_soft = quantify(
            "exists x: P(x)",
            predicates=sample_predicates,
            strategy="soft_differentiable",
            backend=numpy_backend,
        )
        expected = np.array([0.8])  # max(0.8, 0.6, 0.3)
        np.testing.assert_allclose(result_soft, expected, atol=1e-6)

        # Verify same with Gödel
        result_godel = quantify(
            "exists x: P(x)",
            predicates=sample_predicates,
            strategy="godel",
            backend=numpy_backend,
        )
        np.testing.assert_allclose(result_godel, expected, atol=1e-6)

    def test_forall_quantifier_with_strategies(
        self, numpy_backend, sample_predicates
    ) -> None:
        """Test FORALL quantifier with different strategies."""
        # All strategies use min for FORALL
        result_soft = quantify(
            "forall x: P(x)",
            predicates=sample_predicates,
            strategy="soft_differentiable",
            backend=numpy_backend,
        )
        expected = np.array([0.3])  # min(0.8, 0.6, 0.3)
        np.testing.assert_allclose(result_soft, expected, atol=1e-6)

        # Verify same with Łukasiewicz
        result_luk = quantify(
            "forall x: P(x)",
            predicates=sample_predicates,
            strategy="lukasiewicz",
            backend=numpy_backend,
        )
        np.testing.assert_allclose(result_luk, expected, atol=1e-6)


class TestStrategyBackwardCompatibility:
    """Test backward compatibility (strategy parameter is optional)."""

    def test_omitting_strategy_uses_default(
        self, numpy_backend, sample_predicates
    ) -> None:
        """Test that omitting strategy parameter works (backward compatible)."""
        result = quantify(
            "P(x) and Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            backend=numpy_backend,
            # No strategy parameter - should use default
        )

        # Should use default (soft_differentiable)
        expected = np.array([0.8 * 0.9, 0.6 * 0.4, 0.3 * 0.7])
        np.testing.assert_allclose(result, expected, atol=1e-6)


class TestFactoryIntegration:
    """Test factory integration with quantify()."""

    def test_factory_created_strategy_works(
        self, numpy_backend, sample_predicates
    ) -> None:
        """Test that factory-created strategy works with quantify()."""
        # Create strategy via factory
        strategy = create_strategy("godel", backend=numpy_backend)

        # Use with quantify()
        result = quantify(
            "P(x) and Q(x)",
            predicates=sample_predicates,
            bindings={"x": np.array([0, 1, 2])},
            strategy=strategy,
            backend=numpy_backend,
        )

        expected = np.array([min(0.8, 0.9), min(0.6, 0.4), min(0.3, 0.7)])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_all_registered_strategies_work(
        self, numpy_backend, sample_predicates
    ) -> None:
        """Test that all registered strategies work with quantify()."""
        from tensorlogic.compilation import get_available_strategies

        strategies = get_available_strategies()
        assert len(strategies) >= 5  # At least 5 strategies registered

        for strategy_name in strategies:
            # Should not raise any errors
            result = quantify(
                "P(x) and Q(x)",
                predicates=sample_predicates,
                bindings={"x": np.array([0, 1, 2])},
                strategy=strategy_name,
                backend=numpy_backend,
            )

            # Verify result is a valid array
            assert isinstance(result, np.ndarray)
            assert result.shape == (3,)
