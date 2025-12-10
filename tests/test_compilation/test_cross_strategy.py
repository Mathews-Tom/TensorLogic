"""Cross-strategy validation tests.

Comparative tests verifying consistency where semantics overlap and documenting
expected divergence between different compilation strategies.

Key Validations:
- SoftDifferentiable ≈ Product (both use product t-norm)
- HardBoolean produces only {0, 1}
- Gödel uses min/max operations
- Łukasiewicz uses bounded difference
- Strategy-specific behavior documented
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlogic.backends import create_backend
from tensorlogic.compilation import (
    GodelStrategy,
    HardBooleanStrategy,
    LukasiewiczStrategy,
    ProductStrategy,
    SoftDifferentiableStrategy,
)


class TestSoftProductEquivalence:
    """Verify SoftDifferentiable ≈ Product for all operations."""

    def test_and_equivalence(self) -> None:
        """Test AND operations are equivalent."""
        backend = create_backend("numpy")
        soft = SoftDifferentiableStrategy(backend)
        product = ProductStrategy(backend)

        # Test cases spanning [0, 1]
        test_cases = [
            (np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])),
            (np.array([0.2, 0.4, 0.6, 0.8]), np.array([0.3, 0.5, 0.7, 0.9])),
            (np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([[0.5, 0.6], [0.7, 0.8]])),
        ]

        for a, b in test_cases:
            soft_result = soft.compile_and(a, b)
            product_result = product.compile_and(a, b)
            np.testing.assert_allclose(soft_result, product_result, rtol=1e-10, atol=1e-10)

    def test_or_equivalence(self) -> None:
        """Test OR operations are equivalent."""
        backend = create_backend("numpy")
        soft = SoftDifferentiableStrategy(backend)
        product = ProductStrategy(backend)

        test_cases = [
            (np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0])),
            (np.array([0.2, 0.4, 0.6, 0.8]), np.array([0.3, 0.5, 0.7, 0.9])),
        ]

        for a, b in test_cases:
            soft_result = soft.compile_or(a, b)
            product_result = product.compile_or(a, b)
            np.testing.assert_allclose(soft_result, product_result, rtol=1e-10, atol=1e-10)

    def test_not_equivalence(self) -> None:
        """Test NOT operations are equivalent."""
        backend = create_backend("numpy")
        soft = SoftDifferentiableStrategy(backend)
        product = ProductStrategy(backend)

        test_cases = [
            np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
            np.array([[0.1, 0.2], [0.8, 0.9]]),
        ]

        for a in test_cases:
            soft_result = soft.compile_not(a)
            product_result = product.compile_not(a)
            np.testing.assert_allclose(soft_result, product_result, rtol=1e-10, atol=1e-10)

    def test_quantifier_equivalence(self) -> None:
        """Test quantifiers are equivalent."""
        backend = create_backend("numpy")
        soft = SoftDifferentiableStrategy(backend)
        product = ProductStrategy(backend)

        pred = np.array([[0.2, 0.4, 0.6], [0.8, 0.9, 1.0]])

        # EXISTS
        soft_exists = soft.compile_exists(pred, axis=1)
        product_exists = product.compile_exists(pred, axis=1)
        np.testing.assert_allclose(soft_exists, product_exists, rtol=1e-10, atol=1e-10)

        # FORALL
        soft_forall = soft.compile_forall(pred, axis=1)
        product_forall = product.compile_forall(pred, axis=1)
        np.testing.assert_allclose(soft_forall, product_forall, rtol=1e-10, atol=1e-10)


class TestHardBooleanDiscreteness:
    """Verify HardBoolean produces only binary {0, 1} outputs."""

    def test_and_discrete(self) -> None:
        """Test AND produces only 0 or 1."""
        backend = create_backend("numpy")
        strategy = HardBooleanStrategy(backend)

        # Fuzzy inputs
        a = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        b = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

        result = strategy.compile_and(a, b)

        # All values must be exactly 0 or 1
        assert np.all((result == 0) | (result == 1)), \
            f"HardBoolean AND produced non-discrete values: {result}"

    def test_or_discrete(self) -> None:
        """Test OR produces only 0 or 1."""
        backend = create_backend("numpy")
        strategy = HardBooleanStrategy(backend)

        a = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        b = np.array([0.0, 0.2, 0.4, 0.6, 0.8])

        result = strategy.compile_or(a, b)

        assert np.all((result == 0) | (result == 1)), \
            f"HardBoolean OR produced non-discrete values: {result}"

    def test_not_discrete(self) -> None:
        """Test NOT produces only 0 or 1."""
        backend = create_backend("numpy")
        strategy = HardBooleanStrategy(backend)

        a = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        result = strategy.compile_not(a)

        assert np.all((result == 0) | (result == 1)), \
            f"HardBoolean NOT produced non-discrete values: {result}"

    def test_quantifiers_discrete(self) -> None:
        """Test quantifiers produce only 0 or 1."""
        backend = create_backend("numpy")
        strategy = HardBooleanStrategy(backend)

        pred = np.array([[0.1, 0.5, 0.9], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

        exists_result = strategy.compile_exists(pred, axis=1)
        forall_result = strategy.compile_forall(pred, axis=1)

        assert np.all((exists_result == 0) | (exists_result == 1)), \
            f"HardBoolean EXISTS produced non-discrete values: {exists_result}"
        assert np.all((forall_result == 0) | (forall_result == 1)), \
            f"HardBoolean FORALL produced non-discrete values: {forall_result}"


class TestStrategyDivergence:
    """Document expected divergence between strategies."""

    def test_and_divergence_on_midpoint(self) -> None:
        """Test AND behavior at midpoint (0.5, 0.5) varies by strategy."""
        backend = create_backend("numpy")
        a, b = np.array([0.5]), np.array([0.5])

        # Collect results from all strategies
        results = {
            "soft": SoftDifferentiableStrategy(backend).compile_and(a, b)[0],
            "hard": HardBooleanStrategy(backend).compile_and(a, b)[0],
            "godel": GodelStrategy(backend).compile_and(a, b)[0],
            "product": ProductStrategy(backend).compile_and(a, b)[0],
            "lukasiewicz": LukasiewiczStrategy(backend).compile_and(a, b)[0],
        }

        # Document expected behavior
        assert results["soft"] == 0.25, "Soft: 0.5 * 0.5 = 0.25"
        assert results["hard"] == 1.0, "Hard: step(0.5) * step(0.5) = 1"
        assert results["godel"] == 0.5, "Gödel: min(0.5, 0.5) = 0.5"
        assert results["product"] == 0.25, "Product: 0.5 * 0.5 = 0.25"
        assert results["lukasiewicz"] == 0.0, "Łukasiewicz: max(0, 0.5+0.5-1) = 0"

    def test_or_divergence_on_midpoint(self) -> None:
        """Test OR behavior at midpoint (0.5, 0.5) varies by strategy."""
        backend = create_backend("numpy")
        a, b = np.array([0.5]), np.array([0.5])

        results = {
            "soft": SoftDifferentiableStrategy(backend).compile_or(a, b)[0],
            "hard": HardBooleanStrategy(backend).compile_or(a, b)[0],
            "godel": GodelStrategy(backend).compile_or(a, b)[0],
            "product": ProductStrategy(backend).compile_or(a, b)[0],
            "lukasiewicz": LukasiewiczStrategy(backend).compile_or(a, b)[0],
        }

        # Document expected behavior
        assert results["soft"] == pytest.approx(0.75), "Soft: 0.5+0.5-0.5*0.5 = 0.75"
        assert results["hard"] == 1.0, "Hard: max(step(0.5), step(0.5)) = 1"
        assert results["godel"] == 0.5, "Gödel: max(0.5, 0.5) = 0.5"
        assert results["product"] == pytest.approx(0.75), "Product: 0.5+0.5-0.5*0.5 = 0.75"
        assert results["lukasiewicz"] == 1.0, "Łukasiewicz: min(1, 0.5+0.5) = 1.0"

    def test_implies_divergence(self) -> None:
        """Test IMPLIES behavior varies significantly."""
        backend = create_backend("numpy")
        a, b = np.array([0.5]), np.array([0.3])

        results = {
            "soft": SoftDifferentiableStrategy(backend).compile_implies(a, b)[0],
            "hard": HardBooleanStrategy(backend).compile_implies(a, b)[0],
            "godel": GodelStrategy(backend).compile_implies(a, b)[0],
            "product": ProductStrategy(backend).compile_implies(a, b)[0],
            "lukasiewicz": LukasiewiczStrategy(backend).compile_implies(a, b)[0],
        }

        # Document expected behavior
        # Note: Gödel uses residuated implication (1 if a≤b, else b)
        assert results["soft"] == pytest.approx(0.5, abs=0.01), \
            "Soft: max(1-0.5, 0.3) = max(0.5, 0.3) = 0.5"
        assert results["hard"] in [0.0, 1.0], "Hard: discrete boolean implication"
        assert results["godel"] == pytest.approx(0.3, abs=0.01), \
            "Gödel: residuated → gives b when a>b, so 0.5→0.3 = 0.3"
        assert results["product"] == pytest.approx(0.5, abs=0.01), \
            "Product: max(1-0.5, 0.3) = 0.5"
        assert results["lukasiewicz"] == 0.8, \
            "Łukasiewicz: min(1, 1-0.5+0.3) = min(1, 0.8) = 0.8"


class TestConsistencyAtBoundaries:
    """Verify all strategies behave consistently at 0 and 1."""

    def test_and_at_zero(self) -> None:
        """Test AND(0, x) = 0 for all strategies."""
        backend = create_backend("numpy")
        strategies = [
            SoftDifferentiableStrategy(backend),
            HardBooleanStrategy(backend),
            GodelStrategy(backend),
            ProductStrategy(backend),
            LukasiewiczStrategy(backend),
        ]

        zero = np.array([0.0])
        x = np.array([0.7])

        for strategy in strategies:
            result = strategy.compile_and(zero, x)
            np.testing.assert_allclose(result, np.array([0.0]), rtol=1e-10, atol=1e-10,
                                      err_msg=f"{strategy.name} failed AND(0, x) = 0")

    def test_or_at_one(self) -> None:
        """Test OR(1, x) = 1 for all strategies."""
        backend = create_backend("numpy")
        strategies = [
            SoftDifferentiableStrategy(backend),
            HardBooleanStrategy(backend),
            GodelStrategy(backend),
            ProductStrategy(backend),
            LukasiewiczStrategy(backend),
        ]

        one = np.array([1.0])
        x = np.array([0.3])

        for strategy in strategies:
            result = strategy.compile_or(one, x)
            np.testing.assert_allclose(result, np.array([1.0]), rtol=1e-10, atol=1e-10,
                                      err_msg=f"{strategy.name} failed OR(1, x) = 1")

    def test_not_at_boundaries(self) -> None:
        """Test NOT(0) = 1 and NOT(1) = 0 for all strategies."""
        backend = create_backend("numpy")
        strategies = [
            SoftDifferentiableStrategy(backend),
            HardBooleanStrategy(backend),
            GodelStrategy(backend),
            ProductStrategy(backend),
            LukasiewiczStrategy(backend),
        ]

        zero = np.array([0.0])
        one = np.array([1.0])

        for strategy in strategies:
            not_zero = strategy.compile_not(zero)
            not_one = strategy.compile_not(one)

            np.testing.assert_allclose(not_zero, np.array([1.0]), rtol=1e-10, atol=1e-10,
                                      err_msg=f"{strategy.name} failed NOT(0) = 1")
            np.testing.assert_allclose(not_one, np.array([0.0]), rtol=1e-10, atol=1e-10,
                                      err_msg=f"{strategy.name} failed NOT(1) = 0")


class TestGodelMinMaxBehavior:
    """Verify Gödel strategy uses min/max operations."""

    def test_and_is_minimum(self) -> None:
        """Test Gödel AND is minimum operation."""
        backend = create_backend("numpy")
        strategy = GodelStrategy(backend)

        test_cases = [
            (np.array([0.3, 0.7, 0.5]), np.array([0.5, 0.2, 0.5])),
            (np.array([[0.1, 0.9], [0.4, 0.6]]), np.array([[0.8, 0.3], [0.5, 0.7]])),
        ]

        for a, b in test_cases:
            result = strategy.compile_and(a, b)
            expected = np.minimum(a, b)
            np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10,
                                      err_msg="Gödel AND should be minimum")

    def test_or_is_maximum(self) -> None:
        """Test Gödel OR is maximum operation."""
        backend = create_backend("numpy")
        strategy = GodelStrategy(backend)

        test_cases = [
            (np.array([0.3, 0.7, 0.5]), np.array([0.5, 0.2, 0.5])),
            (np.array([[0.1, 0.9], [0.4, 0.6]]), np.array([[0.8, 0.3], [0.5, 0.7]])),
        ]

        for a, b in test_cases:
            result = strategy.compile_or(a, b)
            expected = np.maximum(a, b)
            np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10,
                                      err_msg="Gödel OR should be maximum")


class TestLukasiewiczBoundedOperations:
    """Verify Łukasiewicz uses bounded difference/sum."""

    def test_and_bounded_difference(self) -> None:
        """Test Łukasiewicz AND is max(0, a+b-1)."""
        backend = create_backend("numpy")
        strategy = LukasiewiczStrategy(backend)

        test_cases = [
            (np.array([0.3, 0.7, 0.5]), np.array([0.5, 0.6, 0.5])),
            (np.array([0.8, 0.9, 1.0]), np.array([0.7, 0.8, 0.9])),
        ]

        for a, b in test_cases:
            result = strategy.compile_and(a, b)
            expected = np.maximum(0, a + b - 1)
            np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10,
                                      err_msg="Łukasiewicz AND should be max(0, a+b-1)")

    def test_or_bounded_sum(self) -> None:
        """Test Łukasiewicz OR is min(1, a+b)."""
        backend = create_backend("numpy")
        strategy = LukasiewiczStrategy(backend)

        test_cases = [
            (np.array([0.3, 0.7, 0.5]), np.array([0.5, 0.6, 0.5])),
            (np.array([0.1, 0.2, 0.3]), np.array([0.4, 0.5, 0.6])),
        ]

        for a, b in test_cases:
            result = strategy.compile_or(a, b)
            expected = np.minimum(1, a + b)
            np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10,
                                      err_msg="Łukasiewicz OR should be min(1, a+b)")
