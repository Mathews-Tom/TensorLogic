"""Property-based tests for compilation strategy semantic axioms.

Uses hypothesis to generate test cases verifying mathematical properties:
- Commutativity, associativity, distributivity
- Identity and annihilator laws
- Involution, De Morgan's laws
- Monotonicity and absorption

Each strategy must satisfy its semantic axioms under these property tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tensorlogic.backends import create_backend
from tensorlogic.compilation import (
    GodelStrategy,
    HardBooleanStrategy,
    LukasiewiczStrategy,
    ProductStrategy,
    SoftDifferentiableStrategy,
)

# Hypothesis strategy for fuzzy values in [0, 1]
fuzzy_values = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Hypothesis strategy for arrays of fuzzy values
fuzzy_arrays = st.lists(fuzzy_values, min_size=1, max_size=10).map(lambda x: np.array(x))


# All strategy classes to test
ALL_STRATEGIES = [
    SoftDifferentiableStrategy,
    HardBooleanStrategy,
    GodelStrategy,
    ProductStrategy,
    LukasiewiczStrategy,
]


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
class TestCommutativeProperties:
    """Test commutative properties: op(a, b) = op(b, a)."""

    @given(a=fuzzy_arrays, b=fuzzy_arrays)
    @settings(max_examples=100, deadline=1000)
    def test_and_commutative(self, strategy_class, a, b) -> None:
        """Test AND(a, b) = AND(b, a)."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        # Make arrays same size
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        result_ab = strategy.compile_and(a, b)
        result_ba = strategy.compile_and(b, a)

        np.testing.assert_allclose(result_ab, result_ba, rtol=1e-5, atol=1e-6)

    @given(a=fuzzy_arrays, b=fuzzy_arrays)
    @settings(max_examples=100, deadline=1000)
    def test_or_commutative(self, strategy_class, a, b) -> None:
        """Test OR(a, b) = OR(b, a)."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        result_ab = strategy.compile_or(a, b)
        result_ba = strategy.compile_or(b, a)

        np.testing.assert_allclose(result_ab, result_ba, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
class TestAssociativeProperties:
    """Test associative properties: op(op(a, b), c) = op(a, op(b, c))."""

    @given(a=fuzzy_arrays, b=fuzzy_arrays, c=fuzzy_arrays)
    @settings(max_examples=100, deadline=1000)
    def test_and_associative(self, strategy_class, a, b, c) -> None:
        """Test AND(AND(a, b), c) = AND(a, AND(b, c))."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        min_len = min(len(a), len(b), len(c))
        a, b, c = a[:min_len], b[:min_len], c[:min_len]

        left = strategy.compile_and(strategy.compile_and(a, b), c)
        right = strategy.compile_and(a, strategy.compile_and(b, c))

        np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)

    @given(a=fuzzy_arrays, b=fuzzy_arrays, c=fuzzy_arrays)
    @settings(max_examples=100, deadline=1000)
    def test_or_associative(self, strategy_class, a, b, c) -> None:
        """Test OR(OR(a, b), c) = OR(a, OR(b, c))."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        min_len = min(len(a), len(b), len(c))
        a, b, c = a[:min_len], b[:min_len], c[:min_len]

        left = strategy.compile_or(strategy.compile_or(a, b), c)
        right = strategy.compile_or(a, strategy.compile_or(b, c))

        np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("strategy_class", [
    SoftDifferentiableStrategy,
    GodelStrategy,
    ProductStrategy,
    LukasiewiczStrategy,
])
class TestIdentityProperties:
    """Test identity properties: op(a, identity) = a (fuzzy strategies only)."""

    @given(a=fuzzy_arrays)
    @settings(max_examples=100, deadline=1000)
    def test_and_identity(self, strategy_class, a) -> None:
        """Test AND(a, 1) = a."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        ones = backend.ones(a.shape)
        result = strategy.compile_and(a, ones)

        np.testing.assert_allclose(result, a, rtol=1e-5, atol=1e-6)

    @given(a=fuzzy_arrays)
    @settings(max_examples=100, deadline=1000)
    def test_or_identity(self, strategy_class, a) -> None:
        """Test OR(a, 0) = a."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        zeros = backend.zeros(a.shape)
        result = strategy.compile_or(a, zeros)

        np.testing.assert_allclose(result, a, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
class TestAnnihilatorProperties:
    """Test annihilator properties: op(a, annihilator) = annihilator."""

    @given(a=fuzzy_arrays)
    @settings(max_examples=100, deadline=1000)
    def test_and_annihilator(self, strategy_class, a) -> None:
        """Test AND(a, 0) = 0."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        zeros = backend.zeros(a.shape)
        result = strategy.compile_and(a, zeros)

        np.testing.assert_allclose(result, zeros, rtol=1e-5, atol=1e-6)

    @given(a=fuzzy_arrays)
    @settings(max_examples=100, deadline=1000)
    def test_or_annihilator(self, strategy_class, a) -> None:
        """Test OR(a, 1) = 1."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        ones = backend.ones(a.shape)
        result = strategy.compile_or(a, ones)

        np.testing.assert_allclose(result, ones, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("strategy_class", [
    SoftDifferentiableStrategy,
    GodelStrategy,
    ProductStrategy,
    LukasiewiczStrategy,
])
class TestInvolutionProperty:
    """Test involution: NOT(NOT(a)) = a (fuzzy strategies only)."""

    @given(a=fuzzy_arrays)
    @settings(max_examples=100, deadline=1000)
    def test_not_involution(self, strategy_class, a) -> None:
        """Test NOT(NOT(a)) = a."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        result = strategy.compile_not(strategy.compile_not(a))

        np.testing.assert_allclose(result, a, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
class TestDeMorganLaws:
    """Test De Morgan's laws."""

    @given(a=fuzzy_arrays, b=fuzzy_arrays)
    @settings(max_examples=100, deadline=1000)
    def test_demorgan_and(self, strategy_class, a, b) -> None:
        """Test NOT(a AND b) = NOT(a) OR NOT(b)."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        # NOT(a AND b)
        left = strategy.compile_not(strategy.compile_and(a, b))

        # NOT(a) OR NOT(b)
        right = strategy.compile_or(
            strategy.compile_not(a),
            strategy.compile_not(b)
        )

        np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)

    @given(a=fuzzy_arrays, b=fuzzy_arrays)
    @settings(max_examples=100, deadline=1000)
    def test_demorgan_or(self, strategy_class, a, b) -> None:
        """Test NOT(a OR b) = NOT(a) AND NOT(b)."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        # NOT(a OR b)
        left = strategy.compile_not(strategy.compile_or(a, b))

        # NOT(a) AND NOT(b)
        right = strategy.compile_and(
            strategy.compile_not(a),
            strategy.compile_not(b)
        )

        np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
class TestMonotonicityProperties:
    """Test monotonicity: if a ≤ a' and b ≤ b', then op(a,b) ≤ op(a',b')."""

    @given(a=fuzzy_arrays, b=fuzzy_arrays, delta=st.floats(min_value=0.0, max_value=0.5))
    @settings(max_examples=50, deadline=1000)
    def test_and_monotonic(self, strategy_class, a, b, delta) -> None:
        """Test AND monotonicity."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        # Ensure a' ≥ a and b' ≥ b
        a_prime = np.minimum(a + delta, 1.0)
        b_prime = np.minimum(b + delta, 1.0)

        result_ab = strategy.compile_and(a, b)
        result_ab_prime = strategy.compile_and(a_prime, b_prime)

        # AND should be monotonic: AND(a', b') ≥ AND(a, b)
        assert np.all(result_ab_prime >= result_ab - 1e-6), \
            "AND failed monotonicity"

    @given(a=fuzzy_arrays, b=fuzzy_arrays, delta=st.floats(min_value=0.0, max_value=0.5))
    @settings(max_examples=50, deadline=1000)
    def test_or_monotonic(self, strategy_class, a, b, delta) -> None:
        """Test OR monotonicity."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        # Ensure a' ≥ a and b' ≥ b
        a_prime = np.minimum(a + delta, 1.0)
        b_prime = np.minimum(b + delta, 1.0)

        result_ab = strategy.compile_or(a, b)
        result_ab_prime = strategy.compile_or(a_prime, b_prime)

        # OR should be monotonic: OR(a', b') ≥ OR(a, b)
        assert np.all(result_ab_prime >= result_ab - 1e-6), \
            "OR failed monotonicity"


class TestAbsorptionLaws:
    """Test absorption laws (only for Gödel logic - min/max t-norms)."""

    @given(a=fuzzy_arrays, b=fuzzy_arrays)
    @settings(max_examples=50, deadline=1000)
    def test_absorption_and_or(self, a, b) -> None:
        """Test a OR (a AND b) = a for Gödel logic."""
        backend = create_backend("numpy")
        strategy = GodelStrategy(backend)

        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        # a OR (a AND b)
        result = strategy.compile_or(a, strategy.compile_and(a, b))

        np.testing.assert_allclose(result, a, rtol=1e-5, atol=1e-6)

    @given(a=fuzzy_arrays, b=fuzzy_arrays)
    @settings(max_examples=50, deadline=1000)
    def test_absorption_or_and(self, a, b) -> None:
        """Test a AND (a OR b) = a for Gödel logic."""
        backend = create_backend("numpy")
        strategy = GodelStrategy(backend)

        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        # a AND (a OR b)
        result = strategy.compile_and(a, strategy.compile_or(a, b))

        np.testing.assert_allclose(result, a, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
class TestBoundaryConditions:
    """Test boundary conditions at 0 and 1."""

    @given(a=fuzzy_arrays)
    @settings(max_examples=50, deadline=1000)
    def test_not_boundaries(self, strategy_class, a) -> None:
        """Test NOT(0) = 1 and NOT(1) = 0."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        zeros = backend.zeros(a.shape)
        ones = backend.ones(a.shape)

        not_zero = strategy.compile_not(zeros)
        not_one = strategy.compile_not(ones)

        np.testing.assert_allclose(not_zero, ones, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(not_one, zeros, rtol=1e-5, atol=1e-6)


class TestIdempotentProperties:
    """Test idempotent properties (specific to certain strategies)."""

    @given(a=fuzzy_arrays)
    @settings(max_examples=50, deadline=1000)
    def test_godel_and_idempotent(self, a) -> None:
        """Test Gödel AND is idempotent: AND(a, a) = a."""
        backend = create_backend("numpy")
        strategy = GodelStrategy(backend)
        result = strategy.compile_and(a, a)

        np.testing.assert_allclose(result, a, rtol=1e-5, atol=1e-6)

    @given(a=fuzzy_arrays)
    @settings(max_examples=50, deadline=1000)
    def test_godel_or_idempotent(self, a) -> None:
        """Test Gödel OR is idempotent: OR(a, a) = a."""
        backend = create_backend("numpy")
        strategy = GodelStrategy(backend)
        result = strategy.compile_or(a, a)

        np.testing.assert_allclose(result, a, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("strategy_class", ALL_STRATEGIES)
class TestQuantifierProperties:
    """Test quantifier semantic properties."""

    @given(pred=fuzzy_arrays)
    @settings(max_examples=50, deadline=1000)
    def test_exists_upper_bound(self, strategy_class, pred) -> None:
        """Test EXISTS(P) ≤ 1."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        # Reshape to 2D for axis reduction
        pred_2d = pred.reshape(-1, 1) if pred.ndim == 1 else pred

        result = strategy.compile_exists(pred_2d, axis=0)

        assert np.all(result <= 1.0 + 1e-6), "EXISTS exceeded upper bound"

    @given(pred=fuzzy_arrays)
    @settings(max_examples=50, deadline=1000)
    def test_forall_lower_bound(self, strategy_class, pred) -> None:
        """Test FORALL(P) ≥ 0."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        # Reshape to 2D for axis reduction
        pred_2d = pred.reshape(-1, 1) if pred.ndim == 1 else pred

        result = strategy.compile_forall(pred_2d, axis=0)

        assert np.all(result >= -1e-6), "FORALL went below lower bound"

    @given(pred=fuzzy_arrays)
    @settings(max_examples=50, deadline=1000)
    def test_exists_geq_forall(self, strategy_class, pred) -> None:
        """Test EXISTS(P) ≥ FORALL(P)."""
        backend = create_backend("numpy")
        strategy = strategy_class(backend)

        # Reshape to 2D for axis reduction
        pred_2d = pred.reshape(-1, 1) if pred.ndim == 1 else pred

        exists_result = strategy.compile_exists(pred_2d, axis=0)
        forall_result = strategy.compile_forall(pred_2d, axis=0)

        assert np.all(exists_result >= forall_result - 1e-6), \
            "EXISTS < FORALL violation"


class TestImplicationProperties:
    """Test implication semantic properties (HardBoolean and Łukasiewicz only).

    Note: SoftDifferentiable, Product, and Gödel use max(1-a, b) approximation
    which doesn't satisfy full residuated implication properties. This is a known
    limitation for differentiability and numerical stability.
    """

    @given(a=fuzzy_arrays, b=fuzzy_arrays)
    @settings(max_examples=50, deadline=1000)
    def test_lukasiewicz_modus_ponens(self, a, b) -> None:
        """Test modus ponens for Łukasiewicz: (a AND (a -> b)) ≤ b."""
        backend = create_backend("numpy")
        strategy = LukasiewiczStrategy(backend)

        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        impl = strategy.compile_implies(a, b)
        left = strategy.compile_and(a, impl)

        # Modus ponens: a ∧ (a → b) ≤ b
        assert np.all(left <= b + 1e-5), "Modus ponens failed"

    @given(a=fuzzy_arrays)
    @settings(max_examples=50, deadline=1000)
    def test_lukasiewicz_tautology(self, a) -> None:
        """Test tautology for Łukasiewicz: a -> a = 1."""
        backend = create_backend("numpy")
        strategy = LukasiewiczStrategy(backend)

        result = strategy.compile_implies(a, a)

        np.testing.assert_allclose(result, np.ones_like(a), rtol=1e-5, atol=1e-6)


class TestStrategySpecificProperties:
    """Test strategy-specific mathematical properties."""

    @given(a=fuzzy_arrays, b=fuzzy_arrays)
    @settings(max_examples=50, deadline=1000)
    def test_hard_boolean_discrete(self, a, b) -> None:
        """Test HardBooleanStrategy produces only 0 and 1."""
        backend = create_backend("numpy")
        strategy = HardBooleanStrategy(backend)
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        and_result = strategy.compile_and(a, b)
        or_result = strategy.compile_or(a, b)
        not_result = strategy.compile_not(a)

        # All results should be exactly 0 or 1
        assert np.all((and_result == 0) | (and_result == 1)), \
            "HardBoolean AND not discrete"
        assert np.all((or_result == 0) | (or_result == 1)), \
            "HardBoolean OR not discrete"
        assert np.all((not_result == 0) | (not_result == 1)), \
            "HardBoolean NOT not discrete"

    @given(a=fuzzy_arrays, b=fuzzy_arrays)
    @settings(max_examples=50, deadline=1000)
    def test_soft_product_equivalence(self, a, b) -> None:
        """Test SoftDifferentiable ≈ Product for AND."""
        backend = create_backend("numpy")
        soft = SoftDifferentiableStrategy(backend)
        product = ProductStrategy(backend)

        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        soft_result = soft.compile_and(a, b)
        product_result = product.compile_and(a, b)

        np.testing.assert_allclose(soft_result, product_result, rtol=1e-5, atol=1e-6)

    @given(a=fuzzy_arrays, b=fuzzy_arrays)
    @settings(max_examples=50, deadline=1000)
    def test_lukasiewicz_strict_boundaries(self, a, b) -> None:
        """Test Łukasiewicz enforces strict [0, 1] boundaries."""
        backend = create_backend("numpy")
        strategy = LukasiewiczStrategy(backend)
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]

        and_result = strategy.compile_and(a, b)
        or_result = strategy.compile_or(a, b)

        # Results must be in [0, 1]
        assert np.all((and_result >= -1e-6) & (and_result <= 1.0 + 1e-6)), \
            "Łukasiewicz AND out of bounds"
        assert np.all((or_result >= -1e-6) & (or_result <= 1.0 + 1e-6)), \
            "Łukasiewicz OR out of bounds"
