"""Tests for Gödel fuzzy logic compilation strategy.

Tests mathematical correctness, differentiability, and protocol compliance for
the Gödel fuzzy logic compilation strategy using min/max t-norms.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlogic.backends import create_backend
from tensorlogic.compilation import CompilationStrategy, GodelStrategy


@pytest.fixture
def numpy_backend():
    """Create NumPy backend for testing."""
    return create_backend("numpy")


@pytest.fixture
def strategy(numpy_backend):
    """Create Gödel fuzzy strategy with NumPy backend."""
    return GodelStrategy(numpy_backend)


class TestGodelStrategyOperations:
    """Test core logical operations."""

    def test_compile_and_minimum(self, strategy, numpy_backend) -> None:
        """Test AND operation uses minimum semantics."""
        a = np.array([0.8, 0.6, 0.3])
        b = np.array([0.9, 0.4, 0.7])

        result = strategy.compile_and(a, b)
        numpy_backend.eval(result)

        expected = np.array([0.8, 0.4, 0.3])  # min(a, b)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_compile_or_maximum(self, strategy, numpy_backend) -> None:
        """Test OR operation uses maximum semantics."""
        a = np.array([0.3, 0.5, 0.2])
        b = np.array([0.4, 0.6, 0.3])

        result = strategy.compile_or(a, b)
        numpy_backend.eval(result)

        expected = np.array([0.4, 0.6, 0.3])  # max(a, b)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_compile_not_complement(self, strategy, numpy_backend) -> None:
        """Test NOT operation uses complement (1 - a)."""
        a = np.array([0.8, 0.3, 0.5])

        result = strategy.compile_not(a)
        numpy_backend.eval(result)

        expected = np.array([0.2, 0.7, 0.5])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_compile_implies_godel_semantics(self, strategy, numpy_backend) -> None:
        """Test IMPLIES uses Gödel implication: 1 if a ≤ b, else b."""
        a = np.array([0.9, 0.2, 0.5, 0.6])
        b = np.array([0.3, 0.7, 0.5, 0.6])

        result = strategy.compile_implies(a, b)
        numpy_backend.eval(result)

        # Gödel: if a ≤ b then 1 else b
        # [0.9 ≤ 0.3? No → 0.3, 0.2 ≤ 0.7? Yes → 1.0, 0.5 ≤ 0.5? Yes → 1.0, 0.6 ≤ 0.6? Yes → 1.0]
        expected = np.array([0.3, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_compile_exists_max_reduction(self, strategy, numpy_backend) -> None:
        """Test EXISTS uses max reduction over axis."""
        predicate = np.array([[0.2, 0.8, 0.3], [0.1, 0.4, 0.2]])

        result = strategy.compile_exists(predicate, axis=1)
        numpy_backend.eval(result)

        expected = np.array([0.8, 0.4])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_compile_forall_min_reduction(self, strategy, numpy_backend) -> None:
        """Test FORALL uses min reduction over axis."""
        predicate = np.array([[0.9, 0.8, 0.7], [0.6, 0.5, 0.9]])

        result = strategy.compile_forall(predicate, axis=1)
        numpy_backend.eval(result)

        expected = np.array([0.7, 0.5])
        np.testing.assert_allclose(result, expected, atol=1e-6)


class TestGodelStrategyMathematicalProperties:
    """Test mathematical properties of Gödel fuzzy logic."""

    def test_and_commutative(self, strategy, numpy_backend) -> None:
        """Test AND(a, b) = AND(b, a)."""
        a = np.array([0.7, 0.4])
        b = np.array([0.3, 0.8])

        result1 = strategy.compile_and(a, b)
        result2 = strategy.compile_and(b, a)
        numpy_backend.eval(result1, result2)

        np.testing.assert_allclose(result1, result2, atol=1e-6)

    def test_or_commutative(self, strategy, numpy_backend) -> None:
        """Test OR(a, b) = OR(b, a)."""
        a = np.array([0.7, 0.4])
        b = np.array([0.3, 0.8])

        result1 = strategy.compile_or(a, b)
        result2 = strategy.compile_or(b, a)
        numpy_backend.eval(result1, result2)

        np.testing.assert_allclose(result1, result2, atol=1e-6)

    def test_and_idempotent(self, strategy, numpy_backend) -> None:
        """Test AND(a, a) = a (idempotent)."""
        a = np.array([0.7, 0.4, 0.9])

        result = strategy.compile_and(a, a)
        numpy_backend.eval(result)

        np.testing.assert_allclose(result, a, atol=1e-6)

    def test_or_idempotent(self, strategy, numpy_backend) -> None:
        """Test OR(a, a) = a (idempotent)."""
        a = np.array([0.7, 0.4, 0.9])

        result = strategy.compile_or(a, a)
        numpy_backend.eval(result)

        np.testing.assert_allclose(result, a, atol=1e-6)

    def test_not_involution(self, strategy, numpy_backend) -> None:
        """Test NOT(NOT(a)) = a."""
        a = np.array([0.8, 0.3, 0.5])

        result = strategy.compile_not(strategy.compile_not(a))
        numpy_backend.eval(result)

        np.testing.assert_allclose(result, a, atol=1e-6)

    def test_and_identity(self, strategy, numpy_backend) -> None:
        """Test AND(a, 1) = a."""
        a = np.array([0.7, 0.4, 0.9])
        ones = numpy_backend.ones((3,))

        result = strategy.compile_and(a, ones)
        numpy_backend.eval(result)

        np.testing.assert_allclose(result, a, atol=1e-6)

    def test_and_annihilator(self, strategy, numpy_backend) -> None:
        """Test AND(a, 0) = 0."""
        a = np.array([0.7, 0.4, 0.9])
        zeros = numpy_backend.zeros((3,))

        result = strategy.compile_and(a, zeros)
        numpy_backend.eval(result)

        np.testing.assert_allclose(result, zeros, atol=1e-6)

    def test_or_identity(self, strategy, numpy_backend) -> None:
        """Test OR(a, 0) = a."""
        a = np.array([0.7, 0.4, 0.9])
        zeros = numpy_backend.zeros((3,))

        result = strategy.compile_or(a, zeros)
        numpy_backend.eval(result)

        np.testing.assert_allclose(result, a, atol=1e-6)

    def test_or_annihilator(self, strategy, numpy_backend) -> None:
        """Test OR(a, 1) = 1."""
        a = np.array([0.7, 0.4, 0.9])
        ones = numpy_backend.ones((3,))

        result = strategy.compile_or(a, ones)
        numpy_backend.eval(result)

        np.testing.assert_allclose(result, ones, atol=1e-6)

    def test_demorgan_and(self, strategy, numpy_backend) -> None:
        """Test De Morgan's law: NOT(a AND b) = NOT(a) OR NOT(b) in Gödel logic."""
        a = np.array([0.8, 0.6])
        b = np.array([0.7, 0.5])

        # NOT(a AND b) = NOT(min(a,b)) = 1 - min(a,b)
        left = strategy.compile_not(strategy.compile_and(a, b))
        numpy_backend.eval(left)

        # NOT(a) OR NOT(b) = max(1-a, 1-b) = 1 - min(a,b)
        right = strategy.compile_or(strategy.compile_not(a), strategy.compile_not(b))
        numpy_backend.eval(right)

        # Should be exactly equal in Gödel fuzzy logic
        np.testing.assert_allclose(left, right, atol=1e-6)

    def test_and_monotonic(self, strategy, numpy_backend) -> None:
        """Test monotonicity: if a ≤ a' and b ≤ b', then AND(a,b) ≤ AND(a',b')."""
        a = np.array([0.3, 0.4])
        a_prime = np.array([0.5, 0.6])
        b = np.array([0.2, 0.3])
        b_prime = np.array([0.4, 0.5])

        result1 = strategy.compile_and(a, b)
        result2 = strategy.compile_and(a_prime, b_prime)
        numpy_backend.eval(result1, result2)

        # min(0.3, 0.2) ≤ min(0.5, 0.4), min(0.4, 0.3) ≤ min(0.6, 0.5)
        assert np.all(result1 <= result2)


class TestGodelStrategyBoundaryConditions:
    """Test boundary and edge cases."""

    def test_operations_with_zeros(self, strategy, numpy_backend) -> None:
        """Test operations with zero inputs."""
        zeros = numpy_backend.zeros((3,))

        # NOT(0) = 1
        result_not = strategy.compile_not(zeros)
        numpy_backend.eval(result_not)
        np.testing.assert_allclose(result_not, np.ones(3), atol=1e-6)

        # AND(0, 0) = 0
        result_and = strategy.compile_and(zeros, zeros)
        numpy_backend.eval(result_and)
        np.testing.assert_allclose(result_and, zeros, atol=1e-6)

        # OR(0, 0) = 0
        result_or = strategy.compile_or(zeros, zeros)
        numpy_backend.eval(result_or)
        np.testing.assert_allclose(result_or, zeros, atol=1e-6)

    def test_operations_with_ones(self, strategy, numpy_backend) -> None:
        """Test operations with one inputs."""
        ones = numpy_backend.ones((3,))

        # NOT(1) = 0
        result_not = strategy.compile_not(ones)
        numpy_backend.eval(result_not)
        np.testing.assert_allclose(result_not, np.zeros(3), atol=1e-6)

        # AND(1, 1) = 1
        result_and = strategy.compile_and(ones, ones)
        numpy_backend.eval(result_and)
        np.testing.assert_allclose(result_and, ones, atol=1e-6)

        # OR(1, 1) = 1
        result_or = strategy.compile_or(ones, ones)
        numpy_backend.eval(result_or)
        np.testing.assert_allclose(result_or, ones, atol=1e-6)

    def test_implies_boundary_cases(self, strategy, numpy_backend) -> None:
        """Test boundary cases for Gödel implication."""
        # IMPLIES(0, b) = 1 for any b
        result = strategy.compile_implies(np.array([0.0, 0.0]), np.array([0.5, 0.9]))
        numpy_backend.eval(result)
        np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-6)

        # IMPLIES(a, 1) = 1 for any a
        result = strategy.compile_implies(np.array([0.5, 0.9]), np.array([1.0, 1.0]))
        numpy_backend.eval(result)
        np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-6)

        # IMPLIES(1, 0) = 0 (worst case)
        result = strategy.compile_implies(np.array([1.0]), np.array([0.0]))
        numpy_backend.eval(result)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_quantifiers_single_element(self, strategy, numpy_backend) -> None:
        """Test quantifiers with single element."""
        predicate = np.array([[0.7]])

        exists = strategy.compile_exists(predicate, axis=1)
        forall = strategy.compile_forall(predicate, axis=1)
        numpy_backend.eval(exists, forall)

        # Single element: both should equal the element (max/min of one element)
        np.testing.assert_allclose(exists, [0.7], atol=1e-6)
        np.testing.assert_allclose(forall, [0.7], atol=1e-6)


class TestGodelStrategyProtocolCompliance:
    """Test protocol compliance."""

    def test_is_differentiable_property(self, strategy) -> None:
        """Test is_differentiable property returns True."""
        assert strategy.is_differentiable is True
        assert isinstance(strategy.is_differentiable, bool)

    def test_name_property(self, strategy) -> None:
        """Test name property returns correct identifier."""
        assert strategy.name == "godel"
        assert isinstance(strategy.name, str)

    def test_implements_compilation_strategy_protocol(self, strategy) -> None:
        """Test strategy implements CompilationStrategy protocol."""
        assert isinstance(strategy, CompilationStrategy)

        # Verify all protocol methods exist
        assert hasattr(strategy, "compile_and")
        assert hasattr(strategy, "compile_or")
        assert hasattr(strategy, "compile_not")
        assert hasattr(strategy, "compile_implies")
        assert hasattr(strategy, "compile_exists")
        assert hasattr(strategy, "compile_forall")
        assert hasattr(strategy, "is_differentiable")
        assert hasattr(strategy, "name")


class TestGodelStrategyInitialization:
    """Test strategy initialization."""

    def test_init_with_backend(self, numpy_backend) -> None:
        """Test initialization with explicit backend."""
        strategy = GodelStrategy(numpy_backend)
        assert strategy._backend is numpy_backend

    def test_init_without_backend_creates_default(self) -> None:
        """Test initialization without backend creates default NumPy backend."""
        strategy = GodelStrategy()
        assert strategy._backend is not None
        assert hasattr(strategy._backend, "minimum")  # Check it's a valid backend

    def test_strategy_usable_after_default_init(self) -> None:
        """Test strategy is usable after default initialization."""
        strategy = GodelStrategy()

        a = np.array([0.5, 0.7])
        b = np.array([0.6, 0.3])

        result = strategy.compile_and(a, b)
        strategy._backend.eval(result)

        # Just verify it runs without errors
        assert result is not None
        # Should be min([0.5, 0.7], [0.6, 0.3]) = [0.5, 0.3]
        np.testing.assert_allclose(result, [0.5, 0.3], atol=1e-6)


class TestGodelStrategyIntegration:
    """Integration tests with factory and other components."""

    def test_strategy_via_factory(self) -> None:
        """Test creating strategy via factory."""
        from tensorlogic.compilation import create_strategy

        strategy = create_strategy("godel")

        assert isinstance(strategy, GodelStrategy)
        assert strategy.name == "godel"
        assert strategy.is_differentiable

    def test_strategy_registered_in_factory(self) -> None:
        """Test that godel is registered in factory."""
        from tensorlogic.compilation import get_available_strategies

        strategies = get_available_strategies()
        assert "godel" in strategies

    def test_godel_vs_soft_comparison(self) -> None:
        """Test that Gödel strategy produces different results than soft."""
        from tensorlogic.compilation import create_strategy

        godel = create_strategy("godel")
        soft = create_strategy("soft_differentiable")

        a = np.array([0.5, 0.6])
        b = np.array([0.4, 0.8])

        # Gödel AND: min([0.5, 0.6], [0.4, 0.8]) = [0.4, 0.6]
        godel_result = godel.compile_and(a, b)
        godel._backend.eval(godel_result)

        # Soft AND: multiply([0.5, 0.6], [0.4, 0.8]) = [0.2, 0.48]
        soft_result = soft.compile_and(a, b)
        soft._backend.eval(soft_result)

        # Results should differ: Gödel uses min, soft uses product
        assert not np.allclose(godel_result, soft_result)
        np.testing.assert_allclose(godel_result, [0.4, 0.6], atol=1e-6)
        np.testing.assert_allclose(soft_result, [0.2, 0.48], atol=1e-6)
