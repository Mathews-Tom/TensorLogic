"""Tests for Łukasiewicz fuzzy logic compilation strategy.

Tests mathematical correctness, boundary conditions, and protocol compliance for
the Łukasiewicz (strict) fuzzy logic compilation strategy.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlogic.backends import create_backend
from tensorlogic.compilation import CompilationStrategy, LukasiewiczStrategy


@pytest.fixture
def numpy_backend():
    """Create NumPy backend for testing."""
    return create_backend("numpy")


@pytest.fixture
def strategy(numpy_backend):
    """Create Łukasiewicz fuzzy strategy with NumPy backend."""
    return LukasiewiczStrategy(numpy_backend)


class TestLukasiewiczStrategyOperations:
    """Test core logical operations."""

    def test_compile_and_bounded_difference(self, strategy, numpy_backend) -> None:
        """Test AND operation uses bounded difference: max(0, a + b - 1)."""
        a = np.array([0.8, 0.6, 0.3, 0.5])
        b = np.array([0.9, 0.4, 0.7, 0.6])

        result = strategy.compile_and(a, b)
        numpy_backend.eval(result)

        # max(0, a+b-1): [0.7, 0.0, 0.0, 0.1]
        expected = np.array([0.7, 0.0, 0.0, 0.1])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_compile_or_bounded_sum(self, strategy, numpy_backend) -> None:
        """Test OR operation uses bounded sum: min(1, a + b)."""
        a = np.array([0.3, 0.5, 0.6, 0.2])
        b = np.array([0.4, 0.6, 0.8, 0.3])

        result = strategy.compile_or(a, b)
        numpy_backend.eval(result)

        # min(1, a+b): [0.7, 1.0, 1.0, 0.5]
        expected = np.array([0.7, 1.0, 1.0, 0.5])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_compile_not_complement(self, strategy, numpy_backend) -> None:
        """Test NOT operation uses complement (1 - a)."""
        a = np.array([0.8, 0.3, 0.5])

        result = strategy.compile_not(a)
        numpy_backend.eval(result)

        expected = np.array([0.2, 0.7, 0.5])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_compile_implies_lukasiewicz_semantics(self, strategy, numpy_backend) -> None:
        """Test IMPLIES uses Łukasiewicz implication: min(1, 1 - a + b)."""
        a = np.array([0.9, 0.2, 0.5, 0.6])
        b = np.array([0.3, 0.7, 0.5, 0.6])

        result = strategy.compile_implies(a, b)
        numpy_backend.eval(result)

        # min(1, 1-a+b): [0.4, 1.0, 1.0, 1.0]
        expected = np.array([0.4, 1.0, 1.0, 1.0])
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


class TestLukasiewiczStrategyMathematicalProperties:
    """Test mathematical properties of Łukasiewicz fuzzy logic."""

    def test_and_strict_boundary(self, strategy, numpy_backend) -> None:
        """Test AND returns 0 when a + b ≤ 1 (strict boundary)."""
        a = np.array([0.4, 0.5, 0.3])
        b = np.array([0.6, 0.5, 0.6])

        result = strategy.compile_and(a, b)
        numpy_backend.eval(result)

        # max(0, a+b-1): [0.0, 0.0, 0.0] (all sums ≤ 1)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_or_saturation(self, strategy, numpy_backend) -> None:
        """Test OR saturates at 1 when a + b ≥ 1."""
        a = np.array([0.6, 0.7, 0.9])
        b = np.array([0.5, 0.8, 0.2])

        result = strategy.compile_or(a, b)
        numpy_backend.eval(result)

        # min(1, a+b): [1.0, 1.0, 1.0] (all sums ≥ 1)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

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


class TestLukasiewiczStrategyBoundaryConditions:
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
        """Test boundary cases for Łukasiewicz implication."""
        # IMPLIES(0, b) = 1 for any b
        result = strategy.compile_implies(np.array([0.0, 0.0]), np.array([0.5, 0.9]))
        numpy_backend.eval(result)
        np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-6)

        # IMPLIES(a, 1) = 1 for any a
        result = strategy.compile_implies(np.array([0.5, 0.9]), np.array([1.0, 1.0]))
        numpy_backend.eval(result)
        np.testing.assert_allclose(result, [1.0, 1.0], atol=1e-6)

        # IMPLIES(1, 0) = 0
        result = strategy.compile_implies(np.array([1.0]), np.array([0.0]))
        numpy_backend.eval(result)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_quantifiers_single_element(self, strategy, numpy_backend) -> None:
        """Test quantifiers with single element."""
        predicate = np.array([[0.7]])

        exists = strategy.compile_exists(predicate, axis=1)
        forall = strategy.compile_forall(predicate, axis=1)
        numpy_backend.eval(exists, forall)

        np.testing.assert_allclose(exists, [0.7], atol=1e-6)
        np.testing.assert_allclose(forall, [0.7], atol=1e-6)


class TestLukasiewiczStrategyProtocolCompliance:
    """Test protocol compliance."""

    def test_is_differentiable_property(self, strategy) -> None:
        """Test is_differentiable property returns True."""
        assert strategy.is_differentiable is True
        assert isinstance(strategy.is_differentiable, bool)

    def test_name_property(self, strategy) -> None:
        """Test name property returns correct identifier."""
        assert strategy.name == "lukasiewicz"
        assert isinstance(strategy.name, str)

    def test_implements_compilation_strategy_protocol(self, strategy) -> None:
        """Test strategy implements CompilationStrategy protocol."""
        assert isinstance(strategy, CompilationStrategy)

        assert hasattr(strategy, "compile_and")
        assert hasattr(strategy, "compile_or")
        assert hasattr(strategy, "compile_not")
        assert hasattr(strategy, "compile_implies")
        assert hasattr(strategy, "compile_exists")
        assert hasattr(strategy, "compile_forall")
        assert hasattr(strategy, "is_differentiable")
        assert hasattr(strategy, "name")


class TestLukasiewiczStrategyInitialization:
    """Test strategy initialization."""

    def test_init_with_backend(self, numpy_backend) -> None:
        """Test initialization with explicit backend."""
        strategy = LukasiewiczStrategy(numpy_backend)
        assert strategy._backend is numpy_backend

    def test_init_without_backend_creates_default(self) -> None:
        """Test initialization without backend creates default NumPy backend."""
        strategy = LukasiewiczStrategy()
        assert strategy._backend is not None
        assert hasattr(strategy._backend, "minimum")

    def test_strategy_usable_after_default_init(self) -> None:
        """Test strategy is usable after default initialization."""
        strategy = LukasiewiczStrategy()

        a = np.array([0.8, 0.7])
        b = np.array([0.6, 0.3])

        result = strategy.compile_and(a, b)
        strategy._backend.eval(result)

        assert result is not None
        # max(0, a+b-1): [0.4, 0.0]
        np.testing.assert_allclose(result, [0.4, 0.0], atol=1e-6)


class TestLukasiewiczStrategyIntegration:
    """Integration tests with factory and other components."""

    def test_strategy_via_factory(self) -> None:
        """Test creating strategy via factory."""
        from tensorlogic.compilation import create_strategy

        strategy = create_strategy("lukasiewicz")

        assert isinstance(strategy, LukasiewiczStrategy)
        assert strategy.name == "lukasiewicz"
        assert strategy.is_differentiable

    def test_strategy_registered_in_factory(self) -> None:
        """Test that lukasiewicz is registered in factory."""
        from tensorlogic.compilation import get_available_strategies

        strategies = get_available_strategies()
        assert "lukasiewicz" in strategies

    def test_lukasiewicz_vs_godel_comparison(self) -> None:
        """Test that Łukasiewicz strategy produces different results than Gödel."""
        from tensorlogic.compilation import create_strategy

        lukasiewicz = create_strategy("lukasiewicz")
        godel = create_strategy("godel")

        a = np.array([0.6, 0.8])
        b = np.array([0.5, 0.4])

        # Łukasiewicz AND: max(0, a+b-1) = [0.1, 0.2]
        luk_result = lukasiewicz.compile_and(a, b)
        lukasiewicz._backend.eval(luk_result)

        # Gödel AND: min(a, b) = [0.5, 0.4]
        god_result = godel.compile_and(a, b)
        godel._backend.eval(god_result)

        # Results should differ
        assert not np.allclose(luk_result, god_result)
        np.testing.assert_allclose(luk_result, [0.1, 0.2], atol=1e-6)
        np.testing.assert_allclose(god_result, [0.5, 0.4], atol=1e-6)
