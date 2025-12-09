"""Tests for hard boolean compilation strategy.

Tests mathematical correctness, binary output, and protocol compliance for
the hard boolean (non-differentiable) compilation strategy.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlogic.backends import create_backend
from tensorlogic.compilation import CompilationStrategy, HardBooleanStrategy


@pytest.fixture
def numpy_backend():
    """Create NumPy backend for testing."""
    return create_backend("numpy")


@pytest.fixture
def strategy(numpy_backend):
    """Create hard boolean strategy with NumPy backend."""
    return HardBooleanStrategy(numpy_backend)


class TestHardStrategyOperations:
    """Test core logical operations."""

    def test_compile_and_binary_output(self, strategy, numpy_backend) -> None:
        """Test AND operation produces binary {0, 1} output."""
        a = np.array([0.8, 0.6, 0.3, 0.0])
        b = np.array([0.9, 0.4, 0.7, 0.5])

        result = strategy.compile_and(a, b)
        numpy_backend.eval(result)

        # All outputs should be binary
        assert np.all(np.isin(result, [0.0, 1.0]))
        # step(a*b): [1, 1, 1, 0] (all products > 0 except last)
        expected = np.array([1.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_compile_or_binary_output(self, strategy, numpy_backend) -> None:
        """Test OR operation produces binary {0, 1} output."""
        a = np.array([0.3, 0.5, 0.0, 0.0])
        b = np.array([0.4, 0.0, 0.6, 0.0])

        result = strategy.compile_or(a, b)
        numpy_backend.eval(result)

        # All outputs should be binary
        assert np.all(np.isin(result, [0.0, 1.0]))
        # step(a+b): [1, 1, 1, 0] (all sums > 0 except last)
        expected = np.array([1.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_compile_not_binary_output(self, strategy, numpy_backend) -> None:
        """Test NOT operation produces binary {0, 1} output."""
        a = np.array([0.8, 0.0, -0.5, 0.3])

        result = strategy.compile_not(a)
        numpy_backend.eval(result)

        # All outputs should be binary
        assert np.all(np.isin(result, [0.0, 1.0]))
        # 1 - step(a): [0, 1, 1, 0] (step gives [1, 0, 0, 1])
        expected = np.array([0.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_compile_implies_binary_output(self, strategy, numpy_backend) -> None:
        """Test IMPLIES produces binary {0, 1} output."""
        a = np.array([0.9, 0.0, 0.5, 0.0])
        b = np.array([0.0, 0.7, 0.6, 0.0])

        result = strategy.compile_implies(a, b)
        numpy_backend.eval(result)

        # All outputs should be binary
        assert np.all(np.isin(result, [0.0, 1.0]))
        # (a → b) ≡ (¬a ∨ b): [0, 1, 1, 1]
        # step(a): [1, 0, 1, 0], NOT: [0, 1, 0, 1], step(b): [0, 1, 1, 0]
        # step(NOT(a) + step(b)): [0, 1, 1, 1]
        expected = np.array([0.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_compile_exists_binary_output(self, strategy, numpy_backend) -> None:
        """Test EXISTS produces binary {0, 1} output."""
        predicate = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.9, 0.8, 0.7]])

        result = strategy.compile_exists(predicate, axis=1)
        numpy_backend.eval(result)

        # All outputs should be binary
        assert np.all(np.isin(result, [0.0, 1.0]))
        # step(sum): [0, 1, 1] (sum > 0 for last two rows)
        expected = np.array([0.0, 1.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_compile_forall_binary_output(self, strategy, numpy_backend) -> None:
        """Test FORALL produces binary {0, 1} output."""
        predicate = np.array([[1.0, 1.0, 1.0], [0.9, 0.8, 0.7], [0.5, 0.9, 1.0]])

        result = strategy.compile_forall(predicate, axis=1)
        numpy_backend.eval(result)

        # All outputs should be binary
        assert np.all(np.isin(result, [0.0, 1.0]))
        # step(prod - 0.99): [1, 0, 0]
        # prod: [1.0, 0.504, 0.45], prod - 0.99: [0.01, -0.486, -0.54]
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)


class TestHardStrategyMathematicalProperties:
    """Test mathematical properties of hard boolean logic."""

    def test_and_commutative(self, strategy, numpy_backend) -> None:
        """Test AND(a, b) = AND(b, a)."""
        a = np.array([0.7, 0.4, 0.0])
        b = np.array([0.3, 0.8, 0.5])

        result1 = strategy.compile_and(a, b)
        result2 = strategy.compile_and(b, a)
        numpy_backend.eval(result1, result2)

        np.testing.assert_array_equal(result1, result2)

    def test_or_commutative(self, strategy, numpy_backend) -> None:
        """Test OR(a, b) = OR(b, a)."""
        a = np.array([0.7, 0.4, 0.0])
        b = np.array([0.3, 0.0, 0.5])

        result1 = strategy.compile_or(a, b)
        result2 = strategy.compile_or(b, a)
        numpy_backend.eval(result1, result2)

        np.testing.assert_array_equal(result1, result2)

    def test_not_involution_with_positive(self, strategy, numpy_backend) -> None:
        """Test NOT(NOT(a)) = step(a) for positive values."""
        a = np.array([0.8, 0.3, 0.5])

        result = strategy.compile_not(strategy.compile_not(a))
        expected = numpy_backend.step(a)
        numpy_backend.eval(result, expected)

        np.testing.assert_array_equal(result, expected)

    def test_and_with_zero(self, strategy, numpy_backend) -> None:
        """Test AND(a, 0) = 0 (annihilator)."""
        a = np.array([0.7, 0.4, 0.9])
        zeros = np.array([0.0, 0.0, 0.0])

        result = strategy.compile_and(a, zeros)
        numpy_backend.eval(result)

        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_and_with_positive_identity(self, strategy, numpy_backend) -> None:
        """Test AND(a, 1) = step(a) (identity-like behavior)."""
        a = np.array([0.7, 0.4, 0.0])
        ones = np.array([1.0, 1.0, 1.0])

        result = strategy.compile_and(a, ones)
        expected = numpy_backend.step(a)
        numpy_backend.eval(result, expected)

        np.testing.assert_array_equal(result, expected)

    def test_or_with_zero(self, strategy, numpy_backend) -> None:
        """Test OR(a, 0) = step(a) (identity-like behavior)."""
        a = np.array([0.7, 0.0, 0.9])
        zeros = np.array([0.0, 0.0, 0.0])

        result = strategy.compile_or(a, zeros)
        expected = numpy_backend.step(a)
        numpy_backend.eval(result, expected)

        np.testing.assert_array_equal(result, expected)

    def test_demorgan_and(self, strategy, numpy_backend) -> None:
        """Test De Morgan's law: NOT(a AND b) = NOT(a) OR NOT(b) in boolean logic."""
        a = np.array([0.8, 0.6, 0.0])
        b = np.array([0.7, 0.0, 0.5])

        # NOT(a AND b)
        left = strategy.compile_not(strategy.compile_and(a, b))
        numpy_backend.eval(left)

        # NOT(a) OR NOT(b)
        right = strategy.compile_or(strategy.compile_not(a), strategy.compile_not(b))
        numpy_backend.eval(right)

        # Should be exactly equal in hard boolean logic
        np.testing.assert_array_equal(left, right)


class TestHardStrategyBoundaryConditions:
    """Test boundary and edge cases."""

    def test_operations_with_zeros(self, strategy, numpy_backend) -> None:
        """Test operations with zero inputs."""
        zeros = np.array([0.0, 0.0, 0.0])

        # NOT(0) = 1
        result_not = strategy.compile_not(zeros)
        numpy_backend.eval(result_not)
        np.testing.assert_array_equal(result_not, np.array([1.0, 1.0, 1.0]))

        # AND(0, 0) = 0
        result_and = strategy.compile_and(zeros, zeros)
        numpy_backend.eval(result_and)
        np.testing.assert_array_equal(result_and, zeros)

        # OR(0, 0) = 0
        result_or = strategy.compile_or(zeros, zeros)
        numpy_backend.eval(result_or)
        np.testing.assert_array_equal(result_or, zeros)

    def test_operations_with_ones(self, strategy, numpy_backend) -> None:
        """Test operations with one inputs."""
        ones = np.array([1.0, 1.0, 1.0])

        # NOT(1) = 0
        result_not = strategy.compile_not(ones)
        numpy_backend.eval(result_not)
        np.testing.assert_array_equal(result_not, np.array([0.0, 0.0, 0.0]))

        # AND(1, 1) = 1
        result_and = strategy.compile_and(ones, ones)
        numpy_backend.eval(result_and)
        np.testing.assert_array_equal(result_and, ones)

        # OR(1, 1) = 1
        result_or = strategy.compile_or(ones, ones)
        numpy_backend.eval(result_or)
        np.testing.assert_array_equal(result_or, ones)

    def test_operations_with_negative_values(self, strategy, numpy_backend) -> None:
        """Test operations with negative inputs (should be treated as false)."""
        negative = np.array([-0.5, -1.0, -0.1])

        # step(negative) = 0, so NOT should give 1
        result_not = strategy.compile_not(negative)
        numpy_backend.eval(result_not)
        np.testing.assert_array_equal(result_not, np.array([1.0, 1.0, 1.0]))

        # AND(negative, positive) = 0 (product is negative)
        positive = np.array([0.5, 1.0, 0.1])
        result_and = strategy.compile_and(negative, positive)
        numpy_backend.eval(result_and)
        np.testing.assert_array_equal(result_and, np.array([0.0, 0.0, 0.0]))

    def test_quantifiers_single_element(self, strategy, numpy_backend) -> None:
        """Test quantifiers with single element."""
        predicate = np.array([[0.7]])

        exists = strategy.compile_exists(predicate, axis=1)
        forall = strategy.compile_forall(predicate, axis=1)
        numpy_backend.eval(exists, forall)

        # Single element: both should be 1 (0.7 > 0 and 0.7 - 0.99 < 0 but wait...)
        # Actually: sum([0.7]) = 0.7 > 0 → step = 1
        # prod([0.7]) = 0.7, 0.7 - 0.99 = -0.29 < 0 → step = 0
        np.testing.assert_array_equal(exists, [1.0])
        np.testing.assert_array_equal(forall, [0.0])

    def test_quantifiers_all_ones(self, strategy, numpy_backend) -> None:
        """Test quantifiers with all 1.0 values."""
        predicate = np.array([[1.0, 1.0, 1.0]])

        exists = strategy.compile_exists(predicate, axis=1)
        forall = strategy.compile_forall(predicate, axis=1)
        numpy_backend.eval(exists, forall)

        # All 1.0: sum = 3.0 > 0 → exists = 1, prod = 1.0, 1.0 - 0.99 = 0.01 > 0 → forall = 1
        np.testing.assert_array_equal(exists, [1.0])
        np.testing.assert_array_equal(forall, [1.0])


class TestHardStrategyProtocolCompliance:
    """Test protocol compliance."""

    def test_is_differentiable_property(self, strategy) -> None:
        """Test is_differentiable property returns False."""
        assert strategy.is_differentiable is False
        assert isinstance(strategy.is_differentiable, bool)

    def test_name_property(self, strategy) -> None:
        """Test name property returns correct identifier."""
        assert strategy.name == "hard_boolean"
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


class TestHardStrategyInitialization:
    """Test strategy initialization."""

    def test_init_with_backend(self, numpy_backend) -> None:
        """Test initialization with explicit backend."""
        strategy = HardBooleanStrategy(numpy_backend)
        assert strategy._backend is numpy_backend

    def test_init_without_backend_creates_default(self) -> None:
        """Test initialization without backend creates default NumPy backend."""
        strategy = HardBooleanStrategy()
        assert strategy._backend is not None
        assert hasattr(strategy._backend, "step")  # Check it's a valid backend

    def test_strategy_usable_after_default_init(self) -> None:
        """Test strategy is usable after default initialization."""
        strategy = HardBooleanStrategy()

        a = np.array([0.5, 0.7])
        b = np.array([0.6, 0.0])

        result = strategy.compile_and(a, b)
        strategy._backend.eval(result)

        # Just verify it runs without errors and produces binary output
        assert result is not None
        assert np.all(np.isin(result, [0.0, 1.0]))


class TestHardStrategyIntegration:
    """Integration tests with factory and other components."""

    def test_strategy_via_factory(self) -> None:
        """Test creating strategy via factory."""
        from tensorlogic.compilation import create_strategy

        strategy = create_strategy("hard_boolean")

        assert isinstance(strategy, HardBooleanStrategy)
        assert strategy.name == "hard_boolean"
        assert not strategy.is_differentiable

    def test_strategy_registered_in_factory(self) -> None:
        """Test that hard_boolean is registered in factory."""
        from tensorlogic.compilation import get_available_strategies

        strategies = get_available_strategies()
        assert "hard_boolean" in strategies

    def test_hard_vs_soft_comparison(self) -> None:
        """Test that hard strategy produces different results than soft."""
        from tensorlogic.compilation import create_strategy

        hard = create_strategy("hard_boolean")
        soft = create_strategy("soft_differentiable")

        a = np.array([0.5, 0.6])
        b = np.array([0.4, 0.3])

        # Hard AND: step(a*b) = step([0.2, 0.18]) = [1, 1]
        hard_result = hard.compile_and(a, b)
        hard._backend.eval(hard_result)

        # Soft AND: a*b = [0.2, 0.18]
        soft_result = soft.compile_and(a, b)
        soft._backend.eval(soft_result)

        # Results should differ: hard is binary, soft is continuous
        assert np.all(np.isin(hard_result, [0.0, 1.0]))
        assert not np.allclose(hard_result, soft_result)
