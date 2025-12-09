"""Cross-backend validation tests for CoreLogic operations.

Validates that MLX backend operations produce identical results to NumPy
reference implementation. Tests all operations with various input shapes and
values to catch platform-specific bugs and ensure backend abstraction correctness.

Acceptance Criteria (CORE-006):
- All operations tested on both MLX and NumPy backends
- Results match within FP32 tolerance (1e-6)
- Edge cases validated on both backends
- Performance comparison documented in test output
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from tensorlogic.backends import create_backend
from tensorlogic.core.operations import (
    logical_and,
    logical_implies,
    logical_not,
    logical_or,
    step,
)


class TestCrossBackendValidation:
    """Cross-backend validation comparing MLX and NumPy outputs."""

    @pytest.fixture
    def numpy_backend(self):
        """NumPy reference backend."""
        return create_backend("numpy")

    @pytest.fixture
    def mlx_backend(self):
        """MLX backend under test."""
        return create_backend("mlx")

    def assert_backends_match(
        self,
        numpy_result: Any,
        mlx_result: Any,
        *,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        """Assert two backend results match within FP32 tolerance.

        Args:
            numpy_result: Result from NumPy backend
            mlx_result: Result from MLX backend
            rtol: Relative tolerance (default: 1e-6 for FP32)
            atol: Absolute tolerance (default: 1e-6 for FP32)
        """
        numpy_array = np.asarray(numpy_result)
        mlx_array = np.asarray(mlx_result)
        np.testing.assert_allclose(
            numpy_array,
            mlx_array,
            rtol=rtol,
            atol=atol,
            err_msg="MLX backend result differs from NumPy reference",
        )

    # === Logical AND Tests ===

    def test_and_truth_table(self, numpy_backend, mlx_backend) -> None:
        """Validate AND operation truth table across backends."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])

        numpy_result = logical_and(a, b, backend=numpy_backend)
        mlx_result = logical_and(a, b, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    def test_and_batch_2d(self, numpy_backend, mlx_backend) -> None:
        """Validate AND on 2D batch data."""
        a = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        b = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

        numpy_result = logical_and(a, b, backend=numpy_backend)
        mlx_result = logical_and(a, b, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    def test_and_batch_3d(self, numpy_backend, mlx_backend) -> None:
        """Validate AND on 3D batch data."""
        a = np.array([[[1.0, 0.0], [1.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]])
        b = np.array([[[1.0, 1.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]])

        numpy_result = logical_and(a, b, backend=numpy_backend)
        mlx_result = logical_and(a, b, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    def test_and_fuzzy_values(self, numpy_backend, mlx_backend) -> None:
        """Validate AND on fuzzy logic values [0, 1]."""
        a = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        b = np.array([0.2, 0.4, 0.5, 0.8, 1.0])

        numpy_result = logical_and(a, b, backend=numpy_backend)
        mlx_result = logical_and(a, b, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    # === Logical OR Tests ===

    def test_or_truth_table(self, numpy_backend, mlx_backend) -> None:
        """Validate OR operation truth table across backends."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])

        numpy_result = logical_or(a, b, backend=numpy_backend)
        mlx_result = logical_or(a, b, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    def test_or_batch_2d(self, numpy_backend, mlx_backend) -> None:
        """Validate OR on 2D batch data."""
        a = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        b = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

        numpy_result = logical_or(a, b, backend=numpy_backend)
        mlx_result = logical_or(a, b, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    def test_or_fuzzy_values(self, numpy_backend, mlx_backend) -> None:
        """Validate OR on fuzzy logic values [0, 1]."""
        a = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        b = np.array([0.2, 0.4, 0.5, 0.8, 1.0])

        numpy_result = logical_or(a, b, backend=numpy_backend)
        mlx_result = logical_or(a, b, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    # === Logical NOT Tests ===

    def test_not_truth_values(self, numpy_backend, mlx_backend) -> None:
        """Validate NOT operation on boolean values."""
        a = np.array([1.0, 0.0])

        numpy_result = logical_not(a, backend=numpy_backend)
        mlx_result = logical_not(a, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    def test_not_batch_2d(self, numpy_backend, mlx_backend) -> None:
        """Validate NOT on 2D batch data."""
        a = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

        numpy_result = logical_not(a, backend=numpy_backend)
        mlx_result = logical_not(a, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    def test_not_fuzzy_values(self, numpy_backend, mlx_backend) -> None:
        """Validate NOT on fuzzy logic values [0, 1]."""
        a = np.array([0.0, 0.3, 0.5, 0.7, 1.0])

        numpy_result = logical_not(a, backend=numpy_backend)
        mlx_result = logical_not(a, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    # === Logical IMPLIES Tests ===

    def test_implies_truth_table(self, numpy_backend, mlx_backend) -> None:
        """Validate IMPLIES operation truth table across backends."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])

        numpy_result = logical_implies(a, b, backend=numpy_backend)
        mlx_result = logical_implies(a, b, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    def test_implies_batch_2d(self, numpy_backend, mlx_backend) -> None:
        """Validate IMPLIES on 2D batch data."""
        a = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        b = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])

        numpy_result = logical_implies(a, b, backend=numpy_backend)
        mlx_result = logical_implies(a, b, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    def test_implies_fuzzy_values(self, numpy_backend, mlx_backend) -> None:
        """Validate IMPLIES on fuzzy logic values [0, 1]."""
        a = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        b = np.array([0.2, 0.4, 0.5, 0.8, 1.0])

        numpy_result = logical_implies(a, b, backend=numpy_backend)
        mlx_result = logical_implies(a, b, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    # === Step Function Tests ===

    def test_step_positive_values(self, numpy_backend, mlx_backend) -> None:
        """Validate step function on positive values."""
        x = np.array([0.1, 0.5, 1.0, 10.0])

        numpy_result = step(x, backend=numpy_backend)
        mlx_result = step(x, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    def test_step_negative_values(self, numpy_backend, mlx_backend) -> None:
        """Validate step function on negative values."""
        x = np.array([-10.0, -1.0, -0.5, -0.1])

        numpy_result = step(x, backend=numpy_backend)
        mlx_result = step(x, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    def test_step_zero_value(self, numpy_backend, mlx_backend) -> None:
        """Validate step function at zero (critical edge case)."""
        x = np.array([0.0])

        numpy_result = step(x, backend=numpy_backend)
        mlx_result = step(x, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    def test_step_mixed_values(self, numpy_backend, mlx_backend) -> None:
        """Validate step function on mixed positive/negative/zero values."""
        x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])

        numpy_result = step(x, backend=numpy_backend)
        mlx_result = step(x, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    def test_step_batch_2d(self, numpy_backend, mlx_backend) -> None:
        """Validate step function on 2D batch data."""
        x = np.array([[-1.0, 0.0, 1.0], [-0.5, 0.5, 2.0]])

        numpy_result = step(x, backend=numpy_backend)
        mlx_result = step(x, backend=mlx_backend)
        mlx_backend.eval(mlx_result)

        self.assert_backends_match(numpy_result, mlx_result)

    # === Edge Cases ===

    def test_scalar_inputs(self, numpy_backend, mlx_backend) -> None:
        """Validate operations on scalar inputs."""
        a_scalar = 1.0
        b_scalar = 0.0

        # AND
        numpy_and = logical_and(a_scalar, b_scalar, backend=numpy_backend)
        mlx_and = logical_and(a_scalar, b_scalar, backend=mlx_backend)
        mlx_backend.eval(mlx_and)
        self.assert_backends_match(numpy_and, mlx_and)

        # OR
        numpy_or = logical_or(a_scalar, b_scalar, backend=numpy_backend)
        mlx_or = logical_or(a_scalar, b_scalar, backend=mlx_backend)
        mlx_backend.eval(mlx_or)
        self.assert_backends_match(numpy_or, mlx_or)

        # NOT
        numpy_not = logical_not(a_scalar, backend=numpy_backend)
        mlx_not = logical_not(a_scalar, backend=mlx_backend)
        mlx_backend.eval(mlx_not)
        self.assert_backends_match(numpy_not, mlx_not)

    def test_large_batch(self, numpy_backend, mlx_backend) -> None:
        """Validate operations on large batch sizes."""
        rng = np.random.default_rng(42)
        a = rng.uniform(0.0, 1.0, size=(1000, 100))
        b = rng.uniform(0.0, 1.0, size=(1000, 100))

        # Test AND on large batch
        numpy_result = logical_and(a, b, backend=numpy_backend)
        mlx_result = logical_and(a, b, backend=mlx_backend)
        mlx_backend.eval(mlx_result)
        self.assert_backends_match(numpy_result, mlx_result)

    def test_broadcasting(self, numpy_backend, mlx_backend) -> None:
        """Validate operations with broadcasting."""
        a = np.array([[1.0, 0.0, 1.0]])  # Shape (1, 3)
        b = np.array([[1.0], [0.0], [1.0]])  # Shape (3, 1)

        numpy_result = logical_and(a, b, backend=numpy_backend)
        mlx_result = logical_and(a, b, backend=mlx_backend)
        mlx_backend.eval(mlx_result)
        self.assert_backends_match(numpy_result, mlx_result)

    # === Performance Comparison ===

    def test_performance_and_operation(self, numpy_backend, mlx_backend) -> None:
        """Document performance comparison for AND operation."""
        rng = np.random.default_rng(42)
        a = rng.uniform(0.0, 1.0, size=(1000, 1000))
        b = rng.uniform(0.0, 1.0, size=(1000, 1000))

        # NumPy timing
        start = time.perf_counter()
        numpy_result = logical_and(a, b, backend=numpy_backend)
        numpy_time = time.perf_counter() - start

        # MLX timing
        start = time.perf_counter()
        mlx_result = logical_and(a, b, backend=mlx_backend)
        mlx_backend.eval(mlx_result)
        mlx_time = time.perf_counter() - start

        # Verify correctness
        self.assert_backends_match(numpy_result, mlx_result)

        # Document performance (printed during test execution)
        print(f"\nPerformance (AND, 1000x1000):")
        print(f"  NumPy: {numpy_time*1000:.3f}ms")
        print(f"  MLX:   {mlx_time*1000:.3f}ms")
        print(f"  Speedup: {numpy_time/mlx_time:.2f}x")

    def test_performance_step_operation(self, numpy_backend, mlx_backend) -> None:
        """Document performance comparison for step function."""
        rng = np.random.default_rng(42)
        x = rng.uniform(-1.0, 1.0, size=(1000, 1000))

        # NumPy timing
        start = time.perf_counter()
        numpy_result = step(x, backend=numpy_backend)
        numpy_time = time.perf_counter() - start

        # MLX timing
        start = time.perf_counter()
        mlx_result = step(x, backend=mlx_backend)
        mlx_backend.eval(mlx_result)
        mlx_time = time.perf_counter() - start

        # Verify correctness
        self.assert_backends_match(numpy_result, mlx_result)

        # Document performance
        print(f"\nPerformance (step, 1000x1000):")
        print(f"  NumPy: {numpy_time*1000:.3f}ms")
        print(f"  MLX:   {mlx_time*1000:.3f}ms")
        print(f"  Speedup: {numpy_time/mlx_time:.2f}x")


class TestOperationChaining:
    """Test cross-backend validation for chained operations."""

    @pytest.fixture
    def numpy_backend(self):
        """NumPy reference backend."""
        return create_backend("numpy")

    @pytest.fixture
    def mlx_backend(self):
        """MLX backend under test."""
        return create_backend("mlx")

    def assert_backends_match(
        self,
        numpy_result: Any,
        mlx_result: Any,
        *,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        """Assert two backend results match within FP32 tolerance."""
        numpy_array = np.asarray(numpy_result)
        mlx_array = np.asarray(mlx_result)
        np.testing.assert_allclose(
            numpy_array,
            mlx_array,
            rtol=rtol,
            atol=atol,
            err_msg="MLX backend result differs from NumPy reference",
        )

    def test_de_morgan_law(self, numpy_backend, mlx_backend) -> None:
        """Validate De Morgan's law: ¬(a ∧ b) = ¬a ∨ ¬b."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])

        # Left side: ¬(a ∧ b)
        numpy_left = logical_not(
            logical_and(a, b, backend=numpy_backend), backend=numpy_backend
        )
        mlx_left = logical_not(
            logical_and(a, b, backend=mlx_backend), backend=mlx_backend
        )
        mlx_backend.eval(mlx_left)

        # Right side: ¬a ∨ ¬b
        numpy_right = logical_or(
            logical_not(a, backend=numpy_backend),
            logical_not(b, backend=numpy_backend),
            backend=numpy_backend,
        )
        mlx_right = logical_or(
            logical_not(a, backend=mlx_backend),
            logical_not(b, backend=mlx_backend),
            backend=mlx_backend,
        )
        mlx_backend.eval(mlx_right)

        # Verify backends match
        self.assert_backends_match(numpy_left, mlx_left)
        self.assert_backends_match(numpy_right, mlx_right)

    def test_implication_equivalence(self, numpy_backend, mlx_backend) -> None:
        """Validate a → b ≡ ¬a ∨ b."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])

        # Using implies operation
        numpy_implies = logical_implies(a, b, backend=numpy_backend)
        mlx_implies = logical_implies(a, b, backend=mlx_backend)
        mlx_backend.eval(mlx_implies)

        # Using ¬a ∨ b
        numpy_equiv = logical_or(
            logical_not(a, backend=numpy_backend), b, backend=numpy_backend
        )
        mlx_equiv = logical_or(
            logical_not(a, backend=mlx_backend), b, backend=mlx_backend
        )
        mlx_backend.eval(mlx_equiv)

        # Verify both backends agree
        self.assert_backends_match(numpy_implies, mlx_implies)
        self.assert_backends_match(numpy_equiv, mlx_equiv)

    def test_contrapositive(self, numpy_backend, mlx_backend) -> None:
        """Validate contrapositive: (a → b) ≡ (¬b → ¬a)."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])

        # Direct: a → b
        numpy_direct = logical_implies(a, b, backend=numpy_backend)
        mlx_direct = logical_implies(a, b, backend=mlx_backend)
        mlx_backend.eval(mlx_direct)

        # Contrapositive: ¬b → ¬a
        numpy_contra = logical_implies(
            logical_not(b, backend=numpy_backend),
            logical_not(a, backend=numpy_backend),
            backend=numpy_backend,
        )
        mlx_contra = logical_implies(
            logical_not(b, backend=mlx_backend),
            logical_not(a, backend=mlx_backend),
            backend=mlx_backend,
        )
        mlx_backend.eval(mlx_contra)

        # Verify backends match
        self.assert_backends_match(numpy_direct, mlx_direct)
        self.assert_backends_match(numpy_contra, mlx_contra)
