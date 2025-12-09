"""Tests for temperature-controlled operations.

Tests verify temperature parameter behavior:
- T=0: Hard boolean (deductive reasoning)
- T>0: Soft probabilistic (analogical reasoning)
- Interpolation between hard and soft modes
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from tensorlogic.backends import create_backend
from tensorlogic.core.operations import logical_and, logical_or, logical_not
from tensorlogic.core.temperature import (
    temperature_scaled_operation,
    deductive_operation,
    analogical_operation,
)


@pytest.fixture(params=["numpy", "mlx"])
def backend(request):
    """Parametrized fixture for testing across backends."""
    return create_backend(request.param)


class TestTemperatureScaling:
    """Tests for temperature_scaled_operation function."""

    def test_temperature_zero_gives_hard_boolean(self, backend) -> None:
        """T=0 should apply step function (hard boolean)."""
        # Create operation with T=0
        op = temperature_scaled_operation(
            logical_and, temperature=0.0, backend=backend
        )

        # Test with fuzzy values including values that should map to 0
        a = np.array([0.7, 0.8, 0.0, 0.3])
        b = np.array([0.6, 0.9, 0.5, 0.0])

        result = op(a, b, backend=backend)
        backend.eval(result)

        # With T=0, should get hard boolean AND
        # 0.7*0.6=0.42 -> step(0.42) -> 1.0 (positive)
        # 0.8*0.9=0.72 -> step(0.72) -> 1.0 (positive)
        # 0.0*0.5=0.0 -> step(0.0) -> 0.0 (zero)
        # 0.3*0.0=0.0 -> step(0.0) -> 0.0 (zero)
        expected = np.array([1.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_temperature_nonzero_interpolates(self, backend) -> None:
        """T>0 should interpolate between hard and soft."""
        temperature = 1.0
        op = temperature_scaled_operation(
            logical_and, temperature=temperature, backend=backend
        )

        # Test with fuzzy values
        a = np.array([0.7, 0.8])
        b = np.array([0.6, 0.9])

        result = op(a, b, backend=backend)
        backend.eval(result)

        # Compute expected interpolation
        # α = 1 - exp(-1.0) ≈ 0.632
        alpha = 1.0 - math.exp(-1.0)

        # Soft result: 0.7*0.6=0.42, 0.8*0.9=0.72
        soft = np.array([0.42, 0.72])

        # Hard result: step(0.42)=1.0, step(0.72)=1.0
        hard = np.array([1.0, 1.0])

        # Interpolate: (1-α)*hard + α*soft
        expected = (1 - alpha) * hard + alpha * soft

        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_high_temperature_approaches_soft(self, backend) -> None:
        """T>>0 should approach fully soft (continuous) operations."""
        temperature = 10.0  # Very high temperature
        op = temperature_scaled_operation(
            logical_and, temperature=temperature, backend=backend
        )

        # Test with fuzzy values
        a = np.array([0.7, 0.3])
        b = np.array([0.6, 0.2])

        result = op(a, b, backend=backend)
        backend.eval(result)

        # At T=10, α ≈ 0.9999546 (almost 1.0)
        # Result should be very close to soft result
        soft_expected = np.array([0.42, 0.06])

        # Allow slightly more tolerance due to interpolation
        np.testing.assert_array_almost_equal(result, soft_expected, decimal=3)

    def test_negative_temperature_raises_error(self, backend) -> None:
        """Negative temperature should raise ValueError."""
        with pytest.raises(ValueError, match="Temperature must be non-negative"):
            temperature_scaled_operation(
                logical_and, temperature=-1.0, backend=backend
            )

    def test_temperature_with_or_operation(self, backend) -> None:
        """Temperature scaling should work with logical_or."""
        temperature = 1.0
        op = temperature_scaled_operation(
            logical_or, temperature=temperature, backend=backend
        )

        a = np.array([0.3, 0.4])
        b = np.array([0.2, 0.5])

        result = op(a, b, backend=backend)
        backend.eval(result)

        # Compute expected
        alpha = 1.0 - math.exp(-1.0)
        soft = np.maximum(a, b)  # [0.3, 0.5]
        hard = np.array([1.0, 1.0])  # step([0.3, 0.5]) - both positive
        expected = (1 - alpha) * hard + alpha * soft

        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_temperature_with_not_operation(self, backend) -> None:
        """Temperature scaling should work with logical_not."""
        temperature = 0.5
        op = temperature_scaled_operation(
            logical_not, temperature=temperature, backend=backend
        )

        a = np.array([0.7, 0.3, 0.0, 1.0])

        result = op(a, backend=backend)
        backend.eval(result)

        # Compute expected
        alpha = 1.0 - math.exp(-0.5)
        soft = 1.0 - a  # [0.3, 0.7, 1.0, 0.0]
        hard = np.array([1.0, 1.0, 1.0, 0.0])  # step([0.3, 0.7, 1.0, 0.0]) - all positive except last
        expected = (1 - alpha) * hard + alpha * soft

        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_temperature_preserves_shape(self, backend) -> None:
        """Temperature scaling should preserve tensor shape."""
        op = temperature_scaled_operation(
            logical_and, temperature=1.0, backend=backend
        )

        # Test various shapes
        for shape in [(3,), (2, 3), (2, 3, 4)]:
            a = np.random.rand(*shape)
            b = np.random.rand(*shape)

            result = op(a, b, backend=backend)
            backend.eval(result)

            assert result.shape == shape

    def test_temperature_with_broadcast(self, backend) -> None:
        """Temperature scaling should work with broadcasting."""
        op = temperature_scaled_operation(
            logical_and, temperature=1.0, backend=backend
        )

        # Scalar broadcast
        a = np.array([0.7, 0.8])
        b = 0.5  # Scalar

        result = op(a, b, backend=backend)
        backend.eval(result)

        # Verify broadcast worked (shape should be (2,))
        assert result.shape == (2,)

    def test_temperature_interpolation_monotonic(self, backend) -> None:
        """As temperature increases, result should move from hard to soft."""
        a = np.array([0.6])
        b = np.array([0.4])

        # Soft result: 0.6 * 0.4 = 0.24
        # Hard result: step(0.24) = 1.0 (positive)

        results = []
        temperatures = [0.0, 0.5, 1.0, 2.0, 5.0]

        for temp in temperatures:
            op = temperature_scaled_operation(
                logical_and, temperature=temp, backend=backend
            )
            result = op(a, b, backend=backend)
            backend.eval(result)
            results.append(float(np.asarray(result)[0]))

        # Results should be monotonically decreasing (moving from 1.0 towards 0.24)
        for i in range(len(results) - 1):
            assert results[i] >= results[i + 1] - 1e-6  # Allow tiny numerical errors


class TestConvenienceFunctions:
    """Tests for convenience functions (deductive_operation, analogical_operation)."""

    def test_deductive_operation_is_temperature_zero(self, backend) -> None:
        """deductive_operation should be equivalent to T=0."""
        a = np.array([0.7, 0.3])
        b = np.array([0.6, 0.2])

        # Using deductive_operation
        op1 = deductive_operation(logical_and, backend=backend)
        result1 = op1(a, b, backend=backend)
        backend.eval(result1)

        # Using temperature_scaled_operation with T=0
        op2 = temperature_scaled_operation(
            logical_and, temperature=0.0, backend=backend
        )
        result2 = op2(a, b, backend=backend)
        backend.eval(result2)

        np.testing.assert_array_almost_equal(result1, result2, decimal=5)

    def test_analogical_operation_default_temperature(self, backend) -> None:
        """analogical_operation should default to T=1.0."""
        a = np.array([0.7, 0.3])
        b = np.array([0.6, 0.2])

        # Using analogical_operation (default T=1.0)
        op1 = analogical_operation(logical_and, backend=backend)
        result1 = op1(a, b, backend=backend)
        backend.eval(result1)

        # Using temperature_scaled_operation with T=1.0
        op2 = temperature_scaled_operation(
            logical_and, temperature=1.0, backend=backend
        )
        result2 = op2(a, b, backend=backend)
        backend.eval(result2)

        np.testing.assert_array_almost_equal(result1, result2, decimal=5)

    def test_analogical_operation_custom_temperature(self, backend) -> None:
        """analogical_operation should support custom temperature."""
        a = np.array([0.7, 0.3])
        b = np.array([0.6, 0.2])

        temperature = 2.5

        # Using analogical_operation with custom T
        op1 = analogical_operation(logical_or, temperature=temperature, backend=backend)
        result1 = op1(a, b, backend=backend)
        backend.eval(result1)

        # Using temperature_scaled_operation with same T
        op2 = temperature_scaled_operation(
            logical_or, temperature=temperature, backend=backend
        )
        result2 = op2(a, b, backend=backend)
        backend.eval(result2)

        np.testing.assert_array_almost_equal(result1, result2, decimal=5)

    def test_analogical_operation_rejects_zero_temperature(self, backend) -> None:
        """analogical_operation should reject T=0."""
        with pytest.raises(ValueError, match="Analogical operation requires temperature > 0"):
            analogical_operation(logical_and, temperature=0.0, backend=backend)

    def test_analogical_operation_rejects_negative_temperature(self, backend) -> None:
        """analogical_operation should reject T<0."""
        with pytest.raises(ValueError, match="Analogical operation requires temperature > 0"):
            analogical_operation(logical_and, temperature=-1.0, backend=backend)


class TestCrossBackendConsistency:
    """Tests verifying consistent behavior across NumPy and MLX backends."""

    def test_temperature_zero_cross_backend(self) -> None:
        """T=0 should produce identical results across backends."""
        numpy_backend = create_backend("numpy")
        mlx_backend = create_backend("mlx")

        a = np.array([0.7, 0.4, 0.2])
        b = np.array([0.6, 0.5, 0.1])

        # NumPy result
        op_numpy = temperature_scaled_operation(
            logical_and, temperature=0.0, backend=numpy_backend
        )
        result_numpy = op_numpy(a, b, backend=numpy_backend)
        numpy_backend.eval(result_numpy)

        # MLX result
        op_mlx = temperature_scaled_operation(
            logical_and, temperature=0.0, backend=mlx_backend
        )
        result_mlx = op_mlx(a, b, backend=mlx_backend)
        mlx_backend.eval(result_mlx)

        np.testing.assert_array_almost_equal(
            result_numpy, result_mlx, decimal=5
        )

    def test_temperature_interpolation_cross_backend(self) -> None:
        """T>0 interpolation should match across backends."""
        numpy_backend = create_backend("numpy")
        mlx_backend = create_backend("mlx")

        temperature = 1.5
        a = np.array([0.8, 0.3])
        b = np.array([0.7, 0.4])

        # NumPy result
        op_numpy = temperature_scaled_operation(
            logical_or, temperature=temperature, backend=numpy_backend
        )
        result_numpy = op_numpy(a, b, backend=numpy_backend)
        numpy_backend.eval(result_numpy)

        # MLX result
        op_mlx = temperature_scaled_operation(
            logical_or, temperature=temperature, backend=mlx_backend
        )
        result_mlx = op_mlx(a, b, backend=mlx_backend)
        mlx_backend.eval(result_mlx)

        np.testing.assert_array_almost_equal(
            result_numpy, result_mlx, decimal=5
        )


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_temperature_with_all_zeros(self, backend) -> None:
        """Temperature scaling should handle all-zero inputs."""
        op = temperature_scaled_operation(
            logical_and, temperature=1.0, backend=backend
        )

        a = np.zeros(3)
        b = np.zeros(3)

        result = op(a, b, backend=backend)
        backend.eval(result)

        # Result should be all zeros
        np.testing.assert_array_almost_equal(result, np.zeros(3), decimal=5)

    def test_temperature_with_all_ones(self, backend) -> None:
        """Temperature scaling should handle all-one inputs."""
        op = temperature_scaled_operation(
            logical_and, temperature=1.0, backend=backend
        )

        a = np.ones(3)
        b = np.ones(3)

        result = op(a, b, backend=backend)
        backend.eval(result)

        # Result should be all ones
        np.testing.assert_array_almost_equal(result, np.ones(3), decimal=5)

    def test_temperature_with_single_element(self, backend) -> None:
        """Temperature scaling should work with scalar-like arrays."""
        op = temperature_scaled_operation(
            logical_and, temperature=0.5, backend=backend
        )

        a = np.array([0.6])
        b = np.array([0.7])

        result = op(a, b, backend=backend)
        backend.eval(result)

        # Verify shape and value
        assert result.shape == (1,)
        assert 0.0 <= float(np.asarray(result)[0]) <= 1.0

    def test_temperature_exactly_at_transition(self, backend) -> None:
        """Test temperature at transition point (T=1.0)."""
        # At T=1.0, α = 1 - exp(-1) ≈ 0.632
        temperature = 1.0
        op = temperature_scaled_operation(
            logical_and, temperature=temperature, backend=backend
        )

        a = np.array([0.5])
        b = np.array([0.5])

        result = op(a, b, backend=backend)
        backend.eval(result)

        # soft = 0.5 * 0.5 = 0.25
        # hard = step(0.25) = 1.0 (positive)
        # result = (1-0.632)*1.0 + 0.632*0.25 = 0.368 + 0.158 = 0.526
        alpha = 1.0 - math.exp(-1.0)
        expected = (1 - alpha) * 1.0 + alpha * 0.25

        np.testing.assert_almost_equal(float(np.asarray(result)[0]), expected, decimal=5)
