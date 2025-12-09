"""Comprehensive tests for NumPy backend implementation.

Tests all 23 TensorBackend protocol operations plus protocol compliance.
Target: ≥90% coverage.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlogic.backends import NumpyBackend, TensorBackend


class TestNumpyBackendProtocol:
    """Test protocol compliance."""

    def test_implements_tensor_backend_protocol(self) -> None:
        """Verify NumpyBackend implements TensorBackend protocol."""
        backend = NumpyBackend()
        assert isinstance(backend, TensorBackend)


class TestTensorCreation:
    """Test tensor creation and manipulation operations."""

    @pytest.fixture
    def backend(self) -> NumpyBackend:
        """Create NumPy backend instance."""
        return NumpyBackend()

    def test_einsum_matrix_multiply(self, backend: NumpyBackend) -> None:
        """Test einsum with matrix multiplication pattern."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = backend.einsum("ij,jk->ik", a, b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_equal(result, expected)

    def test_einsum_inner_product(self, backend: NumpyBackend) -> None:
        """Test einsum with inner product pattern."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = backend.einsum("i,i->", a, b)
        expected = 32.0
        assert result == expected

    def test_zeros(self, backend: NumpyBackend) -> None:
        """Test zero tensor creation."""
        result = backend.zeros((2, 3))
        expected = np.zeros((2, 3))
        np.testing.assert_array_equal(result, expected)

    def test_zeros_empty_shape(self, backend: NumpyBackend) -> None:
        """Test zeros with scalar (empty shape)."""
        result = backend.zeros(())
        expected = np.zeros(())
        np.testing.assert_array_equal(result, expected)

    def test_ones(self, backend: NumpyBackend) -> None:
        """Test ones tensor creation."""
        result = backend.ones((3, 2))
        expected = np.ones((3, 2))
        np.testing.assert_array_equal(result, expected)

    def test_ones_large_shape(self, backend: NumpyBackend) -> None:
        """Test ones with large shape."""
        result = backend.ones((10, 10, 10))
        assert result.shape == (10, 10, 10)
        assert np.all(result == 1.0)

    def test_arange(self, backend: NumpyBackend) -> None:
        """Test range tensor creation."""
        result = backend.arange(0, 10, 2)
        expected = np.array([0, 2, 4, 6, 8])
        np.testing.assert_array_equal(result, expected)

    def test_arange_negative_step(self, backend: NumpyBackend) -> None:
        """Test arange with negative step."""
        result = backend.arange(10, 0, -2)
        expected = np.array([10, 8, 6, 4, 2])
        np.testing.assert_array_equal(result, expected)

    def test_arange_single_element(self, backend: NumpyBackend) -> None:
        """Test arange producing single element."""
        result = backend.arange(5, 6)
        expected = np.array([5])
        np.testing.assert_array_equal(result, expected)

    def test_reshape(self, backend: NumpyBackend) -> None:
        """Test reshape operation."""
        array = np.array([1, 2, 3, 4, 5, 6])
        result = backend.reshape(array, (2, 3))
        expected = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_reshape_flatten(self, backend: NumpyBackend) -> None:
        """Test reshape to flatten."""
        array = np.array([[1, 2], [3, 4]])
        result = backend.reshape(array, (4,))
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(result, expected)


class TestLogicalMathOperations:
    """Test logical and mathematical operations."""

    @pytest.fixture
    def backend(self) -> NumpyBackend:
        """Create NumPy backend instance."""
        return NumpyBackend()

    def test_step_positive(self, backend: NumpyBackend) -> None:
        """Test step function with positive values."""
        x = np.array([1.0, 2.0, 3.0])
        result = backend.step(x)
        expected = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_step_negative(self, backend: NumpyBackend) -> None:
        """Test step function with negative values."""
        x = np.array([-1.0, -2.0, -3.0])
        result = backend.step(x)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_step_mixed(self, backend: NumpyBackend) -> None:
        """Test step function with mixed positive/negative/zero."""
        x = np.array([-1.0, 0.0, 1.0])
        result = backend.step(x)
        expected = np.array([0.0, 0.5, 1.0])  # np.heaviside(0, 0.5) = 0.5
        np.testing.assert_array_equal(result, expected)

    def test_step_zero_boundary(self, backend: NumpyBackend) -> None:
        """Test step function at zero boundary."""
        x = np.array([0.0])
        result = backend.step(x)
        # np.heaviside(0, 0.5) returns 0.5
        expected = np.array([0.5])
        np.testing.assert_array_equal(result, expected)

    def test_maximum(self, backend: NumpyBackend) -> None:
        """Test element-wise maximum."""
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([4.0, 2.0, 6.0])
        result = backend.maximum(a, b)
        expected = np.array([4.0, 5.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_maximum_broadcast(self, backend: NumpyBackend) -> None:
        """Test maximum with broadcasting."""
        a = np.array([[1.0], [2.0]])
        b = np.array([3.0, 4.0])
        result = backend.maximum(a, b)
        expected = np.array([[3.0, 4.0], [3.0, 4.0]])
        np.testing.assert_array_equal(result, expected)

    def test_minimum(self, backend: NumpyBackend) -> None:
        """Test element-wise minimum."""
        a = np.array([1.0, 5.0, 3.0])
        b = np.array([4.0, 2.0, 6.0])
        result = backend.minimum(a, b)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_multiply(self, backend: NumpyBackend) -> None:
        """Test element-wise multiplication (Hadamard product)."""
        a = np.array([2.0, 3.0, 4.0])
        b = np.array([5.0, 6.0, 7.0])
        result = backend.multiply(a, b)
        expected = np.array([10.0, 18.0, 28.0])
        np.testing.assert_array_equal(result, expected)

    def test_multiply_broadcast(self, backend: NumpyBackend) -> None:
        """Test multiply with broadcasting."""
        a = np.array([[1.0, 2.0]])
        b = np.array([[3.0], [4.0]])
        result = backend.multiply(a, b)
        expected = np.array([[3.0, 6.0], [4.0, 8.0]])
        np.testing.assert_array_equal(result, expected)

    def test_add(self, backend: NumpyBackend) -> None:
        """Test element-wise addition."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = backend.add(a, b)
        expected = np.array([5.0, 7.0, 9.0])
        np.testing.assert_array_equal(result, expected)

    def test_subtract(self, backend: NumpyBackend) -> None:
        """Test element-wise subtraction."""
        a = np.array([10.0, 8.0, 6.0])
        b = np.array([1.0, 2.0, 3.0])
        result = backend.subtract(a, b)
        expected = np.array([9.0, 6.0, 3.0])
        np.testing.assert_array_equal(result, expected)


class TestQuantifierOperations:
    """Test quantifier operations (sum, prod, any, all)."""

    @pytest.fixture
    def backend(self) -> NumpyBackend:
        """Create NumPy backend instance."""
        return NumpyBackend()

    def test_sum_all_elements(self, backend: NumpyBackend) -> None:
        """Test sum over all elements."""
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.sum(array)
        expected = 10.0
        assert result == expected

    def test_sum_axis_0(self, backend: NumpyBackend) -> None:
        """Test sum over axis 0."""
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.sum(array, axis=0)
        expected = np.array([4.0, 6.0])
        np.testing.assert_array_equal(result, expected)

    def test_sum_axis_1(self, backend: NumpyBackend) -> None:
        """Test sum over axis 1."""
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.sum(array, axis=1)
        expected = np.array([3.0, 7.0])
        np.testing.assert_array_equal(result, expected)

    def test_sum_multiple_axes(self, backend: NumpyBackend) -> None:
        """Test sum over multiple axes."""
        array = np.ones((2, 3, 4))
        result = backend.sum(array, axis=(0, 2))
        expected = np.array([8.0, 8.0, 8.0])
        np.testing.assert_array_equal(result, expected)

    def test_prod_all_elements(self, backend: NumpyBackend) -> None:
        """Test product over all elements."""
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.prod(array)
        expected = 24.0
        assert result == expected

    def test_prod_axis_0(self, backend: NumpyBackend) -> None:
        """Test product over axis 0."""
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.prod(array, axis=0)
        expected = np.array([3.0, 8.0])
        np.testing.assert_array_equal(result, expected)

    def test_any_true(self, backend: NumpyBackend) -> None:
        """Test any with some true values."""
        array = np.array([False, False, True, False])
        result = backend.any(array)
        assert result is np.True_

    def test_any_all_false(self, backend: NumpyBackend) -> None:
        """Test any with all false values."""
        array = np.array([False, False, False])
        result = backend.any(array)
        assert result is np.False_

    def test_any_axis(self, backend: NumpyBackend) -> None:
        """Test any over axis."""
        array = np.array([[True, False], [False, False]])
        result = backend.any(array, axis=1)
        expected = np.array([True, False])
        np.testing.assert_array_equal(result, expected)

    def test_all_true(self, backend: NumpyBackend) -> None:
        """Test all with all true values."""
        array = np.array([True, True, True])
        result = backend.all(array)
        assert result is np.True_

    def test_all_some_false(self, backend: NumpyBackend) -> None:
        """Test all with some false values."""
        array = np.array([True, False, True])
        result = backend.all(array)
        assert result is np.False_

    def test_all_axis(self, backend: NumpyBackend) -> None:
        """Test all over axis."""
        array = np.array([[True, True], [True, False]])
        result = backend.all(array, axis=0)
        expected = np.array([True, False])
        np.testing.assert_array_equal(result, expected)


class TestDifferentiationEvaluation:
    """Test differentiation and evaluation operations."""

    @pytest.fixture
    def backend(self) -> NumpyBackend:
        """Create NumPy backend instance."""
        return NumpyBackend()

    def test_grad_raises_not_implemented(self, backend: NumpyBackend) -> None:
        """Test grad raises NotImplementedError."""

        def dummy_fn(x: np.ndarray) -> np.ndarray:
            return x * 2

        with pytest.raises(NotImplementedError, match="NumPy backend does not support automatic differentiation"):
            backend.grad(dummy_fn)

    def test_eval_no_op(self, backend: NumpyBackend) -> None:
        """Test eval is no-op (doesn't raise)."""
        array = np.array([1.0, 2.0, 3.0])
        backend.eval(array)  # Should not raise

    def test_eval_multiple_arrays(self, backend: NumpyBackend) -> None:
        """Test eval with multiple arrays is no-op."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        backend.eval(a, b)  # Should not raise

    def test_compile_returns_function_unchanged(self, backend: NumpyBackend) -> None:
        """Test compile returns original function."""

        def original_fn(x: int) -> int:
            return x + 1

        compiled_fn = backend.compile(original_fn)
        assert compiled_fn is original_fn
        assert compiled_fn(5) == 6


class TestUtilityOperations:
    """Test utility operations (where, expand_dims, squeeze, transpose, concatenate)."""

    @pytest.fixture
    def backend(self) -> NumpyBackend:
        """Create NumPy backend instance."""
        return NumpyBackend()

    def test_where(self, backend: NumpyBackend) -> None:
        """Test where conditional selection."""
        condition = np.array([True, False, True])
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([4.0, 5.0, 6.0])
        result = backend.where(condition, x, y)
        expected = np.array([1.0, 5.0, 3.0])
        np.testing.assert_array_equal(result, expected)

    def test_where_broadcast(self, backend: NumpyBackend) -> None:
        """Test where with broadcasting."""
        condition = np.array([True, False])
        x = np.array([[1.0, 2.0]])
        y = np.array([[3.0, 4.0]])
        result = backend.where(condition, x, y)
        expected = np.array([[1.0, 4.0]])
        np.testing.assert_array_equal(result, expected)

    def test_expand_dims_axis_0(self, backend: NumpyBackend) -> None:
        """Test expand_dims at axis 0."""
        array = np.array([1.0, 2.0, 3.0])
        result = backend.expand_dims(array, axis=0)
        assert result.shape == (1, 3)
        np.testing.assert_array_equal(result, [[1.0, 2.0, 3.0]])

    def test_expand_dims_axis_1(self, backend: NumpyBackend) -> None:
        """Test expand_dims at axis 1."""
        array = np.array([1.0, 2.0, 3.0])
        result = backend.expand_dims(array, axis=1)
        assert result.shape == (3, 1)

    def test_expand_dims_negative_axis(self, backend: NumpyBackend) -> None:
        """Test expand_dims with negative axis."""
        array = np.array([1.0, 2.0])
        result = backend.expand_dims(array, axis=-1)
        assert result.shape == (2, 1)

    def test_squeeze_all_axes(self, backend: NumpyBackend) -> None:
        """Test squeeze removing all size-1 axes."""
        array = np.array([[[1.0]], [[2.0]]])
        result = backend.squeeze(array)
        expected = np.array([1.0, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_squeeze_specific_axis(self, backend: NumpyBackend) -> None:
        """Test squeeze removing specific axis."""
        array = np.array([[[1.0, 2.0]]])
        result = backend.squeeze(array, axis=0)
        assert result.shape == (1, 2)

    def test_squeeze_no_change(self, backend: NumpyBackend) -> None:
        """Test squeeze on array with no size-1 axes."""
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.squeeze(array)
        np.testing.assert_array_equal(result, array)

    def test_transpose_2d(self, backend: NumpyBackend) -> None:
        """Test transpose on 2D array."""
        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.transpose(array)
        expected = np.array([[1.0, 3.0], [2.0, 4.0]])
        np.testing.assert_array_equal(result, expected)

    def test_transpose_with_axes(self, backend: NumpyBackend) -> None:
        """Test transpose with explicit axes."""
        array = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        result = backend.transpose(array, axes=(1, 0, 2))
        assert result.shape == (1, 2, 2)

    def test_transpose_3d_reverse(self, backend: NumpyBackend) -> None:
        """Test transpose reversing all axes."""
        array = np.ones((2, 3, 4))
        result = backend.transpose(array)
        assert result.shape == (4, 3, 2)

    def test_concatenate_axis_0(self, backend: NumpyBackend) -> None:
        """Test concatenate along axis 0."""
        a = np.array([[1.0, 2.0]])
        b = np.array([[3.0, 4.0]])
        c = np.array([[5.0, 6.0]])
        result = backend.concatenate((a, b, c), axis=0)
        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        np.testing.assert_array_equal(result, expected)

    def test_concatenate_axis_1(self, backend: NumpyBackend) -> None:
        """Test concatenate along axis 1."""
        a = np.array([[1.0], [2.0]])
        b = np.array([[3.0], [4.0]])
        result = backend.concatenate((a, b), axis=1)
        expected = np.array([[1.0, 3.0], [2.0, 4.0]])
        np.testing.assert_array_equal(result, expected)

    def test_concatenate_single_array(self, backend: NumpyBackend) -> None:
        """Test concatenate with single array."""
        a = np.array([[1.0, 2.0]])
        result = backend.concatenate((a,), axis=0)
        np.testing.assert_array_equal(result, a)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def backend(self) -> NumpyBackend:
        """Create NumPy backend instance."""
        return NumpyBackend()

    def test_operations_on_empty_arrays(self, backend: NumpyBackend) -> None:
        """Test operations handle empty arrays."""
        empty = np.array([])
        result = backend.sum(empty)
        assert result == 0.0

    def test_operations_on_scalar(self, backend: NumpyBackend) -> None:
        """Test operations on scalar values."""
        scalar = np.array(5.0)
        result = backend.add(scalar, scalar)
        assert result == 10.0

    def test_large_array_operations(self, backend: NumpyBackend) -> None:
        """Test operations on large arrays."""
        large = np.ones((100, 100))
        result = backend.sum(large)
        assert result == 10000.0

    def test_mixed_dtypes(self, backend: NumpyBackend) -> None:
        """Test operations with mixed dtypes."""
        int_array = np.array([1, 2, 3])
        float_array = np.array([1.5, 2.5, 3.5])
        result = backend.add(int_array, float_array)
        expected = np.array([2.5, 4.5, 6.5])
        np.testing.assert_array_equal(result, expected)
