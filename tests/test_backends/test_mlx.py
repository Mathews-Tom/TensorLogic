"""Unit tests for MLX backend implementation.

Tests all TensorBackend protocol operations for MLX implementation.
Includes MLX-specific tests for lazy evaluation behavior.
"""

from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    from tensorlogic.backends.mlx import MLXBackend
    from tensorlogic.backends.protocol import TensorBackend

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

pytestmark = pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")


class TestMLXBackendProtocolCompliance:
    """Test MLXBackend implements TensorBackend protocol."""

    def test_protocol_compliance(self) -> None:
        """Verify MLXBackend implements TensorBackend protocol."""
        backend = MLXBackend()
        assert isinstance(backend, TensorBackend)


class TestTensorCreation:
    """Test tensor creation and manipulation operations."""

    def test_zeros(self) -> None:
        """Test zeros tensor creation."""
        backend = MLXBackend()
        z = backend.zeros((2, 3))
        backend.eval(z)
        assert z.shape == (2, 3)
        assert mx.array_equal(z, mx.zeros((2, 3)))

    def test_ones(self) -> None:
        """Test ones tensor creation."""
        backend = MLXBackend()
        o = backend.ones((3, 2))
        backend.eval(o)
        assert o.shape == (3, 2)
        assert mx.array_equal(o, mx.ones((3, 2)))

    def test_arange(self) -> None:
        """Test arange tensor creation."""
        backend = MLXBackend()
        r = backend.arange(0, 10, 2)
        backend.eval(r)
        expected = mx.array([0, 2, 4, 6, 8])
        assert mx.array_equal(r, expected)

    def test_arange_default_step(self) -> None:
        """Test arange with default step."""
        backend = MLXBackend()
        r = backend.arange(0, 5)
        backend.eval(r)
        expected = mx.array([0, 1, 2, 3, 4])
        assert mx.array_equal(r, expected)

    def test_reshape(self) -> None:
        """Test reshape operation."""
        backend = MLXBackend()
        x = backend.ones((6,))
        reshaped = backend.reshape(x, (2, 3))
        backend.eval(reshaped)
        assert reshaped.shape == (2, 3)


class TestLogicalOperations:
    """Test logical and mathematical operations."""

    def test_step_positive(self) -> None:
        """Test step function with positive values."""
        backend = MLXBackend()
        x = mx.array([1.0, 2.0, 3.0])
        result = backend.step(x)
        backend.eval(result)
        expected = mx.ones((3,))
        assert mx.array_equal(result, expected)

    def test_step_negative(self) -> None:
        """Test step function with negative values."""
        backend = MLXBackend()
        x = mx.array([-1.0, -2.0, -3.0])
        result = backend.step(x)
        backend.eval(result)
        expected = mx.zeros((3,))
        assert mx.array_equal(result, expected)

    def test_step_zero(self) -> None:
        """Test step function with zero."""
        backend = MLXBackend()
        x = mx.array([0.0])
        result = backend.step(x)
        backend.eval(result)
        expected = mx.array([0.0])
        assert mx.array_equal(result, expected)

    def test_step_mixed(self) -> None:
        """Test step function with mixed values."""
        backend = MLXBackend()
        x = mx.array([-1.0, 0.0, 1.0])
        result = backend.step(x)
        backend.eval(result)
        expected = mx.array([0.0, 0.0, 1.0])
        assert mx.array_equal(result, expected)

    def test_maximum(self) -> None:
        """Test element-wise maximum."""
        backend = MLXBackend()
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([2.0, 1.0, 4.0])
        result = backend.maximum(a, b)
        backend.eval(result)
        expected = mx.array([2.0, 2.0, 4.0])
        assert mx.array_equal(result, expected)

    def test_minimum(self) -> None:
        """Test element-wise minimum."""
        backend = MLXBackend()
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([2.0, 1.0, 4.0])
        result = backend.minimum(a, b)
        backend.eval(result)
        expected = mx.array([1.0, 1.0, 3.0])
        assert mx.array_equal(result, expected)

    def test_multiply(self) -> None:
        """Test element-wise multiplication."""
        backend = MLXBackend()
        a = mx.array([2.0, 3.0, 4.0])
        b = mx.array([1.0, 2.0, 3.0])
        result = backend.multiply(a, b)
        backend.eval(result)
        expected = mx.array([2.0, 6.0, 12.0])
        assert mx.array_equal(result, expected)

    def test_add(self) -> None:
        """Test element-wise addition."""
        backend = MLXBackend()
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        result = backend.add(a, b)
        backend.eval(result)
        expected = mx.array([5.0, 7.0, 9.0])
        assert mx.array_equal(result, expected)

    def test_subtract(self) -> None:
        """Test element-wise subtraction."""
        backend = MLXBackend()
        a = mx.array([5.0, 7.0, 9.0])
        b = mx.array([1.0, 2.0, 3.0])
        result = backend.subtract(a, b)
        backend.eval(result)
        expected = mx.array([4.0, 5.0, 6.0])
        assert mx.array_equal(result, expected)


class TestEinsum:
    """Test Einstein summation operation."""

    def test_einsum_matrix_multiply(self) -> None:
        """Test matrix multiplication via einsum."""
        backend = MLXBackend()
        a = mx.array([[1.0, 2.0], [3.0, 4.0]])
        b = mx.array([[5.0, 6.0], [7.0, 8.0]])
        result = backend.einsum("ij,jk->ik", a, b)
        backend.eval(result)
        expected = mx.array([[19.0, 22.0], [43.0, 50.0]])
        assert mx.allclose(result, expected)

    def test_einsum_dot_product(self) -> None:
        """Test dot product via einsum."""
        backend = MLXBackend()
        a = mx.array([1.0, 2.0, 3.0])
        b = mx.array([4.0, 5.0, 6.0])
        result = backend.einsum("i,i->", a, b)
        backend.eval(result)
        expected = mx.array(32.0)
        assert mx.allclose(result, expected)

    def test_einsum_outer_product(self) -> None:
        """Test outer product via einsum."""
        backend = MLXBackend()
        a = mx.array([1.0, 2.0])
        b = mx.array([3.0, 4.0])
        result = backend.einsum("i,j->ij", a, b)
        backend.eval(result)
        expected = mx.array([[3.0, 4.0], [6.0, 8.0]])
        assert mx.allclose(result, expected)


class TestQuantifierOperations:
    """Test quantifier operations (sum, prod, any, all)."""

    def test_sum_all(self) -> None:
        """Test sum over all elements."""
        backend = MLXBackend()
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.sum(x)
        backend.eval(result)
        expected = mx.array(10.0)
        assert mx.allclose(result, expected)

    def test_sum_axis0(self) -> None:
        """Test sum over axis 0."""
        backend = MLXBackend()
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.sum(x, axis=0)
        backend.eval(result)
        expected = mx.array([4.0, 6.0])
        assert mx.allclose(result, expected)

    def test_sum_axis1(self) -> None:
        """Test sum over axis 1."""
        backend = MLXBackend()
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.sum(x, axis=1)
        backend.eval(result)
        expected = mx.array([3.0, 7.0])
        assert mx.allclose(result, expected)

    def test_prod_all(self) -> None:
        """Test product over all elements."""
        backend = MLXBackend()
        x = mx.array([[2.0, 3.0], [4.0, 5.0]])
        result = backend.prod(x)
        backend.eval(result)
        expected = mx.array(120.0)
        assert mx.allclose(result, expected)

    def test_prod_axis0(self) -> None:
        """Test product over axis 0."""
        backend = MLXBackend()
        x = mx.array([[2.0, 3.0], [4.0, 5.0]])
        result = backend.prod(x, axis=0)
        backend.eval(result)
        expected = mx.array([8.0, 15.0])
        assert mx.allclose(result, expected)

    def test_any_true(self) -> None:
        """Test any with at least one non-zero element."""
        backend = MLXBackend()
        x = mx.array([0.0, 0.0, 1.0])
        result = backend.any(x)
        backend.eval(result)
        assert result.item() is True

    def test_any_false(self) -> None:
        """Test any with all zero elements."""
        backend = MLXBackend()
        x = mx.array([0.0, 0.0, 0.0])
        result = backend.any(x)
        backend.eval(result)
        assert result.item() is False

    def test_any_axis(self) -> None:
        """Test any over specific axis."""
        backend = MLXBackend()
        x = mx.array([[1.0, 0.0], [0.0, 0.0]])
        result = backend.any(x, axis=1)
        backend.eval(result)
        expected = mx.array([True, False])
        assert mx.array_equal(result, expected)

    def test_all_true(self) -> None:
        """Test all with all non-zero elements."""
        backend = MLXBackend()
        x = mx.array([1.0, 2.0, 3.0])
        result = backend.all(x)
        backend.eval(result)
        assert result.item() is True

    def test_all_false(self) -> None:
        """Test all with at least one zero element."""
        backend = MLXBackend()
        x = mx.array([1.0, 0.0, 3.0])
        result = backend.all(x)
        backend.eval(result)
        assert result.item() is False

    def test_all_axis(self) -> None:
        """Test all over specific axis."""
        backend = MLXBackend()
        x = mx.array([[1.0, 2.0], [1.0, 0.0]])
        result = backend.all(x, axis=1)
        backend.eval(result)
        expected = mx.array([True, False])
        assert mx.array_equal(result, expected)


class TestGradient:
    """Test automatic differentiation."""

    def test_grad_simple(self) -> None:
        """Test gradient of simple function."""
        backend = MLXBackend()

        def fn(x: mx.array) -> mx.array:
            return x * x

        grad_fn = backend.grad(fn)
        x = mx.array(3.0)
        result = grad_fn(x)
        backend.eval(result)
        expected = mx.array(6.0)
        assert mx.allclose(result, expected)

    def test_grad_complex(self) -> None:
        """Test gradient of complex function."""
        backend = MLXBackend()

        def fn(x: mx.array) -> mx.array:
            return mx.sum(x * x * x)

        grad_fn = backend.grad(fn)
        x = mx.array([1.0, 2.0, 3.0])
        result = grad_fn(x)
        backend.eval(result)
        expected = mx.array([3.0, 12.0, 27.0])
        assert mx.allclose(result, expected)


class TestEvaluation:
    """Test lazy evaluation and explicit eval()."""

    def test_eval_single_array(self) -> None:
        """Test eval with single array."""
        backend = MLXBackend()
        x = backend.ones((2, 2))
        backend.eval(x)
        assert x.shape == (2, 2)

    def test_eval_multiple_arrays(self) -> None:
        """Test eval with multiple arrays."""
        backend = MLXBackend()
        x = backend.ones((2, 2))
        y = backend.zeros((2, 2))
        backend.eval(x, y)
        assert x.shape == (2, 2)
        assert y.shape == (2, 2)

    def test_lazy_evaluation(self) -> None:
        """Test that operations are lazy without eval()."""
        backend = MLXBackend()
        x = backend.ones((2, 2))
        y = backend.add(x, x)
        assert y.shape == (2, 2)
        backend.eval(y)
        expected = mx.ones((2, 2)) * 2
        assert mx.allclose(y, expected)


class TestCompile:
    """Test JIT compilation."""

    def test_compile_function(self) -> None:
        """Test compiling a function."""
        backend = MLXBackend()

        def compute(x: mx.array) -> mx.array:
            return x * x + x

        compiled_fn = backend.compile(compute)
        x = mx.array([1.0, 2.0, 3.0])
        result = compiled_fn(x)
        backend.eval(result)
        expected = mx.array([2.0, 6.0, 12.0])
        assert mx.allclose(result, expected)


class TestUtilityOperations:
    """Test utility operations."""

    def test_where(self) -> None:
        """Test conditional selection."""
        backend = MLXBackend()
        condition = mx.array([True, False, True])
        x = mx.array([1.0, 2.0, 3.0])
        y = mx.array([4.0, 5.0, 6.0])
        result = backend.where(condition, x, y)
        backend.eval(result)
        expected = mx.array([1.0, 5.0, 3.0])
        assert mx.array_equal(result, expected)

    def test_expand_dims(self) -> None:
        """Test adding dimension."""
        backend = MLXBackend()
        x = mx.array([1.0, 2.0, 3.0])
        result = backend.expand_dims(x, axis=0)
        backend.eval(result)
        assert result.shape == (1, 3)

    def test_expand_dims_axis1(self) -> None:
        """Test adding dimension at axis 1."""
        backend = MLXBackend()
        x = mx.array([1.0, 2.0, 3.0])
        result = backend.expand_dims(x, axis=1)
        backend.eval(result)
        assert result.shape == (3, 1)

    def test_squeeze(self) -> None:
        """Test removing dimension."""
        backend = MLXBackend()
        x = mx.array([[1.0, 2.0, 3.0]])
        result = backend.squeeze(x, axis=0)
        backend.eval(result)
        assert result.shape == (3,)

    def test_squeeze_all(self) -> None:
        """Test removing all single dimensions."""
        backend = MLXBackend()
        x = mx.array([[[1.0]]])
        result = backend.squeeze(x)
        backend.eval(result)
        assert result.shape == ()

    def test_transpose_2d(self) -> None:
        """Test transposing 2D array."""
        backend = MLXBackend()
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.transpose(x, (1, 0))
        backend.eval(result)
        expected = mx.array([[1.0, 3.0], [2.0, 4.0]])
        assert mx.array_equal(result, expected)

    def test_transpose_auto(self) -> None:
        """Test auto-transpose (reverse axes)."""
        backend = MLXBackend()
        x = mx.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.transpose(x)
        backend.eval(result)
        expected = mx.array([[1.0, 3.0], [2.0, 4.0]])
        assert mx.array_equal(result, expected)

    def test_concatenate_axis0(self) -> None:
        """Test concatenation along axis 0."""
        backend = MLXBackend()
        a = mx.array([[1.0, 2.0]])
        b = mx.array([[3.0, 4.0]])
        result = backend.concatenate((a, b), axis=0)
        backend.eval(result)
        expected = mx.array([[1.0, 2.0], [3.0, 4.0]])
        assert mx.array_equal(result, expected)

    def test_concatenate_axis1(self) -> None:
        """Test concatenation along axis 1."""
        backend = MLXBackend()
        a = mx.array([[1.0], [3.0]])
        b = mx.array([[2.0], [4.0]])
        result = backend.concatenate((a, b), axis=1)
        backend.eval(result)
        expected = mx.array([[1.0, 2.0], [3.0, 4.0]])
        assert mx.array_equal(result, expected)

    def test_concatenate_multiple(self) -> None:
        """Test concatenating multiple arrays."""
        backend = MLXBackend()
        a = mx.array([[1.0]])
        b = mx.array([[2.0]])
        c = mx.array([[3.0]])
        result = backend.concatenate((a, b, c), axis=0)
        backend.eval(result)
        expected = mx.array([[1.0], [2.0], [3.0]])
        assert mx.array_equal(result, expected)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_shape_zeros(self) -> None:
        """Test zeros with empty dimension."""
        backend = MLXBackend()
        z = backend.zeros((0,))
        backend.eval(z)
        assert z.shape == (0,)

    def test_single_element(self) -> None:
        """Test operations on single element array."""
        backend = MLXBackend()
        x = mx.array([5.0])
        result = backend.multiply(x, x)
        backend.eval(result)
        expected = mx.array([25.0])
        assert mx.allclose(result, expected)

    def test_broadcasting(self) -> None:
        """Test broadcasting in operations."""
        backend = MLXBackend()
        a = mx.array([[1.0], [2.0], [3.0]])
        b = mx.array([1.0, 2.0, 3.0])
        result = backend.add(a, b)
        backend.eval(result)
        expected = mx.array([[2.0, 3.0, 4.0], [3.0, 4.0, 5.0], [4.0, 5.0, 6.0]])
        assert mx.allclose(result, expected)

    def test_scalar_operations(self) -> None:
        """Test operations with scalar values."""
        backend = MLXBackend()
        x = mx.array([1.0, 2.0, 3.0])
        result = backend.add(x, mx.array(10.0))
        backend.eval(result)
        expected = mx.array([11.0, 12.0, 13.0])
        assert mx.allclose(result, expected)
