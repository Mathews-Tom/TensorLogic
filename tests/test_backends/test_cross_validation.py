"""Cross-backend validation tests for TensorLogic backends.

Ensures MLX and NumPy backends produce equivalent results within FP32 tolerance.
Uses property-based testing with hypothesis to validate mathematical properties.

Testing Strategy:
1. Cross-backend comparison: MLX results match NumPy within tolerance
2. Property-based tests: Mathematical properties hold (commutativity, associativity)
3. Edge cases: Empty tensors, large batches, boundary values
4. Parametric tests: All operations tested across both backends

Coverage Target: ≥90% of backend operations
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings

from tensorlogic.backends import create_backend

# MLX availability check
try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Test configuration
BACKEND_NAMES = ["numpy"]
if MLX_AVAILABLE:
    BACKEND_NAMES.append("mlx")

# Floating-point tolerance for cross-validation
RTOL = 1e-5  # Relative tolerance (FP32 default)
ATOL = 1e-8  # Absolute tolerance


# ============================================================================
# Helper Functions
# ============================================================================


def to_backend_array(backend_name: str, data: Any) -> Any:
    """Convert Python list/nested list to backend-specific array.

    Args:
        backend_name: Name of backend ("numpy" or "mlx")
        data: Python list or nested list to convert

    Returns:
        Backend-specific array
    """
    if backend_name == "numpy":
        return np.array(data)
    elif backend_name == "mlx":
        import mlx.core as mx

        return mx.array(data)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


# ============================================================================
# Section 1: Cross-Backend Comparison Tests
# ============================================================================


class TestCrossBackendElementwise:
    """Test element-wise operations produce identical results across backends."""

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_add(self, backend_name: str) -> None:
        """Test element-wise addition."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [1.0, 2.0, 3.0])
        b = to_backend_array(backend_name, [4.0, 5.0, 6.0])
        result = backend.add(a, b)
        backend.eval(result)
        result_np = np.array(result)
        expected = [5.0, 7.0, 9.0]
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_multiply(self, backend_name: str) -> None:
        """Test element-wise multiplication (Hadamard product)."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [2.0, 3.0, 4.0])
        b = to_backend_array(backend_name, [5.0, 6.0, 7.0])
        result = backend.multiply(a, b)
        backend.eval(result)
        result_np = np.array(result)
        expected = [10.0, 18.0, 28.0]
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_subtract(self, backend_name: str) -> None:
        """Test element-wise subtraction."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [10.0, 20.0, 30.0])
        b = to_backend_array(backend_name, [1.0, 2.0, 3.0])
        result = backend.subtract(a, b)
        backend.eval(result)
        result_np = np.array(result)
        expected = [9.0, 18.0, 27.0]
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_maximum(self, backend_name: str) -> None:
        """Test element-wise maximum (logical OR)."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [1.0, 5.0, 3.0])
        b = to_backend_array(backend_name, [4.0, 2.0, 6.0])
        result = backend.maximum(a, b)
        backend.eval(result)
        result_np = np.array(result)
        expected = [4.0, 5.0, 6.0]
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_minimum(self, backend_name: str) -> None:
        """Test element-wise minimum."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [1.0, 5.0, 3.0])
        b = to_backend_array(backend_name, [4.0, 2.0, 6.0])
        result = backend.minimum(a, b)
        backend.eval(result)
        result_np = np.array(result)
        expected = [1.0, 2.0, 3.0]
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)


class TestCrossBackendEinsum:
    """Test Einstein summation produces identical results across backends."""

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_matrix_multiply(self, backend_name: str) -> None:
        """Test einsum matrix multiplication: ij,jk->ik."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [[1.0, 2.0], [3.0, 4.0]])
        b = to_backend_array(backend_name, [[5.0, 6.0], [7.0, 8.0]])
        result = backend.einsum("ij,jk->ik", a, b)
        backend.eval(result)
        expected = [[19.0, 22.0], [43.0, 50.0]]
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_dot_product(self, backend_name: str) -> None:
        """Test einsum dot product: i,i->."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [1.0, 2.0, 3.0])
        b = to_backend_array(backend_name, [4.0, 5.0, 6.0])
        result = backend.einsum("i,i->", a, b)
        backend.eval(result)
        expected = 32.0
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_outer_product(self, backend_name: str) -> None:
        """Test einsum outer product: i,j->ij."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [1.0, 2.0, 3.0])
        b = to_backend_array(backend_name, [4.0, 5.0])
        result = backend.einsum("i,j->ij", a, b)
        backend.eval(result)
        expected = [[4.0, 5.0], [8.0, 10.0], [12.0, 15.0]]
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_batch_matrix_multiply(self, backend_name: str) -> None:
        """Test einsum batch matrix multiply: bij,bjk->bik."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [[[1.0, 2.0]], [[3.0, 4.0]]])  # shape (2, 1, 2)
        b = to_backend_array(backend_name, [[[5.0], [6.0]], [[7.0], [8.0]]])  # shape (2, 2, 1)
        result = backend.einsum("bij,bjk->bik", a, b)
        backend.eval(result)
        expected = [[[17.0]], [[53.0]]]  # shape (2, 1, 1)
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)


class TestCrossBackendQuantifiers:
    """Test quantifier operations produce identical results across backends."""

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_sum_all(self, backend_name: str) -> None:
        """Test sum reduction over all axes."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [[1.0, 2.0], [3.0, 4.0]])
        result = backend.sum(a)
        backend.eval(result)
        expected = 10.0
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_sum_axis(self, backend_name: str) -> None:
        """Test sum reduction over specific axis."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [[1.0, 2.0], [3.0, 4.0]])
        result = backend.sum(a, axis=1)
        backend.eval(result)
        expected = [3.0, 7.0]
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_prod_all(self, backend_name: str) -> None:
        """Test product reduction over all axes."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [[2.0, 3.0], [4.0, 5.0]])
        result = backend.prod(a)
        backend.eval(result)
        expected = 120.0
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_prod_axis(self, backend_name: str) -> None:
        """Test product reduction over specific axis."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [[2.0, 3.0], [4.0, 5.0]])
        result = backend.prod(a, axis=0)
        backend.eval(result)
        expected = [8.0, 15.0]
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_any_operation(self, backend_name: str) -> None:
        """Test boolean any reduction."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [[1.0, 0.0], [0.0, 0.0]])
        result = backend.any(a, axis=1)
        backend.eval(result)
        expected = [True, False]
        result_np = np.array(result)
        assert np.array_equal(result_np, expected)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_all_operation(self, backend_name: str) -> None:
        """Test boolean all reduction."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [[1.0, 1.0], [1.0, 0.0]])
        result = backend.all(a, axis=1)
        backend.eval(result)
        expected = [True, False]
        result_np = np.array(result)
        assert np.array_equal(result_np, expected)


class TestCrossBackendLogical:
    """Test logical operations produce identical results across backends."""

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_step_function(self, backend_name: str) -> None:
        """Test Heaviside step function.

        Note: NumPy uses np.heaviside(x, 0.5) which returns 0.5 for x=0,
        while MLX uses mx.where(x > 0, 1.0, 0.0) which returns 0.0 for x=0.
        This test only validates non-zero behavior for cross-backend compatibility.
        """
        backend = create_backend(backend_name)
        # Test only non-zero values to avoid backend-specific behavior at x=0
        x = to_backend_array(backend_name, [-2.0, -1.0, 1.0, 2.0])
        result = backend.step(x)
        backend.eval(result)
        result_np = np.array(result)
        expected = [0.0, 0.0, 1.0, 1.0]
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_where_operation(self, backend_name: str) -> None:
        """Test conditional element selection."""
        backend = create_backend(backend_name)
        condition = to_backend_array(backend_name, [True, False, True, False])
        x = to_backend_array(backend_name, [1.0, 2.0, 3.0, 4.0])
        y = to_backend_array(backend_name, [5.0, 6.0, 7.0, 8.0])
        result = backend.where(condition, x, y)
        backend.eval(result)
        expected = [1.0, 6.0, 3.0, 8.0]
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available for cross-validation")
class TestMLXvsNumPyValidation:
    """Direct cross-validation: MLX results match NumPy within tolerance."""

    def test_elementwise_operations_match(self) -> None:
        """Verify MLX and NumPy produce identical elementwise results."""
        numpy_backend = create_backend("numpy")
        mlx_backend = create_backend("mlx")

        a_np = to_backend_array("numpy", [1.5, 2.5, 3.5, 4.5])
        b_np = to_backend_array("numpy", [0.5, 1.0, 1.5, 2.0])
        a_mlx = to_backend_array("mlx", [1.5, 2.5, 3.5, 4.5])
        b_mlx = to_backend_array("mlx", [0.5, 1.0, 1.5, 2.0])

        # Test add
        numpy_result = numpy_backend.add(a_np, b_np)
        mlx_result = mlx_backend.add(a_mlx, b_mlx)
        mlx_backend.eval(mlx_result)
        assert np.allclose(np.array(mlx_result), numpy_result, rtol=RTOL, atol=ATOL)

        # Test multiply
        numpy_result = numpy_backend.multiply(a_np, b_np)
        mlx_result = mlx_backend.multiply(a_mlx, b_mlx)
        mlx_backend.eval(mlx_result)
        assert np.allclose(np.array(mlx_result), numpy_result, rtol=RTOL, atol=ATOL)

    def test_einsum_operations_match(self) -> None:
        """Verify MLX and NumPy produce identical einsum results."""
        numpy_backend = create_backend("numpy")
        mlx_backend = create_backend("mlx")

        a_np = to_backend_array("numpy", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b_np = to_backend_array("numpy", [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        a_mlx = to_backend_array("mlx", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b_mlx = to_backend_array("mlx", [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])

        numpy_result = numpy_backend.einsum("ij,jk->ik", a_np, b_np)
        mlx_result = mlx_backend.einsum("ij,jk->ik", a_mlx, b_mlx)
        mlx_backend.eval(mlx_result)

        assert np.allclose(np.array(mlx_result), numpy_result, rtol=RTOL, atol=ATOL)

    def test_reduction_operations_match(self) -> None:
        """Verify MLX and NumPy produce identical reduction results."""
        numpy_backend = create_backend("numpy")
        mlx_backend = create_backend("mlx")

        a_np = to_backend_array("numpy", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        a_mlx = to_backend_array("mlx", [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        # Test sum
        numpy_result = numpy_backend.sum(a_np, axis=1)
        mlx_result = mlx_backend.sum(a_mlx, axis=1)
        mlx_backend.eval(mlx_result)
        assert np.allclose(np.array(mlx_result), numpy_result, rtol=RTOL, atol=ATOL)

        # Test prod
        numpy_result = numpy_backend.prod(a_np, axis=0)
        mlx_result = mlx_backend.prod(a_mlx, axis=0)
        mlx_backend.eval(mlx_result)
        assert np.allclose(np.array(mlx_result), numpy_result, rtol=RTOL, atol=ATOL)


# ============================================================================
# Section 2: Property-Based Tests (hypothesis)
# ============================================================================


class TestCommutativeProperties:
    """Test commutative properties using property-based testing."""

    @given(
        a=st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        ),
        b=st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=100)
    def test_addition_commutative(self, a: list[float], b: list[float]) -> None:
        """Property: a + b == b + a."""
        backend = create_backend("numpy")
        min_len = min(len(a), len(b))
        a_truncated = a[:min_len]
        b_truncated = b[:min_len]

        result_ab = backend.add(a_truncated, b_truncated)
        result_ba = backend.add(b_truncated, a_truncated)

        assert np.allclose(result_ab, result_ba, rtol=RTOL, atol=ATOL)

    @given(
        a=st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        ),
        b=st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=100)
    def test_multiplication_commutative(self, a: list[float], b: list[float]) -> None:
        """Property: a * b == b * a."""
        backend = create_backend("numpy")
        min_len = min(len(a), len(b))
        a_truncated = a[:min_len]
        b_truncated = b[:min_len]

        result_ab = backend.multiply(a_truncated, b_truncated)
        result_ba = backend.multiply(b_truncated, a_truncated)

        assert np.allclose(result_ab, result_ba, rtol=RTOL, atol=ATOL)

    @given(
        a=st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        ),
        b=st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        ),
    )
    @settings(max_examples=100)
    def test_maximum_commutative(self, a: list[float], b: list[float]) -> None:
        """Property: max(a, b) == max(b, a)."""
        backend = create_backend("numpy")
        min_len = min(len(a), len(b))
        a_truncated = a[:min_len]
        b_truncated = b[:min_len]

        result_ab = backend.maximum(a_truncated, b_truncated)
        result_ba = backend.maximum(b_truncated, a_truncated)

        assert np.allclose(result_ab, result_ba, rtol=RTOL, atol=ATOL)


class TestAssociativeProperties:
    """Test associative properties using property-based testing."""

    @given(
        a=st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        ),
        b=st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        ),
        c=st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=100)
    def test_addition_associative(self, a: list[float], b: list[float], c: list[float]) -> None:
        """Property: (a + b) + c == a + (b + c)."""
        backend = create_backend("numpy")
        min_len = min(len(a), len(b), len(c))
        a_truncated = a[:min_len]
        b_truncated = b[:min_len]
        c_truncated = c[:min_len]

        # (a + b) + c
        temp1 = backend.add(a_truncated, b_truncated)
        result1 = backend.add(temp1, c_truncated)

        # a + (b + c)
        temp2 = backend.add(b_truncated, c_truncated)
        result2 = backend.add(a_truncated, temp2)

        assert np.allclose(result1, result2, rtol=RTOL, atol=ATOL)

    @given(
        a=st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        ),
        b=st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        ),
        c=st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=100)
    def test_multiplication_associative(
        self, a: list[float], b: list[float], c: list[float]
    ) -> None:
        """Property: (a * b) * c == a * (b * c)."""
        backend = create_backend("numpy")
        min_len = min(len(a), len(b), len(c))
        a_truncated = a[:min_len]
        b_truncated = b[:min_len]
        c_truncated = c[:min_len]

        # (a * b) * c
        temp1 = backend.multiply(a_truncated, b_truncated)
        result1 = backend.multiply(temp1, c_truncated)

        # a * (b * c)
        temp2 = backend.multiply(b_truncated, c_truncated)
        result2 = backend.multiply(a_truncated, temp2)

        assert np.allclose(result1, result2, rtol=RTOL, atol=ATOL)


class TestDistributiveProperties:
    """Test distributive properties using property-based testing."""

    @given(
        a=st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        ),
        b=st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        ),
        c=st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=20,
        ),
    )
    @settings(max_examples=100)
    def test_distributive_property(self, a: list[float], b: list[float], c: list[float]) -> None:
        """Property: a * (b + c) == (a * b) + (a * c)."""
        backend = create_backend("numpy")
        min_len = min(len(a), len(b), len(c))
        a_truncated = a[:min_len]
        b_truncated = b[:min_len]
        c_truncated = c[:min_len]

        # a * (b + c)
        temp1 = backend.add(b_truncated, c_truncated)
        result1 = backend.multiply(a_truncated, temp1)

        # (a * b) + (a * c)
        temp2 = backend.multiply(a_truncated, b_truncated)
        temp3 = backend.multiply(a_truncated, c_truncated)
        result2 = backend.add(temp2, temp3)

        assert np.allclose(result1, result2, rtol=RTOL, atol=ATOL)


class TestIdentityProperties:
    """Test identity properties using property-based testing."""

    @given(
        a=st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_additive_identity(self, a: list[float]) -> None:
        """Property: a + 0 == a."""
        backend = create_backend("numpy")
        zeros = [0.0] * len(a)

        result = backend.add(a, zeros)

        result_np = np.array(result)
        assert np.allclose(result_np, a, rtol=RTOL, atol=ATOL)

    @given(
        a=st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_multiplicative_identity(self, a: list[float]) -> None:
        """Property: a * 1 == a."""
        backend = create_backend("numpy")
        ones = [1.0] * len(a)

        result = backend.multiply(a, ones)

        result_np = np.array(result)
        assert np.allclose(result_np, a, rtol=RTOL, atol=ATOL)

    @given(
        a=st.lists(
            st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=100)
    def test_zero_absorption(self, a: list[float]) -> None:
        """Property: a * 0 == 0."""
        backend = create_backend("numpy")
        zeros = [0.0] * len(a)

        result = backend.multiply(a, zeros)

        result_np = np.array(result)
        assert np.allclose(result_np, zeros, rtol=RTOL, atol=ATOL)


# ============================================================================
# Section 3: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases: empty tensors, large batches, boundary values."""

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_empty_tensor_1d(self, backend_name: str) -> None:
        """Test operations on empty 1D tensors."""
        backend = create_backend(backend_name)
        empty = backend.zeros((0,))
        backend.eval(empty)
        assert np.array(empty).shape == (0,)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_empty_tensor_2d_rows(self, backend_name: str) -> None:
        """Test operations on empty 2D tensors (0 rows)."""
        backend = create_backend(backend_name)
        empty = backend.zeros((0, 5))
        backend.eval(empty)
        assert np.array(empty).shape == (0, 5)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_empty_tensor_2d_cols(self, backend_name: str) -> None:
        """Test operations on empty 2D tensors (0 columns)."""
        backend = create_backend(backend_name)
        empty = backend.zeros((5, 0))
        backend.eval(empty)
        assert np.array(empty).shape == (5, 0)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_single_element_scalar(self, backend_name: str) -> None:
        """Test operations on single element tensors."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [5.0])
        b = to_backend_array(backend_name, [3.0])
        result = backend.add(a, b)
        backend.eval(result)
        result_np = np.array(result)
        assert np.allclose(result_np, [8.0], rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_single_element_matrix(self, backend_name: str) -> None:
        """Test operations on (1, 1) matrices."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [[7.0]])
        b = to_backend_array(backend_name, [[2.0]])
        result = backend.multiply(a, b)
        backend.eval(result)
        result_np = np.array(result)
        assert np.allclose(result_np, [[14.0]], rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_large_batch_1d(self, backend_name: str) -> None:
        """Test operations on large 1D batches."""
        backend = create_backend(backend_name)
        size = 1000
        a = to_backend_array(backend_name, list(range(size)))
        b = to_backend_array(backend_name, [1.0] * size)
        result = backend.add(a, b)
        backend.eval(result)
        expected = [i + 1.0 for i in range(size)]
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_large_batch_2d(self, backend_name: str) -> None:
        """Test operations on large 2D batches."""
        backend = create_backend(backend_name)
        a = backend.ones((100, 100))
        b = backend.ones((100, 100))
        result = backend.add(a, b)
        backend.eval(result)
        result_np = np.array(result)
        assert np.allclose(result_np, np.full((100, 100), 2.0), rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_broadcasting_row_col(self, backend_name: str) -> None:
        """Test broadcasting: (3, 1) + (1, 3) -> (3, 3)."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [[1.0], [2.0], [3.0]])  # shape (3, 1)
        b = to_backend_array(backend_name, [[4.0, 5.0, 6.0]])  # shape (1, 3)
        result = backend.add(a, b)
        backend.eval(result)
        expected = [[5.0, 6.0, 7.0], [6.0, 7.0, 8.0], [7.0, 8.0, 9.0]]
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_boundary_value_zero(self, backend_name: str) -> None:
        """Test operations with zero boundary value."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [0.0, 0.0, 0.0])
        b = to_backend_array(backend_name, [1.0, 2.0, 3.0])
        result = backend.add(a, b)
        backend.eval(result)
        result_np = np.array(result)
        assert np.allclose(result_np, b, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_boundary_value_one(self, backend_name: str) -> None:
        """Test operations with one boundary value."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [1.0, 1.0, 1.0])
        b = to_backend_array(backend_name, [5.0, 10.0, 15.0])
        result = backend.multiply(a, b)
        backend.eval(result)
        result_np = np.array(result)
        assert np.allclose(result_np, b, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_boundary_value_negative_one(self, backend_name: str) -> None:
        """Test operations with -1 boundary value."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [-1.0, -1.0, -1.0])
        b = to_backend_array(backend_name, [2.0, 3.0, 4.0])
        result = backend.multiply(a, b)
        backend.eval(result)
        result_np = np.array(result)
        assert np.allclose(result_np, [-2.0, -3.0, -4.0], rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_mixed_signs(self, backend_name: str) -> None:
        """Test operations with mixed positive/negative/zero values."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [-5.0, 0.0, 5.0, -3.0, 7.0])
        b = to_backend_array(backend_name, [2.0, -3.0, 0.0, 4.0, -1.0])
        result = backend.add(a, b)
        backend.eval(result)
        expected = [-3.0, -3.0, 5.0, 1.0, 6.0]
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)


# ============================================================================
# Section 4: Utility Operations Cross-Validation
# ============================================================================


class TestCrossBackendUtilities:
    """Test utility operations produce identical results across backends."""

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_reshape(self, backend_name: str) -> None:
        """Test reshape operation."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        result = backend.reshape(a, (2, 3))
        backend.eval(result)
        expected = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_transpose(self, backend_name: str) -> None:
        """Test transpose operation."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = backend.transpose(a, (1, 0))
        backend.eval(result)
        expected = [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]]
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_expand_dims(self, backend_name: str) -> None:
        """Test expand_dims operation."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [1.0, 2.0, 3.0])
        result = backend.expand_dims(a, axis=0)
        backend.eval(result)
        expected = [[1.0, 2.0, 3.0]]
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_squeeze(self, backend_name: str) -> None:
        """Test squeeze operation."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [[1.0, 2.0, 3.0]])
        result = backend.squeeze(a, axis=0)
        backend.eval(result)
        expected = [1.0, 2.0, 3.0]
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize("backend_name", BACKEND_NAMES)
    def test_concatenate(self, backend_name: str) -> None:
        """Test concatenate operation."""
        backend = create_backend(backend_name)
        a = to_backend_array(backend_name, [[1.0, 2.0], [3.0, 4.0]])
        b = to_backend_array(backend_name, [[5.0, 6.0], [7.0, 8.0]])
        result = backend.concatenate((a, b), axis=0)
        backend.eval(result)
        expected = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
        result_np = np.array(result)
        assert np.allclose(result_np, expected, rtol=RTOL, atol=ATOL)
