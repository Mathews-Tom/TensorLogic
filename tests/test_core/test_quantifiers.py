"""Unit tests for quantifier operations.

Tests verify correctness of existential (∃) and universal (∀) quantification
operations in both hard (boolean) and soft (differentiable) modes.

Acceptance Criteria (CORE-007):
- exists(predicate, axis) via summation + step
- forall(predicate, axis) via product + step
- Soft variants: max (exists), min (forall) for differentiability
- Multi-axis quantification support
- Shape inference correct
- Mathematical soundness verified
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlogic.backends import create_backend
from tensorlogic.core.quantifiers import (
    exists,
    forall,
    soft_exists,
    soft_forall,
)


@pytest.fixture(params=["numpy", "mlx"])
def backend(request):
    """Parametrized fixture for testing across backends."""
    return create_backend(request.param)


class TestExists:
    """Tests for existential quantification (∃)."""

    def test_all_true(self, backend) -> None:
        """∃x.P(x) where all P(x) are true."""
        predicate = np.array([1.0, 1.0, 1.0, 1.0])
        result = exists(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert float(result) == 1.0

    def test_one_true(self, backend) -> None:
        """∃x.P(x) where exactly one P(x) is true."""
        predicate = np.array([0.0, 0.0, 1.0, 0.0])
        result = exists(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert float(result) == 1.0

    def test_all_false(self, backend) -> None:
        """∃x.P(x) where all P(x) are false."""
        predicate = np.array([0.0, 0.0, 0.0, 0.0])
        result = exists(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert float(result) == 0.0

    def test_2d_axis0(self, backend) -> None:
        """∃x.P(x,y) quantifying over rows (axis 0)."""
        predicate = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        result = exists(predicate, axis=0, backend=backend)
        backend.eval(result)
        expected = np.array([1.0, 1.0])  # Each column has at least one true
        np.testing.assert_array_equal(result, expected)

    def test_2d_axis1(self, backend) -> None:
        """∃y.P(x,y) quantifying over columns (axis 1)."""
        predicate = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        result = exists(predicate, axis=1, backend=backend)
        backend.eval(result)
        expected = np.array([0.0, 1.0, 0.0])  # Only middle row has true
        np.testing.assert_array_equal(result, expected)

    def test_multi_axis(self, backend) -> None:
        """∃x,y.P(x,y) quantifying over all axes."""
        predicate = np.array([[0.0, 0.0], [0.0, 1.0]])
        result = exists(predicate, axis=(0, 1), backend=backend)
        backend.eval(result)
        assert float(result) == 1.0  # At least one true in entire matrix

    def test_all_axes_none(self, backend) -> None:
        """∃x.P(x) with axis=None (quantify over all)."""
        predicate = np.array([[[0.0, 0.0], [1.0, 0.0]]])
        result = exists(predicate, axis=None, backend=backend)
        backend.eval(result)
        assert float(result) == 1.0

    def test_shape_preservation_2d(self, backend) -> None:
        """Verify shape reduction along specified axis."""
        predicate = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])  # Shape (2, 3)
        result = exists(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert result.shape == (3,)  # Reduced axis 0

    def test_fuzzy_values(self, backend) -> None:
        """Test with fuzzy (continuous) values in [0, 1]."""
        predicate = np.array([0.0, 0.2, 0.3, 0.0])
        result = exists(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert float(result) == 1.0  # Sum > 0, so exists

    def test_3d_tensor(self, backend) -> None:
        """Test on 3D tensor."""
        predicate = np.array([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]])
        result = exists(predicate, axis=2, backend=backend)
        backend.eval(result)
        expected = np.array([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(result, expected)


class TestForall:
    """Tests for universal quantification (∀)."""

    def test_all_true(self, backend) -> None:
        """∀x.P(x) where all P(x) are true."""
        predicate = np.array([1.0, 1.0, 1.0, 1.0])
        result = forall(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert float(result) == 1.0

    def test_one_false(self, backend) -> None:
        """∀x.P(x) where exactly one P(x) is false."""
        predicate = np.array([1.0, 1.0, 0.0, 1.0])
        result = forall(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert float(result) == 0.0

    def test_all_false(self, backend) -> None:
        """∀x.P(x) where all P(x) are false."""
        predicate = np.array([0.0, 0.0, 0.0, 0.0])
        result = forall(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert float(result) == 0.0

    def test_2d_axis0(self, backend) -> None:
        """∀x.P(x,y) quantifying over rows (axis 0)."""
        predicate = np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        result = forall(predicate, axis=0, backend=backend)
        backend.eval(result)
        expected = np.array([1.0, 0.0])  # First column all true, second has false
        np.testing.assert_array_equal(result, expected)

    def test_2d_axis1(self, backend) -> None:
        """∀y.P(x,y) quantifying over columns (axis 1)."""
        predicate = np.array([[1.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        result = forall(predicate, axis=1, backend=backend)
        backend.eval(result)
        expected = np.array([1.0, 0.0, 1.0])  # First and third rows all true
        np.testing.assert_array_equal(result, expected)

    def test_multi_axis(self, backend) -> None:
        """∀x,y.P(x,y) quantifying over all axes."""
        predicate = np.array([[1.0, 1.0], [1.0, 1.0]])
        result = forall(predicate, axis=(0, 1), backend=backend)
        backend.eval(result)
        assert float(result) == 1.0  # All true

        predicate_with_false = np.array([[1.0, 1.0], [1.0, 0.0]])
        result_false = forall(predicate_with_false, axis=(0, 1), backend=backend)
        backend.eval(result_false)
        assert float(result_false) == 0.0  # One false

    def test_all_axes_none(self, backend) -> None:
        """∀x.P(x) with axis=None (quantify over all)."""
        predicate = np.array([[[1.0, 1.0], [1.0, 1.0]]])
        result = forall(predicate, axis=None, backend=backend)
        backend.eval(result)
        assert float(result) == 1.0

    def test_shape_preservation_2d(self, backend) -> None:
        """Verify shape reduction along specified axis."""
        predicate = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])  # Shape (2, 3)
        result = forall(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert result.shape == (3,)  # Reduced axis 0

    def test_fuzzy_values_all_high(self, backend) -> None:
        """Test with high fuzzy values (product should stay high)."""
        predicate = np.array([1.0, 1.0, 1.0, 1.0])
        result = forall(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert float(result) == 1.0

    def test_fuzzy_values_one_low(self, backend) -> None:
        """Test with one low fuzzy value (product should be low)."""
        predicate = np.array([1.0, 1.0, 0.1, 1.0])
        result = forall(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert float(result) == 0.0  # Product < 0.5

    def test_3d_tensor(self, backend) -> None:
        """Test on 3D tensor."""
        predicate = np.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 0.0]]])
        result = forall(predicate, axis=2, backend=backend)
        backend.eval(result)
        expected = np.array([[1.0, 1.0], [1.0, 0.0]])
        np.testing.assert_array_equal(result, expected)


class TestSoftExists:
    """Tests for soft existential quantification (differentiable)."""

    def test_returns_maximum(self, backend) -> None:
        """Soft-∃ should return maximum value."""
        predicate = np.array([0.0, 0.3, 0.7, 0.2])
        result = soft_exists(predicate, axis=0, backend=backend)
        backend.eval(result)
        np.testing.assert_almost_equal(float(result), 0.7, decimal=5)

    def test_all_zeros(self, backend) -> None:
        """Soft-∃ with all zeros."""
        predicate = np.array([0.0, 0.0, 0.0])
        result = soft_exists(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert float(result) == 0.0

    def test_all_ones(self, backend) -> None:
        """Soft-∃ with all ones."""
        predicate = np.array([1.0, 1.0, 1.0])
        result = soft_exists(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert float(result) == 1.0

    def test_2d_axis0(self, backend) -> None:
        """Soft-∃ over axis 0 returns max per column."""
        predicate = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.5]])
        result = soft_exists(predicate, axis=0, backend=backend)
        backend.eval(result)
        expected = np.array([0.8, 0.9])  # Max of each column
        np.testing.assert_array_almost_equal(result, expected)

    def test_2d_axis1(self, backend) -> None:
        """Soft-∃ over axis 1 returns max per row."""
        predicate = np.array([[0.1, 0.9], [0.8, 0.2]])
        result = soft_exists(predicate, axis=1, backend=backend)
        backend.eval(result)
        expected = np.array([0.9, 0.8])  # Max of each row
        np.testing.assert_array_almost_equal(result, expected)

    def test_multi_axis(self, backend) -> None:
        """Soft-∃ over all axes returns global maximum."""
        predicate = np.array([[0.1, 0.5], [0.3, 0.9]])
        result = soft_exists(predicate, axis=(0, 1), backend=backend)
        backend.eval(result)
        np.testing.assert_almost_equal(float(result), 0.9, decimal=5)

    def test_preserves_gradients(self, backend) -> None:
        """Verify output is differentiable (no step function)."""
        predicate = np.array([0.2, 0.5, 0.8])
        result = soft_exists(predicate, axis=0, backend=backend)
        backend.eval(result)
        # Should be exact max, not stepped to 1.0
        assert 0.79 < float(result) < 0.81


class TestSoftForall:
    """Tests for soft universal quantification (differentiable)."""

    def test_returns_minimum(self, backend) -> None:
        """Soft-∀ should return minimum value."""
        predicate = np.array([1.0, 0.9, 0.7, 0.8])
        result = soft_forall(predicate, axis=0, backend=backend)
        backend.eval(result)
        np.testing.assert_almost_equal(float(result), 0.7, decimal=5)

    def test_all_zeros(self, backend) -> None:
        """Soft-∀ with all zeros."""
        predicate = np.array([0.0, 0.0, 0.0])
        result = soft_forall(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert float(result) == 0.0

    def test_all_ones(self, backend) -> None:
        """Soft-∀ with all ones."""
        predicate = np.array([1.0, 1.0, 1.0])
        result = soft_forall(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert float(result) == 1.0

    def test_2d_axis0(self, backend) -> None:
        """Soft-∀ over axis 0 returns min per column."""
        predicate = np.array([[0.9, 0.3], [0.8, 0.9], [0.7, 0.5]])
        result = soft_forall(predicate, axis=0, backend=backend)
        backend.eval(result)
        expected = np.array([0.7, 0.3])  # Min of each column
        np.testing.assert_array_almost_equal(result, expected)

    def test_2d_axis1(self, backend) -> None:
        """Soft-∀ over axis 1 returns min per row."""
        predicate = np.array([[0.9, 0.3], [0.8, 0.7]])
        result = soft_forall(predicate, axis=1, backend=backend)
        backend.eval(result)
        expected = np.array([0.3, 0.7])  # Min of each row
        np.testing.assert_array_almost_equal(result, expected)

    def test_multi_axis(self, backend) -> None:
        """Soft-∀ over all axes returns global minimum."""
        predicate = np.array([[0.5, 0.9], [0.3, 0.8]])
        result = soft_forall(predicate, axis=(0, 1), backend=backend)
        backend.eval(result)
        np.testing.assert_almost_equal(float(result), 0.3, decimal=5)

    def test_preserves_gradients(self, backend) -> None:
        """Verify output is differentiable (no step function)."""
        predicate = np.array([0.8, 0.5, 0.9])
        result = soft_forall(predicate, axis=0, backend=backend)
        backend.eval(result)
        # Should be exact min, not stepped to 0.0
        assert 0.49 < float(result) < 0.51


class TestQuantifierProperties:
    """Test mathematical properties of quantifiers."""

    def test_exists_tautology(self, backend) -> None:
        """Property: ∃x.True = True."""
        predicate = np.array([1.0, 1.0, 1.0])
        result = exists(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert float(result) == 1.0

    def test_forall_contradiction(self, backend) -> None:
        """Property: ∀x.False = False."""
        predicate = np.array([0.0, 0.0, 0.0])
        result = forall(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert float(result) == 0.0

    def test_exists_vs_forall_all_true(self, backend) -> None:
        """When all true: ∃ = ∀ = 1."""
        predicate = np.array([1.0, 1.0, 1.0])
        exists_result = exists(predicate, axis=0, backend=backend)
        forall_result = forall(predicate, axis=0, backend=backend)
        backend.eval(exists_result, forall_result)
        assert float(exists_result) == float(forall_result) == 1.0

    def test_exists_vs_forall_mixed(self, backend) -> None:
        """When mixed: ∃ = 1, ∀ = 0."""
        predicate = np.array([1.0, 0.0, 1.0])
        exists_result = exists(predicate, axis=0, backend=backend)
        forall_result = forall(predicate, axis=0, backend=backend)
        backend.eval(exists_result, forall_result)
        assert float(exists_result) == 1.0
        assert float(forall_result) == 0.0

    def test_soft_preserves_order(self, backend) -> None:
        """Property: soft-∀(P) ≤ soft-∃(P) (min ≤ max)."""
        predicate = np.array([0.3, 0.7, 0.5])
        soft_exists_result = soft_exists(predicate, axis=0, backend=backend)
        soft_forall_result = soft_forall(predicate, axis=0, backend=backend)
        backend.eval(soft_exists_result, soft_forall_result)
        assert float(soft_forall_result) <= float(soft_exists_result)

    def test_hard_implies_soft_exists(self, backend) -> None:
        """If hard-∃ = 1, then soft-∃ > 0."""
        predicate = np.array([0.0, 0.3, 0.0])
        hard_result = exists(predicate, axis=0, backend=backend)
        soft_result = soft_exists(predicate, axis=0, backend=backend)
        backend.eval(hard_result, soft_result)
        if float(hard_result) == 1.0:
            assert float(soft_result) > 0.0

    def test_hard_implies_soft_forall(self, backend) -> None:
        """If hard-∀ = 1, then soft-∀ = 1 (all must be 1.0)."""
        predicate = np.array([1.0, 1.0, 1.0])
        hard_result = forall(predicate, axis=0, backend=backend)
        soft_result = soft_forall(predicate, axis=0, backend=backend)
        backend.eval(hard_result, soft_result)
        if float(hard_result) == 1.0:
            assert float(soft_result) == 1.0


class TestEdgeCases:
    """Test edge cases for quantifiers."""

    def test_empty_axis(self, backend) -> None:
        """Test behavior with empty tensors."""
        predicate = np.array([])
        # NumPy sum of empty array is 0, so exists should be 0
        result = exists(predicate, axis=0, backend=backend)
        backend.eval(result)
        assert float(result) == 0.0

    def test_scalar_input(self, backend) -> None:
        """Test with scalar (0-d tensor) input."""
        predicate = 1.0
        # Quantifying over "all axes" of a scalar
        result = exists(predicate, axis=None, backend=backend)
        backend.eval(result)
        assert float(result) == 1.0

    def test_negative_axis(self, backend) -> None:
        """Test with negative axis indexing."""
        predicate = np.array([[1.0, 0.0], [0.0, 1.0]])
        # axis=-1 is last axis (columns)
        result = exists(predicate, axis=-1, backend=backend)
        backend.eval(result)
        expected = np.array([1.0, 1.0])  # Each row has at least one true
        np.testing.assert_array_equal(result, expected)

    def test_large_tensor(self, backend) -> None:
        """Test on large tensor for numerical stability."""
        rng = np.random.default_rng(42)
        predicate = rng.uniform(0.0, 1.0, size=(100, 100))

        # Should not overflow or have numerical issues
        exists_result = exists(predicate, axis=0, backend=backend)
        forall_result = forall(predicate, axis=0, backend=backend)
        backend.eval(exists_result, forall_result)

        # Convert to numpy for comparison
        exists_np = np.asarray(exists_result)
        forall_np = np.asarray(forall_result)

        # With 100 random values, exists should be all 1s, forall likely all 0s
        assert np.all(exists_np >= 0.0) and np.all(exists_np <= 1.0)
        assert np.all(forall_np >= 0.0) and np.all(forall_np <= 1.0)
