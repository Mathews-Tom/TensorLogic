"""Tests for rule composition utilities.

This module tests the compose_rules, compose_and, and compose_or functions
for combining multiple logical predicates via AND/OR operations.

Test Coverage:
- Basic composition with 2+ rules (AND, OR)
- Single rule edge case
- Empty rules error handling
- Invalid operation error handling
- Broadcasting compatibility
- Cross-backend validation
- Multi-predicate rule examples (Aunt rule pattern)
- Commutativity and associativity properties
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlogic.backends import create_backend
from tensorlogic.core.composition import compose_and, compose_or, compose_rules
from tensorlogic.core.operations import step


@pytest.fixture(params=["numpy", "mlx"])
def backend(request):
    """Parametrized fixture for testing across backends."""
    return create_backend(request.param)


class TestComposeRules:
    """Test suite for compose_rules function."""

    def test_compose_and_two_rules(self, backend) -> None:
        """Compose two rules with AND operation."""
        rule1 = np.array([1.0, 1.0, 0.0, 0.0])
        rule2 = np.array([1.0, 0.0, 1.0, 0.0])

        result = compose_rules(rule1, rule2, operation="and", backend=backend)
        backend.eval(result)

        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_compose_or_two_rules(self, backend) -> None:
        """Compose two rules with OR operation."""
        rule1 = np.array([1.0, 1.0, 0.0, 0.0])
        rule2 = np.array([1.0, 0.0, 1.0, 0.0])

        result = compose_rules(rule1, rule2, operation="or", backend=backend)
        backend.eval(result)

        expected = np.array([1.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_compose_and_three_rules(self, backend) -> None:
        """Compose three rules with AND operation."""
        rule1 = np.array([1.0, 1.0, 1.0, 0.0])
        rule2 = np.array([1.0, 1.0, 0.0, 1.0])
        rule3 = np.array([1.0, 0.0, 1.0, 1.0])

        result = compose_rules(rule1, rule2, rule3, operation="and", backend=backend)
        backend.eval(result)

        # Only first position has all 1.0s
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_compose_or_three_rules(self, backend) -> None:
        """Compose three rules with OR operation."""
        rule1 = np.array([0.0, 0.0, 0.0, 0.0])
        rule2 = np.array([0.0, 0.5, 0.0, 0.0])
        rule3 = np.array([0.0, 0.0, 0.8, 0.0])

        result = compose_rules(rule1, rule2, rule3, operation="or", backend=backend)
        backend.eval(result)

        expected = np.array([0.0, 0.5, 0.8, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_single_rule_returns_unchanged(self, backend) -> None:
        """Composing a single rule returns the rule unchanged."""
        rule = np.array([1.0, 0.5, 0.0])

        result_and = compose_rules(rule, operation="and", backend=backend)
        result_or = compose_rules(rule, operation="or", backend=backend)

        backend.eval(result_and, result_or)

        np.testing.assert_array_equal(result_and, rule)
        np.testing.assert_array_equal(result_or, rule)

    def test_empty_rules_raises_error(self, backend) -> None:
        """Composing zero rules raises ValueError."""
        with pytest.raises(ValueError, match="At least one rule required"):
            compose_rules(operation="and", backend=backend)

    def test_invalid_operation_raises_error(self, backend) -> None:
        """Invalid operation raises ValueError."""
        rule = np.array([1.0, 0.0])

        with pytest.raises(ValueError, match="Invalid operation 'xor'"):
            compose_rules(rule, operation="xor", backend=backend)

    def test_default_operation_is_and(self, backend) -> None:
        """Default operation is 'and'."""
        rule1 = np.array([1.0, 1.0, 0.0])
        rule2 = np.array([1.0, 0.0, 1.0])

        # Don't specify operation (should default to 'and')
        result = compose_rules(rule1, rule2, backend=backend)
        backend.eval(result)

        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)


class TestComposeAnd:
    """Test suite for compose_and convenience function."""

    def test_compose_and_two_rules(self, backend) -> None:
        """AND composition of two rules."""
        rule1 = np.array([1.0, 1.0, 0.0, 0.0])
        rule2 = np.array([1.0, 0.0, 1.0, 0.0])

        result = compose_and(rule1, rule2, backend=backend)
        backend.eval(result)

        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_compose_and_multiple_rules(self, backend) -> None:
        """AND composition of multiple rules."""
        rules = [
            np.array([1.0, 1.0, 1.0, 1.0]),
            np.array([1.0, 1.0, 1.0, 0.0]),
            np.array([1.0, 1.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
        ]

        result = compose_and(*rules, backend=backend)
        backend.eval(result)

        # Only first position has all 1.0s
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_compose_and_identity(self, backend) -> None:
        """AND with all 1s is identity."""
        rule = np.array([0.5, 0.3, 0.8])
        ones = np.ones(3)

        result = compose_and(rule, ones, backend=backend)
        backend.eval(result)

        np.testing.assert_array_almost_equal(result, rule, decimal=5)

    def test_compose_and_zero_absorbing(self, backend) -> None:
        """AND with all 0s gives all 0s (zero is absorbing element)."""
        rule = np.array([0.5, 0.3, 0.8])
        zeros = np.zeros(3)

        result = compose_and(rule, zeros, backend=backend)
        backend.eval(result)

        np.testing.assert_array_almost_equal(result, zeros, decimal=5)


class TestComposeOr:
    """Test suite for compose_or convenience function."""

    def test_compose_or_two_rules(self, backend) -> None:
        """OR composition of two rules."""
        rule1 = np.array([1.0, 1.0, 0.0, 0.0])
        rule2 = np.array([1.0, 0.0, 1.0, 0.0])

        result = compose_or(rule1, rule2, backend=backend)
        backend.eval(result)

        expected = np.array([1.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_compose_or_multiple_rules(self, backend) -> None:
        """OR composition of multiple rules."""
        rules = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.3, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.5, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.7]),
        ]

        result = compose_or(*rules, backend=backend)
        backend.eval(result)

        expected = np.array([0.0, 0.3, 0.5, 0.7])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_compose_or_identity(self, backend) -> None:
        """OR with all 0s is identity."""
        rule = np.array([0.5, 0.3, 0.8])
        zeros = np.zeros(3)

        result = compose_or(rule, zeros, backend=backend)
        backend.eval(result)

        np.testing.assert_array_almost_equal(result, rule, decimal=5)

    def test_compose_or_one_absorbing(self, backend) -> None:
        """OR with all 1s gives all 1s (one is absorbing element)."""
        rule = np.array([0.5, 0.3, 0.8])
        ones = np.ones(3)

        result = compose_or(rule, ones, backend=backend)
        backend.eval(result)

        np.testing.assert_array_almost_equal(result, ones, decimal=5)


class TestBroadcastingCompatibility:
    """Test broadcasting between rules of different shapes."""

    def test_vector_scalar_broadcasting(self, backend) -> None:
        """Broadcast scalar rule with vector rule."""
        rule1 = np.array([1.0, 0.5, 0.0])
        rule2 = np.array(0.8)

        result = compose_and(rule1, rule2, backend=backend)
        backend.eval(result)

        expected = np.array([0.8, 0.4, 0.0])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_matrix_vector_broadcasting(self, backend) -> None:
        """Broadcast vector rule with matrix rule."""
        rule1 = np.array([[1.0, 0.5], [0.3, 0.0]])
        rule2 = np.array([1.0, 0.5])

        result = compose_and(rule1, rule2, backend=backend)
        backend.eval(result)

        expected = np.array([[1.0, 0.25], [0.3, 0.0]])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_incompatible_shapes_propagate_error(self, backend) -> None:
        """Incompatible shapes should propagate backend error."""
        rule1 = np.array([1.0, 0.5, 0.3])
        rule2 = np.array([1.0, 0.5])

        # Different backends may have different error types
        with pytest.raises(Exception):  # Generic exception for both NumPy/MLX
            result = compose_and(rule1, rule2, backend=backend)
            backend.eval(result)


class TestMultiDimensional:
    """Test composition with multi-dimensional tensors."""

    def test_2d_and_composition(self, backend) -> None:
        """AND composition of 2D tensors."""
        rule1 = np.array([[1.0, 1.0], [1.0, 0.0]])
        rule2 = np.array([[1.0, 0.0], [1.0, 1.0]])

        result = compose_and(rule1, rule2, backend=backend)
        backend.eval(result)

        expected = np.array([[1.0, 0.0], [1.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_2d_or_composition(self, backend) -> None:
        """OR composition of 2D tensors."""
        rule1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        rule2 = np.array([[0.0, 0.5], [0.8, 0.0]])

        result = compose_or(rule1, rule2, backend=backend)
        backend.eval(result)

        expected = np.array([[1.0, 0.5], [0.8, 0.0]])
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_3d_composition(self, backend) -> None:
        """Composition of 3D tensors."""
        rule1 = np.ones((2, 3, 4))
        rule2 = np.ones((2, 3, 4)) * 0.5

        result = compose_and(rule1, rule2, backend=backend)
        backend.eval(result)

        expected = np.ones((2, 3, 4)) * 0.5
        np.testing.assert_array_almost_equal(result, expected, decimal=5)


class TestMathematicalProperties:
    """Test mathematical properties of composition operations."""

    def test_and_commutativity(self, backend) -> None:
        """AND operation is commutative: A ∧ B = B ∧ A."""
        rule_a = np.array([1.0, 0.5, 0.3, 0.0])
        rule_b = np.array([0.8, 0.6, 0.4, 0.2])

        result_ab = compose_and(rule_a, rule_b, backend=backend)
        result_ba = compose_and(rule_b, rule_a, backend=backend)

        backend.eval(result_ab, result_ba)

        np.testing.assert_array_almost_equal(result_ab, result_ba, decimal=5)

    def test_or_commutativity(self, backend) -> None:
        """OR operation is commutative: A ∨ B = B ∨ A."""
        rule_a = np.array([1.0, 0.5, 0.3, 0.0])
        rule_b = np.array([0.8, 0.6, 0.4, 0.2])

        result_ab = compose_or(rule_a, rule_b, backend=backend)
        result_ba = compose_or(rule_b, rule_a, backend=backend)

        backend.eval(result_ab, result_ba)

        np.testing.assert_array_almost_equal(result_ab, result_ba, decimal=5)

    def test_and_associativity(self, backend) -> None:
        """AND operation is associative: (A ∧ B) ∧ C = A ∧ (B ∧ C)."""
        rule_a = np.array([1.0, 0.5, 0.3])
        rule_b = np.array([0.8, 0.6, 0.4])
        rule_c = np.array([0.7, 0.5, 0.2])

        # (A ∧ B) ∧ C
        result_ab = compose_and(rule_a, rule_b, backend=backend)
        result_ab_c = compose_and(result_ab, rule_c, backend=backend)

        # A ∧ (B ∧ C)
        result_bc = compose_and(rule_b, rule_c, backend=backend)
        result_a_bc = compose_and(rule_a, result_bc, backend=backend)

        backend.eval(result_ab_c, result_a_bc)

        np.testing.assert_array_almost_equal(result_ab_c, result_a_bc, decimal=5)

    def test_or_associativity(self, backend) -> None:
        """OR operation is associative: (A ∨ B) ∨ C = A ∨ (B ∨ C)."""
        rule_a = np.array([0.0, 0.3, 0.5])
        rule_b = np.array([0.2, 0.4, 0.1])
        rule_c = np.array([0.1, 0.2, 0.3])

        # (A ∨ B) ∨ C
        result_ab = compose_or(rule_a, rule_b, backend=backend)
        result_ab_c = compose_or(result_ab, rule_c, backend=backend)

        # A ∨ (B ∨ C)
        result_bc = compose_or(rule_b, rule_c, backend=backend)
        result_a_bc = compose_or(rule_a, result_bc, backend=backend)

        backend.eval(result_ab_c, result_a_bc)

        np.testing.assert_array_almost_equal(result_ab_c, result_a_bc, decimal=5)

    def test_and_idempotence_boolean(self, backend) -> None:
        """AND is idempotent for boolean values: A ∧ A = A (when A ∈ {0, 1})."""
        rule = np.array([1.0, 1.0, 0.0, 0.0])

        result = compose_and(rule, rule, backend=backend)
        backend.eval(result)

        np.testing.assert_array_almost_equal(result, rule, decimal=5)

    def test_or_idempotence(self, backend) -> None:
        """OR is idempotent: A ∨ A = A (holds for all values in [0,1])."""
        rule = np.array([1.0, 0.5, 0.3, 0.0])

        result = compose_or(rule, rule, backend=backend)
        backend.eval(result)

        np.testing.assert_array_almost_equal(result, rule, decimal=5)


class TestMultiPredicateRules:
    """Test multi-predicate rule composition patterns (e.g., Aunt rule)."""

    def test_aunt_rule_pattern(self, backend) -> None:
        """Test Aunt(x,z) ← Sister(x,y) ∧ Parent(y,z) pattern.

        This is the canonical example from the spec showing how to compose
        predicates with different arities using einsum for index alignment.
        """
        # Domain: 3 people (Alice=0, Bob=1, Carol=2)
        # Sister[x, y]: x is sister of y
        sister = np.array(
            [
                [0.0, 1.0, 0.0],  # Alice is sister of Bob
                [1.0, 0.0, 0.0],  # Bob is sister of Alice (symmetric)
                [0.0, 0.0, 0.0],  # Carol has no sisters
            ]
        )

        # Parent[y, z]: y is parent of z
        parent = np.array(
            [
                [0.0, 0.0, 0.0],  # Alice is not a parent
                [0.0, 0.0, 1.0],  # Bob is parent of Carol
                [0.0, 0.0, 0.0],  # Carol is not a parent
            ]
        )

        # Use einsum to align indices: Sister[x,y] * Parent[y,z] -> Combined[x,z]
        combined = backend.einsum("xy,yz->xz", sister, parent)

        # Apply step to get boolean result
        aunt = step(combined, backend=backend)
        backend.eval(aunt)

        # Expected: Alice is aunt of Carol (via being sister of Bob, who is parent of Carol)
        expected = np.array(
            [
                [0.0, 0.0, 1.0],  # Alice is aunt of Carol
                [0.0, 0.0, 0.0],  # Bob is not an aunt
                [0.0, 0.0, 0.0],  # Carol is not an aunt
            ]
        )

        np.testing.assert_array_equal(aunt, expected)

    def test_transitive_relation_composition(self, backend) -> None:
        """Test transitive relation: Grandparent(x,z) ← Parent(x,y) ∧ Parent(y,z)."""
        # Parent[x, y]: x is parent of y
        parent = np.array(
            [
                [0.0, 1.0, 0.0, 0.0],  # Person 0 is parent of person 1
                [0.0, 0.0, 1.0, 0.0],  # Person 1 is parent of person 2
                [0.0, 0.0, 0.0, 1.0],  # Person 2 is parent of person 3
                [0.0, 0.0, 0.0, 0.0],  # Person 3 is not a parent
            ]
        )

        # Grandparent[x,z] via einsum: Parent[x,y] * Parent[y,z] -> Grandparent[x,z]
        combined = backend.einsum("xy,yz->xz", parent, parent)
        grandparent = step(combined, backend=backend)
        backend.eval(grandparent)

        # Expected: 0->2 (via 1), 1->3 (via 2)
        expected = np.array(
            [
                [0.0, 0.0, 1.0, 0.0],  # Person 0 is grandparent of person 2
                [0.0, 0.0, 0.0, 1.0],  # Person 1 is grandparent of person 3
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        np.testing.assert_array_equal(grandparent, expected)

    def test_complex_composition_with_three_predicates(self, backend) -> None:
        """Test composition of three predicates with einsum alignment.

        Note: This demonstrates einsum for index alignment. The raw einsum
        results may exceed [0,1] bounds - use step() to convert to boolean.
        """
        # Complex rule: Result(a,d) ← P(a,b) ∧ Q(b,c) ∧ R(c,d)
        # Use smaller values to keep result in reasonable range for demonstration
        p = np.array([[1.0, 0.0], [0.0, 1.0]])  # Identity-like
        q = np.array([[0.5, 0.3], [0.2, 0.4]])  # Shape: (2, 2)
        r = np.array([[0.6, 0.2], [0.3, 0.7]])  # Shape: (2, 2)

        # Chain the einsum operations
        pq = backend.einsum("ab,bc->ac", p, q)
        pqr = backend.einsum("ac,cd->ad", pq, r)

        backend.eval(pqr)

        # Verify shape is correct (index alignment worked)
        assert pqr.shape == (2, 2)

        # Verify all values are non-negative (logical requirement)
        pqr_np = np.asarray(pqr)
        assert np.all(pqr_np >= 0.0)


class TestCrossBackendValidation:
    """Cross-backend validation for composition operations."""

    def test_and_composition_cross_backend(self) -> None:
        """Validate AND composition produces identical results across backends."""
        numpy_backend = create_backend("numpy")
        mlx_backend = create_backend("mlx")

        rule1 = np.array([1.0, 0.8, 0.5, 0.3, 0.0])
        rule2 = np.array([1.0, 0.6, 0.7, 0.2, 0.1])

        numpy_result = compose_and(rule1, rule2, backend=numpy_backend)
        mlx_result = compose_and(rule1, rule2, backend=mlx_backend)

        numpy_backend.eval(numpy_result)
        mlx_backend.eval(mlx_result)

        np.testing.assert_allclose(
            numpy_result,
            np.asarray(mlx_result),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_or_composition_cross_backend(self) -> None:
        """Validate OR composition produces identical results across backends."""
        numpy_backend = create_backend("numpy")
        mlx_backend = create_backend("mlx")

        rule1 = np.array([0.0, 0.2, 0.5, 0.7, 1.0])
        rule2 = np.array([0.1, 0.3, 0.4, 0.6, 0.9])

        numpy_result = compose_or(rule1, rule2, backend=numpy_backend)
        mlx_result = compose_or(rule1, rule2, backend=mlx_backend)

        numpy_backend.eval(numpy_result)
        mlx_backend.eval(mlx_result)

        np.testing.assert_allclose(
            numpy_result,
            np.asarray(mlx_result),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_multi_rule_cross_backend(self) -> None:
        """Validate multi-rule composition across backends."""
        numpy_backend = create_backend("numpy")
        mlx_backend = create_backend("mlx")

        rules = [
            np.array([1.0, 0.9, 0.7, 0.5]),
            np.array([1.0, 0.8, 0.6, 0.4]),
            np.array([1.0, 0.7, 0.5, 0.3]),
        ]

        numpy_result = compose_and(*rules, backend=numpy_backend)
        mlx_result = compose_and(*rules, backend=mlx_backend)

        numpy_backend.eval(numpy_result)
        mlx_backend.eval(mlx_result)

        np.testing.assert_allclose(
            numpy_result,
            np.asarray(mlx_result),
            rtol=1e-6,
            atol=1e-6,
        )
