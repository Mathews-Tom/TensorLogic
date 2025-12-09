"""Unit tests for core logical operations.

Tests verify correctness of AND, OR, NOT operations against truth tables,
mathematical properties (associativity, commutativity, distributivity),
and backend abstraction (MLX and NumPy).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlogic.backends import create_backend
from tensorlogic.core.operations import (
    logical_and,
    logical_implies,
    logical_not,
    logical_or,
)


@pytest.fixture(params=["numpy", "mlx"])
def backend(request):
    """Parametrized fixture for testing across backends."""
    return create_backend(request.param)


class TestLogicalAnd:
    """Tests for logical_and operation."""

    def test_truth_table(self, backend) -> None:
        """Verify AND operation matches truth table."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])
        expected = np.array([1.0, 0.0, 0.0, 0.0])

        result = logical_and(a, b, backend=backend)
        backend.eval(result)  # Force MLX evaluation
        np.testing.assert_array_equal(result, expected)

    def test_commutativity(self, backend) -> None:
        """Verify a ∧ b = b ∧ a."""
        a = np.array([1.0, 0.0, 1.0])
        b = np.array([0.0, 1.0, 1.0])

        result_ab = logical_and(a, b, backend=backend)
        result_ba = logical_and(b, a, backend=backend)
        backend.eval(result_ab, result_ba)

        np.testing.assert_array_equal(result_ab, result_ba)

    def test_associativity(self, backend) -> None:
        """Verify (a ∧ b) ∧ c = a ∧ (b ∧ c)."""
        a = np.array([1.0, 1.0, 1.0, 0.0])
        b = np.array([1.0, 1.0, 0.0, 1.0])
        c = np.array([1.0, 0.0, 1.0, 1.0])

        # (a ∧ b) ∧ c
        ab = logical_and(a, b, backend=backend)
        result_left = logical_and(ab, c, backend=backend)

        # a ∧ (b ∧ c)
        bc = logical_and(b, c, backend=backend)
        result_right = logical_and(a, bc, backend=backend)

        backend.eval(result_left, result_right)
        np.testing.assert_array_equal(result_left, result_right)

    def test_idempotence(self, backend) -> None:
        """Verify a ∧ a = a."""
        a = np.array([1.0, 0.0, 1.0, 0.0])
        result = logical_and(a, a, backend=backend)
        backend.eval(result)
        np.testing.assert_array_equal(result, a)

    def test_identity(self, backend) -> None:
        """Verify a ∧ 1 = a."""
        a = np.array([1.0, 0.0, 1.0, 0.0])
        ones = np.ones_like(a)

        result = logical_and(a, ones, backend=backend)
        backend.eval(result)
        np.testing.assert_array_equal(result, a)

    def test_annihilator(self, backend) -> None:
        """Verify a ∧ 0 = 0."""
        a = np.array([1.0, 0.0, 1.0, 0.0])
        zeros = np.zeros_like(a)

        result = logical_and(a, zeros, backend=backend)
        backend.eval(result)
        np.testing.assert_array_equal(result, zeros)

    def test_broadcasting(self, backend) -> None:
        """Verify AND works with broadcasting."""
        a = np.array([[1.0, 0.0], [1.0, 1.0]])
        b = np.array([1.0, 1.0])  # Broadcasting over first dimension
        expected = np.array([[1.0, 0.0], [1.0, 1.0]])

        result = logical_and(a, b, backend=backend)
        backend.eval(result)
        np.testing.assert_array_equal(result, expected)

    def test_multidimensional(self, backend) -> None:
        """Verify AND works with multidimensional tensors."""
        a = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]])
        b = np.array([[[1.0, 1.0], [0.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]])
        expected = np.array([[[1.0, 0.0], [0.0, 0.0]], [[1.0, 0.0], [0.0, 0.0]]])

        result = logical_and(a, b, backend=backend)
        backend.eval(result)
        np.testing.assert_array_equal(result, expected)


class TestLogicalOr:
    """Tests for logical_or operation."""

    def test_truth_table(self, backend) -> None:
        """Verify OR operation matches truth table."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])
        expected = np.array([1.0, 1.0, 1.0, 0.0])

        result = logical_or(a, b, backend=backend)
        backend.eval(result)
        np.testing.assert_array_equal(result, expected)

    def test_commutativity(self, backend) -> None:
        """Verify a ∨ b = b ∨ a."""
        a = np.array([1.0, 0.0, 1.0])
        b = np.array([0.0, 1.0, 1.0])

        result_ab = logical_or(a, b, backend=backend)
        result_ba = logical_or(b, a, backend=backend)
        backend.eval(result_ab, result_ba)

        np.testing.assert_array_equal(result_ab, result_ba)

    def test_associativity(self, backend) -> None:
        """Verify (a ∨ b) ∨ c = a ∨ (b ∨ c)."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])
        c = np.array([0.0, 1.0, 0.0, 1.0])

        # (a ∨ b) ∨ c
        ab = logical_or(a, b, backend=backend)
        result_left = logical_or(ab, c, backend=backend)

        # a ∨ (b ∨ c)
        bc = logical_or(b, c, backend=backend)
        result_right = logical_or(a, bc, backend=backend)

        backend.eval(result_left, result_right)
        np.testing.assert_array_equal(result_left, result_right)

    def test_idempotence(self, backend) -> None:
        """Verify a ∨ a = a."""
        a = np.array([1.0, 0.0, 1.0, 0.0])
        result = logical_or(a, a, backend=backend)
        backend.eval(result)
        np.testing.assert_array_equal(result, a)

    def test_identity(self, backend) -> None:
        """Verify a ∨ 0 = a."""
        a = np.array([1.0, 0.0, 1.0, 0.0])
        zeros = np.zeros_like(a)

        result = logical_or(a, zeros, backend=backend)
        backend.eval(result)
        np.testing.assert_array_equal(result, a)

    def test_annihilator(self, backend) -> None:
        """Verify a ∨ 1 = 1."""
        a = np.array([1.0, 0.0, 1.0, 0.0])
        ones = np.ones_like(a)

        result = logical_or(a, ones, backend=backend)
        backend.eval(result)
        np.testing.assert_array_equal(result, ones)

    def test_broadcasting(self, backend) -> None:
        """Verify OR works with broadcasting."""
        a = np.array([[1.0, 0.0], [0.0, 0.0]])
        b = np.array([0.0, 1.0])  # Broadcasting over first dimension
        expected = np.array([[1.0, 1.0], [0.0, 1.0]])

        result = logical_or(a, b, backend=backend)
        backend.eval(result)
        np.testing.assert_array_equal(result, expected)

    def test_multidimensional(self, backend) -> None:
        """Verify OR works with multidimensional tensors."""
        a = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [0.0, 0.0]]])
        b = np.array([[[0.0, 0.0], [1.0, 0.0]], [[1.0, 1.0], [0.0, 0.0]]])
        expected = np.array([[[1.0, 0.0], [1.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]])

        result = logical_or(a, b, backend=backend)
        backend.eval(result)
        np.testing.assert_array_equal(result, expected)


class TestLogicalNot:
    """Tests for logical_not operation."""

    def test_truth_table(self, backend) -> None:
        """Verify NOT operation matches truth table."""
        a = np.array([1.0, 0.0, 1.0, 0.0])
        expected = np.array([0.0, 1.0, 0.0, 1.0])

        result = logical_not(a, backend=backend)
        backend.eval(result)
        np.testing.assert_array_equal(result, expected)

    def test_double_negation(self, backend) -> None:
        """Verify ¬¬a = a (involution property)."""
        a = np.array([1.0, 0.0, 1.0, 0.0])

        not_a = logical_not(a, backend=backend)
        not_not_a = logical_not(not_a, backend=backend)
        backend.eval(not_not_a)

        np.testing.assert_allclose(not_not_a, a, atol=1e-7)

    def test_complement(self, backend) -> None:
        """Verify a ∧ ¬a = 0 (law of non-contradiction)."""
        a = np.array([1.0, 0.0, 1.0, 0.0])
        zeros = np.zeros_like(a)

        not_a = logical_not(a, backend=backend)
        result = logical_and(a, not_a, backend=backend)
        backend.eval(result)

        np.testing.assert_array_equal(result, zeros)

    def test_excluded_middle(self, backend) -> None:
        """Verify a ∨ ¬a = 1 (law of excluded middle)."""
        a = np.array([1.0, 0.0, 1.0, 0.0])
        ones = np.ones_like(a)

        not_a = logical_not(a, backend=backend)
        result = logical_or(a, not_a, backend=backend)
        backend.eval(result)

        np.testing.assert_array_equal(result, ones)

    def test_multidimensional(self, backend) -> None:
        """Verify NOT works with multidimensional tensors."""
        a = np.array([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0], [1.0, 1.0]]])
        expected = np.array([[[0.0, 1.0], [1.0, 0.0]], [[1.0, 1.0], [0.0, 0.0]]])

        result = logical_not(a, backend=backend)
        backend.eval(result)
        np.testing.assert_array_equal(result, expected)


class TestDistributivity:
    """Tests for distributive properties between AND and OR."""

    def test_and_over_or(self, backend) -> None:
        """Verify a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])
        c = np.array([0.0, 1.0, 0.0, 1.0])

        # Left side: a ∧ (b ∨ c)
        bc = logical_or(b, c, backend=backend)
        left = logical_and(a, bc, backend=backend)

        # Right side: (a ∧ b) ∨ (a ∧ c)
        ab = logical_and(a, b, backend=backend)
        ac = logical_and(a, c, backend=backend)
        right = logical_or(ab, ac, backend=backend)

        backend.eval(left, right)
        np.testing.assert_array_equal(left, right)

    def test_or_over_and(self, backend) -> None:
        """Verify a ∨ (b ∧ c) = (a ∨ b) ∧ (a ∨ c)."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])
        c = np.array([0.0, 1.0, 0.0, 1.0])

        # Left side: a ∨ (b ∧ c)
        bc = logical_and(b, c, backend=backend)
        left = logical_or(a, bc, backend=backend)

        # Right side: (a ∨ b) ∧ (a ∨ c)
        ab = logical_or(a, b, backend=backend)
        ac = logical_or(a, c, backend=backend)
        right = logical_and(ab, ac, backend=backend)

        backend.eval(left, right)
        np.testing.assert_array_equal(left, right)


class TestDeMorganLaws:
    """Tests for De Morgan's laws."""

    def test_demorgan_and(self, backend) -> None:
        """Verify ¬(a ∧ b) = ¬a ∨ ¬b."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])

        # Left side: ¬(a ∧ b)
        ab = logical_and(a, b, backend=backend)
        left = logical_not(ab, backend=backend)

        # Right side: ¬a ∨ ¬b
        not_a = logical_not(a, backend=backend)
        not_b = logical_not(b, backend=backend)
        right = logical_or(not_a, not_b, backend=backend)

        backend.eval(left, right)
        np.testing.assert_allclose(left, right, atol=1e-7)

    def test_demorgan_or(self, backend) -> None:
        """Verify ¬(a ∨ b) = ¬a ∧ ¬b."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])

        # Left side: ¬(a ∨ b)
        ab = logical_or(a, b, backend=backend)
        left = logical_not(ab, backend=backend)

        # Right side: ¬a ∧ ¬b
        not_a = logical_not(a, backend=backend)
        not_b = logical_not(b, backend=backend)
        right = logical_and(not_a, not_b, backend=backend)

        backend.eval(left, right)
        np.testing.assert_allclose(left, right, atol=1e-7)


class TestLogicalImplies:
    """Tests for logical_implies operation."""

    def test_truth_table(self, backend) -> None:
        """Verify IMPLIES operation matches truth table."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])
        expected = np.array([1.0, 0.0, 1.0, 1.0])

        result = logical_implies(a, b, backend=backend)
        backend.eval(result)
        np.testing.assert_array_equal(result, expected)

    def test_material_implication(self, backend) -> None:
        """Verify a → b = ¬a ∨ b (material implication)."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])

        # Direct implementation: a → b
        result_implies = logical_implies(a, b, backend=backend)

        # Equivalence: ¬a ∨ b
        not_a = logical_not(a, backend=backend)
        result_equiv = logical_or(not_a, b, backend=backend)

        backend.eval(result_implies, result_equiv)
        np.testing.assert_array_equal(result_implies, result_equiv)

    def test_contrapositive(self, backend) -> None:
        """Verify a → b ≡ ¬b → ¬a (contrapositive law)."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])

        # a → b
        result_forward = logical_implies(a, b, backend=backend)

        # ¬b → ¬a
        not_a = logical_not(a, backend=backend)
        not_b = logical_not(b, backend=backend)
        result_contrapositive = logical_implies(not_b, not_a, backend=backend)

        backend.eval(result_forward, result_contrapositive)
        np.testing.assert_array_equal(result_forward, result_contrapositive)

    def test_modus_ponens(self, backend) -> None:
        """Verify (a ∧ (a → b)) → b is a tautology."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])
        ones = np.ones_like(a)

        # a ∧ (a → b)
        a_implies_b = logical_implies(a, b, backend=backend)
        premise = logical_and(a, a_implies_b, backend=backend)

        # (a ∧ (a → b)) → b
        result = logical_implies(premise, b, backend=backend)
        backend.eval(result)

        # Should always be true (tautology)
        np.testing.assert_array_equal(result, ones)

    def test_chain_rule(self, backend) -> None:
        """Verify ((a → b) ∧ (b → c)) → (a → c) is a tautology."""
        a = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        b = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
        c = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        ones = np.ones_like(a)

        # (a → b) ∧ (b → c)
        a_implies_b = logical_implies(a, b, backend=backend)
        b_implies_c = logical_implies(b, c, backend=backend)
        premise = logical_and(a_implies_b, b_implies_c, backend=backend)

        # a → c
        a_implies_c = logical_implies(a, c, backend=backend)

        # ((a → b) ∧ (b → c)) → (a → c)
        result = logical_implies(premise, a_implies_c, backend=backend)
        backend.eval(result)

        # Should always be true (tautology)
        np.testing.assert_array_equal(result, ones)

    def test_non_commutativity(self, backend) -> None:
        """Verify a → b ≠ b → a (not commutative in general)."""
        # Choose specific values where a → b ≠ b → a
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])

        result_ab = logical_implies(a, b, backend=backend)
        result_ba = logical_implies(b, a, backend=backend)
        backend.eval(result_ab, result_ba)

        # a → b = [0.0, 1.0], b → a = [1.0, 0.0]
        # They should NOT be equal
        assert not np.array_equal(result_ab, result_ba)

    def test_tautology_cases(self, backend) -> None:
        """Verify tautology cases: a → a = 1, 0 → a = 1, a → 1 = 1."""
        a = np.array([1.0, 0.0, 1.0, 0.0])
        zeros = np.zeros_like(a)
        ones = np.ones_like(a)

        # a → a = 1 (tautology)
        result_self = logical_implies(a, a, backend=backend)

        # 0 → a = 1 (ex falso quodlibet)
        result_false_antecedent = logical_implies(zeros, a, backend=backend)

        # a → 1 = 1 (verum ex quolibet)
        result_true_consequent = logical_implies(a, ones, backend=backend)

        backend.eval(result_self, result_false_antecedent, result_true_consequent)

        np.testing.assert_array_equal(result_self, ones)
        np.testing.assert_array_equal(result_false_antecedent, ones)
        np.testing.assert_array_equal(result_true_consequent, ones)

    def test_contradiction_case(self, backend) -> None:
        """Verify 1 → 0 = 0 (only false case)."""
        ones = np.array([1.0, 1.0])
        zeros = np.array([0.0, 0.0])
        expected = np.array([0.0, 0.0])

        result = logical_implies(ones, zeros, backend=backend)
        backend.eval(result)

        np.testing.assert_array_equal(result, expected)

    def test_broadcasting(self, backend) -> None:
        """Verify IMPLIES works with broadcasting."""
        a = np.array([[1.0, 0.0], [1.0, 1.0]])
        b = np.array([0.0, 1.0])  # Broadcasting over first dimension
        expected = np.array([[0.0, 1.0], [0.0, 1.0]])

        result = logical_implies(a, b, backend=backend)
        backend.eval(result)
        np.testing.assert_array_equal(result, expected)

    def test_multidimensional(self, backend) -> None:
        """Verify IMPLIES works with multidimensional tensors."""
        a = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]])
        b = np.array([[[1.0, 1.0], [0.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]])
        expected = np.array([[[1.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 1.0]]])

        result = logical_implies(a, b, backend=backend)
        backend.eval(result)
        np.testing.assert_array_equal(result, expected)

    def test_with_other_operations(self, backend) -> None:
        """Verify IMPLIES composes correctly with AND/OR/NOT."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])
        c = np.array([0.0, 1.0, 0.0, 1.0])

        # (a → b) ∧ (b → c)
        ab = logical_implies(a, b, backend=backend)
        bc = logical_implies(b, c, backend=backend)
        result = logical_and(ab, bc, backend=backend)
        backend.eval(result)

        # Manual calculation:
        # a → b = [1, 0, 1, 1]
        # b → c = [0, 1, 0, 1]
        # (a → b) ∧ (b → c) = [0, 0, 0, 1]
        expected = np.array([0.0, 0.0, 0.0, 1.0])
        np.testing.assert_array_equal(result, expected)


class TestEdgeCases:
    """Tests for edge cases and special values."""

    def test_empty_tensor(self, backend) -> None:
        """Verify operations work with empty tensors."""
        a = np.array([])
        b = np.array([])

        result_and = logical_and(a, b, backend=backend)
        result_or = logical_or(a, b, backend=backend)
        result_not = logical_not(a, backend=backend)
        result_implies = logical_implies(a, b, backend=backend)

        backend.eval(result_and, result_or, result_not, result_implies)

        np.testing.assert_array_equal(result_and, np.array([]))
        np.testing.assert_array_equal(result_or, np.array([]))
        np.testing.assert_array_equal(result_not, np.array([]))
        np.testing.assert_array_equal(result_implies, np.array([]))

    def test_scalar_tensors(self, backend) -> None:
        """Verify operations work with scalar tensors."""
        a = np.array(1.0)
        b = np.array(0.0)

        result_and = logical_and(a, b, backend=backend)
        result_or = logical_or(a, b, backend=backend)
        result_not = logical_not(a, backend=backend)
        result_implies = logical_implies(a, b, backend=backend)

        backend.eval(result_and, result_or, result_not, result_implies)

        np.testing.assert_array_equal(result_and, 0.0)
        np.testing.assert_array_equal(result_or, 1.0)
        np.testing.assert_array_equal(result_not, 0.0)
        np.testing.assert_array_equal(result_implies, 0.0)  # 1 → 0 = 0

    def test_large_tensors(self, backend) -> None:
        """Verify operations work efficiently with large tensors."""
        size = 10000
        a = np.ones(size)
        b = np.zeros(size)

        result_and = logical_and(a, b, backend=backend)
        result_or = logical_or(a, b, backend=backend)
        result_not = logical_not(a, backend=backend)
        result_implies = logical_implies(a, b, backend=backend)

        backend.eval(result_and, result_or, result_not, result_implies)

        np.testing.assert_array_equal(result_and, np.zeros(size))
        np.testing.assert_array_equal(result_or, np.ones(size))
        np.testing.assert_array_equal(result_not, np.zeros(size))
        np.testing.assert_array_equal(result_implies, np.zeros(size))  # 1 → 0 = 0
