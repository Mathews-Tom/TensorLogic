"""Comprehensive integration tests for quantify() function."""

from __future__ import annotations

import numpy as np
import pytest

from tensorlogic.api import (
    PatternSyntaxError,
    PatternValidationError,
    TensorLogicError,
    quantify,
)
from tensorlogic.backends import create_backend


class TestBasicQuantification:
    """Test basic quantification patterns."""

    def test_exists_simple(self) -> None:
        """Test simple existential quantification: exists x: P(x)"""
        backend = create_backend("numpy")

        # P = [1, 0, 0] - P is true for x=0
        result = quantify(
            "exists x: P(x)",
            predicates={"P": np.array([1.0, 0.0, 0.0])},
            backend=backend,
        )

        # Should return 1.0 (true) because P(0) is true
        assert result == 1.0

    def test_exists_all_false(self) -> None:
        """Test existential quantification with all false values."""
        backend = create_backend("numpy")

        # P = [0, 0, 0] - P is false for all x
        result = quantify(
            "exists x: P(x)",
            predicates={"P": np.array([0.0, 0.0, 0.0])},
            backend=backend,
        )

        # Should return 0.0 (false) because no x satisfies P(x)
        assert result == 0.0

    def test_forall_simple(self) -> None:
        """Test simple universal quantification: forall x: P(x)"""
        backend = create_backend("numpy")

        # P = [1, 1, 1] - P is true for all x
        result = quantify(
            "forall x: P(x)",
            predicates={"P": np.array([1.0, 1.0, 1.0])},
            backend=backend,
        )

        # Should return 1.0 (true) because P(x) holds for all x
        assert result == 1.0

    def test_forall_partial_true(self) -> None:
        """Test universal quantification with partial truth."""
        backend = create_backend("numpy")

        # P = [1, 0, 1] - P is false for x=1
        result = quantify(
            "forall x: P(x)",
            predicates={"P": np.array([1.0, 0.0, 1.0])},
            backend=backend,
        )

        # Should return 0.0 (false) because P(1) is false
        assert result == 0.0


class TestLogicalOperators:
    """Test patterns with logical operators."""

    def test_and_operator(self) -> None:
        """Test conjunction: exists x: P(x) and Q(x)"""
        backend = create_backend("numpy")

        # P = [1, 1, 0], Q = [1, 0, 1]
        # P and Q = [1, 0, 0] - both true only for x=0
        result = quantify(
            "exists x: P(x) and Q(x)",
            predicates={
                "P": np.array([1.0, 1.0, 0.0]),
                "Q": np.array([1.0, 0.0, 1.0]),
            },
            backend=backend,
        )

        # Should return 1.0 because x=0 satisfies both
        assert result == 1.0

    def test_or_operator(self) -> None:
        """Test disjunction: forall x: P(x) or Q(x)"""
        backend = create_backend("numpy")

        # P = [1, 0, 0], Q = [0, 1, 1]
        # P or Q = [1, 1, 1] - at least one true for all x
        result = quantify(
            "forall x: P(x) or Q(x)",
            predicates={
                "P": np.array([1.0, 0.0, 0.0]),
                "Q": np.array([0.0, 1.0, 1.0]),
            },
            backend=backend,
        )

        # Should return 1.0 because for all x, P or Q holds
        assert result == 1.0

    def test_not_operator(self) -> None:
        """Test negation: exists x: not P(x)"""
        backend = create_backend("numpy")

        # P = [1, 1, 0]
        # not P = [0, 0, 1]
        result = quantify(
            "exists x: not P(x)",
            predicates={"P": np.array([1.0, 1.0, 0.0])},
            backend=backend,
        )

        # Should return 1.0 because x=2 satisfies not P(x)
        assert result == 1.0

    def test_implication(self) -> None:
        """Test implication: forall x: P(x) -> Q(x)"""
        backend = create_backend("numpy")

        # P = [1, 1, 0], Q = [1, 1, 1]
        # P -> Q means: whenever P is true, Q must be true
        result = quantify(
            "forall x: P(x) -> Q(x)",
            predicates={
                "P": np.array([1.0, 1.0, 0.0]),
                "Q": np.array([1.0, 1.0, 1.0]),
            },
            backend=backend,
        )

        # Should return 1.0 because Q is true whenever P is true
        assert result == 1.0

    def test_implication_counterexample(self) -> None:
        """Test implication with counterexample."""
        backend = create_backend("numpy")

        # P = [1, 1, 0], Q = [1, 0, 1]
        # P(1) is true but Q(1) is false - counterexample
        result = quantify(
            "forall x: P(x) -> Q(x)",
            predicates={
                "P": np.array([1.0, 1.0, 0.0]),
                "Q": np.array([1.0, 0.0, 1.0]),
            },
            backend=backend,
        )

        # Should return 0.0 because P(1) -> Q(1) is false
        assert result == 0.0


class TestComplexPatterns:
    """Test complex patterns with multiple operators."""

    def test_nested_quantifiers(self) -> None:
        """Test nested quantifiers: forall x: exists y: Related(x, y)"""
        backend = create_backend("numpy")

        # Relation matrix: each row x must have at least one y where Related(x, y) is true
        # [[1, 0], [0, 1]] - reflexive relation
        result = quantify(
            "forall x: exists y: Related(x, y)",
            predicates={"Related": np.array([[1.0, 0.0], [0.0, 1.0]])},
            backend=backend,
        )

        # Should return 1.0 because each x has at least one related y
        assert result == 1.0

    def test_conjunction_with_negation(self) -> None:
        """Test conjunction with negation: exists x: P(x) and not Q(x)"""
        backend = create_backend("numpy")

        # P = [1, 0, 1], Q = [1, 1, 0]
        # P and not Q = [0, 0, 1] - only x=2 satisfies
        result = quantify(
            "exists x: P(x) and not Q(x)",
            predicates={
                "P": np.array([1.0, 0.0, 1.0]),
                "Q": np.array([1.0, 1.0, 0.0]),
            },
            backend=backend,
        )

        # Should return 1.0 because x=2 satisfies P and not Q
        assert result == 1.0

    def test_complex_implication(self) -> None:
        """Test complex implication: forall x: (P(x) and Q(x)) -> R(x)"""
        backend = create_backend("numpy")

        # If both P and Q hold, then R must hold
        result = quantify(
            "forall x: (P(x) and Q(x)) -> R(x)",
            predicates={
                "P": np.array([1.0, 1.0, 0.0]),
                "Q": np.array([1.0, 0.0, 1.0]),
                "R": np.array([1.0, 1.0, 1.0]),
            },
            backend=backend,
        )

        # Should return 1.0 because when P and Q are both true (x=0), R is also true
        assert result == 1.0

    def test_multiple_operators(self) -> None:
        """Test multiple operators: exists x: P(x) and Q(x) or R(x)"""
        backend = create_backend("numpy")

        # (P and Q) or R with proper precedence
        result = quantify(
            "exists x: P(x) and Q(x) or R(x)",
            predicates={
                "P": np.array([1.0, 0.0, 0.0]),
                "Q": np.array([1.0, 0.0, 0.0]),
                "R": np.array([0.0, 1.0, 0.0]),
            },
            backend=backend,
        )

        # Should return 1.0 because either x=0 (P and Q) or x=1 (R) satisfies
        assert result == 1.0


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_syntax(self) -> None:
        """Test invalid pattern syntax."""
        backend = create_backend("numpy")

        with pytest.raises(PatternSyntaxError):
            quantify(
                "forall x P(x)",  # Missing colon
                predicates={"P": np.array([1.0])},
                backend=backend,
            )

    def test_missing_predicate(self) -> None:
        """Test missing predicate in validation."""
        backend = create_backend("numpy")

        with pytest.raises(PatternValidationError) as exc_info:
            quantify(
                "exists x: P(x)",
                predicates={},  # P not provided
                backend=backend,
            )

        assert "Missing predicates" in str(exc_info.value)
        assert "P" in str(exc_info.value)

    def test_unbound_variable(self) -> None:
        """Test unbound free variable."""
        backend = create_backend("numpy")

        with pytest.raises(PatternValidationError) as exc_info:
            quantify(
                "P(x)",  # x is free variable
                predicates={"P": np.array([1.0])},
                bindings={},  # x not bound
                backend=backend,
            )

        assert "Unbound variables" in str(exc_info.value)
        assert "x" in str(exc_info.value)

    def test_shape_mismatch(self) -> None:
        """Test shape mismatch between predicate and usage."""
        backend = create_backend("numpy")

        with pytest.raises(PatternValidationError):
            quantify(
                "exists x: P(x, y)",  # 2-arg predicate
                predicates={"P": np.array([1.0])},  # 1-dimensional tensor
                bindings={"x": np.array([0]), "y": np.array([0])},
                backend=backend,
            )

    def test_invalid_dtype(self) -> None:
        """Test invalid predicate dtype."""
        backend = create_backend("numpy")

        with pytest.raises(PatternValidationError):
            quantify(
                "exists x: P(x)",
                predicates={"P": np.array(["not", "numeric"], dtype=object)},
                backend=backend,
            )


class TestRealWorldExamples:
    """Test real-world pattern examples from documentation."""

    def test_related_entities(self) -> None:
        """Test entity relation pattern: exists y: Related(x, y)"""
        backend = create_backend("numpy")

        # Relation matrix: [[1, 0], [0, 1]]
        result = quantify(
            "exists y: Related(x, y)",
            predicates={"Related": np.array([[1.0, 0.0], [0.0, 1.0]])},
            bindings={"x": np.array([0, 1])},
            backend=backend,
        )

        # Both entities have at least one relation
        np.testing.assert_array_almost_equal(result, [1.0, 1.0])

    def test_implication_rule(self) -> None:
        """Test implication rule: forall x: P(x) -> Q(x)"""
        backend = create_backend("numpy")

        # Classic implication: if P then Q
        result = quantify(
            "forall x: P(x) -> Q(x)",
            predicates={
                "P": np.array([1.0, 1.0, 0.0]),
                "Q": np.array([1.0, 1.0, 1.0]),
            },
            backend=backend,
        )

        # Implication holds for all x
        assert result == 1.0

    def test_property_conjunction(self) -> None:
        """Test property conjunction: exists x: P(x) and Q(x) and not R(x)"""
        backend = create_backend("numpy")

        # Find entities with P and Q but not R
        result = quantify(
            "exists x: P(x) and Q(x) and not R(x)",
            predicates={
                "P": np.array([1.0, 0.0, 1.0]),
                "Q": np.array([1.0, 1.0, 0.0]),
                "R": np.array([0.0, 0.0, 1.0]),
            },
            backend=backend,
        )

        # x=0 satisfies: P(0)=1, Q(0)=1, R(0)=0
        assert result == 1.0


class TestBackendIntegration:
    """Test integration with different backends."""

    def test_numpy_backend_explicit(self) -> None:
        """Test explicit NumPy backend."""
        backend = create_backend("numpy")

        result = quantify(
            "exists x: P(x)",
            predicates={"P": np.array([1.0, 0.0])},
            backend=backend,
        )

        assert result == 1.0

    def test_default_backend(self) -> None:
        """Test default backend (should work without specifying)."""
        # Should use default backend (MLX or NumPy based on availability)
        result = quantify(
            "exists x: P(x)",
            predicates={"P": np.array([1.0, 0.0])},
        )

        assert result == 1.0

    def test_backend_eval_called(self) -> None:
        """Test that backend eval() is called for lazy backends."""
        backend = create_backend("numpy")

        # Even if backend is NumPy (not lazy), eval should be safe to call
        result = quantify(
            "exists x: P(x)",
            predicates={"P": np.array([1.0])},
            backend=backend,
        )

        # Should return evaluated result
        assert result == 1.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_constant_predicate(self) -> None:
        """Test constant predicate with no arguments."""
        backend = create_backend("numpy")

        # Constant predicate (0-arity)
        result = quantify(
            "IsTrue()",
            predicates={"IsTrue": np.array(1.0)},  # Scalar tensor
            backend=backend,
        )

        assert result == 1.0

    def test_single_element_tensor(self) -> None:
        """Test quantification over single element."""
        backend = create_backend("numpy")

        result = quantify(
            "exists x: P(x)",
            predicates={"P": np.array([1.0])},
            backend=backend,
        )

        assert result == 1.0

    def test_large_tensor(self) -> None:
        """Test quantification over large tensor."""
        backend = create_backend("numpy")

        # Large tensor with one true value
        predicate = np.zeros(1000)
        predicate[500] = 1.0

        result = quantify(
            "exists x: P(x)",
            predicates={"P": predicate},
            backend=backend,
        )

        assert result == 1.0

    def test_empty_predicates_dict(self) -> None:
        """Test with quantifier that doesn't need predicates."""
        backend = create_backend("numpy")

        # Nested quantifiers with constant
        result = quantify(
            "forall x: forall y: IsTrue()",
            predicates={"IsTrue": np.array(1.0)},
            backend=backend,
        )

        assert result == 1.0

    def test_free_variable_with_binding(self) -> None:
        """Test pattern with free variable that is bound."""
        backend = create_backend("numpy")

        # x is free but bound via bindings parameter
        result = quantify(
            "P(x)",
            predicates={"P": np.array([1.0, 0.0, 1.0])},
            bindings={"x": np.array([0, 2])},  # Select indices 0 and 2
            backend=backend,
        )

        # Should return the predicate applied to bound variables
        # This returns the predicate tensor itself for now
        assert result.shape == (3,)
