"""Comprehensive tests for pattern validation."""

from __future__ import annotations

import numpy as np
import pytest

from tensorlogic.api import (
    PatternParser,
    PatternValidationError,
    PatternValidator,
)


# Mock tensor class for testing without backend dependency
class MockTensor:
    """Mock tensor for validation testing."""

    def __init__(self, shape: tuple[int, ...], dtype: str = "float32") -> None:
        """Initialize mock tensor with shape and dtype."""
        self.shape = shape
        self.dtype = dtype


class TestVariableBindingValidation:
    """Test variable binding validation."""

    def test_validate_no_free_variables(self) -> None:
        """Test validation passes when no free variables."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("forall x: P(x)")

        # Should not raise - no free variables
        validator.validate(pattern, predicates={"P": MockTensor((10,))})

    def test_validate_free_variable_bound(self) -> None:
        """Test validation passes when free variable is bound."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x)")

        # Should not raise - x is bound
        validator.validate(
            pattern,
            predicates={"P": MockTensor((10,))},
            bindings={"x": MockTensor((5,))},
        )

    def test_validate_unbound_variable_raises(self) -> None:
        """Test validation fails on unbound variable."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x)")

        with pytest.raises(PatternValidationError) as exc_info:
            validator.validate(
                pattern,
                predicates={"P": MockTensor((10,))},
                bindings={},  # x not bound
            )

        assert "Unbound variables" in str(exc_info.value)
        assert "x" in str(exc_info.value)

    def test_validate_multiple_unbound_variables(self) -> None:
        """Test validation reports all unbound variables."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x, y) and Q(z)")

        with pytest.raises(PatternValidationError) as exc_info:
            validator.validate(
                pattern,
                predicates={"P": MockTensor((10, 10)), "Q": MockTensor((10,))},
                bindings={},
            )

        error_msg = str(exc_info.value)
        assert "x" in error_msg
        assert "y" in error_msg
        assert "z" in error_msg

    def test_validate_partial_binding_raises(self) -> None:
        """Test validation fails when only some variables bound."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x, y)")

        with pytest.raises(PatternValidationError) as exc_info:
            validator.validate(
                pattern,
                predicates={"P": MockTensor((10, 10))},
                bindings={"x": MockTensor((5,))},  # y not bound
            )

        assert "y" in str(exc_info.value)


class TestPredicateAvailabilityValidation:
    """Test predicate availability validation."""

    def test_validate_predicate_available(self) -> None:
        """Test validation passes when predicate available."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x)")

        validator.validate(
            pattern,
            predicates={"P": MockTensor((10,))},
            bindings={"x": MockTensor((5,))},
        )

    def test_validate_missing_predicate_raises(self) -> None:
        """Test validation fails on missing predicate."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x)")

        with pytest.raises(PatternValidationError) as exc_info:
            validator.validate(
                pattern,
                predicates={},  # P not provided
                bindings={"x": MockTensor((5,))},
            )

        assert "Missing predicates" in str(exc_info.value)
        assert "P" in str(exc_info.value)

    def test_validate_multiple_missing_predicates(self) -> None:
        """Test validation reports all missing predicates."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x) and Q(y) or R(z)")

        with pytest.raises(PatternValidationError) as exc_info:
            validator.validate(
                pattern,
                predicates={},
                bindings={"x": MockTensor((5,)), "y": MockTensor((5,)), "z": MockTensor((5,))},
            )

        error_msg = str(exc_info.value)
        assert "P" in error_msg
        assert "Q" in error_msg
        assert "R" in error_msg

    def test_validate_unused_predicates_ignored(self) -> None:
        """Test validation ignores predicates not used in pattern."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x)")

        # Extra predicate Q is ignored
        validator.validate(
            pattern,
            predicates={"P": MockTensor((10,)), "Q": MockTensor((5,))},
            bindings={"x": MockTensor((5,))},
        )


class TestShapeCompatibilityValidation:
    """Test shape compatibility validation."""

    def test_validate_shape_compatible(self) -> None:
        """Test validation passes with compatible shapes."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x, y)")

        # Predicate with 2 dimensions matches 2 arguments
        validator.validate(
            pattern,
            predicates={"P": MockTensor((10, 10))},
            bindings={"x": MockTensor((5,)), "y": MockTensor((5,))},
        )

    def test_validate_shape_too_few_dimensions(self) -> None:
        """Test validation fails when tensor has too few dimensions."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x, y)")

        with pytest.raises(PatternValidationError) as exc_info:
            validator.validate(
                pattern,
                predicates={"P": MockTensor((10,))},  # Only 1 dim, need 2
                bindings={"x": MockTensor((5,)), "y": MockTensor((5,))},
            )

        assert "arity mismatch" in str(exc_info.value)
        assert "P" in str(exc_info.value)

    def test_validate_constant_predicate(self) -> None:
        """Test validation of constant (0-arity) predicates."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("IsTrue()")

        # Scalar tensor for constant predicate
        validator.validate(
            pattern,
            predicates={"IsTrue": MockTensor(())},
        )

    def test_validate_constant_with_wrong_shape_raises(self) -> None:
        """Test validation fails for constant with non-scalar tensor."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("Constant()")

        with pytest.raises(PatternValidationError) as exc_info:
            validator.validate(
                pattern,
                predicates={"Constant": MockTensor((10, 10))},
            )

        assert "constant" in str(exc_info.value).lower()

    def test_validate_extra_dimensions_allowed(self) -> None:
        """Test validation allows extra batch dimensions."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x)")

        # Tensor with shape (batch, entities) for 1-arg predicate is OK
        validator.validate(
            pattern,
            predicates={"P": MockTensor((32, 100))},
            bindings={"x": MockTensor((32,))},
        )


class TestTypeCorrectnessValidation:
    """Test type correctness validation."""

    def test_validate_numeric_dtype(self) -> None:
        """Test validation passes for numeric dtypes."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x)")

        for dtype in ["float32", "float64", "int32", "int64", "bool", "uint8"]:
            validator.validate(
                pattern,
                predicates={"P": MockTensor((10,), dtype=dtype)},
                bindings={"x": MockTensor((5,))},
            )

    def test_validate_invalid_dtype_raises(self) -> None:
        """Test validation fails for non-numeric dtypes."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x)")

        with pytest.raises(PatternValidationError) as exc_info:
            validator.validate(
                pattern,
                predicates={"P": MockTensor((10,), dtype="object")},
                bindings={"x": MockTensor((5,))},
            )

        assert "dtype" in str(exc_info.value)

    def test_validate_non_tensor_predicate_raises(self) -> None:
        """Test validation fails for non-tensor predicates."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x)")

        with pytest.raises(PatternValidationError) as exc_info:
            validator.validate(
                pattern,
                predicates={"P": "not a tensor"},  # String instead of tensor
                bindings={"x": MockTensor((5,))},
            )

        assert "tensor" in str(exc_info.value).lower()

    def test_validate_no_shape_attribute_raises(self) -> None:
        """Test validation fails when predicate lacks .shape."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x)")

        with pytest.raises(PatternValidationError) as exc_info:
            validator.validate(
                pattern,
                predicates={"P": [1, 2, 3]},  # List has no .shape
                bindings={"x": MockTensor((5,))},
            )

        assert "shape" in str(exc_info.value).lower()


class TestValidationIntegration:
    """Integration tests for complete validation."""

    def test_validate_complex_pattern(self) -> None:
        """Test validation of complex pattern with multiple checks."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse(
            "forall x: exists y: Related(x, y) and HasProperty(y)"
        )

        validator.validate(
            pattern,
            predicates={
                "Related": MockTensor((100, 100)),
                "HasProperty": MockTensor((100,)),
            },
        )

    def test_validate_with_numpy_arrays(self) -> None:
        """Test validation with real NumPy arrays."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x) and Q(y)")

        validator.validate(
            pattern,
            predicates={
                "P": np.array([1.0, 0.0, 1.0]),
                "Q": np.array([[1.0, 0.0], [0.0, 1.0]]),
            },
            bindings={
                "x": np.array([0, 1, 2]),
                "y": np.array([0, 1]),
            },
        )

    def test_validate_none_predicates_dict(self) -> None:
        """Test validation handles None predicates dict."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("forall x: P(x)")

        with pytest.raises(PatternValidationError):
            validator.validate(pattern, predicates=None)

    def test_validate_none_bindings_dict(self) -> None:
        """Test validation handles None bindings dict."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x)")

        with pytest.raises(PatternValidationError):
            validator.validate(
                pattern,
                predicates={"P": MockTensor((10,))},
                bindings=None,
            )

    def test_validate_empty_dicts(self) -> None:
        """Test validation with empty predicates and bindings."""
        parser = PatternParser()
        validator = PatternValidator()

        # Pattern with no predicates or variables
        pattern = parser.parse("forall x: forall y: IsTrue()")

        with pytest.raises(PatternValidationError):
            # Missing IsTrue predicate
            validator.validate(pattern, predicates={}, bindings={})

    def test_validate_multiple_errors_reports_first(self) -> None:
        """Test validation reports first error encountered."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x)")

        # Both unbound variable and missing predicate
        with pytest.raises(PatternValidationError) as exc_info:
            validator.validate(pattern, predicates={}, bindings={})

        # Should report unbound variable first
        assert "Unbound" in str(exc_info.value)


class TestValidationErrorMessages:
    """Test validation error message quality."""

    def test_error_includes_pattern(self) -> None:
        """Test error messages include original pattern."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern_str = "P(x)"
        pattern = parser.parse(pattern_str)

        with pytest.raises(PatternValidationError) as exc_info:
            validator.validate(pattern, predicates={}, bindings={"x": MockTensor((5,))})

        assert pattern_str in str(exc_info.value)

    def test_error_includes_suggestion(self) -> None:
        """Test error messages include helpful suggestions."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x)")

        with pytest.raises(PatternValidationError) as exc_info:
            validator.validate(pattern, predicates={}, bindings={"x": MockTensor((5,))})

        assert "Suggestion:" in str(exc_info.value)

    def test_error_includes_context(self) -> None:
        """Test error messages include contextual information."""
        parser = PatternParser()
        validator = PatternValidator()

        pattern = parser.parse("P(x, y)")

        with pytest.raises(PatternValidationError) as exc_info:
            validator.validate(
                pattern,
                predicates={"P": MockTensor((10,))},
                bindings={"x": MockTensor((5,)), "y": MockTensor((5,))},
            )

        error_msg = str(exc_info.value)
        assert "shape" in error_msg.lower()
        assert "arguments" in error_msg.lower() or "dimensions" in error_msg.lower()
