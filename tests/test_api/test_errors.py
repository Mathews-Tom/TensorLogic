"""Tests for enhanced error classes with TensorSensor-style formatting."""

from __future__ import annotations

import pytest

from tensorlogic.api import (
    PatternSyntaxError,
    PatternValidationError,
    TensorLogicError,
)


class TestTensorLogicError:
    """Test suite for TensorLogicError base class."""

    def test_basic_error_creation(self) -> None:
        """Test creating error with just a message."""
        error = TensorLogicError("Something went wrong")
        assert error.message == "Something went wrong"
        assert error.context is None
        assert error.suggestion is None
        assert error.pattern is None
        assert error.highlight is None

    def test_error_with_context(self) -> None:
        """Test error with contextual information."""
        error = TensorLogicError(
            "Operation failed",
            context="Expected tensor shape [32, 256]",
        )
        assert error.message == "Operation failed"
        assert error.context == "Expected tensor shape [32, 256]"

    def test_error_with_suggestion(self) -> None:
        """Test error with actionable suggestion."""
        error = TensorLogicError(
            "Dimension mismatch",
            suggestion="Check input dimensions match predicate expectations",
        )
        assert error.message == "Dimension mismatch"
        assert error.suggestion == "Check input dimensions match predicate expectations"

    def test_error_with_pattern(self) -> None:
        """Test error with pattern display."""
        pattern = "exists x: P(x)"
        error = TensorLogicError(
            "Pattern error",
            pattern=pattern,
        )
        assert error.pattern == pattern

    def test_error_with_highlight(self) -> None:
        """Test error with pattern highlighting positions."""
        error = TensorLogicError(
            "Parse error",
            pattern="exists x: P(x)",
            highlight=(10, 11),
        )
        assert error.highlight == (10, 11)

    def test_basic_error_formatting(self) -> None:
        """Test formatting error with only message."""
        error = TensorLogicError("Basic error")
        formatted = str(error)
        assert formatted == "TensorLogicError: Basic error"

    def test_error_formatting_with_context(self) -> None:
        """Test formatting error with context."""
        error = TensorLogicError(
            "Predicate composition failed",
            context="Predicate 'P' expects embedding dim 256\nReceived tensor with shape [batch=764, dim=200]",
        )
        formatted = str(error)
        assert "TensorLogicError: Predicate composition failed" in formatted
        assert "  Predicate 'P' expects embedding dim 256" in formatted
        assert "  Received tensor with shape [batch=764, dim=200]" in formatted

    def test_error_formatting_with_pattern_no_highlight(self) -> None:
        """Test formatting error with pattern but no highlighting."""
        error = TensorLogicError(
            "Pattern error",
            pattern="exists y: P(x, y) and Q(y)",
        )
        formatted = str(error)
        assert "TensorLogicError: Pattern error" in formatted
        assert "  Pattern: exists y: P(x, y) and Q(y)" in formatted
        # Should not have highlight line
        assert "^" not in formatted

    def test_error_formatting_with_pattern_and_highlight(self) -> None:
        """Test formatting error with pattern highlighting."""
        error = TensorLogicError(
            "Invalid predicate",
            pattern="exists y: P(x, y) and Q(y)",
            highlight=(10, 11),  # Highlight 'P'
        )
        formatted = str(error)
        lines = formatted.split("\n")

        # Find pattern line
        pattern_line_idx = next(
            i for i, line in enumerate(lines) if "Pattern:" in line
        )
        pattern_line = lines[pattern_line_idx]
        highlight_line = lines[pattern_line_idx + 1]

        assert pattern_line == "  Pattern: exists y: P(x, y) and Q(y)"
        # Highlight should be at position 11 (prefix) + 10 (start) = 21
        assert highlight_line.startswith(" " * 21)
        assert "^" in highlight_line

    def test_error_formatting_with_multi_char_highlight(self) -> None:
        """Test formatting error with multi-character highlighting."""
        error = TensorLogicError(
            "Invalid predicate",
            pattern="exists y: HasProperty(y)",
            highlight=(10, 21),  # Highlight 'HasProperty'
        )
        formatted = str(error)
        lines = formatted.split("\n")

        # Find highlight line
        pattern_line_idx = next(
            i for i, line in enumerate(lines) if "Pattern:" in line
        )
        highlight_line = lines[pattern_line_idx + 1]

        # Should have 11 carets (21 - 10)
        assert "^" * 11 in highlight_line

    def test_error_formatting_with_suggestion(self) -> None:
        """Test formatting error with suggestion."""
        error = TensorLogicError(
            "Dimension mismatch",
            suggestion="Check predicate dimensions match",
        )
        formatted = str(error)
        assert "TensorLogicError: Dimension mismatch" in formatted
        assert "  Suggestion: Check predicate dimensions match" in formatted

    def test_complete_error_formatting(self) -> None:
        """Test formatting error with all components."""
        error = TensorLogicError(
            "Predicate composition failed",
            context="Predicate 'HasProperty' expects embedding dim 256\nReceived tensor with shape [batch=764, dim=200]",
            pattern="quantify('exists y: Related(x, y) and HasProperty(y)', ...)",
            highlight=(36, 47),  # Highlight 'HasProperty'
            suggestion="Check HasProperty's input dimension matches Related's output",
        )
        formatted = str(error)

        # Verify all components present
        assert "TensorLogicError: Predicate composition failed" in formatted
        assert "Predicate 'HasProperty' expects embedding dim 256" in formatted
        assert "Received tensor with shape [batch=764, dim=200]" in formatted
        assert "Pattern: quantify('exists y: Related(x, y) and HasProperty(y)', ...)" in formatted
        assert "^" * 11 in formatted  # 11 carets for 'HasProperty'
        assert "Suggestion: Check HasProperty's input dimension matches Related's output" in formatted

    def test_error_inheritance(self) -> None:
        """Test that TensorLogicError inherits from Exception."""
        error = TensorLogicError("Test")
        assert isinstance(error, Exception)

    def test_error_can_be_raised(self) -> None:
        """Test that error can be raised and caught."""
        with pytest.raises(TensorLogicError) as exc_info:
            raise TensorLogicError("Test error")
        assert exc_info.value.message == "Test error"


class TestPatternSyntaxError:
    """Test suite for PatternSyntaxError."""

    def test_inheritance(self) -> None:
        """Test that PatternSyntaxError inherits from TensorLogicError."""
        error = PatternSyntaxError("Syntax error")
        assert isinstance(error, TensorLogicError)
        assert isinstance(error, Exception)

    def test_syntax_error_creation(self) -> None:
        """Test creating syntax error with pattern highlighting."""
        error = PatternSyntaxError(
            "Unexpected token ')'",
            pattern="exists x: P(x))",
            highlight=(15, 16),
            suggestion="Remove extra closing parenthesis",
        )
        assert error.message == "Unexpected token ')'"
        assert error.pattern == "exists x: P(x))"
        assert error.highlight == (15, 16)
        assert error.suggestion == "Remove extra closing parenthesis"

    def test_syntax_error_formatting(self) -> None:
        """Test formatting syntax error."""
        error = PatternSyntaxError(
            "Invalid quantifier syntax",
            pattern="forall: P(x)",
            highlight=(0, 6),
            suggestion="Quantifier must include variable (e.g., 'forall x:')",
        )
        formatted = str(error)
        assert "TensorLogicError: Invalid quantifier syntax" in formatted
        assert "Pattern: forall: P(x)" in formatted
        assert "^" * 6 in formatted
        assert "Suggestion: Quantifier must include variable" in formatted

    def test_syntax_error_can_be_raised(self) -> None:
        """Test that syntax error can be raised and caught."""
        with pytest.raises(PatternSyntaxError) as exc_info:
            raise PatternSyntaxError("Parse failed")
        assert exc_info.value.message == "Parse failed"

    def test_syntax_error_caught_as_base_class(self) -> None:
        """Test that PatternSyntaxError can be caught as TensorLogicError."""
        with pytest.raises(TensorLogicError):
            raise PatternSyntaxError("Test")


class TestPatternValidationError:
    """Test suite for PatternValidationError."""

    def test_inheritance(self) -> None:
        """Test that PatternValidationError inherits from TensorLogicError."""
        error = PatternValidationError("Validation error")
        assert isinstance(error, TensorLogicError)
        assert isinstance(error, Exception)

    def test_validation_error_creation(self) -> None:
        """Test creating validation error with context."""
        error = PatternValidationError(
            "Predicate dimension mismatch",
            context="P expects [batch, 256], got [batch, 128]",
            pattern="forall x: P(x) -> Q(x)",
            highlight=(10, 11),
            suggestion="Ensure P and Q have matching dimensions",
        )
        assert error.message == "Predicate dimension mismatch"
        assert "P expects [batch, 256]" in error.context
        assert error.pattern == "forall x: P(x) -> Q(x)"
        assert error.highlight == (10, 11)

    def test_validation_error_formatting(self) -> None:
        """Test formatting validation error."""
        error = PatternValidationError(
            "Shape mismatch in composition",
            context="Predicate 'Q' expects input shape [32, 256]\nReceived shape [32, 512] from predicate 'P'",
            pattern="exists x: P(x) and Q(x)",
            highlight=(19, 20),
            suggestion="Check that P and Q predicates have compatible output shapes",
        )
        formatted = str(error)
        assert "TensorLogicError: Shape mismatch in composition" in formatted
        assert "Predicate 'Q' expects input shape [32, 256]" in formatted
        assert "Received shape [32, 512] from predicate 'P'" in formatted
        assert "Pattern: exists x: P(x) and Q(x)" in formatted
        assert "Suggestion: Check that P and Q predicates have compatible output shapes" in formatted

    def test_validation_error_can_be_raised(self) -> None:
        """Test that validation error can be raised and caught."""
        with pytest.raises(PatternValidationError) as exc_info:
            raise PatternValidationError("Invalid shape")
        assert exc_info.value.message == "Invalid shape"

    def test_validation_error_caught_as_base_class(self) -> None:
        """Test that PatternValidationError can be caught as TensorLogicError."""
        with pytest.raises(TensorLogicError):
            raise PatternValidationError("Test")


class TestErrorEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_context(self) -> None:
        """Test error with empty context string is ignored."""
        error = TensorLogicError("Error", context="")
        formatted = str(error)
        # Empty context should not be displayed
        assert formatted == "TensorLogicError: Error"

    def test_empty_suggestion(self) -> None:
        """Test error with empty suggestion string is ignored."""
        error = TensorLogicError("Error", suggestion="")
        formatted = str(error)
        # Empty suggestion should not be displayed
        assert "Suggestion: " not in formatted

    def test_empty_pattern(self) -> None:
        """Test error with empty pattern string is ignored."""
        error = TensorLogicError("Error", pattern="")
        formatted = str(error)
        # Empty pattern should not be displayed
        assert "Pattern: " not in formatted

    def test_highlight_at_start(self) -> None:
        """Test highlighting at start of pattern."""
        error = TensorLogicError(
            "Error",
            pattern="exists x: P(x)",
            highlight=(0, 6),
        )
        formatted = str(error)
        lines = formatted.split("\n")
        pattern_line_idx = next(
            i for i, line in enumerate(lines) if "Pattern:" in line
        )
        highlight_line = lines[pattern_line_idx + 1]
        # Should start highlighting at position 11 (length of "  Pattern: ")
        assert highlight_line.startswith(" " * 11 + "^" * 6)

    def test_highlight_at_end(self) -> None:
        """Test highlighting at end of pattern."""
        pattern = "exists x: P(x)"
        error = TensorLogicError(
            "Error",
            pattern=pattern,
            highlight=(10, len(pattern)),
        )
        formatted = str(error)
        assert "^" * 4 in formatted  # "P(x)" is 4 chars

    def test_multiline_context(self) -> None:
        """Test error with multi-line context."""
        error = TensorLogicError(
            "Complex error",
            context="Line 1\nLine 2\nLine 3",
        )
        formatted = str(error)
        assert "  Line 1" in formatted
        assert "  Line 2" in formatted
        assert "  Line 3" in formatted

    def test_zero_length_highlight(self) -> None:
        """Test highlight with start == end (point highlight)."""
        error = TensorLogicError(
            "Error",
            pattern="exists x: P(x)",
            highlight=(10, 10),  # Zero-length highlight
        )
        formatted = str(error)
        # Should not crash, but won't show carets
        assert "Pattern: exists x: P(x)" in formatted
