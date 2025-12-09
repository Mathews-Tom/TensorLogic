"""Enhanced error classes with TensorSensor-style formatting.

Provides rich error messages with pattern highlighting, context display,
and actionable suggestions for debugging TensorLogic operations.
"""

from __future__ import annotations


__all__ = [
    "TensorLogicError",
    "PatternSyntaxError",
    "PatternValidationError",
]


class TensorLogicError(Exception):
    """Base exception for TensorLogic errors with enhanced formatting.

    Provides TensorSensor-style error messages with:
    - Pattern highlighting using caret notation (^^^)
    - Contextual information about the operation
    - Actionable suggestions for fixing the error

    Args:
        message: Primary error description
        context: Additional context about where/how error occurred
        suggestion: Recommended fix or debugging step
        pattern: Pattern string containing the error
        highlight: Character positions (start, end) to highlight in pattern

    Examples:
        >>> raise TensorLogicError(
        ...     "Predicate composition failed",
        ...     context="Predicate 'P' expects dim 256, got dim 200",
        ...     suggestion="Check predicate dimensions match",
        ...     pattern="exists y: P(x, y) and Q(y)",
        ...     highlight=(14, 15)
        ... )
    """

    def __init__(
        self,
        message: str,
        *,
        context: str | None = None,
        suggestion: str | None = None,
        pattern: str | None = None,
        highlight: tuple[int, int] | None = None,
    ) -> None:
        """Initialize error with optional context and formatting."""
        self.message = message
        self.context = context
        self.suggestion = suggestion
        self.pattern = pattern
        self.highlight = highlight
        super().__init__(message)

    def __str__(self) -> str:
        """Format error with TensorSensor-style visualization.

        Returns:
            Formatted multi-line error message with pattern highlighting
        """
        parts = [f"TensorLogicError: {self.message}"]

        # Add context lines
        if self.context:
            for line in self.context.split("\n"):
                parts.append(f"  {line}")

        # Add pattern with highlighting
        if self.pattern:
            parts.append("")
            parts.append(f"  Pattern: {self.pattern}")

            # Add caret highlighting if positions provided
            if self.highlight:
                start, end = self.highlight
                # Account for "  Pattern: " prefix (11 characters)
                prefix_len = 11
                highlight_line = " " * (prefix_len + start) + "^" * (end - start)
                parts.append(highlight_line)

        # Add suggestion
        if self.suggestion:
            parts.append("")
            parts.append(f"  Suggestion: {self.suggestion}")

        return "\n".join(parts)


class PatternSyntaxError(TensorLogicError):
    """Error raised when pattern string has invalid syntax.

    Used for parse errors in logical formulas like:
    - Mismatched parentheses
    - Invalid quantifier syntax
    - Undefined variables
    - Malformed predicates

    Examples:
        >>> raise PatternSyntaxError(
        ...     "Unexpected token ')'",
        ...     pattern="exists x: P(x))",
        ...     highlight=(16, 17),
        ...     suggestion="Remove extra closing parenthesis"
        ... )
    """


class PatternValidationError(TensorLogicError):
    """Error raised when pattern is syntactically valid but semantically invalid.

    Used for shape mismatches, type errors, and logical inconsistencies:
    - Tensor shape incompatibility
    - Type mismatches in operations
    - Undefined predicates
    - Dimension misalignment

    Examples:
        >>> raise PatternValidationError(
        ...     "Predicate dimension mismatch",
        ...     context="P expects [batch, 256], got [batch, 128]",
        ...     pattern="forall x: P(x) -> Q(x)",
        ...     highlight=(10, 11),
        ...     suggestion="Ensure P and Q have matching dimensions"
        ... )
    """
