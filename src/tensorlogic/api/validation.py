"""Pattern validation for parsed logical formulas.

Validates patterns against provided predicates and bindings:
- Variable binding: All free variables bound in bindings
- Predicate availability: All predicates exist in predicates dict
- Shape compatibility: Predicate arities match usage
- Type correctness: Predicates are valid tensors
"""

from __future__ import annotations

from typing import Any

from tensorlogic.api.errors import PatternValidationError
from tensorlogic.api.parser import (
    BinaryOp,
    ParsedPattern,
    Predicate,
    Quantifier,
    UnaryOp,
)

__all__ = ["PatternValidator"]


class PatternValidator:
    """Validator for parsed patterns against predicates and bindings."""

    def validate(
        self,
        pattern: ParsedPattern,
        predicates: dict[str, Any] | None = None,
        bindings: dict[str, Any] | None = None,
    ) -> None:
        """Validate pattern against provided predicates and bindings.

        Args:
            pattern: Parsed pattern AST to validate
            predicates: Available predicate tensors {'P': tensor, ...}
            bindings: Variable bindings {'x': tensor, ...}

        Raises:
            PatternValidationError: On validation failures with detailed context
        """
        # Normalize inputs
        predicates = predicates or {}
        bindings = bindings or {}

        # Run validation checks
        self._validate_variable_binding(pattern, bindings)
        self._validate_predicate_availability(pattern, predicates)
        self._validate_predicate_shapes(pattern, predicates)
        self._validate_types(pattern, predicates)

    def _validate_variable_binding(
        self,
        pattern: ParsedPattern,
        bindings: dict[str, Any],
    ) -> None:
        """Check that all free variables are bound.

        Args:
            pattern: Parsed pattern with free variables
            bindings: Variable bindings dict

        Raises:
            PatternValidationError: If free variables not bound
        """
        unbound_vars = pattern.free_variables - bindings.keys()

        if unbound_vars:
            vars_str = ", ".join(sorted(unbound_vars))
            raise PatternValidationError(
                "Unbound variables in pattern",
                context=f"Variables {{{vars_str}}} used in pattern but not provided in bindings",
                pattern=pattern.pattern,
                suggestion=f"Add bindings for: {vars_str}",
            )

    def _validate_predicate_availability(
        self,
        pattern: ParsedPattern,
        predicates: dict[str, Any],
    ) -> None:
        """Check that all predicates are provided.

        Args:
            pattern: Parsed pattern with predicates
            predicates: Available predicates dict

        Raises:
            PatternValidationError: If predicates missing
        """
        missing_preds = pattern.predicates - predicates.keys()

        if missing_preds:
            preds_str = ", ".join(sorted(missing_preds))
            raise PatternValidationError(
                "Missing predicates",
                context=f"Predicates {{{preds_str}}} used in pattern but not provided",
                pattern=pattern.pattern,
                suggestion=f"Add predicates to dict: {preds_str}",
            )

    def _validate_predicate_shapes(
        self,
        pattern: ParsedPattern,
        predicates: dict[str, Any],
    ) -> None:
        """Validate predicate arities match usage in pattern.

        Args:
            pattern: Parsed pattern AST
            predicates: Available predicate tensors

        Raises:
            PatternValidationError: If arity mismatches found
        """
        # Collect predicate usage from AST
        predicate_usage = self._collect_predicate_usage(pattern.ast)

        for pred_name, expected_arity in predicate_usage.items():
            if pred_name not in predicates:
                # Already caught by availability check
                continue

            pred_tensor = predicates[pred_name]

            # Check if tensor has shape attribute
            if not hasattr(pred_tensor, "shape"):
                raise PatternValidationError(
                    f"Invalid predicate '{pred_name}'",
                    context=f"Predicate must be a tensor with .shape attribute\nGot: {type(pred_tensor).__name__}",
                    pattern=pattern.pattern,
                    suggestion="Ensure predicates are tensor objects (MLX, NumPy, etc.)",
                )

            # Get tensor shape
            shape = pred_tensor.shape

            # For 0-arity predicates (constants), expect scalar or shape ()
            if expected_arity == 0:
                if len(shape) > 0 and shape != (1,):
                    raise PatternValidationError(
                        f"Predicate '{pred_name}' arity mismatch",
                        context=f"Used as constant (0 arguments) but has shape {shape}\nExpected scalar or shape ()",
                        pattern=pattern.pattern,
                        suggestion=f"Use {pred_name}() for constants with scalar tensors",
                    )
            else:
                # For n-arity predicates, expect at least n dimensions
                # (additional dimensions are batch/data dimensions)
                if len(shape) < expected_arity:
                    raise PatternValidationError(
                        f"Predicate '{pred_name}' arity mismatch",
                        context=f"Used with {expected_arity} arguments but tensor has {len(shape)} dimensions\nTensor shape: {shape}",
                        pattern=pattern.pattern,
                        suggestion=f"Ensure {pred_name} tensor has at least {expected_arity} dimensions",
                    )

    def _validate_types(
        self,
        pattern: ParsedPattern,
        predicates: dict[str, Any],
    ) -> None:
        """Validate predicates are numeric tensors.

        Args:
            pattern: Parsed pattern AST
            predicates: Available predicate tensors

        Raises:
            PatternValidationError: If predicates have invalid types
        """
        for pred_name, pred_tensor in predicates.items():
            if pred_name not in pattern.predicates:
                # Predicate not used in pattern, skip
                continue

            # Check for tensor interface (shape and dtype)
            if not hasattr(pred_tensor, "shape"):
                raise PatternValidationError(
                    f"Invalid predicate '{pred_name}' type",
                    context=f"Predicate must be a tensor\nGot: {type(pred_tensor).__name__}",
                    pattern=pattern.pattern,
                    suggestion="Predicates must be tensor objects with .shape attribute",
                )

            # Check for dtype attribute (standard in MLX and NumPy)
            if hasattr(pred_tensor, "dtype"):
                dtype_str = str(pred_tensor.dtype)

                # Check if dtype is numeric (int or float)
                is_numeric = any(
                    t in dtype_str.lower()
                    for t in ["int", "float", "bool", "uint"]
                )

                if not is_numeric:
                    raise PatternValidationError(
                        f"Invalid predicate '{pred_name}' dtype",
                        context=f"Predicate must be numeric tensor\nGot dtype: {dtype_str}",
                        pattern=pattern.pattern,
                        suggestion="Use numeric (int/float/bool) tensors for predicates",
                    )

    def _collect_predicate_usage(
        self,
        node: Any,
    ) -> dict[str, int]:
        """Collect predicate names and their arities from AST.

        Args:
            node: AST node to analyze

        Returns:
            Dict mapping predicate names to number of arguments used
        """
        usage: dict[str, int] = {}

        if isinstance(node, Predicate):
            # Store arity (number of arguments)
            usage[node.name] = len(node.args)
        elif isinstance(node, UnaryOp):
            usage.update(self._collect_predicate_usage(node.operand))
        elif isinstance(node, BinaryOp):
            left_usage = self._collect_predicate_usage(node.left)
            right_usage = self._collect_predicate_usage(node.right)

            # Merge, checking for arity consistency
            for pred_name, arity in left_usage.items():
                if pred_name in right_usage and right_usage[pred_name] != arity:
                    # Same predicate used with different arities
                    # This is actually allowed in some cases (overloading)
                    # but for simplicity, we'll use the maximum arity
                    usage[pred_name] = max(arity, right_usage[pred_name])
                else:
                    usage[pred_name] = arity

            # Add remaining from right
            for pred_name, arity in right_usage.items():
                if pred_name not in usage:
                    usage[pred_name] = arity

        elif isinstance(node, Quantifier):
            usage.update(self._collect_predicate_usage(node.body))

        return usage
