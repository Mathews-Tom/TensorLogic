"""High-level pattern execution API for tensor logic operations.

Implements the quantify() function for executing logical formulas with
quantifiers, integrating parsing, validation, and CoreLogic execution.
"""

from __future__ import annotations

from typing import Any

from tensorlogic.api.errors import TensorLogicError
from tensorlogic.api.parser import (
    ASTNode,
    BinaryOp,
    ParsedPattern,
    PatternParser,
    Predicate,
    Quantifier,
    UnaryOp,
    Variable,
)
from tensorlogic.api.validation import PatternValidator
from tensorlogic.backends import TensorBackend, create_backend
from tensorlogic.core import (
    exists,
    forall,
    logical_and,
    logical_implies,
    logical_not,
    logical_or,
)

__all__ = ["quantify"]


class PatternExecutor:
    """Executes parsed patterns by traversing AST and applying CoreLogic operations."""

    def __init__(
        self,
        predicates: dict[str, Any],
        bindings: dict[str, Any],
        backend: TensorBackend,
    ) -> None:
        """Initialize executor with predicates, bindings, and backend.

        Args:
            predicates: Named predicate tensors
            bindings: Variable bindings
            backend: Tensor backend for operations
        """
        self.predicates = predicates
        self.bindings = bindings
        self.backend = backend
        self.quantified_vars: set[str] = set()  # Track quantified variables

    def execute(self, node: ASTNode) -> Any:
        """Execute AST node and return result tensor.

        Args:
            node: AST node to execute

        Returns:
            Result tensor from executing the node

        Raises:
            TensorLogicError: On execution errors
        """
        if isinstance(node, Variable):
            return self._execute_variable(node)
        elif isinstance(node, Predicate):
            return self._execute_predicate(node)
        elif isinstance(node, UnaryOp):
            return self._execute_unary_op(node)
        elif isinstance(node, BinaryOp):
            return self._execute_binary_op(node)
        elif isinstance(node, Quantifier):
            return self._execute_quantifier(node)
        else:
            raise TensorLogicError(
                f"Unknown AST node type: {type(node).__name__}",
                suggestion="Check pattern parsing implementation",
            )

    def _execute_variable(self, node: Variable) -> Any:
        """Execute variable node by looking up in bindings.

        Args:
            node: Variable node

        Returns:
            Bound tensor value

        Raises:
            TensorLogicError: If variable not in bindings
        """
        if node.name not in self.bindings:
            raise TensorLogicError(
                f"Variable '{node.name}' not found in bindings",
                suggestion="Ensure all free variables are bound",
            )
        return self.bindings[node.name]

    def _execute_predicate(self, node: Predicate) -> Any:
        """Execute predicate node by applying predicate to arguments.

        Args:
            node: Predicate node

        Returns:
            Result of predicate application

        Raises:
            TensorLogicError: If predicate not found or application fails
        """
        if node.name not in self.predicates:
            raise TensorLogicError(
                f"Predicate '{node.name}' not found",
                suggestion="Ensure all predicates are provided",
            )

        predicate_tensor = self.predicates[node.name]

        # If predicate has no arguments, return it directly (constant)
        if len(node.args) == 0:
            return predicate_tensor

        # Check if all arguments are either quantified or bound
        for arg in node.args:
            if arg.name not in self.bindings and arg.name not in self.quantified_vars:
                raise TensorLogicError(
                    f"Argument variable '{arg.name}' not bound",
                    suggestion="Bind all predicate arguments",
                )

        # For quantified variables, return the predicate tensor directly
        # The quantifier will handle aggregation over the appropriate axis
        # For bound variables, we would index the predicate (not implemented yet)
        return predicate_tensor

    def _execute_unary_op(self, node: UnaryOp) -> Any:
        """Execute unary operator (NOT).

        Args:
            node: UnaryOp node

        Returns:
            Result of unary operation

        Raises:
            TensorLogicError: On unsupported operator
        """
        operand_result = self.execute(node.operand)

        if node.operator == "not":
            return logical_not(operand_result, backend=self.backend)
        else:
            raise TensorLogicError(
                f"Unknown unary operator: {node.operator}",
                suggestion="Supported operators: not",
            )

    def _execute_binary_op(self, node: BinaryOp) -> Any:
        """Execute binary operator (AND, OR, IMPLIES).

        Args:
            node: BinaryOp node

        Returns:
            Result of binary operation

        Raises:
            TensorLogicError: On unsupported operator
        """
        left_result = self.execute(node.left)
        right_result = self.execute(node.right)

        if node.operator == "and":
            return logical_and(left_result, right_result, backend=self.backend)
        elif node.operator == "or":
            return logical_or(left_result, right_result, backend=self.backend)
        elif node.operator == "->":
            return logical_implies(left_result, right_result, backend=self.backend)
        else:
            raise TensorLogicError(
                f"Unknown binary operator: {node.operator}",
                suggestion="Supported operators: and, or, ->",
            )

    def _execute_quantifier(self, node: Quantifier) -> Any:
        """Execute quantifier (EXISTS, FORALL).

        Args:
            node: Quantifier node

        Returns:
            Result of quantification

        Raises:
            TensorLogicError: On unsupported quantifier
        """
        # Add quantified variables to scope
        old_quantified_vars = self.quantified_vars.copy()
        for var in node.variables:
            self.quantified_vars.add(var)

        try:
            # Execute the body with quantified variables in scope
            body_result = self.execute(node.body)

            # Determine which axis to quantify over
            # For now, quantify over axis 0 (first dimension)
            axis = 0

            if node.quantifier == "exists":
                return exists(body_result, axis=axis, backend=self.backend)
            elif node.quantifier == "forall":
                return forall(body_result, axis=axis, backend=self.backend)
            else:
                raise TensorLogicError(
                    f"Unknown quantifier: {node.quantifier}",
                    suggestion="Supported quantifiers: exists, forall",
                )
        finally:
            # Restore previous quantified variables scope
            self.quantified_vars = old_quantified_vars


def quantify(
    pattern: str,
    *,
    predicates: dict[str, Any] | None = None,
    bindings: dict[str, Any] | None = None,
    domain: dict[str, range | list[Any]] | None = None,
    backend: TensorBackend | None = None,
) -> Any:
    """Execute quantified logical pattern.

    Parses a logical formula with quantifiers and executes it using tensor operations.
    Supports existential (∃) and universal (∀) quantification, logical operators
    (and, or, not, ->), and predicates over tensors.

    Args:
        pattern: Logical formula string with quantifiers
            Examples: 'forall x: P(x)', 'exists y: P(x, y) and Q(y)'
        predicates: Named predicates as tensors {'P': tensor, ...}
            Predicates must be numeric tensors (int/float/bool)
        bindings: Variable bindings for free variables {'x': tensor, ...}
            All free variables in pattern must be bound
        domain: Quantification domains {'x': range(100), ...}
            (Not yet implemented - reserved for future use)
        backend: Tensor backend (defaults to global/MLX if not specified)
            Use create_backend("mlx") or create_backend("numpy")

    Returns:
        Result tensor after pattern evaluation
            Shape depends on quantifiers and predicates

    Raises:
        PatternSyntaxError: If pattern has invalid syntax
            Error includes character-level highlighting
        PatternValidationError: If predicates/bindings validation fails
            Checks variable binding, predicate availability, shapes, types
        TensorLogicError: On runtime execution errors

    Examples:
        >>> import numpy as np
        >>> from tensorlogic.backends import create_backend
        >>> backend = create_backend("numpy")

        >>> # Existential quantification: ∃y.Related(x,y)
        >>> result = quantify(
        ...     'exists y: Related(x, y)',
        ...     predicates={
        ...         'Related': np.array([[1, 0], [0, 1]]),  # (2, 2) relation matrix
        ...     },
        ...     bindings={'x': np.array([0, 1])},  # 2 entities
        ...     backend=backend,
        ... )
        >>> # Returns [1., 1.] - both entities have at least one relation

        >>> # Universal quantification with implication: ∀x.(P(x) → Q(x))
        >>> result = quantify(
        ...     'forall x: P(x) -> Q(x)',
        ...     predicates={
        ...         'P': np.array([1., 1., 0.]),  # 3 items
        ...         'Q': np.array([1., 1., 1.]),
        ...     },
        ...     backend=backend,
        ... )
        >>> # Returns 1.0 - implication holds for all x

        >>> # Complex pattern with multiple operators
        >>> result = quantify(
        ...     'exists x: P(x) and Q(x) and not R(x)',
        ...     predicates={
        ...         'P': np.array([1., 0., 1.]),
        ...         'Q': np.array([1., 1., 0.]),
        ...         'R': np.array([0., 0., 1.]),
        ...     },
        ...     backend=backend,
        ... )
        >>> # Returns 1.0 - exists x=0 where P∧Q∧¬R is true

    Notes:
        - Pattern language supports:
          * Quantifiers: forall, exists (with optional 'in' scope)
          * Operators: and, or, not, -> (implies)
          * Predicates: P(x, y, ...) with variable arguments
        - Quantification operates on axis 0 by default
        - Predicates must be tensors with .shape and .dtype attributes
        - All operations use the specified backend for execution
        - MLX backend uses lazy evaluation (results auto-evaluated)
    """
    # Normalize inputs
    predicates = predicates or {}
    bindings = bindings or {}
    domain = domain or {}  # Reserved for future use

    # Get or create backend
    if backend is None:
        backend = create_backend()

    # Parse pattern
    parser = PatternParser()
    parsed_pattern: ParsedPattern = parser.parse(pattern)

    # Validate pattern
    validator = PatternValidator()
    validator.validate(parsed_pattern, predicates=predicates, bindings=bindings)

    # Execute pattern
    executor = PatternExecutor(
        predicates=predicates,
        bindings=bindings,
        backend=backend,
    )
    result = executor.execute(parsed_pattern.ast)

    # For MLX backend, ensure result is evaluated
    if hasattr(backend, "eval"):
        backend.eval(result)

    return result
