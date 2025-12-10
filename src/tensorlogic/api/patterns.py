"""High-level pattern execution API for tensor logic operations.

Implements the quantify() function for executing logical formulas with
quantifiers, integrating parsing, validation, and compilation strategies.
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
from tensorlogic.compilation import CompilationStrategy, create_strategy
from tensorlogic.core import (
    exists,
    forall,
    logical_and,
    logical_implies,
    logical_not,
    logical_or,
)
from tensorlogic.core.temperature import temperature_scaled_operation

__all__ = ["quantify", "reason"]


class PatternExecutor:
    """Executes parsed patterns by traversing AST and applying compilation strategy operations."""

    def __init__(
        self,
        predicates: dict[str, Any],
        bindings: dict[str, Any],
        backend: TensorBackend,
        strategy: CompilationStrategy,
    ) -> None:
        """Initialize executor with predicates, bindings, backend, and strategy.

        Args:
            predicates: Named predicate tensors
            bindings: Variable bindings
            backend: Tensor backend for operations
            strategy: Compilation strategy for logical operations
        """
        self.predicates = predicates
        self.bindings = bindings
        self.backend = backend
        self.strategy = strategy
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
            return self.strategy.compile_not(operand_result)
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
            return self.strategy.compile_and(left_result, right_result)
        elif node.operator == "or":
            return self.strategy.compile_or(left_result, right_result)
        elif node.operator == "->":
            return self.strategy.compile_implies(left_result, right_result)
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
                return self.strategy.compile_exists(body_result, axis=axis)
            elif node.quantifier == "forall":
                return self.strategy.compile_forall(body_result, axis=axis)
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
    strategy: str | CompilationStrategy = "soft_differentiable",
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
        strategy: Compilation strategy for logical operations (default: "soft_differentiable")
            Can be a strategy name string or CompilationStrategy instance
            Available strategies: "soft_differentiable", "hard_boolean", "godel",
            "product", "lukasiewicz"
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

    # Resolve strategy
    if isinstance(strategy, str):
        # String name - resolve via factory
        strategy_instance = create_strategy(strategy, backend=backend)
    else:
        # Direct strategy instance
        strategy_instance = strategy

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
        strategy=strategy_instance,
    )
    result = executor.execute(parsed_pattern.ast)

    # For MLX backend, ensure result is evaluated
    if hasattr(backend, "eval"):
        backend.eval(result)

    return result


def reason(
    formula: str,
    *,
    predicates: dict[str, Any] | None = None,
    bindings: dict[str, Any] | None = None,
    temperature: float = 0.0,
    aggregator: str = "product",
    backend: TensorBackend | None = None,
) -> Any:
    """Execute reasoning with temperature control.

    Enables interpolation between deductive (T=0) and analogical (T>0) reasoning.
    Temperature controls the "softness" of logical operations - at T=0, operations
    are crisp boolean functions; as T increases, operations become soft probabilities.

    Temperature Modes:
        T=0.0: Purely deductive (hard boolean, no hallucinations)
            - Operations: AND, OR, NOT use step functions
            - Results are exact {0, 1} values
            - No generalization beyond training data
        T>0.0: Analogical reasoning (soft probabilities)
            - Operations interpolate between hard and soft
            - α = 1 - exp(-T) controls interpolation weight
            - Enables gradual generalization
        T→∞: Maximum entropy (uniform distribution)
            - Operations become fully soft/continuous
            - Maximum generalization capability

    Mathematical Formulation:
        result = (1-α)·step(op(...)) + α·op(...)
        where α = 1 - exp(-T)

    Args:
        formula: Logical formula string
            Examples: 'P(x) and Q(x)', 'exists y: Related(x, y)'
        predicates: Named predicates as tensors {'P': tensor, ...}
            Must be numeric tensors (int/float/bool)
        bindings: Variable bindings for free variables {'x': tensor, ...}
            All free variables in formula must be bound
        temperature: Reasoning temperature (≥0.0, default: 0.0)
            T=0.0 for deductive, T>0.0 for analogical reasoning
        aggregator: Aggregation method (default: 'product')
            - 'product': Łukasiewicz t-norm (strict, conjunctive)
            - 'sum': Probabilistic sum (permissive)
            - 'max': Maximum (disjunctive)
            - 'min': Minimum (conjunctive)
            Note: Current implementation uses 'product' semantics
        backend: Tensor backend (defaults to global/MLX if not specified)
            Use create_backend("mlx") or create_backend("numpy")

    Returns:
        Result tensor after temperature-controlled reasoning
            Shape depends on quantifiers and predicates

    Raises:
        PatternSyntaxError: If formula has invalid syntax
            Error includes character-level highlighting
        PatternValidationError: If predicates/bindings validation fails
            Checks variable binding, predicate availability, shapes, types
        TensorLogicError: On runtime execution errors
        ValueError: If temperature is negative

    Examples:
        >>> import numpy as np
        >>> from tensorlogic.backends import create_backend
        >>> backend = create_backend("numpy")

        >>> # Deductive reasoning (T=0) - hard boolean operations
        >>> result = reason(
        ...     'P(x) and Q(x)',
        ...     predicates={
        ...         'P': np.array([1.0, 0.9, 0.1]),  # 0.9 rounds to 1.0
        ...         'Q': np.array([1.0, 0.8, 0.9]),  # 0.8 rounds to 1.0
        ...     },
        ...     bindings={'x': np.array([0, 1, 2])},
        ...     temperature=0.0,  # Exact boolean: [1, 1, 0]
        ...     backend=backend,
        ... )

        >>> # Analogical reasoning (T=1.0) - soft probabilistic operations
        >>> result = reason(
        ...     'Similar(x, y) -> HasProperty(y)',
        ...     predicates={
        ...         'Similar': np.array([[0.9, 0.1], [0.1, 0.9]]),
        ...         'HasProperty': np.array([1.0, 0.7]),
        ...     },
        ...     bindings={'x': np.array([0, 1])},
        ...     temperature=1.0,  # Soft inference with generalization
        ...     backend=backend,
        ... )

        >>> # Existential quantification with temperature
        >>> result = reason(
        ...     'exists y: Related(x, y) and HasProperty(y)',
        ...     predicates={
        ...         'Related': np.array([[0.8, 0.2], [0.3, 0.9]]),
        ...         'HasProperty': np.array([1.0, 0.6]),
        ...     },
        ...     bindings={'x': np.array([0, 1])},
        ...     temperature=0.5,  # Moderate analogical reasoning
        ...     backend=backend,
        ... )

    Notes:
        - Temperature interpolation: α = 1 - exp(-T)
          * At T=0, α=0 (100% hard boolean)
          * At T=1, α≈0.632 (63% soft, 37% hard)
          * At T≥5, α≈0.993 (99% soft, 1% hard)
        - All operations (AND, OR, NOT, EXISTS, FORALL) are temperature-scaled
        - Predicates must be tensors with .shape and .dtype attributes
        - All operations use the specified backend for execution
        - MLX backend uses lazy evaluation (results auto-evaluated)
        - Aggregator parameter reserved for future implementation
    """
    # Validate temperature
    if temperature < 0.0:
        raise ValueError(f"Temperature must be non-negative, got {temperature}")

    # Validate aggregator (currently unused, but validate for API consistency)
    valid_aggregators = {"product", "sum", "max", "min"}
    if aggregator not in valid_aggregators:
        raise ValueError(
            f"Invalid aggregator '{aggregator}'. Must be one of: {', '.join(sorted(valid_aggregators))}"
        )

    # Normalize inputs
    predicates = predicates or {}
    bindings = bindings or {}

    # Get or create backend
    if backend is None:
        backend = create_backend()

    # Wrap logical operations with temperature scaling
    temp_and = temperature_scaled_operation(logical_and, temperature, backend=backend)
    temp_or = temperature_scaled_operation(logical_or, temperature, backend=backend)
    temp_not = temperature_scaled_operation(logical_not, temperature, backend=backend)
    temp_implies = temperature_scaled_operation(logical_implies, temperature, backend=backend)
    temp_exists = temperature_scaled_operation(exists, temperature, backend=backend)
    temp_forall = temperature_scaled_operation(forall, temperature, backend=backend)

    # Parse pattern
    parser = PatternParser()
    parsed_pattern: ParsedPattern = parser.parse(formula)

    # Validate pattern
    validator = PatternValidator()
    validator.validate(parsed_pattern, predicates=predicates, bindings=bindings)

    # Create temperature-controlled executor
    # We need a modified executor that uses temperature-scaled operations
    class TemperatureControlledExecutor(PatternExecutor):
        """Executor that uses temperature-scaled operations."""

        def _execute_unary_op(self, node: UnaryOp) -> Any:
            """Execute unary operator with temperature control."""
            operand_result = self.execute(node.operand)

            if node.operator == "not":
                return temp_not(operand_result, backend=self.backend)
            else:
                raise TensorLogicError(
                    f"Unknown unary operator: {node.operator}",
                    suggestion="Supported operators: not",
                )

        def _execute_binary_op(self, node: BinaryOp) -> Any:
            """Execute binary operator with temperature control."""
            left_result = self.execute(node.left)
            right_result = self.execute(node.right)

            if node.operator == "and":
                return temp_and(left_result, right_result, backend=self.backend)
            elif node.operator == "or":
                return temp_or(left_result, right_result, backend=self.backend)
            elif node.operator == "->":
                return temp_implies(left_result, right_result, backend=self.backend)
            else:
                raise TensorLogicError(
                    f"Unknown binary operator: {node.operator}",
                    suggestion="Supported operators: and, or, ->",
                )

        def _execute_quantifier(self, node: Quantifier) -> Any:
            """Execute quantifier with temperature control."""
            # Add quantified variables to scope
            old_quantified_vars = self.quantified_vars.copy()
            for var in node.variables:
                self.quantified_vars.add(var)

            try:
                # Execute the body with quantified variables in scope
                body_result = self.execute(node.body)

                # Determine which axis to quantify over
                axis = 0

                if node.quantifier == "exists":
                    return temp_exists(body_result, axis=axis, backend=self.backend)
                elif node.quantifier == "forall":
                    return temp_forall(body_result, axis=axis, backend=self.backend)
                else:
                    raise TensorLogicError(
                        f"Unknown quantifier: {node.quantifier}",
                        suggestion="Supported quantifiers: exists, forall",
                    )
            finally:
                # Restore previous quantified variables scope
                self.quantified_vars = old_quantified_vars

    # Execute pattern with temperature-controlled operations
    # Note: TemperatureControlledExecutor overrides all strategy methods,
    # so we pass a default strategy just to satisfy the signature
    default_strategy = create_strategy("soft_differentiable", backend=backend)
    executor = TemperatureControlledExecutor(
        predicates=predicates,
        bindings=bindings,
        backend=backend,
        strategy=default_strategy,
    )
    result = executor.execute(parsed_pattern.ast)

    # For MLX backend, ensure result is evaluated
    if hasattr(backend, "eval"):
        backend.eval(result)

    return result
