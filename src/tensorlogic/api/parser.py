"""EBNF-based pattern parser for logical formulas.

Implements tokenization, recursive descent parsing, and AST generation
for quantified logical expressions.

Grammar (EBNF):
    formula     ::= quantifier | logical_expr
    quantifier  ::= (forall | exists) var_list scope? ":" formula
    var_list    ::= identifier ("," identifier)*
    scope       ::= "in" identifier
    logical_expr::= term (binary_op term)*
    term        ::= unary_op? (predicate | "(" formula ")")
    predicate   ::= identifier "(" arg_list ")"
    arg_list    ::= identifier ("," identifier)*
    binary_op   ::= "and" | "or" | "->"
    unary_op    ::= "not"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from tensorlogic.api.errors import PatternSyntaxError

__all__ = [
    "Token",
    "TokenType",
    "Tokenizer",
    "ASTNode",
    "Variable",
    "Predicate",
    "UnaryOp",
    "BinaryOp",
    "Quantifier",
    "ParsedPattern",
    "PatternParser",
]


class TokenType(Enum):
    """Token types for pattern language."""

    # Keywords
    FORALL = auto()
    EXISTS = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IMPLIES = auto()  # ->
    IN = auto()

    # Punctuation
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    COLON = auto()  # :
    COMMA = auto()  # ,

    # Literals
    IDENTIFIER = auto()

    # Special
    EOF = auto()


@dataclass(frozen=True)
class Token:
    """Token with type, value, and position information."""

    type: TokenType
    value: str
    position: int  # Character position in original pattern

    def __repr__(self) -> str:
        """Readable token representation for debugging."""
        return f"Token({self.type.name}, {self.value!r}, pos={self.position})"


class Tokenizer:
    """Tokenize pattern strings into tokens."""

    KEYWORDS: dict[str, TokenType] = {
        "forall": TokenType.FORALL,
        "exists": TokenType.EXISTS,
        "and": TokenType.AND,
        "or": TokenType.OR,
        "not": TokenType.NOT,
        "in": TokenType.IN,
    }

    def __init__(self, pattern: str) -> None:
        """Initialize tokenizer with pattern string.

        Args:
            pattern: Logical formula string to tokenize
        """
        self.pattern = pattern
        self.position = 0

    def tokenize(self) -> list[Token]:
        """Tokenize pattern into list of tokens.

        Returns:
            List of tokens including EOF marker

        Raises:
            PatternSyntaxError: On invalid character or malformed token
        """
        tokens: list[Token] = []

        while self.position < len(self.pattern):
            # Skip whitespace
            if self.pattern[self.position].isspace():
                self.position += 1
                continue

            # Single-character tokens
            char = self.pattern[self.position]
            if char == "(":
                tokens.append(Token(TokenType.LPAREN, char, self.position))
                self.position += 1
            elif char == ")":
                tokens.append(Token(TokenType.RPAREN, char, self.position))
                self.position += 1
            elif char == ":":
                tokens.append(Token(TokenType.COLON, char, self.position))
                self.position += 1
            elif char == ",":
                tokens.append(Token(TokenType.COMMA, char, self.position))
                self.position += 1
            # Two-character implication operator
            elif char == "-" and self._peek() == ">":
                tokens.append(Token(TokenType.IMPLIES, "->", self.position))
                self.position += 2
            # Identifiers and keywords
            elif char.isalpha() or char == "_":
                tokens.append(self._read_identifier())
            else:
                raise PatternSyntaxError(
                    f"Unexpected character '{char}'",
                    pattern=self.pattern,
                    highlight=(self.position, self.position + 1),
                    suggestion="Check for invalid characters in pattern",
                )

        # Add EOF marker
        tokens.append(Token(TokenType.EOF, "", self.position))
        return tokens

    def _peek(self) -> str | None:
        """Peek at next character without consuming it."""
        if self.position + 1 < len(self.pattern):
            return self.pattern[self.position + 1]
        return None

    def _read_identifier(self) -> Token:
        """Read identifier or keyword token."""
        start = self.position
        while (
            self.position < len(self.pattern)
            and (self.pattern[self.position].isalnum() or self.pattern[self.position] == "_")
        ):
            self.position += 1

        value = self.pattern[start : self.position]

        # Check if it's a keyword
        token_type = self.KEYWORDS.get(value, TokenType.IDENTIFIER)
        return Token(token_type, value, start)


# AST Node Classes


@dataclass
class ASTNode:
    """Base class for AST nodes."""

    pass


@dataclass
class Variable(ASTNode):
    """Variable reference in pattern."""

    name: str


@dataclass
class Predicate(ASTNode):
    """Predicate application: P(x, y, ...)"""

    name: str
    args: list[Variable]


@dataclass
class UnaryOp(ASTNode):
    """Unary operator application: not P"""

    operator: str  # "not"
    operand: ASTNode


@dataclass
class BinaryOp(ASTNode):
    """Binary operator application: P and Q, P or Q, P -> Q"""

    operator: str  # "and", "or", "->"
    left: ASTNode
    right: ASTNode


@dataclass
class Quantifier(ASTNode):
    """Quantified formula: forall x: P(x), exists y in batch: Q(y)"""

    quantifier: str  # "forall" or "exists"
    variables: list[str]
    scope: str | None  # Optional "in X" scope
    body: ASTNode


@dataclass
class ParsedPattern:
    """Parsed pattern with AST and metadata."""

    ast: ASTNode
    pattern: str
    free_variables: set[str]  # Variables not bound by quantifiers
    predicates: set[str]  # Predicate names used in pattern


class PatternParser:
    """Recursive descent parser for logical formulas."""

    def __init__(self) -> None:
        """Initialize parser."""
        self.tokens: list[Token] = []
        self.current = 0
        self.pattern = ""

    def parse(self, pattern: str) -> ParsedPattern:
        """Parse pattern string into AST.

        Args:
            pattern: Logical formula string

        Returns:
            ParsedPattern with AST and metadata

        Raises:
            PatternSyntaxError: On invalid syntax with highlighted error
        """
        self.pattern = pattern
        tokenizer = Tokenizer(pattern)
        self.tokens = tokenizer.tokenize()
        self.current = 0

        # Parse formula
        ast = self._parse_formula()

        # Ensure we consumed all tokens (except EOF)
        if not self._is_at_end():
            token = self._peek()
            raise PatternSyntaxError(
                f"Unexpected token '{token.value}'",
                pattern=self.pattern,
                highlight=(token.position, token.position + len(token.value)),
                suggestion="Expected end of pattern or operator",
            )

        # Collect metadata
        free_vars = self._collect_free_variables(ast, set())
        predicates = self._collect_predicates(ast)

        return ParsedPattern(
            ast=ast,
            pattern=pattern,
            free_variables=free_vars,
            predicates=predicates,
        )

    def _parse_formula(self) -> ASTNode:
        """Parse formula: quantifier | logical_expr"""
        # Check for quantifier keywords
        if self._match(TokenType.FORALL, TokenType.EXISTS):
            return self._parse_quantifier()
        return self._parse_logical_expr()

    def _parse_quantifier(self) -> Quantifier:
        """Parse quantifier: (forall | exists) var_list scope? ":" formula"""
        # Already consumed quantifier keyword
        quantifier_token = self._previous()
        quantifier = quantifier_token.value

        # Parse variable list
        variables = self._parse_var_list()

        # Optional scope: "in identifier"
        scope = None
        if self._match(TokenType.IN):
            if not self._check(TokenType.IDENTIFIER):
                raise PatternSyntaxError(
                    "Expected scope identifier after 'in'",
                    pattern=self.pattern,
                    highlight=(self._peek().position, self._peek().position + 1),
                    suggestion="Use format 'forall x in batch: ...'",
                )
            scope = self._advance().value

        # Expect colon
        if not self._match(TokenType.COLON):
            token = self._peek()
            raise PatternSyntaxError(
                "Expected ':' after quantifier variables",
                pattern=self.pattern,
                highlight=(token.position, token.position + 1),
                suggestion="Use format 'forall x: P(x)'",
            )

        # Parse body formula
        body = self._parse_formula()

        return Quantifier(
            quantifier=quantifier,
            variables=variables,
            scope=scope,
            body=body,
        )

    def _parse_var_list(self) -> list[str]:
        """Parse var_list: identifier ("," identifier)*"""
        variables: list[str] = []

        if not self._check(TokenType.IDENTIFIER):
            token = self._peek()
            raise PatternSyntaxError(
                "Expected variable name",
                pattern=self.pattern,
                highlight=(token.position, token.position + 1),
                suggestion="Quantifiers require at least one variable",
            )

        variables.append(self._advance().value)

        # Parse additional variables
        while self._match(TokenType.COMMA):
            if not self._check(TokenType.IDENTIFIER):
                token = self._peek()
                raise PatternSyntaxError(
                    "Expected variable name after ','",
                    pattern=self.pattern,
                    highlight=(token.position, token.position + 1),
                    suggestion="Remove trailing comma or add variable",
                )
            variables.append(self._advance().value)

        return variables

    def _parse_logical_expr(self) -> ASTNode:
        """Parse logical_expr: term (binary_op term)*"""
        left = self._parse_term()

        # Parse binary operators with left-associativity
        while self._match(TokenType.AND, TokenType.OR, TokenType.IMPLIES):
            operator_token = self._previous()
            operator = operator_token.value
            right = self._parse_term()
            left = BinaryOp(operator=operator, left=left, right=right)

        return left

    def _parse_term(self) -> ASTNode:
        """Parse term: unary_op? (predicate | "(" formula ")")"""
        # Unary operator
        if self._match(TokenType.NOT):
            operand = self._parse_term()
            return UnaryOp(operator="not", operand=operand)

        # Parenthesized formula
        if self._match(TokenType.LPAREN):
            formula = self._parse_formula()
            if not self._match(TokenType.RPAREN):
                token = self._peek()
                raise PatternSyntaxError(
                    "Expected ')' to close parenthesized expression",
                    pattern=self.pattern,
                    highlight=(token.position, token.position + 1),
                    suggestion="Check for matching parentheses",
                )
            return formula

        # Predicate
        return self._parse_predicate()

    def _parse_predicate(self) -> Predicate:
        """Parse predicate: identifier "(" arg_list ")" """
        if not self._check(TokenType.IDENTIFIER):
            token = self._peek()
            raise PatternSyntaxError(
                "Expected predicate name",
                pattern=self.pattern,
                highlight=(token.position, token.position + 1),
                suggestion="Predicates must start with an identifier",
            )

        name = self._advance().value

        # Expect opening parenthesis
        if not self._match(TokenType.LPAREN):
            token = self._peek()
            raise PatternSyntaxError(
                f"Expected '(' after predicate '{name}'",
                pattern=self.pattern,
                highlight=(token.position, token.position + 1),
                suggestion="Predicates must have argument list: P(x, y)",
            )

        # Parse argument list
        args = self._parse_arg_list()

        # Expect closing parenthesis
        if not self._match(TokenType.RPAREN):
            token = self._peek()
            raise PatternSyntaxError(
                "Expected ')' to close predicate arguments",
                pattern=self.pattern,
                highlight=(token.position, token.position + 1),
                suggestion="Check for matching parentheses in predicate",
            )

        return Predicate(name=name, args=args)

    def _parse_arg_list(self) -> list[Variable]:
        """Parse arg_list: identifier ("," identifier)*"""
        args: list[Variable] = []

        # Empty argument list is allowed
        if self._check(TokenType.RPAREN):
            return args

        if not self._check(TokenType.IDENTIFIER):
            token = self._peek()
            raise PatternSyntaxError(
                "Expected argument name",
                pattern=self.pattern,
                highlight=(token.position, token.position + 1),
                suggestion="Predicate arguments must be identifiers",
            )

        args.append(Variable(name=self._advance().value))

        # Parse additional arguments
        while self._match(TokenType.COMMA):
            if not self._check(TokenType.IDENTIFIER):
                token = self._peek()
                raise PatternSyntaxError(
                    "Expected argument name after ','",
                    pattern=self.pattern,
                    highlight=(token.position, token.position + 1),
                    suggestion="Remove trailing comma or add argument",
                )
            args.append(Variable(name=self._advance().value))

        return args

    # Helper methods for token navigation

    def _match(self, *types: TokenType) -> bool:
        """Check if current token matches any of the given types."""
        for token_type in types:
            if self._check(token_type):
                self._advance()
                return True
        return False

    def _check(self, token_type: TokenType) -> bool:
        """Check if current token is of given type."""
        if self._is_at_end():
            return False
        return self._peek().type == token_type

    def _advance(self) -> Token:
        """Consume and return current token."""
        if not self._is_at_end():
            self.current += 1
        return self._previous()

    def _is_at_end(self) -> bool:
        """Check if we've reached EOF token."""
        return self._peek().type == TokenType.EOF

    def _peek(self) -> Token:
        """Return current token without consuming it."""
        return self.tokens[self.current]

    def _previous(self) -> Token:
        """Return previous token."""
        return self.tokens[self.current - 1]

    def _collect_free_variables(
        self, node: ASTNode, bound_vars: set[str]
    ) -> set[str]:
        """Collect free variables not bound by quantifiers.

        Args:
            node: AST node to analyze
            bound_vars: Variables bound by outer quantifiers

        Returns:
            Set of free variable names
        """
        if isinstance(node, Variable):
            return {node.name} if node.name not in bound_vars else set()
        elif isinstance(node, Predicate):
            free: set[str] = set()
            for arg in node.args:
                if arg.name not in bound_vars:
                    free.add(arg.name)
            return free
        elif isinstance(node, UnaryOp):
            return self._collect_free_variables(node.operand, bound_vars)
        elif isinstance(node, BinaryOp):
            left_free = self._collect_free_variables(node.left, bound_vars)
            right_free = self._collect_free_variables(node.right, bound_vars)
            return left_free | right_free
        elif isinstance(node, Quantifier):
            # Variables in this quantifier are now bound
            new_bound = bound_vars | set(node.variables)
            return self._collect_free_variables(node.body, new_bound)
        return set()

    def _collect_predicates(self, node: ASTNode) -> set[str]:
        """Collect all predicate names used in AST.

        Args:
            node: AST node to analyze

        Returns:
            Set of predicate names
        """
        if isinstance(node, Predicate):
            return {node.name}
        elif isinstance(node, UnaryOp):
            return self._collect_predicates(node.operand)
        elif isinstance(node, BinaryOp):
            left_preds = self._collect_predicates(node.left)
            right_preds = self._collect_predicates(node.right)
            return left_preds | right_preds
        elif isinstance(node, Quantifier):
            return self._collect_predicates(node.body)
        return set()
