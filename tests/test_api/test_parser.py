"""Comprehensive tests for pattern parser."""

from __future__ import annotations

import pytest

from tensorlogic.api import (
    BinaryOp,
    ParsedPattern,
    PatternParser,
    PatternSyntaxError,
    Predicate,
    Quantifier,
    Token,
    Tokenizer,
    TokenType,
    UnaryOp,
    Variable,
)


class TestTokenizer:
    """Test suite for pattern tokenizer."""

    def test_tokenize_simple_predicate(self) -> None:
        """Test tokenizing simple predicate pattern."""
        tokenizer = Tokenizer("P(x)")
        tokens = tokenizer.tokenize()

        assert len(tokens) == 5  # P, (, x, ), EOF
        assert tokens[0].type == TokenType.IDENTIFIER
        assert tokens[0].value == "P"
        assert tokens[1].type == TokenType.LPAREN
        assert tokens[2].type == TokenType.IDENTIFIER
        assert tokens[2].value == "x"
        assert tokens[3].type == TokenType.RPAREN
        assert tokens[4].type == TokenType.EOF

    def test_tokenize_keywords(self) -> None:
        """Test tokenizing reserved keywords."""
        tokenizer = Tokenizer("forall exists and or not in")
        tokens = tokenizer.tokenize()

        assert tokens[0].type == TokenType.FORALL
        assert tokens[1].type == TokenType.EXISTS
        assert tokens[2].type == TokenType.AND
        assert tokens[3].type == TokenType.OR
        assert tokens[4].type == TokenType.NOT
        assert tokens[5].type == TokenType.IN

    def test_tokenize_implication_operator(self) -> None:
        """Test tokenizing -> operator."""
        tokenizer = Tokenizer("P(x) -> Q(x)")
        tokens = tokenizer.tokenize()

        # Find the -> token
        implies_token = next(t for t in tokens if t.type == TokenType.IMPLIES)
        assert implies_token.value == "->"

    def test_tokenize_with_whitespace(self) -> None:
        """Test tokenizer skips whitespace correctly."""
        tokenizer = Tokenizer("  P  (  x  )  ")
        tokens = tokenizer.tokenize()

        # Should have same tokens as without whitespace
        assert len(tokens) == 5
        assert tokens[0].value == "P"

    def test_tokenize_multi_char_identifiers(self) -> None:
        """Test tokenizing multi-character identifiers."""
        tokenizer = Tokenizer("HasProperty(entity_123)")
        tokens = tokenizer.tokenize()

        assert tokens[0].value == "HasProperty"
        assert tokens[2].value == "entity_123"

    def test_tokenize_invalid_character(self) -> None:
        """Test tokenizer raises error on invalid character."""
        tokenizer = Tokenizer("P(x) @ Q(y)")

        with pytest.raises(PatternSyntaxError) as exc_info:
            tokenizer.tokenize()

        assert "Unexpected character '@'" in str(exc_info.value)
        assert exc_info.value.highlight == (5, 6)

    def test_tokenize_positions(self) -> None:
        """Test token positions are correct."""
        tokenizer = Tokenizer("P(x)")
        tokens = tokenizer.tokenize()

        assert tokens[0].position == 0  # P
        assert tokens[1].position == 1  # (
        assert tokens[2].position == 2  # x
        assert tokens[3].position == 3  # )


class TestParserSimplePatterns:
    """Test parsing simple patterns."""

    def test_parse_simple_predicate(self) -> None:
        """Test parsing single predicate."""
        parser = PatternParser()
        result = parser.parse("P(x)")

        assert isinstance(result.ast, Predicate)
        assert result.ast.name == "P"
        assert len(result.ast.args) == 1
        assert result.ast.args[0].name == "x"

    def test_parse_predicate_multiple_args(self) -> None:
        """Test parsing predicate with multiple arguments."""
        parser = PatternParser()
        result = parser.parse("Related(x, y, z)")

        assert isinstance(result.ast, Predicate)
        assert result.ast.name == "Related"
        assert len(result.ast.args) == 3
        assert result.ast.args[0].name == "x"
        assert result.ast.args[1].name == "y"
        assert result.ast.args[2].name == "z"

    def test_parse_predicate_no_args(self) -> None:
        """Test parsing predicate with no arguments."""
        parser = PatternParser()
        result = parser.parse("IsTrue()")

        assert isinstance(result.ast, Predicate)
        assert result.ast.name == "IsTrue"
        assert len(result.ast.args) == 0

    def test_parse_and_operator(self) -> None:
        """Test parsing AND operator."""
        parser = PatternParser()
        result = parser.parse("P(x) and Q(y)")

        assert isinstance(result.ast, BinaryOp)
        assert result.ast.operator == "and"
        assert isinstance(result.ast.left, Predicate)
        assert isinstance(result.ast.right, Predicate)

    def test_parse_or_operator(self) -> None:
        """Test parsing OR operator."""
        parser = PatternParser()
        result = parser.parse("P(x) or Q(y)")

        assert isinstance(result.ast, BinaryOp)
        assert result.ast.operator == "or"

    def test_parse_implies_operator(self) -> None:
        """Test parsing implication operator."""
        parser = PatternParser()
        result = parser.parse("P(x) -> Q(x)")

        assert isinstance(result.ast, BinaryOp)
        assert result.ast.operator == "->"

    def test_parse_not_operator(self) -> None:
        """Test parsing NOT operator."""
        parser = PatternParser()
        result = parser.parse("not P(x)")

        assert isinstance(result.ast, UnaryOp)
        assert result.ast.operator == "not"
        assert isinstance(result.ast.operand, Predicate)

    def test_parse_parenthesized_expr(self) -> None:
        """Test parsing parenthesized expressions."""
        parser = PatternParser()
        result = parser.parse("(P(x))")

        assert isinstance(result.ast, Predicate)

    def test_parse_complex_and_or(self) -> None:
        """Test parsing multiple AND/OR operators."""
        parser = PatternParser()
        result = parser.parse("P(x) and Q(y) or R(z)")

        # Should be left-associative: (P and Q) or R
        assert isinstance(result.ast, BinaryOp)
        assert result.ast.operator == "or"
        assert isinstance(result.ast.left, BinaryOp)
        assert result.ast.left.operator == "and"


class TestParserQuantifiers:
    """Test parsing quantifiers."""

    def test_parse_forall_simple(self) -> None:
        """Test parsing simple forall quantifier."""
        parser = PatternParser()
        result = parser.parse("forall x: P(x)")

        assert isinstance(result.ast, Quantifier)
        assert result.ast.quantifier == "forall"
        assert result.ast.variables == ["x"]
        assert result.ast.scope is None
        assert isinstance(result.ast.body, Predicate)

    def test_parse_exists_simple(self) -> None:
        """Test parsing simple exists quantifier."""
        parser = PatternParser()
        result = parser.parse("exists y: Q(y)")

        assert isinstance(result.ast, Quantifier)
        assert result.ast.quantifier == "exists"
        assert result.ast.variables == ["y"]

    def test_parse_quantifier_multiple_vars(self) -> None:
        """Test parsing quantifier with multiple variables."""
        parser = PatternParser()
        result = parser.parse("forall x, y, z: P(x, y, z)")

        assert isinstance(result.ast, Quantifier)
        assert result.ast.variables == ["x", "y", "z"]

    def test_parse_quantifier_with_scope(self) -> None:
        """Test parsing quantifier with scope."""
        parser = PatternParser()
        result = parser.parse("forall x in batch: P(x)")

        assert isinstance(result.ast, Quantifier)
        assert result.ast.scope == "batch"

    def test_parse_nested_quantifiers(self) -> None:
        """Test parsing nested quantifiers."""
        parser = PatternParser()
        result = parser.parse("forall x: exists y: Related(x, y)")

        assert isinstance(result.ast, Quantifier)
        assert result.ast.quantifier == "forall"
        assert isinstance(result.ast.body, Quantifier)
        assert result.ast.body.quantifier == "exists"

    def test_parse_quantifier_with_complex_body(self) -> None:
        """Test parsing quantifier with complex body expression."""
        parser = PatternParser()
        result = parser.parse("exists y: P(x, y) and Q(y)")

        assert isinstance(result.ast, Quantifier)
        assert isinstance(result.ast.body, BinaryOp)


class TestParserComplexPatterns:
    """Test parsing complex patterns."""

    def test_parse_nested_parentheses(self) -> None:
        """Test parsing nested parenthesized expressions."""
        parser = PatternParser()
        result = parser.parse("((P(x)))")

        assert isinstance(result.ast, Predicate)

    def test_parse_complex_boolean_expr(self) -> None:
        """Test parsing complex boolean expression."""
        parser = PatternParser()
        result = parser.parse("P(x) and (Q(y) or R(z))")

        # P and (Q or R)
        assert isinstance(result.ast, BinaryOp)
        assert result.ast.operator == "and"
        assert isinstance(result.ast.right, BinaryOp)
        assert result.ast.right.operator == "or"

    def test_parse_multiple_not_operators(self) -> None:
        """Test parsing multiple NOT operators."""
        parser = PatternParser()
        result = parser.parse("not not P(x)")

        assert isinstance(result.ast, UnaryOp)
        assert isinstance(result.ast.operand, UnaryOp)

    def test_parse_real_world_pattern_1(self) -> None:
        """Test parsing real-world pattern: property inheritance."""
        parser = PatternParser()
        result = parser.parse(
            "forall x: exists y: Related(x, y) and HasProperty(y)"
        )

        assert isinstance(result.ast, Quantifier)
        assert result.ast.quantifier == "forall"
        assert isinstance(result.ast.body, Quantifier)

    def test_parse_real_world_pattern_2(self) -> None:
        """Test parsing real-world pattern: implication."""
        parser = PatternParser()
        result = parser.parse("forall x: P(x) -> Q(x)")

        assert isinstance(result.ast, Quantifier)
        assert isinstance(result.ast.body, BinaryOp)
        assert result.ast.body.operator == "->"

    def test_parse_real_world_pattern_3(self) -> None:
        """Test parsing real-world pattern: complex composition."""
        parser = PatternParser()
        result = parser.parse(
            "exists x in batch: (P(x) or Q(x)) and not R(x)"
        )

        assert isinstance(result.ast, Quantifier)
        assert result.ast.scope == "batch"


class TestParserErrorHandling:
    """Test parser error handling and reporting."""

    def test_error_missing_colon_after_quantifier(self) -> None:
        """Test error on missing colon after quantifier."""
        parser = PatternParser()

        with pytest.raises(PatternSyntaxError) as exc_info:
            parser.parse("forall x P(x)")

        assert "Expected ':' after quantifier" in str(exc_info.value)

    def test_error_missing_variable_name(self) -> None:
        """Test error on missing variable in quantifier."""
        parser = PatternParser()

        with pytest.raises(PatternSyntaxError) as exc_info:
            parser.parse("forall : P(x)")

        assert "Expected variable name" in str(exc_info.value)

    def test_error_unclosed_parenthesis(self) -> None:
        """Test error on unclosed parenthesis."""
        parser = PatternParser()

        with pytest.raises(PatternSyntaxError) as exc_info:
            parser.parse("P(x")

        assert "Expected ')'" in str(exc_info.value)

    def test_error_extra_closing_parenthesis(self) -> None:
        """Test error on extra closing parenthesis."""
        parser = PatternParser()

        with pytest.raises(PatternSyntaxError) as exc_info:
            parser.parse("P(x))")

        assert "Unexpected token ')'" in str(exc_info.value)

    def test_error_missing_predicate_name(self) -> None:
        """Test error when predicate name missing."""
        parser = PatternParser()

        with pytest.raises(PatternSyntaxError) as exc_info:
            parser.parse("()")

        assert "Expected predicate name" in str(exc_info.value)

    def test_error_trailing_comma_in_args(self) -> None:
        """Test error on trailing comma in arguments."""
        parser = PatternParser()

        with pytest.raises(PatternSyntaxError) as exc_info:
            parser.parse("P(x, y,)")

        assert "Expected argument name after ','" in str(exc_info.value)

    def test_error_missing_argument_list(self) -> None:
        """Test error when predicate missing argument list."""
        parser = PatternParser()

        with pytest.raises(PatternSyntaxError) as exc_info:
            parser.parse("forall x: P")

        assert "Expected '(' after predicate" in str(exc_info.value)

    def test_error_invalid_scope(self) -> None:
        """Test error on invalid scope syntax."""
        parser = PatternParser()

        with pytest.raises(PatternSyntaxError) as exc_info:
            parser.parse("forall x in : P(x)")

        assert "Expected scope identifier after 'in'" in str(exc_info.value)


class TestParserMetadata:
    """Test parser metadata extraction."""

    def test_collect_predicates_simple(self) -> None:
        """Test collecting predicate names from simple pattern."""
        parser = PatternParser()
        result = parser.parse("P(x)")

        assert result.predicates == {"P"}

    def test_collect_predicates_multiple(self) -> None:
        """Test collecting multiple predicate names."""
        parser = PatternParser()
        result = parser.parse("P(x) and Q(y) or R(z)")

        assert result.predicates == {"P", "Q", "R"}

    def test_collect_predicates_nested(self) -> None:
        """Test collecting predicates from nested quantifiers."""
        parser = PatternParser()
        result = parser.parse("forall x: exists y: P(x) and Q(y)")

        assert result.predicates == {"P", "Q"}

    def test_collect_free_variables_none(self) -> None:
        """Test no free variables when all bound."""
        parser = PatternParser()
        result = parser.parse("forall x: P(x)")

        assert result.free_variables == set()

    def test_collect_free_variables_single(self) -> None:
        """Test collecting single free variable."""
        parser = PatternParser()
        result = parser.parse("P(x)")

        assert result.free_variables == {"x"}

    def test_collect_free_variables_multiple(self) -> None:
        """Test collecting multiple free variables."""
        parser = PatternParser()
        result = parser.parse("P(x, y) and Q(z)")

        assert result.free_variables == {"x", "y", "z"}

    def test_collect_free_variables_mixed(self) -> None:
        """Test collecting free variables with quantifiers."""
        parser = PatternParser()
        result = parser.parse("forall x: P(x, y)")

        # x is bound, y is free
        assert result.free_variables == {"y"}

    def test_collect_free_variables_nested_quantifiers(self) -> None:
        """Test free variables with nested quantifiers."""
        parser = PatternParser()
        result = parser.parse("forall x: exists y: P(x, y, z)")

        # x and y are bound, z is free
        assert result.free_variables == {"z"}

    def test_parsed_pattern_stores_original(self) -> None:
        """Test that parsed pattern stores original string."""
        parser = PatternParser()
        pattern = "forall x: P(x)"
        result = parser.parse(pattern)

        assert result.pattern == pattern


class TestParserEdgeCases:
    """Test parser edge cases and boundary conditions."""

    def test_parse_whitespace_variations(self) -> None:
        """Test parsing with various whitespace."""
        parser = PatternParser()
        patterns = [
            "P(x)",
            "P (x)",
            "P( x )",
            "P  (  x  )",
            "  P(x)  ",
        ]

        for pattern in patterns:
            result = parser.parse(pattern)
            assert isinstance(result.ast, Predicate)
            assert result.ast.name == "P"

    def test_parse_long_identifier_names(self) -> None:
        """Test parsing with long identifier names."""
        parser = PatternParser()
        result = parser.parse(
            "VeryLongPredicateName(very_long_variable_name_123)"
        )

        assert isinstance(result.ast, Predicate)
        assert result.ast.name == "VeryLongPredicateName"

    def test_parse_deeply_nested_quantifiers(self) -> None:
        """Test parsing deeply nested quantifiers."""
        parser = PatternParser()
        result = parser.parse(
            "forall a: exists b: forall c: exists d: P(a, b, c, d)"
        )

        # Verify nesting depth
        node = result.ast
        depth = 0
        while isinstance(node, Quantifier):
            depth += 1
            node = node.body
        assert depth == 4

    def test_parse_many_binary_operators(self) -> None:
        """Test parsing many chained binary operators."""
        parser = PatternParser()
        result = parser.parse("P(a) and Q(b) and R(c) and S(d)")

        # Should be left-associative
        node = result.ast
        count = 0
        while isinstance(node, BinaryOp):
            count += 1
            node = node.left
        assert count == 3  # Three AND operators

    def test_parse_empty_predicate_args(self) -> None:
        """Test parsing predicate with no arguments."""
        parser = PatternParser()
        result = parser.parse("Constant()")

        assert isinstance(result.ast, Predicate)
        assert len(result.ast.args) == 0

    def test_parse_single_variable_name(self) -> None:
        """Test parsing single-character variable names."""
        parser = PatternParser()
        result = parser.parse("P(a)")

        assert result.ast.args[0].name == "a"

    def test_parse_underscore_in_identifiers(self) -> None:
        """Test parsing identifiers with underscores."""
        parser = PatternParser()
        result = parser.parse("_pred_123(_var_456)")

        assert result.ast.name == "_pred_123"
        assert result.ast.args[0].name == "_var_456"


class TestParserIntegration:
    """Integration tests for complete parser functionality."""

    def test_parse_50_plus_patterns(self) -> None:
        """Test parsing 50+ diverse patterns successfully."""
        parser = PatternParser()
        patterns = [
            # Simple predicates
            "P(x)",
            "Q(y, z)",
            "R(a, b, c)",
            # Binary operators
            "P(x) and Q(x)",
            "P(x) or Q(x)",
            "P(x) -> Q(x)",
            # Unary operator
            "not P(x)",
            "not not P(x)",
            # Parentheses
            "(P(x))",
            "(P(x) and Q(x))",
            "P(x) and (Q(x) or R(x))",
            # Quantifiers
            "forall x: P(x)",
            "exists y: Q(y)",
            "forall x, y: R(x, y)",
            "exists a, b, c: S(a, b, c)",
            # Quantifiers with scope
            "forall x in batch: P(x)",
            "exists y in domain: Q(y)",
            # Nested quantifiers
            "forall x: exists y: R(x, y)",
            "exists x: forall y: R(x, y)",
            "forall x: forall y: forall z: S(x, y, z)",
            # Complex combinations
            "forall x: P(x) and Q(x)",
            "forall x: P(x) -> Q(x)",
            "forall x: P(x) or not Q(x)",
            "exists x: P(x) and Q(x) and R(x)",
            "forall x: (P(x) and Q(x)) -> R(x)",
            "exists x: (P(x) or Q(x)) and not R(x)",
            # Real-world patterns
            "forall x: exists y: Related(x, y)",
            "forall x: HasProperty(x) -> IsValid(x)",
            "exists x: Predecessor(x, y) and Valid(x)",
            "forall x in entities: Active(x) -> Processed(x)",
            # Multiple predicates
            "P(x) and Q(y) and R(z)",
            "P(x) or Q(y) or R(z)",
            "P(x) -> Q(x) -> R(x)",
            # Deeply nested
            "((P(x)))",
            "(((P(x) and Q(x))))",
            "not (P(x) and Q(x))",
            "not (P(x) or Q(x))",
            # Complex logical expressions
            "P(x) and Q(x) or R(x) and S(x)",
            "(P(x) and Q(x)) or (R(x) and S(x))",
            "P(x) and (Q(x) or R(x)) and S(x)",
            # Quantifiers with complex bodies
            "forall x: exists y: P(x, y) and Q(y) and R(x)",
            "exists x: forall y: (P(x, y) -> Q(y)) and R(x)",
            "forall x, y: exists z: P(x, z) and Q(y, z)",
            # Edge cases
            "EmptyPredicate()",
            "_underscore(_var)",
            "CamelCase(snake_case)",
            # Realistic AI patterns
            "forall entity: HasType(entity, type) -> InDomain(entity)",
            "exists support: Evidence(claim, support) and Verified(support)",
            "forall x in batch: Embedding(x) and Normalized(x)",
            "forall query: exists doc: Relevant(query, doc) -> Retrieved(query, doc)",
        ]

        for pattern in patterns:
            result = parser.parse(pattern)
            assert isinstance(result, ParsedPattern)
            assert result.pattern == pattern
            assert result.ast is not None
