"""PatternAPI: Einops-style pattern notation for tensor logic operations.

High-level API providing string-based pattern notation for self-documenting
logical operations over tensors.
"""

from __future__ import annotations

from tensorlogic.api.compiler import (
    CompiledPattern,
    PatternCompiler,
    get_global_compiler,
)
from tensorlogic.api.errors import (
    PatternSyntaxError,
    PatternValidationError,
    TensorLogicError,
)
from tensorlogic.api.parser import (
    ASTNode,
    BinaryOp,
    ParsedPattern,
    PatternParser,
    Predicate,
    Quantifier,
    Token,
    Tokenizer,
    TokenType,
    UnaryOp,
    Variable,
)
from tensorlogic.api.patterns import quantify, reason
from tensorlogic.api.validation import PatternValidator

__all__ = [
    # Errors
    "TensorLogicError",
    "PatternSyntaxError",
    "PatternValidationError",
    # Parser
    "PatternParser",
    "ParsedPattern",
    "Tokenizer",
    "Token",
    "TokenType",
    # Validation
    "PatternValidator",
    # Compiler
    "PatternCompiler",
    "CompiledPattern",
    "get_global_compiler",
    # AST Nodes
    "ASTNode",
    "Variable",
    "Predicate",
    "UnaryOp",
    "BinaryOp",
    "Quantifier",
    # High-level API
    "quantify",
    "reason",
]
