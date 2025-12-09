# PatternAPI Specification

**Component ID:** API
**Priority:** P0 (Critical - User-Facing Interface)
**Phase:** 2 (Pattern Language)
**Source:** docs/TensorLogic-Overview.md, einops library design philosophy

## 1. Overview

### Purpose and Business Value
Provide einops-style string-based pattern notation for logical operations, making neural-symbolic reasoning self-documenting and accessible. Transforms complex tensor manipulations into readable declarative patterns like `'forall x in batch: P(x) -> Q(x)'`.

**Key Differentiator:** First neural-symbolic framework with einops-inspired API design, prioritizing readability and developer experience over terse tensor notation.

### Success Metrics
- Pattern strings are immediately readable without documentation
- Runtime pattern validation catches 95%+ shape mismatches before execution
- Error messages include context, suggestions, and pattern highlighting
- API works identically across all backends (MLX, NumPy, future PyTorch)
- User survey: ≥90% prefer pattern API over direct tensor operations

### Target Users
- ML researchers implementing neural-symbolic models
- Domain experts encoding logical knowledge without deep tensor expertise
- Library developers building on TensorLogic for specific applications

## 2. Functional Requirements

### FR-1: Pattern Language Syntax
The system **shall** define a string-based pattern language for logical formulas:

**User Story:** As an AI researcher, I want to express logical rules as readable strings so that my code self-documents the reasoning structure.

#### Syntax Specification

**Quantifiers:**
```
'forall x: P(x)'              # Universal quantification
'forall x in batch: P(x)'     # Quantify over specific dimension
'exists y: R(x, y)'           # Existential quantification
'exists x, y: Related(x, y)'  # Multiple quantified variables
```

**Logical Operators:**
```
'P(x) and Q(x)'               # Conjunction
'P(x) or Q(x)'                # Disjunction
'not P(x)'                    # Negation
'P(x) -> Q(x)'                # Implication
'P(x) <-> Q(x)'               # Bi-implication (future)
```

**Composition:**
```
'forall x: P(x) -> Q(x)'                      # Implication with quantifier
'exists y: Related(x, y) and HasProperty(y)'  # Conjunction in scope
'forall x: (P(x) or Q(x)) -> R(x)'           # Complex formula
```

**Reserved Keywords:**
- Quantifiers: `forall`, `exists`
- Operators: `and`, `or`, `not`, `->` (implies)
- Scope: `in`, `:`, `(`, `)`

#### Grammar (EBNF-style)
```
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
```

### FR-2: Pattern Parsing & Validation
The system **shall** provide pattern parser with comprehensive validation:

```python
class PatternParser:
    """Parse and validate logical pattern strings."""

    def parse(self, pattern: str) -> ParsedPattern:
        """Parse pattern string into AST.

        Args:
            pattern: Logical formula string

        Returns:
            ParsedPattern with AST, variable bindings, predicates

        Raises:
            PatternSyntaxError: On invalid syntax with highlighted error
        """

    def validate(
        self,
        pattern: ParsedPattern,
        predicates: dict[str, Any],
        bindings: dict[str, Any],
    ) -> None:
        """Validate pattern against provided predicates.

        Args:
            pattern: Parsed pattern AST
            predicates: Available predicate tensors
            bindings: Variable bindings

        Raises:
            PatternValidationError: On shape/type mismatches
        """
```

**Validation Checks:**
1. **Syntax validation:** Pattern conforms to grammar
2. **Variable binding:** All free variables bound in bindings
3. **Predicate availability:** All predicates provided in predicates dict
4. **Shape compatibility:** Predicate arities match pattern usage
5. **Type correctness:** Predicates are boolean tensors

### FR-3: Quantify API
The system **shall** provide top-level `quantify()` function:

```python
def quantify(
    pattern: str,
    *,
    predicates: dict[str, Any] | None = None,
    bindings: dict[str, Any] | None = None,
    domain: dict[str, range | list] | None = None,
    backend: TensorBackend | None = None,
) -> Array:
    """Execute quantified logical pattern.

    Args:
        pattern: Logical formula with quantifiers
        predicates: Named predicates as tensors {'P': tensor, ...}
        bindings: Variable bindings {'x': tensor, ...}
        domain: Quantification domains {'x': range(100), ...}
        backend: Tensor backend (defaults to global/MLX)

    Returns:
        Result tensor after pattern evaluation

    Raises:
        PatternSyntaxError: Invalid pattern syntax
        PatternValidationError: Shape/type mismatches
        TensorLogicError: Runtime execution errors

    Examples:
        >>> # Existential quantification
        >>> result = quantify(
        ...     'exists y: Related(x, y) and HasProperty(y)',
        ...     predicates={
        ...         'Related': related_tensor,     # shape: (batch, entities)
        ...         'HasProperty': property_tensor, # shape: (entities,)
        ...     },
        ...     bindings={'x': entity_batch},
        ... )

        >>> # Universal quantification with implication
        >>> result = quantify(
        ...     'forall x in batch: P(x) -> Q(x)',
        ...     predicates={'P': predicate_p, 'Q': predicate_q},
        ... )
    """
```

### FR-4: Reason API
The system **shall** provide temperature-controlled `reason()` function:

```python
def reason(
    formula: str,
    *,
    predicates: dict[str, Any],
    bindings: dict[str, Any] | None = None,
    temperature: float = 0.0,
    aggregator: str = "product",
    backend: TensorBackend | None = None,
) -> Array:
    """Execute reasoning with temperature control.

    Temperature Modes:
        T=0.0: Purely deductive (hard boolean, no hallucinations)
        T>0.0: Analogical reasoning (soft probabilities)

    Args:
        formula: Logical formula string
        predicates: Named predicates
        bindings: Variable bindings
        temperature: Reasoning temperature (≥0.0)
        aggregator: Aggregation method ('product', 'sum', 'max', 'min')
        backend: Tensor backend

    Returns:
        Reasoning result tensor

    Examples:
        >>> # Deductive reasoning (T=0)
        >>> result = reason(
        ...     'P(x) and Q(x)',
        ...     predicates={'P': pred_p, 'Q': pred_q},
        ...     temperature=0.0,  # Exact boolean AND
        ... )

        >>> # Analogical reasoning (T=1.0)
        >>> result = reason(
        ...     'Similar(x, y) -> HasProperty(y)',
        ...     predicates={'Similar': similarity, 'HasProperty': properties},
        ...     temperature=1.0,  # Soft probabilistic inference
        ... )
    """
```

**Aggregator Semantics:**
- `"product"`: Łukasiewicz t-norm (default, strict)
- `"sum"`: Probabilistic sum (more permissive)
- `"max"`: Maximum (disjunctive)
- `"min"`: Minimum (conjunctive)

### FR-5: Enhanced Error Messages
The system **shall** provide developer-friendly error messages:

```python
class TensorLogicError(Exception):
    """Base exception with enhanced error reporting."""

    def __init__(
        self,
        message: str,
        *,
        context: str | None = None,
        suggestion: str | None = None,
        pattern: str | None = None,
        highlight: tuple[int, int] | None = None,
    ) -> None:
        """Initialize error with context.

        Args:
            message: Error description
            context: Code context where error occurred
            suggestion: Suggested fix
            pattern: Pattern string with error
            highlight: (start, end) character positions to highlight
        """

    def __str__(self) -> str:
        """Format error with context, pattern highlighting, suggestion."""
```

**Error Message Format:**
```
TensorLogicError: Predicate composition failed
  Predicate 'HasProperty' expects embedding dim 256
  Received tensor with shape [batch=764, dim=200]

  Pattern: quantify('exists y: Related(x, y) and HasProperty(y)', ...)
                                                  ^^^^^^^^^^^

  Suggestion: Check HasProperty's input dimension matches Related's output
```

### FR-6: Pattern Compilation & Caching
The system **shall** cache compiled patterns for performance:

```python
class PatternCompiler:
    """Compile patterns to optimized execution plans."""

    def __init__(self, cache_size: int = 128):
        """Initialize compiler with LRU cache."""

    def compile(
        self,
        pattern: str,
        backend: TensorBackend,
    ) -> CompiledPattern:
        """Compile pattern to execution plan.

        Args:
            pattern: Pattern string
            backend: Target backend

        Returns:
            Compiled pattern with optimized operation sequence
        """

    @lru_cache(maxsize=128)
    def _cached_parse(self, pattern: str) -> ParsedPattern:
        """Cache parsed patterns (parsing is expensive)."""
```

## 3. Non-Functional Requirements

### NFR-1: Readability
- **Self-documenting:** Pattern strings readable without comments
- **Familiar syntax:** Close to standard logical notation
- **IDE support:** Type stubs enable autocomplete for predicates dict

### NFR-2: Performance
- **Pattern caching:** Compiled patterns cached (LRU)
- **Lazy compilation:** Parse/validate only on first use
- **Backend optimization:** Leverage backend JIT compilation (MLX `mx.compile`)
- **Target:** <5ms overhead for pattern parsing (vs direct operation calls)

### NFR-3: Error Quality
- **Parse errors:** Character-level error positioning
- **Validation errors:** Include actual vs expected shapes
- **Runtime errors:** Full stack trace with pattern context
- **Suggestions:** Actionable fixes in 80%+ of errors

### NFR-4: Portability
- **Backend agnostic:** Same patterns work on MLX, NumPy, future PyTorch
- **No backend-specific syntax:** Pure logical notation
- **Version stability:** Pattern syntax backward compatible

## 4. Features & Flows

### Feature 1: Pattern Execution (Priority: P0)
**Flow:**
1. User calls `quantify(pattern, predicates={...}, bindings={...})`
2. System parses pattern string to AST
3. System validates predicates and bindings
4. System compiles pattern to operation sequence
5. System executes operations via CoreLogic
6. System caches compiled pattern for reuse
7. System returns result tensor

**Input:** Pattern string, predicates dict, bindings dict
**Output:** Result tensor

### Feature 2: Error Reporting (Priority: P0)
**Flow:**
1. Parse error detected (syntax) OR validation error (shapes)
2. System identifies error location in pattern string
3. System constructs enhanced error with:
   - Error message
   - Pattern with highlighted error region
   - Context (predicate shapes, variable types)
   - Suggestion based on error type
4. System raises TensorLogicError
5. User sees formatted error in traceback

### Feature 3: Temperature-Controlled Reasoning (Priority: P1)
**Flow:**
1. User calls `reason(formula, temperature=T, ...)`
2. System selects hard (T=0) or soft (T>0) operations
3. System compiles pattern with temperature-scaled ops
4. System executes reasoning
5. System returns results with certainty level

## 5. Code Pattern Requirements

### Naming Conventions
- **Public APIs:** snake_case (`quantify`, `reason`)
- **Classes:** PascalCase (`PatternParser`, `CompiledPattern`)
- **Internal:** Leading underscore (`_cached_parse`, `_validate_shapes`)

### Type Safety Requirements
- **100% type hints** on public APIs
- **Predicate dict:** `dict[str, Any]` (tensors from any backend)
- **Pattern strings:** Validated at runtime (no static typing for DSL)
- **Return types:** Backend-specific `Array` type (use `Any`)

### Testing Approach
- **Framework:** pytest with parametrized tests
- **Pattern test cases:** Valid syntax, invalid syntax, edge cases
- **Error testing:** Verify error messages and suggestions
- **Integration:** Test against CoreLogic operations
- **Coverage:** ≥90% line coverage

**Test Examples:**
```python
@pytest.mark.parametrize(
    "pattern,expected_valid",
    [
        ("forall x: P(x)", True),
        ("exists y: R(x, y) and Q(y)", True),
        ("forall x: P(x -> Q(x)", False),  # Missing )
        ("exists x: x", False),  # x not a predicate
    ],
)
def test_pattern_syntax(pattern, expected_valid):
    if expected_valid:
        parser.parse(pattern)  # Should not raise
    else:
        with pytest.raises(PatternSyntaxError):
            parser.parse(pattern)
```

### Error Handling
- **PatternSyntaxError:** Invalid pattern syntax
- **PatternValidationError:** Shape/type mismatches
- **TensorLogicError:** Runtime execution errors
- All inherit from `TensorLogicError` base class

### Documentation Standards
- **Docstrings:** Google-style with extensive Examples section
- **Pattern gallery:** Example patterns in docs for common use cases
- **Error catalog:** Documentation of all error types with fixes

## 6. Acceptance Criteria

### Definition of Done
- [ ] Pattern parser with EBNF grammar implemented
- [ ] `quantify()` API with predicate/binding/domain support
- [ ] `reason()` API with temperature control
- [ ] Enhanced error messages with pattern highlighting
- [ ] Pattern compilation and caching (LRU)
- [ ] Comprehensive test suite (valid/invalid patterns)
- [ ] Error message test suite (verify suggestions)
- [ ] 100% type hints, passes mypy strict
- [ ] ≥90% test coverage
- [ ] Documentation with pattern gallery

### Validation Approach
1. **Syntax tests:** Parse 50+ valid and invalid patterns
2. **Integration tests:** Execute patterns via CoreLogic
3. **Error tests:** Verify error messages contain suggestions
4. **Performance tests:** <5ms parsing overhead
5. **User testing:** Developers prefer pattern API (survey)

## 7. Dependencies

### Technical Assumptions
- **Python:** >=3.12
- **CoreLogic:** Logical operations implemented
- **TensorBackend:** Backend abstraction available

### External Integrations
- **CoreLogic:** Execute compiled patterns via logical operations
- **TensorBackend:** Backend-agnostic execution
- **No external parsers:** Implement custom parser (no PLY/ANTLR dependency)

### Related Components
- **Upstream:** CoreLogic (provides logical operations)
- **Upstream:** TensorBackend (execution layer)
- **Downstream:** Compilation (uses PatternAPI for semantic strategies)
- **Downstream:** User applications

### Future Enhancements
- **Pattern macros:** Define reusable pattern templates
- **Type inference:** Infer predicate shapes from usage
- **Optimization:** Pattern rewriting for common idioms
- **DSL extensions:** User-defined operators

---

**References:**
- einops Design Philosophy: Readable tensor operations
- TensorLogic Overview: `docs/TensorLogic-Overview.md` (lines 86-109)
- Error Handling Pattern: `.sage/agent/examples/python/api/einops-style-api.md`
- Strategic Intel: `docs/research/intel.md` (developer experience as competitive advantage)
