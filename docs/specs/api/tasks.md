# Tasks: PatternAPI Implementation

**From:** `spec.md`
**Epic:** API-001
**Timeline:** 3 sprints (5-7 days)
**Team:** 1-2 backend engineers
**Created:** 2025-12-09

## Summary
- Total tasks: 9 story tickets
- Estimated effort: 39 story points
- Critical path duration: 7 days
- Key risks: Parser complexity (EBNF grammar), error message quality

**Pattern-Based Estimation:**
- Repository confidence: 85% (Python, pytest, established patterns)
- Complexity factor: Medium
- Estimate buffer: 1.2x (stable codebase with BACKEND/CORE precedent)

## Phase Breakdown

### Phase 1: Foundation (Sprint 1, 15 SP)
**Goal:** Implement error handling and pattern parsing foundation
**Deliverable:** Working pattern parser with validation

#### Tasks

**[API-002] Implement Enhanced Error Classes**

- **Description:** Create TensorLogicError base class with TensorSensor-style formatting
- **Acceptance:**
  - [ ] `TensorLogicError` base exception class
  - [ ] `PatternSyntaxError` for parse errors
  - [ ] `PatternValidationError` for shape/type mismatches
  - [ ] Error formatting with pattern highlighting
  - [ ] Context, suggestion, and pattern display in `__str__`
  - [ ] 100% type hints with modern Python 3.12 syntax
  - [ ] Unit tests for error formatting
- **Effort:** 2 story points (1 day)
- **Owner:** Backend Engineer
- **Dependencies:** None
- **Priority:** P0 (Foundation - blocks all other work)
- **Target Files:**
  - `src/tensorlogic/api/__init__.py` (create)
  - `src/tensorlogic/api/errors.py` (create)
  - `tests/test_api/test_errors.py` (create)

**[API-003] Implement Pattern Parser**

- **Description:** Build EBNF-based pattern parser with AST generation
- **Acceptance:**
  - [ ] Tokenizer for pattern strings
  - [ ] Recursive descent parser implementing EBNF grammar
  - [ ] AST nodes: Quantifier, LogicalExpr, Predicate, etc.
  - [ ] Parse valid patterns: `'forall x: P(x)'`, `'exists y: R(x, y) and Q(y)'`
  - [ ] Raise `PatternSyntaxError` with character-level positioning
  - [ ] Handle nested quantifiers and complex formulas
  - [ ] Support all operators: and, or, not, ->
  - [ ] 100% type hints
  - [ ] Comprehensive parser tests (50+ patterns)
- **Effort:** 8 story points (3 days)
- **Owner:** Backend Engineer
- **Dependencies:** API-002 (error classes)
- **Priority:** P0 (Critical - blocks all API functions)
- **Target Files:**
  - `src/tensorlogic/api/parser.py` (create)
  - `tests/test_api/test_parser.py` (create)

**[API-004] Implement Pattern Validation**

- **Description:** Validate parsed patterns against provided predicates and bindings
- **Acceptance:**
  - [ ] `PatternValidator` class
  - [ ] Variable binding validation (all free variables bound)
  - [ ] Predicate availability checks
  - [ ] Shape compatibility validation (predicate arities)
  - [ ] Type correctness (boolean tensors)
  - [ ] Enhanced error messages with actual/expected shapes
  - [ ] Integration with TensorBackend for shape inference
  - [ ] Validation tests for common errors
- **Effort:** 5 story points (2 days)
- **Owner:** Backend Engineer
- **Dependencies:** API-003 (parser), API-002 (errors)
- **Priority:** P0 (Critical)
- **Target Files:**
  - `src/tensorlogic/api/validation.py` (create)
  - `tests/test_api/test_validation.py` (create)

---

### Phase 2: Core APIs (Sprint 2, 13 SP)
**Goal:** Implement user-facing quantify() and reason() functions
**Deliverable:** Working pattern execution APIs

#### Tasks

**[API-005] Implement quantify() Function**

- **Description:** Top-level API for executing quantified logical patterns
- **Acceptance:**
  - [ ] `quantify(pattern, predicates, bindings, domain, backend)` function
  - [ ] Pattern parsing via API-003
  - [ ] Pattern validation via API-004
  - [ ] Execute patterns via CoreLogic operations
  - [ ] Support existential and universal quantification
  - [ ] Domain specification for quantifiers
  - [ ] Backend selection (default to MLX)
  - [ ] Google-style docstring with extensive examples
  - [ ] Type hints: `dict[str, Any]` for predicates/bindings
  - [ ] Integration tests with CoreLogic
- **Effort:** 5 story points (2 days)
- **Owner:** Backend Engineer
- **Dependencies:** API-004 (validation), CORE-001 (logical operations)
- **Priority:** P0 (Core API)
- **Target Files:**
  - `src/tensorlogic/api/patterns.py` (create)
  - `tests/test_api/test_quantify.py` (create)

**[API-006] Implement reason() Function**

- **Description:** Temperature-controlled reasoning API
- **Acceptance:**
  - [ ] `reason(formula, predicates, bindings, temperature, aggregator, backend)` function
  - [ ] Temperature parameter (T=0 deductive, T>0 analogical)
  - [ ] Aggregator selection: product, sum, max, min
  - [ ] Integration with temperature-controlled operations from CORE
  - [ ] Examples for deductive (T=0) and analogical (T>0) reasoning
  - [ ] Google-style docstring with temperature semantics
  - [ ] Type hints for all parameters
  - [ ] Integration tests with CoreLogic temperature module
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** API-005 (quantify), CORE-001 (temperature control)
- **Priority:** P0 (Core API)
- **Target Files:**
  - `src/tensorlogic/api/patterns.py` (edit)
  - `tests/test_api/test_reason.py` (create)

**[API-007] Implement Pattern Compilation & Caching**

- **Description:** Cache compiled patterns for performance
- **Acceptance:**
  - [ ] `PatternCompiler` class with LRU cache
  - [ ] `compile(pattern, backend)` returns `CompiledPattern`
  - [ ] `@lru_cache(maxsize=128)` for parsed patterns
  - [ ] Lazy compilation (parse on first use)
  - [ ] Cache hit rate tracking (for monitoring)
  - [ ] Integration with quantify() and reason()
  - [ ] Performance tests: <5ms parsing overhead
  - [ ] Cache eviction policy documented
- **Effort:** 5 story points (2 days)
- **Owner:** Backend Engineer
- **Dependencies:** API-005, API-006
- **Priority:** P1 (Performance optimization)
- **Target Files:**
  - `src/tensorlogic/api/compiler.py` (create)
  - `tests/test_api/test_compiler.py` (create)

---

### Phase 3: Testing & Polish (Sprint 3, 11 SP)
**Goal:** Comprehensive testing and quality assurance
**Deliverable:** Production-ready PatternAPI with ≥90% coverage

#### Tasks

**[API-008] Parser Test Suite**

- **Description:** Comprehensive tests for pattern syntax validation
- **Acceptance:**
  - [ ] 50+ parametrized test cases (valid + invalid patterns)
  - [ ] Test all operators: and, or, not, ->
  - [ ] Test quantifiers: forall, exists, with scopes
  - [ ] Test nested quantifiers and complex formulas
  - [ ] Test error positioning (character-level)
  - [ ] Test pattern edge cases (empty, malformed, etc.)
  - [ ] Property-based tests with hypothesis (if applicable)
  - [ ] Coverage: ≥90% for parser.py
- **Effort:** 5 story points (2 days)
- **Owner:** Backend Engineer
- **Dependencies:** API-003 (parser)
- **Priority:** P0 (Quality)
- **Target Files:**
  - `tests/test_api/test_parser.py` (expand)

**[API-009] API Integration Tests**

- **Description:** End-to-end tests for quantify() and reason()
- **Acceptance:**
  - [ ] Test quantify() with existential/universal quantifiers
  - [ ] Test reason() with T=0 and T>0
  - [ ] Test pattern examples from documentation
  - [ ] Test integration with CoreLogic operations
  - [ ] Test backend switching (MLX, NumPy)
  - [ ] Test predicate composition (multi-predicate formulas)
  - [ ] Test aggregator modes (product, sum, max, min)
  - [ ] Coverage: ≥90% for patterns.py
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** API-005, API-006
- **Priority:** P0 (Quality)
- **Target Files:**
  - `tests/test_api/test_integration.py` (create)

**[API-010] Error Message Quality Tests**

- **Description:** Verify error messages contain helpful suggestions
- **Acceptance:**
  - [ ] Test error formatting (pattern highlighting)
  - [ ] Verify error messages include context
  - [ ] Verify error messages include suggestions
  - [ ] Test PatternSyntaxError scenarios
  - [ ] Test PatternValidationError scenarios
  - [ ] Test shape mismatch error messages
  - [ ] Test missing predicate error messages
  - [ ] User testing: developers can fix errors without docs
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** API-002, API-003, API-004
- **Priority:** P0 (Developer experience)
- **Target Files:**
  - `tests/test_api/test_errors.py` (expand)

---

## Critical Path

```plaintext
API-002 → API-003 → API-004 → API-005 → API-006 → API-007
  (1d)      (3d)       (2d)       (2d)       (1d)       (2d)

[Total: 11 days sequential, ~7 days with parallelization]

Testing (API-008, API-009, API-010) runs parallel to implementation
```

**Bottlenecks:**
- API-003: Parser implementation (8 SP, highest complexity)
- API-004: Validation logic (5 SP, shape checking complexity)

**Parallel Tracks:**
- Track 1: API-002 → API-003 → API-004 (Foundation)
- Track 2: API-008 (Parser tests, after API-003)
- Track 3: API-009, API-010 (Integration tests, after API-005/006)

---

## Quick Wins (Days 1-2)

1. **[API-002] Error Classes** - Foundation for all error handling
2. **[API-003] Parser (Basic)** - Start with simple patterns, expand gradually
3. **[API-008] Parser Tests** - TDD approach, write tests as parser develops

---

## Risk Mitigation

| Task | Risk | Mitigation | Contingency |
|------|------|------------|-------------|
| API-003 | Parser complexity (EBNF grammar) | Start with simple recursive descent, expand gradually | Use existing Python parser library (pyparsing) if blocked |
| API-004 | Shape validation complexity | Leverage TensorBackend shape inference | Document shape requirements clearly, fail early |
| API-008 | Test coverage gaps | Generate tests from grammar systematically | Property-based testing with hypothesis |

---

## Testing Strategy

### Automated Testing Tasks
- **[API-008]** Parser Tests (5 SP) - Sprint 3
- **[API-009]** Integration Tests (3 SP) - Sprint 3
- **[API-010]** Error Tests (3 SP) - Sprint 3

### Quality Gates
- 90% code coverage required (enforced by pytest-cov)
- All pattern examples from documentation must pass
- Error messages must contain suggestions (verified by tests)
- Type checking: mypy --strict passes

### Test Categories
1. **Unit tests:** Parser, validator, compiler
2. **Integration tests:** quantify()/reason() with CoreLogic
3. **Error tests:** Verify error messages and suggestions
4. **Performance tests:** <5ms parsing overhead
5. **User acceptance:** Developers prefer pattern API

---

## Team Allocation

**Backend Engineers (1-2)**
- Foundation: API-002, API-003, API-004
- APIs: API-005, API-006
- Optimization: API-007
- Testing: API-008, API-009, API-010

**Optional QA Support (0.5 engineer)**
- Error message user testing
- Pattern gallery expansion
- Documentation review

---

## Sprint Planning

**2-week sprints, ~20 SP velocity per engineer**

| Sprint | Focus | Story Points | Key Deliverables |
|--------|-------|--------------|------------------|
| Sprint 1 | Foundation | 15 SP | Error classes, parser, validation |
| Sprint 2 | APIs | 13 SP | quantify(), reason(), compilation |
| Sprint 3 | Testing | 11 SP | Comprehensive test suite, quality gates |

**Total:** 39 story points over 3 sprints (5-7 days with 1-2 engineers)

---

## Dependencies and Integration

### Upstream Dependencies
- **CORE-001:** Logical operations (COMPLETED)
  - Required: `logical_and`, `logical_or`, `logical_not`, `logical_implies`
  - Required: `exists`, `forall`, `compose_rules`
  - Required: Temperature-controlled operations
- **BACKEND-001:** TensorBackend protocol (COMPLETED)
  - Required: Backend-agnostic execution
  - Required: Shape inference for validation

### Downstream Consumers
- **COMP-001:** Compilation strategies (uses PatternAPI)
- **User applications:** Primary API surface

### Integration Points
- **CoreLogic:** Execute compiled patterns via logical operations
- **TensorBackend:** Backend selection and shape validation
- **Type system:** Modern Python 3.12+ hints throughout
- **Testing:** pytest + parametrized tests

---

## Ticket Import Format

```csv
ID,Title,Description,Estimate,Priority,Assignee,Dependencies,Sprint
API-002,Enhanced Error Classes,TensorLogicError with formatting,2,P0,Backend,,1
API-003,Pattern Parser,EBNF grammar parser,8,P0,Backend,API-002,1
API-004,Pattern Validation,Shape and type validation,5,P0,Backend,API-003,1
API-005,quantify() Function,Quantified pattern execution,5,P0,Backend,API-004,2
API-006,reason() Function,Temperature-controlled reasoning,3,P0,Backend,API-005,2
API-007,Pattern Compilation,LRU caching for patterns,5,P1,Backend,API-005;API-006,2
API-008,Parser Tests,50+ test cases,5,P0,Backend,API-003,3
API-009,API Integration Tests,End-to-end API tests,3,P0,Backend,API-005;API-006,3
API-010,Error Message Tests,Error quality validation,3,P0,Backend,API-002,3
```

---

## Estimation Method

**Approach:** Planning Poker with Fibonacci scale (1,2,3,5,8,13,21)

**Story Point Scale:**
- 1-2 SP: Simple classes, basic tests (e.g., API-002 error classes)
- 3-5 SP: Moderate complexity, integration (e.g., API-004 validation, API-005 quantify)
- 8+ SP: High complexity, research required (e.g., API-003 parser)

**Buffers:**
- 20% general uncertainty buffer included in estimates
- Pattern confidence: 85% → 1.2x multiplier applied
- Parser complexity: API-003 has extra SP for EBNF grammar research

---

## Appendix

### Mathematical References
- **EBNF Grammar:** Standard context-free grammar notation
- **Logical Notation:** First-order logic syntax
- **Error Handling:** TensorSensor design pattern

### Implementation Patterns
- **Parser:** `.sage/agent/examples/python/patterns/recursive-descent-parser.md` (if exists)
- **Error Formatting:** `.sage/agent/examples/python/api/einops-style-api.md`
- **Modern Type Hints:** `.sage/agent/examples/python/types/modern-type-hints.md`

### Definition of Done (Per Task)
- [ ] Code implemented with 100% type hints
- [ ] Unit tests written and passing
- [ ] Integration tests added (where applicable)
- [ ] Error messages include suggestions
- [ ] mypy --strict passes
- [ ] pytest coverage ≥90%
- [ ] Documentation updated (docstrings + examples)
- [ ] Code review approved
- [ ] Merged to feature branch

---

**Next Steps:**
1. Generate ticket files for API-002 through API-010
2. Update `.sage/tickets/index.json` with new tickets
3. Begin implementation with API-002 (error classes)
