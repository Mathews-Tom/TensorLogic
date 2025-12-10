# Tasks: Compilation Strategies (COMP)

**From:** `spec.md`
**Timeline:** 2-3 weeks (3 sprints)
**Team:** 1 ML Engineer
**Created:** 2025-12-09

## Summary

- **Total tasks:** 13 (1 epic + 12 stories)
- **Estimated effort:** 54 story points
- **Critical path duration:** 14 days (sequential) / 8-10 days (with parallelization)
- **Key risks:** Semantic correctness validation, gradient compatibility, performance overhead

## Phase Breakdown

### Phase 1: Foundation (Sprint 1, Days 1-2, 5 SP)

**Goal:** Establish compilation strategy abstraction layer
**Deliverable:** Protocol definition and factory pattern

#### Tasks

**[COMP-002] Define CompilationStrategy Protocol**

- **Description:** Implement Protocol interface for compilation strategies with 6 core methods (compile_and, compile_or, compile_not, compile_implies, compile_exists, compile_forall) plus is_differentiable and name properties
- **Acceptance:**
  - [ ] Protocol defined in `src/tensorlogic/compilation/protocol.py`
  - [ ] All 6 logical operations specified
  - [ ] Properties for differentiability and naming
  - [ ] Type hints using Array from backends
  - [ ] Docstrings with mathematical semantics
- **Effort:** 3 story points (2-3 days)
- **Owner:** ML Engineer
- **Dependencies:** None (requires BACKEND-001, CORE-001 complete)
- **Priority:** P1 (Blocker for all strategies)
- **Files:**
  - `src/tensorlogic/compilation/__init__.py` (create)
  - `src/tensorlogic/compilation/protocol.py` (create)

**[COMP-003] Implement Strategy Factory Pattern**

- **Description:** Create `create_strategy(name: str)` factory function with strategy registry and graceful error handling for unknown strategies
- **Acceptance:**
  - [ ] Factory function in `src/tensorlogic/compilation/factory.py`
  - [ ] Registry of available strategies
  - [ ] ValueError for unknown strategy names
  - [ ] Type-safe return (CompilationStrategy)
  - [ ] Default strategy: "soft_differentiable"
- **Effort:** 2 story points (1-2 days)
- **Owner:** ML Engineer
- **Dependencies:** COMP-002
- **Priority:** P1 (Enables strategy switching)
- **Files:**
  - `src/tensorlogic/compilation/factory.py` (create)

---

### Phase 2: Strategy Implementations (Sprint 1-2, Days 3-7, 25 SP)

**Goal:** Implement 5 compilation strategies (soft, hard, 3 fuzzy variants)
**Deliverable:** Complete strategy implementations with unit tests

#### Tasks

**[COMP-004] Implement SoftDifferentiableStrategy**

- **Description:** Soft probabilistic semantics (differentiable) - AND as product (a*b), OR as probabilistic sum (a+b-a*b), EXISTS as max, FORALL as min
- **Acceptance:**
  - [ ] Class in `src/tensorlogic/compilation/strategies/soft.py`
  - [ ] All 6 operations implemented
  - [ ] is_differentiable = True
  - [ ] Unit tests verify correctness
  - [ ] Gradient flow validated
- **Effort:** 5 story points (3-4 days)
- **Owner:** ML Engineer
- **Dependencies:** COMP-002
- **Priority:** P0 (Default strategy, blocks integration)
- **Files:**
  - `src/tensorlogic/compilation/strategies/__init__.py` (create)
  - `src/tensorlogic/compilation/strategies/soft.py` (create)
  - `tests/test_compilation/test_soft_strategy.py` (create)

**[COMP-005] Implement HardBooleanStrategy**

- **Description:** Exact boolean semantics (non-differentiable) - AND/OR/quantifiers with step function for discrete logic
- **Acceptance:**
  - [ ] Class in `src/tensorlogic/compilation/strategies/hard.py`
  - [ ] All operations use backend.step()
  - [ ] is_differentiable = False
  - [ ] Error on gradient request
  - [ ] Zero hallucinations validated
- **Effort:** 5 story points (3-4 days)
- **Owner:** ML Engineer
- **Dependencies:** COMP-002
- **Priority:** P1 (Production inference mode)
- **Files:**
  - `src/tensorlogic/compilation/strategies/hard.py` (create)
  - `tests/test_compilation/test_hard_strategy.py` (create)

**[COMP-006] Implement GodelStrategy (Fuzzy Logic)**

- **Description:** Gödel fuzzy semantics using min/max t-norms - AND as min(a,b), OR as max(a,b), differentiable via subgradients
- **Acceptance:**
  - [ ] Class in `src/tensorlogic/compilation/strategies/godel.py`
  - [ ] Min/max operations for AND/OR
  - [ ] is_differentiable = True (subgradients)
  - [ ] Fuzzy logic axioms verified
  - [ ] Unit tests pass
- **Effort:** 5 story points (3-4 days)
- **Owner:** ML Engineer
- **Dependencies:** COMP-002
- **Priority:** P1 (Fuzzy logic variant)
- **Files:**
  - `src/tensorlogic/compilation/strategies/godel.py` (create)
  - `tests/test_compilation/test_godel_strategy.py` (create)

**[COMP-007] Implement ProductStrategy (Fuzzy Logic)**

- **Description:** Product fuzzy semantics - AND as product (a*b), OR as probabilistic sum (a+b-a*b), fully differentiable
- **Acceptance:**
  - [ ] Class in `src/tensorlogic/compilation/strategies/product.py`
  - [ ] Product t-norm for AND
  - [ ] Probabilistic t-conorm for OR
  - [ ] is_differentiable = True
  - [ ] Gradient tests pass
- **Effort:** 5 story points (3-4 days)
- **Owner:** ML Engineer
- **Dependencies:** COMP-002
- **Priority:** P2 (Alternative fuzzy semantics)
- **Files:**
  - `src/tensorlogic/compilation/strategies/product.py` (create)
  - `tests/test_compilation/test_product_strategy.py` (create)

**[COMP-008] Implement LukasiewiczStrategy (Fuzzy Logic)**

- **Description:** Łukasiewicz fuzzy semantics (strict) - AND as max(0, a+b-1), OR as min(1, a+b), differentiable
- **Acceptance:**
  - [ ] Class in `src/tensorlogic/compilation/strategies/lukasiewicz.py`
  - [ ] Łukasiewicz t-norm/t-conorm
  - [ ] Boundary conditions (0, 1) handled
  - [ ] is_differentiable = True
  - [ ] Mathematical properties verified
- **Effort:** 5 story points (3-4 days)
- **Owner:** ML Engineer
- **Dependencies:** COMP-002
- **Priority:** P2 (Alternative fuzzy semantics)
- **Files:**
  - `src/tensorlogic/compilation/strategies/lukasiewicz.py` (create)
  - `tests/test_compilation/test_lukasiewicz_strategy.py` (create)

---

### Phase 3: Integration & Testing (Sprint 2-3, Days 8-10, 19 SP)

**Goal:** Integrate with PatternAPI and validate semantic correctness
**Deliverable:** Full test coverage with property-based validation

#### Tasks

**[COMP-009] Integrate with quantify() API**

- **Description:** Add `strategy` parameter to `quantify()` function supporting both string names and strategy instances, update pattern compilation to use selected strategy
- **Acceptance:**
  - [ ] `quantify()` accepts `strategy` parameter
  - [ ] String names resolve via factory
  - [ ] Direct strategy instance support
  - [ ] Default: "soft_differentiable"
  - [ ] Integration tests pass
- **Effort:** 3 story points (2 days)
- **Owner:** ML Engineer
- **Dependencies:** COMP-004 (at least one strategy), API-001
- **Priority:** P0 (User-facing API)
- **Files:**
  - `src/tensorlogic/api/quantify.py` (edit)
  - `tests/test_api/test_quantify_strategies.py` (create)

**[COMP-010] Property-Based Tests for Semantic Axioms**

- **Description:** Implement hypothesis-based property tests verifying each strategy satisfies its semantic axioms (commutativity, associativity, distributivity, De Morgan's laws)
- **Acceptance:**
  - [ ] Property tests for each strategy
  - [ ] Verify semantic axioms per strategy type
  - [ ] Boolean axioms for hard strategy
  - [ ] Fuzzy axioms for fuzzy strategies
  - [ ] 100+ test cases per property
  - [ ] All properties pass
- **Effort:** 8 story points (5 days)
- **Owner:** ML Engineer
- **Dependencies:** COMP-004 through COMP-008
- **Priority:** P0 (Correctness validation)
- **Files:**
  - `tests/test_compilation/test_properties.py` (create)
  - `tests/test_compilation/conftest.py` (strategy fixtures)

**[COMP-011] Cross-Strategy Validation Tests**

- **Description:** Comparative tests across strategies on known examples, verify consistency where semantics overlap, document divergence where expected
- **Acceptance:**
  - [ ] Test suite comparing all 5 strategies
  - [ ] Same inputs, compare outputs
  - [ ] Document expected differences
  - [ ] Validate soft ≈ product for probabilistic
  - [ ] Validate hard produces discrete results
- **Effort:** 5 story points (3 days)
- **Owner:** ML Engineer
- **Dependencies:** COMP-004 through COMP-008
- **Priority:** P1 (Cross-validation)
- **Files:**
  - `tests/test_compilation/test_cross_strategy.py` (create)

**[COMP-012] Gradient Compatibility Tests**

- **Description:** Validate differentiable strategies support backend.grad(), verify non-differentiable strategies raise clear errors, test gradient flow through compiled operations
- **Acceptance:**
  - [ ] Gradient tests for soft, godel, product, lukasiewicz
  - [ ] Error handling for hard strategy gradients
  - [ ] Integration with backend.grad()
  - [ ] Numerical gradient validation
  - [ ] All gradient tests pass
- **Effort:** 3 story points (2 days)
- **Owner:** ML Engineer
- **Dependencies:** COMP-004 through COMP-008
- **Priority:** P1 (Training compatibility)
- **Files:**
  - `tests/test_compilation/test_gradients.py` (create)

---

### Phase 4: Production Readiness (Sprint 3, Days 11-12, 5 SP)

**Goal:** Type safety, documentation, and performance validation
**Deliverable:** Production-ready compilation module

#### Tasks

**[COMP-013] Type Safety Validation**

- **Description:** Ensure 100% type hints, pass mypy --strict, fix any type errors, add py.typed marker
- **Acceptance:**
  - [ ] 100% type hint coverage
  - [ ] mypy --strict passes on compilation module
  - [ ] Protocol typing correct
  - [ ] Strategy return types verified
  - [ ] No type: ignore comments
- **Effort:** 2 story points (1-2 days)
- **Owner:** ML Engineer
- **Dependencies:** COMP-002 through COMP-012
- **Priority:** P1 (Code quality)
- **Files:**
  - All `src/tensorlogic/compilation/*.py` files
  - `src/tensorlogic/py.typed` (verify)

**[COMP-014] Documentation and Examples**

- **Description:** Document each strategy with mathematical semantics, create usage examples, document when to use each strategy, add API reference
- **Acceptance:**
  - [ ] Docstrings for all strategies
  - [ ] Mathematical descriptions
  - [ ] Usage examples in docstrings
  - [ ] Strategy selection guide
  - [ ] API reference updated
  - [ ] ≥90% documentation coverage
- **Effort:** 3 story points (2 days)
- **Owner:** ML Engineer
- **Dependencies:** COMP-002 through COMP-012
- **Priority:** P1 (Developer experience)
- **Files:**
  - All compilation module files (docstrings)
  - `docs/api/compilation.md` (create)
  - `examples/compilation_strategies.py` (create)

---

## Critical Path

```plaintext
COMP-002 → COMP-004 → COMP-009 → COMP-010 → COMP-013 → COMP-014
  (3d)       (4d)       (2d)       (5d)       (2d)       (2d)
                           [18 days total]
```

**Bottlenecks:**
- COMP-002: Protocol blocks all implementation
- COMP-010: Property-based testing is complex, highest effort

**Parallel Tracks:**
- After COMP-002: All strategies (COMP-004 to COMP-008) can run in parallel
- After strategies complete: COMP-011 and COMP-012 can overlap
- COMP-013 and COMP-014 can start once core implementation stable

**Optimized Timeline:** 8-10 days with parallelization

---

## Quick Wins (Days 1-3)

1. **COMP-002: Protocol Definition** - Unblocks all strategy work
2. **COMP-003: Factory Pattern** - Enables strategy switching demos
3. **COMP-004: Soft Strategy** - Default strategy, unblocks API integration

---

## Risk Mitigation

| Task | Risk | Mitigation | Contingency |
|------|------|------------|-------------|
| COMP-010 | Semantic axioms complex to verify | Use hypothesis property-based testing, leverage CORE-001 patterns | Start with basic properties, expand incrementally |
| COMP-005 | Hard boolean non-differentiable | Clear is_differentiable flag, explicit error messages | Document limitation, future: straight-through estimators |
| COMP-009 | API integration breaks existing code | Backward compatible: strategy is optional parameter | Default to soft_differentiable, preserve existing behavior |
| All | Performance overhead >10% | Simple dictionary lookup, no dynamic dispatch | Profile early, optimize hot paths |

---

## Testing Strategy

### Test Organization

```
tests/test_compilation/
├── __init__.py
├── conftest.py                    # Shared fixtures, strategy parametrization
├── test_soft_strategy.py          # Soft strategy unit tests
├── test_hard_strategy.py          # Hard strategy unit tests
├── test_godel_strategy.py         # Gödel fuzzy unit tests
├── test_product_strategy.py       # Product fuzzy unit tests
├── test_lukasiewicz_strategy.py   # Łukasiewicz fuzzy unit tests
├── test_properties.py             # Property-based semantic validation
├── test_cross_strategy.py         # Cross-strategy comparison tests
└── test_gradients.py              # Gradient compatibility tests
```

### Quality Gates

- **≥90% test coverage** on compilation module
- **All strategies pass property-based tests** (semantic axioms)
- **Cross-strategy validation** passes on known examples
- **Gradient tests pass** for differentiable strategies
- **mypy --strict** passes with 100% type hints
- **Performance:** Strategy dispatch adds <10% overhead vs direct CoreLogic

### Testing Approach

1. **Unit tests:** Each strategy has dedicated test file
2. **Property tests:** Hypothesis validates semantic axioms (COMP-010)
3. **Integration tests:** quantify() API with different strategies (COMP-009)
4. **Cross-validation:** Compare strategies on same inputs (COMP-011)
5. **Gradient tests:** Automatic differentiation validation (COMP-012)

---

## Sprint Planning

**2-week sprints, ~20 SP velocity (single developer)**

| Sprint | Focus | Story Points | Key Deliverables |
|--------|-------|--------------|------------------|
| Sprint 1 | Foundation + 2 strategies | 20 SP | Protocol, factory, soft & hard strategies |
| Sprint 2 | Fuzzy strategies + integration | 18 SP | 3 fuzzy variants, API integration |
| Sprint 3 | Testing + polish | 16 SP | Property tests, cross-validation, docs |

**Total:** 54 SP across 3 sprints (6 weeks calendar, ~3 weeks effort for 1 developer)

---

## Dependencies

### Technical Prerequisites
- **BACKEND-001:** Complete (requires TensorBackend abstraction)
- **CORE-001:** Complete (requires logical operations)
- **API-001:** Complete (requires quantify() API for integration)

### External Dependencies
- **hypothesis:** Property-based testing framework (already installed)
- **pytest:** Test runner (already installed)
- **mypy:** Type checker (already installed)

### Integration Points
- `src/tensorlogic/backends/protocol.py` - TensorBackend operations
- `src/tensorlogic/core/operations.py` - Logical operations
- `src/tensorlogic/api/quantify.py` - Pattern execution API

---

## Ticket Hierarchy

```
COMP-001 (Epic: Compilation Strategies)
  ├─ COMP-002 (Story: Protocol Definition)
  ├─ COMP-003 (Story: Factory Pattern)
  ├─ COMP-004 (Story: Soft Strategy)
  ├─ COMP-005 (Story: Hard Strategy)
  ├─ COMP-006 (Story: Gödel Strategy)
  ├─ COMP-007 (Story: Product Strategy)
  ├─ COMP-008 (Story: Łukasiewicz Strategy)
  ├─ COMP-009 (Story: API Integration)
  ├─ COMP-010 (Story: Property Tests)
  ├─ COMP-011 (Story: Cross-Strategy Tests)
  ├─ COMP-012 (Story: Gradient Tests)
  ├─ COMP-013 (Story: Type Safety)
  └─ COMP-014 (Story: Documentation)
```

---

## Acceptance Criteria Summary

Epic COMP-001 is complete when:
- [ ] 5 compilation strategies implemented (soft, hard, gödel, product, lukasiewicz)
- [ ] CompilationStrategy protocol defined
- [ ] Factory function for strategy creation
- [ ] Integration with quantify() API
- [ ] Property-based tests verify semantic axioms
- [ ] Cross-strategy validation passes
- [ ] Gradient compatibility validated
- [ ] 100% type hints, passes mypy strict
- [ ] ≥90% test coverage
- [ ] Documentation complete with usage examples
- [ ] Performance: <10% overhead vs direct CoreLogic operations

---

**Estimation Method:** Based on BACKEND-001 and CORE-001 patterns
**Story Point Scale:** Fibonacci (1, 2, 3, 5, 8, 13, 21)
**Confidence Level:** High (90%) - Similar to completed BACKEND and CORE components

**References:**
- Spec: `docs/specs/compilation/spec.md`
- Cool-japan benchmark: 6 compilation strategies
- TensorLogic Overview: Phase 3 (Compilation Strategies)
