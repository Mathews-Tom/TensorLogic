# Tasks: CoreLogic Implementation

**From:** `spec.md`
**Epic:** CORE-001
**Timeline:** 2 sprints (5-7 days)
**Team:** 1-2 backend engineers
**Created:** 2025-12-09

## Summary
- Total tasks: 9 story tickets
- Estimated effort: 41 story points
- Critical path duration: 7 days
- Key risks: Temperature scaling complexity, property-based test coverage

**Pattern-Based Estimation:**
- Repository confidence: 85% (Python, pytest)
- Complexity factor: Medium
- Estimate buffer: 1.2x (stable patterns)

## Phase Breakdown

### Phase 1: Core Operations (Sprint 1, 18 SP)
**Goal:** Implement fundamental tensor-to-logic primitives
**Deliverable:** Working logical operations (AND, OR, NOT, IMPLIES) with property tests

#### Tasks

**[CORE-002] Implement Basic Logical Operations**

- **Description:** Implement AND, OR, NOT operations as tensor primitives using backend abstraction
- **Acceptance:**
  - [ ] `logical_and(a, b)` via Hadamard product
  - [ ] `logical_or(a, b)` via element-wise maximum
  - [ ] `logical_not(a)` via complement (1-a)
  - [ ] All operations use TensorBackend protocol
  - [ ] 100% type hints with modern syntax
  - [ ] Unit tests for basic functionality
- **Effort:** 5 story points (2 days)
- **Owner:** Backend Engineer
- **Dependencies:** BACKEND-001 (completed)
- **Priority:** P0 (Critical - Foundation)
- **Target Files:**
  - `src/tensorlogic/core/__init__.py` (create)
  - `src/tensorlogic/core/operations.py` (create)
  - `tests/test_core/test_operations.py` (create)

**[CORE-003] Implement Logical Implication**

- **Description:** Implement IMPLIES operation via max(1-a, b) formula
- **Acceptance:**
  - [ ] `logical_implies(a, b)` correct for all truth table combinations
  - [ ] Mathematical equivalence: a → b = ¬a ∨ b verified
  - [ ] Shape validation and error handling
  - [ ] Type-safe implementation
- **Effort:** 3 story points (1 day)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-002
- **Priority:** P0 (Critical)
- **Target Files:**
  - `src/tensorlogic/core/operations.py` (edit)
  - `tests/test_core/test_operations.py` (edit)

**[CORE-004] Property-Based Tests for Operations**

- **Description:** Implement hypothesis-based property tests for logical axioms
- **Acceptance:**
  - [ ] Associativity: (a ∧ b) ∧ c = a ∧ (b ∧ c)
  - [ ] Commutativity: a ∧ b = b ∧ a, a ∨ b = b ∨ a
  - [ ] Distributivity: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)
  - [ ] De Morgan's laws: ¬(a ∧ b) = ¬a ∨ ¬b
  - [ ] Double negation: ¬¬a = a
  - [ ] Tests use hypothesis with boolean strategies
  - [ ] Coverage ≥90% for operations.py
- **Effort:** 5 story points (2 days)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-003
- **Priority:** P0 (Critical - Correctness)
- **Target Files:**
  - `tests/test_core/test_properties.py` (create)

**[CORE-005] Implement Step Function**

- **Description:** Implement Heaviside step function for boolean conversion
- **Acceptance:**
  - [ ] `step(x)` returns 1.0 for x > 0, else 0.0
  - [ ] Handles edge cases (x=0, NaN, inf)
  - [ ] Backend-agnostic via TensorBackend
  - [ ] Numerical stability verified
- **Effort:** 2 story points (1 day)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-002
- **Priority:** P0 (Required for quantifiers)
- **Target Files:**
  - `src/tensorlogic/core/operations.py` (edit)
  - `tests/test_core/test_operations.py` (edit)

**[CORE-006] Cross-Backend Validation for Operations**

- **Description:** Validate MLX operations match NumPy reference implementation
- **Acceptance:**
  - [ ] All operations tested on both MLX and NumPy backends
  - [ ] Results match within FP32 tolerance (1e-6)
  - [ ] Parametrized tests across backends
  - [ ] Performance comparison documented
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-004, CORE-005
- **Priority:** P0 (Validation)
- **Target Files:**
  - `tests/test_core/test_cross_backend.py` (create)

---

### Phase 2: Quantifiers and Composition (Sprint 1-2, 15 SP)
**Goal:** Implement quantification and rule composition
**Deliverable:** Working EXISTS/FORALL with composition utilities

#### Tasks

**[CORE-007] Implement Quantifier Operations**

- **Description:** Implement existential (∃) and universal (∀) quantification
- **Acceptance:**
  - [ ] `exists(predicate, axis)` via summation + step
  - [ ] `forall(predicate, axis)` via product + step
  - [ ] Soft variants: max (exists), min (forall) for differentiability
  - [ ] Multi-axis quantification support
  - [ ] Shape inference correct
  - [ ] Mathematical soundness verified
- **Effort:** 5 story points (2 days)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-005 (step function required)
- **Priority:** P0 (Critical)
- **Target Files:**
  - `src/tensorlogic/core/quantifiers.py` (create)
  - `tests/test_core/test_quantifiers.py` (create)

**[CORE-008] Rule Composition Utilities**

- **Description:** Implement utilities for composing logical rules
- **Acceptance:**
  - [ ] `compose_rules(*rules, operation='and')` function
  - [ ] Support 'and', 'or' composition modes
  - [ ] Handle variable number of rule inputs
  - [ ] Einstein summation for multi-predicate rules
  - [ ] Example: Aunt(x,z) ← Sister(x,y) ∧ Parent(y,z)
- **Effort:** 5 story points (2 days)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-007
- **Priority:** P0 (Core API)
- **Target Files:**
  - `src/tensorlogic/core/composition.py` (create)
  - `tests/test_core/test_composition.py` (create)

**[CORE-009] Property Tests for Quantifiers**

- **Description:** Hypothesis-based tests for quantifier correctness
- **Acceptance:**
  - [ ] ∃x.True = True (tautology)
  - [ ] ∀x.False = False (contradiction)
  - [ ] ¬∃x.P(x) = ∀x.¬P(x) (quantifier negation)
  - [ ] Soft quantifiers differentiable (gradient test)
  - [ ] Edge cases: empty domain, single element
- **Effort:** 5 story points (2 days)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-007, CORE-008
- **Priority:** P0 (Correctness)
- **Target Files:**
  - `tests/test_core/test_properties.py` (edit)

---

### Phase 3: Temperature Control and Integration (Sprint 2, 8 SP)
**Goal:** Implement temperature-controlled reasoning and final integration
**Deliverable:** Complete CoreLogic module with deductive/analogical modes

#### Tasks

**[CORE-010] Temperature-Controlled Operations**

- **Description:** Implement temperature parameter for interpolating hard/soft reasoning
- **Acceptance:**
  - [ ] `temperature_scaled_operation(op, temperature)` wrapper
  - [ ] T=0.0: Hard boolean (step function applied)
  - [ ] T>0.0: Soft probabilistic (continuous operations)
  - [ ] Interpolation strategy documented
  - [ ] Examples: deductive (T=0) vs analogical (T>0)
  - [ ] Integration with existing operations
- **Effort:** 5 story points (2 days)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-009
- **Priority:** P1 (High - Differentiator)
- **Target Files:**
  - `src/tensorlogic/core/temperature.py` (create)
  - `tests/test_core/test_temperature.py` (create)

**[CORE-011] End-to-End Integration Tests**

- **Description:** Full integration tests demonstrating complete rule execution
- **Acceptance:**
  - [ ] Aunt example: Aunt(x,z) ← Sister(x,y) ∧ Parent(y,z)
  - [ ] Multi-hop reasoning: Ancestor via transitive closure
  - [ ] Quantified rules: ∃y: Related(x,y) ∧ HasProperty(y)
  - [ ] Temperature switching tested (T=0 vs T>0)
  - [ ] Performance benchmark: 12-equation transformer baseline
  - [ ] Documentation examples executable
- **Effort:** 3 story points (1-2 days)
- **Owner:** Backend Engineer
- **Dependencies:** CORE-010
- **Priority:** P0 (Validation)
- **Target Files:**
  - `tests/test_core/test_integration.py` (create)

---

## Critical Path

```plaintext
CORE-002 → CORE-003 → CORE-004 → CORE-006
  (2d)      (1d)       (2d)       (1d)
              ↓
          CORE-005 → CORE-007 → CORE-008 → CORE-009 → CORE-010 → CORE-011
            (1d)      (2d)       (2d)       (2d)       (2d)       (1d)

[Total: 16 days sequential, ~7 days with parallelization]
```

**Bottlenecks:**
- CORE-004: Property tests (highest complexity, critical for correctness)
- CORE-010: Temperature scaling (novel implementation, requires research)

**Parallel Tracks:**
- Track 1: CORE-002 → CORE-003 → CORE-004 (Operations + properties)
- Track 2: CORE-005 → CORE-007 → CORE-009 (Step + quantifiers)
- Track 3: CORE-008 (Composition, parallel with CORE-009)

---

## Quick Wins (Days 1-3)

1. **[CORE-002] Basic Operations** - Unblocks all downstream work
2. **[CORE-005] Step Function** - Enables quantifiers
3. **[CORE-003] Implication** - Completes operation set early

---

## Risk Mitigation

| Task | Risk | Mitigation | Contingency |
|------|------|------------|-------------|
| CORE-004 | Property test complexity | Start with basic axioms, expand gradually | Use parametrized tests as fallback |
| CORE-007 | Quantifier numerical stability | Validate against NumPy reference | Clamp values before aggregation |
| CORE-010 | Temperature formulation unclear | Research softmax temperature scaling | Implement simple linear interpolation |
| CORE-011 | Performance benchmarks fail | Profile and optimize einsum usage | Document performance targets as future work |

---

## Testing Strategy

### Automated Testing Tasks
- **[CORE-004]** Unit + Property Tests - Operations (5 SP) - Sprint 1
- **[CORE-006]** Cross-Backend Validation (3 SP) - Sprint 1
- **[CORE-009]** Property Tests - Quantifiers (5 SP) - Sprint 2
- **[CORE-011]** Integration Tests (3 SP) - Sprint 2

### Quality Gates
- 90% code coverage required (enforced by pytest-cov)
- All property tests pass (hypothesis)
- Cross-backend validation: MLX matches NumPy (FP32 tolerance)
- Type checking: mypy --strict passes

### Test Categories
1. **Unit tests:** Individual operations, edge cases
2. **Property tests:** Logical axioms via hypothesis
3. **Integration tests:** Multi-operation rule composition
4. **Cross-backend:** MLX vs NumPy equivalence
5. **Performance:** Benchmark simple vs complex rules

---

## Team Allocation

**Backend Engineers (1-2)**
- Core operations: CORE-002, CORE-003, CORE-005
- Quantifiers: CORE-007, CORE-008
- Temperature control: CORE-010
- Testing: CORE-004, CORE-006, CORE-009, CORE-011

**Optional QA Support (0.5 engineer)**
- Property test expansion
- Integration test scenarios
- Performance baseline validation

---

## Sprint Planning

**2-week sprints, ~20 SP velocity per engineer**

| Sprint | Focus | Story Points | Key Deliverables |
|--------|-------|--------------|------------------|
| Sprint 1 | Operations & Quantifiers | 23 SP | AND/OR/NOT, EXISTS/FORALL, property tests |
| Sprint 2 | Composition & Temperature | 18 SP | Rule composition, temperature control, integration |

**Total:** 41 story points over 2 sprints (5-7 days with 1-2 engineers)

---

## Dependencies and Integration

### Upstream Dependencies
- **BACKEND-001:** TensorBackend protocol (COMPLETED)
  - Required: `TensorBackend.multiply`, `maximum`, `subtract`, `einsum`, `sum`, `prod`, `step`
  - Used by: All CORE operations

### Downstream Consumers
- **API-001:** PatternAPI (uses CoreLogic for pattern execution)
- **COMP-001:** Compilation (uses CoreLogic for semantic strategies)

### Integration Points
- **TensorBackend:** All operations backend-agnostic
- **Type system:** Modern Python 3.12+ hints throughout
- **Testing:** pytest + hypothesis framework
- **Documentation:** Google-style docstrings with mathematical formulations

---

## Ticket Import Format

```csv
ID,Title,Description,Estimate,Priority,Assignee,Dependencies,Sprint
CORE-002,Basic Logical Operations,Implement AND/OR/NOT via tensors,5,P0,Backend,BACKEND-001,1
CORE-003,Logical Implication,Implement IMPLIES operation,3,P0,Backend,CORE-002,1
CORE-004,Property Tests - Operations,Hypothesis tests for axioms,5,P0,Backend,CORE-003,1
CORE-005,Step Function,Heaviside step for boolean conversion,2,P0,Backend,CORE-002,1
CORE-006,Cross-Backend Validation,MLX vs NumPy equivalence,3,P0,Backend,CORE-004;CORE-005,1
CORE-007,Quantifier Operations,EXISTS and FORALL implementation,5,P0,Backend,CORE-005,1
CORE-008,Rule Composition,compose_rules utility,5,P0,Backend,CORE-007,2
CORE-009,Property Tests - Quantifiers,Hypothesis tests for quantifiers,5,P0,Backend,CORE-007;CORE-008,2
CORE-010,Temperature Control,Temperature-scaled operations,5,P1,Backend,CORE-009,2
CORE-011,Integration Tests,End-to-end rule execution,3,P0,Backend,CORE-010,2
```

---

## Estimation Method

**Approach:** Planning Poker with Fibonacci scale (1,2,3,5,8,13,21)

**Story Point Scale:**
- 1-2 SP: Simple functions, unit tests (e.g., CORE-005 step function)
- 3-5 SP: Moderate complexity, property tests (e.g., CORE-002 operations)
- 8+ SP: High complexity, research required (none in this breakdown)

**Buffers:**
- 20% general uncertainty buffer included in estimates
- Pattern confidence: 85% → 1.2x multiplier applied
- Research buffer: CORE-010 (temperature) has extra 1 SP for formulation research

---

## Appendix

### Mathematical References
- **Domingos Paper:** arXiv:2510.12269, Sections 2-3 (logical operations)
- **Property Axioms:** Standard first-order logic textbooks
- **Temperature Scaling:** Softmax temperature in neural networks

### Implementation Patterns
- **Backend abstraction:** `.sage/agent/examples/python/patterns/protocol-based.md`
- **Modern type hints:** `.sage/agent/examples/python/types/modern-type-hints.md`
- **Property testing:** `.sage/agent/examples/python/testing/property-based.md`

### Definition of Done (Per Task)
- [ ] Code implemented with 100% type hints
- [ ] Unit tests written and passing
- [ ] Property tests added (where applicable)
- [ ] Cross-backend validation passing
- [ ] mypy --strict passes
- [ ] pytest coverage ≥90%
- [ ] Documentation updated (docstrings + examples)
- [ ] Code review approved
- [ ] Merged to feature branch

---

**Next Steps:**
1. Generate ticket files for CORE-002 through CORE-011
2. Update `.sage/tickets/index.json` with new tickets
3. Begin implementation with CORE-002 (basic operations)
