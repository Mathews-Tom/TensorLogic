# Verification Component Implementation Blueprint (PRP)

**Format:** Product Requirements Prompt (Context Engineering)
**Generated:** 2025-12-10
**Specification:** `docs/specs/verification/spec.md`
**Component ID:** VERIF-001
**Priority:** P1 (High - Strategic Differentiator)

---

## 📖 Context & Documentation

### Traceability Chain

**Strategic Assessment → Research Intelligence → Specification → This Plan**

1. **Strategic Assessment:** docs/TensorLogic-Overview.md
   - Lean 4 integration as unique differentiator (no competitor has this)
   - Harmonic AI $100M raise validates formal verification market
   - First neural-symbolic framework with provably correct reasoning

2. **Research & Intelligence:** docs/research/intel.md
   - LeanDojo provides bidirectional Python↔Lean bridge
   - Lean Copilot achieves 74.2% proof automation (vs 40.1% baseline)
   - Mathlib provides extensive theorem library for foundations
   - Lean 4 kernel provides trusted verification core

3. **Formal Specification:** docs/specs/verification/spec.md
   - LeanDojo integration for theorem proving
   - Core operation verification (AND, OR, NOT, IMPLIES, EXISTS, FORALL)
   - Proof-guided training with constraint penalties
   - Neural tactic suggestion for proof automation

### Related Documentation

**System Context:**
- Architecture: `.sage/agent/system/architecture.md` - 4-layer architecture (Verification layer 4)
- Tech Stack: `.sage/agent/system/tech-stack.md` - Python 3.12+, Lean 4, LeanDojo
- Patterns: `.sage/agent/system/patterns.md` - Protocol-based abstractions

**Code Examples:**
- `.sage/agent/examples/python/types/modern-type-hints.md` - Modern Python 3.12+ syntax
- `.sage/agent/examples/python/testing/pytest-hypothesis.md` - Property-based testing

**Dependencies:**
- Upstream: CORE-001 (operations to verify), BACKEND-001 (TensorBackend)
- Integration: API-001 (optional pattern verification)
- External: Lean 4 system installation, LeanDojo Python package

---

## 📊 Executive Summary

### Business Alignment

**Purpose:** Provide formal verification of tensor logic operations and learned neural predicates using Lean 4 theorem prover, enabling provably correct neural-symbolic AI.

**Value Proposition:**
- **Unique differentiator:** First neural-symbolic framework with formal verification
- **Mathematical guarantees:** Prove operations satisfy logical properties before deployment
- **Market validation:** Harmonic AI $100M raise demonstrates verification market demand
- **Safety-critical applications:** Enable deployment in high-stakes domains requiring correctness proofs
- **Research enabler:** Publish verified neural-symbolic models with proof certificates

**Target Users:**
- Researchers requiring mathematical guarantees for publications
- Production systems with safety-critical reasoning requirements
- Academic users needing formal verification for peer review
- Enterprise users in regulated industries (healthcare, finance, aerospace)

### Technical Approach

**Architecture Pattern:** Bidirectional Bridge with Optional Verification

**Bridge Layer:**
- LeanDojo Python package for Lean 4 communication
- Lean project with TensorLogic theorem definitions
- Result dataclasses for verification outcomes
- Async verification during training (minimize overhead)

**Verification Levels:**
1. **Operation verification:** Prove core ops satisfy logical axioms
2. **Predicate verification:** Check learned predicates satisfy constraints
3. **Pattern verification:** Validate pattern semantics (future)
4. **Compilation verification:** Prove compilation strategies correct (future)

**Implementation Strategy:**
- Phase 1: LeanDojo bridge + basic theorem definitions (Week 1)
- Phase 2: Core operation verification (Week 2)
- Phase 3: Predicate verification API (Week 3)
- Phase 4: Proof-guided training integration (Week 4)

### Key Success Metrics

**Service Level Objectives (SLOs):**
- **Verification overhead:** <15% added to training pipeline
- **Proof automation:** ≥70% success rate (Lean Copilot: 74.2%)
- **Theorem coverage:** 100% of core operations verified
- **Test coverage:** ≥85% (Lean integration hard to test exhaustively)

**Key Performance Indicators (KPIs):**
- **Kernel validation:** All proofs checked by Lean's trusted kernel
- **Counterexample detection:** Failed verifications provide concrete examples
- **Tactic suggestion latency:** <1 second for neural proof search
- **Proof caching:** Avoid recomputation of verified theorems

---

## 💻 Code Examples & Patterns

### Repository Patterns

**1. Modern Type Hints:** `.sage/agent/examples/python/types/modern-type-hints.md`

**Application:** 100% type hint coverage for verification APIs

```python
from __future__ import annotations
from typing import TypeAlias, Any
from collections.abc import Callable
from dataclasses import dataclass, field

# Type aliases for clarity
TheoremName: TypeAlias = str
ProofScript: TypeAlias = str
TacticSuggestion: TypeAlias = tuple[str, float]  # (tactic, confidence)

@dataclass
class VerificationResult:
    """Result of Lean verification."""

    verified: bool
    theorem_name: str
    proof_trace: str | None = None
    counterexample: Any | None = None
    verification_time: float = 0.0
    tactic_suggestions: list[TacticSuggestion] = field(default_factory=list)

    def __str__(self) -> str:
        """Human-readable verification report."""
        status = "✓ VERIFIED" if self.verified else "✗ FAILED"
        report = f"{status}: {self.theorem_name}\n"

        if self.counterexample is not None:
            report += f"Counterexample: {self.counterexample}\n"

        if self.tactic_suggestions:
            report += "Suggested tactics:\n"
            for tactic, conf in self.tactic_suggestions[:3]:
                report += f"  - {tactic} (confidence: {conf:.2%})\n"

        return report
```

**2. Async Integration Pattern**

**Application:** Background verification to minimize training overhead

```python
import asyncio
from typing import Protocol

class AsyncVerifiable(Protocol):
    """Protocol for components supporting async verification."""

    async def verify_async(
        self,
        lean_bridge: LeanBridge,
        constraints: list[str],
    ) -> VerificationResult:
        """Verify asynchronously during training."""
        ...

# Usage in training loop
async def train_with_verification(
    predicate: Callable,
    examples: Array,
    lean_bridge: LeanBridge,
) -> None:
    # Start verification in background
    verification_task = asyncio.create_task(
        verify_predicate_async(predicate, ["transitivity"], lean_bridge)
    )

    # Continue training
    loss = compute_loss(predicate, examples)
    update_parameters(loss)

    # Check verification result
    result = await verification_task
    if not result.verified:
        logger.warning(f"Verification failed: {result}")
```

### Implementation Reference Examples

**From Verification Spec (lines 32-70):**

```python
class LeanBridge:
    """Bridge to Lean 4 theorem prover via LeanDojo."""

    def __init__(self, lean_project_path: str):
        """Initialize Lean bridge.

        Args:
            lean_project_path: Path to Lean 4 project with theorems

        Raises:
            RuntimeError: If Lean 4 not installed or project invalid
        """
        self.project_path = Path(lean_project_path)
        self._validate_lean_installation()
        self._load_project()

    def verify_theorem(
        self,
        theorem_name: str,
        proof_script: str | None = None,
    ) -> VerificationResult:
        """Verify theorem in Lean.

        Args:
            theorem_name: Lean theorem identifier (e.g., 'TensorLogic.and_commutative')
            proof_script: Optional proof tactics (use auto if None)

        Returns:
            VerificationResult with success/failure and proof trace

        Examples:
            >>> bridge = LeanBridge("lean/")
            >>> result = bridge.verify_theorem("TensorLogic.and_commutative")
            >>> assert result.verified
        """

    def suggest_tactics(
        self,
        goal_state: str,
    ) -> list[tuple[str, float]]:
        """Neural proof search for tactics.

        Args:
            goal_state: Current Lean goal state (e.g., "⊢ a ∧ b = b ∧ a")

        Returns:
            List of (tactic, confidence) suggestions sorted by confidence

        Examples:
            >>> suggestions = bridge.suggest_tactics("⊢ a ∧ b = b ∧ a")
            >>> print(suggestions[0])  # ("apply and.comm", 0.87)
        """
```

**Key Takeaways:**
- **Optional integration:** Verification is opt-in (doesn't block basic usage)
- **Clear error reporting:** Failed verifications include counterexamples
- **Async-first design:** Minimize overhead on training pipeline

**Anti-patterns to Avoid:**
- ❌ Blocking verification (use async to avoid training delays)
- ❌ Silent verification failures (always report counterexamples)
- ❌ Axiom abuse (prove constructively, avoid sorry/admit)

---

## 🔧 Technology Stack

### Recommended Stack

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Runtime | Python | ≥3.12 | Modern type hints, async/await |
| Theorem Prover | Lean 4 | ≥4.0 | Trusted kernel, extensive mathlib |
| Python Bridge | LeanDojo | latest | Bidirectional Python↔Lean communication |
| Backend | TensorBackend | - | Execute operations to verify |
| Testing | pytest | ≥7.0 | Unit and integration tests |
| Property Testing | hypothesis | ≥6.0 | Generate test cases for theorems |
| Type Checking | mypy | ≥1.0 | Strict mode for type safety |

**Key Technology Decisions:**

1. **Lean 4 over Coq/Isabelle/Agda:**
   - **Rationale:** LeanDojo provides Python bridge, Mathlib has extensive foundations, Lean Copilot achieves 74.2% automation
   - **Source:** Strategic intel - Harmonic AI $100M raise validates Lean 4 ecosystem
   - **Trade-off:** Smaller community than Coq, but rapidly growing

2. **LeanDojo over direct Lean API:**
   - **Rationale:** Higher-level Python API, handles subprocess management, proof state tracking
   - **Source:** Verification spec FR-1 (LeanDojo Integration)
   - **Trade-off:** Additional dependency, but significantly simpler integration

3. **Optional verification (not mandatory):**
   - **Rationale:** Don't block basic usage, verification is value-add for safety-critical applications
   - **Source:** Specification NFR-3 (Usability - optional integration)
   - **Trade-off:** Users might skip verification, but better than forcing it

### Alternatives Considered

**Option 2: Coq + coq-python**
- **Pros:** Mature ecosystem, extensive tactics, strong academic support
- **Cons:** Steeper learning curve, no LeanDojo equivalent, less active Python integration
- **Why Not Chosen:** Lean 4 has better Python tooling and growing momentum

**Option 3: Isabelle/HOL**
- **Pros:** Very mature, extensive libraries, excellent automation (sledgehammer)
- **Cons:** ML-based language barrier, no modern Python bridge, smaller AI community
- **Why Not Chosen:** Poor Python integration, less momentum in ML community

**Option 4: No Formal Verification**
- **Pros:** Simpler implementation, no Lean dependency
- **Cons:** No mathematical guarantees, missed market differentiator
- **Why Not Chosen:** Strategic value too high (Harmonic AI $100M validates market)

### Alignment with Existing System

**From `.sage/agent/system/tech-stack.md`:**
- **Consistent With:** Python 3.12+, pytest+hypothesis testing, mypy strict
- **New Additions:** Lean 4, LeanDojo
- **Migration Considerations:** Lean 4 system installation required for verification features

---

## 🏗️ Architecture Design

### System Context

**From `.sage/agent/system/architecture.md`:**

Current architecture is 4-layer:
1. **Core:** Tensor logic primitives (depends on Backends)
2. **Backends:** TensorBackend abstraction
3. **API:** High-level patterns (depends on Core)
4. **Verification:** This component (depends on Core + Backends)

**Integration Points:**
- **Upstream (depends on Verification):** None (leaf component)
- **Downstream (Verification depends on):** CoreLogic (operations to verify), TensorBackend (execution)

### Component Architecture

**Architecture Pattern:** Bridge with Optional Verification Hooks

```
┌──────────────────────────────────────────────────┐
│              LeanBridge                          │
│  (Python ↔ Lean 4 communication)                 │
└──────────────┬───────────────────────────────────┘
               │
               │ calls Lean theorems
               │
    ┌──────────▼──────────┐
    │  Lean 4 Project     │
    │  (TensorLogic.lean) │
    │  - Theorem defs     │
    │  - Proof scripts    │
    │  - Tactics          │
    └──────────┬──────────┘
               │
               │ verification results
               │
┌──────────────▼───────────────────────────────────┐
│           VerificationResult                     │
│  (verified, proof_trace, counterexample)         │
└──────────────┬───────────────────────────────────┘
               │
               │ consumed by
               │
    ┌──────────▼──────────┐    ┌────────────────┐
    │ verify_predicate()  │    │ ProofGuided    │
    │ (API function)      │    │ Trainer        │
    └─────────────────────┘    └────────────────┘
```

**Verification Flow:**
1. User calls `verify_predicate(pred, constraints, lean_bridge)`
2. Python extracts predicate behavior as test cases
3. LeanBridge sends theorem + test cases to Lean 4
4. Lean 4 kernel attempts proof
5. Bridge returns VerificationResult (verified or counterexample)
6. User inspects result, updates predicate if needed

**Rationale:**
- Bridge pattern decouples Python from Lean implementation details
- Optional hooks allow verification without modifying core code
- Async design minimizes training overhead

### Architecture Decisions

**Decision 1: Bidirectional Bridge vs One-Way Verification**
- **Choice:** Bidirectional (Python→Lean for verification, Lean→Python for tactic suggestions)
- **Rationale:** Enable proof-guided learning with neural tactic suggestion
- **Implementation:** LeanDojo supports both directions
- **Trade-offs:**
  - ✅ Pro: Neural proof search improves automation
  - ✅ Pro: Richer feedback for debugging
  - ❌ Con: More complex integration

**Decision 2: Theorem Definitions in Lean vs Python**
- **Choice:** Define theorems in Lean, wrap in Python
- **Rationale:** Lean kernel validates proofs (trusted core), Python layer is convenience
- **Implementation:** `lean/TensorLogic.lean` with Python wrappers in `theorems.py`
- **Trade-offs:**
  - ✅ Pro: Proofs checked by trusted kernel
  - ✅ Pro: Lean community can review theorems
  - ⚠️ Con: Users need basic Lean knowledge for advanced usage

**Decision 3: Sync vs Async Verification**
- **Choice:** Both (sync for simple cases, async for training)
- **Rationale:** Flexibility - sync for debugging, async for production training
- **Implementation:** `verify_theorem()` sync, `verify_theorem_async()` async
- **Trade-offs:**
  - ✅ Pro: Best of both worlds
  - ✅ Pro: <15% overhead with async
  - ❌ Con: Two APIs to maintain

**Decision 4: Constructive Proofs vs Axioms**
- **Choice:** Constructive (avoid sorry/admit)
- **Rationale:** Mathematical rigor, no trust holes
- **Implementation:** Mark axiom-based proofs clearly, work toward constructive versions
- **Trade-offs:**
  - ✅ Pro: Stronger guarantees
  - ⚠️ Con: Harder to prove (may need placeholders initially)

### Component Breakdown

**1. LeanBridge** (`src/tensorlogic/verification/lean_bridge.py`)
- **Purpose:** Manage Lean 4 communication via LeanDojo
- **Technology:** LeanDojo Python package
- **Pattern:** Bridge pattern for external system integration
- **Interfaces:** `verify_theorem()`, `suggest_tactics()`, `load_project()`
- **Dependencies:** LeanDojo, Lean 4 system installation

**2. Theorem Definitions** (`lean/TensorLogic.lean`)
- **Purpose:** Define logical properties as Lean theorems
- **Technology:** Lean 4 language
- **Pattern:** Mathematical theorem library
- **Interfaces:** Theorem declarations (and_commutative, demorgan_and, etc.)
- **Dependencies:** Lean 4 Mathlib (for foundations)

**3. Theorem Wrappers** (`src/tensorlogic/verification/theorems.py`)
- **Purpose:** Python-friendly wrappers for Lean theorems
- **Technology:** Python dataclasses, LeanBridge
- **Pattern:** Facade pattern over Lean theorems
- **Interfaces:** Python functions calling `lean_bridge.verify_theorem()`
- **Dependencies:** LeanBridge

**4. Predicate Verification** (`src/tensorlogic/verification/predicates.py`)
- **Purpose:** High-level API for verifying neural predicates
- **Technology:** Python, hypothesis for test case generation
- **Pattern:** Template Method (extract behavior, verify, report)
- **Interfaces:** `verify_predicate(predicate, constraints, lean_bridge)`
- **Dependencies:** LeanBridge, hypothesis, TensorBackend

**5. Proof-Guided Training** (`src/tensorlogic/verification/training.py`)
- **Purpose:** Integrate verification into training loop
- **Technology:** Async Python, MLX gradients
- **Pattern:** Observer pattern (verification observes training)
- **Interfaces:** `ProofGuidedTrainer` class
- **Dependencies:** LeanBridge, TensorBackend, CoreLogic

**6. Verification Results** (`src/tensorlogic/verification/results.py`)
- **Purpose:** Structured verification outcomes
- **Technology:** Python dataclasses
- **Pattern:** Value Object
- **Interfaces:** `VerificationResult` dataclass
- **Dependencies:** None (pure Python)

### Data Flow & Boundaries

**Verification Request Flow:**
1. User calls `verify_predicate(similarity, ["transitivity"], bridge)`
2. Extract predicate behavior as test cases (hypothesis generates)
3. Encode test cases in Lean format
4. LeanBridge sends to Lean 4: `verify TensorLogic.transitivity with examples`
5. Lean 4 kernel attempts proof
6. Lean returns result (proof trace or counterexample)
7. Bridge parses result into VerificationResult
8. User inspects result

**Component Boundaries:**
- **Public Interface:** `verify_predicate()`, `LeanBridge`, `ProofGuidedTrainer`
- **Internal Implementation:** LeanDojo subprocess management, Lean parsing
- **Cross-Component Contracts:** CoreLogic operations must be verifiable

---

## ⚠️ Error Handling & Edge Cases

### Error Scenarios

**1. Lean 4 Not Installed**
- **Cause:** User tries verification but Lean 4 not in PATH
- **Impact:** RuntimeError on bridge initialization
- **Handling:** Check Lean installation in `__init__`, provide clear error
- **Recovery:** Suggest installation command for platform
- **User Experience:**
  ```python
  raise RuntimeError(
      "Lean 4 not found. Install from: https://lean-lang.org/lean4/doc/setup.html",
      suggestion="Or disable verification with verify=False"
  )
  ```

**2. LeanDojo Import Failure**
- **Cause:** LeanDojo package not installed
- **Impact:** ImportError when importing verification module
- **Handling:** Lazy import, clear error message
- **Recovery:** Suggest `pip install lean-dojo`
- **Pattern:** Optional dependency (verification is opt-in)

**3. Theorem Proof Fails**
- **Cause:** Operation doesn't satisfy claimed property
- **Impact:** VerificationResult.verified = False
- **Handling:** Return counterexample from Lean
- **Recovery:** User updates operation or theorem
- **User Experience:**
  ```python
  result = verify_theorem("and_commutative")
  if not result.verified:
      print(f"Counterexample: {result.counterexample}")
      # Shows: a=True, b=False fails commutativity
  ```

**4. Proof Timeout**
- **Cause:** Proof search exceeds time limit
- **Impact:** Lean subprocess hangs
- **Handling:** Set timeout in LeanDojo, catch TimeoutError
- **Recovery:** Suggest manual proof or simpler property
- **Configuration:** `verify_theorem(..., timeout_seconds=30)`

### Edge Cases

**1. Empty Predicate (Always False)**
- **Detection:** Predicate returns all False for test cases
- **Handling:** Special case in verification (trivially satisfies some properties)
- **Testing:** Parametrized test with degenerate predicates

**2. Axiom-Based Proofs (sorry/admit)**
- **Detection:** Proof script contains `sorry` or `admit`
- **Handling:** Mark in VerificationResult, warn user
- **Testing:** Check proof trace for axiom usage

**3. Circular Theorem Dependencies**
- **Detection:** Theorem A proves B, B proves A
- **Handling:** Lean kernel rejects circular proofs
- **Testing:** Attempt circular proofs in test suite

**4. Counterexample Too Large**
- **Detection:** Lean returns massive counterexample (large tensors)
- **Handling:** Truncate in VerificationResult, log full version
- **User Experience:** Show summary, link to full output

### Input Validation

**Validation Strategy:** Fail fast with clear errors

- **Theorem name validation:** Check exists in Lean project before calling
- **Predicate validation:** Ensure callable, correct signature
- **Constraint validation:** Verify constraint list non-empty, all strings

**Example:**
```python
def verify_predicate(
    predicate: Callable,
    constraints: list[str],
    *,
    lean_bridge: LeanBridge,
) -> VerificationResult:
    # Validate inputs
    if not callable(predicate):
        raise TypeError(f"Predicate must be callable, got {type(predicate)}")

    if not constraints:
        raise ValueError("Constraints list cannot be empty")

    if not all(isinstance(c, str) for c in constraints):
        raise TypeError("All constraints must be theorem name strings")

    # Proceed with verification
    ...
```

### Graceful Degradation

**Verification unavailable → Skip verification:**
```python
def verify_if_available(
    predicate: Callable,
    constraints: list[str],
) -> VerificationResult | None:
    try:
        bridge = LeanBridge("lean/")
        return verify_predicate(predicate, constraints, lean_bridge=bridge)
    except (ImportError, RuntimeError):
        logger.warning("Verification unavailable, skipping")
        return None
```

---

## 📚 Implementation Roadmap

### Phase 1: LeanDojo Bridge + Basic Theorems (Week 1)

**Days 1-2: Lean 4 Setup**
- [ ] Install Lean 4 and verify installation
- [ ] Create `lean/` directory structure
- [ ] Initialize Lean project with `lakefile.lean`
- [ ] Add Mathlib dependency for foundations
- [ ] Test basic Lean theorem (hello world)

**Days 3-4: LeanBridge Implementation**
- [ ] Install LeanDojo Python package
- [ ] Implement `LeanBridge.__init__()` with project loading
- [ ] Implement `verify_theorem()` sync method
- [ ] Add timeout handling for proof search
- [ ] Write unit tests for bridge communication

**Day 5: VerificationResult**
- [ ] Define `VerificationResult` dataclass
- [ ] Add `__str__()` for human-readable reports
- [ ] Add JSON serialization for logging
- [ ] Write tests for result formatting

### Phase 2: Core Operation Verification (Week 2)

**Days 6-8: Lean Theorem Definitions**
- [ ] Define `TensorLogic` namespace in Lean
- [ ] Theorem: `and_commutative` (a ∧ b = b ∧ a)
- [ ] Theorem: `and_associative` ((a ∧ b) ∧ c = a ∧ (b ∧ c))
- [ ] Theorem: `or_commutative` (a ∨ b = b ∨ a)
- [ ] Theorem: `demorgan_and` (¬(a ∧ b) = ¬a ∨ ¬b)
- [ ] Theorem: `demorgan_or` (¬(a ∨ b) = ¬a ∧ ¬b)
- [ ] Theorem: `and_distributes_or` (a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c))
- [ ] Theorem: `exists_distributes_or` (∃x. (P x ∨ Q x) = (∃x. P x) ∨ (∃x. Q x))

**Days 9-10: Python Theorem Wrappers**
- [ ] Implement `theorems.py` with wrapper functions
- [ ] Function: `verify_and_commutative()`
- [ ] Function: `verify_demorgan_laws()`
- [ ] Integration tests: Verify CoreLogic operations match theorems
- [ ] Property tests: Generate random inputs, verify properties hold

### Phase 3: Predicate Verification API (Week 3)

**Days 11-12: Test Case Generation**
- [ ] Implement `predicates.py` module
- [ ] Use hypothesis to generate test cases for predicates
- [ ] Encode Python predicate behavior in Lean format
- [ ] Function: `extract_predicate_behavior()`
- [ ] Unit tests for encoding

**Days 13-14: Verification API**
- [ ] Implement `verify_predicate()` function
- [ ] Handle constraint list (multiple theorem names)
- [ ] Return VerificationResult with all constraint outcomes
- [ ] Integration tests: Verify sample predicates
- [ ] Documentation with examples

**Day 15: Counterexample Reporting**
- [ ] Parse Lean counterexamples into Python objects
- [ ] Add counterexample visualization
- [ ] Test with intentionally wrong predicates
- [ ] User-friendly error messages

### Phase 4: Proof-Guided Training (Week 4)

**Days 16-18: ProofGuidedTrainer**
- [ ] Implement `ProofGuidedTrainer` class
- [ ] Training step with async verification
- [ ] Loss augmentation for constraint violations
- [ ] Constraint satisfaction metrics
- [ ] Unit tests for trainer

**Days 19-20: Neural Tactic Suggestion**
- [ ] Implement `suggest_tactics()` in LeanBridge
- [ ] Use LeanDojo's proof search API
- [ ] Cache tactic suggestions
- [ ] Integration with ProofGuidedTrainer
- [ ] Benchmarks against Lean Copilot (target: ≥70%)

**Day 21: Integration & Performance**
- [ ] End-to-end test: Train predicate with constraints
- [ ] Measure verification overhead (<15% target)
- [ ] Async optimization (background verification)
- [ ] Performance profiling

### Phase 5: Documentation & Hardening (Week 4+)

**Days 22-24: Documentation**
- [ ] User guide: Getting started with verification
- [ ] Lean theorem reference documentation
- [ ] Example: Verify similarity predicate (transitivity, symmetry)
- [ ] Example: Train with proof guidance
- [ ] API reference (Sphinx autodoc)

**Days 25-26: Production Readiness**
- [ ] Achieve ≥85% test coverage
- [ ] Pass mypy strict mode (100% type hints)
- [ ] Error message improvements
- [ ] Optional verification flag (disable for CI)
- [ ] Installation guide (Lean 4 + LeanDojo)

**Day 27+: Optional Enhancements**
- [ ] Automated proof search (neural tactics)
- [ ] Pattern verification (verify pattern semantics)
- [ ] Certified predicate export (proof certificates)
- [ ] Integration with theorem libraries (Mathlib extensions)

---

## 🧪 Quality Assurance

### Testing Strategy

**Unit Tests (pytest):**
- LeanBridge initialization and project loading
- Verification of individual theorems
- VerificationResult serialization
- Error handling (Lean not installed, theorem not found)
- Timeout handling for long proofs

**Integration Tests:**
```python
def test_and_commutative_verified():
    """Verify CoreLogic AND satisfies commutativity."""
    bridge = LeanBridge("lean/")

    # Test with CoreLogic implementation
    result = verify_theorem("TensorLogic.and_commutative")

    assert result.verified
    assert "and.comm" in result.proof_trace
    assert result.verification_time < 1.0
```

**Property-Based Tests (hypothesis):**
```python
from hypothesis import given, strategies as st

@given(
    a=st.booleans(),
    b=st.booleans(),
)
def test_and_commutativity_property(a, b):
    """Property: AND(a, b) == AND(b, a) for all a, b."""
    from tensorlogic.core import logical_and

    result_ab = logical_and(a, b)
    result_ba = logical_and(b, a)

    assert result_ab == result_ba
```

**End-to-End Tests:**
```python
async def test_proof_guided_training():
    """Train predicate with transitivity constraint."""
    bridge = LeanBridge("lean/")
    trainer = ProofGuidedTrainer(bridge, backend)

    # Initial predicate (not transitive)
    predicate = lambda x, y: random_similarity(x, y)

    # Train with transitivity constraint
    for epoch in range(100):
        loss, metrics = await trainer.train_step(
            predicate,
            examples,
            constraints=["transitivity"]
        )

    # Final verification
    result = verify_predicate(
        predicate,
        constraints=["transitivity"],
        lean_bridge=bridge
    )

    assert result.verified
    assert metrics["constraint_satisfaction"] > 0.95
```

**Code Quality Gates:**
- **Type Checking:** `mypy --strict src/tensorlogic/verification` (0 errors)
- **Linting:** `ruff check src/tensorlogic/verification` (0 warnings)
- **Coverage:** `pytest --cov=tensorlogic.verification --cov-report=term-missing` (≥85%)
- **Lean Check:** `lake build` in `lean/` directory (0 errors)

### Deployment Verification

- [ ] LeanBridge loads on macOS with Lean 4 installed
- [ ] LeanBridge loads on Linux with Lean 4 installed
- [ ] Graceful degradation on platforms without Lean (skip verification)
- [ ] All core operation theorems verified
- [ ] Proof automation ≥70% (Lean Copilot benchmark: 74.2%)
- [ ] Verification overhead <15% on training pipeline
- [ ] Async verification completes in background
- [ ] Counterexamples provided for failed verifications

---

## 📚 References & Traceability

### Source Documentation

**Strategic Assessment:**
- docs/TensorLogic-Overview.md (lines 110-120)
  - Lean 4 integration as differentiator
  - LeanDojo bidirectional bridge
  - Proof-guided learning strategy

**Research & Intelligence:**
- docs/research/intel.md
  - Harmonic AI $100M raise (validates verification market)
  - Lean Copilot 74.2% automation benchmark
  - LeanDojo architecture and capabilities

**Specification:**
- docs/specs/verification/spec.md
  - Functional requirements (bridge, theorems, predicates, training)
  - Non-functional requirements (performance, correctness, usability)
  - Acceptance criteria

**Ticket:**
- .sage/tickets/VERIF-001.md
  - Epic description and acceptance criteria
  - Target files and dependencies
  - Estimated complexity (Very High, 7-10 days)

### System Context

**Architecture & Patterns:**
- `.sage/agent/system/architecture.md` - Verification layer in 4-layer architecture
- `.sage/agent/system/tech-stack.md` - Python 3.12+, Lean 4, LeanDojo
- `.sage/agent/system/patterns.md` - Protocol-based abstractions, fail-fast errors

**Code Examples:**
- `.sage/agent/examples/python/types/modern-type-hints.md` - Modern type hints
- `.sage/agent/examples/python/testing/pytest-hypothesis.md` - Property-based testing

### Technology References

**Lean 4:**
- Website: https://lean-lang.org/
- Documentation: https://lean-lang.org/lean4/doc/
- Theorem Proving: https://lean-lang.org/theorem_proving_in_lean4/

**LeanDojo:**
- Website: https://leandojo.org/
- Paper: https://arxiv.org/abs/2306.15626
- GitHub: https://github.com/lean-dojo/LeanDojo

**Lean Copilot:**
- Paper: https://arxiv.org/abs/2404.04954
- Benchmark: 74.2% proof automation (vs 40.1% aesop baseline)

**Harmonic AI:**
- Funding: $100M raise (validates formal verification market)
- Focus: Lean 4-based AI verification infrastructure

### Related Components

**Dependents:**
- User training pipelines: Optional verification hooks
- Documentation: Verification examples and guides

**Dependencies:**
- CORE-001 (CoreLogic): Operations to verify (AND, OR, NOT, etc.)
- BACKEND-001 (TensorBackend): Execute operations for test case generation
- API-001 (PatternAPI): Optional pattern verification (future)

---

**Plan Status:** Ready for implementation
**Estimated Duration:** 4 weeks (27 working days)
**Risk Level:** High (external Lean 4 dependency, complex integration)
**Strategic Value:** Critical differentiator (no competitor has this)
