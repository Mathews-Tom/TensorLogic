# Verification Specification

**Component ID:** VERIF
**Priority:** P1 (High - Strategic Differentiator)
**Phase:** 5 (Lean 4 Bridge)
**Source:** docs/TensorLogic-Overview.md, strategic intelligence (Harmonic AI $100M raise)

## 1. Overview

### Purpose and Business Value
Integrate Lean 4 formal verification for tensor logic operations, enabling provably correct neural-symbolic reasoning. Provides mathematical guarantees that learned predicates satisfy logical constraints before deployment.

**Key Differentiator:** First neural-symbolic framework with formal verification integration. Harmonic AI raised $100M for Lean 4-based verification—validates strategic value.

### Success Metrics
- Bidirectional Python↔Lean communication via LeanDojo
- Core operations verified against formal theorems (associativity, distributivity)
- Proof-guided learning achieves ≥70% automation (Lean Copilot benchmark: 74.2%)
- Verified predicates pass Lean kernel validation
- Integration adds <15% overhead to training pipeline

### Target Users
- Researchers requiring mathematical guarantees
- Production systems with safety-critical reasoning
- Academic users publishing verified neural-symbolic models

## 2. Functional Requirements

### FR-1: LeanDojo Integration
The system **shall** use LeanDojo Python bridge for Lean 4 communication:

```python
class LeanBridge:
    """Bridge to Lean 4 theorem prover via LeanDojo."""

    def __init__(self, lean_project_path: str):
        """Initialize Lean bridge.

        Args:
            lean_project_path: Path to Lean 4 project with theorems
        """

    def verify_theorem(
        self,
        theorem_name: str,
        proof_script: str | None = None,
    ) -> VerificationResult:
        """Verify theorem in Lean.

        Args:
            theorem_name: Lean theorem identifier
            proof_script: Optional proof tactics

        Returns:
            VerificationResult with success/failure and proof trace
        """

    def suggest_tactics(
        self,
        goal_state: str,
    ) -> list[tuple[str, float]]:
        """Neural proof search for tactics.

        Args:
            goal_state: Current Lean goal state

        Returns:
            List of (tactic, confidence) suggestions
        """
```

### FR-2: Operation Theorems
The system **shall** define Lean theorems for core operations:

**Lean 4 Definitions:**
```lean
-- In TensorLogic.lean

namespace TensorLogic

-- Logical AND properties
theorem and_commutative (a b : Tensor Bool) :
  tensor_and a b = tensor_and b a :=
  sorry

theorem and_associative (a b c : Tensor Bool) :
  tensor_and (tensor_and a b) c = tensor_and a (tensor_and b c) :=
  sorry

theorem and_idempotent (a : Tensor Bool) :
  tensor_and a a = a :=
  sorry

-- Logical OR properties
theorem or_commutative (a b : Tensor Bool) :
  tensor_or a b = tensor_or b a :=
  sorry

-- De Morgan's Laws
theorem demorgan_and (a b : Tensor Bool) :
  tensor_not (tensor_and a b) = tensor_or (tensor_not a) (tensor_not b) :=
  sorry

theorem demorgan_or (a b : Tensor Bool) :
  tensor_not (tensor_or a b) = tensor_and (tensor_not a) (tensor_not b) :=
  sorry

-- Distributivity
theorem and_distributes_or (a b c : Tensor Bool) :
  tensor_and a (tensor_or b c) =
    tensor_or (tensor_and a b) (tensor_and a c) :=
  sorry

-- Quantifier properties
theorem exists_distributes_or (P Q : Tensor Bool) :
  tensor_exists (tensor_or P Q) =
    tensor_or (tensor_exists P) (tensor_exists Q) :=
  sorry

end TensorLogic
```

### FR-3: Predicate Verification
The system **shall** verify learned neural predicates satisfy constraints:

```python
def verify_predicate(
    predicate: Callable,
    constraints: list[str],
    *,
    lean_bridge: LeanBridge,
) -> VerificationResult:
    """Verify predicate satisfies logical constraints.

    Args:
        predicate: Neural predicate function
        constraints: List of Lean theorem names to verify
        lean_bridge: Lean 4 bridge

    Returns:
        VerificationResult with verified/failed constraints

    Examples:
        >>> # Verify transitivity
        >>> result = verify_predicate(
        ...     similarity_predicate,
        ...     constraints=["transitivity", "symmetry"],
        ...     lean_bridge=bridge,
        ... )
        >>> assert result.verified
    """
```

### FR-4: Proof-Guided Learning
The system **shall** use Lean proofs to guide neural predicate training:

```python
class ProofGuidedTrainer:
    """Train neural predicates with proof guidance."""

    def __init__(
        self,
        lean_bridge: LeanBridge,
        backend: TensorBackend,
    ):
        """Initialize proof-guided trainer."""

    def train_step(
        self,
        predicate: Callable,
        examples: Array,
        constraints: list[str],
    ) -> tuple[Array, dict[str, Any]]:
        """Training step with proof guidance.

        Args:
            predicate: Neural predicate to train
            examples: Training examples
            constraints: Logical constraints (Lean theorems)

        Returns:
            (loss, metrics) tuple with proof satisfaction metrics
        """
```

**Training Flow:**
1. Forward pass: Compute predicate outputs
2. Constraint checking: Verify outputs satisfy Lean theorems
3. Loss augmentation: Add penalty for constraint violations
4. Backpropagation: Update predicate parameters
5. Proof validation: Check if updated predicate passes Lean kernel

### FR-5: Verification Result Reporting
```python
@dataclass
class VerificationResult:
    """Result of Lean verification."""

    verified: bool
    theorem_name: str
    proof_trace: str | None = None
    counterexample: Any | None = None
    verification_time: float = 0.0
    tactic_suggestions: list[tuple[str, float]] = field(default_factory=list)

    def __str__(self) -> str:
        """Human-readable verification report."""
```

## 3. Non-Functional Requirements

### NFR-1: Performance
- **Verification overhead:** <15% added to training time
- **Proof search:** Tactic suggestion within 1 second
- **Caching:** Verified proofs cached to avoid recomputation
- **Async verification:** Background verification during training

### NFR-2: Correctness
- **Lean kernel validation:** All proofs checked by Lean's trusted kernel
- **No axioms:** Proofs constructive (avoid sorry/admit)
- **Sound abstractions:** Tensor operations correctly modeled in Lean

### NFR-3: Usability
- **Optional integration:** Verification is opt-in (not required for basic use)
- **Clear errors:** Verification failures include counterexamples
- **Incremental adoption:** Verify incrementally (start with core ops, expand to predicates)

## 4. Code Pattern Requirements

### Naming Conventions
- **Lean theorems:** snake_case (`and_commutative`, `demorgan_and`)
- **Python classes:** PascalCase (`LeanBridge`, `ProofGuidedTrainer`)
- **Verification methods:** `verify_<target>` pattern

### Type Safety Requirements
- **100% type hints** on Python APIs
- **Lean types:** Match Python tensor operations to Lean definitions
- **Result types:** `VerificationResult` dataclass

### Testing Approach
- **Unit tests:** Each Lean theorem has Python test
- **Integration tests:** End-to-end verification workflows
- **Property tests:** Verify Python operations match Lean axioms
- **Coverage:** ≥85% (Lean integration hard to test exhaustively)

## 5. Acceptance Criteria

### Definition of Done
- [ ] LeanDojo Python bridge integrated
- [ ] Lean 4 project with TensorLogic theorems
- [ ] Core operations verified (AND, OR, NOT, IMPLIES, EXISTS, FORALL)
- [ ] `verify_predicate()` API for neural predicate verification
- [ ] `ProofGuidedTrainer` for training with constraints
- [ ] Verification result reporting with counterexamples
- [ ] Documentation with Lean theorem examples
- [ ] 100% type hints, passes mypy strict
- [ ] ≥85% test coverage
- [ ] Performance: <15% overhead

### Validation Approach
1. **Theorem tests:** Verify each Lean theorem from Python
2. **Property validation:** Python operations satisfy Lean axioms
3. **Predicate tests:** Train neural predicate with constraints
4. **Performance:** Measure verification overhead
5. **Usability:** User study (optional verification workflow)

## 6. Dependencies

### Technical Assumptions
- **Lean 4:** Version ≥4.0 installed
- **LeanDojo:** Python package for Lean integration
- **Mathlib:** Lean mathematics library (optional)
- **Platform:** macOS or Linux (Lean 4 support)

### External Integrations
- **LeanDojo:** `pip install lean4-python` (future package)
- **Lean 4:** System installation required
- **CoreLogic:** Operations to verify
- **TensorBackend:** For tensor operations in proofs

### Related Components
- **Upstream:** CoreLogic (operations to verify)
- **Upstream:** TensorBackend (execution layer)
- **Integration:** PatternAPI (optional verification of patterns)
- **Downstream:** User training pipelines

### Future Enhancements
- **Automated proof search:** Neural tactics for proof automation
- **Verified compilation:** Prove compilation strategies correct
- **Pattern verification:** Verify pattern language semantics
- **Certified neural predicates:** Export predicates with proof certificates

---

**References:**
- LeanDojo: https://leandojo.org
- Lean Copilot: 74.2% proof automation (vs 40.1% aesop)
- Harmonic AI: $100M raise for Lean 4-based verification
- TensorLogic Overview: `docs/TensorLogic-Overview.md` (lines 110-120)
- Strategic Intel: `docs/research/intel.md` (formal verification market opportunity)
