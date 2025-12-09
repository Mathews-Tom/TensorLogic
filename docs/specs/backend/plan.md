# TensorBackend Implementation Blueprint (PRP)

**Format:** Product Requirements Prompt (Context Engineering)
**Generated:** 2025-12-08
**Specification:** `docs/specs/backend/spec.md`
**Component ID:** BACKEND-001
**Priority:** P0 (Critical - Foundation)

---

## 📖 Context & Documentation

### Traceability Chain

**Strategic Assessment → Research Intelligence → Specification → This Plan**

1. **Strategic Assessment:** docs/TensorLogic-Overview.md
   - MLX-first strategy validated as viable
   - Backend abstraction pattern: einops philosophy (~25-30 operations)
   - Apple Silicon development with CUDA scaling path

2. **Research & Intelligence:** docs/research/intel.md
   - MLX CUDA backend added March 2025 (Apple-sponsored)
   - MLX 0.30.0+ provides comprehensive einsum, grad, compile support
   - Competitor analysis: cool-japan locked to custom SciRS2 backend (avoid this)

3. **Formal Specification:** docs/specs/backend/spec.md
   - Protocol-based abstraction (no inheritance)
   - MLX primary, NumPy fallback for testing
   - ~25 core operations, runtime validation

### Related Documentation

**System Context:**
- Architecture: `.sage/agent/system/architecture.md` - 4-layer architecture (Core, Backends, API, Verification)
- Tech Stack: `.sage/agent/system/tech-stack.md` - Python 3.12+, MLX 0.30.0+, pytest+hypothesis
- Patterns: `.sage/agent/system/patterns.md` - Protocol-based abstraction, modern type hints

**Code Examples:**
- `.sage/agent/examples/python/classes/protocol-pattern.md` - TensorBackend Protocol template
- `.sage/agent/examples/python/types/modern-type-hints.md` - Modern Python 3.12+ syntax

**Dependencies:**
- Downstream: CORE-001 (CoreLogic), API-001 (PatternAPI)
- No upstream dependencies (foundation component)

---

## 📊 Executive Summary

### Business Alignment

**Purpose:** Enable framework-portable tensor operations across MLX and NumPy backends, supporting Apple Silicon-first development with CUDA scaling.

**Value Proposition:**
- Zero vendor lock-in: Users can switch backends without code changes
- MLX-first strategy: Leverage unified memory and Metal GPU on M1/M2/M3
- Future-proof: CUDA scaling via `mlx[cuda]` when needed
- Developer experience: einops-style minimal abstraction (not heavyweight wrapper)

**Target Users:**
- TensorLogic library developers (internal - implementing CoreLogic)
- Advanced users needing performance-critical custom operations
- Backend maintainers extending to PyTorch/JAX

### Technical Approach

**Architecture Pattern:** Protocol-based structural typing (PEP 544)
- No inheritance hierarchy (avoid OOP tax)
- Duck typing with runtime validation (`@runtime_checkable`)
- Minimal surface area (~25 operations)

**Technology Stack:**
- **Primary:** MLX 0.30.0+ (Apple ML Research maintained, 22,800+ stars)
- **Fallback:** NumPy 1.24.0+ (reference implementation)
- **Type System:** Python 3.12+ modern type hints
- **Testing:** pytest + hypothesis for property-based tests

**Implementation Strategy:**
- Phase 1: Protocol definition + NumPy backend (Week 1)
- Phase 2: MLX backend implementation (Week 1-2)
- Phase 3: Cross-backend validation tests (Week 2)

### Key Success Metrics

**Service Level Objectives (SLOs):**
- **Zero-overhead abstraction:** <1% performance penalty vs direct MLX calls
- **Protocol compliance:** 100% of backends pass runtime validation
- **Test coverage:** ≥90% line coverage
- **Cross-validation:** MLX results match NumPy within FP32 tolerance

**Key Performance Indicators (KPIs):**
- **Lazy evaluation support:** MLX builds computation graph, evaluates on `eval()`
- **Memory efficiency:** Utilize MLX unified memory (200 GB/s bandwidth on M1 Pro)
- **Batch processing:** Support batch sizes 4-32 for M1 Pro development

---

## 💻 Code Examples & Patterns

### Repository Patterns

**1. Protocol Pattern:** `.sage/agent/examples/python/classes/protocol-pattern.md`

**Application:** Structural typing for TensorBackend interface

```python
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable
from collections.abc import Callable

@runtime_checkable
class TensorBackend(Protocol):
    """Protocol defining tensor backend interface."""

    def einsum(self, pattern: str, *tensors: Any) -> Any:
        """Execute Einstein summation."""
        ...

    def grad(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Create gradient function."""
        ...

    def eval(self, *arrays: Any) -> None:
        """Force evaluation of lazy tensors (MLX-specific, no-op for eager)."""
        ...

    def step(self, x: Any) -> Any:
        """Heaviside step function (critical for boolean logic)."""
        ...
```

**Adaptation Notes:**
- Add ~20 more operations (maximum, minimum, sum, prod, etc.)
- Keep Protocol lean (no implementation in Protocol itself)
- Use `@runtime_checkable` for `isinstance()` validation

**2. Modern Type Hints:** `.sage/agent/examples/python/types/modern-type-hints.md`

**Application:** 100% type hint coverage with Python 3.12+ syntax

```python
from __future__ import annotations
from typing import TypeAlias

# Use built-in generics (not typing.List)
TensorShape: TypeAlias = tuple[int, ...]

# Use | for unions (not Optional)
def create_backend(name: str = "mlx") -> TensorBackend:
    """Create backend by name."""
    ...

def validate_backend(backend: Any) -> None:
    """Validate protocol implementation."""
    if not isinstance(backend, TensorBackend):
        raise TypeError(f"{type(backend)} doesn't implement TensorBackend")
```

### Implementation Reference Examples

**From TensorLogic Overview (lines 69-81):**

```python
class TensorBackend(Protocol):
    def einsum(self, pattern: str, *tensors) -> Array
    def grad(self, fn: Callable) -> Callable
    def eval(self, *arrays) -> None  # Critical: MLX's lazy evaluation
    # ~25-30 core operations total
```

**Key Takeaways:**
- **Minimal abstraction:** Only essential operations, no bloat
- **Backend-specific behavior:** Document in docstrings (e.g., MLX lazy eval)
- **Performance-critical escape hatch:** Users can drop to native APIs

**Anti-patterns to Avoid (from cool-japan analysis):**
- ❌ Custom backend lock-in (SciRS2)
- ❌ Over-modularization (11 crates, high cognitive overhead)
- ❌ Missing GPU support despite "production-ready" claims

---

## 🔧 Technology Stack

### Recommended Stack

| Component | Technology | Version | Rationale |
|-----------|------------|---------|-----------|
| Runtime | Python | ≥3.12 | Modern type hints (| unions, built-in generics) |
| Primary Backend | MLX | ≥0.30.0 | Apple Silicon-first, CUDA scaling, 22,800+ stars |
| Fallback Backend | NumPy | ≥1.24.0 | Reference implementation, bit-accurate validation |
| Type Checking | mypy | ≥1.0 | Strict mode for Protocol validation |
| Testing | pytest | ≥7.0 | Unit and integration tests |
| Property Testing | hypothesis | ≥6.0 | Mathematical property validation |
| Linting | ruff | ≥0.1.0 | Fast Python linter |

**Key Technology Decisions:**

1. **MLX over PyTorch for primary backend:**
   - **Rationale:** Unified memory architecture (zero-copy CPU/GPU), Metal GPU acceleration, CUDA backend available
   - **Source:** Strategic intel - MLX CUDA backend added March 2025 (Apple-sponsored)
   - **Trade-off:** Smaller ecosystem than PyTorch, but growing rapidly

2. **Protocol-based abstraction (not inheritance):**
   - **Rationale:** einops philosophy - minimal abstraction, no OOP overhead
   - **Source:** TensorLogic Overview - "abstract at operation level, not model level"
   - **Trade-off:** No polymorphism, but cleaner interface

3. **NumPy as fallback (not PyTorch):**
   - **Rationale:** Simpler, bit-accurate reference for validation
   - **Source:** Specification NFR-3 (Portability)
   - **Trade-off:** No GPU for fallback, but sufficient for testing

### Alternatives Considered

**Option 2: PyTorch-first**
- **Pros:** Larger ecosystem (75% NeurIPS dominance), comprehensive operations
- **Cons:** Heavier dependency, MPS backend less mature than MLX Metal
- **Why Not Chosen:** MLX-first aligns with Apple Silicon strategy, PyTorch can be added later

**Option 3: JAX-first**
- **Pros:** Functional programming style, excellent for research
- **Cons:** Steeper learning curve, less mature Apple Silicon support
- **Why Not Chosen:** MLX provides better M1/M2/M3 experience

### Alignment with Existing System

**From `.sage/agent/system/tech-stack.md`:**
- **Consistent With:** Python 3.12+, pytest+hypothesis testing, mypy strict type checking
- **New Additions:** MLX framework (primary tensor backend)
- **Migration Considerations:** None (greenfield implementation)

---

## 🏗️ Architecture Design

### System Context

**From `.sage/agent/system/architecture.md`:**

Current architecture is 4-layer:
1. **Core:** Tensor logic primitives (depends on Backends)
2. **Backends:** This component (TensorBackend abstraction)
3. **API:** High-level patterns (depends on Core → Backends)
4. **Verification:** Lean 4 integration (depends on Core → Backends)

**Integration Points:**
- **Upstream (depends on Backend):** CoreLogic, PatternAPI, Compilation
- **Downstream (Backend depends on):** None (foundation layer)

### Component Architecture

**Architecture Pattern:** Protocol-based Interface Segregation

```
┌────────────────────────────────────────────┐
│         TensorBackend Protocol             │
│  (Structural typing, ~25 operations)       │
└────────────────────────────────────────────┘
                    ▲
                    │ implements
        ┌───────────┴───────────┐
        │                       │
┌───────┴────────┐    ┌────────┴────────┐
│  MLXBackend    │    │  NumpyBackend   │
│  (primary)     │    │  (fallback)     │
└────────────────┘    └─────────────────┘
        │                       │
        │                       │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │  create_backend()     │
        │  (factory function)   │
        └───────────────────────┘
```

**Rationale:**
- Protocol defines interface contract (no implementation)
- Concrete backends implement operations using native frameworks
- Factory provides dependency injection point

**Alignment:** Matches einops approach - minimal abstraction, no heavyweight wrappers

### Architecture Decisions

**Decision 1: Protocol vs Abstract Base Class**
- **Choice:** Protocol (structural typing)
- **Rationale:** No inheritance hierarchy, duck typing with runtime validation
- **Implementation:** `@runtime_checkable` decorator for `isinstance()` checks
- **Trade-offs:**
  - ✅ Pro: No OOP overhead, cleaner interface
  - ✅ Pro: Easier to add new backends (no inheritance)
  - ❌ Con: No default implementations (each backend implements all methods)

**Decision 2: Lazy vs Eager Evaluation**
- **Choice:** Support both (MLX lazy, NumPy eager)
- **Rationale:** MLX's lazy evaluation critical for performance, but NumPy eager for testing
- **Implementation:** `eval(*arrays)` method - MLX calls `mx.eval()`, NumPy no-op
- **Trade-offs:**
  - ✅ Pro: Best of both worlds (performance + simplicity)
  - ⚠️ Con: Users must call `eval()` explicitly for MLX (documented in pattern)

**Decision 3: Backend Discovery**
- **Choice:** Explicit factory, no auto-detection
- **Rationale:** Clear dependency injection, avoid import-time side effects
- **Implementation:** `create_backend(name="mlx")` with graceful fallback
- **Trade-offs:**
  - ✅ Pro: Testability (inject mock backends)
  - ✅ Pro: Explicit > implicit
  - ❌ Con: User must specify backend (acceptable)

### Component Breakdown

**1. TensorBackend Protocol** (`src/tensorlogic/backends/protocol.py`)
- **Purpose:** Define structural interface for all backends
- **Technology:** Python Protocol (PEP 544)
- **Pattern:** `.sage/agent/examples/python/classes/protocol-pattern.md`
- **Interfaces:** ~25 operation methods (einsum, grad, eval, step, etc.)
- **Dependencies:** None (pure Python typing)

**2. MLXBackend** (`src/tensorlogic/backends/mlx.py`)
- **Purpose:** Primary backend for Apple Silicon
- **Technology:** MLX 0.30.0+ (`import mlx.core as mx`)
- **Pattern:** Concrete implementation of TensorBackend Protocol
- **Interfaces:** Implements all Protocol methods using MLX operations
- **Dependencies:** `mlx>=0.30.0`

**3. NumpyBackend** (`src/tensorlogic/backends/numpy.py`)
- **Purpose:** Reference implementation and testing fallback
- **Technology:** NumPy 1.24.0+ (`import numpy as np`)
- **Pattern:** Concrete implementation of TensorBackend Protocol
- **Interfaces:** Implements all Protocol methods using NumPy operations
- **Dependencies:** `numpy>=1.24.0`

**4. Backend Factory** (`src/tensorlogic/backends/factory.py`)
- **Purpose:** Create backend instances by name
- **Technology:** Factory pattern with import-time validation
- **Pattern:** Dependency injection point
- **Interfaces:** `create_backend(name: str) -> TensorBackend`
- **Dependencies:** MLXBackend, NumpyBackend

### Data Flow & Boundaries

**Request Flow (User → Backend → Framework):**
1. User calls `backend = create_backend("mlx")`
2. Factory imports MLXBackend, instantiates, validates
3. User calls `result = backend.einsum("ij,jk->ik", A, B)`
4. Backend delegates to `mx.einsum()`
5. MLX builds computation graph (lazy)
6. User calls `backend.eval(result)` to force execution
7. MLX executes graph, returns result

**Component Boundaries:**
- **Public Interface:** TensorBackend Protocol methods
- **Internal Implementation:** Native framework calls (mx.*, np.*)
- **Cross-Component Contracts:** CoreLogic depends on TensorBackend Protocol

---

## ⚠️ Error Handling & Edge Cases

### Error Scenarios

**1. Missing Backend Dependency**
- **Cause:** User tries `create_backend("mlx")` but MLX not installed
- **Impact:** ImportError, application cannot start
- **Handling:** Catch ImportError, raise ValueError with helpful message
- **Recovery:** Suggest `pip install mlx>=0.30.0` or fall back to NumPy
- **User Experience:**
  ```python
  raise ValueError(
      "MLX backend not available. Install with: pip install mlx>=0.30.0",
      suggestion="Use create_backend('numpy') for CPU-only fallback"
  )
  ```

**2. Protocol Validation Failure**
- **Cause:** Backend implementation missing required methods
- **Impact:** TypeError at runtime when using backend
- **Handling:** Runtime validation in `create_backend()` using `isinstance()`
- **Recovery:** Fail fast with clear error
- **Pattern:** From `.sage/agent/examples/python/classes/protocol-pattern.md`

**3. Lazy Evaluation Not Triggered**
- **Cause:** User forgets to call `backend.eval()` for MLX
- **Impact:** Computation not executed, stale results
- **Handling:** Document requirement in all MLX-specific operations
- **Recovery:** CoreLogic layer must call `eval()` at appropriate boundaries
- **User Experience:** Docstrings emphasize "Critical: MLX's lazy evaluation"

### Edge Cases

**1. Empty Tensors**
- **Detection:** Shape contains 0 (e.g., `(0, 10)`)
- **Handling:** Pass through to backend (NumPy/MLX handle gracefully)
- **Testing:** Parametrized tests with empty shapes

**2. Mismatched Shapes in einsum**
- **Detection:** einsum pattern incompatible with tensor shapes
- **Handling:** Backend raises native error (mx.einsum raises, np.einsum raises)
- **Testing:** Verify error messages are passed through

**3. Unsupported Operations on Backend**
- **Detection:** Backend doesn't support specific operation (e.g., step function)
- **Handling:** Implement workaround (e.g., MLX step via `mx.where(x > 0, 1.0, 0.0)`)
- **Testing:** Cross-backend tests verify equivalent behavior

### Input Validation

**Validation Strategy:** Minimal - trust backend frameworks

- **Pattern validation:** Delegated to native einsum
- **Type validation:** Python type hints + mypy
- **Shape validation:** Delegated to native operations

**Rationale:** Zero-overhead abstraction - don't duplicate framework validation

### Graceful Degradation

**MLX unavailable → NumPy fallback:**
```python
def create_backend(name: str = "mlx") -> TensorBackend:
    try:
        if name == "mlx":
            from .mlx import MLXBackend
            return MLXBackend()
    except ImportError:
        logger.warning("MLX not available, falling back to NumPy")
        name = "numpy"

    if name == "numpy":
        from .numpy import NumpyBackend
        return NumpyBackend()

    raise ValueError(f"Unknown backend: {name}")
```

---

## 📚 Implementation Roadmap

### Phase 1: Protocol & NumPy (Week 1)

**Days 1-2: Protocol Definition**
- [ ] Define `TensorBackend` Protocol with ~25 operations
- [ ] Add `@runtime_checkable` decorator
- [ ] Write docstrings with Args/Returns/Raises (Google style)
- [ ] Add type hints (100% coverage)
- [ ] Create `py.typed` marker file

**Days 3-4: NumPy Backend**
- [ ] Implement `NumpyBackend` class
- [ ] Map all Protocol methods to NumPy operations
- [ ] Handle `step()` via `np.heaviside()`
- [ ] Implement `eval()` as no-op (NumPy is eager)
- [ ] Unit tests for each operation

**Day 5: Factory & Validation**
- [ ] Implement `create_backend()` factory
- [ ] Add runtime Protocol validation
- [ ] Error handling for missing dependencies
- [ ] Integration tests

### Phase 2: MLX Backend (Week 1-2)

**Days 6-8: MLX Implementation**
- [ ] Implement `MLXBackend` class
- [ ] Map Protocol methods to MLX operations (`mx.einsum`, `mx.grad`, etc.)
- [ ] Handle `step()` via `mx.where(x > 0, 1.0, 0.0)` workaround
- [ ] Implement `eval()` calling `mx.eval(*arrays)`
- [ ] Unit tests for MLX-specific behavior (lazy evaluation)

**Days 9-10: Cross-Backend Validation**
- [ ] Property-based tests (commutativity, associativity)
- [ ] Cross-validation: MLX results match NumPy within tolerance
- [ ] Performance tests: <1% overhead vs direct MLX calls
- [ ] Documentation with usage examples

### Phase 3: Hardening (Week 2)

**Days 11-12: Performance Optimization**
- [ ] Benchmark MLX vs direct calls
- [ ] Profile memory usage (unified memory utilization)
- [ ] Test batch sizes 4-32 on M1 Pro

**Days 13-14: Production Readiness**
- [ ] Achieve ≥90% test coverage
- [ ] Pass mypy strict mode
- [ ] Generate type stubs (.pyi files)
- [ ] Complete documentation

### Phase 4: Integration (Week 2+)

**Day 15+: CoreLogic Integration**
- [ ] Validate CoreLogic can use TensorBackend
- [ ] Test lazy evaluation boundaries
- [ ] Performance integration tests

---

## 🧪 Quality Assurance

### Testing Strategy

**Unit Tests (pytest):**
- Each Protocol method tested independently
- Each backend (MLX, NumPy) tested in isolation
- Edge cases: empty tensors, single element, large batches

**Property-Based Tests (hypothesis):**
```python
from hypothesis import given, strategies as st

@given(
    a=st.lists(st.floats(), min_size=1, max_size=100),
    b=st.lists(st.floats(), min_size=1, max_size=100),
)
def test_multiply_commutative(a, b):
    """Property: a * b == b * a"""
    backend = create_backend("numpy")
    min_len = min(len(a), len(b))
    result_ab = backend.multiply(a[:min_len], b[:min_len])
    result_ba = backend.multiply(b[:min_len], a[:min_len])
    assert np.allclose(result_ab, result_ba)
```

**Cross-Backend Validation:**
```python
@pytest.mark.parametrize("backend_name", ["mlx", "numpy"])
def test_einsum_cross_backend(backend_name):
    backend = create_backend(backend_name)
    A = [[1.0, 2.0], [3.0, 4.0]]
    B = [[5.0, 6.0], [7.0, 8.0]]
    result = backend.einsum("ij,jk->ik", A, B)
    backend.eval(result)  # No-op for NumPy, critical for MLX
    # Verify against known result
    expected = [[19.0, 22.0], [43.0, 50.0]]
    assert np.allclose(result, expected)
```

**Code Quality Gates:**
- **Type Checking:** `mypy --strict src/tensorlogic/backends` (0 errors)
- **Linting:** `ruff check src/tensorlogic/backends` (0 warnings)
- **Coverage:** `pytest --cov=tensorlogic.backends --cov-report=term-missing` (≥90%)

### Deployment Verification

- [ ] MLX backend loads on M1/M2/M3 Macs
- [ ] NumPy backend loads on all platforms (Linux, macOS, Windows)
- [ ] Protocol validation catches incomplete implementations
- [ ] Factory gracefully falls back to NumPy if MLX unavailable
- [ ] Performance: <1% overhead vs direct MLX calls
- [ ] Memory: Unified memory utilized on Apple Silicon

---

## 📚 References & Traceability

### Source Documentation

**Strategic Assessment:**
- docs/TensorLogic-Overview.md (lines 46-81)
  - MLX viability analysis
  - Backend abstraction pattern (einops philosophy)
  - Performance targets (M1 Pro batch sizes 4-32)

**Research & Intelligence:**
- docs/research/intel.md
  - MLX CUDA backend (March 2025, Apple-sponsored)
  - cool-japan analysis (avoid custom backend lock-in)
  - MLX 0.30.0+ capabilities (einsum, grad, compile)

**Specification:**
- docs/specs/backend/spec.md
  - Functional requirements (Protocol, MLX, NumPy, factory)
  - Non-functional requirements (performance, type safety, portability)
  - Acceptance criteria

### System Context

**Architecture & Patterns:**
- `.sage/agent/system/architecture.md` - 4-layer architecture context
- `.sage/agent/system/tech-stack.md` - Python 3.12+, MLX 0.30.0+, pytest+hypothesis
- `.sage/agent/system/patterns.md` - Protocol-based abstraction, fail-fast error handling

**Code Examples:**
- `.sage/agent/examples/python/classes/protocol-pattern.md` - TensorBackend Protocol template
- `.sage/agent/examples/python/types/modern-type-hints.md` - Modern type hints (| unions)

### Technology References

**MLX Framework:**
- GitHub: https://github.com/ml-explore/mlx (22,800+ stars)
- Docs: https://ml-explore.github.io/mlx/
- CUDA Backend: Added March 2025

**Python Protocols:**
- PEP 544: https://peps.python.org/pep-0544/
- Runtime Checkable: https://docs.python.org/3/library/typing.html#typing.runtime_checkable

### Related Components

**Dependents:**
- CORE-001 (CoreLogic): docs/specs/core/spec.md - Uses TensorBackend for all operations
- API-001 (PatternAPI): docs/specs/api/spec.md - Indirect via CoreLogic
- COMP-001 (Compilation): docs/specs/compilation/spec.md - Uses backend for gradient computation
- VERIF-001 (Verification): docs/specs/verification/spec.md - Verifies backend operations

---

**Plan Status:** Ready for `/sage.tasks` breakdown
**Estimated Duration:** 2 weeks (10 working days)
**Risk Level:** Low (well-defined scope, proven technologies)
