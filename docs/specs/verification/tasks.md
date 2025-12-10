# Verification Engine (VERIF-001) - SMART Task Breakdown

**Epic ID:** VERIF-001
**Priority:** P1 (High - Strategic Differentiator)
**Phase:** 5 (Lean 4 Bridge)
**Estimated Duration:** 4 weeks (34 story points)
**Generated:** 2025-12-10

---

## Epic Summary

Implement Lean 4 formal verification integration for TensorLogic operations and learned neural predicates. Provides mathematical guarantees through LeanDojo bridge, theorem definitions, and proof-guided training.

**Strategic Value:** First neural-symbolic framework with formal verification (no competitor has this). Harmonic AI $100M raise validates market demand.

**Key Deliverables:**
- LeanDojo Python bridge for Lean 4 communication
- Lean theorem library for core operations
- Predicate verification API
- Proof-guided training integration
- Verification result reporting with counterexamples

---

## Story Breakdown

### VERIF-002: Define TensorBackend Protocol (5 SP)
**Priority:** P0 (Blocks all verification work)
**Complexity:** Medium
**Estimated Duration:** 2-3 days

#### Description
Define core TensorBackend Protocol interface that verification system will use to execute tensor operations. This is the foundation for abstracting backend implementations.

#### Acceptance Criteria
- [ ] Protocol defined in `src/tensorlogic/verification/protocol.py`
- [ ] ~25-30 operation signatures (matching BACKEND-001 spec)
- [ ] Include `eval()` method for lazy evaluation (critical for MLX)
- [ ] Type hints: 100% coverage, passes mypy --strict
- [ ] Docstrings: Google style for all methods
- [ ] `py.typed` marker file included

#### Technical Requirements
```python
from __future__ import annotations
from typing import Protocol, Any

class TensorBackend(Protocol):
    """Minimal tensor operation protocol for verification."""

    def array(self, data: Any) -> Any:
        """Create array from Python data."""
        ...

    def eval(self, *arrays: Any) -> None:
        """Force evaluation of lazy tensors (critical for MLX)."""
        ...

    # ~23 more operations (see BACKEND-001 spec)
```

#### SMART Breakdown

**Task 2.1: Create protocol.py module**
- **Specific:** Create `src/tensorlogic/verification/protocol.py` with TensorBackend Protocol
- **Measurable:** File exists, imports without errors
- **Achievable:** Straightforward Protocol definition
- **Relevant:** Foundation for verification system
- **Time-Bound:** 2 hours
- **Command:** `touch src/tensorlogic/verification/protocol.py`

**Task 2.2: Define core array operations**
- **Specific:** Add array(), eval(), zeros(), ones(), full() methods to Protocol
- **Measurable:** 5 methods with type hints and docstrings
- **Achievable:** Standard tensor operations
- **Relevant:** Required for tensor creation
- **Time-Bound:** 3 hours

**Task 2.3: Define logical operations**
- **Specific:** Add logical_and(), logical_or(), logical_not(), where() methods
- **Measurable:** 4 methods with type hints
- **Achievable:** Map to tensor operations
- **Relevant:** Core verification operations
- **Time-Bound:** 2 hours

**Task 2.4: Define reduction operations**
- **Specific:** Add sum(), prod(), max(), min(), all(), any() with axis support
- **Measurable:** 6 methods with axis parameter
- **Achievable:** Standard reductions
- **Relevant:** Quantifier implementation
- **Time-Bound:** 3 hours

**Task 2.5: Define shape operations**
- **Specific:** Add reshape(), transpose(), expand_dims(), squeeze(), broadcast_to()
- **Measurable:** 5 methods for shape manipulation
- **Achievable:** Standard operations
- **Relevant:** Tensor transformations
- **Time-Bound:** 2 hours

**Task 2.6: Define comparison operations**
- **Specific:** Add equal(), not_equal(), less(), greater(), isclose()
- **Measurable:** 5 comparison methods
- **Achievable:** Standard comparisons
- **Relevant:** Verification assertions
- **Time-Bound:** 2 hours

**Task 2.7: Add type checking and documentation**
- **Specific:** Run mypy --strict, add Google-style docstrings
- **Measurable:** 0 mypy errors, all methods documented
- **Achievable:** Type hints already in place
- **Relevant:** Code quality gate
- **Time-Bound:** 2 hours
- **Command:** `uv run mypy src/tensorlogic/verification/protocol.py --strict`

**Task 2.8: Create py.typed marker**
- **Specific:** Add `src/tensorlogic/py.typed` for PEP 561 compliance
- **Measurable:** File exists
- **Achievable:** Touch empty file
- **Relevant:** Type checking in downstream packages
- **Time-Bound:** 5 minutes
- **Command:** `touch src/tensorlogic/py.typed`

#### Dependencies
- **Upstream:** None (foundation component)
- **Downstream:** VERIF-003, VERIF-004, VERIF-005

#### Files Modified
- `src/tensorlogic/verification/__init__.py` (create)
- `src/tensorlogic/verification/protocol.py` (create)
- `src/tensorlogic/py.typed` (create)

---

### VERIF-003: Implement NumPy Backend (5 SP)
**Priority:** P0 (Needed for testing verification)
**Complexity:** Low-Medium
**Estimated Duration:** 2-3 days

#### Description
Implement TensorBackend Protocol using NumPy as reference implementation. Used for testing and as fallback when MLX unavailable.

#### Acceptance Criteria
- [ ] NumpyBackend class implements TensorBackend Protocol
- [ ] All 25-30 operations implemented
- [ ] Test suite: ≥90% coverage
- [ ] Type hints: passes mypy --strict
- [ ] Documentation: NumPy-specific notes

#### Technical Requirements
```python
import numpy as np
from tensorlogic.verification.protocol import TensorBackend

class NumpyBackend:
    """NumPy implementation of TensorBackend."""

    def array(self, data: Any) -> np.ndarray:
        return np.array(data)

    def eval(self, *arrays: np.ndarray) -> None:
        pass  # NumPy is eager, no-op

    # Implement remaining operations
```

#### SMART Breakdown

**Task 3.1: Create numpy.py module**
- **Specific:** Create `src/tensorlogic/backends/numpy.py` with NumpyBackend class
- **Measurable:** File exists, imports NumPy
- **Achievable:** Straightforward file creation
- **Relevant:** Backend implementation
- **Time-Bound:** 1 hour

**Task 3.2: Implement array creation operations**
- **Specific:** Implement array(), zeros(), ones(), full(), eval()
- **Measurable:** 5 methods pass unit tests
- **Achievable:** Direct NumPy wrappers
- **Relevant:** Foundation operations
- **Time-Bound:** 2 hours

**Task 3.3: Implement logical operations**
- **Specific:** Implement logical_and(), logical_or(), logical_not(), where()
- **Measurable:** 4 methods with numpy logical ops
- **Achievable:** np.logical_and, np.logical_or, np.logical_not, np.where
- **Relevant:** Core verification operations
- **Time-Bound:** 2 hours

**Task 3.4: Implement reduction operations**
- **Specific:** Implement sum(), prod(), max(), min(), all(), any()
- **Measurable:** 6 methods with axis support
- **Achievable:** np.sum, np.prod, etc.
- **Relevant:** Quantifier support
- **Time-Bound:** 2 hours

**Task 3.5: Implement shape operations**
- **Specific:** Implement reshape(), transpose(), expand_dims(), squeeze(), broadcast_to()
- **Measurable:** 5 methods pass shape tests
- **Achievable:** Direct NumPy calls
- **Relevant:** Tensor transformations
- **Time-Bound:** 2 hours

**Task 3.6: Implement comparison operations**
- **Specific:** Implement equal(), not_equal(), less(), greater(), isclose()
- **Measurable:** 5 methods with np comparisons
- **Achievable:** np.equal, np.isclose, etc.
- **Relevant:** Verification assertions
- **Time-Bound:** 1 hour

**Task 3.7: Write unit tests**
- **Specific:** Create `tests/test_backends/test_numpy.py` with pytest tests
- **Measurable:** ≥90% coverage, all operations tested
- **Achievable:** Parametrized tests for each operation
- **Relevant:** Quality gate
- **Time-Bound:** 4 hours
- **Command:** `uv run pytest tests/test_backends/test_numpy.py --cov=tensorlogic.backends.numpy`

**Task 3.8: Type checking and documentation**
- **Specific:** Run mypy, add docstrings
- **Measurable:** 0 mypy errors, all methods documented
- **Achievable:** Type hints in place
- **Relevant:** Code quality
- **Time-Bound:** 2 hours
- **Command:** `uv run mypy src/tensorlogic/backends/numpy.py --strict`

#### Dependencies
- **Upstream:** VERIF-002 (TensorBackend Protocol)
- **Downstream:** VERIF-004 (factory pattern needs backend)

#### Files Modified
- `src/tensorlogic/backends/__init__.py` (create)
- `src/tensorlogic/backends/numpy.py` (create)
- `tests/test_backends/test_numpy.py` (create)

---

### VERIF-004: Factory Pattern & Validation (3 SP)
**Priority:** P0 (Needed before MLX backend)
**Complexity:** Low
**Estimated Duration:** 1-2 days

#### Description
Implement factory pattern for backend instantiation with graceful fallback (MLX → NumPy). Include cross-backend validation helpers.

#### Acceptance Criteria
- [ ] `get_backend()` factory function with fallback logic
- [ ] Cross-backend validation helpers (compare results)
- [ ] Tolerance configuration for FP32 comparisons
- [ ] Test suite for factory logic
- [ ] Documentation: Backend selection guide

#### Technical Requirements
```python
def get_backend(prefer: str = "mlx") -> TensorBackend:
    """Get tensor backend with graceful fallback.

    Args:
        prefer: Preferred backend ("mlx" or "numpy")

    Returns:
        TensorBackend instance (MLX if available, else NumPy)
    """
    if prefer == "mlx":
        try:
            from tensorlogic.backends.mlx import MLXBackend
            return MLXBackend()
        except ImportError:
            logger.warning("MLX not available, falling back to NumPy")

    from tensorlogic.backends.numpy import NumpyBackend
    return NumpyBackend()
```

#### SMART Breakdown

**Task 4.1: Create factory.py module**
- **Specific:** Create `src/tensorlogic/backends/factory.py` with get_backend()
- **Measurable:** Function returns backend instance
- **Achievable:** Simple factory pattern
- **Relevant:** Backend abstraction
- **Time-Bound:** 2 hours

**Task 4.2: Implement fallback logic**
- **Specific:** Try MLX import, catch ImportError, fallback to NumPy
- **Measurable:** Tests verify fallback behavior
- **Achievable:** Standard try/except pattern
- **Relevant:** Graceful degradation
- **Time-Bound:** 1 hour

**Task 4.3: Add backend validation helpers**
- **Specific:** Implement `validate_backends()` to compare MLX vs NumPy results
- **Measurable:** Function accepts two arrays, returns bool
- **Achievable:** Use numpy.allclose with tolerance
- **Relevant:** Cross-backend testing
- **Time-Bound:** 2 hours

**Task 4.4: Implement tolerance configuration**
- **Specific:** Add `ValidationConfig` dataclass with rtol/atol parameters
- **Measurable:** Config used in validation
- **Achievable:** Simple dataclass
- **Relevant:** FP32 precision handling
- **Time-Bound:** 1 hour

**Task 4.5: Write factory tests**
- **Specific:** Create `tests/test_backends/test_factory.py`
- **Measurable:** Test MLX preference, NumPy fallback, validation
- **Achievable:** Mock MLX import failure
- **Relevant:** Factory logic testing
- **Time-Bound:** 2 hours
- **Command:** `uv run pytest tests/test_backends/test_factory.py`

**Task 4.6: Documentation**
- **Specific:** Add docstrings and user guide for backend selection
- **Measurable:** All functions documented
- **Achievable:** Google-style docstrings
- **Relevant:** User-facing API
- **Time-Bound:** 1 hour

#### Dependencies
- **Upstream:** VERIF-003 (NumPy backend)
- **Downstream:** VERIF-005 (MLX backend uses factory)

#### Files Modified
- `src/tensorlogic/backends/factory.py` (create)
- `src/tensorlogic/backends/__init__.py` (update exports)
- `tests/test_backends/test_factory.py` (create)

---

### VERIF-005: Implement MLX Backend (8 SP)
**Priority:** P1 (Primary backend)
**Complexity:** High
**Estimated Duration:** 3-4 days

#### Description
Implement TensorBackend Protocol using MLX (Apple Silicon optimized). Handles lazy evaluation, unified memory, and Metal GPU acceleration.

#### Acceptance Criteria
- [ ] MLXBackend class implements TensorBackend Protocol
- [ ] All operations support lazy evaluation with explicit eval()
- [ ] Test suite: ≥90% coverage
- [ ] Cross-backend validation: matches NumPy within tolerance
- [ ] Performance: <1% overhead vs direct MLX calls
- [ ] Documentation: MLX-specific notes (lazy eval, unified memory)

#### Technical Requirements
```python
import mlx.core as mx
from tensorlogic.verification.protocol import TensorBackend

class MLXBackend:
    """MLX implementation of TensorBackend."""

    def array(self, data: Any) -> mx.array:
        return mx.array(data)

    def eval(self, *arrays: mx.array) -> None:
        """Force evaluation of lazy MLX tensors."""
        mx.eval(*arrays)

    # Implement remaining operations
```

#### SMART Breakdown

**Task 5.1: Create mlx.py module**
- **Specific:** Create `src/tensorlogic/backends/mlx.py` with MLXBackend class
- **Measurable:** File exists, imports mlx.core
- **Achievable:** Standard file creation
- **Relevant:** Primary backend
- **Time-Bound:** 1 hour

**Task 5.2: Implement array creation operations**
- **Specific:** Implement array(), zeros(), ones(), full() with mx.array, mx.zeros, etc.
- **Measurable:** 4 methods return mx.array
- **Achievable:** Direct MLX wrappers
- **Relevant:** Foundation operations
- **Time-Bound:** 2 hours

**Task 5.3: Implement eval() for lazy evaluation**
- **Specific:** Implement eval() calling mx.eval(*arrays)
- **Measurable:** Test verifies lazy tensors evaluated
- **Achievable:** Single MLX call
- **Relevant:** Critical for MLX correctness
- **Time-Bound:** 1 hour

**Task 5.4: Implement logical operations**
- **Specific:** Implement logical_and(), logical_or(), logical_not(), where()
- **Measurable:** 4 methods with MLX logical ops
- **Achievable:** mx.logical_and, mx.logical_or, etc.
- **Relevant:** Core verification operations
- **Time-Bound:** 2 hours

**Task 5.5: Implement reduction operations**
- **Specific:** Implement sum(), prod(), max(), min(), all(), any()
- **Measurable:** 6 methods with axis support
- **Achievable:** mx.sum, mx.prod, etc.
- **Relevant:** Quantifier support
- **Time-Bound:** 3 hours

**Task 5.6: Implement shape operations**
- **Specific:** Implement reshape(), transpose(), expand_dims(), squeeze(), broadcast_to()
- **Measurable:** 5 methods pass shape tests
- **Achievable:** mx.reshape, mx.transpose, etc.
- **Relevant:** Tensor transformations
- **Time-Bound:** 3 hours

**Task 5.7: Implement comparison operations**
- **Specific:** Implement equal(), not_equal(), less(), greater(), isclose()
- **Measurable:** 5 methods with MLX comparisons
- **Achievable:** mx.equal, mx.isclose, etc.
- **Relevant:** Verification assertions
- **Time-Bound:** 2 hours

**Task 5.8: Write unit tests**
- **Specific:** Create `tests/test_backends/test_mlx.py` with pytest tests
- **Measurable:** ≥90% coverage, all operations tested
- **Achievable:** Parametrized tests for each operation
- **Relevant:** Quality gate
- **Time-Bound:** 4 hours
- **Command:** `uv run pytest tests/test_backends/test_mlx.py --cov=tensorlogic.backends.mlx`

**Task 5.9: Cross-backend validation tests**
- **Specific:** Compare MLX vs NumPy results with tolerance
- **Measurable:** All operations match within FP32 precision
- **Achievable:** Use validation helpers from VERIF-004
- **Relevant:** Correctness verification
- **Time-Bound:** 3 hours

**Task 5.10: Performance benchmarks**
- **Specific:** Measure overhead vs direct MLX calls
- **Measurable:** <1% overhead target
- **Achievable:** Microbenchmarks with timeit
- **Relevant:** Performance validation
- **Time-Bound:** 2 hours

**Task 5.11: Type checking and documentation**
- **Specific:** Run mypy, add MLX-specific notes (lazy eval, unified memory)
- **Measurable:** 0 mypy errors, all methods documented
- **Achievable:** Type hints in place
- **Relevant:** Code quality
- **Time-Bound:** 2 hours
- **Command:** `uv run mypy src/tensorlogic/backends/mlx.py --strict`

#### Dependencies
- **Upstream:** VERIF-002 (TensorBackend Protocol), VERIF-004 (factory pattern)
- **Downstream:** VERIF-006 (cross-backend validation)

#### Files Modified
- `src/tensorlogic/backends/mlx.py` (create)
- `tests/test_backends/test_mlx.py` (create)

---

### VERIF-006: Cross-Backend Validation (5 SP)
**Priority:** P1 (Quality gate)
**Complexity:** Medium
**Estimated Duration:** 2-3 days

#### Description
Comprehensive cross-backend validation ensuring MLX and NumPy implementations produce equivalent results within FP32 tolerance. Property-based testing with hypothesis.

#### Acceptance Criteria
- [ ] Property tests: All operations match across backends
- [ ] Tolerance validation: FP32 precision (rtol=1e-5, atol=1e-8)
- [ ] Edge case testing: NaN, inf, empty arrays, large dimensions
- [ ] Test suite: ≥90% coverage
- [ ] Documentation: Validation methodology

#### Technical Requirements
```python
from hypothesis import given, strategies as st
import numpy as np
import mlx.core as mx

@given(
    a=st.lists(st.floats(min_value=-100, max_value=100), min_size=1, max_size=100),
    b=st.lists(st.floats(min_value=-100, max_value=100), min_size=1, max_size=100),
)
def test_logical_and_equivalence(a, b):
    """Property: MLX and NumPy produce same logical_and results."""
    numpy_backend = NumpyBackend()
    mlx_backend = MLXBackend()

    np_result = numpy_backend.logical_and(np.array(a), np.array(b))
    mlx_result = mlx_backend.logical_and(mx.array(a), mx.array(b))
    mlx_backend.eval(mlx_result)

    assert validate_backends(np_result, np.array(mlx_result))
```

#### SMART Breakdown

**Task 6.1: Create validation test module**
- **Specific:** Create `tests/test_backends/test_cross_validation.py`
- **Measurable:** File exists with pytest structure
- **Achievable:** Standard test file creation
- **Relevant:** Cross-backend testing
- **Time-Bound:** 1 hour

**Task 6.2: Property tests for logical operations**
- **Specific:** Write hypothesis tests for logical_and, logical_or, logical_not
- **Measurable:** 3 property tests with 100+ generated cases each
- **Achievable:** Use hypothesis strategies
- **Relevant:** Core operations validation
- **Time-Bound:** 3 hours

**Task 6.3: Property tests for reductions**
- **Specific:** Write tests for sum, prod, max, min, all, any
- **Measurable:** 6 property tests with axis variations
- **Achievable:** Parametrize over axis values
- **Relevant:** Quantifier operations
- **Time-Bound:** 4 hours

**Task 6.4: Property tests for shape operations**
- **Specific:** Write tests for reshape, transpose, expand_dims, squeeze, broadcast_to
- **Measurable:** 5 property tests
- **Achievable:** Generate random shapes
- **Relevant:** Tensor transformations
- **Time-Bound:** 3 hours

**Task 6.5: Edge case testing**
- **Specific:** Test NaN, inf, -inf, empty arrays, large dimensions (10000+)
- **Measurable:** 10+ edge case tests pass
- **Achievable:** Explicit test cases
- **Relevant:** Robustness validation
- **Time-Bound:** 3 hours

**Task 6.6: Tolerance validation tests**
- **Specific:** Verify FP32 precision (rtol=1e-5, atol=1e-8) sufficient
- **Measurable:** All comparisons within tolerance
- **Achievable:** Use numpy.allclose
- **Relevant:** Correctness threshold
- **Time-Bound:** 2 hours

**Task 6.7: Run full validation suite**
- **Specific:** Execute all cross-backend tests
- **Measurable:** ≥90% coverage, all tests pass
- **Achievable:** Tests written incrementally
- **Relevant:** Quality gate
- **Time-Bound:** 1 hour
- **Command:** `uv run pytest tests/test_backends/test_cross_validation.py --cov`

**Task 6.8: Documentation**
- **Specific:** Document validation methodology and tolerance rationale
- **Measurable:** Markdown doc with examples
- **Achievable:** Describe testing approach
- **Relevant:** Maintainability
- **Time-Bound:** 2 hours

#### Dependencies
- **Upstream:** VERIF-005 (MLX backend)
- **Downstream:** VERIF-007 (performance validation)

#### Files Modified
- `tests/test_backends/test_cross_validation.py` (create)
- `docs/verification/validation-methodology.md` (create)

---

### VERIF-007: Performance Validation & Optimization (5 SP)
**Priority:** P1 (Performance gate)
**Complexity:** Medium-High
**Estimated Duration:** 2-3 days

#### Description
Validate performance targets and optimize critical paths. Ensure backend abstraction overhead <1% vs direct MLX/NumPy calls.

#### Acceptance Criteria
- [ ] Microbenchmarks: <1% overhead vs direct calls
- [ ] Batch operation benchmarks: Various sizes (1, 10, 100, 1000)
- [ ] Lazy evaluation validation: MLX eval() called appropriately
- [ ] Profiling: No unnecessary copies or allocations
- [ ] Documentation: Performance characteristics

#### Technical Requirements
```python
import timeit
import mlx.core as mx

def benchmark_overhead():
    """Measure backend abstraction overhead."""
    backend = MLXBackend()

    # Direct MLX call
    direct_time = timeit.timeit(
        lambda: mx.sum(mx.array([1, 2, 3, 4, 5])),
        number=10000
    )

    # Backend call
    backend_time = timeit.timeit(
        lambda: backend.sum(backend.array([1, 2, 3, 4, 5])),
        number=10000
    )

    overhead = (backend_time - direct_time) / direct_time
    assert overhead < 0.01  # <1% overhead
```

#### SMART Breakdown

**Task 7.1: Create benchmark module**
- **Specific:** Create `tests/benchmarks/test_backend_performance.py`
- **Measurable:** File exists with pytest-benchmark structure
- **Achievable:** Standard benchmark file
- **Relevant:** Performance validation
- **Time-Bound:** 1 hour

**Task 7.2: Microbenchmarks for array operations**
- **Specific:** Benchmark array(), zeros(), ones() vs direct MLX/NumPy
- **Measurable:** <1% overhead for each operation
- **Achievable:** Use timeit with 10000 iterations
- **Relevant:** Creation overhead validation
- **Time-Bound:** 2 hours

**Task 7.3: Microbenchmarks for logical operations**
- **Specific:** Benchmark logical_and, logical_or, logical_not
- **Measurable:** <1% overhead
- **Achievable:** Timeit comparisons
- **Relevant:** Core operation performance
- **Time-Bound:** 2 hours

**Task 7.4: Batch operation benchmarks**
- **Specific:** Test batch sizes: 1, 10, 100, 1000, 10000 elements
- **Measurable:** Overhead remains <1% across sizes
- **Achievable:** Parametrize over sizes
- **Relevant:** Scalability validation
- **Time-Bound:** 3 hours

**Task 7.5: Lazy evaluation validation**
- **Specific:** Verify MLX eval() called only when needed
- **Measurable:** Profiling shows no premature evaluation
- **Achievable:** Add eval counter to backend
- **Relevant:** MLX correctness
- **Time-Bound:** 2 hours

**Task 7.6: Memory profiling**
- **Specific:** Profile for unnecessary copies/allocations
- **Measurable:** No redundant array copies
- **Achievable:** Use memory_profiler
- **Relevant:** Efficiency validation
- **Time-Bound:** 3 hours

**Task 7.7: Optimization pass**
- **Specific:** Optimize any operations >1% overhead
- **Measurable:** All operations meet target
- **Achievable:** Profile-guided optimization
- **Relevant:** Performance gate
- **Time-Bound:** 4 hours

**Task 7.8: Documentation**
- **Specific:** Document performance characteristics and optimization notes
- **Measurable:** Markdown doc with benchmark results
- **Achievable:** Summarize findings
- **Relevant:** User guidance
- **Time-Bound:** 2 hours

#### Dependencies
- **Upstream:** VERIF-006 (validation complete)
- **Downstream:** VERIF-008 (production readiness)

#### Files Modified
- `tests/benchmarks/test_backend_performance.py` (create)
- `docs/verification/performance-characteristics.md` (create)

---

### VERIF-008: Production Readiness (3 SP)
**Priority:** P0 (Release blocker)
**Complexity:** Low-Medium
**Estimated Duration:** 1-2 days

#### Description
Final production readiness checks: documentation, type coverage, test coverage, error messages, installation guide.

#### Acceptance Criteria
- [ ] Test coverage: ≥90% line coverage
- [ ] Type coverage: 100% (mypy --strict passes)
- [ ] Documentation: Complete API reference
- [ ] Error messages: All error paths have clear messages
- [ ] Installation guide: Lean 4 + LeanDojo setup
- [ ] CI/CD: All checks pass

#### SMART Breakdown

**Task 8.1: Measure test coverage**
- **Specific:** Run pytest with coverage report
- **Measurable:** ≥90% line coverage
- **Achievable:** Write tests for uncovered lines
- **Relevant:** Quality gate
- **Time-Bound:** 3 hours
- **Command:** `uv run pytest --cov=tensorlogic.verification --cov-report=html`

**Task 8.2: Fill coverage gaps**
- **Specific:** Write tests for uncovered lines
- **Measurable:** Reach ≥90% coverage
- **Achievable:** Targeted test writing
- **Relevant:** Completeness
- **Time-Bound:** 4 hours

**Task 8.3: Type checking validation**
- **Specific:** Run mypy --strict on all modules
- **Measurable:** 0 type errors
- **Achievable:** Fix any remaining issues
- **Relevant:** Type safety
- **Time-Bound:** 2 hours
- **Command:** `uv run mypy src/tensorlogic/verification --strict`

**Task 8.4: Error message audit**
- **Specific:** Review all error paths for clarity
- **Measurable:** All exceptions have context and suggestions
- **Achievable:** Add enhanced error messages
- **Relevant:** User experience
- **Time-Bound:** 3 hours

**Task 8.5: API reference documentation**
- **Specific:** Generate Sphinx autodoc for all public APIs
- **Measurable:** Complete API docs published
- **Achievable:** Docstrings already in place
- **Relevant:** User documentation
- **Time-Bound:** 2 hours

**Task 8.6: Installation guide**
- **Specific:** Write guide for Lean 4 + LeanDojo setup
- **Measurable:** Guide tested on clean macOS system
- **Achievable:** Document installation steps
- **Relevant:** User onboarding
- **Time-Bound:** 2 hours

**Task 8.7: CI/CD validation**
- **Specific:** Verify all checks pass (tests, types, linting)
- **Measurable:** Green CI pipeline
- **Achievable:** Fix any failures
- **Relevant:** Release readiness
- **Time-Bound:** 1 hour

#### Dependencies
- **Upstream:** VERIF-007 (performance validation)
- **Downstream:** None (release ready)

#### Files Modified
- `tests/` (coverage improvements)
- `docs/verification/installation.md` (create)
- `docs/verification/api-reference.md` (create)

---

## Summary Statistics

**Total Story Points:** 34
**Total Stories:** 7
**Estimated Duration:** 4 weeks (20 working days)

**Complexity Breakdown:**
- Low: 1 story (VERIF-004)
- Low-Medium: 2 stories (VERIF-003, VERIF-008)
- Medium: 2 stories (VERIF-002, VERIF-006)
- Medium-High: 1 story (VERIF-007)
- High: 1 story (VERIF-005)

**Priority Breakdown:**
- P0 (Blocking): 4 stories (VERIF-002, VERIF-003, VERIF-004, VERIF-008)
- P1 (High): 3 stories (VERIF-005, VERIF-006, VERIF-007)

**Critical Path:**
VERIF-002 → VERIF-003 → VERIF-004 → VERIF-005 → VERIF-006 → VERIF-007 → VERIF-008

---

## Implementation Notes

### Developer Guidelines

**Before Starting:**
1. Review specification: `docs/specs/verification/spec.md`
2. Review implementation plan: `docs/specs/verification/plan.md`
3. Check architecture: `.sage/agent/system/architecture.md`
4. Review code patterns: `.sage/agent/examples/python/`

**During Implementation:**
1. Follow SMART task breakdown for each story
2. Run tests after each task: `uv run pytest`
3. Type check after each task: `uv run mypy src/tensorlogic/verification --strict`
4. Update documentation as you go
5. Commit frequently with conventional commit messages

**Quality Gates:**
- All tests pass: `uv run pytest`
- Type checking: `uv run mypy --strict` (0 errors)
- Linting: `uv run ruff check .` (0 warnings)
- Coverage: ≥90% (`uv run pytest --cov`)
- Performance: <1% overhead (microbenchmarks)

### Testing Strategy

**Unit Tests:**
- Test each backend operation independently
- Parametrize over input shapes and types
- Test error cases (invalid inputs)

**Integration Tests:**
- Cross-backend validation (MLX vs NumPy)
- Factory pattern fallback logic
- End-to-end workflows

**Property Tests (hypothesis):**
- Generate random inputs
- Verify operations match across backends
- Test mathematical properties (commutativity, associativity)

**Performance Tests:**
- Microbenchmarks for overhead validation
- Batch operation scaling
- Memory profiling

### Common Pitfalls

**MLX-Specific:**
- ⚠️ Remember to call `eval()` on lazy tensors
- ⚠️ MLX uses unified memory (no explicit device management)
- ⚠️ Some operations may have different broadcasting rules

**NumPy-Specific:**
- ⚠️ NumPy is eager (eval() is no-op)
- ⚠️ Type conversions may differ from MLX

**Cross-Backend:**
- ⚠️ FP32 precision requires tolerance in comparisons
- ⚠️ NaN handling may differ between backends
- ⚠️ Empty array edge cases need explicit tests

---

## References

**Specifications:**
- Verification Spec: `docs/specs/verification/spec.md`
- Implementation Plan: `docs/specs/verification/plan.md`
- Architecture: `.sage/agent/system/architecture.md`

**Dependencies:**
- Backend Epic: `docs/specs/backend/tasks.md` (BACKEND-001)
- Core Operations: (CORE-001, future epic)
- API Layer: (API-001, future epic)

**External Resources:**
- Lean 4: https://lean-lang.org/
- LeanDojo: https://leandojo.org/
- MLX: https://ml-explore.github.io/mlx/
- NumPy: https://numpy.org/doc/

**Project Context:**
- Vision: `docs/TensorLogic-Overview.md`
- Tech Stack: `.sage/agent/system/tech-stack.md`
- Code Patterns: `.sage/agent/examples/python/`
