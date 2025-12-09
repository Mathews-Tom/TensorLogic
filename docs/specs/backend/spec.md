# TensorBackend Specification

**Component ID:** BACKEND
**Priority:** P0 (Critical - Foundation)
**Phase:** 1 (Core Operations)
**Source:** docs/TensorLogic-Overview.md, .sage/agent/system/architecture.md

## 1. Overview

### Purpose and Business Value
Provide minimal, protocol-based abstraction layer for tensor operations across MLX, NumPy, and future backends (PyTorch, JAX). Enables framework portability while maintaining zero-overhead access to backend-specific optimizations.

**Key Differentiator:** Following einops' philosophy of "abstract at operation level, not model level" with ~25-30 core operations instead of heavyweight compatibility layers.

### Success Metrics
- Protocol implementation for MLX backend passes 100% of test suite
- NumPy fallback backend provides bit-accurate reference implementation
- Zero-overhead abstraction: <1% performance penalty vs direct MLX calls
- All backends support lazy evaluation (critical for MLX)

### Target Users
- TensorLogic library developers implementing core logic operations
- Advanced users needing performance-critical custom operations
- Backend maintainers extending support to new frameworks

## 2. Functional Requirements

### FR-1: Protocol Definition
The system **shall** define a `TensorBackend` Protocol with the following core operations:

#### Tensor Creation & Manipulation
- `einsum(pattern: str, *tensors) -> Array`: Einstein summation with string pattern
- `zeros(shape: tuple[int, ...]) -> Array`: Create zero tensor
- `ones(shape: tuple[int, ...]) -> Array`: Create ones tensor
- `arange(start: int, stop: int) -> Array`: Create range tensor
- `reshape(array: Array, shape: tuple[int, ...]) -> Array`: Reshape operation

#### Logical & Mathematical Operations
- `step(x: Array) -> Array`: Heaviside step function (critical for boolean logic)
- `maximum(a: Array, b: Array) -> Array`: Element-wise max (for OR operation)
- `minimum(a: Array, b: Array) -> Array`: Element-wise min
- `multiply(a: Array, b: Array) -> Array`: Hadamard product (for AND operation)
- `add(a: Array, b: Array) -> Array`: Element-wise addition
- `subtract(a: Array, b: Array) -> Array`: Element-wise subtraction

#### Quantifier Operations
- `sum(array: Array, axis: int | tuple[int, ...] | None = None) -> Array`: Summation (for exists)
- `prod(array: Array, axis: int | tuple[int, ...] | None = None) -> Array`: Product (for forall)
- `any(array: Array, axis: int | tuple[int, ...] | None = None) -> Array`: Boolean any
- `all(array: Array, axis: int | tuple[int, ...] | None = None) -> Array`: Boolean all

#### Differentiation & Evaluation
- `grad(fn: Callable) -> Callable`: Create gradient function
- `eval(*arrays: Array) -> None`: Force evaluation of lazy tensors (MLX-specific but no-op for eager backends)
- `compile(fn: Callable) -> Callable`: JIT compilation hint

#### Utility Operations
- `where(condition: Array, x: Array, y: Array) -> Array`: Conditional selection
- `expand_dims(array: Array, axis: int) -> Array`: Add dimension
- `squeeze(array: Array, axis: int | None = None) -> Array`: Remove dimension
- `transpose(array: Array, axes: tuple[int, ...] | None = None) -> Array`: Transpose

**Total:** ~25 operations (expandable to 30 for additional optimizations)

### FR-2: MLX Backend Implementation
The system **shall** provide an `MLXBackend` class implementing `TensorBackend` protocol:

**User Story:** As a TensorLogic developer, I want native MLX support so that I can leverage unified memory and Metal GPU acceleration on Apple Silicon.

#### MLX-Specific Requirements
- Import `mlx.core as mx` for all operations
- Implement `eval()` to call `mx.eval(*arrays)` for lazy evaluation
- Map `step(x)` to `mx.where(x > 0, 1.0, 0.0)` (no native step in MLX)
- Support BF16 precision for 8B parameter model development
- Handle MLX's lazy evaluation in gradient computation

### FR-3: NumPy Fallback Backend
The system **shall** provide a `NumpyBackend` class for testing and reference:

**User Story:** As a TensorLogic developer, I want a NumPy reference backend so that I can verify correctness without GPU dependencies.

#### NumPy-Specific Requirements
- Import `numpy as np` for all operations
- Implement `eval()` as no-op (NumPy is eager)
- Map `step(x)` to `np.heaviside(x, 0.5)` (native support)
- Provide bit-accurate reference for validating MLX implementation

### FR-4: Backend Factory & Registration
The system **shall** provide a factory function for backend creation:

```python
def create_backend(name: str = "mlx") -> TensorBackend:
    """Create tensor backend by name.

    Args:
        name: Backend identifier ('mlx', 'numpy')

    Returns:
        Backend instance conforming to TensorBackend protocol

    Raises:
        ValueError: If backend name unknown or dependencies missing
    """
```

**Business Rule:** Default to MLX backend if available, fall back to NumPy if MLX import fails.

### FR-5: Runtime Protocol Validation
The system **shall** provide runtime validation using `@runtime_checkable`:

```python
def validate_backend(backend: Any) -> None:
    """Validate object implements TensorBackend protocol."""
    if not isinstance(backend, TensorBackend):
        raise TypeError(f"{type(backend)} doesn't implement TensorBackend")
```

## 3. Non-Functional Requirements

### NFR-1: Performance
- **Zero-overhead abstraction:** Protocol dispatch adds <1% overhead vs direct calls
- **Lazy evaluation support:** MLX operations build computation graph without immediate execution
- **Memory efficiency:** Support MLX unified memory (200 GB/s bandwidth on M1 Pro)
- **Batch size targets:** Support batch sizes 4-32 for M1 Pro development

### NFR-2: Type Safety
- **100% type hint coverage** on all public APIs
- **Protocol-based typing:** Use `typing.Protocol` with `@runtime_checkable`
- **Modern Python syntax:** Use `list[int] | None` not `Optional[List[int]]`
- **Static analysis:** Pass mypy strict mode without errors

### NFR-3: Portability
- **Backend isolation:** No MLX-specific code outside MLXBackend implementation
- **Import safety:** Gracefully handle missing backend dependencies
- **Platform support:** macOS (MLX primary), Linux (NumPy fallback), future CUDA via `mlx[cuda]`

### NFR-4: Maintainability
- **Minimal abstraction:** ~25-30 operations only, no bloat
- **Clear documentation:** Docstrings for all protocol methods with Args/Returns/Raises
- **Version compatibility:** MLX >=0.30.0, NumPy >=1.24.0, Python >=3.12

## 4. Features & Flows

### Feature 1: Backend Selection (Priority: P0)
**Flow:**
1. User calls `create_backend("mlx")`
2. Factory attempts MLX import
3. If successful, instantiate MLXBackend and validate
4. If failed, raise ValueError with suggestion to install mlx
5. Return validated backend

**Input:** Backend name string
**Output:** TensorBackend instance

### Feature 2: Lazy Evaluation Cycle (Priority: P0)
**Flow (MLX-specific):**
1. User calls `backend.einsum("ij,jk->ik", A, B)`
2. MLX builds computation graph (no execution)
3. User calls additional operations (graph extends)
4. User calls `backend.eval(result)` explicitly
5. MLX executes entire graph, returns result

**Key Constraint:** CoreLogic layer must call `eval()` at appropriate boundaries

### Feature 3: Gradient Computation (Priority: P0)
**Flow:**
1. User defines function using backend operations
2. User calls `grad_fn = backend.grad(fn)`
3. User calls `grad_fn(inputs)` to compute gradients
4. Backend returns gradient values

**MLX Detail:** `mx.grad` supports automatic differentiation through custom operations

## 5. Code Pattern Requirements

### Naming Conventions
- **Module names:** lowercase with underscores (`tensor_backend.py`)
- **Class names:** PascalCase (`MLXBackend`, `TensorBackend`)
- **Function names:** snake_case (`create_backend`, `validate_backend`)
- **Constants:** SCREAMING_SNAKE_CASE (`DEFAULT_BACKEND = "mlx"`)

### Type Safety Requirements
- **Type hint coverage:** 100% (all public APIs)
- **Union syntax:** Use `|` operator (`Array | None`)
- **Generics:** Use builtin generics (`list[str]`, `tuple[int, ...]`)
- **Null handling:** Explicit (`TensorBackend | None`)
- **Protocol usage:** `@runtime_checkable` decorator for runtime validation

### Testing Approach
- **Framework:** pytest with hypothesis for property-based tests
- **Coverage requirement:** ≥90% line coverage
- **Property tests:** Commutativity, associativity of operations
- **Cross-backend validation:** NumPy results validate MLX correctness
- **Test organization:** `tests/test_backends/{test_protocol.py, test_mlx.py, test_numpy.py}`

### Error Handling
- **Strategy:** Fail fast with explicit errors
- **Custom exceptions:** `BackendError(message, context, suggestion)`
- **Validation:** Input validation at protocol boundaries
- **No silent failures:** Raise on missing dependencies, invalid inputs

### Architecture Patterns
- **Module system:** Standard Python packages with `__init__.py`
- **Export style:** Named exports in `__init__.py` (`__all__ = ["TensorBackend", "create_backend"]`)
- **Import style:** Absolute imports (`from tensorlogic.backends import TensorBackend`)
- **Protocol pattern:** Structural typing, no inheritance required

### Documentation Standards
- **Docstring style:** Google-style with Args/Returns/Raises/Examples
- **Type stubs:** Generate `.pyi` files for IDE support
- **PEP 561:** Include `py.typed` marker file
- **Inline comments:** Only where logic non-obvious (e.g., MLX step function workaround)

## 6. Acceptance Criteria

### Definition of Done
- [ ] `TensorBackend` Protocol defined with ~25 core operations
- [ ] `MLXBackend` implementation passes all tests on M1 Pro
- [ ] `NumpyBackend` implementation provides bit-accurate reference
- [ ] `create_backend()` factory with graceful fallback
- [ ] Runtime validation with `@runtime_checkable`
- [ ] 100% type hints, passes mypy strict
- [ ] ≥90% test coverage with pytest + hypothesis
- [ ] Property-based tests for operation correctness
- [ ] Cross-backend validation (NumPy validates MLX)
- [ ] Documentation with examples for all public APIs
- [ ] `py.typed` marker file included

### Validation Approach
1. **Unit tests:** Each backend operation tested independently
2. **Property tests:** Mathematical properties (associativity, distributivity)
3. **Cross-validation:** MLX results match NumPy within floating-point tolerance
4. **Integration tests:** CoreLogic can switch backends without code changes
5. **Performance tests:** <1% overhead vs direct MLX calls
6. **Type checking:** `mypy --strict src/tensorlogic/backends` passes

## 7. Dependencies

### Technical Assumptions
- **Python version:** >=3.12 (for modern type hints)
- **MLX version:** >=0.30.0 (for einsum, grad, compile support)
- **NumPy version:** >=1.24.0 (for stable API)
- **Platform:** macOS with Apple Silicon for MLX (M1/M2/M3)

### External Integrations
- **MLX:** `import mlx.core as mx`
- **NumPy:** `import numpy as np`
- **No other external dependencies** (stdlib only for protocol/factory)

### Related Components
- **Downstream:** CoreLogic (depends on TensorBackend)
- **Downstream:** PatternAPI (indirect dependency via CoreLogic)
- **Downstream:** Compilation (uses backend for gradient computation)
- **Upstream:** None (foundational component)

### Future Extensions
- **PyTorch backend:** `create_backend("torch")` for broader ecosystem
- **JAX backend:** `create_backend("jax")` for research compatibility
- **MLX CUDA backend:** Via `mlx[cuda]` optional dependency
- **Custom operations:** Allow users to extend protocol with domain-specific ops

---

**References:**
- TensorLogic Overview: `docs/TensorLogic-Overview.md` (lines 69-81)
- Architecture: `.sage/agent/system/architecture.md`
- Pattern Template: `.sage/agent/examples/python/classes/protocol-pattern.md`
- Strategic Intelligence: `docs/research/intel.md` (MLX CUDA backend, einops philosophy)
