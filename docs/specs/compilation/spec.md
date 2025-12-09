# Compilation Specification

**Component ID:** COMP
**Priority:** P1 (High - Semantic Flexibility)
**Phase:** 3 (Compilation Strategies)
**Source:** docs/TensorLogic-Overview.md, cool-japan/tensorlogic analysis

## 1. Overview

### Purpose and Business Value
Provide multiple compilation strategies for logical patterns, supporting different semantic interpretations (differentiable, Boolean, fuzzy logic). Enables users to choose between strict deductive reasoning and soft probabilistic inference based on application needs.

**Key Innovation:** Support for 6 compilation strategies matching cool-japan's breadth while maintaining simplicity.

### Success Metrics
- Support 4+ compilation strategies (soft differentiable, hard Boolean, fuzzy variants)
- Gradient flow works for differentiable strategies
- Hard Boolean strategy produces zero hallucinations
- Performance: <10% overhead vs direct CoreLogic operations
- Users can switch strategies without code changes

### Target Users
- Researchers comparing semantic interpretations
- ML engineers training neural predicates (need gradients)
- Production deployments requiring verified boolean logic

## 2. Functional Requirements

### FR-1: Compilation Strategy Interface
The system **shall** define a `CompilationStrategy` protocol:

```python
class CompilationStrategy(Protocol):
    """Protocol for logical compilation strategies."""

    def compile_and(self, a: Array, b: Array) -> Array:
        """Compile AND operation."""

    def compile_or(self, a: Array, b: Array) -> Array:
        """Compile OR operation."""

    def compile_not(self, a: Array) -> Array:
        """Compile NOT operation."""

    def compile_implies(self, a: Array, b: Array) -> Array:
        """Compile implication."""

    def compile_exists(self, predicate: Array, axis: int) -> Array:
        """Compile existential quantifier."""

    def compile_forall(self, predicate: Array, axis: int) -> Array:
        """Compile universal quantifier."""

    @property
    def is_differentiable(self) -> bool:
        """Whether strategy supports gradient computation."""

    @property
    def name(self) -> str:
        """Strategy identifier."""
```

### FR-2: Soft Differentiable Strategy
**Mathematics:** Continuous operations without step function

```python
class SoftDifferentiableStrategy:
    """Soft probabilistic semantics (differentiable)."""

    def compile_and(self, a: Array, b: Array) -> Array:
        """Soft AND: a * b (product)"""
        return backend.multiply(a, b)

    def compile_or(self, a: Array, b: Array) -> Array:
        """Soft OR: a + b - a*b (probabilistic sum)"""
        return backend.add(a, b) - backend.multiply(a, b)

    def compile_exists(self, predicate: Array, axis: int) -> Array:
        """Soft exists: max(predicate, axis)"""
        return backend.max(predicate, axis=axis)

    def compile_forall(self, predicate: Array, axis: int) -> Array:
        """Soft forall: min(predicate, axis)"""
        return backend.min(predicate, axis=axis)

    is_differentiable = True
    name = "soft_differentiable"
```

### FR-3: Hard Boolean Strategy
**Mathematics:** Discrete boolean with step function

```python
class HardBooleanStrategy:
    """Exact boolean semantics (non-differentiable)."""

    def compile_and(self, a: Array, b: Array) -> Array:
        """Hard AND: step(a * b)"""
        return backend.step(backend.multiply(a, b))

    def compile_or(self, a: Array, b: Array) -> Array:
        """Hard OR: step(a + b)"""
        return backend.step(backend.add(a, b))

    def compile_exists(self, predicate: Array, axis: int) -> Array:
        """Hard exists: step(sum(predicate, axis))"""
        return backend.step(backend.sum(predicate, axis=axis))

    def compile_forall(self, predicate: Array, axis: int) -> Array:
        """Hard forall: step(prod(predicate, axis) - threshold)"""
        prod = backend.prod(predicate, axis=axis)
        return backend.step(prod - 0.99)  # Near 1.0 threshold

    is_differentiable = False
    name = "hard_boolean"
```

### FR-4: Fuzzy Logic Strategies
**Mathematics:** Fuzzy t-norms and t-conorms

```python
class GodelStrategy:
    """Gödel fuzzy semantics (min/max)."""

    def compile_and(self, a: Array, b: Array) -> Array:
        """Gödel AND: min(a, b)"""
        return backend.minimum(a, b)

    def compile_or(self, a: Array, b: Array) -> Array:
        """Gödel OR: max(a, b)"""
        return backend.maximum(a, b)

    is_differentiable = True  # Subgradients exist
    name = "godel"

class ProductStrategy:
    """Product fuzzy semantics."""

    def compile_and(self, a: Array, b: Array) -> Array:
        """Product AND: a * b"""
        return backend.multiply(a, b)

    def compile_or(self, a: Array, b: Array) -> Array:
        """Product OR: a + b - a*b"""
        return backend.add(a, b) - backend.multiply(a, b)

    is_differentiable = True
    name = "product"

class LukasiewiczStrategy:
    """Łukasiewicz fuzzy semantics (strict)."""

    def compile_and(self, a: Array, b: Array) -> Array:
        """Łukasiewicz AND: max(0, a + b - 1)"""
        return backend.maximum(backend.zeros_like(a),
                              backend.add(a, b) - 1.0)

    def compile_or(self, a: Array, b: Array) -> Array:
        """Łukasiewicz OR: min(1, a + b)"""
        return backend.minimum(backend.ones_like(a),
                              backend.add(a, b))

    is_differentiable = True
    name = "lukasiewicz"
```

### FR-5: Strategy Selection API
```python
def create_strategy(name: str = "soft_differentiable") -> CompilationStrategy:
    """Create compilation strategy by name.

    Args:
        name: Strategy identifier
            - "soft_differentiable": Continuous, gradient-friendly (default)
            - "hard_boolean": Discrete, exact boolean logic
            - "godel": Gödel fuzzy logic (min/max)
            - "product": Product fuzzy logic
            - "lukasiewicz": Łukasiewicz fuzzy logic

    Returns:
        Compilation strategy instance

    Raises:
        ValueError: Unknown strategy name
    """
```

### FR-6: Integration with PatternAPI
```python
def quantify(
    pattern: str,
    *,
    predicates: dict[str, Any],
    strategy: str | CompilationStrategy = "soft_differentiable",
    backend: TensorBackend | None = None,
) -> Array:
    """Execute pattern with specified compilation strategy.

    Args:
        pattern: Logical pattern string
        predicates: Named predicates
        strategy: Compilation strategy name or instance
        backend: Tensor backend

    Returns:
        Compiled and executed result
    """
```

## 3. Non-Functional Requirements

### NFR-1: Gradient Compatibility
- Differentiable strategies support `backend.grad()`
- Non-differentiable strategies raise clear errors when gradients requested
- Straight-through estimators for mixed training (future)

### NFR-2: Performance
- Strategy dispatch: O(1) lookup
- Compilation overhead: <10% vs direct operations
- Backend JIT compilation preserved

### NFR-3: Semantic Correctness
- Each strategy satisfies its semantic axioms
- Property-based tests for each strategy
- Cross-strategy comparison tests

## 4. Code Pattern Requirements

### Naming Conventions
- **Strategies:** PascalCase with "Strategy" suffix (`SoftDifferentiableStrategy`)
- **Methods:** `compile_<operation>` pattern
- **Properties:** `is_differentiable`, `name`

### Type Safety Requirements
- **Protocol:** `CompilationStrategy` defines interface
- **100% type hints** on all methods
- **Backend-agnostic:** Use `Array` type alias

### Testing Approach
- **Property tests:** Verify semantic axioms per strategy
- **Cross-strategy tests:** Compare results across strategies
- **Gradient tests:** Verify differentiable strategies work with `grad()`
- **Coverage:** ≥90%

## 5. Acceptance Criteria

### Definition of Done
- [ ] `CompilationStrategy` protocol defined
- [ ] 5 strategies implemented (soft, hard, godel, product, lukasiewicz)
- [ ] `create_strategy()` factory function
- [ ] Integration with `quantify()` API
- [ ] Property-based tests for each strategy
- [ ] Gradient tests for differentiable strategies
- [ ] 100% type hints, passes mypy strict
- [ ] ≥90% test coverage
- [ ] Documentation with semantic descriptions

## 6. Dependencies

### Technical Assumptions
- **CoreLogic:** Provides logical operations
- **TensorBackend:** Backend abstraction layer
- **PatternAPI:** Pattern execution framework

### Related Components
- **Upstream:** CoreLogic, TensorBackend
- **Upstream:** PatternAPI (uses compilation strategies)
- **Downstream:** User training pipelines

---

**References:**
- cool-japan/tensorlogic: 6 compilation strategies benchmark
- TensorLogic Overview: `docs/TensorLogic-Overview.md` (Phase 3)
- Strategic Intel: `docs/research/intel.md` (fuzzy logic variants)
