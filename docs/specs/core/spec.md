# CoreLogic Specification

**Component ID:** CORE
**Priority:** P0 (Critical - Core Primitives)
**Phase:** 1 (Core Operations)
**Source:** docs/TensorLogic-Overview.md, arXiv:2510.12269 (Domingos paper)

## 1. Overview

### Purpose and Business Value
Implement the foundational tensor-to-logic primitives that realize Pedro Domingos' mathematical equivalence between logical rules and Einstein summation. Provides the computational core for neural-symbolic reasoning with temperature-controlled interpolation between deductive (T=0) and analogical (T>0) inference.

**Key Innovation:** First production implementation of Tensor Logic theory, enabling verified neural-symbolic AI with controllable reasoning modes.

### Success Metrics
- Logical operations (AND, OR, NOT, IMPLIES) map correctly to tensor operations
- Quantifiers (EXISTS, FOR ALL) produce mathematically sound results
- Temperature parameter T=0 produces zero hallucinations (purely deductive)
- Performance: 12 tensor equations implement complete transformer architecture
- Correctness: Pass property-based tests for logical axioms (associativity, distributivity)

### Target Users
- TensorLogic library developers building PatternAPI layer
- Researchers implementing neural-symbolic models
- ML engineers training neural predicates with logical constraints

## 2. Functional Requirements

### FR-1: Sparse Boolean Tensor Representation
The system **shall** represent logical relations as sparse Boolean tensors:

**User Story:** As a neural-symbolic developer, I want to represent logical relations as tensors so that I can apply differentiable operations to symbolic knowledge.

#### Requirements
- Boolean values: 0.0 (False), 1.0 (True)
- Sparse representation for efficiency (most relations are sparse)
- Support n-ary relations: unary `P(x)`, binary `R(x,y)`, ternary `T(x,y,z)`
- Shape inference from predicate arity and domain sizes

**Mathematical Foundation:**
```
Datalog: Aunt(x,z) ← Sister(x,y), Parent(y,z)
Tensor:  Aunt[x,z] = step(Sister[x,y] ⊙ Parent[y,z])  # sum over y implicit
```

### FR-2: Logical Operations as Tensor Operations
The system **shall** implement logical operations using tensor primitives:

#### AND Operation (Logical Conjunction)
```python
def logical_and(a: Array, b: Array, *, backend: TensorBackend) -> Array:
    """Logical AND via Hadamard product.

    Mathematical: a ∧ b = a ⊙ b (element-wise multiply)

    Args:
        a: Boolean tensor (values in {0.0, 1.0})
        b: Boolean tensor (same shape as a)
        backend: Tensor backend for operations

    Returns:
        Boolean tensor: 1.0 where both true, 0.0 otherwise
    """
```

#### OR Operation (Logical Disjunction)
```python
def logical_or(a: Array, b: Array, *, backend: TensorBackend) -> Array:
    """Logical OR via element-wise maximum.

    Mathematical: a ∨ b = max(a, b)

    Args:
        a: Boolean tensor
        b: Boolean tensor (same shape as a)
        backend: Tensor backend

    Returns:
        Boolean tensor: 1.0 where at least one true, 0.0 otherwise
    """
```

#### NOT Operation (Logical Negation)
```python
def logical_not(a: Array, *, backend: TensorBackend) -> Array:
    """Logical NOT via complement.

    Mathematical: ¬a = 1 - a

    Args:
        a: Boolean tensor
        backend: Tensor backend

    Returns:
        Boolean tensor: 1.0 where input false, 0.0 otherwise
    """
```

#### IMPLIES Operation (Logical Implication)
```python
def logical_implies(a: Array, b: Array, *, backend: TensorBackend) -> Array:
    """Logical implication via max(1-a, b).

    Mathematical: a → b = ¬a ∨ b = max(1-a, b)

    Args:
        a: Boolean tensor (antecedent)
        b: Boolean tensor (consequent, same shape)
        backend: Tensor backend

    Returns:
        Boolean tensor: 0.0 only where a true and b false
    """
```

**Property Requirements:**
- Associativity: `(a ∧ b) ∧ c = a ∧ (b ∧ c)`
- Commutativity: `a ∧ b = b ∧ a`, `a ∨ b = b ∨ a`
- Distributivity: `a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)`
- De Morgan's laws: `¬(a ∧ b) = ¬a ∨ ¬b`

### FR-3: Quantifier Operations
The system **shall** implement existential and universal quantifiers:

#### Existential Quantification (∃)
```python
def exists(
    predicate: Array,
    axis: int | tuple[int, ...],
    *,
    backend: TensorBackend,
) -> Array:
    """Existential quantification via summation + step.

    Mathematical: ∃x.P(x) = step(Σ_x P(x))

    Args:
        predicate: Boolean tensor over domain
        axis: Axis/axes to quantify over
        backend: Tensor backend

    Returns:
        Boolean tensor: 1.0 if at least one true along axis
    """
```

#### Universal Quantification (∀)
```python
def forall(
    predicate: Array,
    axis: int | tuple[int, ...],
    *,
    backend: TensorBackend,
) -> Array:
    """Universal quantification via product + step.

    Mathematical: ∀x.P(x) = step(Π_x P(x) - threshold)

    Args:
        predicate: Boolean tensor over domain
        axis: Axis/axes to quantify over
        backend: Tensor backend

    Returns:
        Boolean tensor: 1.0 if all true along axis
    """
```

**Alternative Formulation (Soft):**
- Soft exists: `max_x P(x)` (no step, differentiable)
- Soft forall: `min_x P(x)` (no step, differentiable)

### FR-4: Step Function (Heaviside)
The system **shall** provide step function for boolean conversion:

```python
def step(x: Array, *, backend: TensorBackend) -> Array:
    """Heaviside step function.

    Mathematical: step(x) = 1.0 if x > 0, else 0.0

    Args:
        x: Input tensor (any real values)
        backend: Tensor backend

    Returns:
        Boolean tensor {0.0, 1.0}

    Note:
        Critical for converting continuous values to discrete boolean.
        Used in quantifiers and rule application.
    """
```

### FR-5: Temperature-Controlled Reasoning
The system **shall** support temperature parameter for reasoning control:

```python
def temperature_scaled_operation(
    operation: Callable,
    temperature: float,
    *,
    backend: TensorBackend,
) -> Callable:
    """Scale logical operation by temperature.

    Temperature Modes:
        T=0.0: Purely deductive (hard boolean, no hallucinations)
        T>0.0: Analogical reasoning (soft probabilities, generalization)
        T→∞: Maximum entropy (uniform distribution)

    Args:
        operation: Logical operation function
        temperature: Temperature parameter (≥0.0)
        backend: Tensor backend

    Returns:
        Temperature-scaled operation

    Examples:
        >>> # Deductive reasoning
        >>> op = temperature_scaled_operation(logical_and, temperature=0.0)
        >>> result = op(a, b)  # Exact boolean AND

        >>> # Analogical reasoning
        >>> op = temperature_scaled_operation(logical_and, temperature=1.0)
        >>> result = op(a, b)  # Soft probabilistic AND
    """
```

**Mathematical Formulation:**
- T=0: Use step function (hard boolean)
- T>0: Apply softmax with temperature scaling
- Interpolation: `soft_op = (1-α)·hard_op + α·continuous_op` where α=f(T)

### FR-6: Rule Composition
The system **shall** support composition of logical rules:

```python
def compose_rules(
    *rules: Array,
    operation: str = "and",
    backend: TensorBackend,
) -> Array:
    """Compose multiple logical rules.

    Args:
        rules: Variable number of boolean tensors
        operation: Composition operator ('and', 'or')
        backend: Tensor backend

    Returns:
        Composed boolean tensor

    Examples:
        >>> # Aunt(x,z) ← Sister(x,y) ∧ Parent(y,z)
        >>> aunt = compose_rules(sister, parent, operation="and")
    """
```

## 3. Non-Functional Requirements

### NFR-1: Mathematical Correctness
- **Logical axioms:** All operations satisfy standard logical properties
- **Equivalence:** Results match Domingos paper formulation
- **Numerical stability:** Operations stable for values in [0,1]
- **Precision:** Use FP32 minimum, BF16 acceptable for large models

### NFR-2: Performance
- **Tensor operations:** Leverage backend vectorization (MLX/NumPy)
- **Memory efficiency:** Sparse representations for large knowledge bases
- **Batch processing:** Support batch dimensions for parallel inference
- **12-equation transformer:** Complete transformer in ~12 tensor equations

### NFR-3: Differentiability
- **Gradient flow:** All operations support automatic differentiation
- **Soft alternatives:** Provide differentiable versions of hard operations
- **Training compatibility:** Enable end-to-end training of neural predicates

### NFR-4: Verification
- **Property-based testing:** Validate logical axioms with hypothesis
- **Cross-backend validation:** NumPy validates MLX implementation
- **Lean 4 integration:** Operations verifiable against formal theorems (future)

## 4. Features & Flows

### Feature 1: Rule Application (Priority: P0)
**Flow:**
1. User defines predicates as boolean tensors
2. User applies logical operations to compose rules
3. System validates tensor shapes match predicate arities
4. System applies operation using backend primitives
5. System returns result tensor

**Example:**
```python
# Define predicates
sister = backend.zeros((10, 10))  # Sister[x, y]
parent = backend.zeros((10, 10))  # Parent[y, z]

# Apply rule: Aunt(x,z) ← Sister(x,y) ∧ Parent(y,z)
combined = logical_and(sister, parent, backend=backend)
aunt = backend.einsum("xy,yz->xz", sister, parent)  # Compose via einsum
aunt = step(aunt, backend=backend)  # Convert to boolean
```

### Feature 2: Quantified Inference (Priority: P0)
**Flow:**
1. User specifies quantified formula (exists/forall)
2. User provides predicate tensor and quantification axis
3. System aggregates along specified axis
4. System applies step function for hard boolean or skip for soft
5. System returns quantified result

**Example:**
```python
# ∃y: Sister(x,y)
has_sister = exists(sister, axis=1, backend=backend)

# ∀x: Mortal(x)
all_mortal = forall(mortal, axis=0, backend=backend)
```

### Feature 3: Temperature-Controlled Reasoning (Priority: P1)
**Flow:**
1. User specifies temperature parameter (0.0 for deductive, >0.0 for analogical)
2. System selects hard (step) or soft (continuous) operations
3. System applies operations with temperature scaling
4. System returns results with corresponding certainty mode

## 5. Code Pattern Requirements

### Naming Conventions
- **Functions:** snake_case (`logical_and`, `exists`, `temperature_scaled_operation`)
- **Parameters:** descriptive (`predicate`, `axis`, `temperature` not `x`, `i`, `t`)
- **Private functions:** leading underscore (`_validate_boolean_tensor`)

### Type Safety Requirements
- **100% type hints** on all public functions
- **Backend parameter:** Use `TensorBackend` protocol type
- **Array type:** Use `Any` for backend-agnostic tensor type (MLX Array, NumPy ndarray)
- **Union syntax:** `int | tuple[int, ...]` for axis parameter

### Testing Approach
- **Framework:** pytest with hypothesis
- **Property tests:** Logical axioms (associativity, commutativity, distributivity)
- **Parametrized tests:** Edge cases (empty tensors, single element, all-true, all-false)
- **Cross-backend:** Validate MLX matches NumPy
- **Coverage:** ≥90% line coverage

**Property Test Examples:**
```python
from hypothesis import given, strategies as st

@given(
    a=st.lists(st.sampled_from([0.0, 1.0]), min_size=1, max_size=100),
    b=st.lists(st.sampled_from([0.0, 1.0]), min_size=1, max_size=100),
)
def test_and_commutative(a, b):
    """Property: a ∧ b = b ∧ a"""
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]
    assert logical_and(a, b) == logical_and(b, a)

@given(a=st.lists(st.sampled_from([0.0, 1.0]), min_size=1, max_size=100))
def test_double_negation(a):
    """Property: ¬¬a = a"""
    assert logical_not(logical_not(a)) == a
```

### Error Handling
- **Shape validation:** Raise `ValueError` on incompatible tensor shapes
- **Type validation:** Raise `TypeError` on non-boolean tensors
- **Enhanced errors:** Provide context and suggestions

```python
if a.shape != b.shape:
    raise ValueError(
        f"Tensor shapes incompatible: {a.shape} vs {b.shape}",
        context="logical_and(a, b)",
        suggestion="Ensure predicates have matching arities",
    )
```

### Architecture Patterns
- **Functional style:** Pure functions with no side effects
- **Backend injection:** Pass backend explicitly (no global state)
- **Composability:** Operations compose naturally

## 6. Acceptance Criteria

### Definition of Done
- [ ] Logical operations (AND, OR, NOT, IMPLIES) implemented and tested
- [ ] Quantifiers (EXISTS, FORALL) with hard and soft variants
- [ ] Step function via backend abstraction
- [ ] Temperature-controlled operation wrapper
- [ ] Rule composition utilities
- [ ] Property-based tests for all logical axioms
- [ ] Cross-backend validation (NumPy validates MLX)
- [ ] 100% type hints, passes mypy strict
- [ ] ≥90% test coverage
- [ ] Documentation with mathematical formulations

### Validation Approach
1. **Property tests:** Verify logical axioms hold
2. **Equivalence tests:** Match Domingos paper formulations
3. **Cross-backend:** MLX results match NumPy within tolerance
4. **Integration:** PatternAPI can compose rules via CoreLogic
5. **Performance:** Benchmark 12-equation transformer implementation

## 7. Dependencies

### Technical Assumptions
- **Python:** >=3.12
- **Backend:** TensorBackend protocol implemented (BACKEND component)
- **Precision:** FP32 or BF16 for boolean tensors {0.0, 1.0}

### External Integrations
- **TensorBackend:** All operations via backend abstraction
- **No direct framework dependencies:** MLX/NumPy accessed only through backend

### Related Components
- **Upstream:** TensorBackend (foundation)
- **Downstream:** PatternAPI (uses CoreLogic for pattern execution)
- **Downstream:** Compilation (uses CoreLogic for semantic strategies)
- **Future:** Verification (Lean 4 theorems for operation correctness)

### References
- **Domingos Paper:** arXiv:2510.12269 "Tensor Logic"
- **TensorLogic Overview:** `docs/TensorLogic-Overview.md` (lines 7-14)
- **Strategic Intel:** `docs/research/intel.md` (first production implementation)
- **Pattern Template:** `.sage/agent/examples/python/types/modern-type-hints.md`

---

**Implementation Priority:**
1. Basic logical operations (AND, OR, NOT) - Week 1
2. Quantifiers (EXISTS, FORALL) - Week 1
3. Rule composition utilities - Week 2
4. Temperature-controlled reasoning - Week 2
5. Comprehensive property-based testing - Week 3
