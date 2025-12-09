# Compilation Strategies API Reference

## Overview

The compilation module implements the **Strategy Pattern** for translating logical operations into tensor operations with different semantic interpretations. This enables switching between boolean logic, fuzzy logic, and differentiable probabilistic semantics without changing your code.

**Key Concept:** Logical operations (AND, OR, NOT, IMPLIES, EXISTS, FORALL) can be interpreted in multiple mathematically valid ways, each with different trade-offs for accuracy, differentiability, and computational efficiency.

## Quick Start

```python
from tensorlogic.compilation import create_strategy
from tensorlogic.backends import create_backend

# Create backend
backend = create_backend("mlx")

# Create strategy
strategy = create_strategy("soft_differentiable", backend=backend)

# Use strategy for logical operations
a = backend.array([0.8, 0.6, 0.3])
b = backend.array([0.9, 0.4, 0.7])

# Soft AND (product)
result = strategy.compile_and(a, b)  # [0.72, 0.24, 0.21]

# Soft OR (probabilistic sum)
result = strategy.compile_or(a, b)   # [0.98, 0.76, 0.79]
```

## Strategy Selection Guide

### When to Use Each Strategy

| Strategy | Use Case | Differentiable | Semantic | Training | Inference |
|----------|----------|----------------|----------|----------|-----------|
| **soft_differentiable** | Neural predicate training | ✅ Yes | Probabilistic | ✅ Best | ⚠️ Approximate |
| **hard_boolean** | Exact logical inference | ❌ No | Boolean | ❌ N/A | ✅ Exact |
| **godel** | Fuzzy reasoning (conservative) | ⚠️ Subgradients | Min/Max | ✅ Yes | ✅ Interpretable |
| **product** | Fuzzy reasoning (probabilistic) | ✅ Yes | Product | ✅ Yes | ✅ Interpretable |
| **lukasiewicz** | Fuzzy reasoning (strict) | ✅ Yes | Bounded sum | ✅ Yes | ✅ Interpretable |

### Decision Tree

```
Do you need exact boolean logic?
├─ YES → hard_boolean
└─ NO → Do you need to train neural predicates?
    ├─ YES → Do you need fuzzy logic semantics?
    │   ├─ YES → product (most common) or lukasiewicz (strict bounds)
    │   └─ NO → soft_differentiable (default, fastest training)
    └─ NO → Do you need interpretable fuzzy values?
        ├─ YES → godel (conservative) or product (probabilistic)
        └─ NO → hard_boolean (fastest inference)
```

### Detailed Strategy Descriptions

#### soft_differentiable (Default)

**Best for:** Neural-symbolic training, gradient-based optimization

**Semantics:** Probabilistic interpretation of logic
- AND: `a * b` (product)
- OR: `a + b - a*b` (probabilistic sum)
- NOT: `1 - a` (complement)
- EXISTS: `max(predicate)` (soft maximum)
- FORALL: `min(predicate)` (soft minimum)

**Properties:**
- ✅ Fully differentiable (smooth gradients)
- ✅ Fast training convergence
- ⚠️ May produce approximate results (not exact boolean)
- ✅ Natural probabilistic interpretation

**Example:**
```python
strategy = create_strategy("soft_differentiable")

# Train neural predicates
predicates = {"P": neural_predicate_p, "Q": neural_predicate_q}
result = quantify("forall x: P(x) -> Q(x)", predicates=predicates, strategy=strategy)

# Backpropagate through logical operations
loss = compute_loss(result, target)
loss.backward()  # Gradients flow through AND, OR, quantifiers
```

#### hard_boolean

**Best for:** Production inference, exact logical reasoning, zero hallucinations

**Semantics:** Discrete boolean logic (0 or 1 only)
- AND: `step(a * b)` (discrete conjunction)
- OR: `step(max(a, b))` (discrete disjunction)
- NOT: `step(1 - a)` (discrete negation)
- EXISTS: `step(max(predicate))` (at least one true)
- FORALL: `step(min(predicate))` (all true)

**Properties:**
- ❌ Not differentiable (raises error on gradient request)
- ✅ Exact boolean results (no approximation)
- ✅ Zero hallucinations
- ✅ Fastest inference (no soft operations)

**Example:**
```python
strategy = create_strategy("hard_boolean")

# Exact logical inference (no training)
result = strategy.compile_and(a, b)  # Produces 0 or 1, never 0.5

# Use for production inference after training with soft strategy
# 1. Train with soft_differentiable
# 2. Switch to hard_boolean for deployment
```

#### godel (Gödel Fuzzy Logic)

**Best for:** Fuzzy reasoning with conservative semantics, subgradient optimization

**Semantics:** Min/max fuzzy logic
- AND: `min(a, b)` (pessimistic conjunction)
- OR: `max(a, b)` (optimistic disjunction)
- NOT: `1 - a` (standard complement)
- EXISTS: `max(predicate)` (at least one)
- FORALL: `min(predicate)` (all must hold)

**Properties:**
- ⚠️ Differentiable via subgradients (non-smooth at boundaries)
- ✅ Idempotent: `AND(a, a) = a`, `OR(a, a) = a`
- ✅ Conservative: Always takes minimum/maximum
- ⚠️ May slow training convergence (flat gradients)

**Example:**
```python
strategy = create_strategy("godel")

# Fuzzy rule evaluation with conservative semantics
# "If temperature is high AND pressure is low, THEN ..."
temp_high = backend.array([0.8])
pressure_low = backend.array([0.6])
condition = strategy.compile_and(temp_high, pressure_low)  # min(0.8, 0.6) = 0.6
```

#### product (Product Fuzzy Logic)

**Best for:** Fuzzy reasoning with probabilistic semantics, smooth training

**Semantics:** Product t-norm
- AND: `a * b` (product)
- OR: `a + b - a*b` (probabilistic sum)
- NOT: `1 - a` (standard complement)
- EXISTS: `max(predicate)` (at least one)
- FORALL: `min(predicate)` (all must hold)

**Properties:**
- ✅ Fully differentiable (smooth gradients everywhere)
- ✅ Probabilistic interpretation: independent events
- ✅ Fast training convergence
- ✅ Most popular fuzzy logic variant

**Example:**
```python
strategy = create_strategy("product")

# Fuzzy rule with probabilistic interpretation
# Treats logical conjunction like independent probability
expert_1_confidence = backend.array([0.9])
expert_2_confidence = backend.array([0.8])
combined = strategy.compile_and(expert_1_confidence, expert_2_confidence)  # 0.72
```

#### lukasiewicz (Łukasiewicz Fuzzy Logic)

**Best for:** Fuzzy reasoning with strict boundary conditions, sum-based semantics

**Semantics:** Łukasiewicz t-norm (bounded arithmetic)
- AND: `max(0, a + b - 1)` (bounded sum)
- OR: `min(1, a + b)` (bounded addition)
- NOT: `1 - a` (standard complement)
- EXISTS: `max(predicate)` (at least one)
- FORALL: `min(predicate)` (all must hold)

**Properties:**
- ✅ Fully differentiable (smooth gradients)
- ✅ Strict boundary enforcement (values never exceed [0, 1])
- ✅ Linear operations (addition-based)
- ⚠️ Less common than product fuzzy logic

**Example:**
```python
strategy = create_strategy("lukasiewicz")

# Fuzzy logic with strict arithmetic semantics
a = backend.array([0.6])
b = backend.array([0.7])
result = strategy.compile_and(a, b)  # max(0, 0.6 + 0.7 - 1) = 0.3
```

## API Reference

### Factory Function

#### `create_strategy`

```python
def create_strategy(
    name: str = "soft_differentiable",
    backend: TensorBackend | None = None
) -> CompilationStrategy
```

Create a compilation strategy by name.

**Parameters:**
- `name` (str): Strategy identifier. Default: `"soft_differentiable"`
  - Available: `"soft_differentiable"`, `"hard_boolean"`, `"godel"`, `"product"`, `"lukasiewicz"`
- `backend` (TensorBackend | None): Optional backend instance. If `None`, creates default NumPy backend.

**Returns:**
- `CompilationStrategy`: Strategy instance implementing the CompilationStrategy protocol

**Raises:**
- `ValueError`: If strategy name not recognized

**Example:**
```python
from tensorlogic.compilation import create_strategy
from tensorlogic.backends import create_backend

# Default strategy (soft_differentiable with NumPy)
strategy = create_strategy()

# Explicit strategy with MLX backend
backend = create_backend("mlx")
strategy = create_strategy("hard_boolean", backend=backend)
```

#### `get_available_strategies`

```python
def get_available_strategies() -> list[str]
```

Get list of all registered strategy names.

**Returns:**
- `list[str]`: List of available strategy identifiers

**Example:**
```python
from tensorlogic.compilation import get_available_strategies

strategies = get_available_strategies()
print(strategies)
# ['soft_differentiable', 'hard_boolean', 'godel', 'product', 'lukasiewicz']
```

### CompilationStrategy Protocol

All strategies implement this protocol with 6 core operations and 2 properties.

#### Operations

##### `compile_and(a, b)`

Compile logical AND operation.

**Parameters:**
- `a`: First input tensor (values in [0, 1])
- `b`: Second input tensor (broadcastable with `a`)

**Returns:**
- Tensor representing AND(a, b) according to strategy semantics

**Strategy-specific semantics:**
- `soft_differentiable`: `a * b`
- `hard_boolean`: `step(a * b)`
- `godel`: `min(a, b)`
- `product`: `a * b`
- `lukasiewicz`: `max(0, a + b - 1)`

##### `compile_or(a, b)`

Compile logical OR operation.

**Parameters:**
- `a`: First input tensor (values in [0, 1])
- `b`: Second input tensor (broadcastable with `a`)

**Returns:**
- Tensor representing OR(a, b) according to strategy semantics

**Strategy-specific semantics:**
- `soft_differentiable`: `a + b - a*b`
- `hard_boolean`: `step(max(a, b))`
- `godel`: `max(a, b)`
- `product`: `a + b - a*b`
- `lukasiewicz`: `min(1, a + b)`

##### `compile_not(a)`

Compile logical NOT operation.

**Parameters:**
- `a`: Input tensor (values in [0, 1])

**Returns:**
- Tensor representing NOT(a) (consistent across all strategies: `1 - a`)

##### `compile_implies(a, b)`

Compile logical implication (a → b).

**Parameters:**
- `a`: Antecedent tensor (values in [0, 1])
- `b`: Consequent tensor (broadcastable with `a`)

**Returns:**
- Tensor representing IMPLIES(a, b)

**Strategy-specific semantics:**
- All strategies: `max(1 - a, b)` (equivalent to `OR(NOT(a), b)`)

##### `compile_exists(predicate, axis)`

Compile existential quantification (∃x: P(x)).

**Parameters:**
- `predicate`: Tensor of predicate values
- `axis`: Axis or axes to quantify over

**Returns:**
- Tensor representing "at least one true"

**Strategy-specific semantics:**
- Most strategies: `max(predicate, axis=axis)`
- `hard_boolean`: `step(max(predicate, axis=axis))`

##### `compile_forall(predicate, axis)`

Compile universal quantification (∀x: P(x)).

**Parameters:**
- `predicate`: Tensor of predicate values
- `axis`: Axis or axes to quantify over

**Returns:**
- Tensor representing "all true"

**Strategy-specific semantics:**
- Most strategies: `min(predicate, axis=axis)`
- `hard_boolean`: `step(min(predicate, axis=axis))`

#### Properties

##### `is_differentiable`

```python
@property
def is_differentiable(self) -> bool
```

Whether strategy supports gradient computation.

**Returns:**
- `True` for differentiable strategies: `soft_differentiable`, `godel`, `product`, `lukasiewicz`
- `False` for non-differentiable: `hard_boolean`

##### `name`

```python
@property
def name(self) -> str
```

Strategy identifier name.

**Returns:**
- Strategy name string (e.g., `"soft_differentiable"`)

## Integration with quantify() API

Strategies integrate seamlessly with the high-level `quantify()` API:

```python
from tensorlogic.api import quantify
from tensorlogic.compilation import create_strategy

# Option 1: Pass strategy name (string)
result = quantify(
    "forall x: P(x) -> Q(x)",
    predicates={"P": pred_p, "Q": pred_q},
    strategy="soft_differentiable"  # Default
)

# Option 2: Pass strategy instance
strategy = create_strategy("hard_boolean")
result = quantify(
    "exists y: Related(x, y) and HasProperty(y)",
    bindings={"x": entity_batch},
    strategy=strategy  # Direct instance
)

# Option 3: Default (soft_differentiable)
result = quantify("forall x: P(x)", predicates={"P": pred})
# Automatically uses soft_differentiable
```

## Common Patterns

### Training with Soft, Inference with Hard

Train neural predicates with differentiable strategy, then deploy with exact boolean logic:

```python
# Training phase
train_strategy = create_strategy("soft_differentiable", backend=backend)

for epoch in range(num_epochs):
    result = quantify(formula, predicates=predicates, strategy=train_strategy)
    loss = compute_loss(result, labels)
    loss.backward()
    optimizer.step()

# Inference phase (after training)
inference_strategy = create_strategy("hard_boolean", backend=backend)
result = quantify(formula, predicates=predicates, strategy=inference_strategy)
# Result is exact boolean (0 or 1), no approximation
```

### Strategy Comparison

Compare outputs across different strategies:

```python
from tensorlogic.compilation import get_available_strategies, create_strategy

strategies = {
    name: create_strategy(name, backend=backend)
    for name in get_available_strategies()
}

for name, strategy in strategies.items():
    result = strategy.compile_and(a, b)
    print(f"{name:20s}: {result}")

# Output:
# soft_differentiable : [0.72 0.24 0.21]
# hard_boolean        : [1 0 0]
# godel               : [0.8 0.4 0.3]
# product             : [0.72 0.24 0.21]
# lukasiewicz         : [0.7 0.0 0.0]
```

### Custom Strategy Registration

Register your own strategy:

```python
from tensorlogic.compilation import register_strategy, CompilationStrategy

class MyCustomStrategy:
    def __init__(self, backend=None):
        self._backend = backend or create_backend("numpy")

    @property
    def name(self) -> str:
        return "custom"

    @property
    def is_differentiable(self) -> bool:
        return True

    def compile_and(self, a, b):
        # Custom AND semantics
        return self._backend.custom_and(a, b)

    # ... implement other operations

# Register custom strategy
register_strategy("custom", MyCustomStrategy)

# Now available via factory
strategy = create_strategy("custom")
```

## Mathematical Properties

### Semantic Axioms by Strategy

Each strategy satisfies different logical/fuzzy axioms:

#### soft_differentiable & product

- Commutativity: `AND(a, b) = AND(b, a)`, `OR(a, b) = OR(b, a)`
- Associativity: `AND(AND(a, b), c) = AND(a, AND(b, c))`
- Identity: `AND(a, 1) = a`, `OR(a, 0) = a`
- Annihilator: `AND(a, 0) = 0`, `OR(a, 1) = 1`
- ⚠️ NOT idempotent: `AND(a, a) = a²` (not `a`)

#### hard_boolean

- All classical boolean axioms
- Law of excluded middle: `OR(a, NOT(a)) = 1`
- Law of non-contradiction: `AND(a, NOT(a)) = 0`
- Idempotence: `AND(a, a) = a`, `OR(a, a) = a`

#### godel

- All fuzzy logic axioms
- Idempotence: `AND(a, a) = a`, `OR(a, a) = a`
- Monotonicity: If `a ≤ b`, then `AND(a, c) ≤ AND(b, c)`
- Boundary conditions: `AND(a, 0) = 0`, `OR(a, 1) = 1`

#### lukasiewicz

- All fuzzy logic axioms
- Nilpotent: `AND(a, a, ..., a)` (n times) → 0 as n → ∞
- Strict boundary enforcement
- Continuity: Small changes in input → small changes in output

## Performance Considerations

### Strategy Performance Characteristics

| Strategy | Training Speed | Inference Speed | Memory | Gradient Quality |
|----------|---------------|-----------------|--------|------------------|
| **soft_differentiable** | Fast | Fast | Low | Excellent |
| **hard_boolean** | N/A | Fastest | Lowest | N/A |
| **godel** | Slow (subgradients) | Fast | Low | Fair (non-smooth) |
| **product** | Fast | Fast | Low | Excellent |
| **lukasiewicz** | Medium | Medium | Low | Good |

### Optimization Tips

1. **Use soft_differentiable for training**: Fastest convergence and best gradients
2. **Switch to hard_boolean for inference**: Exact results, fastest execution
3. **Avoid godel for training**: Subgradients can slow convergence
4. **Batch operations**: All strategies support batched operations efficiently
5. **Backend selection**: MLX backend provides best performance on Apple Silicon

## Error Handling

### Common Errors

#### Gradient on Non-Differentiable Strategy

```python
strategy = create_strategy("hard_boolean")
result = strategy.compile_and(a, b)
# Attempting to compute gradients will raise:
# TensorLogicError: Cannot compute gradients through hard_boolean strategy
```

**Solution:** Use differentiable strategy for training.

#### Shape Mismatch

```python
a = backend.array([0.8, 0.6])
b = backend.array([0.9, 0.4, 0.7])
result = strategy.compile_and(a, b)
# ValueError: Operands could not be broadcast together
```

**Solution:** Ensure tensors are broadcastable.

#### Invalid Strategy Name

```python
strategy = create_strategy("nonexistent")
# ValueError: Unknown compilation strategy: 'nonexistent'.
# Available strategies: 'soft_differentiable', 'hard_boolean', ...
```

**Solution:** Use `get_available_strategies()` to list valid names.

## See Also

- **Backend API**: `docs/api/backends.md` - Tensor backend abstraction
- **Core Operations API**: `docs/api/core.md` - Low-level logical operations
- **Pattern API**: `docs/api/patterns.md` - High-level quantify() API
- **Examples**: `examples/compilation_strategies.py` - Complete working examples

## References

- **Domingos (2025)**: "Tensor Logic: Unifying Neural and Symbolic AI" (arXiv:2510.12269)
  - Section 3: "Compilation Strategies and Semantic Variants"
- **Fuzzy Logic Theory**:
  - Gödel t-norm: `min(a, b)`
  - Product t-norm: `a * b`
  - Łukasiewicz t-norm: `max(0, a + b - 1)`
- **Probabilistic Logic**: Product semantics as independent event probabilities
