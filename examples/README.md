# TensorLogic Examples

Practical examples demonstrating TensorLogic's neural-symbolic reasoning capabilities.

## Running Examples

```bash
# Run any example
uv run python examples/<example_name>.py

# Run all examples
for f in examples/*.py; do uv run python "$f"; done
```

## Available Examples

### 1. Compilation Strategies (`compilation_strategies.py`)

Demonstrates TensorLogic's compilation strategy system for different reasoning modes.

**Features covered:**
- Strategy creation and selection
- Comparing soft, hard, Godel, product, and Lukasiewicz semantics
- Quantifier compilation (EXISTS, FORALL)
- Nested formula compilation
- Training vs inference patterns
- Strategy selection guidelines

**Run:**
```bash
uv run python examples/compilation_strategies.py
```

**Key concepts:**
| Strategy | Use Case |
|----------|----------|
| `soft_differentiable` | Neural network training (default) |
| `hard_boolean` | Exact logical inference |
| `godel` | Fuzzy logic with min/max semantics |
| `product` | Probabilistic reasoning |
| `lukasiewicz` | Bounded arithmetic fuzzy logic |

---

### 2. Knowledge Graph Reasoning (`knowledge_graph_reasoning.py`)

Comprehensive example demonstrating neural-symbolic reasoning over a family knowledge graph.

**Features covered:**
- Backend abstraction (MLX/NumPy)
- Core logical operations (AND, OR, NOT, IMPLIES)
- Relation inference with rules
- Quantified queries (exists, forall)
- Temperature-controlled reasoning
- Compilation strategy comparison
- Uncertain knowledge handling

**Run:**
```bash
uv run python examples/knowledge_graph_reasoning.py
```

**Sections:**

| Section | Topic | Description |
|---------|-------|-------------|
| 1 | Knowledge Graph Setup | Create family tree with 8 entities and 4 relation types |
| 2 | Basic Logical Operations | Direct queries, composed relations, negation |
| 3 | Relation Inference | Grandparent and Aunt/Uncle rules via logical implication |
| 4 | Quantified Queries | EXISTS ("has children?") and FORALL ("loves all children?") |
| 5 | Temperature Control | T=0 deductive vs T>0 analogical reasoning |
| 6 | Strategy Comparison | Same query across different compilation strategies |
| 7 | Uncertain Knowledge | Fuzzy relations with uncertainty propagation |
| 8 | Best Practices | When to use each feature |

**Sample output:**
```
================================================================================
TENSORLOGIC: KNOWLEDGE GRAPH REASONING EXAMPLE
================================================================================

Section 1: Family Knowledge Graph Setup
--------------------------------------------------------------------------------
Backend: numpy
Entities: ['Alice', 'Bob', 'Carol', 'David', 'Eve', 'Frank', 'Grace', 'Henry']

Parent Relation Matrix (8x8):
         Alice    Bob  Carol  David    Eve  Frank  Grace  Henry
Alice     0.00   1.00   1.00   0.00   0.00   0.00   0.00   0.00
Bob       0.00   0.00   0.00   1.00   1.00   0.00   0.00   0.00
...
```

## API Reference

### Backend Creation

```python
from tensorlogic.backends import create_backend

# Auto-select (MLX if available, else NumPy)
backend = create_backend()

# Explicit selection
backend = create_backend("numpy")
backend = create_backend("mlx")
```

### Core Operations

```python
from tensorlogic.core import logical_and, logical_or, logical_not, logical_implies

# Element-wise logical operations
result = logical_and(a, b, backend=backend)  # a AND b
result = logical_or(a, b, backend=backend)   # a OR b
result = logical_not(a, backend=backend)     # NOT a
result = logical_implies(a, b, backend=backend)  # a -> b
```

### Quantifiers

```python
from tensorlogic.core.quantifiers import exists, forall

# Existential: "exists x such that P(x)"
result = exists(predicate, axis=0, backend=backend)

# Universal: "for all x, P(x)"
result = forall(predicate, axis=0, backend=backend)
```

### Compilation Strategies

```python
from tensorlogic.compilation import create_strategy

# Create strategy
strategy = create_strategy("soft_differentiable")
strategy = create_strategy("hard_boolean")
strategy = create_strategy("godel")
strategy = create_strategy("product")
strategy = create_strategy("lukasiewicz")

# Compile operations
result = strategy.compile_and(a, b)
result = strategy.compile_or(a, b)
result = strategy.compile_not(a)
result = strategy.compile_implies(a, b)
```

### High-Level API

```python
from tensorlogic.api import quantify, reason

# Pattern-based quantified queries
result = quantify(
    'exists y: Parent(x, y)',
    predicates={'Parent': parent_tensor},
    backend=backend
)

# Temperature-controlled reasoning
result = reason(
    'Parent(x, y) and Sibling(y, z)',
    bindings={'x': alice_idx},
    temperature=0.0,  # 0.0 = deductive, >0 = analogical
    backend=backend
)
```

## Contributing Examples

When adding new examples:

1. Follow the existing file structure with clear section headers
2. Include comprehensive docstrings
3. Use `print("=" * 80)` for section separators
4. Add type hints to all functions
5. Test with both MLX and NumPy backends
6. Update this README with the new example

## Requirements

- Python 3.12+
- TensorLogic package
- Optional: MLX for Apple Silicon acceleration
