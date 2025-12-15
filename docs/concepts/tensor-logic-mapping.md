# Tensor Logic: The Rosetta Stone

A conceptual guide mapping logical operations to tensor operations for researchers from logic, AI, and mathematics backgrounds.

## The Core Insight

Pedro Domingos' Tensor Logic (arXiv:2510.12269) reveals a fundamental equivalence: **logical rules and Einstein summation are mathematically identical operations**, differing only in atomic data types. This enables neural networks to perform exact logical reasoning while maintaining differentiability for training.

## How Logic Becomes Tensors

| Logical Operation | Tensor Operation | Mathematical Form | TensorLogic Code |
|-------------------|------------------|-------------------|------------------|
| AND (a ∧ b) | Hadamard product | a ⊙ b | `logical_and(a, b)` |
| OR (a ∨ b) | Maximum | max(a, b) | `logical_or(a, b)` |
| NOT (¬a) | Complement | 1 - a | `logical_not(a)` |
| IMPLIES (a → b) | Fuzzy implication | max(1-a, b) | `logical_implies(a, b)` |
| EXISTS (∃x.P(x)) | Sum + Step | step(Σ_x P(x)) | `exists(P, axis)` |
| FORALL (∀x.P(x)) | Product + Step | step(Π_x P(x)) | `forall(P, axis)` |

## Relations as Tensors

A binary relation R(x, y) between entities becomes a matrix:
- Rows represent the first argument (subject)
- Columns represent the second argument (object)
- Value 1.0 means the relation holds; 0.0 means it doesn't

```python
# Family tree as tensor
# Entities: [Alice, Bob, Carol, David]
parent = backend.asarray([
    #  Alice  Bob  Carol  David
    [   0.,   1.,    1.,    0.],  # Alice is parent of Bob, Carol
    [   0.,   0.,    0.,    1.],  # Bob is parent of David
    [   0.,   0.,    0.,    0.],  # Carol has no children
    [   0.,   0.,    0.,    0.],  # David has no children
])
# parent[0, 1] = 1.0 means Parent(Alice, Bob) is TRUE
```

## How Quantifiers Work

### EXISTS via Einsum

The existential quantifier "exists y such that P(x, y)" becomes a sum over the y axis:

```python
# "Exists y such that Parent(Alice, y)"
# In logic: ∃y. Parent(Alice, y)
# In tensor: step(Σ_y Parent[Alice, y])

alice_has_children = backend.step(
    backend.sum(parent[0, :], axis=0)
)  # Returns 1.0 (TRUE)
```

### Multi-hop Reasoning with Einsum

The grandparent relation requires composing two parent relations:

```
Grandparent(x, z) ← ∃y. Parent(x, y) ∧ Parent(y, z)
```

This becomes an einsum operation:

```python
# Einsum contracts over the shared variable y
# 'xy,yz->xz' means: sum over y for each (x,z) pair
composition = backend.einsum('xy,yz->xz', parent, parent)
grandparent = backend.step(composition)

# Result: grandparent[0, 3] = 1.0
# Meaning: Grandparent(Alice, David) is TRUE
```

**Why einsum?** The pattern `xy,yz->xz` specifies:
- `xy`: First relation has indices x (row) and y (column)
- `yz`: Second relation has indices y (row) and z (column)
- `->xz`: Output has indices x and z
- The shared index `y` is summed over (existential quantification)

### FORALL via Product

The universal quantifier "for all x, P(x)" becomes a product:

```python
# "Alice loves all her children"
# In logic: ∀c. Parent(Alice, c) → Loves(Alice, c)
# Simplified: If Alice has children, does she love them all?

# Get Alice's children
children_mask = parent[0, :]  # [0, 1, 1, 0] - Bob and Carol

# Check if Alice loves each child
loves = backend.asarray([...])  # loves[0, :] for Alice's loves
alice_loves = loves[0, :]

# Universal quantification: product over all true children
# Only consider children (mask out non-children)
relevant_loves = logical_implies(children_mask, alice_loves, backend=backend)
all_loved = forall(relevant_loves, backend=backend)
```

## Temperature Control Explained

Temperature controls the reasoning mode, transitioning between exact logic and soft inference:

| Temperature | Reasoning Mode | Mathematical Effect | Use Case |
|-------------|----------------|---------------------|----------|
| T = 0 | Hard deductive | Step function (0 or 1) | Exact inference, no uncertainty |
| T = 0.1 | Soft deductive | Sharp sigmoid | Minor uncertainty tolerance |
| T = 1.0 | Analogical | Smooth sigmoid | Generalization, similarity-based reasoning |
| T > 1.0 | Exploratory | Very smooth | Creative reasoning, hypothesis generation |

```python
from tensorlogic.core.temperature import with_temperature

# Hard deductive (T=0): Binary output
hard_result = with_temperature(
    logical_and(a, b, backend=backend),
    temperature=0.0,
    backend=backend
)  # Returns exactly 0.0 or 1.0

# Analogical (T=1.0): Soft output
soft_result = with_temperature(
    logical_and(a, b, backend=backend),
    temperature=1.0,
    backend=backend
)  # Returns values in [0, 1] based on similarity
```

**Key Insight:** At T=0, TensorLogic performs exact logical reasoning with no hallucinations. As temperature increases, reasoning becomes more flexible, enabling generalization but potentially introducing uncertainty.

## Compilation Strategy Selection

Different logical semantics produce different behaviors:

| Strategy | AND | OR | NOT | Key Property |
|----------|-----|-----|-----|--------------|
| `hard_boolean` | min(a,b) → step | max(a,b) → step | 1-a | Exact Boolean, not differentiable |
| `soft_differentiable` | sigmoid(a+b) | sigmoid(max) | 1-a | Smooth, trainable |
| `godel` | min(a,b) | max(a,b) | 1-a | Standard fuzzy logic |
| `product` | a × b | a + b - a×b | 1-a | Probabilistic interpretation |
| `lukasiewicz` | max(0, a+b-1) | min(1, a+b) | 1-a | Bounded arithmetic |

### When to Use Each Strategy

```plaintext
┌─────────────────────────────────────────────────────────────┐
│                  Strategy Selection Guide                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Need exact logical inference?                              │
│  ├── YES → Use hard_boolean                                 │
│  └── NO ↓                                                   │
│                                                             │
│  Training a neural network?                                 │
│  ├── YES → Use soft_differentiable (default)                │
│  └── NO ↓                                                   │
│                                                             │
│  Working with fuzzy/uncertain knowledge?                    │
│  ├── YES → Which semantics?                                 │
│  │   ├── Standard (min/max) → Use godel                     │
│  │   ├── Probabilistic → Use product                        │
│  │   └── Bounded arithmetic → Use lukasiewicz               │
│  └── NO → Use soft_differentiable                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

```python
from tensorlogic.compilation import create_strategy

# For neural network training
strategy = create_strategy("soft_differentiable")

# For exact logical inference
strategy = create_strategy("hard_boolean")

# For fuzzy logic with probabilistic interpretation
strategy = create_strategy("product")
```

## Complete Example: Family Reasoning

```python
from tensorlogic.backends import create_backend
from tensorlogic.core import logical_and, logical_or, logical_not, logical_implies
from tensorlogic.core.quantifiers import exists, forall

backend = create_backend()

# Define entities
entities = ['Alice', 'Bob', 'Carol', 'David', 'Eve', 'Frank']
N = len(entities)

# Define relations as tensors
parent = backend.zeros((N, N))
# Alice is parent of Bob and Carol
parent = backend.asarray([
    [0, 1, 1, 0, 0, 0],  # Alice
    [0, 0, 0, 1, 1, 0],  # Bob (parent of David, Eve)
    [0, 0, 0, 0, 0, 1],  # Carol (parent of Frank)
    [0, 0, 0, 0, 0, 0],  # David
    [0, 0, 0, 0, 0, 0],  # Eve
    [0, 0, 0, 0, 0, 0],  # Frank
])

sibling = backend.asarray([
    [0, 0, 0, 0, 0, 0],  # Alice
    [0, 0, 1, 0, 0, 0],  # Bob - sibling of Carol
    [0, 1, 0, 0, 0, 0],  # Carol - sibling of Bob
    [0, 0, 0, 0, 1, 0],  # David - sibling of Eve
    [0, 0, 0, 1, 0, 0],  # Eve - sibling of David
    [0, 0, 0, 0, 0, 0],  # Frank
])

# Infer grandparent: ∃y. Parent(x,y) ∧ Parent(y,z)
grandparent = backend.step(
    backend.einsum('xy,yz->xz', parent, parent)
)
# Alice is grandparent of David, Eve, Frank

# Infer aunt/uncle: ∃y. Sibling(x,y) ∧ Parent(y,z)
aunt_uncle = backend.step(
    backend.einsum('xy,yz->xz', sibling, parent)
)
# Carol is aunt of David, Eve; Bob is uncle of Frank

# Query: "Who has grandchildren?"
has_grandchildren = exists(grandparent, axis=1, backend=backend)
# Returns [1, 0, 0, 0, 0, 0] - only Alice

# Query: "Is Alice an ancestor of everyone?"
ancestor = logical_or(parent, grandparent, backend=backend)
alice_ancestor_of_all = forall(ancestor[0, 1:], backend=backend)
# Check if Alice is ancestor of all non-Alice entities
```

## Mathematical Foundations

For those interested in the formal theory:

### Tensor Logic Equation
A Datalog rule: `R(x,z) ← S(x,y), T(y,z)`

Becomes: `R[x,z] = step(S[x,y] · T[y,z])`

Where the Einstein summation convention implies summation over repeated index y.

### Key Theorems

1. **Soundness:** If tensor operations return 1.0 at T=0, the logical formula is provable
2. **Completeness:** If a logical formula is provable, tensor operations return 1.0 at T=0
3. **Differentiability:** At T>0, all operations are differentiable for gradient-based learning

## Further Reading

- **Original Paper:** arXiv:2510.12269 (Domingos, 2025) - Theoretical foundations
- **Knowledge Graph Example:** `examples/knowledge_graph_reasoning.py` - Full implementation
- **Compilation Strategies:** `docs/api/compilation.md` - Detailed semantics
- **Core Operations Spec:** `docs/specs/core/spec.md` - Mathematical specifications
