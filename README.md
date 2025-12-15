# TensorLogic

Neural-symbolic AI framework unifying logical reasoning and tensor computation. Bridge neural networks and symbolic reasoning through tensor operations based on Pedro Domingos' Tensor Logic paper (arXiv:2510.12269).

**Core Insight:** Logical operations map directly to tensor operations:
- Logical AND → Hadamard product
- Logical OR → Maximum operation
- Implications → `max(1-a, b)`
- Quantifiers → Einsum summation with Heaviside step

## Quick Start

### Installation

```bash
# Basic Installation (NumPy backend)
uv add tensorlogic

# Recommended (MLX backend for Apple Silicon)
uv add tensorlogic mlx>=0.30.0
```

### Logical Reasoning in Tensors

```python
from tensorlogic.core import logical_and, logical_or, logical_not, logical_implies
from tensorlogic.core.quantifiers import exists, forall
from tensorlogic.backends import create_backend

backend = create_backend()

# Define relations as tensors (family knowledge graph)
# Rows = subject, Columns = object
parent = backend.asarray([
    [0., 1., 1., 0.],  # Alice is parent of Bob, Carol
    [0., 0., 0., 1.],  # Bob is parent of David
    [0., 0., 0., 0.],  # Carol has no children
    [0., 0., 0., 0.],  # David has no children
])

# Infer grandparent: exists y: Parent(x,y) AND Parent(y,z)
# Using einsum: sum over intermediate variable y
composition = backend.einsum('xy,yz->xz', parent, parent)
grandparent = backend.step(composition)  # Alice is grandparent of David

# Quantified query: "Does Alice have any children?"
has_children = exists(parent[0, :], backend=backend)  # True

# Logical implication: Parent(x,y) -> Ancestor(x,y)
ancestor = logical_implies(parent, parent, backend=backend)
```

## Knowledge Graph Reasoning

TensorLogic's flagship capability: neural-symbolic reasoning over knowledge graphs with temperature-controlled inference.

```python
from tensorlogic.api import quantify, reason

# Pattern-based quantified queries
result = quantify(
    'exists y: Parent(x, y) and Parent(y, z)',
    predicates={'Parent': parent_tensor},
    backend=backend
)

# Temperature-controlled reasoning
# T=0: Pure deductive (no hallucinations)
# T>0: Analogical reasoning (generalization)
inference = reason(
    'Grandparent(x, z)',
    bindings={'x': alice_idx, 'z': david_idx},
    temperature=0.0,  # Strict deductive mode
    backend=backend
)
```

### Comprehensive Example

Run the full knowledge graph reasoning example:

```bash
uv run python examples/knowledge_graph_reasoning.py
```

**Demonstrates:**
- Family knowledge graph with 8 entities and 4 relation types
- Logical operations: AND, OR, NOT, IMPLIES
- Relation inference: Grandparent, Aunt/Uncle rules via implication
- Quantified queries: EXISTS ("has children?"), FORALL ("loves all?")
- Temperature control: T=0 deductive vs T>0 analogical reasoning
- Compilation strategy comparison across 5 semantic modes
- Uncertain knowledge handling with fuzzy relations

See [`examples/README.md`](examples/README.md) for detailed documentation.

## Compilation Strategies

TensorLogic supports multiple semantic interpretations:

| Strategy | Use Case | Differentiable |
|----------|----------|----------------|
| `soft_differentiable` | Neural network training | Yes |
| `hard_boolean` | Exact logical inference | No |
| `godel` | Fuzzy logic (min/max) | Yes |
| `product` | Probabilistic reasoning | Yes |
| `lukasiewicz` | Bounded arithmetic logic | Yes |

```python
from tensorlogic.compilation import create_strategy

# Choose semantics based on use case
strategy = create_strategy("soft_differentiable")  # Training
strategy = create_strategy("hard_boolean")         # Inference

# Compile logical operations
result = strategy.compile_and(a, b)
result = strategy.compile_implies(premise, conclusion)
```

## API Reference

### Core Operations

```python
from tensorlogic.core import logical_and, logical_or, logical_not, logical_implies

# Element-wise logical operations on tensors
result = logical_and(a, b, backend=backend)      # a AND b
result = logical_or(a, b, backend=backend)       # a OR b
result = logical_not(a, backend=backend)         # NOT a
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

### High-Level Pattern API

```python
from tensorlogic.api import quantify, reason

# Pattern-based quantified queries
result = quantify(
    'forall x: P(x) -> Q(x)',
    predicates={'P': predicate_p, 'Q': predicate_q},
    backend=backend
)

# Temperature-controlled reasoning
result = reason(
    'exists y: Related(x, y) and HasProperty(y)',
    bindings={'x': entity_batch},
    temperature=0.0,  # 0.0 = deductive, >0 = analogical
    backend=backend
)
```

## Backend System

TensorLogic uses a minimal Protocol-based abstraction (~25-30 operations) supporting multiple tensor frameworks:

- **MLX Backend** (Primary): GPU/Apple Silicon optimized with lazy evaluation
- **NumPy Backend** (Fallback): Universal CPU compatibility

### Backend Selection

```python
from tensorlogic.backends import create_backend

# Automatic selection (MLX → NumPy fallback)
backend = create_backend()

# Explicit backend selection
numpy_backend = create_backend("numpy")
mlx_backend = create_backend("mlx")
```

### Lazy Evaluation (MLX)

MLX uses lazy evaluation - operations are not computed until explicitly evaluated:

```python
backend = create_backend("mlx")

# Operations are lazy - not computed yet
a = backend.ones((100, 100))
result = backend.einsum('ij,jk->ik', a, a)

# Force evaluation
backend.eval(result)  # Now computed
```

### Backend Protocol Operations

All backends implement the `TensorBackend` Protocol:

**Creation:** `zeros`, `ones`, `arange`, `full`, `asarray`

**Transformation:** `reshape`, `broadcast_to`, `transpose`, `squeeze`, `expand_dims`

**Operations:** `einsum`, `maximum`, `add`, `subtract`, `multiply`, `divide`, `matmul`

**Reductions:** `sum`, `max`, `min`, `mean`, `prod`

**Utilities:** `eval`, `step`, `clip`, `abs`, `exp`, `log`, `sqrt`, `power`, `astype`

See [`docs/backends/API.md`](docs/backends/API.md) for complete API reference.

## Project Status

**Current Phase:** Core Framework Complete (97%)

**Completed:**
- BACKEND-001: TensorBackend Protocol with MLX + NumPy (PR #6)
- CORE-001: Logical Operations & Quantifiers (PR #7)
- API-001: Pattern Language & Compilation (PR #8)
- 817 tests, 99.76% pass rate, 100% type coverage

**Next Phase:** Advanced Applications
- COMP-001: Compilation Strategy Optimization
- VERIF-001: Lean 4 Verification Bridge
- RAG Integration: Scalable symbolic-aware retrieval

See [`docs/research/rag-goals.md`](docs/research/rag-goals.md) for research roadmap.

## Development

### Running Tests

```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=tensorlogic --cov-report=html

# Specific component
uv run pytest tests/test_core/
uv run pytest tests/test_backends/
uv run pytest tests/test_api/
```

### Type Checking

```bash
uv run mypy --strict src/tensorlogic/
# Current status: 0 errors
```

### Code Quality

```bash
uv run ruff check .   # Linting
uv run ruff format .  # Formatting
```

## Documentation

- **Conceptual Guide:** [`docs/concepts/tensor-logic-mapping.md`](docs/concepts/tensor-logic-mapping.md) - How logic becomes tensors
- **Examples:** [`examples/README.md`](examples/README.md) - Practical usage examples
- **Backend API:** [`docs/backends/API.md`](docs/backends/API.md) - Comprehensive API reference
- **Research Goals:** [`docs/research/rag-goals.md`](docs/research/rag-goals.md) - RAG research roadmap
- **Original Paper:** arXiv:2510.12269 (Domingos, 2025)

## License

MIT License
