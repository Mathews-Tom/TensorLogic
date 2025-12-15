# TensorLogic

Neural-symbolic AI framework unifying logical reasoning and tensor computation. Bridge neural networks and symbolic reasoning through tensor operations based on Pedro Domingos' Tensor Logic paper (arXiv:2510.12269).

**Core Insight:** Logical operations map directly to tensor operations:
- Logical AND → Hadamard product
- Logical OR → Maximum operation
- Implications → `max(1-a, b)`
- Quantifiers → Einsum summation with Heaviside step

## Beyond Deduction: Enabling Generalization with Analogical Reasoning

TensorLogic's breakthrough capability: **temperature-controlled reasoning** that bridges pure logic and neural approximation.

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| T=0 | Pure deductive inference | Verification, provable correctness, zero hallucinations |
| T=0.1-0.5 | Cautious generalization | Robust inference with uncertainty |
| T=1.0 | Analogical reasoning | Pattern completion, missing link prediction |
| T>1.0 | Exploratory | Creative hypotheses, knowledge graph expansion |

**Why this matters:** Standard logical solvers give you T=0 only. Standard neural networks give you T>0 only with no guarantees. TensorLogic gives you the entire spectrum—from mathematically provable deduction to neural-style generalization—in a unified framework.

```python
from tensorlogic.api import reason

# Pure deduction: mathematically provable, zero hallucinations
result = reason('Grandparent(x, z)', temperature=0.0, ...)

# Analogical: can infer "likely grandparent" even with incomplete data
result = reason('Grandparent(x, z)', temperature=0.5, ...)
```

This capability is theoretically grounded in Pedro Domingos' Tensor Logic paper ([arXiv:2510.12269](https://arxiv.org/abs/2510.12269)). For a deep dive on temperature semantics, see the [Temperature-Controlled Inference Guide](docs/concepts/tensor-logic-mapping.md#temperature-controlled-inference).

## Quick Start

### Installation

```bash
# Basic Installation (NumPy backend)
uv add tensorlogic

# Recommended (MLX backend for Apple Silicon)
uv add tensorlogic mlx>=0.30.0
```

### Performance Architecture

TensorLogic is built for scale. The MLX backend enables 1M+ entity knowledge graphs on Apple Silicon:

```python
from tensorlogic.backends import create_backend

# Auto-selects MLX (GPU) on Apple Silicon, NumPy fallback elsewhere
backend = create_backend()  # ← This step selects your hardware backend
```

| Backend | Hardware | Key Advantage |
|---------|----------|---------------|
| **MLX** | Apple Silicon (M1/M2/M3) | Unified memory + Metal GPU, lazy evaluation |
| **NumPy** | Universal CPU | Compatibility fallback |

The MLX backend's lazy evaluation enables 10-100x speedups for complex knowledge graph queries. See [Performance Benchmarks](docs/PERFORMANCE.md) for detailed metrics.

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

TensorLogic supports multiple semantic interpretations—choose based on your problem, not your logic background:

### soft_differentiable — Train neural networks that respect logical rules
**Problem:** "I want to train a model where the loss includes logical constraints"
**Example:** Learning embeddings where `Parent(x,y) ∧ Parent(y,z) → Grandparent(x,z)` is enforced during training

### hard_boolean — Provable, exact inference
**Problem:** "I need mathematically guaranteed answers with no approximation"
**Example:** Verifying that a knowledge graph satisfies business rules (integrates with [Lean 4 verification](docs/specs/verification/spec.md))

### godel — Score similarity on a continuous spectrum
**Problem:** "I need a grade (0.0-1.0), not just true/false"
**Example:** Scoring product similarity in a recommendation engine

### product — Probabilistic reasoning with independent events
**Problem:** "I'm combining probabilities and want P(A∧B) = P(A) × P(B)"
**Example:** Computing joint probabilities in a Bayesian knowledge graph

### lukasiewicz — Bounded arithmetic with saturation
**Problem:** "I need bounded confidence scores that don't explode"
**Example:** Multi-hop reasoning where confidence degrades gracefully

| Strategy | Differentiable | Best For |
|----------|----------------|----------|
| `soft_differentiable` | Yes | Neural network training with logic constraints |
| `hard_boolean` | No | Exact verification, theorem proving |
| `godel` | Yes | Similarity scoring, fuzzy matching |
| `product` | Yes | Probabilistic inference |
| `lukasiewicz` | Yes | Bounded multi-hop reasoning |

```python
from tensorlogic.compilation import create_strategy

# Choose based on your problem
strategy = create_strategy("soft_differentiable")  # Training with logic constraints
strategy = create_strategy("hard_boolean")         # Exact verification
strategy = create_strategy("godel")                # Continuous scoring
```

See [Compilation Strategies Guide](docs/api/compilation.md) for detailed API reference and mathematical semantics.

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

TensorLogic uses a minimal Protocol-based abstraction (~25-30 operations) supporting multiple tensor frameworks. See [Performance Architecture](#performance-architecture) for hardware selection.

```python
from tensorlogic.backends import create_backend

# Explicit backend selection
numpy_backend = create_backend("numpy")
mlx_backend = create_backend("mlx")
```

**MLX Lazy Evaluation:** Operations are not computed until `backend.eval(result)` is called—critical for batching complex knowledge graph queries.

**Protocol Operations:**
- **Creation:** `zeros`, `ones`, `arange`, `full`, `asarray`
- **Transformation:** `reshape`, `broadcast_to`, `transpose`, `squeeze`, `expand_dims`
- **Operations:** `einsum`, `maximum`, `add`, `subtract`, `multiply`, `divide`, `matmul`
- **Reductions:** `sum`, `max`, `min`, `mean`, `prod`
- **Utilities:** `eval`, `step`, `clip`, `abs`, `exp`, `log`, `sqrt`, `power`, `astype`

See [`docs/backends/API.md`](docs/backends/API.md) for complete API reference.

## Project Status

**Current Phase:** Production Ready

**Completed:**
- BACKEND-001: TensorBackend Protocol with MLX + NumPy (PR #6)
- CORE-001: Logical Operations & Quantifiers (PR #7)
- API-001: Pattern Language & Compilation (PR #8)
- VERIF-001: Lean 4 Verification Bridge (15 theorems proven)
- RAG-001: Integration module with LangChain adapter
- 1,257 tests, 99%+ pass rate, 100% type coverage

**Features:**
- Sparse tensor support for 1M+ entity knowledge graphs
- LangChain-compatible retriever with hybrid neural-symbolic scoring
- 4 Jupyter notebooks for interactive learning
- Benchmark suite for scale validation

See [`docs/tutorials/index.md`](docs/tutorials/index.md) for tutorials and [`docs/research/rag-goals.md`](docs/research/rag-goals.md) for research roadmap.

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
uv run pytest tests/test_integrations/
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
