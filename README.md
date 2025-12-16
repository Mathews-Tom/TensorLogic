# TensorLogic

**The Temperature Dial for AI Reasoning** — Go from provable deduction to creative inference in one parameter.

[![PyPI](https://img.shields.io/pypi/v/python-tensorlogic)](https://pypi.org/project/python-tensorlogic/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mathews-Tom/TensorLogic/blob/main/notebooks/05_google_colab_cuda.ipynb)

## What Is This? (30-Second Pitch)

TensorLogic is a **reasoning framework** with a single parameter that controls how strictly your AI follows logic:

```
             THE TEMPERATURE DIAL

    STRICT                              CREATIVE
    T=0.0 ─────────────────────────────── T=2.0
      │                                     │
      ▼                                     ▼
  "Alice is Bob's grandparent"     "Alice might be related
   ONLY if the data proves it       to Bob based on patterns"
      │                                     │
      ▼                                     ▼
   Zero hallucinations              Generalizes from
   Provably correct                 incomplete data
```

**Think of it like a volume knob:**
- **Turn it to 0** → Strict database query (only returns what's explicitly true)
- **Turn it up** → Fuzzy search (finds likely matches even with missing data)

**No PhD required.** If you can write Python lists, you can use TensorLogic.

## The 3-Minute Tour

Try these examples in order. Each builds on the previous.

### Level 0: Hello World (5 lines of logic)

```bash
uv run python examples/00_hello_world.py   # or: pip install python-tensorlogic && python ...
```

```python
from tensorlogic import create_backend, logical_and

backend = create_backend()  # Auto-selects best GPU/CPU
facts_a = [1.0, 0.0, 1.0, 0.0]  # TRUE, FALSE, TRUE, FALSE
facts_b = [1.0, 1.0, 0.0, 0.0]  # TRUE, TRUE, FALSE, FALSE

result = logical_and(facts_a, facts_b, backend=backend)
print(result)  # [1. 0. 0. 0.] - Only position 0 is TRUE in BOTH
```

### Level 1: Family Tree (Multi-Hop Reasoning)

```bash
uv run python examples/01_family_tree_minimal.py
```

```python
# "Alice is parent of Bob, Bob is parent of Carol"
# Who is Alice's grandchild? (We never stated this directly!)

parent = [
    [0, 1, 0],  # Alice → Bob
    [0, 0, 1],  # Bob → Carol
    [0, 0, 0],  # Carol → (no one)
]

# Matrix multiply = "follow two parent edges" = grandparent!
grandparent = parent @ parent  # Alice → Carol (inferred!)
```

### Level 2: Temperature Control

```bash
uv run python examples/02_temperature_demo.py
```

```python
# User preferences (some uncertain)
likes_action = [1.0, 0.6, 0.0, 0.5]  # User 3 = unknown (0.5)
likes_comedy = [0.8, 0.5, 0.9, 0.7]

# T=0: Only recommend when CERTAIN → User 0 only
# T=1: Recommend with uncertainty  → Users 0, 1, 3 (graded scores)
```

**Ready for more?** See [examples/README.md](examples/README.md) for the complete progression.

---

## The Problem TensorLogic Solves

Traditional AI forces a choice: **logical solvers** give you provable correctness but can't generalize beyond their rules. **Neural networks** generalize beautifully but hallucinate with no guarantees. You've had to pick one.

**TensorLogic gives you both.** One framework. One API. One parameter to control the trade-off.

```python
from tensorlogic.api import reason

# Pure deduction: mathematically provable, zero hallucinations
certain = reason('Grandparent(x, z)', temperature=0.0, ...)  # T=0: Exact logic

# Analogical: infers "likely grandparent" even with incomplete data
creative = reason('Grandparent(x, z)', temperature=0.5, ...)  # T>0: Generalization
```

**That's the entire value proposition.** The temperature dial bridges symbolic AI and neural AI.

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

## Technical Validation

Before diving in, here's why you can trust TensorLogic for production:

| Capability | Status | What It Means |
|------------|--------|---------------|
| **1M+ Entity Graphs** | Sparse tensor support | Handle enterprise-scale knowledge graphs |
| **Up to 700x Speedups** | MLX + CUDA backends | Real-time inference on GPU hardware |
| **15 Theorems Proven** | Lean 4 verification | Mathematically verified core operations |
| **99%+ Test Coverage** | 1,257 tests | Production-grade reliability |
| **LangChain Integration** | RAG-ready | Drop into existing LLM pipelines |

This isn't an academic prototype. It's built for production ML pipelines.

## Quick Start

### Installation

```bash
# Basic Installation (NumPy backend)
pip install python-tensorlogic

# Apple Silicon (MLX backend - recommended for M1/M2/M3)
pip install python-tensorlogic mlx>=0.30.0

# NVIDIA GPU / Google Colab (CUDA backend)
pip install python-tensorlogic cupy-cuda12x  # CUDA 12.x (Colab, recommended)
pip install python-tensorlogic cupy-cuda11x  # CUDA 11.x (legacy systems)
```

**Try it on Google Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mathews-Tom/TensorLogic/blob/main/notebooks/05_google_colab_cuda.ipynb)

### Performance Architecture

TensorLogic is built for scale across all major GPU platforms:

```python
from tensorlogic.backends import create_backend

# Auto-detect best available: MLX (Apple) -> CUDA (NVIDIA) -> NumPy (CPU)
backend = create_backend()  # Automatic hardware selection

# Or explicitly choose your backend
backend = create_backend("cuda")   # NVIDIA GPUs (T4, V100, A100, Colab)
backend = create_backend("mlx")    # Apple Silicon (M1/M2/M3)
backend = create_backend("numpy")  # Universal CPU fallback
```

| Backend | Hardware | Key Advantage |
|---------|----------|---------------|
| **CUDA** | NVIDIA GPUs (T4, V100, A100) | Data center scale, Google Colab support |
| **MLX** | Apple Silicon (M1/M2/M3) | Unified memory + Metal GPU, lazy evaluation |
| **NumPy** | Universal CPU | Compatibility fallback |

### CUDA Performance Benchmarks (Tesla T4)

Benchmarked on Google Colab with Tesla T4 GPU (15GB VRAM):

| Knowledge Graph Size | CUDA (ms) | NumPy (ms) | Speedup |
|---------------------|-----------|------------|---------|
| 100 entities | 0.54 | 0.26 | 0.5x |
| 500 entities | 0.54 | 20.42 | **37.5x** |
| 1,000 entities | 1.37 | 181.62 | **132.5x** |
| 2,000 entities | 7.93 | 1,574.37 | **198.5x** |
| 5,000 entities | 59.57 | 42,167.71 | **707.8x** |

**Average speedup: 215x** for knowledge graph reasoning. See [Performance Benchmarks](docs/PERFORMANCE.md) for detailed metrics.

### Logical Reasoning in Tensors

```python
from tensorlogic.core import logical_and, logical_or, logical_not, logical_implies
from tensorlogic.core.quantifiers import exists, forall
from tensorlogic.backends import create_backend

backend = create_backend()

# Define relations as tensors (family knowledge graph)
# Rows = subject, Columns = object
import numpy as np
parent = np.array([
    [0., 1., 1., 0.],  # Alice is parent of Bob, Carol
    [0., 0., 0., 1.],  # Bob is parent of David
    [0., 0., 0., 0.],  # Carol has no children
    [0., 0., 0., 0.],  # David has no children
], dtype=np.float32)

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

# Auto-detection (recommended)
backend = create_backend()  # MLX -> CUDA -> NumPy

# Explicit backend selection
numpy_backend = create_backend("numpy")   # CPU reference implementation
mlx_backend = create_backend("mlx")       # Apple Silicon GPU
cuda_backend = create_backend("cuda")     # NVIDIA GPU (Colab, data centers)
```

**Lazy Evaluation (MLX):** Operations build computation graphs, executed on `backend.eval(result)`—critical for batching complex queries.

**CUDA Backend:** Uses CuPy for NVIDIA GPUs. Install with `pip install cupy-cuda12x` (Colab & modern GPUs) or `cupy-cuda11x` (legacy systems).

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

## FAQ

### Do I need to know logic notation (∀, ∃, →)?

**No.** TensorLogic uses plain Python. The mathematical notation in the docs is just for those who want to understand the theory. You can use the library without ever seeing a logic symbol.

### Do I need to understand tensors?

**Think spreadsheets.** A tensor is just a table of numbers. A 2D tensor is like an Excel sheet where:
- Rows = subjects (people, products, etc.)
- Columns = objects (other people, categories, etc.)
- Cell value = how strongly the relationship holds (0.0 to 1.0)

```
         Bob  Carol  David
Alice [  0.0   1.0    0.0 ]   ← "Alice is parent of Carol"
Bob   [  0.0   0.0    1.0 ]   ← "Bob is parent of David"
```

### How is this different from Prolog/Datalog?

| Feature | Prolog/Datalog | TensorLogic |
|---------|----------------|-------------|
| Execution | CPU, sequential | GPU, parallel |
| Uncertainty | No (true/false only) | Yes (0.0-1.0 confidence) |
| Gradients | No | Yes (train neural networks) |
| Temperature | No | Yes (control reasoning style) |

### How is this different from knowledge graph embeddings (TransE, etc.)?

| Feature | KG Embeddings | TensorLogic |
|---------|---------------|-------------|
| Interpretable | No (black box) | Yes (explicit rules) |
| Provable | No | Yes (T=0 gives exact logic) |
| Training required | Yes | Optional |
| Missing links | Predicted | Predicted (T>0) or not (T=0) |

**TensorLogic at T=0** = Classical logic solver (provable, no hallucinations)
**TensorLogic at T>0** = Embedding-style generalization (pattern-based inference)

### When should I use T=0 vs T>0?

```
T=0 (Deductive):
  ✓ Legal/medical/safety applications
  ✓ Database-style queries
  ✓ When false positives are costly
  ✓ Complete, reliable data

T>0 (Analogical):
  ✓ Recommendations and ranking
  ✓ Incomplete or uncertain data
  ✓ Training neural networks
  ✓ Exploratory analysis
```

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

- **Google Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mathews-Tom/TensorLogic/blob/main/notebooks/05_google_colab_cuda.ipynb) - Try TensorLogic on T4 GPU
- **Conceptual Guide:** [`docs/concepts/tensor-logic-mapping.md`](docs/concepts/tensor-logic-mapping.md) - How logic becomes tensors
- **Examples:** [`examples/README.md`](examples/README.md) - Practical usage examples
- **Backend API:** [`docs/backends/API.md`](docs/backends/API.md) - Comprehensive API reference
- **Research Goals:** [`docs/research/rag-goals.md`](docs/research/rag-goals.md) - RAG research roadmap
- **Original Paper:** arXiv:2510.12269 (Domingos, 2025)

### Jupyter Notebooks

1. **Getting Started:** `notebooks/01_getting_started.ipynb`
2. **Knowledge Graphs:** `notebooks/02_knowledge_graphs.ipynb`
3. **Compilation Strategies:** `notebooks/03_compilation_strategies.ipynb`
4. **Temperature Control:** `notebooks/04_temperature_control.ipynb`
5. **Google Colab (CUDA):** `notebooks/05_google_colab_cuda.ipynb`

## License

MIT License
