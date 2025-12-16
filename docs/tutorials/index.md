# TensorLogic Tutorials

Welcome to TensorLogic! These tutorials will guide you from basic concepts to advanced neural-symbolic reasoning.

---

## Choose Your Starting Point

### I'm New to AI and Logic
**Path:** Newcomers Guide → Hello World → Family Tree → Temperature Demo
**Time:** 30-60 minutes to productivity

1. [Newcomers Guide](00_newcomers_guide.md) - Zero-prerequisite introduction
2. `examples/00_hello_world.py` - Your first 10 lines
3. `examples/01_family_tree_minimal.py` - Multi-hop reasoning
4. `examples/02_temperature_demo.py` - The temperature dial

### I Know Prolog/Datalog
**Path:** From Datalog Guide → Temperature Guide → Knowledge Graph Example
**Time:** 20-30 minutes to productivity

1. [From Datalog](01_from_datalog.md) - Translation guide *(coming soon)*
2. [Temperature Guide](03_temperature_guide.md) - Temperature semantics
3. `examples/knowledge_graph_reasoning.py` - Full 8-entity example

### I Know PyTorch/Deep Learning
**Path:** From PyTorch Guide → Gradient Training → Compilation Strategies
**Time:** 20-30 minutes to productivity

1. [From PyTorch](02_from_pytorch.md) - Concept mapping *(coming soon)*
2. `examples/gradient_training.py` - Differentiable logic
3. `notebooks/03_compilation_strategies.ipynb` - Train-soft/infer-hard

### I'm a Researcher
**Path:** Conceptual Guide → Original Paper → Specs
**Time:** 2-3 hours for deep understanding

1. [Conceptual Guide](../concepts/tensor-logic-mapping.md) - Mathematical foundations
2. [arXiv:2510.12269](https://arxiv.org/abs/2510.12269) - Original paper
3. `docs/specs/` - Technical specifications

---

## Getting Help

**Something not working?** See [Troubleshooting Guide](04_troubleshooting.md)

---

## Getting Started

### Prerequisites

- Python 3.12+
- Basic Python knowledge (no logic/tensor background required)

### Installation

```bash
# Clone and install
git clone https://github.com/your-username/TensorLogic.git
cd TensorLogic
uv sync
```

## Tutorial Path

### 1. Introduction (Beginner)

**Notebook**: `notebooks/01_getting_started.ipynb`

Learn the fundamentals:
- Installing TensorLogic
- Basic logical operations (AND, OR, NOT)
- Creating your first knowledge graph
- Simple queries with `quantify()`

**Key concepts**: Tensor-logic mapping, backend abstraction

### 2. Knowledge Graph Reasoning (Intermediate)

**Notebook**: `notebooks/02_knowledge_graphs.ipynb`

Build reasoning systems:
- Multi-hop reasoning (grandparent inference)
- Relation composition (uncle/aunt relationships)
- EXISTS and FORALL quantifiers
- Temperature-controlled inference

**Key concepts**: Quantification, relation traversal, soft vs hard logic

### 3. Compilation Strategies (Intermediate)

**Notebook**: `notebooks/03_compilation_strategies.ipynb`

Master compilation:
- `soft_differentiable` for training
- `hard_boolean` for inference
- Train-soft/infer-hard pattern
- T-norms: Gödel, product, Łukasiewicz

**Key concepts**: Differentiable logic, T-norm semantics

### 4. Temperature Control (Advanced)

**Notebook**: `notebooks/04_temperature_control.ipynb`

Control reasoning behavior:
- T=0: Strict deductive reasoning
- T>0: Soft analogical reasoning
- Precision-recall tradeoffs
- Visual comparisons

**Key concepts**: Reasoning temperature, fuzzy logic

## Example Scripts

### Gradient Training

**File**: `examples/gradient_training.py`

Train neural networks through logical operations:
- Differentiable predicate learning
- Knowledge base embedding
- Loss computation via logical constraints

```bash
uv run python examples/gradient_training.py
```

### Real-World Datasets

**File**: `examples/freebase_reasoning.py`

Work with real knowledge graphs:
- FB15k-237 dataset subset
- 10K+ entity reasoning
- Performance benchmarking

```bash
uv run python examples/freebase_reasoning.py
```

### LangChain Integration

**File**: `examples/langchain_integration.py`

Integrate with RAG systems:
- TensorLogicLangChainRetriever
- Hybrid neural-symbolic scoring
- 10K+ entity demonstration

```bash
uv run python examples/langchain_integration.py
```

## Quick Reference

### Core Operations

```python
from tensorlogic.core import (
    logical_and,    # Conjunction: a ∧ b
    logical_or,     # Disjunction: a ∨ b
    logical_not,    # Negation: ¬a
    logical_implies,# Implication: a → b
    exists,         # ∃x: P(x)
    forall,         # ∀x: P(x)
)
```

### High-Level API

```python
from tensorlogic import quantify, reason

# Pattern-based quantification
result = quantify(
    'forall x: P(x) -> Q(x)',
    predicates={'P': pred_p, 'Q': pred_q}
)

# Temperature-controlled reasoning
result = reason(
    'exists y: Related(x, y) and HasProperty(y)',
    bindings={'x': entities},
    temperature=0.3  # Soft reasoning
)
```

### Backends

```python
from tensorlogic.backends import create_backend

# Auto-detect best backend (MLX on Apple Silicon, NumPy otherwise)
backend = create_backend()

# Explicit backend
from tensorlogic.backends import MLXBackend, NumpyBackend
mlx_backend = MLXBackend()
numpy_backend = NumpyBackend()
```

### Compilation Strategies

```python
from tensorlogic.compilation import create_strategy

# For training
soft = create_strategy('soft_differentiable')

# For inference
hard = create_strategy('hard_boolean')

# T-norms
godel = create_strategy('gödel')
product = create_strategy('product')
lukasiewicz = create_strategy('łukasiewicz')
```

### RAG Integration

```python
from tensorlogic.integrations import (
    TensorLogicRetriever,
    TensorLogicLangChainRetriever,
    create_langchain_retriever,
)

# Basic retriever
retriever = TensorLogicRetriever(
    temperature=0.0,
    lambda_neural=0.5,
)
retriever.index_entities(embeddings)

# LangChain adapter
lc_retriever = create_langchain_retriever(
    documents=docs,
    embedding_fn=embed,
)
results = lc_retriever.get_relevant_documents("query")
```

## Further Reading

- [Conceptual Guide](../concepts/tensor-logic-mapping.md): Deep dive into tensor-logic equivalence
- [RAG Research Goals](../research/rag-goals.md): Integration roadmap
- [API Documentation](../api/): Complete API reference
- [Original Paper](https://arxiv.org/abs/2510.12269): Domingos (2025) Tensor Logic

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-username/TensorLogic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/TensorLogic/discussions)
- **Contributing**: See [CONTRIBUTING.md](../../CONTRIBUTING.md)
