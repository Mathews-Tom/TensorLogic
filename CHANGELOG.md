# Changelog

All notable changes to TensorLogic are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2025-12-16

### Added

- **CUDA Backend** (`tensorlogic.backends.cuda`)
  - CuPy-based implementation for NVIDIA GPUs
  - Full Protocol compliance with ~25 operations
  - Auto-detection via `create_backend()` (MLX → CUDA → NumPy priority)
  - Explicit selection: `create_backend("cuda")`
  - Google Colab compatible with `cupy-cuda12x`
  - Benchmarked on Tesla T4 GPU (15GB VRAM, CUDA 12.4)

- **CUDA Performance Benchmarks**
  - Knowledge graph reasoning: up to **707.8x speedup** vs NumPy at 5,000 entities
  - Average speedup: **215.4x** across tested scales (100-5,000 entities)
  - Large-scale demo: 10,000 entities with 99,917 edges
  - 2-hop inference: 485.4ms, 3-hop inference: 8,217.1ms
  - GPU memory usage: 381.5MB for 10K×10K relation matrix

- **Google Colab Notebook** (`notebooks/05_google_colab_cuda.ipynb`)
  - Complete CUDA backend testing and benchmarking
  - Relation composition benchmarks at multiple scales
  - Quantified query examples
  - Large-scale knowledge graph inference demo

- **Lean 4 Verification**: Formal verification of 15 logical operation theorems
  - Commutativity and associativity of AND/OR
  - De Morgan's laws and idempotence
  - Identity and annihilator properties
  - Quantifier negation (negation of EXISTS equals FORALL of NOT)
  - LeanDojo bridge for theorem verification

- **RAG Integration Module** (`tensorlogic.integrations`)
  - `TensorLogicRetriever`: Core symbolic-aware retrieval
  - `HybridRanker`: Neural-symbolic hybrid scoring with RRF fusion
  - `ConstraintFilter`: Logical constraint filtering with EXISTS/FORALL
  - `TensorLogicLangChainRetriever`: LangChain BaseRetriever adapter
  - `create_langchain_retriever`: Factory function for easy setup

- **Sparse Tensor Support**
  - `sparse_tensor()` method in backend protocol
  - `sparse_einsum()` for memory-efficient computation
  - Coordinate (COO) format support
  - Enables 1M+ entity knowledge graphs

- **Benchmark Suite** (`benchmarks/`)
  - Scale tests (1K to 1M entities)
  - Memory profiling
  - Strategy comparison benchmarks
  - MLX vs NumPy performance comparison

- **Jupyter Notebooks** (`notebooks/`)
  - `01_getting_started.ipynb`: Installation and first knowledge graph
  - `02_knowledge_graphs.ipynb`: Multi-hop reasoning patterns
  - `03_compilation_strategies.ipynb`: Train-soft/infer-hard pattern
  - `04_temperature_control.ipynb`: Deductive vs analogical reasoning

- **Examples**
  - `gradient_training.py`: Neural network training through logical operations
  - `freebase_reasoning.py`: Real-world knowledge graph (FB15k-237 subset)
  - `langchain_integration.py`: 10K+ entity LangChain demo

### Changed

- Updated top-level exports in `__init__.py` for easier imports
  - Can now use `from tensorlogic import quantify, reason`
  - Core operations directly accessible at package level

### Fixed

- Array conversion between MLX and NumPy backends
- Type hints updated to Python 3.12+ syntax throughout codebase

## [0.1.0] - 2024-12-15

### Added

- **Core Operations** (`tensorlogic.core`)
  - `logical_and`: Tensor-based conjunction (Hadamard product)
  - `logical_or`: Tensor-based disjunction (max operation)
  - `logical_not`: Tensor-based negation (1 - x)
  - `logical_implies`: Tensor-based implication (max(1-a, b))
  - `exists`: Existential quantification (summation)
  - `forall`: Universal quantification (product)

- **Backend Abstraction** (`tensorlogic.backends`)
  - `TensorBackend` Protocol with ~25 core operations
  - `MLXBackend`: Apple Silicon optimized (lazy evaluation)
  - `NumpyBackend`: Reference implementation and fallback
  - `create_backend()`: Auto-detection factory

- **Compilation Strategies** (`tensorlogic.compilation`)
  - `soft_differentiable`: Gradient-friendly (sigmoid-based)
  - `hard_boolean`: Strict boolean (threshold-based)
  - `gödel`: Gödel t-norm (min/max)
  - `product`: Product t-norm (multiplication)
  - `lukasiewicz`: Łukasiewicz t-norm (bounded)
  - `create_strategy()`: Strategy factory function

- **High-Level API** (`tensorlogic.api`)
  - `quantify()`: Pattern-based quantification
  - `reason()`: Temperature-controlled reasoning

- **Verification Module** (`tensorlogic.verification`)
  - Lean 4 project structure
  - `lean_bridge.py`: LeanDojo integration
  - `theorems.py`: Theorem registry

- **Examples**
  - `compilation_strategies.py`: Strategy comparison demo
  - `knowledge_graph_reasoning.py`: Family tree reasoning

### Documentation

- Comprehensive README with quickstart guide
- CLAUDE.md for AI assistant guidance
- docs/TensorLogic-Overview.md: Vision and strategy
- docs/concepts/tensor-logic-mapping.md: Conceptual guide
- docs/research/rag-goals.md: RAG integration roadmap

### Testing

- 1,200+ tests with 99%+ pass rate
- Property-based testing with hypothesis
- Cross-backend validation tests
- Type checking with mypy --strict

[Unreleased]: https://github.com/Mathews-Tom/TensorLogic/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Mathews-Tom/TensorLogic/compare/v0.1.0...v0.3.0
[0.1.0]: https://github.com/Mathews-Tom/TensorLogic/releases/tag/v0.1.0
