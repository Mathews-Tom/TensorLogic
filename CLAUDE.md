# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TensorLogic is a neural-symbolic AI framework that unifies logical reasoning and tensor computation. Based on Pedro Domingos' Tensor Logic paper (arXiv:2510.12269), it implements the mathematical equivalence between logical rules and Einstein summation.

**Current Status:** Core framework complete (97%). Active development on advanced features (sparse tensors, RAG integration, Lean 4 verification).

**Completed Components:**
- Backend abstraction: MLX + CUDA + NumPy with Protocol-based design (25+ operations)
- Core operations: AND, OR, NOT, IMPLIES with full test coverage
- Quantifiers: EXISTS, FORALL (hard and soft variants)
- Compilation strategies: soft_differentiable, hard_boolean, gödel, product, łukasiewicz
- Pattern API: `quantify()` and `reason()` with temperature control
- Test suite: 1,100+ tests with property-based testing (hypothesis)

**Core Concept:** Logical operations map to tensor operations:
- Logical AND → Hadamard product
- Logical OR → max operations
- Existential quantification → summation over axes
- Implications → `max(1-a, b)`
- Temperature-controlled reasoning (T=0 for deductive, higher for analogical)

## Package Management & Development Commands

This project uses `uv` as the package manager. **Never use `pip` directly.**

```bash
# Add dependencies
uv add <package>              # Production dependency
uv add --dev <package>        # Development dependency

# Run commands with environment
uv run <command>              # Execute with uv-managed environment
uv sync                       # Sync dependencies from lock file

# Testing
uv run pytest                 # Run all tests
uv run pytest tests/test_core/test_operations.py  # Single test file
uv run pytest tests/test_core/test_operations.py::test_and_operation  # Single test
uv run pytest --cov=tensorlogic --cov-report=html  # With coverage

# Type checking
uv run mypy src/tensorlogic   # Type check source

# Linting and formatting
uv run ruff check .           # Lint
uv run ruff format .          # Format code
```

## Architecture

### Directory Structure (Implemented)

```
src/tensorlogic/
├── __init__.py           # Package entry point with top-level exports
├── py.typed              # PEP 561 type marker
├── backends/             # ✅ COMPLETE - Backend abstraction
│   ├── __init__.py
│   ├── protocol.py       # TensorBackend Protocol (25+ operations)
│   ├── mlx.py            # MLX implementation (Apple Silicon)
│   ├── cuda.py           # CUDA implementation (NVIDIA GPUs via CuPy)
│   ├── numpy.py          # NumPy fallback
│   └── factory.py        # create_backend() factory (auto-detects best)
├── core/                 # ✅ COMPLETE - Core tensor logic primitives
│   ├── __init__.py
│   ├── operations.py     # logical_and, logical_or, logical_not, logical_implies
│   ├── quantifiers.py    # exists, forall (hard & soft variants)
│   ├── composition.py    # Predicate composition
│   └── temperature.py    # Temperature-scaled reasoning
├── api/                  # ✅ COMPLETE - High-level einops-style API
│   ├── __init__.py
│   ├── patterns.py       # quantify() and reason() functions
│   ├── parser.py         # PatternParser with AST
│   ├── validation.py     # PatternValidator
│   ├── errors.py         # TensorLogicError hierarchy
│   └── compiler.py       # Pattern compilation & caching
├── compilation/          # ✅ COMPLETE - Multiple semantic strategies
│   ├── __init__.py
│   ├── protocol.py       # CompilationStrategy Protocol
│   ├── factory.py        # create_strategy()
│   └── strategies/       # 5 compilation strategies
│       ├── hard.py       # HardBooleanStrategy
│       ├── soft.py       # SoftDifferentiableStrategy
│       ├── godel.py      # GodelFuzzyStrategy
│       ├── product.py    # ProductFuzzyStrategy
│       └── lukasiewicz.py # LukasiewiczFuzzyStrategy
└── verification/         # 🔄 IN PROGRESS - Lean 4 integration
    ├── __init__.py
    ├── lean_bridge.py    # LeanBridge class (placeholder implementation)
    └── results.py        # VerificationResult dataclass
```

### Key Design Principles

**1. Multi-GPU Backend Strategy**
- Auto-detection priority: MLX (Apple Silicon) → CUDA (NVIDIA) → NumPy (CPU)
- MLX: Apple Silicon optimized with lazy evaluation, unified memory
- CUDA: NVIDIA GPU support via CuPy (up to 700x speedup for large KGs)
- NumPy: Universal CPU fallback for compatibility
- Protocol-based abstraction (~25 operations, minimal overhead)

**2. Einops-Style API Design**
String-based pattern notation for self-documenting operations:

```python
from tensorlogic import quantify, reason, create_backend

backend = create_backend()  # Auto-selects MLX → CUDA → NumPy

# Pattern-based quantified queries
result = quantify(
    'forall x: P(x) -> Q(x)',
    predicates={'P': predicate_p, 'Q': predicate_q},
    backend=backend
)

# Temperature-controlled reasoning
inference = reason(
    'exists y: Related(x, y) and HasProperty(y)',
    bindings={'x': entity_batch},
    temperature=0.0,  # 0.0 = deductive, higher = analogical
    backend=backend
)
```

**3. Enhanced Error Messages**
Implement TensorSensor-style error visualization:

```python
# Instead of: "RuntimeError: size mismatch, m1: [764 x 256], m2: [764 x 200]"
# Provide:
TensorLogicError: Predicate composition failed
  Predicate 'HasProperty' expects embedding dim 256
  Received tensor with shape [batch=764, dim=200]
  Context: quantify('exists y: Related(x, y) and HasProperty(y)', ...)
                                              ^^^^^^^^^^^
  Suggestion: Check HasProperty's input dimension matches Related's output
```

**4. Lean 4 Integration (Future Differentiator)**
- First neural-symbolic framework with formal verification
- LeanDojo Python bridge for bidirectional communication
- Verify tensor operations against theorems
- Proof-guided learning

## Type System Requirements

**Mandatory modern Python 3.12+ type hints:**

```python
from __future__ import annotations
from typing import Protocol, Self, Any
from collections.abc import Callable

# ✅ REQUIRED - Modern syntax
def process(data: list[float] | None) -> dict[str, Any] | None:
    pass

# ❌ FORBIDDEN - Deprecated syntax
from typing import List, Dict, Optional, Union
def old_process(data: Optional[List[float]]) -> Union[Dict[str, Any], None]:
    pass
```

**Type hint rules:**
- Built-in generics: `list[int]`, `dict[str, Any]`, `tuple[int, ...]`, `set[str]`
- Union syntax: `int | None`, `str | int`
- Keep from typing: `Any`, `Protocol`, `TypeAlias`, `Self`, `Literal`, `TypeVar`, `Final`, `ClassVar`
- From collections.abc: `Callable`, `Mapping`, `Sequence`, `Iterable`
- All public functions/classes must have type hints
- Include `py.typed` marker file for PEP 561 compliance

## Testing Requirements

**Framework:** pytest + hypothesis

**Coverage:** Minimum 90% line coverage required

**Testing patterns:**

```python
# Property-based tests for mathematical operations
from hypothesis import given, strategies as st

@given(
    a=st.lists(st.floats(min_value=-100, max_value=100), min_size=1, max_size=100),
    b=st.lists(st.floats(min_value=-100, max_value=100), min_size=1, max_size=100),
)
def test_operation_commutativity(a: list[float], b: list[float]) -> None:
    """Property: op(a, b) == op(b, a) for commutative operations."""
    pass

# Parametrized tests for edge cases
@pytest.mark.parametrize(
    "pattern,expected_shape",
    [
        ("exists x: P(x)", (1,)),
        ("forall x: P(x) -> Q(x)", (1,)),
    ],
)
def test_quantifier_shapes(pattern: str, expected_shape: tuple[int, ...]) -> None:
    pass
```

**Test organization:**
- `tests/test_core/` - Core operations
- `tests/test_backends/` - Backend implementations
- `tests/test_api/` - High-level API
- All error paths must be tested
- Mathematical properties tested with hypothesis

## Error Handling Philosophy

**Fail fast. No fallbacks or graceful degradation.**

```python
# ✅ REQUIRED - Explicit errors
def validate_pattern(pattern: str) -> None:
    if not is_valid(pattern):
        raise TensorLogicError(
            "Invalid pattern syntax",
            context=f"Pattern: {pattern}",
            suggestion="Use format 'exists/forall var: predicate'"
        )

# ❌ FORBIDDEN - Silent failures
def bad_validate(pattern: str) -> None:
    try:
        parse(pattern)
    except Exception:
        return None  # Silent failure - NEVER DO THIS
```

## Code Organization Standards

**Module structure:**

```python
from __future__ import annotations

# Standard library imports
import sys
from pathlib import Path

# Third-party imports
import mlx.core as mx

# Local imports
from tensorlogic.core import operations
from tensorlogic.backends import TensorBackend

# Constants
DEFAULT_TEMPERATURE: float = 0.0

# Type aliases
TensorShape: TypeAlias = tuple[int, ...]

# Public API
__all__ = ["quantify", "reason"]

# Implementation
...
```

**Docstring format (Google style):**

```python
def quantify(
    pattern: str,
    *,
    predicates: dict[str, Any] | None = None,
) -> Any:
    """Execute quantified logic pattern.

    Args:
        pattern: Logic formula as string (e.g., 'exists y: P(x, y)')
        predicates: Named predicates to use in pattern

    Returns:
        Result tensor after pattern evaluation

    Raises:
        TensorLogicError: If pattern syntax invalid or shapes incompatible

    Examples:
        >>> quantify('forall x: P(x)', predicates={'P': pred})
    """
```

## MLX Backend Specifics

**Critical: MLX uses lazy evaluation**

```python
import mlx.core as mx

# Must explicitly evaluate lazy tensors
result = mx.einsum('ij,jk->ik', a, b)
mx.eval(result)  # Force evaluation

# TensorBackend protocol must include eval()
class TensorBackend(Protocol):
    def eval(self, *arrays: Any) -> None:
        """Force evaluation of lazy tensors (critical for MLX)."""
        ...
```

**Performance targets:**
- Development (M1 Pro): Batch sizes 4-32, 8B models in BF16
- Focus on developer experience and correctness over raw speed

## CUDA Backend Specifics

**CuPy-based implementation for NVIDIA GPUs:**

```python
import cupy as cp

# CuPy mirrors NumPy API for easy porting
result = cp.einsum('ij,jk->ik', a, b)

# Unlike MLX, CuPy evaluates eagerly (no manual eval needed)
# Memory management is automatic with GPU memory pool
```

**Installation options:**
```bash
pip install cupy-cuda12x  # CUDA 12.x (Google Colab, modern GPUs)
pip install cupy-cuda11x  # CUDA 11.x (legacy systems)
```

**Performance benchmarks (Tesla T4):**

| Knowledge Graph Size | CUDA (ms) | NumPy (ms) | Speedup |
|---------------------|-----------|------------|---------|
| 500 entities | 0.54 | 20.42 | **37.5x** |
| 1,000 entities | 1.37 | 181.62 | **132.5x** |
| 5,000 entities | 59.57 | 42,167.71 | **707.8x** |

**Best for:** Large knowledge graphs (1K+ entities), data center deployments, Google Colab

## Implementation Phases

### Completed (98%)
1. ✅ **Core operations** - Tensor-to-logic primitives with multi-backend support
2. ✅ **Pattern language** - Einops-style string patterns for logical formulas
3. ✅ **Compilation strategies** - Boolean, fuzzy (Gödel, product, Łukasiewicz), differentiable semantics
4. ✅ **Developer tools** - Enhanced errors, type stubs (py.typed), documentation
5. ✅ **CUDA backend** - NVIDIA GPU support via CuPy (up to 700x speedup, benchmarked on T4)

### In Progress
6. 🔄 **Lean 4 bridge** - LeanDojo integration for verified operations (skeleton implemented)
7. 🔄 **Sparse tensors** - Support for 1M+ entity knowledge graphs
8. 🔄 **RAG integration** - Scalable symbolic-aware retrieval

### Planned
9. ⏳ **Proof-guided learning** - Train neural components with theorem prover feedback

## Anti-Patterns to Avoid

**From competitor analysis (cool-japan/tensorlogic):**
- ❌ Over-modularization with too many interconnected modules
- ❌ Custom backend lock-in (use standard frameworks)
- ❌ GPU backend as "future work" (TensorLogic: MLX + CUDA production-ready)

**From existing neural-symbolic frameworks:**
- ❌ Poor error messages (implement TensorSensor-style errors)
- ❌ Heavy compatibility layers (use minimal Protocol abstraction)
- ❌ CPU-only operations (leverage GPU from start)

## Documentation References

- **Vision Document:** `docs/TensorLogic-Overview.md` - Comprehensive strategic assessment
- **Conceptual Guide:** `docs/concepts/tensor-logic-mapping.md` - Tensor-to-logic mappings
- **Performance:** `docs/PERFORMANCE.md` - MLX and CUDA benchmark results
- **RAG Research:** `docs/research/rag-goals.md` - RAG integration roadmap
- **Backend API:** `docs/backends/API.md` - Backend protocol reference
- **Examples:** `examples/README.md` - Working code examples
- **Colab Notebook:** `notebooks/05_google_colab_cuda.ipynb` - CUDA testing on T4 GPU
- **Original Paper:** arXiv:2510.12269 (Domingos, 2025)

## Sage-Dev Integration

This repository uses the sage-dev system for development workflow:

```bash
# Workflow selection
/sage.workflow          # Choose Traditional vs Ticket-Based workflow

# Traditional workflow
/sage.init-feature      # Create feature request
/sage.intel             # Research and analyze
/sage.specify           # Generate specifications
/sage.plan              # Create implementation plan

# Ticket-based workflow
/sage.migrate           # Set up ticket system
/sage.tasks             # Generate SMART tasks
/sage.implement         # Execute tickets
```

Pattern templates in `.sage/agent/examples/python/` provide implementation guidance.
