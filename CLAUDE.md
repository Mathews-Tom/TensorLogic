# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TensorLogic is a neural-symbolic AI framework that unifies logical reasoning and tensor computation. Based on Pedro Domingos' Tensor Logic paper (arXiv:2510.12269), it implements the mathematical equivalence between logical rules and Einstein summation.

**Current Status:** Pre-implementation planning phase. No source code exists yet, but comprehensive architecture is documented.

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

# Testing (once implemented)
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

### Planned Directory Structure

```
src/tensorlogic/
├── core/              # Core tensor logic primitives
│   ├── operations.py  # Logical ops as tensors (AND, OR, implications)
│   ├── quantifiers.py # Existential/universal quantification
│   └── semantics.py   # Boolean, fuzzy, differentiable semantics
├── backends/          # Backend abstraction (Protocol-based, ~25-30 ops)
│   ├── protocol.py    # TensorBackend Protocol
│   ├── mlx.py         # MLX implementation (primary)
│   └── numpy.py       # NumPy fallback
├── api/               # High-level einops-style API
│   ├── patterns.py    # Pattern language parser
│   ├── quantify.py    # quantify() function
│   └── reason.py      # reason() with temperature control
├── verification/      # Lean 4 integration (future)
│   ├── bridge.py      # LeanDojo bridge
│   └── theorems.py    # Verified tensor operation theorems
└── utils/
    ├── errors.py      # Enhanced error messages (TensorLogicError)
    └── validation.py  # Runtime validation
```

### Key Design Principles

**1. MLX-First Backend Strategy**
- Primary backend: MLX (Apple Silicon optimized, lazy evaluation)
- Scaling path: MLX CUDA backend for production
- Fallback: NumPy for compatibility
- Protocol-based abstraction (~25-30 operations, not heavy compatibility layer)

**2. Einops-Style API Design**
String-based pattern notation for self-documenting operations:

```python
# API style to implement
quantify(
    'forall x in batch: P(x) -> Q(x)',
    predicates={'P': predicate_p, 'Q': predicate_q},
    aggregator='product'
)

reason(
    'exists y: Related(x, y) and HasProperty(y)',
    bindings={'x': entity_batch},
    temperature=0.0  # 0.0 = deductive, higher = analogical
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

## Implementation Phases

1. **Core operations** - Tensor-to-logic primitives with MLX backend
2. **Pattern language** - Einops-style string patterns for logical formulas
3. **Compilation strategies** - Boolean, fuzzy (Gödel, product, Łukasiewicz), differentiable semantics
4. **Developer tools** - Enhanced errors, type stubs, documentation
5. **Lean 4 bridge** - LeanDojo integration for verified operations
6. **CUDA scaling** - Test and optimize MLX CUDA backend
7. **Proof-guided learning** - Train neural components with theorem prover feedback

## Anti-Patterns to Avoid

**From competitor analysis (cool-japan/tensorlogic):**
- ❌ Over-modularization with too many interconnected modules
- ❌ Custom backend lock-in (use standard frameworks)
- ❌ GPU backend as "future work" (MLX supports it now)

**From existing neural-symbolic frameworks:**
- ❌ Poor error messages (implement TensorSensor-style errors)
- ❌ Heavy compatibility layers (use minimal Protocol abstraction)
- ❌ CPU-only operations (leverage GPU from start)

## Documentation References

- **Vision Document:** `docs/TensorLogic-Overview.md` - Comprehensive strategic assessment
- **Architecture:** `.sage/agent/system/architecture.md` - System architecture
- **Tech Stack:** `.sage/agent/system/tech-stack.md` - Technology decisions
- **Code Patterns:** `.sage/agent/examples/python/` - Template patterns (pre-implementation)
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
