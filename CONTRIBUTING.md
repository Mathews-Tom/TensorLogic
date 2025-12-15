# Contributing to TensorLogic

Thank you for your interest in contributing to TensorLogic! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.12+
- uv (package manager)
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/TensorLogic.git
cd TensorLogic

# Install dependencies
uv sync

# Run tests to verify setup
uv run pytest
```

## Code Standards

### Type Hints

TensorLogic uses modern Python 3.12+ type hints. All code must:

- Include `from __future__ import annotations` at the top of each file
- Use built-in generics: `list[int]`, `dict[str, Any]`, `tuple[int, ...]`
- Use `|` for unions: `int | None`, `str | int`
- Avoid deprecated typing imports: `typing.List`, `typing.Optional`, `typing.Union`

```python
# Correct
from __future__ import annotations

def process(data: list[float] | None) -> dict[str, Any]:
    pass

# Incorrect
from typing import List, Optional, Dict, Any

def process(data: Optional[List[float]]) -> Dict[str, Any]:
    pass
```

### Docstrings

Use Google-style docstrings for all public functions and classes:

```python
def compute_scores(
    embeddings: np.ndarray,
    query: np.ndarray,
    temperature: float = 0.0,
) -> np.ndarray:
    """Compute similarity scores between embeddings and query.

    Args:
        embeddings: Entity embeddings array [num_entities, embedding_dim]
        query: Query vector [embedding_dim]
        temperature: Reasoning temperature (0=strict, >0=relaxed)

    Returns:
        Similarity scores [num_entities]

    Raises:
        ValueError: If embedding dimensions don't match

    Example:
        >>> scores = compute_scores(embeddings, query)
        >>> top_k = np.argsort(scores)[-10:]
    """
```

### Error Handling

- Fail fast with explicit errors
- No fallbacks or graceful degradation
- Provide clear, actionable error messages

```python
# Correct
if not is_valid(pattern):
    raise TensorLogicError(
        f"Invalid pattern: {pattern}",
        suggestion="Use format 'exists/forall var: predicate'"
    )

# Incorrect
try:
    parse(pattern)
except Exception:
    return None  # Silent failure
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=tensorlogic --cov-report=html

# Run specific test file
uv run pytest tests/test_core/test_operations.py

# Run single test
uv run pytest tests/test_core/test_operations.py::test_and_operation
```

### Writing Tests

- Place tests in `tests/` mirroring the source structure
- Use pytest fixtures for shared test data
- Include property-based tests using hypothesis for mathematical properties
- Test both success and error paths

```python
import pytest
import numpy as np
from hypothesis import given, strategies as st

from tensorlogic.core import logical_and


class TestLogicalAnd:
    """Tests for logical_and operation."""

    def test_basic_and(self) -> None:
        """Test basic AND operation."""
        a = np.array([1.0, 0.0, 1.0])
        b = np.array([1.0, 1.0, 0.0])
        result = logical_and(a, b)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    @given(st.lists(st.floats(0, 1), min_size=1, max_size=100))
    def test_and_commutativity(self, values: list[float]) -> None:
        """Property: AND is commutative."""
        a = np.array(values)
        b = np.random.rand(len(values))
        result1 = logical_and(a, b)
        result2 = logical_and(b, a)
        np.testing.assert_allclose(result1, result2)
```

### Coverage Requirements

- Minimum 90% line coverage
- All new code must have tests
- Coverage must not decrease with new changes

## Pull Request Process

### Before Submitting

1. **Run tests locally**
   ```bash
   uv run pytest
   ```

2. **Run type checker**
   ```bash
   uv run mypy src/tensorlogic
   ```

3. **Format code**
   ```bash
   uv run ruff format .
   uv run ruff check . --fix
   ```

4. **Update documentation if needed**

### PR Guidelines

- Create feature branches from `main`
- Use descriptive branch names: `feat/add-sparse-tensors`, `fix/relation-traversal`
- Keep PRs focused on a single change
- Include tests for new functionality
- Update relevant documentation

### Commit Messages

Follow conventional commits format:

```
feat(core): add sparse tensor support

Implement sparse tensor operations for memory-efficient
knowledge graph processing.

- Add SparseBackend protocol extension
- Implement sparse einsum for MLX
- Add tests for 1M+ entity graphs

Closes #123
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `chore`

## Architecture Overview

```
src/tensorlogic/
├── core/              # Core tensor logic primitives
│   ├── operations.py  # AND, OR, NOT, IMPLIES
│   └── quantifiers.py # EXISTS, FORALL
├── backends/          # Backend abstraction
│   ├── protocol.py    # TensorBackend Protocol
│   ├── mlx.py         # MLX implementation
│   └── numpy.py       # NumPy fallback
├── compilation/       # Compilation strategies
│   └── strategies/    # soft_differentiable, hard_boolean, etc.
├── api/               # High-level API
│   ├── quantify.py    # quantify() function
│   └── reason.py      # reason() with temperature
├── integrations/      # Framework integrations
│   ├── rag/           # RAG components
│   └── langchain/     # LangChain adapter
└── verification/      # Lean 4 integration
```

## Key Concepts

### Temperature-Controlled Reasoning

- `temperature=0`: Strict deductive reasoning (boolean logic)
- `temperature>0`: Soft analogical reasoning (fuzzy logic)

### Compilation Strategies

- `soft_differentiable`: Gradient-friendly for training
- `hard_boolean`: Strict boolean for inference
- `gödel`, `product`, `łukasiewicz`: T-norm based fuzzy logic

### Backend Protocol

All backends implement ~25 core operations. Use the protocol to ensure compatibility:

```python
from tensorlogic.backends import TensorBackend

def process(data: Any, backend: TensorBackend) -> Any:
    return backend.einsum('ij,jk->ik', data, weights)
```

## Getting Help

- **Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Documentation**: See `docs/` for detailed guides

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Welcome newcomers and help them contribute

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
