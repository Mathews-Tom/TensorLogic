# TensorLogic

Neural-symbolic AI framework unifying logical reasoning and tensor computation. Bridge neural networks and symbolic reasoning through tensor operations based on Pedro Domingos' Tensor Logic paper (arXiv:2510.12269).

## Quick Start

### Installation

**Basic Installation (NumPy backend):**
```bash
uv add tensorlogic
```

**Recommended (MLX backend for Apple Silicon):**
```bash
uv add tensorlogic mlx>=0.30.0
```

### Basic Usage

```python
from tensorlogic.backends import create_backend

# Default: try MLX, fallback to NumPy
backend = create_backend()

# Create tensors
x = backend.zeros((2, 3))
y = backend.ones((2, 3))

# Tensor operations
result = backend.einsum('ij,ij->ij', x, y)
backend.eval(result)  # Force evaluation (critical for MLX)
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

# Explicit NumPy backend
numpy_backend = create_backend("numpy")

# Explicit MLX backend (raises if unavailable)
mlx_backend = create_backend("mlx")
```

### Lazy Evaluation (MLX)

MLX uses lazy evaluation - operations are not computed until explicitly evaluated:

```python
backend = create_backend("mlx")

# These operations are lazy - not computed yet
a = backend.ones((100, 100))
b = backend.zeros((100, 100))
result = backend.einsum('ij,jk->ik', a, b)

# Force evaluation
backend.eval(result)  # Now computed
```

### Backend Protocol

All backends implement the `TensorBackend` Protocol with these operations:

**Creation:**
- `zeros(shape)` - Zero-filled tensor
- `ones(shape)` - One-filled tensor
- `arange(start, stop)` - Sequential values
- `full(shape, fill_value)` - Constant-filled tensor

**Transformation:**
- `reshape(tensor, shape)` - Change tensor shape
- `broadcast_to(tensor, shape)` - Broadcast to shape
- `transpose(tensor, axes)` - Permute axes
- `squeeze(tensor, axis)` - Remove size-1 dimensions
- `expand_dims(tensor, axis)` - Add size-1 dimension

**Operations:**
- `einsum(pattern, *tensors)` - Einstein summation
- `maximum(a, b)` - Element-wise maximum
- `add(a, b)` - Element-wise addition
- `subtract(a, b)` - Element-wise subtraction
- `multiply(a, b)` - Element-wise multiplication
- `divide(a, b)` - Element-wise division
- `matmul(a, b)` - Matrix multiplication
- `sum(tensor, axis, keepdims)` - Sum reduction
- `max(tensor, axis, keepdims)` - Max reduction
- `min(tensor, axis, keepdims)` - Min reduction
- `mean(tensor, axis, keepdims)` - Mean reduction
- `prod(tensor, axis, keepdims)` - Product reduction

**Utilities:**
- `eval(*tensors)` - Force evaluation (MLX lazy execution)
- `step(x)` - Heaviside step function
- `clip(tensor, min, max)` - Clamp values
- `abs(tensor)` - Absolute value
- `exp(tensor)` - Exponential
- `log(tensor)` - Natural logarithm
- `sqrt(tensor)` - Square root
- `power(tensor, exponent)` - Power operation
- `astype(tensor, dtype)` - Type conversion
- `asarray(data)` - Convert to tensor

## Examples

TensorLogic includes practical examples demonstrating neural-symbolic reasoning:

### Compilation Strategies

```bash
uv run python examples/compilation_strategies.py
```

Compare soft, hard, Godel, product, and Lukasiewicz semantics for logical operations.

### Knowledge Graph Reasoning

```bash
uv run python examples/knowledge_graph_reasoning.py
```

Comprehensive example demonstrating:
- Family knowledge graph with 8 entities
- Logical operations (AND, OR, NOT, IMPLIES)
- Relation inference (Grandparent, Aunt/Uncle rules)
- Quantified queries (EXISTS, FORALL)
- Temperature-controlled reasoning (deductive vs analogical)
- Compilation strategy comparison
- Uncertain knowledge handling

See [`examples/README.md`](examples/README.md) for detailed documentation.

## Development

### Running Tests

```bash
# All tests
uv run pytest tests/test_backends/

# With coverage
uv run pytest tests/test_backends/ --cov=tensorlogic.backends --cov-report=html

# Single test file
uv run pytest tests/test_backends/test_mlx.py

# Specific test
uv run pytest tests/test_backends/test_mlx.py::test_einsum_matrix_multiply
```

### Type Checking

```bash
# Strict type checking
uv run mypy --strict src/tensorlogic/backends/

# Current status: 0 errors
```

### Code Quality

```bash
# Linting
uv run ruff check .

# Formatting
uv run ruff format .
```

## Project Status

**Current Phase:** Backend Implementation Complete

**Completed:**
- TensorBackend Protocol definition
- NumPy backend implementation (100% coverage)
- MLX backend implementation (100% coverage)
- Factory pattern with validation
- Cross-backend validation tests
- Performance benchmarks
- Production readiness (99% coverage, 0 mypy errors)

**Next Phase:** Core Logic Implementation (CORE-001)
- Logical operations (AND, OR, NOT, IMPLIES)
- Quantifiers (EXISTS, FORALL)
- Temperature-controlled reasoning

## Documentation

- **Examples**: [`examples/README.md`](examples/README.md) (practical usage examples)
- **Backend API**: `docs/backends/API.md` (comprehensive API reference)
- **Architecture**: `.sage/agent/system/architecture.md`
- **Tech Stack**: `.sage/agent/system/tech-stack.md`
- **Original Paper**: arXiv:2510.12269 (Domingos, 2025)

## License

MIT License
