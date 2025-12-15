# Backend API Reference

Comprehensive API documentation for TensorLogic's backend abstraction layer.

## Overview

TensorLogic uses a Protocol-based backend abstraction with ~25-30 core operations, following the einops philosophy of minimal abstraction. This design allows seamless switching between MLX (Apple Silicon), CUDA (NVIDIA GPUs), and NumPy (CPU fallback) backends.

| Backend | Hardware | Performance | Use Case |
|---------|----------|-------------|----------|
| **CUDA** | NVIDIA GPUs | Up to 700x vs CPU | Production, data centers, Google Colab |
| **MLX** | Apple Silicon | Up to 100x vs CPU | Development, Apple devices |
| **NumPy** | Any CPU | Baseline | Testing, compatibility, small graphs |

## Table of Contents

1. [Factory Functions](#factory-functions)
2. [TensorBackend Protocol](#tensorbackend-protocol)
3. [Backend Implementations](#backend-implementations)
4. [Validation](#validation)
5. [Type System](#type-system)
6. [Error Handling](#error-handling)

---

## Factory Functions

### `create_backend(name: str = "auto") -> TensorBackend`

Create a tensor backend by name with graceful fallback.

**Parameters:**
- `name` (str, optional): Backend identifier. Options: `"auto"`, `"cuda"`, `"mlx"`, `"numpy"`. Defaults to `"auto"`.

**Returns:**
- `TensorBackend`: Backend instance conforming to TensorBackend protocol

**Raises:**
- `ValueError`: If backend name is unknown or required dependencies are missing
- `ImportError`: If requested backend is unavailable

**Behavior:**
- When `name="auto"`: Auto-detects best available backend in priority order: MLX → CUDA → NumPy
- When `name="cuda"`: Creates CUDA backend using CuPy. Requires NVIDIA GPU and CuPy installation.
- When `name="mlx"`: Attempts to import MLX backend. If MLX is unavailable, issues a warning and falls back to NumPy.
- When `name="numpy"`: Directly creates NumPy backend. Raises if NumPy unavailable.
- All created backends are validated against the TensorBackend protocol before returning.

**Examples:**

```python
from tensorlogic.backends import create_backend

# Auto-detect best available backend (recommended)
backend = create_backend()  # or create_backend("auto")

# Explicitly request CUDA (NVIDIA GPUs)
cuda_backend = create_backend("cuda")

# Explicitly request MLX (Apple Silicon)
mlx_backend = create_backend("mlx")

# Explicitly request NumPy (CPU fallback)
numpy_backend = create_backend("numpy")
```

**Installation by Backend:**

```bash
# CUDA backend (NVIDIA GPUs)
pip install cupy-cuda12x  # CUDA 12.x (recommended, Google Colab)
pip install cupy-cuda11x  # CUDA 11.x (legacy systems)

# MLX backend (Apple Silicon)
pip install mlx>=0.30.0

# NumPy backend (included by default)
pip install numpy>=1.24.0
```

---

### `validate_backend(backend: Any) -> None`

Validate an object implements the TensorBackend protocol.

**Parameters:**
- `backend` (Any): Object to validate

**Returns:**
- None (raises on validation failure)

**Raises:**
- `TypeError`: If backend doesn't implement all required TensorBackend protocol methods

**Purpose:**
Runtime protocol checking to ensure backend provides all required operations. Critical for catching incomplete implementations since Protocol checking is structural.

**Examples:**

```python
from tensorlogic.backends import validate_backend, create_backend

# Valid backend passes silently
backend = create_backend()
validate_backend(backend)  # No error

# Invalid backend raises TypeError
class IncompleteBackend:
    def einsum(self, pattern: str, *tensors: Any) -> Any:
        pass
    # Missing other required methods

validate_backend(IncompleteBackend())
# Raises: TypeError: Backend validation failed: IncompleteBackend doesn't implement TensorBackend protocol...
```

---

## TensorBackend Protocol

The `TensorBackend` Protocol defines the interface all backends must implement. It uses Python's structural typing (Protocol) for duck-typing compatibility.

### Tensor Creation Operations

#### `zeros(shape: tuple[int, ...]) -> Any`

Create a tensor filled with zeros.

**Parameters:**
- `shape` (tuple[int, ...]): Shape of output tensor

**Returns:**
- Tensor of specified shape filled with 0.0

**Example:**
```python
backend = create_backend()
x = backend.zeros((2, 3))  # 2x3 tensor of zeros
```

---

#### `ones(shape: tuple[int, ...]) -> Any`

Create a tensor filled with ones.

**Parameters:**
- `shape` (tuple[int, ...]): Shape of output tensor

**Returns:**
- Tensor of specified shape filled with 1.0

**Example:**
```python
backend = create_backend()
x = backend.ones((3, 3))  # 3x3 tensor of ones
```

---

#### `arange(start: float, stop: float) -> Any`

Create a 1D tensor with sequential values.

**Parameters:**
- `start` (float): Starting value (inclusive)
- `stop` (float): Ending value (exclusive)

**Returns:**
- 1D tensor with values [start, start+1, ..., stop-1]

**Example:**
```python
backend = create_backend()
x = backend.arange(0.0, 5.0)  # [0., 1., 2., 3., 4.]
```

---

#### `full(shape: tuple[int, ...], fill_value: float) -> Any`

Create a tensor filled with a constant value.

**Parameters:**
- `shape` (tuple[int, ...]): Shape of output tensor
- `fill_value` (float): Value to fill tensor with

**Returns:**
- Tensor of specified shape filled with `fill_value`

**Example:**
```python
backend = create_backend()
x = backend.full((2, 2), 3.14)  # 2x2 tensor filled with 3.14
```

---

### Shape Manipulation Operations

#### `reshape(tensor: Any, shape: tuple[int, ...]) -> Any`

Reshape a tensor to a new shape.

**Parameters:**
- `tensor` (Any): Input tensor
- `shape` (tuple[int, ...]): New shape (must have same total elements)

**Returns:**
- Tensor with new shape

**Example:**
```python
backend = create_backend()
x = backend.ones((6,))
y = backend.reshape(x, (2, 3))  # Shape: (2, 3)
```

---

#### `broadcast_to(tensor: Any, shape: tuple[int, ...]) -> Any`

Broadcast tensor to a new shape.

**Parameters:**
- `tensor` (Any): Input tensor
- `shape` (tuple[int, ...]): Target shape (must be broadcast-compatible)

**Returns:**
- Broadcasted tensor

**Example:**
```python
backend = create_backend()
x = backend.ones((3,))
y = backend.broadcast_to(x, (4, 3))  # Shape: (4, 3)
```

---

#### `transpose(tensor: Any, axes: tuple[int, ...] | None = None) -> Any`

Permute tensor axes.

**Parameters:**
- `tensor` (Any): Input tensor
- `axes` (tuple[int, ...] | None): Permutation of axes. If None, reverses all axes.

**Returns:**
- Transposed tensor

**Example:**
```python
backend = create_backend()
x = backend.ones((2, 3, 4))
y = backend.transpose(x, (2, 0, 1))  # Shape: (4, 2, 3)
z = backend.transpose(x)  # Shape: (4, 3, 2) - reversed axes
```

---

#### `squeeze(tensor: Any, axis: int | None = None) -> Any`

Remove size-1 dimensions.

**Parameters:**
- `tensor` (Any): Input tensor
- `axis` (int | None): Axis to squeeze. If None, squeeze all size-1 dimensions.

**Returns:**
- Squeezed tensor

**Example:**
```python
backend = create_backend()
x = backend.ones((2, 1, 3, 1))
y = backend.squeeze(x)  # Shape: (2, 3)
z = backend.squeeze(x, axis=1)  # Shape: (2, 3, 1)
```

---

#### `expand_dims(tensor: Any, axis: int) -> Any`

Add a size-1 dimension at specified axis.

**Parameters:**
- `tensor` (Any): Input tensor
- `axis` (int): Position to insert new axis

**Returns:**
- Tensor with expanded dimensions

**Example:**
```python
backend = create_backend()
x = backend.ones((2, 3))
y = backend.expand_dims(x, axis=0)  # Shape: (1, 2, 3)
z = backend.expand_dims(x, axis=2)  # Shape: (2, 3, 1)
```

---

### Core Tensor Operations

#### `einsum(pattern: str, *tensors: Any) -> Any`

Einstein summation convention for tensor operations.

**Parameters:**
- `pattern` (str): Einstein summation pattern (e.g., `"ij,jk->ik"`)
- `*tensors` (Any): Input tensors matching the pattern

**Returns:**
- Result of Einstein summation

**Examples:**

```python
backend = create_backend()

# Matrix multiplication
a = backend.ones((2, 3))
b = backend.ones((3, 4))
c = backend.einsum('ij,jk->ik', a, b)  # Shape: (2, 4)

# Batch matrix multiplication
a = backend.ones((10, 2, 3))
b = backend.ones((10, 3, 4))
c = backend.einsum('bij,bjk->bik', a, b)  # Shape: (10, 2, 4)

# Hadamard product (element-wise)
a = backend.ones((2, 3))
b = backend.ones((2, 3))
c = backend.einsum('ij,ij->ij', a, b)  # Shape: (2, 3)

# Tensor contraction
a = backend.ones((2, 3, 4))
b = backend.ones((4, 5))
c = backend.einsum('ijk,kl->ijl', a, b)  # Shape: (2, 3, 5)
```

---

#### `maximum(a: Any, b: Any) -> Any`

Element-wise maximum of two tensors.

**Parameters:**
- `a` (Any): First tensor
- `b` (Any): Second tensor (must be broadcast-compatible)

**Returns:**
- Element-wise maximum

**Example:**
```python
backend = create_backend()
a = backend.asarray([1.0, 2.0, 3.0])
b = backend.asarray([2.0, 1.5, 4.0])
c = backend.maximum(a, b)  # [2.0, 2.0, 4.0]
```

---

#### `add(a: Any, b: Any) -> Any`

Element-wise addition.

**Parameters:**
- `a` (Any): First tensor
- `b` (Any): Second tensor (must be broadcast-compatible)

**Returns:**
- Element-wise sum

**Example:**
```python
backend = create_backend()
a = backend.ones((2, 3))
b = backend.full((2, 3), 2.0)
c = backend.add(a, b)  # All elements are 3.0
```

---

#### `subtract(a: Any, b: Any) -> Any`

Element-wise subtraction.

**Parameters:**
- `a` (Any): First tensor
- `b` (Any): Second tensor (must be broadcast-compatible)

**Returns:**
- Element-wise difference (a - b)

**Example:**
```python
backend = create_backend()
a = backend.full((2, 3), 5.0)
b = backend.ones((2, 3))
c = backend.subtract(a, b)  # All elements are 4.0
```

---

#### `multiply(a: Any, b: Any) -> Any`

Element-wise multiplication.

**Parameters:**
- `a` (Any): First tensor
- `b` (Any): Second tensor (must be broadcast-compatible)

**Returns:**
- Element-wise product

**Example:**
```python
backend = create_backend()
a = backend.full((2, 3), 2.0)
b = backend.full((2, 3), 3.0)
c = backend.multiply(a, b)  # All elements are 6.0
```

---

#### `divide(a: Any, b: Any) -> Any`

Element-wise division.

**Parameters:**
- `a` (Any): Numerator tensor
- `b` (Any): Denominator tensor (must be broadcast-compatible)

**Returns:**
- Element-wise quotient (a / b)

**Example:**
```python
backend = create_backend()
a = backend.full((2, 3), 6.0)
b = backend.full((2, 3), 2.0)
c = backend.divide(a, b)  # All elements are 3.0
```

---

#### `matmul(a: Any, b: Any) -> Any`

Matrix multiplication.

**Parameters:**
- `a` (Any): First matrix
- `b` (Any): Second matrix (must have compatible dimensions)

**Returns:**
- Matrix product

**Example:**
```python
backend = create_backend()
a = backend.ones((2, 3))
b = backend.ones((3, 4))
c = backend.matmul(a, b)  # Shape: (2, 4)
```

---

### Reduction Operations

All reduction operations support optional `axis` and `keepdims` parameters:
- `axis` (int | None): Axis along which to reduce. If None, reduce over all axes.
- `keepdims` (bool): If True, keep reduced dimensions as size 1. Defaults to False.

#### `sum(tensor: Any, axis: int | None = None, keepdims: bool = False) -> Any`

Sum reduction along specified axis.

**Example:**
```python
backend = create_backend()
x = backend.ones((2, 3, 4))

# Sum all elements
total = backend.sum(x)  # Scalar: 24.0

# Sum along axis 1
row_sums = backend.sum(x, axis=1)  # Shape: (2, 4)

# Sum with keepdims
row_sums_keep = backend.sum(x, axis=1, keepdims=True)  # Shape: (2, 1, 4)
```

---

#### `max(tensor: Any, axis: int | None = None, keepdims: bool = False) -> Any`

Maximum reduction along specified axis.

**Example:**
```python
backend = create_backend()
x = backend.asarray([[1.0, 3.0], [2.0, 4.0]])

max_all = backend.max(x)  # Scalar: 4.0
max_cols = backend.max(x, axis=0)  # [2.0, 4.0]
```

---

#### `min(tensor: Any, axis: int | None = None, keepdims: bool = False) -> Any`

Minimum reduction along specified axis.

**Example:**
```python
backend = create_backend()
x = backend.asarray([[1.0, 3.0], [2.0, 4.0]])

min_all = backend.min(x)  # Scalar: 1.0
min_cols = backend.min(x, axis=0)  # [1.0, 3.0]
```

---

#### `mean(tensor: Any, axis: int | None = None, keepdims: bool = False) -> Any`

Mean (average) reduction along specified axis.

**Example:**
```python
backend = create_backend()
x = backend.asarray([[1.0, 3.0], [2.0, 4.0]])

mean_all = backend.mean(x)  # Scalar: 2.5
mean_rows = backend.mean(x, axis=1)  # [2.0, 3.0]
```

---

#### `prod(tensor: Any, axis: int | None = None, keepdims: bool = False) -> Any`

Product reduction along specified axis.

**Example:**
```python
backend = create_backend()
x = backend.asarray([[1.0, 2.0], [3.0, 4.0]])

prod_all = backend.prod(x)  # Scalar: 24.0
prod_cols = backend.prod(x, axis=0)  # [3.0, 8.0]
```

---

### Utility Operations

#### `eval(*tensors: Any) -> None`

Force evaluation of lazy tensors.

**Critical for MLX backend:** MLX uses lazy evaluation, so operations are not computed until explicitly evaluated.

**Parameters:**
- `*tensors` (Any): Tensors to evaluate

**Returns:**
- None (tensors are evaluated in-place)

**Example:**
```python
backend = create_backend("mlx")

# These operations are lazy - not computed yet
a = backend.ones((100, 100))
b = backend.zeros((100, 100))
result = backend.einsum('ij,jk->ik', a, b)

# Force evaluation
backend.eval(result)  # Now computed
```

---

#### `step(x: Any) -> Any`

Heaviside step function (0 if x < 0, 1 if x >= 0).

**Parameters:**
- `x` (Any): Input tensor

**Returns:**
- Step function output

**Example:**
```python
backend = create_backend()
x = backend.asarray([-2.0, -0.5, 0.0, 0.5, 2.0])
y = backend.step(x)  # [0., 0., 1., 1., 1.]
```

---

#### `clip(tensor: Any, min_val: float, max_val: float) -> Any`

Clamp tensor values to range [min_val, max_val].

**Parameters:**
- `tensor` (Any): Input tensor
- `min_val` (float): Minimum value
- `max_val` (float): Maximum value

**Returns:**
- Clipped tensor

**Example:**
```python
backend = create_backend()
x = backend.asarray([-2.0, 0.0, 5.0, 10.0])
y = backend.clip(x, 0.0, 5.0)  # [0.0, 0.0, 5.0, 5.0]
```

---

#### `abs(tensor: Any) -> Any`

Element-wise absolute value.

**Example:**
```python
backend = create_backend()
x = backend.asarray([-2.0, -1.0, 0.0, 1.0, 2.0])
y = backend.abs(x)  # [2.0, 1.0, 0.0, 1.0, 2.0]
```

---

#### `exp(tensor: Any) -> Any`

Element-wise exponential (e^x).

**Example:**
```python
backend = create_backend()
x = backend.asarray([0.0, 1.0, 2.0])
y = backend.exp(x)  # [1.0, 2.718..., 7.389...]
```

---

#### `log(tensor: Any) -> Any`

Element-wise natural logarithm.

**Example:**
```python
backend = create_backend()
x = backend.asarray([1.0, 2.718, 7.389])
y = backend.log(x)  # [0.0, 1.0, 2.0]
```

---

#### `sqrt(tensor: Any) -> Any`

Element-wise square root.

**Example:**
```python
backend = create_backend()
x = backend.asarray([1.0, 4.0, 9.0, 16.0])
y = backend.sqrt(x)  # [1.0, 2.0, 3.0, 4.0]
```

---

#### `power(tensor: Any, exponent: float) -> Any`

Element-wise power operation.

**Parameters:**
- `tensor` (Any): Base tensor
- `exponent` (float): Exponent value

**Returns:**
- Element-wise tensor^exponent

**Example:**
```python
backend = create_backend()
x = backend.asarray([1.0, 2.0, 3.0, 4.0])
y = backend.power(x, 2.0)  # [1.0, 4.0, 9.0, 16.0]
z = backend.power(x, 0.5)  # [1.0, 1.414, 1.732, 2.0]
```

---

#### `astype(tensor: Any, dtype: str) -> Any`

Convert tensor to specified data type.

**Parameters:**
- `tensor` (Any): Input tensor
- `dtype` (str): Target dtype ("float32", "float64", "int32", etc.)

**Returns:**
- Tensor with converted dtype

**Example:**
```python
backend = create_backend()
x = backend.ones((2, 3))  # Default float32
y = backend.astype(x, "float64")
```

---

#### `asarray(data: Any) -> Any`

Convert Python data to tensor.

**Parameters:**
- `data` (Any): Python list, tuple, scalar, or other array-like data

**Returns:**
- Tensor representation of input data

**Example:**
```python
backend = create_backend()
x = backend.asarray([1, 2, 3])  # 1D tensor
y = backend.asarray([[1, 2], [3, 4]])  # 2D tensor
z = backend.asarray(3.14)  # Scalar tensor
```

---

## Backend Implementations

### MLXBackend

MLX backend implementation optimized for Apple Silicon with GPU acceleration.

**Key Features:**
- GPU/Metal acceleration on Apple Silicon (M1/M2/M3)
- Lazy evaluation (operations deferred until `eval()`)
- Memory-efficient for large tensors
- Automatic graph optimization

**Installation:**
```bash
uv add mlx>=0.30.0
```

**Usage:**
```python
from tensorlogic.backends import create_backend

backend = create_backend("mlx")

# Operations are lazy
x = backend.ones((1000, 1000))
y = backend.zeros((1000, 1000))
result = backend.einsum('ij,jk->ik', x, y)

# Force evaluation for computation
backend.eval(result)
```

---

### NumpyBackend

NumPy backend implementation for universal CPU compatibility.

**Key Features:**
- Universal compatibility (works on all platforms)
- Eager evaluation (operations computed immediately)
- CPU-based computation
- Stable and well-tested

**Installation:**
```bash
uv add numpy>=1.24.0
```

**Usage:**
```python
from tensorlogic.backends import create_backend

backend = create_backend("numpy")

# Operations are eager - computed immediately
x = backend.ones((100, 100))
y = backend.zeros((100, 100))
result = backend.einsum('ij,jk->ik', x, y)  # Already computed
```

---

### CUDABackend

CUDA backend implementation for NVIDIA GPU acceleration using CuPy.

**Key Features:**
- High-performance GPU acceleration (up to 700x vs CPU)
- Google Colab support (T4, V100, A100 GPUs)
- Eager evaluation with explicit synchronization
- CuPy ecosystem compatibility

**Installation:**
```bash
# CUDA 12.x (recommended for Google Colab and modern systems)
pip install cupy-cuda12x

# CUDA 11.x (legacy systems)
pip install cupy-cuda11x
```

**Usage:**
```python
from tensorlogic.backends import create_backend
import cupy as cp

backend = create_backend("cuda")

# Create tensors on GPU
x = cp.ones((1000, 1000), dtype=cp.float32)
y = cp.ones((1000, 1000), dtype=cp.float32)

# Perform operations on GPU
result = backend.einsum('ij,jk->ik', x, y)

# Synchronize for timing (optional)
cp.cuda.Stream.null.synchronize()

# Transfer back to CPU if needed
result_cpu = cp.asnumpy(result)
```

**Performance Benchmarks (Tesla T4):**

| Graph Size | CUDA Time | NumPy Time | Speedup |
|------------|-----------|------------|---------|
| 500 entities | 0.54 ms | 20.42 ms | 37.5x |
| 1,000 entities | 1.37 ms | 181.62 ms | 132.5x |
| 5,000 entities | 59.57 ms | 42,167.71 ms | 707.8x |

**Best Practices:**
- Use CUDA for knowledge graphs with 500+ entities
- Use NumPy for small graphs (<500 entities) to avoid GPU overhead
- Call `cp.cuda.Stream.null.synchronize()` before timing operations
- Use `cp.asnumpy()` to transfer results back to CPU

---

## Type System

All backends use modern Python 3.12+ type hints:

```python
from __future__ import annotations
from typing import Any, Protocol

class TensorBackend(Protocol):
    """Protocol defining tensor backend interface."""

    def zeros(self, shape: tuple[int, ...]) -> Any: ...
    def ones(self, shape: tuple[int, ...]) -> Any: ...
    # ... other methods
```

**Key typing decisions:**
- Operations return `Any` because tensor types are backend-specific (mlx.core.array vs numpy.ndarray)
- Shape parameters use `tuple[int, ...]` for variable-length tuples
- Optional parameters use `| None` syntax (not `Optional`)
- Protocol-based duck typing for backend abstraction

---

## Error Handling

### ValueError

Raised for invalid backend names or configuration errors:

```python
# Unknown backend name
backend = create_backend("pytorch")  # Raises ValueError

# Missing dependencies
backend = create_backend("numpy")  # Raises ValueError if NumPy not installed
```

**Error message format:**
```
ValueError: Unknown backend: 'pytorch'. Available backends: 'mlx', 'numpy'
```

---

### TypeError

Raised when backend validation fails:

```python
class IncompleteBackend:
    def einsum(self, pattern: str, *tensors: Any) -> Any:
        pass

validate_backend(IncompleteBackend())
# Raises: TypeError: Backend validation failed: IncompleteBackend doesn't implement TensorBackend protocol...
```

**Error message includes:**
- Backend class name
- List of missing protocol methods
- Total number of required protocol methods

---

### ImportError

Raised when MLX backend is explicitly requested but unavailable (after fallback warning):

```python
# If MLX is unavailable, warning is issued and fallback to NumPy occurs
# ImportError only raised if NumPy also unavailable
```

---

## Performance Considerations

### MLX Backend

**Strengths:**
- GPU acceleration on Apple Silicon
- Lazy evaluation enables graph optimization
- Memory-efficient for large models

**Limitations:**
- Requires `eval()` calls for computation
- Apple Silicon only (M1/M2/M3)
- Smaller ecosystem than PyTorch/JAX

**Best for:**
- Large-scale tensor operations
- GPU-accelerated inference
- Apple Silicon development

---

### NumPy Backend

**Strengths:**
- Universal compatibility
- No GPU dependencies
- Eager evaluation (no surprise delays)
- Mature ecosystem

**Limitations:**
- CPU-only (slower for large operations)
- No automatic graph optimization

**Best for:**
- Development and testing
- CPU-only environments
- Prototyping and debugging

---

## Migration Guide

### From NumPy to MLX

```python
# Before (NumPy)
import numpy as np
x = np.ones((100, 100))
y = np.zeros((100, 100))
result = np.einsum('ij,jk->ik', x, y)

# After (TensorLogic with MLX)
from tensorlogic.backends import create_backend
backend = create_backend("mlx")
x = backend.ones((100, 100))
y = backend.zeros((100, 100))
result = backend.einsum('ij,jk->ik', x, y)
backend.eval(result)  # Critical: force evaluation
```

---

### From PyTorch to TensorLogic

```python
# Before (PyTorch)
import torch
x = torch.ones(100, 100)
y = torch.zeros(100, 100)
result = torch.einsum('ij,jk->ik', x, y)

# After (TensorLogic)
from tensorlogic.backends import create_backend
backend = create_backend()
x = backend.ones((100, 100))
y = backend.zeros((100, 100))
result = backend.einsum('ij,jk->ik', x, y)
backend.eval(result)  # If using MLX
```

---

## Examples

### Example 1: Cross-Backend Compatibility

```python
from tensorlogic.backends import create_backend

def compute_with_backend(backend_name: str) -> float:
    """Demonstrate cross-backend compatibility."""
    backend = create_backend(backend_name)

    # Create tensors
    x = backend.ones((10, 20))
    y = backend.ones((20, 30))

    # Matrix multiplication
    result = backend.einsum('ij,jk->ik', x, y)

    # Force evaluation (no-op for NumPy)
    backend.eval(result)

    # Compute mean
    mean_val = backend.mean(result)
    backend.eval(mean_val)

    return float(mean_val)

# Both backends produce identical results
mlx_result = compute_with_backend("mlx")
numpy_result = compute_with_backend("numpy")
assert abs(mlx_result - numpy_result) < 1e-6
```

---

### Example 2: Lazy Evaluation with MLX

```python
from tensorlogic.backends import create_backend

backend = create_backend("mlx")

# All operations are lazy - no computation yet
a = backend.ones((1000, 1000))
b = backend.ones((1000, 1000))
c = backend.einsum('ij,jk->ik', a, b)
d = backend.sum(c)

# Single eval call computes entire graph
backend.eval(d)
print(f"Result: {d}")  # Now computed
```

---

### Example 3: Custom Backend Validation

```python
from typing import Any
from tensorlogic.backends import TensorBackend, validate_backend

class CustomBackend:
    """Example of implementing a custom backend."""

    def zeros(self, shape: tuple[int, ...]) -> Any:
        return [0.0] * (shape[0] if shape else 1)

    def ones(self, shape: tuple[int, ...]) -> Any:
        return [1.0] * (shape[0] if shape else 1)

    # ... implement all other TensorBackend methods

# Validate before use
try:
    validate_backend(CustomBackend())
except TypeError as e:
    print(f"Validation failed: {e}")
```

---

## FAQ

**Q: Which backend should I use?**

A:
- **CUDA**: Best for production on NVIDIA GPUs (T4, V100, A100), Google Colab, or data centers. Up to 700x faster than CPU.
- **MLX**: Best for Apple Silicon development (M1/M2/M3). Up to 100x faster than CPU with unified memory benefits.
- **NumPy**: Best for small graphs (<500 entities), testing, or when GPU is unavailable.

---

**Q: Do I need to call eval() after every operation?**

A: Only for MLX backend. NumPy and CUDA use eager evaluation. Best practice: call `eval()` on final result tensors when using MLX.

---

**Q: Can I mix backends in the same code?**

A: Not recommended. Tensors from different backends are incompatible. Choose one backend per execution context.

---

**Q: How do I convert between backend tensor types?**

A: Use `asarray()` to convert from Python data, or use backend-specific conversion:

```python
# CUDA (CuPy) to NumPy
import cupy as cp
cuda_tensor = cp.ones((2, 3))
numpy_array = cp.asnumpy(cuda_tensor)

# NumPy to CUDA
numpy_array = np.ones((2, 3))
cuda_tensor = cp.asarray(numpy_array)

# MLX to NumPy
mlx_tensor = mlx_backend.ones((2, 3))
numpy_tensor = numpy_backend.asarray(mlx_tensor.tolist())
```

---

**Q: What's the performance crossover point?**

A: Based on benchmarks:
- **<500 entities**: NumPy is competitive or faster (GPU overhead dominates)
- **500-1,000 entities**: CUDA 30-130x faster
- **1,000+ entities**: CUDA 100-700x faster (scales super-linearly)

---

**Q: How do I use TensorLogic on Google Colab?**

A: See the [Google Colab notebook](https://colab.research.google.com/github/Mathews-Tom/TensorLogic/blob/main/notebooks/05_google_colab_cuda.ipynb):

```python
# Install
!pip install git+https://github.com/Mathews-Tom/TensorLogic.git
!pip install cupy-cuda12x

# Use
from tensorlogic import create_backend
backend = create_backend("cuda")  # or "auto" to auto-detect
```

---

## See Also

- [README.md](/Users/druk/WorkSpace/AetherForge/TensorLogic/README.md) - Quick start guide
- [Architecture](/Users/druk/WorkSpace/AetherForge/TensorLogic/.sage/agent/system/architecture.md) - System design
- [Tech Stack](/Users/druk/WorkSpace/AetherForge/TensorLogic/.sage/agent/system/tech-stack.md) - Technology decisions
