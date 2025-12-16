# TensorLogic Troubleshooting Guide

Quick fixes for common issues, organized by symptom.

---

## Installation Issues

### "Module 'tensorlogic' not found"

**Cause:** Package not installed or wrong Python environment.

```bash
# Install the package
pip install python-tensorlogic

# If using uv
uv add python-tensorlogic

# Verify installation
python -c "import tensorlogic; print('OK')"
```

### "Using NumpyBackend" (Expected GPU)

**Cause:** GPU backend not installed.

**For Apple Silicon (M1/M2/M3):**
```bash
pip install mlx>=0.30.0

# Verify
python -c "import mlx; print('MLX OK')"
```

**For NVIDIA GPU:**
```bash
# CUDA 12.x (Google Colab, modern GPUs)
pip install cupy-cuda12x

# CUDA 11.x (older systems)
pip install cupy-cuda11x

# Verify
python -c "import cupy; print(f'CuPy OK, CUDA {cupy.cuda.runtime.runtimeGetVersion()}')"
```

**Check which backend TensorLogic detected:**
```python
from tensorlogic import create_backend
backend = create_backend()
print(f"Using: {type(backend).__name__}")
# Expected: MLXBackend (Apple) or CUDABackend (NVIDIA) or NumpyBackend (CPU)
```

### "No module named 'mlx'" on Intel Mac

**Cause:** MLX only works on Apple Silicon (M1/M2/M3).

```bash
# Check your chip
uname -m
# arm64 = Apple Silicon (MLX works)
# x86_64 = Intel (use NumPy or CUDA)
```

**Solution:** Use NumPy backend or an NVIDIA GPU.

---

## Shape Mismatch Errors

### "operands could not be broadcast together"

**Cause:** Arrays have incompatible shapes.

```python
# ❌ WRONG: Different lengths
a = np.array([1.0, 0.0, 1.0])      # 3 elements
b = np.array([1.0, 0.0])           # 2 elements
logical_and(a, b, backend=backend)  # Error!
```

**Visual explanation:**
```
a: [1.0, 0.0, 1.0]
b: [1.0, 0.0,  ? ]  ← What should go here?
              ↑
         Mismatch!
```

**Fix:** Ensure arrays have the same shape.

```python
# ✅ CORRECT: Same length
a = np.array([1.0, 0.0, 1.0])
b = np.array([1.0, 0.0, 0.0])  # 3 elements
result = logical_and(a, b, backend=backend)  # Works!
```

### "einsum path not supported"

**Cause:** Einsum notation mismatch with tensor dimensions.

```python
# ❌ WRONG: 'xy,yz->xz' expects 2D tensors
parent = np.array([1.0, 0.0, 1.0])  # 1D tensor!
backend.einsum('xy,yz->xz', parent, parent)  # Error!
```

**Visual explanation:**
```
Expected 2D:              Got 1D:
     y=0  y=1  y=2
x=0 [  _    _    _  ]     [  _    _    _  ]
x=1 [  _    _    _  ]
x=2 [  _    _    _  ]
```

**Fix:** Reshape to 2D or use correct einsum notation.

```python
# ✅ CORRECT: 2D tensor
parent = np.array([
    [0., 1., 0.],
    [0., 0., 1.],
    [0., 0., 0.],
])
backend.einsum('xy,yz->xz', parent, parent)  # Works!
```

---

## Unexpected Results

### All Zeros at T=0 (Deductive Mode)

**Cause:** Values below 0.5 become 0 at T=0.

```python
# At T=0, the step function applies:
# step(x) = 1 if x > 0.5 else 0

confidences = np.array([0.3, 0.4, 0.5, 0.6])
# At T=0:     [0.0, 0.0, 0.0, 1.0]
#              ↑    ↑    ↑    ✓
#           All below 0.5 become 0
```

**Solution:** Use T>0 for graded values, or ensure your input values are above 0.5 for "true".

### Results Don't Change With Temperature

**Cause:** Values are already 0.0 or 1.0 (fully binary).

```python
# These won't change with temperature
binary_facts = np.array([1.0, 0.0, 1.0, 0.0])

# Temperature only affects values BETWEEN 0 and 1
uncertain_facts = np.array([0.8, 0.3, 0.6, 0.5])  # Temperature matters here
```

### AND Returns All Ones

**Cause:** The soft AND operation uses `min(a, b)` which can behave unexpectedly.

```python
# Soft AND at T>0 uses: result = alpha * min(a, b) + (1-alpha) * hard_result
# Check your input values
```

**Debug steps:**
```python
print(f"Input a: {a}")
print(f"Input b: {b}")
print(f"Temperature: {temperature}")
print(f"Result: {result}")
```

---

## Performance Issues

### "Out of memory" on GPU

**Cause:** Tensor too large for GPU memory.

```python
# Check your tensor sizes
print(f"Tensor shape: {tensor.shape}")
print(f"Memory (approx): {tensor.size * 4 / 1024**2:.1f} MB")  # for float32
```

**Solutions:**

1. **Use smaller batches:**
```python
# Instead of processing all at once
for batch in np.array_split(large_tensor, 10):
    result = process(batch)
```

2. **Use NumPy backend for very large tensors:**
```python
backend = create_backend("numpy")  # Uses CPU RAM
```

3. **On Apple Silicon:** MLX handles memory pressure well via unified memory.

### Slow Performance

**Check backend:**
```python
from tensorlogic import create_backend
backend = create_backend()
print(type(backend).__name__)
# NumpyBackend = CPU only (slower for large tensors)
# MLXBackend or CUDABackend = GPU accelerated
```

**For knowledge graphs:**

| Entities | Recommended Backend |
|----------|---------------------|
| < 100 | NumPy (GPU overhead not worth it) |
| 100-1000 | MLX or CUDA |
| > 1000 | CUDA (up to 700x faster) |

---

## Import Errors

### "cannot import name 'quantify' from 'tensorlogic'"

**Cause:** Using wrong import path.

```python
# ❌ WRONG
from tensorlogic import quantify

# ✅ CORRECT
from tensorlogic.api import quantify
```

### "cannot import name 'TensorBackend'"

**Cause:** TensorBackend is a Protocol (for type hints), not a class to instantiate.

```python
# ❌ WRONG: Don't import TensorBackend directly
from tensorlogic.backends import TensorBackend

# ✅ CORRECT: Use the factory function
from tensorlogic import create_backend
backend = create_backend()
```

---

## Type Errors

### "Expected float32, got float64"

**Cause:** NumPy defaults to float64.

```python
# ❌ WRONG: Default is float64
a = np.array([1.0, 0.0, 1.0])

# ✅ CORRECT: Explicitly use float32
a = np.array([1.0, 0.0, 1.0], dtype=np.float32)
```

### "Object of type 'Array' is not JSON serializable"

**Cause:** Trying to serialize MLX arrays directly.

```python
# ❌ WRONG
json.dumps({"result": mlx_result})

# ✅ CORRECT: Convert to numpy first
json.dumps({"result": np.asarray(mlx_result).tolist()})
```

---

## Backend-Specific Issues

### MLX (Apple Silicon)

**"Lazy evaluation didn't run"**

MLX uses lazy evaluation. Force evaluation explicitly:

```python
result = backend.einsum('xy,yz->xz', a, b)
backend.eval(result)  # Force computation
print(np.asarray(result))  # Now safe to use
```

**"Metal framework not available"**

```bash
# Update macOS to 12.0 or later
# MLX requires Metal GPU support
```

### CUDA (NVIDIA)

**"CUDA out of memory"**

```python
import cupy as cp
# Clear GPU memory
cp.get_default_memory_pool().free_all_blocks()
```

**"CUDA version mismatch"**

```bash
# Check CUDA version
nvidia-smi

# Install matching CuPy
pip install cupy-cuda12x  # For CUDA 12.x
pip install cupy-cuda11x  # For CUDA 11.x
```

---

## Common Patterns

### Safe Backend Detection

```python
from tensorlogic import create_backend

def get_best_backend():
    """Get best available backend with fallback."""
    backend = create_backend()  # Auto-detects
    backend_name = type(backend).__name__

    if backend_name == "NumpyBackend":
        print("Warning: Using CPU. Install mlx or cupy for GPU acceleration.")
    else:
        print(f"Using GPU: {backend_name}")

    return backend
```

### Safe Array Conversion

```python
import numpy as np

def to_numpy(arr):
    """Convert any tensor to numpy array safely."""
    if hasattr(arr, '__array__'):
        return np.asarray(arr)
    return np.array(arr)

# Works with MLX, CuPy, or NumPy arrays
result_np = to_numpy(result)
```

### Debug Mode

```python
def debug_operation(name, a, b, result, backend):
    """Print debug info for logical operations."""
    print(f"=== {name} ===")
    print(f"Backend: {type(backend).__name__}")
    print(f"Input a: shape={getattr(a, 'shape', len(a))}, values={a[:5]}...")
    print(f"Input b: shape={getattr(b, 'shape', len(b))}, values={b[:5]}...")
    print(f"Result:  shape={getattr(result, 'shape', len(result))}, values={result[:5]}...")
```

---

## Getting Help

### Check Version

```python
import tensorlogic
print(tensorlogic.__version__)
```

### Minimal Reproducible Example

When reporting issues, include:

```python
# 1. Version info
import tensorlogic
import numpy as np
print(f"tensorlogic: {tensorlogic.__version__}")
print(f"numpy: {np.__version__}")

# 2. Backend info
from tensorlogic import create_backend
backend = create_backend()
print(f"backend: {type(backend).__name__}")

# 3. Minimal code that reproduces the issue
a = np.array([1.0, 0.0, 1.0], dtype=np.float32)
b = np.array([1.0, 1.0, 0.0], dtype=np.float32)
result = tensorlogic.logical_and(a, b, backend=backend)
print(f"result: {result}")  # What you got
print(f"expected: [1.0, 0.0, 0.0]")  # What you expected
```

### Resources

- **GitHub Issues:** [github.com/Mathews-Tom/TensorLogic/issues](https://github.com/Mathews-Tom/TensorLogic/issues)
- **Examples:** `examples/` directory
- **Tests:** `tests/` directory (see how features are tested)
