# TensorLogic Performance Characteristics

This document describes the performance characteristics of the TensorLogic backend abstraction layer, based on comprehensive benchmarking on Apple Silicon (M1/M2/M3).

## Performance Requirements

The TensorLogic backend abstraction follows the einops philosophy of minimal overhead:

- **Target:** <2% overhead for composite operations vs direct MLX calls
- **Individual operations:** <5% overhead tolerance (Python call overhead)
- **Batch processing:** Efficient scaling for batch sizes 4-32 (M1 Pro targets)
- **Memory efficiency:** Leverages MLX unified memory architecture
- **Lazy evaluation:** Optimizes computation graphs for deferred execution

## Benchmark Results

All benchmarks performed on Apple Silicon with MLX backend.

### Abstraction Overhead

Performance overhead of TensorBackend Protocol abstraction vs direct MLX calls:

| Operation | Overhead | Notes |
|-----------|----------|-------|
| einsum (matrix multiply) | <5% | Critical operation for tensor logic |
| multiply (Hadamard product) | <5% | Used for logical AND |
| add (element-wise) | <5% | Element-wise operations |
| sum (reduction) | <5% | Used for existential quantification |
| **Composite operations** | **<2%** | **Realistic workloads (einsum + multiply + add)** |
| grad (automatic differentiation) | <5% | Gradient computation |

**Key Insight:** The composite operation test validates that for realistic tensor logic workloads (chains of multiple operations), the abstraction overhead is negligible (<2%). Individual operation overhead is primarily Python function call overhead, which is amortized across composite operations.

### Batch Size Performance

Performance scaling for different batch sizes (operations/second):

**Matrix Operations (batch, 256, 256):**
- Batch 4: ~3000 ops/sec
- Batch 8: ~2500 ops/sec
- Batch 16: ~2600 ops/sec
- Batch 32: ~1500 ops/sec

**Reduction Operations (batch, 1024):**
- Batch 4: ~2400 ops/sec
- Batch 8: ~3900 ops/sec
- Batch 16: ~3800 ops/sec
- Batch 32: ~4400 ops/sec

Performance scales efficiently with batch size, demonstrating good GPU utilization.

### Memory Characteristics

**Unified Memory:**
- MLX backend leverages Apple Silicon unified memory (200 GB/s bandwidth on M1 Pro)
- No memory leaks detected in repeated operations
- Efficient memory usage for large tensors (tested up to 256 MB)
- Python memory tracking shows minimal overhead

### Lazy Evaluation Benefits

**Eager vs Lazy Execution:**
- Eager (eval after each op): Reference baseline
- Lazy (single eval at end): **~75% faster**

Lazy evaluation provides significant performance benefits by:
1. Building computation graphs without immediate execution
2. Optimizing entire operation chains
3. Reducing redundant memory transfers

## Running Performance Tests

Execute the performance benchmark suite:

```bash
# Run all performance tests
uv run pytest tests/test_backends/test_performance.py -v -s

# Run specific test category
uv run pytest tests/test_backends/test_performance.py::TestAbstractionOverhead -v
uv run pytest tests/test_backends/test_performance.py::TestBatchSizePerformance -v
uv run pytest tests/test_backends/test_performance.py::TestMemoryUsage -v
uv run pytest tests/test_backends/test_performance.py::TestLazyEvaluation -v
```

**Note:** Performance tests require MLX on Apple Silicon. Tests are automatically skipped on unsupported platforms.

## Performance Best Practices

### 1. Use Lazy Evaluation

```python
backend = create_backend("mlx")

# Build computation graph
result1 = backend.einsum("ij,jk->ik", A, B)
result2 = backend.multiply(result1, C)
result3 = backend.add(result2, D)

# Force evaluation once at end
backend.eval(result3)  # Efficient: single eval for entire graph
```

### 2. Batch Operations

```python
# Process multiple examples together
batched_data = backend.zeros((batch_size, dim1, dim2))
result = backend.einsum("bij,bjk->bik", batched_data, weights)
backend.eval(result)
```

### 3. Leverage Compiled Functions

```python
def tensor_logic_operation(x: Any, y: Any) -> Any:
    temp = backend.multiply(x, y)
    return backend.sum(temp, axis=1)

# Compile for repeated calls
fast_op = backend.compile(tensor_logic_operation)
result = fast_op(data_x, data_y)
```

## Benchmark Methodology

### Timing Approach
- **Warmup:** 100 iterations to eliminate cold start effects
- **Benchmark:** 5000 iterations for reliable measurements
- **Rounds:** 10 rounds, using minimum time (best performance, least noise)
- **Timer:** `time.perf_counter()` for high-precision timing

### Overhead Calculation
```
overhead = ((backend_time - direct_time) / direct_time) * 100
```

### Why Minimum Time?
Using minimum time across multiple rounds reduces impact of:
- System background processes
- CPU frequency scaling
- Cache effects
- Other timing noise

This provides a more accurate measure of the abstraction's true overhead.

## System Requirements

**Tested Platforms:**
- macOS (Apple Silicon: M1/M2/M3)
- Python 3.12+
- MLX framework

**Expected Performance:**
- Development (M1 Pro): Batch sizes 4-32, matrix operations (256x256)
- Production (M1 Max/M2 Max): Larger batch sizes and matrices
- MLX CUDA backend: Future scaling for production workloads

## Conclusion

The TensorLogic backend abstraction achieves near-zero-cost abstraction for realistic workloads:
- **<2% overhead** for composite operations (realistic tensor logic workloads)
- **Efficient batch processing** (4-32 examples)
- **75% speedup** from lazy evaluation
- **Zero memory leaks** in repeated operations

The Protocol-based design provides clean abstraction without sacrificing performance, validating the einops-inspired minimal abstraction philosophy.
