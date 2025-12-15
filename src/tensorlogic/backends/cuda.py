"""CuPy CUDA backend implementation for tensor operations.

This module provides a CuPy-based implementation of the TensorBackend protocol.
CuPy provides GPU-accelerated NumPy-compatible operations for NVIDIA GPUs.

Key characteristics:
- CUDA GPU execution (NVIDIA GPUs, including Google Colab T4)
- NumPy-compatible API
- Eager evaluation (eval() is no-op, like NumPy)
- Automatic memory management on GPU
- Supports automatic differentiation via cupy.gradient (limited)

Requirements:
    Install with: uv add cupy-cuda12x  # For CUDA 12.x (Colab, recommended)
    Or: pip install cupy-cuda11x  # For CUDA 11.x (legacy systems)

Note: CuPy requires NVIDIA GPU with CUDA support. For Apple Silicon,
use the MLX backend instead.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

try:
    import cupy as cp
except ImportError as e:
    raise ImportError(
        "CuPy is required for CUDA backend. "
        "Install with: pip install cupy-cuda12x (for CUDA 12.x, Colab) "
        "or pip install cupy-cuda11x (for CUDA 11.x legacy systems)"
    ) from e


class CUDABackend:
    """CuPy CUDA implementation of TensorBackend protocol.

    Provides GPU-accelerated tensor operations using CuPy on NVIDIA GPUs.
    Compatible with Google Colab T4 GPUs and other CUDA-capable devices.

    Example:
        >>> backend = CUDABackend()
        >>> a = backend.zeros((2, 3))  # Creates tensor on GPU
        >>> b = backend.ones((2, 3))
        >>> c = backend.add(a, b)  # GPU-accelerated addition
    """

    # Tensor Creation & Manipulation

    def einsum(self, pattern: str, *tensors: Any) -> Any:
        """Execute Einstein summation with string pattern notation.

        Implements generalized tensor contraction using Einstein notation.
        Core operation for tensor logic, enabling complex index manipulation.

        Args:
            pattern: Einstein summation pattern (e.g., "ij,jk->ik" for matrix multiply)
            *tensors: Variable number of input tensors matching pattern

        Returns:
            Result tensor after pattern evaluation (on GPU)

        Raises:
            ValueError: If pattern syntax invalid or incompatible with tensor shapes
            TypeError: If tensors have incompatible types

        Example:
            >>> # Matrix multiplication: C[i,k] = A[i,j] * B[j,k]
            >>> C = backend.einsum("ij,jk->ik", A, B)
        """
        return cp.einsum(pattern, *tensors)

    def zeros(self, shape: tuple[int, ...]) -> Any:
        """Create tensor filled with zeros on GPU.

        Args:
            shape: Tuple specifying tensor dimensions (e.g., (2, 3) for 2x3 matrix)

        Returns:
            Zero tensor with specified shape (on GPU)

        Raises:
            ValueError: If shape contains negative dimensions

        Example:
            >>> z = backend.zeros((3, 4))  # Creates 3x4 zero matrix on GPU
        """
        return cp.zeros(shape)

    def ones(self, shape: tuple[int, ...]) -> Any:
        """Create tensor filled with ones on GPU.

        Args:
            shape: Tuple specifying tensor dimensions

        Returns:
            Ones tensor with specified shape (on GPU)

        Raises:
            ValueError: If shape contains negative dimensions

        Example:
            >>> o = backend.ones((2, 2))  # Creates 2x2 ones matrix on GPU
        """
        return cp.ones(shape)

    def arange(self, start: int, stop: int, step: int = 1) -> Any:
        """Create 1D tensor with evenly spaced values on GPU.

        Args:
            start: Start value (inclusive)
            stop: End value (exclusive)
            step: Spacing between values (default: 1)

        Returns:
            1D tensor with values [start, start+step, ..., stop-step] (on GPU)

        Raises:
            ValueError: If step is zero

        Example:
            >>> r = backend.arange(0, 10, 2)  # [0, 2, 4, 6, 8] on GPU
        """
        return cp.arange(start, stop, step)

    def reshape(self, array: Any, shape: tuple[int, ...]) -> Any:
        """Reshape tensor to new dimensions without changing data.

        Args:
            array: Input tensor (on GPU)
            shape: New shape (must have same total size as input)

        Returns:
            Reshaped tensor (view when possible)

        Raises:
            ValueError: If new shape incompatible with array size

        Example:
            >>> x = backend.reshape(x, (2, 3))  # Reshape to 2x3
        """
        return cp.reshape(array, shape)

    # Logical & Mathematical Operations

    def step(self, x: Any) -> Any:
        """Heaviside step function (critical for boolean logic).

        Maps x > 0 -> 1.0, x <= 0 -> 0.0. Essential for converting continuous
        values to discrete boolean logic.

        Args:
            x: Input tensor (on GPU)

        Returns:
            Tensor with 1.0 where x > 0, 0.0 elsewhere (on GPU)

        Example:
            >>> s = backend.step(cp.array([-1.0, 0.0, 1.0]))  # [0.0, 0.0, 1.0]
        """
        return cp.where(x > 0, 1.0, 0.0)

    def maximum(self, a: Any, b: Any) -> Any:
        """Element-wise maximum (used for logical OR).

        Args:
            a: First input tensor (on GPU)
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor with element-wise max(a, b) (on GPU)

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> m = backend.maximum(a, b)  # Logical OR via max
        """
        return cp.maximum(a, b)

    def minimum(self, a: Any, b: Any) -> Any:
        """Element-wise minimum.

        Args:
            a: First input tensor (on GPU)
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor with element-wise min(a, b) (on GPU)

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> m = backend.minimum(a, b)
        """
        return cp.minimum(a, b)

    def multiply(self, a: Any, b: Any) -> Any:
        """Element-wise multiplication / Hadamard product (used for logical AND).

        Args:
            a: First input tensor (on GPU)
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor with element-wise a * b (on GPU)

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> p = backend.multiply(a, b)  # Logical AND via product
        """
        return cp.multiply(a, b)

    def add(self, a: Any, b: Any) -> Any:
        """Element-wise addition.

        Args:
            a: First input tensor (on GPU)
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor with element-wise a + b (on GPU)

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> s = backend.add(a, b)
        """
        return cp.add(a, b)

    def subtract(self, a: Any, b: Any) -> Any:
        """Element-wise subtraction.

        Args:
            a: First input tensor (on GPU)
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor with element-wise a - b (on GPU)

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> d = backend.subtract(a, b)
        """
        return cp.subtract(a, b)

    # Quantifier Operations

    def sum(self, array: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        """Sum reduction (used for existential quantification).

        Args:
            array: Input tensor (on GPU)
            axis: Axis or axes to sum over (None = sum all elements)

        Returns:
            Reduced tensor with summed values (on GPU)

        Example:
            >>> s = backend.sum(array, axis=1)  # Sum over axis 1
        """
        return cp.sum(array, axis=axis)

    def prod(self, array: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        """Product reduction (used for universal quantification).

        Args:
            array: Input tensor (on GPU)
            axis: Axis or axes to product over (None = product all elements)

        Returns:
            Reduced tensor with product values (on GPU)

        Example:
            >>> p = backend.prod(array, axis=0)  # Product over axis 0
        """
        return cp.prod(array, axis=axis)

    def any(self, array: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        """Boolean any reduction.

        Args:
            array: Input tensor (boolean or numeric, on GPU)
            axis: Axis or axes to apply any over (None = all elements)

        Returns:
            Boolean tensor with True where any element is True/non-zero

        Example:
            >>> a = backend.any(array, axis=1)  # Any true per row
        """
        return cp.any(array, axis=axis)

    def all(self, array: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        """Boolean all reduction.

        Args:
            array: Input tensor (boolean or numeric, on GPU)
            axis: Axis or axes to apply all over (None = all elements)

        Returns:
            Boolean tensor with True where all elements are True/non-zero

        Example:
            >>> a = backend.all(array, axis=0)  # All true per column
        """
        return cp.all(array, axis=axis)

    def max(self, array: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        """Maximum reduction (used for soft existential quantification).

        Args:
            array: Input tensor (on GPU)
            axis: Axis or axes to take maximum over (None = max all elements)

        Returns:
            Reduced tensor with maximum values (on GPU)

        Example:
            >>> m = backend.max(array, axis=1)  # Max over axis 1
        """
        return cp.max(array, axis=axis)

    def min(self, array: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        """Minimum reduction (used for soft universal quantification).

        Args:
            array: Input tensor (on GPU)
            axis: Axis or axes to take minimum over (None = min all elements)

        Returns:
            Reduced tensor with minimum values (on GPU)

        Example:
            >>> m = backend.min(array, axis=0)  # Min over axis 0
        """
        return cp.min(array, axis=axis)

    # Differentiation & Evaluation

    def grad(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Create gradient function for automatic differentiation.

        CuPy does not natively support automatic differentiation like MLX.
        For gradient computation on CUDA, consider using PyTorch or JAX.

        Args:
            fn: Function to differentiate

        Returns:
            Never returns (always raises)

        Raises:
            NotImplementedError: CuPy backend doesn't support autodiff

        Example:
            >>> grad_fn = backend.grad(loss_function)  # Raises NotImplementedError
        """
        raise NotImplementedError(
            "CuPy backend does not support automatic differentiation. "
            "Use MLX backend (Apple Silicon) for gradient computation, "
            "or integrate with PyTorch/JAX for CUDA autodiff."
        )

    def eval(self, *arrays: Any) -> None:
        """Force evaluation of lazy tensors.

        CuPy uses eager evaluation like NumPy, so this is a no-op.
        Provided for protocol compatibility with lazy evaluation backends.

        Args:
            *arrays: Variable number of tensors (ignored)

        Example:
            >>> result = backend.einsum("ij,jk->ik", A, B)
            >>> backend.eval(result)  # No-op for CuPy
        """
        # Synchronize GPU to ensure all operations are complete
        cp.cuda.Stream.null.synchronize()

    def compile(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """JIT compilation hint for performance optimization.

        CuPy provides fusion optimizations that are applied automatically.
        This returns the function unchanged but could be extended to use
        cupy.fuse for kernel fusion.

        Args:
            fn: Function to compile

        Returns:
            Original function unchanged

        Example:
            >>> fast_fn = backend.compile(compute_fn)  # Returns compute_fn as-is
        """
        return fn

    # Utility Operations

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """Conditional element selection.

        Args:
            condition: Boolean tensor for selection (on GPU)
            x: Tensor to select from where condition is True
            y: Tensor to select from where condition is False

        Returns:
            Tensor with elements from x where condition True, y otherwise (on GPU)

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> result = backend.where(mask, a, b)  # a if mask else b
        """
        return cp.where(condition, x, y)

    def expand_dims(self, array: Any, axis: int) -> Any:
        """Add new axis to tensor.

        Args:
            array: Input tensor (on GPU)
            axis: Position to insert new axis

        Returns:
            Tensor with additional dimension of size 1 (on GPU)

        Raises:
            ValueError: If axis out of valid range

        Example:
            >>> x = backend.expand_dims(x, axis=0)  # Add batch dimension
        """
        return cp.expand_dims(array, axis=axis)

    def squeeze(self, array: Any, axis: int | None = None) -> Any:
        """Remove single-dimensional axes.

        Args:
            array: Input tensor (on GPU)
            axis: Specific axis to remove (None = remove all size-1 axes)

        Returns:
            Tensor with specified dimensions removed (on GPU)

        Raises:
            ValueError: If specified axis not size 1

        Example:
            >>> x = backend.squeeze(x, axis=0)  # Remove batch dimension
        """
        return cp.squeeze(array, axis=axis)

    def transpose(self, array: Any, axes: tuple[int, ...] | None = None) -> Any:
        """Permute tensor dimensions.

        Args:
            array: Input tensor (on GPU)
            axes: Permutation of axes (None = reverse all axes)

        Returns:
            Transposed tensor (on GPU)

        Raises:
            ValueError: If axes invalid permutation

        Example:
            >>> x_t = backend.transpose(x, (1, 0))  # Swap first two axes
        """
        return cp.transpose(array, axes=axes)

    def concatenate(self, arrays: tuple[Any, ...], axis: int = 0) -> Any:
        """Concatenate tensors along existing axis.

        Args:
            arrays: Tuple of tensors to concatenate (on GPU)
            axis: Axis along which to concatenate

        Returns:
            Concatenated tensor (on GPU)

        Raises:
            ValueError: If tensors have incompatible shapes

        Example:
            >>> combined = backend.concatenate((a, b, c), axis=0)
        """
        return cp.concatenate(arrays, axis=axis)

    # Additional CuPy-specific utilities

    def asarray(self, data: Any) -> Any:
        """Convert data to CuPy array on GPU.

        Args:
            data: Input data (list, NumPy array, or CuPy array)

        Returns:
            CuPy array on GPU

        Example:
            >>> arr = backend.asarray([[1, 2], [3, 4]])
        """
        return cp.asarray(data)

    def to_numpy(self, array: Any) -> Any:
        """Transfer CuPy array from GPU to NumPy array on CPU.

        Args:
            array: CuPy array on GPU

        Returns:
            NumPy array on CPU

        Example:
            >>> np_arr = backend.to_numpy(gpu_arr)
        """
        return cp.asnumpy(array)

    def get_device_info(self) -> dict[str, Any]:
        """Get information about the current CUDA device.

        Returns:
            Dictionary with device name, memory info, compute capability

        Example:
            >>> info = backend.get_device_info()
            >>> print(info['name'])  # 'Tesla T4'
        """
        device = cp.cuda.Device()
        mem_info = device.mem_info
        return {
            "device_id": device.id,
            "name": cp.cuda.runtime.getDeviceProperties(device.id)["name"].decode(),
            "compute_capability": device.compute_capability,
            "total_memory_gb": mem_info[1] / (1024**3),
            "free_memory_gb": mem_info[0] / (1024**3),
        }


__all__ = ["CUDABackend"]
