"""NumPy backend implementation for tensor operations.

This module provides a NumPy-based implementation of the TensorBackend protocol.
Serves as reference implementation and fallback for platforms without MLX support.

Key characteristics:
- Eager evaluation (eval() is no-op)
- CPU-only execution
- Bit-accurate reference for validating other backends
- Uses np.heaviside for step function
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

import numpy as np


class NumpyBackend:
    """NumPy implementation of TensorBackend protocol.

    Provides reference implementation using NumPy for CPU-based tensor operations.
    Uses eager evaluation, so eval() is a no-op. Suitable for testing, validation,
    and platforms without GPU acceleration.

    Example:
        >>> backend = NumpyBackend()
        >>> a = backend.zeros((2, 3))
        >>> b = backend.ones((2, 3))
        >>> c = backend.add(a, b)  # Immediately evaluated
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
            Result tensor after pattern evaluation

        Raises:
            ValueError: If pattern syntax invalid or incompatible with tensor shapes
            TypeError: If tensors have incompatible types

        Example:
            >>> # Matrix multiplication: C[i,k] = A[i,j] * B[j,k]
            >>> C = backend.einsum("ij,jk->ik", A, B)
        """
        return np.einsum(pattern, *tensors)

    def zeros(self, shape: tuple[int, ...]) -> Any:
        """Create tensor filled with zeros.

        Args:
            shape: Tuple specifying tensor dimensions (e.g., (2, 3) for 2x3 matrix)

        Returns:
            Zero tensor with specified shape

        Raises:
            ValueError: If shape contains negative dimensions

        Example:
            >>> z = backend.zeros((3, 4))  # Creates 3x4 zero matrix
        """
        return np.zeros(shape)

    def ones(self, shape: tuple[int, ...]) -> Any:
        """Create tensor filled with ones.

        Args:
            shape: Tuple specifying tensor dimensions

        Returns:
            Ones tensor with specified shape

        Raises:
            ValueError: If shape contains negative dimensions

        Example:
            >>> o = backend.ones((2, 2))  # Creates 2x2 identity-like matrix
        """
        return np.ones(shape)

    def arange(self, start: int, stop: int, step: int = 1) -> Any:
        """Create 1D tensor with evenly spaced values.

        Args:
            start: Start value (inclusive)
            stop: End value (exclusive)
            step: Spacing between values (default: 1)

        Returns:
            1D tensor with values [start, start+step, ..., stop-step]

        Raises:
            ValueError: If step is zero

        Example:
            >>> r = backend.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
        """
        return np.arange(start, stop, step)

    def reshape(self, array: Any, shape: tuple[int, ...]) -> Any:
        """Reshape tensor to new dimensions without changing data.

        Args:
            array: Input tensor
            shape: New shape (must have same total size as input)

        Returns:
            Reshaped tensor (view or copy depending on backend)

        Raises:
            ValueError: If new shape incompatible with array size

        Example:
            >>> x = backend.reshape(x, (2, 3))  # Reshape to 2x3
        """
        return np.reshape(array, shape)

    # Logical & Mathematical Operations

    def step(self, x: Any) -> Any:
        """Heaviside step function (critical for boolean logic).

        Maps x > 0 -> 1.0, x <= 0 -> 0.0. Essential for converting continuous
        values to discrete boolean logic.

        NumPy implementation uses np.where(x > 0, 1.0, 0.0) to handle edge cases
        (NaN -> 0.0, since NaN > 0 is False).

        Args:
            x: Input tensor

        Returns:
            Tensor with 1.0 where x > 0, 0.0 elsewhere

        Example:
            >>> s = backend.step([-1.0, 0.0, 1.0])  # [0.0, 0.0, 1.0]
        """
        return np.where(x > 0, 1.0, 0.0)

    def maximum(self, a: Any, b: Any) -> Any:
        """Element-wise maximum (used for logical OR).

        Args:
            a: First input tensor
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor with element-wise max(a, b)

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> m = backend.maximum(a, b)  # Logical OR via max
        """
        return np.maximum(a, b)

    def minimum(self, a: Any, b: Any) -> Any:
        """Element-wise minimum.

        Args:
            a: First input tensor
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor with element-wise min(a, b)

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> m = backend.minimum(a, b)
        """
        return np.minimum(a, b)

    def multiply(self, a: Any, b: Any) -> Any:
        """Element-wise multiplication / Hadamard product (used for logical AND).

        Args:
            a: First input tensor
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor with element-wise a * b

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> p = backend.multiply(a, b)  # Logical AND via product
        """
        return np.multiply(a, b)

    def add(self, a: Any, b: Any) -> Any:
        """Element-wise addition.

        Args:
            a: First input tensor
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor with element-wise a + b

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> s = backend.add(a, b)
        """
        return np.add(a, b)

    def subtract(self, a: Any, b: Any) -> Any:
        """Element-wise subtraction.

        Args:
            a: First input tensor
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor with element-wise a - b

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> d = backend.subtract(a, b)
        """
        return np.subtract(a, b)

    # Quantifier Operations

    def sum(self, array: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        """Sum reduction (used for existential quantification).

        Args:
            array: Input tensor
            axis: Axis or axes to sum over (None = sum all elements)

        Returns:
            Reduced tensor with summed values

        Example:
            >>> s = backend.sum(array, axis=1)  # Sum over axis 1
        """
        return np.sum(array, axis=axis)

    def prod(self, array: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        """Product reduction (used for universal quantification).

        Args:
            array: Input tensor
            axis: Axis or axes to product over (None = product all elements)

        Returns:
            Reduced tensor with product values

        Example:
            >>> p = backend.prod(array, axis=0)  # Product over axis 0
        """
        return np.prod(array, axis=axis)

    def any(self, array: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        """Boolean any reduction.

        Args:
            array: Input tensor (boolean or numeric)
            axis: Axis or axes to apply any over (None = all elements)

        Returns:
            Boolean tensor with True where any element is True/non-zero

        Example:
            >>> a = backend.any(array, axis=1)  # Any true per row
        """
        return np.any(array, axis=axis)

    def all(self, array: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        """Boolean all reduction.

        Args:
            array: Input tensor (boolean or numeric)
            axis: Axis or axes to apply all over (None = all elements)

        Returns:
            Boolean tensor with True where all elements are True/non-zero

        Example:
            >>> a = backend.all(array, axis=0)  # All true per column
        """
        return np.all(array, axis=axis)

    # Differentiation & Evaluation

    def grad(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Create gradient function for automatic differentiation.

        NumPy does not support automatic differentiation. Use MLX backend
        for gradient computation.

        Args:
            fn: Function to differentiate

        Returns:
            Never returns (always raises)

        Raises:
            NotImplementedError: NumPy backend doesn't support autodiff

        Example:
            >>> grad_fn = backend.grad(loss_function)  # Raises NotImplementedError
        """
        raise NotImplementedError(
            "NumPy backend does not support automatic differentiation. "
            "Use MLX backend for gradient computation."
        )

    def eval(self, *arrays: Any) -> None:
        """Force evaluation of lazy tensors.

        NumPy uses eager evaluation, so this is a no-op. Provided for
        protocol compatibility with lazy evaluation backends (MLX).

        Args:
            *arrays: Variable number of tensors (ignored)

        Example:
            >>> result = backend.einsum("ij,jk->ik", A, B)
            >>> backend.eval(result)  # No-op for NumPy
        """
        pass

    def compile(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """JIT compilation hint for performance optimization.

        NumPy does not support JIT compilation. Returns function unchanged.
        Provided for protocol compatibility.

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
            condition: Boolean tensor for selection
            x: Tensor to select from where condition is True
            y: Tensor to select from where condition is False

        Returns:
            Tensor with elements from x where condition True, y otherwise

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> result = backend.where(mask, a, b)  # a if mask else b
        """
        return np.where(condition, x, y)

    def expand_dims(self, array: Any, axis: int) -> Any:
        """Add new axis to tensor.

        Args:
            array: Input tensor
            axis: Position to insert new axis

        Returns:
            Tensor with additional dimension of size 1

        Raises:
            ValueError: If axis out of valid range

        Example:
            >>> x = backend.expand_dims(x, axis=0)  # Add batch dimension
        """
        return np.expand_dims(array, axis=axis)

    def squeeze(self, array: Any, axis: int | None = None) -> Any:
        """Remove single-dimensional axes.

        Args:
            array: Input tensor
            axis: Specific axis to remove (None = remove all size-1 axes)

        Returns:
            Tensor with specified dimensions removed

        Raises:
            ValueError: If specified axis not size 1

        Example:
            >>> x = backend.squeeze(x, axis=0)  # Remove batch dimension
        """
        return np.squeeze(array, axis=axis)

    def transpose(self, array: Any, axes: tuple[int, ...] | None = None) -> Any:
        """Permute tensor dimensions.

        Args:
            array: Input tensor
            axes: Permutation of axes (None = reverse all axes)

        Returns:
            Transposed tensor

        Raises:
            ValueError: If axes invalid permutation

        Example:
            >>> x_t = backend.transpose(x, (1, 0))  # Swap first two axes
        """
        return np.transpose(array, axes=axes)

    def concatenate(self, arrays: tuple[Any, ...], axis: int = 0) -> Any:
        """Concatenate tensors along existing axis.

        Args:
            arrays: Tuple of tensors to concatenate
            axis: Axis along which to concatenate

        Returns:
            Concatenated tensor

        Raises:
            ValueError: If tensors have incompatible shapes

        Example:
            >>> combined = backend.concatenate((a, b, c), axis=0)
        """
        return np.concatenate(arrays, axis=axis)


__all__ = ["NumpyBackend"]
