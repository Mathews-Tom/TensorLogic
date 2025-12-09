"""MLX backend implementation for tensor operations.

This module provides an MLX-based implementation of the TensorBackend protocol.
Leverages Apple Silicon's unified memory architecture and Metal GPU acceleration.

Key characteristics:
- Lazy evaluation (operations build computation graph)
- Must call eval() to force execution
- GPU-accelerated via Metal
- Uses mx.where() workaround for step function (no native step)
- Supports automatic differentiation via mx.grad
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

import mlx.core as mx


class MLXBackend:
    """MLX implementation of TensorBackend protocol.

    Provides GPU-accelerated tensor operations using Apple's MLX framework.
    Uses lazy evaluation - operations build computation graphs that must be
    explicitly evaluated via eval() method.

    Critical: Unlike NumPy, MLX operations do not execute immediately.
    Always call eval() to force execution of computation graphs.

    Example:
        >>> backend = MLXBackend()
        >>> a = backend.zeros((2, 3))
        >>> b = backend.ones((2, 3))
        >>> c = backend.add(a, b)  # Computation graph built, not executed
        >>> backend.eval(c)        # Force execution
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
            Result tensor after pattern evaluation (lazy, call eval() to execute)

        Raises:
            ValueError: If pattern syntax invalid or incompatible with tensor shapes
            TypeError: If tensors have incompatible types

        Example:
            >>> # Matrix multiplication: C[i,k] = A[i,j] * B[j,k]
            >>> C = backend.einsum("ij,jk->ik", A, B)
            >>> backend.eval(C)  # Force execution
        """
        return mx.einsum(pattern, *tensors)

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
        return mx.zeros(shape)

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
        return mx.ones(shape)

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
        return mx.arange(start, stop, step)

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
        return mx.reshape(array, shape)

    # Logical & Mathematical Operations

    def step(self, x: Any) -> Any:
        """Heaviside step function (critical for boolean logic).

        Maps x > 0 -> 1.0, x <= 0 -> 0.0. Essential for converting continuous
        values to discrete boolean logic.

        MLX implementation uses mx.where() workaround as there is no native step function.

        Args:
            x: Input tensor

        Returns:
            Tensor with 1.0 where x > 0, 0.0 elsewhere (lazy, call eval() to execute)

        Example:
            >>> s = backend.step([-1.0, 0.0, 1.0])  # [0.0, 0.0, 1.0]
            >>> backend.eval(s)
        """
        return mx.where(x > 0, 1.0, 0.0)

    def maximum(self, a: Any, b: Any) -> Any:
        """Element-wise maximum (used for logical OR).

        Args:
            a: First input tensor
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor with element-wise max(a, b) (lazy, call eval() to execute)

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> m = backend.maximum(a, b)  # Logical OR via max
            >>> backend.eval(m)
        """
        return mx.maximum(a, b)

    def minimum(self, a: Any, b: Any) -> Any:
        """Element-wise minimum.

        Args:
            a: First input tensor
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor with element-wise min(a, b) (lazy, call eval() to execute)

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> m = backend.minimum(a, b)
            >>> backend.eval(m)
        """
        return mx.minimum(a, b)

    def multiply(self, a: Any, b: Any) -> Any:
        """Element-wise multiplication / Hadamard product (used for logical AND).

        Args:
            a: First input tensor
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor with element-wise a * b (lazy, call eval() to execute)

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> p = backend.multiply(a, b)  # Logical AND via product
            >>> backend.eval(p)
        """
        return mx.multiply(a, b)

    def add(self, a: Any, b: Any) -> Any:
        """Element-wise addition.

        Args:
            a: First input tensor
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor with element-wise a + b (lazy, call eval() to execute)

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> s = backend.add(a, b)
            >>> backend.eval(s)
        """
        return mx.add(a, b)

    def subtract(self, a: Any, b: Any) -> Any:
        """Element-wise subtraction.

        Args:
            a: First input tensor
            b: Second input tensor (must be broadcastable with a)

        Returns:
            Tensor with element-wise a - b (lazy, call eval() to execute)

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> d = backend.subtract(a, b)
            >>> backend.eval(d)
        """
        return mx.subtract(a, b)

    # Quantifier Operations

    def sum(self, array: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        """Sum reduction (used for existential quantification).

        Args:
            array: Input tensor
            axis: Axis or axes to sum over (None = sum all elements)

        Returns:
            Reduced tensor with summed values (lazy, call eval() to execute)

        Example:
            >>> s = backend.sum(array, axis=1)  # Sum over axis 1
            >>> backend.eval(s)
        """
        return mx.sum(array, axis=axis)

    def prod(self, array: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        """Product reduction (used for universal quantification).

        Args:
            array: Input tensor
            axis: Axis or axes to product over (None = product all elements)

        Returns:
            Reduced tensor with product values (lazy, call eval() to execute)

        Example:
            >>> p = backend.prod(array, axis=0)  # Product over axis 0
            >>> backend.eval(p)
        """
        return mx.prod(array, axis=axis)

    def any(self, array: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        """Boolean any reduction.

        MLX does not have native any() function. Implemented using sum() workaround:
        any(x) = sum(x != 0) > 0

        Args:
            array: Input tensor (boolean or numeric)
            axis: Axis or axes to apply any over (None = all elements)

        Returns:
            Boolean tensor with True where any element is True/non-zero (lazy)

        Example:
            >>> a = backend.any(array, axis=1)  # Any true per row
            >>> backend.eval(a)
        """
        return mx.sum(array != 0, axis=axis) > 0

    def all(self, array: Any, axis: int | tuple[int, ...] | None = None) -> Any:
        """Boolean all reduction.

        MLX does not have native all() function. Implemented using prod() workaround:
        all(x) = prod(x != 0) != 0

        Args:
            array: Input tensor (boolean or numeric)
            axis: Axis or axes to apply all over (None = all elements)

        Returns:
            Boolean tensor with True where all elements are True/non-zero (lazy)

        Example:
            >>> a = backend.all(array, axis=0)  # All true per column
            >>> backend.eval(a)
        """
        return mx.prod(array != 0, axis=axis) != 0

    # Differentiation & Evaluation

    def grad(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Create gradient function for automatic differentiation.

        Wraps function to compute gradients of outputs with respect to inputs.
        Essential for differentiable logical reasoning.

        MLX uses lazy evaluation for gradients. The returned function builds
        gradient computation graphs that must be evaluated via eval().

        Args:
            fn: Function to differentiate (must be differentiable)

        Returns:
            Gradient function that computes derivatives

        Example:
            >>> grad_fn = backend.grad(loss_function)
            >>> gradients = grad_fn(params)
            >>> backend.eval(gradients)
        """
        return mx.grad(fn)

    def eval(self, *arrays: Any) -> None:
        """Force evaluation of lazy tensors.

        Critical for MLX backend which uses lazy evaluation. Operations build
        computation graphs that are only executed when eval() is called.

        This is the key difference between MLX and eager backends like NumPy.
        Always call eval() after operations to ensure execution.

        Args:
            *arrays: Variable number of tensors to evaluate

        Example:
            >>> result = backend.einsum("ij,jk->ik", A, B)
            >>> backend.eval(result)  # Critical: force MLX execution
        """
        mx.eval(*arrays)

    def compile(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """JIT compilation hint for performance optimization.

        MLX supports JIT compilation for optimizing repeated function calls.
        Compiled functions can be significantly faster for complex operations.

        Args:
            fn: Function to compile

        Returns:
            Compiled function optimized for repeated execution

        Example:
            >>> fast_fn = backend.compile(compute_fn)
            >>> result = fast_fn(inputs)
        """
        return mx.compile(fn)

    # Utility Operations

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """Conditional element selection.

        Args:
            condition: Boolean tensor for selection
            x: Tensor to select from where condition is True
            y: Tensor to select from where condition is False

        Returns:
            Tensor with elements from x where condition True, y otherwise (lazy)

        Raises:
            ValueError: If shapes not broadcastable

        Example:
            >>> result = backend.where(mask, a, b)  # a if mask else b
            >>> backend.eval(result)
        """
        return mx.where(condition, x, y)

    def expand_dims(self, array: Any, axis: int) -> Any:
        """Add new axis to tensor.

        Args:
            array: Input tensor
            axis: Position to insert new axis

        Returns:
            Tensor with additional dimension of size 1 (lazy)

        Raises:
            ValueError: If axis out of valid range

        Example:
            >>> x = backend.expand_dims(x, axis=0)  # Add batch dimension
            >>> backend.eval(x)
        """
        return mx.expand_dims(array, axis=axis)

    def squeeze(self, array: Any, axis: int | None = None) -> Any:
        """Remove single-dimensional axes.

        Args:
            array: Input tensor
            axis: Specific axis to remove (None = remove all size-1 axes)

        Returns:
            Tensor with specified dimensions removed (lazy)

        Raises:
            ValueError: If specified axis not size 1

        Example:
            >>> x = backend.squeeze(x, axis=0)  # Remove batch dimension
            >>> backend.eval(x)
        """
        return mx.squeeze(array, axis=axis)

    def transpose(self, array: Any, axes: tuple[int, ...] | None = None) -> Any:
        """Permute tensor dimensions.

        Args:
            array: Input tensor
            axes: Permutation of axes (None = reverse all axes)

        Returns:
            Transposed tensor (lazy)

        Raises:
            ValueError: If axes invalid permutation

        Example:
            >>> x_t = backend.transpose(x, (1, 0))  # Swap first two axes
            >>> backend.eval(x_t)
        """
        return mx.transpose(array, axes=axes)

    def concatenate(self, arrays: tuple[Any, ...], axis: int = 0) -> Any:
        """Concatenate tensors along existing axis.

        Args:
            arrays: Tuple of tensors to concatenate
            axis: Axis along which to concatenate

        Returns:
            Concatenated tensor (lazy)

        Raises:
            ValueError: If tensors have incompatible shapes

        Example:
            >>> combined = backend.concatenate((a, b, c), axis=0)
            >>> backend.eval(combined)
        """
        return mx.concatenate(list(arrays), axis=axis)


__all__ = ["MLXBackend"]
