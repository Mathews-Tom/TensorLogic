"""TensorBackend Protocol defining tensor operation interface.

This module defines a Protocol-based abstraction for tensor operations across
different backends (MLX, NumPy, future PyTorch/JAX). Following the einops
philosophy of minimal abstraction with ~25-30 core operations.

Critical design note: MLX backend uses lazy evaluation. All operations build
computation graphs that must be explicitly evaluated via the eval() method.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from collections.abc import Callable


@runtime_checkable
class TensorBackend(Protocol):
    """Protocol defining minimal tensor operation interface for backend abstraction.

    This Protocol defines ~25 core operations required for tensor logic computation.
    Backends implement these methods using their native frameworks (MLX, NumPy, etc.).

    Design Philosophy:
        - Abstract at operation level, not model level (einops-style)
        - Zero-overhead abstraction (<1% performance penalty)
        - Support both lazy (MLX) and eager (NumPy) evaluation
        - Structural typing via Protocol (no inheritance required)

    Implementation Requirements:
        - All methods must be implemented by concrete backends
        - Type hints must use modern Python 3.12+ syntax (| unions, built-in generics)
        - MLX backends must handle lazy evaluation (eval() is critical)
        - NumPy backends implement eval() as no-op

    Example:
        >>> backend = create_backend("mlx")
        >>> result = backend.einsum("ij,jk->ik", A, B)
        >>> backend.eval(result)  # Critical for MLX, no-op for NumPy
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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

    # Logical & Mathematical Operations

    def step(self, x: Any) -> Any:
        """Heaviside step function (critical for boolean logic).

        Maps x > 0 -> 1.0, x <= 0 -> 0.0. Essential for converting continuous
        values to discrete boolean logic.

        Backend-Specific Notes:
            - MLX: No native step, implement as where(x > 0, 1.0, 0.0)
            - NumPy: Use np.heaviside(x, 0.5)

        Args:
            x: Input tensor

        Returns:
            Tensor with 1.0 where x > 0, 0.0 elsewhere

        Example:
            >>> s = backend.step([-1.0, 0.0, 1.0])  # [0.0, 0.0, 1.0]
        """
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...

    # Differentiation & Evaluation

    def grad(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Create gradient function for automatic differentiation.

        Wraps function to compute gradients of outputs with respect to inputs.
        Essential for differentiable logical reasoning.

        Backend-Specific Notes:
            - MLX: Uses mx.grad with lazy evaluation
            - NumPy: Not supported (raises NotImplementedError)

        Args:
            fn: Function to differentiate (must be differentiable)

        Returns:
            Gradient function that computes derivatives

        Raises:
            NotImplementedError: If backend doesn't support autodiff

        Example:
            >>> grad_fn = backend.grad(loss_function)
            >>> gradients = grad_fn(params)
        """
        ...

    def eval(self, *arrays: Any) -> None:
        """Force evaluation of lazy tensors.

        Critical for MLX backend which uses lazy evaluation. Operations build
        computation graphs that are only executed when eval() is called.
        NumPy backend implements this as no-op (eager evaluation).

        Backend-Specific Notes:
            - MLX: Calls mx.eval(*arrays) to execute computation graph
            - NumPy: No-op, arrays already evaluated

        Args:
            *arrays: Variable number of tensors to evaluate

        Example:
            >>> result = backend.einsum("ij,jk->ik", A, B)
            >>> backend.eval(result)  # Critical: force MLX execution
        """
        ...

    def compile(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """JIT compilation hint for performance optimization.

        Provides hint to backend that function should be compiled. May be
        no-op for backends without JIT support.

        Args:
            fn: Function to compile

        Returns:
            Compiled function (or original if compilation not supported)

        Example:
            >>> fast_fn = backend.compile(compute_fn)
        """
        ...

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
        ...

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
        ...

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
        ...

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
        ...

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
        ...


__all__ = ["TensorBackend"]
