"""Backend factory with validation and graceful fallback.

This module provides factory functions for creating and validating tensor backends.
Implements graceful fallback from MLX to NumPy when MLX is unavailable, with
runtime protocol validation to ensure backend compliance.

Design Philosophy:
    - Default to MLX for performance (GPU/Apple Silicon optimized)
    - Graceful fallback to NumPy for compatibility
    - Runtime validation via isinstance() protocol checking
    - Helpful error messages with installation suggestions
"""

from __future__ import annotations

import warnings
from typing import Any

from tensorlogic.backends.protocol import TensorBackend


def create_backend(name: str = "mlx") -> TensorBackend:
    """Create tensor backend by name with graceful fallback.

    Attempts to create the requested backend, falling back to NumPy if MLX
    dependencies are unavailable. All backends are validated against the
    TensorBackend protocol before returning.

    Args:
        name: Backend identifier ('mlx', 'numpy'). Defaults to 'mlx'.

    Returns:
        Backend instance conforming to TensorBackend protocol

    Raises:
        ValueError: If backend name unknown or NumPy dependencies missing
        ImportError: If NumPy backend requested but unavailable

    Example:
        >>> # Try MLX, fallback to NumPy if unavailable
        >>> backend = create_backend()
        >>> # Explicitly request NumPy
        >>> backend = create_backend("numpy")
    """
    backend: TensorBackend

    if name == "mlx":
        try:
            from tensorlogic.backends.mlx import MLXBackend  # type: ignore[import-not-found]

            backend = MLXBackend()
        except ImportError as e:
            warnings.warn(
                f"MLX backend unavailable ({e}), falling back to NumPy. "
                "Install with: uv add mlx>=0.30.0",
                stacklevel=2,
            )
            name = "numpy"

    if name == "numpy":
        try:
            from tensorlogic.backends.numpy import NumpyBackend

            backend = NumpyBackend()
        except ImportError as e:
            msg = (
                f"NumPy backend unavailable ({e}). "
                "Install with: uv add numpy>=1.24.0"
            )
            raise ValueError(msg) from e
    elif name != "mlx":  # name is not "numpy" and not "mlx"
        available = ["mlx", "numpy"]
        msg = (
            f"Unknown backend: '{name}'. "
            f"Available backends: {', '.join(repr(b) for b in available)}"
        )
        raise ValueError(msg)

    validate_backend(backend)
    return backend


def validate_backend(backend: Any) -> None:
    """Validate object implements TensorBackend protocol.

    Uses runtime protocol checking to ensure backend provides all required
    operations. This is critical for catching incomplete implementations at
    runtime since Protocol checking is structural.

    Args:
        backend: Object to validate

    Raises:
        TypeError: If backend doesn't implement TensorBackend protocol

    Example:
        >>> class IncompleteBackend:
        ...     def einsum(self, pattern: str, *tensors: Any) -> Any:
        ...         pass
        >>> validate_backend(IncompleteBackend())  # Raises TypeError
    """
    if not isinstance(backend, TensorBackend):
        backend_type = type(backend).__name__
        msg = (
            f"Backend validation failed: {backend_type} doesn't implement TensorBackend protocol. "
            f"Missing required operations. Backend must implement all {len(TensorBackend.__annotations__)} "
            "protocol methods (einsum, zeros, ones, arange, reshape, step, maximum, etc.)"
        )
        raise TypeError(msg)


__all__ = ["create_backend", "validate_backend"]
