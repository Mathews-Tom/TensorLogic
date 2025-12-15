"""Backend factory with validation and graceful fallback.

This module provides factory functions for creating and validating tensor backends.
Implements graceful fallback from MLX to NumPy when MLX is unavailable, with
runtime protocol validation to ensure backend compliance.

Design Philosophy:
    - Default to MLX for performance (GPU/Apple Silicon optimized)
    - Support CUDA via CuPy for NVIDIA GPUs (T4, V100, A100, etc.)
    - Graceful fallback to NumPy for compatibility
    - Runtime validation via isinstance() protocol checking
    - Helpful error messages with installation suggestions

Backend Selection:
    - 'mlx': Apple Silicon (M1/M2/M3) with Metal GPU acceleration
    - 'cuda': NVIDIA GPUs via CuPy (Google Colab T4, data center GPUs)
    - 'numpy': CPU fallback, universal compatibility
    - 'auto': Auto-detect best available backend
"""

from __future__ import annotations

import warnings
from typing import Any

from tensorlogic.backends.protocol import TensorBackend


def create_backend(name: str = "auto") -> TensorBackend:
    """Create tensor backend by name with graceful fallback.

    Attempts to create the requested backend, with intelligent auto-detection
    and fallback. All backends are validated against the TensorBackend protocol
    before returning.

    Args:
        name: Backend identifier. Options:
            - 'auto' (default): Auto-detect best available (MLX -> CUDA -> NumPy)
            - 'mlx': Apple Silicon with Metal GPU acceleration
            - 'cuda': NVIDIA GPUs via CuPy (T4, V100, A100, Colab)
            - 'numpy': CPU fallback, universal compatibility

    Returns:
        Backend instance conforming to TensorBackend protocol

    Raises:
        ValueError: If backend name unknown or no backend available
        ImportError: If explicitly requested backend unavailable

    Example:
        >>> # Auto-detect best available backend
        >>> backend = create_backend()
        >>> # Explicitly request CUDA for Google Colab
        >>> backend = create_backend("cuda")
        >>> # Request NumPy for CPU-only environments
        >>> backend = create_backend("numpy")
    """
    backend: TensorBackend

    # Auto-detection: try MLX (Apple) -> CUDA (NVIDIA) -> NumPy (fallback)
    if name == "auto":
        # Try MLX first (Apple Silicon)
        try:
            from tensorlogic.backends.mlx import MLXBackend

            backend = MLXBackend()
            validate_backend(backend)
            return backend
        except ImportError:
            pass

        # Try CUDA second (NVIDIA GPUs)
        try:
            from tensorlogic.backends.cuda import CUDABackend

            backend = CUDABackend()
            validate_backend(backend)
            return backend
        except ImportError:
            pass

        # Fall back to NumPy
        name = "numpy"

    if name == "mlx":
        try:
            from tensorlogic.backends.mlx import MLXBackend

            backend = MLXBackend()
        except ImportError as e:
            warnings.warn(
                f"MLX backend unavailable ({e}), falling back to NumPy. "
                "Install with: uv add mlx>=0.30.0",
                stacklevel=2,
            )
            name = "numpy"

    if name == "cuda":
        try:
            from tensorlogic.backends.cuda import CUDABackend

            backend = CUDABackend()
        except ImportError as e:
            warnings.warn(
                f"CUDA backend unavailable ({e}), falling back to NumPy. "
                "Install with: pip install cupy-cuda12x (CUDA 12) or cupy-cuda11x (CUDA 11/Colab)",
                stacklevel=2,
            )
            name = "numpy"

    if name == "numpy":
        try:
            from tensorlogic.backends.numpy import NumpyBackend

            backend = NumpyBackend()
        except ImportError as e:
            msg = f"NumPy backend unavailable ({e}). Install with: uv add numpy>=1.24.0"
            raise ValueError(msg) from e
    elif name not in ("mlx", "cuda"):  # name is not a known backend
        available = ["auto", "mlx", "cuda", "numpy"]
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
