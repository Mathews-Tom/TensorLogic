"""Backend abstraction layer for tensor operations.

This module provides a Protocol-based abstraction for tensor operations across
different frameworks (MLX, NumPy, future PyTorch/JAX). Following the einops
philosophy of minimal abstraction with ~25-30 core operations.

Usage:
    >>> from tensorlogic.backends import create_backend
    >>> # Default: try MLX, fallback to NumPy
    >>> backend = create_backend()
    >>> # Explicit backend selection
    >>> numpy_backend = create_backend("numpy")

Components:
    - TensorBackend: Protocol defining tensor operation interface
    - NumpyBackend: NumPy-based reference implementation
    - create_backend: Factory function with graceful fallback
    - validate_backend: Runtime protocol validation
"""

from __future__ import annotations

from tensorlogic.backends.protocol import TensorBackend
from tensorlogic.backends.numpy import NumpyBackend
from tensorlogic.backends.factory import create_backend, validate_backend

try:
    from tensorlogic.backends.mlx import MLXBackend

    __all__ = [
        "TensorBackend",
        "NumpyBackend",
        "MLXBackend",
        "create_backend",
        "validate_backend",
    ]
except ImportError:
    __all__ = ["TensorBackend", "NumpyBackend", "create_backend", "validate_backend"]
