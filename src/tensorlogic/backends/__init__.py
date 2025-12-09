"""Backend abstraction layer for tensor operations.

This module provides a Protocol-based abstraction for tensor operations across
different frameworks (MLX, NumPy, future PyTorch/JAX). Following the einops
philosophy of minimal abstraction with ~25-30 core operations.
"""

from __future__ import annotations

from tensorlogic.backends.protocol import TensorBackend

__all__ = ["TensorBackend"]
