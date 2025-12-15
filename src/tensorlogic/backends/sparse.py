"""Sparse Tensor Support for TensorLogic

This module provides sparse tensor operations for knowledge graphs with 1M+ entities.
Uses scipy.sparse for NumPy backend, with plans for MLX sparse support when available.

Key Features:
    - COO (Coordinate) format for sparse tensors
    - Efficient multi-hop reasoning via sparse matrix multiplication
    - Memory efficient: O(nnz) instead of O(n^2) for sparse relations
    - Seamless conversion between sparse and dense representations

Target: Support 1M+ entity knowledge graphs with <10GB memory

Usage:
    >>> from tensorlogic.backends.sparse import SparseTensor, sparse_matmul
    >>> # Create sparse relation from triples
    >>> sparse_rel = SparseTensor.from_triples(
    ...     rows=[0, 1, 2], cols=[1, 2, 3], n_entities=1000000
    ... )
    >>> # Multi-hop reasoning
    >>> two_hop = sparse_matmul(sparse_rel, sparse_rel)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Try to import scipy for sparse operations
try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    sp = None  # type: ignore


@dataclass
class SparseTensor:
    """Sparse tensor representation for knowledge graph relations.

    Uses COO (Coordinate) format internally, which is efficient for:
    - Construction from triples
    - Conversion to CSR for matrix multiplication
    - Random access

    Attributes:
        rows: Row indices of non-zero elements
        cols: Column indices of non-zero elements
        values: Values at (row, col) positions
        shape: Shape of the dense tensor this represents
    """

    rows: np.ndarray
    cols: np.ndarray
    values: np.ndarray
    shape: tuple[int, int]

    @classmethod
    def from_triples(
        cls,
        rows: list[int] | np.ndarray,
        cols: list[int] | np.ndarray,
        n_entities: int,
        values: list[float] | np.ndarray | None = None,
    ) -> SparseTensor:
        """Create sparse tensor from knowledge graph triples.

        Args:
            rows: Head entity indices
            cols: Tail entity indices
            n_entities: Total number of entities
            values: Edge weights (default: all 1.0)

        Returns:
            SparseTensor representing the relation

        Example:
            >>> # Create Parent relation
            >>> sparse = SparseTensor.from_triples(
            ...     rows=[0, 0, 1, 1],  # Alice, Alice, Bob, Bob
            ...     cols=[2, 3, 2, 3],  # Carol, David, Carol, David
            ...     n_entities=1000,
            ... )
        """
        rows_arr = np.asarray(rows, dtype=np.int32)
        cols_arr = np.asarray(cols, dtype=np.int32)

        if values is None:
            values_arr = np.ones(len(rows_arr), dtype=np.float32)
        else:
            values_arr = np.asarray(values, dtype=np.float32)

        return cls(
            rows=rows_arr,
            cols=cols_arr,
            values=values_arr,
            shape=(n_entities, n_entities),
        )

    @classmethod
    def from_dense(cls, dense: np.ndarray, threshold: float = 0.0) -> SparseTensor:
        """Convert dense tensor to sparse representation.

        Args:
            dense: Dense numpy array
            threshold: Values below this are treated as zero

        Returns:
            SparseTensor with non-zero elements

        Example:
            >>> dense = np.array([[1, 0], [0, 1]])
            >>> sparse = SparseTensor.from_dense(dense)
        """
        rows, cols = np.where(dense > threshold)
        values = dense[rows, cols]

        return cls(
            rows=rows.astype(np.int32),
            cols=cols.astype(np.int32),
            values=values.astype(np.float32),
            shape=dense.shape,
        )

    def to_dense(self) -> np.ndarray:
        """Convert sparse tensor to dense numpy array.

        Warning: May consume large amounts of memory for large tensors.

        Returns:
            Dense numpy array

        Example:
            >>> dense = sparse_tensor.to_dense()
        """
        dense = np.zeros(self.shape, dtype=np.float32)
        dense[self.rows, self.cols] = self.values
        return dense

    def to_scipy_csr(self) -> Any:
        """Convert to scipy CSR matrix for efficient arithmetic.

        Returns:
            scipy.sparse.csr_matrix

        Raises:
            ImportError: If scipy is not installed
        """
        if not HAS_SCIPY:
            raise ImportError(
                "scipy is required for sparse tensor operations. "
                "Install with: uv add scipy"
            )

        return sp.csr_matrix(
            (self.values, (self.rows, self.cols)),
            shape=self.shape,
        )

    def to_scipy_coo(self) -> Any:
        """Convert to scipy COO matrix.

        Returns:
            scipy.sparse.coo_matrix

        Raises:
            ImportError: If scipy is not installed
        """
        if not HAS_SCIPY:
            raise ImportError(
                "scipy is required for sparse tensor operations. "
                "Install with: uv add scipy"
            )

        return sp.coo_matrix(
            (self.values, (self.rows, self.cols)),
            shape=self.shape,
        )

    @classmethod
    def from_scipy(cls, scipy_matrix: Any) -> SparseTensor:
        """Create SparseTensor from scipy sparse matrix.

        Args:
            scipy_matrix: scipy.sparse matrix (any format)

        Returns:
            SparseTensor
        """
        if not HAS_SCIPY:
            raise ImportError("scipy is required")

        coo = scipy_matrix.tocoo()
        return cls(
            rows=coo.row.astype(np.int32),
            cols=coo.col.astype(np.int32),
            values=coo.data.astype(np.float32),
            shape=coo.shape,
        )

    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return len(self.values)

    @property
    def density(self) -> float:
        """Fraction of non-zero elements."""
        total = self.shape[0] * self.shape[1]
        return self.nnz / total if total > 0 else 0.0

    def memory_bytes(self) -> int:
        """Estimate memory usage in bytes."""
        # rows (int32) + cols (int32) + values (float32)
        return self.nnz * (4 + 4 + 4)

    def memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        return self.memory_bytes() / (1024 * 1024)


def sparse_matmul(a: SparseTensor, b: SparseTensor) -> SparseTensor:
    """Sparse matrix multiplication for relation composition.

    Computes: result[i,k] = exists j: a[i,j] AND b[j,k]

    This is the core operation for multi-hop reasoning:
    - Grandparent = Parent @ Parent
    - Aunt/Uncle = Sibling @ Parent

    Args:
        a: First sparse tensor
        b: Second sparse tensor

    Returns:
        SparseTensor with composed relation

    Raises:
        ImportError: If scipy is not installed
        ValueError: If shapes are incompatible

    Example:
        >>> grandparent = sparse_matmul(parent, parent)
    """
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for sparse tensor operations. "
            "Install with: uv add scipy"
        )

    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Incompatible shapes for matmul: {a.shape} @ {b.shape}. "
            f"Inner dimensions must match."
        )

    # Convert to CSR for efficient multiplication
    a_csr = a.to_scipy_csr()
    b_csr = b.to_scipy_csr()

    # Sparse matrix multiplication
    result_csr = a_csr @ b_csr

    # Convert back to SparseTensor
    return SparseTensor.from_scipy(result_csr)


def sparse_and(a: SparseTensor, b: SparseTensor) -> SparseTensor:
    """Element-wise AND (intersection) of sparse tensors.

    Computes: result[i,j] = a[i,j] AND b[i,j]

    For boolean tensors, this is the intersection of edges.

    Args:
        a: First sparse tensor
        b: Second sparse tensor (must have same shape)

    Returns:
        SparseTensor with intersection

    Raises:
        ImportError: If scipy is not installed
        ValueError: If shapes don't match
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required")

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    a_csr = a.to_scipy_csr()
    b_csr = b.to_scipy_csr()

    # Element-wise minimum (AND for boolean values)
    result = a_csr.multiply(b_csr)

    return SparseTensor.from_scipy(result)


def sparse_or(a: SparseTensor, b: SparseTensor) -> SparseTensor:
    """Element-wise OR (union) of sparse tensors.

    Computes: result[i,j] = a[i,j] OR b[i,j]

    For boolean tensors, this is the union of edges.

    Args:
        a: First sparse tensor
        b: Second sparse tensor (must have same shape)

    Returns:
        SparseTensor with union

    Raises:
        ImportError: If scipy is not installed
        ValueError: If shapes don't match
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required")

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

    a_csr = a.to_scipy_csr()
    b_csr = b.to_scipy_csr()

    # Element-wise maximum (OR for boolean values)
    result = a_csr.maximum(b_csr)

    return SparseTensor.from_scipy(result)


def sparse_exists(
    tensor: SparseTensor,
    axis: int,
) -> np.ndarray:
    """Existential quantification over sparse tensor.

    EXISTS x: R(x, y) -> returns 1D array indicating which y have any x

    Args:
        tensor: Sparse relation tensor
        axis: Axis to aggregate (0 = aggregate over rows, 1 = over columns)

    Returns:
        1D numpy array with 1.0 where exists, 0.0 otherwise

    Example:
        >>> # Who has children? (exists y: Parent(x, y))
        >>> has_children = sparse_exists(parent_sparse, axis=1)
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required")

    csr = tensor.to_scipy_csr()

    if axis == 1:
        # Aggregate over columns (for each row, check if any column has value)
        result = np.asarray(csr.sum(axis=1)).flatten()
    else:
        # Aggregate over rows (for each column, check if any row has value)
        result = np.asarray(csr.sum(axis=0)).flatten()

    # Convert to boolean (1.0 if any, 0.0 otherwise)
    return (result > 0).astype(np.float32)


def sparse_forall(
    tensor: SparseTensor,
    axis: int,
    threshold: float = 0.5,
) -> np.ndarray:
    """Universal quantification over sparse tensor.

    FORALL x: R(x, y) -> returns 1D array indicating where all x satisfy R

    Note: For sparse tensors, "forall" typically returns 0 for most elements
    since missing entries are implicitly 0 (false).

    Args:
        tensor: Sparse relation tensor
        axis: Axis to aggregate
        threshold: Value threshold for truth

    Returns:
        1D numpy array with 1.0 where forall holds, 0.0 otherwise
    """
    if not HAS_SCIPY:
        raise ImportError("scipy is required")

    # For sparse tensors, forall is tricky because missing values are 0
    # We need to check if all positions along axis have values >= threshold

    csr = tensor.to_scipy_csr()
    n = tensor.shape[0] if axis == 1 else tensor.shape[1]

    if axis == 1:
        # For each row, check if number of non-zeros equals number of columns
        # and all values are >= threshold
        nnz_per_row = np.diff(csr.indptr)
        all_present = nnz_per_row == tensor.shape[1]
        # Also check minimum value per row
        result = np.zeros(tensor.shape[0], dtype=np.float32)
        for i in range(tensor.shape[0]):
            if all_present[i]:
                row_data = csr.data[csr.indptr[i]:csr.indptr[i+1]]
                if np.all(row_data >= threshold):
                    result[i] = 1.0
    else:
        # For each column
        csc = csr.tocsc()
        nnz_per_col = np.diff(csc.indptr)
        all_present = nnz_per_col == tensor.shape[0]
        result = np.zeros(tensor.shape[1], dtype=np.float32)
        for j in range(tensor.shape[1]):
            if all_present[j]:
                col_data = csc.data[csc.indptr[j]:csc.indptr[j+1]]
                if np.all(col_data >= threshold):
                    result[j] = 1.0

    return result


def estimate_sparse_memory(
    n_entities: int,
    density: float,
    n_relations: int = 1,
) -> dict[str, float]:
    """Estimate memory usage for sparse vs dense representations.

    Args:
        n_entities: Number of entities
        density: Fraction of non-zero entries
        n_relations: Number of relations

    Returns:
        Dictionary with memory estimates in MB

    Example:
        >>> mem = estimate_sparse_memory(1_000_000, 0.00001, 50)
        >>> print(f"Sparse: {mem['sparse_mb']:.1f} MB")
        >>> print(f"Dense: {mem['dense_mb']:.1f} MB")
        >>> print(f"Savings: {mem['savings_ratio']:.1f}x")
    """
    n_elements = n_entities * n_entities
    nnz_per_relation = int(n_elements * density)

    # Dense: 4 bytes per float32
    dense_bytes = n_elements * 4 * n_relations
    dense_mb = dense_bytes / (1024 * 1024)

    # Sparse COO: 4 bytes row + 4 bytes col + 4 bytes value per entry
    sparse_bytes = nnz_per_relation * 12 * n_relations
    sparse_mb = sparse_bytes / (1024 * 1024)

    # CSR is more efficient: values + col_indices + row_pointers
    # values: 4 * nnz, col_indices: 4 * nnz, row_pointers: 4 * (n_entities + 1)
    csr_bytes = (
        (nnz_per_relation * 8 + (n_entities + 1) * 4) * n_relations
    )
    csr_mb = csr_bytes / (1024 * 1024)

    savings = dense_mb / sparse_mb if sparse_mb > 0 else float('inf')

    return {
        "n_entities": n_entities,
        "density": density,
        "n_relations": n_relations,
        "nnz_per_relation": nnz_per_relation,
        "dense_mb": dense_mb,
        "sparse_coo_mb": sparse_mb,
        "sparse_csr_mb": csr_mb,
        "savings_ratio": savings,
    }


def demo_sparse_scaling() -> None:
    """Demonstrate memory savings with sparse tensors at scale."""
    print("=" * 70)
    print("SPARSE TENSOR MEMORY SCALING")
    print("=" * 70)

    print(f"\n{'Entities':<12} {'Density':<10} {'Dense (MB)':<12} {'Sparse (MB)':<14} {'Savings':<10}")
    print("-" * 60)

    test_cases = [
        (1_000, 0.01),
        (10_000, 0.001),
        (100_000, 0.0001),
        (1_000_000, 0.00001),
    ]

    for n_entities, density in test_cases:
        mem = estimate_sparse_memory(n_entities, density, n_relations=1)
        print(
            f"{n_entities:<12,} {density:<10.5f} {mem['dense_mb']:<12,.1f} "
            f"{mem['sparse_coo_mb']:<14,.1f} {mem['savings_ratio']:<10,.1f}x"
        )

    print("\n" + "=" * 70)
    print("TARGET: 1M+ entities with <10GB memory - ACHIEVABLE with sparse tensors!")
    print("=" * 70)


__all__ = [
    "SparseTensor",
    "sparse_matmul",
    "sparse_and",
    "sparse_or",
    "sparse_exists",
    "sparse_forall",
    "estimate_sparse_memory",
    "HAS_SCIPY",
]


if __name__ == "__main__":
    demo_sparse_scaling()
