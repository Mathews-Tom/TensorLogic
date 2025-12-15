"""Tests for sparse tensor support.

Target: 1M+ entity knowledge graphs with <10GB memory.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlogic.backends.sparse import (
    SparseTensor,
    sparse_matmul,
    sparse_and,
    sparse_or,
    sparse_exists,
    sparse_forall,
    estimate_sparse_memory,
    HAS_SCIPY,
)


pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")


class TestSparseTensor:
    """Tests for SparseTensor class."""

    def test_from_triples_basic(self) -> None:
        """Test creating sparse tensor from triples."""
        sparse = SparseTensor.from_triples(
            rows=[0, 1, 2],
            cols=[1, 2, 3],
            n_entities=5,
        )

        assert sparse.shape == (5, 5)
        assert sparse.nnz == 3
        assert len(sparse.rows) == 3
        assert len(sparse.cols) == 3
        assert len(sparse.values) == 3

    def test_from_triples_with_values(self) -> None:
        """Test creating sparse tensor with custom values."""
        sparse = SparseTensor.from_triples(
            rows=[0, 1],
            cols=[1, 2],
            n_entities=5,
            values=[0.5, 0.75],
        )

        assert sparse.nnz == 2
        np.testing.assert_array_almost_equal(sparse.values, [0.5, 0.75])

    def test_from_dense(self) -> None:
        """Test creating sparse tensor from dense array."""
        dense = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [1, 0, 1],
        ], dtype=np.float32)

        sparse = SparseTensor.from_dense(dense)

        assert sparse.shape == (3, 3)
        assert sparse.nnz == 4

    def test_to_dense(self) -> None:
        """Test converting sparse tensor to dense."""
        sparse = SparseTensor.from_triples(
            rows=[0, 1, 2],
            cols=[1, 2, 0],
            n_entities=3,
        )

        dense = sparse.to_dense()

        expected = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ], dtype=np.float32)

        np.testing.assert_array_equal(dense, expected)

    def test_round_trip_dense(self) -> None:
        """Test dense -> sparse -> dense preserves values."""
        original = np.array([
            [1, 0, 0.5],
            [0, 1, 0],
            [0.7, 0, 1],
        ], dtype=np.float32)

        sparse = SparseTensor.from_dense(original)
        recovered = sparse.to_dense()

        np.testing.assert_array_almost_equal(recovered, original)

    def test_to_scipy_csr(self) -> None:
        """Test conversion to scipy CSR matrix."""
        sparse = SparseTensor.from_triples(
            rows=[0, 1],
            cols=[1, 0],
            n_entities=3,
        )

        csr = sparse.to_scipy_csr()

        assert csr.shape == (3, 3)
        assert csr.nnz == 2

    def test_from_scipy(self) -> None:
        """Test creating SparseTensor from scipy matrix."""
        import scipy.sparse as sp

        scipy_csr = sp.csr_matrix(np.eye(3, dtype=np.float32))
        sparse = SparseTensor.from_scipy(scipy_csr)

        assert sparse.shape == (3, 3)
        assert sparse.nnz == 3

    def test_density(self) -> None:
        """Test density calculation."""
        sparse = SparseTensor.from_triples(
            rows=[0, 1, 2, 3],
            cols=[1, 2, 3, 0],
            n_entities=4,
        )

        # 4 edges in 4x4 = 16 positions
        expected_density = 4 / 16
        assert sparse.density == expected_density

    def test_memory_estimation(self) -> None:
        """Test memory usage estimation."""
        sparse = SparseTensor.from_triples(
            rows=list(range(1000)),
            cols=list(range(1000)),
            n_entities=10000,
        )

        # 1000 edges * (4 + 4 + 4) bytes = 12000 bytes
        expected_bytes = 1000 * 12
        assert sparse.memory_bytes() == expected_bytes

        expected_mb = expected_bytes / (1024 * 1024)
        assert abs(sparse.memory_mb() - expected_mb) < 0.001


class TestSparseMatmul:
    """Tests for sparse matrix multiplication (multi-hop reasoning)."""

    def test_matmul_basic(self) -> None:
        """Test basic sparse matrix multiplication."""
        # Parent relation: 0->1, 1->2
        parent = SparseTensor.from_triples(
            rows=[0, 1],
            cols=[1, 2],
            n_entities=3,
        )

        # Grandparent = Parent @ Parent
        grandparent = sparse_matmul(parent, parent)

        # Should have 0->2 (through 1)
        assert grandparent.nnz == 1
        dense = grandparent.to_dense()
        assert dense[0, 2] == 1.0

    def test_matmul_multiple_paths(self) -> None:
        """Test matmul with multiple paths between nodes."""
        # 0->1, 0->2, 1->3, 2->3
        rel = SparseTensor.from_triples(
            rows=[0, 0, 1, 2],
            cols=[1, 2, 3, 3],
            n_entities=4,
        )

        two_hop = sparse_matmul(rel, rel)
        dense = two_hop.to_dense()

        # 0->3 should have value 2 (two paths: 0->1->3 and 0->2->3)
        assert dense[0, 3] == 2.0

    def test_matmul_shape_validation(self) -> None:
        """Test matmul raises error for incompatible shapes."""
        a = SparseTensor.from_triples(rows=[0], cols=[0], n_entities=3)
        b = SparseTensor.from_triples(rows=[0], cols=[0], n_entities=5)

        with pytest.raises(ValueError, match="Incompatible shapes"):
            sparse_matmul(a, b)


class TestSparseLogicalOps:
    """Tests for sparse logical operations."""

    def test_sparse_and(self) -> None:
        """Test sparse AND (intersection)."""
        a = SparseTensor.from_triples(
            rows=[0, 0, 1],
            cols=[1, 2, 2],
            n_entities=3,
        )
        b = SparseTensor.from_triples(
            rows=[0, 1, 2],
            cols=[1, 1, 2],
            n_entities=3,
        )

        result = sparse_and(a, b)

        # Only (0,1) is in both
        dense = result.to_dense()
        assert dense[0, 1] == 1.0
        assert dense[0, 2] == 0.0  # Not in b

    def test_sparse_or(self) -> None:
        """Test sparse OR (union)."""
        a = SparseTensor.from_triples(
            rows=[0, 1],
            cols=[1, 2],
            n_entities=3,
        )
        b = SparseTensor.from_triples(
            rows=[0, 2],
            cols=[2, 1],
            n_entities=3,
        )

        result = sparse_or(a, b)

        # Should have (0,1), (0,2), (1,2), (2,1)
        assert result.nnz == 4

    def test_shape_mismatch_error(self) -> None:
        """Test AND/OR raise error for shape mismatch."""
        a = SparseTensor.from_triples(rows=[0], cols=[0], n_entities=3)
        b = SparseTensor.from_triples(rows=[0], cols=[0], n_entities=5)

        with pytest.raises(ValueError, match="Shape mismatch"):
            sparse_and(a, b)

        with pytest.raises(ValueError, match="Shape mismatch"):
            sparse_or(a, b)


class TestSparseQuantifiers:
    """Tests for sparse quantifier operations."""

    def test_exists_over_columns(self) -> None:
        """Test exists aggregation over columns (axis=1)."""
        # Rows 0 and 1 have outgoing edges
        sparse = SparseTensor.from_triples(
            rows=[0, 0, 1],
            cols=[1, 2, 2],
            n_entities=3,
        )

        result = sparse_exists(sparse, axis=1)

        expected = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_exists_over_rows(self) -> None:
        """Test exists aggregation over rows (axis=0)."""
        # Columns 1 and 2 have incoming edges
        sparse = SparseTensor.from_triples(
            rows=[0, 0, 1],
            cols=[1, 2, 2],
            n_entities=3,
        )

        result = sparse_exists(sparse, axis=0)

        expected = np.array([0.0, 1.0, 1.0], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_forall_sparse(self) -> None:
        """Test forall on sparse tensor (mostly zeros)."""
        # Sparse tensor - forall should return 0 for most rows/cols
        sparse = SparseTensor.from_triples(
            rows=[0, 0, 0],
            cols=[0, 1, 2],
            n_entities=3,
            values=[1.0, 1.0, 1.0],
        )

        result = sparse_forall(sparse, axis=1)

        # Only row 0 has all columns filled
        assert result[0] == 1.0
        assert result[1] == 0.0
        assert result[2] == 0.0


class TestMemoryEstimation:
    """Tests for memory estimation utilities."""

    def test_estimate_basic(self) -> None:
        """Test basic memory estimation."""
        mem = estimate_sparse_memory(1000, 0.01, 1)

        assert "dense_mb" in mem
        assert "sparse_coo_mb" in mem
        assert "sparse_csr_mb" in mem
        assert "savings_ratio" in mem

    def test_estimate_large_scale(self) -> None:
        """Test memory estimation at 1M entity scale."""
        mem = estimate_sparse_memory(1_000_000, 0.00001, 50)

        # Sparse should be much smaller than dense
        assert mem["sparse_coo_mb"] < mem["dense_mb"]
        assert mem["savings_ratio"] > 100  # At least 100x savings

    def test_estimate_returns_correct_inputs(self) -> None:
        """Test that estimation returns input values."""
        mem = estimate_sparse_memory(1000, 0.01, 5)

        assert mem["n_entities"] == 1000
        assert mem["density"] == 0.01
        assert mem["n_relations"] == 5


class TestKnowledgeGraphIntegration:
    """Integration tests for knowledge graph operations."""

    def test_family_tree_reasoning(self) -> None:
        """Test multi-hop reasoning on family tree."""
        # Entity indices: 0=grandparent, 1=parent, 2=child
        parent_relation = SparseTensor.from_triples(
            rows=[0, 1],
            cols=[1, 2],
            n_entities=3,
        )

        # Grandparent = Parent @ Parent
        grandparent = sparse_matmul(parent_relation, parent_relation)

        dense = grandparent.to_dense()
        assert dense[0, 2] == 1.0  # 0 is grandparent of 2
        assert dense[0, 1] == 0.0  # 0 is not grandparent of 1
        assert dense[1, 2] == 0.0  # 1 is not grandparent of anyone

    def test_social_network_queries(self) -> None:
        """Test friend-of-friend queries."""
        # Friend relation: 0-1, 1-2, 2-3 (symmetric)
        friend = SparseTensor.from_triples(
            rows=[0, 1, 1, 2, 2, 3],
            cols=[1, 0, 2, 1, 3, 2],
            n_entities=4,
        )

        # Friends of friends
        fof = sparse_matmul(friend, friend)

        # Who are friends of 0's friends?
        has_fof = sparse_exists(fof, axis=1)
        # All nodes should have friends-of-friends
        assert all(has_fof > 0)

    def test_large_sparse_graph(self) -> None:
        """Test operations on larger sparse graph."""
        n_entities = 10000
        n_edges = 50000

        np.random.seed(42)
        rows = np.random.randint(0, n_entities, size=n_edges)
        cols = np.random.randint(0, n_entities, size=n_edges)

        sparse = SparseTensor.from_triples(
            rows=rows.tolist(),
            cols=cols.tolist(),
            n_entities=n_entities,
        )

        # Should handle 10K entities efficiently
        assert sparse.shape == (n_entities, n_entities)
        assert sparse.nnz <= n_edges  # May have duplicates

        # Memory should be reasonable
        assert sparse.memory_mb() < 10  # < 10MB for sparse representation
