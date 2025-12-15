"""Tests for RAG integration module."""

from __future__ import annotations

import numpy as np
import pytest

from tensorlogic.integrations import (
    TensorLogicRetriever,
    RetrievalResult,
    HybridRanker,
    ConstraintFilter,
)
from tensorlogic.integrations.rag.ranker import RankingConfig


class TestTensorLogicRetriever:
    """Tests for TensorLogicRetriever class."""

    @pytest.fixture
    def sample_embeddings(self) -> np.ndarray:
        """Create sample entity embeddings."""
        np.random.seed(42)
        return np.random.randn(100, 64).astype(np.float32)

    @pytest.fixture
    def sample_retriever(self, sample_embeddings: np.ndarray) -> TensorLogicRetriever:
        """Create retriever with sample data."""
        retriever = TensorLogicRetriever(temperature=0.0, lambda_neural=0.5)
        metadata = [{"name": f"entity_{i}"} for i in range(100)]
        retriever.index_entities(sample_embeddings, metadata=metadata)
        return retriever

    def test_retriever_initialization(self) -> None:
        """Test retriever can be initialized."""
        retriever = TensorLogicRetriever()
        assert retriever.num_entities == 0
        assert retriever.temperature == 0.0

    def test_index_entities(self, sample_embeddings: np.ndarray) -> None:
        """Test entity indexing."""
        retriever = TensorLogicRetriever()
        retriever.index_entities(sample_embeddings)
        assert retriever.num_entities == 100

    def test_index_entities_with_ids(self, sample_embeddings: np.ndarray) -> None:
        """Test entity indexing with custom IDs."""
        retriever = TensorLogicRetriever()
        ids = [f"entity_{i}" for i in range(100)]
        retriever.index_entities(sample_embeddings, entity_ids=ids)
        assert retriever._entity_ids[0] == "entity_0"

    def test_neural_retrieval(self, sample_retriever: TensorLogicRetriever) -> None:
        """Test neural similarity retrieval."""
        query = np.random.randn(64).astype(np.float32)
        results = sample_retriever.retrieve(query_embedding=query, top_k=5)
        assert len(results) <= 5
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Scores should be sorted descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_type_filtering(self, sample_retriever: TensorLogicRetriever) -> None:
        """Test type-based filtering."""
        # Add type mask (first 50 are "Person")
        type_mask = np.zeros(100)
        type_mask[:50] = 1
        sample_retriever.add_type_mask("Person", type_mask)

        query = np.random.randn(64).astype(np.float32)
        results = sample_retriever.retrieve(
            query_embedding=query, type_filter="Person", top_k=10
        )

        # All results should be in first 50 entities
        for result in results:
            entity_idx = sample_retriever._entity_ids.index(result.entity_id)
            assert entity_idx < 50, f"Entity {result.entity_id} should be type Person"

    def test_relation_traversal(
        self, sample_retriever: TensorLogicRetriever
    ) -> None:
        """Test multi-hop relation traversal."""
        # Create simple relation: entity_i -> entity_(i+1) for i < 50
        relation = np.zeros((100, 100))
        for i in range(50):
            relation[i, i + 1] = 1
        sample_retriever.add_relation("NextTo", relation)

        # Start from entity 0, should reach entity 1
        results = sample_retriever.retrieve(
            relation_chain=["NextTo"],
            source_entities=[0],
            top_k=5,
        )

        # Entity 1 should be reachable
        result_ids = [r.entity_id for r in results]
        assert 1 in result_ids

    def test_batch_retrieval(self, sample_retriever: TensorLogicRetriever) -> None:
        """Test batch retrieval for multiple queries."""
        queries = np.random.randn(3, 64).astype(np.float32)
        results = sample_retriever.batch_retrieve(queries, top_k=5)
        assert len(results) == 3
        assert all(len(r) <= 5 for r in results)


class TestConstraintFilter:
    """Tests for ConstraintFilter class."""

    @pytest.fixture
    def sample_filter(self) -> ConstraintFilter:
        """Create filter with sample predicates."""
        filter = ConstraintFilter(temperature=0.0)
        # Add predicates
        filter.add_predicate("IsActive", np.array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0]))
        filter.add_predicate("HasPermission", np.array([1, 0, 1, 0, 1, 1, 0, 0, 1, 0]))
        return filter

    def test_evaluate_predicate(self, sample_filter: ConstraintFilter) -> None:
        """Test single predicate evaluation."""
        mask = sample_filter.evaluate_predicate("IsActive")
        assert float(mask[0]) == 1.0
        assert float(mask[2]) == 0.0

    def test_evaluate_negated_predicate(self, sample_filter: ConstraintFilter) -> None:
        """Test negated predicate evaluation."""
        mask = sample_filter.evaluate_predicate("IsActive", negated=True)
        assert float(mask[0]) == 0.0
        assert float(mask[2]) == 1.0

    def test_compose_and(self, sample_filter: ConstraintFilter) -> None:
        """Test AND composition of predicates."""
        active = sample_filter.evaluate_predicate("IsActive")
        perm = sample_filter.evaluate_predicate("HasPermission")
        combined = sample_filter.compose_and(active, perm)

        # Only entity 0 and 4 have both predicates
        assert float(combined[0]) == 1.0
        assert float(combined[4]) == 1.0
        assert float(combined[1]) == 0.0

    def test_compose_or(self, sample_filter: ConstraintFilter) -> None:
        """Test OR composition of predicates."""
        active = sample_filter.evaluate_predicate("IsActive")
        perm = sample_filter.evaluate_predicate("HasPermission")
        combined = sample_filter.compose_or(active, perm)

        # Entities with either predicate
        assert float(combined[0]) == 1.0
        assert float(combined[2]) == 1.0
        assert float(combined[3]) == 0.0

    def test_apply_filter(self, sample_filter: ConstraintFilter) -> None:
        """Test applying filter to candidates."""
        candidates = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        filtered = sample_filter.apply(
            candidates, predicate_names=["IsActive"], composition="and"
        )
        filtered_np = np.array(filtered)

        # Inactive entities should be zeroed
        assert filtered_np[2] == 0.0
        assert filtered_np[3] == 0.0
        # Active entities keep scores
        assert filtered_np[0] > 0

    def test_relation_exists(self) -> None:
        """Test existential quantification over relation."""
        filter = ConstraintFilter()
        # Create relation: 0->1, 0->2, 1->3
        relation = np.array([
            [0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ], dtype=np.float32)
        filter.add_relation("Links", relation)

        # Start from entity 0
        source = np.array([1, 0, 0, 0, 0], dtype=np.float32)
        result = filter.evaluate_relation_exists("Links", source)
        result_np = np.array(result)

        # Entities 1 and 2 should be reachable from 0
        assert result_np[1] > 0.5
        assert result_np[2] > 0.5
        assert result_np[3] < 0.5


class TestHybridRanker:
    """Tests for HybridRanker class."""

    def test_ranker_initialization(self) -> None:
        """Test ranker initialization with default config."""
        ranker = HybridRanker()
        assert ranker.config.lambda_neural == 0.5

    def test_ranker_with_custom_config(self) -> None:
        """Test ranker with custom configuration."""
        config = RankingConfig(lambda_neural=0.8, temperature=0.5)
        ranker = HybridRanker(config=config)
        assert ranker.config.lambda_neural == 0.8
        assert ranker.config.temperature == 0.5

    def test_rank_hard_filtering(self) -> None:
        """Test hard filtering mode (temperature=0)."""
        ranker = HybridRanker(config=RankingConfig(temperature=0.0))

        neural = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        logical = np.array([1.0, 0.0, 1.0, 0.0, 1.0])

        scores = ranker.rank(neural, logical)
        scores_np = np.array(scores)

        # Entities with logical=0 should be filtered out
        assert scores_np[1] == 0.0
        assert scores_np[3] == 0.0
        assert scores_np[0] > 0

    def test_rank_soft_combination(self) -> None:
        """Test soft combination mode (temperature>0)."""
        ranker = HybridRanker(
            config=RankingConfig(lambda_neural=0.5, temperature=0.5)
        )

        neural = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        logical = np.array([0.1, 0.9, 0.5, 0.3, 0.7])

        scores = ranker.rank(neural, logical)
        scores_np = np.array(scores)

        # Soft combination: 0.5 * neural + 0.5 * logical
        expected_0 = 0.5 * 0.9 + 0.5 * 0.1  # 0.5
        expected_1 = 0.5 * 0.8 + 0.5 * 0.9  # 0.85
        # Normalized, so entity 1 should be highest
        assert scores_np[1] == pytest.approx(1.0, rel=0.01)

    def test_get_top_k(self) -> None:
        """Test top-k retrieval."""
        ranker = HybridRanker()
        scores = np.array([0.3, 0.9, 0.1, 0.7, 0.5])

        indices, top_scores = ranker.get_top_k(scores, k=3)

        assert indices[0] == 1  # Highest score
        assert indices[1] == 3  # Second highest
        assert len(indices) == 3

    def test_reciprocal_rank_fusion(self) -> None:
        """Test RRF combination of multiple rankings."""
        ranker = HybridRanker()

        # Two different rankings
        scores1 = np.array([0.9, 0.5, 0.3, 0.8, 0.1])
        scores2 = np.array([0.2, 0.9, 0.4, 0.1, 0.8])

        fused = ranker.rank_reciprocal_fusion([scores1, scores2])
        fused_np = np.array(fused)

        # All scores should be non-negative
        assert all(s >= 0 for s in fused_np)
        # Should be normalized
        assert max(fused_np) == pytest.approx(1.0, rel=0.01)

    def test_rerank(self) -> None:
        """Test reranking a candidate set."""
        ranker = HybridRanker(config=RankingConfig(lambda_neural=0.7))

        all_neural = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        all_logical = np.array([1.0, 0.5, 1.0, 0.0, 1.0])

        # Rerank candidates [0, 2, 4]
        results = ranker.rerank([0, 2, 4], all_neural, all_logical)

        # Results should be sorted by score
        assert results[0][0] == 2  # Entity 2 has highest neural score and logical=1


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating a retrieval result."""
        result = RetrievalResult(
            entity_id=42,
            score=0.95,
            neural_score=0.9,
            logical_score=1.0,
            metadata={"name": "test"},
            explanation="Test entity",
        )
        assert result.entity_id == 42
        assert result.score == 0.95
        assert result.metadata["name"] == "test"

    def test_result_defaults(self) -> None:
        """Test result default values."""
        result = RetrievalResult(entity_id="test", score=0.5)
        assert result.neural_score == 0.0
        assert result.logical_score == 1.0
        assert result.metadata == {}
        assert result.explanation == ""
