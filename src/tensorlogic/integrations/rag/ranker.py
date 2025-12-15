"""Hybrid neural-symbolic ranker for RAG.

Combines dense embedding similarity with logical constraint satisfaction
for improved retrieval ranking. Supports temperature-controlled fusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tensorlogic.backends import create_backend, TensorBackend


@dataclass
class RankingConfig:
    """Configuration for hybrid ranking.

    Attributes:
        lambda_neural: Weight for neural scores (0-1)
        temperature: Reasoning temperature for soft constraints
        min_logical_score: Minimum logical score threshold
        normalize_scores: Whether to normalize output scores
    """

    lambda_neural: float = 0.5
    temperature: float = 0.0
    min_logical_score: float = 0.0
    normalize_scores: bool = True


class HybridRanker:
    """Hybrid neural-symbolic ranking for RAG retrieval.

    Implements the scoring function:
        Score(doc, query) = λ · Neural(doc, query) + (1-λ) · Logic(doc, query)

    Where:
        - Neural(doc, query) = cosine_similarity(embed(doc), embed(query))
        - Logic(doc, query) = logical constraint satisfaction
        - λ (lambda_neural) balances neural vs symbolic contribution

    Args:
        backend: TensorBackend instance
        config: RankingConfig with hyperparameters

    Example:
        >>> ranker = HybridRanker(config=RankingConfig(lambda_neural=0.7))
        >>> scores = ranker.rank(
        ...     neural_scores=embedding_similarities,
        ...     logical_scores=constraint_scores,
        ... )
    """

    def __init__(
        self,
        backend: TensorBackend | None = None,
        config: RankingConfig | None = None,
    ) -> None:
        """Initialize hybrid ranker.

        Args:
            backend: TensorBackend instance (auto-detected if None)
            config: RankingConfig (defaults used if None)
        """
        self.backend = backend or create_backend()
        self.config = config or RankingConfig()

    def rank(
        self,
        neural_scores: Any,
        logical_scores: Any,
        temperature: float | None = None,
    ) -> np.ndarray:
        """Compute hybrid ranking scores.

        Args:
            neural_scores: Neural similarity scores [num_entities]
            logical_scores: Logical constraint scores [num_entities]
            temperature: Override config temperature

        Returns:
            Combined ranking scores [num_entities]
        """
        neural = np.asarray(neural_scores, dtype=np.float32)
        logical = np.asarray(logical_scores, dtype=np.float32)

        temp = temperature if temperature is not None else self.config.temperature
        lam = self.config.lambda_neural

        if temp > 0:
            # Soft combination: weighted average
            combined = lam * neural + (1 - lam) * logical
        else:
            # Hard filtering: logical gates neural
            # If logical_score < threshold, zero out
            logical_gate = (logical > self.config.min_logical_score).astype(np.float32)
            combined = neural * logical_gate

        # Normalize if requested
        if self.config.normalize_scores:
            max_score = np.max(combined)
            if max_score > 0:
                combined = combined / max_score

        return combined

    def rank_with_boost(
        self,
        neural_scores: Any,
        logical_scores: Any,
        boost_mask: Any,
        boost_factor: float = 1.5,
    ) -> np.ndarray:
        """Rank with boosting for specific entities.

        Args:
            neural_scores: Neural similarity scores [num_entities]
            logical_scores: Logical constraint scores [num_entities]
            boost_mask: Binary mask for entities to boost [num_entities]
            boost_factor: Multiplicative boost factor

        Returns:
            Boosted ranking scores [num_entities]
        """
        base_scores = self.rank(neural_scores, logical_scores)
        boost = np.asarray(boost_mask, dtype=np.float32)

        # Apply multiplicative boost
        boosted = base_scores * (1 + (boost_factor - 1) * boost)

        # Normalize
        if self.config.normalize_scores:
            max_score = np.max(boosted)
            if max_score > 0:
                boosted = boosted / max_score

        return boosted

    def rank_reciprocal_fusion(
        self,
        score_lists: list[Any],
        k: int = 60,
    ) -> np.ndarray:
        """Reciprocal Rank Fusion (RRF) for combining multiple rankings.

        RRF score = sum(1 / (k + rank_i)) for each ranking list

        Args:
            score_lists: List of score arrays from different rankers
            k: RRF constant (default 60)

        Returns:
            Fused ranking scores [num_entities]
        """
        if len(score_lists) == 0:
            raise ValueError("At least one score list required")

        # Get number of entities from first list
        first_scores = np.asarray(score_lists[0])
        num_entities = len(first_scores)

        # Initialize RRF scores
        rrf_scores = np.zeros(num_entities, dtype=np.float32)

        for scores in score_lists:
            scores_np = np.asarray(scores)
            if len(scores_np) != num_entities:
                raise ValueError("All score lists must have same length")

            # Get ranks (1-indexed, lower is better)
            ranks = np.argsort(np.argsort(-scores_np)) + 1

            # Add RRF contribution
            rrf_scores += 1.0 / (k + ranks)

        # Normalize
        if self.config.normalize_scores:
            max_score = np.max(rrf_scores)
            if max_score > 0:
                rrf_scores = rrf_scores / max_score

        return rrf_scores

    def get_top_k(
        self,
        scores: Any,
        k: int = 10,
    ) -> tuple[list[int], list[float]]:
        """Get top-k indices and scores.

        Args:
            scores: Ranking scores [num_entities]
            k: Number of top results

        Returns:
            Tuple of (indices, scores) for top-k entities
        """
        scores_np = np.asarray(scores)
        top_indices = np.argsort(scores_np)[::-1][:k]
        top_scores = scores_np[top_indices]

        return list(top_indices), list(top_scores)

    def rerank(
        self,
        candidate_indices: list[int],
        all_neural_scores: Any,
        all_logical_scores: Any,
    ) -> list[tuple[int, float]]:
        """Rerank a candidate set using hybrid scoring.

        Args:
            candidate_indices: Indices of candidate entities
            all_neural_scores: Neural scores for all entities
            all_logical_scores: Logical scores for all entities

        Returns:
            List of (index, score) tuples sorted by score descending
        """
        neural = np.asarray(all_neural_scores)
        logical = np.asarray(all_logical_scores)

        # Get scores for candidates
        candidate_neural = neural[candidate_indices]
        candidate_logical = logical[candidate_indices]

        # Compute hybrid scores
        combined = self.rank(candidate_neural, candidate_logical)

        # Sort and return
        sorted_indices = np.argsort(combined)[::-1]
        results = [
            (candidate_indices[i], float(combined[i])) for i in sorted_indices
        ]

        return results
