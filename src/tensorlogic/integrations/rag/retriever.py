"""TensorLogic RAG Retriever.

Core retriever for symbolic-aware retrieval with logical constraints.
Supports hybrid neural-symbolic scoring and temperature-controlled reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tensorlogic.backends import create_backend, TensorBackend
from tensorlogic.core import logical_and, logical_or, exists, forall


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert array-like to numpy array."""
    if isinstance(arr, np.ndarray):
        return arr
    # Handle MLX arrays
    if hasattr(arr, "tolist"):
        return np.array(arr.tolist())
    return np.asarray(arr)


@dataclass
class RetrievalResult:
    """Result from TensorLogic retrieval.

    Attributes:
        entity_id: Identifier of the retrieved entity
        score: Combined retrieval score (0-1)
        neural_score: Dense embedding similarity score
        logical_score: Logical constraint satisfaction score
        metadata: Additional entity metadata
        explanation: Human-readable reasoning trace
    """

    entity_id: int | str
    score: float
    neural_score: float = 0.0
    logical_score: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


class TensorLogicRetriever:
    """Symbolic-aware retriever using tensor logic operations.

    Combines dense embeddings with logical constraints for hybrid retrieval.
    Supports temperature-controlled reasoning for precision-recall tradeoffs.

    Args:
        backend: TensorBackend instance (defaults to auto-detected)
        temperature: Reasoning temperature (0=strict, >0=relaxed)
        lambda_neural: Weight for neural scores (0-1)

    Example:
        >>> retriever = TensorLogicRetriever()
        >>> retriever.index_entities(embeddings, metadata=entity_info)
        >>> retriever.add_relation("WorksAt", works_at_matrix)
        >>> results = retriever.retrieve(
        ...     query_embedding=query_vec,
        ...     constraints="WorksAt(x, y) and LocatedIn(y, 'Seattle')",
        ...     top_k=10,
        ... )
    """

    def __init__(
        self,
        backend: TensorBackend | None = None,
        temperature: float = 0.0,
        lambda_neural: float = 0.5,
    ) -> None:
        """Initialize retriever.

        Args:
            backend: TensorBackend instance (auto-detected if None)
            temperature: Reasoning temperature (0=deductive, >0=analogical)
            lambda_neural: Weight for neural vs logical scores
        """
        self.backend = backend or create_backend()
        self.temperature = temperature
        self.lambda_neural = lambda_neural

        # Entity storage (using numpy for storage, backend for operations)
        self._entity_embeddings: np.ndarray | None = None
        self._entity_ids: list[int | str] = []
        self._entity_metadata: list[dict[str, Any]] = []

        # Relation tensors (numpy arrays)
        self._relations: dict[str, np.ndarray] = {}

        # Type masks for efficient filtering (numpy arrays)
        self._type_masks: dict[str, np.ndarray] = {}

    @property
    def num_entities(self) -> int:
        """Number of indexed entities."""
        return len(self._entity_ids)

    def index_entities(
        self,
        embeddings: Any,
        entity_ids: list[int | str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        """Index entities for retrieval.

        Args:
            embeddings: Entity embeddings array [num_entities, embedding_dim]
            entity_ids: Optional entity identifiers (defaults to indices)
            metadata: Optional metadata for each entity
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        num_entities = embeddings.shape[0]

        # Normalize embeddings for cosine similarity
        norms = np.sqrt(np.sum(embeddings * embeddings, axis=1, keepdims=True))
        self._entity_embeddings = embeddings / (norms + 1e-8)

        # Set entity IDs
        if entity_ids is not None:
            if len(entity_ids) != num_entities:
                raise ValueError(
                    f"entity_ids length {len(entity_ids)} != embeddings count {num_entities}"
                )
            self._entity_ids = list(entity_ids)
        else:
            self._entity_ids = list(range(num_entities))

        # Set metadata
        if metadata is not None:
            if len(metadata) != num_entities:
                raise ValueError(
                    f"metadata length {len(metadata)} != embeddings count {num_entities}"
                )
            self._entity_metadata = list(metadata)
        else:
            self._entity_metadata = [{} for _ in range(num_entities)]

    def add_relation(self, name: str, tensor: Any) -> None:
        """Add a relation tensor.

        Args:
            name: Relation name (e.g., 'WorksAt', 'LocatedIn')
            tensor: Binary relation matrix [num_entities, num_entities]
        """
        tensor = np.asarray(tensor, dtype=np.float32)
        expected_shape = (self.num_entities, self.num_entities)
        if tensor.shape != expected_shape:
            raise ValueError(f"Relation tensor shape {tensor.shape} != {expected_shape}")
        self._relations[name] = tensor

    def add_type_mask(self, type_name: str, mask: Any) -> None:
        """Add a type mask for entity filtering.

        Args:
            type_name: Type name (e.g., 'Person', 'Organization')
            mask: Binary mask [num_entities] where 1 = has type
        """
        mask = np.asarray(mask, dtype=np.float32)
        if mask.shape[0] != self.num_entities:
            raise ValueError(f"Mask length {mask.shape[0]} != {self.num_entities}")
        self._type_masks[type_name] = mask

    def compute_neural_scores(self, query_embedding: Any) -> np.ndarray:
        """Compute neural similarity scores.

        Args:
            query_embedding: Query embedding vector [embedding_dim]

        Returns:
            Similarity scores [num_entities]
        """
        if self._entity_embeddings is None:
            raise ValueError("No entities indexed. Call index_entities() first.")

        query = np.asarray(query_embedding, dtype=np.float32)

        # Normalize query
        query_norm = np.sqrt(np.sum(query * query))
        query = query / (query_norm + 1e-8)

        # Cosine similarity via dot product (embeddings already normalized)
        scores = np.einsum("d,nd->n", query, self._entity_embeddings)

        # Clamp to [0, 1]
        scores = np.maximum(scores, 0.0)
        scores = np.minimum(scores, 1.0)

        return scores

    def compute_logical_scores(
        self,
        type_filter: str | None = None,
        relation_chain: list[str] | None = None,
        source_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute logical constraint satisfaction scores.

        Args:
            type_filter: Entity type to filter by
            relation_chain: List of relations for multi-hop reasoning
            source_mask: Starting entity mask for relation traversal

        Returns:
            Logical scores [num_entities]
        """
        num_entities = self.num_entities
        scores = np.ones(num_entities, dtype=np.float32)

        # Apply type filter
        if type_filter is not None:
            if type_filter not in self._type_masks:
                raise ValueError(f"Unknown type: {type_filter}")
            type_mask = self._type_masks[type_filter]
            scores = _to_numpy(
                logical_and(scores, type_mask, backend=self.backend)
            )

        # Apply relation chain (multi-hop reasoning)
        if relation_chain is not None and source_mask is not None:
            current_mask = np.asarray(source_mask, dtype=np.float32)

            for relation_name in relation_chain:
                if relation_name not in self._relations:
                    raise ValueError(f"Unknown relation: {relation_name}")

                relation = self._relations[relation_name]

                # EXISTS: find entities reachable through relation
                # result[j] = exists(current[i] AND relation[i,j])
                expanded = current_mask[:, np.newaxis] * np.ones((1, num_entities))
                conjunction = _to_numpy(
                    logical_and(expanded, relation, backend=self.backend)
                )
                current_mask = _to_numpy(
                    exists(conjunction, axis=0, backend=self.backend)
                )

            scores = _to_numpy(
                logical_and(scores, current_mask, backend=self.backend)
            )

        return scores

    def retrieve(
        self,
        query_embedding: Any | None = None,
        type_filter: str | None = None,
        relation_chain: list[str] | None = None,
        source_entities: list[int | str] | None = None,
        top_k: int = 10,
        temperature: float | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve entities matching query and constraints.

        Args:
            query_embedding: Query vector for neural similarity
            type_filter: Filter by entity type
            relation_chain: Multi-hop relation path
            source_entities: Starting entities for relation traversal
            top_k: Number of results to return
            temperature: Override default temperature

        Returns:
            List of RetrievalResult sorted by score
        """
        if self._entity_embeddings is None:
            raise ValueError("No entities indexed. Call index_entities() first.")

        temp = temperature if temperature is not None else self.temperature
        num_entities = self.num_entities

        # Neural scores
        if query_embedding is not None:
            neural_scores = self.compute_neural_scores(query_embedding)
        else:
            neural_scores = np.ones(num_entities, dtype=np.float32)

        # Create source mask for relation traversal
        source_mask = None
        if source_entities is not None:
            source_mask = np.zeros(num_entities, dtype=np.float32)
            for entity in source_entities:
                if entity in self._entity_ids:
                    idx = self._entity_ids.index(entity)
                    source_mask[idx] = 1.0

        # Logical scores
        logical_scores = self.compute_logical_scores(
            type_filter=type_filter,
            relation_chain=relation_chain,
            source_mask=source_mask,
        )

        # Temperature-controlled combination
        if temp > 0:
            # Soft combination with temperature
            combined = (
                self.lambda_neural * neural_scores
                + (1 - self.lambda_neural) * logical_scores
            )
        else:
            # Hard logical filtering, then neural ranking
            combined = neural_scores * logical_scores

        # Get top-k indices
        top_indices = np.argsort(combined)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            if combined[idx] > 0:  # Only include non-zero scores
                results.append(
                    RetrievalResult(
                        entity_id=self._entity_ids[idx],
                        score=float(combined[idx]),
                        neural_score=float(neural_scores[idx]),
                        logical_score=float(logical_scores[idx]),
                        metadata=self._entity_metadata[idx].copy(),
                        explanation=self._generate_explanation(
                            idx, type_filter, relation_chain
                        ),
                    )
                )

        return results

    def _generate_explanation(
        self,
        entity_idx: int,
        type_filter: str | None,
        relation_chain: list[str] | None,
    ) -> str:
        """Generate human-readable explanation for retrieval.

        Args:
            entity_idx: Index of retrieved entity
            type_filter: Applied type filter
            relation_chain: Applied relation chain

        Returns:
            Explanation string
        """
        entity_id = self._entity_ids[entity_idx]
        parts = [f"Entity {entity_id}"]

        if type_filter:
            parts.append(f"has type '{type_filter}'")

        if relation_chain:
            chain_str = " → ".join(relation_chain)
            parts.append(f"reachable via {chain_str}")

        return " ".join(parts)

    def batch_retrieve(
        self,
        query_embeddings: Any,
        top_k: int = 10,
        **kwargs: Any,
    ) -> list[list[RetrievalResult]]:
        """Batch retrieval for multiple queries.

        Args:
            query_embeddings: Query vectors [num_queries, embedding_dim]
            top_k: Number of results per query
            **kwargs: Additional arguments passed to retrieve()

        Returns:
            List of result lists, one per query
        """
        queries = np.asarray(query_embeddings)
        num_queries = queries.shape[0]

        all_results = []
        for i in range(num_queries):
            query = queries[i]
            results = self.retrieve(query_embedding=query, top_k=top_k, **kwargs)
            all_results.append(results)

        return all_results
