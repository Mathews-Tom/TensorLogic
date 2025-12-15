"""LangChain BaseRetriever adapter for TensorLogic.

Provides a LangChain-compatible retriever that leverages TensorLogic's
neural-symbolic reasoning capabilities for hybrid retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tensorlogic.integrations.rag import (
    TensorLogicRetriever,
    ConstraintFilter,
    HybridRanker,
)
from tensorlogic.integrations.rag.ranker import RankingConfig


@dataclass
class Document:
    """Simple document representation compatible with LangChain.

    Attributes:
        page_content: The document text content
        metadata: Additional document metadata
    """

    page_content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorLogicLangChainRetriever:
    """LangChain-compatible retriever using TensorLogic.

    Wraps TensorLogicRetriever to provide a LangChain BaseRetriever-compatible
    interface with neural-symbolic hybrid retrieval.

    Attributes:
        retriever: Underlying TensorLogicRetriever instance
        documents: List of indexed documents
        embedding_fn: Function to compute embeddings from text
        top_k: Default number of results to return

    Example:
        >>> def embed(text):
        ...     # Your embedding function
        ...     return np.random.randn(64)
        >>> retriever = TensorLogicLangChainRetriever(
        ...     embedding_fn=embed,
        ...     top_k=5,
        ... )
        >>> retriever.add_documents(documents)
        >>> results = retriever.get_relevant_documents("query text")
    """

    embedding_fn: Any  # Callable[[str], np.ndarray]
    top_k: int = 10
    lambda_neural: float = 0.5
    temperature: float = 0.0

    def __post_init__(self) -> None:
        """Initialize internal retriever."""
        self._retriever = TensorLogicRetriever(
            temperature=self.temperature,
            lambda_neural=self.lambda_neural,
        )
        self._constraint_filter = ConstraintFilter(temperature=self.temperature)
        self._ranker = HybridRanker(
            config=RankingConfig(
                lambda_neural=self.lambda_neural,
                temperature=self.temperature,
            )
        )
        self._documents: list[Document] = []
        self._indexed = False

    @property
    def documents(self) -> list[Document]:
        """Get indexed documents."""
        return self._documents

    def add_documents(
        self,
        documents: list[Document],
        entity_ids: list[int | str] | None = None,
    ) -> None:
        """Add documents to the retriever index.

        Args:
            documents: List of Document objects to index
            entity_ids: Optional custom IDs (defaults to indices)
        """
        if len(documents) == 0:
            return

        self._documents.extend(documents)

        # Compute embeddings for all documents
        embeddings = []
        for doc in documents:
            emb = self.embedding_fn(doc.page_content)
            embeddings.append(emb)

        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Extract metadata for indexing
        metadata_list = [doc.metadata for doc in documents]

        # Index in underlying retriever
        self._retriever.index_entities(
            embeddings=embeddings_array,
            entity_ids=entity_ids,
            metadata=metadata_list,
        )
        self._indexed = True

    def add_type_mask(self, type_name: str, document_indices: list[int]) -> None:
        """Add type mask for document filtering.

        Args:
            type_name: Name of the type (e.g., 'technical', 'legal')
            document_indices: Indices of documents with this type
        """
        mask = np.zeros(len(self._documents), dtype=np.float32)
        for idx in document_indices:
            if 0 <= idx < len(self._documents):
                mask[idx] = 1.0
        self._retriever.add_type_mask(type_name, mask)
        self._constraint_filter.add_predicate(type_name, mask)

    def add_relation(self, name: str, matrix: Any) -> None:
        """Add a relation between documents.

        Args:
            name: Relation name (e.g., 'cites', 'related_to')
            matrix: Binary relation matrix [num_docs, num_docs]
        """
        self._retriever.add_relation(name, matrix)
        self._constraint_filter.add_relation(name, np.asarray(matrix))

    def get_relevant_documents(
        self,
        query: str,
        *,
        type_filter: str | None = None,
        relation_chain: list[str] | None = None,
        source_documents: list[int] | None = None,
        top_k: int | None = None,
    ) -> list[Document]:
        """Retrieve relevant documents for a query.

        LangChain-compatible interface method.

        Args:
            query: Query text
            type_filter: Filter by document type
            relation_chain: Multi-hop relation path
            source_documents: Starting document indices for relation traversal
            top_k: Number of results (defaults to self.top_k)

        Returns:
            List of relevant Document objects
        """
        if not self._indexed:
            return []

        k = top_k if top_k is not None else self.top_k

        # Compute query embedding
        query_embedding = self.embedding_fn(query)

        # Retrieve using TensorLogicRetriever
        results = self._retriever.retrieve(
            query_embedding=query_embedding,
            type_filter=type_filter,
            relation_chain=relation_chain,
            source_entities=source_documents,
            top_k=k,
        )

        # Convert to Documents with scores in metadata
        documents = []
        for result in results:
            idx = result.entity_id
            if isinstance(idx, int) and 0 <= idx < len(self._documents):
                doc = self._documents[idx]
                # Add retrieval scores to metadata
                enriched_metadata = {
                    **doc.metadata,
                    "score": result.score,
                    "neural_score": result.neural_score,
                    "logical_score": result.logical_score,
                    "explanation": result.explanation,
                }
                documents.append(
                    Document(page_content=doc.page_content, metadata=enriched_metadata)
                )

        return documents

    async def aget_relevant_documents(
        self,
        query: str,
        **kwargs: Any,
    ) -> list[Document]:
        """Async version of get_relevant_documents.

        Currently runs synchronously as TensorLogic operations are CPU/GPU bound.

        Args:
            query: Query text
            **kwargs: Additional arguments passed to get_relevant_documents

        Returns:
            List of relevant Document objects
        """
        return self.get_relevant_documents(query, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[Document]:
        """Perform similarity search (alias for get_relevant_documents).

        Args:
            query: Query text
            k: Number of results
            **kwargs: Additional arguments

        Returns:
            List of similar Document objects
        """
        return self.get_relevant_documents(query, top_k=k, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Perform similarity search with scores.

        Args:
            query: Query text
            k: Number of results
            **kwargs: Additional arguments

        Returns:
            List of (Document, score) tuples
        """
        docs = self.get_relevant_documents(query, top_k=k, **kwargs)
        return [(doc, doc.metadata.get("score", 0.0)) for doc in docs]

    def apply_logical_constraints(
        self,
        candidates: list[Document],
        predicate_names: list[str],
        composition: str = "and",
    ) -> list[Document]:
        """Filter candidates using logical constraints.

        Args:
            candidates: Candidate documents
            predicate_names: Predicates to apply
            composition: How to compose ('and' or 'or')

        Returns:
            Filtered documents satisfying constraints
        """
        if not candidates:
            return []

        # Build content to index mapping for lookup
        content_to_idx = {
            doc.page_content: i for i, doc in enumerate(self._documents)
        }

        # Get candidate indices by matching content
        candidate_indices = []
        candidate_docs = []
        for doc in candidates:
            idx = content_to_idx.get(doc.page_content)
            if idx is not None:
                candidate_indices.append(idx)
                candidate_docs.append(doc)

        if not candidate_indices:
            return candidates

        # Create scores array for candidates
        scores = np.ones(len(self._documents), dtype=np.float32)

        # Apply constraints
        filtered_scores = self._constraint_filter.apply(
            candidates=scores,
            predicate_names=predicate_names,
            composition=composition,
        )

        # Return documents that pass the filter
        result = []
        for idx in candidate_indices:
            if filtered_scores[idx] > 0.5:
                result.append(self._documents[idx])

        return result


def create_langchain_retriever(
    documents: list[Document] | list[str],
    embedding_fn: Any,
    top_k: int = 10,
    lambda_neural: float = 0.5,
    temperature: float = 0.0,
) -> TensorLogicLangChainRetriever:
    """Factory function to create a LangChain-compatible retriever.

    Args:
        documents: Documents to index (strings or Document objects)
        embedding_fn: Function to compute embeddings from text
        top_k: Default number of results
        lambda_neural: Weight for neural scores (0-1)
        temperature: Reasoning temperature (0=strict, >0=relaxed)

    Returns:
        Configured TensorLogicLangChainRetriever instance

    Example:
        >>> docs = ["First document.", "Second document."]
        >>> def embed(text): return np.random.randn(64)
        >>> retriever = create_langchain_retriever(docs, embed)
        >>> results = retriever.get_relevant_documents("query")
    """
    # Convert strings to Document objects if needed
    doc_objects = []
    for i, doc in enumerate(documents):
        if isinstance(doc, str):
            doc_objects.append(Document(page_content=doc, metadata={"index": i}))
        else:
            doc_objects.append(doc)

    # Create and configure retriever
    retriever = TensorLogicLangChainRetriever(
        embedding_fn=embedding_fn,
        top_k=top_k,
        lambda_neural=lambda_neural,
        temperature=temperature,
    )

    # Index documents
    retriever.add_documents(doc_objects)

    return retriever
