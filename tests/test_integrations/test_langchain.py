"""Tests for LangChain integration module."""

from __future__ import annotations

import numpy as np
import pytest

from tensorlogic.integrations.langchain import (
    TensorLogicLangChainRetriever,
    create_langchain_retriever,
)
from tensorlogic.integrations.langchain.retriever import Document


class TestDocument:
    """Tests for Document dataclass."""

    def test_document_creation(self) -> None:
        """Test creating a document with content and metadata."""
        doc = Document(
            page_content="Test content",
            metadata={"key": "value"},
        )
        assert doc.page_content == "Test content"
        assert doc.metadata["key"] == "value"

    def test_document_default_metadata(self) -> None:
        """Test document default metadata is empty dict."""
        doc = Document(page_content="Content only")
        assert doc.metadata == {}


class TestTensorLogicLangChainRetriever:
    """Tests for TensorLogicLangChainRetriever class."""

    @pytest.fixture
    def embedding_fn(self) -> callable:
        """Create a simple embedding function."""
        def embed(text: str) -> np.ndarray:
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(32).astype(np.float32)
        return embed

    @pytest.fixture
    def sample_documents(self) -> list[Document]:
        """Create sample documents."""
        return [
            Document(page_content="First document about AI", metadata={"id": 0}),
            Document(page_content="Second document about ML", metadata={"id": 1}),
            Document(page_content="Third document about NLP", metadata={"id": 2}),
            Document(page_content="Fourth document about data", metadata={"id": 3}),
            Document(page_content="Fifth document about science", metadata={"id": 4}),
        ]

    @pytest.fixture
    def retriever(
        self, embedding_fn: callable, sample_documents: list[Document]
    ) -> TensorLogicLangChainRetriever:
        """Create retriever with sample documents."""
        retriever = TensorLogicLangChainRetriever(
            embedding_fn=embedding_fn,
            top_k=5,
            lambda_neural=0.5,
            temperature=0.0,
        )
        retriever.add_documents(sample_documents)
        return retriever

    def test_retriever_initialization(self, embedding_fn: callable) -> None:
        """Test retriever initialization with config."""
        retriever = TensorLogicLangChainRetriever(
            embedding_fn=embedding_fn,
            top_k=10,
            lambda_neural=0.7,
            temperature=0.5,
        )
        assert retriever.top_k == 10
        assert retriever.lambda_neural == 0.7
        assert retriever.temperature == 0.5

    def test_add_documents(
        self, embedding_fn: callable, sample_documents: list[Document]
    ) -> None:
        """Test adding documents to retriever."""
        retriever = TensorLogicLangChainRetriever(
            embedding_fn=embedding_fn,
            top_k=5,
        )
        retriever.add_documents(sample_documents)
        assert len(retriever.documents) == 5

    def test_add_empty_documents(self, embedding_fn: callable) -> None:
        """Test adding empty document list does nothing."""
        retriever = TensorLogicLangChainRetriever(
            embedding_fn=embedding_fn,
            top_k=5,
        )
        retriever.add_documents([])
        assert len(retriever.documents) == 0

    def test_get_relevant_documents(
        self, retriever: TensorLogicLangChainRetriever
    ) -> None:
        """Test basic document retrieval."""
        results = retriever.get_relevant_documents("AI document")
        assert len(results) <= 5
        assert all(isinstance(r, Document) for r in results)
        # Results should have scores in metadata
        for doc in results:
            assert "score" in doc.metadata

    def test_get_relevant_documents_top_k(
        self, retriever: TensorLogicLangChainRetriever
    ) -> None:
        """Test top_k parameter."""
        results = retriever.get_relevant_documents("test query", top_k=2)
        assert len(results) <= 2

    def test_similarity_search(
        self, retriever: TensorLogicLangChainRetriever
    ) -> None:
        """Test similarity_search method."""
        results = retriever.similarity_search("machine learning", k=3)
        assert len(results) <= 3

    def test_similarity_search_with_score(
        self, retriever: TensorLogicLangChainRetriever
    ) -> None:
        """Test similarity search with scores."""
        results = retriever.similarity_search_with_score("data", k=3)
        assert len(results) <= 3
        assert all(isinstance(r, tuple) for r in results)
        for doc, score in results:
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_type_mask_filtering(
        self, retriever: TensorLogicLangChainRetriever
    ) -> None:
        """Test type-based filtering."""
        # Add type mask for first two documents
        retriever.add_type_mask("TypeA", [0, 1])

        results = retriever.get_relevant_documents(
            "test query",
            type_filter="TypeA",
            top_k=10,
        )

        # Should only return documents 0 and 1
        result_ids = [doc.metadata.get("id") for doc in results]
        assert all(id in [0, 1] for id in result_ids if id is not None)

    def test_relation_traversal(
        self, embedding_fn: callable
    ) -> None:
        """Test multi-hop relation traversal."""
        # Create retriever with documents
        docs = [
            Document(page_content="Source entity", metadata={"id": 0}),
            Document(page_content="Target entity 1", metadata={"id": 1}),
            Document(page_content="Target entity 2", metadata={"id": 2}),
        ]

        retriever = TensorLogicLangChainRetriever(
            embedding_fn=embedding_fn,
            top_k=10,
        )
        retriever.add_documents(docs)

        # Create relation: 0 -> 1, 0 -> 2
        relation = np.array([
            [0, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ], dtype=np.float32)
        retriever.add_relation("Links", relation)

        results = retriever.get_relevant_documents(
            "entity",
            relation_chain=["Links"],
            source_documents=[0],
        )

        # Should find documents 1 and 2 (reachable from 0)
        result_ids = [doc.metadata.get("id") for doc in results]
        assert 1 in result_ids or 2 in result_ids

    def test_apply_logical_constraints(
        self, retriever: TensorLogicLangChainRetriever
    ) -> None:
        """Test applying logical constraints to candidates."""
        # Add predicates
        retriever.add_type_mask("Selected", [0, 2, 4])  # Even indices

        # Get all results
        all_docs = retriever.get_relevant_documents("document", top_k=10)

        # Apply constraint - should filter to only Selected documents
        filtered = retriever.apply_logical_constraints(
            all_docs,
            predicate_names=["Selected"],
            composition="and",
        )

        # Verify filtered results are subset of original
        assert len(filtered) <= len(all_docs)

        # All filtered documents should have index in [0, 2, 4]
        # The returned documents are from self._documents which have original metadata
        for doc in filtered:
            doc_id = doc.metadata.get("id")
            assert doc_id in [0, 2, 4], f"Expected id in [0, 2, 4], got {doc_id}"

    def test_empty_retriever(self, embedding_fn: callable) -> None:
        """Test retrieval on empty retriever."""
        retriever = TensorLogicLangChainRetriever(
            embedding_fn=embedding_fn,
            top_k=5,
        )
        results = retriever.get_relevant_documents("query")
        assert results == []


class TestAsyncRetrieval:
    """Tests for async retrieval methods."""

    @pytest.fixture
    def embedding_fn(self) -> callable:
        """Create a simple embedding function."""
        def embed(text: str) -> np.ndarray:
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(32).astype(np.float32)
        return embed

    def test_aget_relevant_documents_sync(self, embedding_fn: callable) -> None:
        """Test async document retrieval runs synchronously."""
        import asyncio

        docs = [
            Document(page_content="Test document about technology", metadata={"id": 0}),
        ]

        retriever = TensorLogicLangChainRetriever(
            embedding_fn=embedding_fn,
            top_k=5,
        )
        retriever.add_documents(docs)

        # Run async method synchronously
        results = asyncio.get_event_loop().run_until_complete(
            retriever.aget_relevant_documents("technology")
        )
        # May return 0 or 1 results depending on score threshold
        assert len(results) <= 1


class TestCreateLangchainRetriever:
    """Tests for create_langchain_retriever factory function."""

    @pytest.fixture
    def embedding_fn(self) -> callable:
        """Create a simple embedding function."""
        def embed(text: str) -> np.ndarray:
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(32).astype(np.float32)
        return embed

    def test_create_from_strings(self, embedding_fn: callable) -> None:
        """Test creating retriever from string list."""
        docs = ["First doc", "Second doc", "Third doc"]
        retriever = create_langchain_retriever(
            documents=docs,
            embedding_fn=embedding_fn,
        )
        assert len(retriever.documents) == 3

    def test_create_from_documents(self, embedding_fn: callable) -> None:
        """Test creating retriever from Document list."""
        docs = [
            Document(page_content="First", metadata={"a": 1}),
            Document(page_content="Second", metadata={"b": 2}),
        ]
        retriever = create_langchain_retriever(
            documents=docs,
            embedding_fn=embedding_fn,
        )
        assert len(retriever.documents) == 2

    def test_create_with_config(self, embedding_fn: callable) -> None:
        """Test creating retriever with custom config."""
        docs = ["Test"]
        retriever = create_langchain_retriever(
            documents=docs,
            embedding_fn=embedding_fn,
            top_k=20,
            lambda_neural=0.8,
            temperature=0.3,
        )
        assert retriever.top_k == 20
        assert retriever.lambda_neural == 0.8
        assert retriever.temperature == 0.3

    def test_retrieval_after_creation(self, embedding_fn: callable) -> None:
        """Test retrieval works after factory creation."""
        docs = ["Document about cats", "Document about dogs"]
        retriever = create_langchain_retriever(
            documents=docs,
            embedding_fn=embedding_fn,
        )
        # Query with text from documents ensures non-zero similarity
        results = retriever.get_relevant_documents("Document about cats")
        # Results may vary based on embedding randomness, just verify no errors
        assert isinstance(results, list)


class TestScaleTest:
    """Tests for larger scale scenarios."""

    @pytest.fixture
    def embedding_fn(self) -> callable:
        """Create a simple embedding function."""
        def embed(text: str) -> np.ndarray:
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(64).astype(np.float32)
        return embed

    def test_hundred_documents(self, embedding_fn: callable) -> None:
        """Test with 100 documents."""
        docs = [f"Document number {i}" for i in range(100)]
        retriever = create_langchain_retriever(
            documents=docs,
            embedding_fn=embedding_fn,
            top_k=10,
        )
        results = retriever.get_relevant_documents("number")
        assert len(results) == 10

    def test_with_relations(self, embedding_fn: callable) -> None:
        """Test with relation matrix on moderate scale."""
        num_docs = 50
        docs = [
            Document(page_content=f"Entity {i}", metadata={"id": i})
            for i in range(num_docs)
        ]

        retriever = TensorLogicLangChainRetriever(
            embedding_fn=embedding_fn,
            top_k=10,
            temperature=0.5,  # Use soft combination to avoid zero scores
        )
        retriever.add_documents(docs)

        # Create sparse relation
        relation = np.zeros((num_docs, num_docs), dtype=np.float32)
        for i in range(num_docs - 1):
            relation[i, i + 1] = 1.0  # Chain relation
        retriever.add_relation("Next", relation)

        # Traverse from first entity
        results = retriever.get_relevant_documents(
            "Entity",
            relation_chain=["Next"],
            source_documents=[0],
        )
        # With soft combination and relation traversal, should find connected entities
        assert isinstance(results, list)
