"""TensorLogic integrations with RAG frameworks.

This module provides adapters for popular RAG (Retrieval-Augmented Generation)
frameworks, enabling symbolic-aware retrieval and hybrid neural-symbolic reasoning.

**Key Features:**
- TensorLogicRetriever: Core symbolic retrieval with logical constraints
- LangChain adapter: Drop-in BaseRetriever for LangChain pipelines
- Hybrid scoring: Combine dense embeddings with logical constraints

**Usage:**

Basic retrieval::

    from tensorlogic.integrations import TensorLogicRetriever

    retriever = TensorLogicRetriever(
        entities=entity_embeddings,
        relations=relation_tensors,
    )
    results = retriever.retrieve(
        query="People who work at tech companies in Seattle",
        constraints="HasType(x, 'Person') and WorksAt(x, y) and LocatedIn(y, 'Seattle')",
        top_k=10,
    )

With LangChain::

    from tensorlogic.integrations.langchain import TensorLogicLangChainRetriever
    from langchain.chains import RetrievalQA

    retriever = TensorLogicLangChainRetriever(
        tensorlogic_retriever=retriever,
        temperature=0.3,
    )
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

See Also:
    - docs/research/rag-goals.md for research roadmap
    - examples/langchain_integration.py for full examples
"""

from __future__ import annotations

from tensorlogic.integrations.rag import (
    TensorLogicRetriever,
    RetrievalResult,
    HybridRanker,
    ConstraintFilter,
)
from tensorlogic.integrations.langchain import (
    TensorLogicLangChainRetriever,
    create_langchain_retriever,
)

__all__ = [
    # RAG core
    "TensorLogicRetriever",
    "RetrievalResult",
    "HybridRanker",
    "ConstraintFilter",
    # LangChain adapter
    "TensorLogicLangChainRetriever",
    "create_langchain_retriever",
]
