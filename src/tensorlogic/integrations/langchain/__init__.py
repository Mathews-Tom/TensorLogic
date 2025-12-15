"""LangChain integration for TensorLogic.

Provides LangChain-compatible retrievers using TensorLogic's
neural-symbolic retrieval capabilities.

Components:
    - TensorLogicLangChainRetriever: LangChain BaseRetriever adapter
    - create_langchain_retriever: Factory function for easy setup
"""

from __future__ import annotations

from tensorlogic.integrations.langchain.retriever import (
    TensorLogicLangChainRetriever,
    create_langchain_retriever,
)

__all__ = [
    "TensorLogicLangChainRetriever",
    "create_langchain_retriever",
]
