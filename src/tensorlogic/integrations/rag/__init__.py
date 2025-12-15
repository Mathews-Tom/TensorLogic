"""RAG integration components for TensorLogic.

This module provides the core retrieval components for integrating TensorLogic
with Retrieval-Augmented Generation systems.

Components:
    - TensorLogicRetriever: Main retrieval interface with logical constraints
    - RetrievalResult: Structured result with scores and explanations
    - HybridRanker: Neural + symbolic hybrid scoring
    - ConstraintFilter: Logical constraint-based filtering
"""

from __future__ import annotations

from tensorlogic.integrations.rag.retriever import (
    TensorLogicRetriever,
    RetrievalResult,
)
from tensorlogic.integrations.rag.ranker import HybridRanker
from tensorlogic.integrations.rag.constraints import ConstraintFilter

__all__ = [
    "TensorLogicRetriever",
    "RetrievalResult",
    "HybridRanker",
    "ConstraintFilter",
]
