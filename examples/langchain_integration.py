"""LangChain integration example with 10K+ entity knowledge graph.

This example demonstrates TensorLogic's hybrid neural-symbolic retrieval
integrated with LangChain's retrieval interface.

Features demonstrated:
- 10K+ entity knowledge graph construction
- LangChain-compatible retriever interface
- Hybrid neural-symbolic scoring
- Logical constraint filtering
- Multi-hop relation traversal

Run with:
    uv run python examples/langchain_integration.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from tensorlogic.integrations import (
    TensorLogicLangChainRetriever,
    create_langchain_retriever,
)
from tensorlogic.integrations.langchain.retriever import Document


# =============================================================================
# Configuration
# =============================================================================

EMBEDDING_DIM = 64
NUM_COMPANIES = 100
NUM_PEOPLE = 10000  # 10K entities
NUM_LOCATIONS = 50
SPARSITY_WORKS_AT = 0.01  # 1% density for WorksAt relation
SPARSITY_LOCATED_IN = 0.3  # 30% density for LocatedIn


# =============================================================================
# Synthetic Data Generation
# =============================================================================


@dataclass
class Entity:
    """Entity in the knowledge graph."""

    id: int
    name: str
    type: str  # 'Person', 'Company', 'Location'
    embedding: np.ndarray
    description: str


def generate_entity_embedding(entity_type: str, seed: int) -> np.ndarray:
    """Generate embeddings with type-based clustering.

    Entities of the same type will have similar embeddings, enabling
    meaningful semantic search.
    """
    rng = np.random.RandomState(seed)

    # Type-based cluster centers
    type_offsets = {
        "Person": np.array([1.0, 0.0, 0.0]),
        "Company": np.array([0.0, 1.0, 0.0]),
        "Location": np.array([0.0, 0.0, 1.0]),
    }

    # Base embedding with noise
    base = rng.randn(EMBEDDING_DIM - 3).astype(np.float32) * 0.3
    type_signal = type_offsets[entity_type]

    # Combine type signal with noise
    embedding = np.concatenate([type_signal, base])
    return embedding / (np.linalg.norm(embedding) + 1e-8)


def generate_knowledge_graph() -> tuple[
    list[Entity],
    np.ndarray,  # WorksAt relation
    np.ndarray,  # LocatedIn relation
]:
    """Generate synthetic knowledge graph with 10K+ entities.

    Returns:
        entities: List of Entity objects
        works_at: Binary matrix [num_entities, num_entities]
        located_in: Binary matrix [num_entities, num_entities]
    """
    print("Generating knowledge graph...")
    start_time = time.time()

    entities: list[Entity] = []
    entity_id = 0

    # Generate locations
    location_ids: list[int] = []
    for i in range(NUM_LOCATIONS):
        entities.append(
            Entity(
                id=entity_id,
                name=f"City_{i}",
                type="Location",
                embedding=generate_entity_embedding("Location", entity_id),
                description=f"City {i} is a major metropolitan area.",
            )
        )
        location_ids.append(entity_id)
        entity_id += 1

    # Generate companies
    company_ids: list[int] = []
    for i in range(NUM_COMPANIES):
        entities.append(
            Entity(
                id=entity_id,
                name=f"TechCorp_{i}",
                type="Company",
                embedding=generate_entity_embedding("Company", entity_id),
                description=f"TechCorp {i} is a technology company specializing in AI.",
            )
        )
        company_ids.append(entity_id)
        entity_id += 1

    # Generate people (the bulk of entities)
    person_ids: list[int] = []
    for i in range(NUM_PEOPLE):
        entities.append(
            Entity(
                id=entity_id,
                name=f"Person_{i}",
                type="Person",
                embedding=generate_entity_embedding("Person", entity_id),
                description=f"Person {i} is a professional in the technology industry.",
            )
        )
        person_ids.append(entity_id)
        entity_id += 1

    total_entities = len(entities)
    print(f"  Created {total_entities:,} entities")
    print(f"    - {NUM_LOCATIONS} locations")
    print(f"    - {NUM_COMPANIES} companies")
    print(f"    - {NUM_PEOPLE:,} people")

    # Generate WorksAt relation (Person -> Company)
    print("  Generating WorksAt relation...")
    works_at = np.zeros((total_entities, total_entities), dtype=np.float32)
    rng = np.random.RandomState(42)

    for person_id in person_ids:
        # Each person works at 0-3 companies (sparse)
        if rng.random() < SPARSITY_WORKS_AT * 100:  # ~10% of people work
            num_jobs = rng.randint(1, 4)
            employers = rng.choice(company_ids, size=min(num_jobs, NUM_COMPANIES), replace=False)
            for company_id in employers:
                works_at[person_id, company_id] = 1.0

    num_work_relations = int(works_at.sum())
    print(f"    WorksAt relations: {num_work_relations:,}")

    # Generate LocatedIn relation (Company -> Location)
    print("  Generating LocatedIn relation...")
    located_in = np.zeros((total_entities, total_entities), dtype=np.float32)

    for company_id in company_ids:
        # Each company is in 1-3 locations
        num_locations = rng.randint(1, 4)
        locations = rng.choice(location_ids, size=min(num_locations, NUM_LOCATIONS), replace=False)
        for loc_id in locations:
            located_in[company_id, loc_id] = 1.0

    num_location_relations = int(located_in.sum())
    print(f"    LocatedIn relations: {num_location_relations:,}")

    elapsed = time.time() - start_time
    print(f"  Knowledge graph generated in {elapsed:.2f}s")

    return entities, works_at, located_in


# =============================================================================
# LangChain Integration Demo
# =============================================================================


def demo_langchain_retrieval() -> None:
    """Demonstrate LangChain-compatible retrieval with logical constraints."""
    print("\n" + "=" * 70)
    print("LangChain Integration Demo with 10K+ Entities")
    print("=" * 70)

    # Generate knowledge graph
    entities, works_at, located_in = generate_knowledge_graph()

    # Convert entities to Documents
    documents = [
        Document(
            page_content=e.description,
            metadata={
                "id": e.id,
                "name": e.name,
                "type": e.type,
            },
        )
        for e in entities
    ]

    # Create embeddings lookup (simulating embedding function)
    entity_embeddings = {e.id: e.embedding for e in entities}

    def embed_text(text: str) -> np.ndarray:
        """Simple embedding function using entity lookup or query embedding."""
        # Check if this is a known entity description
        for e in entities:
            if e.description == text:
                return e.embedding

        # For queries, create a query embedding
        # In real usage, this would call an embedding model
        np.random.seed(hash(text) % 2**32)
        query_emb = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        return query_emb / (np.linalg.norm(query_emb) + 1e-8)

    # Create LangChain-compatible retriever
    print("\nCreating TensorLogicLangChainRetriever...")
    start_time = time.time()

    retriever = TensorLogicLangChainRetriever(
        embedding_fn=embed_text,
        top_k=10,
        lambda_neural=0.6,  # 60% neural, 40% logical
        temperature=0.0,  # Strict logical filtering
    )

    # Add documents
    retriever.add_documents(documents)

    # Add type masks
    person_indices = [e.id for e in entities if e.type == "Person"]
    company_indices = [e.id for e in entities if e.type == "Company"]
    location_indices = [e.id for e in entities if e.type == "Location"]

    retriever.add_type_mask("Person", person_indices)
    retriever.add_type_mask("Company", company_indices)
    retriever.add_type_mask("Location", location_indices)

    # Add relations
    retriever.add_relation("WorksAt", works_at)
    retriever.add_relation("LocatedIn", located_in)

    setup_time = time.time() - start_time
    print(f"  Retriever setup completed in {setup_time:.2f}s")

    # ==========================================================================
    # Demo 1: Basic similarity search
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Demo 1: Basic Similarity Search")
    print("-" * 50)

    query = "technology professional"
    start_time = time.time()
    results = retriever.similarity_search(query, k=5)
    query_time = (time.time() - start_time) * 1000

    print(f"  Query: '{query}'")
    print(f"  Query time: {query_time:.1f}ms")
    print(f"  Results ({len(results)}):")
    for i, doc in enumerate(results[:5], 1):
        print(f"    {i}. {doc.metadata.get('name', 'Unknown')} "
              f"(type={doc.metadata.get('type')}, score={doc.metadata.get('score', 0):.3f})")

    # ==========================================================================
    # Demo 2: Type-filtered retrieval
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Demo 2: Type-Filtered Retrieval (Person only)")
    print("-" * 50)

    query = "AI specialist"
    start_time = time.time()
    results = retriever.get_relevant_documents(query, type_filter="Person", top_k=5)
    query_time = (time.time() - start_time) * 1000

    print(f"  Query: '{query}' with type_filter='Person'")
    print(f"  Query time: {query_time:.1f}ms")
    print(f"  Results ({len(results)}, all should be Person):")
    for i, doc in enumerate(results[:5], 1):
        entity_type = doc.metadata.get("type", "Unknown")
        assert entity_type == "Person", f"Expected Person, got {entity_type}"
        print(f"    {i}. {doc.metadata.get('name')} (type={entity_type})")
    print("  All results verified as Person type")

    # ==========================================================================
    # Demo 3: Multi-hop relation traversal
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Demo 3: Multi-Hop Relation Traversal")
    print("-" * 50)
    print("  Query: People who work at companies in specific locations")
    print("  Relation chain: Person --WorksAt--> Company --LocatedIn--> Location")

    # Find a location that has companies
    source_location = None
    for loc_id in location_indices:
        # Check if any company is located here
        if located_in[:, loc_id].sum() > 0:
            source_location = loc_id
            break

    if source_location is not None:
        location_name = entities[source_location].name

        start_time = time.time()
        results = retriever.get_relevant_documents(
            query="technology professional",
            relation_chain=["WorksAt", "LocatedIn"],
            source_documents=[source_location],
            top_k=10,
        )
        query_time = (time.time() - start_time) * 1000

        print(f"  Source location: {location_name}")
        print(f"  Query time: {query_time:.1f}ms")
        print(f"  Found {len(results)} people connected to {location_name}")

    # ==========================================================================
    # Demo 4: Similarity search with scores
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Demo 4: Similarity Search with Scores")
    print("-" * 50)

    query = "company specializing"
    start_time = time.time()
    results_with_scores = retriever.similarity_search_with_score(query, k=5)
    query_time = (time.time() - start_time) * 1000

    print(f"  Query: '{query}'")
    print(f"  Query time: {query_time:.1f}ms")
    print("  Results with scores:")
    for doc, score in results_with_scores[:5]:
        print(f"    - {doc.metadata.get('name')}: {score:.4f}")

    # ==========================================================================
    # Demo 5: Factory function
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Demo 5: Using create_langchain_retriever Factory")
    print("-" * 50)

    # Create retriever using factory function
    simple_docs = [e.description for e in entities[:100]]  # First 100 entities

    start_time = time.time()
    simple_retriever = create_langchain_retriever(
        documents=simple_docs,
        embedding_fn=embed_text,
        top_k=5,
    )
    factory_time = time.time() - start_time

    print(f"  Created retriever with {len(simple_docs)} documents in {factory_time:.2f}s")

    results = simple_retriever.get_relevant_documents("metropolitan area")
    print(f"  Query 'metropolitan area' returned {len(results)} results")

    # ==========================================================================
    # Performance Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(f"  Total entities indexed: {len(entities):,}")
    print(f"  Embedding dimension: {EMBEDDING_DIM}")
    print(f"  Relations: WorksAt ({int(works_at.sum()):,}), LocatedIn ({int(located_in.sum()):,})")
    print(f"  Memory for embeddings: ~{len(entities) * EMBEDDING_DIM * 4 / 1024 / 1024:.1f} MB")
    print(f"  Memory for relations: ~{(works_at.nbytes + located_in.nbytes) / 1024 / 1024:.1f} MB")


def demo_logical_constraints() -> None:
    """Demonstrate logical constraint composition."""
    print("\n" + "=" * 70)
    print("Logical Constraint Composition Demo")
    print("=" * 70)

    # Simple example with smaller dataset
    documents = [
        Document(page_content="Technical documentation for API", metadata={"type": "technical", "id": 0}),
        Document(page_content="Legal contract for service", metadata={"type": "legal", "id": 1}),
        Document(page_content="Technical guide for developers", metadata={"type": "technical", "id": 2}),
        Document(page_content="Legal policy document", metadata={"type": "legal", "id": 3}),
        Document(page_content="Marketing brochure", metadata={"type": "marketing", "id": 4}),
    ]

    def simple_embed(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(32).astype(np.float32)

    retriever = TensorLogicLangChainRetriever(
        embedding_fn=simple_embed,
        top_k=10,
    )

    retriever.add_documents(documents)

    # Add type constraints
    retriever.add_type_mask("technical", [0, 2])  # Documents 0 and 2 are technical
    retriever.add_type_mask("legal", [1, 3])  # Documents 1 and 3 are legal

    # Demo: Filter to only technical documents
    print("\n  Filtering to technical documents only:")
    all_docs = retriever.get_relevant_documents("documentation")
    technical_docs = retriever.apply_logical_constraints(
        all_docs, predicate_names=["technical"], composition="and"
    )

    print(f"    All results: {len(all_docs)}")
    print(f"    After 'technical' constraint: {len(technical_docs)}")
    for doc in technical_docs:
        print(f"      - {doc.page_content[:50]}...")


if __name__ == "__main__":
    demo_langchain_retrieval()
    demo_logical_constraints()
    print("\nDemo completed successfully!")
