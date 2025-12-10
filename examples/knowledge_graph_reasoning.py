"""Knowledge Graph Reasoning with TensorLogic

This comprehensive example demonstrates neural-symbolic reasoning over a family
knowledge graph using TensorLogic. It showcases how logical rules can be
expressed as tensor operations for efficient inference.

Features Demonstrated:
    - Backend abstraction (MLX/NumPy)
    - Core logical operations (AND, OR, NOT, IMPLIES)
    - Quantifiers (EXISTS, FORALL)
    - High-level quantify() and reason() API
    - Temperature-controlled reasoning (deductive vs analogical)
    - Multiple compilation strategies comparison
    - Handling uncertain/fuzzy knowledge

Run example:
    uv run python examples/knowledge_graph_reasoning.py
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tensorlogic.api import quantify, reason
from tensorlogic.backends import TensorBackend, create_backend
from tensorlogic.compilation import create_strategy
from tensorlogic.core import logical_and, logical_implies, logical_not, logical_or
from tensorlogic.core.quantifiers import exists, forall


# =============================================================================
# SECTION 1: Knowledge Graph Setup
# =============================================================================


def create_family_knowledge_graph(backend: TensorBackend) -> dict[str, Any]:
    """Create a family knowledge graph with entities and relations.

    Family Tree Structure:
        Generation 1: Alice, Bob (married)
        Generation 2: Carol (child of Alice & Bob), David (Carol's spouse)
                      Eve (child of Alice & Bob)
        Generation 3: Frank (child of Carol & David), Grace (child of Carol & David)
                      Henry (child of Eve)

    Relations defined:
        - Parent[x, y]: x is parent of y
        - Sibling[x, y]: x and y are siblings
        - Married[x, y]: x and y are married
        - Loves[x, y]: x loves y (soft relation for fuzzy reasoning)

    Args:
        backend: Tensor backend for creating tensors

    Returns:
        Dictionary containing entities list and relation tensors
    """
    # Define entities (8 people in 3 generations)
    entities = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry"]
    n = len(entities)
    entity_to_idx = {name: i for i, name in enumerate(entities)}

    # Create Parent relation tensor (8x8 matrix)
    # Parent[i, j] = 1.0 means entity i is parent of entity j
    parent = np.zeros((n, n), dtype=np.float32)
    parent_facts = [
        ("Alice", "Carol"),
        ("Alice", "Eve"),
        ("Bob", "Carol"),
        ("Bob", "Eve"),
        ("Carol", "Frank"),
        ("Carol", "Grace"),
        ("David", "Frank"),
        ("David", "Grace"),
        ("Eve", "Henry"),
    ]
    for p, c in parent_facts:
        parent[entity_to_idx[p], entity_to_idx[c]] = 1.0

    # Create Sibling relation tensor (symmetric)
    sibling = np.zeros((n, n), dtype=np.float32)
    sibling_pairs = [
        ("Carol", "Eve"),
        ("Frank", "Grace"),
    ]
    for s1, s2 in sibling_pairs:
        sibling[entity_to_idx[s1], entity_to_idx[s2]] = 1.0
        sibling[entity_to_idx[s2], entity_to_idx[s1]] = 1.0

    # Create Married relation tensor (symmetric)
    married = np.zeros((n, n), dtype=np.float32)
    married_pairs = [
        ("Alice", "Bob"),
        ("Carol", "David"),
    ]
    for m1, m2 in married_pairs:
        married[entity_to_idx[m1], entity_to_idx[m2]] = 1.0
        married[entity_to_idx[m2], entity_to_idx[m1]] = 1.0

    # Create Loves relation tensor (soft/fuzzy values for demonstration)
    # Values in [0, 1] represent degree of love
    loves = np.zeros((n, n), dtype=np.float32)
    # Parents love their children
    for p, c in parent_facts:
        loves[entity_to_idx[p], entity_to_idx[c]] = 0.95
    # Married couples love each other
    for m1, m2 in married_pairs:
        loves[entity_to_idx[m1], entity_to_idx[m2]] = 0.9
        loves[entity_to_idx[m2], entity_to_idx[m1]] = 0.9
    # Siblings have affection
    for s1, s2 in sibling_pairs:
        loves[entity_to_idx[s1], entity_to_idx[s2]] = 0.7
        loves[entity_to_idx[s2], entity_to_idx[s1]] = 0.7

    return {
        "entities": entities,
        "entity_to_idx": entity_to_idx,
        "Parent": parent,
        "Sibling": sibling,
        "Married": married,
        "Loves": loves,
    }


def example_knowledge_graph_setup() -> dict[str, Any]:
    """Section 1: Set up and display the family knowledge graph."""
    print("=" * 80)
    print("SECTION 1: Family Knowledge Graph Setup")
    print("=" * 80)

    backend = create_backend()
    kg = create_family_knowledge_graph(backend)

    print(f"\nBackend: {type(backend).__name__}")
    print(f"\nEntities ({len(kg['entities'])} people):")
    for i, name in enumerate(kg["entities"]):
        print(f"  [{i}] {name}")

    print("\n" + "-" * 80)
    print("Parent Relation Matrix (Parent[i,j] = 1.0 means i is parent of j):")
    print("-" * 80)
    entities = kg["entities"]
    print(f"{'':>8}", end="")
    for name in entities:
        print(f"{name:>7}", end="")
    print()
    for i, row_name in enumerate(entities):
        print(f"{row_name:>8}", end="")
        for j in range(len(entities)):
            val = kg["Parent"][i, j]
            print(f"{val:>7.1f}", end="")
        print()

    print("\n" + "-" * 80)
    print("Sibling Relation Matrix (symmetric):")
    print("-" * 80)
    sibling_pairs = []
    for i, name1 in enumerate(entities):
        for j, name2 in enumerate(entities):
            if i < j and kg["Sibling"][i, j] > 0:
                sibling_pairs.append(f"{name1} <-> {name2}")
    print(f"  {', '.join(sibling_pairs)}")

    print("\n" + "-" * 80)
    print("Married Relation Matrix (symmetric):")
    print("-" * 80)
    married_pairs = []
    for i, name1 in enumerate(entities):
        for j, name2 in enumerate(entities):
            if i < j and kg["Married"][i, j] > 0:
                married_pairs.append(f"{name1} <-> {name2}")
    print(f"  {', '.join(married_pairs)}")

    return kg


# =============================================================================
# SECTION 2: Basic Logical Operations
# =============================================================================


def example_basic_logical_operations(kg: dict[str, Any]) -> None:
    """Section 2: Demonstrate basic logical operations on relations."""
    print("\n" + "=" * 80)
    print("SECTION 2: Basic Logical Operations")
    print("=" * 80)

    backend = create_backend()
    entities = kg["entities"]
    entity_to_idx = kg["entity_to_idx"]

    # Direct relation queries
    print("\n" + "-" * 80)
    print("Direct Relation Queries:")
    print("-" * 80)

    # Is Alice parent of Carol?
    alice_idx, carol_idx = entity_to_idx["Alice"], entity_to_idx["Carol"]
    is_parent = kg["Parent"][alice_idx, carol_idx]
    print(
        f"  Is Alice parent of Carol? {is_parent:.1f} (Yes)"
        if is_parent
        else f"  Is Alice parent of Carol? {is_parent:.1f} (No)"
    )

    # Is Alice parent of Henry?
    henry_idx = entity_to_idx["Henry"]
    is_parent = kg["Parent"][alice_idx, henry_idx]
    print(f"  Is Alice parent of Henry? {is_parent:.1f} (No - Henry is grandchild)")

    # Logical AND: Who is both a parent AND married?
    print("\n" + "-" * 80)
    print("Logical AND: Who is both a Parent AND Married?")
    print("-" * 80)

    # Sum across columns to check if person is a parent of anyone
    is_parent_of_someone = np.sum(kg["Parent"], axis=1) > 0
    is_married_to_someone = np.sum(kg["Married"], axis=1) > 0

    # Convert to float tensors
    is_parent_tensor = is_parent_of_someone.astype(np.float32)
    is_married_tensor = is_married_to_someone.astype(np.float32)

    # Apply logical AND
    parent_and_married = logical_and(
        is_parent_tensor, is_married_tensor, backend=backend
    )

    print("  Rule: IsParent(x) AND IsMarried(x)")
    for i, name in enumerate(entities):
        if parent_and_married[i] > 0:
            print(
                f"    {name}: Parent={is_parent_tensor[i]:.0f}, "
                f"Married={is_married_tensor[i]:.0f} -> AND={parent_and_married[i]:.0f}"
            )

    # Logical OR: Who is either a sibling OR a parent?
    print("\n" + "-" * 80)
    print("Logical OR: Who is either a Sibling OR a Parent?")
    print("-" * 80)

    is_sibling_of_someone = np.sum(kg["Sibling"], axis=1) > 0
    is_sibling_tensor = is_sibling_of_someone.astype(np.float32)

    sibling_or_parent = logical_or(is_sibling_tensor, is_parent_tensor, backend=backend)

    print("  Rule: IsSibling(x) OR IsParent(x)")
    for i, name in enumerate(entities):
        if sibling_or_parent[i] > 0:
            print(
                f"    {name}: Sibling={is_sibling_tensor[i]:.0f}, "
                f"Parent={is_parent_tensor[i]:.0f} -> OR={sibling_or_parent[i]:.0f}"
            )

    # Logical NOT: Who is NOT a parent?
    print("\n" + "-" * 80)
    print("Logical NOT: Who is NOT a Parent?")
    print("-" * 80)

    not_parent = logical_not(is_parent_tensor, backend=backend)

    print("  Rule: NOT IsParent(x)")
    for i, name in enumerate(entities):
        if not_parent[i] > 0:
            print(f"    {name}: NOT Parent = {not_parent[i]:.0f}")


# =============================================================================
# SECTION 3: Relation Inference with Rules
# =============================================================================


def example_relation_inference(kg: dict[str, Any]) -> None:
    """Section 3: Infer new relations using logical rules as tensor operations."""
    print("\n" + "=" * 80)
    print("SECTION 3: Relation Inference with Logical Rules")
    print("=" * 80)

    backend = create_backend()
    entities = kg["entities"]
    parent = kg["Parent"]

    # Grandparent Rule: Grandparent(x,z) = exists y: Parent(x,y) AND Parent(y,z)
    # This is equivalent to matrix multiplication: Grandparent = Parent @ Parent
    print("\n" + "-" * 80)
    print("Grandparent Rule: Grandparent(x,z) <- exists y: Parent(x,y) AND Parent(y,z)")
    print("-" * 80)

    # Matrix multiplication gives the count of paths
    grandparent_paths = np.matmul(parent, parent)
    # Convert to boolean (1.0 if any path exists)
    grandparent = (grandparent_paths > 0).astype(np.float32)

    print("\n  Inferred Grandparent relations:")
    for i, gp_name in enumerate(entities):
        for j, gc_name in enumerate(entities):
            if grandparent[i, j] > 0:
                # Find the intermediate parent
                for k, p_name in enumerate(entities):
                    if parent[i, k] > 0 and parent[k, j] > 0:
                        print(f"    {gp_name} -> {gc_name} (via {p_name})")

    # Aunt/Uncle Rule: AuntUncle(x,z) = exists y: Sibling(x,y) AND Parent(y,z)
    print("\n" + "-" * 80)
    print("Aunt/Uncle Rule: AuntUncle(x,z) <- exists y: Sibling(x,y) AND Parent(y,z)")
    print("-" * 80)

    sibling = kg["Sibling"]
    aunt_uncle = (np.matmul(sibling, parent) > 0).astype(np.float32)

    print("\n  Inferred Aunt/Uncle relations:")
    for i, au_name in enumerate(entities):
        for j, niece_name in enumerate(entities):
            if aunt_uncle[i, j] > 0:
                # Find the sibling who is the parent
                for k, sib_name in enumerate(entities):
                    if sibling[i, k] > 0 and parent[k, j] > 0:
                        print(
                            f"    {au_name} is aunt/uncle of {niece_name} "
                            f"(sibling of {sib_name})"
                        )

    # Implication Rule: If someone is a Parent, they should Love their child
    # Parent(x,y) -> Loves(x,y)
    print("\n" + "-" * 80)
    print("Implication: Parent(x,y) -> Loves(x,y)")
    print("-" * 80)

    loves = kg["Loves"]
    implication_satisfied = logical_implies(parent, loves, backend=backend)

    print("\n  Checking if all parents love their children:")
    print("  (Implication P->Q = max(1-P, Q), satisfied when >= threshold)")
    threshold = 0.5  # Implication satisfied if >= 0.5
    for i, p_name in enumerate(entities):
        for j, c_name in enumerate(entities):
            if parent[i, j] > 0:
                imp_val = float(implication_satisfied[i, j])
                status = "satisfied" if imp_val >= threshold else "violated"
                print(
                    f"    {p_name} -> {c_name}: "
                    f"Parent={parent[i, j]:.1f}, Loves={loves[i, j]:.2f}, "
                    f"Implication={imp_val:.2f} ({status})"
                )


# =============================================================================
# SECTION 4: Quantified Queries with quantify() API
# =============================================================================


def example_quantified_queries(kg: dict[str, Any]) -> None:
    """Section 4: Demonstrate quantified queries using core operations.

    This section shows how to express logical queries using TensorLogic's
    core quantifier operations. These patterns can be wrapped with the
    quantify() API for more readable code.
    """
    print("\n" + "=" * 80)
    print("SECTION 4: Quantified Queries")
    print("=" * 80)

    backend = create_backend()
    entities = kg["entities"]
    parent = kg["Parent"]

    # Query 1: EXISTS y: Parent(x, y) - "Does x have any children?"
    # For each person x, check if there exists y such that Parent(x, y)
    print("\n" + "-" * 80)
    print("Query 1: exists y: Parent(x, y) - 'Who has children?'")
    print("-" * 80)

    # Parent[x, y] = 1 if x is parent of y
    # exists y: Parent(x, y) = check each row for any 1s
    has_children = exists(parent, axis=1, backend=backend)  # aggregate over y (columns)

    print("\n  Results (1.0 = has children):")
    for i, name in enumerate(entities):
        val = float(has_children[i])
        children = [entities[j] for j in range(len(entities)) if parent[i, j] > 0]
        if val > 0:
            print(f"    {name}: {val:.1f} (children: {', '.join(children)})")
        else:
            print(f"    {name}: {val:.1f}")

    # Query 2: FORALL y: Parent(x, y) -> Loves(x, y)
    # "Does x love ALL their children?"
    print("\n" + "-" * 80)
    print("Query 2: forall y: Parent(x, y) -> Loves(x, y)")
    print("         'Does everyone love all their children?'")
    print("-" * 80)

    loves = kg["Loves"]

    # Compute implication element-wise: Parent(x,y) -> Loves(x,y)
    implication = logical_implies(parent, loves, backend=backend)

    # For each x, check if ALL y satisfy the implication
    # Using forall over axis 1 (y dimension)
    forall_result = forall(implication, axis=1, backend=backend)

    print("\n  Results (1.0 = loves all children):")
    for i, name in enumerate(entities):
        val = float(forall_result[i])
        print(f"    {name}: {val:.1f}")

    # Query 3: EXISTS y: Sibling(x, y) AND Married(y, z)
    # "Does x have a married sibling?"
    print("\n" + "-" * 80)
    print("Query 3: exists y: Sibling(x, y) and IsMarried(y)")
    print("         'Who has a married sibling?'")
    print("-" * 80)

    sibling = kg["Sibling"]
    married = kg["Married"]

    # For each x, check if there exists y where Sibling(x,y) AND Married(y, someone)
    is_married = (np.sum(married, axis=1) > 0).astype(np.float32)

    # Sibling(x, y) AND IsMarried(y) - broadcast is_married across rows
    sibling_and_married = logical_and(
        sibling, is_married.reshape(1, -1), backend=backend
    )

    # Exists over y (axis 1)
    has_married_sibling = exists(sibling_and_married, axis=1, backend=backend)

    print("\n  Results (1.0 = has married sibling):")
    for i, name in enumerate(entities):
        val = float(has_married_sibling[i])
        if val > 0:
            # Find the married sibling
            for j, sib_name in enumerate(entities):
                if sibling[i, j] > 0 and is_married[j] > 0:
                    print(f"    {name}: {val:.1f} (married sibling: {sib_name})")
                    break
        else:
            print(f"    {name}: {val:.1f}")

    # Query 4: Using quantify() API with simple patterns
    print("\n" + "-" * 80)
    print("Query 4: Using quantify() API")
    print("-" * 80)

    # Simple single-predicate pattern
    result = quantify(
        "exists x: P(x)",
        predicates={"P": has_children},  # Reuse computed result
        backend=backend,
    )
    print("\n  quantify('exists x: P(x)', predicates={'P': has_children})")
    print(f"  'Does anyone have children?' -> {float(result):.1f}")


# =============================================================================
# SECTION 5: Temperature-Controlled Reasoning
# =============================================================================


def example_temperature_reasoning(kg: dict[str, Any]) -> None:
    """Section 5: Compare deductive (T=0) vs analogical (T>0) reasoning."""
    print("\n" + "=" * 80)
    print("SECTION 5: Temperature-Controlled Reasoning")
    print("=" * 80)

    backend = create_backend()
    entities = kg["entities"]

    print("""
    Temperature controls reasoning mode:
    - T=0.0: Deductive (hard boolean, exact logic, no hallucinations)
    - T>0.0: Analogical (soft probabilities, generalization)
    - Higher T = more analogical, smoother interpolation
    """)

    # Create uncertain/fuzzy relations for demonstration
    # "Possible Parent" relation with varying confidence
    possible_parent = np.zeros((len(entities), len(entities)), dtype=np.float32)
    possible_parent[0, 2] = 0.9  # Alice possibly parent of Carol (high confidence)
    possible_parent[0, 3] = 0.4  # Alice possibly parent of David (low - actually not)
    possible_parent[1, 2] = 0.85  # Bob possibly parent of Carol
    possible_parent[2, 5] = 0.75  # Carol possibly parent of Frank

    print("-" * 80)
    print("Uncertain Knowledge: PossibleParent relation with confidence values")
    print("-" * 80)
    print("\n  PossibleParent facts:")
    for i, p_name in enumerate(entities):
        for j, c_name in enumerate(entities):
            if possible_parent[i, j] > 0:
                print(f"    {p_name} -> {c_name}: {possible_parent[i, j]:.2f}")

    # Query: Does Alice have any children? (using exists)
    print("\n" + "-" * 80)
    print("Query: exists y: PossibleParent(Alice, y)")
    print("       'Does Alice possibly have children?' at different temperatures")
    print("-" * 80)

    alice_row = possible_parent[0:1, :]  # Shape (1, 8)

    temperatures = [0.0, 0.5, 1.0, 2.0, 5.0]

    print("\n  Temperature | Result | Interpretation")
    print("  " + "-" * 50)

    for temp in temperatures:
        result = reason(
            "exists y: P(y)",
            predicates={"P": alice_row.flatten()},
            bindings={"y": np.arange(len(entities))},
            temperature=temp,
            backend=backend,
        )
        result_val = (
            float(result) if np.ndim(result) == 0 else float(result.flatten()[0])
        )

        if temp == 0.0:
            interp = "Hard boolean: step(sum) = 1 if any > 0"
        elif temp < 1.0:
            interp = f"63% soft blend (alpha={1 - np.exp(-temp):.2f})"
        else:
            interp = f"More analogical (alpha={1 - np.exp(-temp):.2f})"

        print(f"  T={temp:>4.1f}      | {result_val:>5.3f}  | {interp}")

    # Compare deductive vs analogical on implication
    print("\n" + "-" * 80)
    print("Implication with Uncertainty: PossibleParent(x,y) -> Loves(x,y)")
    print("-" * 80)

    print("\n  Comparing how uncertain implications are evaluated:")
    print("\n  Fact: Alice PossibleParent of David = 0.4, Alice Loves David = 0.0")
    print("  Implication: 0.4 -> 0.0 = max(1-0.4, 0.0) = 0.6 (soft)")

    for temp in [0.0, 1.0, 2.0]:
        result = reason(
            "P(x) -> Q(x)",
            predicates={
                "P": np.array([0.4]),  # PossibleParent confidence
                "Q": np.array([0.0]),  # Loves value
            },
            bindings={"x": np.array([0])},
            temperature=temp,
            backend=backend,
        )
        result_val = (
            float(result) if np.ndim(result) == 0 else float(result.flatten()[0])
        )
        print(f"\n  T={temp:.1f}: Implication result = {result_val:.3f}")
        if temp == 0.0:
            print("         Hard: step(0.6) = 1.0 (implication technically holds)")
        else:
            alpha = 1 - np.exp(-temp)
            print(f"         Soft blend: {1 - alpha:.2f}*hard + {alpha:.2f}*soft")


# =============================================================================
# SECTION 6: Compilation Strategy Comparison
# =============================================================================


def example_strategy_comparison(kg: dict[str, Any]) -> None:
    """Section 6: Compare different compilation strategies."""
    print("\n" + "=" * 80)
    print("SECTION 6: Compilation Strategy Comparison")
    print("=" * 80)

    backend = create_backend()

    print("""
    TensorLogic supports multiple compilation strategies:

    1. soft_differentiable: Product semantics (a*b), gradient-friendly
       Use for: Training neural predicates

    2. hard_boolean: Step function discretization
       Use for: Production inference with exact results

    3. godel: Min/max fuzzy logic (conservative)
       Use for: Fuzzy reasoning where idempotence matters

    4. product: Probabilistic (a*b for AND, a+b-ab for OR)
       Use for: Independent probability modeling

    5. lukasiewicz: Bounded arithmetic (max(0, a+b-1) for AND)
       Use for: Strict boundary conditions
    """)

    # Test data: Uncertain predicates
    p = np.array([0.8, 0.6, 0.4])
    q = np.array([0.9, 0.5, 0.3])

    print("-" * 80)
    print("Test: AND operation on uncertain predicates")
    print("-" * 80)
    print(f"\n  P = {p}")
    print(f"  Q = {q}")

    print("\n  Strategy Comparison for P AND Q:")
    print("  " + "-" * 60)
    print(f"  {'Strategy':<20} | {'Result':<25} | Semantics")
    print("  " + "-" * 60)

    strategies_info = [
        ("soft_differentiable", "a * b"),
        ("hard_boolean", "step(a * b)"),
        ("godel", "min(a, b)"),
        ("product", "a * b"),
        ("lukasiewicz", "max(0, a+b-1)"),
    ]

    for strategy_name, semantics in strategies_info:
        strategy = create_strategy(strategy_name, backend=backend)
        result = strategy.compile_and(p, q)
        result_str = np.array2string(np.asarray(result), precision=2, separator=", ")
        print(f"  {strategy_name:<20} | {result_str:<25} | {semantics}")

    # Compare OR operation
    print("\n" + "-" * 80)
    print("Test: OR operation on uncertain predicates")
    print("-" * 80)

    print("\n  Strategy Comparison for P OR Q:")
    print("  " + "-" * 60)
    print(f"  {'Strategy':<20} | {'Result':<25} | Semantics")
    print("  " + "-" * 60)

    strategies_or_info = [
        ("soft_differentiable", "a + b - a*b"),
        ("hard_boolean", "step(max(a, b))"),
        ("godel", "max(a, b)"),
        ("product", "a + b - a*b"),
        ("lukasiewicz", "min(1, a+b)"),
    ]

    for strategy_name, semantics in strategies_or_info:
        strategy = create_strategy(strategy_name, backend=backend)
        result = strategy.compile_or(p, q)
        result_str = np.array2string(np.asarray(result), precision=2, separator=", ")
        print(f"  {strategy_name:<20} | {result_str:<25} | {semantics}")

    # Training vs Inference Pattern
    print("\n" + "-" * 80)
    print("Best Practice: Train Soft, Infer Hard")
    print("-" * 80)

    print("""
    Recommended workflow:

    TRAINING PHASE:
        strategy = create_strategy("soft_differentiable")
        # Gradients flow through product operations
        # Neural predicates can learn from logical constraints

    INFERENCE PHASE:
        strategy = create_strategy("hard_boolean")
        # Exact 0/1 outputs, no ambiguity
        # No hallucinations from soft interpolation
    """)


# =============================================================================
# SECTION 7: Advanced - Uncertain Knowledge
# =============================================================================


def example_uncertain_knowledge(kg: dict[str, Any]) -> None:
    """Section 7: Handle uncertain/fuzzy knowledge in reasoning."""
    print("\n" + "=" * 80)
    print("SECTION 7: Reasoning with Uncertain Knowledge")
    print("=" * 80)

    backend = create_backend()
    entities = kg["entities"]

    print("""
    Real-world knowledge is often uncertain. TensorLogic handles this through:
    1. Fuzzy relation values (confidence in [0, 1])
    2. Different compilation strategies for uncertainty propagation
    3. Temperature control for soft vs hard inference
    """)

    # Create uncertain DNA/paternity evidence
    print("-" * 80)
    print("Scenario: Uncertain Paternity from DNA Evidence")
    print("-" * 80)

    # DNA match probability (uncertain parent relation)
    dna_match = np.zeros((len(entities), len(entities)), dtype=np.float32)
    dna_match[0, 2] = 0.95  # Alice -> Carol: 95% match
    dna_match[0, 4] = 0.92  # Alice -> Eve: 92% match
    dna_match[1, 2] = 0.88  # Bob -> Carol: 88% match
    dna_match[1, 4] = 0.90  # Bob -> Eve: 90% match
    dna_match[2, 5] = 0.85  # Carol -> Frank: 85% match

    print("\n  DNA Match Probabilities:")
    for i, p_name in enumerate(entities):
        for j, c_name in enumerate(entities):
            if dna_match[i, j] > 0:
                print(f"    {p_name} -> {c_name}: {dna_match[i, j]:.0%}")

    # Query: What's the probability of grandparent relationship via uncertain parent?
    print("\n" + "-" * 80)
    print("Query: Uncertain Grandparent = exists y: DNAMatch(x,y) AND DNAMatch(y,z)")
    print("-" * 80)

    # Compare strategies on uncertain grandparent inference
    strategies = ["soft_differentiable", "godel", "product"]

    print("\n  Comparing uncertain grandparent inference strategies:")
    print("\n  Alice -> Frank (via Carol):")
    print(
        f"    Path: Alice->Carol ({dna_match[0, 2]:.2f}) AND Carol->Frank ({dna_match[2, 5]:.2f})"
    )

    for strategy_name in strategies:
        strategy = create_strategy(strategy_name, backend=backend)

        # Compute path confidence
        path_confidence = strategy.compile_and(
            np.array([dna_match[0, 2]]),  # Alice -> Carol
            np.array([dna_match[2, 5]]),  # Carol -> Frank
        )

        print(f"\n    {strategy_name}:")
        print(f"      Grandparent confidence: {float(path_confidence[0]):.3f}")

        if strategy_name == "soft_differentiable":
            print(f"      Formula: 0.95 * 0.85 = {0.95 * 0.85:.3f}")
        elif strategy_name == "godel":
            print(f"      Formula: min(0.95, 0.85) = {min(0.95, 0.85):.3f}")
        elif strategy_name == "product":
            print(f"      Formula: 0.95 * 0.85 = {0.95 * 0.85:.3f}")

    # Combining multiple evidence sources
    print("\n" + "-" * 80)
    print("Combining Evidence: DNA OR WitnessTestimony")
    print("-" * 80)

    witness_testimony = np.array([0.7])  # Witness 70% confident
    dna_evidence = np.array([0.85])  # DNA 85% confident

    print("\n  Evidence sources:")
    print(f"    DNA evidence: {float(dna_evidence[0]):.0%}")
    print(f"    Witness testimony: {float(witness_testimony[0]):.0%}")

    print("\n  Combined confidence (Evidence1 OR Evidence2):")

    for strategy_name in strategies:
        strategy = create_strategy(strategy_name, backend=backend)
        combined = strategy.compile_or(dna_evidence, witness_testimony)

        print(f"    {strategy_name}: {float(combined[0]):.3f}")


# =============================================================================
# SECTION 8: Summary and Best Practices
# =============================================================================


def example_summary() -> None:
    """Section 8: Summary of TensorLogic features and best practices."""
    print("\n" + "=" * 80)
    print("SECTION 8: Summary and Best Practices")
    print("=" * 80)

    print("""
    KEY TENSORLOGIC FEATURES:

    1. LOGICAL OPERATIONS AS TENSORS
       - AND: Hadamard product (element-wise multiply)
       - OR: Element-wise maximum
       - NOT: Complement (1 - x)
       - IMPLIES: max(1-a, b)

    2. QUANTIFIERS
       - exists: "At least one" via summation + step
       - forall: "All" via product + threshold
       - Soft variants (soft_exists, soft_forall) for differentiable reasoning

    3. HIGH-LEVEL API
       - quantify(): Pattern-based quantified queries
       - reason(): Temperature-controlled inference

    4. COMPILATION STRATEGIES
       - soft_differentiable: Training neural predicates
       - hard_boolean: Production inference
       - godel/product/lukasiewicz: Different fuzzy semantics

    5. TEMPERATURE CONTROL
       - T=0: Deductive (exact boolean, no hallucinations)
       - T>0: Analogical (soft probabilities, generalization)

    BEST PRACTICES:

    - Start with hard_boolean for exact logical queries
    - Use soft_differentiable when training neural components
    - Apply temperature control for uncertainty handling
    - Choose strategy based on uncertainty semantics needed
    - Use quantify() for readable, maintainable logical queries

    PERFORMANCE TIPS:

    - Prefer MLX backend on Apple Silicon (GPU acceleration)
    - Use batch operations over loops
    - Call backend.eval() after MLX computations
    - Leverage einsum for complex tensor operations
    """)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    """Run all knowledge graph reasoning examples."""
    print("\n" + "=" * 80)
    print("TENSORLOGIC: KNOWLEDGE GRAPH REASONING EXAMPLE")
    print("=" * 80)
    print("""
    This example demonstrates neural-symbolic reasoning over a family
    knowledge graph using TensorLogic.

    Family Tree:
        Generation 1: Alice & Bob (married)
        Generation 2: Carol (+ David), Eve
        Generation 3: Frank, Grace, Henry
    """)

    # Run all sections
    kg = example_knowledge_graph_setup()
    example_basic_logical_operations(kg)
    example_relation_inference(kg)
    example_quantified_queries(kg)
    example_temperature_reasoning(kg)
    example_strategy_comparison(kg)
    example_uncertain_knowledge(kg)
    example_summary()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
