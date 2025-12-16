"""TensorLogic Family Tree: Multi-Hop Reasoning with Einsum

This example shows how to infer relationships that aren't directly stated.
For example: "Alice is parent of Bob, Bob is parent of Carol" implies
"Alice is grandparent of Carol" - but we never stated that directly!

What you'll learn:
    1. How to represent relationships as matrices
    2. How einsum computes multi-hop inferences (grandparent = parent of parent)
    3. How to query "who has children?" using EXISTS

The Key Insight:
    A relationship like "Parent(x, y)" becomes a matrix where:
    - Row = the first person (x)
    - Column = the second person (y)
    - Value 1.0 = the relationship holds

Run this example:
    uv run python examples/01_family_tree_minimal.py
"""

from __future__ import annotations

import numpy as np

from tensorlogic import create_backend
from tensorlogic.core.quantifiers import exists

# =============================================================================
# STEP 1: Define our family
# =============================================================================
# A simple 3-generation family: Alice -> Bob -> Carol

entities = ["Alice", "Bob", "Carol"]
print("Family members:", entities)
print()

# =============================================================================
# STEP 2: Create the Parent relationship as a matrix
# =============================================================================
# Parent[i, j] = 1.0 means "person i is parent of person j"
#
#              Alice  Bob  Carol
#   Alice   [   0     1     0   ]   <- Alice is parent of Bob
#   Bob     [   0     0     1   ]   <- Bob is parent of Carol
#   Carol   [   0     0     0   ]   <- Carol has no children

parent = np.array([
    [0., 1., 0.],  # Alice is parent of Bob
    [0., 0., 1.],  # Bob is parent of Carol
    [0., 0., 0.],  # Carol has no children
], dtype=np.float32)

print("Parent relationship matrix:")
print("             Alice  Bob  Carol")
for i, name in enumerate(entities):
    row = "  ".join(f"{v:.0f}" for v in parent[i])
    print(f"  {name:>6}   [  {row}  ]")
print()

# =============================================================================
# STEP 3: Infer the Grandparent relationship
# =============================================================================
# Logical rule: Grandparent(x, z) = EXISTS y: Parent(x, y) AND Parent(y, z)
#
# In tensor form, this is matrix multiplication (or einsum):
#   - 'xy' means Parent[x, y] (first matrix)
#   - 'yz' means Parent[y, z] (second matrix)
#   - '->xz' means the result has indices x and z
#   - The shared 'y' is summed over (this is the EXISTS)

backend = create_backend()

# Compute grandparent via einsum
grandparent_raw = backend.einsum('xy,yz->xz', parent, parent)

# Apply step function to get boolean (1.0 or 0.0)
grandparent = backend.step(grandparent_raw)

print("Grandparent relationship (inferred via einsum):")
print("             Alice  Bob  Carol")
for i, name in enumerate(entities):
    row = "  ".join(f"{v:.0f}" for v in np.asarray(grandparent)[i])
    print(f"  {name:>6}   [  {row}  ]")
print()

# Verify: Alice should be grandparent of Carol
alice_idx, carol_idx = 0, 2
is_grandparent = grandparent[alice_idx, carol_idx]
print(f"Is Alice grandparent of Carol? {bool(is_grandparent)}")
print("  (Inferred from: Alice -> Bob -> Carol)")
print()

# =============================================================================
# STEP 4: Query "Who has children?" using EXISTS
# =============================================================================
# The EXISTS quantifier checks: "is there any y where Parent(x, y) is true?"
# For each person x, we check if they are parent of anyone.

has_children = exists(parent, axis=1, backend=backend)

print("Who has children? (EXISTS y: Parent(x, y))")
for i, name in enumerate(entities):
    result = "Yes" if has_children[i] > 0 else "No"
    print(f"  {name}: {result}")
print()

# =============================================================================
# STEP 5: Query "Who has grandchildren?" using composed reasoning
# =============================================================================
# We already computed grandparent, now use EXISTS on that

has_grandchildren = exists(grandparent, axis=1, backend=backend)

print("Who has grandchildren?")
for i, name in enumerate(entities):
    result = "Yes" if has_grandchildren[i] > 0 else "No"
    print(f"  {name}: {result}")

# =============================================================================
# WHAT'S NEXT?
# =============================================================================
# Now that you understand multi-hop reasoning, try:
#   - examples/02_temperature_demo.py (deductive vs analogical reasoning)
#   - examples/knowledge_graph_reasoning.py (comprehensive 8-person family)
