"""TensorLogic Recommendation System: Item Similarity with Temperature

This example shows how to build a product recommendation system:
- "Given a user likes Product A, what else might they like?"
- How temperature controls recommendation diversity

What you'll learn:
    1. How to encode items and categories as matrices
    2. How to compute item similarity via shared categories
    3. How temperature affects recommendation diversity

The Key Insight:
    Items sharing categories are similar. Temperature controls how
    strictly we require similarity before recommending.

Run this example:
    uv run python examples/04_recommendation_system.py
"""

from __future__ import annotations

import numpy as np

from tensorlogic import create_backend, logical_and
from tensorlogic.core.temperature import temperature_scaled_operation

# =============================================================================
# SETUP: Product Catalog
# =============================================================================

backend = create_backend()
print(f"Backend: {type(backend).__name__}")
print()

# Products in our catalog
products = [
    "Python Book",      # 0
    "JavaScript Book",  # 1
    "Data Science Kit", # 2
    "Web Dev Course",   # 3
    "ML Course",        # 4
    "Excel Guide",      # 5
]

# Categories
categories = ["Programming", "Data Science", "Web Dev", "Beginner", "Advanced"]

# product_in_category[product, category] = strength of association (0.0 to 1.0)
#                    Programming  DataSci  WebDev  Beginner  Advanced
product_category = np.array([
    [1.0, 0.3, 0.0, 0.8, 0.2],  # Python Book: strong programming, some data sci
    [1.0, 0.0, 0.9, 0.7, 0.3],  # JS Book: programming + web dev
    [0.5, 1.0, 0.0, 0.3, 0.7],  # Data Science Kit: strong data sci
    [0.6, 0.2, 1.0, 0.9, 0.1],  # Web Dev Course: web focused, beginner
    [0.7, 1.0, 0.1, 0.2, 0.9],  # ML Course: data sci + advanced
    [0.2, 0.5, 0.0, 1.0, 0.0],  # Excel Guide: beginner data analysis
], dtype=np.float32)

print("=== Product Catalog ===")
print()
for i, prod in enumerate(products):
    cats = [categories[j] for j in range(len(categories)) if product_category[i, j] > 0.5]
    print(f"  {prod}: {', '.join(cats)}")
print()

# =============================================================================
# PART 1: Compute Item Similarity
# =============================================================================

print("=" * 60)
print("PART 1: Item Similarity Matrix")
print("=" * 60)
print()

# Items are similar if they share categories
# similarity[i, j] = how many categories i and j share (normalized)

# Method: product_category @ product_category.T gives co-occurrence
similarity_raw = backend.einsum('ic,jc->ij', product_category, product_category)

# Normalize by the geometric mean of category counts
category_counts = np.sum(product_category, axis=1, keepdims=True)
normalization = np.sqrt(category_counts @ category_counts.T)
similarity = np.asarray(similarity_raw) / (normalization + 1e-6)

# Zero out self-similarity (don't recommend item to itself)
np.fill_diagonal(similarity, 0)

print("Similarity matrix (top matches):")
for i, prod in enumerate(products):
    top_idx = np.argmax(similarity[i])
    print(f"  {prod} → most similar: {products[top_idx]} ({similarity[i, top_idx]:.2f})")
print()

# =============================================================================
# PART 2: Recommend With Temperature
# =============================================================================

print("=" * 60)
print("PART 2: Recommendations with Temperature Control")
print("=" * 60)
print()

# User has purchased Python Book (index 0)
purchased_idx = 0
print(f"User purchased: {products[purchased_idx]}")
print()

# Get similarity scores for this product
scores = similarity[purchased_idx].copy()

# Apply temperature to control recommendation diversity
for temp in [0.0, 0.5, 1.0]:
    print(f"Temperature T={temp}:")

    if temp == 0.0:
        # T=0: Only recommend items with similarity > threshold
        threshold = 0.5
        recommendations = [(products[i], scores[i]) for i in range(len(products))
                         if scores[i] > threshold and i != purchased_idx]
    else:
        # T>0: Include softer matches with graded scores
        # Apply softmax-like temperature scaling
        alpha = 1.0 - np.exp(-temp)
        scaled_scores = (1 - alpha) * (scores > 0.5).astype(float) + alpha * scores
        recommendations = [(products[i], scaled_scores[i]) for i in range(len(products))
                         if scaled_scores[i] > 0.1 and i != purchased_idx]
        recommendations = sorted(recommendations, key=lambda x: -x[1])

    if recommendations:
        for name, score in recommendations[:3]:
            print(f"    - {name}: {score:.2f}")
    else:
        print("    (no recommendations meet threshold)")
    print()

# =============================================================================
# PART 3: Recommendation Rules with Logical Ops
# =============================================================================

print("=" * 60)
print("PART 3: Rule-Based Recommendations")
print("=" * 60)
print()

# Rule: Recommend items that are in BOTH "Programming" AND "Beginner" categories
# This targets newcomers to programming

programming_idx = categories.index("Programming")
beginner_idx = categories.index("Beginner")

is_programming = product_category[:, programming_idx]
is_beginner = product_category[:, beginner_idx]

print("Rule: Programming AND Beginner (for coding newcomers)")
print()

# At T=0 (strict): Only items with strong signals in BOTH
deductive_and = temperature_scaled_operation(
    logical_and, temperature=0.0, backend=backend
)
strict_matches = np.asarray(deductive_and(is_programming, is_beginner, backend=backend))

print("T=0 (Strict):")
for i, prod in enumerate(products):
    if strict_matches[i] > 0.5:
        print(f"  - {prod}")

# At T=1 (soft): Items with any signal in both
soft_and = temperature_scaled_operation(
    logical_and, temperature=1.0, backend=backend
)
soft_matches = np.asarray(soft_and(is_programming, is_beginner, backend=backend))

print()
print("T=1 (Soft) - with scores:")
ranked = sorted(enumerate(soft_matches), key=lambda x: -x[1])
for i, score in ranked:
    if score > 0.1:
        print(f"  - {products[i]}: {score:.2f}")

print()

# =============================================================================
# PART 4: Collaborative Filtering Pattern
# =============================================================================

print("=" * 60)
print("PART 4: Collaborative Filtering (Users Who Bought X Also Bought Y)")
print("=" * 60)
print()

# User purchase history (which products each user bought)
# user_purchases[user, product] = 1.0 if user bought that product
users = ["User1", "User2", "User3", "User4"]
user_purchases = np.array([
    [1., 0., 1., 0., 1., 0.],  # User1: Python Book, Data Science Kit, ML Course
    [1., 1., 0., 1., 0., 0.],  # User2: Python Book, JS Book, Web Dev Course
    [0., 1., 0., 1., 0., 0.],  # User3: JS Book, Web Dev Course
    [0., 0., 1., 0., 1., 1.],  # User4: Data Science Kit, ML Course, Excel Guide
], dtype=np.float32)

# Co-purchase matrix: how often items are bought together
# co_purchase[i, j] = number of users who bought both i and j
co_purchase = backend.einsum('ui,uj->ij', user_purchases, user_purchases)
co_purchase = np.asarray(co_purchase)

# Normalize by product popularity
popularity = np.sum(user_purchases, axis=0, keepdims=True)
co_purchase_norm = co_purchase / (np.sqrt(popularity.T @ popularity) + 1e-6)
np.fill_diagonal(co_purchase_norm, 0)

print("Co-purchase patterns:")
for i, prod in enumerate(products):
    top_idx = np.argmax(co_purchase_norm[i])
    if co_purchase_norm[i, top_idx] > 0:
        print(f"  Bought {prod} → also bought: {products[top_idx]}")

print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 60)
print("SUMMARY: Recommendation Patterns")
print("=" * 60)
print("""
┌─────────────────────────────────────────────────────────────┐
│  PATTERN                        TENSORLOGIC                 │
├─────────────────────────────────────────────────────────────┤
│  Content-based similarity       einsum('ic,jc->ij', items)  │
│  Rule-based filtering           logical_and(cat_A, cat_B)   │
│  Collaborative filtering        einsum('ui,uj->ij', users)  │
│  Diversity control              temperature parameter       │
├─────────────────────────────────────────────────────────────┤
│  T=0: Conservative (only strong matches)                    │
│  T>0: Diverse (includes softer matches with scores)         │
└─────────────────────────────────────────────────────────────┘
""")

# =============================================================================
# WHAT'S NEXT?
# =============================================================================
# Now that you understand recommendations:
#   - examples/knowledge_graph_reasoning.py (full reasoning example)
#   - examples/gradient_training.py (learn embeddings from data)
#   - docs/concepts/tensor-logic-mapping.md (mathematical foundations)
