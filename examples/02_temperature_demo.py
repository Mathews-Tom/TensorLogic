"""TensorLogic Temperature Demo: Deductive vs Analogical Reasoning

Temperature is TensorLogic's killer feature - it's like a dial that controls
how strictly your AI follows logical rules.

    T=0.0  →  DEDUCTIVE (strict logic, no guessing, no hallucinations)
    T>0.0  →  ANALOGICAL (soft reasoning, can generalize from incomplete data)

What you'll learn:
    1. How T=0 gives exact boolean results (like a database query)
    2. How T>0 allows "fuzzy" reasoning with uncertain knowledge
    3. When to use each mode in real applications

The Key Insight:
    At T=0: Values snap to 0.0 or 1.0 (TRUE or FALSE, nothing in between)
    At T>0: Values stay continuous, allowing gradual confidence levels

Run this example:
    uv run python examples/02_temperature_demo.py
"""

from __future__ import annotations

import math

import numpy as np

from tensorlogic import create_backend, logical_and
from tensorlogic.core.temperature import temperature_scaled_operation

# =============================================================================
# SETUP
# =============================================================================

backend = create_backend()
print(f"Backend: {type(backend).__name__}")
print()

# =============================================================================
# PART 1: Temperature Effect on Values
# =============================================================================
# Let's see how temperature transforms the same input values

print("=" * 60)
print("PART 1: How Temperature Transforms Values")
print("=" * 60)
print()

# These are "confidence" values - not quite 0 or 1
# Think: "I'm 30% sure Alice is tall, 70% sure Bob is tall"
confidences = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float32)

print(f"Input confidences: {confidences}")
print("  (30%, 50%, 70%, 90% confident)")
print()

# Create operations at different temperatures
for temp in [0.0, 0.5, 1.0, 2.0]:
    # At T=0, anything > 0.5 becomes 1, else becomes 0
    # At T>0, values blend between hard (0/1) and soft (original)

    # Calculate interpolation weight: α = 1 - exp(-T)
    alpha = 1.0 - math.exp(-temp)

    # Simulate what temperature does:
    # result = (1-α)·step(x) + α·x
    hard = np.where(confidences > 0.5, 1.0, 0.0)  # step function
    soft = confidences  # original values
    result = (1 - alpha) * hard + alpha * soft

    print(f"T={temp:.1f} (α={alpha:.2f}): {np.array2string(result, precision=2)}")

    if temp == 0.0:
        print("     ↑ Binary: below 0.5 → 0, above 0.5 → 1")
    elif temp == 1.0:
        print("     ↑ Blend: 37% hard + 63% soft")

print()

# =============================================================================
# PART 2: Temperature with Logical Operations
# =============================================================================

print("=" * 60)
print("PART 2: Temperature with Logical AND")
print("=" * 60)
print()

# Two uncertain predicates
# "Is person X tall?" and "Is person X strong?"
is_tall = np.array([0.8, 0.3, 0.6, 0.9], dtype=np.float32)
is_strong = np.array([0.7, 0.4, 0.8, 0.5], dtype=np.float32)

print(f"Is Tall:   {is_tall}  (80%, 30%, 60%, 90% confident)")
print(f"Is Strong: {is_strong}  (70%, 40%, 80%, 50% confident)")
print()
print("Query: 'Tall AND Strong' at different temperatures:")
print()

for temp in [0.0, 1.0]:
    # Create temperature-scaled AND operation
    scaled_and = temperature_scaled_operation(
        logical_and, temperature=temp, backend=backend
    )

    result = scaled_and(is_tall, is_strong, backend=backend)
    result_arr = np.asarray(result)

    print(f"T={temp:.1f}: {np.array2string(result_arr, precision=2)}")

    if temp == 0.0:
        print("     ↑ DEDUCTIVE: Only person 0 is definitely both (>0.5)")
    else:
        print("     ↑ ANALOGICAL: Shows graded confidence levels")

print()

# =============================================================================
# PART 3: Real-World Scenario - Missing Data
# =============================================================================

print("=" * 60)
print("PART 3: Handling Incomplete Knowledge")
print("=" * 60)
print()

print("Scenario: Product recommendation based on incomplete user data")
print()

# User preferences (some are uncertain/incomplete)
# 1.0 = definitely likes, 0.0 = definitely dislikes, 0.5 = unknown
likes_action = np.array([1.0, 0.6, 0.0, 0.5], dtype=np.float32)
likes_comedy = np.array([0.8, 0.5, 0.9, 0.7], dtype=np.float32)

print(f"Likes Action:  {likes_action}  (User 0: yes, User 1: maybe, User 2: no, User 3: unknown)")
print(f"Likes Comedy:  {likes_comedy}  (User 0: yes, User 1: unknown, User 2: yes, User 3: somewhat)")
print()

# Query: "Recommend action-comedy movie" (user likes BOTH genres)
print("Query: Who should we recommend an action-comedy to?")
print("       (Likes Action AND Likes Comedy)")
print()

# Deductive (T=0): Only recommend when CERTAIN
deductive_and = temperature_scaled_operation(
    logical_and, temperature=0.0, backend=backend
)
result_deductive = np.asarray(deductive_and(likes_action, likes_comedy, backend=backend))

# Analogical (T=1): Can recommend with uncertainty
analogical_and = temperature_scaled_operation(
    logical_and, temperature=1.0, backend=backend
)
result_analogical = np.asarray(analogical_and(likes_action, likes_comedy, backend=backend))

print(f"T=0 (Deductive):  {np.array2string(result_deductive, precision=2)}")
print("  → Only User 0 gets recommendation (certain they like both)")
print()
print(f"T=1 (Analogical): {np.array2string(result_analogical, precision=2)}")
print("  → User 0 strong rec (0.78), User 1/3 weak rec (0.46-0.55)")
print()

# =============================================================================
# PART 4: When to Use Each Mode
# =============================================================================

print("=" * 60)
print("PART 4: When to Use Each Temperature")
print("=" * 60)
print("""
┌─────────────────────────────────────────────────────────────┐
│  USE T=0 (DEDUCTIVE) WHEN:                                  │
│  • You need provable correctness (legal, medical, safety)   │
│  • You have complete, reliable data                         │
│  • False positives are costly (fraud detection)             │
│  • You're doing database-style queries                      │
│                                                             │
│  USE T>0 (ANALOGICAL) WHEN:                                 │
│  • You have incomplete or uncertain data                    │
│  • You want to generalize beyond explicit rules             │
│  • You're doing recommendations or ranking                  │
│  • You want soft/fuzzy matches                              │
│                                                             │
│  TYPICAL WORKFLOW:                                          │
│  • Training neural networks: T>0 (need gradients)           │
│  • Production inference: T=0 (need reliability)             │
│  • Recommendations: T=0.3-1.0 (want diversity)              │
└─────────────────────────────────────────────────────────────┘
""")

# =============================================================================
# WHAT'S NEXT?
# =============================================================================
# Now that you understand temperature:
#   - examples/03_knowledge_graph_query.py (realistic KG queries)
#   - examples/knowledge_graph_reasoning.py (full 8-section walkthrough)
#   - docs/tutorials/03_temperature_guide.md (deep dive)
