# TensorLogic Newcomers Guide

Welcome! This guide assumes you know Python but nothing about logic programming, tensors, or AI. You'll go from zero to productive in about 30 minutes.

## What You'll Learn

1. What TensorLogic actually does (in plain English)
2. Your first program (5 minutes)
3. Multi-hop reasoning (10 minutes)
4. The temperature dial (10 minutes)
5. Where to go next

---

## Part 1: What Is TensorLogic?

### The 30-Second Explanation

TensorLogic answers questions about relationships in data:

```
Given: "Alice is parent of Bob, Bob is parent of Carol"
Query: "Is Alice a grandparent of Carol?"
Answer: Yes (inferred from the chain: Alice → Bob → Carol)
```

**The key insight:** We never explicitly said "Alice is grandparent of Carol". TensorLogic *figured it out* by chaining the relationships.

### Why Is This Useful?

Real-world examples:
- **Knowledge graphs:** "Which companies are connected to Person X through investments?"
- **Recommendations:** "Which products are similar based on shared categories?"
- **Compliance:** "Does User Y have access to Resource Z through any group membership?"

### What Makes TensorLogic Special?

The **temperature dial**:

```
T=0.0 (DEDUCTIVE)          T=1.0 (ANALOGICAL)
     │                           │
     ▼                           ▼
"Only return TRUE if        "Return confidence scores
 the data PROVES it"         even with incomplete data"
```

Turn it to 0 → exact database query (no false positives)
Turn it up → fuzzy search (finds likely matches)

---

## Part 2: Your First Program

### Step 1: Installation

```bash
pip install python-tensorlogic

# Optional: For GPU acceleration
pip install mlx>=0.30.0        # Apple Silicon (M1/M2/M3)
pip install cupy-cuda12x       # NVIDIA GPU
```

### Step 2: Hello World

Create a file called `hello.py`:

```python
"""My first TensorLogic program."""
import numpy as np
from tensorlogic import create_backend, logical_and, logical_or, logical_not

# Step 1: Create a backend (handles GPU/CPU automatically)
backend = create_backend()
print(f"Using: {type(backend).__name__}")

# Step 2: Create some facts
# TRUE = 1.0, FALSE = 0.0
# Think of these as columns in a spreadsheet
is_tall = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)   # [Alice, Bob, Carol, David]
is_strong = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)

print(f"Is tall:   {is_tall}")    # [1. 0. 1. 0.]
print(f"Is strong: {is_strong}")  # [1. 1. 0. 0.]

# Step 3: Apply logical operations
tall_and_strong = logical_and(is_tall, is_strong, backend=backend)
tall_or_strong = logical_or(is_tall, is_strong, backend=backend)
not_tall = logical_not(is_tall, backend=backend)

print(f"Tall AND Strong: {tall_and_strong}")  # [1. 0. 0. 0.] - Only Alice
print(f"Tall OR Strong:  {tall_or_strong}")   # [1. 1. 1. 0.] - Alice, Bob, Carol
print(f"NOT Tall:        {not_tall}")         # [0. 1. 0. 1.] - Bob, David
```

Run it:
```bash
python hello.py
```

### What Just Happened?

```
Input:                           Output:
is_tall   = [1, 0, 1, 0]        tall_and_strong = [1, 0, 0, 0]
is_strong = [1, 1, 0, 0]                             │
                                 Only Alice (position 0) is BOTH
```

**Key insight:** Logical operations work element-by-element, just like spreadsheet formulas.

---

## Part 3: Multi-Hop Reasoning

Now for the real power: inferring relationships that aren't explicitly stated.

### The Family Tree Example

```python
"""Infer grandparent relationships."""
import numpy as np
from tensorlogic import create_backend
from tensorlogic.core.quantifiers import exists

backend = create_backend()

# Family: Alice → Bob → Carol (3 generations)
entities = ["Alice", "Bob", "Carol"]

# Parent relationship as a matrix
# parent[i][j] = 1.0 means "person i is parent of person j"
parent = np.array([
    [0., 1., 0.],  # Alice is parent of Bob
    [0., 0., 1.],  # Bob is parent of Carol
    [0., 0., 0.],  # Carol has no children
], dtype=np.float32)

print("Parent matrix:")
print("         Alice  Bob  Carol")
print(f"Alice  {parent[0]}")
print(f"Bob    {parent[1]}")
print(f"Carol  {parent[2]}")
```

### The Magic: Matrix Multiplication = Chained Relationships

```python
# Grandparent = Parent of Parent
# In math: Grandparent(x,z) = exists y: Parent(x,y) AND Parent(y,z)
# In tensors: Matrix multiplication!

grandparent_raw = backend.einsum('xy,yz->xz', parent, parent)
grandparent = backend.step(grandparent_raw)  # Convert to 0/1

print("\nGrandparent matrix (inferred!):")
print("         Alice  Bob  Carol")
print(f"Alice  {np.asarray(grandparent)[0]}")  # [0. 0. 1.] - Alice is grandparent of Carol!
print(f"Bob    {np.asarray(grandparent)[1]}")
print(f"Carol  {np.asarray(grandparent)[2]}")
```

### Visualizing the Chain

```
PARENT MATRIX              GRANDPARENT (computed)
       Alice Bob Carol            Alice Bob Carol
Alice [  0    1    0  ]     Alice [  0    0    1  ]  ← Alice → Carol!
Bob   [  0    0    1  ]  ×  Bob   [  0    0    0  ]
Carol [  0    0    0  ]     Carol [  0    0    0  ]

              │
              ▼
     Matrix multiplication
     follows the chain:
     Alice → Bob → Carol
```

### Query: "Who Has Children?"

```python
# EXISTS = "is there any y where Parent(x,y) is true?"
has_children = exists(parent, axis=1, backend=backend)

for i, name in enumerate(entities):
    result = "Yes" if has_children[i] > 0 else "No"
    print(f"{name} has children: {result}")

# Output:
# Alice has children: Yes
# Bob has children: Yes
# Carol has children: No
```

---

## Part 4: The Temperature Dial

This is TensorLogic's superpower: controlling how strictly the system follows logic.

### Setup: Uncertain Knowledge

```python
"""Temperature-controlled reasoning."""
import numpy as np
from tensorlogic import create_backend, logical_and
from tensorlogic.core.temperature import temperature_scaled_operation

backend = create_backend()

# User movie preferences (some uncertain!)
# 1.0 = definitely likes, 0.0 = definitely dislikes, 0.5 = unknown
likes_action = np.array([1.0, 0.6, 0.0, 0.5], dtype=np.float32)
likes_comedy = np.array([0.8, 0.5, 0.9, 0.7], dtype=np.float32)

print("Likes Action:", likes_action, " (User 0: yes, 1: maybe, 2: no, 3: unknown)")
print("Likes Comedy:", likes_comedy, " (User 0: yes, 1: unknown, 2: yes, 3: somewhat)")
```

### Deductive Mode (T=0): "Only Recommend When Certain"

```python
# T=0: Values snap to 0 or 1 based on > 0.5 threshold
deductive_and = temperature_scaled_operation(
    logical_and, temperature=0.0, backend=backend
)

result_deductive = np.asarray(deductive_and(likes_action, likes_comedy, backend=backend))
print(f"\nT=0 (Deductive): {result_deductive}")
# Output: [1. 0. 0. 0.] or similar
# Only User 0 gets a recommendation (certain they like BOTH genres)
```

### Analogical Mode (T=1): "Recommend With Confidence Scores"

```python
# T=1: Keep the graded values, show confidence levels
analogical_and = temperature_scaled_operation(
    logical_and, temperature=1.0, backend=backend
)

result_analogical = np.asarray(analogical_and(likes_action, likes_comedy, backend=backend))
print(f"T=1 (Analogical): {result_analogical}")
# Output: [0.78, 0.56, 0.0, 0.59] or similar
# User 0: strong recommendation (0.78)
# User 1, 3: weak recommendations (0.56, 0.59)
# User 2: no recommendation (0.0 - definitely doesn't like action)
```

### When to Use Each Mode

```
┌─────────────────────────────────────────────────────────────┐
│  T=0 (DEDUCTIVE):                                           │
│  • Legal, medical, safety applications                      │
│  • Database-style queries ("show me exactly what's true")   │
│  • When false positives are costly                          │
│                                                             │
│  T>0 (ANALOGICAL):                                          │
│  • Recommendations and ranking                              │
│  • Incomplete or uncertain data                             │
│  • Training neural networks (need gradients)                │
│  • Exploratory analysis ("what might be related?")          │
└─────────────────────────────────────────────────────────────┘
```

---

## Part 5: Key Concepts Summary

### Tensors = Spreadsheets

```
# 1D tensor: A single column
is_tall = [1.0, 0.0, 1.0, 0.0]  # Alice, Bob, Carol, David

# 2D tensor: A relationship table
#        Alice  Bob  Carol
# Alice [  0     1     0  ]  ← Alice is parent of Bob
# Bob   [  0     0     1  ]  ← Bob is parent of Carol
# Carol [  0     0     0  ]
```

### Logical Operations = Element-Wise

```
AND: Both must be 1.0 → returns 1.0
OR:  Either is 1.0 → returns 1.0
NOT: Flips 1.0 ↔ 0.0
```

### Multi-Hop = Matrix Multiplication

```
Parent × Parent = Grandparent
(Follow one edge, then another)
```

### Temperature = Strictness Dial

```
T=0: Binary (TRUE/FALSE only)
T>0: Graded (confidence 0.0 to 1.0)
```

---

## Where to Go Next

### Example Files (In Order)

1. `examples/00_hello_world.py` - What you just learned
2. `examples/01_family_tree_minimal.py` - Multi-hop reasoning
3. `examples/02_temperature_demo.py` - Temperature control
4. `examples/knowledge_graph_reasoning.py` - Full 8-entity example

### Run Them

```bash
uv run python examples/00_hello_world.py
uv run python examples/01_family_tree_minimal.py
uv run python examples/02_temperature_demo.py
```

### Deeper Dives

- **How Logic Becomes Tensors:** [docs/concepts/tensor-logic-mapping.md](../concepts/tensor-logic-mapping.md)
- **All Compilation Strategies:** [docs/api/compilation.md](../api/compilation.md)
- **Performance on GPU:** [docs/PERFORMANCE.md](../PERFORMANCE.md)
- **The Original Paper:** [arXiv:2510.12269](https://arxiv.org/abs/2510.12269)

### Coming From Other Backgrounds?

- **Prolog/Datalog:** See [docs/tutorials/01_from_datalog.md](01_from_datalog.md)
- **PyTorch/ML:** See [docs/tutorials/02_from_pytorch.md](02_from_pytorch.md)

---

## Troubleshooting

### "Module not found" Error

```bash
pip install python-tensorlogic
```

### "NumpyBackend" Instead of GPU

```bash
# For Apple Silicon
pip install mlx>=0.30.0

# For NVIDIA
pip install cupy-cuda12x
```

Then verify:
```python
from tensorlogic import create_backend
backend = create_backend()
print(type(backend).__name__)  # Should show MLXBackend or CUDABackend
```

### "Shape mismatch" Errors

The arrays in a logical operation must have the same length:

```python
# ❌ WRONG: Different lengths
a = [1.0, 0.0, 1.0]
b = [1.0, 0.0]        # Only 2 elements!
logical_and(a, b)     # Error!

# ✅ CORRECT: Same length
a = [1.0, 0.0, 1.0]
b = [1.0, 0.0, 1.0]   # 3 elements
logical_and(a, b)     # Works!
```

---

**You're ready.** Go run the examples and build something!
