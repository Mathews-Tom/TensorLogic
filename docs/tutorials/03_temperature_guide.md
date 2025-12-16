# Temperature Guide: The TensorLogic Control Dial

Temperature is TensorLogic's distinguishing feature. This guide explains what it does, when to use different values, and how it works under the hood.

---

## Quick Reference

```
T = 0.0          T = 0.5          T = 1.0          T = 2.0
  │                │                │                │
  ▼                ▼                ▼                ▼
DEDUCTIVE        CAUTIOUS        ANALOGICAL      EXPLORATORY
Binary           Mixed            Graded          Creative
(0/1 only)       (mostly 0/1)     (0.0-1.0)       (soft boundaries)
```

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| **T = 0** | Pure deduction, binary results | Verification, database queries, safety-critical |
| **T = 0.1-0.5** | Conservative generalization | Production systems with uncertainty tolerance |
| **T = 1.0** | Balanced analogical reasoning | Recommendations, exploratory queries |
| **T > 1.0** | Creative/exploratory | Hypothesis generation, brainstorming |

---

## Part 1: Intuitive Explanation

### The Volume Knob Analogy

Think of temperature as a **strictness dial**:

```
STRICT                                        CREATIVE
  │                                              │
  ▼                                              ▼
  ┌─────────────────────────────────────────────┐
  │ ○──────────────────●─────────────────────○ │
  │      T=0         T=1                  T=2   │
  └─────────────────────────────────────────────┘
  │                    │                        │
  │                    │                        │
  ▼                    ▼                        ▼
"Show me ONLY       "Show me what's        "Show me what
 what's proven"      likely, with scores"   MIGHT be related"
```

### Concrete Example: Movie Recommendations

Imagine a user's preferences:
- Likes Action: **80%** confident
- Likes Comedy: **70%** confident

**Query:** "Should we recommend an Action-Comedy movie?"

```python
# At T=0 (Deductive)
# 80% > 50% → TRUE, 70% > 50% → TRUE
# TRUE AND TRUE = TRUE → Recommend!
result = 1.0  # Binary: recommend or don't

# At T=1 (Analogical)
# Keep the actual confidence values
# 80% AND 70% ≈ 56% confidence
result = 0.56  # Graded: how strongly to recommend
```

**T=0** answers: "Yes or No?"
**T>0** answers: "How much?"

---

## Part 2: Mathematical Foundation

### The Temperature Formula

Temperature controls interpolation between **hard** (binary) and **soft** (continuous) results:

```
α = 1 - exp(-T)       # Interpolation weight (0 to 1)

result = (1 - α) × hard_result + α × soft_result
```

| Temperature | α (weight) | Behavior |
|-------------|------------|----------|
| T = 0 | α = 0.00 | 100% hard (binary) |
| T = 0.5 | α = 0.39 | 61% hard, 39% soft |
| T = 1.0 | α = 0.63 | 37% hard, 63% soft |
| T = 2.0 | α = 0.86 | 14% hard, 86% soft |

### The Step Function (Hard Logic)

At T=0, values snap to 0 or 1:

```
step(x) = 1  if x > 0.5
          0  otherwise

Input:  [0.3, 0.5, 0.7, 0.9]
Output: [0.0, 0.0, 1.0, 1.0]
         ↑    ↑    ↑    ↑
      Below threshold → 0
      Above threshold → 1
```

### Visual: Temperature Effect

```
Input: [0.3, 0.5, 0.7, 0.9]

T=0.0:  [0.00, 0.00, 1.00, 1.00]  ← Binary
         │     │     │     │
T=0.5:  [0.12, 0.20, 0.88, 0.96]  ← Mostly binary
         │     │     │     │
T=1.0:  [0.19, 0.32, 0.81, 0.94]  ← Blended
         │     │     │     │
T=2.0:  [0.26, 0.43, 0.74, 0.91]  ← Close to input
         ▲     ▲     ▲     ▲
         │     │     │     │
        Low values stay   High values stay
        relatively low    relatively high
```

---

## Part 3: Practical Usage

### Creating Temperature-Scaled Operations

```python
from tensorlogic import create_backend, logical_and
from tensorlogic.core.temperature import temperature_scaled_operation

backend = create_backend()

# Create operations at different temperatures
deductive_and = temperature_scaled_operation(
    logical_and, temperature=0.0, backend=backend
)

analogical_and = temperature_scaled_operation(
    logical_and, temperature=1.0, backend=backend
)

# Use them
a = [0.8, 0.3, 0.6]
b = [0.7, 0.4, 0.9]

result_strict = deductive_and(a, b, backend=backend)   # [1.0, 0.0, 1.0]
result_soft = analogical_and(a, b, backend=backend)    # [0.56, 0.12, 0.54]
```

### Using the High-Level API

```python
from tensorlogic.api import reason

# Deductive reasoning
certain_result = reason(
    'Grandparent(x, z)',
    bindings={'x': alice_idx, 'z': carol_idx},
    temperature=0.0,
    backend=backend
)

# Analogical reasoning
likely_result = reason(
    'Grandparent(x, z)',
    bindings={'x': alice_idx, 'z': carol_idx},
    temperature=0.5,
    backend=backend
)
```

---

## Part 4: When to Use Each Temperature

### T = 0: Pure Deduction

**Use when:**
- Provable correctness is required
- False positives are costly
- Data is complete and reliable
- Auditing/compliance scenarios

**Examples:**
```
✓ "Is this user authorized to access this resource?"
✓ "Does this transaction violate compliance rules?"
✓ "Is this patient allergic to this medication?"
```

**Code pattern:**
```python
# Safety-critical query
result = reason('HasPermission(user, resource)', temperature=0.0, ...)
if result > 0.5:
    grant_access()
else:
    deny_access()
```

### T = 0.1-0.5: Conservative Generalization

**Use when:**
- Some uncertainty is acceptable
- Want to catch edge cases
- Production systems with fallbacks

**Examples:**
```
✓ "Find potentially fraudulent transactions"
✓ "Which products might need restocking?"
✓ "Flag accounts that may need review"
```

**Code pattern:**
```python
# Production system with uncertainty
results = reason('MayNeedReview(account)', temperature=0.3, ...)
flagged = [acc for acc, score in results if score > 0.4]
```

### T = 1.0: Balanced Analogical

**Use when:**
- Ranking/scoring is needed
- Data is incomplete
- User-facing recommendations

**Examples:**
```
✓ "Rank products by relevance to user"
✓ "Score candidates for job matching"
✓ "Find similar documents"
```

**Code pattern:**
```python
# Recommendation system
scores = reason('SimilarTo(item, query)', temperature=1.0, ...)
ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:10]
```

### T > 1.0: Exploratory

**Use when:**
- Generating hypotheses
- Creative exploration
- Research/experimentation

**Examples:**
```
✓ "What entities might be connected?"
✓ "Generate potential research directions"
✓ "Explore latent relationships"
```

**Code pattern:**
```python
# Hypothesis generation
hypotheses = reason('MightBeRelated(x, y)', temperature=2.0, ...)
candidates = [(i, j, score) for (i, j), score in hypotheses if score > 0.1]
```

---

## Part 5: Decision Tree

```
Start Here
    │
    ├── Is correctness critical (legal, medical, safety)?
    │   └── YES → T = 0.0
    │
    ├── Is data complete and reliable?
    │   └── YES → T = 0.0 to 0.3
    │
    ├── Need ranking/scores (not just yes/no)?
    │   └── YES → T = 0.5 to 1.0
    │
    ├── Training a neural network?
    │   └── YES → T = 1.0+ (need gradients)
    │
    ├── Exploratory analysis?
    │   └── YES → T = 1.0 to 2.0
    │
    └── Default recommendation → T = 0.5
```

---

## Part 6: Common Patterns

### Pattern 1: Train Soft, Infer Hard

```python
# Training: Use soft semantics (differentiable)
for epoch in range(epochs):
    loss = compute_loss(
        reason('Rule(x)', temperature=1.0, ...)  # Gradients flow
    )
    loss.backward()

# Inference: Use hard semantics (exact)
with torch.no_grad():
    result = reason('Rule(x)', temperature=0.0, ...)  # Binary output
```

### Pattern 2: Threshold Adjustment

```python
# Lower temperature + lower threshold = conservative
result = reason('Query(x)', temperature=0.3, ...)
matches = [x for x, score in result if score > 0.7]  # High confidence only

# Higher temperature + lower threshold = permissive
result = reason('Query(x)', temperature=1.0, ...)
matches = [x for x, score in result if score > 0.3]  # Include uncertain
```

### Pattern 3: Temperature Sweep

```python
# Find optimal temperature for your task
for T in [0.0, 0.3, 0.5, 0.7, 1.0]:
    result = reason('Query(x)', temperature=T, ...)
    precision, recall = evaluate(result, ground_truth)
    print(f"T={T}: P={precision:.2f}, R={recall:.2f}")
```

---

## Part 7: Troubleshooting

### "All results are 0 at T=0"

Your input values are all below 0.5. At T=0, step(x < 0.5) = 0.

**Fix:** Use T > 0 for graded inputs, or ensure inputs are above 0.5 for "true".

### "Results don't change with temperature"

Your inputs are already binary (0.0 or 1.0). Temperature only affects intermediate values.

**Check:**
```python
print(f"Input range: min={a.min()}, max={a.max()}")
# If min=0 and max=1 with nothing in between, temperature has no effect
```

### "Results seem too permissive"

Temperature is too high for your use case.

**Fix:** Lower temperature or raise the threshold:
```python
result = reason('Query(x)', temperature=0.3, ...)  # Lower T
matches = [x for x, s in result if s > 0.6]        # Higher threshold
```

---

## Summary

| Scenario | Temperature | Threshold |
|----------|-------------|-----------|
| Safety-critical | 0.0 | 0.5 |
| Production (conservative) | 0.3 | 0.5-0.7 |
| Recommendations | 1.0 | 0.3-0.5 |
| Training neural networks | 1.0+ | N/A |
| Exploration | 1.5-2.0 | 0.1-0.3 |

**Remember:**
- T=0 → Binary logic (provable, no hallucinations)
- T>0 → Graded logic (can generalize, may over-generalize)
- When in doubt, start with T=0.5 and adjust

---

## Next Steps

- **Code example:** `examples/02_temperature_demo.py`
- **Mathematical details:** [Tensor-Logic Mapping](../concepts/tensor-logic-mapping.md)
- **Original paper:** [arXiv:2510.12269](https://arxiv.org/abs/2510.12269)
