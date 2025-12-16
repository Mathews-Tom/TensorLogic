"""TensorLogic Hello World: Your First Neural-Symbolic Program

This is the simplest possible TensorLogic program. It demonstrates that
logical operations (AND, OR, NOT) work directly on arrays of truth values.

What you'll learn:
    1. How to create a TensorLogic backend
    2. How to represent TRUE/FALSE as 1.0/0.0 in arrays
    3. How to apply logical operations to arrays

Run this example:
    uv run python examples/00_hello_world.py

Expected output:
    Backend: NumpyBackend (or MLXBackend on Apple Silicon)

    Facts A: [1. 0. 1. 0.]  (TRUE, FALSE, TRUE, FALSE)
    Facts B: [1. 1. 0. 0.]  (TRUE, TRUE, FALSE, FALSE)

    A AND B: [1. 0. 0. 0.]  <- Only position 0 is TRUE in BOTH
    A OR B:  [1. 1. 1. 0.]  <- Positions 0,1,2 are TRUE in at least one
    NOT A:   [0. 1. 0. 1.]  <- Flips TRUE to FALSE and vice versa
"""

from __future__ import annotations

import numpy as np

from tensorlogic import create_backend, logical_and, logical_not, logical_or

# =============================================================================
# STEP 1: Create a backend
# =============================================================================
# The backend handles all tensor operations. TensorLogic auto-selects
# the best available: MLX (Apple Silicon) -> CUDA (NVIDIA) -> NumPy (CPU)

backend = create_backend()
print(f"Backend: {type(backend).__name__}")
print()

# =============================================================================
# STEP 2: Create arrays of truth values
# =============================================================================
# In TensorLogic, TRUE = 1.0 and FALSE = 0.0
# Think of these as columns in a spreadsheet with checkmarks

facts_a = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)  # TRUE, FALSE, TRUE, FALSE
facts_b = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)  # TRUE, TRUE, FALSE, FALSE

print(f"Facts A: {facts_a}  (TRUE, FALSE, TRUE, FALSE)")
print(f"Facts B: {facts_b}  (TRUE, TRUE, FALSE, FALSE)")
print()

# =============================================================================
# STEP 3: Apply logical operations
# =============================================================================
# These work element-by-element, just like in a spreadsheet formula

# AND: Both must be TRUE
result_and = logical_and(facts_a, facts_b, backend=backend)
print(f"A AND B: {result_and}  <- Only position 0 is TRUE in BOTH")

# OR: At least one must be TRUE
result_or = logical_or(facts_a, facts_b, backend=backend)
print(f"A OR B:  {result_or}  <- Positions 0,1,2 are TRUE in at least one")

# NOT: Flip TRUE to FALSE and vice versa
result_not = logical_not(facts_a, backend=backend)
print(f"NOT A:   {result_not}  <- Flips TRUE to FALSE and vice versa")

# =============================================================================
# WHAT'S NEXT?
# =============================================================================
# Now that you understand the basics, try:
#   - examples/01_family_tree_minimal.py  (multi-hop reasoning)
#   - examples/02_temperature_demo.py     (deductive vs analogical reasoning)
#
# Or read the newcomers guide:
#   - docs/tutorials/00_newcomers_guide.md
