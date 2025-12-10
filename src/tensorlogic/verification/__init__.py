"""Formal verification integration for TensorLogic.

This package provides Lean 4 formal verification capabilities for tensor logic
operations and learned neural predicates. Enables provably correct neural-symbolic
reasoning with mathematical guarantees.

**Key Features:**
- Theorem verification via Lean 4 kernel
- Proof-guided training with logical constraints
- Neural tactic suggestion for automated proof search
- Counterexample generation for failed proofs

**Strategic Value:**
- First neural-symbolic framework with formal verification
- Mathematical guarantees for safety-critical applications
- Enables publication-ready verified models

**External Dependencies:**
- Lean 4 system installation (required for verification features)
- LeanDojo Python package (optional, enables verification APIs)

**Usage:**

Basic verification::

    from tensorlogic.verification import LeanBridge, VerificationResult

    bridge = LeanBridge("lean/")
    result = bridge.verify_theorem("TensorLogic.and_commutative")

    if result.verified:
        print(f"✓ Theorem verified: {result.theorem_name}")
    else:
        print(f"✗ Verification failed: {result.counterexample}")

Context manager usage::

    with LeanBridge("lean/") as bridge:
        result = bridge.verify_theorem("TensorLogic.transitivity")
        print(result)

**Note:** Verification is optional. If Lean 4 or LeanDojo are not installed,
TensorLogic core functionality remains available. Verification features will
raise LeanNotInstalledError if dependencies are missing.

See specification: docs/specs/verification/spec.md
See implementation plan: docs/specs/verification/plan.md
"""

from __future__ import annotations

from tensorlogic.verification.lean_bridge import (
    LeanBridge,
    LeanNotInstalledError,
    LeanProjectError,
)
from tensorlogic.verification.results import VerificationResult

__all__ = [
    "LeanBridge",
    "LeanNotInstalledError",
    "LeanProjectError",
    "VerificationResult",
]

__version__ = "0.1.0"
