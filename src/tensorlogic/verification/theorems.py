"""Verified TensorLogic Theorems.

This module provides access to formally verified theorems about tensor logic
operations. All theorems are proven in Lean 4 using constructive proofs.

**Verification Status:** All theorems listed here have been formally verified
with `lake build` successfully completing on the Lean project.

**Strategic Value:**
- First neural-symbolic framework with formal verification
- Mathematical guarantees for logical operations
- Foundation for proof-guided learning

Usage:
    >>> from tensorlogic.verification import PROVEN_THEOREMS, verify_build
    >>> print(len(PROVEN_THEOREMS))
    14
    >>> # Check if Lean project builds successfully
    >>> result = verify_build()
    >>> print(result.verified)
    True
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

__all__ = [
    "ProvenTheorem",
    "PROVEN_THEOREMS",
    "get_theorem",
    "list_theorems_by_category",
    "verify_build",
]


@dataclass(frozen=True)
class ProvenTheorem:
    """A formally verified theorem.

    Attributes:
        name: Lean theorem identifier
        statement: Human-readable theorem statement
        category: Category of the theorem
        lean_proof: The Lean 4 proof tactic
    """

    name: str
    statement: str
    category: Literal[
        "and_properties",
        "or_properties",
        "demorgan",
        "distributivity",
        "implication",
        "additional",
    ]
    lean_proof: str


# All theorems proven in lean/TensorLogic.lean
PROVEN_THEOREMS: tuple[ProvenTheorem, ...] = (
    # AND Properties
    ProvenTheorem(
        name="and_commutative",
        statement="a ∧ b = b ∧ a",
        category="and_properties",
        lean_proof="simp [tensor_and, Bool.and_comm]",
    ),
    ProvenTheorem(
        name="and_associative",
        statement="(a ∧ b) ∧ c = a ∧ (b ∧ c)",
        category="and_properties",
        lean_proof="simp [tensor_and, Bool.and_assoc]",
    ),
    ProvenTheorem(
        name="and_idempotent",
        statement="a ∧ a = a",
        category="and_properties",
        lean_proof="simp [tensor_and]",
    ),
    # OR Properties
    ProvenTheorem(
        name="or_commutative",
        statement="a ∨ b = b ∨ a",
        category="or_properties",
        lean_proof="simp [tensor_or, Bool.or_comm]",
    ),
    ProvenTheorem(
        name="or_associative",
        statement="(a ∨ b) ∨ c = a ∨ (b ∨ c)",
        category="or_properties",
        lean_proof="simp [tensor_or, Bool.or_assoc]",
    ),
    ProvenTheorem(
        name="or_idempotent",
        statement="a ∨ a = a",
        category="or_properties",
        lean_proof="simp [tensor_or]",
    ),
    # De Morgan's Laws
    ProvenTheorem(
        name="demorgan_and",
        statement="¬(a ∧ b) = ¬a ∨ ¬b",
        category="demorgan",
        lean_proof="cases a <;> cases b <;> rfl",
    ),
    ProvenTheorem(
        name="demorgan_or",
        statement="¬(a ∨ b) = ¬a ∧ ¬b",
        category="demorgan",
        lean_proof="cases a <;> cases b <;> rfl",
    ),
    # Distributivity
    ProvenTheorem(
        name="and_distributes_or",
        statement="a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)",
        category="distributivity",
        lean_proof="cases a <;> cases b <;> cases c <;> rfl",
    ),
    ProvenTheorem(
        name="or_distributes_and",
        statement="a ∨ (b ∧ c) = (a ∨ b) ∧ (a ∨ c)",
        category="distributivity",
        lean_proof="cases a <;> cases b <;> cases c <;> rfl",
    ),
    # Implication Properties
    ProvenTheorem(
        name="implies_elimination",
        statement="(a → b) = (¬a ∨ b)",
        category="implication",
        lean_proof="rfl",
    ),
    ProvenTheorem(
        name="contraposition",
        statement="(a → b) = (¬b → ¬a)",
        category="implication",
        lean_proof="cases a <;> cases b <;> rfl",
    ),
    # Additional Properties
    ProvenTheorem(
        name="double_negation",
        statement="¬¬a = a",
        category="additional",
        lean_proof="cases a <;> rfl",
    ),
    ProvenTheorem(
        name="absorption_and",
        statement="a ∧ (a ∨ b) = a",
        category="additional",
        lean_proof="cases a <;> cases b <;> rfl",
    ),
    ProvenTheorem(
        name="absorption_or",
        statement="a ∨ (a ∧ b) = a",
        category="additional",
        lean_proof="cases a <;> cases b <;> rfl",
    ),
)


def get_theorem(name: str) -> ProvenTheorem | None:
    """Get a proven theorem by name.

    Args:
        name: Theorem name (e.g., 'and_commutative')

    Returns:
        ProvenTheorem if found, None otherwise

    Example:
        >>> thm = get_theorem('demorgan_and')
        >>> print(thm.statement)
        ¬(a ∧ b) = ¬a ∨ ¬b
    """
    for theorem in PROVEN_THEOREMS:
        if theorem.name == name:
            return theorem
    return None


def list_theorems_by_category(
    category: str | None = None,
) -> list[ProvenTheorem]:
    """List theorems, optionally filtered by category.

    Args:
        category: Optional category filter

    Returns:
        List of matching theorems

    Example:
        >>> theorems = list_theorems_by_category('demorgan')
        >>> print(len(theorems))
        2
    """
    if category is None:
        return list(PROVEN_THEOREMS)
    return [t for t in PROVEN_THEOREMS if t.category == category]


@dataclass
class BuildResult:
    """Result of Lean project build.

    Attributes:
        verified: Whether all theorems compiled successfully
        build_time: Time taken to build in seconds
        output: Build output or error message
        lean_version: Lean version used
    """

    verified: bool
    build_time: float
    output: str
    lean_version: str | None = None


def verify_build(lean_project_path: str | Path | None = None) -> BuildResult:
    """Verify that all theorems compile successfully.

    This runs `lake build` on the Lean project and returns the result.
    Requires Lean 4 to be installed.

    Args:
        lean_project_path: Path to Lean project (defaults to bundled lean/)

    Returns:
        BuildResult with verification status

    Example:
        >>> result = verify_build()
        >>> if result.verified:
        ...     print(f"All {len(PROVEN_THEOREMS)} theorems verified!")
    """
    # Find Lean project path
    if lean_project_path is None:
        # Default to bundled lean directory
        # Path: src/tensorlogic/verification/theorems.py -> lean/
        module_dir = Path(__file__).parent.parent.parent.parent
        lean_project_path = module_dir / "lean"

    lean_project_path = Path(lean_project_path)

    # Check if Lean is installed
    lean_path = shutil.which("lean")
    if lean_path is None:
        return BuildResult(
            verified=False,
            build_time=0.0,
            output="Lean 4 not installed. Install from: https://lean-lang.org/lean4/doc/setup.html",
            lean_version=None,
        )

    # Get Lean version
    try:
        version_result = subprocess.run(
            ["lean", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lean_version = version_result.stdout.strip()
    except Exception:
        lean_version = None

    # Check if project exists
    if not lean_project_path.exists():
        return BuildResult(
            verified=False,
            build_time=0.0,
            output=f"Lean project not found at: {lean_project_path}",
            lean_version=lean_version,
        )

    # Run lake build
    start_time = time.perf_counter()
    try:
        result = subprocess.run(
            ["lake", "build"],
            cwd=lean_project_path,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        build_time = time.perf_counter() - start_time

        if result.returncode == 0:
            return BuildResult(
                verified=True,
                build_time=build_time,
                output=result.stdout + result.stderr,
                lean_version=lean_version,
            )
        else:
            return BuildResult(
                verified=False,
                build_time=build_time,
                output=result.stderr or result.stdout,
                lean_version=lean_version,
            )

    except subprocess.TimeoutExpired:
        return BuildResult(
            verified=False,
            build_time=300.0,
            output="Build timed out after 5 minutes",
            lean_version=lean_version,
        )
    except Exception as e:
        return BuildResult(
            verified=False,
            build_time=0.0,
            output=str(e),
            lean_version=lean_version,
        )


def print_theorem_summary() -> None:
    """Print a summary of all proven theorems."""
    print("=" * 60)
    print("TENSORLOGIC VERIFIED THEOREMS")
    print("=" * 60)

    categories = {}
    for theorem in PROVEN_THEOREMS:
        if theorem.category not in categories:
            categories[theorem.category] = []
        categories[theorem.category].append(theorem)

    for category, theorems in categories.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for theorem in theorems:
            print(f"  ✓ {theorem.name}: {theorem.statement}")

    print(f"\nTotal: {len(PROVEN_THEOREMS)} theorems proven")
    print("=" * 60)


if __name__ == "__main__":
    print_theorem_summary()
    print("\nVerifying Lean build...")
    result = verify_build()
    if result.verified:
        print(f"✓ All theorems verified in {result.build_time:.2f}s")
    else:
        print(f"✗ Verification failed: {result.output}")
