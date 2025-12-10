"""Verification result types for Lean 4 integration.

This module defines structured result types for theorem verification and proof
validation. Results include success/failure status, proof traces, counterexamples,
and tactic suggestions for automated proof search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["VerificationResult"]


TacticSuggestion = tuple[str, float]  # (tactic_name, confidence_score)


@dataclass
class VerificationResult:
    """Result of Lean 4 theorem verification.

    Attributes:
        verified: Whether the theorem was successfully proven
        theorem_name: Full Lean theorem identifier (e.g., 'TensorLogic.and_commutative')
        proof_trace: Optional Lean proof trace/script
        counterexample: Optional counterexample if verification failed
        verification_time: Time taken for verification in seconds
        tactic_suggestions: List of suggested tactics with confidence scores

    Examples:
        >>> result = VerificationResult(
        ...     verified=True,
        ...     theorem_name="TensorLogic.and_commutative",
        ...     proof_trace="apply and.comm",
        ...     verification_time=0.15
        ... )
        >>> print(result)
        ✓ VERIFIED: TensorLogic.and_commutative

        >>> failed_result = VerificationResult(
        ...     verified=False,
        ...     theorem_name="TensorLogic.transitivity",
        ...     counterexample={"a": True, "b": False, "c": True},
        ...     tactic_suggestions=[("apply trans", 0.82), ("simp", 0.65)]
        ... )
        >>> print(failed_result)
        ✗ FAILED: TensorLogic.transitivity
        Counterexample: {'a': True, 'b': False, 'c': True}
        Suggested tactics:
          - apply trans (confidence: 82.00%)
          - simp (confidence: 65.00%)
    """

    verified: bool
    theorem_name: str
    proof_trace: str | None = None
    counterexample: Any | None = None
    verification_time: float = 0.0
    tactic_suggestions: list[TacticSuggestion] = field(default_factory=list)

    def __str__(self) -> str:
        """Human-readable verification report.

        Returns:
            Formatted string with verification status, counterexamples, and
            tactic suggestions.
        """
        status = "✓ VERIFIED" if self.verified else "✗ FAILED"
        report = f"{status}: {self.theorem_name}\n"

        if self.counterexample is not None:
            report += f"Counterexample: {self.counterexample}\n"

        if self.proof_trace and self.verified:
            report += f"Proof: {self.proof_trace}\n"

        if self.tactic_suggestions:
            report += "Suggested tactics:\n"
            for tactic, confidence in self.tactic_suggestions[:3]:
                report += f"  - {tactic} (confidence: {confidence:.2%})\n"

        report += f"Verification time: {self.verification_time:.3f}s"

        return report

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation of verification result.
        """
        return {
            "verified": self.verified,
            "theorem_name": self.theorem_name,
            "proof_trace": self.proof_trace,
            "counterexample": self.counterexample,
            "verification_time": self.verification_time,
            "tactic_suggestions": [
                {"tactic": tactic, "confidence": conf}
                for tactic, conf in self.tactic_suggestions
            ],
        }
