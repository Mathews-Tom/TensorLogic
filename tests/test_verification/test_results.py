"""Tests for tensorlogic.verification.results module.

Tests cover VerificationResult dataclass functionality including string
representation, dictionary conversion, and result formatting.
"""

from __future__ import annotations


from tensorlogic.verification.results import VerificationResult


class TestVerificationResult:
    """Test suite for VerificationResult dataclass."""

    def test_verified_result_creation(self) -> None:
        """Test creation of successful verification result."""
        result = VerificationResult(
            verified=True,
            theorem_name="TensorLogic.and_commutative",
            proof_trace="apply and.comm",
            verification_time=0.15,
        )

        assert result.verified is True
        assert result.theorem_name == "TensorLogic.and_commutative"
        assert result.proof_trace == "apply and.comm"
        assert result.counterexample is None
        assert result.verification_time == 0.15
        assert result.tactic_suggestions == []

    def test_failed_result_creation(self) -> None:
        """Test creation of failed verification result."""
        counterexample = {"a": True, "b": False, "c": True}
        tactics = [("apply trans", 0.82), ("simp", 0.65)]

        result = VerificationResult(
            verified=False,
            theorem_name="TensorLogic.transitivity",
            counterexample=counterexample,
            verification_time=1.23,
            tactic_suggestions=tactics,
        )

        assert result.verified is False
        assert result.theorem_name == "TensorLogic.transitivity"
        assert result.proof_trace is None
        assert result.counterexample == counterexample
        assert result.verification_time == 1.23
        assert result.tactic_suggestions == tactics

    def test_verified_result_str_representation(self) -> None:
        """Test string representation of verified result."""
        result = VerificationResult(
            verified=True,
            theorem_name="TensorLogic.and_commutative",
            proof_trace="apply and.comm",
            verification_time=0.15,
        )

        str_repr = str(result)

        assert "✓ VERIFIED" in str_repr
        assert "TensorLogic.and_commutative" in str_repr
        assert "Proof: apply and.comm" in str_repr
        assert "0.150s" in str_repr

    def test_failed_result_str_representation(self) -> None:
        """Test string representation of failed result."""
        result = VerificationResult(
            verified=False,
            theorem_name="TensorLogic.transitivity",
            counterexample={"a": True, "b": False},
            tactic_suggestions=[("apply trans", 0.82), ("simp", 0.65)],
            verification_time=1.23,
        )

        str_repr = str(result)

        assert "✗ FAILED" in str_repr
        assert "TensorLogic.transitivity" in str_repr
        assert "Counterexample:" in str_repr
        assert "{'a': True, 'b': False}" in str_repr
        assert "Suggested tactics:" in str_repr
        assert "apply trans" in str_repr
        assert "82.00%" in str_repr
        assert "1.230s" in str_repr

    def test_str_representation_tactic_limit(self) -> None:
        """Test string representation limits tactics to top 3."""
        tactics = [
            ("tactic1", 0.9),
            ("tactic2", 0.8),
            ("tactic3", 0.7),
            ("tactic4", 0.6),
            ("tactic5", 0.5),
        ]

        result = VerificationResult(
            verified=False,
            theorem_name="TensorLogic.test",
            tactic_suggestions=tactics,
        )

        str_repr = str(result)

        # Only first 3 tactics should be shown
        assert "tactic1" in str_repr
        assert "tactic2" in str_repr
        assert "tactic3" in str_repr
        assert "tactic4" not in str_repr
        assert "tactic5" not in str_repr

    def test_to_dict_conversion(self) -> None:
        """Test dictionary conversion for JSON serialization."""
        tactics = [("apply trans", 0.82), ("simp", 0.65)]
        result = VerificationResult(
            verified=False,
            theorem_name="TensorLogic.transitivity",
            counterexample={"a": True, "b": False},
            tactic_suggestions=tactics,
            verification_time=1.23,
        )

        result_dict = result.to_dict()

        assert result_dict == {
            "verified": False,
            "theorem_name": "TensorLogic.transitivity",
            "proof_trace": None,
            "counterexample": {"a": True, "b": False},
            "verification_time": 1.23,
            "tactic_suggestions": [
                {"tactic": "apply trans", "confidence": 0.82},
                {"tactic": "simp", "confidence": 0.65},
            ],
        }

    def test_to_dict_verified_result(self) -> None:
        """Test dictionary conversion for verified result."""
        result = VerificationResult(
            verified=True,
            theorem_name="TensorLogic.and_commutative",
            proof_trace="apply and.comm",
            verification_time=0.15,
        )

        result_dict = result.to_dict()

        assert result_dict == {
            "verified": True,
            "theorem_name": "TensorLogic.and_commutative",
            "proof_trace": "apply and.comm",
            "counterexample": None,
            "verification_time": 0.15,
            "tactic_suggestions": [],
        }

    def test_empty_tactic_suggestions(self) -> None:
        """Test result with empty tactic suggestions list."""
        result = VerificationResult(
            verified=False,
            theorem_name="TensorLogic.test",
            tactic_suggestions=[],
        )

        str_repr = str(result)

        assert "Suggested tactics:" not in str_repr

    def test_minimal_result(self) -> None:
        """Test result with only required fields."""
        result = VerificationResult(
            verified=True,
            theorem_name="TensorLogic.minimal",
        )

        assert result.verified is True
        assert result.theorem_name == "TensorLogic.minimal"
        assert result.proof_trace is None
        assert result.counterexample is None
        assert result.verification_time == 0.0
        assert result.tactic_suggestions == []
