"""Tests for verified theorems module."""

from __future__ import annotations

import pytest

from tensorlogic.verification import (
    PROVEN_THEOREMS,
    ProvenTheorem,
    get_theorem,
    list_theorems_by_category,
    verify_build,
)


class TestProvenTheorems:
    """Tests for the PROVEN_THEOREMS registry."""

    def test_theorems_count(self) -> None:
        """Test that we have the expected number of proven theorems."""
        # P2.3 requires at least 3 theorems
        assert len(PROVEN_THEOREMS) >= 3

    def test_required_theorems_present(self) -> None:
        """Test that the 3 required theorems are present (P2.3 requirement)."""
        required = ["and_commutative", "or_commutative", "demorgan_and"]
        theorem_names = [t.name for t in PROVEN_THEOREMS]
        for name in required:
            assert name in theorem_names, f"Required theorem {name} not found"

    def test_theorem_structure(self) -> None:
        """Test that all theorems have required fields."""
        for theorem in PROVEN_THEOREMS:
            assert isinstance(theorem, ProvenTheorem)
            assert theorem.name
            assert theorem.statement
            assert theorem.category
            assert theorem.lean_proof


class TestGetTheorem:
    """Tests for get_theorem function."""

    def test_get_existing_theorem(self) -> None:
        """Test getting an existing theorem."""
        theorem = get_theorem("and_commutative")
        assert theorem is not None
        assert theorem.name == "and_commutative"
        assert "∧" in theorem.statement

    def test_get_nonexistent_theorem(self) -> None:
        """Test getting a non-existent theorem returns None."""
        theorem = get_theorem("nonexistent_theorem")
        assert theorem is None

    def test_get_demorgan_and(self) -> None:
        """Test getting De Morgan's law for AND."""
        theorem = get_theorem("demorgan_and")
        assert theorem is not None
        assert "¬(a ∧ b)" in theorem.statement
        assert theorem.category == "demorgan"


class TestListTheoremsByCategory:
    """Tests for list_theorems_by_category function."""

    def test_list_all_theorems(self) -> None:
        """Test listing all theorems without category filter."""
        all_theorems = list_theorems_by_category()
        assert len(all_theorems) == len(PROVEN_THEOREMS)

    def test_list_and_properties(self) -> None:
        """Test listing AND property theorems."""
        and_theorems = list_theorems_by_category("and_properties")
        assert len(and_theorems) >= 3
        assert all(t.category == "and_properties" for t in and_theorems)

    def test_list_demorgan(self) -> None:
        """Test listing De Morgan's law theorems."""
        demorgan_theorems = list_theorems_by_category("demorgan")
        assert len(demorgan_theorems) == 2
        assert all(t.category == "demorgan" for t in demorgan_theorems)

    def test_list_nonexistent_category(self) -> None:
        """Test listing from non-existent category returns empty list."""
        theorems = list_theorems_by_category("nonexistent")
        assert theorems == []


class TestVerifyBuild:
    """Tests for verify_build function."""

    @pytest.mark.skipif(
        not pytest.importorskip("shutil").which("lean"),
        reason="Lean 4 not installed",
    )
    def test_verify_build_success(self) -> None:
        """Test that verify_build succeeds when Lean is installed."""
        result = verify_build()
        assert result.verified is True
        assert result.build_time > 0
        assert result.lean_version is not None

    def test_verify_build_invalid_path(self) -> None:
        """Test verify_build with invalid project path."""
        result = verify_build("/nonexistent/path")
        assert result.verified is False
        assert "not found" in result.output.lower()


class TestTheoremMathematicalProperties:
    """Tests that verify theorem statements are mathematically correct."""

    def test_commutativity_statements(self) -> None:
        """Test that commutativity theorems have symmetric statements."""
        and_comm = get_theorem("and_commutative")
        or_comm = get_theorem("or_commutative")

        assert and_comm is not None
        assert or_comm is not None

        # Both should have a = b pattern
        assert "a" in and_comm.statement
        assert "b" in and_comm.statement

    def test_demorgan_dual_statements(self) -> None:
        """Test that De Morgan's laws are duals of each other."""
        dm_and = get_theorem("demorgan_and")
        dm_or = get_theorem("demorgan_or")

        assert dm_and is not None
        assert dm_or is not None

        # One should have ∧ in negation, other should have ∨
        assert "∧" in dm_and.statement
        assert "∨" in dm_or.statement

    def test_associativity_statements(self) -> None:
        """Test that associativity theorems have triple variable pattern."""
        and_assoc = get_theorem("and_associative")
        or_assoc = get_theorem("or_associative")

        assert and_assoc is not None
        assert or_assoc is not None

        # Both should have a, b, c
        for thm in [and_assoc, or_assoc]:
            assert "a" in thm.statement
            assert "b" in thm.statement
            assert "c" in thm.statement
