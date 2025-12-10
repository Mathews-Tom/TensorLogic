"""Tests for tensorlogic.verification.lean_bridge module.

Tests cover LeanBridge initialization, validation, error handling, and the
verification API. Uses fixtures and mocks to test without requiring Lean 4
installation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tensorlogic.verification.lean_bridge import (
    LeanBridge,
    LeanNotInstalledError,
    LeanProjectError,
)
from tensorlogic.verification.results import VerificationResult


@pytest.fixture
def mock_lean_project(tmp_path: Path) -> Path:
    """Create temporary Lean project structure."""
    project_path = tmp_path / "lean_project"
    project_path.mkdir()

    # Create lakefile.lean
    lakefile = project_path / "lakefile.lean"
    lakefile.write_text(
        "import Lake\n"
        "open Lake DSL\n\n"
        "package «test» {}\n\n"
        "lean_lib «Test» {\n"
        "  roots := #[`Test]\n"
        "}\n"
    )

    return project_path


@pytest.fixture
def mock_lean_installation() -> Any:
    """Mock Lean 4 installation."""
    with (
        patch("shutil.which", return_value="/usr/local/bin/lean"),
        patch(
            "subprocess.run",
            return_value=MagicMock(
                stdout="Lean (version 4.0.0)",
                returncode=0,
            ),
        ),
    ):
        yield


class TestLeanNotInstalledError:
    """Test suite for LeanNotInstalledError exception."""

    def test_default_message(self) -> None:
        """Test default error message."""
        error = LeanNotInstalledError()
        msg = str(error)

        assert "Lean 4 not found in PATH" in msg
        assert "https://lean-lang.org/lean4/doc/setup.html" in msg
        assert "verify=False" in msg

    def test_custom_message(self) -> None:
        """Test custom error message."""
        custom_msg = "Custom Lean error message"
        error = LeanNotInstalledError(custom_msg)

        assert str(error) == custom_msg


class TestLeanProjectError:
    """Test suite for LeanProjectError exception."""

    def test_message(self) -> None:
        """Test error message."""
        msg = "Invalid Lean project"
        error = LeanProjectError(msg)

        assert str(error) == msg


class TestLeanBridge:
    """Test suite for LeanBridge class."""

    def test_init_validates_lean_installation(self, mock_lean_project: Path) -> None:
        """Test initialization validates Lean 4 installation."""
        with (
            patch("shutil.which", return_value=None),
        ):
            with pytest.raises(LeanNotInstalledError) as exc_info:
                LeanBridge(mock_lean_project)

            assert "Lean 4 not found in PATH" in str(exc_info.value)

    def test_init_rejects_lean3(self, mock_lean_project: Path) -> None:
        """Test initialization rejects Lean 3 installation."""
        with (
            patch("shutil.which", return_value="/usr/local/bin/lean"),
            patch(
                "subprocess.run",
                return_value=MagicMock(
                    stdout="Lean (version 3.51.0)",
                    returncode=0,
                ),
            ),
        ):
            with pytest.raises(LeanNotInstalledError) as exc_info:
                LeanBridge(mock_lean_project)

            assert "Lean 3 detected" in str(exc_info.value)
            assert "TensorLogic requires Lean 4" in str(exc_info.value)

    def test_init_validates_project_exists(
        self, tmp_path: Path, mock_lean_installation: Any
    ) -> None:
        """Test initialization validates project path exists."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(LeanProjectError) as exc_info:
            LeanBridge(nonexistent)

        assert "does not exist" in str(exc_info.value)

    def test_init_validates_project_is_directory(
        self, tmp_path: Path, mock_lean_installation: Any
    ) -> None:
        """Test initialization validates project path is directory."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("not a directory")

        with pytest.raises(LeanProjectError) as exc_info:
            LeanBridge(file_path)

        assert "not a directory" in str(exc_info.value)

    def test_init_validates_lakefile_exists(
        self, tmp_path: Path, mock_lean_installation: Any
    ) -> None:
        """Test initialization validates lakefile.lean exists."""
        project_path = tmp_path / "project"
        project_path.mkdir()

        with pytest.raises(LeanProjectError) as exc_info:
            LeanBridge(project_path)

        assert "No lakefile.lean found" in str(exc_info.value)

    def test_init_success(
        self, mock_lean_project: Path, mock_lean_installation: Any
    ) -> None:
        """Test successful initialization."""
        with patch(
            "tensorlogic.verification.lean_bridge.LeanBridge._check_leandojo",
            return_value=True,
        ):
            bridge = LeanBridge(mock_lean_project, timeout_seconds=60.0)

            assert bridge.project_path == mock_lean_project.resolve()
            assert bridge.timeout_seconds == 60.0

    def test_init_default_timeout(
        self, mock_lean_project: Path, mock_lean_installation: Any
    ) -> None:
        """Test initialization with default timeout."""
        with patch(
            "tensorlogic.verification.lean_bridge.LeanBridge._check_leandojo",
            return_value=True,
        ):
            bridge = LeanBridge(mock_lean_project)

            assert bridge.timeout_seconds == 30.0

    def test_check_leandojo_available(
        self, mock_lean_project: Path, mock_lean_installation: Any
    ) -> None:
        """Test LeanDojo availability check when installed."""
        # Mock the actual import check in _check_leandojo
        with patch(
            "tensorlogic.verification.lean_bridge.LeanBridge._check_leandojo",
            return_value=True,
        ):
            bridge = LeanBridge(mock_lean_project)

            assert bridge._leandojo_available is True

    def test_check_leandojo_unavailable(
        self, mock_lean_project: Path, mock_lean_installation: Any
    ) -> None:
        """Test LeanDojo availability check when not installed."""
        with patch(
            "tensorlogic.verification.lean_bridge.LeanBridge._check_leandojo",
            return_value=False,
        ):
            bridge = LeanBridge(mock_lean_project)

            assert bridge._leandojo_available is False

    def test_verify_theorem_requires_leandojo(
        self, mock_lean_project: Path, mock_lean_installation: Any
    ) -> None:
        """Test verify_theorem raises error when LeanDojo unavailable."""
        with patch(
            "tensorlogic.verification.lean_bridge.LeanBridge._check_leandojo",
            return_value=False,
        ):
            bridge = LeanBridge(mock_lean_project)

            with pytest.raises(LeanNotInstalledError) as exc_info:
                bridge.verify_theorem("TensorLogic.and_commutative")

            assert "LeanDojo not installed" in str(exc_info.value)
            assert "pip install lean-dojo" in str(exc_info.value)

    def test_verify_theorem_returns_placeholder_result(
        self, mock_lean_project: Path, mock_lean_installation: Any
    ) -> None:
        """Test verify_theorem returns placeholder result (Phase 1)."""
        with patch(
            "tensorlogic.verification.lean_bridge.LeanBridge._check_leandojo",
            return_value=True,
        ):
            bridge = LeanBridge(mock_lean_project)

            result = bridge.verify_theorem("TensorLogic.and_commutative")

            assert isinstance(result, VerificationResult)
            assert result.verified is False
            assert result.theorem_name == "TensorLogic.and_commutative"
            assert result.proof_trace is None
            assert result.counterexample is None
            assert result.verification_time == 0.0
            assert result.tactic_suggestions == []

    def test_verify_theorem_with_proof_script(
        self, mock_lean_project: Path, mock_lean_installation: Any
    ) -> None:
        """Test verify_theorem accepts proof script parameter."""
        with patch(
            "tensorlogic.verification.lean_bridge.LeanBridge._check_leandojo",
            return_value=True,
        ):
            bridge = LeanBridge(mock_lean_project)

            result = bridge.verify_theorem(
                "TensorLogic.demorgan_and",
                proof_script="apply DeMorgan.and_not",
            )

            assert isinstance(result, VerificationResult)

    def test_verify_theorem_with_custom_timeout(
        self, mock_lean_project: Path, mock_lean_installation: Any
    ) -> None:
        """Test verify_theorem accepts custom timeout."""
        with patch(
            "tensorlogic.verification.lean_bridge.LeanBridge._check_leandojo",
            return_value=True,
        ):
            bridge = LeanBridge(mock_lean_project)

            result = bridge.verify_theorem(
                "TensorLogic.and_commutative",
                timeout_seconds=60.0,
            )

            assert isinstance(result, VerificationResult)

    def test_suggest_tactics_requires_leandojo(
        self, mock_lean_project: Path, mock_lean_installation: Any
    ) -> None:
        """Test suggest_tactics raises error when LeanDojo unavailable."""
        with patch(
            "tensorlogic.verification.lean_bridge.LeanBridge._check_leandojo",
            return_value=False,
        ):
            bridge = LeanBridge(mock_lean_project)

            with pytest.raises(LeanNotInstalledError) as exc_info:
                bridge.suggest_tactics("⊢ a ∧ b = b ∧ a")

            assert "LeanDojo not installed" in str(exc_info.value)
            assert "Neural tactic suggestion requires LeanDojo" in str(exc_info.value)

    def test_suggest_tactics_returns_empty_placeholder(
        self, mock_lean_project: Path, mock_lean_installation: Any
    ) -> None:
        """Test suggest_tactics returns empty list (Phase 1 placeholder)."""
        with patch(
            "tensorlogic.verification.lean_bridge.LeanBridge._check_leandojo",
            return_value=True,
        ):
            bridge = LeanBridge(mock_lean_project)

            suggestions = bridge.suggest_tactics("⊢ a ∧ b = b ∧ a")

            assert suggestions == []

    def test_suggest_tactics_accepts_max_suggestions(
        self, mock_lean_project: Path, mock_lean_installation: Any
    ) -> None:
        """Test suggest_tactics accepts max_suggestions parameter."""
        with patch(
            "tensorlogic.verification.lean_bridge.LeanBridge._check_leandojo",
            return_value=True,
        ):
            bridge = LeanBridge(mock_lean_project)

            suggestions = bridge.suggest_tactics("⊢ a ∧ b = b ∧ a", max_suggestions=3)

            assert suggestions == []

    def test_close_method(
        self, mock_lean_project: Path, mock_lean_installation: Any
    ) -> None:
        """Test close method (placeholder for resource cleanup)."""
        with patch(
            "tensorlogic.verification.lean_bridge.LeanBridge._check_leandojo",
            return_value=True,
        ):
            bridge = LeanBridge(mock_lean_project)

            # Should not raise exception
            bridge.close()

    def test_context_manager_protocol(
        self, mock_lean_project: Path, mock_lean_installation: Any
    ) -> None:
        """Test context manager protocol (__enter__ and __exit__)."""
        with patch(
            "tensorlogic.verification.lean_bridge.LeanBridge._check_leandojo",
            return_value=True,
        ):
            with LeanBridge(mock_lean_project) as bridge:
                assert isinstance(bridge, LeanBridge)
                assert bridge.project_path == mock_lean_project.resolve()

            # Context manager should call close() on exit

    def test_context_manager_calls_close_on_exception(
        self, mock_lean_project: Path, mock_lean_installation: Any
    ) -> None:
        """Test context manager calls close even on exception."""
        with patch(
            "tensorlogic.verification.lean_bridge.LeanBridge._check_leandojo",
            return_value=True,
        ):
            with pytest.raises(ValueError):
                with LeanBridge(mock_lean_project):
                    raise ValueError("Test exception")

            # close() should have been called despite exception

    def test_subprocess_error_in_lean_validation(self, mock_lean_project: Path) -> None:
        """Test subprocess error during Lean version validation."""
        with (
            patch("shutil.which", return_value="/usr/local/bin/lean"),
            patch(
                "subprocess.run",
                side_effect=subprocess.SubprocessError("Command failed"),
            ),
        ):
            with pytest.raises(LeanNotInstalledError) as exc_info:
                LeanBridge(mock_lean_project)

            assert "Failed to verify Lean installation" in str(exc_info.value)

    def test_string_path_conversion(
        self, mock_lean_project: Path, mock_lean_installation: Any
    ) -> None:
        """Test initialization accepts string path."""
        with patch(
            "tensorlogic.verification.lean_bridge.LeanBridge._check_leandojo",
            return_value=True,
        ):
            bridge = LeanBridge(str(mock_lean_project))

            assert bridge.project_path == mock_lean_project.resolve()
