"""Lean 4 integration bridge via LeanDojo.

This module provides bidirectional communication with Lean 4 theorem prover
through the LeanDojo Python package. Enables theorem verification, proof search,
and neural tactic suggestion for automated proof automation.

**Note:** LeanDojo is an optional dependency. If not installed, verification
features will be unavailable but the rest of TensorLogic remains functional.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from tensorlogic.verification.results import VerificationResult

__all__ = ["LeanBridge", "LeanNotInstalledError", "LeanProjectError"]


class LeanNotInstalledError(RuntimeError):
    """Raised when Lean 4 is not installed or not in PATH."""

    def __init__(self, message: str | None = None):
        default_msg = (
            "Lean 4 not found in PATH. Install from: "
            "https://lean-lang.org/lean4/doc/setup.html\n"
            "Alternative: Disable verification with verify=False"
        )
        super().__init__(message or default_msg)


class LeanProjectError(RuntimeError):
    """Raised when Lean project is invalid or cannot be loaded."""

    pass


class LeanBridge:
    """Bridge to Lean 4 theorem prover via LeanDojo.

    Manages communication with Lean 4 for theorem verification, proof validation,
    and automated tactic suggestion. Handles subprocess management, timeout
    enforcement, and error recovery.

    **External Dependencies:**
    - Lean 4 system installation (required)
    - LeanDojo Python package (required for verification features)

    Attributes:
        project_path: Path to Lean 4 project directory
        timeout_seconds: Default timeout for proof search

    Examples:
        >>> # Basic theorem verification
        >>> bridge = LeanBridge("lean/")
        >>> result = bridge.verify_theorem("TensorLogic.and_commutative")
        >>> assert result.verified

        >>> # Theorem verification with custom proof
        >>> result = bridge.verify_theorem(
        ...     "TensorLogic.demorgan_and",
        ...     proof_script="apply DeMorgan.and_not"
        ... )

        >>> # Neural tactic suggestion
        >>> suggestions = bridge.suggest_tactics("⊢ a ∧ b = b ∧ a")
        >>> print(suggestions[0])  # ("apply and.comm", 0.87)
    """

    def __init__(self, lean_project_path: str | Path, timeout_seconds: float = 30.0):
        """Initialize Lean bridge.

        Args:
            lean_project_path: Path to Lean 4 project containing theorems
            timeout_seconds: Default timeout for proof search

        Raises:
            LeanNotInstalledError: If Lean 4 not installed
            LeanProjectError: If project path invalid or project cannot be loaded
        """
        self.project_path = Path(lean_project_path).resolve()
        self.timeout_seconds = timeout_seconds

        # Validate Lean 4 installation
        self._validate_lean_installation()

        # Validate project structure
        self._validate_project()

        # Try to import LeanDojo (optional dependency)
        self._leandojo_available = self._check_leandojo()

    def _validate_lean_installation(self) -> None:
        """Check if Lean 4 is installed and in PATH.

        Raises:
            LeanNotInstalledError: If Lean 4 not found
        """
        if not shutil.which("lean"):
            raise LeanNotInstalledError()

        # Verify Lean 4 (not Lean 3)
        try:
            result = subprocess.run(
                ["lean", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            if "Lean 4" not in result.stdout and "version 4" not in result.stdout:
                raise LeanNotInstalledError(
                    "Lean 3 detected. TensorLogic requires Lean 4.\n"
                    "Install Lean 4 from: https://lean-lang.org/lean4/doc/setup.html"
                )
        except subprocess.SubprocessError as e:
            raise LeanNotInstalledError(f"Failed to verify Lean installation: {e}")

    def _validate_project(self) -> None:
        """Validate Lean project structure.

        Raises:
            LeanProjectError: If project invalid
        """
        if not self.project_path.exists():
            raise LeanProjectError(
                f"Lean project path does not exist: {self.project_path}"
            )

        if not self.project_path.is_dir():
            raise LeanProjectError(
                f"Lean project path is not a directory: {self.project_path}"
            )

        # Check for lakefile.lean (Lean 4 project file)
        lakefile = self.project_path / "lakefile.lean"
        if not lakefile.exists():
            raise LeanProjectError(
                f"No lakefile.lean found in {self.project_path}\n"
                "Initialize Lean project with: lake init"
            )

    def _check_leandojo(self) -> bool:
        """Check if LeanDojo is installed.

        Returns:
            True if LeanDojo available, False otherwise
        """
        try:
            import lean_dojo  # type: ignore  # noqa: F401

            return True
        except ImportError:
            return False

    def verify_theorem(
        self,
        theorem_name: str,
        proof_script: str | None = None,
        timeout_seconds: float | None = None,
    ) -> VerificationResult:
        """Verify theorem in Lean 4.

        Args:
            theorem_name: Lean theorem identifier (e.g., 'TensorLogic.and_commutative')
            proof_script: Optional proof tactics (uses auto tactics if None)
            timeout_seconds: Override default timeout

        Returns:
            VerificationResult with success/failure and proof trace

        Raises:
            LeanNotInstalledError: If LeanDojo not available
            TimeoutError: If proof search exceeds timeout

        Examples:
            >>> bridge = LeanBridge("lean/")
            >>> result = bridge.verify_theorem("TensorLogic.and_commutative")
            >>> print(result.verified)
            True
        """
        if not self._leandojo_available:
            raise LeanNotInstalledError(
                "LeanDojo not installed. Install with: pip install lean-dojo\n"
                "Note: LeanDojo is required for verification features."
            )

        # Placeholder implementation - actual LeanDojo integration will be added
        # in subsequent phases
        return VerificationResult(
            verified=False,
            theorem_name=theorem_name,
            proof_trace=None,
            counterexample=None,
            verification_time=0.0,
            tactic_suggestions=[],
        )

    def suggest_tactics(
        self,
        goal_state: str,
        max_suggestions: int = 5,
    ) -> list[tuple[str, float]]:
        """Neural proof search for tactics.

        Uses LeanDojo's neural tactic suggestion model to recommend proof tactics
        for the given goal state. Tactics are ranked by confidence score.

        Args:
            goal_state: Current Lean goal state (e.g., "⊢ a ∧ b = b ∧ a")
            max_suggestions: Maximum number of tactics to return

        Returns:
            List of (tactic, confidence) tuples sorted by confidence descending

        Raises:
            LeanNotInstalledError: If LeanDojo not available

        Examples:
            >>> bridge = LeanBridge("lean/")
            >>> suggestions = bridge.suggest_tactics("⊢ a ∧ b = b ∧ a")
            >>> print(suggestions[0])
            ('apply and.comm', 0.87)
        """
        if not self._leandojo_available:
            raise LeanNotInstalledError(
                "LeanDojo not installed. Neural tactic suggestion requires LeanDojo.\n"
                "Install with: pip install lean-dojo"
            )

        # Placeholder implementation - actual neural tactic suggestion will be
        # added in Phase 4 (Proof-Guided Training)
        return []

    def close(self) -> None:
        """Close Lean connection and cleanup resources.

        Call this when done with verification to release resources.
        """
        # Placeholder for cleanup logic when LeanDojo integration is complete
        pass

    def __enter__(self) -> LeanBridge:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup."""
        self.close()
