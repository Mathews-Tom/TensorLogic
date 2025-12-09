"""Tests for backend factory and validation.

Tests factory pattern with graceful fallback, runtime validation, and error handling.
Verifies protocol compliance checking and helpful error messages.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pytest

from tensorlogic.backends import TensorBackend, NumpyBackend, create_backend, validate_backend


class TestCreateBackend:
    """Tests for create_backend factory function."""

    def test_create_numpy_backend(self) -> None:
        """Test explicit NumPy backend creation."""
        backend = create_backend("numpy")
        assert isinstance(backend, NumpyBackend)
        assert isinstance(backend, TensorBackend)

    def test_create_backend_unknown_raises_value_error(self) -> None:
        """Test unknown backend name raises ValueError with helpful message."""
        with pytest.raises(ValueError, match=r"Unknown backend: 'pytorch'"):
            create_backend("pytorch")

        with pytest.raises(ValueError, match=r"Available backends:.*'mlx'.*'numpy'"):
            create_backend("jax")

    @patch("tensorlogic.backends.factory.warnings.warn")
    def test_mlx_fallback_to_numpy_on_import_error(self, mock_warn: Mock) -> None:
        """Test graceful fallback from MLX to NumPy when MLX unavailable."""
        # Patch the import statement inside create_backend to raise ImportError for MLX
        import builtins
        original_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "tensorlogic.backends.mlx":
                raise ImportError("No module named 'mlx'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            backend = create_backend("mlx")

            # Should fallback to NumPy
            assert isinstance(backend, NumpyBackend)

            # Should warn about fallback
            mock_warn.assert_called_once()
            warning_msg = mock_warn.call_args[0][0]
            assert "MLX backend unavailable" in warning_msg
            assert "falling back to NumPy" in warning_msg
            assert "uv add mlx>=0.30.0" in warning_msg

    def test_default_backend_attempts_mlx(self) -> None:
        """Test default parameter attempts MLX backend."""
        # Since MLX is not available in test env, should fallback to NumPy
        with patch("tensorlogic.backends.factory.warnings.warn") as mock_warn:
            backend = create_backend()  # Default to "mlx"

            # Either MLX worked (unlikely in test env) or fell back to NumPy
            assert isinstance(backend, TensorBackend)

            # If MLX not available, warning should have been issued
            if isinstance(backend, NumpyBackend):
                assert mock_warn.called

    def test_numpy_import_error_raises_value_error(self) -> None:
        """Test ImportError for NumPy raises ValueError with installation hint."""
        # Patch the import to raise ImportError for numpy backend
        import builtins
        original_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "tensorlogic.backends.numpy":
                raise ImportError("No module named 'numpy'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ValueError, match=r"NumPy backend unavailable"):
                create_backend("numpy")

            with pytest.raises(ValueError, match=r"uv add numpy>=1.24.0"):
                create_backend("numpy")

    def test_backend_validated_before_return(self) -> None:
        """Test backend is validated against protocol before return."""
        # Create incomplete backend that will fail validation
        class IncompleteBackend:
            def einsum(self, pattern: str, *tensors: Any) -> Any:
                pass

        # Patch the NumpyBackend class in the numpy module before it's imported
        import tensorlogic.backends.numpy as numpy_module
        original_backend = numpy_module.NumpyBackend

        try:
            # Replace NumpyBackend with IncompleteBackend
            numpy_module.NumpyBackend = IncompleteBackend  # type: ignore

            with pytest.raises(TypeError, match=r"doesn't implement TensorBackend protocol"):
                create_backend("numpy")
        finally:
            # Restore original
            numpy_module.NumpyBackend = original_backend


class TestValidateBackend:
    """Tests for validate_backend validation function."""

    def test_valid_backend_passes(self) -> None:
        """Test valid backend passes validation without error."""
        backend = NumpyBackend()
        validate_backend(backend)  # Should not raise

    def test_incomplete_backend_raises_type_error(self) -> None:
        """Test incomplete backend implementation raises TypeError."""
        class IncompleteBackend:
            """Backend missing most protocol methods."""
            def einsum(self, pattern: str, *tensors: Any) -> Any:
                pass

        incomplete = IncompleteBackend()
        with pytest.raises(TypeError, match=r"doesn't implement TensorBackend protocol"):
            validate_backend(incomplete)

    def test_validation_error_message_helpful(self) -> None:
        """Test validation error message includes helpful context."""
        class EmptyBackend:
            """Backend with no methods."""
            pass

        empty = EmptyBackend()
        with pytest.raises(TypeError) as exc_info:
            validate_backend(empty)

        error_msg = str(exc_info.value)
        assert "EmptyBackend" in error_msg
        assert "doesn't implement TensorBackend protocol" in error_msg
        assert "Missing required operations" in error_msg

    def test_non_backend_object_raises_type_error(self) -> None:
        """Test arbitrary object raises TypeError."""
        not_a_backend = "just a string"
        with pytest.raises(TypeError, match=r"doesn't implement TensorBackend protocol"):
            validate_backend(not_a_backend)

        not_a_backend_int = 42
        with pytest.raises(TypeError):
            validate_backend(not_a_backend_int)

    def test_numpy_backend_is_valid(self) -> None:
        """Test NumpyBackend passes protocol validation."""
        backend = NumpyBackend()
        assert isinstance(backend, TensorBackend)
        validate_backend(backend)  # Should not raise


class TestFactoryIntegration:
    """Integration tests for factory pattern."""

    def test_factory_returns_working_backend(self) -> None:
        """Test factory returns backend that can perform operations."""
        backend = create_backend("numpy")

        # Test basic operations work
        zeros = backend.zeros((2, 3))
        assert zeros.shape == (2, 3)

        ones = backend.ones((2, 3))
        result = backend.add(zeros, ones)
        assert result.shape == (2, 3)

    def test_factory_backend_conforms_to_protocol(self) -> None:
        """Test factory-created backend implements all protocol methods."""
        backend = create_backend("numpy")

        # Verify protocol methods exist
        assert hasattr(backend, "einsum")
        assert hasattr(backend, "zeros")
        assert hasattr(backend, "ones")
        assert hasattr(backend, "arange")
        assert hasattr(backend, "reshape")
        assert hasattr(backend, "step")
        assert hasattr(backend, "maximum")
        assert hasattr(backend, "minimum")
        assert hasattr(backend, "multiply")
        assert hasattr(backend, "add")
        assert hasattr(backend, "subtract")
        assert hasattr(backend, "sum")
        assert hasattr(backend, "prod")
        assert hasattr(backend, "any")
        assert hasattr(backend, "all")
        assert hasattr(backend, "grad")
        assert hasattr(backend, "eval")
        assert hasattr(backend, "compile")
        assert hasattr(backend, "where")
        assert hasattr(backend, "expand_dims")
        assert hasattr(backend, "squeeze")
        assert hasattr(backend, "transpose")
        assert hasattr(backend, "concatenate")

    def test_multiple_backends_independent(self) -> None:
        """Test creating multiple backends returns independent instances."""
        backend1 = create_backend("numpy")
        backend2 = create_backend("numpy")

        # Should be different instances
        assert backend1 is not backend2

        # But both should be valid
        validate_backend(backend1)
        validate_backend(backend2)


def test_factory_exports_in_init() -> None:
    """Test factory functions exported from backends package."""
    from tensorlogic.backends import create_backend, validate_backend

    # Should be importable
    assert callable(create_backend)
    assert callable(validate_backend)
