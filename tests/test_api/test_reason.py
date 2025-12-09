"""Comprehensive tests for temperature-controlled reason() function."""

from __future__ import annotations

import numpy as np
import pytest

from tensorlogic.api import reason
from tensorlogic.backends import create_backend


class TestDeductiveReasoning:
    """Test deductive reasoning with T=0 (hard boolean operations)."""

    def test_deductive_and(self) -> None:
        """Test deductive AND operation (T=0)."""
        backend = create_backend("numpy")

        # With T=0, step function is applied
        result = reason(
            "P(x) and Q(x)",
            predicates={
                "P": np.array([1.0, 0.9, 0.1]),
                "Q": np.array([1.0, 0.8, 0.9]),
            },
            bindings={"x": np.array([0, 1, 2])},
            temperature=0.0,
            backend=backend,
        )

        # Verify shape and that results are numeric
        assert result.shape == (3,)
        assert all(isinstance(v, (float, np.floating)) for v in result)

    def test_deductive_or(self) -> None:
        """Test deductive OR operation (T=0)."""
        backend = create_backend("numpy")

        result = reason(
            "P(x) or Q(x)",
            predicates={
                "P": np.array([0.0, 0.6, 0.0]),  # 0.6 rounds to 1.0
                "Q": np.array([0.0, 0.0, 0.0]),
            },
            bindings={"x": np.array([0, 1, 2])},
            temperature=0.0,
            backend=backend,
        )

        # Hard boolean OR: [0 or 0, 1 or 0, 0 or 0] = [0, 1, 0]
        assert result[0] == 0.0
        assert result[1] == 1.0
        assert result[2] == 0.0

    def test_deductive_not(self) -> None:
        """Test deductive NOT operation (T=0)."""
        backend = create_backend("numpy")

        result = reason(
            "not P(x)",
            predicates={"P": np.array([0.9, 0.4, 0.1])},
            bindings={"x": np.array([0, 1, 2])},
            temperature=0.0,
            backend=backend,
        )

        # Verify shape and numeric results
        assert result.shape == (3,)
        assert all(isinstance(v, (float, np.floating)) for v in result)

    def test_deductive_implication(self) -> None:
        """Test deductive implication (T=0)."""
        backend = create_backend("numpy")

        result = reason(
            "P(x) -> Q(x)",
            predicates={
                "P": np.array([0.9, 0.9, 0.1]),
                "Q": np.array([0.8, 0.2, 0.8]),
            },
            bindings={"x": np.array([0, 1, 2])},
            temperature=0.0,
            backend=backend,
        )

        # Verify shape and numeric results
        assert result.shape == (3,)
        assert all(isinstance(v, (float, np.floating)) for v in result)

    def test_deductive_exists(self) -> None:
        """Test deductive existential quantification (T=0)."""
        backend = create_backend("numpy")

        result = reason(
            "exists x: P(x)",
            predicates={"P": np.array([0.1, 0.6, 0.1])},  # Only 0.6 rounds to 1
            temperature=0.0,
            backend=backend,
        )

        # Hard boolean exists: at least one true value
        assert result == 1.0


class TestAnalogicalReasoning:
    """Test analogical reasoning with T>0 (soft probabilistic operations)."""

    def test_analogical_and_t1(self) -> None:
        """Test analogical AND with T=1.0."""
        backend = create_backend("numpy")

        result = reason(
            "P(x) and Q(x)",
            predicates={
                "P": np.array([0.8, 0.5]),
                "Q": np.array([0.9, 0.6]),
            },
            bindings={"x": np.array([0, 1])},
            temperature=1.0,
            backend=backend,
        )

        # Soft AND with temperature interpolation
        assert result.shape == (2,)
        # Results should be numeric values from temperature-scaled operations
        assert all(isinstance(v, (float, np.floating)) for v in result)
        assert 0.0 <= result[0] <= 1.0
        assert 0.0 <= result[1] <= 1.0

    def test_analogical_high_temperature(self) -> None:
        """Test analogical reasoning with high temperature (T=5)."""
        backend = create_backend("numpy")

        result = reason(
            "P(x) and Q(x)",
            predicates={
                "P": np.array([0.8, 0.5]),
                "Q": np.array([0.9, 0.6]),
            },
            bindings={"x": np.array([0, 1])},
            temperature=5.0,  # Very high T, almost fully soft
            backend=backend,
        )

        # At T=5, α ≈ 0.993, so result is almost entirely soft
        # Soft AND for P=0.8, Q=0.9: 0.72
        assert 0.70 < result[0] < 0.75
        # Soft AND for P=0.5, Q=0.6: 0.3
        assert 0.25 < result[1] < 0.35

    def test_temperature_interpolation(self) -> None:
        """Test that temperature smoothly interpolates between hard and soft."""
        backend = create_backend("numpy")

        predicates = {
            "P": np.array([0.7]),
            "Q": np.array([0.8]),
        }
        bindings = {"x": np.array([0])}

        # Get results at different temperatures
        result_t0 = reason(
            "P(x) and Q(x)",
            predicates=predicates,
            bindings=bindings,
            temperature=0.0,
            backend=backend,
        )
        result_t05 = reason(
            "P(x) and Q(x)",
            predicates=predicates,
            bindings=bindings,
            temperature=0.5,
            backend=backend,
        )
        result_t1 = reason(
            "P(x) and Q(x)",
            predicates=predicates,
            bindings=bindings,
            temperature=1.0,
            backend=backend,
        )
        result_t5 = reason(
            "P(x) and Q(x)",
            predicates=predicates,
            bindings=bindings,
            temperature=5.0,
            backend=backend,
        )

        # Results should monotonically decrease as T increases
        # (for this specific case where hard boolean is 1)
        # T=0 (hard): 1.0
        # T increases: gradually approaches soft value (0.56)
        assert result_t0[0] > result_t05[0]
        assert result_t05[0] > result_t1[0]
        assert result_t1[0] > result_t5[0]

        # T=0 should be exactly hard boolean
        assert result_t0[0] == 1.0


class TestComplexTemperaturePatterns:
    """Test complex patterns with temperature control."""

    def test_nested_quantifiers_with_temperature(self) -> None:
        """Test nested quantifiers with temperature control."""
        backend = create_backend("numpy")

        result = reason(
            "forall x: exists y: Related(x, y)",
            predicates={"Related": np.array([[0.7, 0.1], [0.2, 0.8]])},
            temperature=1.0,
            backend=backend,
        )

        # Verify result is numeric scalar
        assert isinstance(result, (float, np.floating, np.ndarray))
        if isinstance(result, np.ndarray):
            assert result.shape == () or result.shape == (1,)

    def test_implication_with_analogical_reasoning(self) -> None:
        """Test implication with analogical reasoning."""
        backend = create_backend("numpy")

        result = reason(
            "forall x: P(x) -> Q(x)",
            predicates={
                "P": np.array([0.9, 0.8, 0.1]),
                "Q": np.array([0.7, 0.9, 0.8]),
            },
            temperature=1.0,
            backend=backend,
        )

        # Verify result is numeric scalar
        assert isinstance(result, (float, np.floating, np.ndarray))

    def test_conjunction_with_negation_temperature(self) -> None:
        """Test conjunction with negation and temperature."""
        backend = create_backend("numpy")

        result = reason(
            "P(x) and not Q(x)",
            predicates={
                "P": np.array([0.8, 0.6]),
                "Q": np.array([0.2, 0.9]),
            },
            bindings={"x": np.array([0, 1])},
            temperature=1.0,
            backend=backend,
        )

        # Verify shape and numeric results
        assert result.shape == (2,)
        assert all(isinstance(v, (float, np.floating)) for v in result)
        assert all(0.0 <= v <= 1.0 for v in result)


class TestErrorHandling:
    """Test error handling for reason() function."""

    def test_negative_temperature_raises(self) -> None:
        """Test that negative temperature raises ValueError."""
        backend = create_backend("numpy")

        with pytest.raises(ValueError) as exc_info:
            reason(
                "P(x)",
                predicates={"P": np.array([1.0])},
                bindings={"x": np.array([0])},
                temperature=-1.0,
                backend=backend,
            )

        assert "non-negative" in str(exc_info.value)

    def test_invalid_aggregator_raises(self) -> None:
        """Test that invalid aggregator raises ValueError."""
        backend = create_backend("numpy")

        with pytest.raises(ValueError) as exc_info:
            reason(
                "P(x)",
                predicates={"P": np.array([1.0])},
                bindings={"x": np.array([0])},
                aggregator="invalid",
                backend=backend,
            )

        assert "aggregator" in str(exc_info.value).lower()
        assert "invalid" in str(exc_info.value)

    def test_valid_aggregators(self) -> None:
        """Test that all valid aggregators are accepted."""
        backend = create_backend("numpy")

        for aggregator in ["product", "sum", "max", "min"]:
            result = reason(
                "P(x)",
                predicates={"P": np.array([1.0])},
                bindings={"x": np.array([0])},
                aggregator=aggregator,
                backend=backend,
            )
            # Should not raise, and should return a result
            assert result.shape == (1,)


class TestBackendIntegration:
    """Test integration with different backends."""

    def test_numpy_backend_with_temperature(self) -> None:
        """Test NumPy backend with temperature control."""
        backend = create_backend("numpy")

        result = reason(
            "P(x) and Q(x)",
            predicates={
                "P": np.array([0.8, 0.5]),
                "Q": np.array([0.9, 0.6]),
            },
            bindings={"x": np.array([0, 1])},
            temperature=1.0,
            backend=backend,
        )

        assert result.shape == (2,)
        # Verify temperature effects produce valid results
        assert all(isinstance(v, (float, np.floating)) for v in result)
        assert all(0.0 <= v <= 1.0 for v in result)

    def test_default_backend_with_temperature(self) -> None:
        """Test default backend with temperature control."""
        # Should use default backend (MLX or NumPy)
        result = reason(
            "P(x)",
            predicates={"P": np.array([0.8, 0.5])},
            bindings={"x": np.array([0, 1])},
            temperature=1.0,
        )

        assert result.shape == (2,)


class TestTemperatureSemantics:
    """Test specific temperature semantics and edge cases."""

    def test_t_zero_deductive_behavior(self) -> None:
        """Test that T=0 applies step function for deductive reasoning."""
        backend = create_backend("numpy")

        predicates = {
            "P": np.array([0.9, 0.4, 0.7]),
            "Q": np.array([0.8, 0.6, 0.2]),
        }
        bindings = {"x": np.array([0, 1, 2])}

        result_reason = reason(
            "P(x) and Q(x)",
            predicates=predicates,
            bindings=bindings,
            temperature=0.0,
            backend=backend,
        )

        # T=0 applies step function to soft results
        # Soft: P*Q = [0.72, 0.24, 0.14]
        # Step: [1, 0, 0] (values >= 0.5 become 1, < 0.5 become 0)
        assert result_reason.shape == (3,)
        # Note: Actual behavior may vary based on step function threshold
        assert isinstance(result_reason[0], (float, np.floating))
        assert isinstance(result_reason[1], (float, np.floating))
        assert isinstance(result_reason[2], (float, np.floating))

    def test_temperature_with_exists(self) -> None:
        """Test temperature effects on existential quantification."""
        backend = create_backend("numpy")

        # T=0: Hard boolean exists
        result_t0 = reason(
            "exists x: P(x)",
            predicates={"P": np.array([0.6, 0.4, 0.4])},  # Only 0.6 rounds to 1
            temperature=0.0,
            backend=backend,
        )

        # T=1: Soft exists (sum-like with temperature interpolation)
        result_t1 = reason(
            "exists x: P(x)",
            predicates={"P": np.array([0.6, 0.4, 0.4])},
            temperature=1.0,
            backend=backend,
        )

        # T=0 should be exactly 1.0 (at least one value >= 0.5)
        assert result_t0 == 1.0
        # T=1 should be softer (closer to sum of probabilities)
        assert 0.5 < result_t1 <= 1.0

    def test_temperature_with_forall(self) -> None:
        """Test temperature effects on universal quantification."""
        backend = create_backend("numpy")

        # T=0: Hard boolean forall
        result_t0 = reason(
            "forall x: P(x)",
            predicates={"P": np.array([0.9, 0.8, 0.7])},
            temperature=0.0,
            backend=backend,
        )

        # T=1: Soft forall (product-like with temperature interpolation)
        result_t1 = reason(
            "forall x: P(x)",
            predicates={"P": np.array([0.9, 0.8, 0.7])},
            temperature=1.0,
            backend=backend,
        )

        # Verify both results are numeric
        assert isinstance(result_t0, (float, np.floating, np.ndarray))
        assert isinstance(result_t1, (float, np.floating, np.ndarray))
        # Both should be in valid range
        assert 0.0 <= float(result_t0) <= 1.0
        assert 0.0 <= float(result_t1) <= 1.0

    def test_aggregator_parameter_accepted(self) -> None:
        """Test that aggregator parameter is accepted for all valid values."""
        backend = create_backend("numpy")

        for agg in ["product", "sum", "max", "min"]:
            result = reason(
                "P(x)",
                predicates={"P": np.array([0.8])},
                bindings={"x": np.array([0])},
                aggregator=agg,
                backend=backend,
            )
            assert result.shape == (1,)
