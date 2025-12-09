"""Tests for compilation strategy factory pattern.

Tests factory function, strategy registry, and error handling for unknown strategies.
"""

from __future__ import annotations

from typing import Any

import pytest

from tensorlogic.compilation import (
    CompilationStrategy,
    create_strategy,
    get_available_strategies,
    register_strategy,
    unregister_strategy,
)


@pytest.fixture(autouse=True)
def preserve_registry():
    """Preserve and restore strategy registry after each test."""
    from tensorlogic.compilation.factory import _STRATEGY_REGISTRY

    # Save current registry
    saved_registry = _STRATEGY_REGISTRY.copy()

    yield

    # Restore original registry
    _STRATEGY_REGISTRY.clear()
    _STRATEGY_REGISTRY.update(saved_registry)


# Mock strategy for testing
class MockStrategy:
    """Mock compilation strategy for factory tests."""

    def __init__(self, backend: Any = None) -> None:
        """Initialize mock strategy (backend parameter accepted but ignored)."""
        self.backend = backend

    def compile_and(self, a: Any, b: Any) -> Any:
        return None

    def compile_or(self, a: Any, b: Any) -> Any:
        return None

    def compile_not(self, a: Any) -> Any:
        return None

    def compile_implies(self, a: Any, b: Any) -> Any:
        return None

    def compile_exists(self, predicate: Any, axis: int) -> Any:
        return None

    def compile_forall(self, predicate: Any, axis: int) -> Any:
        return None

    @property
    def is_differentiable(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "mock_strategy"


class TestFactoryRegistration:
    """Test strategy registration and unregistration."""

    def test_register_strategy(self) -> None:
        """Test registering a new strategy."""
        register_strategy("test_strategy", MockStrategy)
        assert "test_strategy" in get_available_strategies()

        # Cleanup
        unregister_strategy("test_strategy")

    def test_register_duplicate_strategy_raises_error(self) -> None:
        """Test that registering duplicate strategy names raises ValueError."""
        register_strategy("test_dup", MockStrategy)

        with pytest.raises(ValueError, match="already registered"):
            register_strategy("test_dup", MockStrategy)

        # Cleanup
        unregister_strategy("test_dup")

    def test_unregister_strategy(self) -> None:
        """Test unregistering a strategy."""
        register_strategy("test_unreg", MockStrategy)
        assert "test_unreg" in get_available_strategies()

        unregister_strategy("test_unreg")
        assert "test_unreg" not in get_available_strategies()

    def test_unregister_unknown_strategy_raises_error(self) -> None:
        """Test that unregistering unknown strategy raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            unregister_strategy("nonexistent_strategy")


class TestGetAvailableStrategies:
    """Test querying available strategies."""

    def test_get_available_strategies_empty(self) -> None:
        """Test getting available strategies when none registered."""
        strategies = get_available_strategies()
        assert isinstance(strategies, list)
        # May be empty or contain strategies from other tests

    def test_get_available_strategies_with_registered(self) -> None:
        """Test getting available strategies after registration."""
        register_strategy("test_available", MockStrategy)

        strategies = get_available_strategies()
        assert "test_available" in strategies

        # Cleanup
        unregister_strategy("test_available")


class TestCreateStrategy:
    """Test strategy creation via factory."""

    def test_create_strategy_with_unknown_name_raises_error(self) -> None:
        """Test that unknown strategy name raises helpful ValueError."""
        with pytest.raises(ValueError, match="Unknown compilation strategy"):
            create_strategy("nonexistent_strategy")

    def test_create_strategy_error_message_shows_available(self) -> None:
        """Test that error message lists available strategies."""
        register_strategy("test_listed", MockStrategy)

        with pytest.raises(ValueError, match="test_listed"):
            create_strategy("wrong_name")

        # Cleanup
        unregister_strategy("test_listed")

    def test_create_strategy_with_no_registered_strategies(self) -> None:
        """Test error message when no strategies are registered."""
        # Ensure registry is empty for this test
        available = get_available_strategies()
        for name in available:
            unregister_strategy(name)

        with pytest.raises(
            ValueError, match="No strategies are currently registered"
        ):
            create_strategy("anything")

    def test_create_strategy_success(self) -> None:
        """Test successful strategy creation."""
        register_strategy("test_create", MockStrategy)

        strategy = create_strategy("test_create")
        assert isinstance(strategy, MockStrategy)
        assert strategy.name == "mock_strategy"

        # Cleanup
        unregister_strategy("test_create")

    def test_create_strategy_default_parameter(self) -> None:
        """Test that create_strategy has default parameter."""
        # Default should be "soft_differentiable" (now implemented)
        strategy = create_strategy()
        assert strategy.name == "soft_differentiable"
        assert isinstance(strategy, CompilationStrategy)

    def test_create_strategy_returns_protocol_instance(self) -> None:
        """Test that created strategy implements CompilationStrategy protocol."""
        register_strategy("test_protocol", MockStrategy)

        strategy = create_strategy("test_protocol")

        # Check protocol methods exist
        assert hasattr(strategy, "compile_and")
        assert hasattr(strategy, "compile_or")
        assert hasattr(strategy, "compile_not")
        assert hasattr(strategy, "compile_implies")
        assert hasattr(strategy, "compile_exists")
        assert hasattr(strategy, "compile_forall")
        assert hasattr(strategy, "is_differentiable")
        assert hasattr(strategy, "name")

        # Verify it's a runtime instance of the protocol
        assert isinstance(strategy, CompilationStrategy)

        # Cleanup
        unregister_strategy("test_protocol")


class TestFactoryIntegration:
    """Integration tests for factory pattern."""

    def test_multiple_strategy_registration(self) -> None:
        """Test registering multiple strategies."""

        class Strategy1(MockStrategy):
            @property
            def name(self) -> str:
                return "strategy1"

        class Strategy2(MockStrategy):
            @property
            def name(self) -> str:
                return "strategy2"

        register_strategy("strat1", Strategy1)
        register_strategy("strat2", Strategy2)

        available = get_available_strategies()
        assert "strat1" in available
        assert "strat2" in available

        # Verify both can be created
        s1 = create_strategy("strat1")
        s2 = create_strategy("strat2")
        assert s1.name == "strategy1"
        assert s2.name == "strategy2"

        # Cleanup
        unregister_strategy("strat1")
        unregister_strategy("strat2")

    def test_factory_isolation(self) -> None:
        """Test that factory registry is isolated per test."""
        # Register a strategy
        register_strategy("isolation_test", MockStrategy)

        # Create instance
        strategy = create_strategy("isolation_test")
        assert strategy.name == "mock_strategy"

        # Cleanup
        unregister_strategy("isolation_test")

        # Verify it's gone
        assert "isolation_test" not in get_available_strategies()
