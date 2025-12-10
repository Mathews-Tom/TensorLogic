"""Factory function for creating compilation strategies by name.

This module provides a simple factory pattern for instantiating compilation strategies
without requiring direct imports of strategy classes. Enables runtime strategy selection
and easier testing.
"""

from __future__ import annotations

from typing import Any

from tensorlogic.compilation.protocol import CompilationStrategy


# Strategy registry mapping names to strategy classes
# Uses type[Any] to allow flexible __init__ signatures
_STRATEGY_REGISTRY: dict[str, type[Any]] = {}


def register_strategy(name: str, strategy_class: type[CompilationStrategy]) -> None:
    """Register a compilation strategy in the factory registry.

    This function is used internally by strategy implementations to register themselves
    with the factory. Users typically don't need to call this directly.

    Args:
        name: Strategy identifier (e.g., "soft_differentiable", "hard_boolean")
        strategy_class: Strategy class implementing CompilationStrategy protocol

    Raises:
        ValueError: If strategy name already registered

    Example:
        >>> class MySoftStrategy:
        ...     @property
        ...     def name(self) -> str:
        ...         return "soft_differentiable"
        ...     # ... implement other protocol methods
        >>> register_strategy("soft_differentiable", MySoftStrategy)
    """
    if name in _STRATEGY_REGISTRY:
        raise ValueError(
            f"Strategy '{name}' is already registered. "
            f"Use a different name or unregister the existing strategy first."
        )
    _STRATEGY_REGISTRY[name] = strategy_class


def unregister_strategy(name: str) -> None:
    """Remove a strategy from the factory registry.

    Args:
        name: Strategy identifier to unregister

    Raises:
        KeyError: If strategy name not found in registry

    Example:
        >>> unregister_strategy("soft_differentiable")
    """
    if name not in _STRATEGY_REGISTRY:
        raise KeyError(f"Strategy '{name}' not found in registry")
    del _STRATEGY_REGISTRY[name]


def get_available_strategies() -> list[str]:
    """Get list of all registered strategy names.

    Returns:
        List of strategy names currently available in registry

    Example:
        >>> strategies = get_available_strategies()
        >>> print(strategies)
        ['soft_differentiable', 'hard_boolean', 'godel', 'product', 'lukasiewicz']
    """
    return list(_STRATEGY_REGISTRY.keys())


def create_strategy(name: str = "soft_differentiable", backend: Any = None) -> Any:
    """Create compilation strategy by name.

    Factory function for instantiating compilation strategies without direct imports.
    Enables runtime strategy selection based on string identifiers.

    Args:
        name: Strategy identifier (default: "soft_differentiable")
            Available strategies:
            - "soft_differentiable": Continuous, gradient-friendly (default)
            - "hard_boolean": Discrete, exact boolean logic
            - "godel": Gödel fuzzy logic (min/max)
            - "product": Product fuzzy logic
            - "lukasiewicz": Łukasiewicz fuzzy logic
        backend: Optional TensorBackend instance to use for operations
            If None, strategy creates its own default NumPy backend

    Returns:
        Compilation strategy instance implementing CompilationStrategy protocol

    Raises:
        ValueError: If strategy name not recognized

    Example:
        >>> # Using default strategy
        >>> strategy = create_strategy()
        >>> assert strategy.name == "soft_differentiable"
        >>>
        >>> # Explicit strategy selection with backend
        >>> from tensorlogic.backends import create_backend
        >>> backend = create_backend("mlx")
        >>> strategy = create_strategy("hard_boolean", backend=backend)
        >>> result = strategy.compile_and(a, b)
        >>>
        >>> # Check available strategies
        >>> strategies = get_available_strategies()
        >>> if "godel" in strategies:
        ...     strategy = create_strategy("godel")
    """
    if name not in _STRATEGY_REGISTRY:
        available = get_available_strategies()
        if available:
            available_list = ", ".join(f"'{s}'" for s in sorted(available))
            raise ValueError(
                f"Unknown compilation strategy: '{name}'. "
                f"Available strategies: {available_list}. "
                f"Did you mean one of these?"
            )
        else:
            raise ValueError(
                f"Unknown compilation strategy: '{name}'. "
                f"No strategies are currently registered."
            )

    strategy_class = _STRATEGY_REGISTRY[name]
    return strategy_class(backend=backend)


__all__ = [
    "create_strategy",
    "register_strategy",
    "unregister_strategy",
    "get_available_strategies",
]
