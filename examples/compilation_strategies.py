"""Compilation Strategies Usage Examples

This module demonstrates practical usage of TensorLogic's compilation strategies,
showing when and how to use each strategy variant (soft, hard, fuzzy) for different
neural-symbolic reasoning tasks.

Run examples:
    uv run python examples/compilation_strategies.py
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tensorlogic.backends import create_backend
from tensorlogic.compilation import create_strategy, get_available_strategies


def example_basic_strategy_usage() -> None:
    """Basic usage: Creating and using compilation strategies."""
    print("=" * 80)
    print("Example 1: Basic Strategy Usage")
    print("=" * 80)

    # Create backend
    backend = create_backend("mlx")

    # Create default strategy (soft_differentiable)
    strategy = create_strategy()
    print(f"Default strategy: {strategy.name}")
    print(f"Differentiable: {strategy.is_differentiable}\n")

    # Create test data
    a = np.array([0.8, 0.6, 0.3])
    b = np.array([0.9, 0.4, 0.7])

    # Logical AND
    result_and = strategy.compile_and(a, b)
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"AND(a, b) = {result_and} (product: a * b)\n")

    # Logical OR
    result_or = strategy.compile_or(a, b)
    print(f"OR(a, b) = {result_or} (probabilistic sum: a + b - a*b)\n")

    # Logical NOT
    result_not = strategy.compile_not(a)
    print(f"NOT(a) = {result_not} (complement: 1 - a)\n")

    # Logical IMPLIES
    result_implies = strategy.compile_implies(a, b)
    print(f"IMPLIES(a, b) = {result_implies} (max(1 - a, b))\n")


def example_compare_strategies() -> None:
    """Compare all strategies on the same input."""
    print("=" * 80)
    print("Example 2: Strategy Comparison")
    print("=" * 80)

    backend = create_backend("mlx")
    a = np.array([0.8, 0.6, 0.3])
    b = np.array([0.9, 0.4, 0.7])

    print(f"Input: a = {a}")
    print(f"       b = {b}\n")

    # Compare AND operation across all strategies
    print("AND Operation Results:")
    print("-" * 80)
    for strategy_name in get_available_strategies():
        strategy = create_strategy(strategy_name, backend=backend)
        result = strategy.compile_and(a, b)
        print(f"{strategy_name:20s}: {result}")
    print()

    # Compare OR operation across all strategies
    print("OR Operation Results:")
    print("-" * 80)
    for strategy_name in get_available_strategies():
        strategy = create_strategy(strategy_name, backend=backend)
        result = strategy.compile_or(a, b)
        print(f"{strategy_name:20s}: {result}")
    print()


def example_quantifiers() -> None:
    """Demonstrate quantifier compilation with different strategies."""
    print("=" * 80)
    print("Example 3: Quantifiers (EXISTS and FORALL)")
    print("=" * 80)

    backend = create_backend("mlx")

    # Predicate values for multiple entities
    predicate = np.array([[0.9, 0.3, 0.7],  # Entity 1 properties
                          [0.2, 0.8, 0.4],  # Entity 2 properties
                          [0.6, 0.5, 0.9]]) # Entity 3 properties

    print(f"Predicate values (3 entities, 3 properties each):")
    print(f"{predicate}\n")

    # EXISTS: "There exists at least one entity with this property"
    print("EXISTS (at least one entity satisfies predicate):")
    print("-" * 80)
    for strategy_name in ["soft_differentiable", "hard_boolean", "godel"]:
        strategy = create_strategy(strategy_name, backend=backend)
        result = strategy.compile_exists(predicate, axis=0)
        print(f"{strategy_name:20s}: {result}")
    print()

    # FORALL: "All entities satisfy this property"
    print("FORALL (all entities satisfy predicate):")
    print("-" * 80)
    for strategy_name in ["soft_differentiable", "hard_boolean", "godel"]:
        strategy = create_strategy(strategy_name, backend=backend)
        result = strategy.compile_forall(predicate, axis=0)
        print(f"{strategy_name:20s}: {result}")
    print()


def example_soft_differentiable() -> None:
    """Detailed example: Soft differentiable strategy for training."""
    print("=" * 80)
    print("Example 4: Soft Differentiable Strategy (Training)")
    print("=" * 80)

    backend = create_backend("mlx")
    strategy = create_strategy("soft_differentiable", backend=backend)

    print(f"Strategy: {strategy.name}")
    print(f"Differentiable: {strategy.is_differentiable}")
    print(f"Use case: Neural predicate training, gradient-based optimization\n")

    # Simulate neural predicate outputs (continuous values [0, 1])
    predicate_p = np.array([0.85, 0.60, 0.30])  # P(x) confidence
    predicate_q = np.array([0.90, 0.45, 0.75])  # Q(x) confidence

    print("Neural predicate outputs:")
    print(f"P(x) = {predicate_p}")
    print(f"Q(x) = {predicate_q}\n")

    # Logical rule: P(x) AND Q(x)
    conjunction = strategy.compile_and(predicate_p, predicate_q)
    print(f"P(x) AND Q(x) = {conjunction}")
    print(f"Semantics: Product (independent events)\n")

    # Logical rule: P(x) -> Q(x) (implication)
    implication = strategy.compile_implies(predicate_p, predicate_q)
    print(f"P(x) -> Q(x) = {implication}")
    print(f"Semantics: max(1 - P(x), Q(x))")
    print(f"Interpretation: Implication holds when P is false OR Q is true\n")

    # Universal quantification: forall x: P(x)
    forall_p = strategy.compile_forall(predicate_p, axis=0)
    print(f"forall x: P(x) = {forall_p}")
    print(f"Semantics: min(P(x)) = {np.min(predicate_p)}")
    print(f"Interpretation: All entities must satisfy P\n")


def example_hard_boolean() -> None:
    """Detailed example: Hard boolean strategy for exact inference."""
    print("=" * 80)
    print("Example 5: Hard Boolean Strategy (Exact Inference)")
    print("=" * 80)

    backend = create_backend("mlx")
    strategy = create_strategy("hard_boolean", backend=backend)

    print(f"Strategy: {strategy.name}")
    print(f"Differentiable: {strategy.is_differentiable}")
    print(f"Use case: Production inference, exact logical reasoning\n")

    # Continuous inputs get discretized to 0 or 1
    soft_values = np.array([0.9, 0.6, 0.3, 0.1])
    print(f"Input (continuous): {soft_values}\n")

    # Hard boolean operations produce exact 0/1
    hard_and = strategy.compile_and(soft_values, np.array([0.8, 0.7, 0.5, 0.2]))
    print(f"AND result (discrete): {hard_and}")
    print(f"Semantics: step(a * b) -> produces only 0 or 1\n")

    hard_or = strategy.compile_or(soft_values, np.array([0.8, 0.7, 0.5, 0.2]))
    print(f"OR result (discrete): {hard_or}")
    print(f"Semantics: step(max(a, b)) -> produces only 0 or 1\n")

    print("Key benefit: Zero hallucinations, exact boolean logic")
    print("Trade-off: Not differentiable (cannot use for training)\n")


def example_fuzzy_godel() -> None:
    """Detailed example: Gödel fuzzy logic strategy."""
    print("=" * 80)
    print("Example 6: Gödel Fuzzy Logic Strategy")
    print("=" * 80)

    backend = create_backend("mlx")
    strategy = create_strategy("godel", backend=backend)

    print(f"Strategy: {strategy.name}")
    print(f"Differentiable: {strategy.is_differentiable} (via subgradients)")
    print(f"Use case: Fuzzy reasoning with conservative (min/max) semantics\n")

    # Fuzzy membership values
    temperature_high = np.array([0.8])  # Temperature is "highly" high
    pressure_low = np.array([0.6])      # Pressure is "moderately" low

    print(f"Temperature is high: {temperature_high}")
    print(f"Pressure is low: {pressure_low}\n")

    # Fuzzy AND: Takes minimum (most conservative)
    condition = strategy.compile_and(temperature_high, pressure_low)
    print(f"Temperature is high AND pressure is low: {condition}")
    print(f"Semantics: min(0.8, 0.6) = 0.6")
    print(f"Interpretation: Rule fires with confidence of weakest condition\n")

    # Fuzzy OR: Takes maximum (most optimistic)
    either = strategy.compile_or(temperature_high, pressure_low)
    print(f"Temperature is high OR pressure is low: {either}")
    print(f"Semantics: max(0.8, 0.6) = 0.8")
    print(f"Interpretation: Rule fires with confidence of strongest condition\n")

    # Idempotence property: AND(a, a) = a
    idempotent = strategy.compile_and(temperature_high, temperature_high)
    print(f"AND(a, a) = {idempotent} (idempotent: equals a)")
    print(f"Unique to Gödel fuzzy logic (not true for product semantics)\n")


def example_fuzzy_product() -> None:
    """Detailed example: Product fuzzy logic strategy."""
    print("=" * 80)
    print("Example 7: Product Fuzzy Logic Strategy")
    print("=" * 80)

    backend = create_backend("mlx")
    strategy = create_strategy("product", backend=backend)

    print(f"Strategy: {strategy.name}")
    print(f"Differentiable: {strategy.is_differentiable}")
    print(f"Use case: Fuzzy reasoning with probabilistic semantics\n")

    # Expert system: Multiple expert opinions
    expert_1_confidence = np.array([0.9])  # Expert 1: 90% confident
    expert_2_confidence = np.array([0.8])  # Expert 2: 80% confident

    print(f"Expert 1 confidence: {expert_1_confidence}")
    print(f"Expert 2 confidence: {expert_2_confidence}\n")

    # Fuzzy AND: Product (treats as independent probabilities)
    combined = strategy.compile_and(expert_1_confidence, expert_2_confidence)
    print(f"Combined confidence (AND): {combined}")
    print(f"Semantics: 0.9 * 0.8 = 0.72")
    print(f"Interpretation: Independent expert agreement probability\n")

    # Fuzzy OR: Probabilistic sum
    either_expert = strategy.compile_or(expert_1_confidence, expert_2_confidence)
    print(f"Either expert (OR): {either_expert}")
    print(f"Semantics: 0.9 + 0.8 - (0.9 * 0.8) = 0.98")
    print(f"Interpretation: Probability at least one expert is correct\n")


def example_fuzzy_lukasiewicz() -> None:
    """Detailed example: Łukasiewicz fuzzy logic strategy."""
    print("=" * 80)
    print("Example 8: Łukasiewicz Fuzzy Logic Strategy")
    print("=" * 80)

    backend = create_backend("mlx")
    strategy = create_strategy("lukasiewicz", backend=backend)

    print(f"Strategy: {strategy.name}")
    print(f"Differentiable: {strategy.is_differentiable}")
    print(f"Use case: Fuzzy reasoning with strict boundary conditions\n")

    # Fuzzy values
    a = np.array([0.6, 0.4, 0.8])
    b = np.array([0.7, 0.3, 0.5])

    print(f"a = {a}")
    print(f"b = {b}\n")

    # Łukasiewicz AND: max(0, a + b - 1)
    result_and = strategy.compile_and(a, b)
    print(f"AND(a, b) = {result_and}")
    print(f"Semantics: max(0, a + b - 1)")
    print(f"  [0]: max(0, 0.6 + 0.7 - 1) = 0.3")
    print(f"  [1]: max(0, 0.4 + 0.3 - 1) = 0.0 (bounded at 0)")
    print(f"  [2]: max(0, 0.8 + 0.5 - 1) = 0.3\n")

    # Łukasiewicz OR: min(1, a + b)
    result_or = strategy.compile_or(a, b)
    print(f"OR(a, b) = {result_or}")
    print(f"Semantics: min(1, a + b)")
    print(f"  [0]: min(1, 0.6 + 0.7) = 1.0 (bounded at 1)")
    print(f"  [1]: min(1, 0.4 + 0.3) = 0.7")
    print(f"  [2]: min(1, 0.8 + 0.5) = 1.0 (bounded at 1)\n")

    print("Key property: Strict boundary enforcement (never exceeds [0, 1])")
    print("Use case: When you need linear arithmetic semantics with bounds\n")


def example_train_with_soft_infer_with_hard() -> None:
    """Example: Train with soft strategy, deploy with hard strategy."""
    print("=" * 80)
    print("Example 9: Training vs Inference Strategy Pattern")
    print("=" * 80)

    backend = create_backend("mlx")

    # Simulate neural predicate outputs during training
    print("TRAINING PHASE: Use soft_differentiable for gradient flow")
    print("-" * 80)
    train_strategy = create_strategy("soft_differentiable", backend=backend)

    # Neural outputs are continuous [0, 1]
    neural_output_p = np.array([0.82, 0.65, 0.48])
    neural_output_q = np.array([0.91, 0.38, 0.73])

    print(f"Neural P(x): {neural_output_p}")
    print(f"Neural Q(x): {neural_output_q}\n")

    # Logical rule: P(x) AND Q(x)
    train_result = train_strategy.compile_and(neural_output_p, neural_output_q)
    print(f"Training result (soft AND): {train_result}")
    print(f"Gradients flow through product operation for backprop\n")

    # After training, switch to hard boolean for exact inference
    print("INFERENCE PHASE: Use hard_boolean for exact results")
    print("-" * 80)
    infer_strategy = create_strategy("hard_boolean", backend=backend)

    # Same neural outputs, but discretized to exact boolean
    infer_result = infer_strategy.compile_and(neural_output_p, neural_output_q)
    print(f"Inference result (hard AND): {infer_result}")
    print(f"Produces exact 0/1 values (no ambiguity)\n")

    print("Pattern: Train differentiable, deploy discrete")
    print("Benefits: Best of both worlds - smooth training, exact inference\n")


def example_strategy_properties() -> None:
    """Demonstrate mathematical properties of different strategies."""
    print("=" * 80)
    print("Example 10: Mathematical Properties by Strategy")
    print("=" * 80)

    backend = create_backend("mlx")
    a = np.array([0.7])

    print(f"Test value: a = {a}\n")

    # Test idempotence: AND(a, a) = a?
    print("Idempotence Test: AND(a, a) = a?")
    print("-" * 80)
    for strategy_name in ["soft_differentiable", "hard_boolean", "godel", "product"]:
        strategy = create_strategy(strategy_name, backend=backend)
        result = strategy.compile_and(a, a)
        is_idempotent = abs(float(result[0]) - float(a[0])) < 0.01
        print(
            f"{strategy_name:20s}: AND(a, a) = {result} "
            f"[{'✓ Idempotent' if is_idempotent else '✗ Not idempotent'}]"
        )
    print()

    # Test law of excluded middle: OR(a, NOT(a)) = 1?
    print("Law of Excluded Middle: OR(a, NOT(a)) = 1?")
    print("-" * 80)
    for strategy_name in ["soft_differentiable", "hard_boolean", "godel", "product"]:
        strategy = create_strategy(strategy_name, backend=backend)
        not_a = strategy.compile_not(a)
        result = strategy.compile_or(a, not_a)
        is_true = abs(float(result[0]) - 1.0) < 0.01
        print(
            f"{strategy_name:20s}: OR(a, NOT(a)) = {result} "
            f"[{'✓ Always 1' if is_true else '✗ Not always 1'}]"
        )
    print()


def example_integration_with_quantify() -> None:
    """Example: Integration with high-level quantify() API."""
    print("=" * 80)
    print("Example 11: Integration with quantify() API")
    print("=" * 80)

    # Note: This example demonstrates the API pattern
    # Actual quantify() implementation depends on API-001 completion

    print("Integration Pattern:")
    print("-" * 80)
    print("""
# Option 1: Pass strategy name
from tensorlogic.api import quantify

result = quantify(
    "forall x: P(x) -> Q(x)",
    predicates={"P": predicate_p, "Q": predicate_q},
    strategy="soft_differentiable"  # String name
)

# Option 2: Pass strategy instance
from tensorlogic.compilation import create_strategy

strategy = create_strategy("hard_boolean")
result = quantify(
    "exists y: Related(x, y) and HasProperty(y)",
    bindings={"x": entity_batch},
    strategy=strategy  # Direct instance
)

# Option 3: Use default (soft_differentiable)
result = quantify(
    "forall x: P(x)",
    predicates={"P": predicate}
)  # Automatically uses soft_differentiable
    """)


def main() -> None:
    """Run all compilation strategy examples."""
    examples: list[tuple[str, Any]] = [
        ("Basic Usage", example_basic_strategy_usage),
        ("Strategy Comparison", example_compare_strategies),
        ("Quantifiers", example_quantifiers),
        ("Soft Differentiable", example_soft_differentiable),
        ("Hard Boolean", example_hard_boolean),
        ("Gödel Fuzzy", example_fuzzy_godel),
        ("Product Fuzzy", example_fuzzy_product),
        ("Łukasiewicz Fuzzy", example_fuzzy_lukasiewicz),
        ("Train Soft / Infer Hard", example_train_with_soft_infer_with_hard),
        ("Mathematical Properties", example_strategy_properties),
        ("Integration with quantify()", example_integration_with_quantify),
    ]

    print("\n" + "=" * 80)
    print("TENSORLOGIC COMPILATION STRATEGIES - COMPREHENSIVE EXAMPLES")
    print("=" * 80)
    print()

    for i, (name, example_func) in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\n⚠️  Example {i} ({name}) failed: {e}\n")

        if i < len(examples):
            print("\n" + "=" * 80)
            print()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
