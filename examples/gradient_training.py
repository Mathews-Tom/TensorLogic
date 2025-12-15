"""Gradient-Based Training with TensorLogic

This example demonstrates that TensorLogic's soft_differentiable compilation
strategy enables gradient-based training through logical operations.

Key Concepts:
    - Soft semantics allow gradients to flow through AND/OR/NOT operations
    - Neural predicates can be trained using logical constraints as loss
    - The "Train Soft, Infer Hard" pattern for production deployment

Run example:
    uv run python examples/gradient_training.py
"""

from __future__ import annotations

import numpy as np

from tensorlogic import (
    create_backend,
    create_strategy,
    logical_and,
    logical_or,
    logical_not,
    logical_implies,
    exists,
    forall,
)


def gradient_check_and() -> None:
    """Demonstrate gradient flow through AND operation.

    AND(a, b) = a * b in soft semantics
    Gradients:
        d(AND)/da = b
        d(AND)/db = a
    """
    print("=" * 70)
    print("GRADIENT CHECK: AND Operation")
    print("=" * 70)

    backend = create_backend()
    strategy = create_strategy("soft_differentiable", backend=backend)

    # Test values
    a = np.array([0.7, 0.3, 0.9])
    b = np.array([0.8, 0.6, 0.4])

    print(f"\nInputs:")
    print(f"  a = {a}")
    print(f"  b = {b}")

    # Forward pass
    result = strategy.compile_and(a, b)
    print(f"\nForward: AND(a, b) = a * b = {np.asarray(result)}")

    # Analytical gradients
    grad_a = b  # d(a*b)/da = b
    grad_b = a  # d(a*b)/db = a

    print(f"\nAnalytical Gradients:")
    print(f"  d(AND)/da = b = {grad_a}")
    print(f"  d(AND)/db = a = {grad_b}")

    # Numerical gradient check
    epsilon = 1e-5
    numerical_grad_a = []
    for i in range(len(a)):
        a_plus = a.copy()
        a_plus[i] += epsilon
        a_minus = a.copy()
        a_minus[i] -= epsilon
        grad_i = (
            float(strategy.compile_and(a_plus, b)[i])
            - float(strategy.compile_and(a_minus, b)[i])
        ) / (2 * epsilon)
        numerical_grad_a.append(grad_i)

    print(f"\nNumerical Gradient Check:")
    print(f"  d(AND)/da (numerical) = {np.array(numerical_grad_a).round(4)}")
    print(f"  d(AND)/da (analytical) = {grad_a}")
    print(f"  Match: {np.allclose(numerical_grad_a, grad_a, atol=1e-4)}")


def gradient_check_or() -> None:
    """Demonstrate gradient flow through OR operation.

    OR(a, b) = a + b - a*b in soft semantics
    Gradients:
        d(OR)/da = 1 - b
        d(OR)/db = 1 - a
    """
    print("\n" + "=" * 70)
    print("GRADIENT CHECK: OR Operation")
    print("=" * 70)

    backend = create_backend()
    strategy = create_strategy("soft_differentiable", backend=backend)

    # Test values
    a = np.array([0.7, 0.3, 0.9])
    b = np.array([0.8, 0.6, 0.4])

    print(f"\nInputs:")
    print(f"  a = {a}")
    print(f"  b = {b}")

    # Forward pass
    result = strategy.compile_or(a, b)
    expected = a + b - a * b
    print(f"\nForward: OR(a, b) = a + b - a*b = {np.asarray(result)}")
    print(f"  Expected: {expected}")

    # Analytical gradients
    grad_a = 1 - b  # d(a + b - a*b)/da = 1 - b
    grad_b = 1 - a  # d(a + b - a*b)/db = 1 - a

    print(f"\nAnalytical Gradients:")
    print(f"  d(OR)/da = 1 - b = {grad_a}")
    print(f"  d(OR)/db = 1 - a = {grad_b}")

    # Numerical gradient check
    epsilon = 1e-5
    numerical_grad_a = []
    for i in range(len(a)):
        a_plus = a.copy()
        a_plus[i] += epsilon
        a_minus = a.copy()
        a_minus[i] -= epsilon
        grad_i = (
            float(strategy.compile_or(a_plus, b)[i])
            - float(strategy.compile_or(a_minus, b)[i])
        ) / (2 * epsilon)
        numerical_grad_a.append(grad_i)

    print(f"\nNumerical Gradient Check:")
    print(f"  d(OR)/da (numerical) = {np.array(numerical_grad_a).round(4)}")
    print(f"  d(OR)/da (analytical) = {grad_a}")
    print(f"  Match: {np.allclose(numerical_grad_a, grad_a, atol=1e-4)}")


def trainable_predicate_simulation() -> None:
    """Simulate training a neural predicate using logical constraints.

    Scenario:
        - We have a predicate P(x) that should satisfy: forall x: P(x) -> Q(x)
        - Q(x) is given (ground truth)
        - P(x) starts random and learns to satisfy the constraint

    This demonstrates the core "logic as loss function" concept.
    """
    print("\n" + "=" * 70)
    print("TRAINABLE PREDICATE SIMULATION")
    print("=" * 70)

    backend = create_backend()
    strategy = create_strategy("soft_differentiable", backend=backend)

    # Ground truth: Q(x) - these are fixed
    q = np.array([1.0, 1.0, 0.0, 1.0, 0.0])

    # Trainable: P(x) - starts random, should learn to imply Q
    np.random.seed(42)
    p = np.random.uniform(0.3, 0.7, size=5)

    print(f"\nSetup:")
    print(f"  Ground truth Q(x) = {q}")
    print(f"  Initial P(x) = {p.round(3)}")
    print(f"\nConstraint: forall x: P(x) -> Q(x)")
    print("  (P should only be true where Q is true)")

    # Training loop
    learning_rate = 0.5
    n_iterations = 20

    print("\n" + "-" * 60)
    print(f"{'Iteration':>10} {'Loss':>10} {'P values':>30}")
    print("-" * 60)

    for iteration in range(n_iterations):
        # Forward pass: compute implication P(x) -> Q(x) = max(1-P, Q)
        # For soft semantics: 1 - P + P*Q (derived from max(1-P, Q))
        implication = np.maximum(1 - p, q)

        # Loss: we want forall x: implication(x) to be 1
        # Using soft forall as product: prod(implication)
        forall_value = np.prod(implication)

        # Loss = 1 - forall_value (we want to maximize forall_value)
        loss = 1 - forall_value

        if iteration % 5 == 0 or iteration == n_iterations - 1:
            print(f"{iteration:>10} {loss:>10.4f} {np.array2string(p, precision=3):>30}")

        # Compute gradient of loss w.r.t. P
        # d(loss)/dP = d(1 - prod(max(1-P, Q)))/dP
        # For simplicity, we use numerical gradient here
        epsilon = 1e-5
        grad_p = np.zeros_like(p)
        for i in range(len(p)):
            p_plus = p.copy()
            p_plus[i] += epsilon
            p_minus = p.copy()
            p_minus[i] -= epsilon

            impl_plus = np.maximum(1 - p_plus, q)
            impl_minus = np.maximum(1 - p_minus, q)

            loss_plus = 1 - np.prod(impl_plus)
            loss_minus = 1 - np.prod(impl_minus)

            grad_p[i] = (loss_plus - loss_minus) / (2 * epsilon)

        # Update P
        p = p - learning_rate * grad_p

        # Clip to valid range [0, 1]
        p = np.clip(p, 0, 1)

    print("-" * 60)

    print(f"\nFinal Results:")
    print(f"  Final P(x) = {p.round(3)}")
    print(f"  Ground truth Q(x) = {q}")
    print(f"\nAnalysis:")
    print("  Where Q=0 (indices 2, 4), P learned to be ~0 (can't imply false)")
    print("  Where Q=1 (indices 0, 1, 3), P can be any value (P->True is always true)")


def train_soft_infer_hard_pattern() -> None:
    """Demonstrate the Train Soft, Infer Hard pattern.

    1. Training: Use soft_differentiable for gradient flow
    2. Inference: Use hard_boolean for exact outputs
    """
    print("\n" + "=" * 70)
    print("TRAIN SOFT, INFER HARD PATTERN")
    print("=" * 70)

    backend = create_backend()
    soft_strategy = create_strategy("soft_differentiable", backend=backend)
    hard_strategy = create_strategy("hard_boolean", backend=backend)

    # Simulated "trained" predicate values
    # (In reality, these would come from a neural network)
    trained_p = np.array([0.92, 0.78, 0.45, 0.31, 0.08])
    trained_q = np.array([0.88, 0.65, 0.52, 0.28, 0.15])

    labels = ["Entity A", "Entity B", "Entity C", "Entity D", "Entity E"]

    print("\nSimulated trained predicate values:")
    print(f"  P(x) = {trained_p}")
    print(f"  Q(x) = {trained_q}")

    print("\n" + "-" * 60)
    print("TRAINING MODE: soft_differentiable")
    print("-" * 60)

    soft_and = soft_strategy.compile_and(trained_p, trained_q)
    soft_or = soft_strategy.compile_or(trained_p, trained_q)

    print(f"\nP AND Q (soft): {np.asarray(soft_and).round(3)}")
    print("  (Continuous values, gradients can flow)")
    print(f"\nP OR Q (soft): {np.asarray(soft_or).round(3)}")
    print("  (Continuous values, gradients can flow)")

    print("\n" + "-" * 60)
    print("INFERENCE MODE: hard_boolean")
    print("-" * 60)

    hard_and = hard_strategy.compile_and(trained_p, trained_q)
    hard_or = hard_strategy.compile_or(trained_p, trained_q)

    print(f"\nP AND Q (hard): {np.asarray(hard_and).round(3)}")
    print("  (Binary outputs: 0 or 1)")
    print(f"\nP OR Q (hard): {np.asarray(hard_or).round(3)}")
    print("  (Binary outputs: 0 or 1)")

    print("\n" + "-" * 60)
    print("Side-by-side Comparison:")
    print("-" * 60)
    print(f"\n{'Entity':>10} {'P':>6} {'Q':>6} {'AND(soft)':>10} {'AND(hard)':>10}")
    print("-" * 50)
    for i, label in enumerate(labels):
        print(
            f"{label:>10} {trained_p[i]:>6.2f} {trained_q[i]:>6.2f} "
            f"{float(soft_and[i]):>10.3f} {float(hard_and[i]):>10.0f}"
        )


def logical_constraint_as_loss() -> None:
    """Show how logical constraints become differentiable loss functions.

    Key insight: Soft logical operations are differentiable, so logical
    constraints can be directly used as loss functions for training.
    """
    print("\n" + "=" * 70)
    print("LOGICAL CONSTRAINTS AS LOSS FUNCTIONS")
    print("=" * 70)

    backend = create_backend()
    strategy = create_strategy("soft_differentiable", backend=backend)

    print("""
    Key Insight:
    ------------
    Any logical constraint can become a differentiable loss function!

    Constraint                   | Loss Function
    -----------------------------|----------------------------
    forall x: P(x)               | 1 - prod(P)
    exists x: P(x)               | 1 - max(P)
    P(x) -> Q(x)                 | ReLU(P - Q) or 1 - max(1-P, Q)
    forall x: P(x) -> Q(x)       | 1 - prod(max(1-P, Q))
    P(x) AND Q(x)                | 1 - P*Q (want to maximize)
    P(x) OR Q(x)                 | 1 - (P + Q - P*Q) (want to maximize)
    """)

    # Example: Constraint satisfaction scoring
    p = np.array([0.9, 0.7, 0.3, 0.8])
    q = np.array([0.85, 0.6, 0.4, 0.9])

    print("\nExample: Scoring constraint satisfaction")
    print(f"  P = {p}")
    print(f"  Q = {q}")

    # Different constraints as losses
    print("\nConstraint Satisfaction Scores:")

    # 1. forall x: P(x) AND Q(x)
    and_values = strategy.compile_and(p, q)
    forall_and = float(np.prod(np.asarray(and_values)))
    print(f"  forall x: P(x) AND Q(x) = {forall_and:.3f}")

    # 2. exists x: P(x) AND Q(x)
    exists_and = float(np.max(np.asarray(and_values)))
    print(f"  exists x: P(x) AND Q(x) = {exists_and:.3f}")

    # 3. forall x: P(x) -> Q(x)
    impl_values = np.maximum(1 - p, q)
    forall_impl = float(np.prod(impl_values))
    print(f"  forall x: P(x) -> Q(x) = {forall_impl:.3f}")

    print("\n  (Higher scores = better constraint satisfaction)")
    print("  (These can be used as rewards in training)")


def summary() -> None:
    """Print summary of gradient training concepts."""
    print("\n" + "=" * 70)
    print("SUMMARY: Gradient Training with TensorLogic")
    print("=" * 70)

    print("""
    KEY CONCEPTS:

    1. SOFT DIFFERENTIABLE SEMANTICS
       - AND(a,b) = a * b
       - OR(a,b) = a + b - a*b
       - NOT(a) = 1 - a
       - All operations have well-defined gradients

    2. GRADIENTS FLOW THROUGH LOGIC
       - d(AND)/da = b, d(AND)/db = a
       - d(OR)/da = 1-b, d(OR)/db = 1-a
       - d(NOT)/da = -1

    3. TRAIN SOFT, INFER HARD
       - Training: soft_differentiable strategy (gradients flow)
       - Inference: hard_boolean strategy (exact 0/1 outputs)

    4. LOGICAL CONSTRAINTS AS LOSS
       - Any logical formula becomes a differentiable loss
       - forall, exists, implications all work
       - Enable "logic-guided" neural network training

    PRACTICAL APPLICATIONS:
       - Neural knowledge graph completion
       - Constraint-based learning
       - Neuro-symbolic reasoning
       - Rule learning with gradient descent
    """)


def main() -> None:
    """Run all gradient training examples."""
    print("\n" + "=" * 70)
    print("TENSORLOGIC: GRADIENT-BASED TRAINING EXAMPLE")
    print("=" * 70)
    print("""
    This example demonstrates that TensorLogic's soft_differentiable
    compilation strategy enables gradient-based training through
    logical operations.
    """)

    gradient_check_and()
    gradient_check_or()
    trainable_predicate_simulation()
    train_soft_infer_hard_pattern()
    logical_constraint_as_loss()
    summary()

    print("\n" + "=" * 70)
    print("ALL GRADIENT EXAMPLES COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
