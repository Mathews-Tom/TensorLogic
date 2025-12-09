"""End-to-end integration tests for TensorLogic CoreLogic module.

This test suite validates the complete CoreLogic implementation by demonstrating
real-world logical reasoning patterns. Tests cover:
- Multi-predicate rule composition (Aunt, Grandparent patterns)
- Multi-hop reasoning with transitive closure (Ancestor chains)
- Quantified rules with existential/universal quantifiers
- Temperature-controlled reasoning (deductive vs analogical)
- Performance benchmarks against transformer baselines

All tests use realistic domain examples to verify the framework works end-to-end.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from tensorlogic.backends import create_backend
from tensorlogic.core import (
    logical_and,
    logical_or,
    logical_not,
    exists,
    forall,
    compose_and,
    compose_or,
    step,
)
from tensorlogic.core.temperature import (
    temperature_scaled_operation,
    deductive_operation,
    analogical_operation,
)


@pytest.fixture(params=["numpy", "mlx"])
def backend(request):
    """Parametrized fixture for testing across backends."""
    return create_backend(request.param)


class TestMultiPredicateRules:
    """Integration tests for multi-predicate rule composition."""

    def test_aunt_rule_complete(self, backend) -> None:
        """Test complete Aunt rule: Aunt(x,z) ← Sister(x,y) ∧ Parent(y,z).

        Domain: Family relationships
        - Alice (0) is sister of Bob (1)
        - Bob (1) is parent of Carol (2)
        - Therefore: Alice is aunt of Carol
        """
        # Domain: 4 people (Alice=0, Bob=1, Carol=2, Dave=3)
        sister = np.array([
            [0.0, 1.0, 0.0, 0.0],  # Alice is sister of Bob
            [1.0, 0.0, 0.0, 0.0],  # Bob is sister of Alice
            [0.0, 0.0, 0.0, 0.0],  # Carol has no sisters
            [0.0, 0.0, 0.0, 0.0],  # Dave has no sisters
        ])

        parent = np.array([
            [0.0, 0.0, 0.0, 0.0],  # Alice is not a parent
            [0.0, 0.0, 1.0, 0.0],  # Bob is parent of Carol
            [0.0, 0.0, 0.0, 0.0],  # Carol is not a parent
            [0.0, 0.0, 0.0, 0.0],  # Dave is not a parent
        ])

        # Use einsum to align indices: Sister[x,y] ⊙ Parent[y,z] -> Combined[x,z]
        combined = backend.einsum("xy,yz->xz", sister, parent)
        aunt = step(combined, backend=backend)
        backend.eval(aunt)

        # Expected: Alice is aunt of Carol
        expected = np.array([
            [0.0, 0.0, 1.0, 0.0],  # Alice is aunt of Carol
            [0.0, 0.0, 0.0, 0.0],  # Bob is not an aunt
            [0.0, 0.0, 0.0, 0.0],  # Carol is not an aunt
            [0.0, 0.0, 0.0, 0.0],  # Dave is not an aunt
        ])

        np.testing.assert_array_equal(aunt, expected)

    def test_grandparent_rule(self, backend) -> None:
        """Test Grandparent rule: Grandparent(x,z) ← Parent(x,y) ∧ Parent(y,z).

        Domain: Multi-generational family
        - Alice (0) is parent of Bob (1)
        - Bob (1) is parent of Carol (2)
        - Therefore: Alice is grandparent of Carol
        """
        # Domain: 3 people (Alice=0, Bob=1, Carol=2)
        parent = np.array([
            [0.0, 1.0, 0.0],  # Alice is parent of Bob
            [0.0, 0.0, 1.0],  # Bob is parent of Carol
            [0.0, 0.0, 0.0],  # Carol is not a parent
        ])

        # Grandparent[x,z] = Parent[x,y] ⊙ Parent[y,z]
        grandparent_soft = backend.einsum("xy,yz->xz", parent, parent)
        grandparent = step(grandparent_soft, backend=backend)
        backend.eval(grandparent)

        # Expected: Alice is grandparent of Carol
        expected = np.array([
            [0.0, 0.0, 1.0],  # Alice is grandparent of Carol
            [0.0, 0.0, 0.0],  # Bob is not a grandparent
            [0.0, 0.0, 0.0],  # Carol is not a grandparent
        ])

        np.testing.assert_array_equal(grandparent, expected)

    def test_uncle_rule_composition(self, backend) -> None:
        """Test Uncle rule: Uncle(x,z) ← Brother(x,y) ∧ Parent(y,z).

        More complex than Aunt because brother relationship is non-symmetric.
        """
        # Domain: 4 people (Alice=0, Bob=1, Carol=2, Dave=3)
        # Brother[x, y] means "x is brother of y"
        brother = np.array([
            [0.0, 0.0, 0.0, 0.0],  # Alice has no brothers
            [1.0, 0.0, 0.0, 0.0],  # Bob is brother of Alice
            [0.0, 0.0, 0.0, 0.0],  # Carol has no brothers
            [0.0, 0.0, 1.0, 0.0],  # Dave is brother of Carol
        ])

        parent = np.array([
            [0.0, 0.0, 1.0, 0.0],  # Alice is parent of Carol
            [0.0, 0.0, 0.0, 0.0],  # Bob is not a parent
            [0.0, 0.0, 0.0, 0.0],  # Carol is not a parent
            [0.0, 0.0, 0.0, 0.0],  # Dave is not a parent
        ])

        # Uncle[x,z] = Brother[x,y] ⊙ Parent[y,z]
        # For Bob to be uncle of Carol: Bob is brother of Alice (Brother[1,0]=1) AND Alice is parent of Carol (Parent[0,2]=1)
        uncle_soft = backend.einsum("xy,yz->xz", brother, parent)
        uncle = step(uncle_soft, backend=backend)
        backend.eval(uncle)

        # Expected: Bob is uncle of Carol (Bob is brother of Alice, Alice is parent of Carol)
        expected = np.array([
            [0.0, 0.0, 0.0, 0.0],  # Alice is not an uncle
            [0.0, 0.0, 1.0, 0.0],  # Bob is uncle of Carol
            [0.0, 0.0, 0.0, 0.0],  # Carol is not an uncle
            [0.0, 0.0, 0.0, 0.0],  # Dave is not an uncle
        ])

        np.testing.assert_array_equal(uncle, expected)


class TestMultiHopReasoning:
    """Integration tests for multi-hop reasoning with transitive closure."""

    def test_ancestor_two_hops(self, backend) -> None:
        """Test 2-hop ancestor chain: Ancestor includes grandparents.

        Ancestor(x,z) ← Parent(x,z) ∨ (Parent(x,y) ∧ Parent(y,z))
        """
        # Domain: 3 people (Alice=0, Bob=1, Carol=2)
        parent = np.array([
            [0.0, 1.0, 0.0],  # Alice is parent of Bob
            [0.0, 0.0, 1.0],  # Bob is parent of Carol
            [0.0, 0.0, 0.0],  # Carol is not a parent
        ])

        # Direct ancestors (parents)
        direct = parent

        # 2-hop ancestors (grandparents)
        two_hop_soft = backend.einsum("xy,yz->xz", parent, parent)
        two_hop = step(two_hop_soft, backend=backend)

        # Combine: Ancestor = Parent ∨ Grandparent
        ancestor = compose_or(direct, two_hop, backend=backend)
        backend.eval(ancestor)

        # Expected: Alice is ancestor of Bob and Carol, Bob is ancestor of Carol
        expected = np.array([
            [0.0, 1.0, 1.0],  # Alice is ancestor of Bob and Carol
            [0.0, 0.0, 1.0],  # Bob is ancestor of Carol
            [0.0, 0.0, 0.0],  # Carol is not an ancestor
        ])

        np.testing.assert_array_equal(ancestor, expected)

    def test_ancestor_three_hops(self, backend) -> None:
        """Test 3-hop ancestor chain: Great-grandparents included.

        Domain: 4 generations
        """
        # Domain: 4 people (Alice=0, Bob=1, Carol=2, Dave=3)
        parent = np.array([
            [0.0, 1.0, 0.0, 0.0],  # Alice parent of Bob
            [0.0, 0.0, 1.0, 0.0],  # Bob parent of Carol
            [0.0, 0.0, 0.0, 1.0],  # Carol parent of Dave
            [0.0, 0.0, 0.0, 0.0],  # Dave not a parent
        ])

        # 1-hop: parents
        hop1 = parent

        # 2-hop: grandparents
        hop2_soft = backend.einsum("xy,yz->xz", parent, parent)
        hop2 = step(hop2_soft, backend=backend)

        # 3-hop: great-grandparents
        hop3_soft = backend.einsum("xy,yz->xz", hop2, parent)
        hop3 = step(hop3_soft, backend=backend)

        # Combine all hops
        ancestor_temp = compose_or(hop1, hop2, backend=backend)
        ancestor = compose_or(ancestor_temp, hop3, backend=backend)
        backend.eval(ancestor)

        # Expected: Alice is ancestor of Bob, Carol, Dave
        assert float(ancestor[0, 1]) == 1.0  # Alice -> Bob
        assert float(ancestor[0, 2]) == 1.0  # Alice -> Carol
        assert float(ancestor[0, 3]) == 1.0  # Alice -> Dave
        assert float(ancestor[1, 3]) == 1.0  # Bob -> Dave


class TestQuantifiedRules:
    """Integration tests for rules with quantifiers."""

    def test_exists_with_property(self, backend) -> None:
        """Test ∃y: Related(x,y) ∧ HasProperty(y).

        Question: Which entities x have at least one related entity with property P?
        """
        # Domain: 4 entities
        # Related[x, y]: entity x is related to entity y
        related = np.array([
            [0.0, 1.0, 1.0, 0.0],  # Entity 0 related to 1, 2
            [1.0, 0.0, 0.0, 1.0],  # Entity 1 related to 0, 3
            [0.0, 0.0, 0.0, 1.0],  # Entity 2 related to 3
            [0.0, 0.0, 0.0, 0.0],  # Entity 3 not related to anyone
        ])

        # HasProperty[y]: entity y has property P
        has_property = np.array([0.0, 1.0, 0.0, 1.0])  # Entities 1 and 3 have property

        # Related(x,y) ∧ HasProperty(y)
        combined = backend.multiply(related, has_property)

        # ∃y: Related(x,y) ∧ HasProperty(y)
        result = exists(combined, axis=1, backend=backend)
        backend.eval(result)

        # Expected:
        # - Entity 0: related to 1 (has property) ✓
        # - Entity 1: related to 3 (has property) ✓
        # - Entity 2: related to 3 (has property) ✓
        # - Entity 3: not related to anyone ✗
        expected = np.array([1.0, 1.0, 1.0, 0.0])

        np.testing.assert_array_equal(result, expected)

    def test_forall_constraint(self, backend) -> None:
        """Test ∀y: Related(x,y) → HasProperty(y).

        Question: Which entities x only relate to entities with property P?
        """
        # Domain: 3 entities
        related = np.array([
            [0.0, 1.0, 1.0],  # Entity 0 related to 1, 2
            [0.0, 0.0, 1.0],  # Entity 1 related to 2
            [0.0, 0.0, 0.0],  # Entity 2 not related to anyone
        ])

        has_property = np.array([0.0, 1.0, 1.0])  # Entities 1 and 2 have property

        # Implication: Related(x,y) → HasProperty(y) = max(1 - Related(x,y), HasProperty(y))
        not_related = logical_not(related, backend=backend)
        implication = backend.maximum(not_related, has_property)

        # ∀y: Related(x,y) → HasProperty(y)
        result = forall(implication, axis=1, backend=backend)
        backend.eval(result)

        # Expected:
        # - Entity 0: relates to 1,2 (both have property) ✓
        # - Entity 1: relates to 2 (has property) ✓
        # - Entity 2: relates to no one (vacuously true) ✓
        expected = np.array([1.0, 1.0, 1.0])

        np.testing.assert_array_equal(result, expected)

    def test_nested_quantifiers(self, backend) -> None:
        """Test nested quantifiers: ∀x: ∃y: Related(x,y).

        Question: Does every entity have at least one relation?
        """
        # Domain: 3 entities
        related = np.array([
            [0.0, 1.0, 0.0],  # Entity 0 related to 1
            [0.0, 0.0, 1.0],  # Entity 1 related to 2
            [0.0, 0.0, 0.0],  # Entity 2 not related to anyone
        ])

        # ∃y: Related(x,y) for each x
        has_relation = exists(related, axis=1, backend=backend)

        # ∀x: has_relation(x)
        all_have_relations = forall(has_relation, axis=None, backend=backend)
        backend.eval(all_have_relations)

        # Expected: False (entity 2 has no relations)
        assert float(all_have_relations) == 0.0


class TestTemperatureControlled:
    """Integration tests for temperature-controlled reasoning."""

    def test_deductive_vs_analogical_and(self, backend) -> None:
        """Compare deductive (T=0) vs analogical (T=1) reasoning for AND."""
        a = np.array([0.7, 0.4])
        b = np.array([0.8, 0.3])

        # Deductive: T=0 (hard boolean)
        deductive_op = deductive_operation(logical_and, backend=backend)
        deductive_result = deductive_op(a, b, backend=backend)
        backend.eval(deductive_result)

        # Analogical: T=1.0 (soft probabilistic)
        analogical_op = analogical_operation(logical_and, temperature=1.0, backend=backend)
        analogical_result = analogical_op(a, b, backend=backend)
        backend.eval(analogical_result)

        # Deductive should give hard boolean (step of 0.56 and 0.12 = 1.0 and 1.0)
        np.testing.assert_array_equal(deductive_result, np.array([1.0, 1.0]))

        # Analogical should be interpolated (between hard and soft)
        # Verify it's different from deductive
        assert not np.array_equal(analogical_result, deductive_result)

        # Verify analogical is closer to soft result
        soft_result = a * b  # [0.56, 0.12]
        analog_diff = np.abs(analogical_result - soft_result).sum()
        deductive_diff = np.abs(deductive_result - soft_result).sum()
        assert analog_diff < deductive_diff

    def test_multi_rule_with_temperature(self, backend) -> None:
        """Test complete multi-predicate rule with temperature control.

        Aunt rule with temperature: allows fuzzy family relationships.
        """
        # Fuzzy family relationships (probabilistic)
        sister = np.array([
            [0.0, 0.9, 0.3],  # Alice 90% sister of Bob, 30% of Carol
            [0.9, 0.0, 0.2],  # Bob 90% sister of Alice, 20% of Carol
            [0.3, 0.2, 0.0],  # Carol 30% sister of Alice, 20% of Bob
        ])

        parent = np.array([
            [0.0, 0.0, 0.0],  # Alice not a parent
            [0.0, 0.0, 0.8],  # Bob 80% parent of Carol
            [0.0, 0.0, 0.0],  # Carol not a parent
        ])

        # Deductive (T=0): Hard boolean logic
        combined_soft = backend.einsum("xy,yz->xz", sister, parent)
        deductive_result = step(combined_soft, backend=backend)
        backend.eval(deductive_result)

        # Expected: Alice[0,2] = step(0.9*0.8) = step(0.72) = 1.0
        assert float(deductive_result[0, 2]) == 1.0

        # Analogical (T=5.0): Soft probabilistic (high temperature)
        temp_op = temperature_scaled_operation(
            lambda a, b, backend: backend.multiply(a, b),
            temperature=5.0,
            backend=backend,
        )

        # Apply temperature-scaled multiplication
        sister_x_parent = backend.einsum("xy,yz->xz", sister, parent)
        analogical_result = temp_op(sister_x_parent, 1.0, backend=backend)
        backend.eval(analogical_result)

        # Analogical should preserve more fuzzy information
        # Result should be closer to 0.72 than 1.0
        analogical_value = float(analogical_result[0, 2])
        assert 0.5 < analogical_value < 1.0  # Between soft and hard


class TestMLXGradients:
    """MLX-specific tests for gradient computation (autodiff).

    These tests verify differentiability of logical operations for neural-symbolic
    learning. They require MLX backend's automatic differentiation capabilities.
    """

    def test_temperature_gradient_for_learning(self) -> None:
        """Test that soft temperature operations are differentiable.

        Verifies that analogical reasoning operations support gradient computation
        for end-to-end learning in neural-symbolic AI systems.
        """
        import mlx.core as mx

        backend = create_backend("mlx")

        a = mx.array([0.6, 0.7])
        b = mx.array([0.5, 0.8])

        # Create differentiable operation
        def loss_fn(a_param):
            op = analogical_operation(logical_and, temperature=1.0, backend=backend)
            result = op(a_param, b, backend=backend)
            return backend.sum(result, axis=None)

        # Compute gradient
        grad_fn = backend.grad(loss_fn)
        gradients = grad_fn(a)
        backend.eval(gradients)

        # Verify gradient exists and is finite
        assert gradients.shape == a.shape
        grad_np = np.asarray(gradients)
        assert np.all(np.isfinite(grad_np))


class TestPerformance:
    """Integration tests for performance benchmarks."""

    def test_einsum_performance_baseline(self, backend) -> None:
        """Benchmark einsum performance (core operation).

        Target: <10ms for 12 equation evaluations (transformer baseline).
        """
        # Create realistic tensor sizes (batch=32, dim=64)
        a = np.random.rand(32, 64)
        b = np.random.rand(64, 32)

        # Warm-up
        result = backend.einsum("ij,jk->ik", a, b)
        backend.eval(result)

        # Benchmark 12 einsum operations (transformer-like)
        start = time.time()
        for _ in range(12):
            result = backend.einsum("ij,jk->ik", a, b)
            backend.eval(result)
        elapsed = time.time() - start

        # Report performance
        per_op = elapsed / 12 * 1000  # ms per operation
        print(f"\n[{backend.__class__.__name__}] Einsum performance: {per_op:.2f}ms per operation")

        # Sanity check: should complete in reasonable time (<1s for 12 ops)
        assert elapsed < 1.0, f"Performance regression: {elapsed:.3f}s for 12 ops"

    def test_quantifier_performance(self, backend) -> None:
        """Benchmark quantifier operations."""
        # Large tensor (100x100)
        tensor = np.random.rand(100, 100)

        # Warm-up
        result = exists(tensor, axis=1, backend=backend)
        backend.eval(result)

        # Benchmark exists operation
        start = time.time()
        for _ in range(100):
            result = exists(tensor, axis=1, backend=backend)
            backend.eval(result)
        elapsed = time.time() - start

        per_op = elapsed / 100 * 1000
        print(f"\n[{backend.__class__.__name__}] Quantifier performance: {per_op:.2f}ms per operation")

        # Sanity check
        assert elapsed < 1.0, f"Performance regression: {elapsed:.3f}s for 100 ops"


class TestDocumentationExamples:
    """Validate that all documentation examples work correctly."""

    def test_basic_and_example(self, backend) -> None:
        """Test basic AND example from documentation."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])

        result = logical_and(a, b, backend=backend)
        backend.eval(result)

        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_basic_or_example(self, backend) -> None:
        """Test basic OR example from documentation."""
        a = np.array([1.0, 1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 1.0, 0.0])

        result = logical_or(a, b, backend=backend)
        backend.eval(result)

        expected = np.array([1.0, 1.0, 1.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_exists_example(self, backend) -> None:
        """Test EXISTS quantifier example from documentation."""
        # Predicate matrix: P[x, y] indicates if P(x, y) is true
        predicate = np.array([
            [0.0, 1.0, 0.0],  # ∃y: P(x=0, y) = True (y=1)
            [0.0, 0.0, 0.0],  # ∃y: P(x=1, y) = False
            [0.0, 0.0, 1.0],  # ∃y: P(x=2, y) = True (y=2)
        ])

        result = exists(predicate, axis=1, backend=backend)
        backend.eval(result)

        expected = np.array([1.0, 0.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_composition_example(self, backend) -> None:
        """Test rule composition example from documentation."""
        rule1 = np.array([1.0, 0.0, 1.0])
        rule2 = np.array([1.0, 1.0, 0.0])
        rule3 = np.array([1.0, 0.0, 0.0])

        # Compose with AND
        result = compose_and(rule1, rule2, rule3, backend=backend)
        backend.eval(result)

        # Only first element satisfies all three rules
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)
