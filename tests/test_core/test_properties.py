"""Property-based tests for core logical operations.

Uses hypothesis library to generate random test cases verifying mathematical
properties hold universally across all possible inputs and tensor shapes.
Tests verify:
- Associativity: (a ∧ b) ∧ c = a ∧ (b ∧ c)
- Commutativity: a ∧ b = b ∧ a, a ∨ b = b ∨ a
- Distributivity: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)
- De Morgan's laws: ¬(a ∧ b) = ¬a ∨ ¬b
- Double negation: ¬¬a = a
- Identity/Annihilator: a ∧ 1 = a, a ∧ 0 = 0, etc.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, strategies as st

from tensorlogic.backends import create_backend
from tensorlogic.core.operations import (
    logical_and,
    logical_implies,
    logical_not,
    logical_or,
)
from tensorlogic.core.quantifiers import (
    exists,
    forall,
    soft_exists,
    soft_forall,
)


# Custom strategies for generating boolean tensors
@st.composite
def boolean_arrays(
    draw: st.DrawFn,
    min_size: int = 1,
    max_size: int = 20,
    allow_multidim: bool = True,
) -> np.ndarray:
    """Generate random boolean arrays (values in {0.0, 1.0})."""
    if allow_multidim and draw(st.booleans()):
        # Generate multidimensional array (keep small to avoid explosion)
        ndim = draw(st.integers(min_value=1, max_value=3))
        shape = draw(
            st.lists(
                st.integers(min_value=1, max_value=5),  # Small dimensions
                min_size=ndim,
                max_size=ndim,
            )
        )
        size = int(np.prod(shape))
        values = draw(
            st.lists(
                st.sampled_from([0.0, 1.0]),
                min_size=size,
                max_size=size,
            )
        )
        return np.array(values).reshape(shape)
    else:
        # Generate 1D array
        size = draw(st.integers(min_value=min_size, max_value=max_size))
        values = draw(
            st.lists(
                st.sampled_from([0.0, 1.0]),
                min_size=size,
                max_size=size,
            )
        )
        return np.array(values)


@st.composite
def matching_boolean_arrays(
    draw: st.DrawFn,
    min_size: int = 1,
    max_size: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two boolean arrays with matching shapes."""
    arr_a = draw(boolean_arrays(min_size=min_size, max_size=max_size))
    # Generate second array with same shape
    values = draw(
        st.lists(
            st.sampled_from([0.0, 1.0]),
            min_size=arr_a.size,
            max_size=arr_a.size,
        )
    )
    arr_b = np.array(values).reshape(arr_a.shape)
    return arr_a, arr_b


@st.composite
def triple_matching_arrays(
    draw: st.DrawFn,
    min_size: int = 1,
    max_size: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate three boolean arrays with matching shapes."""
    arr_a = draw(boolean_arrays(min_size=min_size, max_size=max_size))
    # Generate second and third arrays with same shape
    values_b = draw(
        st.lists(
            st.sampled_from([0.0, 1.0]),
            min_size=arr_a.size,
            max_size=arr_a.size,
        )
    )
    values_c = draw(
        st.lists(
            st.sampled_from([0.0, 1.0]),
            min_size=arr_a.size,
            max_size=arr_a.size,
        )
    )
    arr_b = np.array(values_b).reshape(arr_a.shape)
    arr_c = np.array(values_c).reshape(arr_a.shape)
    return arr_a, arr_b, arr_c


class TestCommutativity:
    """Property-based tests for commutativity: a ⊕ b = b ⊕ a."""

    @settings(deadline=None, max_examples=100)
    @given(arrays=matching_boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_and_commutative(
        self,
        arrays: tuple[np.ndarray, np.ndarray],
        backend_name: str,
    ) -> None:
        """Property: a ∧ b = b ∧ a for all boolean tensors a, b."""
        a, b = arrays
        backend = create_backend(backend_name)

        result_ab = logical_and(a, b, backend=backend)
        result_ba = logical_and(b, a, backend=backend)
        backend.eval(result_ab, result_ba)

        np.testing.assert_array_equal(
            result_ab,
            result_ba,
            err_msg=f"AND not commutative for a={a}, b={b} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(arrays=matching_boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_or_commutative(
        self,
        arrays: tuple[np.ndarray, np.ndarray],
        backend_name: str,
    ) -> None:
        """Property: a ∨ b = b ∨ a for all boolean tensors a, b."""
        a, b = arrays
        backend = create_backend(backend_name)

        result_ab = logical_or(a, b, backend=backend)
        result_ba = logical_or(b, a, backend=backend)
        backend.eval(result_ab, result_ba)

        np.testing.assert_array_equal(
            result_ab,
            result_ba,
            err_msg=f"OR not commutative for a={a}, b={b} (backend={backend_name})",
        )


class TestAssociativity:
    """Property-based tests for associativity: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)."""

    @settings(deadline=None, max_examples=100)
    @given(arrays=triple_matching_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_and_associative(
        self,
        arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
        backend_name: str,
    ) -> None:
        """Property: (a ∧ b) ∧ c = a ∧ (b ∧ c) for all boolean tensors."""
        a, b, c = arrays
        backend = create_backend(backend_name)

        # Left: (a ∧ b) ∧ c
        ab = logical_and(a, b, backend=backend)
        left = logical_and(ab, c, backend=backend)

        # Right: a ∧ (b ∧ c)
        bc = logical_and(b, c, backend=backend)
        right = logical_and(a, bc, backend=backend)

        backend.eval(left, right)

        np.testing.assert_array_equal(
            left,
            right,
            err_msg=f"AND not associative for a={a}, b={b}, c={c} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(arrays=triple_matching_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_or_associative(
        self,
        arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
        backend_name: str,
    ) -> None:
        """Property: (a ∨ b) ∨ c = a ∨ (b ∨ c) for all boolean tensors."""
        a, b, c = arrays
        backend = create_backend(backend_name)

        # Left: (a ∨ b) ∨ c
        ab = logical_or(a, b, backend=backend)
        left = logical_or(ab, c, backend=backend)

        # Right: a ∨ (b ∨ c)
        bc = logical_or(b, c, backend=backend)
        right = logical_or(a, bc, backend=backend)

        backend.eval(left, right)

        np.testing.assert_array_equal(
            left,
            right,
            err_msg=f"OR not associative for a={a}, b={b}, c={c} (backend={backend_name})",
        )


class TestDistributivity:
    """Property-based tests for distributivity: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c)."""

    @settings(deadline=None, max_examples=100)
    @given(arrays=triple_matching_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_and_over_or(
        self,
        arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
        backend_name: str,
    ) -> None:
        """Property: a ∧ (b ∨ c) = (a ∧ b) ∨ (a ∧ c) for all boolean tensors."""
        a, b, c = arrays
        backend = create_backend(backend_name)

        # Left: a ∧ (b ∨ c)
        bc = logical_or(b, c, backend=backend)
        left = logical_and(a, bc, backend=backend)

        # Right: (a ∧ b) ∨ (a ∧ c)
        ab = logical_and(a, b, backend=backend)
        ac = logical_and(a, c, backend=backend)
        right = logical_or(ab, ac, backend=backend)

        backend.eval(left, right)

        np.testing.assert_array_equal(
            left,
            right,
            err_msg=f"AND not distributive over OR for a={a}, b={b}, c={c} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(arrays=triple_matching_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_or_over_and(
        self,
        arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
        backend_name: str,
    ) -> None:
        """Property: a ∨ (b ∧ c) = (a ∨ b) ∧ (a ∨ c) for all boolean tensors."""
        a, b, c = arrays
        backend = create_backend(backend_name)

        # Left: a ∨ (b ∧ c)
        bc = logical_and(b, c, backend=backend)
        left = logical_or(a, bc, backend=backend)

        # Right: (a ∨ b) ∧ (a ∨ c)
        ab = logical_or(a, b, backend=backend)
        ac = logical_or(a, c, backend=backend)
        right = logical_and(ab, ac, backend=backend)

        backend.eval(left, right)

        np.testing.assert_array_equal(
            left,
            right,
            err_msg=f"OR not distributive over AND for a={a}, b={b}, c={c} (backend={backend_name})",
        )


class TestDeMorganLaws:
    """Property-based tests for De Morgan's laws."""

    @settings(deadline=None, max_examples=100)
    @given(arrays=matching_boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_demorgan_and(
        self,
        arrays: tuple[np.ndarray, np.ndarray],
        backend_name: str,
    ) -> None:
        """Property: ¬(a ∧ b) = ¬a ∨ ¬b for all boolean tensors."""
        a, b = arrays
        backend = create_backend(backend_name)

        # Left: ¬(a ∧ b)
        ab = logical_and(a, b, backend=backend)
        left = logical_not(ab, backend=backend)

        # Right: ¬a ∨ ¬b
        not_a = logical_not(a, backend=backend)
        not_b = logical_not(b, backend=backend)
        right = logical_or(not_a, not_b, backend=backend)

        backend.eval(left, right)

        np.testing.assert_allclose(
            left,
            right,
            atol=1e-7,
            err_msg=f"De Morgan's AND law violated for a={a}, b={b} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(arrays=matching_boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_demorgan_or(
        self,
        arrays: tuple[np.ndarray, np.ndarray],
        backend_name: str,
    ) -> None:
        """Property: ¬(a ∨ b) = ¬a ∧ ¬b for all boolean tensors."""
        a, b = arrays
        backend = create_backend(backend_name)

        # Left: ¬(a ∨ b)
        ab = logical_or(a, b, backend=backend)
        left = logical_not(ab, backend=backend)

        # Right: ¬a ∧ ¬b
        not_a = logical_not(a, backend=backend)
        not_b = logical_not(b, backend=backend)
        right = logical_and(not_a, not_b, backend=backend)

        backend.eval(left, right)

        np.testing.assert_allclose(
            left,
            right,
            atol=1e-7,
            err_msg=f"De Morgan's OR law violated for a={a}, b={b} (backend={backend_name})",
        )


class TestDoubleNegation:
    """Property-based tests for double negation: ¬¬a = a."""

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_double_negation(self, arr: np.ndarray, backend_name: str) -> None:
        """Property: ¬¬a = a for all boolean tensors (involution)."""
        backend = create_backend(backend_name)
        not_a = logical_not(arr, backend=backend)
        not_not_a = logical_not(not_a, backend=backend)
        backend.eval(not_not_a)

        np.testing.assert_allclose(
            not_not_a,
            arr,
            atol=1e-7,
            err_msg=f"Double negation failed for a={arr} (backend={backend_name})",
        )


class TestIdentityElements:
    """Property-based tests for identity elements."""

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_and_identity(self, arr: np.ndarray, backend_name: str) -> None:
        """Property: a ∧ 1 = a for all boolean tensors."""
        backend = create_backend(backend_name)
        ones = np.ones_like(arr)
        result = logical_and(arr, ones, backend=backend)
        backend.eval(result)

        np.testing.assert_array_equal(
            result,
            arr,
            err_msg=f"AND identity violated for a={arr} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_or_identity(self, arr: np.ndarray, backend_name: str) -> None:
        """Property: a ∨ 0 = a for all boolean tensors."""
        backend = create_backend(backend_name)
        zeros = np.zeros_like(arr)
        result = logical_or(arr, zeros, backend=backend)
        backend.eval(result)

        np.testing.assert_array_equal(
            result,
            arr,
            err_msg=f"OR identity violated for a={arr} (backend={backend_name})",
        )


class TestAnnihilators:
    """Property-based tests for annihilator elements."""

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_and_annihilator(self, arr: np.ndarray, backend_name: str) -> None:
        """Property: a ∧ 0 = 0 for all boolean tensors."""
        backend = create_backend(backend_name)
        zeros = np.zeros_like(arr)
        result = logical_and(arr, zeros, backend=backend)
        backend.eval(result)

        np.testing.assert_array_equal(
            result,
            zeros,
            err_msg=f"AND annihilator violated for a={arr} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_or_annihilator(self, arr: np.ndarray, backend_name: str) -> None:
        """Property: a ∨ 1 = 1 for all boolean tensors."""
        backend = create_backend(backend_name)
        ones = np.ones_like(arr)
        result = logical_or(arr, ones, backend=backend)
        backend.eval(result)

        np.testing.assert_array_equal(
            result,
            ones,
            err_msg=f"OR annihilator violated for a={arr} (backend={backend_name})",
        )


class TestIdempotence:
    """Property-based tests for idempotence: a ⊕ a = a."""

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_and_idempotent(self, arr: np.ndarray, backend_name: str) -> None:
        """Property: a ∧ a = a for all boolean tensors."""
        backend = create_backend(backend_name)
        result = logical_and(arr, arr, backend=backend)
        backend.eval(result)

        np.testing.assert_array_equal(
            result,
            arr,
            err_msg=f"AND not idempotent for a={arr} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_or_idempotent(self, arr: np.ndarray, backend_name: str) -> None:
        """Property: a ∨ a = a for all boolean tensors."""
        backend = create_backend(backend_name)
        result = logical_or(arr, arr, backend=backend)
        backend.eval(result)

        np.testing.assert_array_equal(
            result,
            arr,
            err_msg=f"OR not idempotent for a={arr} (backend={backend_name})",
        )


class TestAbsorption:
    """Property-based tests for absorption laws."""

    @settings(deadline=None, max_examples=100)
    @given(arrays=matching_boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_absorption_and_or(
        self,
        arrays: tuple[np.ndarray, np.ndarray],
        backend_name: str,
    ) -> None:
        """Property: a ∧ (a ∨ b) = a for all boolean tensors."""
        a, b = arrays
        backend = create_backend(backend_name)

        # a ∨ b
        ab = logical_or(a, b, backend=backend)
        # a ∧ (a ∨ b)
        result = logical_and(a, ab, backend=backend)
        backend.eval(result)

        np.testing.assert_array_equal(
            result,
            a,
            err_msg=f"Absorption (AND-OR) violated for a={a}, b={b} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(arrays=matching_boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_absorption_or_and(
        self,
        arrays: tuple[np.ndarray, np.ndarray],
        backend_name: str,
    ) -> None:
        """Property: a ∨ (a ∧ b) = a for all boolean tensors."""
        a, b = arrays
        backend = create_backend(backend_name)

        # a ∧ b
        ab = logical_and(a, b, backend=backend)
        # a ∨ (a ∧ b)
        result = logical_or(a, ab, backend=backend)
        backend.eval(result)

        np.testing.assert_array_equal(
            result,
            a,
            err_msg=f"Absorption (OR-AND) violated for a={a}, b={b} (backend={backend_name})",
        )


class TestComplementLaws:
    """Property-based tests for complement laws."""

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_non_contradiction(self, arr: np.ndarray, backend_name: str) -> None:
        """Property: a ∧ ¬a = 0 for all boolean tensors (law of non-contradiction)."""
        backend = create_backend(backend_name)
        not_a = logical_not(arr, backend=backend)
        result = logical_and(arr, not_a, backend=backend)
        backend.eval(result)

        zeros = np.zeros_like(arr)
        np.testing.assert_array_equal(
            result,
            zeros,
            err_msg=f"Non-contradiction violated for a={arr} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_excluded_middle(self, arr: np.ndarray, backend_name: str) -> None:
        """Property: a ∨ ¬a = 1 for all boolean tensors (law of excluded middle)."""
        backend = create_backend(backend_name)
        not_a = logical_not(arr, backend=backend)
        result = logical_or(arr, not_a, backend=backend)
        backend.eval(result)

        ones = np.ones_like(arr)
        np.testing.assert_array_equal(
            result,
            ones,
            err_msg=f"Excluded middle violated for a={arr} (backend={backend_name})",
        )


class TestImplicationProperties:
    """Property-based tests for implication properties."""

    @settings(deadline=None, max_examples=100)
    @given(arrays=matching_boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_material_implication(
        self,
        arrays: tuple[np.ndarray, np.ndarray],
        backend_name: str,
    ) -> None:
        """Property: a → b = ¬a ∨ b for all boolean tensors."""
        a, b = arrays
        backend = create_backend(backend_name)

        # Direct: a → b
        result_implies = logical_implies(a, b, backend=backend)

        # Equivalence: ¬a ∨ b
        not_a = logical_not(a, backend=backend)
        result_equiv = logical_or(not_a, b, backend=backend)

        backend.eval(result_implies, result_equiv)

        np.testing.assert_array_equal(
            result_implies,
            result_equiv,
            err_msg=f"Material implication violated for a={a}, b={b} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(arrays=matching_boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_contrapositive(
        self,
        arrays: tuple[np.ndarray, np.ndarray],
        backend_name: str,
    ) -> None:
        """Property: a → b = ¬b → ¬a for all boolean tensors."""
        a, b = arrays
        backend = create_backend(backend_name)

        # Forward: a → b
        result_forward = logical_implies(a, b, backend=backend)

        # Contrapositive: ¬b → ¬a
        not_a = logical_not(a, backend=backend)
        not_b = logical_not(b, backend=backend)
        result_contra = logical_implies(not_b, not_a, backend=backend)

        backend.eval(result_forward, result_contra)

        np.testing.assert_array_equal(
            result_forward,
            result_contra,
            err_msg=f"Contrapositive violated for a={a}, b={b} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_self_implication(self, arr: np.ndarray, backend_name: str) -> None:
        """Property: a → a = 1 for all boolean tensors (tautology)."""
        backend = create_backend(backend_name)
        result = logical_implies(arr, arr, backend=backend)
        backend.eval(result)

        ones = np.ones_like(arr)
        np.testing.assert_array_equal(
            result,
            ones,
            err_msg=f"Self-implication violated for a={arr} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(arrays=triple_matching_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_transitivity(
        self,
        arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
        backend_name: str,
    ) -> None:
        """Property: ((a → b) ∧ (b → c)) → (a → c) = 1 (chain rule)."""
        a, b, c = arrays
        backend = create_backend(backend_name)

        # (a → b) ∧ (b → c)
        ab = logical_implies(a, b, backend=backend)
        bc = logical_implies(b, c, backend=backend)
        premise = logical_and(ab, bc, backend=backend)

        # a → c
        ac = logical_implies(a, c, backend=backend)

        # ((a → b) ∧ (b → c)) → (a → c)
        result = logical_implies(premise, ac, backend=backend)
        backend.eval(result)

        ones = np.ones_like(a)
        np.testing.assert_array_equal(
            result,
            ones,
            err_msg=f"Transitivity violated for a={a}, b={b}, c={c} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(arrays=matching_boolean_arrays(), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_modus_ponens(
        self,
        arrays: tuple[np.ndarray, np.ndarray],
        backend_name: str,
    ) -> None:
        """Property: (a ∧ (a → b)) → b = 1 (modus ponens tautology)."""
        a, b = arrays
        backend = create_backend(backend_name)

        # a ∧ (a → b)
        ab_implies = logical_implies(a, b, backend=backend)
        premise = logical_and(a, ab_implies, backend=backend)

        # (a ∧ (a → b)) → b
        result = logical_implies(premise, b, backend=backend)
        backend.eval(result)

        ones = np.ones_like(a)
        np.testing.assert_array_equal(
            result,
            ones,
            err_msg=f"Modus ponens violated for a={a}, b={b} (backend={backend_name})",
        )


class TestCrossBackendConsistency:
    """Property-based tests ensuring consistency across backends."""

    @settings(deadline=None, max_examples=50)
    @given(arrays=matching_boolean_arrays())
    def test_and_cross_backend(
        self,
        arrays: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify AND produces identical results across NumPy and MLX."""
        a, b = arrays
        backend_numpy = create_backend("numpy")
        backend_mlx = create_backend("mlx")

        result_numpy = logical_and(a, b, backend=backend_numpy)
        result_mlx = logical_and(a, b, backend=backend_mlx)
        backend_mlx.eval(result_mlx)

        np.testing.assert_array_equal(
            result_numpy,
            result_mlx,
            err_msg=f"AND inconsistent across backends for a={a}, b={b}",
        )

    @settings(deadline=None, max_examples=50)
    @given(arrays=matching_boolean_arrays())
    def test_or_cross_backend(
        self,
        arrays: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify OR produces identical results across NumPy and MLX."""
        a, b = arrays
        backend_numpy = create_backend("numpy")
        backend_mlx = create_backend("mlx")

        result_numpy = logical_or(a, b, backend=backend_numpy)
        result_mlx = logical_or(a, b, backend=backend_mlx)
        backend_mlx.eval(result_mlx)

        np.testing.assert_array_equal(
            result_numpy,
            result_mlx,
            err_msg=f"OR inconsistent across backends for a={a}, b={b}",
        )

    @settings(deadline=None, max_examples=50)
    @given(arr=boolean_arrays())
    def test_not_cross_backend(self, arr: np.ndarray) -> None:
        """Verify NOT produces identical results across NumPy and MLX."""
        backend_numpy = create_backend("numpy")
        backend_mlx = create_backend("mlx")

        result_numpy = logical_not(arr, backend=backend_numpy)
        result_mlx = logical_not(arr, backend=backend_mlx)
        backend_mlx.eval(result_mlx)

        np.testing.assert_allclose(
            result_numpy,
            result_mlx,
            atol=1e-7,
            err_msg=f"NOT inconsistent across backends for a={arr}",
        )

    @settings(deadline=None, max_examples=50)
    @given(arrays=matching_boolean_arrays())
    def test_implies_cross_backend(
        self,
        arrays: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify IMPLIES produces identical results across NumPy and MLX."""
        a, b = arrays
        backend_numpy = create_backend("numpy")
        backend_mlx = create_backend("mlx")

        result_numpy = logical_implies(a, b, backend=backend_numpy)
        result_mlx = logical_implies(a, b, backend=backend_mlx)
        backend_mlx.eval(result_mlx)

        np.testing.assert_array_equal(
            result_numpy,
            result_mlx,
            err_msg=f"IMPLIES inconsistent across backends for a={a}, b={b}",
        )


class TestQuantifierTautologies:
    """Property-based tests for quantifier tautologies and contradictions."""

    @settings(deadline=None, max_examples=100)
    @given(
        arr_shape=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=3),
        backend_name=st.sampled_from(["numpy", "mlx"]),
    )
    def test_exists_true_tautology(
        self,
        arr_shape: list[int],
        backend_name: str,
    ) -> None:
        """Property: ∃x.True = True (tautology)."""
        backend = create_backend(backend_name)
        shape = tuple(arr_shape)
        ones = backend.ones(shape)

        # ∃x.True over any axis should be True
        result = exists(ones, axis=0, backend=backend)
        backend.eval(result)

        # Result should be 1.0 (True)
        result_np = np.asarray(result)
        assert np.all(result_np == 1.0), (
            f"∃x.True != True for shape={shape} (backend={backend_name})"
        )

    @settings(deadline=None, max_examples=100)
    @given(
        arr_shape=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=3),
        backend_name=st.sampled_from(["numpy", "mlx"]),
    )
    def test_forall_false_contradiction(
        self,
        arr_shape: list[int],
        backend_name: str,
    ) -> None:
        """Property: ∀x.False = False (contradiction)."""
        backend = create_backend(backend_name)
        shape = tuple(arr_shape)
        zeros = backend.zeros(shape)

        # ∀x.False over any axis should be False
        result = forall(zeros, axis=0, backend=backend)
        backend.eval(result)

        # Result should be 0.0 (False)
        result_np = np.asarray(result)
        assert np.all(result_np == 0.0), (
            f"∀x.False != False for shape={shape} (backend={backend_name})"
        )

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(min_size=2, max_size=20), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_self_existence(self, arr: np.ndarray, backend_name: str) -> None:
        """Property: ∃x.P(x) = True if any P(x) is true."""
        backend = create_backend(backend_name)

        result = exists(arr, axis=None, backend=backend)
        backend.eval(result)

        # Should be True iff at least one element is true
        expected = 1.0 if np.any(arr > 0) else 0.0
        np.testing.assert_almost_equal(
            float(result),
            expected,
            decimal=5,
            err_msg=f"∃x.P(x) incorrect for arr={arr} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(min_size=2, max_size=20), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_universal_requirement(self, arr: np.ndarray, backend_name: str) -> None:
        """Property: ∀x.P(x) = True if all P(x) are true."""
        backend = create_backend(backend_name)

        result = forall(arr, axis=None, backend=backend)
        backend.eval(result)

        # Should be True iff all elements are true
        expected = 1.0 if np.all(arr == 1.0) else 0.0
        np.testing.assert_almost_equal(
            float(result),
            expected,
            decimal=5,
            err_msg=f"∀x.P(x) incorrect for arr={arr} (backend={backend_name})",
        )


class TestQuantifierNegation:
    """Property-based tests for quantifier negation equivalence."""

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(min_size=2, max_size=15), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_negation_equivalence_exists(
        self,
        arr: np.ndarray,
        backend_name: str,
    ) -> None:
        """Property: ¬∃x.P(x) = ∀x.¬P(x) for all boolean tensors."""
        backend = create_backend(backend_name)

        # Left: ¬∃x.P(x)
        exists_result = exists(arr, axis=None, backend=backend)
        left = logical_not(exists_result, backend=backend)

        # Right: ∀x.¬P(x)
        not_arr = logical_not(arr, backend=backend)
        right = forall(not_arr, axis=None, backend=backend)

        backend.eval(left, right)

        np.testing.assert_almost_equal(
            float(left),
            float(right),
            decimal=5,
            err_msg=f"¬∃x.P(x) != ∀x.¬P(x) for arr={arr} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(min_size=2, max_size=15), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_negation_equivalence_forall(
        self,
        arr: np.ndarray,
        backend_name: str,
    ) -> None:
        """Property: ¬∀x.P(x) = ∃x.¬P(x) for all boolean tensors."""
        backend = create_backend(backend_name)

        # Left: ¬∀x.P(x)
        forall_result = forall(arr, axis=None, backend=backend)
        left = logical_not(forall_result, backend=backend)

        # Right: ∃x.¬P(x)
        not_arr = logical_not(arr, backend=backend)
        right = exists(not_arr, axis=None, backend=backend)

        backend.eval(left, right)

        np.testing.assert_almost_equal(
            float(left),
            float(right),
            decimal=5,
            err_msg=f"¬∀x.P(x) != ∃x.¬P(x) for arr={arr} (backend={backend_name})",
        )


class TestSoftQuantifierProperties:
    """Property-based tests for soft (differentiable) quantifier properties."""

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(min_size=2, max_size=15), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_soft_exists_bounds(self, arr: np.ndarray, backend_name: str) -> None:
        """Property: soft_exists(P) ∈ [0, 1] for all P."""
        backend = create_backend(backend_name)

        result = soft_exists(arr, axis=None, backend=backend)
        backend.eval(result)

        result_val = float(result)
        assert 0.0 <= result_val <= 1.0, (
            f"soft_exists out of bounds: {result_val} for arr={arr} (backend={backend_name})"
        )

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(min_size=2, max_size=15), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_soft_forall_bounds(self, arr: np.ndarray, backend_name: str) -> None:
        """Property: soft_forall(P) ∈ [0, 1] for all P."""
        backend = create_backend(backend_name)

        result = soft_forall(arr, axis=None, backend=backend)
        backend.eval(result)

        result_val = float(result)
        assert 0.0 <= result_val <= 1.0, (
            f"soft_forall out of bounds: {result_val} for arr={arr} (backend={backend_name})"
        )

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(min_size=2, max_size=15), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_soft_exists_upper_bound(self, arr: np.ndarray, backend_name: str) -> None:
        """Property: soft_exists(P) = max(P) ≤ hard exists result."""
        backend = create_backend(backend_name)

        soft_result = soft_exists(arr, axis=None, backend=backend)
        hard_result = exists(arr, axis=None, backend=backend)
        backend.eval(soft_result, hard_result)

        # Soft exists should equal max value
        expected_max = np.max(arr)
        np.testing.assert_almost_equal(
            float(soft_result),
            expected_max,
            decimal=5,
            err_msg=f"soft_exists != max for arr={arr} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(arr=boolean_arrays(min_size=2, max_size=15), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_soft_forall_lower_bound(self, arr: np.ndarray, backend_name: str) -> None:
        """Property: soft_forall(P) = min(P) ≤ hard forall result."""
        backend = create_backend(backend_name)

        soft_result = soft_forall(arr, axis=None, backend=backend)
        hard_result = forall(arr, axis=None, backend=backend)
        backend.eval(soft_result, hard_result)

        # Soft forall should equal min value
        expected_min = np.min(arr)
        np.testing.assert_almost_equal(
            float(soft_result),
            expected_min,
            decimal=5,
            err_msg=f"soft_forall != min for arr={arr} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=50)
    @given(arr=boolean_arrays(min_size=2, max_size=10, allow_multidim=False))
    def test_soft_exists_gradient_mlx(self, arr: np.ndarray) -> None:
        """Property: soft_exists is differentiable (gradient exists for MLX)."""
        import mlx.core as mx

        backend = create_backend("mlx")

        # Convert NumPy array to MLX array for gradient computation
        mlx_arr = mx.array(arr)

        # Define function for gradient computation
        def loss_fn(x):
            return soft_exists(x, axis=None, backend=backend)

        # Compute gradient
        grad_fn = backend.grad(loss_fn)
        gradients = grad_fn(mlx_arr)
        backend.eval(gradients)

        # Gradient should exist and have same shape as input
        assert gradients.shape == arr.shape, (
            f"Gradient shape mismatch: {gradients.shape} != {arr.shape}"
        )

        # Gradient should be finite
        grad_np = np.asarray(gradients)
        assert np.all(np.isfinite(grad_np)), (
            f"Non-finite gradients for arr={arr}"
        )

    @settings(deadline=None, max_examples=50)
    @given(arr=boolean_arrays(min_size=2, max_size=10, allow_multidim=False))
    def test_soft_forall_gradient_mlx(self, arr: np.ndarray) -> None:
        """Property: soft_forall is differentiable (gradient exists for MLX)."""
        import mlx.core as mx

        backend = create_backend("mlx")

        # Convert NumPy array to MLX array for gradient computation
        mlx_arr = mx.array(arr)

        # Define function for gradient computation
        def loss_fn(x):
            return soft_forall(x, axis=None, backend=backend)

        # Compute gradient
        grad_fn = backend.grad(loss_fn)
        gradients = grad_fn(mlx_arr)
        backend.eval(gradients)

        # Gradient should exist and have same shape as input
        assert gradients.shape == arr.shape, (
            f"Gradient shape mismatch: {gradients.shape} != {arr.shape}"
        )

        # Gradient should be finite
        grad_np = np.asarray(gradients)
        assert np.all(np.isfinite(grad_np)), (
            f"Non-finite gradients for arr={arr}"
        )


class TestQuantifierEdgeCases:
    """Property-based tests for quantifier edge cases."""

    @settings(deadline=None, max_examples=100)
    @given(value=st.sampled_from([0.0, 1.0]), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_single_element_exists(self, value: float, backend_name: str) -> None:
        """Property: ∃x.P(x) with single element P = P."""
        backend = create_backend(backend_name)
        arr = np.array([value])

        result = exists(arr, axis=None, backend=backend)
        backend.eval(result)

        expected = 1.0 if value > 0 else 0.0
        np.testing.assert_almost_equal(
            float(result),
            expected,
            decimal=5,
            err_msg=f"Single element exists failed for value={value} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(value=st.sampled_from([0.0, 1.0]), backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_single_element_forall(self, value: float, backend_name: str) -> None:
        """Property: ∀x.P(x) with single element P = P."""
        backend = create_backend(backend_name)
        arr = np.array([value])

        result = forall(arr, axis=None, backend=backend)
        backend.eval(result)

        expected = 1.0 if value == 1.0 else 0.0
        np.testing.assert_almost_equal(
            float(result),
            expected,
            decimal=5,
            err_msg=f"Single element forall failed for value={value} (backend={backend_name})",
        )

    @settings(deadline=None, max_examples=100)
    @given(backend_name=st.sampled_from(["numpy", "mlx"]))
    def test_empty_axis_quantification(self, backend_name: str) -> None:
        """Property: Quantification over all axes reduces to scalar."""
        backend = create_backend(backend_name)
        arr = np.array([[1.0, 0.0], [1.0, 1.0]])

        # Quantify over all axes (axis=None)
        exists_result = exists(arr, axis=None, backend=backend)
        forall_result = forall(arr, axis=None, backend=backend)
        backend.eval(exists_result, forall_result)

        # Results should be scalars
        assert np.asarray(exists_result).shape == (), (
            f"exists with axis=None should be scalar (backend={backend_name})"
        )
        assert np.asarray(forall_result).shape == (), (
            f"forall with axis=None should be scalar (backend={backend_name})"
        )


class TestQuantifierCrossBackend:
    """Property-based tests ensuring quantifier consistency across backends."""

    @settings(deadline=None, max_examples=50)
    @given(arr=boolean_arrays(min_size=2, max_size=15))
    def test_exists_cross_backend(self, arr: np.ndarray) -> None:
        """Verify exists produces identical results across NumPy and MLX."""
        backend_numpy = create_backend("numpy")
        backend_mlx = create_backend("mlx")

        result_numpy = exists(arr, axis=None, backend=backend_numpy)
        result_mlx = exists(arr, axis=None, backend=backend_mlx)
        backend_mlx.eval(result_mlx)

        np.testing.assert_almost_equal(
            float(result_numpy),
            float(result_mlx),
            decimal=5,
            err_msg=f"exists inconsistent across backends for arr={arr}",
        )

    @settings(deadline=None, max_examples=50)
    @given(arr=boolean_arrays(min_size=2, max_size=15))
    def test_forall_cross_backend(self, arr: np.ndarray) -> None:
        """Verify forall produces identical results across NumPy and MLX."""
        backend_numpy = create_backend("numpy")
        backend_mlx = create_backend("mlx")

        result_numpy = forall(arr, axis=None, backend=backend_numpy)
        result_mlx = forall(arr, axis=None, backend=backend_mlx)
        backend_mlx.eval(result_mlx)

        np.testing.assert_almost_equal(
            float(result_numpy),
            float(result_mlx),
            decimal=5,
            err_msg=f"forall inconsistent across backends for arr={arr}",
        )

    @settings(deadline=None, max_examples=50)
    @given(arr=boolean_arrays(min_size=2, max_size=15))
    def test_soft_exists_cross_backend(self, arr: np.ndarray) -> None:
        """Verify soft_exists produces identical results across NumPy and MLX."""
        backend_numpy = create_backend("numpy")
        backend_mlx = create_backend("mlx")

        result_numpy = soft_exists(arr, axis=None, backend=backend_numpy)
        result_mlx = soft_exists(arr, axis=None, backend=backend_mlx)
        backend_mlx.eval(result_mlx)

        np.testing.assert_almost_equal(
            float(result_numpy),
            float(result_mlx),
            decimal=5,
            err_msg=f"soft_exists inconsistent across backends for arr={arr}",
        )

    @settings(deadline=None, max_examples=50)
    @given(arr=boolean_arrays(min_size=2, max_size=15))
    def test_soft_forall_cross_backend(self, arr: np.ndarray) -> None:
        """Verify soft_forall produces identical results across NumPy and MLX."""
        backend_numpy = create_backend("numpy")
        backend_mlx = create_backend("mlx")

        result_numpy = soft_forall(arr, axis=None, backend=backend_numpy)
        result_mlx = soft_forall(arr, axis=None, backend=backend_mlx)
        backend_mlx.eval(result_mlx)

        np.testing.assert_almost_equal(
            float(result_numpy),
            float(result_mlx),
            decimal=5,
            err_msg=f"soft_forall inconsistent across backends for arr={arr}",
        )
