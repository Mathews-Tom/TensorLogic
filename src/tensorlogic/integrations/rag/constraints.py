"""Logical constraint filtering for RAG retrieval.

Provides constraint-based filtering using TensorLogic operations.
Supports both hard (boolean) and soft (fuzzy) constraint evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tensorlogic.backends import create_backend, TensorBackend
from tensorlogic.core import logical_and, logical_or, logical_not, exists, forall


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert array-like to numpy array."""
    if isinstance(arr, np.ndarray):
        return arr
    # Handle MLX arrays
    if hasattr(arr, "tolist"):
        return np.array(arr.tolist())
    return np.asarray(arr)


@dataclass
class Constraint:
    """A logical constraint for filtering.

    Attributes:
        name: Constraint identifier
        predicate: Predicate name or expression
        negated: Whether to negate the constraint
        quantifier: 'exists', 'forall', or None
        variables: Variables in the constraint
    """

    name: str
    predicate: str
    negated: bool = False
    quantifier: str | None = None
    variables: list[str] | None = None


class ConstraintFilter:
    """Filter entities using logical constraints.

    Applies TensorLogic operations to filter candidates based on
    logical predicates with support for composition and quantification.

    Args:
        backend: TensorBackend instance
        temperature: Softness of constraint evaluation (0=hard)

    Example:
        >>> filter = ConstraintFilter()
        >>> filter.add_predicate("IsActive", activity_mask)
        >>> filter.add_predicate("HasPermission", permission_matrix)
        >>> filtered = filter.apply(
        ...     candidates=entity_scores,
        ...     constraints=["IsActive", "HasPermission"],
        ... )
    """

    def __init__(
        self,
        backend: TensorBackend | None = None,
        temperature: float = 0.0,
    ) -> None:
        """Initialize constraint filter.

        Args:
            backend: TensorBackend instance (auto-detected if None)
            temperature: Constraint softness (0=boolean, >0=fuzzy)
        """
        self.backend = backend or create_backend()
        self.temperature = temperature
        self._predicates: dict[str, np.ndarray] = {}
        self._relations: dict[str, np.ndarray] = {}

    def add_predicate(self, name: str, mask: Any) -> None:
        """Add a unary predicate (entity mask).

        Args:
            name: Predicate name
            mask: Boolean mask [num_entities]
        """
        self._predicates[name] = np.asarray(mask, dtype=np.float32)

    def add_relation(self, name: str, matrix: Any) -> None:
        """Add a binary relation.

        Args:
            name: Relation name
            matrix: Relation matrix [num_entities, num_entities]
        """
        self._relations[name] = np.asarray(matrix, dtype=np.float32)

    def evaluate_predicate(self, name: str, negated: bool = False) -> np.ndarray:
        """Evaluate a single predicate.

        Args:
            name: Predicate name
            negated: Whether to negate the result

        Returns:
            Predicate mask [num_entities]
        """
        if name not in self._predicates:
            raise ValueError(f"Unknown predicate: {name}")

        mask = self._predicates[name]
        if negated:
            mask = _to_numpy(logical_not(mask, backend=self.backend))
        return mask

    def evaluate_relation_exists(
        self,
        relation_name: str,
        source_mask: Any,
        negated: bool = False,
    ) -> np.ndarray:
        """Evaluate existential quantification over a relation.

        Computes: exists y: source(y) AND relation(y, x)

        Args:
            relation_name: Relation name
            source_mask: Source entity mask
            negated: Whether to negate the result

        Returns:
            Target entity mask [num_entities]
        """
        if relation_name not in self._relations:
            raise ValueError(f"Unknown relation: {relation_name}")

        relation = self._relations[relation_name]
        source = np.asarray(source_mask, dtype=np.float32)

        # Broadcast source mask and compute exists
        num_entities = relation.shape[0]
        source_expanded = source[:, np.newaxis] * np.ones((1, num_entities))
        conjunction = _to_numpy(
            logical_and(source_expanded, relation, backend=self.backend)
        )

        # EXISTS: any source entity has relation to target
        result = _to_numpy(exists(conjunction, axis=0, backend=self.backend))

        if negated:
            result = _to_numpy(logical_not(result, backend=self.backend))

        return result

    def evaluate_relation_forall(
        self,
        relation_name: str,
        source_mask: Any,
        negated: bool = False,
    ) -> np.ndarray:
        """Evaluate universal quantification over a relation.

        Computes: forall y: source(y) -> relation(y, x)

        Args:
            relation_name: Relation name
            source_mask: Source entity mask
            negated: Whether to negate the result

        Returns:
            Target entity mask [num_entities]
        """
        if relation_name not in self._relations:
            raise ValueError(f"Unknown relation: {relation_name}")

        relation = self._relations[relation_name]
        source = np.asarray(source_mask, dtype=np.float32)

        # Broadcast source mask
        num_entities = relation.shape[0]
        source_expanded = source[:, np.newaxis] * np.ones((1, num_entities))

        # FORALL: source -> relation (equivalently: NOT source OR relation)
        not_source = _to_numpy(logical_not(source_expanded, backend=self.backend))
        implication = _to_numpy(logical_or(not_source, relation, backend=self.backend))

        # FORALL: all implications hold
        result = _to_numpy(forall(implication, axis=0, backend=self.backend))

        if negated:
            result = _to_numpy(logical_not(result, backend=self.backend))

        return result

    def compose_and(self, *masks: Any) -> np.ndarray:
        """Compose masks with logical AND.

        Args:
            *masks: Variable number of masks

        Returns:
            Conjunction of all masks
        """
        if len(masks) == 0:
            raise ValueError("At least one mask required")

        result = np.asarray(masks[0], dtype=np.float32)
        for mask in masks[1:]:
            result = _to_numpy(logical_and(result, mask, backend=self.backend))
        return result

    def compose_or(self, *masks: Any) -> np.ndarray:
        """Compose masks with logical OR.

        Args:
            *masks: Variable number of masks

        Returns:
            Disjunction of all masks
        """
        if len(masks) == 0:
            raise ValueError("At least one mask required")

        result = np.asarray(masks[0], dtype=np.float32)
        for mask in masks[1:]:
            result = _to_numpy(logical_or(result, mask, backend=self.backend))
        return result

    def apply(
        self,
        candidates: Any,
        predicate_names: list[str] | None = None,
        composition: str = "and",
    ) -> np.ndarray:
        """Apply constraints to filter candidates.

        Args:
            candidates: Candidate scores [num_entities]
            predicate_names: Predicates to apply
            composition: How to compose ('and' or 'or')

        Returns:
            Filtered scores [num_entities]
        """
        candidates = np.asarray(candidates, dtype=np.float32)

        if predicate_names is None or len(predicate_names) == 0:
            return candidates

        # Evaluate predicates
        masks = [self.evaluate_predicate(name) for name in predicate_names]

        # Compose
        if composition == "and":
            combined_mask = self.compose_and(*masks)
        elif composition == "or":
            combined_mask = self.compose_or(*masks)
        else:
            raise ValueError(f"Unknown composition: {composition}")

        # Apply temperature
        if self.temperature > 0:
            # Soft filtering: multiply by mask (allows partial satisfaction)
            return candidates * combined_mask
        else:
            # Hard filtering: zero out non-matching
            return candidates * (combined_mask > 0.5).astype(np.float32)

    def filter_by_relation_chain(
        self,
        candidates: Any,
        chain: list[str],
        source_mask: Any,
    ) -> np.ndarray:
        """Filter candidates reachable through a relation chain.

        Args:
            candidates: Candidate scores [num_entities]
            chain: List of relation names to traverse
            source_mask: Starting entity mask

        Returns:
            Filtered scores [num_entities]
        """
        candidates = np.asarray(candidates, dtype=np.float32)
        current_mask = np.asarray(source_mask, dtype=np.float32)

        for relation_name in chain:
            current_mask = self.evaluate_relation_exists(relation_name, current_mask)

        return candidates * current_mask
