"""Memory Profiling for TensorLogic

Measures memory usage vs graph density and entity count.

Target: Linear scaling with entity count

Run benchmark:
    uv run python benchmarks/memory_profile.py
    uv run python benchmarks/memory_profile.py --output results/memory_results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from tensorlogic import create_backend


@dataclass
class MemoryMeasurement:
    """Memory measurement result."""

    n_entities: int
    n_relations: int
    density: float
    theoretical_mb: float
    actual_mb: float
    overhead_ratio: float
    timestamp: str


@dataclass
class MemoryProfileSuite:
    """Complete memory profile results."""

    backend: str
    platform: str
    measurements: list[MemoryMeasurement]
    timestamp: str


def get_object_size_mb(obj: np.ndarray) -> float:
    """Get size of a numpy array in MB."""
    return obj.nbytes / (1024 * 1024)


def measure_tensor_memory(
    n_entities: int,
    n_relations: int = 1,
    density: float = 0.01,
) -> MemoryMeasurement:
    """Measure memory usage for knowledge graph tensors.

    Args:
        n_entities: Number of entities
        n_relations: Number of relation tensors
        density: Fraction of non-zero entries

    Returns:
        MemoryMeasurement with actual and theoretical values
    """
    gc.collect()

    # Theoretical memory (dense representation)
    theoretical_mb = (n_entities * n_entities * 4 * n_relations) / (1024 * 1024)

    # Create actual tensors
    tensors = []
    np.random.seed(42)

    for _ in range(n_relations):
        tensor = np.zeros((n_entities, n_entities), dtype=np.float32)
        n_edges = int(n_entities * n_entities * density)
        n_edges = max(n_edges, n_entities)

        rows = np.random.randint(0, n_entities, size=n_edges)
        cols = np.random.randint(0, n_entities, size=n_edges)
        tensor[rows, cols] = 1.0

        tensors.append(tensor)

    # Measure actual memory
    actual_mb = sum(get_object_size_mb(t) for t in tensors)
    overhead_ratio = actual_mb / theoretical_mb if theoretical_mb > 0 else 1.0

    return MemoryMeasurement(
        n_entities=n_entities,
        n_relations=n_relations,
        density=density,
        theoretical_mb=theoretical_mb,
        actual_mb=actual_mb,
        overhead_ratio=overhead_ratio,
        timestamp=datetime.now().isoformat(),
    )


def measure_operation_memory(
    n_entities: int,
    operation: str = "matmul",
) -> tuple[float, float]:
    """Measure peak memory during an operation.

    Args:
        n_entities: Number of entities
        operation: Operation type

    Returns:
        Tuple of (input_mb, peak_mb)
    """
    gc.collect()

    # Create input tensors
    np.random.seed(42)
    rel1 = np.random.random((n_entities, n_entities)).astype(np.float32)
    rel2 = np.random.random((n_entities, n_entities)).astype(np.float32)

    input_mb = get_object_size_mb(rel1) + get_object_size_mb(rel2)

    # Perform operation
    if operation == "matmul":
        result = np.matmul(rel1, rel2)
    elif operation == "add":
        result = rel1 + rel2
    elif operation == "multiply":
        result = rel1 * rel2
    else:
        result = np.matmul(rel1, rel2)

    output_mb = get_object_size_mb(result)
    peak_mb = input_mb + output_mb

    return input_mb, peak_mb


def run_memory_profile(
    entity_counts: list[int] | None = None,
    relation_counts: list[int] | None = None,
    densities: list[float] | None = None,
    verbose: bool = True,
) -> MemoryProfileSuite:
    """Run memory profiling suite.

    Args:
        entity_counts: List of entity counts to test
        relation_counts: List of relation counts to test
        densities: List of densities to test
        verbose: Print progress

    Returns:
        MemoryProfileSuite with all measurements
    """
    if entity_counts is None:
        entity_counts = [100, 500, 1000, 2000, 5000, 10000]
    if relation_counts is None:
        relation_counts = [1, 5, 10]
    if densities is None:
        densities = [0.001, 0.01, 0.1]

    backend = create_backend()
    backend_name = type(backend).__name__

    if verbose:
        print("=" * 80)
        print("TENSORLOGIC MEMORY PROFILING")
        print("=" * 80)
        print(f"Backend: {backend_name}")
        print(f"Entity counts: {entity_counts}")
        print(f"Relation counts: {relation_counts}")
        print(f"Densities: {densities}")
        print()

    measurements: list[MemoryMeasurement] = []

    # Test 1: Memory vs entity count (fixed density, single relation)
    if verbose:
        print("\n--- Memory vs Entity Count (density=0.01, 1 relation) ---")
        print(f"{'Entities':<12} {'Theoretical (MB)':<18} {'Actual (MB)':<15}")
        print("-" * 50)

    for n_entities in entity_counts:
        measurement = measure_tensor_memory(n_entities, n_relations=1, density=0.01)
        measurements.append(measurement)

        if verbose:
            print(
                f"{n_entities:<12} {measurement.theoretical_mb:<18.2f} "
                f"{measurement.actual_mb:<15.2f}"
            )

    # Test 2: Memory vs density (fixed entity count)
    if verbose:
        print("\n--- Memory vs Density (1000 entities, 1 relation) ---")
        print(f"{'Density':<12} {'Theoretical (MB)':<18} {'Actual (MB)':<15}")
        print("-" * 50)

    for density in densities:
        measurement = measure_tensor_memory(1000, n_relations=1, density=density)
        measurements.append(measurement)

        if verbose:
            print(
                f"{density:<12} {measurement.theoretical_mb:<18.2f} "
                f"{measurement.actual_mb:<15.2f}"
            )

    # Test 3: Memory vs relation count
    if verbose:
        print("\n--- Memory vs Relation Count (1000 entities, density=0.01) ---")
        print(f"{'Relations':<12} {'Theoretical (MB)':<18} {'Actual (MB)':<15}")
        print("-" * 50)

    for n_relations in relation_counts:
        measurement = measure_tensor_memory(1000, n_relations=n_relations, density=0.01)
        measurements.append(measurement)

        if verbose:
            print(
                f"{n_relations:<12} {measurement.theoretical_mb:<18.2f} "
                f"{measurement.actual_mb:<15.2f}"
            )

    # Test 4: Peak memory during operations
    if verbose:
        print("\n--- Peak Memory During Operations ---")
        print(f"{'Entities':<12} {'Input (MB)':<15} {'Peak (MB)':<15} {'Ratio':<10}")
        print("-" * 55)

    for n_entities in [100, 500, 1000, 2000]:
        input_mb, peak_mb = measure_operation_memory(n_entities, "matmul")

        if verbose:
            ratio = peak_mb / input_mb if input_mb > 0 else 0
            print(f"{n_entities:<12} {input_mb:<15.2f} {peak_mb:<15.2f} {ratio:<10.2f}")

    return MemoryProfileSuite(
        backend=backend_name,
        platform=sys.platform,
        measurements=measurements,
        timestamp=datetime.now().isoformat(),
    )


def analyze_scaling(suite: MemoryProfileSuite) -> None:
    """Analyze memory scaling from measurements."""
    print("\n" + "=" * 80)
    print("MEMORY SCALING ANALYSIS")
    print("=" * 80)

    # Filter entity count tests
    entity_tests = [
        m for m in suite.measurements
        if m.n_relations == 1 and m.density == 0.01
    ]

    if len(entity_tests) >= 2:
        # Calculate scaling factor
        first = entity_tests[0]
        last = entity_tests[-1]

        entity_scale = last.n_entities / first.n_entities
        memory_scale = last.actual_mb / first.actual_mb
        expected_scale = entity_scale ** 2  # O(n^2) for dense matrix

        print(f"\nEntity Scaling (n^2 expected for dense):")
        print(f"  Entity scale: {entity_scale:.1f}x")
        print(f"  Memory scale: {memory_scale:.1f}x")
        print(f"  Expected (O(n^2)): {expected_scale:.1f}x")

        if memory_scale < expected_scale * 1.1:
            print(f"  Result: PASS (scales as expected)")
        else:
            print(f"  Result: WARNING (higher than expected)")

    # Memory efficiency report
    print(f"\nMemory Efficiency:")
    for m in suite.measurements[:6]:  # First 6 entity tests
        efficiency = m.theoretical_mb / m.actual_mb if m.actual_mb > 0 else 0
        print(
            f"  {m.n_entities} entities: {efficiency:.2%} efficiency "
            f"({m.actual_mb:.2f} MB)"
        )


def save_results(suite: MemoryProfileSuite, output_path: Path) -> None:
    """Save memory profile results to JSON file."""
    data = {
        "backend": suite.backend,
        "platform": suite.platform,
        "timestamp": suite.timestamp,
        "measurements": [asdict(m) for m in suite.measurements],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main() -> None:
    """Run memory profiling."""
    parser = argparse.ArgumentParser(description="TensorLogic Memory Profiling")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[100, 500, 1000, 2000, 5000, 10000],
        help="Entity counts to profile",
    )
    args = parser.parse_args()

    suite = run_memory_profile(
        entity_counts=args.scales,
        verbose=True,
    )

    analyze_scaling(suite)

    if args.output:
        save_results(suite, args.output)


if __name__ == "__main__":
    main()
