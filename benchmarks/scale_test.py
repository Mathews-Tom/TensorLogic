"""Scale Benchmarks for TensorLogic

Measures query latency vs entity count across different operation types.

Target: <100ms for 3-hop patterns on 100K entities

Run benchmark:
    uv run python benchmarks/scale_test.py
    uv run python benchmarks/scale_test.py --output results/scale_results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from tensorlogic import (
    create_backend,
    create_strategy,
    logical_and,
    logical_or,
    exists,
    forall,
)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    operation: str
    n_entities: int
    n_relations: int
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    memory_mb: float
    timestamp: str


@dataclass
class ScaleBenchmarkSuite:
    """Complete benchmark suite results."""

    backend: str
    platform: str
    results: list[BenchmarkResult]
    timestamp: str


def generate_sparse_relation(
    n_entities: int,
    density: float = 0.01,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a sparse relation tensor.

    Args:
        n_entities: Number of entities
        density: Fraction of non-zero entries
        seed: Random seed

    Returns:
        n_entities x n_entities float32 array
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate sparse indices
    n_edges = int(n_entities * n_entities * density)
    n_edges = max(n_edges, n_entities)  # Minimum connectivity

    # Random edge positions
    rows = np.random.randint(0, n_entities, size=n_edges)
    cols = np.random.randint(0, n_entities, size=n_edges)

    # Create dense tensor
    tensor = np.zeros((n_entities, n_entities), dtype=np.float32)
    tensor[rows, cols] = 1.0

    return tensor


def time_operation(
    operation: Any,
    n_warmup: int = 2,
    n_runs: int = 10,
) -> tuple[float, float, float, float]:
    """Time an operation with warmup.

    Args:
        operation: Callable to time
        n_warmup: Number of warmup runs (not counted)
        n_runs: Number of timed runs

    Returns:
        Tuple of (mean_ms, std_ms, min_ms, max_ms)
    """
    # Warmup
    for _ in range(n_warmup):
        operation()

    # Force garbage collection
    gc.collect()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        operation()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return (
        float(np.mean(times)),
        float(np.std(times)),
        float(np.min(times)),
        float(np.max(times)),
    )


def estimate_memory_mb(n_entities: int, n_relations: int = 1) -> float:
    """Estimate memory usage in MB for relation tensors.

    Args:
        n_entities: Number of entities
        n_relations: Number of relation tensors

    Returns:
        Estimated memory in MB
    """
    # Each float32 is 4 bytes
    bytes_per_tensor = n_entities * n_entities * 4
    total_bytes = bytes_per_tensor * n_relations
    return total_bytes / (1024 * 1024)


def benchmark_single_hop(
    n_entities: int,
    density: float = 0.01,
    n_runs: int = 10,
) -> BenchmarkResult:
    """Benchmark single-hop exists query.

    Query: exists y: R(x, y) for all x
    """
    backend = create_backend()
    rel = generate_sparse_relation(n_entities, density, seed=42)

    mean_ms, std_ms, min_ms, max_ms = time_operation(
        lambda: exists(rel, axis=1, backend=backend),
        n_runs=n_runs,
    )

    return BenchmarkResult(
        operation="single_hop_exists",
        n_entities=n_entities,
        n_relations=1,
        mean_time_ms=mean_ms,
        std_time_ms=std_ms,
        min_time_ms=min_ms,
        max_time_ms=max_ms,
        memory_mb=estimate_memory_mb(n_entities, 1),
        timestamp=datetime.now().isoformat(),
    )


def benchmark_two_hop(
    n_entities: int,
    density: float = 0.01,
    n_runs: int = 10,
) -> BenchmarkResult:
    """Benchmark 2-hop relation composition.

    Query: R1 @ R2 (matrix multiplication)
    """
    rel1 = generate_sparse_relation(n_entities, density, seed=42)
    rel2 = generate_sparse_relation(n_entities, density, seed=43)

    mean_ms, std_ms, min_ms, max_ms = time_operation(
        lambda: np.matmul(rel1, rel2),
        n_runs=n_runs,
    )

    return BenchmarkResult(
        operation="two_hop_composition",
        n_entities=n_entities,
        n_relations=2,
        mean_time_ms=mean_ms,
        std_time_ms=std_ms,
        min_time_ms=min_ms,
        max_time_ms=max_ms,
        memory_mb=estimate_memory_mb(n_entities, 3),  # 2 input + 1 output
        timestamp=datetime.now().isoformat(),
    )


def benchmark_three_hop(
    n_entities: int,
    density: float = 0.01,
    n_runs: int = 10,
) -> BenchmarkResult:
    """Benchmark 3-hop relation composition.

    Query: R1 @ R2 @ R3
    """
    rel1 = generate_sparse_relation(n_entities, density, seed=42)
    rel2 = generate_sparse_relation(n_entities, density, seed=43)
    rel3 = generate_sparse_relation(n_entities, density, seed=44)

    mean_ms, std_ms, min_ms, max_ms = time_operation(
        lambda: np.matmul(np.matmul(rel1, rel2), rel3),
        n_runs=n_runs,
    )

    return BenchmarkResult(
        operation="three_hop_composition",
        n_entities=n_entities,
        n_relations=3,
        mean_time_ms=mean_ms,
        std_time_ms=std_ms,
        min_time_ms=min_ms,
        max_time_ms=max_ms,
        memory_mb=estimate_memory_mb(n_entities, 4),  # 3 input + 1 output
        timestamp=datetime.now().isoformat(),
    )


def benchmark_logical_and(
    n_entities: int,
    density: float = 0.01,
    n_runs: int = 10,
) -> BenchmarkResult:
    """Benchmark logical AND operation.

    Query: R1 AND R2 (element-wise)
    """
    backend = create_backend()
    rel1 = generate_sparse_relation(n_entities, density, seed=42)
    rel2 = generate_sparse_relation(n_entities, density, seed=43)

    mean_ms, std_ms, min_ms, max_ms = time_operation(
        lambda: logical_and(rel1, rel2, backend=backend),
        n_runs=n_runs,
    )

    return BenchmarkResult(
        operation="logical_and",
        n_entities=n_entities,
        n_relations=2,
        mean_time_ms=mean_ms,
        std_time_ms=std_ms,
        min_time_ms=min_ms,
        max_time_ms=max_ms,
        memory_mb=estimate_memory_mb(n_entities, 3),
        timestamp=datetime.now().isoformat(),
    )


def benchmark_logical_or(
    n_entities: int,
    density: float = 0.01,
    n_runs: int = 10,
) -> BenchmarkResult:
    """Benchmark logical OR operation.

    Query: R1 OR R2 (element-wise)
    """
    backend = create_backend()
    rel1 = generate_sparse_relation(n_entities, density, seed=42)
    rel2 = generate_sparse_relation(n_entities, density, seed=43)

    mean_ms, std_ms, min_ms, max_ms = time_operation(
        lambda: logical_or(rel1, rel2, backend=backend),
        n_runs=n_runs,
    )

    return BenchmarkResult(
        operation="logical_or",
        n_entities=n_entities,
        n_relations=2,
        mean_time_ms=mean_ms,
        std_time_ms=std_ms,
        min_time_ms=min_ms,
        max_time_ms=max_ms,
        memory_mb=estimate_memory_mb(n_entities, 3),
        timestamp=datetime.now().isoformat(),
    )


def benchmark_forall(
    n_entities: int,
    density: float = 0.01,
    n_runs: int = 10,
) -> BenchmarkResult:
    """Benchmark universal quantifier.

    Query: forall y: R(x, y) for all x
    """
    backend = create_backend()
    rel = generate_sparse_relation(n_entities, density, seed=42)

    mean_ms, std_ms, min_ms, max_ms = time_operation(
        lambda: forall(rel, axis=1, backend=backend),
        n_runs=n_runs,
    )

    return BenchmarkResult(
        operation="forall",
        n_entities=n_entities,
        n_relations=1,
        mean_time_ms=mean_ms,
        std_time_ms=std_ms,
        min_time_ms=min_ms,
        max_time_ms=max_ms,
        memory_mb=estimate_memory_mb(n_entities, 1),
        timestamp=datetime.now().isoformat(),
    )


def run_scale_benchmarks(
    entity_counts: list[int] | None = None,
    density: float = 0.01,
    n_runs: int = 10,
    verbose: bool = True,
) -> ScaleBenchmarkSuite:
    """Run full scale benchmark suite.

    Args:
        entity_counts: List of entity counts to test
        density: Relation density
        n_runs: Number of runs per benchmark
        verbose: Print progress

    Returns:
        ScaleBenchmarkSuite with all results
    """
    if entity_counts is None:
        entity_counts = [100, 500, 1000, 2000, 5000, 10000]

    backend = create_backend()
    backend_name = type(backend).__name__

    if verbose:
        print("=" * 80)
        print("TENSORLOGIC SCALE BENCHMARKS")
        print("=" * 80)
        print(f"Backend: {backend_name}")
        print(f"Density: {density}")
        print(f"Runs per benchmark: {n_runs}")
        print(f"Entity counts: {entity_counts}")
        print()

    results: list[BenchmarkResult] = []
    benchmark_funcs = [
        ("single_hop_exists", benchmark_single_hop),
        ("two_hop_composition", benchmark_two_hop),
        ("three_hop_composition", benchmark_three_hop),
        ("logical_and", benchmark_logical_and),
        ("logical_or", benchmark_logical_or),
        ("forall", benchmark_forall),
    ]

    for n_entities in entity_counts:
        if verbose:
            print(f"\n--- {n_entities} entities ---")

        for bench_name, bench_func in benchmark_funcs:
            # Skip very large benchmarks that would take too long
            if n_entities > 20000 and "three_hop" in bench_name:
                if verbose:
                    print(f"  {bench_name}: skipped (too large)")
                continue

            try:
                result = bench_func(n_entities, density=density, n_runs=n_runs)
                results.append(result)

                if verbose:
                    status = "PASS" if result.mean_time_ms < 100 else "SLOW"
                    print(
                        f"  {bench_name}: {result.mean_time_ms:.2f}ms "
                        f"(+/- {result.std_time_ms:.2f}ms) [{status}]"
                    )

            except Exception as e:
                if verbose:
                    print(f"  {bench_name}: FAILED ({e})")

        # Clean up memory between scales
        gc.collect()

    return ScaleBenchmarkSuite(
        backend=backend_name,
        platform="darwin",  # Could detect this dynamically
        results=results,
        timestamp=datetime.now().isoformat(),
    )


def print_summary(suite: ScaleBenchmarkSuite) -> None:
    """Print summary of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    # Group by operation
    operations = {}
    for result in suite.results:
        if result.operation not in operations:
            operations[result.operation] = []
        operations[result.operation].append(result)

    print(f"\n{'Operation':<25} {'Scale':<12} {'Time (ms)':<15} {'Memory (MB)':<12}")
    print("-" * 70)

    for op_name, op_results in operations.items():
        for result in op_results:
            time_str = f"{result.mean_time_ms:.2f} +/- {result.std_time_ms:.2f}"
            print(
                f"{op_name:<25} {result.n_entities:<12} "
                f"{time_str:<15} {result.memory_mb:<12.2f}"
            )
        print()

    # Check against targets
    print("\nTarget Assessment (3-hop <100ms):")
    three_hop_results = [r for r in suite.results if "three_hop" in r.operation]
    for result in three_hop_results:
        status = "PASS" if result.mean_time_ms < 100 else "FAIL"
        print(f"  {result.n_entities} entities: {result.mean_time_ms:.2f}ms [{status}]")


def save_results(suite: ScaleBenchmarkSuite, output_path: Path) -> None:
    """Save benchmark results to JSON file."""
    data = {
        "backend": suite.backend,
        "platform": suite.platform,
        "timestamp": suite.timestamp,
        "results": [asdict(r) for r in suite.results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main() -> None:
    """Run scale benchmarks."""
    parser = argparse.ArgumentParser(description="TensorLogic Scale Benchmarks")
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
        help="Entity counts to benchmark",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.01,
        help="Relation density (default: 0.01)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs per benchmark",
    )
    args = parser.parse_args()

    suite = run_scale_benchmarks(
        entity_counts=args.scales,
        density=args.density,
        n_runs=args.runs,
        verbose=True,
    )

    print_summary(suite)

    if args.output:
        save_results(suite, args.output)


if __name__ == "__main__":
    main()
