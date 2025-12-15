"""Compilation Strategy Benchmarks for TensorLogic

Compares performance across the 5 compilation strategies:
- soft_differentiable: a*b (AND), a+b-ab (OR)
- hard_boolean: step(a*b), step(max(a,b))
- godel: min(a,b), max(a,b)
- product: a*b, a+b-ab
- lukasiewicz: max(0,a+b-1), min(1,a+b)

Run benchmark:
    uv run python benchmarks/strategy_comparison.py
    uv run python benchmarks/strategy_comparison.py --output results/strategy_results.json
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

from tensorlogic import create_backend, create_strategy, get_available_strategies


@dataclass
class StrategyBenchmark:
    """Benchmark result for a single strategy."""

    strategy: str
    operation: str
    n_entities: int
    mean_time_ms: float
    std_time_ms: float
    result_sum: float  # Checksum to verify correctness
    timestamp: str


@dataclass
class StrategyComparisonSuite:
    """Complete strategy comparison results."""

    backend: str
    strategies: list[str]
    results: list[StrategyBenchmark]
    timestamp: str


def generate_test_data(
    n_entities: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate test data for strategy benchmarks.

    Args:
        n_entities: Number of entities
        seed: Random seed

    Returns:
        Tuple of (a, b) arrays with values in [0, 1]
    """
    np.random.seed(seed)
    a = np.random.random((n_entities, n_entities)).astype(np.float32)
    b = np.random.random((n_entities, n_entities)).astype(np.float32)
    return a, b


def time_operation(
    operation: Any,
    n_warmup: int = 2,
    n_runs: int = 10,
) -> tuple[float, float]:
    """Time an operation.

    Returns:
        Tuple of (mean_ms, std_ms)
    """
    # Warmup
    for _ in range(n_warmup):
        operation()

    gc.collect()

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        operation()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return float(np.mean(times)), float(np.std(times))


def benchmark_strategy_operation(
    strategy_name: str,
    operation: str,
    n_entities: int,
    n_runs: int = 10,
) -> StrategyBenchmark:
    """Benchmark a specific strategy and operation combination.

    Args:
        strategy_name: Name of the compilation strategy
        operation: "and", "or", or "not"
        n_entities: Size of test data
        n_runs: Number of runs

    Returns:
        StrategyBenchmark result
    """
    backend = create_backend()
    strategy = create_strategy(strategy_name, backend=backend)

    a, b = generate_test_data(n_entities)

    if operation == "and":
        op_func = lambda: strategy.compile_and(a, b)
    elif operation == "or":
        op_func = lambda: strategy.compile_or(a, b)
    elif operation == "not":
        op_func = lambda: strategy.compile_not(a)
    else:
        raise ValueError(f"Unknown operation: {operation}")

    mean_ms, std_ms = time_operation(op_func, n_runs=n_runs)

    # Compute result checksum for verification
    result = op_func()
    result_sum = float(np.sum(np.asarray(result)))

    return StrategyBenchmark(
        strategy=strategy_name,
        operation=operation,
        n_entities=n_entities,
        mean_time_ms=mean_ms,
        std_time_ms=std_ms,
        result_sum=result_sum,
        timestamp=datetime.now().isoformat(),
    )


def run_strategy_comparison(
    entity_counts: list[int] | None = None,
    operations: list[str] | None = None,
    n_runs: int = 10,
    verbose: bool = True,
) -> StrategyComparisonSuite:
    """Run strategy comparison benchmarks.

    Args:
        entity_counts: List of entity counts to test
        operations: List of operations to test
        n_runs: Number of runs per benchmark
        verbose: Print progress

    Returns:
        StrategyComparisonSuite with all results
    """
    if entity_counts is None:
        entity_counts = [100, 500, 1000, 2000]
    if operations is None:
        operations = ["and", "or", "not"]

    backend = create_backend()
    backend_name = type(backend).__name__
    strategies = get_available_strategies()

    if verbose:
        print("=" * 80)
        print("TENSORLOGIC STRATEGY COMPARISON")
        print("=" * 80)
        print(f"Backend: {backend_name}")
        print(f"Strategies: {strategies}")
        print(f"Operations: {operations}")
        print(f"Entity counts: {entity_counts}")
        print(f"Runs per benchmark: {n_runs}")
        print()

    results: list[StrategyBenchmark] = []

    for n_entities in entity_counts:
        if verbose:
            print(f"\n--- {n_entities} entities ---")

        for operation in operations:
            if verbose:
                print(f"\n  Operation: {operation.upper()}")
                print(f"  {'Strategy':<25} {'Time (ms)':<20} {'Result Sum':<15}")
                print("  " + "-" * 60)

            for strategy_name in strategies:
                try:
                    result = benchmark_strategy_operation(
                        strategy_name,
                        operation,
                        n_entities,
                        n_runs=n_runs,
                    )
                    results.append(result)

                    if verbose:
                        time_str = f"{result.mean_time_ms:.3f} +/- {result.std_time_ms:.3f}"
                        print(
                            f"  {strategy_name:<25} {time_str:<20} "
                            f"{result.result_sum:<15.2f}"
                        )

                except Exception as e:
                    if verbose:
                        print(f"  {strategy_name:<25} FAILED: {e}")

        gc.collect()

    return StrategyComparisonSuite(
        backend=backend_name,
        strategies=strategies,
        results=results,
        timestamp=datetime.now().isoformat(),
    )


def analyze_results(suite: StrategyComparisonSuite) -> None:
    """Analyze and print strategy comparison results."""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON ANALYSIS")
    print("=" * 80)

    # Group by operation
    operations = set(r.operation for r in suite.results)
    entity_counts = sorted(set(r.n_entities for r in suite.results))

    for operation in operations:
        print(f"\n--- {operation.upper()} Operation ---")

        # Find fastest strategy at each scale
        print(f"\n{'Scale':<10}", end="")
        for strategy in suite.strategies:
            print(f"{strategy:<18}", end="")
        print("Fastest")
        print("-" * (10 + 18 * len(suite.strategies) + 10))

        for n_entities in entity_counts:
            op_results = [
                r for r in suite.results
                if r.operation == operation and r.n_entities == n_entities
            ]

            if not op_results:
                continue

            print(f"{n_entities:<10}", end="")

            fastest = min(op_results, key=lambda r: r.mean_time_ms)

            for strategy in suite.strategies:
                result = next(
                    (r for r in op_results if r.strategy == strategy),
                    None,
                )
                if result:
                    marker = "*" if result == fastest else " "
                    print(f"{result.mean_time_ms:>6.3f}ms{marker:<10}", end="")
                else:
                    print(f"{'N/A':<18}", end="")

            print(f"{fastest.strategy}")

    # Compute average speedup vs soft_differentiable (baseline)
    print("\n--- Average Performance (relative to soft_differentiable) ---")
    baseline_strategy = "soft_differentiable"

    for operation in operations:
        print(f"\n{operation.upper()}:")

        for strategy in suite.strategies:
            strategy_times = [
                r.mean_time_ms for r in suite.results
                if r.strategy == strategy and r.operation == operation
            ]
            baseline_times = [
                r.mean_time_ms for r in suite.results
                if r.strategy == baseline_strategy and r.operation == operation
            ]

            if strategy_times and baseline_times:
                avg_ratio = np.mean(
                    [b / s for s, b in zip(strategy_times, baseline_times)]
                )
                if strategy == baseline_strategy:
                    print(f"  {strategy:<25} 1.00x (baseline)")
                else:
                    faster_slower = "faster" if avg_ratio > 1 else "slower"
                    print(f"  {strategy:<25} {avg_ratio:.2f}x ({faster_slower})")

    # Verify correctness (results should differ by strategy)
    print("\n--- Correctness Verification ---")
    print("(Different strategies should produce different result sums)")

    for operation in ["and", "or"]:
        results_1000 = [
            r for r in suite.results
            if r.operation == operation and r.n_entities == 1000
        ]

        if results_1000:
            sums = {r.strategy: r.result_sum for r in results_1000}
            unique_sums = len(set(sums.values()))
            print(f"\n{operation.upper()} at 1000 entities:")
            for strategy, result_sum in sums.items():
                print(f"  {strategy:<25} sum = {result_sum:.2f}")
            print(f"  Unique values: {unique_sums} (expected: varies by strategy)")


def save_results(suite: StrategyComparisonSuite, output_path: Path) -> None:
    """Save strategy comparison results to JSON file."""
    data = {
        "backend": suite.backend,
        "strategies": suite.strategies,
        "timestamp": suite.timestamp,
        "results": [asdict(r) for r in suite.results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main() -> None:
    """Run strategy comparison benchmarks."""
    parser = argparse.ArgumentParser(description="TensorLogic Strategy Comparison")
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
        default=[100, 500, 1000, 2000],
        help="Entity counts to benchmark",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of runs per benchmark",
    )
    args = parser.parse_args()

    suite = run_strategy_comparison(
        entity_counts=args.scales,
        n_runs=args.runs,
        verbose=True,
    )

    analyze_results(suite)

    if args.output:
        save_results(suite, args.output)


if __name__ == "__main__":
    main()
