#!/usr/bin/env python3
"""
NumPy Benchmark Script for NumRS2 Comparison

This script runs the same benchmarks as the Rust numpy_comparison_benchmark.rs
to enable direct performance comparison between NumRS2 and NumPy.

Usage:
    python bench/numpy_benchmark.py

Output:
    JSON file with benchmark results at bench/numpy/numpy_benchmark_results.json
"""

import numpy as np
import time
import json
import os
from typing import Dict, List, Callable
from dataclasses import dataclass, asdict
import statistics

@dataclass
class BenchmarkResult:
    """Results from a single benchmark"""
    name: str
    size: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    iterations: int

def benchmark(func: Callable, iterations: int = 100, warmup: int = 10) -> tuple:
    """Run a benchmark function multiple times and return statistics"""
    # Warmup
    for _ in range(warmup):
        func()

    # Actual timing
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    return (
        statistics.mean(times),
        statistics.stdev(times) if len(times) > 1 else 0,
        min(times),
        max(times),
        iterations
    )

# Standard sizes matching Rust benchmarks
SIZES = [1000, 10000, 100000, 1000000]
MATRIX_SIZES = [32, 64, 128, 256]

def run_creation_benchmarks() -> List[BenchmarkResult]:
    """Array creation benchmarks"""
    results = []

    for size in SIZES:
        # np.zeros
        mean, std, min_t, max_t, iters = benchmark(lambda: np.zeros(size))
        results.append(BenchmarkResult("zeros", size, mean, std, min_t, max_t, iters))

        # np.ones
        mean, std, min_t, max_t, iters = benchmark(lambda: np.ones(size))
        results.append(BenchmarkResult("ones", size, mean, std, min_t, max_t, iters))

        # np.full
        mean, std, min_t, max_t, iters = benchmark(lambda: np.full(size, np.pi))
        results.append(BenchmarkResult("full", size, mean, std, min_t, max_t, iters))

        # np.empty
        mean, std, min_t, max_t, iters = benchmark(lambda: np.empty(size))
        results.append(BenchmarkResult("empty", size, mean, std, min_t, max_t, iters))

        if size <= 100000:
            # np.arange
            mean, std, min_t, max_t, iters = benchmark(lambda s=size: np.arange(s, dtype=np.float64))
            results.append(BenchmarkResult("arange", size, mean, std, min_t, max_t, iters))

            # np.linspace
            mean, std, min_t, max_t, iters = benchmark(lambda s=size: np.linspace(0, 1, s))
            results.append(BenchmarkResult("linspace", size, mean, std, min_t, max_t, iters))

    # Matrix creation
    for size in MATRIX_SIZES:
        # np.eye
        mean, std, min_t, max_t, iters = benchmark(lambda s=size: np.eye(s))
        results.append(BenchmarkResult("eye", size, mean, std, min_t, max_t, iters))

        # np.identity
        mean, std, min_t, max_t, iters = benchmark(lambda s=size: np.identity(s))
        results.append(BenchmarkResult("identity", size, mean, std, min_t, max_t, iters))

    return results

def run_arithmetic_benchmarks() -> List[BenchmarkResult]:
    """Arithmetic operation benchmarks"""
    results = []

    for size in SIZES:
        a = np.arange(size, dtype=np.float64) * 0.001 + 1.0
        b = np.arange(size, dtype=np.float64) * 0.002 + 0.5

        # np.add
        mean, std, min_t, max_t, iters = benchmark(lambda: np.add(a, b))
        results.append(BenchmarkResult("add", size, mean, std, min_t, max_t, iters))

        # np.subtract
        mean, std, min_t, max_t, iters = benchmark(lambda: np.subtract(a, b))
        results.append(BenchmarkResult("subtract", size, mean, std, min_t, max_t, iters))

        # np.multiply
        mean, std, min_t, max_t, iters = benchmark(lambda: np.multiply(a, b))
        results.append(BenchmarkResult("multiply", size, mean, std, min_t, max_t, iters))

        # np.divide
        mean, std, min_t, max_t, iters = benchmark(lambda: np.divide(a, b))
        results.append(BenchmarkResult("divide", size, mean, std, min_t, max_t, iters))

        # Scalar add
        mean, std, min_t, max_t, iters = benchmark(lambda: a + 2.5)
        results.append(BenchmarkResult("add_scalar", size, mean, std, min_t, max_t, iters))

        # Scalar mul
        mean, std, min_t, max_t, iters = benchmark(lambda: a * 2.5)
        results.append(BenchmarkResult("mul_scalar", size, mean, std, min_t, max_t, iters))

    return results

def run_ufunc_benchmarks() -> List[BenchmarkResult]:
    """Universal function benchmarks"""
    results = []

    for size in SIZES:
        if size > 100000:
            continue

        pos = np.arange(1, size + 1, dtype=np.float64)
        small = np.arange(size, dtype=np.float64) * 0.0001
        angles = np.arange(size, dtype=np.float64) * 0.001 * np.pi
        mixed = np.arange(size, dtype=np.float64) - size / 2.0

        # np.sqrt
        mean, std, min_t, max_t, iters = benchmark(lambda: np.sqrt(pos))
        results.append(BenchmarkResult("sqrt", size, mean, std, min_t, max_t, iters))

        # np.exp
        mean, std, min_t, max_t, iters = benchmark(lambda: np.exp(small))
        results.append(BenchmarkResult("exp", size, mean, std, min_t, max_t, iters))

        # np.log
        mean, std, min_t, max_t, iters = benchmark(lambda: np.log(pos))
        results.append(BenchmarkResult("log", size, mean, std, min_t, max_t, iters))

        # np.sin
        mean, std, min_t, max_t, iters = benchmark(lambda: np.sin(angles))
        results.append(BenchmarkResult("sin", size, mean, std, min_t, max_t, iters))

        # np.cos
        mean, std, min_t, max_t, iters = benchmark(lambda: np.cos(angles))
        results.append(BenchmarkResult("cos", size, mean, std, min_t, max_t, iters))

        # np.tan
        mean, std, min_t, max_t, iters = benchmark(lambda: np.tan(angles))
        results.append(BenchmarkResult("tan", size, mean, std, min_t, max_t, iters))

        # np.abs
        mean, std, min_t, max_t, iters = benchmark(lambda: np.abs(mixed))
        results.append(BenchmarkResult("abs", size, mean, std, min_t, max_t, iters))

        # np.power
        mean, std, min_t, max_t, iters = benchmark(lambda: np.power(pos, 2.0))
        results.append(BenchmarkResult("power_2", size, mean, std, min_t, max_t, iters))

        # np.square
        mean, std, min_t, max_t, iters = benchmark(lambda: np.square(pos))
        results.append(BenchmarkResult("square", size, mean, std, min_t, max_t, iters))

    return results

def run_reduction_benchmarks() -> List[BenchmarkResult]:
    """Reduction operation benchmarks"""
    results = []

    for size in SIZES:
        a = np.arange(size, dtype=np.float64) * 0.001

        # np.sum
        mean, std, min_t, max_t, iters = benchmark(lambda: np.sum(a))
        results.append(BenchmarkResult("sum", size, mean, std, min_t, max_t, iters))

        if size <= 10000:
            # np.prod
            prod_arr = 1.0 + np.arange(min(size, 1000), dtype=np.float64) * 0.0001
            mean, std, min_t, max_t, iters = benchmark(lambda: np.prod(prod_arr))
            results.append(BenchmarkResult("prod", size, mean, std, min_t, max_t, iters))

        # np.mean
        mean, std, min_t, max_t, iters = benchmark(lambda: np.mean(a))
        results.append(BenchmarkResult("mean", size, mean, std, min_t, max_t, iters))

        # np.std
        mean, std, min_t, max_t, iters = benchmark(lambda: np.std(a))
        results.append(BenchmarkResult("std", size, mean, std, min_t, max_t, iters))

        # np.var
        mean, std, min_t, max_t, iters = benchmark(lambda: np.var(a))
        results.append(BenchmarkResult("var", size, mean, std, min_t, max_t, iters))

        # np.min
        mean, std, min_t, max_t, iters = benchmark(lambda: np.min(a))
        results.append(BenchmarkResult("min", size, mean, std, min_t, max_t, iters))

        # np.max
        mean, std, min_t, max_t, iters = benchmark(lambda: np.max(a))
        results.append(BenchmarkResult("max", size, mean, std, min_t, max_t, iters))

        # np.argmin
        mean, std, min_t, max_t, iters = benchmark(lambda: np.argmin(a))
        results.append(BenchmarkResult("argmin", size, mean, std, min_t, max_t, iters))

        # np.argmax
        mean, std, min_t, max_t, iters = benchmark(lambda: np.argmax(a))
        results.append(BenchmarkResult("argmax", size, mean, std, min_t, max_t, iters))

    return results

def run_linalg_benchmarks() -> List[BenchmarkResult]:
    """Linear algebra benchmarks"""
    results = []

    for size in MATRIX_SIZES:
        a = ((np.arange(size * size, dtype=np.float64).reshape(size, size) * 0.73) % 10.0) + 0.1
        v = np.arange(size, dtype=np.float64) + 1.0

        # matmul
        mean, std, min_t, max_t, iters = benchmark(lambda: np.matmul(a, a), iterations=50)
        results.append(BenchmarkResult("matmul", size, mean, std, min_t, max_t, iters))

        # dot
        mean, std, min_t, max_t, iters = benchmark(lambda: np.dot(v, v))
        results.append(BenchmarkResult("dot", size, mean, std, min_t, max_t, iters))

        if size <= 128:
            # det
            mean, std, min_t, max_t, iters = benchmark(lambda: np.linalg.det(a), iterations=50)
            results.append(BenchmarkResult("det", size, mean, std, min_t, max_t, iters))

            # inv (for SPD matrix)
            spd = a @ a.T + np.eye(size) * size * 0.1
            mean, std, min_t, max_t, iters = benchmark(lambda: np.linalg.inv(spd), iterations=50)
            results.append(BenchmarkResult("inv", size, mean, std, min_t, max_t, iters))

            # qr
            mean, std, min_t, max_t, iters = benchmark(lambda: np.linalg.qr(a), iterations=50)
            results.append(BenchmarkResult("qr", size, mean, std, min_t, max_t, iters))

            # cholesky
            mean, std, min_t, max_t, iters = benchmark(lambda: np.linalg.cholesky(spd), iterations=50)
            results.append(BenchmarkResult("cholesky", size, mean, std, min_t, max_t, iters))

        if size <= 64:
            # svd
            mean, std, min_t, max_t, iters = benchmark(lambda: np.linalg.svd(a), iterations=20)
            results.append(BenchmarkResult("svd", size, mean, std, min_t, max_t, iters))

            # eig
            mean, std, min_t, max_t, iters = benchmark(lambda: np.linalg.eig(a), iterations=20)
            results.append(BenchmarkResult("eig", size, mean, std, min_t, max_t, iters))

        # trace
        mean, std, min_t, max_t, iters = benchmark(lambda: np.trace(a))
        results.append(BenchmarkResult("trace", size, mean, std, min_t, max_t, iters))

    return results

def run_statistics_benchmarks() -> List[BenchmarkResult]:
    """Statistics benchmarks"""
    results = []

    for size in SIZES:
        if size > 100000:
            continue

        a = ((np.arange(size, dtype=np.float64) * 0.73 + 17.0) % 100.0)

        # median
        mean, std, min_t, max_t, iters = benchmark(lambda: np.median(a))
        results.append(BenchmarkResult("median", size, mean, std, min_t, max_t, iters))

        # percentile
        mean, std, min_t, max_t, iters = benchmark(lambda: np.percentile(a, 75))
        results.append(BenchmarkResult("percentile", size, mean, std, min_t, max_t, iters))

        if size <= 10000:
            # histogram
            mean, std, min_t, max_t, iters = benchmark(lambda: np.histogram(a, bins=50))
            results.append(BenchmarkResult("histogram", size, mean, std, min_t, max_t, iters))

        # cumsum
        mean, std, min_t, max_t, iters = benchmark(lambda: np.cumsum(a))
        results.append(BenchmarkResult("cumsum", size, mean, std, min_t, max_t, iters))

        # diff
        mean, std, min_t, max_t, iters = benchmark(lambda: np.diff(a))
        results.append(BenchmarkResult("diff", size, mean, std, min_t, max_t, iters))

        # sort
        a_copy = a.copy()
        mean, std, min_t, max_t, iters = benchmark(lambda: np.sort(a_copy))
        results.append(BenchmarkResult("sort", size, mean, std, min_t, max_t, iters))

        # argsort
        mean, std, min_t, max_t, iters = benchmark(lambda: np.argsort(a_copy))
        results.append(BenchmarkResult("argsort", size, mean, std, min_t, max_t, iters))

    return results

def run_polynomial_benchmarks() -> List[BenchmarkResult]:
    """Polynomial benchmarks"""
    results = []

    degrees = [5, 10, 20, 50]

    for degree in degrees:
        coeffs = np.array([1.0 / (i + 1) for i in range(degree + 1)])

        # polyval single point
        mean, std, min_t, max_t, iters = benchmark(lambda: np.polyval(coeffs, 2.5))
        results.append(BenchmarkResult("polyval_single", degree, mean, std, min_t, max_t, iters))

        # polyder
        mean, std, min_t, max_t, iters = benchmark(lambda: np.polyder(coeffs))
        results.append(BenchmarkResult("polyder", degree, mean, std, min_t, max_t, iters))

        # polyint
        mean, std, min_t, max_t, iters = benchmark(lambda: np.polyint(coeffs))
        results.append(BenchmarkResult("polyint", degree, mean, std, min_t, max_t, iters))

        # polymul
        coeffs2 = np.arange(degree + 1, dtype=np.float64) * 0.5
        mean, std, min_t, max_t, iters = benchmark(lambda: np.polymul(coeffs, coeffs2))
        results.append(BenchmarkResult("polymul", degree, mean, std, min_t, max_t, iters))

        # polyadd
        mean, std, min_t, max_t, iters = benchmark(lambda: np.polyadd(coeffs, coeffs2))
        results.append(BenchmarkResult("polyadd", degree, mean, std, min_t, max_t, iters))

    # polyfit
    fit_sizes = [10, 50, 100, 500]
    for size in fit_sizes:
        x = np.arange(size, dtype=np.float64)
        y = x * x * 0.5 + x * 2.0 + 1.0 + np.sin(x * 0.1)

        mean, std, min_t, max_t, iters = benchmark(lambda: np.polyfit(x, y, 3))
        results.append(BenchmarkResult("polyfit_deg3", size, mean, std, min_t, max_t, iters))

    return results

def main():
    """Run all benchmarks and save results"""
    print("NumRS2 vs NumPy Benchmark Comparison")
    print("=" * 50)
    print(f"NumPy version: {np.__version__}")
    print()

    all_results = {
        "numpy_version": np.__version__,
        "benchmarks": {}
    }

    # Run each benchmark category
    categories = [
        ("creation", run_creation_benchmarks),
        ("arithmetic", run_arithmetic_benchmarks),
        ("ufuncs", run_ufunc_benchmarks),
        ("reductions", run_reduction_benchmarks),
        ("linalg", run_linalg_benchmarks),
        ("statistics", run_statistics_benchmarks),
        ("polynomial", run_polynomial_benchmarks),
    ]

    for category_name, benchmark_func in categories:
        print(f"Running {category_name} benchmarks...")
        results = benchmark_func()
        all_results["benchmarks"][category_name] = [asdict(r) for r in results]

        # Print summary
        for r in results:
            print(f"  {r.name}[{r.size}]: {r.mean_ms:.4f} ms (±{r.std_ms:.4f})")
        print()

    # Save results
    os.makedirs("bench/numpy", exist_ok=True)
    output_path = "bench/numpy/numpy_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to {output_path}")
    print("Done!")

if __name__ == "__main__":
    main()
