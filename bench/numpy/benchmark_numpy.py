#!/usr/bin/env python3
"""
NumPy Benchmark Suite for comparison with NumRS2

This script runs various NumPy operations and times them to provide
reference performance data to compare against NumRS2's equivalent operations.

Results are saved to a JSON file that can be loaded by the Rust benchmark suite.
"""

import numpy as np
import json
import os
import time
import sys
from datetime import datetime

# Sample sizes for benchmarks
SMALL_SIZE = 1_000
MEDIUM_SIZE = 10_000
LARGE_SIZE = 100_000
MATRIX_SMALL = (100, 100)
MATRIX_MEDIUM = (1000, 1000)
MATRIX_LARGE = (5000, 5000)

# Number of iterations to run each benchmark
ITERATIONS = 10

# Directory to save benchmark results
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BENCH_DIR, "numpy_benchmark_results.json")

# System information
def get_system_info():
    """Get information about the system running the benchmarks"""
    import platform
    import psutil
    
    try:
        # Get CPU information
        cpu_info = {
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "physical_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
        }
        return cpu_info
    except ImportError:
        # If psutil is not installed
        return {
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count()
        }

# Benchmark timing function
def time_function(func, *args, **kwargs):
    """
    Time a function execution for multiple iterations and return statistics
    
    Returns:
        dict: Statistics about the timing (mean, min, max, etc.)
    """
    times = []
    
    # Warmup
    func(*args, **kwargs)
    
    # Timed runs
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds
    
    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "median": sorted(times)[len(times) // 2],
        "iterations": ITERATIONS
    }

# Benchmark functions
def benchmark_array_creation():
    """Benchmark array creation operations"""
    results = {}
    
    # Time array creation with different sizes
    results["zeros_small"] = time_function(np.zeros, SMALL_SIZE)
    results["zeros_medium"] = time_function(np.zeros, MEDIUM_SIZE)
    results["zeros_large"] = time_function(np.zeros, LARGE_SIZE)
    
    results["ones_small"] = time_function(np.ones, SMALL_SIZE)
    results["ones_medium"] = time_function(np.ones, MEDIUM_SIZE)
    results["ones_large"] = time_function(np.ones, LARGE_SIZE)
    
    results["random_small"] = time_function(np.random.random, (SMALL_SIZE,))
    results["random_medium"] = time_function(np.random.random, (MEDIUM_SIZE,))
    results["random_large"] = time_function(np.random.random, (LARGE_SIZE,))
    
    # 2D arrays
    results["zeros_matrix_small"] = time_function(np.zeros, MATRIX_SMALL)
    results["zeros_matrix_medium"] = time_function(np.zeros, MATRIX_MEDIUM)
    
    results["ones_matrix_small"] = time_function(np.ones, MATRIX_SMALL)
    results["ones_matrix_medium"] = time_function(np.ones, MATRIX_MEDIUM)
    
    results["random_matrix_small"] = time_function(np.random.random, MATRIX_SMALL)
    results["random_matrix_medium"] = time_function(np.random.random, MATRIX_MEDIUM)
    
    # Special array creations
    results["identity_small"] = time_function(np.identity, 100)
    results["identity_medium"] = time_function(np.identity, 1000)
    
    results["arange_small"] = time_function(np.arange, SMALL_SIZE)
    results["arange_medium"] = time_function(np.arange, MEDIUM_SIZE)
    results["arange_large"] = time_function(np.arange, LARGE_SIZE)
    
    return results

def benchmark_array_operations():
    """Benchmark array operations"""
    results = {}
    
    # Create arrays for operations
    a_small = np.random.random(SMALL_SIZE)
    a_medium = np.random.random(MEDIUM_SIZE)
    a_large = np.random.random(LARGE_SIZE)
    
    b_small = np.random.random(SMALL_SIZE)
    b_medium = np.random.random(MEDIUM_SIZE)
    b_large = np.random.random(LARGE_SIZE)
    
    # Basic operations
    results["add_small"] = time_function(np.add, a_small, b_small)
    results["add_medium"] = time_function(np.add, a_medium, b_medium)
    results["add_large"] = time_function(np.add, a_large, b_large)
    
    results["multiply_small"] = time_function(np.multiply, a_small, b_small)
    results["multiply_medium"] = time_function(np.multiply, a_medium, b_medium)
    results["multiply_large"] = time_function(np.multiply, a_large, b_large)
    
    results["sqrt_small"] = time_function(np.sqrt, a_small)
    results["sqrt_medium"] = time_function(np.sqrt, a_medium)
    results["sqrt_large"] = time_function(np.sqrt, a_large)
    
    results["exp_small"] = time_function(np.exp, a_small)
    results["exp_medium"] = time_function(np.exp, a_medium)
    results["exp_large"] = time_function(np.exp, a_large)
    
    results["log_small"] = time_function(np.log, a_small)
    results["log_medium"] = time_function(np.log, a_medium)
    results["log_large"] = time_function(np.log, a_large)
    
    # Reductions
    results["sum_small"] = time_function(np.sum, a_small)
    results["sum_medium"] = time_function(np.sum, a_medium)
    results["sum_large"] = time_function(np.sum, a_large)
    
    results["mean_small"] = time_function(np.mean, a_small)
    results["mean_medium"] = time_function(np.mean, a_medium)
    results["mean_large"] = time_function(np.mean, a_large)
    
    results["std_small"] = time_function(np.std, a_small)
    results["std_medium"] = time_function(np.std, a_medium)
    results["std_large"] = time_function(np.std, a_large)
    
    # Array manipulation
    results["reshape_small"] = time_function(np.reshape, a_small, (SMALL_SIZE // 10, 10))
    results["reshape_medium"] = time_function(np.reshape, a_medium, (MEDIUM_SIZE // 100, 100))
    results["reshape_large"] = time_function(np.reshape, a_large, (LARGE_SIZE // 1000, 1000))
    
    results["transpose_small"] = time_function(lambda x: x.T, np.reshape(a_small, (SMALL_SIZE // 10, 10)))
    results["transpose_medium"] = time_function(lambda x: x.T, np.reshape(a_medium, (MEDIUM_SIZE // 100, 100)))
    
    return results

def benchmark_linear_algebra():
    """Benchmark linear algebra operations"""
    results = {}
    
    # Create matrices for operations
    A_small = np.random.random(MATRIX_SMALL)
    A_medium = np.random.random(MATRIX_MEDIUM)
    
    B_small = np.random.random(MATRIX_SMALL)
    B_medium = np.random.random(MATRIX_MEDIUM)
    
    # Square matrices for decompositions
    S_small = np.random.random((100, 100))
    S_medium = np.random.random((500, 500))
    
    # Matrix operations
    results["matmul_small"] = time_function(np.matmul, A_small, B_small)
    results["matmul_medium"] = time_function(np.matmul, A_medium[:100, :100], B_medium[:100, :100])
    
    results["dot_small"] = time_function(np.dot, A_small.flatten()[:100], B_small.flatten()[:100])
    results["dot_medium"] = time_function(np.dot, A_medium.flatten()[:1000], B_medium.flatten()[:1000])
    
    # Matrix decompositions
    results["svd_small"] = time_function(np.linalg.svd, S_small)
    results["svd_medium"] = time_function(np.linalg.svd, S_medium)
    
    results["qr_small"] = time_function(np.linalg.qr, S_small)
    results["qr_medium"] = time_function(np.linalg.qr, S_medium)
    
    results["cholesky_small"] = time_function(np.linalg.cholesky, S_small @ S_small.T)  # Ensure positive definite
    results["cholesky_medium"] = time_function(np.linalg.cholesky, S_medium @ S_medium.T)
    
    results["det_small"] = time_function(np.linalg.det, S_small)
    results["det_medium"] = time_function(np.linalg.det, S_medium)
    
    results["inv_small"] = time_function(np.linalg.inv, S_small)
    results["inv_medium"] = time_function(np.linalg.inv, S_medium)
    
    return results

def benchmark_distributions():
    """Benchmark random distributions"""
    results = {}
    
    # Basic distributions
    results["normal_small"] = time_function(np.random.normal, 0, 1, SMALL_SIZE)
    results["normal_medium"] = time_function(np.random.normal, 0, 1, MEDIUM_SIZE)
    results["normal_large"] = time_function(np.random.normal, 0, 1, LARGE_SIZE)
    
    results["uniform_small"] = time_function(np.random.uniform, 0, 1, SMALL_SIZE)
    results["uniform_medium"] = time_function(np.random.uniform, 0, 1, MEDIUM_SIZE)
    results["uniform_large"] = time_function(np.random.uniform, 0, 1, LARGE_SIZE)
    
    results["exponential_small"] = time_function(np.random.exponential, 1, SMALL_SIZE)
    results["exponential_medium"] = time_function(np.random.exponential, 1, MEDIUM_SIZE)
    results["exponential_large"] = time_function(np.random.exponential, 1, LARGE_SIZE)
    
    results["beta_small"] = time_function(np.random.beta, 2, 5, SMALL_SIZE)
    results["beta_medium"] = time_function(np.random.beta, 2, 5, MEDIUM_SIZE)
    results["beta_large"] = time_function(np.random.beta, 2, 5, LARGE_SIZE)
    
    results["gamma_small"] = time_function(np.random.gamma, 2, 2, SMALL_SIZE)
    results["gamma_medium"] = time_function(np.random.gamma, 2, 2, MEDIUM_SIZE)
    results["gamma_large"] = time_function(np.random.gamma, 2, 2, LARGE_SIZE)
    
    # Import scipy.stats for advanced distributions
    try:
        import scipy.stats as stats
        
        # Advanced distributions
        results["noncentral_chisquare_small"] = time_function(stats.ncx2.rvs, 2, 1, size=SMALL_SIZE)
        results["noncentral_chisquare_medium"] = time_function(stats.ncx2.rvs, 2, 1, size=MEDIUM_SIZE)
        
        results["noncentral_f_small"] = time_function(stats.ncf.rvs, 2, 5, 1, size=SMALL_SIZE)
        results["noncentral_f_medium"] = time_function(stats.ncf.rvs, 2, 5, 1, size=MEDIUM_SIZE)
        
        results["vonmises_small"] = time_function(stats.vonmises.rvs, 1, size=SMALL_SIZE)
        results["vonmises_medium"] = time_function(stats.vonmises.rvs, 1, size=MEDIUM_SIZE)
        
    except ImportError:
        print("SciPy not available. Skipping advanced distribution benchmarks.")
    
    return results

def run_benchmarks():
    """Run all benchmarks and save results to a JSON file"""
    results = {}
    
    print("Running NumPy Benchmarks for comparison with NumRS2")
    print("-" * 50)
    
    # Get system information
    results["system_info"] = get_system_info()
    results["timestamp"] = datetime.now().isoformat()
    
    # Run benchmarks
    print("Benchmarking array creation...")
    results["array_creation"] = benchmark_array_creation()
    
    print("Benchmarking array operations...")
    results["array_operations"] = benchmark_array_operations()
    
    print("Benchmarking linear algebra...")
    results["linear_algebra"] = benchmark_linear_algebra()
    
    print("Benchmarking distributions...")
    results["distributions"] = benchmark_distributions()
    
    # Save results to JSON file
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark results saved to {RESULTS_FILE}")
    return results

if __name__ == "__main__":
    run_benchmarks()
    
    # Print summary of results
    with open(RESULTS_FILE, 'r') as f:
        results = json.load(f)
    
    print("\nBenchmark Summary (mean times in ms):")
    print("-" * 50)
    
    # Print a few key results
    print("Array Creation:")
    print(f"  zeros_medium: {results['array_creation']['zeros_medium']['mean']:.2f} ms")
    print(f"  random_medium: {results['array_creation']['random_medium']['mean']:.2f} ms")
    
    print("\nArray Operations:")
    print(f"  add_medium: {results['array_operations']['add_medium']['mean']:.2f} ms")
    print(f"  multiply_medium: {results['array_operations']['multiply_medium']['mean']:.2f} ms")
    print(f"  sqrt_medium: {results['array_operations']['sqrt_medium']['mean']:.2f} ms")
    
    print("\nLinear Algebra:")
    print(f"  matmul_small: {results['linear_algebra']['matmul_small']['mean']:.2f} ms")
    print(f"  svd_small: {results['linear_algebra']['svd_small']['mean']:.2f} ms")
    print(f"  qr_small: {results['linear_algebra']['qr_small']['mean']:.2f} ms")
    
    print("\nDistributions:")
    print(f"  normal_medium: {results['distributions']['normal_medium']['mean']:.2f} ms")
    print(f"  beta_medium: {results['distributions']['beta_medium']['mean']:.2f} ms")
    
    if "noncentral_chisquare_medium" in results["distributions"]:
        print(f"  noncentral_chisquare_medium: {results['distributions']['noncentral_chisquare_medium']['mean']:.2f} ms")
        
    print("\nRun this benchmark on the same system as NumRS2 benchmarks for valid comparisons.")