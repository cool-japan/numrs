#!/usr/bin/env python3
"""
NumRS2 NumPy Migration Guide

Side-by-side comparison of NumPy and NumRS2 syntax to help with migration.
"""

import numpy as np
import numrs2 as nr


def main():
    print("NumPy to NumRS2 Migration Guide")
    print("=" * 80)
    print()

    print("This example shows equivalent operations in NumPy and NumRS2")
    print()

    # Array creation
    print("1. Array Creation")
    print("-" * 80)

    print("NumPy:  np.array([1, 2, 3])")
    np_arr = np.array([1.0, 2.0, 3.0])
    print(f"Result: {np_arr}")

    print("NumRS2: nr.array([1, 2, 3])")
    nr_arr = nr.array([1.0, 2.0, 3.0])
    print(f"Result: {nr_arr.tolist()}")
    print()

    # Zeros and ones
    print("2. Zeros and Ones")
    print("-" * 80)

    print("NumPy:  np.zeros([2, 3])")
    print(f"Result shape: {np.zeros([2, 3]).shape}")

    print("NumRS2: nr.zeros([2, 3])")
    print(f"Result shape: {nr.zeros([2, 3]).shape}")
    print()

    # Linspace
    print("3. Linspace")
    print("-" * 80)

    print("NumPy:  np.linspace(0, 1, 5)")
    np_lin = np.linspace(0, 1, 5)
    print(f"Result: {np_lin}")

    print("NumRS2: nr.linspace(0.0, 1.0, 5)")
    nr_lin = nr.linspace(0.0, 1.0, 5)
    print(f"Result: {nr_lin.tolist()}")
    print()

    # Reshape
    print("4. Reshape")
    print("-" * 80)

    print("NumPy:  arr.reshape([2, 3])")
    np_reshape = np.array([1, 2, 3, 4, 5, 6]).reshape([2, 3])
    print(f"Result shape: {np_reshape.shape}")

    print("NumRS2: arr.reshape([2, 3])")
    nr_reshape = nr.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([2, 3])
    print(f"Result shape: {nr_reshape.shape}")
    print()

    # Matrix multiplication
    print("5. Matrix Multiplication")
    print("-" * 80)

    print("NumPy:  np.matmul(A, B) or A @ B")
    np_A = np.array([[1, 2], [3, 4]])
    np_B = np.array([[5, 6], [7, 8]])
    np_C = np.matmul(np_A, np_B)
    print(f"Result: {np_C}")

    print("NumRS2: nr.matmul(A, B)")
    nr_A = nr.array([1.0, 2.0, 3.0, 4.0]).reshape([2, 2])
    nr_B = nr.array([5.0, 6.0, 7.0, 8.0]).reshape([2, 2])
    nr_C = nr.matmul(nr_A, nr_B)
    print(f"Result: {nr_C.tolist()}")
    print()

    # Aggregations
    print("6. Aggregations")
    print("-" * 80)

    data_np = np.array([1, 2, 3, 4, 5])
    data_nr = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])

    print("NumPy:  np.mean(arr), np.sum(arr), np.min(arr), np.max(arr)")
    print(f"Mean: {np.mean(data_np)}, Sum: {np.sum(data_np)}")

    print("NumRS2: arr.mean(), arr.sum(), arr.min(), arr.max()")
    print(f"Mean: {data_nr.mean()}, Sum: {data_nr.sum()}")
    print()

    # Linear algebra
    print("7. Linear Algebra")
    print("-" * 80)

    print("NumPy:  np.linalg.det(M), np.linalg.inv(M)")
    M_np = np.array([[4, 7], [2, 6]])
    print(f"det(M): {np.linalg.det(M_np):.6f}")

    print("NumRS2: nr.linalg.det(M), nr.linalg.inv(M)")
    M_nr = nr.array([4.0, 7.0, 2.0, 6.0]).reshape([2, 2])
    print(f"det(M): {nr.linalg.det(M_nr):.6f}")
    print()

    # Statistics
    print("8. Statistics")
    print("-" * 80)

    print("NumPy:  np.std(arr), np.var(arr), np.median(arr)")
    stats_np = np.array([1, 2, 3, 4, 5])
    print(f"Std: {np.std(stats_np):.6f}, Median: {np.median(stats_np)}")

    print("NumRS2: nr.stats.std(arr), nr.stats.var(arr), nr.stats.median(arr)")
    stats_nr = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Std: {nr.stats.std(stats_nr):.6f}, Median: {nr.stats.median(stats_nr)}")
    print()

    # Interoperability
    print("9. NumPy Interoperability")
    print("-" * 80)

    print("Converting between NumPy and NumRS2:")
    print()

    # NumPy -> NumRS2
    print("NumPy to NumRS2:")
    np_original = np.array([1, 2, 3, 4, 5])
    print(f"NumPy array: {np_original}")

    nr_from_np = nr.array(np_original)
    print(f"NumRS2 array: {nr_from_np.tolist()}")
    print()

    # NumRS2 -> NumPy
    print("NumRS2 to NumPy:")
    nr_original = nr.array([6.0, 7.0, 8.0, 9.0, 10.0])
    print(f"NumRS2 array: {nr_original.tolist()}")

    np_from_nr = nr_original.to_numpy(None)
    print(f"NumPy array: {np_from_nr}")
    print()

    # Key differences
    print("10. Key Differences")
    print("-" * 80)

    print("Differences to be aware of when migrating:")
    print()
    print("1. All NumRS2 operations use float64 by default")
    print("2. Some NumPy functions are in submodules (e.g., nr.linalg.*, nr.stats.*)")
    print("3. Axis parameters may not be supported for all operations yet")
    print("4. Use .tolist() to convert NumRS2 arrays to Python lists")
    print("5. Use .to_numpy() to convert NumRS2 arrays to NumPy arrays")
    print()

    print("=" * 80)
    print("Migration guide completed!")
    print()
    print("For more information, see the NumRS2 documentation:")
    print("https://github.com/cool-japan/numrs")


if __name__ == "__main__":
    main()
