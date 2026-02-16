#!/usr/bin/env python3
"""
NumRS2 Basic Usage Example

Demonstrates core array operations and creation functions.
"""

import numrs2 as nr
import numpy as np


def main():
    print("NumRS2 Basic Usage Examples")
    print("=" * 60)
    print(f"NumRS2 version: {nr.__version__}")
    print()

    # Array creation
    print("1. Array Creation")
    print("-" * 60)

    # From Python list
    a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Array from list: {a}")
    print(f"Shape: {a.shape}, Size: {a.size}, Dimensions: {a.ndim}")
    print()

    # From NumPy array
    np_arr = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
    b = nr.array(np_arr)
    print(f"Array from NumPy: {b}")
    print()

    # Zeros and ones
    zeros = nr.zeros([2, 3])
    print(f"Zeros (2x3): {zeros}")
    print(f"Shape: {zeros.shape}")
    print()

    ones = nr.ones([3, 2])
    print(f"Ones (3x2): {ones}")
    print()

    # Identity matrix
    identity = nr.eye(3)
    print(f"Identity (3x3): {identity}")
    print()

    # Linspace
    lin = nr.linspace(0.0, 1.0, 11)
    print(f"Linspace [0, 1] with 11 points: {lin.tolist()}")
    print()

    # Arange
    rng = nr.arange(0.0, 10.0, 2.0)
    print(f"Arange [0, 10) step 2: {rng.tolist()}")
    print()

    # Array operations
    print("2. Array Operations")
    print("-" * 60)

    # Reshape
    flat = nr.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    matrix = flat.reshape([2, 3])
    print(f"Original: {flat}")
    print(f"Reshaped to 2x3: {matrix}")
    print()

    # Transpose
    transposed = matrix.transpose()
    print(f"Transposed: {transposed}")
    print(f"Shape: {transposed.shape}")
    print()

    # Element-wise operations
    print("3. Element-wise Operations")
    print("-" * 60)

    x = nr.array([1.0, 2.0, 3.0])
    y = nr.array([4.0, 5.0, 6.0])

    add = x + y
    print(f"{x.tolist()} + {y.tolist()} = {add.tolist()}")

    sub = y - x
    print(f"{y.tolist()} - {x.tolist()} = {sub.tolist()}")

    mul = x * y
    print(f"{x.tolist()} * {y.tolist()} = {mul.tolist()}")

    div = y / x
    print(f"{y.tolist()} / {x.tolist()} = {div.tolist()}")
    print()

    # Aggregations
    print("4. Aggregation Operations")
    print("-" * 60)

    data = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"Data: {data.tolist()}")
    print(f"Sum: {data.sum()}")
    print(f"Mean: {data.mean()}")
    print(f"Min: {data.min()}")
    print(f"Max: {data.max()}")
    print()

    # NumPy interoperability
    print("5. NumPy Interoperability")
    print("-" * 60)

    nr_array = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
    print(f"NumRS2 array: {nr_array}")

    # Convert to NumPy
    numpy_array = nr_array.to_numpy(None)
    print(f"Converted to NumPy: {numpy_array}")
    print(f"Type: {type(numpy_array)}")
    print()

    # Back to NumRS2
    back_to_nr = nr.array(numpy_array)
    print(f"Back to NumRS2: {back_to_nr}")
    print()

    print("=" * 60)
    print("Examples completed successfully!")


if __name__ == "__main__":
    main()
