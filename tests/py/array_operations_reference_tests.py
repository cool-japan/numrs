#!/usr/bin/env python3
"""
NumPy Reference Tests for NumRS2 Array Operations

This script generates reference values from NumPy array operations
to compare against NumRS2 implementations. The output is used by Rust
reference tests to validate array operation implementations.

The test strategy:
1. Create various array configurations using fixed seeds
2. Apply NumPy operations and record results
3. Save these reference values to a JSON file for Rust tests to use
"""

import numpy as np
import json
import os
from typing import Dict, Any, List, Union

# Set a fixed seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Directory to save reference data
REFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_FILE = os.path.join(REFERENCE_DIR, "array_operations_reference_data.json")

# Dictionary to store reference data
reference_data: Dict[str, Any] = {}

def serialize_array(arr: np.ndarray) -> Dict[str, Any]:
    """Convert NumPy array to serializable format"""
    return {
        "data": arr.flatten().tolist(),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype)
    }

def create_test_arrays() -> Dict[str, np.ndarray]:
    """Create standard test arrays used across multiple tests"""
    return {
        "small_1d": np.array([1, 2, 3, 4, 5], dtype=np.float64),
        "medium_1d": np.arange(10, dtype=np.float64),
        "small_2d": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64),
        "square_2d": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64),
        "rect_2d": np.arange(12, dtype=np.float64).reshape(3, 4),
        "small_3d": np.arange(24, dtype=np.float64).reshape(2, 3, 4),
        "zeros_2d": np.zeros((3, 3), dtype=np.float64),
        "ones_2d": np.ones((2, 4), dtype=np.float64),
        "random_2d": np.random.rand(4, 4),
        "negative_vals": np.array([-2, -1, 0, 1, 2], dtype=np.float64),
        "mixed_vals": np.array([1.5, -2.3, 0, 4.7, -1.1], dtype=np.float64)
    }

print("Generating reference data for array creation operations...")

# Array creation operations
creation_tests = {}
shapes_to_test = [(5,), (3, 4), (2, 3, 4), (1, 10), (0,), (0, 5)]

for i, shape in enumerate(shapes_to_test):
    # zeros
    zeros_arr = np.zeros(shape, dtype=np.float64)
    creation_tests[f"zeros_shape_{i}"] = {
        "operation": "zeros",
        "shape": list(shape),
        "result": serialize_array(zeros_arr)
    }
    
    # ones
    ones_arr = np.ones(shape, dtype=np.float64)
    creation_tests[f"ones_shape_{i}"] = {
        "operation": "ones", 
        "shape": list(shape),
        "result": serialize_array(ones_arr)
    }
    
    # full
    if np.prod(shape) > 0:  # Skip empty arrays for full
        full_arr = np.full(shape, 3.14, dtype=np.float64)
        creation_tests[f"full_shape_{i}"] = {
            "operation": "full",
            "shape": list(shape),
            "fill_value": 3.14,
            "result": serialize_array(full_arr)
        }

# arange tests
arange_tests = [
    {"start": 0, "stop": 10, "step": 1},
    {"start": 5, "stop": 0, "step": -1},
    {"start": 0, "stop": 5, "step": 0.5},
    {"start": -5, "stop": 5, "step": 2},
]

for i, params in enumerate(arange_tests):
    arr = np.arange(params["start"], params["stop"], params["step"], dtype=np.float64)
    creation_tests[f"arange_{i}"] = {
        "operation": "arange",
        "params": params,
        "result": serialize_array(arr)
    }

# linspace tests
linspace_tests = [
    {"start": 0, "stop": 10, "num": 11},
    {"start": -1, "stop": 1, "num": 5},
    {"start": 0, "stop": 1, "num": 101},
]

for i, params in enumerate(linspace_tests):
    arr = np.linspace(params["start"], params["stop"], params["num"], dtype=np.float64)
    creation_tests[f"linspace_{i}"] = {
        "operation": "linspace",
        "params": params,
        "result": serialize_array(arr)
    }

reference_data["array_creation"] = creation_tests

print("Generating reference data for array manipulation operations...")

# Array manipulation operations
test_arrays = create_test_arrays()
manipulation_tests = {}

# Reshape operations
reshape_tests = [
    {"array": "small_1d", "new_shape": [5, 1]},
    {"array": "small_1d", "new_shape": [1, 5]},
    {"array": "medium_1d", "new_shape": [2, 5]},
    {"array": "rect_2d", "new_shape": [4, 3]},
    {"array": "rect_2d", "new_shape": [2, 6]},
    {"array": "small_3d", "new_shape": [6, 4]},
    {"array": "small_3d", "new_shape": [24]},
]

for i, test in enumerate(reshape_tests):
    arr = test_arrays[test["array"]]
    reshaped = arr.reshape(test["new_shape"])
    manipulation_tests[f"reshape_{i}"] = {
        "operation": "reshape",
        "input": serialize_array(arr),
        "new_shape": test["new_shape"],
        "result": serialize_array(reshaped)
    }

# Transpose operations
transpose_arrays = ["small_2d", "square_2d", "rect_2d", "small_3d"]
for i, arr_name in enumerate(transpose_arrays):
    arr = test_arrays[arr_name]
    transposed = arr.T
    manipulation_tests[f"transpose_{i}"] = {
        "operation": "transpose",
        "input": serialize_array(arr),
        "result": serialize_array(transposed)
    }

# Flatten operations
for i, arr_name in enumerate(["small_2d", "square_2d", "small_3d"]):
    arr = test_arrays[arr_name]
    flattened = arr.flatten()
    manipulation_tests[f"flatten_{i}"] = {
        "operation": "flatten",
        "input": serialize_array(arr),
        "result": serialize_array(flattened)
    }

# Squeeze operations
squeeze_test_arrays = [
    np.array([[[1, 2, 3]]]),  # shape (1, 1, 3)
    np.array([[1], [2], [3]]),  # shape (3, 1)
    np.array([[[1]], [[2]]]),  # shape (2, 1, 1)
]

for i, arr in enumerate(squeeze_test_arrays):
    squeezed = arr.squeeze()
    manipulation_tests[f"squeeze_{i}"] = {
        "operation": "squeeze",
        "input": serialize_array(arr),
        "result": serialize_array(squeezed)
    }

reference_data["array_manipulation"] = manipulation_tests

print("Generating reference data for arithmetic operations...")

# Arithmetic operations
arithmetic_tests = {}

# Element-wise operations
test_pairs = [
    ("small_1d", "small_1d"),
    ("small_2d", "small_2d"), 
    ("square_2d", "square_2d"),
]

operations = ["add", "subtract", "multiply", "divide"]

for op in operations:
    for i, (arr1_name, arr2_name) in enumerate(test_pairs):
        arr1 = test_arrays[arr1_name]
        arr2 = test_arrays[arr2_name]
        
        if op == "add":
            result = arr1 + arr2
        elif op == "subtract":
            result = arr1 - arr2
        elif op == "multiply":
            result = arr1 * arr2
        elif op == "divide":
            # Avoid division by zero
            arr2_safe = arr2 + 1e-10
            result = arr1 / arr2_safe
        
        arithmetic_tests[f"{op}_{i}"] = {
            "operation": op,
            "input1": serialize_array(arr1),
            "input2": serialize_array(arr2 if op != "divide" else arr2_safe),
            "result": serialize_array(result)
        }

# Scalar operations
scalar_ops = [
    {"op": "add", "scalar": 5.0},
    {"op": "subtract", "scalar": 2.5},
    {"op": "multiply", "scalar": 3.0},
    {"op": "divide", "scalar": 2.0},
]

for i, test in enumerate(scalar_ops):
    arr = test_arrays["small_2d"]
    scalar = test["scalar"]
    
    if test["op"] == "add":
        result = arr + scalar
    elif test["op"] == "subtract":
        result = arr - scalar
    elif test["op"] == "multiply":
        result = arr * scalar
    elif test["op"] == "divide":
        result = arr / scalar
    
    arithmetic_tests[f"scalar_{test['op']}_{i}"] = {
        "operation": f"scalar_{test['op']}",
        "input": serialize_array(arr),
        "scalar": scalar,
        "result": serialize_array(result)
    }

reference_data["arithmetic_operations"] = arithmetic_tests

print("Generating reference data for mathematical functions...")

# Mathematical functions
math_tests = {}

test_arrays_math = {
    "positive": np.array([1, 4, 9, 16, 25], dtype=np.float64),
    "mixed": np.array([-2, -1, 0, 1, 2], dtype=np.float64),
    "angles": np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2], dtype=np.float64),
    "small_positive": np.array([0.1, 0.5, 1.0, 2.0, 5.0], dtype=np.float64),
}

# Basic math functions
math_functions = {
    "sqrt": lambda x: np.sqrt(np.abs(x)),  # Use abs to avoid complex numbers
    "exp": lambda x: np.exp(x),
    "log": lambda x: np.log(np.abs(x) + 1e-10),  # Avoid log of negative/zero
    "abs": lambda x: np.abs(x),
    "sin": lambda x: np.sin(x),
    "cos": lambda x: np.cos(x),
    "tan": lambda x: np.tan(x),
}

for func_name, func in math_functions.items():
    for arr_name, arr in test_arrays_math.items():
        try:
            result = func(arr)
            math_tests[f"{func_name}_{arr_name}"] = {
                "operation": func_name,
                "input": serialize_array(arr),
                "result": serialize_array(result)
            }
        except (ValueError, RuntimeWarning):
            # Skip problematic combinations
            continue

# Power operations
power_tests = [
    {"base": "positive", "exponent": 2.0},
    {"base": "small_positive", "exponent": 0.5},
    {"base": "positive", "exponent": -1.0},
]

for i, test in enumerate(power_tests):
    base_arr = test_arrays_math[test["base"]]
    result = np.power(base_arr, test["exponent"])
    math_tests[f"power_{i}"] = {
        "operation": "power",
        "base": serialize_array(base_arr),
        "exponent": test["exponent"],
        "result": serialize_array(result)
    }

# Rounding operations
rounding_arr = np.array([1.2, 2.7, -1.5, -2.3, 3.9], dtype=np.float64)
rounding_functions = {
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
}

for func_name, func in rounding_functions.items():
    result = func(rounding_arr)
    math_tests[f"{func_name}"] = {
        "operation": func_name,
        "input": serialize_array(rounding_arr),
        "result": serialize_array(result)
    }

reference_data["mathematical_functions"] = math_tests

print("Generating reference data for statistical operations...")

# Statistical operations
stats_tests = {}

stat_arrays = {
    "simple": np.array([1, 2, 3, 4, 5], dtype=np.float64),
    "with_negative": np.array([-2, -1, 0, 1, 2], dtype=np.float64),
    "2d": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64),
    "3x3": np.arange(9, dtype=np.float64).reshape(3, 3),
}

# Basic statistics
basic_stats = ["mean", "sum", "min", "max", "std", "var"]

for stat_name in basic_stats:
    for arr_name, arr in stat_arrays.items():
        stat_func = getattr(np, stat_name)
        
        # Overall statistic
        result_overall = stat_func(arr)
        stats_tests[f"{stat_name}_{arr_name}_overall"] = {
            "operation": stat_name,
            "input": serialize_array(arr),
            "axis": None,
            "result": float(result_overall)
        }
        
        # Axis-wise statistics for multidimensional arrays
        if arr.ndim > 1:
            for axis in range(arr.ndim):
                try:
                    result_axis = stat_func(arr, axis=axis)
                    stats_tests[f"{stat_name}_{arr_name}_axis{axis}"] = {
                        "operation": stat_name,
                        "input": serialize_array(arr),
                        "axis": axis,
                        "result": serialize_array(result_axis) if hasattr(result_axis, "shape") else result_axis.tolist()
                    }
                except:
                    continue

# Percentiles
percentile_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64)
percentiles = [0, 25, 50, 75, 100]

for p in percentiles:
    result = np.percentile(percentile_arr, p)
    stats_tests[f"percentile_{p}"] = {
        "operation": "percentile",
        "input": serialize_array(percentile_arr),
        "percentile": p,
        "result": float(result)
    }

reference_data["statistical_operations"] = stats_tests

print("Generating reference data for comparison operations...")

# Comparison operations
comparison_tests = {}

comp_arrays = {
    "arr1": np.array([1, 3, 5, 7], dtype=np.float64),
    "arr2": np.array([2, 3, 4, 8], dtype=np.float64),
    "equal_arr": np.array([1, 3, 5, 7], dtype=np.float64),
}

comparison_ops = {
    "greater": np.greater,
    "greater_equal": np.greater_equal,
    "less": np.less,
    "less_equal": np.less_equal,
    "equal": np.equal,
    "not_equal": np.not_equal,
}

for op_name, op_func in comparison_ops.items():
    # Array vs Array
    result = op_func(comp_arrays["arr1"], comp_arrays["arr2"])
    comparison_tests[f"{op_name}_arrays"] = {
        "operation": op_name,
        "input1": serialize_array(comp_arrays["arr1"]),
        "input2": serialize_array(comp_arrays["arr2"]),
        "result": result.tolist()
    }
    
    # Array vs Scalar
    scalar = 3.0
    result_scalar = op_func(comp_arrays["arr1"], scalar)
    comparison_tests[f"{op_name}_scalar"] = {
        "operation": f"{op_name}_scalar",
        "input": serialize_array(comp_arrays["arr1"]),
        "scalar": scalar,
        "result": result_scalar.tolist()
    }

# Array equality functions
arr_equal_result = np.array_equal(comp_arrays["arr1"], comp_arrays["equal_arr"])
comparison_tests["array_equal_true"] = {
    "operation": "array_equal",
    "input1": serialize_array(comp_arrays["arr1"]),
    "input2": serialize_array(comp_arrays["equal_arr"]),
    "result": bool(arr_equal_result)
}

arr_not_equal_result = np.array_equal(comp_arrays["arr1"], comp_arrays["arr2"])
comparison_tests["array_equal_false"] = {
    "operation": "array_equal",
    "input1": serialize_array(comp_arrays["arr1"]),
    "input2": serialize_array(comp_arrays["arr2"]),
    "result": bool(arr_not_equal_result)
}

# allclose
close_arr1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
close_arr2 = np.array([1.0000001, 2.0000002, 3.0000003], dtype=np.float64)
not_close_arr = np.array([1.01, 2.02, 3.03], dtype=np.float64)

allclose_true = np.allclose(close_arr1, close_arr2)
comparison_tests["allclose_true"] = {
    "operation": "allclose",
    "input1": serialize_array(close_arr1),
    "input2": serialize_array(close_arr2),
    "result": bool(allclose_true)
}

allclose_false = np.allclose(close_arr1, not_close_arr)
comparison_tests["allclose_false"] = {
    "operation": "allclose",
    "input1": serialize_array(close_arr1),
    "input2": serialize_array(not_close_arr),
    "result": bool(allclose_false)
}

reference_data["comparison_operations"] = comparison_tests

print("Generating reference data for indexing and slicing...")

# Indexing and slicing operations
indexing_tests = {}

# Basic indexing
test_2d = np.arange(12, dtype=np.float64).reshape(3, 4)
test_3d = np.arange(24, dtype=np.float64).reshape(2, 3, 4)

# Single element access
indexing_tests["get_2d_00"] = {
    "operation": "get",
    "input": serialize_array(test_2d),
    "indices": [0, 0],
    "result": float(test_2d[0, 0])
}

indexing_tests["get_2d_12"] = {
    "operation": "get",
    "input": serialize_array(test_2d),
    "indices": [1, 2],
    "result": float(test_2d[1, 2])
}

indexing_tests["get_3d_123"] = {
    "operation": "get",
    "input": serialize_array(test_3d),
    "indices": [1, 2, 3],
    "result": float(test_3d[1, 2, 3])
}

# Row/column slicing
row_slice = test_2d[0, :]
indexing_tests["slice_row_0"] = {
    "operation": "slice_row",
    "input": serialize_array(test_2d),
    "row": 0,
    "result": serialize_array(row_slice)
}

col_slice = test_2d[:, 1]
indexing_tests["slice_col_1"] = {
    "operation": "slice_col",
    "input": serialize_array(test_2d),
    "col": 1,
    "result": serialize_array(col_slice)
}

reference_data["indexing_operations"] = indexing_tests

print("Saving reference data...")

# Save reference data to JSON file
with open(REFERENCE_FILE, 'w') as f:
    json.dump(reference_data, f, indent=2)

print(f"Reference data saved to {REFERENCE_FILE}")
print("Total test categories:", len(reference_data))
for category, tests in reference_data.items():
    print(f"  {category}: {len(tests)} tests")
print("This data can be used by Rust tests to validate array operation implementations.")