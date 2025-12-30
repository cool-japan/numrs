# NumRS2 Python Reference Tests

This directory contains Python scripts for generating reference data to validate NumRS2 implementations against NumPy and SciPy.

## Purpose

These scripts are used to:

1. Generate reference values from NumPy/SciPy functions
2. Save these reference values in a format that can be loaded by Rust tests
3. Provide a baseline for validating the correctness of NumRS2 implementations

## Available Reference Tests

- `distribution_reference_tests.py`: Generates reference data for NumRS2's random distributions by sampling from equivalent NumPy/SciPy distributions with fixed seeds.

## How to Run

1. Ensure you have NumPy and SciPy installed:
   ```
   pip install numpy scipy
   ```

2. Run the reference generation script:
   ```
   python distribution_reference_tests.py
   ```

3. This will generate a JSON file with reference data that will be used by Rust tests in `tests/test_distribution_reference.rs`.

## How to Update

You should update the reference data whenever:

1. NumRS2 distributions are modified
2. New distributions are added to NumRS2
3. You want to test against a newer version of NumPy/SciPy

After running the Python script, the Rust tests should automatically use the updated reference data.

## Test Strategy

The reference tests work by:

1. Generating a large number of samples from both NumRS2 and NumPy/SciPy distributions with the same seed
2. Computing statistical properties (mean, variance, etc.) for both sets of samples
3. Asserting that the properties from NumRS2 are within acceptable tolerances of the NumPy/SciPy values

This approach is more robust than comparing individual samples because:
- It allows for small implementation differences that don't affect statistical properties
- It verifies that the distributions behave correctly in aggregate
- It's less sensitive to minor platform-specific floating-point differences

## Adding New Reference Tests

To add a new reference test:

1. Create a new Python script in this directory
2. Generate reference data and save it to a JSON file
3. Create a corresponding Rust test that loads and validates against this data