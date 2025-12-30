# NumRS2 Testing Guide

This document provides an overview of the testing approach for NumRS2, a high-performance numerical computing library for Rust.

## Testing Philosophy

NumRS2 employs multiple testing strategies to ensure correct behavior and numerical stability:

1. **Unit Tests**: Verify individual function behavior and correctness
2. **Property-Based Testing**: Validate mathematical properties and invariants
3. **Reference Testing**: Compare against known reference values
4. **Statistical Testing**: Verify statistical properties of random distributions
5. **Integration Tests**: Ensure components work together correctly
6. **Stability Tests**: Verify numerical stability for edge cases

## Test Files

### Core Functionality Tests
- `test_array_ops.rs`: Tests for array manipulation operations
- `test_fixed_features.rs`: Tests for fixed-point features
- `test_indexing.rs`: Tests for array indexing operations
- `test_masked.rs`: Tests for masked array operations
- `test_matrix_stability.rs`: Tests for matrix numerical stability

### Random Module Tests
- `test_random_properties.rs`: Basic property-based tests for random distributions
- `test_random_advanced.rs`: Advanced distribution tests focusing on relationships between distributions
- `test_random_reference.rs`: Tests against fixed reference values
- `test_random_statistical.rs`: Tests for statistical properties of distributions
- `test_random_state.rs`: Tests for RandomState implementation
- `test_random_values_collector.rs`: Helper to generate reference values

### Linear Algebra Tests
- `test_linalg_properties.rs`: Property-based tests for linear algebra operations
- `test_linalg_reference.rs`: Reference tests for linear algebra operations

### Special Functions Tests
- `test_special_properties.rs`: Property-based tests for special functions
- `test_special_reference.rs`: Reference tests for special functions
- [Special Functions Testing Guide](SPECIAL_FUNCTIONS_TESTING.md): Detailed guide for testing special functions

## Running Tests

```bash
# Run all tests
cargo test

# Run tests for a specific module
cargo test random

# Run a specific test file
cargo test --test test_random_statistical
cargo test --test test_linalg_properties
cargo test --test test_special_reference

# Run a specific test
cargo test test_normal_distribution_statistics
cargo test test_eigendecomposition_properties
cargo test test_erf_reference
```

Additionally, benchmarks can be run with:

```bash
# Run all benchmarks
cargo bench

# Run benchmarks for specific functionality
cargo bench --bench linear_algebra_benchmark
cargo bench --bench special_functions_benchmark
```

## Adding New Tests

### Unit Tests

Add unit tests in the module file itself using the standard Rust test approach:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_function_name() {
        // Test code
    }
}
```

### New Test Files

When adding a new test file, follow these guidelines:

1. Name the file with a `test_` prefix followed by the feature being tested
2. Import required modules at the top of the file
3. Write test functions with descriptive names prefixed with `test_`
4. Include comments explaining the purpose of each test

## Property-Based Testing

For property-based testing, we focus on two approaches:

1. **Mathematical Properties**: Testing that functions satisfy mathematical properties
2. **Statistical Properties**: Testing that distributions satisfy their statistical properties

### Example: Statistical Properties

```rust
#[test]
fn test_normal_distribution_statistics() {
    // Test that normal distribution satisfies expected statistical properties
    let mean = 3.0;
    let std_dev = 2.0;
    
    set_seed(42);
    let samples = random::normal(mean, std_dev, &[SAMPLE_SIZE]).unwrap();
    
    // Calculate statistics
    let sample_mean = calculate_mean(&samples);
    let sample_variance = calculate_variance(&samples, sample_mean);
    
    // Check if mean is within expected range
    assert!((sample_mean - mean).abs() < 0.1 * std_dev);
    
    // Check if variance is within expected range
    assert!((sample_variance - std_dev * std_dev).abs() < 0.15 * std_dev * std_dev);
}
```

## Reference Testing

Reference testing compares results against known values to ensure consistent behavior:

```rust
#[test]
fn test_function_reference_values() {
    // Test against known reference values
    let result = my_function(input);
    assert_eq!(result, expected_output);
}
```

## Contributing New Tests

When contributing new tests:

1. Ensure tests are deterministic and don't depend on external state
2. For random distributions, always set a fixed seed
3. Include tests for edge cases and error conditions
4. Document the purpose of each test with comments
5. Use tolerance for floating-point comparisons
6. Verify that tests pass consistently

## Test Coverage

The goal is to have comprehensive test coverage for all functions in the library. Priority is given to:

1. Core numerical operations
2. Mathematical transformations
3. Random distributions
4. Linear algebra operations
5. Special functions

## Reference Resources

For testing mathematical functions, refer to these resources:

1. NumPy documentation for expected behavior
2. Mathematical/statistical textbooks for properties and formulas
3. NIST resources for numerical algorithms