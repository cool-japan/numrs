# Special Functions Testing Guide

This document provides an overview of the testing approach for special functions in NumRS2.

## Testing Strategy

The special functions in NumRS2 are tested using a multi-faceted approach:

1. **Property-based testing**: Testing mathematical properties and relationships
2. **Reference testing**: Testing against known reference values
3. **Performance benchmarking**: Measuring and optimizing performance

### Property-Based Testing

Property-based testing focuses on verifying the mathematical properties and relationships that special functions should satisfy. These include:

- **Functional identities**: e.g., erf(-x) = -erf(x)
- **Recurrence relations**: e.g., gamma(x+1) = x * gamma(x)
- **Limiting behaviors**: e.g., erf(x) → 1 as x → ∞
- **Inverse relationships**: e.g., erf(erfinv(x)) = x
- **Compound function relationships**: e.g., erfc(x) = 1 - erf(x)

Property-based testing helps ensure that the implementations are mathematically correct beyond just matching specific reference values.

### Reference Testing

Reference tests compare the computed values of special functions against known reference values. These tests:

- Verify function values at well-known points (e.g., gamma(1) = 1, gamma(0.5) = √π)
- Test function behavior at special inputs (e.g., integer and half-integer arguments)
- Check edge cases and boundary values
- Ensure numerical stability in challenging regions

### Performance Benchmarking

Performance benchmarks measure the execution time of special functions with various input sizes. These benchmarks:

- Track performance across different function implementations
- Compare vectorized vs. scalar implementations
- Identify performance bottlenecks
- Help optimize for different input sizes and scenarios

## Testing Strategy for Specific Functions

### Error Functions

Error functions (erf, erfc, erfinv, erfcinv) are tested for:

- Antisymmetry/symmetry properties
- Complementary relationship: erf(x) + erfc(x) = 1
- Inverse relationships: erf(erfinv(x)) = x, erfc(erfcinv(x)) = x
- Limiting behaviors as x approaches ±∞
- Reference values from mathematical tables

### Gamma Functions

Gamma functions (gamma, gammaln, digamma, gammainc) are tested for:

- Recurrence relations: gamma(x+1) = x * gamma(x)
- Special values: gamma(n) = (n-1)! for positive integers
- Half-integer values: gamma(n+0.5) for integers n
- Logarithmic properties: gammaln(x) = ln(gamma(x))
- Digamma recurrence: digamma(x+1) = digamma(x) + 1/x
- Numerical stability for large values

### Bessel Functions

Bessel functions (bessel_j, bessel_y, bessel_i, bessel_k) are tested for:

- Recurrence relations
- Special values at x=0
- Symmetry properties for negative orders
- Limiting behaviors for large arguments
- Wronskian and other differential relationships
- Reference values from mathematical tables

### Elliptic Integrals

Elliptic integrals (ellipk, ellipe) are tested for:

- Special values at m=0 and m=1
- Relationship between first and second kind
- Monotonicity properties
- Reference values from mathematical tables

## Running the Tests

To run tests for special functions, use the following commands:

```bash
# Run property-based tests for special functions
cargo test --test test_special_properties

# Run reference tests against known values
cargo test --test test_special_reference

# Run performance benchmarks
cargo bench --bench special_functions_benchmark
```

## Contributing New Tests

When adding new special functions, please also add corresponding tests:

1. **Add property tests** in `tests/test_special_properties.rs`
2. **Add reference tests** in `tests/test_special_reference.rs`
3. **Add benchmarks** in `benches/special_functions_benchmark.rs`

Focus on testing:
- Mathematical properties and identities
- Well-known reference values
- Edge cases and numerical stability
- Performance characteristics

## References

For verifying special function values, we use the following references:

1. Abramowitz, M. and Stegun, I. A. (1972), *Handbook of Mathematical Functions*
2. NIST Digital Library of Mathematical Functions, https://dlmf.nist.gov/
3. SciPy special functions implementation
4. Numerical Recipes in C