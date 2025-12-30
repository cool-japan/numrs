# Random Distributions Testing Guide

This document outlines the approach to testing random distributions in NumRS2, focusing on property-based testing for verifying the correctness of the implementations.

## Testing Philosophy

Testing random distributions is inherently challenging because they're, well, random. However, we can use statistical properties and relationships between distributions to verify that our implementations behave correctly. Our testing approach combines:

1. **Statistical property testing**: Verifying that distributions exhibit their expected statistical properties (mean, variance, etc.)
2. **Distribution relationships**: Testing known mathematical relationships between different distributions
3. **Boundary behavior**: Testing how distributions behave at parameter boundaries
4. **Edge cases**: Testing specific edge cases where distributions should exhibit predictable behavior

## Test File Structure

The random distributions testing suite consists of several files:

1. `test_random_properties.rs`: Basic property tests for common distributions
2. `test_random_statistical.rs`: More comprehensive statistical tests
3. `test_random_advanced.rs`: Tests for relationships between distributions and complex properties
4. `test_random_advanced_distributions.rs`: Tests focused on less common distributions
5. `test_random_reference.rs`: Tests against reference values
6. `test_random_state.rs`: Tests for random state management
7. `test_random_values_collector.rs`: Helper tools for collecting distribution outputs

## Key Testing Approaches

### Statistical Property Testing

Statistical property testing verifies that random samples exhibit the expected statistical properties of their theoretical distributions:

```rust
// Example: Testing normal distribution properties
let samples = random::normal(mean, std_dev, &[SAMPLE_SIZE]).unwrap();
let sample_mean = calculate_mean(&samples);
let sample_variance = calculate_variance(&samples, sample_mean);

// Verify that mean is close to expected
assert!(is_within_bounds(sample_mean, mean, 0.1 * std_dev));

// Verify that variance is close to expected
assert!(is_within_bounds(sample_variance, std_dev * std_dev, 0.2 * std_dev * std_dev));
```

### Distribution Relationships

Many distributions have mathematical relationships with each other. Testing these relationships provides additional validation:

```rust
// Example: Chi-square is the sum of squared standard normal variables
let normal_samples = random::standard_normal::<f64>(&[k]).unwrap();
let sum_of_squares: f64 = normal_samples.to_vec().iter().map(|x| x * x).sum();
// sum_of_squares should follow chi-square with k degrees of freedom
```

### Conformance Testing

For some distributions, we use goodness-of-fit tests like Kolmogorov-Smirnov to test conformance to theoretical distributions:

```rust
// Example: Testing if samples follow standard normal using KS test
let ks_statistic = calculate_ks_statistic(&samples, normal_cdf);
assert!(ks_statistic <= critical_value);
```

### Parameter Boundary Testing

We test how distributions behave with extreme parameter values:

```rust
// Example: Testing exponential with very small scale
let small_scale = 1e-5;
let samples = random::exponential(small_scale, &[1000]).unwrap();
let sample_mean = calculate_mean(&samples);
assert!(is_within_bounds(sample_mean, small_scale, small_scale * 0.5));
```

## Distribution Specific Tests

### Continuous Distributions

1. **Normal Distribution**:
   - Mean should match specified mean
   - Variance should match square of specified standard deviation
   - Skewness should be close to 0
   - Kurtosis should be close to 0

2. **Uniform Distribution**:
   - Mean should be (low + high) / 2
   - Variance should be (high - low)² / 12
   - All values should be within [low, high]

3. **Beta Distribution**:
   - Mean should be α / (α + β)
   - Variance should match theoretical formula
   - All values should be within [0, 1]
   - Beta(1,1) should be equivalent to Uniform(0,1)

4. **Exponential Distribution**:
   - Mean should be scale
   - Variance should be scale²
   - All values should be positive

5. **Cauchy Distribution**:
   - Median should equal location parameter
   - 25th and 75th percentiles should be at expected distances

### Discrete Distributions

1. **Binomial Distribution**:
   - Mean should be n*p
   - Variance should be n*p*(1-p)
   - All values should be integers in [0, n]

2. **Poisson Distribution**:
   - Mean should be lambda
   - Variance should be lambda
   - All values should be non-negative integers

3. **Zipf Distribution**:
   - Frequency of each value k should be proportional to k^(-α)
   - All values should be positive integers

## Tolerance Levels

Testing random distributions requires appropriate tolerance levels. Our guidelines:

- Mean: Usually within 10-15% of expected value for most distributions
- Variance: Usually within 20-30% of expected value
- Percentiles: Usually within 10-20% for central percentiles
- Heavy-tailed distributions require wider tolerances
- Discrete distributions may require special handling

## Adding New Distribution Tests

When adding a new distribution test:

1. Identify the key statistical properties to test
2. Determine appropriate tolerance levels based on distribution characteristics
3. Test both common and edge cases
4. If applicable, test relationships with other distributions
5. Consider performance implications for large sample sizes

## Running the Tests

Run all random distribution tests:

```bash
cargo test random
```

Run tests for a specific distribution:

```bash
cargo test test_normal_distribution
```

## Reference

Statistical properties for each distribution are based on standard mathematical references:

- **Normal(μ, σ)**: Mean = μ, Variance = σ²
- **Uniform(a, b)**: Mean = (a+b)/2, Variance = (b-a)²/12
- **Beta(α, β)**: Mean = α/(α+β), Variance = αβ/((α+β)²(α+β+1))
- **Exponential(λ)**: Mean = 1/λ, Variance = 1/λ²
- **Gamma(k, θ)**: Mean = kθ, Variance = kθ²
- **Cauchy(x₀, γ)**: Mean = undefined, Median = x₀
- **Binomial(n, p)**: Mean = np, Variance = np(1-p)
- **Poisson(λ)**: Mean = λ, Variance = λ

For complete reference, see statistics textbooks or the documentation for each distribution function.