# Random Distributions Reference for NumRS2

This document provides an overview of all the probability distributions available in the NumRS2 random module, along with their parameters and typical use cases.

## Continuous Distributions

### Uniform

```rust
// Using global generator
random::uniform(low, high, &[shape])?;

// Using specific generator
rng.uniform(low, high, &[shape])?;
```

**Parameters:**
- `low`: Lower bound (inclusive)
- `high`: Upper bound (inclusive for uniform, exclusive for integers)
- `shape`: Shape of the output array

**Use cases:** When all values in a range are equally likely. Used for simulations, game mechanics, and initializing weights.

### Normal (Gaussian)

```rust
// Using global generator
random::normal(mean, std, &[shape])?;
random::standard_normal(&[shape])?;  // Mean 0, std 1

// Using specific generator
rng.normal(mean, std, &[shape])?;
rng.standard_normal(&[shape])?;
```

**Parameters:**
- `mean`: Center of the distribution
- `std`: Standard deviation
- `shape`: Shape of the output array

**Use cases:** The most commonly used distribution in statistics. Models natural phenomena, measurement errors, and many other random processes. Central Limit Theorem makes it widely applicable.

### Beta

```rust
// Using global generator
random::beta(a, b, &[shape])?;

// Using specific generator
rng.beta(a, b, &[shape])?;
```

**Parameters:**
- `a`: Alpha shape parameter (>0)
- `b`: Beta shape parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Modeling random probabilities, proportions, and percentages. Used in Bayesian statistics and project management (PERT).

### Gamma

```rust
// Using global generator
random::gamma(shape_param, scale, &[shape])?;

// Using specific generator
rng.gamma(shape_param, scale, &[shape])?;
```

**Parameters:**
- `shape_param`: Shape parameter (>0)
- `scale`: Scale parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Modeling waiting times, in particular when events occur randomly. Used in finance, reliability analysis, and queueing theory.

### Chi-Square

```rust
// Using global generator
random::chisquare(df, &[shape])?;

// Using specific generator
rng.chisquare(df, &[shape])?;
```

**Parameters:**
- `df`: Degrees of freedom (>0)
- `shape`: Shape of the output array

**Use cases:** Statistical hypothesis testing, confidence intervals, and quality control. Special case of gamma distribution.

### Exponential

```rust
// Using global generator
random::exponential(scale, &[shape])?;

// Using specific generator
rng.exponential(scale, &[shape])?;
```

**Parameters:**
- `scale`: Scale parameter (>0), inverse of rate
- `shape`: Shape of the output array

**Use cases:** Modeling time between independent events occurring at a constant rate, like radioactive decay or arrival times.

### Weibull

```rust
// Using global generator
random::weibull(shape_param, scale, &[shape])?;

// Using specific generator
rng.weibull(shape_param, scale, &[shape])?;
```

**Parameters:**
- `shape_param`: Shape parameter (>0)
- `scale`: Scale parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Reliability engineering and survival analysis. Models time-to-failure and material strength.

### Lognormal

```rust
// Using global generator
random::lognormal(mean, sigma, &[shape])?;

// Using specific generator
rng.lognormal(mean, sigma, &[shape])?;
```

**Parameters:**
- `mean`: Mean of the underlying normal distribution
- `sigma`: Standard deviation of the underlying normal distribution
- `shape`: Shape of the output array

**Use cases:** Modeling growth processes, income distribution, stock prices, and biological processes.

### Pareto

```rust
// Using global generator
random::pareto(alpha, &[shape])?;

// Using specific generator
rng.pareto(alpha, &[shape])?;
```

**Parameters:**
- `alpha`: Shape parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Modeling the 80/20 rule, distribution of wealth, city sizes, and many other "power law" phenomena.

### Cauchy (Lorentz)

```rust
// Using global generator
random::cauchy(loc, scale, &[shape])?;

// Using specific generator
rng.cauchy(loc, scale, &[shape])?;
```

**Parameters:**
- `loc`: Location parameter
- `scale`: Scale parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Modeling resonance behavior, spectral lines, and certain physical phenomena.

### Student's t

```rust
// Using global generator
random::student_t(df, &[shape])?;

// Using specific generator
rng.student_t(df, &[shape])?;
```

**Parameters:**
- `df`: Degrees of freedom (>0)
- `shape`: Shape of the output array

**Use cases:** Statistical hypothesis testing, especially when sample sizes are small.

### Laplace

```rust
// Using global generator
random::laplace(loc, scale, &[shape])?;

// Using specific generator
rng.laplace(loc, scale, &[shape])?;
```

**Parameters:**
- `loc`: Location parameter
- `scale`: Scale parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Modeling error distributions in machine learning and other areas where errors are double exponentially distributed.

### Gumbel

```rust
// Using global generator
random::gumbel(loc, scale, &[shape])?;

// Using specific generator
rng.gumbel(loc, scale, &[shape])?;
```

**Parameters:**
- `loc`: Location parameter
- `scale`: Scale parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Extreme value analysis like flooding, rainfall, and maximum wind speeds.

### Logistic

```rust
// Using global generator
random::logistic(loc, scale, &[shape])?;

// Using specific generator
rng.logistic(loc, scale, &[shape])?;
```

**Parameters:**
- `loc`: Location parameter
- `scale`: Scale parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Growth models, neural networks (sigmoid activation), and logistic regression.

### Rayleigh

```rust
// Using global generator
random::rayleigh(scale, &[shape])?;

// Using specific generator
rng.rayleigh(scale, &[shape])?;
```

**Parameters:**
- `scale`: Scale parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Modeling wind speeds, wave heights, and in communication theory.

### Triangular

```rust
// Using global generator
random::triangular(low, mode, high, &[shape])?;

// Using specific generator
rng.triangular(low, mode, high, &[shape])?;
```

**Parameters:**
- `low`: Lower limit
- `mode`: Mode (peak of the distribution)
- `high`: Upper limit
- `shape`: Shape of the output array

**Use cases:** Project management, risk analysis when minimum, maximum, and most likely values are known.

### PERT

```rust
// Using global generator
random::pert(min, mode, max, &[shape])?;

// Using specific generator
rng.pert(min, mode, max, &[shape])?;
```

**Parameters:**
- `min`: Minimum value
- `mode`: Most likely value
- `max`: Maximum value
- `shape`: Shape of the output array

**Use cases:** Project management and risk analysis, similar to triangular but gives more weight to the mode.

### Wald (Inverse Gaussian)

```rust
// Using global generator
random::wald(mean, scale, &[shape])?;

// Using specific generator
rng.wald(mean, scale, &[shape])?;
```

**Parameters:**
- `mean`: Mean parameter (>0)
- `scale`: Shape parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Modeling waiting times for Brownian motion to reach a certain level.

## Discrete Distributions

### Binomial

```rust
// Using global generator
random::binomial::<T>(n, p, &[shape])?;

// Using specific generator
rng.binomial::<T>(n, p, &[shape])?;
```

**Parameters:**
- `n`: Number of trials
- `p`: Probability of success in each trial (between 0 and 1)
- `shape`: Shape of the output array

**Use cases:** Modeling number of successes in a fixed number of independent trials, like coin flips or yes/no surveys.

### Poisson

```rust
// Using global generator
random::poisson::<T>(lambda, &[shape])?;

// Using specific generator
rng.poisson::<T>(lambda, &[shape])?;
```

**Parameters:**
- `lambda`: Rate parameter (average number of events, >0)
- `shape`: Shape of the output array

**Use cases:** Modeling counts of events occurring in a fixed time interval, like number of calls per hour or website visits.

### Bernoulli

```rust
// Using global generator
random::bernoulli(p, &[shape])?;

// Using specific generator
rng.bernoulli(p, &[shape])?;
```

**Parameters:**
- `p`: Probability of success (between 0 and 1)
- `shape`: Shape of the output array

**Use cases:** Modeling binary outcomes like success/failure, yes/no, true/false.

### Negative Binomial

```rust
// Using global generator
random::negative_binomial::<T>(n, p, &[shape])?;

// Using specific generator
rng.negative_binomial::<T>(n, p, &[shape])?;
```

**Parameters:**
- `n`: Number of successes
- `p`: Probability of success in each trial (between 0 and 1)
- `shape`: Shape of the output array

**Use cases:** Modeling the number of trials needed to achieve a specified number of successes.

### Geometric

```rust
// Using global generator
random::geometric::<T>(p, &[shape])?;

// Using specific generator
rng.geometric::<T>(p, &[shape])?;
```

**Parameters:**
- `p`: Probability of success (between 0 and 1)
- `shape`: Shape of the output array

**Use cases:** Modeling the number of trials until the first success, like number of coin flips until first heads.

### Hypergeometric

```rust
// Using global generator
random::hypergeometric::<T>(ngood, nbad, nsample, &[shape])?;

// Using specific generator
rng.hypergeometric::<T>(ngood, nbad, nsample, &[shape])?;
```

**Parameters:**
- `ngood`: Number of success states in the population
- `nbad`: Number of failure states in the population
- `nsample`: Number of samples drawn
- `shape`: Shape of the output array

**Use cases:** Modeling sampling without replacement, like drawing cards from a deck.

### Zipf

```rust
// Using global generator
random::zipf::<T>(a, &[shape])?;

// Using specific generator
rng.zipf::<T>(a, &[shape])?;
```

**Parameters:**
- `a`: Distribution parameter (>1)
- `shape`: Shape of the output array

**Use cases:** Modeling frequency distributions like word frequencies in text or city population ranks.

### Logseries

```rust
// Using global generator
random::logseries::<T>(p, &[shape])?;

// Using specific generator
rng.logseries::<T>(p, &[shape])?;
```

**Parameters:**
- `p`: Distribution parameter (between 0 and 1)
- `shape`: Shape of the output array

**Use cases:** Modeling certain biological processes and frequency distributions.

## Multivariate Distributions

### Multivariate Normal

```rust
// Using global generator
random::multivariate_normal(&mean, &cov, size)?;

// Using specific generator
rng.multivariate_normal(&mean, &cov, size)?;
```

**Parameters:**
- `mean`: Mean vector
- `cov`: Covariance matrix
- `size`: Optional shape of output array

**Use cases:** Modeling correlated random variables in finance, machine learning, and many scientific applications.

### Multinomial

```rust
// Using global generator
random::multinomial::<T>(n, &pvals, size)?;

// Using specific generator
rng.multinomial::<T>(n, &pvals, size)?;
```

**Parameters:**
- `n`: Number of trials
- `pvals`: Probability vector (must sum to 1)
- `size`: Optional shape of output array

**Use cases:** Modeling outcomes of experiments with multiple possible results, like dice rolls.

### Dirichlet

```rust
// Using global generator
random::dirichlet::<T>(&alpha, &[shape])?;

// Using specific generator
rng.dirichlet::<T>(&alpha, &[shape])?;
```

**Parameters:**
- `alpha`: Concentration parameters
- `shape`: Shape of the output array

**Use cases:** Modeling distributions over probability vectors, widely used in Bayesian statistics and topic modeling.

## Statistical Properties Reference

For property-based testing, the following statistical properties are useful:

### Normal(μ, σ)
- Mean = μ
- Variance = σ²
- Skewness = 0
- Kurtosis = 0

### Uniform(a, b)
- Mean = (a+b)/2
- Variance = (b-a)²/12
- Skewness = 0
- Kurtosis = -6/5

### Beta(α, β)
- Mean = α/(α+β)
- Variance = αβ/((α+β)²(α+β+1))
- Special case: Beta(1,1) = Uniform(0,1)

### Gamma(k, θ)
- Mean = kθ
- Variance = kθ²
- Special case: Gamma(1, λ⁻¹) = Exponential(λ)

### Exponential(λ)
- Mean = 1/λ
- Variance = 1/λ²
- All values > 0

### Chi-Square(k)
- Mean = k
- Variance = 2k
- Equals sum of k squared standard normal variables

### Student's t(ν)
- Mean = 0 for ν > 1
- Variance = ν/(ν-2) for ν > 2
- Approaches Normal(0,1) as ν increases

### Binomial(n, p)
- Mean = np
- Variance = np(1-p)
- Approaches Normal(np, np(1-p)) for large n

### Poisson(λ)
- Mean = λ
- Variance = λ
- All values are non-negative integers