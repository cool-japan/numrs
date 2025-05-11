# NumRS Random Distributions Reference

This document provides an overview of all the probability distributions available in the NumRS random module, along with their parameters and typical use cases.

## Continuous Distributions

### Uniform

```rust
// Modern interface
rng.random::<f64>(&[shape])?;            // Range [0, 1)
rng.uniform(low, high, &[shape])?;       // Range [low, high)

// Legacy interface
uniform(low, high, &[shape])?;
```

**Parameters:**
- `low`: Lower bound (inclusive)
- `high`: Upper bound (inclusive for uniform, exclusive for integers)
- `shape`: Shape of the output array

**Use cases:** When all values in a range are equally likely. Used for simulations, game mechanics, and initializing weights.

### Normal (Gaussian)

```rust
// Modern interface
rng.normal(mean, std, &[shape])?;
rng.standard_normal(&[shape])?;          // Mean 0, std 1

// Legacy interface
normal(mean, std, &[shape])?;
standard_normal(&[shape])?;
```

**Parameters:**
- `mean`: Center of the distribution
- `std`: Standard deviation
- `shape`: Shape of the output array

**Use cases:** The most commonly used distribution in statistics. Models natural phenomena, measurement errors, and many other random processes. Central Limit Theorem makes it widely applicable.

### Beta

```rust
// Modern interface
rng.beta(a, b, &[shape])?;

// Legacy interface
beta(a, b, &[shape])?;
```

**Parameters:**
- `a`: Alpha shape parameter (>0)
- `b`: Beta shape parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Modeling random probabilities, proportions, and percentages. Used in Bayesian statistics and project management (PERT).

### Gamma

```rust
// Modern interface
rng.gamma(shape_param, scale, &[shape])?;

// Legacy interface
gamma(shape_param, scale, &[shape])?;
```

**Parameters:**
- `shape_param`: Shape parameter (>0)
- `scale`: Scale parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Modeling waiting times, in particular when events occur randomly. Used in finance, reliability analysis, and queueing theory.

### Chi-Square

```rust
// Modern interface
rng.chisquare(df, &[shape])?;

// Legacy interface
chisquare(df, &[shape])?;
```

**Parameters:**
- `df`: Degrees of freedom (>0)
- `shape`: Shape of the output array

**Use cases:** Statistical hypothesis testing, confidence intervals, and quality control. Special case of gamma distribution.

### Exponential

```rust
// Modern interface
rng.exponential(scale, &[shape])?;

// Legacy interface
exponential(scale, &[shape])?;
```

**Parameters:**
- `scale`: Scale parameter (>0), inverse of rate
- `shape`: Shape of the output array

**Use cases:** Modeling time between independent events occurring at a constant rate, like radioactive decay or arrival times.

### Weibull

```rust
// Modern interface
rng.weibull(shape_param, scale, &[shape])?;

// Legacy interface
weibull(shape_param, scale, &[shape])?;
```

**Parameters:**
- `shape_param`: Shape parameter (>0)
- `scale`: Scale parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Reliability engineering and survival analysis. Models time-to-failure and material strength.

### Lognormal

```rust
// Modern interface
rng.lognormal(mean, sigma, &[shape])?;

// Legacy interface
lognormal(mean, sigma, &[shape])?;
```

**Parameters:**
- `mean`: Mean of the underlying normal distribution
- `sigma`: Standard deviation of the underlying normal distribution
- `shape`: Shape of the output array

**Use cases:** Modeling growth processes, income distribution, stock prices, and biological processes.

### Pareto

```rust
// Modern interface
rng.pareto(alpha, &[shape])?;

// Legacy interface
pareto(alpha, &[shape])?;
```

**Parameters:**
- `alpha`: Shape parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Modeling the 80/20 rule, distribution of wealth, city sizes, and many other "power law" phenomena.

### Cauchy (Lorentz)

```rust
// Modern interface
rng.cauchy(loc, scale, &[shape])?;

// Legacy interface
cauchy(loc, scale, &[shape])?;
```

**Parameters:**
- `loc`: Location parameter
- `scale`: Scale parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Modeling resonance behavior, spectral lines, and certain physical phenomena.

### Student's t

```rust
// Modern interface
rng.student_t(df, &[shape])?;

// Legacy interface
student_t(df, &[shape])?;
```

**Parameters:**
- `df`: Degrees of freedom (>0)
- `shape`: Shape of the output array

**Use cases:** Statistical hypothesis testing, especially when sample sizes are small.

### Laplace

```rust
// Legacy interface
laplace(loc, scale, &[shape])?;
```

**Parameters:**
- `loc`: Location parameter
- `scale`: Scale parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Modeling error distributions in machine learning and other areas where errors are double exponentially distributed.

### Gumbel

```rust
// Legacy interface
gumbel(loc, scale, &[shape])?;
```

**Parameters:**
- `loc`: Location parameter
- `scale`: Scale parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Extreme value analysis like flooding, rainfall, and maximum wind speeds.

### Logistic

```rust
// Legacy interface
logistic(loc, scale, &[shape])?;
```

**Parameters:**
- `loc`: Location parameter
- `scale`: Scale parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Growth models, neural networks (sigmoid activation), and logistic regression.

### Rayleigh

```rust
// Legacy interface
rayleigh(scale, &[shape])?;
```

**Parameters:**
- `scale`: Scale parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Modeling wind speeds, wave heights, and in communication theory.

### Triangular

```rust
// Legacy interface
triangular(low, mode, high, &[shape])?;
```

**Parameters:**
- `low`: Lower limit
- `mode`: Mode (peak of the distribution)
- `high`: Upper limit
- `shape`: Shape of the output array

**Use cases:** Project management, risk analysis when minimum, maximum, and most likely values are known.

### PERT

```rust
// Legacy interface
pert(min, mode, max, &[shape])?;
```

**Parameters:**
- `min`: Minimum value
- `mode`: Most likely value
- `max`: Maximum value
- `shape`: Shape of the output array

**Use cases:** Project management and risk analysis, similar to triangular but gives more weight to the mode.

## Discrete Distributions

### Binomial

```rust
// Modern interface
rng.binomial::<T>(n, p, &[shape])?;

// Legacy interface
binomial::<T>(n, p, &[shape])?;
```

**Parameters:**
- `n`: Number of trials
- `p`: Probability of success in each trial (between 0 and 1)
- `shape`: Shape of the output array

**Use cases:** Modeling number of successes in a fixed number of independent trials, like coin flips or yes/no surveys.

### Poisson

```rust
// Modern interface
rng.poisson::<T>(lambda, &[shape])?;

// Legacy interface
poisson::<T>(lambda, &[shape])?;
```

**Parameters:**
- `lambda`: Rate parameter (average number of events, >0)
- `shape`: Shape of the output array

**Use cases:** Modeling counts of events occurring in a fixed time interval, like number of calls per hour or website visits.

### Bernoulli

```rust
// Modern interface
rng.bernoulli(p, &[shape])?;

// Legacy interface
bernoulli(p, &[shape])?;
```

**Parameters:**
- `p`: Probability of success (between 0 and 1)
- `shape`: Shape of the output array

**Use cases:** Modeling binary outcomes like success/failure, yes/no, true/false.

### Negative Binomial

```rust
// Legacy interface
negative_binomial::<T>(n, p, &[shape])?;
```

**Parameters:**
- `n`: Number of successes
- `p`: Probability of success in each trial (between 0 and 1)
- `shape`: Shape of the output array

**Use cases:** Modeling the number of trials needed to achieve a specified number of successes.

### Geometric

```rust
// Legacy interface
geometric::<T>(p, &[shape])?;
```

**Parameters:**
- `p`: Probability of success (between 0 and 1)
- `shape`: Shape of the output array

**Use cases:** Modeling the number of trials until the first success, like number of coin flips until first heads.

### Hypergeometric

```rust
// Legacy interface
hypergeometric::<T>(ngood, nbad, nsample, &[shape])?;
```

**Parameters:**
- `ngood`: Number of success states in the population
- `nbad`: Number of failure states in the population
- `nsample`: Number of samples drawn
- `shape`: Shape of the output array

**Use cases:** Modeling sampling without replacement, like drawing cards from a deck.

### Zipf

```rust
// Legacy interface
zipf::<T>(a, &[shape])?;
```

**Parameters:**
- `a`: Distribution parameter (>1)
- `shape`: Shape of the output array

**Use cases:** Modeling frequency distributions like word frequencies in text or city population ranks.

### Logseries

```rust
// Legacy interface
logseries::<T>(p, &[shape])?;
```

**Parameters:**
- `p`: Distribution parameter (between 0 and 1)
- `shape`: Shape of the output array

**Use cases:** Modeling certain biological processes and frequency distributions.

## Multivariate Distributions

### Multivariate Normal

```rust
// Legacy interface
multivariate_normal(&mean, &cov, size)?;
```

**Parameters:**
- `mean`: Mean vector
- `cov`: Covariance matrix
- `size`: Optional shape of output array

**Use cases:** Modeling correlated random variables in finance, machine learning, and many scientific applications.

### Multinomial

```rust
// Legacy interface
multinomial::<T>(n, &pvals, size)?;
```

**Parameters:**
- `n`: Number of trials
- `pvals`: Probability vector (must sum to 1)
- `size`: Optional shape of output array

**Use cases:** Modeling outcomes of experiments with multiple possible results, like dice rolls.

### Dirichlet

```rust
// Legacy interface
dirichlet::<T>(&alpha, &[shape])?;
```

**Parameters:**
- `alpha`: Concentration parameters
- `shape`: Shape of the output array

**Use cases:** Modeling distributions over probability vectors, widely used in Bayesian statistics and topic modeling.

## Advanced Distributions

### Noncentral Chi-Square

```rust
// Legacy interface
noncentral_chisquare::<T>(df, nonc, &[shape])?;
```

**Parameters:**
- `df`: Degrees of freedom (>0)
- `nonc`: Non-centrality parameter (≥0)
- `shape`: Shape of the output array

**Use cases:** More general version of chi-square, used in certain statistical power calculations.

### Noncentral F

```rust
// Legacy interface
noncentral_f::<T>(dfnum, dfden, nonc, &[shape])?;
```

**Parameters:**
- `dfnum`: Numerator degrees of freedom (>0)
- `dfden`: Denominator degrees of freedom (>0)
- `nonc`: Non-centrality parameter (≥0)
- `shape`: Shape of the output array

**Use cases:** Statistical power calculations and certain ANOVA models.

### Von Mises

```rust
// Legacy interface
vonmises::<T>(mu, kappa, &[shape])?;
```

**Parameters:**
- `mu`: Mode (location)
- `kappa`: Concentration parameter (≥0)
- `shape`: Shape of the output array

**Use cases:** Modeling circular data like angles, wind directions, or phases.

### Maxwell

```rust
// Legacy interface
maxwell::<T>(scale, &[shape])?;
```

**Parameters:**
- `scale`: Scale parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Modeling speed distributions in statistical mechanics and molecular physics.

### Wald (Inverse Gaussian)

```rust
// Legacy interface
wald::<T>(mean, scale, &[shape])?;
```

**Parameters:**
- `mean`: Mean parameter (>0)
- `scale`: Shape parameter (>0)
- `shape`: Shape of the output array

**Use cases:** Modeling waiting times and financial returns.