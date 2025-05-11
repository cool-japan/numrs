#!/usr/bin/env python3
"""
NumPy Reference Tests for NumRS2 Distributions

This script generates reference values from NumPy/SciPy distributions
to compare against NumRS2 implementations. The output is used by Rust
reference tests to validate distribution implementations.

The test strategy:
1. Generate samples from various distributions using fixed seeds
2. Calculate statistical properties (mean, variance, min, max, etc.)
3. Save these reference values to a JSON file for Rust tests to use
"""

import numpy as np
import scipy.stats as stats
from scipy import special
import json
import os

# Set a fixed seed for reproducibility
SEED = 12345

# Directory to save reference data
REFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_FILE = os.path.join(REFERENCE_DIR, "distribution_reference_data.json")

# Sample size for generating distribution data
SAMPLE_SIZE = 10000

# Initialize the RNG with fixed seed
np.random.seed(SEED)

# Dictionary to store reference data
reference_data = {}

def generate_basic_stats(samples):
    """Calculate basic statistics for a sample"""
    return {
        "mean": float(np.mean(samples)),
        "variance": float(np.var(samples)),
        "min": float(np.min(samples)),
        "max": float(np.max(samples)),
        "median": float(np.median(samples)),
        "first_10": samples[:10].tolist()  # First 10 samples for exact comparison
    }

# Test standard continuous distributions available in both NumRS2 and NumPy
print("Generating reference data for standard continuous distributions...")

# Normal distribution
samples = np.random.normal(0, 1, SAMPLE_SIZE)
reference_data["normal"] = generate_basic_stats(samples)

# Beta distribution
samples = np.random.beta(2, 5, SAMPLE_SIZE)
reference_data["beta"] = generate_basic_stats(samples)

# Cauchy distribution
samples = stats.cauchy.rvs(loc=0, scale=1, size=SAMPLE_SIZE)
reference_data["cauchy"] = generate_basic_stats(samples)

# Chi-square distribution
samples = np.random.chisquare(2, SAMPLE_SIZE)
reference_data["chisquare"] = generate_basic_stats(samples)

# Exponential distribution
samples = np.random.exponential(1, SAMPLE_SIZE)
reference_data["exponential"] = generate_basic_stats(samples)

# Gamma distribution
samples = np.random.gamma(2, 2, SAMPLE_SIZE)
reference_data["gamma"] = generate_basic_stats(samples)

# Gumbel distribution
samples = np.random.gumbel(0, 1, SAMPLE_SIZE)
reference_data["gumbel"] = generate_basic_stats(samples)

# Laplace distribution
samples = np.random.laplace(0, 1, SAMPLE_SIZE)
reference_data["laplace"] = generate_basic_stats(samples)

# Logistic distribution
samples = stats.logistic.rvs(loc=0, scale=1, size=SAMPLE_SIZE)
reference_data["logistic"] = generate_basic_stats(samples)

# Log-normal distribution
samples = np.random.lognormal(0, 1, SAMPLE_SIZE)
reference_data["lognormal"] = generate_basic_stats(samples)

# Pareto distribution
samples = np.random.pareto(2, SAMPLE_SIZE) + 1  # Add 1 because NumPy's Pareto adds 1
reference_data["pareto"] = generate_basic_stats(samples)

# Student's t distribution
samples = np.random.standard_t(5, SAMPLE_SIZE)
reference_data["student_t"] = generate_basic_stats(samples)

# Uniform distribution
samples = np.random.uniform(0, 1, SAMPLE_SIZE)
reference_data["uniform"] = generate_basic_stats(samples)

# Weibull distribution
samples = np.random.weibull(1, SAMPLE_SIZE)
reference_data["weibull"] = generate_basic_stats(samples)

# Test discrete distributions
print("Generating reference data for discrete distributions...")

# Binomial distribution
samples = np.random.binomial(10, 0.5, SAMPLE_SIZE)
reference_data["binomial"] = generate_basic_stats(samples)

# Poisson distribution
samples = np.random.poisson(5, SAMPLE_SIZE)
reference_data["poisson"] = generate_basic_stats(samples)

# Bernoulli distribution (special case of binomial with n=1)
samples = np.random.binomial(1, 0.5, SAMPLE_SIZE)
reference_data["bernoulli"] = generate_basic_stats(samples)

# Geometric distribution
samples = np.random.geometric(0.5, SAMPLE_SIZE)
reference_data["geometric"] = generate_basic_stats(samples)

# Test advanced distributions from SciPy that are in SciRS2
print("Generating reference data for advanced SciPy distributions...")

# Noncentral chi-square distribution
samples = stats.ncx2.rvs(df=2, nc=1, size=SAMPLE_SIZE)
reference_data["noncentral_chisquare"] = generate_basic_stats(samples)

# Noncentral F distribution
samples = stats.ncf.rvs(dfn=2, dfd=5, nc=1, size=SAMPLE_SIZE)
reference_data["noncentral_f"] = generate_basic_stats(samples)

# Von Mises distribution
samples = stats.vonmises.rvs(kappa=1, loc=0, size=SAMPLE_SIZE)
reference_data["vonmises"] = generate_basic_stats(samples)

# Maxwell distribution
samples = stats.maxwell.rvs(scale=1, size=SAMPLE_SIZE)
reference_data["maxwell"] = generate_basic_stats(samples)

# Truncated normal distribution (using truncnorm)
a, b = -2, 2  # truncation range in standardized scale
samples = stats.truncnorm.rvs(a, b, loc=0, scale=1, size=SAMPLE_SIZE)
reference_data["truncated_normal"] = generate_basic_stats(samples)

# Multivariate normal
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
samples = np.random.multivariate_normal(mean, cov, SAMPLE_SIZE)
mvn_stats = {
    "mean": np.mean(samples, axis=0).tolist(),
    "cov": np.cov(samples, rowvar=False).tolist(),
    "first_10": samples[:10].tolist()
}
reference_data["multivariate_normal"] = mvn_stats

# Save reference data to JSON file
with open(REFERENCE_FILE, 'w') as f:
    json.dump(reference_data, f, indent=2)

print(f"Reference data saved to {REFERENCE_FILE}")
print("This data can be used by Rust tests to validate distribution implementations.")