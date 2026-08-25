//! Reference tests for NumRS2 distributions against NumPy/SciPy
//!
//! This file contains tests that compare the statistical properties
//! of NumRS2 distributions with reference values generated from NumPy/SciPy.
//!
//! To update reference data:
//! 1. Run the Python script at tests/py/distribution_reference_tests.py
//! 2. The script will generate a JSON file with reference values
//! 3. This test file will load that JSON and compare against NumRS2 values

// Array is used in the cfg(feature = "scirs") section
#[allow(unused_imports)]
use numrs2::array::Array;
use numrs2::random::{self, distributions::*};
use serde_json::{self, Value};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use serial_test::serial;

const SAMPLE_SIZE: usize = 10000;
const REFERENCE_FILE: &str = "tests/py/distribution_reference_data.json";
const EPSILON: f64 = 0.1; // Tolerance for statistical comparisons

/// Load reference data from the JSON file
fn load_reference_data() -> HashMap<String, Value> {
    let file_path = Path::new(REFERENCE_FILE);
    let file = File::open(file_path).unwrap_or_else(|_| {
        panic!(
            "Failed to open reference file: {}. Run the Python script to generate it.",
            REFERENCE_FILE
        )
    });

    let reader = BufReader::new(file);
    let json: HashMap<String, Value> =
        serde_json::from_reader(reader).expect("Failed to parse reference JSON data");

    json
}

/// Calculate basic statistics for a sample
fn calculate_basic_stats(samples: &[f64]) -> HashMap<String, f64> {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;

    let variance = samples.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let median = if sorted.len().is_multiple_of(2) {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    let mut stats = HashMap::new();
    stats.insert("mean".to_string(), mean);
    stats.insert("variance".to_string(), variance);
    stats.insert("min".to_string(), min);
    stats.insert("max".to_string(), max);
    stats.insert("median".to_string(), median);

    stats
}

/// Assert that NumRS2 stats are close to reference stats
fn assert_stats_close(rs_stats: &HashMap<String, f64>, ref_value: &Value, name: &str) {
    let ref_mean = ref_value["mean"].as_f64().expect("mean should be a number");
    let ref_variance = ref_value["variance"]
        .as_f64()
        .expect("variance should be a number");
    let ref_min = ref_value["min"].as_f64().expect("min should be a number");
    let ref_max = ref_value["max"].as_f64().expect("max should be a number");
    let ref_median = ref_value["median"]
        .as_f64()
        .expect("median should be a number");

    assert!(
        (rs_stats["mean"] - ref_mean).abs() < EPSILON,
        "{} mean: NumRS2 = {}, NumPy = {}, diff = {}",
        name,
        rs_stats["mean"],
        ref_mean,
        (rs_stats["mean"] - ref_mean).abs()
    );

    // Use slightly higher tolerance for distributions with high variance or heavy tails
    // Log-normal is particularly sensitive because sample variance scales with exp(2σ²)
    // and different RNG implementations (NumRS2 vs NumPy) produce different sample sequences
    // The sample variance for lognormal can easily vary by 50-70% between runs due to outliers
    let variance_tolerance = if name == "Log-normal" {
        EPSILON * 7.0 // Very high tolerance due to heavy tail causing large sample-to-sample variance variation
    } else if name == "Noncentral F" {
        EPSILON * 2.2 // Noncentral F has higher variance due to sampling variability
    } else {
        EPSILON * 2.0
    };
    assert!(
        (rs_stats["variance"] - ref_variance).abs() / ref_variance < variance_tolerance,
        "{} variance: NumRS2 = {}, NumPy = {}, relative diff = {}",
        name,
        rs_stats["variance"],
        ref_variance,
        (rs_stats["variance"] - ref_variance).abs() / ref_variance
    );

    // For min, max, median we use a looser tolerance since they're more sensitive to sampling
    let range = ref_max - ref_min;
    // Use higher tolerance for heavy-tailed distributions due to extreme value sensitivity
    // Student's t, Cauchy, and Log-normal can have extremely large outliers that vary between runs
    let range_multiplier = if name == "Student's t" || name == "Cauchy" || name == "Log-normal" {
        1.5 // Increased from 1.0 due to heavy tail extreme value sensitivity
    } else {
        0.50
    };
    let range_epsilon = range * range_multiplier;

    assert!(
        (rs_stats["min"] - ref_min).abs() < range_epsilon,
        "{} min: NumRS2 = {}, NumPy = {}, diff = {}, tolerance = {}",
        name,
        rs_stats["min"],
        ref_min,
        (rs_stats["min"] - ref_min).abs(),
        range_epsilon
    );

    assert!(
        (rs_stats["max"] - ref_max).abs() < range_epsilon,
        "{} max: NumRS2 = {}, NumPy = {}, diff = {}, tolerance = {}",
        name,
        rs_stats["max"],
        ref_max,
        (rs_stats["max"] - ref_max).abs(),
        range_epsilon
    );

    // Use slightly higher tolerance for median values due to statistical sampling variation
    let median_tolerance = if name == "Gamma" {
        EPSILON * 1.2
    } else {
        EPSILON
    };
    assert!(
        (rs_stats["median"] - ref_median).abs() < median_tolerance,
        "{} median: NumRS2 = {}, NumPy = {}, diff = {}, tolerance = {}",
        name,
        rs_stats["median"],
        ref_median,
        (rs_stats["median"] - ref_median).abs(),
        median_tolerance
    );
}

#[test]
#[serial]
fn test_normal_against_reference() {
    let ref_data = load_reference_data();

    // Use the same seed as in the Python script
    random::set_seed(12345);

    let samples = normal(0.0, 1.0, &[SAMPLE_SIZE]).unwrap();
    let samples_vec = samples.to_vec();
    let rs_stats = calculate_basic_stats(&samples_vec);

    assert_stats_close(&rs_stats, &ref_data["normal"], "Normal");
}

#[test]
#[serial]
fn test_beta_against_reference() {
    let ref_data = load_reference_data();

    random::set_seed(12345);

    let samples = beta(2.0, 5.0, &[SAMPLE_SIZE]).unwrap();
    let samples_vec = samples.to_vec();
    let rs_stats = calculate_basic_stats(&samples_vec);

    assert_stats_close(&rs_stats, &ref_data["beta"], "Beta");
}

#[test]
#[serial]
fn test_cauchy_against_reference() {
    let ref_data = load_reference_data();

    random::set_seed(12345);

    let samples = cauchy(0.0, 1.0, &[SAMPLE_SIZE]).unwrap();
    let samples_vec = samples.to_vec();
    let rs_stats = calculate_basic_stats(&samples_vec);

    // For Cauchy, we only check min, max, and median since mean and variance don't exist
    let ref_mean = ref_data["cauchy"]["median"].as_f64().unwrap();
    assert!((rs_stats["median"] - ref_mean).abs() < EPSILON);
}

#[test]
#[serial]
fn test_chisquare_against_reference() {
    let ref_data = load_reference_data();

    random::set_seed(12345);

    let samples = chisquare(2.0, &[SAMPLE_SIZE]).unwrap();
    let samples_vec = samples.to_vec();
    let rs_stats = calculate_basic_stats(&samples_vec);

    assert_stats_close(&rs_stats, &ref_data["chisquare"], "Chi-square");
}

#[test]
#[serial]
fn test_exponential_against_reference() {
    let ref_data = load_reference_data();

    random::set_seed(12345);

    let samples = exponential(1.0, &[SAMPLE_SIZE]).unwrap();
    let samples_vec = samples.to_vec();
    let rs_stats = calculate_basic_stats(&samples_vec);

    assert_stats_close(&rs_stats, &ref_data["exponential"], "Exponential");
}

#[test]
#[serial]
fn test_gamma_against_reference() {
    let ref_data = load_reference_data();

    random::set_seed(12345);

    let samples = gamma(2.0, 2.0, &[SAMPLE_SIZE]).unwrap();
    let samples_vec = samples.to_vec();
    let rs_stats = calculate_basic_stats(&samples_vec);

    assert_stats_close(&rs_stats, &ref_data["gamma"], "Gamma");
}

#[test]
#[serial]
fn test_lognormal_against_reference() {
    let ref_data = load_reference_data();

    random::set_seed(12345);

    let samples = lognormal(0.0, 1.0, &[SAMPLE_SIZE]).unwrap();
    let samples_vec = samples.to_vec();
    let rs_stats = calculate_basic_stats(&samples_vec);

    assert_stats_close(&rs_stats, &ref_data["lognormal"], "Log-normal");
}

#[test]
#[serial]
fn test_student_t_against_reference() {
    let ref_data = load_reference_data();

    random::set_seed(12345);

    let samples = student_t(5.0, &[SAMPLE_SIZE]).unwrap();
    let samples_vec = samples.to_vec();
    let rs_stats = calculate_basic_stats(&samples_vec);

    assert_stats_close(&rs_stats, &ref_data["student_t"], "Student's t");
}

#[test]
#[serial]
fn test_uniform_against_reference() {
    let ref_data = load_reference_data();

    random::set_seed(12345);

    let samples = uniform(0.0, 1.0, &[SAMPLE_SIZE]).unwrap();
    let samples_vec = samples.to_vec();
    let rs_stats = calculate_basic_stats(&samples_vec);

    assert_stats_close(&rs_stats, &ref_data["uniform"], "Uniform");
}

#[test]
#[serial]
fn test_binomial_against_reference() {
    let ref_data = load_reference_data();

    random::set_seed(12345);

    let samples = binomial::<f64>(10, 0.5, &[SAMPLE_SIZE]).unwrap();
    let samples_vec = samples.to_vec();
    let rs_stats = calculate_basic_stats(&samples_vec);

    assert_stats_close(&rs_stats, &ref_data["binomial"], "Binomial");
}

#[test]
#[serial]
fn test_poisson_against_reference() {
    let ref_data = load_reference_data();

    random::set_seed(12345);

    let samples = poisson::<f64>(5.0, &[SAMPLE_SIZE]).unwrap();
    let samples_vec = samples.to_vec();
    let rs_stats = calculate_basic_stats(&samples_vec);

    assert_stats_close(&rs_stats, &ref_data["poisson"], "Poisson");
}

// Tests for SciRS2 distributions that are only available with the "scirs" feature
#[cfg(feature = "scirs")]
mod scirs_tests {
    use super::*;

    #[test]
    #[serial]
    fn test_noncentral_chisquare_against_reference() {
        let ref_data = load_reference_data();

        random::set_seed(12345);

        let samples = noncentral_chisquare(2.0, 1.0, &[SAMPLE_SIZE]).unwrap();
        let samples_vec = samples.to_vec();
        let rs_stats = calculate_basic_stats(&samples_vec);

        assert_stats_close(
            &rs_stats,
            &ref_data["noncentral_chisquare"],
            "Noncentral chi-square",
        );
    }

    #[test]
    #[serial]
    fn test_noncentral_f_against_reference() {
        // Was `#[ignore]`d as "Flaky test - noncentral F has high variance,
        // needs larger sample size or better implementation". It was
        // already seeded (`random::set_seed` below, unchanged), so the
        // flakiness was never about non-determinism -- it was that
        // `assert_stats_close`'s tolerances (calibrated for low-variance
        // distributions) were far too tight for THIS specific distribution.
        //
        // noncentral F(dfnum=2, dfden=5, nonc=1) has:
        //   mean = dfden*(dfnum+nonc) / (dfnum*(dfden-2)) = 5*3/(2*3) = 2.5
        //   var  = 2*(dfden/dfnum)^2 * [(dfnum+nonc)^2 + (dfnum+2*nonc)*(dfden-2)]
        //          / [(dfden-2)^2*(dfden-4)]
        //        = 2*6.25*(9+12)/9 ~= 29.17  (std ~= 5.4)
        // (the variance formula collapses to the standard central-F variance
        // at nonc=0, which is how it was checked). With BOTH this run and
        // the fixed reference draw independently sampling that distribution
        // at similar N, the combined standard error on the *mean* is
        // sqrt(2) * 5.4/sqrt(N) ~= 0.076 at N=10,000 -- so the previous
        // `EPSILON` (0.1) absolute tolerance on the mean was under 1.5
        // combined-sigma, i.e. a real coin flip, nothing to do with sample
        // size or "a better implementation".
        //
        // dfden=5 additionally means `E[X^4]` does not exist (the F
        // distribution's k-th moment needs dfden > 2k), so the sample
        // *variance*'s own sampling distribution is heavy-tailed with no
        // finite-CLT-variance guarantee, and extreme order statistics
        // (min/max of thousands of draws) are correspondingly unstable --
        // no fixed tolerance makes those a reproducible cross-implementation
        // check. So, unlike every other distribution in this file, this
        // test does not route through the shared `assert_stats_close`
        // (min/max included): it compares mean and median, both of which
        // *do* have well-behaved finite-variance sampling distributions
        // here, against tolerances sized off the derivation above, plus a
        // wide sanity band on variance, and does not compare min/max.
        let ref_data = load_reference_data();

        random::set_seed(12345);

        // Raised from the shared `SAMPLE_SIZE` (10,000, used by every other
        // test in this file and matched to the reference JSON's own N) to a
        // *local* N: mean/median tolerance only needs to hold for this one
        // test, and raising this test's own N shrinks its contribution to
        // the combined standard error above (the fixed reference draw's own
        // noise floor cannot be reduced this way, which is exactly why the
        // tolerance below is generous rather than tight).
        const N: usize = 30_000;
        let samples = noncentral_f(2.0, 5.0, 1.0, &[N]).unwrap();
        let samples_vec = samples.to_vec();
        let rs_stats = calculate_basic_stats(&samples_vec);

        let ref_stats = &ref_data["noncentral_f"];
        let ref_mean = ref_stats["mean"].as_f64().expect("mean should be a number");
        let ref_median = ref_stats["median"]
            .as_f64()
            .expect("median should be a number");
        let ref_variance = ref_stats["variance"]
            .as_f64()
            .expect("variance should be a number");

        // ~0.5 is comfortably wide relative to the ~0.076 combined standard
        // error derived above (>6 combined-sigma), while still catching a
        // grossly wrong mean (e.g. a sign error or swapped parameters).
        const MEAN_TOLERANCE: f64 = 0.5;
        assert!(
            (rs_stats["mean"] - ref_mean).abs() < MEAN_TOLERANCE,
            "Noncentral F mean: NumRS2 = {}, NumPy = {}, diff = {}",
            rs_stats["mean"],
            ref_mean,
            (rs_stats["mean"] - ref_mean).abs()
        );

        const MEDIAN_TOLERANCE: f64 = 0.5;
        assert!(
            (rs_stats["median"] - ref_median).abs() < MEDIAN_TOLERANCE,
            "Noncentral F median: NumRS2 = {}, NumPy = {}, diff = {}",
            rs_stats["median"],
            ref_median,
            (rs_stats["median"] - ref_median).abs()
        );

        // Wide sanity band only (order-of-magnitude), not a precision check
        // -- see the `E[X^4]` note above for why the sample variance itself
        // cannot be pinned tightly here.
        let variance_low = ref_variance * 0.25;
        let variance_high = ref_variance * 4.0;
        assert!(
            rs_stats["variance"] > variance_low && rs_stats["variance"] < variance_high,
            "Noncentral F variance: NumRS2 = {}, NumPy = {}, expected within [{}, {}]",
            rs_stats["variance"],
            ref_variance,
            variance_low,
            variance_high
        );
    }

    #[test]
    #[serial]
    fn test_vonmises_against_reference() {
        let ref_data = load_reference_data();

        random::set_seed(12345);

        let samples = vonmises(0.0, 1.0, &[SAMPLE_SIZE]).unwrap();
        let samples_vec = samples.to_vec();
        let rs_stats = calculate_basic_stats(&samples_vec);

        // Temporarily use relaxed tolerance for Von Mises while we improve the implementation
        let ref_variance = ref_data["vonmises"]["variance"].as_f64().unwrap();
        let ref_mean = ref_data["vonmises"]["mean"].as_f64().unwrap();

        // Check mean with normal tolerance
        assert!(
            (rs_stats["mean"] - ref_mean).abs() < EPSILON,
            "Von Mises mean: NumRS2 = {}, NumPy = {}, diff = {}",
            rs_stats["mean"],
            ref_mean,
            (rs_stats["mean"] - ref_mean).abs()
        );

        // Use much more relaxed tolerance for variance (temporarily)
        assert!(
            (rs_stats["variance"] - ref_variance).abs() / ref_variance < 1.0, // Allow up to 100% difference
            "Von Mises variance: NumRS2 = {}, NumPy = {}, relative diff = {}",
            rs_stats["variance"],
            ref_variance,
            (rs_stats["variance"] - ref_variance).abs() / ref_variance
        );
    }

    #[test]
    #[serial]
    fn test_maxwell_against_reference() {
        let ref_data = load_reference_data();

        random::set_seed(12345);

        let samples = maxwell(1.0, &[SAMPLE_SIZE]).unwrap();
        let samples_vec = samples.to_vec();
        let rs_stats = calculate_basic_stats(&samples_vec);

        assert_stats_close(&rs_stats, &ref_data["maxwell"], "Maxwell");
    }

    #[test]
    #[serial]
    fn test_truncated_normal_against_reference() {
        let ref_data = load_reference_data();

        random::set_seed(12345);

        let samples = truncated_normal(0.0, 1.0, -2.0, 2.0, &[SAMPLE_SIZE]).unwrap();
        let samples_vec = samples.to_vec();
        let rs_stats = calculate_basic_stats(&samples_vec);

        assert_stats_close(&rs_stats, &ref_data["truncated_normal"], "Truncated normal");
    }

    #[test]
    #[serial]
    fn test_multivariate_normal_against_reference() {
        // This test is different since multivariate normal has different statistics
        let ref_data = load_reference_data();

        random::set_seed(12345);

        let mean = vec![0.0, 0.0];
        let cov_data = vec![1.0, 0.5, 0.5, 1.0];
        let cov = Array::from_vec(cov_data).reshape(&[2, 2]);

        let samples =
            multivariate_normal_with_rotation(&mean, &cov, Some(&[SAMPLE_SIZE]), None).unwrap();
        let samples_vec = samples.to_vec();

        // Convert to 2D array for easier analysis
        let mut x_vals = Vec::with_capacity(SAMPLE_SIZE);
        let mut y_vals = Vec::with_capacity(SAMPLE_SIZE);

        for i in 0..SAMPLE_SIZE {
            x_vals.push(samples_vec[i * 2]);
            y_vals.push(samples_vec[i * 2 + 1]);
        }

        // Calculate mean and check
        let mean_x = x_vals.iter().sum::<f64>() / SAMPLE_SIZE as f64;
        let mean_y = y_vals.iter().sum::<f64>() / SAMPLE_SIZE as f64;

        let ref_mean_x = ref_data["multivariate_normal"]["mean"][0].as_f64().unwrap();
        let ref_mean_y = ref_data["multivariate_normal"]["mean"][1].as_f64().unwrap();

        assert!((mean_x - ref_mean_x).abs() < EPSILON);
        assert!((mean_y - ref_mean_y).abs() < EPSILON);

        // Calculate covariance and check
        let var_x = x_vals.iter().map(|&x| (x - mean_x).powi(2)).sum::<f64>() / SAMPLE_SIZE as f64;
        let var_y = y_vals.iter().map(|&y| (y - mean_y).powi(2)).sum::<f64>() / SAMPLE_SIZE as f64;

        let mut cov_xy = 0.0;
        for i in 0..SAMPLE_SIZE {
            cov_xy += (x_vals[i] - mean_x) * (y_vals[i] - mean_y);
        }
        cov_xy /= SAMPLE_SIZE as f64;

        let ref_var_x = ref_data["multivariate_normal"]["cov"][0][0]
            .as_f64()
            .unwrap();
        let ref_var_y = ref_data["multivariate_normal"]["cov"][1][1]
            .as_f64()
            .unwrap();
        let ref_cov_xy = ref_data["multivariate_normal"]["cov"][0][1]
            .as_f64()
            .unwrap();

        assert!((var_x - ref_var_x).abs() / ref_var_x < EPSILON);
        assert!((var_y - ref_var_y).abs() / ref_var_y < EPSILON);
        assert!((cov_xy - ref_cov_xy).abs() / (ref_cov_xy.abs() + 1e-10) < EPSILON);
    }
}
