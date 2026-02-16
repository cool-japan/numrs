//! Comprehensive tests for statistical distribution functions
//!
//! This test suite validates the PDF, CDF, and quantile functions for various
//! probability distributions against known values and mathematical properties.

use numrs2::stats::distributions::*;

const TOL: f64 = 1e-6;
const LOOSER_TOL: f64 = 1e-4;

// ============================================================================
// Beta Distribution Tests
// ============================================================================

#[test]
fn test_beta_pdf_basic() {
    // Beta(2, 3) PDF at x=0.5
    // Expected: B(2,3) = Γ(2)Γ(3)/Γ(5) = 1*2/24 = 1/12
    // PDF = x^(a-1) * (1-x)^(b-1) / B(a,b) = 0.5^1 * 0.5^2 / (1/12) = 0.125 * 12 = 1.5
    let pdf: f64 = beta_pdf(0.5, 2.0, 3.0).expect("beta_pdf should succeed");
    assert!((pdf - 1.5).abs() < TOL, "Beta(2,3) PDF at 0.5 = {}", pdf);
}

#[test]
fn test_beta_pdf_edge_cases() {
    // PDF at x=0 with a>1
    let pdf: f64 = beta_pdf(0.0, 2.0, 3.0).expect("beta_pdf should succeed");
    assert_eq!(pdf, 0.0, "Beta PDF at x=0 should be 0");

    // PDF at x=1 with b>1
    let pdf: f64 = beta_pdf(1.0, 2.0, 3.0).expect("beta_pdf should succeed");
    assert_eq!(pdf, 0.0, "Beta PDF at x=1 should be 0");

    // PDF outside [0,1]
    let pdf: f64 = beta_pdf(1.5, 2.0, 3.0).expect("beta_pdf should succeed");
    assert_eq!(pdf, 0.0, "Beta PDF outside [0,1] should be 0");
}

#[test]
fn test_beta_cdf_properties() {
    // CDF at 0 should be 0
    let cdf: f64 = beta_cdf(0.0, 2.0, 3.0).expect("beta_cdf should succeed");
    assert!((cdf - 0.0).abs() < TOL, "Beta CDF at 0 = {}", cdf);

    // CDF at 1 should be 1
    let cdf: f64 = beta_cdf(1.0, 2.0, 3.0).expect("beta_cdf should succeed");
    assert!((cdf - 1.0).abs() < TOL, "Beta CDF at 1 = {}", cdf);

    // CDF should be monotonically increasing
    let cdf1 = beta_cdf(0.3, 2.0, 3.0).expect("beta_cdf should succeed");
    let cdf2 = beta_cdf(0.5, 2.0, 3.0).expect("beta_cdf should succeed");
    let cdf3 = beta_cdf(0.7, 2.0, 3.0).expect("beta_cdf should succeed");
    assert!(cdf1 < cdf2 && cdf2 < cdf3, "Beta CDF should be increasing");
}

#[test]
fn test_beta_ppf_cdf_inverse() {
    // PPF should be inverse of CDF
    let x = 0.5;
    let cdf: f64 = beta_cdf(x, 2.0, 3.0).expect("beta_cdf should succeed");
    let ppf: f64 = beta_ppf(cdf, 2.0, 3.0).expect("beta_ppf should succeed");
    assert!(
        (ppf - x).abs() < LOOSER_TOL,
        "Beta PPF(CDF(x)) ≈ x, got {}",
        ppf
    );
}

#[test]
fn test_beta_logpdf_consistency() {
    // log PDF should equal log of PDF
    let x = 0.5;
    let a = 2.0;
    let b = 3.0;
    let pdf: f64 = beta_pdf(x, a, b).expect("beta_pdf should succeed");
    let logpdf = beta_logpdf(x, a, b).expect("beta_logpdf should succeed");
    assert!(
        (logpdf - pdf.ln()).abs() < TOL,
        "log PDF should match ln(PDF)"
    );
}

// ============================================================================
// Gamma Distribution Tests
// ============================================================================

#[test]
fn test_gamma_pdf_basic() {
    // Gamma(1, 1) is exponential distribution with rate 1
    // PDF at x=0 should be 1.0
    let pdf: f64 = gamma_pdf(0.0001, 1.0, 1.0).expect("gamma_pdf should succeed");
    assert!((pdf - 1.0).abs() < 0.01, "Gamma(1,1) PDF at x≈0 ≈ 1.0");
}

#[test]
fn test_gamma_pdf_edge_cases() {
    // PDF at x=0
    let pdf: f64 = gamma_pdf(0.0, 2.0, 1.0).expect("gamma_pdf should succeed");
    assert_eq!(pdf, 0.0, "Gamma PDF at x=0 should be 0");

    // PDF at negative x
    let pdf: f64 = gamma_pdf(-1.0, 2.0, 1.0).expect("gamma_pdf should succeed");
    assert_eq!(pdf, 0.0, "Gamma PDF at x<0 should be 0");
}

#[test]
fn test_gamma_cdf_properties() {
    // CDF at 0 should be 0
    let cdf: f64 = gamma_cdf(0.0, 2.0, 1.0).expect("gamma_cdf should succeed");
    assert!((cdf - 0.0).abs() < TOL, "Gamma CDF at 0 = {}", cdf);

    // CDF should be monotonically increasing
    let cdf1 = gamma_cdf(1.0, 2.0, 1.0).expect("gamma_cdf should succeed");
    let cdf2 = gamma_cdf(2.0, 2.0, 1.0).expect("gamma_cdf should succeed");
    let cdf3 = gamma_cdf(3.0, 2.0, 1.0).expect("gamma_cdf should succeed");
    assert!(cdf1 < cdf2 && cdf2 < cdf3, "Gamma CDF should be increasing");
}

#[test]
fn test_gamma_ppf_cdf_inverse() {
    // PPF should be inverse of CDF
    let x = 2.0;
    let cdf: f64 = gamma_cdf(x, 2.0, 1.0).expect("gamma_cdf should succeed");
    let ppf: f64 = gamma_ppf(cdf, 2.0, 1.0).expect("gamma_ppf should succeed");
    assert!(
        (ppf - x).abs() < LOOSER_TOL,
        "Gamma PPF(CDF(x)) ≈ x, got {}",
        ppf
    );
}

#[test]
fn test_gamma_logpdf_consistency() {
    // log PDF should equal log of PDF
    let x = 2.0;
    let shape = 2.0;
    let scale = 1.0;
    let pdf: f64 = gamma_pdf(x, shape, scale).expect("gamma_pdf should succeed");
    let logpdf = gamma_logpdf(x, shape, scale).expect("gamma_logpdf should succeed");
    assert!(
        (logpdf - pdf.ln()).abs() < TOL,
        "log PDF should match ln(PDF)"
    );
}

// ============================================================================
// Student's t-Distribution Tests
// ============================================================================

#[test]
fn test_student_t_pdf_basic() {
    // Student's t PDF at x=0 with any df should be symmetric
    let pdf: f64 = student_t_pdf(0.0, 10.0).expect("student_t_pdf should succeed");
    assert!(pdf > 0.0, "Student's t PDF at 0 should be positive");

    // PDF should be symmetric
    let pdf_pos: f64 = student_t_pdf(1.5, 10.0).expect("student_t_pdf should succeed");
    let pdf_neg: f64 = student_t_pdf(-1.5, 10.0).expect("student_t_pdf should succeed");
    assert!(
        (pdf_pos - pdf_neg).abs() < TOL,
        "Student's t PDF should be symmetric"
    );
}

#[test]
fn test_student_t_cdf_properties() {
    // CDF at 0 should be 0.5
    let cdf: f64 = student_t_cdf(0.0, 10.0).expect("student_t_cdf should succeed");
    assert!((cdf - 0.5).abs() < TOL, "Student's t CDF at 0 = {}", cdf);

    // CDF should be monotonically increasing
    let cdf1 = student_t_cdf(-1.0, 10.0).expect("student_t_cdf should succeed");
    let cdf2 = student_t_cdf(0.0, 10.0).expect("student_t_cdf should succeed");
    let cdf3 = student_t_cdf(1.0, 10.0).expect("student_t_cdf should succeed");
    assert!(
        cdf1 < cdf2 && cdf2 < cdf3,
        "Student's t CDF should be increasing"
    );
}

#[test]
fn test_student_t_ppf_cdf_inverse() {
    // PPF should be inverse of CDF
    let x = 1.5;
    let cdf: f64 = student_t_cdf(x, 10.0).expect("student_t_cdf should succeed");
    let ppf: f64 = student_t_ppf(cdf, 10.0).expect("student_t_ppf should succeed");
    assert!(
        (ppf - x).abs() < LOOSER_TOL,
        "Student's t PPF(CDF(x)) ≈ x, got {}",
        ppf
    );
}

#[test]
fn test_student_t_critical_values() {
    // Test known critical values for t-distribution
    // For df=10, two-tailed 95% critical value ≈ ±2.228
    let upper: f64 = student_t_ppf(0.975, 10.0).expect("student_t_ppf should succeed");
    assert!(
        (upper - 2.228).abs() < 0.01,
        "t(10) 97.5% quantile ≈ 2.228, got {}",
        upper
    );
}

// ============================================================================
// Cauchy Distribution Tests
// ============================================================================

#[test]
fn test_cauchy_pdf_basic() {
    // Standard Cauchy PDF at x=0 should be 1/π
    let pdf: f64 = cauchy_pdf(0.0, 0.0, 1.0).expect("cauchy_pdf should succeed");
    let expected = 1.0 / std::f64::consts::PI;
    assert!((pdf - expected).abs() < TOL, "Cauchy PDF at 0 = {}", pdf);

    // PDF should be symmetric
    let pdf_pos: f64 = cauchy_pdf(2.0, 0.0, 1.0).expect("cauchy_pdf should succeed");
    let pdf_neg: f64 = cauchy_pdf(-2.0, 0.0, 1.0).expect("cauchy_pdf should succeed");
    assert!(
        (pdf_pos - pdf_neg).abs() < TOL,
        "Cauchy PDF should be symmetric"
    );
}

#[test]
fn test_cauchy_cdf_properties() {
    // CDF at median (loc) should be 0.5
    let cdf: f64 = cauchy_cdf(0.0, 0.0, 1.0).expect("cauchy_cdf should succeed");
    assert!((cdf - 0.5).abs() < TOL, "Cauchy CDF at median = {}", cdf);

    // CDF should be monotonically increasing
    let cdf1 = cauchy_cdf(-2.0, 0.0, 1.0).expect("cauchy_cdf should succeed");
    let cdf2 = cauchy_cdf(0.0, 0.0, 1.0).expect("cauchy_cdf should succeed");
    let cdf3 = cauchy_cdf(2.0, 0.0, 1.0).expect("cauchy_cdf should succeed");
    assert!(
        cdf1 < cdf2 && cdf2 < cdf3,
        "Cauchy CDF should be increasing"
    );
}

#[test]
fn test_cauchy_ppf_cdf_inverse() {
    // PPF should be inverse of CDF
    let x = 1.5;
    let cdf: f64 = cauchy_cdf(x, 0.0, 1.0).expect("cauchy_cdf should succeed");
    let ppf: f64 = cauchy_ppf(cdf, 0.0, 1.0).expect("cauchy_ppf should succeed");
    assert!((ppf - x).abs() < TOL, "Cauchy PPF(CDF(x)) ≈ x, got {}", ppf);
}

// ============================================================================
// Laplace Distribution Tests
// ============================================================================

#[test]
fn test_laplace_pdf_basic() {
    // Standard Laplace PDF at x=0 (mean) should be 1/(2*scale) = 0.5
    let pdf: f64 = laplace_pdf(0.0, 0.0, 1.0).expect("laplace_pdf should succeed");
    assert!((pdf - 0.5).abs() < TOL, "Laplace PDF at mean = {}", pdf);

    // PDF should be symmetric
    let pdf_pos: f64 = laplace_pdf(1.5, 0.0, 1.0).expect("laplace_pdf should succeed");
    let pdf_neg: f64 = laplace_pdf(-1.5, 0.0, 1.0).expect("laplace_pdf should succeed");
    assert!(
        (pdf_pos - pdf_neg).abs() < TOL,
        "Laplace PDF should be symmetric"
    );
}

#[test]
fn test_laplace_cdf_properties() {
    // CDF at mean should be 0.5
    let cdf: f64 = laplace_cdf(0.0, 0.0, 1.0).expect("laplace_cdf should succeed");
    assert!((cdf - 0.5).abs() < TOL, "Laplace CDF at mean = {}", cdf);

    // CDF should be monotonically increasing
    let cdf1 = laplace_cdf(-2.0, 0.0, 1.0).expect("laplace_cdf should succeed");
    let cdf2 = laplace_cdf(0.0, 0.0, 1.0).expect("laplace_cdf should succeed");
    let cdf3 = laplace_cdf(2.0, 0.0, 1.0).expect("laplace_cdf should succeed");
    assert!(
        cdf1 < cdf2 && cdf2 < cdf3,
        "Laplace CDF should be increasing"
    );
}

#[test]
fn test_laplace_ppf_cdf_inverse() {
    // PPF should be inverse of CDF
    let x = 1.0;
    let cdf: f64 = laplace_cdf(x, 0.0, 1.0).expect("laplace_cdf should succeed");
    let ppf: f64 = laplace_ppf(cdf, 0.0, 1.0).expect("laplace_ppf should succeed");
    assert!(
        (ppf - x).abs() < TOL,
        "Laplace PPF(CDF(x)) ≈ x, got {}",
        ppf
    );
}

// ============================================================================
// Logistic Distribution Tests
// ============================================================================

#[test]
fn test_logistic_pdf_basic() {
    // Standard Logistic PDF at x=0 (mean) should be 1/4
    let pdf: f64 = logistic_pdf(0.0, 0.0, 1.0).expect("logistic_pdf should succeed");
    assert!(
        (pdf - 0.25_f64).abs() < TOL,
        "Logistic PDF at mean = {}",
        pdf
    );

    // PDF should be symmetric
    let pdf_pos: f64 = logistic_pdf(1.5, 0.0, 1.0).expect("logistic_pdf should succeed");
    let pdf_neg: f64 = logistic_pdf(-1.5, 0.0, 1.0).expect("logistic_pdf should succeed");
    assert!(
        (pdf_pos - pdf_neg).abs() < TOL,
        "Logistic PDF should be symmetric"
    );
}

#[test]
fn test_logistic_cdf_properties() {
    // CDF at mean should be 0.5
    let cdf: f64 = logistic_cdf(0.0, 0.0, 1.0).expect("logistic_cdf should succeed");
    assert!((cdf - 0.5).abs() < TOL, "Logistic CDF at mean = {}", cdf);

    // CDF should be monotonically increasing
    let cdf1 = logistic_cdf(-2.0, 0.0, 1.0).expect("logistic_cdf should succeed");
    let cdf2 = logistic_cdf(0.0, 0.0, 1.0).expect("logistic_cdf should succeed");
    let cdf3 = logistic_cdf(2.0, 0.0, 1.0).expect("logistic_cdf should succeed");
    assert!(
        cdf1 < cdf2 && cdf2 < cdf3,
        "Logistic CDF should be increasing"
    );
}

#[test]
fn test_logistic_ppf_cdf_inverse() {
    // PPF should be inverse of CDF
    let x = 1.0;
    let cdf: f64 = logistic_cdf(x, 0.0, 1.0).expect("logistic_cdf should succeed");
    let ppf: f64 = logistic_ppf(cdf, 0.0, 1.0).expect("logistic_ppf should succeed");
    assert!(
        (ppf - x).abs() < TOL,
        "Logistic PPF(CDF(x)) ≈ x, got {}",
        ppf
    );
}

// ============================================================================
// Pareto Distribution Tests
// ============================================================================

#[test]
fn test_pareto_pdf_basic() {
    // Pareto(2, 1) PDF at x=2
    // f(2; α=2, xₘ=1) = 2 * 1^2 / 2^3 = 2/8 = 0.25
    let pdf: f64 = pareto_pdf(2.0, 2.0, 1.0).expect("pareto_pdf should succeed");
    assert!((pdf - 0.25).abs() < TOL, "Pareto(2,1) PDF at x=2 = {}", pdf);

    // PDF below xm should be 0
    let pdf: f64 = pareto_pdf(0.5, 2.0, 1.0).expect("pareto_pdf should succeed");
    assert_eq!(pdf, 0.0, "Pareto PDF below xm should be 0");
}

#[test]
fn test_pareto_cdf_properties() {
    // CDF at xm should be 0
    let cdf: f64 = pareto_cdf(1.0, 2.0, 1.0).expect("pareto_cdf should succeed");
    assert!((cdf - 0.0).abs() < TOL, "Pareto CDF at xm = {}", cdf);

    // CDF at x=2 with α=2, xₘ=1
    // F(2; 2, 1) = 1 - (1/2)^2 = 0.75
    let cdf: f64 = pareto_cdf(2.0, 2.0, 1.0).expect("pareto_cdf should succeed");
    assert!((cdf - 0.75).abs() < TOL, "Pareto(2,1) CDF at x=2 = {}", cdf);

    // CDF should be monotonically increasing
    let cdf1 = pareto_cdf(1.5, 2.0, 1.0).expect("pareto_cdf should succeed");
    let cdf2 = pareto_cdf(2.0, 2.0, 1.0).expect("pareto_cdf should succeed");
    let cdf3 = pareto_cdf(3.0, 2.0, 1.0).expect("pareto_cdf should succeed");
    assert!(
        cdf1 < cdf2 && cdf2 < cdf3,
        "Pareto CDF should be increasing"
    );
}

#[test]
fn test_pareto_ppf_cdf_inverse() {
    // PPF should be inverse of CDF
    let x = 2.0;
    let cdf: f64 = pareto_cdf(x, 2.0, 1.0).expect("pareto_cdf should succeed");
    let ppf: f64 = pareto_ppf(cdf, 2.0, 1.0).expect("pareto_ppf should succeed");
    assert!((ppf - x).abs() < TOL, "Pareto PPF(CDF(x)) ≈ x, got {}", ppf);
}

// ============================================================================
// Extreme Value Tests
// ============================================================================

#[test]
fn test_numerical_stability() {
    // Test with extreme parameter values for numerical stability

    // Beta with very small parameters
    let pdf: f64 = beta_pdf(0.5, 0.1, 0.1).expect("beta_pdf should succeed");
    assert!(
        pdf.is_finite(),
        "Beta PDF should be finite for small parameters"
    );

    // Gamma with large shape parameter
    let pdf: f64 = gamma_pdf(10.0, 100.0, 1.0).expect("gamma_pdf should succeed");
    assert!(
        pdf.is_finite(),
        "Gamma PDF should be finite for large shape"
    );

    // Student's t with very small df
    let pdf: f64 = student_t_pdf(0.0, 1.0).expect("student_t_pdf should succeed");
    assert!(pdf.is_finite(), "Student's t PDF should be finite for df=1");

    // Student's t with very large df (should approach normal)
    let pdf: f64 = student_t_pdf(0.0, 1000.0).expect("student_t_pdf should succeed");
    assert!(
        pdf.is_finite(),
        "Student's t PDF should be finite for large df"
    );
}

#[test]
fn test_error_handling() {
    // Test invalid parameters

    // Beta with non-positive parameters
    assert!(
        beta_pdf(0.5, -1.0, 3.0).is_err(),
        "Beta PDF should reject negative alpha"
    );
    assert!(
        beta_pdf(0.5, 2.0, 0.0).is_err(),
        "Beta PDF should reject zero beta"
    );

    // Gamma with non-positive parameters
    assert!(
        gamma_pdf(1.0, 0.0, 1.0).is_err(),
        "Gamma PDF should reject zero shape"
    );
    assert!(
        gamma_pdf(1.0, 2.0, -1.0).is_err(),
        "Gamma PDF should reject negative scale"
    );

    // Student's t with non-positive df
    assert!(
        student_t_pdf(0.0, 0.0).is_err(),
        "Student's t PDF should reject zero df"
    );
    assert!(
        student_t_pdf(0.0, -5.0).is_err(),
        "Student's t PDF should reject negative df"
    );

    // Cauchy with non-positive scale
    assert!(
        cauchy_pdf(0.0, 0.0, 0.0).is_err(),
        "Cauchy PDF should reject zero scale"
    );

    // Laplace with non-positive scale
    assert!(
        laplace_pdf(0.0, 0.0, -1.0).is_err(),
        "Laplace PDF should reject negative scale"
    );

    // Logistic with non-positive scale
    assert!(
        logistic_pdf(0.0, 0.0, 0.0).is_err(),
        "Logistic PDF should reject zero scale"
    );

    // Pareto with non-positive parameters
    assert!(
        pareto_pdf(2.0, 0.0, 1.0).is_err(),
        "Pareto PDF should reject zero alpha"
    );
    assert!(
        pareto_pdf(2.0, 2.0, -1.0).is_err(),
        "Pareto PDF should reject negative xm"
    );

    // Probability out of range
    assert!(beta_ppf(1.5, 2.0, 3.0).is_err(), "PPF should reject p > 1");
    assert!(
        gamma_ppf(-0.1, 2.0, 1.0).is_err(),
        "PPF should reject p < 0"
    );
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_pdf_cdf_consistency() {
    // Verify that integrating PDF gives CDF (numerically)
    // This is a basic sanity check using trapezoidal rule

    // Beta distribution
    let a = 2.0;
    let b = 3.0;
    let x = 0.6;
    let n = 1000;
    let dx = x / n as f64;
    let mut integral = 0.0;
    for i in 1..n {
        let xi = i as f64 * dx;
        let pdf_val = beta_pdf(xi, a, b).expect("beta_pdf should succeed");
        integral += pdf_val * dx;
    }
    let cdf: f64 = beta_cdf(x, a, b).expect("beta_cdf should succeed");
    assert!((integral - cdf).abs() < 0.01, "Integral of Beta PDF ≈ CDF");
}

#[test]
fn test_distribution_relationships() {
    // Test known relationships between distributions

    // Chi-square(2) is Gamma(1, 2)
    // (Not directly tested here, but worth noting)

    // Cauchy(0,1) is Student's t with df=1
    // Compare PDFs at multiple points
    for x in &[0.0, 0.5, 1.0, 2.0] {
        let cauchy: f64 = cauchy_pdf(*x, 0.0, 1.0).expect("cauchy_pdf should succeed");
        let t1: f64 = student_t_pdf(*x, 1.0).expect("student_t_pdf should succeed");
        let diff = (cauchy - t1).abs();
        assert!(
            diff < TOL,
            "Cauchy(0,1) ≈ t(1) at x={}: cauchy={}, t1={}, diff={}, TOL={}",
            x,
            cauchy,
            t1,
            diff,
            TOL
        );
    }
}
