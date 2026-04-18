//! Quantum Gate Unitarity Verification
//!
//! This module provides functions to verify the unitarity of quantum gate matrices.
//! A matrix U is unitary if U * U† = I, where U† is the conjugate transpose of U.
//!
//! # Overview
//!
//! Unitarity is a fundamental property of quantum gates. Any valid quantum gate
//! must be represented by a unitary matrix, meaning:
//!
//! - U†U = I (preserves inner products)
//! - ||U|ψ⟩|| = |||ψ⟩|| for all states |ψ⟩ (preserves norms)
//! - All eigenvalues have absolute value 1
//!
//! # Functions
//!
//! - [`is_unitary`]: Check whether a gate matrix is unitary within a given tolerance
//! - [`unitarity_error`]: Compute the Frobenius norm of the deviation from unitarity
//! - [`validate_gate_unitarity`]: Validate a gate and return a descriptive error if not unitary
//! - [`validate_all_standard_gates`]: Validate all built-in standard gates
//!
//! # Runtime Validation
//!
//! The module provides a global flag to enable or disable runtime unitarity
//! checks during gate creation. When enabled, custom gate creation functions
//! will verify unitarity before accepting the gate matrix.
//!
//! # Examples
//!
//! ```
//! use numrs2::new_modules::quantum::unitarity::{is_unitary, unitarity_error};
//! use numrs2::new_modules::quantum::gates::hadamard;
//!
//! let h = hadamard::<f64>().expect("valid Hadamard gate");
//!
//! // Check if the Hadamard gate is unitary
//! assert!(is_unitary(&h, 1e-10).expect("valid unitarity check"));
//!
//! // Compute the deviation from unitarity
//! let error = unitarity_error(&h).expect("valid unitarity error computation");
//! assert!(error < 1e-10);
//! ```
//!
//! # References
//!
//! - Nielsen & Chuang, "Quantum Computation and Quantum Information" (2010), Chapter 2

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use scirs2_core::Complex;
use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, Ordering};

/// Global flag to enable or disable runtime unitarity validation.
///
/// When set to `true`, custom gate creation functions will verify that
/// the provided matrix is unitary before accepting it.
///
/// Default: `false` (disabled for performance)
static RUNTIME_UNITARITY_CHECK: AtomicBool = AtomicBool::new(false);

/// Enable runtime unitarity validation for gate creation.
///
/// When enabled, [`create_validated_gate`] will check that matrices
/// satisfy the unitarity condition U†U ≈ I before accepting them.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::quantum::unitarity::{enable_runtime_validation, is_runtime_validation_enabled};
///
/// enable_runtime_validation();
/// assert!(is_runtime_validation_enabled());
/// ```
pub fn enable_runtime_validation() {
    RUNTIME_UNITARITY_CHECK.store(true, Ordering::SeqCst);
}

/// Disable runtime unitarity validation for gate creation.
///
/// When disabled, gate creation functions will skip the unitarity check
/// for better performance.
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::quantum::unitarity::{disable_runtime_validation, is_runtime_validation_enabled};
///
/// disable_runtime_validation();
/// assert!(!is_runtime_validation_enabled());
/// ```
pub fn disable_runtime_validation() {
    RUNTIME_UNITARITY_CHECK.store(false, Ordering::SeqCst);
}

/// Check whether runtime unitarity validation is currently enabled.
///
/// # Returns
///
/// `true` if runtime validation is enabled, `false` otherwise.
pub fn is_runtime_validation_enabled() -> bool {
    RUNTIME_UNITARITY_CHECK.load(Ordering::SeqCst)
}

/// Check whether a gate matrix is unitary within a given tolerance.
///
/// A matrix U is unitary if U†U ≈ I, where the deviation of each
/// element from the identity matrix is within the specified tolerance.
///
/// # Arguments
///
/// * `gate_matrix` - The square complex matrix to check
/// * `tolerance` - Maximum allowed absolute deviation per element from identity
///
/// # Returns
///
/// * `Ok(true)` if the matrix is unitary within tolerance
/// * `Ok(false)` if the matrix is not unitary
/// * `Err(...)` if the matrix is not square or indexing fails
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::quantum::unitarity::is_unitary;
/// use numrs2::new_modules::quantum::gates::{hadamard, pauli_x};
///
/// let h = hadamard::<f64>().expect("valid Hadamard gate");
/// assert!(is_unitary(&h, 1e-10).expect("valid unitarity check"));
///
/// let x = pauli_x::<f64>().expect("valid Pauli-X gate");
/// assert!(is_unitary(&x, 1e-10).expect("valid unitarity check"));
/// ```
pub fn is_unitary<T>(gate_matrix: &Array<Complex<T>>, tolerance: f64) -> Result<bool>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    validate_square_matrix(gate_matrix)?;

    let shape = gate_matrix.shape();
    let n = shape[0];

    // Compute U†U and check each element against identity
    for i in 0..n {
        for j in 0..n {
            let mut sum = Complex::new(T::zero(), T::zero());
            for k in 0..n {
                let u_ki = gate_matrix.get(&[k, i]).map_err(|e| {
                    NumRs2Error::ComputationError(format!(
                        "Failed to access gate element [{}, {}]: {}",
                        k, i, e
                    ))
                })?;
                let u_kj = gate_matrix.get(&[k, j]).map_err(|e| {
                    NumRs2Error::ComputationError(format!(
                        "Failed to access gate element [{}, {}]: {}",
                        k, j, e
                    ))
                })?;
                // U†[i,k] * U[k,j] = conj(U[k,i]) * U[k,j]
                sum = sum + u_ki.conj() * u_kj;
            }

            let expected_re = if i == j { T::one() } else { T::zero() };
            let diff_re: f64 = (sum.re - expected_re).abs().into();
            let diff_im: f64 = sum.im.abs().into();

            if diff_re > tolerance || diff_im > tolerance {
                return Ok(false);
            }
        }
    }

    Ok(true)
}

/// Compute the unitarity error of a gate matrix.
///
/// Calculates ||U†U - I||_F, the Frobenius norm of the deviation from unitarity.
/// A perfectly unitary matrix will have an error of 0.0.
///
/// The Frobenius norm is computed as:
///   ||M||_F = sqrt(sum_{i,j} |M_{i,j}|²)
///
/// # Arguments
///
/// * `gate_matrix` - The square complex matrix to evaluate
///
/// # Returns
///
/// * `Ok(f64)` - The Frobenius norm of (U†U - I)
/// * `Err(...)` if the matrix is not square or indexing fails
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::quantum::unitarity::unitarity_error;
/// use numrs2::new_modules::quantum::gates::hadamard;
///
/// let h = hadamard::<f64>().expect("valid Hadamard gate");
/// let error = unitarity_error(&h).expect("valid unitarity error computation");
/// assert!(error < 1e-14);
/// ```
pub fn unitarity_error<T>(gate_matrix: &Array<Complex<T>>) -> Result<f64>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    validate_square_matrix(gate_matrix)?;

    let shape = gate_matrix.shape();
    let n = shape[0];

    let mut frobenius_sq: f64 = 0.0;

    // Compute ||U†U - I||_F²
    for i in 0..n {
        for j in 0..n {
            let mut sum = Complex::new(T::zero(), T::zero());
            for k in 0..n {
                let u_ki = gate_matrix.get(&[k, i]).map_err(|e| {
                    NumRs2Error::ComputationError(format!(
                        "Failed to access gate element [{}, {}]: {}",
                        k, i, e
                    ))
                })?;
                let u_kj = gate_matrix.get(&[k, j]).map_err(|e| {
                    NumRs2Error::ComputationError(format!(
                        "Failed to access gate element [{}, {}]: {}",
                        k, j, e
                    ))
                })?;
                sum = sum + u_ki.conj() * u_kj;
            }

            // Subtract identity element
            let identity_val = if i == j { T::one() } else { T::zero() };
            let diff_re: f64 = (sum.re - identity_val).into();
            let diff_im: f64 = sum.im.into();

            frobenius_sq += diff_re * diff_re + diff_im * diff_im;
        }
    }

    Ok(frobenius_sq.sqrt())
}

/// Validate a gate matrix for unitarity and return a descriptive error if not unitary.
///
/// This function performs a complete unitarity check and returns a detailed
/// error message if the matrix fails the test.
///
/// # Arguments
///
/// * `gate_matrix` - The square complex matrix to validate
/// * `tolerance` - Maximum allowed Frobenius norm deviation from identity
/// * `gate_name` - Optional name of the gate for error reporting
///
/// # Returns
///
/// * `Ok(())` if the gate is unitary within tolerance
/// * `Err(NumRs2Error::InvalidInput(...))` if the gate is not unitary
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::quantum::unitarity::validate_gate_unitarity;
/// use numrs2::new_modules::quantum::gates::pauli_z;
///
/// let z = pauli_z::<f64>().expect("valid Pauli-Z gate");
/// validate_gate_unitarity(&z, 1e-10, Some("Pauli-Z")).expect("valid unitarity validation");
/// ```
pub fn validate_gate_unitarity<T>(
    gate_matrix: &Array<Complex<T>>,
    tolerance: f64,
    gate_name: Option<&str>,
) -> Result<()>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let error = unitarity_error(gate_matrix)?;

    if error > tolerance {
        let name = gate_name.unwrap_or("unnamed");
        return Err(NumRs2Error::InvalidInput(format!(
            "Gate '{}' is not unitary: ||U†U - I||_F = {:.6e} (tolerance: {:.6e})",
            name, error, tolerance
        )));
    }

    Ok(())
}

/// Create a validated custom gate with optional runtime unitarity checking.
///
/// If runtime validation is enabled (via [`enable_runtime_validation`]),
/// this function will verify the matrix is unitary before returning it.
/// Otherwise, it only checks that the matrix is square.
///
/// # Arguments
///
/// * `matrix` - The square complex matrix representing the gate
/// * `tolerance` - Tolerance for unitarity check (used only if runtime validation is enabled)
/// * `gate_name` - Optional name for error reporting
///
/// # Returns
///
/// * `Ok(Array<Complex<T>>)` - The validated gate matrix
/// * `Err(...)` if validation fails
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::quantum::unitarity::{
///     create_validated_gate, enable_runtime_validation, disable_runtime_validation,
/// };
/// use numrs2::new_modules::quantum::gates::hadamard;
///
/// // Get the Hadamard gate matrix
/// let h = hadamard::<f64>().expect("valid Hadamard gate");
///
/// // With runtime validation enabled
/// enable_runtime_validation();
/// let validated = create_validated_gate(h.clone(), 1e-10, Some("Custom-H"));
/// assert!(validated.is_ok());
///
/// // Clean up
/// disable_runtime_validation();
/// ```
pub fn create_validated_gate<T>(
    matrix: Array<Complex<T>>,
    tolerance: f64,
    gate_name: Option<&str>,
) -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    validate_square_matrix(&matrix)?;

    if is_runtime_validation_enabled() {
        validate_gate_unitarity(&matrix, tolerance, gate_name)?;
    }

    Ok(matrix)
}

/// Result of validating all standard gates.
///
/// Contains per-gate validation results including gate name,
/// whether it passed, and the unitarity error.
#[derive(Debug, Clone)]
pub struct GateValidationResult {
    /// Name of the gate
    pub gate_name: String,
    /// Whether the gate passed the unitarity check
    pub is_unitary: bool,
    /// The Frobenius norm of U†U - I
    pub unitarity_error: f64,
    /// Dimension of the gate matrix
    pub dimension: usize,
}

/// Result of validating all standard gates.
#[derive(Debug, Clone)]
pub struct StandardGatesValidation {
    /// Per-gate validation results
    pub results: Vec<GateValidationResult>,
    /// Whether all gates passed
    pub all_passed: bool,
    /// Number of gates that passed
    pub num_passed: usize,
    /// Number of gates that failed
    pub num_failed: usize,
}

/// Validate all built-in standard quantum gates.
///
/// Tests the following gates for unitarity:
/// - Pauli gates: X, Y, Z
/// - Hadamard gate: H
/// - Phase gates: S, T
/// - Rotation gates: Rx(pi/4), Ry(pi/4), Rz(pi/4)
/// - Two-qubit gates: CNOT, SWAP, CZ, CY
/// - Multi-qubit gates: Toffoli, Fredkin
/// - Identity gates: 1-qubit, 2-qubit
///
/// # Arguments
///
/// * `tolerance` - Maximum allowed Frobenius norm deviation from identity
///
/// # Returns
///
/// * `Ok(StandardGatesValidation)` - Comprehensive validation results
/// * `Err(...)` if gate creation or validation fails unexpectedly
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::quantum::unitarity::validate_all_standard_gates;
///
/// let validation = validate_all_standard_gates(1e-10).expect("valid gate validation");
/// assert!(validation.all_passed);
/// ```
pub fn validate_all_standard_gates(tolerance: f64) -> Result<StandardGatesValidation> {
    use super::gates;

    let mut results = Vec::new();

    // Helper closure to validate a single gate
    let mut validate_one = |name: &str, gate_result: Result<Array<Complex<f64>>>| -> Result<()> {
        let gate = gate_result?;
        let dim = gate.shape()[0];
        let error = unitarity_error(&gate)?;
        let passed = error <= tolerance;

        results.push(GateValidationResult {
            gate_name: name.to_string(),
            is_unitary: passed,
            unitarity_error: error,
            dimension: dim,
        });

        Ok(())
    };

    // Single-qubit gates
    validate_one("Pauli-X", gates::pauli_x::<f64>())?;
    validate_one("Pauli-Y", gates::pauli_y::<f64>())?;
    validate_one("Pauli-Z", gates::pauli_z::<f64>())?;
    validate_one("Hadamard", gates::hadamard::<f64>())?;
    validate_one("Phase (S)", gates::phase_gate::<f64>())?;
    validate_one("T gate", gates::t_gate::<f64>())?;

    // Rotation gates at pi/4
    let theta = std::f64::consts::PI / 4.0;
    validate_one("Rx(pi/4)", gates::rx(theta))?;
    validate_one("Ry(pi/4)", gates::ry(theta))?;
    validate_one("Rz(pi/4)", gates::rz(theta))?;

    // Two-qubit gates
    validate_one("CNOT", gates::cnot::<f64>())?;
    validate_one("SWAP", gates::swap::<f64>())?;
    validate_one("CZ", gates::cz::<f64>())?;
    validate_one("CY", gates::cy::<f64>())?;

    // Multi-qubit gates
    validate_one("Toffoli", gates::toffoli::<f64>())?;
    validate_one("Fredkin", gates::fredkin::<f64>())?;

    // Identity gates
    validate_one("Identity-1q", gates::identity::<f64>(1))?;
    validate_one("Identity-2q", gates::identity::<f64>(2))?;

    let num_passed = results.iter().filter(|r| r.is_unitary).count();
    let num_failed = results.len() - num_passed;
    let all_passed = num_failed == 0;

    Ok(StandardGatesValidation {
        results,
        all_passed,
        num_passed,
        num_failed,
    })
}

/// Validate that a matrix is square and at least 1x1.
///
/// # Arguments
///
/// * `matrix` - The matrix to validate
///
/// # Returns
///
/// * `Ok(())` if the matrix is square
/// * `Err(NumRs2Error::DimensionMismatch(...))` if the matrix is not square or not 2D
fn validate_square_matrix<T>(matrix: &Array<T>) -> Result<()>
where
    T: Clone,
{
    let shape = matrix.shape();

    if shape.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Gate matrix must be 2-dimensional, got {}-dimensional",
            shape.len()
        )));
    }

    if shape[0] != shape[1] {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Gate matrix must be square, got {}x{}",
            shape[0], shape[1]
        )));
    }

    if shape[0] == 0 {
        return Err(NumRs2Error::DimensionMismatch(
            "Gate matrix must have at least dimension 1x1".to_string(),
        ));
    }

    Ok(())
}

/// Compute the product of two square complex matrices.
///
/// Used internally for checking that the product of unitary matrices is unitary.
///
/// # Arguments
///
/// * `a` - First matrix
/// * `b` - Second matrix
///
/// # Returns
///
/// * `Ok(Array<Complex<T>>)` - The matrix product A * B
/// * `Err(...)` if dimensions are incompatible or access fails
pub fn matrix_product<T>(a: &Array<Complex<T>>, b: &Array<Complex<T>>) -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let shape_a = a.shape();
    let shape_b = b.shape();

    if shape_a.len() != 2 || shape_b.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "Both matrices must be 2-dimensional".to_string(),
        ));
    }

    if shape_a[1] != shape_b[0] {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "Matrix dimensions incompatible for multiplication: {}x{} * {}x{}",
            shape_a[0], shape_a[1], shape_b[0], shape_b[1]
        )));
    }

    let m = shape_a[0];
    let n = shape_b[1];
    let p = shape_a[1];

    let mut result_data = vec![Complex::new(T::zero(), T::zero()); m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = Complex::new(T::zero(), T::zero());
            for k in 0..p {
                let a_ik = a.get(&[i, k]).map_err(|e| {
                    NumRs2Error::ComputationError(format!(
                        "Failed to access matrix A element [{}, {}]: {}",
                        i, k, e
                    ))
                })?;
                let b_kj = b.get(&[k, j]).map_err(|e| {
                    NumRs2Error::ComputationError(format!(
                        "Failed to access matrix B element [{}, {}]: {}",
                        k, j, e
                    ))
                })?;
                sum = sum + a_ik * b_kj;
            }
            result_data[i * n + j] = sum;
        }
    }

    Ok(Array::from_vec(result_data).reshape(&[m, n]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::new_modules::quantum::gates;

    /// Test 1: Verify all standard single-qubit gates are unitary
    #[test]
    fn test_single_qubit_gates_unitary() {
        let tolerance = 1e-12;

        let x = gates::pauli_x::<f64>().expect("pauli_x should succeed");
        assert!(
            is_unitary(&x, tolerance).expect("is_unitary check should not fail"),
            "Pauli-X should be unitary"
        );

        let y = gates::pauli_y::<f64>().expect("pauli_y should succeed");
        assert!(
            is_unitary(&y, tolerance).expect("is_unitary check should not fail"),
            "Pauli-Y should be unitary"
        );

        let z = gates::pauli_z::<f64>().expect("pauli_z should succeed");
        assert!(
            is_unitary(&z, tolerance).expect("is_unitary check should not fail"),
            "Pauli-Z should be unitary"
        );

        let h = gates::hadamard::<f64>().expect("hadamard should succeed");
        assert!(
            is_unitary(&h, tolerance).expect("is_unitary check should not fail"),
            "Hadamard should be unitary"
        );

        let s = gates::phase_gate::<f64>().expect("phase_gate should succeed");
        assert!(
            is_unitary(&s, tolerance).expect("is_unitary check should not fail"),
            "Phase (S) gate should be unitary"
        );

        let t = gates::t_gate::<f64>().expect("t_gate should succeed");
        assert!(
            is_unitary(&t, tolerance).expect("is_unitary check should not fail"),
            "T gate should be unitary"
        );
    }

    /// Test 2: Verify multi-qubit gates (CNOT, SWAP, Toffoli, Fredkin) are unitary
    #[test]
    fn test_multi_qubit_gates_unitary() {
        let tolerance = 1e-12;

        let cnot_gate = gates::cnot::<f64>().expect("cnot should succeed");
        assert!(
            is_unitary(&cnot_gate, tolerance).expect("is_unitary check should not fail"),
            "CNOT should be unitary"
        );

        let swap_gate = gates::swap::<f64>().expect("swap should succeed");
        assert!(
            is_unitary(&swap_gate, tolerance).expect("is_unitary check should not fail"),
            "SWAP should be unitary"
        );

        let cz_gate = gates::cz::<f64>().expect("cz should succeed");
        assert!(
            is_unitary(&cz_gate, tolerance).expect("is_unitary check should not fail"),
            "CZ should be unitary"
        );

        let cy_gate = gates::cy::<f64>().expect("cy should succeed");
        assert!(
            is_unitary(&cy_gate, tolerance).expect("is_unitary check should not fail"),
            "CY should be unitary"
        );

        let toffoli_gate = gates::toffoli::<f64>().expect("toffoli should succeed");
        assert!(
            is_unitary(&toffoli_gate, tolerance).expect("is_unitary check should not fail"),
            "Toffoli should be unitary"
        );

        let fredkin_gate = gates::fredkin::<f64>().expect("fredkin should succeed");
        assert!(
            is_unitary(&fredkin_gate, tolerance).expect("is_unitary check should not fail"),
            "Fredkin should be unitary"
        );
    }

    /// Test 3: Detect a non-unitary matrix
    #[test]
    fn test_non_unitary_matrix_detected() {
        // A non-unitary matrix (scaling matrix)
        let data = vec![
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.0, 0.0),
        ];
        let non_unitary = Array::from_vec(data).reshape(&[2, 2]);

        let result = is_unitary(&non_unitary, 1e-10).expect("is_unitary check should not fail");
        assert!(!result, "Scaling matrix should NOT be detected as unitary");

        let error = unitarity_error(&non_unitary).expect("unitarity_error should not fail");
        assert!(
            error > 1.0,
            "Scaling matrix should have a large unitarity error, got {}",
            error
        );
    }

    /// Test 4: Tolerance handling - tight vs loose tolerance
    #[test]
    fn test_tolerance_handling() {
        let h = gates::hadamard::<f64>().expect("hadamard should succeed");

        // Very tight tolerance should still pass for exact gates
        assert!(
            is_unitary(&h, 1e-15).expect("is_unitary check should not fail"),
            "Hadamard should pass with very tight tolerance"
        );

        // Create a slightly non-unitary matrix (Hadamard with small perturbation)
        let data = vec![
            Complex::new(1.0 / 2.0_f64.sqrt() + 1e-6, 0.0),
            Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex::new(-1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        let perturbed = Array::from_vec(data).reshape(&[2, 2]);

        // Should fail with tight tolerance
        assert!(
            !is_unitary(&perturbed, 1e-10).expect("is_unitary check should not fail"),
            "Perturbed Hadamard should fail with tight tolerance"
        );

        // Should pass with loose tolerance
        assert!(
            is_unitary(&perturbed, 1e-3).expect("is_unitary check should not fail"),
            "Perturbed Hadamard should pass with loose tolerance"
        );
    }

    /// Test 5: Identity gate is unitary
    #[test]
    fn test_identity_gate_unitary() {
        let tolerance = 1e-14;

        // 1-qubit identity
        let id1 = gates::identity::<f64>(1).expect("identity(1) should succeed");
        assert!(
            is_unitary(&id1, tolerance).expect("is_unitary check should not fail"),
            "1-qubit identity should be unitary"
        );
        let error1 = unitarity_error(&id1).expect("unitarity_error should not fail");
        assert!(
            error1 < 1e-15,
            "Identity unitarity error should be essentially zero, got {}",
            error1
        );

        // 2-qubit identity
        let id2 = gates::identity::<f64>(2).expect("identity(2) should succeed");
        assert!(
            is_unitary(&id2, tolerance).expect("is_unitary check should not fail"),
            "2-qubit identity should be unitary"
        );

        // 3-qubit identity
        let id3 = gates::identity::<f64>(3).expect("identity(3) should succeed");
        assert!(
            is_unitary(&id3, tolerance).expect("is_unitary check should not fail"),
            "3-qubit identity should be unitary"
        );
    }

    /// Test 6: Composed gates - product of unitary gates should be unitary
    #[test]
    fn test_composed_gates_unitary() {
        let tolerance = 1e-10;

        // H * X should be unitary (product of two unitaries)
        let h = gates::hadamard::<f64>().expect("hadamard should succeed");
        let x = gates::pauli_x::<f64>().expect("pauli_x should succeed");
        let hx = matrix_product(&h, &x).expect("matrix_product should succeed");
        assert!(
            is_unitary(&hx, tolerance).expect("is_unitary check should not fail"),
            "H*X should be unitary"
        );

        // X * Y * Z should be unitary
        let y = gates::pauli_y::<f64>().expect("pauli_y should succeed");
        let z = gates::pauli_z::<f64>().expect("pauli_z should succeed");
        let xy = matrix_product(&x, &y).expect("matrix_product should succeed");
        let xyz = matrix_product(&xy, &z).expect("matrix_product should succeed");
        assert!(
            is_unitary(&xyz, tolerance).expect("is_unitary check should not fail"),
            "X*Y*Z should be unitary"
        );

        // H * H should be approximately identity (H is self-inverse)
        let hh = matrix_product(&h, &h).expect("matrix_product should succeed");
        let hh_error = unitarity_error(&hh).expect("unitarity_error should not fail");
        assert!(
            hh_error < tolerance,
            "H*H unitarity error should be small, got {}",
            hh_error
        );
    }

    /// Test 7: Edge case - 1x1 gate (global phase)
    #[test]
    fn test_1x1_gate() {
        let tolerance = 1e-14;

        // e^(i*pi/4) as a 1x1 unitary
        let phase = std::f64::consts::PI / 4.0;
        let data = vec![Complex::new(phase.cos(), phase.sin())];
        let gate = Array::from_vec(data).reshape(&[1, 1]);

        assert!(
            is_unitary(&gate, tolerance).expect("is_unitary check should not fail"),
            "1x1 phase gate should be unitary"
        );

        let error = unitarity_error(&gate).expect("unitarity_error should not fail");
        assert!(
            error < 1e-15,
            "1x1 phase gate error should be ~0, got {}",
            error
        );

        // Non-unitary 1x1 (magnitude != 1)
        let data_non_unitary = vec![Complex::new(0.5, 0.0)];
        let non_unitary_gate = Array::from_vec(data_non_unitary).reshape(&[1, 1]);

        assert!(
            !is_unitary(&non_unitary_gate, tolerance).expect("is_unitary check should not fail"),
            "1x1 scaling gate should NOT be unitary"
        );
    }

    /// Test 8: Larger gates (4-qubit identity = 16x16)
    #[test]
    fn test_larger_gates() {
        let tolerance = 1e-12;

        // 4-qubit identity (16x16 matrix)
        let id4 = gates::identity::<f64>(4).expect("identity(4) should succeed");
        assert_eq!(id4.shape(), vec![16, 16]);
        assert!(
            is_unitary(&id4, tolerance).expect("is_unitary check should not fail"),
            "4-qubit identity should be unitary"
        );

        let error = unitarity_error(&id4).expect("unitarity_error should not fail");
        assert!(
            error < 1e-15,
            "4-qubit identity error should be ~0, got {}",
            error
        );
    }

    /// Test 9: Rotation gates at various angles are always unitary
    #[test]
    fn test_rotation_gates_various_angles() {
        let tolerance = 1e-12;

        let angles = [
            0.0,
            std::f64::consts::PI / 6.0,
            std::f64::consts::PI / 4.0,
            std::f64::consts::PI / 3.0,
            std::f64::consts::PI / 2.0,
            std::f64::consts::PI,
            std::f64::consts::TAU,
            -std::f64::consts::PI / 4.0,
            3.7, // arbitrary angle
        ];

        for &angle in &angles {
            let rx_gate = gates::rx(angle).expect("rx should succeed");
            assert!(
                is_unitary(&rx_gate, tolerance).expect("is_unitary check should not fail"),
                "Rx({}) should be unitary",
                angle
            );

            let ry_gate = gates::ry(angle).expect("ry should succeed");
            assert!(
                is_unitary(&ry_gate, tolerance).expect("is_unitary check should not fail"),
                "Ry({}) should be unitary",
                angle
            );

            let rz_gate = gates::rz(angle).expect("rz should succeed");
            assert!(
                is_unitary(&rz_gate, tolerance).expect("is_unitary check should not fail"),
                "Rz({}) should be unitary",
                angle
            );
        }
    }

    /// Test 10: validate_all_standard_gates comprehensive check
    #[test]
    fn test_validate_all_standard_gates() {
        let validation = validate_all_standard_gates(1e-10).expect("validation should not fail");

        assert!(
            validation.all_passed,
            "All standard gates should pass unitarity check. Failed gates: {:?}",
            validation
                .results
                .iter()
                .filter(|r| !r.is_unitary)
                .map(|r| format!("{}: error={:.6e}", r.gate_name, r.unitarity_error))
                .collect::<Vec<_>>()
        );

        assert_eq!(validation.num_failed, 0, "No standard gates should fail");

        assert!(
            validation.num_passed >= 17,
            "Should validate at least 17 gates, got {}",
            validation.num_passed
        );
    }

    /// Test 11: Non-square matrix is rejected
    #[test]
    fn test_non_square_matrix_rejected() {
        let data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let non_square = Array::from_vec(data).reshape(&[2, 3]);

        let result = is_unitary(&non_square, 1e-10);
        assert!(result.is_err(), "Non-square matrix should return an error");

        let error_result = unitarity_error(&non_square);
        assert!(
            error_result.is_err(),
            "unitarity_error should fail for non-square matrix"
        );
    }

    /// Test 12: validate_gate_unitarity provides meaningful error messages
    #[test]
    fn test_validate_gate_unitarity_error_message() {
        // Create a non-unitary matrix
        let data = vec![
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
        ];
        let non_unitary = Array::from_vec(data).reshape(&[2, 2]);

        let result = validate_gate_unitarity(&non_unitary, 1e-10, Some("BadGate"));
        assert!(result.is_err(), "Non-unitary gate should fail validation");

        let err_msg = format!("{}", result.expect_err("should be error"));
        assert!(
            err_msg.contains("BadGate"),
            "Error message should contain gate name, got: {}",
            err_msg
        );
        assert!(
            err_msg.contains("not unitary"),
            "Error message should indicate non-unitarity, got: {}",
            err_msg
        );
    }

    /// Test 13: Runtime validation flag works correctly
    #[test]
    fn test_runtime_validation_flag() {
        // Ensure we start with validation disabled
        disable_runtime_validation();
        assert!(!is_runtime_validation_enabled());

        // Non-unitary matrix should pass when validation is disabled
        let data = vec![
            Complex::new(2.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.0, 0.0),
        ];
        let non_unitary = Array::from_vec(data.clone()).reshape(&[2, 2]);
        let result = create_validated_gate(non_unitary, 1e-10, Some("Test"));
        assert!(
            result.is_ok(),
            "Non-unitary should pass when validation is disabled"
        );

        // Enable validation
        enable_runtime_validation();
        assert!(is_runtime_validation_enabled());

        // Same non-unitary matrix should fail now
        let non_unitary_2 = Array::from_vec(data).reshape(&[2, 2]);
        let result_2 = create_validated_gate(non_unitary_2, 1e-10, Some("Test"));
        assert!(
            result_2.is_err(),
            "Non-unitary should fail when validation is enabled"
        );

        // Valid unitary should still pass
        let h = gates::hadamard::<f64>().expect("hadamard should succeed");
        let result_3 = create_validated_gate(h, 1e-10, Some("Hadamard"));
        assert!(result_3.is_ok(), "Hadamard should pass validation");

        // Clean up
        disable_runtime_validation();
    }

    /// Test 14: Controlled gate preserves unitarity
    #[test]
    fn test_controlled_gate_unitarity() {
        let tolerance = 1e-12;

        let x = gates::pauli_x::<f64>().expect("pauli_x should succeed");
        let cx = gates::controlled_gate(&x).expect("controlled_gate should succeed");

        assert!(
            is_unitary(&cx, tolerance).expect("is_unitary check should not fail"),
            "Controlled-X should be unitary"
        );

        let h = gates::hadamard::<f64>().expect("hadamard should succeed");
        let ch = gates::controlled_gate(&h).expect("controlled_gate should succeed");

        assert!(
            is_unitary(&ch, tolerance).expect("is_unitary check should not fail"),
            "Controlled-H should be unitary"
        );

        let s = gates::phase_gate::<f64>().expect("phase_gate should succeed");
        let cs = gates::controlled_gate(&s).expect("controlled_gate should succeed");

        assert!(
            is_unitary(&cs, tolerance).expect("is_unitary check should not fail"),
            "Controlled-S should be unitary"
        );
    }

    /// Test 15: Unitarity error is zero for exact identity
    #[test]
    fn test_exact_identity_zero_error() {
        let id1 = gates::identity::<f64>(1).expect("identity(1) should succeed");
        let error = unitarity_error(&id1).expect("unitarity_error should not fail");
        assert!(
            error < f64::EPSILON,
            "Identity gate should have zero unitarity error, got {}",
            error
        );
    }
}
