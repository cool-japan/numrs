//! Quantum Gates
//!
//! This module provides standard quantum gates for quantum circuit simulation.
//! All gates are represented as unitary matrices operating on quantum states.
//!
//! # Gate Categories
//!
//! - **Single-qubit gates**: X, Y, Z, H, S, T, Rx, Ry, Rz
//! - **Two-qubit gates**: CNOT, SWAP, CZ, CY
//! - **Multi-qubit gates**: Toffoli (CCX), Fredkin (CSWAP)
//! - **Custom gates**: User-defined unitary operations
//!
//! # Mathematical Background
//!
//! All quantum gates are unitary transformations U satisfying U†U = I.
//! Single-qubit gates are 2×2 unitary matrices, two-qubit gates are 4×4, etc.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use scirs2_core::Complex;
use std::fmt::Debug;

/// Pauli-X gate (NOT gate)
///
/// Matrix representation:
/// ```text
/// X = |0 1|
///     |1 0|
/// ```
///
/// # Examples
///
/// ```
/// use numrs2::new_modules::quantum::gates::pauli_x;
///
/// let x_gate = pauli_x::<f64>().unwrap();
/// assert_eq!(x_gate.shape(), &[2, 2]);
/// ```
pub fn pauli_x<T>() -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let data = vec![
        Complex::new(T::zero(), T::zero()),
        Complex::new(T::one(), T::zero()),
        Complex::new(T::one(), T::zero()),
        Complex::new(T::zero(), T::zero()),
    ];
    Ok(Array::from_vec(data).reshape(&[2, 2]))
}

/// Pauli-Y gate
///
/// Matrix representation:
/// ```text
/// Y = |0  -i|
///     |i   0|
/// ```
pub fn pauli_y<T>() -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let data = vec![
        Complex::new(T::zero(), T::zero()),
        Complex::new(T::zero(), -T::one()),
        Complex::new(T::zero(), T::one()),
        Complex::new(T::zero(), T::zero()),
    ];
    Ok(Array::from_vec(data).reshape(&[2, 2]))
}

/// Pauli-Z gate
///
/// Matrix representation:
/// ```text
/// Z = |1   0|
///     |0  -1|
/// ```
pub fn pauli_z<T>() -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let data = vec![
        Complex::new(T::one(), T::zero()),
        Complex::new(T::zero(), T::zero()),
        Complex::new(T::zero(), T::zero()),
        Complex::new(-T::one(), T::zero()),
    ];
    Ok(Array::from_vec(data).reshape(&[2, 2]))
}

/// Hadamard gate
///
/// Creates equal superposition: H|0⟩ = (|0⟩ + |1⟩)/√2
///
/// Matrix representation:
/// ```text
/// H = 1/√2 |1   1|
///          |1  -1|
/// ```
pub fn hadamard<T>() -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let inv_sqrt2 = <T as From<f64>>::from(1.0 / 2.0_f64.sqrt());
    let data = vec![
        Complex::new(inv_sqrt2, T::zero()),
        Complex::new(inv_sqrt2, T::zero()),
        Complex::new(inv_sqrt2, T::zero()),
        Complex::new(-inv_sqrt2, T::zero()),
    ];
    Ok(Array::from_vec(data).reshape(&[2, 2]))
}

/// Phase gate (S gate)
///
/// Matrix representation:
/// ```text
/// S = |1  0|
///     |0  i|
/// ```
pub fn phase_gate<T>() -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let data = vec![
        Complex::new(T::one(), T::zero()),
        Complex::new(T::zero(), T::zero()),
        Complex::new(T::zero(), T::zero()),
        Complex::new(T::zero(), T::one()),
    ];
    Ok(Array::from_vec(data).reshape(&[2, 2]))
}

/// T gate (π/8 gate)
///
/// Matrix representation:
/// ```text
/// T = |1      0    |
///     |0  e^(iπ/4)|
/// ```
pub fn t_gate<T>() -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let phase = <T as From<f64>>::from(std::f64::consts::PI / 4.0);
    let data = vec![
        Complex::new(T::one(), T::zero()),
        Complex::new(T::zero(), T::zero()),
        Complex::new(T::zero(), T::zero()),
        Complex::new(phase.cos(), phase.sin()),
    ];
    Ok(Array::from_vec(data).reshape(&[2, 2]))
}

/// Rotation around X-axis
///
/// Matrix representation:
/// ```text
/// Rx(θ) = |cos(θ/2)   -i·sin(θ/2)|
///         |-i·sin(θ/2)  cos(θ/2) |
/// ```
///
/// # Arguments
///
/// * `theta` - Rotation angle in radians
pub fn rx<T>(theta: T) -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let half_theta = theta / <T as From<f64>>::from(2.0);
    let cos_val = half_theta.cos();
    let sin_val = half_theta.sin();

    let data = vec![
        Complex::new(cos_val, T::zero()),
        Complex::new(T::zero(), -sin_val),
        Complex::new(T::zero(), -sin_val),
        Complex::new(cos_val, T::zero()),
    ];
    Ok(Array::from_vec(data).reshape(&[2, 2]))
}

/// Rotation around Y-axis
///
/// Matrix representation:
/// ```text
/// Ry(θ) = |cos(θ/2)  -sin(θ/2)|
///         |sin(θ/2)   cos(θ/2)|
/// ```
///
/// # Arguments
///
/// * `theta` - Rotation angle in radians
pub fn ry<T>(theta: T) -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let half_theta = theta / <T as From<f64>>::from(2.0);
    let cos_val = half_theta.cos();
    let sin_val = half_theta.sin();

    let data = vec![
        Complex::new(cos_val, T::zero()),
        Complex::new(-sin_val, T::zero()),
        Complex::new(sin_val, T::zero()),
        Complex::new(cos_val, T::zero()),
    ];
    Ok(Array::from_vec(data).reshape(&[2, 2]))
}

/// Rotation around Z-axis
///
/// Matrix representation:
/// ```text
/// Rz(θ) = |e^(-iθ/2)      0     |
///         |    0      e^(iθ/2) |
/// ```
///
/// # Arguments
///
/// * `theta` - Rotation angle in radians
pub fn rz<T>(theta: T) -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let half_theta = theta / <T as From<f64>>::from(2.0);
    let cos_val = half_theta.cos();
    let sin_val = half_theta.sin();

    let data = vec![
        Complex::new(cos_val, -sin_val),
        Complex::new(T::zero(), T::zero()),
        Complex::new(T::zero(), T::zero()),
        Complex::new(cos_val, sin_val),
    ];
    Ok(Array::from_vec(data).reshape(&[2, 2]))
}

/// CNOT gate (Controlled-NOT)
///
/// Matrix representation (4×4):
/// ```text
/// CNOT = |1 0 0 0|
///        |0 1 0 0|
///        |0 0 0 1|
///        |0 0 1 0|
/// ```
///
/// Control qubit at index 0, target at index 1 in the gate's qubit space
pub fn cnot<T>() -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let mut data = vec![Complex::new(T::zero(), T::zero()); 16];
    // Identity for |00⟩ and |01⟩
    data[0] = Complex::new(T::one(), T::zero()); // |00⟩ -> |00⟩
    data[5] = Complex::new(T::one(), T::zero()); // |01⟩ -> |01⟩
                                                 // NOT for |10⟩ and |11⟩
    data[11] = Complex::new(T::one(), T::zero()); // |10⟩ -> |11⟩
    data[14] = Complex::new(T::one(), T::zero()); // |11⟩ -> |10⟩

    Ok(Array::from_vec(data).reshape(&[4, 4]))
}

/// SWAP gate
///
/// Swaps two qubits
///
/// Matrix representation (4×4):
/// ```text
/// SWAP = |1 0 0 0|
///        |0 0 1 0|
///        |0 1 0 0|
///        |0 0 0 1|
/// ```
pub fn swap<T>() -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let mut data = vec![Complex::new(T::zero(), T::zero()); 16];
    data[0] = Complex::new(T::one(), T::zero()); // |00⟩ -> |00⟩
    data[6] = Complex::new(T::one(), T::zero()); // |01⟩ -> |10⟩
    data[9] = Complex::new(T::one(), T::zero()); // |10⟩ -> |01⟩
    data[15] = Complex::new(T::one(), T::zero()); // |11⟩ -> |11⟩

    Ok(Array::from_vec(data).reshape(&[4, 4]))
}

/// Controlled-Z gate
///
/// Matrix representation (4×4):
/// ```text
/// CZ = |1 0 0  0|
///      |0 1 0  0|
///      |0 0 1  0|
///      |0 0 0 -1|
/// ```
pub fn cz<T>() -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let mut data = vec![Complex::new(T::zero(), T::zero()); 16];
    data[0] = Complex::new(T::one(), T::zero());
    data[5] = Complex::new(T::one(), T::zero());
    data[10] = Complex::new(T::one(), T::zero());
    data[15] = Complex::new(-T::one(), T::zero());

    Ok(Array::from_vec(data).reshape(&[4, 4]))
}

/// Controlled-Y gate
///
/// Matrix representation (4×4):
/// ```text
/// CY = |1 0  0   0|
///      |0 1  0   0|
///      |0 0  0  -i|
///      |0 0  i   0|
/// ```
pub fn cy<T>() -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let mut data = vec![Complex::new(T::zero(), T::zero()); 16];
    data[0] = Complex::new(T::one(), T::zero());
    data[5] = Complex::new(T::one(), T::zero());
    data[11] = Complex::new(T::zero(), -T::one());
    data[14] = Complex::new(T::zero(), T::one());

    Ok(Array::from_vec(data).reshape(&[4, 4]))
}

/// Toffoli gate (CCNOT, CCX)
///
/// Double-controlled NOT gate. Flips target qubit if both control qubits are |1⟩.
///
/// Matrix representation (8×8):
/// Only flips |110⟩ ↔ |111⟩, all other states unchanged
pub fn toffoli<T>() -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let mut data = vec![Complex::new(T::zero(), T::zero()); 64];

    // Identity for all states except |110⟩ and |111⟩
    for i in 0..6 {
        data[i * 8 + i] = Complex::new(T::one(), T::zero());
    }

    // Swap |110⟩ ↔ |111⟩
    data[6 * 8 + 7] = Complex::new(T::one(), T::zero());
    data[7 * 8 + 6] = Complex::new(T::one(), T::zero());

    Ok(Array::from_vec(data).reshape(&[8, 8]))
}

/// Fredkin gate (CSWAP)
///
/// Controlled-SWAP gate. Swaps two target qubits if control qubit is |1⟩.
///
/// Matrix representation (8×8):
/// Swaps |101⟩ ↔ |110⟩, all other states unchanged
pub fn fredkin<T>() -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let mut data = vec![Complex::new(T::zero(), T::zero()); 64];

    // Identity for states |000⟩ to |100⟩
    for i in 0..5 {
        data[i * 8 + i] = Complex::new(T::one(), T::zero());
    }

    // Swap |101⟩ ↔ |110⟩ (indices 5 and 6)
    data[5 * 8 + 6] = Complex::new(T::one(), T::zero());
    data[6 * 8 + 5] = Complex::new(T::one(), T::zero());

    // Identity for |111⟩
    data[7 * 8 + 7] = Complex::new(T::one(), T::zero());

    Ok(Array::from_vec(data).reshape(&[8, 8]))
}

/// Create a custom single-qubit gate from matrix elements
///
/// # Arguments
///
/// * `matrix` - 2×2 complex matrix (must be unitary)
///
/// # Returns
///
/// Custom single-qubit gate
pub fn custom_single_qubit_gate<T>(matrix: Array<Complex<T>>) -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let shape = matrix.shape();
    if shape.len() != 2 || shape[0] != 2 || shape[1] != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "Custom single-qubit gate must be 2×2".to_string(),
        ));
    }

    // Perform unitarity check if runtime validation is enabled
    if super::unitarity::is_runtime_validation_enabled() {
        super::unitarity::validate_gate_unitarity(
            &matrix,
            1e-10,
            Some("custom single-qubit gate"),
        )?;
    }

    Ok(matrix)
}

/// Create a controlled version of a single-qubit gate
///
/// # Arguments
///
/// * `gate` - Single-qubit gate (2×2 matrix)
///
/// # Returns
///
/// Controlled gate (4×4 matrix)
pub fn controlled_gate<T>(gate: &Array<Complex<T>>) -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let shape = gate.shape();
    if shape.len() != 2 || shape[0] != 2 || shape[1] != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "Input gate must be 2×2".to_string(),
        ));
    }

    let mut data = vec![Complex::new(T::zero(), T::zero()); 16];

    // Upper left 2×2 block is identity (control qubit = 0)
    data[0] = Complex::new(T::one(), T::zero());
    data[5] = Complex::new(T::one(), T::zero());

    // Lower right 2×2 block is the gate (control qubit = 1)
    for i in 0..2 {
        for j in 0..2 {
            data[(i + 2) * 4 + (j + 2)] = gate
                .get(&[i, j])
                .map_err(|_| NumRs2Error::IndexOutOfBounds("Invalid gate access".to_string()))?;
        }
    }

    Ok(Array::from_vec(data).reshape(&[4, 4]))
}

/// Identity gate for n qubits
///
/// # Arguments
///
/// * `num_qubits` - Number of qubits
///
/// # Returns
///
/// Identity matrix of dimension 2^n × 2^n
pub fn identity<T>(num_qubits: usize) -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let dim = 2_usize.pow(num_qubits as u32);
    let mut data = vec![Complex::new(T::zero(), T::zero()); dim * dim];

    for i in 0..dim {
        data[i * dim + i] = Complex::new(T::one(), T::zero());
    }

    Ok(Array::from_vec(data).reshape(&[dim, dim]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn is_unitary<T>(gate: &Array<Complex<T>>, epsilon: f64) -> bool
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let shape = gate.shape();
        let n = shape[0];

        // Compute U†U
        for i in 0..n {
            for j in 0..n {
                let mut sum = Complex::new(T::zero(), T::zero());
                for k in 0..n {
                    let u_ki = gate.get(&[k, i]).unwrap();
                    let u_kj = gate.get(&[k, j]).unwrap();
                    sum = sum + u_ki.conj() * u_kj;
                }

                let expected = if i == j { T::one() } else { T::zero() };
                let diff_re: f64 = (sum.re - expected).abs().into();
                let diff_im: f64 = sum.im.abs().into();

                if diff_re > epsilon || diff_im > epsilon {
                    return false;
                }
            }
        }

        true
    }

    #[test]
    fn test_pauli_gates_unitary() {
        let x = pauli_x::<f64>().unwrap();
        let y = pauli_y::<f64>().unwrap();
        let z = pauli_z::<f64>().unwrap();

        assert!(is_unitary(&x, 1e-10));
        assert!(is_unitary(&y, 1e-10));
        assert!(is_unitary(&z, 1e-10));
    }

    #[test]
    fn test_hadamard_unitary() {
        let h = hadamard::<f64>().unwrap();
        assert!(is_unitary(&h, 1e-10));
    }

    #[test]
    fn test_pauli_x_properties() {
        let x = pauli_x::<f64>().unwrap();

        // X² = I
        let mut x_squared = Complex::new(0.0, 0.0);
        for k in 0..2 {
            let xk0 = x.get(&[0, k]).unwrap();
            let x0k = x.get(&[k, 0]).unwrap();
            x_squared += xk0 * x0k;
        }
        assert_relative_eq!(x_squared.re, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_rotation_gates() {
        let theta = std::f64::consts::PI / 4.0;
        let rx_gate = rx(theta).unwrap();
        let ry_gate = ry(theta).unwrap();
        let rz_gate = rz(theta).unwrap();

        assert!(is_unitary(&rx_gate, 1e-10));
        assert!(is_unitary(&ry_gate, 1e-10));
        assert!(is_unitary(&rz_gate, 1e-10));
    }

    #[test]
    fn test_cnot_unitary() {
        let cnot_gate = cnot::<f64>().unwrap();
        assert!(is_unitary(&cnot_gate, 1e-10));
        assert_eq!(cnot_gate.shape(), &[4, 4]);
    }

    #[test]
    fn test_swap_unitary() {
        let swap_gate = swap::<f64>().unwrap();
        assert!(is_unitary(&swap_gate, 1e-10));
        assert_eq!(swap_gate.shape(), &[4, 4]);
    }

    #[test]
    fn test_cz_unitary() {
        let cz_gate = cz::<f64>().unwrap();
        assert!(is_unitary(&cz_gate, 1e-10));
    }

    #[test]
    fn test_toffoli_unitary() {
        let toffoli_gate = toffoli::<f64>().unwrap();
        assert!(is_unitary(&toffoli_gate, 1e-10));
        assert_eq!(toffoli_gate.shape(), &[8, 8]);
    }

    #[test]
    fn test_fredkin_unitary() {
        let fredkin_gate = fredkin::<f64>().unwrap();
        assert!(is_unitary(&fredkin_gate, 1e-10));
        assert_eq!(fredkin_gate.shape(), &[8, 8]);
    }

    #[test]
    fn test_phase_gate() {
        let s = phase_gate::<f64>().unwrap();
        assert!(is_unitary(&s, 1e-10));
    }

    #[test]
    fn test_t_gate() {
        let t = t_gate::<f64>().unwrap();
        assert!(is_unitary(&t, 1e-10));
    }

    #[test]
    fn test_controlled_gate() {
        let x = pauli_x::<f64>().unwrap();
        let cx = controlled_gate(&x).unwrap();

        assert!(is_unitary(&cx, 1e-10));
        assert_eq!(cx.shape(), &[4, 4]);
    }

    #[test]
    fn test_identity_gate() {
        let id2 = identity::<f64>(2).unwrap();
        assert!(is_unitary(&id2, 1e-10));
        assert_eq!(id2.shape(), &[4, 4]);
    }

    #[test]
    fn test_rotation_angle_zero() {
        let rx0 = rx(0.0).unwrap();
        let id = identity::<f64>(1).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                let rx_val = rx0.get(&[i, j]).unwrap();
                let id_val = id.get(&[i, j]).unwrap();
                assert_relative_eq!(rx_val.re, id_val.re, epsilon = 1e-10);
                assert_relative_eq!(rx_val.im, id_val.im, epsilon = 1e-10);
            }
        }
    }
}
