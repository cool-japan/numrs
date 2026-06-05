//! Quantum State Vector Simulation
//!
//! This module provides quantum state vector representation and operations.
//! State vectors represent pure quantum states using complex probability amplitudes.
//!
//! # Mathematical Background
//!
//! A quantum state of n qubits is represented by a normalized complex vector in 2^n dimensional
//! Hilbert space: |ψ⟩ = Σ αᵢ|i⟩, where Σ|αᵢ|² = 1
//!
//! # Examples
//!
//! ```
//! use numrs2::new_modules::quantum::statevector::StateVector;
//!
//! // Create a 2-qubit state |00⟩
//! let state = StateVector::<f64>::new(2).expect("valid qubit count");
//! assert_eq!(state.num_qubits(), 2);
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use scirs2_core::Complex;
use std::fmt::Debug;

/// Quantum state vector representing pure quantum states
///
/// The state vector contains 2^n complex amplitudes for n qubits,
/// where each amplitude represents the probability amplitude of
/// measuring the corresponding computational basis state.
#[derive(Clone, Debug)]
pub struct StateVector<T: Clone> {
    /// Complex amplitudes of the quantum state
    amplitudes: Array<Complex<T>>,
    /// Number of qubits in the state
    num_qubits: usize,
}

impl<T> StateVector<T>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    /// Create a new quantum state initialized to |0...0⟩
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - Number of qubits in the state
    ///
    /// # Returns
    ///
    /// A new state vector initialized to the computational basis state |0...0⟩
    ///
    /// # Examples
    ///
    /// ```
    /// use numrs2::new_modules::quantum::statevector::StateVector;
    ///
    /// let state = StateVector::<f64>::new(3).expect("valid qubit count");
    /// assert_eq!(state.num_qubits(), 3);
    /// ```
    pub fn new(num_qubits: usize) -> Result<Self> {
        if num_qubits == 0 {
            return Err(NumRs2Error::InvalidOperation(
                "Number of qubits must be positive".to_string(),
            ));
        }

        let dim = 2_usize.pow(num_qubits as u32);
        let mut amplitudes_vec = vec![Complex::new(T::zero(), T::zero()); dim];
        amplitudes_vec[0] = Complex::new(T::one(), T::zero()); // |0...0⟩ state

        Ok(Self {
            amplitudes: Array::from_vec(amplitudes_vec),
            num_qubits,
        })
    }

    /// Create a state vector from a vector of complex amplitudes
    ///
    /// # Arguments
    ///
    /// * `amplitudes` - Vector of complex amplitudes
    ///
    /// # Returns
    ///
    /// A new state vector with the given amplitudes (normalized)
    pub fn from_amplitudes(amplitudes: Vec<Complex<T>>) -> Result<Self> {
        let dim = amplitudes.len();

        // Check if dimension is a power of 2
        if dim == 0 || (dim & (dim - 1)) != 0 {
            return Err(NumRs2Error::InvalidOperation(
                "Amplitude vector length must be a power of 2".to_string(),
            ));
        }

        let num_qubits = (dim as f64).log2() as usize;
        let mut state = Self {
            amplitudes: Array::from_vec(amplitudes),
            num_qubits,
        };

        // Normalize the state
        state.normalize()?;
        Ok(state)
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the dimension of the state vector (2^n)
    pub fn dim(&self) -> usize {
        2_usize.pow(self.num_qubits as u32)
    }

    /// Get the amplitudes as a reference
    pub fn amplitudes(&self) -> &Array<Complex<T>> {
        &self.amplitudes
    }

    /// Get a mutable reference to amplitudes
    pub fn amplitudes_mut(&mut self) -> &mut Array<Complex<T>> {
        &mut self.amplitudes
    }

    /// Normalize the state vector
    ///
    /// Ensures Σ|αᵢ|² = 1
    pub fn normalize(&mut self) -> Result<()> {
        let norm_squared = self.probability_norm_squared();

        if norm_squared <= T::zero() {
            return Err(NumRs2Error::InvalidOperation(
                "Cannot normalize zero state".to_string(),
            ));
        }

        let norm = norm_squared.sqrt();
        let amps = self.amplitudes.to_vec();
        let normalized: Vec<Complex<T>> = amps
            .iter()
            .map(|&a| a / Complex::new(norm, T::zero()))
            .collect();

        self.amplitudes = Array::from_vec(normalized);
        Ok(())
    }

    /// Calculate the squared norm of probabilities Σ|αᵢ|²
    pub fn probability_norm_squared(&self) -> T {
        let amps = self.amplitudes.to_vec();
        amps.iter()
            .map(|a| {
                let re: f64 = a.re.into();
                let im: f64 = a.im.into();
                <T as From<f64>>::from(re * re + im * im)
            })
            .fold(T::zero(), |acc, x| acc + x)
    }

    /// Get the probability of measuring a specific computational basis state
    ///
    /// # Arguments
    ///
    /// * `state` - The basis state index (0 to 2^n - 1)
    ///
    /// # Returns
    ///
    /// Probability |α\[state\]|²
    pub fn get_probability(&self, state: usize) -> Result<T> {
        if state >= self.dim() {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "State index {} out of bounds for {} qubits",
                state, self.num_qubits
            )));
        }

        let amp = &self.amplitudes.to_vec()[state];
        let re: f64 = amp.re.into();
        let im: f64 = amp.im.into();
        Ok(<T as From<f64>>::from(re * re + im * im))
    }

    /// Get all probabilities
    ///
    /// # Returns
    ///
    /// Array of probabilities for all basis states
    pub fn get_probabilities(&self) -> Array<T> {
        let amps = self.amplitudes.to_vec();
        let probs: Vec<T> = amps
            .iter()
            .map(|a| {
                let re: f64 = a.re.into();
                let im: f64 = a.im.into();
                <T as From<f64>>::from(re * re + im * im)
            })
            .collect();
        Array::from_vec(probs)
    }

    /// Compute the partial trace over specified qubits
    ///
    /// Returns the reduced density matrix by tracing out the specified qubits.
    ///
    /// # Arguments
    ///
    /// * `traced_qubits` - Indices of qubits to trace out
    ///
    /// # Returns
    ///
    /// Density matrix of the remaining system
    pub fn partial_trace(&self, traced_qubits: &[usize]) -> Result<DensityMatrix<T>> {
        for &qubit in traced_qubits {
            if qubit >= self.num_qubits {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "Qubit index {} out of bounds for {} qubits",
                    qubit, self.num_qubits
                )));
            }
        }

        // For a pure state, first convert to density matrix then trace
        let rho = self.to_density_matrix()?;
        rho.partial_trace(traced_qubits)
    }

    /// Convert state vector to density matrix representation
    ///
    /// ρ = |ψ⟩⟨ψ|
    pub fn to_density_matrix(&self) -> Result<DensityMatrix<T>> {
        DensityMatrix::from_state_vector(self)
    }

    /// Apply a unitary gate to the state
    ///
    /// # Arguments
    ///
    /// * `unitary` - Unitary matrix (2^k × 2^k for k-qubit gate)
    /// * `target_qubits` - Qubits the gate acts on
    pub fn apply_gate(
        &mut self,
        unitary: &Array<Complex<T>>,
        target_qubits: &[usize],
    ) -> Result<()> {
        let shape = unitary.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(NumRs2Error::DimensionMismatch(
                "Unitary must be a square matrix".to_string(),
            ));
        }

        let gate_size = shape[0];
        let num_gate_qubits = (gate_size as f64).log2() as usize;

        if gate_size != 2_usize.pow(num_gate_qubits as u32) {
            return Err(NumRs2Error::InvalidOperation(
                "Gate dimension must be a power of 2".to_string(),
            ));
        }

        if target_qubits.len() != num_gate_qubits {
            return Err(NumRs2Error::InvalidOperation(
                "Number of target qubits must match gate dimension".to_string(),
            ));
        }

        for &qubit in target_qubits {
            if qubit >= self.num_qubits {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "Qubit index {} out of bounds",
                    qubit
                )));
            }
        }

        // Apply gate by computing new amplitudes
        let old_amps = self.amplitudes.to_vec();
        let mut new_amps = vec![Complex::new(T::zero(), T::zero()); self.dim()];

        for i in 0..self.dim() {
            for j in 0..gate_size {
                // Check if state i has the right bits for target qubits
                let mut target_state = 0;

                for (k, &qubit) in target_qubits.iter().enumerate() {
                    let bit = (i >> qubit) & 1;
                    target_state |= bit << k;
                }

                if target_state == j {
                    // This row of the unitary contributes to this amplitude
                    for k in 0..gate_size {
                        // Find the corresponding initial state
                        let mut init_state = i;
                        for (m, &qubit) in target_qubits.iter().enumerate() {
                            let old_bit = (i >> qubit) & 1;
                            let new_bit = (k >> m) & 1;
                            if old_bit != new_bit {
                                init_state ^= 1 << qubit;
                            }
                        }

                        let u_elem = unitary.get(&[j, k]).map_err(|_| {
                            NumRs2Error::IndexOutOfBounds("Invalid unitary access".to_string())
                        })?;
                        new_amps[i] = new_amps[i] + u_elem * old_amps[init_state];
                    }
                }
            }
        }

        self.amplitudes = Array::from_vec(new_amps);
        Ok(())
    }
}

/// Density matrix representation of quantum states
///
/// Represents both pure and mixed quantum states using density matrix formalism.
/// For a pure state: ρ = |ψ⟩⟨ψ|
/// For a mixed state: ρ = Σᵢ pᵢ|ψᵢ⟩⟨ψᵢ|
#[derive(Clone, Debug)]
pub struct DensityMatrix<T: Clone> {
    /// Density matrix elements
    matrix: Array<Complex<T>>,
    /// Number of qubits
    num_qubits: usize,
}

impl<T> DensityMatrix<T>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    /// Create density matrix from a pure state vector
    ///
    /// ρ = |ψ⟩⟨ψ|
    pub fn from_state_vector(state: &StateVector<T>) -> Result<Self> {
        let dim = state.dim();
        let amps = state.amplitudes().to_vec();

        let mut matrix_data = vec![Complex::new(T::zero(), T::zero()); dim * dim];

        for i in 0..dim {
            for j in 0..dim {
                matrix_data[i * dim + j] = amps[i] * amps[j].conj();
            }
        }

        Ok(Self {
            matrix: Array::from_vec(matrix_data).reshape(&[dim, dim]),
            num_qubits: state.num_qubits(),
        })
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the density matrix
    pub fn matrix(&self) -> &Array<Complex<T>> {
        &self.matrix
    }

    /// Compute the trace of the density matrix
    ///
    /// For a valid density matrix, Tr(ρ) = 1
    pub fn trace(&self) -> Complex<T> {
        let dim = 2_usize.pow(self.num_qubits as u32);
        let mut tr = Complex::new(T::zero(), T::zero());

        for i in 0..dim {
            if let Ok(val) = self.matrix.get(&[i, i]) {
                tr = tr + val;
            }
        }

        tr
    }

    /// Compute partial trace over specified qubits
    ///
    /// # Arguments
    ///
    /// * `traced_qubits` - Indices of qubits to trace out
    ///
    /// # Returns
    ///
    /// Reduced density matrix
    pub fn partial_trace(&self, traced_qubits: &[usize]) -> Result<Self> {
        for &qubit in traced_qubits {
            if qubit >= self.num_qubits {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "Qubit index {} out of bounds",
                    qubit
                )));
            }
        }

        // Build sorted lists for remaining and traced qubit positions.
        // Qubit convention: qubit q corresponds to bit q of the state index (LSB = qubit 0),
        // consistent with the gate application convention: `(idx >> q) & 1`.
        let mut traced_sorted: Vec<usize> = traced_qubits.to_vec();
        traced_sorted.sort_unstable();
        traced_sorted.dedup();

        let remaining_list: Vec<usize> = (0..self.num_qubits)
            .filter(|q| !traced_sorted.contains(q))
            .collect();

        let num_remaining = remaining_list.len();
        let new_dim = 2_usize.pow(num_remaining as u32);
        let traced_dim = 2_usize.pow(traced_sorted.len() as u32);
        let old_dim = 2_usize.pow(self.num_qubits as u32);

        // Helper closure: reconstruct a full state index from a reduced-system index
        // and a traced-qubit index.  Both `rem_idx` and `traced_idx` are compact
        // indices whose bits correspond to the qubits listed in `remaining_list` and
        // `traced_sorted` respectively (bit 0 of the compact index ↔ first qubit in
        // the sorted list).
        let reconstruct_full = |rem_idx: usize, traced_idx: usize| -> usize {
            let mut full_idx = 0usize;
            // Insert bits for remaining qubits
            for (rank, &qubit_pos) in remaining_list.iter().enumerate() {
                let bit = (rem_idx >> rank) & 1;
                full_idx |= bit << qubit_pos;
            }
            // Insert bits for traced qubits
            for (rank, &qubit_pos) in traced_sorted.iter().enumerate() {
                let bit = (traced_idx >> rank) & 1;
                full_idx |= bit << qubit_pos;
            }
            full_idx
        };

        let mut reduced = vec![Complex::new(T::zero(), T::zero()); new_dim * new_dim];

        for i in 0..new_dim {
            for j in 0..new_dim {
                let mut sum = Complex::new(T::zero(), T::zero());

                // Sum ρ[full_i, full_j] over all traced-qubit basis states k
                for k in 0..traced_dim {
                    let full_i = reconstruct_full(i, k);
                    let full_j = reconstruct_full(j, k);

                    if full_i < old_dim && full_j < old_dim {
                        if let Ok(val) = self.matrix.get(&[full_i, full_j]) {
                            sum = sum + val;
                        }
                    }
                }

                reduced[i * new_dim + j] = sum;
            }
        }

        Ok(Self {
            matrix: Array::from_vec(reduced).reshape(&[new_dim, new_dim]),
            num_qubits: num_remaining,
        })
    }

    /// Calculate the purity of the state
    ///
    /// Purity = Tr(ρ²)
    /// - Pure states: purity = 1
    /// - Maximally mixed states: purity = 1/d
    pub fn purity(&self) -> T {
        let dim = 2_usize.pow(self.num_qubits as u32);
        let mut sum = T::zero();

        // Compute Tr(ρ²)
        for i in 0..dim {
            for k in 0..dim {
                if let (Ok(rho_ik), Ok(rho_ki)) =
                    (self.matrix.get(&[i, k]), self.matrix.get(&[k, i]))
                {
                    let product = rho_ik * rho_ki;
                    let re: f64 = product.re.into();
                    sum = sum + <T as From<f64>>::from(re);
                }
            }
        }

        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_statevector_creation() {
        let state = StateVector::<f64>::new(2).expect("test: valid qubit count");
        assert_eq!(state.num_qubits(), 2);
        assert_eq!(state.dim(), 4);

        // Should be initialized to |00⟩
        let prob = state.get_probability(0).expect("test: valid state index");
        assert_relative_eq!(prob, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_statevector_normalization() {
        let amplitudes = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        let state = StateVector::from_amplitudes(amplitudes).expect("test: valid amplitudes");

        let norm_sq = state.probability_norm_squared();
        assert_relative_eq!(norm_sq, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_probabilities() {
        let amplitudes = vec![
            Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        let state = StateVector::from_amplitudes(amplitudes).expect("test: valid amplitudes");

        let prob0 = state.get_probability(0).expect("test: valid state index");
        let prob1 = state.get_probability(1).expect("test: valid state index");

        assert_relative_eq!(prob0, 0.5, epsilon = 1e-10);
        assert_relative_eq!(prob1, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_density_matrix_from_pure_state() {
        let state = StateVector::<f64>::new(1).expect("test: valid qubit count");
        let rho = state
            .to_density_matrix()
            .expect("test: valid density matrix conversion");

        // Check trace = 1
        let tr = rho.trace();
        assert_relative_eq!(tr.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(tr.im, 0.0, epsilon = 1e-10);

        // Check purity = 1 for pure state
        let purity = rho.purity();
        assert_relative_eq!(purity, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_invalid_qubit_count() {
        let result = StateVector::<f64>::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_amplitude_dimension() {
        let amplitudes = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let result = StateVector::from_amplitudes(amplitudes);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_probabilities() {
        let amplitudes = vec![
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
            Complex::new(0.5, 0.0),
        ];
        let state = StateVector::from_amplitudes(amplitudes).expect("test: valid amplitudes");
        let probs = state.get_probabilities();

        assert_eq!(probs.shape()[0], 4);
        for i in 0..4 {
            let p = probs.get(&[i]).expect("test: valid probability index");
            assert_relative_eq!(p, 0.25, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_complex_amplitudes() {
        // Test with complex amplitudes
        let amplitudes = vec![Complex::new(0.5, 0.5), Complex::new(0.5, -0.5)];
        let state = StateVector::from_amplitudes(amplitudes).expect("test: valid amplitudes");
        let norm = state.probability_norm_squared();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_three_qubit_state() {
        let state = StateVector::<f64>::new(3).expect("test: valid qubit count");
        assert_eq!(state.num_qubits(), 3);
        assert_eq!(state.dim(), 8);
    }

    #[test]
    fn test_large_qubit_state() {
        let state = StateVector::<f64>::new(4).expect("test: valid qubit count");
        assert_eq!(state.dim(), 16);
    }

    #[test]
    fn test_probability_sum() {
        let state = StateVector::<f64>::new(2).expect("test: valid qubit count");
        let probs = state.get_probabilities();
        let sum: f64 = probs.to_vec().iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);
    }

    // Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2.
    // Qubit convention: qubit 0 = LSB, qubit 1 = MSB.
    //   |00⟩ → index 0  (bit1=0, bit0=0)
    //   |01⟩ → index 1  (bit1=0, bit0=1)
    //   |10⟩ → index 2  (bit1=1, bit0=0)
    //   |11⟩ → index 3  (bit1=1, bit0=1)
    // So Bell amplitudes: amp[0] = 1/√2, amp[3] = 1/√2, rest zero.
    //
    // Partial trace over qubit 0 should yield the 2×2 maximally mixed state I/2.
    // Purity of I/2 is Tr((I/2)²) = 1/2.
    #[test]
    fn test_partial_trace_bell_state() {
        let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();
        // Index 0 = |00⟩, index 3 = |11⟩
        let amplitudes = vec![
            Complex::new(inv_sqrt2, 0.0), // |00⟩
            Complex::new(0.0, 0.0),       // |01⟩
            Complex::new(0.0, 0.0),       // |10⟩
            Complex::new(inv_sqrt2, 0.0), // |11⟩
        ];
        let state =
            StateVector::from_amplitudes(amplitudes).expect("test: valid Bell state amplitudes");
        assert_eq!(state.num_qubits(), 2);

        // Trace out qubit 0 (LSB) → reduced system is qubit 1
        let rho_full = state
            .to_density_matrix()
            .expect("test: density matrix from Bell state");

        let rho_red = rho_full
            .partial_trace(&[0])
            .expect("test: partial trace over qubit 0");

        assert_eq!(rho_red.num_qubits(), 1);

        // Should be I/2: diagonal entries = 0.5, off-diagonal = 0
        let m00 = rho_red.matrix().get(&[0, 0]).expect("test: rho_red[0,0]");
        let m01 = rho_red.matrix().get(&[0, 1]).expect("test: rho_red[0,1]");
        let m10 = rho_red.matrix().get(&[1, 0]).expect("test: rho_red[1,0]");
        let m11 = rho_red.matrix().get(&[1, 1]).expect("test: rho_red[1,1]");

        assert_relative_eq!(m00.re, 0.5, epsilon = 1e-10);
        assert_relative_eq!(m00.im, 0.0, epsilon = 1e-10);
        assert_relative_eq!(m01.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(m01.im, 0.0, epsilon = 1e-10);
        assert_relative_eq!(m10.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(m10.im, 0.0, epsilon = 1e-10);
        assert_relative_eq!(m11.re, 0.5, epsilon = 1e-10);
        assert_relative_eq!(m11.im, 0.0, epsilon = 1e-10);

        // Purity = Tr(ρ²) = 0.5 for maximally mixed single-qubit state
        let purity = rho_red.purity();
        assert_relative_eq!(purity, 0.5, epsilon = 1e-10);
    }

    // Product state |ψ⟩ ⊗ |φ⟩ where |ψ⟩ = (|0⟩ + |1⟩)/√2 (qubit 1, MSB)
    // and |φ⟩ = |0⟩ (qubit 0, LSB).
    // Amplitude vector (qubit1 ⊗ qubit0 basis):
    //   amp[0] = 1/√2 (|00⟩), amp[1] = 0 (|01⟩), amp[2] = 1/√2 (|10⟩), amp[3] = 0 (|11⟩)
    //
    // Trace out qubit 0 → remaining qubit 1 should be |ψ⟩ = (|0⟩+|1⟩)/√2, purity = 1.
    #[test]
    fn test_partial_trace_product_state() {
        let inv_sqrt2 = 1.0_f64 / 2.0_f64.sqrt();
        // qubit 1 = (|0⟩+|1⟩)/√2, qubit 0 = |0⟩
        // combined: amp[0]=1/√2, amp[1]=0, amp[2]=1/√2, amp[3]=0
        let amplitudes = vec![
            Complex::new(inv_sqrt2, 0.0), // |00⟩
            Complex::new(0.0, 0.0),       // |01⟩
            Complex::new(inv_sqrt2, 0.0), // |10⟩
            Complex::new(0.0, 0.0),       // |11⟩
        ];
        let state =
            StateVector::from_amplitudes(amplitudes).expect("test: valid product state amplitudes");

        let rho_full = state
            .to_density_matrix()
            .expect("test: density matrix from product state");

        // Trace out qubit 0 (|0⟩ part) → should get pure state of qubit 1
        let rho_red = rho_full
            .partial_trace(&[0])
            .expect("test: partial trace over qubit 0");

        assert_eq!(rho_red.num_qubits(), 1);

        // Expected: ρ_1 = |+⟩⟨+| = [[0.5, 0.5],[0.5, 0.5]]
        let m00 = rho_red.matrix().get(&[0, 0]).expect("test: rho_red[0,0]");
        let m11 = rho_red.matrix().get(&[1, 1]).expect("test: rho_red[1,1]");
        let m01 = rho_red.matrix().get(&[0, 1]).expect("test: rho_red[0,1]");

        assert_relative_eq!(m00.re, 0.5, epsilon = 1e-10);
        assert_relative_eq!(m11.re, 0.5, epsilon = 1e-10);
        assert_relative_eq!(m01.re, 0.5, epsilon = 1e-10);
        assert_relative_eq!(m01.im, 0.0, epsilon = 1e-10);

        // Pure state: purity = 1
        let purity = rho_red.purity();
        assert_relative_eq!(purity, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_density_matrix_hermitian() {
        let state = StateVector::<f64>::new(1).expect("test: valid qubit count");
        let rho = state
            .to_density_matrix()
            .expect("test: valid density matrix conversion");

        // Check if Hermitian: rho[i,j] = conj(rho[j,i])
        for i in 0..2 {
            for j in 0..2 {
                let rho_ij = rho.matrix().get(&[i, j]).expect("test: valid matrix index");
                let rho_ji = rho.matrix().get(&[j, i]).expect("test: valid matrix index");
                assert_relative_eq!(rho_ij.re, rho_ji.re, epsilon = 1e-10);
                assert_relative_eq!(rho_ij.im, -rho_ji.im, epsilon = 1e-10);
            }
        }
    }
}
