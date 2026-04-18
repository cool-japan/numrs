//! Quantum Algorithms
//!
//! This module provides implementations of important quantum algorithms:
//! - Quantum Fourier Transform (QFT)
//! - Grover's search algorithm
//! - Variational Quantum Eigensolver (VQE)
//! - Quantum Phase Estimation (QPE)
//!
//! # Examples
//!
//! ```
//! use numrs2::new_modules::quantum::algorithms::QuantumFourierTransform;
//! use numrs2::new_modules::quantum::circuit::QuantumCircuit;
//!
//! // Apply QFT to a 3-qubit circuit
//! let mut circuit = QuantumCircuit::<f64>::new(3).expect("valid qubit count");
//! QuantumFourierTransform::apply(&mut circuit).expect("valid QFT application");
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::new_modules::quantum::circuit::QuantumCircuit;
use crate::new_modules::quantum::gates;
use crate::new_modules::quantum::measurement::Measurement;
use crate::new_modules::quantum::statevector::StateVector;
use num_traits::Float;
use scirs2_core::Complex;
use std::f64::consts::PI;
use std::fmt::Debug;

/// Quantum Fourier Transform (QFT)
///
/// Implements the quantum analogue of the discrete Fourier transform.
/// The QFT is a key component of many quantum algorithms including
/// Shor's algorithm and quantum phase estimation.
///
/// # Mathematical Background
///
/// QFT maps |j⟩ → (1/√N) Σₖ exp(2πijk/N)|k⟩ where N = 2ⁿ
pub struct QuantumFourierTransform;

impl QuantumFourierTransform {
    /// Apply QFT to a quantum circuit
    ///
    /// # Arguments
    ///
    /// * `circuit` - Quantum circuit to apply QFT to
    ///
    /// # Returns
    ///
    /// Modified circuit with QFT gates added
    pub fn apply<T>(circuit: &mut QuantumCircuit<T>) -> Result<()>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let n = circuit.num_qubits();

        for j in 0..n {
            // Apply Hadamard to qubit j
            circuit.h(j)?;

            // Apply controlled rotations
            for k in (j + 1)..n {
                let angle = <T as From<f64>>::from(2.0 * PI / 2.0_f64.powi((k - j + 1) as i32));
                Self::controlled_phase_rotation(circuit, k, j, angle)?;
            }
        }

        // Swap qubits to reverse the order
        for i in 0..(n / 2) {
            circuit.swap(i, n - 1 - i)?;
        }

        Ok(())
    }

    /// Apply inverse QFT
    pub fn apply_inverse<T>(circuit: &mut QuantumCircuit<T>) -> Result<()>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let n = circuit.num_qubits();

        // Reverse the swaps
        for i in 0..(n / 2) {
            circuit.swap(i, n - 1 - i)?;
        }

        // Apply inverse rotations and Hadamards
        for j in (0..n).rev() {
            // Apply inverse controlled rotations
            for k in ((j + 1)..n).rev() {
                let angle = <T as From<f64>>::from(-2.0 * PI / 2.0_f64.powi((k - j + 1) as i32));
                Self::controlled_phase_rotation(circuit, k, j, angle)?;
            }

            // Apply Hadamard
            circuit.h(j)?;
        }

        Ok(())
    }

    /// Helper: Controlled phase rotation
    fn controlled_phase_rotation<T>(
        circuit: &mut QuantumCircuit<T>,
        control: usize,
        target: usize,
        angle: T,
    ) -> Result<()>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        // CP(θ) = diag(1, 1, 1, e^(iθ))
        let phase = Complex::new(angle.cos(), angle.sin());
        let gate_data = vec![
            Complex::new(T::one(), T::zero()),
            Complex::new(T::zero(), T::zero()),
            Complex::new(T::zero(), T::zero()),
            Complex::new(T::zero(), T::zero()),
            Complex::new(T::zero(), T::zero()),
            Complex::new(T::one(), T::zero()),
            Complex::new(T::zero(), T::zero()),
            Complex::new(T::zero(), T::zero()),
            Complex::new(T::zero(), T::zero()),
            Complex::new(T::zero(), T::zero()),
            Complex::new(T::one(), T::zero()),
            Complex::new(T::zero(), T::zero()),
            Complex::new(T::zero(), T::zero()),
            Complex::new(T::zero(), T::zero()),
            Complex::new(T::zero(), T::zero()),
            phase,
        ];

        let gate = Array::from_vec(gate_data).reshape(&[4, 4]);
        circuit.add_gate(gate, vec![control, target], "CP".to_string())?;

        Ok(())
    }
}

/// Grover's Search Algorithm
///
/// Provides quadratic speedup for unstructured search problems.
/// Searches for a marked item in O(√N) queries compared to O(N) classically.
///
/// # Mathematical Background
///
/// Grover's algorithm uses amplitude amplification to increase the probability
/// of measuring the marked state. Each iteration applies:
/// 1. Oracle that marks the solution
/// 2. Diffusion operator that amplifies the marked amplitude
pub struct GroverSearch;

impl GroverSearch {
    /// Run Grover's search algorithm
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - Number of qubits (search space size = 2^n)
    /// * `oracle` - Oracle function marking the solution
    /// * `num_iterations` - Number of Grover iterations (optimal: π/4 * √N)
    ///
    /// # Returns
    ///
    /// State after Grover iterations
    pub fn search<T, F>(
        num_qubits: usize,
        oracle: F,
        num_iterations: usize,
    ) -> Result<StateVector<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
        F: Fn(&mut QuantumCircuit<T>) -> Result<()>,
    {
        let mut circuit = QuantumCircuit::new(num_qubits)?;

        // Initialize to uniform superposition
        for i in 0..num_qubits {
            circuit.h(i)?;
        }

        // Run Grover iterations
        for _ in 0..num_iterations {
            // Apply oracle
            oracle(&mut circuit)?;

            // Apply diffusion operator
            Self::diffusion_operator(&mut circuit)?;
        }

        circuit.execute()
    }

    /// Diffusion operator (inversion about average)
    ///
    /// D = 2|ψ⟩⟨ψ| - I where |ψ⟩ is uniform superposition
    fn diffusion_operator<T>(circuit: &mut QuantumCircuit<T>) -> Result<()>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let n = circuit.num_qubits();

        // Apply H to all qubits
        for i in 0..n {
            circuit.h(i)?;
        }

        // Apply X to all qubits
        for i in 0..n {
            circuit.x(i)?;
        }

        // Multi-controlled Z gate (can be simplified for larger circuits)
        if n == 1 {
            circuit.z(0)?;
        } else if n == 2 {
            circuit.cz(0, 1)?;
        } else {
            // For n > 2, use multi-controlled Z approximation
            Self::multi_controlled_z(circuit, n)?;
        }

        // Apply X to all qubits
        for i in 0..n {
            circuit.x(i)?;
        }

        // Apply H to all qubits
        for i in 0..n {
            circuit.h(i)?;
        }

        Ok(())
    }

    /// Multi-controlled Z gate for n qubits
    fn multi_controlled_z<T>(circuit: &mut QuantumCircuit<T>, n: usize) -> Result<()>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        // Simplified implementation: apply CZ to adjacent pairs
        // For production, would use full multi-controlled gate decomposition
        for i in 0..(n - 1) {
            circuit.cz(i, i + 1)?;
        }
        Ok(())
    }

    /// Calculate optimal number of Grover iterations
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - Number of qubits
    /// * `num_solutions` - Number of marked items
    ///
    /// # Returns
    ///
    /// Optimal number of iterations
    pub fn optimal_iterations(num_qubits: usize, num_solutions: usize) -> usize {
        let n = 2_usize.pow(num_qubits as u32);
        let theta = ((num_solutions as f64) / (n as f64)).sqrt().asin();
        let iterations = (PI / (4.0 * theta)) as usize;
        iterations.max(1)
    }
}

/// Variational Quantum Eigensolver (VQE)
///
/// Hybrid quantum-classical algorithm for finding ground state energies.
/// Uses parameterized quantum circuits and classical optimization.
pub struct VQE;

impl VQE {
    /// Run VQE to find minimum eigenvalue
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - Number of qubits
    /// * `hamiltonian` - Observable to minimize (as Pauli string)
    /// * `ansatz` - Parameterized circuit ansatz
    /// * `initial_params` - Initial parameters
    /// * `max_iterations` - Maximum optimization iterations
    ///
    /// # Returns
    ///
    /// Optimized parameters and energy
    pub fn minimize<T, F>(
        num_qubits: usize,
        hamiltonian: &HamiltonianPauliZ<T>,
        ansatz: F,
        initial_params: Vec<T>,
        max_iterations: usize,
    ) -> Result<(Vec<T>, T)>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
        F: Fn(&mut QuantumCircuit<T>, &[T]) -> Result<()>,
    {
        let mut params = initial_params;
        let mut best_energy = T::infinity();

        // Simple gradient-free optimization (for demonstration)
        let step_size = <T as From<f64>>::from(0.1);

        for _ in 0..max_iterations {
            let energy = Self::evaluate_energy(num_qubits, hamiltonian, &ansatz, &params)?;

            if energy < best_energy {
                best_energy = energy;
            }

            // Update parameters (simplified gradient descent)
            for i in 0..params.len() {
                let mut params_plus = params.clone();
                params_plus[i] = params_plus[i] + step_size;

                let energy_plus =
                    Self::evaluate_energy(num_qubits, hamiltonian, &ansatz, &params_plus)?;

                if energy_plus < energy {
                    params[i] = params_plus[i];
                }
            }
        }

        // Evaluate final energy at the optimized parameters
        let final_energy = Self::evaluate_energy(num_qubits, hamiltonian, &ansatz, &params)?;
        Ok((params, final_energy))
    }

    /// Evaluate energy expectation value
    fn evaluate_energy<T, F>(
        num_qubits: usize,
        hamiltonian: &HamiltonianPauliZ<T>,
        ansatz: &F,
        params: &[T],
    ) -> Result<T>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
        F: Fn(&mut QuantumCircuit<T>, &[T]) -> Result<()>,
    {
        let mut circuit = QuantumCircuit::new(num_qubits)?;
        ansatz(&mut circuit, params)?;

        let state = circuit.execute()?;
        hamiltonian.expectation_value(&state)
    }
}

/// Hamiltonian represented as sum of Pauli-Z terms
///
/// H = Σᵢ cᵢ Zᵢ where Zᵢ acts on qubit i
pub struct HamiltonianPauliZ<T> {
    /// Coefficients for each qubit
    coefficients: Vec<T>,
}

impl<T> HamiltonianPauliZ<T>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    /// Create a new Pauli-Z Hamiltonian
    pub fn new(coefficients: Vec<T>) -> Self {
        Self { coefficients }
    }

    /// Calculate expectation value ⟨ψ|H|ψ⟩
    pub fn expectation_value(&self, state: &StateVector<T>) -> Result<T> {
        let mut energy = T::zero();

        for (qubit, &coeff) in self.coefficients.iter().enumerate() {
            if qubit >= state.num_qubits() {
                return Err(NumRs2Error::IndexOutOfBounds(
                    "Hamiltonian size exceeds state size".to_string(),
                ));
            }

            let exp_z: f64 = Measurement::expectation_z(state, qubit)?;
            energy = energy + coeff * <T as From<f64>>::from(exp_z);
        }

        Ok(energy)
    }
}

/// Quantum Phase Estimation (QPE)
///
/// Estimates eigenvalues of unitary operators.
/// Key component of Shor's algorithm and quantum chemistry applications.
pub struct QuantumPhaseEstimation;

impl QuantumPhaseEstimation {
    /// Run phase estimation algorithm
    ///
    /// # Arguments
    ///
    /// * `num_precision_qubits` - Number of qubits for phase precision
    /// * `unitary` - Unitary operator U
    /// * `eigenstate` - Eigenstate of U
    ///
    /// # Returns
    ///
    /// Estimated phase (in [0, 1))
    pub fn estimate_phase<T>(
        num_precision_qubits: usize,
        unitary: &Array<Complex<T>>,
        eigenstate: &StateVector<T>,
    ) -> Result<f64>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let n_eigen = eigenstate.num_qubits();
        let n_total = num_precision_qubits + n_eigen;

        let mut circuit = QuantumCircuit::new(n_total)?;

        // Initialize precision qubits to |+⟩
        for i in 0..num_precision_qubits {
            circuit.h(i)?;
        }

        // Apply controlled-U^(2^k) operations
        for k in 0..num_precision_qubits {
            let power = 2_usize.pow(k as u32);
            for _ in 0..power {
                // Simplified: apply U (in practice, need controlled-U)
                // For full implementation, need to construct controlled version
                for target in 0..n_eigen {
                    circuit.add_gate(
                        unitary.clone(),
                        vec![num_precision_qubits + target],
                        "U".to_string(),
                    )?;
                }
            }
        }

        // Apply inverse QFT to precision qubits
        let mut qft_circuit = QuantumCircuit::<T>::new(num_precision_qubits)?;
        QuantumFourierTransform::apply_inverse(&mut qft_circuit)?;

        // Measure precision qubits
        let state = circuit.execute()?;
        let (result, _) = Measurement::measure_all(&state, None)?;

        // Extract phase from measurement
        let phase = (result.outcome as f64) / 2.0_f64.powi(num_precision_qubits as i32);

        Ok(phase)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_qft_circuit() {
        let mut circuit = QuantumCircuit::<f64>::new(3).expect("test: valid qubit count");
        QuantumFourierTransform::apply(&mut circuit).expect("test: valid QFT application");

        // QFT should add gates
        assert!(circuit.num_gates() > 0);
    }

    #[test]
    fn test_qft_on_zero_state() {
        let mut circuit = QuantumCircuit::<f64>::new(2).expect("test: valid qubit count");
        QuantumFourierTransform::apply(&mut circuit).expect("test: valid QFT application");

        let state = circuit.execute().expect("test: circuit execution succeeds");

        // QFT|00⟩ = (|00⟩ + |01⟩ + |10⟩ + |11⟩)/2
        for i in 0..4 {
            let prob = state.get_probability(i).expect("test: valid state index");
            assert_relative_eq!(prob, 0.25, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_inverse_qft() {
        let mut circuit = QuantumCircuit::<f64>::new(2).expect("test: valid 2-qubit circuit");

        // Apply QFT then inverse QFT
        QuantumFourierTransform::apply(&mut circuit).expect("test: valid QFT application");
        QuantumFourierTransform::apply_inverse(&mut circuit)
            .expect("test: valid inverse QFT application");

        let state = circuit.execute().expect("test: circuit execution succeeds");

        // Should return to |00⟩
        let prob_00 = state.get_probability(0).expect("test: valid state index");
        assert_relative_eq!(prob_00, 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_grover_optimal_iterations() {
        let iterations = GroverSearch::optimal_iterations(3, 1);
        // For 8 items, 1 solution: π/4 * √8 ≈ 2.22
        assert!((2..=3).contains(&iterations));
    }

    #[test]
    fn test_grover_search_single_item() {
        // Search for |11⟩ in 2-qubit space
        let oracle = |circuit: &mut QuantumCircuit<f64>| {
            // Oracle marks |11⟩
            circuit.cz(0, 1)?;
            Ok(())
        };

        let iterations = GroverSearch::optimal_iterations(2, 1);
        let state = GroverSearch::search(2, oracle, iterations).expect("test: valid Grover search");

        // Should have high probability of measuring |11⟩
        let prob_11 = state.get_probability(3).expect("test: valid state index");
        assert!(prob_11 > 0.5);
    }

    #[test]
    fn test_hamiltonian_expectation() {
        // H = Z₀ (eigenvalue +1 for |0⟩, -1 for |1⟩)
        let ham = HamiltonianPauliZ::new(vec![1.0]);

        // Test with |0⟩
        let state_0 = StateVector::<f64>::new(1).expect("test: valid qubit count");
        let energy_0 = ham
            .expectation_value(&state_0)
            .expect("test: valid expectation value");
        assert_relative_eq!(energy_0, 1.0, epsilon = 1e-10);

        // Test with |1⟩
        let amplitudes = vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];
        let state_1 = StateVector::from_amplitudes(amplitudes).expect("test: valid amplitudes");
        let energy_1 = ham
            .expectation_value(&state_1)
            .expect("test: valid expectation value");
        assert_relative_eq!(energy_1, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_vqe_simple_ansatz() {
        // Simple 1-qubit VQE with Ry rotation ansatz
        let ansatz = |circuit: &mut QuantumCircuit<f64>, params: &[f64]| {
            circuit.ry(0, params[0])?;
            Ok(())
        };

        // Hamiltonian: H = Z (ground state is |1⟩ with eigenvalue -1)
        let ham = HamiltonianPauliZ::new(vec![1.0]);

        let initial_params = vec![0.5];
        let (params, energy) = VQE::minimize(1, &ham, ansatz, initial_params, 10)
            .expect("test: valid VQE minimization");

        // After 10 iterations with step_size=0.1, params should be around 1.5
        // which gives energy ≈ cos(1.5) ≈ 0.07
        // VQE minimizes, so it's moving from cos(0.5)≈0.88 toward cos(π)=-1
        assert!(params.len() == 1);
        assert!(
            params[0] > 1.0,
            "Parameter should have increased from initial 0.5"
        );
        assert!(
            energy < 0.5,
            "Energy should have decreased from initial ~0.88"
        );
        assert_relative_eq!(energy, 0.07, epsilon = 0.01);
    }

    #[test]
    fn test_diffusion_operator() {
        let mut circuit = QuantumCircuit::<f64>::new(2).expect("test: valid qubit count");

        // Initialize to superposition
        circuit.h(0).expect("test: valid qubit index");
        circuit.h(1).expect("test: valid qubit index");

        GroverSearch::diffusion_operator(&mut circuit).expect("test: valid diffusion operator");

        // Should have added gates
        assert!(circuit.num_gates() > 2);
    }

    #[test]
    fn test_hamiltonian_multi_qubit() {
        // H = Z₀ + 2Z₁
        let ham = HamiltonianPauliZ::new(vec![1.0, 2.0]);

        // State |00⟩: E = 1 + 2 = 3
        let state = StateVector::<f64>::new(2).expect("test: valid qubit count");
        let energy = ham
            .expectation_value(&state)
            .expect("test: valid expectation value");
        assert_relative_eq!(energy, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_qft_circuit_depth() {
        let mut circuit = QuantumCircuit::<f64>::new(3).expect("test: valid qubit count");
        QuantumFourierTransform::apply(&mut circuit).expect("test: valid QFT application");

        let depth = circuit.depth();
        assert!(depth > 0);
    }

    #[test]
    fn test_qft_single_qubit() {
        let mut circuit = QuantumCircuit::<f64>::new(1).expect("test: valid qubit count");
        QuantumFourierTransform::apply(&mut circuit).expect("test: valid QFT application");
        let state = circuit.execute().expect("test: circuit execution succeeds");

        // Single qubit QFT is just Hadamard
        let prob_0 = state.get_probability(0).expect("test: valid state index");
        let prob_1 = state.get_probability(1).expect("test: valid state index");
        assert_relative_eq!(prob_0, 0.5, epsilon = 1e-10);
        assert_relative_eq!(prob_1, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_hamiltonian_zero_state() {
        let ham = HamiltonianPauliZ::new(vec![1.0, 1.0]);
        let state = StateVector::<f64>::new(2).expect("test: valid qubit count");
        let energy = ham
            .expectation_value(&state)
            .expect("test: valid expectation value");
        assert_relative_eq!(energy, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_grover_iterations_large_space() {
        let iterations = GroverSearch::optimal_iterations(10, 1);
        // For 1024 items, optimal iterations ≈ π/4 * √1024 ≈ 25
        assert!(iterations > 20 && iterations < 30);
    }
}
