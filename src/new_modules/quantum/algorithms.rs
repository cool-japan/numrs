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

        let gate = Array::from_vec_shape(gate_data, &[4, 4])?;
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

        // Multi-controlled Z gate: flips the phase of the all-ones state only.
        if n == 1 {
            circuit.z(0)?;
        } else if n == 2 {
            circuit.cz(0, 1)?;
        } else {
            // For n > 2, use the exact ancilla-free multi-controlled-Z.
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

    /// Multi-controlled Z gate over all `n` qubits.
    ///
    /// Flips the global phase of the all-ones basis state `|11...1⟩` and leaves
    /// every other computational basis state unchanged, implementing the diagonal
    /// unitary `diag(1, …, 1, -1)` of dimension `2^n`.
    ///
    /// The construction uses the identity
    /// `MCZ(c_0,…,c_{n-1}) = H(t) · MCX(c_0,…,c_{n-2}; t) · H(t)` with `t = n-1`,
    /// where `MCX` is a multi-controlled-X (generalized Toffoli). The `MCX` is
    /// realised with the textbook ancilla-free recursive √X decomposition of
    /// Barenco et al. (1995), so no scratch qubits are required — important here
    /// because Grover's diffusion operator has no ancillas available.
    fn multi_controlled_z<T>(circuit: &mut QuantumCircuit<T>, n: usize) -> Result<()>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        if n == 0 {
            return Ok(());
        }
        if n == 1 {
            // Single-qubit MCZ is just Z.
            return circuit.z(0).map(|_| ());
        }

        let target = n - 1;
        let controls: Vec<usize> = (0..target).collect();

        // MCZ = H · MCX · H on the chosen target qubit.
        circuit.h(target)?;
        multi_controlled_x(circuit, &controls, target)?;
        circuit.h(target)?;

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

/// Build the 2×2 matrix of `X^p`, the real power `p` of the Pauli-X gate.
///
/// `X^p = e^{iπp/2} · (cos(πp/2) · I − i·sin(πp/2) · X)`, which evaluates to
/// ```text
/// X^p = e^{iπp/2} · | cos(πp/2)      -i·sin(πp/2) |
///                   | -i·sin(πp/2)    cos(πp/2)   |
/// ```
/// For `p = 1` this is exactly the Pauli-X gate, for `p = 0` the identity, and
/// `X^{p/2} · X^{p/2} = X^p`, so `X^{p}` and `X^{p/2}` form the square-root pairs
/// required by the recursive multi-controlled-X decomposition. The leading phase
/// `e^{iπp/2}` is the principal branch and makes `X^p` a genuine unitary (rather
/// than merely a rotation), which is what the Barenco identity assumes.
fn x_power_matrix<T>(p: f64) -> Array<Complex<T>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let half = std::f64::consts::PI * p / 2.0;
    let global = Complex::new(half.cos(), half.sin()); // e^{iπp/2}
    let cos_h = Complex::new(half.cos(), 0.0_f64);
    let neg_i_sin = Complex::new(0.0_f64, -half.sin()); // -i·sin(πp/2)

    let m00 = global * cos_h;
    let m01 = global * neg_i_sin;
    let m10 = global * neg_i_sin;
    let m11 = global * cos_h;

    let conv =
        |z: Complex<f64>| Complex::new(<T as From<f64>>::from(z.re), <T as From<f64>>::from(z.im));

    let data = vec![conv(m00), conv(m01), conv(m10), conv(m11)];
    Array::from_vec_shape(data, &[2, 2]).unwrap_or_else(|e| panic!("{e}"))
}

/// Build the controlled version (4×4) of a single-qubit gate.
///
/// The resulting matrix is `diag(I₂, U)` in the basis where the control qubit is
/// the most-significant gate bit. When applied via [`QuantumCircuit::add_gate`]
/// with `target_qubits = [target, control]`, gate bit 0 maps to `target` and gate
/// bit 1 maps to `control`, so the lower-right block `U` acts on the target iff the
/// control is `|1⟩`.
fn controlled_2x2<T>(gate: &Array<Complex<T>>) -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let mut data = vec![Complex::new(T::zero(), T::zero()); 16];
    // Control = 0 → identity on the target.
    data[0] = Complex::new(T::one(), T::zero());
    data[5] = Complex::new(T::one(), T::zero());
    // Control = 1 → apply the gate to the target.
    for i in 0..2 {
        for j in 0..2 {
            let elem = gate
                .get(&[i, j])
                .map_err(|_| NumRs2Error::IndexOutOfBounds("Invalid gate access".to_string()))?;
            data[(i + 2) * 4 + (j + 2)] = elem;
        }
    }
    Array::from_vec_shape(data, &[4, 4])
}

/// Apply a multi-controlled `X^p` gate using the ancilla-free recursive
/// decomposition of Barenco et al. (1995), Lemma 7.5.
///
/// `controls` is the list of control qubits and `target` the flipped qubit; `p`
/// is the power of Pauli-X applied to the target when all controls are `|1⟩`.
/// The recursion is
/// ```text
/// C^k(X^p) = C(X^{p/2})(c_{k-1}, t)
///          · C^{k-1}(X)(c_0..c_{k-2}, c_{k-1})
///          · C(X^{-p/2})(c_{k-1}, t)
///          · C^{k-1}(X)(c_0..c_{k-2}, c_{k-1})
///          · C^{k-1}(X^{p/2})(c_0..c_{k-2}, t)
/// ```
/// with base cases `C^0(X^p) = X^p` (a bare single-qubit gate) and
/// `C^1(X^p)` (a single controlled gate). No ancilla qubits are used.
fn multi_controlled_x_power<T>(
    circuit: &mut QuantumCircuit<T>,
    controls: &[usize],
    target: usize,
    p: f64,
) -> Result<()>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    match controls.len() {
        0 => {
            // No controls: apply X^p directly to the target.
            let gate = x_power_matrix::<T>(p);
            circuit.add_gate(gate, vec![target], "Xp".to_string())?;
        }
        1 => {
            // Single control: apply controlled-X^p.
            let gate = controlled_2x2(&x_power_matrix::<T>(p))?;
            circuit.add_gate(gate, vec![target, controls[0]], "CXp".to_string())?;
        }
        _ => {
            let k = controls.len();
            let last = controls[k - 1];
            let rest = &controls[..k - 1];

            // C(X^{p/2}) on (last → target)
            let cv = controlled_2x2(&x_power_matrix::<T>(p / 2.0))?;
            circuit.add_gate(cv, vec![target, last], "CV".to_string())?;

            // C^{k-1}(X) on (rest → last)
            multi_controlled_x_power(circuit, rest, last, 1.0)?;

            // C(X^{-p/2}) on (last → target)
            let cv_dag = controlled_2x2(&x_power_matrix::<T>(-p / 2.0))?;
            circuit.add_gate(cv_dag, vec![target, last], "CVdg".to_string())?;

            // C^{k-1}(X) on (rest → last)
            multi_controlled_x_power(circuit, rest, last, 1.0)?;

            // C^{k-1}(X^{p/2}) on (rest → target)
            multi_controlled_x_power(circuit, rest, target, p / 2.0)?;
        }
    }
    Ok(())
}

/// Apply a multi-controlled-X (generalized Toffoli) gate.
///
/// Flips `target` iff every qubit in `controls` is `|1⟩`. Implemented via the
/// ancilla-free recursive √X decomposition (see [`multi_controlled_x_power`]).
fn multi_controlled_x<T>(
    circuit: &mut QuantumCircuit<T>,
    controls: &[usize],
    target: usize,
) -> Result<()>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    multi_controlled_x_power(circuit, controls, target, 1.0)
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

        // ----------------------------------------------------------------
        // Gradient-based optimization using the analytic parameter-shift rule
        // driven by the Adam optimizer (with bias correction).
        //
        // For an ansatz built from rotation gates (Rx/Ry/Rz, generator G with
        // eigenvalues ±1/2) the gradient of the energy with respect to each
        // parameter is exact:
        //
        //     ∂⟨H⟩/∂θ_i = ( E(θ_i + π/2) − E(θ_i − π/2) ) / 2
        //
        // Adam adapts the per-parameter step size from the first and second
        // moments of the gradient, which converges far more reliably than a
        // blind fixed step. Optimization stops early once the energy change
        // between consecutive iterations falls below a tolerance.
        // ----------------------------------------------------------------
        let shift = <T as From<f64>>::from(std::f64::consts::FRAC_PI_2);
        let half = <T as From<f64>>::from(0.5);

        // Adam hyper-parameters.
        let learning_rate = <T as From<f64>>::from(0.1);
        let beta1 = <T as From<f64>>::from(0.9);
        let beta2 = <T as From<f64>>::from(0.999);
        let epsilon = <T as From<f64>>::from(1e-8);
        let convergence_tol = <T as From<f64>>::from(1e-8);

        let n_params = params.len();
        let mut m = vec![T::zero(); n_params]; // first moment estimate
        let mut v = vec![T::zero(); n_params]; // second moment estimate

        let mut prev_energy = Self::evaluate_energy(num_qubits, hamiltonian, &ansatz, &params)?;
        let mut best_params = params.clone();
        let mut best_energy = prev_energy;

        for iteration in 0..max_iterations {
            // Analytic gradient via the parameter-shift rule.
            let mut gradient = vec![T::zero(); n_params];
            for i in 0..n_params {
                let mut params_plus = params.clone();
                let mut params_minus = params.clone();
                params_plus[i] = params_plus[i] + shift;
                params_minus[i] = params_minus[i] - shift;

                let energy_plus =
                    Self::evaluate_energy(num_qubits, hamiltonian, &ansatz, &params_plus)?;
                let energy_minus =
                    Self::evaluate_energy(num_qubits, hamiltonian, &ansatz, &params_minus)?;

                gradient[i] = (energy_plus - energy_minus) * half;
            }

            // Adam update with bias correction.
            // Bias-correction denominators 1 − β^t use the 1-based step index.
            let step = <T as From<f64>>::from((iteration + 1) as f64);
            let bias1 = T::one() - beta1.powf(step);
            let bias2 = T::one() - beta2.powf(step);

            for i in 0..n_params {
                let g = gradient[i];
                m[i] = beta1 * m[i] + (T::one() - beta1) * g;
                v[i] = beta2 * v[i] + (T::one() - beta2) * g * g;

                let m_hat = m[i] / bias1;
                let v_hat = v[i] / bias2;

                params[i] = params[i] - learning_rate * m_hat / (v_hat.sqrt() + epsilon);
            }

            let energy = Self::evaluate_energy(num_qubits, hamiltonian, &ansatz, &params)?;

            if energy < best_energy {
                best_energy = energy;
                best_params = params.clone();
            }

            // Convergence check on the change in energy.
            let delta = (energy - prev_energy).abs();
            prev_energy = energy;
            if delta < convergence_tol {
                break;
            }
        }

        Ok((best_params, best_energy))
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

        // Validate that the unitary acts on the full eigenstate register.
        let u_shape = unitary.shape();
        let expected_dim = 2_usize.pow(n_eigen as u32);
        if u_shape.len() != 2 || u_shape[0] != expected_dim || u_shape[1] != expected_dim {
            return Err(NumRs2Error::DimensionMismatch(format!(
                "Unitary must be {expected_dim}×{expected_dim} for a {n_eigen}-qubit eigenstate"
            )));
        }

        // Prepare the full register: load the supplied eigenstate into the high
        // qubits (qubits num_precision_qubits..n_total) and leave the precision
        // qubits in |0⟩, then put the precision qubits into uniform superposition.
        let init_state = Self::embed_eigenstate(eigenstate, num_precision_qubits, n_total)?;
        let mut circuit = QuantumCircuit::with_initial_state(init_state);
        for i in 0..num_precision_qubits {
            circuit.h(i)?;
        }

        // Apply controlled-U^(2^p) with a counting qubit as the control acting on
        // the whole eigenstate register. Applying the controlled-U gate `2^p`
        // times in sequence (all sharing the same control) realises controlled-U^(2^p)
        // exactly, because successive controlled gates with a common control compose
        // into the controlled product of the gates.
        //
        // Bit-order convention: this project's `QuantumFourierTransform` produces a
        // *bit-reversed* output, i.e. `QFT|j⟩ = (1/√N) Σ_x e^{2πi·j·rev(x)/N}|x⟩`
        // (verified against `QuantumFourierTransform::apply`). Its inverse therefore
        // collapses the precision register onto the exact eigenstate `|j⟩` only when
        // the register is prepared as `(1/√N) Σ_x e^{2πi·rev(x)·φ}|x⟩`. To achieve
        // that, counting qubit `k` must accumulate the phase of `U^(2^{p-1-k})`
        // (the most-significant counting qubit carries the smallest power), so that
        // the power weighting matches the inverse QFT's bit ordering. With this
        // pairing an exactly representable φ = j/N yields a deterministic, exact
        // measurement of the bin `j`.
        let controlled_u = Self::build_controlled_unitary(unitary, n_eigen)?;
        // Gate qubit order: eigenstate qubits occupy gate bits 0..n_eigen and the
        // control occupies the most-significant gate bit, matching the block layout
        // diag(I, U) produced by `build_controlled_unitary`.
        let eigen_qubits: Vec<usize> = (0..n_eigen).map(|q| num_precision_qubits + q).collect();

        for k in 0..num_precision_qubits {
            let power = 2_usize.pow((num_precision_qubits - 1 - k) as u32);
            let mut targets = eigen_qubits.clone();
            targets.push(k); // control is the high gate bit
            for _ in 0..power {
                circuit.add_gate(controlled_u.clone(), targets.clone(), "CU".to_string())?;
            }
        }

        // Apply the inverse QFT to the precision qubits (qubits 0..num_precision_qubits),
        // appended directly to the main circuit. This mirrors
        // `QuantumFourierTransform::apply_inverse` but restricted to the precision
        // register so the eigenstate qubits are left untouched.
        Self::inverse_qft_on_precision(&mut circuit, num_precision_qubits)?;

        // Measure all qubits, then keep only the precision-register bits.
        let state = circuit.execute()?;
        let (result, _) = Measurement::measure_all(&state, None)?;

        // Extract the raw precision register (the low `num_precision_qubits` bits).
        let mask = if num_precision_qubits >= usize::BITS as usize {
            usize::MAX
        } else {
            (1_usize << num_precision_qubits) - 1
        };
        let raw_outcome = result.outcome & mask;

        // The inverse QFT (whose forward counterpart in this project emits a
        // bit-reversed output) recovers the *bit-reversed* phase bin: a register
        // prepared as `(1/√N) Σ_x e^{2πi·rev(x)·φ}|x⟩` collapses to `|rev(j)⟩` for
        // φ = j/N. Reversing the `num_precision_qubits` measured bits therefore
        // recovers the true bin `j`, and the phase is `j / 2^p`. For exactly
        // representable phases this is deterministic and exact.
        let mut phase_bin = 0_usize;
        for bit in 0..num_precision_qubits {
            if (raw_outcome >> bit) & 1 == 1 {
                phase_bin |= 1 << (num_precision_qubits - 1 - bit);
            }
        }
        let phase = (phase_bin as f64) / 2.0_f64.powi(num_precision_qubits as i32);

        Ok(phase)
    }

    /// Build the controlled version of an `n`-qubit unitary `U` as the
    /// `2^{n+1} × 2^{n+1}` block matrix `diag(I, U)`.
    ///
    /// The control is the most-significant gate bit, so when applied with
    /// `target_qubits = [eigen_0, …, eigen_{n-1}, control]` the lower-right block
    /// `U` acts on the eigenstate register iff the control qubit is `|1⟩`.
    fn build_controlled_unitary<T>(
        unitary: &Array<Complex<T>>,
        n_eigen: usize,
    ) -> Result<Array<Complex<T>>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let dim = 2_usize.pow(n_eigen as u32);
        let big = 2 * dim;
        let mut data = vec![Complex::new(T::zero(), T::zero()); big * big];

        // Control = 0 → identity on the eigenstate register (top-left block).
        for i in 0..dim {
            data[i * big + i] = Complex::new(T::one(), T::zero());
        }
        // Control = 1 → apply U (bottom-right block).
        for i in 0..dim {
            for j in 0..dim {
                let elem = unitary.get(&[i, j]).map_err(|_| {
                    NumRs2Error::IndexOutOfBounds("Invalid unitary access".to_string())
                })?;
                data[(i + dim) * big + (j + dim)] = elem;
            }
        }

        Array::from_vec_shape(data, &[big, big])
    }

    /// Embed an `n_eigen`-qubit eigenstate into the high qubits of an
    /// `n_total`-qubit register, with the low `num_precision_qubits` qubits in `|0⟩`.
    ///
    /// In the LSB-first convention the precision qubits are the low bits, so the
    /// eigenstate amplitude at index `e` is placed at full-register index
    /// `e << num_precision_qubits`.
    fn embed_eigenstate<T>(
        eigenstate: &StateVector<T>,
        num_precision_qubits: usize,
        n_total: usize,
    ) -> Result<StateVector<T>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let total_dim = 2_usize.pow(n_total as u32);
        let mut amps = vec![Complex::new(T::zero(), T::zero()); total_dim];
        let eigen_amps = eigenstate.amplitudes().to_vec();

        for (e, &amp) in eigen_amps.iter().enumerate() {
            let full_index = e << num_precision_qubits;
            amps[full_index] = amp;
        }

        StateVector::from_amplitudes(amps)
    }

    /// Apply the inverse Quantum Fourier Transform to the precision register
    /// (qubits `0..num_precision_qubits`) of an existing circuit.
    ///
    /// This replicates [`QuantumFourierTransform::apply_inverse`] but acts only on
    /// the precision qubits, leaving any additional (eigenstate) qubits untouched.
    fn inverse_qft_on_precision<T>(
        circuit: &mut QuantumCircuit<T>,
        num_precision_qubits: usize,
    ) -> Result<()>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let n = num_precision_qubits;

        // Reverse the bit-order swaps.
        for i in 0..(n / 2) {
            circuit.swap(i, n - 1 - i)?;
        }

        // Inverse controlled rotations and Hadamards.
        for j in (0..n).rev() {
            for k in ((j + 1)..n).rev() {
                let angle = <T as From<f64>>::from(-2.0 * PI / 2.0_f64.powi((k - j + 1) as i32));
                QuantumPhaseEstimation::controlled_phase_rotation(circuit, k, j, angle)?;
            }
            circuit.h(j)?;
        }

        Ok(())
    }

    /// Helper: controlled phase rotation `CP(θ) = diag(1, 1, 1, e^{iθ})`.
    ///
    /// Mirrors the controlled-phase gate used by the Quantum Fourier Transform so
    /// the inverse QFT inside phase estimation uses an identical convention.
    fn controlled_phase_rotation<T>(
        circuit: &mut QuantumCircuit<T>,
        control: usize,
        target: usize,
        angle: T,
    ) -> Result<()>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
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

        let gate = Array::from_vec_shape(gate_data, &[4, 4])?;
        circuit.add_gate(gate, vec![control, target], "CP".to_string())?;

        Ok(())
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

        // ⟨Z⟩ for the Ry(θ) ansatz on |0⟩ is cos(θ); the true minimum is at θ = π
        // with energy −1. The parameter-shift + Adam optimizer should converge
        // there from the initial θ = 0.5.
        let initial_params = vec![0.5];
        let (params, energy) = VQE::minimize(1, &ham, ansatz, initial_params, 200)
            .expect("test: valid VQE minimization");

        assert!(params.len() == 1);
        // Parameter should move from 0.5 toward π (≈3.1416).
        assert!(
            params[0] > 2.5,
            "Parameter should have converged toward π, got {}",
            params[0]
        );
        // Energy should converge close to the true ground-state energy of -1.
        assert!(
            energy < -0.95,
            "Energy should converge toward -1, got {energy}"
        );
        assert_relative_eq!(energy, -1.0, epsilon = 0.05);
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

    // Multi-controlled-X (generalized Toffoli) must flip the target iff every
    // control is |1⟩. Controls = {0, 1}, target = 2.
    #[test]
    fn test_multi_controlled_x_all_controls_one() {
        // Prepare |0_2 1_1 1_0⟩ = index 0b011 = 3 (controls set, target clear).
        let mut circuit = QuantumCircuit::<f64>::new(3).expect("test: valid qubit count");
        circuit.x(0).expect("test: x");
        circuit.x(1).expect("test: x");
        multi_controlled_x(&mut circuit, &[0, 1], 2).expect("test: mcx");

        let state = circuit.execute().expect("test: execute");
        // Target should flip → |111⟩ = index 7.
        let prob_111 = state.get_probability(7).expect("test: prob");
        assert_relative_eq!(prob_111, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_controlled_x_one_control_zero() {
        // Prepare |001⟩ = index 1 (only control 0 set, control 1 clear).
        let mut circuit = QuantumCircuit::<f64>::new(3).expect("test: valid qubit count");
        circuit.x(0).expect("test: x");
        multi_controlled_x(&mut circuit, &[0, 1], 2).expect("test: mcx");

        let state = circuit.execute().expect("test: execute");
        // Not all controls set → target unchanged, state stays |001⟩ = index 1.
        let prob_001 = state.get_probability(1).expect("test: prob");
        assert_relative_eq!(prob_001, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_controlled_x_three_controls() {
        // Controls = {0,1,2}, target = 3. Prepare |0111⟩ (all controls set).
        let mut circuit = QuantumCircuit::<f64>::new(4).expect("test: valid qubit count");
        circuit.x(0).expect("test: x");
        circuit.x(1).expect("test: x");
        circuit.x(2).expect("test: x");
        multi_controlled_x(&mut circuit, &[0, 1, 2], 3).expect("test: mcx");

        let state = circuit.execute().expect("test: execute");
        // Target flips → |1111⟩ = index 15.
        let prob_1111 = state.get_probability(15).expect("test: prob");
        assert_relative_eq!(prob_1111, 1.0, epsilon = 1e-10);

        // With one control cleared, the target must NOT flip.
        let mut circuit2 = QuantumCircuit::<f64>::new(4).expect("test: valid qubit count");
        circuit2.x(0).expect("test: x");
        circuit2.x(1).expect("test: x");
        multi_controlled_x(&mut circuit2, &[0, 1, 2], 3).expect("test: mcx");
        let state2 = circuit2.execute().expect("test: execute");
        // Stays |0011⟩ = index 3.
        let prob_0011 = state2.get_probability(3).expect("test: prob");
        assert_relative_eq!(prob_0011, 1.0, epsilon = 1e-10);
    }

    // Multi-controlled-Z must flip the phase of |11...1⟩ only and leave the
    // magnitudes of every basis state unchanged (it is diagonal).
    #[test]
    fn test_multi_controlled_z_flips_only_all_ones() {
        // |111⟩: amplitude should become -1.
        let mut circuit = QuantumCircuit::<f64>::new(3).expect("test: valid qubit count");
        circuit.x(0).expect("test: x");
        circuit.x(1).expect("test: x");
        circuit.x(2).expect("test: x");
        GroverSearch::multi_controlled_z(&mut circuit, 3).expect("test: mcz");
        let state = circuit.execute().expect("test: execute");

        let amp = state.amplitudes().to_vec()[7];
        assert_relative_eq!(amp.re, -1.0, epsilon = 1e-10);
        assert_relative_eq!(amp.im, 0.0, epsilon = 1e-10);

        // |110⟩ (index 6): amplitude must remain +1 (phase untouched).
        let mut circuit2 = QuantumCircuit::<f64>::new(3).expect("test: valid qubit count");
        circuit2.x(1).expect("test: x");
        circuit2.x(2).expect("test: x");
        GroverSearch::multi_controlled_z(&mut circuit2, 3).expect("test: mcz");
        let state2 = circuit2.execute().expect("test: execute");

        let amp2 = state2.amplitudes().to_vec()[6];
        assert_relative_eq!(amp2.re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(amp2.im, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_multi_controlled_z_diagonal_on_superposition() {
        // Apply MCZ to the uniform superposition over 3 qubits and confirm only
        // the all-ones amplitude is negated; all magnitudes stay equal.
        let mut circuit = QuantumCircuit::<f64>::new(3).expect("test: valid qubit count");
        for q in 0..3 {
            circuit.h(q).expect("test: h");
        }
        GroverSearch::multi_controlled_z(&mut circuit, 3).expect("test: mcz");
        let state = circuit.execute().expect("test: execute");

        let amps = state.amplitudes().to_vec();
        let inv_sqrt8 = 1.0_f64 / 8.0_f64.sqrt();
        for (idx, amp) in amps.iter().enumerate() {
            // All magnitudes equal to 1/√8.
            assert_relative_eq!(amp.norm(), inv_sqrt8, epsilon = 1e-10);
            if idx == 7 {
                assert_relative_eq!(amp.re, -inv_sqrt8, epsilon = 1e-10);
            } else {
                assert_relative_eq!(amp.re, inv_sqrt8, epsilon = 1e-10);
            }
            assert_relative_eq!(amp.im, 0.0, epsilon = 1e-10);
        }
    }

    // End-to-end QPE: estimate the phase of a single-qubit phase gate U = diag(1, e^{2πiφ})
    // acting on its eigenstate |1⟩. With φ = 1/4 and 3 precision qubits the result is exact.
    #[test]
    fn test_qpe_phase_gate_quarter() {
        let phi = 0.25_f64;
        // U|1⟩ = e^{2πiφ}|1⟩, U = diag(1, e^{2πiφ}).
        let angle = 2.0 * PI * phi;
        let u_data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(angle.cos(), angle.sin()),
        ];
        let unitary = Array::from_vec(u_data).reshape(&[2, 2]);

        // Eigenstate |1⟩.
        let eigen =
            StateVector::from_amplitudes(vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)])
                .expect("test: eigenstate");

        let estimated =
            QuantumPhaseEstimation::estimate_phase(3, &unitary, &eigen).expect("test: qpe");

        // 3-bit exact representation of 1/4 = 0.010₂.
        assert_relative_eq!(estimated, 0.25, epsilon = 1e-9);
    }

    #[test]
    fn test_qpe_phase_gate_half() {
        let phi = 0.5_f64;
        let angle = 2.0 * PI * phi;
        let u_data = vec![
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(angle.cos(), angle.sin()),
        ];
        let unitary = Array::from_vec(u_data).reshape(&[2, 2]);
        let eigen =
            StateVector::from_amplitudes(vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)])
                .expect("test: eigenstate");

        let estimated =
            QuantumPhaseEstimation::estimate_phase(3, &unitary, &eigen).expect("test: qpe");
        assert_relative_eq!(estimated, 0.5, epsilon = 1e-9);
    }
}
