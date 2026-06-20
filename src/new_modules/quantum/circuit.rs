//! Quantum Circuit Builder
//!
//! This module provides a builder API for constructing and manipulating quantum circuits.
//! Circuits are sequences of quantum gates applied to qubits.
//!
//! # Examples
//!
//! ```
//! use numrs2::new_modules::quantum::circuit::QuantumCircuit;
//!
//! // Create a 2-qubit circuit
//! let mut circuit = QuantumCircuit::<f64>::new(2).expect("valid qubit count");
//!
//! // Add a Hadamard gate on qubit 0
//! circuit.h(0).expect("valid qubit index");
//!
//! // Add a CNOT gate with control=0, target=1
//! circuit.cnot(0, 1).expect("valid qubit indices");
//!
//! // Execute the circuit
//! let final_state = circuit.execute().expect("circuit execution succeeds");
//! ```

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::new_modules::quantum::gates;
use crate::new_modules::quantum::statevector::StateVector;
use num_traits::Float;
use scirs2_core::Complex;
use std::fmt::Debug;

/// Represents a single gate operation in a quantum circuit
#[derive(Clone, Debug)]
struct GateOperation<T: Clone> {
    /// The unitary matrix representing the gate
    gate: Array<Complex<T>>,
    /// Qubits the gate acts on
    target_qubits: Vec<usize>,
    /// Name of the gate (for display)
    name: String,
}

/// Quantum circuit builder
///
/// Represents a quantum circuit as a sequence of gate operations.
/// Provides a fluent API for adding gates and executing the circuit.
#[derive(Clone, Debug)]
pub struct QuantumCircuit<T: Clone> {
    /// Number of qubits in the circuit
    num_qubits: usize,
    /// Sequence of gate operations
    operations: Vec<GateOperation<T>>,
    /// Initial state (default: |0...0⟩)
    initial_state: StateVector<T>,
}

impl<T> QuantumCircuit<T>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    /// Create a new quantum circuit
    ///
    /// # Arguments
    ///
    /// * `num_qubits` - Number of qubits in the circuit
    ///
    /// # Returns
    ///
    /// A new empty circuit with the given number of qubits
    pub fn new(num_qubits: usize) -> Result<Self> {
        let initial_state = StateVector::new(num_qubits)?;
        Ok(Self {
            num_qubits,
            operations: Vec::new(),
            initial_state,
        })
    }

    /// Create a circuit with a custom initial state
    ///
    /// # Arguments
    ///
    /// * `initial_state` - Initial quantum state
    pub fn with_initial_state(initial_state: StateVector<T>) -> Self {
        let num_qubits = initial_state.num_qubits();
        Self {
            num_qubits,
            operations: Vec::new(),
            initial_state,
        }
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get the number of gates in the circuit
    pub fn num_gates(&self) -> usize {
        self.operations.len()
    }

    /// Calculate the circuit depth
    ///
    /// Depth is the longest path of dependent gates (gates acting on overlapping qubits).
    pub fn depth(&self) -> usize {
        if self.operations.is_empty() {
            return 0;
        }

        let mut last_gate_time = vec![0; self.num_qubits];
        let mut max_depth = 0;

        for op in &self.operations {
            // Find the latest time any of the target qubits was used
            let mut start_time = 0;
            for &qubit in &op.target_qubits {
                start_time = start_time.max(last_gate_time[qubit]);
            }

            // This gate happens at start_time + 1
            let gate_time = start_time + 1;

            // Update all target qubits
            for &qubit in &op.target_qubits {
                last_gate_time[qubit] = gate_time;
            }

            max_depth = max_depth.max(gate_time);
        }

        max_depth
    }

    /// Add a custom gate to the circuit
    ///
    /// # Arguments
    ///
    /// * `gate` - Unitary matrix representing the gate
    /// * `target_qubits` - Qubits the gate acts on
    /// * `name` - Name of the gate for display
    pub fn add_gate(
        &mut self,
        gate: Array<Complex<T>>,
        target_qubits: Vec<usize>,
        name: String,
    ) -> Result<&mut Self> {
        // Validate target qubits
        for &qubit in &target_qubits {
            if qubit >= self.num_qubits {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "Qubit index {} out of bounds for {} qubits",
                    qubit, self.num_qubits
                )));
            }
        }

        self.operations.push(GateOperation {
            gate,
            target_qubits,
            name,
        });

        Ok(self)
    }

    /// Add a Hadamard gate
    pub fn h(&mut self, qubit: usize) -> Result<&mut Self> {
        let gate = gates::hadamard()?;
        self.add_gate(gate, vec![qubit], "H".to_string())
    }

    /// Add a Pauli-X gate
    pub fn x(&mut self, qubit: usize) -> Result<&mut Self> {
        let gate = gates::pauli_x()?;
        self.add_gate(gate, vec![qubit], "X".to_string())
    }

    /// Add a Pauli-Y gate
    pub fn y(&mut self, qubit: usize) -> Result<&mut Self> {
        let gate = gates::pauli_y()?;
        self.add_gate(gate, vec![qubit], "Y".to_string())
    }

    /// Add a Pauli-Z gate
    pub fn z(&mut self, qubit: usize) -> Result<&mut Self> {
        let gate = gates::pauli_z()?;
        self.add_gate(gate, vec![qubit], "Z".to_string())
    }

    /// Add a Phase gate (S gate)
    pub fn s(&mut self, qubit: usize) -> Result<&mut Self> {
        let gate = gates::phase_gate()?;
        self.add_gate(gate, vec![qubit], "S".to_string())
    }

    /// Add a T gate
    pub fn t(&mut self, qubit: usize) -> Result<&mut Self> {
        let gate = gates::t_gate()?;
        self.add_gate(gate, vec![qubit], "T".to_string())
    }

    /// Add an Rx rotation gate
    pub fn rx(&mut self, qubit: usize, theta: T) -> Result<&mut Self> {
        let gate = gates::rx(theta)?;
        self.add_gate(gate, vec![qubit], format!("Rx({:.3})", theta.into()))
    }

    /// Add an Ry rotation gate
    pub fn ry(&mut self, qubit: usize, theta: T) -> Result<&mut Self> {
        let gate = gates::ry(theta)?;
        self.add_gate(gate, vec![qubit], format!("Ry({:.3})", theta.into()))
    }

    /// Add an Rz rotation gate
    pub fn rz(&mut self, qubit: usize, theta: T) -> Result<&mut Self> {
        let gate = gates::rz(theta)?;
        self.add_gate(gate, vec![qubit], format!("Rz({:.3})", theta.into()))
    }

    /// Add a CNOT gate
    pub fn cnot(&mut self, control: usize, target: usize) -> Result<&mut Self> {
        if control == target {
            return Err(NumRs2Error::InvalidOperation(
                "Control and target qubits must be different".to_string(),
            ));
        }
        let gate = gates::cnot()?;
        // Note: Gate matrix uses MSB-first ordering, but state vector uses LSB-first.
        // We reverse the qubit order to match the conventions.
        self.add_gate(gate, vec![target, control], "CNOT".to_string())
    }

    /// Add a SWAP gate
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) -> Result<&mut Self> {
        if qubit1 == qubit2 {
            return Err(NumRs2Error::InvalidOperation(
                "SWAP qubits must be different".to_string(),
            ));
        }
        let gate = gates::swap()?;
        // Note: Gate matrix uses MSB-first ordering, but state vector uses LSB-first.
        // We reverse the qubit order to match the conventions.
        self.add_gate(gate, vec![qubit2, qubit1], "SWAP".to_string())
    }

    /// Add a CZ gate
    pub fn cz(&mut self, control: usize, target: usize) -> Result<&mut Self> {
        if control == target {
            return Err(NumRs2Error::InvalidOperation(
                "Control and target qubits must be different".to_string(),
            ));
        }
        let gate = gates::cz()?;
        self.add_gate(gate, vec![control, target], "CZ".to_string())
    }

    /// Add a CY gate
    pub fn cy(&mut self, control: usize, target: usize) -> Result<&mut Self> {
        if control == target {
            return Err(NumRs2Error::InvalidOperation(
                "Control and target qubits must be different".to_string(),
            ));
        }
        let gate = gates::cy()?;
        self.add_gate(gate, vec![control, target], "CY".to_string())
    }

    /// Add a Toffoli gate (CCNOT)
    pub fn toffoli(
        &mut self,
        control1: usize,
        control2: usize,
        target: usize,
    ) -> Result<&mut Self> {
        if control1 == control2 || control1 == target || control2 == target {
            return Err(NumRs2Error::InvalidOperation(
                "Toffoli qubits must all be different".to_string(),
            ));
        }
        let gate = gates::toffoli()?;
        self.add_gate(
            gate,
            vec![control1, control2, target],
            "Toffoli".to_string(),
        )
    }

    /// Add a Fredkin gate (CSWAP)
    pub fn fredkin(&mut self, control: usize, target1: usize, target2: usize) -> Result<&mut Self> {
        if control == target1 || control == target2 || target1 == target2 {
            return Err(NumRs2Error::InvalidOperation(
                "Fredkin qubits must all be different".to_string(),
            ));
        }
        let gate = gates::fredkin()?;
        self.add_gate(gate, vec![control, target1, target2], "Fredkin".to_string())
    }

    /// Add a controlled-U gate with arbitrary unitary U.
    ///
    /// The gate applies `u` to `targets` when all `controls` qubits are |1⟩.
    ///
    /// # Arguments
    ///
    /// * `u` - Unitary matrix of shape [2^k, 2^k] for k = targets.len()
    /// * `controls` - Control qubit indices
    /// * `targets` - Target qubit indices
    pub fn controlled_u(
        &mut self,
        u: Array<Complex<T>>,
        controls: Vec<usize>,
        targets: Vec<usize>,
    ) -> Result<&mut Self> {
        // Validate no overlap between controls and targets
        for &c in &controls {
            if targets.contains(&c) {
                return Err(NumRs2Error::InvalidOperation(
                    "controlled_u: control and target qubits must not overlap".to_string(),
                ));
            }
        }

        let cu = gates::controlled_u_gate(&u, controls.len())?;

        // Qubit ordering: targets first (gate indices 0..k), then controls (gate indices k..k+m).
        // This matches the controlled_u_gate convention where low bits are target bits.
        let mut combined = targets.clone();
        combined.extend_from_slice(&controls);

        self.add_gate(cu, combined, "Controlled-U".to_string())
    }

    /// Execute the circuit and return the final state
    ///
    /// Applies all gates in sequence to the initial state.
    pub fn execute(&self) -> Result<StateVector<T>> {
        let mut state = self.initial_state.clone();

        for op in &self.operations {
            state.apply_gate(&op.gate, &op.target_qubits)?;
        }

        Ok(state)
    }

    /// Clear all gates from the circuit
    pub fn clear(&mut self) {
        self.operations.clear();
    }

    /// Create a copy of this circuit
    pub fn clone_circuit(&self) -> Self {
        self.clone()
    }

    /// Optimize the circuit by fusing adjacent gates.
    ///
    /// Phase 1: Fuse consecutive single-qubit gates on the same qubit.
    /// Phase 2: Fuse adjacent gate pairs whose qubit supports overlap.
    pub fn optimize(&mut self) -> Result<()> {
        // Phase 1: single-qubit gate fusion (fast path, same as before)
        self.fuse_single_qubit_chains()?;
        // Phase 2: multi-qubit gate fusion (generalized)
        self.fuse_adjacent_gates()?;
        Ok(())
    }

    /// Fuse consecutive single-qubit gates on the same qubit.
    fn fuse_single_qubit_chains(&mut self) -> Result<()> {
        let mut optimized_ops: Vec<GateOperation<T>> = Vec::new();
        let mut i = 0;

        while i < self.operations.len() {
            let current = &self.operations[i];

            if current.target_qubits.len() == 1 {
                let qubit = current.target_qubits[0];
                let mut fused_gate = current.gate.clone();
                let mut fused_name = current.name.clone();
                let mut j = i + 1;

                while j < self.operations.len() {
                    let next = &self.operations[j];
                    if next.target_qubits.len() == 1 && next.target_qubits[0] == qubit {
                        fused_gate = multiply_square_gates(&next.gate, &fused_gate)?;
                        fused_name = format!("Fused({}+{})", fused_name, next.name);
                        j += 1;
                    } else {
                        break;
                    }
                }

                optimized_ops.push(GateOperation {
                    gate: fused_gate,
                    target_qubits: vec![qubit],
                    name: if j > i + 1 {
                        fused_name
                    } else {
                        current.name.clone()
                    },
                });
                i = j;
            } else {
                optimized_ops.push(current.clone());
                i += 1;
            }
        }

        self.operations = optimized_ops;
        Ok(())
    }

    /// Fuse adjacent gate pairs that share overlapping qubit support.
    /// Runs to fixed-point (repeats until no more fusions are possible).
    fn fuse_adjacent_gates(&mut self) -> Result<()> {
        loop {
            let mut fused_any = false;
            let mut new_ops: Vec<GateOperation<T>> = Vec::new();
            let mut i = 0;

            while i < self.operations.len() {
                if i + 1 < self.operations.len() {
                    let g1 = &self.operations[i];
                    let g2 = &self.operations[i + 1];

                    // Check if their qubit sets overlap
                    let overlaps = g1
                        .target_qubits
                        .iter()
                        .any(|q| g2.target_qubits.contains(q));

                    if overlaps {
                        // Fuse: compute union of qubit sets (sorted)
                        let mut all_qubits = g1.target_qubits.clone();
                        for &q in &g2.target_qubits {
                            if !all_qubits.contains(&q) {
                                all_qubits.push(q);
                            }
                        }
                        all_qubits.sort_unstable();

                        // Embed both gates into the combined space
                        let g1_big =
                            embed_gate_in_space(&g1.gate, &g1.target_qubits, &all_qubits)?;
                        let g2_big =
                            embed_gate_in_space(&g2.gate, &g2.target_qubits, &all_qubits)?;

                        // Fused = G2 * G1 (G1 applied first)
                        let fused_gate = multiply_square_gates(&g2_big, &g1_big)?;
                        let fused_name = format!("Fused({}+{})", g1.name, g2.name);

                        new_ops.push(GateOperation {
                            gate: fused_gate,
                            target_qubits: all_qubits,
                            name: fused_name,
                        });

                        fused_any = true;
                        i += 2; // consumed both
                        continue;
                    }
                }

                new_ops.push(self.operations[i].clone());
                i += 1;
            }

            self.operations = new_ops;

            if !fused_any {
                break;
            }
        }

        Ok(())
    }

    /// Get a summary of the circuit
    pub fn summary(&self) -> String {
        format!(
            "QuantumCircuit(qubits={}, gates={}, depth={})",
            self.num_qubits,
            self.num_gates(),
            self.depth()
        )
    }
}

/// Embed a gate acting on `gate_qubits` into the full Hilbert space spanned by `all_qubits`.
///
/// `all_qubits` must be sorted ascending. `gate_qubits` must be a subset of `all_qubits`.
///
/// The result is a 2^|all_qubits| × 2^|all_qubits| matrix consistent with the
/// `statevector::apply_gate` convention: for `apply_gate(M, all_qubits)`, bit k of
/// the gate index = value of circuit qubit `all_qubits[k]`.
fn embed_gate_in_space<T>(
    gate: &Array<Complex<T>>,
    gate_qubits: &[usize],
    all_qubits: &[usize],
) -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let num_all = all_qubits.len();
    let num_gate = gate_qubits.len();
    let big_dim = 1usize << num_all;

    // Map each gate qubit to its position in all_qubits
    let mut pos_in_all = vec![0usize; num_gate];
    for (g, &gq) in gate_qubits.iter().enumerate() {
        let pos = all_qubits.iter().position(|&q| q == gq).ok_or_else(|| {
            NumRs2Error::InvalidOperation(format!(
                "embed_gate_in_space: gate qubit {} not found in all_qubits",
                gq
            ))
        })?;
        pos_in_all[g] = pos;
    }

    let mut data = vec![Complex::new(T::zero(), T::zero()); big_dim * big_dim];

    for big_row in 0..big_dim {
        for big_col in 0..big_dim {
            // Extract gate-qubit bits from big_row and big_col
            let mut gate_row = 0usize;
            let mut gate_col = 0usize;
            let mut identity_matches = true;

            for g in 0..num_gate {
                let bit_pos = pos_in_all[g];
                let row_bit = (big_row >> bit_pos) & 1;
                let col_bit = (big_col >> bit_pos) & 1;
                gate_row |= row_bit << g;
                gate_col |= col_bit << g;
            }

            // For non-gate qubits (positions not in pos_in_all), row and col must match
            for p in 0..num_all {
                if !pos_in_all.contains(&p) {
                    let row_bit = (big_row >> p) & 1;
                    let col_bit = (big_col >> p) & 1;
                    if row_bit != col_bit {
                        identity_matches = false;
                        break;
                    }
                }
            }

            if identity_matches {
                let val = gate.get(&[gate_row, gate_col]).map_err(|_| {
                    NumRs2Error::IndexOutOfBounds(
                        "embed_gate_in_space: invalid gate access".to_string(),
                    )
                })?;
                data[big_row * big_dim + big_col] = val;
            }
        }
    }

    Ok(Array::from_vec(data).reshape(&[big_dim, big_dim]))
}

/// Multiply two square gate matrices of the same size: result = a * b
fn multiply_square_gates<T>(
    a: &Array<Complex<T>>,
    b: &Array<Complex<T>>,
) -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let shape_a = a.shape();
    let shape_b = b.shape();

    if shape_a.len() != 2 || shape_b.len() != 2 {
        return Err(NumRs2Error::DimensionMismatch(
            "multiply_square_gates: both matrices must be 2D".to_string(),
        ));
    }
    if shape_a[0] != shape_a[1] || shape_b[0] != shape_b[1] {
        return Err(NumRs2Error::DimensionMismatch(
            "multiply_square_gates: both matrices must be square".to_string(),
        ));
    }
    let n = shape_a[0];
    if n != shape_b[0] {
        return Err(NumRs2Error::DimensionMismatch(format!(
            "multiply_square_gates: size mismatch {}x{} vs {}x{}",
            shape_a[0], shape_a[1], shape_b[0], shape_b[1]
        )));
    }

    let mut result = vec![Complex::new(T::zero(), T::zero()); n * n];

    for i in 0..n {
        for j in 0..n {
            let mut sum = Complex::new(T::zero(), T::zero());
            for k in 0..n {
                let a_ik = a.get(&[i, k]).map_err(|_| {
                    NumRs2Error::IndexOutOfBounds(
                        "multiply_square_gates: invalid a access".to_string(),
                    )
                })?;
                let b_kj = b.get(&[k, j]).map_err(|_| {
                    NumRs2Error::IndexOutOfBounds(
                        "multiply_square_gates: invalid b access".to_string(),
                    )
                })?;
                sum = sum + a_ik * b_kj;
            }
            result[i * n + j] = sum;
        }
    }

    Ok(Array::from_vec(result).reshape(&[n, n]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_circuit_creation() {
        let circuit = QuantumCircuit::<f64>::new(2).expect("test: valid qubit count");
        assert_eq!(circuit.num_qubits(), 2);
        assert_eq!(circuit.num_gates(), 0);
    }

    #[test]
    fn test_add_single_qubit_gates() {
        let mut circuit = QuantumCircuit::<f64>::new(2).expect("test: valid qubit count");
        circuit.h(0).expect("test: valid qubit index");
        circuit.x(1).expect("test: valid qubit index");
        circuit.y(0).expect("test: valid qubit index");
        circuit.z(1).expect("test: valid qubit index");

        assert_eq!(circuit.num_gates(), 4);
    }

    #[test]
    fn test_add_two_qubit_gates() {
        let mut circuit = QuantumCircuit::<f64>::new(2).expect("test: valid qubit count");
        circuit.cnot(0, 1).expect("test: valid qubit indices");
        circuit.swap(0, 1).expect("test: valid qubit indices");
        circuit.cz(0, 1).expect("test: valid qubit indices");

        assert_eq!(circuit.num_gates(), 3);
    }

    #[test]
    fn test_bell_state_circuit() {
        let mut circuit = QuantumCircuit::<f64>::new(2).expect("test: valid qubit count");
        circuit.h(0).expect("test: valid qubit index");
        circuit.cnot(0, 1).expect("test: valid qubit indices");

        let state = circuit.execute().expect("test: circuit execution succeeds");

        // Bell state: (|00⟩ + |11⟩)/√2
        let prob_00 = state.get_probability(0).expect("test: valid state index");
        let prob_11 = state.get_probability(3).expect("test: valid state index");

        assert_relative_eq!(prob_00, 0.5, epsilon = 1e-10);
        assert_relative_eq!(prob_11, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_circuit_depth() {
        let mut circuit = QuantumCircuit::<f64>::new(3).expect("test: valid qubit count");
        circuit.h(0).expect("test: valid qubit index");
        circuit.h(1).expect("test: valid qubit index");
        circuit.h(2).expect("test: valid qubit index");
        // All three H gates can be parallel, depth = 1

        assert_eq!(circuit.depth(), 1);

        circuit.cnot(0, 1).expect("test: valid qubit indices"); // Depth 2
        circuit.cnot(1, 2).expect("test: valid qubit indices"); // Depth 3

        assert_eq!(circuit.depth(), 3);
    }

    #[test]
    fn test_rotation_gates() {
        let mut circuit = QuantumCircuit::<f64>::new(1).expect("test: valid qubit count");
        let theta = std::f64::consts::PI;

        circuit.rx(0, theta).expect("test: valid rotation gate");
        circuit.ry(0, theta).expect("test: valid rotation gate");
        circuit.rz(0, theta).expect("test: valid rotation gate");

        assert_eq!(circuit.num_gates(), 3);
    }

    #[test]
    fn test_toffoli_gate() {
        let mut circuit = QuantumCircuit::<f64>::new(3).expect("test: valid qubit count");
        circuit.toffoli(0, 1, 2).expect("test: valid qubit indices");

        assert_eq!(circuit.num_gates(), 1);
    }

    #[test]
    fn test_invalid_qubit_index() {
        let mut circuit = QuantumCircuit::<f64>::new(2).expect("test: valid qubit count");
        let result = circuit.h(5);

        assert!(result.is_err());
    }

    #[test]
    fn test_clear_circuit() {
        let mut circuit = QuantumCircuit::<f64>::new(2).expect("test: valid qubit count");
        circuit.h(0).expect("test: valid qubit index");
        circuit.cnot(0, 1).expect("test: valid qubit indices");

        assert_eq!(circuit.num_gates(), 2);

        circuit.clear();
        assert_eq!(circuit.num_gates(), 0);
    }

    #[test]
    fn test_circuit_optimize() {
        let mut circuit = QuantumCircuit::<f64>::new(1).expect("test: valid qubit count");
        circuit.x(0).expect("test: valid qubit index");
        circuit.y(0).expect("test: valid qubit index");
        circuit.z(0).expect("test: valid qubit index");

        assert_eq!(circuit.num_gates(), 3);

        circuit
            .optimize()
            .expect("test: circuit optimization succeeds");

        // Should be fused into a single gate
        assert_eq!(circuit.num_gates(), 1);
    }

    #[test]
    fn test_summary() {
        let mut circuit = QuantumCircuit::<f64>::new(2).expect("test: valid qubit count");
        circuit.h(0).expect("test: valid qubit index");
        circuit.cnot(0, 1).expect("test: valid qubit indices");

        let summary = circuit.summary();
        assert!(summary.contains("qubits=2"));
        assert!(summary.contains("gates=2"));
    }

    #[test]
    fn test_same_qubit_cnot_error() {
        let mut circuit = QuantumCircuit::<f64>::new(2).expect("test: valid qubit count");
        let result = circuit.cnot(0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_phase_and_t_gates() {
        let mut circuit = QuantumCircuit::<f64>::new(1).expect("test: valid qubit count");
        circuit.s(0).expect("test: valid qubit index");
        circuit.t(0).expect("test: valid qubit index");

        assert_eq!(circuit.num_gates(), 2);
    }

    #[test]
    fn test_fredkin_gate() {
        let mut circuit = QuantumCircuit::<f64>::new(3).expect("test: valid qubit count");
        circuit.fredkin(0, 1, 2).expect("test: valid qubit indices");

        assert_eq!(circuit.num_gates(), 1);
    }

    #[test]
    fn test_controlled_u_hadamard() {
        use crate::new_modules::quantum::gates::hadamard;

        let h = hadamard::<f64>().expect("test: valid Hadamard gate");

        let mut circuit = QuantumCircuit::<f64>::new(2).expect("test: valid qubit count");
        circuit.x(0).expect("test: flip qubit 0");
        circuit.x(1).expect("test: flip qubit 1");
        // Controlled-H: control=qubit 1, target=qubit 0
        circuit
            .controlled_u(h, vec![1], vec![0])
            .expect("test: valid controlled_u");

        let state = circuit.execute().expect("test: circuit execution succeeds");

        // |10⟩ is index 2, |11⟩ is index 3 (qubit 1 is MSB)
        let prob_10 = state.get_probability(2).expect("test: valid state index");
        let prob_11 = state.get_probability(3).expect("test: valid state index");

        assert_relative_eq!(prob_10, 0.5, epsilon = 1e-10);
        assert_relative_eq!(prob_11, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_multiqubit_fusion_correctness() {
        // H on q0, then CNOT(control=0, target=1) creates Bell state
        // After fusion, result should be identical to unfused circuit
        let mut circuit_unfused = QuantumCircuit::<f64>::new(2).expect("test: valid qubit count");
        circuit_unfused.h(0).expect("test: valid H gate");
        circuit_unfused.cnot(0, 1).expect("test: valid CNOT gate");
        let state_unfused = circuit_unfused.execute().expect("test: circuit execution");

        let mut circuit_fused = QuantumCircuit::<f64>::new(2).expect("test: valid qubit count");
        circuit_fused.h(0).expect("test: valid H gate");
        circuit_fused.cnot(0, 1).expect("test: valid CNOT gate");
        circuit_fused
            .optimize()
            .expect("test: circuit optimization succeeds");
        let state_fused = circuit_fused.execute().expect("test: circuit execution");

        // Bell state probabilities should match
        let p00_unfused = state_unfused
            .get_probability(0)
            .expect("test: valid state index");
        let p11_unfused = state_unfused
            .get_probability(3)
            .expect("test: valid state index");
        let p00_fused = state_fused
            .get_probability(0)
            .expect("test: valid state index");
        let p11_fused = state_fused
            .get_probability(3)
            .expect("test: valid state index");

        assert_relative_eq!(p00_unfused, p00_fused, epsilon = 1e-10);
        assert_relative_eq!(p11_unfused, p11_fused, epsilon = 1e-10);
        assert_relative_eq!(p00_fused, 0.5, epsilon = 1e-10);
        assert_relative_eq!(p11_fused, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_fusion_reduces_op_count() {
        // 4 consecutive H gates on the same qubit → fuse to 1 op
        let mut circuit = QuantumCircuit::<f64>::new(1).expect("test: valid qubit count");
        circuit.h(0).expect("test: valid gate");
        circuit.h(0).expect("test: valid gate");
        circuit.h(0).expect("test: valid gate");
        circuit.h(0).expect("test: valid gate");

        assert_eq!(circuit.num_gates(), 4);

        circuit
            .optimize()
            .expect("test: circuit optimization succeeds");

        // After fusion: 4 H gates on same qubit collapse to 1 (structurally 1 op)
        assert_eq!(circuit.num_gates(), 1);
    }
}
