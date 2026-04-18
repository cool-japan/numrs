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

    /// Optimize the circuit by fusing adjacent gates on the same qubits
    ///
    /// This is a simple optimization that combines consecutive single-qubit gates
    /// on the same qubit into a single gate.
    pub fn optimize(&mut self) -> Result<()> {
        // Simple gate fusion optimization
        let mut optimized_ops = Vec::new();
        let mut i = 0;

        while i < self.operations.len() {
            let current = &self.operations[i];

            // Only fuse single-qubit gates for now
            if current.target_qubits.len() == 1 {
                let qubit = current.target_qubits[0];
                let mut fused_gate = current.gate.clone();
                let mut j = i + 1;

                // Look for consecutive single-qubit gates on the same qubit
                while j < self.operations.len() {
                    let next = &self.operations[j];
                    if next.target_qubits.len() == 1 && next.target_qubits[0] == qubit {
                        // Multiply gates: result = next * fused
                        fused_gate = multiply_2x2_gates(&next.gate, &fused_gate)?;
                        j += 1;
                    } else {
                        break;
                    }
                }

                optimized_ops.push(GateOperation {
                    gate: fused_gate,
                    target_qubits: vec![qubit],
                    name: "Fused".to_string(),
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

/// Multiply two 2×2 gate matrices
fn multiply_2x2_gates<T>(a: &Array<Complex<T>>, b: &Array<Complex<T>>) -> Result<Array<Complex<T>>>
where
    T: Float + Clone + Debug + Into<f64> + From<f64>,
{
    let mut result = vec![Complex::new(T::zero(), T::zero()); 4];

    for i in 0..2 {
        for j in 0..2 {
            let mut sum = Complex::new(T::zero(), T::zero());
            for k in 0..2 {
                let a_ik = a.get(&[i, k]).map_err(|_| {
                    NumRs2Error::IndexOutOfBounds("Invalid gate access".to_string())
                })?;
                let b_kj = b.get(&[k, j]).map_err(|_| {
                    NumRs2Error::IndexOutOfBounds("Invalid gate access".to_string())
                })?;
                sum = sum + a_ik * b_kj;
            }
            result[i * 2 + j] = sum;
        }
    }

    Ok(Array::from_vec(result).reshape(&[2, 2]))
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
}
