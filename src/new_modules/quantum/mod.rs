//! Quantum Computing Simulation
//!
//! This module provides comprehensive quantum computing simulation capabilities for NumRS2,
//! including state vector simulation, quantum gates, circuit construction, measurements,
//! and quantum algorithms.
//!
//! # Overview
//!
//! The quantum module enables:
//! - **State Vector Simulation**: Exact simulation of quantum states using complex amplitudes
//! - **Quantum Gates**: Complete set of single and multi-qubit gates
//! - **Circuit Building**: Fluent API for constructing quantum circuits
//! - **Measurements**: Computational and Pauli basis measurements with sampling
//! - **Quantum Algorithms**: QFT, Grover's search, VQE, and QPE implementations
//!
//! # Module Organization
//!
//! - [`statevector`]: Quantum state representation and density matrices
//! - [`gates`]: Standard quantum gates (Pauli, Hadamard, rotations, controlled gates)
//! - [`circuit`]: Quantum circuit builder with optimization
//! - [`measurement`]: Measurement operations and statistics
//! - [`algorithms`]: Quantum algorithms (QFT, Grover, VQE, QPE)
//!
//! # Examples
//!
//! ## Creating a Bell State
//!
//! ```
//! use numrs2::new_modules::quantum::circuit::QuantumCircuit;
//!
//! // Create Bell state (|00⟩ + |11⟩)/√2
//! let mut circuit = QuantumCircuit::<f64>::new(2).expect("valid qubit count");
//! circuit.h(0).expect("valid qubit index");  // Hadamard on qubit 0
//! circuit.cnot(0, 1).expect("valid qubit indices");  // CNOT with control=0, target=1
//!
//! let state = circuit.execute().expect("circuit execution succeeds");
//! ```
//!
//! ## Running Grover's Search
//!
//! ```
//! use numrs2::new_modules::quantum::algorithms::GroverSearch;
//! use numrs2::new_modules::quantum::circuit::QuantumCircuit;
//!
//! // Oracle that marks state |11⟩
//! let oracle = |circuit: &mut QuantumCircuit<f64>| {
//!     circuit.cz(0, 1)?;
//!     Ok(())
//! };
//!
//! let iterations = GroverSearch::optimal_iterations(2, 1);
//! let result_state = GroverSearch::search(2, oracle, iterations).expect("valid Grover search");
//! ```
//!
//! ## Quantum Fourier Transform
//!
//! ```
//! use numrs2::new_modules::quantum::circuit::QuantumCircuit;
//! use numrs2::new_modules::quantum::algorithms::QuantumFourierTransform;
//!
//! let mut circuit = QuantumCircuit::<f64>::new(3).expect("valid qubit count");
//! QuantumFourierTransform::apply(&mut circuit).expect("valid QFT application");
//!
//! let state = circuit.execute().expect("circuit execution succeeds");
//! ```
//!
//! # Mathematical Background
//!
//! ## Quantum States
//!
//! A quantum state of n qubits is represented as a normalized vector in 2ⁿ-dimensional
//! complex Hilbert space:
//!
//! |ψ⟩ = Σᵢ αᵢ|i⟩ where Σᵢ|αᵢ|² = 1
//!
//! ## Quantum Gates
//!
//! Quantum gates are unitary transformations U satisfying U†U = I.
//! Common gates include:
//!
//! - **Pauli-X**: Bit flip, X|0⟩ = |1⟩, X|1⟩ = |0⟩
//! - **Pauli-Y**: Y = iXZ
//! - **Pauli-Z**: Phase flip, Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
//! - **Hadamard**: Creates superposition, H|0⟩ = (|0⟩ + |1⟩)/√2
//! - **CNOT**: Controlled-NOT, entanglement gate
//!
//! ## Measurement
//!
//! Measurement follows the Born rule:
//! P(outcome i) = |⟨i|ψ⟩|² = |αᵢ|²
//!
//! # SCIRS2 Integration
//!
//! This module strictly follows NumRS2's SCIRS2 ecosystem policy:
//! - Complex numbers via `scirs2_core::Complex`
//! - Random number generation via `scirs2_core::random`
//! - Array operations via NumRS2's Array type
//! - No direct external dependencies (ndarray, rand, etc.)
//!
//! # Performance Considerations
//!
//! - State vector simulation is exact but memory-intensive: O(2ⁿ) for n qubits
//! - Practical limit: ~20-25 qubits on typical systems
//! - Gate operations: O(2ⁿ) time complexity
//! - Measurement sampling: O(1) per shot after O(2ⁿ) probability computation
//!
//! # References
//!
//! - Nielsen & Chuang, "Quantum Computation and Quantum Information" (2010)
//! - Grover, "Quantum Mechanics Helps in Searching for a Needle in a Haystack" (1997)
//! - Peruzzo et al., "A variational eigenvalue solver on a photonic quantum processor" (2014)
//! - Shor, "Polynomial-Time Algorithms for Prime Factorization" (1997)

pub mod algorithms;
pub mod circuit;
pub mod gates;
pub mod measurement;
pub mod statevector;
pub mod unitarity;

// Re-export commonly used types for convenience
pub use algorithms::{GroverSearch, QuantumFourierTransform, QuantumPhaseEstimation, VQE};
pub use circuit::QuantumCircuit;
pub use measurement::{Measurement, MeasurementResult, MeasurementStatistics};
pub use statevector::{DensityMatrix, StateVector};
pub use unitarity::{
    create_validated_gate, disable_runtime_validation, enable_runtime_validation, is_unitary,
    unitarity_error, validate_all_standard_gates, validate_gate_unitarity, GateValidationResult,
    StandardGatesValidation,
};
