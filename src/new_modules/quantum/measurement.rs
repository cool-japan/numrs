//! Quantum Measurement Operations
//!
//! This module provides measurement operations for quantum states, including:
//! - Computational basis measurements
//! - Pauli basis measurements
//! - Measurement statistics and sampling
//!
//! # Mathematical Background
//!
//! Measurement in quantum mechanics is a projective operation that collapses
//! the quantum state according to the Born rule: P(outcome) = |⟨outcome|ψ⟩|²

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use crate::new_modules::quantum::statevector::StateVector;
use num_traits::Float;
use scirs2_core::random::{Rng, RngExt, SeedableRng, StdRng};
use scirs2_core::Complex;
use std::collections::HashMap;
use std::fmt::Debug;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Result of a single measurement
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeasurementResult {
    /// Measured bit string (as integer)
    pub outcome: usize,
    /// Number of qubits measured
    pub num_qubits: usize,
}

impl MeasurementResult {
    /// Get the outcome as a binary string
    pub fn as_binary_string(&self) -> String {
        format!("{:0width$b}", self.outcome, width = self.num_qubits)
    }

    /// Get the bit value of a specific qubit
    pub fn get_bit(&self, qubit: usize) -> Result<u8> {
        if qubit >= self.num_qubits {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Qubit {} out of bounds",
                qubit
            )));
        }
        Ok(((self.outcome >> qubit) & 1) as u8)
    }
}

/// Measurement statistics from multiple shots
#[derive(Clone, Debug)]
pub struct MeasurementStatistics {
    /// Counts for each outcome
    pub counts: HashMap<usize, usize>,
    /// Total number of shots
    pub total_shots: usize,
    /// Number of qubits
    pub num_qubits: usize,
}

impl MeasurementStatistics {
    /// Create new measurement statistics
    pub fn new(num_qubits: usize) -> Self {
        Self {
            counts: HashMap::new(),
            total_shots: 0,
            num_qubits,
        }
    }

    /// Add a measurement result
    pub fn add_result(&mut self, outcome: usize) {
        *self.counts.entry(outcome).or_insert(0) += 1;
        self.total_shots += 1;
    }

    /// Get probability of an outcome
    pub fn get_probability(&self, outcome: usize) -> f64 {
        if self.total_shots == 0 {
            return 0.0;
        }
        let count = self.counts.get(&outcome).unwrap_or(&0);
        *count as f64 / self.total_shots as f64
    }

    /// Get all probabilities
    pub fn get_probabilities(&self) -> HashMap<usize, f64> {
        let mut probs = HashMap::new();
        for (&outcome, &count) in &self.counts {
            probs.insert(outcome, count as f64 / self.total_shots as f64);
        }
        probs
    }

    /// Get the most frequent outcome
    pub fn most_frequent(&self) -> Option<usize> {
        self.counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&outcome, _)| outcome)
    }

    /// Get entropy of the measurement distribution
    pub fn entropy(&self) -> f64 {
        if self.total_shots == 0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &count in self.counts.values() {
            if count > 0 {
                let p = count as f64 / self.total_shots as f64;
                entropy -= p * p.log2();
            }
        }
        entropy
    }
}

/// Measurement operations on quantum states
pub struct Measurement;

impl Measurement {
    /// Measure all qubits in computational basis
    ///
    /// # Arguments
    ///
    /// * `state` - Quantum state to measure
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Measurement result and post-measurement state
    pub fn measure_all<T>(
        state: &StateVector<T>,
        seed: Option<u64>,
    ) -> Result<(MeasurementResult, StateVector<T>)>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_else(|_| Duration::from_secs(1));
                StdRng::seed_from_u64(now.as_secs())
            }
        };

        // Get probabilities
        let probs = state.get_probabilities();
        let prob_vec = probs.to_vec();
        let prob_vec_f64: Vec<f64> = prob_vec.iter().map(|p| (*p).into()).collect();

        // Sample from probability distribution
        let outcome = sample_discrete(&mut rng, &prob_vec_f64)?;

        // Create post-measurement state (collapsed to measured state)
        let dim = state.dim();
        let mut post_amplitudes = vec![Complex::new(T::zero(), T::zero()); dim];
        post_amplitudes[outcome] = Complex::new(T::one(), T::zero());

        let post_state = StateVector::from_amplitudes(post_amplitudes)?;

        Ok((
            MeasurementResult {
                outcome,
                num_qubits: state.num_qubits(),
            },
            post_state,
        ))
    }

    /// Measure specific qubits in computational basis
    ///
    /// # Arguments
    ///
    /// * `state` - Quantum state to measure
    /// * `qubits` - Qubits to measure
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Measurement results for specified qubits
    pub fn measure_qubits<T>(
        state: &StateVector<T>,
        qubits: &[usize],
        seed: Option<u64>,
    ) -> Result<Vec<u8>>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        // Validate qubits
        for &qubit in qubits {
            if qubit >= state.num_qubits() {
                return Err(NumRs2Error::IndexOutOfBounds(format!(
                    "Qubit {} out of bounds",
                    qubit
                )));
            }
        }

        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_else(|_| Duration::from_secs(1));
                StdRng::seed_from_u64(now.as_secs())
            }
        };

        let mut results = Vec::new();

        // Measure each qubit independently
        for &qubit in qubits {
            // Calculate probability of measuring |1⟩
            let mut prob_one = T::zero();
            for i in 0..state.dim() {
                if (i >> qubit) & 1 == 1 {
                    prob_one = prob_one + state.get_probability(i)?;
                }
            }

            let prob_one_f64: f64 = prob_one.into();
            let random_val: f64 = rng.random();

            results.push(if random_val < prob_one_f64 { 1 } else { 0 });
        }

        Ok(results)
    }

    /// Perform shot-based sampling
    ///
    /// # Arguments
    ///
    /// * `state` - Quantum state to sample from
    /// * `num_shots` - Number of measurement shots
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    ///
    /// Measurement statistics over all shots
    pub fn sample<T>(
        state: &StateVector<T>,
        num_shots: usize,
        seed: Option<u64>,
    ) -> Result<MeasurementStatistics>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_else(|_| Duration::from_secs(1));
                StdRng::seed_from_u64(now.as_secs())
            }
        };

        let mut stats = MeasurementStatistics::new(state.num_qubits());
        let probs = state.get_probabilities();
        let prob_vec = probs.to_vec();
        let prob_vec_f64: Vec<f64> = prob_vec.iter().map(|p| (*p).into()).collect();

        for _ in 0..num_shots {
            let outcome = sample_discrete(&mut rng, &prob_vec_f64)?;
            stats.add_result(outcome);
        }

        Ok(stats)
    }

    /// Measure in Pauli-X basis
    ///
    /// Transforms to X-basis before measurement: |±⟩ = (|0⟩ ± |1⟩)/√2
    ///
    /// # Arguments
    ///
    /// * `state` - Quantum state
    /// * `qubit` - Qubit to measure
    /// * `seed` - Random seed
    ///
    /// # Returns
    ///
    /// Measurement outcome (0 for |+⟩, 1 for |-⟩)
    pub fn measure_x<T>(state: &StateVector<T>, qubit: usize, seed: Option<u64>) -> Result<u8>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        if qubit >= state.num_qubits() {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Qubit {} out of bounds",
                qubit
            )));
        }

        // Apply Hadamard to transform to X basis, then measure in Z basis
        let mut state_copy = state.clone();
        let h_gate = crate::new_modules::quantum::gates::hadamard()?;
        state_copy.apply_gate(&h_gate, &[qubit])?;

        let results = Self::measure_qubits(&state_copy, &[qubit], seed)?;
        Ok(results[0])
    }

    /// Measure in Pauli-Y basis
    ///
    /// # Arguments
    ///
    /// * `state` - Quantum state
    /// * `qubit` - Qubit to measure
    /// * `seed` - Random seed
    ///
    /// # Returns
    ///
    /// Measurement outcome
    pub fn measure_y<T>(state: &StateVector<T>, qubit: usize, seed: Option<u64>) -> Result<u8>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        if qubit >= state.num_qubits() {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Qubit {} out of bounds",
                qubit
            )));
        }

        // Apply S†H to transform to Y basis
        let mut state_copy = state.clone();

        // S† = S^3 for 2×2 unitary
        let s_gate = crate::new_modules::quantum::gates::phase_gate()?;
        state_copy.apply_gate(&s_gate, &[qubit])?;
        state_copy.apply_gate(&s_gate, &[qubit])?;
        state_copy.apply_gate(&s_gate, &[qubit])?;

        let h_gate = crate::new_modules::quantum::gates::hadamard()?;
        state_copy.apply_gate(&h_gate, &[qubit])?;

        let results = Self::measure_qubits(&state_copy, &[qubit], seed)?;
        Ok(results[0])
    }

    /// Measure in Pauli-Z basis (standard computational basis)
    ///
    /// # Arguments
    ///
    /// * `state` - Quantum state
    /// * `qubit` - Qubit to measure
    /// * `seed` - Random seed
    ///
    /// # Returns
    ///
    /// Measurement outcome (0 or 1)
    pub fn measure_z<T>(state: &StateVector<T>, qubit: usize, seed: Option<u64>) -> Result<u8>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        let results = Self::measure_qubits(state, &[qubit], seed)?;
        Ok(results[0])
    }

    /// Calculate expectation value of Pauli-Z on a qubit
    ///
    /// ⟨Z⟩ = P(0) - P(1)
    ///
    /// # Arguments
    ///
    /// * `state` - Quantum state
    /// * `qubit` - Qubit to measure
    ///
    /// # Returns
    ///
    /// Expectation value in range [-1, 1]
    pub fn expectation_z<T>(state: &StateVector<T>, qubit: usize) -> Result<f64>
    where
        T: Float + Clone + Debug + Into<f64> + From<f64>,
    {
        if qubit >= state.num_qubits() {
            return Err(NumRs2Error::IndexOutOfBounds(format!(
                "Qubit {} out of bounds",
                qubit
            )));
        }

        let mut prob_zero = 0.0;
        let mut prob_one = 0.0;

        for i in 0..state.dim() {
            let prob: f64 = state.get_probability(i)?.into();
            if (i >> qubit) & 1 == 0 {
                prob_zero += prob;
            } else {
                prob_one += prob;
            }
        }

        Ok(prob_zero - prob_one)
    }
}

/// Sample from a discrete probability distribution
fn sample_discrete<R: Rng>(rng: &mut R, probabilities: &[f64]) -> Result<usize> {
    let random_val: f64 = rng.random();
    let mut cumulative = 0.0;

    for (i, &prob) in probabilities.iter().enumerate() {
        cumulative += prob;
        if random_val < cumulative {
            return Ok(i);
        }
    }

    // Due to floating point errors, return last index
    Ok(probabilities.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_measurement_result_binary_string() {
        let result = MeasurementResult {
            outcome: 5,
            num_qubits: 3,
        };
        assert_eq!(result.as_binary_string(), "101");
    }

    #[test]
    fn test_measurement_result_get_bit() {
        let result = MeasurementResult {
            outcome: 6, // 110 in binary
            num_qubits: 3,
        };
        assert_eq!(result.get_bit(0).unwrap(), 0);
        assert_eq!(result.get_bit(1).unwrap(), 1);
        assert_eq!(result.get_bit(2).unwrap(), 1);
    }

    #[test]
    fn test_measure_all_zero_state() {
        let state = StateVector::<f64>::new(2).unwrap();
        let (result, _) = Measurement::measure_all(&state, Some(42)).unwrap();

        assert_eq!(result.outcome, 0);
        assert_eq!(result.num_qubits, 2);
    }

    #[test]
    fn test_measure_all_superposition() {
        // Create |+⟩ state
        let amplitudes = vec![
            Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        let state = StateVector::from_amplitudes(amplitudes).unwrap();

        // Measure many times
        let mut zeros = 0;
        let mut ones = 0;

        for i in 0..100 {
            let (result, _) = Measurement::measure_all(&state, Some(i as u64)).unwrap();
            if result.outcome == 0 {
                zeros += 1;
            } else {
                ones += 1;
            }
        }

        // Should be roughly 50/50
        assert!(zeros > 30 && zeros < 70);
        assert!(ones > 30 && ones < 70);
    }

    #[test]
    fn test_sample() {
        let amplitudes = vec![
            Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ];
        let state = StateVector::from_amplitudes(amplitudes).unwrap();

        let stats = Measurement::sample(&state, 1000, Some(42)).unwrap();

        assert_eq!(stats.total_shots, 1000);

        let prob_0 = stats.get_probability(0);
        let prob_1 = stats.get_probability(1);

        // Should be close to 0.5 each
        assert_relative_eq!(prob_0, 0.5, epsilon = 0.1);
        assert_relative_eq!(prob_1, 0.5, epsilon = 0.1);
    }

    #[test]
    fn test_measurement_statistics() {
        let mut stats = MeasurementStatistics::new(2);

        stats.add_result(0);
        stats.add_result(0);
        stats.add_result(1);
        stats.add_result(3);

        assert_eq!(stats.total_shots, 4);
        assert_eq!(stats.get_probability(0), 0.5);
        assert_eq!(stats.get_probability(1), 0.25);
        assert_eq!(stats.most_frequent(), Some(0));
    }

    #[test]
    fn test_entropy() {
        let mut stats = MeasurementStatistics::new(2);

        // Uniform distribution
        stats.add_result(0);
        stats.add_result(1);
        stats.add_result(2);
        stats.add_result(3);

        let entropy = stats.entropy();
        assert_relative_eq!(entropy, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_measure_qubits() {
        let state = StateVector::<f64>::new(3).unwrap();
        let results = Measurement::measure_qubits(&state, &[0, 1, 2], Some(42)).unwrap();

        assert_eq!(results.len(), 3);
        // Should all be 0 for |000⟩ state
        assert_eq!(results, vec![0, 0, 0]);
    }

    #[test]
    fn test_measure_x_basis() {
        // |0⟩ state
        let state = StateVector::<f64>::new(1).unwrap();

        // |0⟩ in X basis is |+⟩, should measure 0 half the time
        let mut count_0 = 0;
        for i in 0..100 {
            if Measurement::measure_x(&state, 0, Some(i)).unwrap() == 0 {
                count_0 += 1;
            }
        }

        assert!(count_0 > 30 && count_0 < 70);
    }

    #[test]
    fn test_expectation_z() {
        // |0⟩ state
        let state = StateVector::<f64>::new(1).unwrap();
        let exp = Measurement::expectation_z(&state, 0).unwrap();
        assert_relative_eq!(exp, 1.0, epsilon = 1e-10);

        // |1⟩ state
        let amplitudes = vec![Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)];
        let state = StateVector::from_amplitudes(amplitudes).unwrap();
        let exp = Measurement::expectation_z(&state, 0).unwrap();
        assert_relative_eq!(exp, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_invalid_qubit_measurement() {
        let state = StateVector::<f64>::new(2).unwrap();
        let result = Measurement::measure_qubits(&state, &[5], Some(42));
        assert!(result.is_err());
    }
}
