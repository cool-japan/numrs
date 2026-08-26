//! State Space Models and Kalman Filtering
//!
//! Implements linear and nonlinear state space models with various Kalman filter variants.

use crate::error::{NumRs2Error, Result};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::thread_rng;

/// Kalman filter state.
#[derive(Debug, Clone)]
pub struct KalmanState {
    /// State estimate
    pub x: Array1<f64>,
    /// State covariance matrix
    pub p: Array2<f64>,
}

/// Linear Kalman Filter for state estimation.
///
/// State space model:
/// - x_{t+1} = F x_t + w_t,  w_t ~ N(0, Q)
/// - y_t = H x_t + v_t,      v_t ~ N(0, R)
#[derive(Debug, Clone)]
pub struct KalmanFilter {
    /// State transition matrix F
    pub f: Array2<f64>,
    /// Observation matrix H
    pub h: Array2<f64>,
    /// Process noise covariance Q
    pub q: Array2<f64>,
    /// Measurement noise covariance R
    pub r: Array2<f64>,
}

impl KalmanFilter {
    /// Create new Kalman filter.
    pub fn new(f: Array2<f64>, h: Array2<f64>, q: Array2<f64>, r: Array2<f64>) -> Result<Self> {
        // Validate dimensions
        let n = f.nrows();
        if f.ncols() != n {
            return Err(NumRs2Error::DimensionMismatch(
                "F must be square".to_string(),
            ));
        }
        if q.nrows() != n || q.ncols() != n {
            return Err(NumRs2Error::DimensionMismatch(
                "Q must match F dimensions".to_string(),
            ));
        }

        Ok(Self { f, h, q, r })
    }

    /// Predict step: x_{t|t-1} = F x_{t-1|t-1}
    pub fn predict(&self, state: &KalmanState) -> Result<KalmanState> {
        let x_pred = self.f.dot(&state.x);
        let p_pred = self.f.dot(&state.p).dot(&self.f.t()) + &self.q;

        Ok(KalmanState {
            x: x_pred,
            p: p_pred,
        })
    }

    /// Update step with new measurement.
    pub fn update(&self, state: &KalmanState, y: &ArrayView1<f64>) -> Result<KalmanState> {
        // Innovation: y - H x
        let y_pred = self.h.dot(&state.x);
        let innovation = y - &y_pred;

        // Innovation covariance: S = H P H' + R
        let s = self.h.dot(&state.p).dot(&self.h.t()) + &self.r;

        // Kalman gain: K = P H' S^{-1}
        let pht = state.p.dot(&self.h.t());
        let s_inv_pht: Array2<f64> = scirs2_linalg::solve_multiple(&s.view(), &pht.view(), None)
            .map_err(|_| NumRs2Error::ComputationError("Singular matrix".to_string()))?;
        let k = s_inv_pht.t().to_owned();

        // Update state: x = x + K * innovation
        let x_updated = &state.x + &k.dot(&innovation);

        // Update covariance: P = (I - K H) P
        let n = state.p.nrows();
        let i = Array2::eye(n);
        let kh = k.dot(&self.h);
        let p_updated = (i - kh).dot(&state.p);

        Ok(KalmanState {
            x: x_updated,
            p: p_updated,
        })
    }

    /// Filter entire time series.
    pub fn filter(
        &self,
        observations: &ArrayView2<f64>,
        initial_state: KalmanState,
    ) -> Result<Vec<KalmanState>> {
        let t = observations.nrows();
        let mut states = Vec::with_capacity(t);
        let mut state = initial_state;

        for i in 0..t {
            let y = observations.row(i);
            state = self.predict(&state)?;
            state = self.update(&state, &y)?;
            states.push(state.clone());
        }

        Ok(states)
    }
}

/// Particle Filter for nonlinear/non-Gaussian state estimation.
#[derive(Debug, Clone)]
pub struct ParticleFilter {
    /// Number of particles
    pub n_particles: usize,
    /// State dimension
    pub state_dim: usize,
}

impl ParticleFilter {
    /// Create new particle filter.
    pub fn new(n_particles: usize, state_dim: usize) -> Self {
        Self {
            n_particles,
            state_dim,
        }
    }

    /// Run particle filter (simplified implementation).
    pub fn filter<F, G>(
        &self,
        observations: &ArrayView2<f64>,
        transition_fn: F,
        observation_fn: G,
        initial_particles: Array2<f64>,
    ) -> Result<Vec<Array1<f64>>>
    where
        F: Fn(&ArrayView1<f64>) -> Array1<f64>,
        G: Fn(&ArrayView1<f64>, &ArrayView1<f64>) -> f64,
    {
        let t = observations.nrows();
        let mut particles = initial_particles;
        let mut estimates = Vec::with_capacity(t);

        for i in 0..t {
            let y = observations.row(i);

            // Predict: apply transition to each particle
            for j in 0..self.n_particles {
                let particle = particles.row(j);
                let new_particle = transition_fn(&particle);
                particles.row_mut(j).assign(&new_particle);
            }

            // Update: compute weights based on observation likelihood
            let mut weights = Array1::zeros(self.n_particles);
            for j in 0..self.n_particles {
                let particle = particles.row(j);
                weights[j] = observation_fn(&particle, &y);
            }

            // Normalize weights
            let sum_weights: f64 = weights.iter().sum();
            if sum_weights > 1e-10 {
                weights /= sum_weights;
            } else {
                weights = Array1::from_elem(self.n_particles, 1.0 / self.n_particles as f64);
            }

            // Compute weighted mean as state estimate
            let mut estimate = Array1::zeros(self.state_dim);
            for j in 0..self.n_particles {
                let particle = particles.row(j);
                estimate += &(weights[j] * &particle.to_owned());
            }
            estimates.push(estimate);

            // Resample particles
            particles = self.resample(&particles.view(), &weights.view())?;
        }

        Ok(estimates)
    }

    /// Systematic resampling of particles.
    fn resample(
        &self,
        particles: &ArrayView2<f64>,
        weights: &ArrayView1<f64>,
    ) -> Result<Array2<f64>> {
        let mut rng = thread_rng();
        let mut new_particles = Array2::zeros((self.n_particles, self.state_dim));

        // Compute cumulative weights
        let mut cumsum = Array1::zeros(self.n_particles);
        cumsum[0] = weights[0];
        for i in 1..self.n_particles {
            cumsum[i] = cumsum[i - 1] + weights[i];
        }

        // Systematic resampling
        let u0: f64 = rng.gen_range(0.0..1.0 / self.n_particles as f64);

        for i in 0..self.n_particles {
            let u = u0 + i as f64 / self.n_particles as f64;

            // Find index where cumsum >= u
            let idx = cumsum
                .iter()
                .position(|&x| x >= u)
                .unwrap_or(self.n_particles - 1);
            new_particles.row_mut(i).assign(&particles.row(idx));
        }

        Ok(new_particles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::{arr2, Array1, Array2};

    #[test]
    fn test_kalman_filter_creation() {
        let f = Array2::eye(2);
        let h = Array2::eye(2);
        let q = Array2::eye(2) * 0.1;
        let r = Array2::eye(2) * 1.0;

        let kf = KalmanFilter::new(f, h, q, r);
        assert!(kf.is_ok());
    }

    #[test]
    fn test_kalman_predict() {
        let f = arr2(&[[1.0, 1.0], [0.0, 1.0]]);
        let h = Array2::eye(2);
        let q = Array2::eye(2) * 0.01;
        let r = Array2::eye(2) * 0.1;

        let kf = KalmanFilter::new(f, h, q, r).expect("creation should succeed");

        let state = KalmanState {
            x: Array1::from_vec(vec![0.0, 1.0]),
            p: Array2::eye(2),
        };

        let pred = kf.predict(&state).expect("predict should succeed");
        assert_relative_eq!(pred.x[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(pred.x[1], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_particle_filter_creation() {
        let pf = ParticleFilter::new(100, 2);
        assert_eq!(pf.n_particles, 100);
        assert_eq!(pf.state_dim, 2);
    }
}
