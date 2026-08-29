//! State-Space Representation
//!
//! This module provides state-space representation and analysis for control systems.
//!
//! A state-space system is represented as:
//! ```text
//! dx/dt = Ax + Bu  (state equation)
//! y = Cx + Du      (output equation)
//! ```
//! where:
//! - x is the state vector (n×1)
//! - u is the input vector (m×1)
//! - y is the output vector (p×1)
//! - A is the state matrix (n×n)
//! - B is the input matrix (n×m)
//! - C is the output matrix (p×n)
//! - D is the feedthrough matrix (p×m)

use super::{ControlError, ControlResult, SystemType};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::num_complex::Complex64;

/// Type alias for simulation results: (time_points, states, outputs)
type SimulationResult = ControlResult<(Vec<f64>, Vec<Array1<f64>>, Vec<Array1<f64>>)>;

/// State-space system representation
#[derive(Debug, Clone)]
pub struct StateSpace {
    /// State matrix A (n×n)
    a: Array2<f64>,
    /// Input matrix B (n×m)
    b: Array2<f64>,
    /// Output matrix C (p×n)
    c: Array2<f64>,
    /// Feedthrough matrix D (p×m)
    d: Array2<f64>,
    /// System type (continuous or discrete)
    system_type: SystemType,
}

impl StateSpace {
    /// Create a new state-space system
    ///
    /// # Arguments
    /// * `a` - State matrix (n×n)
    /// * `b` - Input matrix (n×m)
    /// * `c` - Output matrix (p×n)
    /// * `d` - Feedthrough matrix (p×m)
    pub fn new(
        a: Array2<f64>,
        b: Array2<f64>,
        c: Array2<f64>,
        d: Array2<f64>,
    ) -> ControlResult<Self> {
        // Validate dimensions
        let (n_a, m_a) = a.dim();
        if n_a != m_a {
            return Err(ControlError::DimensionMismatch {
                expected: "A matrix must be square".to_string(),
                actual: format!("{}×{}", n_a, m_a),
            });
        }

        let (n_b, m_b) = b.dim();
        if n_b != n_a {
            return Err(ControlError::DimensionMismatch {
                expected: format!("B matrix rows must match A dimension ({})", n_a),
                actual: format!("{}", n_b),
            });
        }

        let (p_c, n_c) = c.dim();
        if n_c != n_a {
            return Err(ControlError::DimensionMismatch {
                expected: format!("C matrix columns must match A dimension ({})", n_a),
                actual: format!("{}", n_c),
            });
        }

        let (p_d, m_d) = d.dim();
        if p_d != p_c || m_d != m_b {
            return Err(ControlError::DimensionMismatch {
                expected: format!("D matrix must be {}×{}", p_c, m_b),
                actual: format!("{}×{}", p_d, m_d),
            });
        }

        Ok(Self {
            a,
            b,
            c,
            d,
            system_type: SystemType::Continuous,
        })
    }

    /// Get the state matrix A
    pub fn a(&self) -> &Array2<f64> {
        &self.a
    }

    /// Get the input matrix B
    pub fn b(&self) -> &Array2<f64> {
        &self.b
    }

    /// Get the output matrix C
    pub fn c(&self) -> &Array2<f64> {
        &self.c
    }

    /// Get the feedthrough matrix D
    pub fn d(&self) -> &Array2<f64> {
        &self.d
    }

    /// Get the system type
    pub fn system_type(&self) -> SystemType {
        self.system_type
    }

    /// Get the number of states
    pub fn num_states(&self) -> usize {
        self.a.nrows()
    }

    /// Get the number of inputs
    pub fn num_inputs(&self) -> usize {
        self.b.ncols()
    }

    /// Get the number of outputs
    pub fn num_outputs(&self) -> usize {
        self.c.nrows()
    }

    /// Compute the controllability matrix: [B AB A^2B ... A^(n-1)B]
    pub fn controllability_matrix(&self) -> ControlResult<Array2<f64>> {
        let n = self.num_states();
        let m = self.num_inputs();
        let mut ctrb = Array2::zeros((n, n * m));

        // First block: B
        for i in 0..n {
            for j in 0..m {
                ctrb[[i, j]] = self.b[[i, j]];
            }
        }

        // Subsequent blocks: A^k B
        let mut a_power = self.a.clone();
        for k in 1..n {
            let ab = matrix_multiply(&a_power, &self.b)?;
            for i in 0..n {
                for j in 0..m {
                    ctrb[[i, k * m + j]] = ab[[i, j]];
                }
            }
            if k < n - 1 {
                a_power = matrix_multiply(&a_power, &self.a)?;
            }
        }

        Ok(ctrb)
    }

    /// Test if the system is controllable
    pub fn is_controllable(&self) -> ControlResult<bool> {
        let ctrb = self.controllability_matrix()?;
        let rank = matrix_rank(&ctrb)?;
        Ok(rank == self.num_states())
    }

    /// Compute the observability matrix: [C; CA; CA^2; ...; CA^(n-1)]
    pub fn observability_matrix(&self) -> ControlResult<Array2<f64>> {
        let n = self.num_states();
        let p = self.num_outputs();
        let mut obsv = Array2::zeros((n * p, n));

        // First block: C
        for i in 0..p {
            for j in 0..n {
                obsv[[i, j]] = self.c[[i, j]];
            }
        }

        // Subsequent blocks: CA^k
        let mut a_power = self.a.clone();
        for k in 1..n {
            let ca = matrix_multiply(&self.c, &a_power)?;
            for i in 0..p {
                for j in 0..n {
                    obsv[[k * p + i, j]] = ca[[i, j]];
                }
            }
            if k < n - 1 {
                a_power = matrix_multiply(&a_power, &self.a)?;
            }
        }

        Ok(obsv)
    }

    /// Test if the system is observable
    pub fn is_observable(&self) -> ControlResult<bool> {
        let obsv = self.observability_matrix()?;
        let rank = matrix_rank(&obsv)?;
        Ok(rank == self.num_states())
    }

    /// Compute eigenvalues of the A matrix (system poles)
    pub fn eigenvalues(&self) -> ControlResult<Vec<Complex64>> {
        compute_eigenvalues(&self.a)
    }

    /// Test if the system is stable
    ///
    /// For continuous systems: all eigenvalues must have negative real parts
    /// For discrete systems: all eigenvalues must be inside the unit circle
    pub fn is_stable(&self) -> ControlResult<bool> {
        let eigenvalues = self.eigenvalues()?;

        match self.system_type {
            SystemType::Continuous => Ok(eigenvalues.iter().all(|&e| e.re < 0.0)),
            SystemType::Discrete { .. } => {
                Ok(eigenvalues.iter().all(|&e: &Complex64| e.norm() < 1.0))
            }
        }
    }

    /// Simulate the system response using Euler method
    ///
    /// # Arguments
    /// * `x0` - Initial state vector
    /// * `u_func` - Function providing input at each time step
    /// * `time_span` - (start_time, end_time)
    /// * `dt` - Time step
    pub fn simulate_euler<F>(
        &self,
        x0: &Array1<f64>,
        u_func: F,
        time_span: (f64, f64),
        dt: f64,
    ) -> SimulationResult
    where
        F: Fn(f64) -> Array1<f64>,
    {
        if x0.len() != self.num_states() {
            return Err(ControlError::DimensionMismatch {
                expected: format!("initial state dimension {}", self.num_states()),
                actual: format!("{}", x0.len()),
            });
        }

        if dt <= 0.0 {
            return Err(ControlError::InvalidParameters(
                "Time step must be positive".to_string(),
            ));
        }

        let (t_start, t_end) = time_span;
        let num_steps = ((t_end - t_start) / dt).ceil() as usize;

        let mut time = Vec::with_capacity(num_steps);
        let mut states = Vec::with_capacity(num_steps);
        let mut outputs = Vec::with_capacity(num_steps);

        let mut x = x0.clone();
        let mut t = t_start;

        for _ in 0..num_steps {
            time.push(t);
            states.push(x.clone());

            let u = u_func(t);
            let y = compute_output(&self.c, &self.d, &x, &u)?;
            outputs.push(y);

            // dx/dt = Ax + Bu
            let ax = matrix_vector_multiply(&self.a, &x)?;
            let bu = matrix_vector_multiply(&self.b, &u)?;
            let dx = &ax + &bu;

            // x(t+dt) = x(t) + dx*dt
            x = &x + &(&dx * dt);
            t += dt;
        }

        Ok((time, states, outputs))
    }

    /// Simulate the system response using 4th-order Runge-Kutta method
    pub fn simulate_rk4<F>(
        &self,
        x0: &Array1<f64>,
        u_func: F,
        time_span: (f64, f64),
        dt: f64,
    ) -> SimulationResult
    where
        F: Fn(f64) -> Array1<f64>,
    {
        if x0.len() != self.num_states() {
            return Err(ControlError::DimensionMismatch {
                expected: format!("initial state dimension {}", self.num_states()),
                actual: format!("{}", x0.len()),
            });
        }

        if dt <= 0.0 {
            return Err(ControlError::InvalidParameters(
                "Time step must be positive".to_string(),
            ));
        }

        let (t_start, t_end) = time_span;
        let num_steps = ((t_end - t_start) / dt).ceil() as usize;

        let mut time = Vec::with_capacity(num_steps);
        let mut states = Vec::with_capacity(num_steps);
        let mut outputs = Vec::with_capacity(num_steps);

        let mut x = x0.clone();
        let mut t = t_start;

        for _ in 0..num_steps {
            time.push(t);
            states.push(x.clone());

            let u = u_func(t);
            let y = compute_output(&self.c, &self.d, &x, &u)?;
            outputs.push(y);

            // RK4 method
            let k1 = self.state_derivative(&x, &u)?;
            let k2 = self.state_derivative(&(&x + &(&k1 * (dt / 2.0))), &u_func(t + dt / 2.0))?;
            let k3 = self.state_derivative(&(&x + &(&k2 * (dt / 2.0))), &u_func(t + dt / 2.0))?;
            let k4 = self.state_derivative(&(&x + &(&k3 * dt)), &u_func(t + dt))?;

            x = &x + &((&k1 + &(&k2 * 2.0) + &(&k3 * 2.0) + &k4) * (dt / 6.0));
            t += dt;
        }

        Ok((time, states, outputs))
    }

    /// Compute state derivative: dx/dt = Ax + Bu
    fn state_derivative(&self, x: &Array1<f64>, u: &Array1<f64>) -> ControlResult<Array1<f64>> {
        let ax = matrix_vector_multiply(&self.a, x)?;
        let bu = matrix_vector_multiply(&self.b, u)?;
        Ok(&ax + &bu)
    }

    /// Convert to discrete-time using Zero-Order Hold (ZOH)
    ///
    /// # Arguments
    /// * `sample_time` - Sampling period (seconds)
    pub fn discretize_zoh(&self, sample_time: f64) -> ControlResult<StateSpace> {
        if sample_time <= 0.0 {
            return Err(ControlError::InvalidParameters(
                "Sample time must be positive".to_string(),
            ));
        }

        // Ad = exp(A * Ts)
        let ad = matrix_exponential(&self.a, sample_time)?;

        // Bd = A^(-1) * (Ad - I) * B
        // For numerical stability, we use: Bd = ∫[0,Ts] exp(A*τ) dτ * B
        let bd = compute_bd_zoh(&self.a, &self.b, sample_time)?;

        let mut discrete = StateSpace::new(ad, bd, self.c.clone(), self.d.clone())?;
        discrete.system_type = SystemType::Discrete {
            sample_time: (sample_time * 1_000_000.0) as u64,
        };

        Ok(discrete)
    }

    /// Convert to discrete-time using Tustin (bilinear) transform
    ///
    /// # Arguments
    /// * `sample_time` - Sampling period (seconds)
    pub fn discretize_tustin(&self, sample_time: f64) -> ControlResult<StateSpace> {
        if sample_time <= 0.0 {
            return Err(ControlError::InvalidParameters(
                "Sample time must be positive".to_string(),
            ));
        }

        // Tustin transform: s = 2/Ts * (z-1)/(z+1)
        // Ad = (I + A*Ts/2)^(-1) * (I - A*Ts/2)
        // Bd = (I + A*Ts/2)^(-1) * B * Ts

        let n = self.num_states();
        let eye: Array2<f64> = Array2::eye(n);

        let a_half = &self.a * (sample_time / 2.0);
        let i_plus_a_half = &eye + &a_half;
        let i_minus_a_half = &eye - &a_half;

        let i_plus_a_half_inv = matrix_inverse(&i_plus_a_half)?;
        let ad = matrix_multiply(&i_plus_a_half_inv, &i_minus_a_half)?;
        let bd = matrix_multiply(&i_plus_a_half_inv, &(&self.b * sample_time))?;

        let mut discrete = StateSpace::new(ad, bd, self.c.clone(), self.d.clone())?;
        discrete.system_type = SystemType::Discrete {
            sample_time: (sample_time * 1_000_000.0) as u64,
        };

        Ok(discrete)
    }
}

/// Compute output: y = Cx + Du
fn compute_output(
    c: &Array2<f64>,
    d: &Array2<f64>,
    x: &Array1<f64>,
    u: &Array1<f64>,
) -> ControlResult<Array1<f64>> {
    let cx = matrix_vector_multiply(c, x)?;
    let du = matrix_vector_multiply(d, u)?;
    Ok(&cx + &du)
}

/// Matrix-vector multiplication
fn matrix_vector_multiply(a: &Array2<f64>, v: &Array1<f64>) -> ControlResult<Array1<f64>> {
    let (rows, cols) = a.dim();
    if cols != v.len() {
        return Err(ControlError::DimensionMismatch {
            expected: format!("vector length {}", cols),
            actual: format!("{}", v.len()),
        });
    }

    let mut result = Array1::zeros(rows);
    for i in 0..rows {
        let mut sum = 0.0;
        for j in 0..cols {
            sum += a[[i, j]] * v[j];
        }
        result[i] = sum;
    }

    Ok(result)
}

/// Matrix-matrix multiplication
fn matrix_multiply(a: &Array2<f64>, b: &Array2<f64>) -> ControlResult<Array2<f64>> {
    let (m, n) = a.dim();
    let (p, q) = b.dim();

    if n != p {
        return Err(ControlError::DimensionMismatch {
            expected: format!("second matrix rows {}", n),
            actual: format!("{}", p),
        });
    }

    let mut result = Array2::zeros((m, q));
    for i in 0..m {
        for j in 0..q {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[[i, k]] * b[[k, j]];
            }
            result[[i, j]] = sum;
        }
    }

    Ok(result)
}

/// Compute matrix rank using SVD-like approach
fn matrix_rank(a: &Array2<f64>) -> ControlResult<usize> {
    const TOL: f64 = 1e-10;
    let (m, n) = a.dim();
    let mut u = a.clone();

    // Simple Gaussian elimination to find rank
    let mut rank = 0;
    for col in 0..n.min(m) {
        // Find pivot
        let mut max_row = col;
        let mut max_val = u[[col, col]].abs();

        for row in (col + 1)..m {
            if u[[row, col]].abs() > max_val {
                max_val = u[[row, col]].abs();
                max_row = row;
            }
        }

        if max_val < TOL {
            continue;
        }

        // Swap rows
        if max_row != col {
            for j in 0..n {
                let tmp = u[[col, j]];
                u[[col, j]] = u[[max_row, j]];
                u[[max_row, j]] = tmp;
            }
        }

        rank += 1;

        // Eliminate
        for row in (col + 1)..m {
            let factor = u[[row, col]] / u[[col, col]];
            for j in col..n {
                u[[row, j]] -= factor * u[[col, j]];
            }
        }
    }

    Ok(rank)
}

/// Compute eigenvalues using QR algorithm
fn compute_eigenvalues(a: &Array2<f64>) -> ControlResult<Vec<Complex64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(ControlError::DimensionMismatch {
            expected: "square matrix".to_string(),
            actual: format!("{}×{}", n, a.ncols()),
        });
    }

    const MAX_ITER: usize = 1000;
    const TOL: f64 = 1e-10;

    let mut ak = a.clone();

    // QR algorithm
    for _ in 0..MAX_ITER {
        let (q, r) = qr_decomposition(&ak)?;
        let ak_new = matrix_multiply(&r, &q)?;

        // Check convergence
        let mut converged = true;
        for i in 0..n {
            if (ak_new[[i, i]] - ak[[i, i]]).abs() > TOL {
                converged = false;
                break;
            }
        }

        ak = ak_new;

        if converged {
            break;
        }
    }

    // Extract eigenvalues from diagonal
    let mut eigenvalues = Vec::new();
    let mut i = 0;
    while i < n {
        // Check for 2x2 block (complex conjugate pair)
        if i < n - 1 && ak[[i + 1, i]].abs() > TOL {
            let a_val = ak[[i, i]];
            let b = ak[[i, i + 1]];
            let c = ak[[i + 1, i]];
            let d = ak[[i + 1, i + 1]];

            let trace = a_val + d;
            let det = a_val * d - b * c;
            let discriminant = trace * trace - 4.0 * det;

            if discriminant < 0.0 {
                let real = trace / 2.0;
                let imag = (-discriminant).sqrt() / 2.0;
                eigenvalues.push(Complex64::new(real, imag));
                eigenvalues.push(Complex64::new(real, -imag));
            } else {
                let sqrt_d = discriminant.sqrt();
                eigenvalues.push(Complex64::new((trace + sqrt_d) / 2.0, 0.0));
                eigenvalues.push(Complex64::new((trace - sqrt_d) / 2.0, 0.0));
            }
            i += 2;
        } else {
            eigenvalues.push(Complex64::new(ak[[i, i]], 0.0));
            i += 1;
        }
    }

    Ok(eigenvalues)
}

/// QR decomposition using Gram-Schmidt
fn qr_decomposition(a: &Array2<f64>) -> ControlResult<(Array2<f64>, Array2<f64>)> {
    let (m, n) = a.dim();
    let mut q = Array2::zeros((m, n));
    let mut r = Array2::zeros((n, n));

    for j in 0..n {
        let mut v = a.column(j).to_owned();

        for i in 0..j {
            let q_col = q.column(i);
            let dot_product: f64 = v.iter().zip(q_col.iter()).map(|(&a, &b)| a * b).sum();
            r[[i, j]] = dot_product;

            for k in 0..m {
                v[k] -= r[[i, j]] * q[[k, i]];
            }
        }

        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        r[[j, j]] = norm;

        if norm > 1e-15 {
            for k in 0..m {
                q[[k, j]] = v[k] / norm;
            }
        }
    }

    Ok((q, r))
}

/// Matrix inverse using Gauss-Jordan elimination
fn matrix_inverse(a: &Array2<f64>) -> ControlResult<Array2<f64>> {
    let n = a.nrows();
    if n != a.ncols() {
        return Err(ControlError::LinAlgError(
            "Matrix must be square for inversion".to_string(),
        ));
    }

    // Create augmented matrix [A | I]
    let mut aug = Array2::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    // Forward elimination
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            if aug[[row, col]].abs() > max_val {
                max_val = aug[[row, col]].abs();
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            return Err(ControlError::LinAlgError(
                "Matrix is singular and cannot be inverted".to_string(),
            ));
        }

        // Swap rows
        if max_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        // Scale pivot row
        let pivot = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row != col {
                let factor = aug[[row, col]];
                for j in 0..(2 * n) {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }
    }

    // Extract inverse from right half
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }

    Ok(inv)
}

/// Matrix exponential using Padé approximation
fn matrix_exponential(a: &Array2<f64>, t: f64) -> ControlResult<Array2<f64>> {
    let at = a * t;
    let n = at.nrows();

    // Use Padé approximation with scaling and squaring
    let norm = matrix_norm(&at);
    let j = (norm.log2().ceil().max(0.0)) as i32;
    let scaled = &at / (2.0_f64.powi(j));

    // Padé approximation of order 6
    let eye: Array2<f64> = Array2::eye(n);
    let a2 = matrix_multiply(&scaled, &scaled)?;
    let a4 = matrix_multiply(&a2, &a2)?;
    let a6 = matrix_multiply(&a4, &a2)?;

    let u = &(&(&a6 * 1.0 / 30240.0 + &(&a4 * 1.0 / 840.0)) + &(&a2 * 1.0 / 42.0))
        + &(&scaled * 1.0 / 2.0);
    let u = matrix_multiply(&u, &scaled)?;
    let u = &u + &eye;

    let v = &(&(&a6 * 1.0 / 30240.0 + &(&a4 * 1.0 / 840.0)) + &(&a2 * 1.0 / 42.0))
        - &(&scaled * 1.0 / 2.0);
    let v = matrix_multiply(&v, &scaled)?;
    let v = &v + &eye;

    let v_inv = matrix_inverse(&v)?;
    let mut result = matrix_multiply(&v_inv, &u)?;

    // Squaring
    for _ in 0..j {
        result = matrix_multiply(&result, &result)?;
    }

    Ok(result)
}

/// Compute Bd for ZOH discretization
fn compute_bd_zoh(a: &Array2<f64>, b: &Array2<f64>, ts: f64) -> ControlResult<Array2<f64>> {
    // Bd ≈ (I + A*Ts/2 + A^2*Ts^2/6) * B * Ts
    let n = a.nrows();
    let eye: Array2<f64> = Array2::eye(n);

    let a_ts = a * ts;
    let a2 = matrix_multiply(a, a)?;
    let a2_ts2 = &a2 * (ts * ts / 6.0);

    let approx = &(&eye + &(&a_ts * 0.5)) + &a2_ts2;
    let bd = matrix_multiply(&approx, &(b * ts))?;

    Ok(bd)
}

/// Matrix norm (Frobenius norm)
fn matrix_norm(a: &Array2<f64>) -> f64 {
    a.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_state_space_creation() {
        let a = array![[0.0, 1.0], [-2.0, -3.0]];
        let b = array![[0.0], [1.0]];
        let c = array![[1.0, 0.0]];
        let d = array![[0.0]];

        let ss = StateSpace::new(a, b, c, d);
        assert!(ss.is_ok());
    }

    #[test]
    fn test_invalid_dimensions() {
        let a = array![[0.0, 1.0]]; // Not square
        let b = array![[0.0], [1.0]];
        let c = array![[1.0, 0.0]];
        let d = array![[0.0]];

        let ss = StateSpace::new(a, b, c, d);
        assert!(ss.is_err());
    }

    #[test]
    fn test_controllability_controllable_system() {
        // Controllable system
        let a = array![[0.0, 1.0], [-2.0, -3.0]];
        let b = array![[0.0], [1.0]];
        let c = array![[1.0, 0.0]];
        let d = array![[0.0]];

        let ss = StateSpace::new(a, b, c, d).expect("test: valid state-space system");
        let is_controllable = ss
            .is_controllable()
            .expect("test: valid controllability check");
        assert!(is_controllable);
    }

    #[test]
    fn test_observability_observable_system() {
        // Observable system
        let a = array![[0.0, 1.0], [-2.0, -3.0]];
        let b = array![[0.0], [1.0]];
        let c = array![[1.0, 0.0]];
        let d = array![[0.0]];

        let ss = StateSpace::new(a, b, c, d).expect("test: valid state-space system");
        let is_observable = ss.is_observable().expect("test: valid observability check");
        assert!(is_observable);
    }

    #[test]
    fn test_matrix_multiply() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let c = matrix_multiply(&a, &b).expect("test: valid matrix multiplication");

        assert!((c[[0, 0]] - 19.0).abs() < 1e-10);
        assert!((c[[0, 1]] - 22.0).abs() < 1e-10);
        assert!((c[[1, 0]] - 43.0).abs() < 1e-10);
        assert!((c[[1, 1]] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_vector_multiply() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let v = array![5.0, 6.0];

        let result =
            matrix_vector_multiply(&a, &v).expect("test: valid matrix-vector multiplication");

        assert!((result[0] - 17.0).abs() < 1e-10);
        assert!((result[1] - 39.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_rank() {
        let a = array![[1.0, 2.0], [2.0, 4.0]]; // Rank 1
        let rank = matrix_rank(&a).expect("test: valid matrix rank computation");
        assert_eq!(rank, 1);

        let b = array![[1.0, 0.0], [0.0, 1.0]]; // Rank 2
        let rank = matrix_rank(&b).expect("test: valid matrix rank computation");
        assert_eq!(rank, 2);
    }
}
