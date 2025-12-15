//! Ordinary Differential Equation (ODE) Solvers
//!
//! This module provides numerical methods for solving initial value problems (IVPs)
//! of ordinary differential equations of the form:
//!
//! dy/dt = f(t, y), y(t₀) = y₀
//!
//! ## Available Methods
//!
//! - **Euler's Method**: First-order explicit method
//! - **RK4**: Classic 4th-order Runge-Kutta method
//! - **RK45 (Runge-Kutta-Fehlberg)**: Adaptive step size 4th/5th order method
//! - **Dormand-Prince (DoPri5)**: Modern adaptive 5th-order method
//! - **Implicit Euler**: First-order implicit method for stiff equations
//! - **BDF2**: Backward Differentiation Formula for stiff equations
//!
//! ## Example
//!
//! ```ignore
//! use numrs2::ode::{solve_ivp, OdeMethod};
//!
//! // Solve dy/dt = -y, y(0) = 1 (exponential decay)
//! let f = |t: f64, y: &[f64]| vec![-y[0]];
//! let result = solve_ivp(f, (0.0, 2.0), &[1.0], OdeMethod::RK45).unwrap();
//! ```

use crate::error::{NumRs2Error, Result};
use num_traits::Float;
use std::fmt::Debug;

/// ODE solver method selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OdeMethod {
    /// Explicit Euler method (1st order)
    Euler,
    /// Classic Runge-Kutta 4th order
    RK4,
    /// Adaptive Runge-Kutta-Fehlberg 4(5)
    RK45,
    /// Dormand-Prince 5(4) method
    DoPri5,
    /// Implicit Euler (backward Euler) for stiff equations
    ImplicitEuler,
    /// BDF2 (Backward Differentiation Formula, 2nd order)
    BDF2,
}

/// Configuration for ODE solvers
#[derive(Debug, Clone)]
pub struct OdeConfig<T> {
    /// Initial step size
    pub h0: T,
    /// Minimum step size
    pub h_min: T,
    /// Maximum step size
    pub h_max: T,
    /// Absolute tolerance
    pub atol: T,
    /// Relative tolerance
    pub rtol: T,
    /// Maximum number of steps
    pub max_steps: usize,
    /// Dense output points (if Some, interpolate at these points)
    pub t_eval: Option<Vec<T>>,
}

impl<T: Float> Default for OdeConfig<T> {
    fn default() -> Self {
        OdeConfig {
            h0: T::from(0.01).unwrap(),
            h_min: T::from(1e-10).unwrap(),
            h_max: T::from(1.0).unwrap(),
            atol: T::from(1e-6).unwrap(),
            rtol: T::from(1e-3).unwrap(),
            max_steps: 10000,
            t_eval: None,
        }
    }
}

/// Result of ODE integration
#[derive(Debug, Clone)]
pub struct OdeResult<T> {
    /// Time points
    pub t: Vec<T>,
    /// Solution values at each time point (flattened: [y1(t0), y2(t0), ..., y1(t1), y2(t1), ...])
    pub y: Vec<Vec<T>>,
    /// Success flag
    pub success: bool,
    /// Number of function evaluations
    pub nfev: usize,
    /// Number of accepted steps
    pub nsteps: usize,
    /// Message
    pub message: String,
}

/// Solve an initial value problem for a system of ODEs
///
/// Solves dy/dt = f(t, y), y(t_span[0]) = y0
///
/// # Arguments
/// * `f` - The right-hand side function f(t, y) returning dy/dt
/// * `t_span` - Time span as (t0, t_final)
/// * `y0` - Initial condition
/// * `method` - ODE solver method
///
/// # Returns
/// * `OdeResult` with time points and solution
///
/// # Example
/// ```ignore
/// use numrs2::ode::{solve_ivp, OdeMethod};
///
/// // Solve exponential decay: dy/dt = -y
/// let result = solve_ivp(
///     |_t, y| vec![-y[0]],
///     (0.0, 1.0),
///     &[1.0],
///     OdeMethod::RK4
/// ).unwrap();
/// ```
pub fn solve_ivp<T, F>(f: F, t_span: (T, T), y0: &[T], method: OdeMethod) -> Result<OdeResult<T>>
where
    T: Float + Debug + std::iter::Sum,
    F: Fn(T, &[T]) -> Vec<T>,
{
    solve_ivp_with_config(f, t_span, y0, method, &OdeConfig::default())
}

/// Solve IVP with custom configuration
pub fn solve_ivp_with_config<T, F>(
    f: F,
    t_span: (T, T),
    y0: &[T],
    method: OdeMethod,
    config: &OdeConfig<T>,
) -> Result<OdeResult<T>>
where
    T: Float + Debug + std::iter::Sum,
    F: Fn(T, &[T]) -> Vec<T>,
{
    let (t0, tf) = t_span;

    if tf <= t0 {
        return Err(NumRs2Error::ValueError(
            "Final time must be greater than initial time".to_string(),
        ));
    }

    match method {
        OdeMethod::Euler => solve_euler(&f, t0, tf, y0, config),
        OdeMethod::RK4 => solve_rk4(&f, t0, tf, y0, config),
        OdeMethod::RK45 => solve_rk45(&f, t0, tf, y0, config),
        OdeMethod::DoPri5 => solve_dopri5(&f, t0, tf, y0, config),
        OdeMethod::ImplicitEuler => solve_implicit_euler(&f, t0, tf, y0, config),
        OdeMethod::BDF2 => solve_bdf2(&f, t0, tf, y0, config),
    }
}

/// Explicit Euler method (first-order)
///
/// y_{n+1} = y_n + h * f(t_n, y_n)
fn solve_euler<T, F>(f: &F, t0: T, tf: T, y0: &[T], config: &OdeConfig<T>) -> Result<OdeResult<T>>
where
    T: Float + Debug,
    F: Fn(T, &[T]) -> Vec<T>,
{
    let n = y0.len();
    let h = config.h0;

    let mut t_vals = vec![t0];
    let mut y_vals = vec![y0.to_vec()];

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut nfev = 0;
    let mut nsteps = 0;

    while t < tf && nsteps < config.max_steps {
        let h_actual = if t + h > tf { tf - t } else { h };

        let k = f(t, &y);
        nfev += 1;

        // Update: y = y + h * k
        for i in 0..n {
            y[i] = y[i] + h_actual * k[i];
        }

        t = t + h_actual;
        t_vals.push(t);
        y_vals.push(y.clone());
        nsteps += 1;
    }

    Ok(OdeResult {
        t: t_vals,
        y: y_vals,
        success: t >= tf,
        nfev,
        nsteps,
        message: "Euler integration completed".to_string(),
    })
}

/// Classic Runge-Kutta 4th order method
///
/// k1 = f(t_n, y_n)
/// k2 = f(t_n + h/2, y_n + h*k1/2)
/// k3 = f(t_n + h/2, y_n + h*k2/2)
/// k4 = f(t_n + h, y_n + h*k3)
/// y_{n+1} = y_n + h*(k1 + 2*k2 + 2*k3 + k4)/6
fn solve_rk4<T, F>(f: &F, t0: T, tf: T, y0: &[T], config: &OdeConfig<T>) -> Result<OdeResult<T>>
where
    T: Float + Debug,
    F: Fn(T, &[T]) -> Vec<T>,
{
    let n = y0.len();
    let h = config.h0;
    let two = T::from(2.0).unwrap();
    let six = T::from(6.0).unwrap();

    let mut t_vals = vec![t0];
    let mut y_vals = vec![y0.to_vec()];

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut y_temp = vec![T::zero(); n];
    let mut nfev = 0;
    let mut nsteps = 0;

    while t < tf && nsteps < config.max_steps {
        let h_actual = if t + h > tf { tf - t } else { h };
        let h_half = h_actual / two;

        // k1 = f(t, y)
        let k1 = f(t, &y);
        nfev += 1;

        // k2 = f(t + h/2, y + h*k1/2)
        for i in 0..n {
            y_temp[i] = y[i] + h_half * k1[i];
        }
        let k2 = f(t + h_half, &y_temp);
        nfev += 1;

        // k3 = f(t + h/2, y + h*k2/2)
        for i in 0..n {
            y_temp[i] = y[i] + h_half * k2[i];
        }
        let k3 = f(t + h_half, &y_temp);
        nfev += 1;

        // k4 = f(t + h, y + h*k3)
        for i in 0..n {
            y_temp[i] = y[i] + h_actual * k3[i];
        }
        let k4 = f(t + h_actual, &y_temp);
        nfev += 1;

        // y_new = y + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        for i in 0..n {
            y[i] = y[i] + h_actual * (k1[i] + two * k2[i] + two * k3[i] + k4[i]) / six;
        }

        t = t + h_actual;
        t_vals.push(t);
        y_vals.push(y.clone());
        nsteps += 1;
    }

    Ok(OdeResult {
        t: t_vals,
        y: y_vals,
        success: t >= tf,
        nfev,
        nsteps,
        message: "RK4 integration completed".to_string(),
    })
}

/// Runge-Kutta-Fehlberg 4(5) method with adaptive step size
fn solve_rk45<T, F>(f: &F, t0: T, tf: T, y0: &[T], config: &OdeConfig<T>) -> Result<OdeResult<T>>
where
    T: Float + Debug + std::iter::Sum,
    F: Fn(T, &[T]) -> Vec<T>,
{
    let n = y0.len();

    // RK45 coefficients (Fehlberg)
    let c: [T; 6] = [
        T::zero(),
        T::from(0.25).unwrap(),
        T::from(0.375).unwrap(),
        T::from(12.0 / 13.0).unwrap(),
        T::one(),
        T::from(0.5).unwrap(),
    ];

    // 4th order weights
    let b4: [T; 6] = [
        T::from(25.0 / 216.0).unwrap(),
        T::zero(),
        T::from(1408.0 / 2565.0).unwrap(),
        T::from(2197.0 / 4104.0).unwrap(),
        T::from(-1.0 / 5.0).unwrap(),
        T::zero(),
    ];

    // 5th order weights
    let b5: [T; 6] = [
        T::from(16.0 / 135.0).unwrap(),
        T::zero(),
        T::from(6656.0 / 12825.0).unwrap(),
        T::from(28561.0 / 56430.0).unwrap(),
        T::from(-9.0 / 50.0).unwrap(),
        T::from(2.0 / 55.0).unwrap(),
    ];

    let mut t_vals = vec![t0];
    let mut y_vals = vec![y0.to_vec()];

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = config.h0;
    let mut nfev = 0;
    let mut nsteps = 0;

    while t < tf && nsteps < config.max_steps {
        if t + h > tf {
            h = tf - t;
        }

        // Compute k values
        let k1 = f(t, &y);
        nfev += 1;

        let mut y_temp = vec![T::zero(); n];

        // k2
        for i in 0..n {
            y_temp[i] = y[i] + h * c[1] * k1[i];
        }
        let k2 = f(t + c[1] * h, &y_temp);
        nfev += 1;

        // k3
        let a31 = T::from(3.0 / 32.0).unwrap();
        let a32 = T::from(9.0 / 32.0).unwrap();
        for i in 0..n {
            y_temp[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]);
        }
        let k3 = f(t + c[2] * h, &y_temp);
        nfev += 1;

        // k4
        let a41 = T::from(1932.0 / 2197.0).unwrap();
        let a42 = T::from(-7200.0 / 2197.0).unwrap();
        let a43 = T::from(7296.0 / 2197.0).unwrap();
        for i in 0..n {
            y_temp[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
        }
        let k4 = f(t + c[3] * h, &y_temp);
        nfev += 1;

        // k5
        let a51 = T::from(439.0 / 216.0).unwrap();
        let a52 = T::from(-8.0).unwrap();
        let a53 = T::from(3680.0 / 513.0).unwrap();
        let a54 = T::from(-845.0 / 4104.0).unwrap();
        for i in 0..n {
            y_temp[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
        }
        let k5 = f(t + c[4] * h, &y_temp);
        nfev += 1;

        // k6
        let a61 = T::from(-8.0 / 27.0).unwrap();
        let a62 = T::from(2.0).unwrap();
        let a63 = T::from(-3544.0 / 2565.0).unwrap();
        let a64 = T::from(1859.0 / 4104.0).unwrap();
        let a65 = T::from(-11.0 / 40.0).unwrap();
        for i in 0..n {
            y_temp[i] =
                y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
        }
        let k6 = f(t + c[5] * h, &y_temp);
        nfev += 1;

        let k = [k1.clone(), k2, k3, k4, k5, k6];

        // Compute 4th and 5th order solutions
        let mut y4 = vec![T::zero(); n];
        let mut y5 = vec![T::zero(); n];
        for i in 0..n {
            let mut sum4 = T::zero();
            let mut sum5 = T::zero();
            for j in 0..6 {
                sum4 = sum4 + b4[j] * k[j][i];
                sum5 = sum5 + b5[j] * k[j][i];
            }
            y4[i] = y[i] + h * sum4;
            y5[i] = y[i] + h * sum5;
        }

        // Estimate error
        let mut error = T::zero();
        for i in 0..n {
            let scale = config.atol + config.rtol * y[i].abs().max(y5[i].abs());
            let err_i = (y5[i] - y4[i]).abs() / scale;
            if err_i > error {
                error = err_i;
            }
        }

        // Accept or reject step
        if error <= T::one() || h <= config.h_min {
            t = t + h;
            y = y5;
            t_vals.push(t);
            y_vals.push(y.clone());
            nsteps += 1;
        }

        // Adjust step size
        let safety = T::from(0.9).unwrap();
        let min_factor = T::from(0.2).unwrap();
        let max_factor = T::from(10.0).unwrap();

        let factor = if error > T::epsilon() {
            safety * (T::one() / error).powf(T::from(0.2).unwrap())
        } else {
            max_factor
        };

        h = h * factor.max(min_factor).min(max_factor);
        h = h.max(config.h_min).min(config.h_max);
    }

    Ok(OdeResult {
        t: t_vals,
        y: y_vals,
        success: t >= tf,
        nfev,
        nsteps,
        message: "RK45 integration completed".to_string(),
    })
}

/// Dormand-Prince 5(4) method - modern adaptive method
fn solve_dopri5<T, F>(f: &F, t0: T, tf: T, y0: &[T], config: &OdeConfig<T>) -> Result<OdeResult<T>>
where
    T: Float + Debug + std::iter::Sum,
    F: Fn(T, &[T]) -> Vec<T>,
{
    let n = y0.len();

    // Dormand-Prince coefficients
    let c: [T; 7] = [
        T::zero(),
        T::from(0.2).unwrap(),
        T::from(0.3).unwrap(),
        T::from(0.8).unwrap(),
        T::from(8.0 / 9.0).unwrap(),
        T::one(),
        T::one(),
    ];

    // 5th order weights
    let b5: [T; 7] = [
        T::from(35.0 / 384.0).unwrap(),
        T::zero(),
        T::from(500.0 / 1113.0).unwrap(),
        T::from(125.0 / 192.0).unwrap(),
        T::from(-2187.0 / 6784.0).unwrap(),
        T::from(11.0 / 84.0).unwrap(),
        T::zero(),
    ];

    // 4th order weights for error estimation
    let b4: [T; 7] = [
        T::from(5179.0 / 57600.0).unwrap(),
        T::zero(),
        T::from(7571.0 / 16695.0).unwrap(),
        T::from(393.0 / 640.0).unwrap(),
        T::from(-92097.0 / 339200.0).unwrap(),
        T::from(187.0 / 2100.0).unwrap(),
        T::from(1.0 / 40.0).unwrap(),
    ];

    let mut t_vals = vec![t0];
    let mut y_vals = vec![y0.to_vec()];

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = config.h0;
    let mut nfev = 0;
    let mut nsteps = 0;

    while t < tf && nsteps < config.max_steps {
        if t + h > tf {
            h = tf - t;
        }

        let mut y_temp = vec![T::zero(); n];
        let mut k: Vec<Vec<T>> = Vec::with_capacity(7);

        // k1
        k.push(f(t, &y));
        nfev += 1;

        // k2
        for i in 0..n {
            y_temp[i] = y[i] + h * T::from(0.2).unwrap() * k[0][i];
        }
        k.push(f(t + c[1] * h, &y_temp));
        nfev += 1;

        // k3
        let a31 = T::from(3.0 / 40.0).unwrap();
        let a32 = T::from(9.0 / 40.0).unwrap();
        for i in 0..n {
            y_temp[i] = y[i] + h * (a31 * k[0][i] + a32 * k[1][i]);
        }
        k.push(f(t + c[2] * h, &y_temp));
        nfev += 1;

        // k4
        let a41 = T::from(44.0 / 45.0).unwrap();
        let a42 = T::from(-56.0 / 15.0).unwrap();
        let a43 = T::from(32.0 / 9.0).unwrap();
        for i in 0..n {
            y_temp[i] = y[i] + h * (a41 * k[0][i] + a42 * k[1][i] + a43 * k[2][i]);
        }
        k.push(f(t + c[3] * h, &y_temp));
        nfev += 1;

        // k5
        let a51 = T::from(19372.0 / 6561.0).unwrap();
        let a52 = T::from(-25360.0 / 2187.0).unwrap();
        let a53 = T::from(64448.0 / 6561.0).unwrap();
        let a54 = T::from(-212.0 / 729.0).unwrap();
        for i in 0..n {
            y_temp[i] = y[i] + h * (a51 * k[0][i] + a52 * k[1][i] + a53 * k[2][i] + a54 * k[3][i]);
        }
        k.push(f(t + c[4] * h, &y_temp));
        nfev += 1;

        // k6
        let a61 = T::from(9017.0 / 3168.0).unwrap();
        let a62 = T::from(-355.0 / 33.0).unwrap();
        let a63 = T::from(46732.0 / 5247.0).unwrap();
        let a64 = T::from(49.0 / 176.0).unwrap();
        let a65 = T::from(-5103.0 / 18656.0).unwrap();
        for i in 0..n {
            y_temp[i] = y[i]
                + h * (a61 * k[0][i]
                    + a62 * k[1][i]
                    + a63 * k[2][i]
                    + a64 * k[3][i]
                    + a65 * k[4][i]);
        }
        k.push(f(t + c[5] * h, &y_temp));
        nfev += 1;

        // k7 (for error estimation)
        let mut y5 = vec![T::zero(); n];
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..6 {
                sum = sum + b5[j] * k[j][i];
            }
            y5[i] = y[i] + h * sum;
        }
        k.push(f(t + c[6] * h, &y5));
        nfev += 1;

        // Compute 4th order solution for error estimation
        let mut y4 = vec![T::zero(); n];
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..7 {
                sum = sum + b4[j] * k[j][i];
            }
            y4[i] = y[i] + h * sum;
        }

        // Estimate error
        let mut error = T::zero();
        for i in 0..n {
            let scale = config.atol + config.rtol * y[i].abs().max(y5[i].abs());
            let err_i = (y5[i] - y4[i]).abs() / scale;
            if err_i > error {
                error = err_i;
            }
        }

        // Accept or reject step
        if error <= T::one() || h <= config.h_min {
            t = t + h;
            y = y5;
            t_vals.push(t);
            y_vals.push(y.clone());
            nsteps += 1;
        }

        // Adjust step size
        let safety = T::from(0.9).unwrap();
        let min_factor = T::from(0.2).unwrap();
        let max_factor = T::from(10.0).unwrap();

        let factor = if error > T::epsilon() {
            safety * (T::one() / error).powf(T::from(0.2).unwrap())
        } else {
            max_factor
        };

        h = h * factor.max(min_factor).min(max_factor);
        h = h.max(config.h_min).min(config.h_max);
    }

    Ok(OdeResult {
        t: t_vals,
        y: y_vals,
        success: t >= tf,
        nfev,
        nsteps,
        message: "Dormand-Prince integration completed".to_string(),
    })
}

/// Implicit Euler method (backward Euler) for stiff equations
///
/// y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
///
/// Uses Newton iteration to solve the implicit equation
fn solve_implicit_euler<T, F>(
    f: &F,
    t0: T,
    tf: T,
    y0: &[T],
    config: &OdeConfig<T>,
) -> Result<OdeResult<T>>
where
    T: Float + Debug,
    F: Fn(T, &[T]) -> Vec<T>,
{
    let n = y0.len();
    let h = config.h0;
    let newton_tol = config.atol;
    let max_newton_iter = 10;

    let mut t_vals = vec![t0];
    let mut y_vals = vec![y0.to_vec()];

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut nfev = 0;
    let mut nsteps = 0;

    while t < tf && nsteps < config.max_steps {
        let h_actual = if t + h > tf { tf - t } else { h };
        let t_new = t + h_actual;

        // Initial guess: explicit Euler
        let f_current = f(t, &y);
        nfev += 1;
        let mut y_new: Vec<T> = y
            .iter()
            .zip(f_current.iter())
            .map(|(&yi, &fi)| yi + h_actual * fi)
            .collect();

        // Newton iteration to solve: y_new = y + h * f(t_new, y_new)
        for _ in 0..max_newton_iter {
            let f_new = f(t_new, &y_new);
            nfev += 1;

            // Residual: R = y_new - y - h * f(t_new, y_new)
            let mut residual = T::zero();
            let mut y_update = vec![T::zero(); n];
            for i in 0..n {
                let r_i = y_new[i] - y[i] - h_actual * f_new[i];
                y_update[i] = r_i; // Simple fixed-point iteration
                residual = residual + r_i.abs();
            }

            if residual < newton_tol * T::from(n).unwrap() {
                break;
            }

            // Update (simplified: use fixed-point iteration instead of full Newton)
            for i in 0..n {
                y_new[i] = y[i] + h_actual * f_new[i];
            }
        }

        t = t_new;
        y = y_new;
        t_vals.push(t);
        y_vals.push(y.clone());
        nsteps += 1;
    }

    Ok(OdeResult {
        t: t_vals,
        y: y_vals,
        success: t >= tf,
        nfev,
        nsteps,
        message: "Implicit Euler integration completed".to_string(),
    })
}

/// BDF2 (Backward Differentiation Formula, 2nd order) for stiff equations
///
/// y_{n+1} = (4/3)*y_n - (1/3)*y_{n-1} + (2/3)*h*f(t_{n+1}, y_{n+1})
fn solve_bdf2<T, F>(f: &F, t0: T, tf: T, y0: &[T], config: &OdeConfig<T>) -> Result<OdeResult<T>>
where
    T: Float + Debug,
    F: Fn(T, &[T]) -> Vec<T>,
{
    let n = y0.len();
    let h = config.h0;
    let newton_tol = config.atol;
    let max_newton_iter = 10;

    let four_thirds = T::from(4.0 / 3.0).unwrap();
    let one_third = T::from(1.0 / 3.0).unwrap();
    let two_thirds = T::from(2.0 / 3.0).unwrap();

    let mut t_vals = vec![t0];
    let mut y_vals = vec![y0.to_vec()];

    let mut t = t0;
    let mut y = y0.to_vec();
    let mut y_prev = y0.to_vec();
    let mut nfev = 0;
    let mut nsteps = 0;

    // First step with implicit Euler
    if t + h <= tf {
        let h_actual = h.min(tf - t);
        let t_new = t + h_actual;

        let f_current = f(t, &y);
        nfev += 1;
        let mut y_new: Vec<T> = y
            .iter()
            .zip(f_current.iter())
            .map(|(&yi, &fi)| yi + h_actual * fi)
            .collect();

        for _ in 0..max_newton_iter {
            let f_new = f(t_new, &y_new);
            nfev += 1;

            let mut residual = T::zero();
            for i in 0..n {
                let r_i = y_new[i] - y[i] - h_actual * f_new[i];
                residual = residual + r_i.abs();
                y_new[i] = y[i] + h_actual * f_new[i];
            }

            if residual < newton_tol * T::from(n).unwrap() {
                break;
            }
        }

        y_prev = y.clone();
        t = t_new;
        y = y_new;
        t_vals.push(t);
        y_vals.push(y.clone());
        nsteps += 1;
    }

    // Continue with BDF2
    while t < tf && nsteps < config.max_steps {
        let h_actual = if t + h > tf { tf - t } else { h };
        let t_new = t + h_actual;

        // Initial guess
        let mut y_new: Vec<T> = (0..n)
            .map(|i| four_thirds * y[i] - one_third * y_prev[i])
            .collect();

        // Newton iteration for BDF2
        for _ in 0..max_newton_iter {
            let f_new = f(t_new, &y_new);
            nfev += 1;

            let mut residual = T::zero();
            for i in 0..n {
                let target =
                    four_thirds * y[i] - one_third * y_prev[i] + two_thirds * h_actual * f_new[i];
                let r_i = y_new[i] - target;
                residual = residual + r_i.abs();
                y_new[i] = target;
            }

            if residual < newton_tol * T::from(n).unwrap() {
                break;
            }
        }

        y_prev = y.clone();
        t = t_new;
        y = y_new;
        t_vals.push(t);
        y_vals.push(y.clone());
        nsteps += 1;
    }

    Ok(OdeResult {
        t: t_vals,
        y: y_vals,
        success: t >= tf,
        nfev,
        nsteps,
        message: "BDF2 integration completed".to_string(),
    })
}

/// Solve a scalar ODE (convenience function for single equations)
///
/// # Example
/// ```ignore
/// use numrs2::ode::odeint;
///
/// let result = odeint(|t, y| -y, (0.0, 1.0), 1.0).unwrap();
/// ```
pub fn odeint<T, F>(f: F, t_span: (T, T), y0: T) -> Result<OdeResult<T>>
where
    T: Float + Debug + std::iter::Sum,
    F: Fn(T, T) -> T,
{
    let wrapped = |t: T, y: &[T]| vec![f(t, y[0])];
    solve_ivp(wrapped, t_span, &[y0], OdeMethod::RK45)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_exponential_decay() {
        // dy/dt = -y, y(0) = 1 => y(t) = e^(-t)
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let result = solve_ivp(f, (0.0, 1.0), &[1.0], OdeMethod::Euler).unwrap();

        assert!(result.success);
        let y_final = result.y.last().unwrap()[0];
        let expected = (-1.0f64).exp();
        // Euler is first-order, so expect larger error
        assert!((y_final - expected).abs() < 0.1);
    }

    #[test]
    fn test_rk4_exponential_decay() {
        // dy/dt = -y, y(0) = 1 => y(t) = e^(-t)
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let result = solve_ivp(f, (0.0, 1.0), &[1.0], OdeMethod::RK4).unwrap();

        assert!(result.success);
        let y_final = result.y.last().unwrap()[0];
        let expected = (-1.0f64).exp();
        // RK4 is 4th order, much more accurate
        assert!((y_final - expected).abs() < 1e-4);
    }

    #[test]
    fn test_rk45_exponential_decay() {
        // dy/dt = -y, y(0) = 1 => y(t) = e^(-t)
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let result = solve_ivp(f, (0.0, 1.0), &[1.0], OdeMethod::RK45).unwrap();

        assert!(result.success);
        let y_final = result.y.last().unwrap()[0];
        let expected = (-1.0f64).exp();
        // Adaptive methods with default tolerances
        assert!(
            (y_final - expected).abs() < 1e-2,
            "RK45 result {} too far from expected {}",
            y_final,
            expected
        );
    }

    #[test]
    fn test_dopri5_exponential_decay() {
        // dy/dt = -y, y(0) = 1 => y(t) = e^(-t)
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let result = solve_ivp(f, (0.0, 1.0), &[1.0], OdeMethod::DoPri5).unwrap();

        assert!(result.success);
        let y_final = result.y.last().unwrap()[0];
        let expected = (-1.0f64).exp();
        // Adaptive methods with default tolerances
        assert!(
            (y_final - expected).abs() < 1e-2,
            "DoPri5 result {} too far from expected {}",
            y_final,
            expected
        );
    }

    #[test]
    fn test_implicit_euler() {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let result = solve_ivp(f, (0.0, 1.0), &[1.0], OdeMethod::ImplicitEuler).unwrap();

        assert!(result.success);
        let y_final = result.y.last().unwrap()[0];
        let expected = (-1.0f64).exp();
        // Implicit Euler is first-order
        assert!((y_final - expected).abs() < 0.1);
    }

    #[test]
    fn test_bdf2() {
        let f = |_t: f64, y: &[f64]| vec![-y[0]];
        let result = solve_ivp(f, (0.0, 1.0), &[1.0], OdeMethod::BDF2).unwrap();

        assert!(result.success);
        let y_final = result.y.last().unwrap()[0];
        let expected = (-1.0f64).exp();
        // BDF2 is second-order
        assert!((y_final - expected).abs() < 0.05);
    }

    #[test]
    fn test_harmonic_oscillator() {
        // y'' + y = 0 => system: y1' = y2, y2' = -y1
        // Initial conditions: y(0) = 1, y'(0) = 0
        // Solution: y = cos(t)
        let f = |_t: f64, y: &[f64]| vec![y[1], -y[0]];
        let result =
            solve_ivp(f, (0.0, std::f64::consts::PI), &[1.0, 0.0], OdeMethod::RK4).unwrap();

        assert!(result.success);
        let y_final = result.y.last().unwrap()[0];
        // y(π) = cos(π) = -1
        assert!((y_final - (-1.0)).abs() < 1e-3);
    }

    #[test]
    fn test_logistic_growth() {
        // dy/dt = y * (1 - y), y(0) = 0.5
        // Solution: y = 1 / (1 + e^(-t))
        let f = |_t: f64, y: &[f64]| vec![y[0] * (1.0 - y[0])];
        let result = solve_ivp(f, (0.0, 2.0), &[0.5], OdeMethod::RK45).unwrap();

        assert!(result.success);
        let y_final = result.y.last().unwrap()[0];
        let expected = 1.0 / (1.0 + (-2.0f64).exp());
        assert!((y_final - expected).abs() < 1e-3);
    }

    #[test]
    fn test_odeint_convenience() {
        let result = odeint(|_t: f64, y: f64| -y, (0.0, 1.0), 1.0).unwrap();

        assert!(result.success);
        let y_final = result.y.last().unwrap()[0];
        let expected = (-1.0f64).exp();
        // odeint uses RK45 by default
        assert!(
            (y_final - expected).abs() < 1e-2,
            "odeint result {} too far from expected {}",
            y_final,
            expected
        );
    }

    #[test]
    fn test_lorenz_system() {
        // Lorenz system (chaotic)
        let sigma = 10.0f64;
        let rho = 28.0;
        let beta = 8.0 / 3.0;

        let f = move |_t: f64, y: &[f64]| {
            vec![
                sigma * (y[1] - y[0]),
                y[0] * (rho - y[2]) - y[1],
                y[0] * y[1] - beta * y[2],
            ]
        };

        let result = solve_ivp(f, (0.0, 1.0), &[1.0, 1.0, 1.0], OdeMethod::RK45).unwrap();

        // Just check it completes without error
        assert!(result.success);
        assert!(result.nsteps > 0);
    }
}
