//! PID Controller Design and Tuning
//!
//! This module provides PID (Proportional-Integral-Derivative) controller
//! implementation and auto-tuning methods.
//!
//! ## PID Control Law
//!
//! The continuous PID controller is defined as:
//! ```text
//! u(t) = Kp*e(t) + Ki*∫e(t)dt + Kd*de(t)/dt
//! ```
//!
//! The discrete PID controller uses:
//! ```text
//! u[k] = Kp*e[k] + Ki*Ts*Σe[k] + Kd*(e[k] - e[k-1])/Ts
//! ```
//!
//! where:
//! - Kp: Proportional gain
//! - Ki: Integral gain
//! - Kd: Derivative gain
//! - Ts: Sample time
//! - e(t) or e\[k\]: Error signal

use super::{ControlError, ControlResult, SystemType, TransferFunction};
use scirs2_core::ndarray::Array1;

/// PID controller tuning method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuningMethod {
    /// Ziegler-Nichols method
    ZieglerNichols,
    /// Cohen-Coon method
    CohenCoon,
    /// Manual tuning
    Manual,
}

/// Anti-windup method for integral term
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntiWindup {
    /// No anti-windup
    None,
    /// Clamp integral to limits
    Clamp,
    /// Back-calculation
    BackCalculation,
}

/// PID controller
#[derive(Debug, Clone)]
pub struct PIDController {
    /// Proportional gain
    kp: f64,
    /// Integral gain
    ki: f64,
    /// Derivative gain
    kd: f64,
    /// Sample time (seconds)
    sample_time: f64,
    /// Integral sum
    integral: f64,
    /// Previous error
    prev_error: f64,
    /// Output limits (min, max)
    output_limits: Option<(f64, f64)>,
    /// Integral limits (min, max)
    integral_limits: Option<(f64, f64)>,
    /// Anti-windup method
    anti_windup: AntiWindup,
    /// Derivative filter coefficient (0 = no filter, 1 = full filter)
    derivative_filter: f64,
    /// Filtered derivative
    filtered_derivative: f64,
}

impl PIDController {
    /// Create a new PID controller
    ///
    /// # Arguments
    /// * `kp` - Proportional gain
    /// * `ki` - Integral gain
    /// * `kd` - Derivative gain
    /// * `sample_time` - Sample time in seconds
    pub fn new(kp: f64, ki: f64, kd: f64, sample_time: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            sample_time,
            integral: 0.0,
            prev_error: 0.0,
            output_limits: None,
            integral_limits: None,
            anti_windup: AntiWindup::None,
            derivative_filter: 0.0,
            filtered_derivative: 0.0,
        }
    }

    /// Set output limits
    pub fn set_output_limits(&mut self, min: f64, max: f64) -> ControlResult<()> {
        if min >= max {
            return Err(ControlError::InvalidParameters(
                "Min limit must be less than max limit".to_string(),
            ));
        }
        self.output_limits = Some((min, max));
        Ok(())
    }

    /// Set integral limits
    pub fn set_integral_limits(&mut self, min: f64, max: f64) -> ControlResult<()> {
        if min >= max {
            return Err(ControlError::InvalidParameters(
                "Min limit must be less than max limit".to_string(),
            ));
        }
        self.integral_limits = Some((min, max));
        Ok(())
    }

    /// Set anti-windup method
    pub fn set_anti_windup(&mut self, method: AntiWindup) {
        self.anti_windup = method;
    }

    /// Set derivative filter coefficient (0-1)
    ///
    /// Higher values provide more filtering but slower response
    pub fn set_derivative_filter(&mut self, alpha: f64) -> ControlResult<()> {
        if !(0.0..=1.0).contains(&alpha) {
            return Err(ControlError::InvalidParameters(
                "Derivative filter coefficient must be between 0 and 1".to_string(),
            ));
        }
        self.derivative_filter = alpha;
        Ok(())
    }

    /// Reset the controller state
    pub fn reset(&mut self) {
        self.integral = 0.0;
        self.prev_error = 0.0;
        self.filtered_derivative = 0.0;
    }

    /// Update the controller with new error value
    ///
    /// # Arguments
    /// * `error` - Current error (setpoint - measurement)
    ///
    /// # Returns
    /// Control output
    pub fn update(&mut self, error: f64) -> f64 {
        // Proportional term
        let p_term = self.kp * error;

        // Integral term
        self.integral += error * self.sample_time;

        // Apply integral limits if set
        if let Some((min, max)) = self.integral_limits {
            self.integral = self.integral.clamp(min, max);
        }

        let i_term = self.ki * self.integral;

        // Derivative term with filtering
        let raw_derivative = (error - self.prev_error) / self.sample_time;
        self.filtered_derivative = self.derivative_filter * self.filtered_derivative
            + (1.0 - self.derivative_filter) * raw_derivative;
        let d_term = self.kd * self.filtered_derivative;

        // Compute output
        let mut output = p_term + i_term + d_term;

        // Apply output limits and anti-windup
        if let Some((min, max)) = self.output_limits {
            let unclamped_output = output;
            output = output.clamp(min, max);

            // Anti-windup
            match self.anti_windup {
                AntiWindup::Clamp => {
                    // Already handled by integral limits
                }
                AntiWindup::BackCalculation => {
                    // Back-calculate integral to prevent windup
                    if output != unclamped_output {
                        let excess = unclamped_output - output;
                        self.integral -= excess / self.ki.max(1e-10);
                    }
                }
                AntiWindup::None => {}
            }
        }

        self.prev_error = error;
        output
    }

    /// Get current gains
    pub fn gains(&self) -> (f64, f64, f64) {
        (self.kp, self.ki, self.kd)
    }

    /// Set gains manually
    pub fn set_gains(&mut self, kp: f64, ki: f64, kd: f64) {
        self.kp = kp;
        self.ki = ki;
        self.kd = kd;
    }

    /// Auto-tune using Ziegler-Nichols method
    ///
    /// # Arguments
    /// * `ultimate_gain` - Ultimate gain Ku (gain at stability limit)
    /// * `ultimate_period` - Ultimate period Pu (oscillation period at Ku)
    ///
    /// # Tuning Rules
    /// - Kp = 0.6 * Ku
    /// - Ki = 1.2 * Ku / Pu
    /// - Kd = 0.075 * Ku * Pu
    pub fn tune_ziegler_nichols(
        &mut self,
        ultimate_gain: f64,
        ultimate_period: f64,
    ) -> ControlResult<()> {
        if ultimate_gain <= 0.0 || ultimate_period <= 0.0 {
            return Err(ControlError::InvalidParameters(
                "Ultimate gain and period must be positive".to_string(),
            ));
        }

        self.kp = 0.6 * ultimate_gain;
        self.ki = 1.2 * ultimate_gain / ultimate_period;
        self.kd = 0.075 * ultimate_gain * ultimate_period;

        Ok(())
    }

    /// Auto-tune using Cohen-Coon method
    ///
    /// # Arguments
    /// * `process_gain` - Steady-state process gain K
    /// * `time_constant` - Process time constant τ
    /// * `dead_time` - Process dead time θ
    ///
    /// # Tuning Rules
    /// ```text
    /// Kp = (1/K) * (τ/θ) * (1 + θ/(3*τ))
    /// Ki = Kp * (30 + 3*θ/τ) / (9 + 20*θ/τ) / τ
    /// Kd = Kp * 4 / (11 + 2*θ/τ) * θ
    /// ```
    pub fn tune_cohen_coon(
        &mut self,
        process_gain: f64,
        time_constant: f64,
        dead_time: f64,
    ) -> ControlResult<()> {
        if process_gain.abs() < 1e-10 {
            return Err(ControlError::InvalidParameters(
                "Process gain cannot be zero".to_string(),
            ));
        }

        if time_constant <= 0.0 || dead_time <= 0.0 {
            return Err(ControlError::InvalidParameters(
                "Time constant and dead time must be positive".to_string(),
            ));
        }

        let theta_tau = dead_time / time_constant;

        self.kp = (1.0 / process_gain) * (time_constant / dead_time) * (1.0 + theta_tau / 3.0);

        let ki_factor = (30.0 + 3.0 * theta_tau) / (9.0 + 20.0 * theta_tau);
        self.ki = self.kp * ki_factor / time_constant;

        let kd_factor = 4.0 / (11.0 + 2.0 * theta_tau);
        self.kd = self.kp * kd_factor * dead_time;

        Ok(())
    }

    /// Convert PID to transfer function representation
    ///
    /// C(s) = Kp + Ki/s + Kd*s
    pub fn to_transfer_function(&self) -> ControlResult<TransferFunction> {
        // C(s) = (Kd*s^2 + Kp*s + Ki) / s
        let num = vec![self.kd, self.kp, self.ki];
        let den = vec![1.0, 0.0];

        TransferFunction::new(num, den)
    }

    /// Simulate step response of closed-loop system
    ///
    /// # Arguments
    /// * `plant_tf` - Plant transfer function G(s)
    /// * `setpoint` - Desired setpoint
    /// * `time_span` - (start_time, end_time)
    /// * `num_points` - Number of simulation points
    pub fn simulate_step_response(
        &mut self,
        plant_tf: &TransferFunction,
        setpoint: f64,
        time_span: (f64, f64),
        num_points: usize,
    ) -> ControlResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let (t_start, t_end) = time_span;
        let dt = (t_end - t_start) / (num_points - 1) as f64;

        if dt <= 0.0 {
            return Err(ControlError::InvalidParameters(
                "Invalid time span or number of points".to_string(),
            ));
        }

        let mut time = Vec::with_capacity(num_points);
        let mut output = Vec::with_capacity(num_points);
        let mut control = Vec::with_capacity(num_points);

        // Simple simulation using discrete approximation
        let mut y = 0.0; // Current output
        let mut u; // Control signal

        // Get plant poles and zeros for simple simulation
        let poles = plant_tf.poles()?;
        let dc_gain = plant_tf.dc_gain();

        // Use first-order approximation if plant is complex
        let tau = if !poles.is_empty() {
            1.0 / poles[0].norm().max(0.1)
        } else {
            1.0
        };

        self.reset();

        for i in 0..num_points {
            let t = t_start + i as f64 * dt;
            time.push(t);

            // Compute error
            let error = setpoint - y;

            // Update controller
            u = self.update(error);

            // Simple first-order plant response: dy/dt = (K*u - y)/tau
            let dydt = (dc_gain * u - y) / tau;
            y += dydt * dt;

            output.push(y);
            control.push(u);
        }

        Ok((time, output, control))
    }
}

/// Compute optimal PID gains using IMC (Internal Model Control) tuning
///
/// # Arguments
/// * `process_gain` - Process gain K
/// * `time_constant` - Process time constant τ
/// * `dead_time` - Process dead time θ
/// * `desired_closed_loop_time_constant` - Desired closed-loop time constant λ
pub fn tune_imc(
    process_gain: f64,
    time_constant: f64,
    dead_time: f64,
    desired_closed_loop_time_constant: f64,
) -> ControlResult<(f64, f64, f64)> {
    if process_gain.abs() < 1e-10 {
        return Err(ControlError::InvalidParameters(
            "Process gain cannot be zero".to_string(),
        ));
    }

    if time_constant <= 0.0 || dead_time < 0.0 || desired_closed_loop_time_constant <= 0.0 {
        return Err(ControlError::InvalidParameters(
            "Invalid time constants".to_string(),
        ));
    }

    let lambda = desired_closed_loop_time_constant;

    // IMC tuning rules for first-order plus dead time (FOPDT) model
    let kp = time_constant / (process_gain * (lambda + dead_time));
    let ki = kp / time_constant;
    let kd = 0.0; // IMC typically doesn't use derivative for FOPDT

    Ok((kp, ki, kd))
}

/// Compute step response characteristics
#[derive(Debug, Clone)]
pub struct StepResponseMetrics {
    /// Rise time (10% to 90%)
    pub rise_time: f64,
    /// Settling time (within 2% of final value)
    pub settling_time: f64,
    /// Overshoot percentage
    pub overshoot_percent: f64,
    /// Peak time
    pub peak_time: f64,
    /// Steady-state error
    pub steady_state_error: f64,
}

/// Analyze step response characteristics
pub fn analyze_step_response(
    time: &[f64],
    output: &[f64],
    setpoint: f64,
) -> ControlResult<StepResponseMetrics> {
    if time.len() != output.len() || time.is_empty() {
        return Err(ControlError::InvalidParameters(
            "Time and output arrays must have same non-zero length".to_string(),
        ));
    }

    let final_value = output[output.len() - 1];
    let steady_state_error = (setpoint - final_value).abs();

    // Find peak
    let mut peak_value = f64::NEG_INFINITY;
    let mut peak_time = 0.0;
    for (i, &y) in output.iter().enumerate() {
        if y > peak_value {
            peak_value = y;
            peak_time = time[i];
        }
    }

    let overshoot_percent = if final_value.abs() > 1e-10 {
        ((peak_value - final_value) / final_value * 100.0).max(0.0)
    } else {
        0.0
    };

    // Find rise time (10% to 90%)
    let threshold_low = 0.1 * final_value;
    let threshold_high = 0.9 * final_value;

    let mut t_low = None;
    let mut t_high = None;

    for (i, &y) in output.iter().enumerate() {
        if t_low.is_none() && y >= threshold_low {
            t_low = Some(time[i]);
        }
        if y >= threshold_high {
            t_high = Some(time[i]);
            break;
        }
    }

    let rise_time = match (t_low, t_high) {
        (Some(tl), Some(th)) => th - tl,
        _ => 0.0,
    };

    // Find settling time (within 2% of final value)
    let tolerance = 0.02 * final_value.abs();
    let mut settling_time = time[time.len() - 1];

    for i in (0..output.len()).rev() {
        if (output[i] - final_value).abs() > tolerance {
            if i + 1 < time.len() {
                settling_time = time[i + 1];
            }
            break;
        }
    }

    Ok(StepResponseMetrics {
        rise_time,
        settling_time,
        overshoot_percent,
        peak_time,
        steady_state_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_creation() {
        let pid = PIDController::new(1.0, 0.5, 0.1, 0.01);
        let (kp, ki, kd) = pid.gains();

        assert_eq!(kp, 1.0);
        assert_eq!(ki, 0.5);
        assert_eq!(kd, 0.1);
    }

    #[test]
    fn test_pid_update() {
        let mut pid = PIDController::new(1.0, 0.0, 0.0, 0.01);
        let output = pid.update(1.0);

        // Pure proportional, should be Kp * error = 1.0 * 1.0 = 1.0
        assert!((output - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_output_limits() {
        let mut pid = PIDController::new(10.0, 0.0, 0.0, 0.01);
        pid.set_output_limits(-5.0, 5.0).unwrap();

        let output = pid.update(1.0);
        assert!((-5.0..=5.0).contains(&output));
    }

    #[test]
    fn test_pid_reset() {
        let mut pid = PIDController::new(1.0, 1.0, 0.0, 0.01);

        pid.update(1.0);
        pid.update(1.0);

        pid.reset();

        let (_, _, _) = pid.gains();
        // After reset, integral should be zero
        // Next update with error 0 should give 0 output
        let output = pid.update(0.0);
        assert!((output).abs() < 1e-10);
    }

    #[test]
    fn test_ziegler_nichols_tuning() {
        let mut pid = PIDController::new(0.0, 0.0, 0.0, 0.01);
        pid.tune_ziegler_nichols(4.0, 2.0).unwrap();

        let (kp, ki, kd) = pid.gains();

        assert!((kp - 2.4).abs() < 1e-10); // 0.6 * 4.0
        assert!((ki - 2.4).abs() < 1e-10); // 1.2 * 4.0 / 2.0
        assert!((kd - 0.6).abs() < 1e-10); // 0.075 * 4.0 * 2.0
    }

    #[test]
    fn test_cohen_coon_tuning() {
        let mut pid = PIDController::new(0.0, 0.0, 0.0, 0.01);
        pid.tune_cohen_coon(1.0, 10.0, 1.0).unwrap();

        let (kp, ki, kd) = pid.gains();

        // Just check that values are computed (not zero)
        assert!(kp > 0.0);
        assert!(ki > 0.0);
        assert!(kd > 0.0);
    }

    #[test]
    fn test_imc_tuning() {
        let result = tune_imc(1.0, 10.0, 1.0, 5.0);
        assert!(result.is_ok());

        let (kp, ki, kd) = result.unwrap();
        assert!(kp > 0.0);
        assert!(ki > 0.0);
        assert_eq!(kd, 0.0); // IMC doesn't use derivative for FOPDT
    }

    #[test]
    fn test_analyze_step_response() {
        let time = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let output = vec![0.0, 0.5, 0.9, 1.1, 1.0, 1.0];
        let setpoint = 1.0;

        let metrics = analyze_step_response(&time, &output, setpoint).unwrap();

        assert!(metrics.rise_time > 0.0);
        assert!(metrics.overshoot_percent >= 0.0);
        assert!(metrics.steady_state_error >= 0.0);
    }

    #[test]
    fn test_pid_to_transfer_function() {
        let pid = PIDController::new(1.0, 0.5, 0.1, 0.01);
        let tf = pid.to_transfer_function();
        assert!(tf.is_ok());
    }
}
