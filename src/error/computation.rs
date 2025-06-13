//! Computation and numerical errors - instability, convergence, overflow

use super::context::{ErrorSeverity, OperationContext};
use thiserror::Error;

/// Computation and numerical errors
#[derive(Error, Debug, Clone)]
pub enum ComputationError {
    /// Numerical instability detected
    #[error("Numerical instability in {operation}: {details}")]
    NumericalInstability {
        operation: String,
        details: String,
        condition_number: Option<f64>,
        context: OperationContext,
    },

    /// Algorithm failed to converge
    #[error("Convergence failure in {algorithm}: {reason} after {iterations} iterations")]
    ConvergenceFailure {
        algorithm: String,
        reason: String,
        iterations: usize,
        tolerance: Option<f64>,
        residual: Option<f64>,
        context: OperationContext,
    },

    /// Numerical overflow
    #[error("Numerical overflow in {operation}: {details}")]
    Overflow {
        operation: String,
        details: String,
        value: Option<String>,
        context: OperationContext,
    },

    /// Numerical underflow
    #[error("Numerical underflow in {operation}: {details}")]
    Underflow {
        operation: String,
        details: String,
        value: Option<String>,
        context: OperationContext,
    },

    /// Invalid mathematical operation
    #[error("Invalid mathematical operation: {operation} - {reason}")]
    InvalidMathOperation {
        operation: String,
        reason: String,
        suggested_alternative: Option<String>,
        context: OperationContext,
    },

    /// Singular matrix or non-invertible operation
    #[error("Singular matrix in {operation}: matrix is not invertible")]
    SingularMatrix {
        operation: String,
        condition_number: Option<f64>,
        rank: Option<usize>,
        expected_rank: Option<usize>,
        context: OperationContext,
    },

    /// BLAS/LAPACK library error
    #[error("BLAS/LAPACK error in {operation}: {library} returned code {error_code}")]
    LinearAlgebraLibraryError {
        operation: String,
        library: String, // "BLAS", "LAPACK", etc.
        error_code: i32,
        details: Option<String>,
        context: OperationContext,
    },

    /// Eigenvalue/eigenvector computation error
    #[error("Eigenvalue computation failed in {operation}: {reason}")]
    EigenvalueError {
        operation: String,
        reason: String,
        matrix_properties: Option<String>, // "symmetric", "hermitian", etc.
        context: OperationContext,
    },

    /// SVD computation error
    #[error("SVD computation failed: {reason}")]
    SVDError {
        reason: String,
        matrix_shape: Vec<usize>,
        requested_components: Option<usize>,
        context: OperationContext,
    },

    /// FFT computation error
    #[error("FFT computation failed: {reason}")]
    FFTError {
        reason: String,
        signal_length: usize,
        transform_type: String, // "DFT", "RFFT", "DCT", etc.
        context: OperationContext,
    },

    /// Optimization algorithm error
    #[error("Optimization failed in {algorithm}: {reason}")]
    OptimizationError {
        algorithm: String,
        reason: String,
        objective_value: Option<f64>,
        gradient_norm: Option<f64>,
        context: OperationContext,
    },

    /// Integration/ODE solver error
    #[error("Integration failed: {reason}")]
    IntegrationError {
        reason: String,
        method: String,
        step_size: Option<f64>,
        error_estimate: Option<f64>,
        context: OperationContext,
    },

    /// Interpolation error
    #[error("Interpolation failed: {reason}")]
    InterpolationError {
        reason: String,
        method: String,
        data_points: usize,
        context: OperationContext,
    },

    /// Random number generation error
    #[error("Random number generation failed: {reason}")]
    RandomGenerationError {
        reason: String,
        distribution: Option<String>,
        parameters: Option<String>,
        context: OperationContext,
    },
}

impl ComputationError {
    /// Get the severity level of this error
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            ComputationError::NumericalInstability { condition_number, .. } => {
                // High condition number indicates severe instability
                if let Some(cond) = condition_number {
                    if *cond > 1e12 {
                        ErrorSeverity::High
                    } else {
                        ErrorSeverity::Medium
                    }
                } else {
                    ErrorSeverity::Medium
                }
            },
            ComputationError::ConvergenceFailure { .. } => ErrorSeverity::Medium,
            ComputationError::Overflow { .. } => ErrorSeverity::High,
            ComputationError::Underflow { .. } => ErrorSeverity::Medium,
            ComputationError::InvalidMathOperation { .. } => ErrorSeverity::High,
            ComputationError::SingularMatrix { .. } => ErrorSeverity::High,
            ComputationError::LinearAlgebraLibraryError { error_code, .. } => {
                // Critical LAPACK errors are typically > 0
                if *error_code > 0 {
                    ErrorSeverity::High
                } else {
                    ErrorSeverity::Medium
                }
            },
            ComputationError::EigenvalueError { .. } => ErrorSeverity::Medium,
            ComputationError::SVDError { .. } => ErrorSeverity::Medium,
            ComputationError::FFTError { .. } => ErrorSeverity::Medium,
            ComputationError::OptimizationError { .. } => ErrorSeverity::Medium,
            ComputationError::IntegrationError { .. } => ErrorSeverity::Medium,
            ComputationError::InterpolationError { .. } => ErrorSeverity::Medium,
            ComputationError::RandomGenerationError { .. } => ErrorSeverity::Low,
        }
    }

    /// Get the operation context
    pub fn context(&self) -> &OperationContext {
        match self {
            ComputationError::NumericalInstability { context, .. } => context,
            ComputationError::ConvergenceFailure { context, .. } => context,
            ComputationError::Overflow { context, .. } => context,
            ComputationError::Underflow { context, .. } => context,
            ComputationError::InvalidMathOperation { context, .. } => context,
            ComputationError::SingularMatrix { context, .. } => context,
            ComputationError::LinearAlgebraLibraryError { context, .. } => context,
            ComputationError::EigenvalueError { context, .. } => context,
            ComputationError::SVDError { context, .. } => context,
            ComputationError::FFTError { context, .. } => context,
            ComputationError::OptimizationError { context, .. } => context,
            ComputationError::IntegrationError { context, .. } => context,
            ComputationError::InterpolationError { context, .. } => context,
            ComputationError::RandomGenerationError { context, .. } => context,
        }
    }

    /// Check if this error might be recoverable with different parameters
    pub fn is_potentially_recoverable(&self) -> bool {
        matches!(self,
            ComputationError::ConvergenceFailure { .. } |
            ComputationError::NumericalInstability { .. } |
            ComputationError::OptimizationError { .. } |
            ComputationError::IntegrationError { .. }
        )
    }

    /// Get suggested recovery actions
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            ComputationError::NumericalInstability { condition_number, .. } => {
                let mut suggestions = vec![
                    "Use higher precision arithmetic (f64 instead of f32)".to_string(),
                    "Consider iterative refinement techniques".to_string(),
                    "Check input data for extreme values".to_string(),
                ];
                if let Some(cond) = condition_number {
                    if *cond > 1e12 {
                        suggestions.push("Matrix is severely ill-conditioned, consider regularization".to_string());
                    }
                }
                suggestions
            },
            ComputationError::SingularMatrix { .. } => {
                vec![
                    "Use pseudo-inverse (pinv) instead of regular inverse".to_string(),
                    "Add regularization to make matrix invertible".to_string(),
                    "Check for linearly dependent rows/columns".to_string(),
                    "Use SVD-based solution methods".to_string(),
                ]
            },
            ComputationError::ConvergenceFailure { algorithm, iterations, tolerance, .. } => {
                vec![
                    format!("Increase iteration limit (current: {})", iterations),
                    if let Some(tol) = tolerance {
                        format!("Relax tolerance (current: {:.2e})", tol)
                    } else {
                        "Adjust convergence tolerance".to_string()
                    },
                    format!("Try different algorithm instead of {}", algorithm),
                    "Improve initial guess or starting conditions".to_string(),
                ]
            },
            ComputationError::Overflow { .. } => {
                vec![
                    "Scale input data to prevent overflow".to_string(),
                    "Use logarithmic representation for large numbers".to_string(),
                    "Check for infinite or very large intermediate values".to_string(),
                ]
            },
            ComputationError::LinearAlgebraLibraryError { error_code, library, .. } => {
                vec![
                    format!("Check {} documentation for error code {}", library, error_code),
                    "Verify input matrix properties (dimensions, data type)".to_string(),
                    "Ensure matrix storage format is correct".to_string(),
                ]
            },
            ComputationError::FFTError { signal_length, .. } => {
                vec![
                    "Ensure signal length is supported by FFT implementation".to_string(),
                    format!("Try padding signal to power of 2 (current length: {})", signal_length),
                    "Check for NaN or infinite values in input signal".to_string(),
                ]
            },
            _ => vec!["Check algorithm parameters and input data validity".to_string()],
        }
    }
}

/// Convenience constructors for common computation errors
impl ComputationError {
    /// Create a numerical instability error
    pub fn numerical_instability(operation: &str, details: &str) -> Self {
        ComputationError::NumericalInstability {
            operation: operation.to_string(),
            details: details.to_string(),
            condition_number: None,
            context: OperationContext::new(operation),
        }
    }

    /// Create a convergence failure error
    pub fn convergence_failure(algorithm: &str, reason: &str, iterations: usize) -> Self {
        ComputationError::ConvergenceFailure {
            algorithm: algorithm.to_string(),
            reason: reason.to_string(),
            iterations,
            tolerance: None,
            residual: None,
            context: OperationContext::new(algorithm),
        }
    }

    /// Create a singular matrix error
    pub fn singular_matrix(operation: &str) -> Self {
        ComputationError::SingularMatrix {
            operation: operation.to_string(),
            condition_number: None,
            rank: None,
            expected_rank: None,
            context: OperationContext::new(operation),
        }
    }

    /// Create a BLAS error
    pub fn blas_error(operation: &str, error_code: i32) -> Self {
        ComputationError::LinearAlgebraLibraryError {
            operation: operation.to_string(),
            library: "BLAS".to_string(),
            error_code,
            details: None,
            context: OperationContext::new(operation),
        }
    }

    /// Create a LAPACK error
    pub fn lapack_error(operation: &str, error_code: i32) -> Self {
        ComputationError::LinearAlgebraLibraryError {
            operation: operation.to_string(),
            library: "LAPACK".to_string(),
            error_code,
            details: None,
            context: OperationContext::new(operation),
        }
    }

    /// Create an overflow error
    pub fn overflow(operation: &str, details: &str) -> Self {
        ComputationError::Overflow {
            operation: operation.to_string(),
            details: details.to_string(),
            value: None,
            context: OperationContext::new(operation),
        }
    }

    /// Create an FFT error
    pub fn fft_error(reason: &str, signal_length: usize, transform_type: &str) -> Self {
        ComputationError::FFTError {
            reason: reason.to_string(),
            signal_length,
            transform_type: transform_type.to_string(),
            context: OperationContext::new("FFT"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numerical_instability_severity() {
        let stable_err = ComputationError::NumericalInstability {
            operation: "test".to_string(),
            details: "test".to_string(),
            condition_number: Some(1e6),
            context: OperationContext::default(),
        };
        assert_eq!(stable_err.severity(), ErrorSeverity::Medium);

        let unstable_err = ComputationError::NumericalInstability {
            operation: "test".to_string(),
            details: "test".to_string(),
            condition_number: Some(1e15),
            context: OperationContext::default(),
        };
        assert_eq!(unstable_err.severity(), ErrorSeverity::High);
    }

    #[test]
    fn test_singular_matrix_suggestions() {
        let err = ComputationError::singular_matrix("matrix_inverse");
        let suggestions = err.recovery_suggestions();
        assert!(suggestions.iter().any(|s| s.contains("pseudo-inverse")));
        assert!(suggestions.iter().any(|s| s.contains("regularization")));
    }

    #[test]
    fn test_convergence_failure() {
        let err = ComputationError::convergence_failure("newton_raphson", "gradient too small", 100);
        assert!(err.is_potentially_recoverable());
        
        if let ComputationError::ConvergenceFailure { iterations, .. } = &err {
            assert_eq!(*iterations, 100);
        } else {
            panic!("Expected ConvergenceFailure variant");
        }
    }

    #[test]
    fn test_blas_error_severity() {
        let info_err = ComputationError::blas_error("dgemm", -1);
        assert_eq!(info_err.severity(), ErrorSeverity::Medium);

        let critical_err = ComputationError::blas_error("dgemm", 5);
        assert_eq!(critical_err.severity(), ErrorSeverity::High);
    }

    #[test]
    fn test_fft_error() {
        let err = ComputationError::fft_error("Invalid signal length", 1023, "DFT");
        let suggestions = err.recovery_suggestions();
        assert!(suggestions.iter().any(|s| s.contains("power of 2")));
        assert!(suggestions.iter().any(|s| s.contains("1023")));
    }
}