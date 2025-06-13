//! Core library errors - shape mismatches, indexing, basic operations

use super::context::{ErrorSeverity, OperationContext};
use thiserror::Error;

/// Core library errors
#[derive(Error, Debug, Clone)]
pub enum CoreError {
    /// Array shape mismatch
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
        context: OperationContext,
    },

    /// Dimension mismatch between arrays
    #[error("Dimension mismatch: {message}")]
    DimensionMismatch {
        message: String,
        expected_dims: Option<usize>,
        actual_dims: Option<usize>,
        context: OperationContext,
    },

    /// Invalid array index
    #[error("Index out of bounds: index {index} is out of bounds for axis {axis} with size {size}")]
    IndexOutOfBounds {
        index: isize,
        axis: usize,
        size: usize,
        context: OperationContext,
    },

    /// Invalid operation for the given array state
    #[error("Invalid operation: {operation} cannot be performed - {reason}")]
    InvalidOperation {
        operation: String,
        reason: String,
        context: OperationContext,
    },

    /// Invalid value provided to a function
    #[error("Invalid value: {message}")]
    ValueError {
        message: String,
        expected_range: Option<String>,
        actual_value: Option<String>,
        context: OperationContext,
    },

    /// Type conversion error
    #[error("Type conversion failed: cannot convert from {from_type} to {to_type} - {reason}")]
    TypeConversion {
        from_type: String,
        to_type: String,
        reason: String,
        context: OperationContext,
    },

    /// Broadcasting error
    #[error("Broadcasting error: shapes {shape1:?} and {shape2:?} cannot be broadcast together")]
    BroadcastError {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
        context: OperationContext,
    },

    /// Axis specification error
    #[error("Invalid axis: axis {axis} is out of bounds for array with {ndim} dimensions")]
    InvalidAxis {
        axis: isize,
        ndim: usize,
        context: OperationContext,
    },

    /// Feature not implemented
    #[error("Feature not implemented: {feature}")]
    NotImplemented {
        feature: String,
        planned_version: Option<String>,
        alternative: Option<String>,
        context: OperationContext,
    },

    /// Feature not enabled
    #[error("Feature not enabled: {feature} requires compilation with feature '{feature_flag}'")]
    FeatureNotEnabled {
        feature: String,
        feature_flag: String,
        context: OperationContext,
    },

    /// Array view error
    #[error("View error: {message}")]
    ViewError {
        message: String,
        view_type: String,
        context: OperationContext,
    },

    /// Stride configuration error
    #[error("Stride error: invalid stride configuration - {message}")]
    StrideError {
        message: String,
        expected_strides: Option<Vec<isize>>,
        actual_strides: Option<Vec<isize>>,
        context: OperationContext,
    },
}

impl CoreError {
    /// Get the severity level of this error
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            CoreError::ShapeMismatch { .. } => ErrorSeverity::High,
            CoreError::DimensionMismatch { .. } => ErrorSeverity::High,
            CoreError::IndexOutOfBounds { .. } => ErrorSeverity::Medium,
            CoreError::InvalidOperation { .. } => ErrorSeverity::Medium,
            CoreError::ValueError { .. } => ErrorSeverity::Medium,
            CoreError::TypeConversion { .. } => ErrorSeverity::Medium,
            CoreError::BroadcastError { .. } => ErrorSeverity::High,
            CoreError::InvalidAxis { .. } => ErrorSeverity::Medium,
            CoreError::NotImplemented { .. } => ErrorSeverity::Low,
            CoreError::FeatureNotEnabled { .. } => ErrorSeverity::Low,
            CoreError::ViewError { .. } => ErrorSeverity::Medium,
            CoreError::StrideError { .. } => ErrorSeverity::Medium,
        }
    }

    /// Get the operation context
    pub fn context(&self) -> &OperationContext {
        match self {
            CoreError::ShapeMismatch { context, .. } => context,
            CoreError::DimensionMismatch { context, .. } => context,
            CoreError::IndexOutOfBounds { context, .. } => context,
            CoreError::InvalidOperation { context, .. } => context,
            CoreError::ValueError { context, .. } => context,
            CoreError::TypeConversion { context, .. } => context,
            CoreError::BroadcastError { context, .. } => context,
            CoreError::InvalidAxis { context, .. } => context,
            CoreError::NotImplemented { context, .. } => context,
            CoreError::FeatureNotEnabled { context, .. } => context,
            CoreError::ViewError { context, .. } => context,
            CoreError::StrideError { context, .. } => context,
        }
    }

    /// Check if this error suggests a programming mistake vs runtime condition
    pub fn is_programming_error(&self) -> bool {
        matches!(self, 
            CoreError::ShapeMismatch { .. } |
            CoreError::DimensionMismatch { .. } |
            CoreError::BroadcastError { .. } |
            CoreError::InvalidAxis { .. } |
            CoreError::StrideError { .. }
        )
    }

    /// Get suggested recovery actions
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            CoreError::ShapeMismatch { expected, actual, .. } => {
                vec![
                    format!("Reshape one of the arrays to match: expected {:?}", expected),
                    format!("Current shape is {:?}, use .reshape() or .view()", actual),
                    "Check array dimensions before the operation".to_string(),
                ]
            },
            CoreError::IndexOutOfBounds { index, axis, size, .. } => {
                vec![
                    format!("Use index in range [0, {}) for axis {}", size, axis),
                    format!("Index {} is invalid, maximum valid index is {}", index, size - 1),
                    "Consider using .get() method for safe indexing".to_string(),
                ]
            },
            CoreError::BroadcastError { shape1, shape2, .. } => {
                vec![
                    format!("Reshape arrays to compatible shapes (current: {:?} vs {:?})", shape1, shape2),
                    "Use explicit broadcasting with .broadcast_to()".to_string(),
                    "Check NumPy broadcasting rules".to_string(),
                ]
            },
            CoreError::FeatureNotEnabled { feature_flag, .. } => {
                vec![
                    format!("Recompile with --features {}", feature_flag),
                    "Check Cargo.toml for available features".to_string(),
                ]
            },
            CoreError::TypeConversion { from_type, to_type, .. } => {
                vec![
                    format!("Use explicit conversion methods from {} to {}", from_type, to_type),
                    "Check if the conversion is mathematically valid".to_string(),
                    "Consider using .try_into() for fallible conversions".to_string(),
                ]
            },
            _ => vec!["Check function documentation for valid parameters".to_string()],
        }
    }
}

/// Convenience constructors for common core errors
impl CoreError {
    /// Create a shape mismatch error
    pub fn shape_mismatch(
        expected: Vec<usize>, 
        actual: Vec<usize>, 
        operation: &str
    ) -> Self {
        CoreError::ShapeMismatch {
            expected,
            actual,
            context: OperationContext::new(operation),
        }
    }

    /// Create an index out of bounds error
    pub fn index_out_of_bounds(
        index: isize, 
        axis: usize, 
        size: usize, 
        operation: &str
    ) -> Self {
        CoreError::IndexOutOfBounds {
            index,
            axis,
            size,
            context: OperationContext::new(operation),
        }
    }

    /// Create a dimension mismatch error
    pub fn dimension_mismatch(
        message: &str, 
        expected: Option<usize>, 
        actual: Option<usize>
    ) -> Self {
        CoreError::DimensionMismatch {
            message: message.to_string(),
            expected_dims: expected,
            actual_dims: actual,
            context: OperationContext::default(),
        }
    }

    /// Create an invalid operation error
    pub fn invalid_operation(operation: &str, reason: &str) -> Self {
        CoreError::InvalidOperation {
            operation: operation.to_string(),
            reason: reason.to_string(),
            context: OperationContext::new(operation),
        }
    }

    /// Create a broadcasting error
    pub fn broadcast_error(shape1: Vec<usize>, shape2: Vec<usize>, operation: &str) -> Self {
        CoreError::BroadcastError {
            shape1,
            shape2,
            context: OperationContext::new(operation),
        }
    }

    /// Create a feature not implemented error
    pub fn not_implemented(feature: &str) -> Self {
        CoreError::NotImplemented {
            feature: feature.to_string(),
            planned_version: None,
            alternative: None,
            context: OperationContext::default(),
        }
    }

    /// Create a feature not enabled error
    pub fn feature_not_enabled(feature: &str, feature_flag: &str) -> Self {
        CoreError::FeatureNotEnabled {
            feature: feature.to_string(),
            feature_flag: feature_flag.to_string(),
            context: OperationContext::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_mismatch_error() {
        let err = CoreError::shape_mismatch(vec![2, 3], vec![3, 2], "matrix_multiply");
        assert_eq!(err.severity(), ErrorSeverity::High);
        assert!(err.is_programming_error());
        
        let suggestions = err.recovery_suggestions();
        assert!(!suggestions.is_empty());
        assert!(suggestions[0].contains("Reshape"));
    }

    #[test]
    fn test_index_out_of_bounds_error() {
        let err = CoreError::index_out_of_bounds(5, 0, 3, "array_access");
        assert_eq!(err.severity(), ErrorSeverity::Medium);
        
        let suggestions = err.recovery_suggestions();
        assert!(suggestions[0].contains("range [0, 3)"));
    }

    #[test]
    fn test_broadcast_error() {
        let err = CoreError::broadcast_error(vec![2, 3], vec![4, 5], "element_wise_add");
        assert_eq!(err.severity(), ErrorSeverity::High);
        
        if let CoreError::BroadcastError { shape1, shape2, .. } = &err {
            assert_eq!(*shape1, vec![2, 3]);
            assert_eq!(*shape2, vec![4, 5]);
        } else {
            panic!("Expected BroadcastError variant");
        }
    }

    #[test]
    fn test_feature_not_enabled() {
        let err = CoreError::feature_not_enabled("GPU acceleration", "gpu");
        assert_eq!(err.severity(), ErrorSeverity::Low);
        
        let suggestions = err.recovery_suggestions();
        assert!(suggestions[0].contains("--features gpu"));
    }

    #[test]
    fn test_error_context_access() {
        let err = CoreError::shape_mismatch(vec![2, 3], vec![3, 2], "test_operation");
        let context = err.context();
        assert_eq!(context.operation, Some("test_operation".to_string()));
    }
}