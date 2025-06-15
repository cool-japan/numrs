# NumRS2 Error Handling Guide

## Overview

NumRS2 features a comprehensive hierarchical error system that provides rich context, recovery suggestions, and backward compatibility. This guide covers the error system architecture, usage patterns, and best practices.

## Error System Architecture

### Hierarchical Structure

```
NumRs2Error (Legacy Root)
├── Core Errors (Shape, indexing, basic operations)
├── Computation Errors (Numerical issues, convergence)
├── Memory Errors (Allocation, memory pressure)
└── I/O Errors (File operations, serialization)
```

### Error Categories

#### 1. Core Errors (`CoreError`)
Fundamental library errors related to data structure operations:

```rust
pub enum CoreError {
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize>, context: OperationContext },
    DimensionMismatch { expected: usize, actual: usize, context: OperationContext },
    IndexOutOfBounds { index: Vec<usize>, shape: Vec<usize>, context: OperationContext },
    InvalidOperation { operation: String, reason: String, context: OperationContext },
    TypeError { expected: String, actual: String, context: OperationContext },
    ValueError { value: String, constraint: String, context: OperationContext },
}
```

#### 2. Computation Errors (`ComputationError`)
Numerical and mathematical computation issues:

```rust
pub enum ComputationError {
    NumericalInstability { algorithm: String, condition_number: Option<f64>, context: OperationContext },
    ConvergenceFailure { algorithm: String, iterations: usize, tolerance: f64, context: OperationContext },
    SingularMatrix { matrix_info: MatrixInfo, context: OperationContext },
    DivisionByZero { location: String, context: OperationContext },
    Overflow { operation: String, inputs: Vec<String>, context: OperationContext },
    Underflow { operation: String, threshold: f64, context: OperationContext },
}
```

#### 3. Memory Errors (`MemoryError`)
Memory allocation and management issues:

```rust
pub enum MemoryError {
    AllocationFailed { size: usize, alignment: usize, context: OperationContext },
    OutOfMemory { requested: usize, available: Option<usize>, context: OperationContext },
    MemoryCorruption { corruption_type: CorruptionType, location: String, context: OperationContext },
    InvalidAlignment { required: usize, actual: usize, context: OperationContext },
    BufferOverflow { buffer_size: usize, access_size: usize, context: OperationContext },
}
```

#### 4. I/O Errors (`IOError`)
File operations, serialization, and external data handling:

```rust
pub enum IOError {
    FileOperation { operation: String, path: PathBuf, reason: String, context: OperationContext },
    InvalidFormat { format: String, details: String, context: OperationContext },
    Serialization { format: String, reason: String, context: OperationContext },
    Network { operation: String, reason: String, context: OperationContext },
}
```

## Error Context System

### OperationContext

Rich context information for errors:

```rust
pub struct OperationContext {
    pub operation: Option<String>,
    pub parameters: HashMap<String, String>,
    pub shapes: Vec<Vec<usize>>,
    pub dtypes: Vec<String>,
    pub memory_info: Option<MemoryInfo>,
    pub thread_info: Option<ThreadInfo>,
    pub performance_hints: Vec<String>,
    pub timestamp: u64,
}
```

### ErrorContext Wrapper

Enhanced error with full context:

```rust
pub struct ErrorContext<E> {
    error: E,
    context: OperationContext,
    location: Option<ErrorLocation>,
    chain: Vec<Box<dyn std::error::Error + Send + Sync>>,
    recovery_suggestions: Vec<String>,
}
```

## Usage Patterns

### Basic Error Handling

```rust
use numrs::{Array, error::{Result, NumRs2Error}};

fn basic_array_operation() -> Result<Array<f64>> {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0], [3])?;
    let b = Array::from_vec(vec![4.0, 5.0], [2])?; // Wrong shape!
    
    // This will return a ShapeMismatch error
    a.add(&b)
}

fn handle_basic_errors() {
    match basic_array_operation() {
        Ok(result) => println!("Success: {:?}", result),
        Err(e) => {
            println!("Error: {}", e);
            println!("Category: {}", e.category());
            println!("Severity: {}", e.severity());
        }
    }
}
```

### Enhanced Error Handling with Context

```rust
use numrs::error::{ContextResult, ErrorContext, OperationContext, CoreError};

fn enhanced_matrix_multiply(a: &Array<f64>, b: &Array<f64>) -> ContextResult<Array<f64>> {
    let context = OperationContext::new("matrix_multiply")
        .with_shape(a.shape().to_vec())
        .with_shape(b.shape().to_vec())
        .with_parameter("algorithm", "BLAS")
        .with_parameter("threading", "parallel");

    // Validate dimensions
    if a.shape()[1] != b.shape()[0] {
        return Err(ErrorContext::new(
            CoreError::shape_mismatch_with_context(
                vec![a.shape()[1]],
                vec![b.shape()[0]],
                context.clone()
            ),
            context
        ).with_suggestion("Ensure inner dimensions match for matrix multiplication")
         .with_suggestion("Consider transposing one of the matrices"));
    }

    // Perform operation with error context
    a.matmul(b).map_err(|e| {
        ErrorContext::new(e, context)
            .with_suggestion("Try using a different BLAS implementation")
            .with_suggestion("Check for numerical instability in input matrices")
    })
}
```

### Error Recovery and Retry Logic

```rust
use numrs::error::{ErrorSeverity, IOError, MemoryError};

fn robust_file_operation(path: &str) -> Result<Array<f64>> {
    const MAX_RETRIES: usize = 3;
    let mut attempts = 0;

    loop {
        match load_array_from_file(path) {
            Ok(array) => return Ok(array),
            Err(e) => {
                attempts += 1;
                
                match &e {
                    // Retry on transient errors
                    NumRs2Error::IO(io_err) if io_err.is_transient() && attempts < MAX_RETRIES => {
                        std::thread::sleep(std::time::Duration::from_millis(100 * attempts));
                        continue;
                    },
                    
                    // Try recovery on memory errors
                    NumRs2Error::Memory(mem_err) if attempts < MAX_RETRIES => {
                        match mem_err {
                            MemoryError::OutOfMemory { .. } => {
                                garbage_collect();
                                continue;
                            },
                            _ => return Err(e),
                        }
                    },
                    
                    // No recovery possible
                    _ => return Err(e),
                }
            }
        }
    }
}
```

### Error Analysis and Debugging

```rust
use numrs::error::{ErrorCategory, ErrorSeverity};

fn analyze_error(error: &NumRs2Error) {
    println!("Error Analysis:");
    println!("  Type: {}", error);
    println!("  Category: {}", error.category());
    println!("  Severity: {}", error.severity());
    
    // Category-specific handling
    match error.category() {
        ErrorCategory::Core => {
            println!("  Issue: Problem with basic array operations");
            println!("  Action: Check array shapes and indices");
        },
        ErrorCategory::Computation => {
            println!("  Issue: Numerical computation problem");
            println!("  Action: Check input validity and algorithm parameters");
        },
        ErrorCategory::Memory => {
            println!("  Issue: Memory management problem");
            println!("  Action: Monitor memory usage and consider optimization");
        },
        ErrorCategory::IO => {
            println!("  Issue: Input/output operation failed");
            println!("  Action: Check file permissions and format compatibility");
        },
    }
    
    // Severity-specific handling
    match error.severity() {
        ErrorSeverity::Critical => {
            println!("  Response: Immediate termination required");
            std::process::exit(1);
        },
        ErrorSeverity::High => {
            println!("  Response: Operation failed, try alternative approach");
        },
        ErrorSeverity::Medium => {
            println!("  Response: Warning logged, operation may continue with fallback");
        },
        ErrorSeverity::Low => {
            println!("  Response: Minor issue, operation continues normally");
        },
    }
}
```

## Error Creation Patterns

### Using Convenience Constructors

```rust
use numrs::error::{CoreError, ComputationError, MemoryError, IOError};

// Core errors
let shape_error = CoreError::shape_mismatch(vec![3, 3], vec![2, 2]);
let index_error = CoreError::index_out_of_bounds(vec![5, 5], vec![3, 3]);
let type_error = CoreError::type_error("f64", "i32");

// Computation errors
let convergence_error = ComputationError::convergence_failure("newton_raphson", 1000, 1e-6);
let singular_error = ComputationError::singular_matrix("LU decomposition");
let overflow_error = ComputationError::overflow("exponential", vec!["x=1000".to_string()]);

// Memory errors
let allocation_error = MemoryError::allocation_failed(1024*1024*1024, 32);
let oom_error = MemoryError::out_of_memory(2048, Some(1024));

// I/O errors
let file_error = IOError::file_operation("read", "/path/to/file.npy", "Permission denied");
let format_error = IOError::invalid_format("NPY", "Invalid magic bytes");
```

### Creating Errors with Rich Context

```rust
use numrs::error::{OperationContext, ErrorLocation, MemoryInfo, MemoryPressure};

fn create_contextual_error() -> NumRs2Error {
    let memory_info = MemoryInfo {
        total_allocated: 1024 * 1024 * 500, // 500MB
        peak_usage: 1024 * 1024 * 750,     // 750MB
        available_memory: Some(1024 * 1024 * 256), // 256MB available
        pressure_level: MemoryPressure::High,
    };

    let context = OperationContext::new("large_matrix_multiply")
        .with_shape(vec![10000, 10000])
        .with_shape(vec![10000, 5000])
        .with_parameter("algorithm", "blocked_multiply")
        .with_parameter("block_size", "512")
        .with_dtype("f64")
        .with_memory_info(memory_info)
        .with_performance_hint("Consider using sparse matrices for large operations");

    let location = ErrorLocation::with_module(
        "matrix.rs",
        142,
        "blocked_matrix_multiply",
        "numrs::matrix::operations"
    );

    ComputationError::numerical_instability_with_context(
        "blocked_matrix_multiply",
        Some(1e12), // High condition number
        context
    ).with_location(location)
     .with_suggestion("Use double precision arithmetic")
     .with_suggestion("Apply matrix preconditioning")
     .with_suggestion("Consider iterative methods for large matrices")
     .into()
}
```

## Advanced Error Handling

### Error Chaining and Causality

```rust
use numrs::error::ErrorContext;

fn nested_operation() -> Result<Array<f64>> {
    read_matrix_from_file("data.csv")
        .map_err(|io_err| {
            ErrorContext::new(io_err, OperationContext::new("data_loading"))
                .with_suggestion("Check file format and permissions")
        })
        .and_then(|matrix| {
            validate_matrix(&matrix)
                .map_err(|validation_err| {
                    ErrorContext::new(validation_err, OperationContext::new("validation"))
                        .with_source(io_err) // Chain the original I/O error
                        .with_suggestion("Verify data integrity")
                })
        })
        .and_then(|matrix| {
            process_matrix(&matrix)
                .map_err(|proc_err| {
                    ErrorContext::new(proc_err, OperationContext::new("processing"))
                        .with_suggestion("Try different processing parameters")
                })
        })
}
```

### Custom Error Types

```rust
use numrs::error::{ErrorCategory, ErrorSeverity, OperationContext};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CustomAnalysisError {
    #[error("Statistical test failed: {test_name} with p-value {p_value}")]
    StatisticalTestFailed {
        test_name: String,
        p_value: f64,
        significance_level: f64,
        context: OperationContext,
    },
    
    #[error("Model convergence failed after {iterations} iterations")]
    ModelConvergenceFailed {
        model_type: String,
        iterations: usize,
        final_loss: f64,
        context: OperationContext,
    },
}

impl CustomAnalysisError {
    pub fn category(&self) -> ErrorCategory {
        ErrorCategory::Computation
    }
    
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            CustomAnalysisError::StatisticalTestFailed { p_value, significance_level, .. } => {
                if p_value < significance_level / 10.0 {
                    ErrorSeverity::High
                } else {
                    ErrorSeverity::Medium
                }
            },
            CustomAnalysisError::ModelConvergenceFailed { final_loss, .. } => {
                if *final_loss > 1.0 {
                    ErrorSeverity::High
                } else {
                    ErrorSeverity::Medium
                }
            },
        }
    }
}

// Integration with NumRS2 error system
impl From<CustomAnalysisError> for NumRs2Error {
    fn from(err: CustomAnalysisError) -> Self {
        NumRs2Error::ComputationError(err.to_string())
    }
}
```

### Error Reporting and Logging

```rust
use log::{error, warn, info};
use numrs::error::{ErrorSeverity, ErrorCategory};

fn log_error_with_context(error: &NumRs2Error, operation: &str) {
    let log_level = match error.severity() {
        ErrorSeverity::Critical => "CRITICAL",
        ErrorSeverity::High => "ERROR",
        ErrorSeverity::Medium => "WARN",
        ErrorSeverity::Low => "INFO",
    };

    let category = error.category();
    let message = format!(
        "[{}:{}] {} failed: {}",
        category, log_level, operation, error
    );

    match error.severity() {
        ErrorSeverity::Critical | ErrorSeverity::High => error!("{}", message),
        ErrorSeverity::Medium => warn!("{}", message),
        ErrorSeverity::Low => info!("{}", message),
    }

    // Additional context logging
    if let Some(context) = get_error_context(error) {
        info!("Operation context: {:?}", context);
        
        if !context.shapes.is_empty() {
            info!("Array shapes involved: {:?}", context.shapes);
        }
        
        if !context.parameters.is_empty() {
            info!("Parameters: {:?}", context.parameters);
        }
        
        if let Some(memory_info) = &context.memory_info {
            warn!("Memory pressure: {:?}", memory_info.pressure_level);
        }
    }
}

fn get_error_context(error: &NumRs2Error) -> Option<&OperationContext> {
    match error {
        NumRs2Error::Core(e) => Some(e.context()),
        NumRs2Error::Computation(e) => Some(e.context()),
        NumRs2Error::Memory(e) => Some(e.context()),
        NumRs2Error::IO(e) => Some(e.context()),
        _ => None,
    }
}
```

## Error Testing Strategies

### Unit Testing Error Conditions

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use numrs::error::{CoreError, ErrorCategory, ErrorSeverity};

    #[test]
    fn test_shape_mismatch_error() {
        let a = Array::from_vec(vec![1.0, 2.0], [2]);
        let b = Array::from_vec(vec![1.0, 2.0, 3.0], [3]);
        
        let result = a.add(&b);
        assert!(result.is_err());
        
        let error = result.unwrap_err();
        assert_eq!(error.category(), ErrorCategory::Core);
        assert_eq!(error.severity(), ErrorSeverity::High);
        
        if let NumRs2Error::Core(CoreError::ShapeMismatch { expected, actual, .. }) = error {
            assert_eq!(expected, vec![2]);
            assert_eq!(actual, vec![3]);
        } else {
            panic!("Expected ShapeMismatch error");
        }
    }

    #[test]
    fn test_error_recovery_suggestions() {
        let error = CoreError::shape_mismatch(vec![3, 3], vec![2, 2]);
        let suggestions = error.recovery_suggestions();
        
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("reshape")));
    }

    #[test]
    fn test_error_context_chaining() {
        let base_error = IOError::file_operation("read", "/tmp/test.npy", "Not found");
        let context = OperationContext::new("load_dataset");
        
        let contextual_error = ErrorContext::new(base_error, context)
            .with_suggestion("Check file path")
            .with_suggestion("Verify file exists");
        
        assert_eq!(contextual_error.recovery_suggestions().len(), 2);
        assert_eq!(contextual_error.context().operation, Some("load_dataset".to_string()));
    }
}
```

### Integration Testing with Error Scenarios

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_memory_pressure_handling() {
        // Create a scenario that triggers memory pressure
        let large_size = 10000;
        let arrays: Vec<Array<f64>> = (0..100)
            .map(|_| Array::zeros([large_size, large_size]))
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_else(|e| {
                // Verify we get appropriate memory errors
                match e {
                    NumRs2Error::Memory(MemoryError::OutOfMemory { .. }) => {
                        println!("Memory error caught as expected: {}", e);
                        vec![] // Return empty vec for test
                    },
                    _ => panic!("Expected memory error, got: {}", e),
                }
            });
    }

    #[test]
    fn test_numerical_stability_detection() {
        // Create an ill-conditioned matrix
        let mut matrix = Array::eye(100);
        matrix[[99, 99]] = 1e-16; // Make it nearly singular
        
        let result = matrix.inverse();
        
        match result {
            Err(NumRs2Error::Computation(ComputationError::NumericalInstability { .. })) => {
                // Expected numerical instability error
            },
            Err(NumRs2Error::Computation(ComputationError::SingularMatrix { .. })) => {
                // Also acceptable - matrix is effectively singular
            },
            Ok(_) => panic!("Expected numerical error for ill-conditioned matrix"),
            Err(e) => panic!("Unexpected error type: {}", e),
        }
    }
}
```

## Best Practices

### 1. Error Granularity
- Use specific error types rather than generic strings
- Include relevant context information
- Provide actionable recovery suggestions

### 2. Performance Considerations
- Error creation should be fast for hot paths
- Use lazy evaluation for expensive context gathering
- Consider error frequency in API design

### 3. User Experience
- Provide clear, helpful error messages
- Include suggestions for fixing common issues
- Use appropriate severity levels

### 4. Backward Compatibility
- Maintain legacy error types during transitions
- Use gradual migration strategies
- Provide clear migration paths

### 5. Testing
- Test both success and error paths
- Verify error context information
- Test error recovery mechanisms

This error handling system provides robust error management while maintaining usability and performance. Use the hierarchical structure to handle errors appropriately based on their category and severity.