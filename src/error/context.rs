//! Error context and metadata for enhanced error reporting

use std::collections::HashMap;
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    /// Low severity - warnings, missing optional features
    Low,
    /// Medium severity - recoverable errors, invalid input
    Medium,
    /// High severity - mathematical errors, dimension mismatches
    High,
    /// Critical severity - memory allocation failures, system errors
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Low => write!(f, "LOW"),
            ErrorSeverity::Medium => write!(f, "MEDIUM"),
            ErrorSeverity::High => write!(f, "HIGH"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Location information for errors
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorLocation {
    /// Source file name
    pub file: String,
    /// Line number
    pub line: u32,
    /// Function or method name
    pub function: String,
    /// Optional module path
    pub module_path: Option<String>,
}

impl ErrorLocation {
    /// Create a new error location
    pub fn new(file: &str, line: u32, function: &str) -> Self {
        Self {
            file: file.to_string(),
            line,
            function: function.to_string(),
            module_path: None,
        }
    }

    /// Create an error location with module path
    pub fn with_module(file: &str, line: u32, function: &str, module_path: &str) -> Self {
        Self {
            file: file.to_string(),
            line,
            function: function.to_string(),
            module_path: Some(module_path.to_string()),
        }
    }
}

impl fmt::Display for ErrorLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref module) = self.module_path {
            write!(f, "{}::{} ({}:{})", module, self.function, self.file, self.line)
        } else {
            write!(f, "{} ({}:{})", self.function, self.file, self.line)
        }
    }
}

/// Operation context for errors
#[derive(Debug, Clone)]
pub struct OperationContext {
    /// The operation being performed
    pub operation: Option<String>,
    /// Input parameters or relevant data
    pub parameters: HashMap<String, String>,
    /// Array shapes involved in the operation
    pub shapes: Vec<Vec<usize>>,
    /// Data types involved
    pub dtypes: Vec<String>,
    /// Memory usage information
    pub memory_info: Option<MemoryInfo>,
    /// Thread information
    pub thread_info: Option<ThreadInfo>,
    /// Performance hints
    pub performance_hints: Vec<String>,
    /// Timestamp when the operation started
    pub timestamp: u64,
}

impl Default for OperationContext {
    fn default() -> Self {
        Self {
            operation: None,
            parameters: HashMap::new(),
            shapes: Vec::new(),
            dtypes: Vec::new(),
            memory_info: None,
            thread_info: None,
            performance_hints: Vec::new(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        }
    }
}

impl OperationContext {
    /// Create a new operation context
    pub fn new(operation: &str) -> Self {
        Self {
            operation: Some(operation.to_string()),
            ..Default::default()
        }
    }

    /// Add a parameter to the context
    pub fn with_parameter<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: fmt::Display,
    {
        self.parameters.insert(key.into(), value.to_string());
        self
    }

    /// Add array shape information
    pub fn with_shape(mut self, shape: Vec<usize>) -> Self {
        self.shapes.push(shape);
        self
    }

    /// Add array shapes information
    pub fn with_shapes(mut self, shapes: &[Vec<usize>]) -> Self {
        self.shapes.extend_from_slice(shapes);
        self
    }

    /// Add data type information
    pub fn with_dtype(mut self, dtype: &str) -> Self {
        self.dtypes.push(dtype.to_string());
        self
    }

    /// Add memory information
    pub fn with_memory_info(mut self, memory_info: MemoryInfo) -> Self {
        self.memory_info = Some(memory_info);
        self
    }

    /// Add thread information
    pub fn with_thread_info(mut self, thread_info: ThreadInfo) -> Self {
        self.thread_info = Some(thread_info);
        self
    }

    /// Add a performance hint
    pub fn with_performance_hint(mut self, hint: &str) -> Self {
        self.performance_hints.push(hint.to_string());
        self
    }
}

/// Memory usage information
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total bytes allocated
    pub total_allocated: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Available memory
    pub available_memory: Option<usize>,
    /// Memory pressure level
    pub pressure_level: MemoryPressure,
}

/// Memory pressure levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryPressure {
    /// Plenty of memory available
    Low,
    /// Moderate memory usage
    Medium,
    /// High memory usage, consider optimization
    High,
    /// Critical memory usage, operations may fail
    Critical,
}

/// Thread execution information
#[derive(Debug, Clone)]
pub struct ThreadInfo {
    /// Current thread ID
    pub thread_id: String,
    /// Whether the operation is running in parallel
    pub is_parallel: bool,
    /// Number of threads being used
    pub thread_count: Option<usize>,
    /// Thread pool information
    pub pool_info: Option<String>,
}

/// Enhanced error wrapper with context
#[derive(Debug)]
pub struct ErrorContext<E> {
    /// The underlying error
    error: E,
    /// Operation context when the error occurred
    context: OperationContext,
    /// Location where the error occurred
    location: Option<ErrorLocation>,
    /// Error chain (parent errors)
    chain: Vec<Box<dyn std::error::Error + Send + Sync>>,
    /// Suggested recovery actions
    recovery_suggestions: Vec<String>,
}

impl<E> ErrorContext<E> {
    /// Create a new error context
    pub fn new(error: E, context: OperationContext) -> Self {
        Self {
            error,
            context,
            location: None,
            chain: Vec::new(),
            recovery_suggestions: Vec::new(),
        }
    }

    /// Add location information
    pub fn with_location(mut self, location: ErrorLocation) -> Self {
        self.location = Some(location);
        self
    }

    /// Add a parent error to the chain
    pub fn with_source<S>(mut self, source: S) -> Self 
    where
        S: std::error::Error + Send + Sync + 'static,
    {
        self.chain.push(Box::new(source));
        self
    }

    /// Add a recovery suggestion
    pub fn with_suggestion(mut self, suggestion: &str) -> Self {
        self.recovery_suggestions.push(suggestion.to_string());
        self
    }

    /// Get the underlying error
    pub fn error(&self) -> &E {
        &self.error
    }

    /// Get the operation context
    pub fn context(&self) -> &OperationContext {
        &self.context
    }

    /// Get the error location
    pub fn location(&self) -> Option<&ErrorLocation> {
        self.location.as_ref()
    }

    /// Get the error chain
    pub fn chain(&self) -> &[Box<dyn std::error::Error + Send + Sync>] {
        &self.chain
    }

    /// Get recovery suggestions
    pub fn recovery_suggestions(&self) -> &[String] {
        &self.recovery_suggestions
    }

    /// Consume the wrapper and return the underlying error
    pub fn into_inner(self) -> E {
        self.error
    }
}

impl<E: fmt::Display> fmt::Display for ErrorContext<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.error)?;

        if let Some(ref location) = self.location {
            write!(f, " at {}", location)?;
        }

        if let Some(ref operation) = self.context.operation {
            write!(f, " during '{}'", operation)?;
        }

        if !self.context.shapes.is_empty() {
            write!(f, " [shapes: {:?}]", self.context.shapes)?;
        }

        if !self.recovery_suggestions.is_empty() {
            write!(f, "\nSuggestions: {}", self.recovery_suggestions.join(", "))?;
        }

        Ok(())
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ErrorContext<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.chain.first().map(|e| e.as_ref() as &(dyn std::error::Error + 'static))
    }
}

/// Macro for creating error location from current position
#[macro_export]
macro_rules! error_location {
    () => {
        $crate::error::ErrorLocation::with_module(
            file!(), 
            line!(), 
            stringify!($crate::error::error_location),
            module_path!()
        )
    };
    ($func:expr) => {
        $crate::error::ErrorLocation::with_module(
            file!(), 
            line!(), 
            $func,
            module_path!()
        )
    };
}

/// Macro for creating operation context with current function
#[macro_export]
macro_rules! operation_context {
    ($op:expr) => {
        $crate::error::OperationContext::new($op)
    };
    ($op:expr, $($key:expr => $value:expr),*) => {
        {
            let mut ctx = $crate::error::OperationContext::new($op);
            $(
                ctx = ctx.with_parameter($key, $value);
            )*
            ctx
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_location() {
        let location = ErrorLocation::new("test.rs", 42, "test_function");
        assert_eq!(location.file, "test.rs");
        assert_eq!(location.line, 42);
        assert_eq!(location.function, "test_function");
    }

    #[test]
    fn test_operation_context() {
        let ctx = OperationContext::new("matrix_multiply")
            .with_parameter("method", "BLAS")
            .with_shape(vec![100, 50])
            .with_shape(vec![50, 75])
            .with_dtype("f64");

        assert_eq!(ctx.operation, Some("matrix_multiply".to_string()));
        assert_eq!(ctx.shapes.len(), 2);
        assert_eq!(ctx.dtypes, vec!["f64"]);
        assert!(ctx.parameters.contains_key("method"));
    }

    #[test]
    fn test_error_context() {
        let error = "Test error";
        let ctx = OperationContext::new("test_operation");
        let location = ErrorLocation::new("test.rs", 100, "test_function");

        let error_ctx = ErrorContext::new(error, ctx)
            .with_location(location)
            .with_suggestion("Try a different approach");

        assert_eq!(*error_ctx.error(), "Test error");
        assert!(error_ctx.location().is_some());
        assert_eq!(error_ctx.recovery_suggestions().len(), 1);
    }

    #[test]
    fn test_memory_info() {
        let memory = MemoryInfo {
            total_allocated: 1024,
            peak_usage: 2048,
            available_memory: Some(8192),
            pressure_level: MemoryPressure::Medium,
        };

        let ctx = OperationContext::default().with_memory_info(memory);
        assert!(ctx.memory_info.is_some());
        assert_eq!(ctx.memory_info.unwrap().total_allocated, 1024);
    }
}