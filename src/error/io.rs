//! I/O and serialization errors - file operations, network, data format issues

use super::context::{ErrorSeverity, OperationContext};
use std::path::PathBuf;
use thiserror::Error;

/// I/O and serialization errors
#[derive(Error, Debug, Clone)]
pub enum IOError {
    /// File system operation failed
    #[error("File operation failed: {operation} on '{path}' - {reason}")]
    FileOperation {
        operation: String, // "read", "write", "create", "delete", etc.
        path: PathBuf,
        reason: String,
        io_error_kind: Option<std::io::ErrorKind>,
        context: OperationContext,
    },

    /// File format not supported or invalid
    #[error("Invalid file format: {format} - {details}")]
    InvalidFormat {
        format: String, // "NPY", "CSV", "HDF5", etc.
        details: String,
        file_path: Option<PathBuf>,
        expected_format: Option<String>,
        context: OperationContext,
    },

    /// Serialization failed
    #[error("Serialization failed: {format} - {reason}")]
    Serialization {
        format: String, // "bincode", "json", "messagepack", etc.
        reason: String,
        data_type: Option<String>,
        context: OperationContext,
    },

    /// Deserialization failed
    #[error("Deserialization failed: {format} - {reason}")]
    Deserialization {
        format: String,
        reason: String,
        file_path: Option<PathBuf>,
        data_size: Option<usize>,
        context: OperationContext,
    },

    /// Network operation failed
    #[error("Network error: {operation} failed - {reason}")]
    Network {
        operation: String, // "download", "upload", "connect", etc.
        reason: String,
        url: Option<String>,
        status_code: Option<u16>,
        context: OperationContext,
    },

    /// Data encoding/decoding error
    #[error("Encoding error: {encoding} - {details}")]
    Encoding {
        encoding: String, // "UTF-8", "Base64", "Hex", etc.
        details: String,
        byte_position: Option<usize>,
        context: OperationContext,
    },

    /// Compression/decompression error
    #[error("Compression error: {algorithm} - {reason}")]
    Compression {
        algorithm: String, // "gzip", "zstd", "lz4", etc.
        reason: String,
        compression_ratio: Option<f64>,
        context: OperationContext,
    },

    /// Database operation error
    #[error("Database error: {operation} - {reason}")]
    Database {
        operation: String,
        reason: String,
        database_type: Option<String>, // "SQLite", "PostgreSQL", etc.
        connection_string: Option<String>,
        context: OperationContext,
    },

    /// Memory-mapped file error
    #[error("Memory mapping error: {reason}")]
    MemoryMapping {
        reason: String,
        file_path: PathBuf,
        mapping_size: Option<usize>,
        offset: Option<usize>,
        context: OperationContext,
    },

    /// Stream operation error
    #[error("Stream error: {reason}")]
    Stream {
        reason: String,
        stream_type: String, // "buffered", "async", "compressed", etc.
        bytes_processed: Option<usize>,
        context: OperationContext,
    },

    /// Checksum or validation error
    #[error("Data validation failed: {validation_type} - {details}")]
    Validation {
        validation_type: String, // "checksum", "hash", "signature", etc.
        details: String,
        expected_value: Option<String>,
        actual_value: Option<String>,
        context: OperationContext,
    },

    /// Temporary file/directory operation error
    #[error("Temporary resource error: {reason}")]
    TemporaryResource {
        reason: String,
        resource_type: String, // "file", "directory"
        cleanup_attempted: bool,
        context: OperationContext,
    },

    /// Permission/access error
    #[error("Access denied: {operation} on '{resource}' - {reason}")]
    AccessDenied {
        operation: String,
        resource: String,
        reason: String,
        required_permissions: Option<String>,
        context: OperationContext,
    },
}

impl IOError {
    /// Get the severity level of this error
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            IOError::FileOperation { io_error_kind, .. } => {
                match io_error_kind {
                    Some(std::io::ErrorKind::NotFound) => ErrorSeverity::Medium,
                    Some(std::io::ErrorKind::PermissionDenied) => ErrorSeverity::High,
                    Some(std::io::ErrorKind::OutOfMemory) => ErrorSeverity::Critical,
                    Some(std::io::ErrorKind::StorageFull) => ErrorSeverity::High,
                    _ => ErrorSeverity::Medium,
                }
            },
            IOError::InvalidFormat { .. } => ErrorSeverity::Medium,
            IOError::Serialization { .. } => ErrorSeverity::Medium,
            IOError::Deserialization { .. } => ErrorSeverity::Medium,
            IOError::Network { status_code, .. } => {
                match status_code {
                    Some(500..=599) => ErrorSeverity::High, // Server errors
                    Some(400..=499) => ErrorSeverity::Medium, // Client errors
                    _ => ErrorSeverity::Medium,
                }
            },
            IOError::Encoding { .. } => ErrorSeverity::Medium,
            IOError::Compression { .. } => ErrorSeverity::Medium,
            IOError::Database { .. } => ErrorSeverity::High,
            IOError::MemoryMapping { .. } => ErrorSeverity::High,
            IOError::Stream { .. } => ErrorSeverity::Medium,
            IOError::Validation { .. } => ErrorSeverity::High,
            IOError::TemporaryResource { .. } => ErrorSeverity::Low,
            IOError::AccessDenied { .. } => ErrorSeverity::High,
        }
    }

    /// Get the operation context
    pub fn context(&self) -> &OperationContext {
        match self {
            IOError::FileOperation { context, .. } => context,
            IOError::InvalidFormat { context, .. } => context,
            IOError::Serialization { context, .. } => context,
            IOError::Deserialization { context, .. } => context,
            IOError::Network { context, .. } => context,
            IOError::Encoding { context, .. } => context,
            IOError::Compression { context, .. } => context,
            IOError::Database { context, .. } => context,
            IOError::MemoryMapping { context, .. } => context,
            IOError::Stream { context, .. } => context,
            IOError::Validation { context, .. } => context,
            IOError::TemporaryResource { context, .. } => context,
            IOError::AccessDenied { context, .. } => context,
        }
    }

    /// Check if this error might be transient (worth retrying)
    pub fn is_transient(&self) -> bool {
        match self {
            IOError::Network { status_code, .. } => {
                // Temporary network issues
                matches!(status_code, Some(429) | Some(500..=503) | Some(504) | None)
            },
            IOError::FileOperation { io_error_kind, .. } => {
                // Some file operations might be transient
                matches!(io_error_kind, Some(std::io::ErrorKind::Interrupted) | Some(std::io::ErrorKind::TimedOut))
            },
            IOError::Database { .. } => true, // Database connections can be transient
            IOError::TemporaryResource { .. } => true,
            _ => false,
        }
    }

    /// Get suggested recovery actions
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            IOError::FileOperation { operation, path, io_error_kind, .. } => {
                let mut suggestions = vec![];
                
                match io_error_kind {
                    Some(std::io::ErrorKind::NotFound) => {
                        suggestions.push(format!("Check if file exists: {}", path.display()));
                        suggestions.push("Verify the file path is correct".to_string());
                        if operation == "read" {
                            suggestions.push("Create the file if it should exist".to_string());
                        }
                    },
                    Some(std::io::ErrorKind::PermissionDenied) => {
                        suggestions.push("Check file permissions".to_string());
                        suggestions.push("Run with appropriate privileges".to_string());
                        suggestions.push(format!("Ensure write access to directory: {}", 
                            path.parent().map(|p| p.display().to_string()).unwrap_or_default()));
                    },
                    Some(std::io::ErrorKind::StorageFull) => {
                        suggestions.push("Free up disk space".to_string());
                        suggestions.push("Use a different storage location".to_string());
                        suggestions.push("Consider compression or data cleanup".to_string());
                    },
                    _ => {
                        suggestions.push(format!("Retry the {} operation", operation));
                        suggestions.push("Check system resources and file locks".to_string());
                    }
                }
                suggestions
            },
            IOError::InvalidFormat { format, expected_format, file_path, .. } => {
                vec![
                    if let Some(expected) = expected_format {
                        format!("Convert file to {} format (currently: {})", expected, format)
                    } else {
                        format!("Verify the file format (detected: {})", format)
                    },
                    if let Some(path) = file_path {
                        format!("Check file contents: {}", path.display())
                    } else {
                        "Validate input data format".to_string()
                    },
                    "Use appropriate loader for the file type".to_string(),
                ]
            },
            IOError::Network { status_code, url, .. } => {
                let mut suggestions = vec!["Check network connectivity".to_string()];
                
                if let Some(code) = status_code {
                    match *code {
                        400..=499 => suggestions.push("Check request parameters and authentication".to_string()),
                        500..=599 => suggestions.push("Server error - retry later".to_string()),
                        429 => suggestions.push("Rate limited - reduce request frequency".to_string()),
                        _ => {},
                    }
                }
                
                if let Some(url_str) = url {
                    suggestions.push(format!("Verify URL is accessible: {}", url_str));
                }
                
                suggestions.push("Consider using offline data if available".to_string());
                suggestions
            },
            IOError::Serialization { format, data_type, .. } => {
                vec![
                    format!("Check data compatibility with {} format", format),
                    if let Some(dtype) = data_type {
                        format!("Verify {} data can be serialized", dtype)
                    } else {
                        "Check data types are serializable".to_string()
                    },
                    "Use alternative serialization format".to_string(),
                    "Implement custom serialization if needed".to_string(),
                ]
            },
            IOError::Deserialization { format, file_path, data_size, .. } => {
                vec![
                    format!("Verify file was created with compatible {} version", format),
                    if let Some(path) = file_path {
                        format!("Check file integrity: {}", path.display())
                    } else {
                        "Validate input data integrity".to_string()
                    },
                    if let Some(size) = data_size {
                        format!("File size: {} bytes - check if complete", size)
                    } else {
                        "Ensure complete data transfer".to_string()
                    },
                    "Try alternative deserialization method".to_string(),
                ]
            },
            IOError::Validation { validation_type, expected_value, actual_value, .. } => {
                vec![
                    format!("Data {} validation failed", validation_type),
                    if let (Some(expected), Some(actual)) = (expected_value, actual_value) {
                        format!("Expected: {}, Got: {}", expected, actual)
                    } else {
                        "Check data integrity and source".to_string()
                    },
                    "Re-download or regenerate the data".to_string(),
                    "Use data without validation if corruption is acceptable".to_string(),
                ]
            },
            IOError::MemoryMapping { file_path, mapping_size, .. } => {
                vec![
                    format!("Check file accessibility: {}", file_path.display()),
                    if let Some(size) = mapping_size {
                        format!("Verify mapping size ({} bytes) is valid", size)
                    } else {
                        "Check memory mapping parameters".to_string()
                    },
                    "Ensure sufficient virtual memory available".to_string(),
                    "Use regular file I/O instead of memory mapping".to_string(),
                ]
            },
            IOError::AccessDenied { operation, resource, required_permissions, .. } => {
                vec![
                    format!("Grant {} permissions for {}", operation, resource),
                    if let Some(perms) = required_permissions {
                        format!("Required permissions: {}", perms)
                    } else {
                        "Check user permissions and ownership".to_string()
                    },
                    "Run with administrator/root privileges if necessary".to_string(),
                    "Move files to accessible location".to_string(),
                ]
            },
            _ => vec!["Check I/O operation parameters and retry".to_string()],
        }
    }
}

/// Convenience constructors for common I/O errors
impl IOError {
    /// Create a file operation error
    pub fn file_operation(operation: &str, path: PathBuf, reason: &str) -> Self {
        IOError::FileOperation {
            operation: operation.to_string(),
            path,
            reason: reason.to_string(),
            io_error_kind: None,
            context: OperationContext::new(operation),
        }
    }

    /// Create an invalid format error
    pub fn invalid_format(format: &str, details: &str) -> Self {
        IOError::InvalidFormat {
            format: format.to_string(),
            details: details.to_string(),
            file_path: None,
            expected_format: None,
            context: OperationContext::default(),
        }
    }

    /// Create a serialization error
    pub fn serialization(format: &str, reason: &str) -> Self {
        IOError::Serialization {
            format: format.to_string(),
            reason: reason.to_string(),
            data_type: None,
            context: OperationContext::new("serialize"),
        }
    }

    /// Create a deserialization error
    pub fn deserialization(format: &str, reason: &str) -> Self {
        IOError::Deserialization {
            format: format.to_string(),
            reason: reason.to_string(),
            file_path: None,
            data_size: None,
            context: OperationContext::new("deserialize"),
        }
    }

    /// Create a network error
    pub fn network(operation: &str, reason: &str) -> Self {
        IOError::Network {
            operation: operation.to_string(),
            reason: reason.to_string(),
            url: None,
            status_code: None,
            context: OperationContext::new(operation),
        }
    }

    /// Create an access denied error
    pub fn access_denied(operation: &str, resource: &str, reason: &str) -> Self {
        IOError::AccessDenied {
            operation: operation.to_string(),
            resource: resource.to_string(),
            reason: reason.to_string(),
            required_permissions: None,
            context: OperationContext::new(operation),
        }
    }
}

impl From<std::io::Error> for IOError {
    fn from(err: std::io::Error) -> Self {
        IOError::FileOperation {
            operation: "io_operation".to_string(),
            path: PathBuf::from("unknown"),
            reason: err.to_string(),
            io_error_kind: Some(err.kind()),
            context: OperationContext::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_file_operation_severity() {
        let not_found = IOError::FileOperation {
            operation: "read".to_string(),
            path: PathBuf::from("test.txt"),
            reason: "File not found".to_string(),
            io_error_kind: Some(std::io::ErrorKind::NotFound),
            context: OperationContext::default(),
        };
        assert_eq!(not_found.severity(), ErrorSeverity::Medium);

        let permission_denied = IOError::FileOperation {
            operation: "write".to_string(),
            path: PathBuf::from("test.txt"),
            reason: "Permission denied".to_string(),
            io_error_kind: Some(std::io::ErrorKind::PermissionDenied),
            context: OperationContext::default(),
        };
        assert_eq!(permission_denied.severity(), ErrorSeverity::High);
    }

    #[test]
    fn test_network_error_transient() {
        let server_error = IOError::Network {
            operation: "download".to_string(),
            reason: "Server error".to_string(),
            url: None,
            status_code: Some(503),
            context: OperationContext::default(),
        };
        assert!(server_error.is_transient());

        let client_error = IOError::Network {
            operation: "download".to_string(),
            reason: "Bad request".to_string(),
            url: None,
            status_code: Some(400),
            context: OperationContext::default(),
        };
        assert!(!client_error.is_transient());
    }

    #[test]
    fn test_file_operation_suggestions() {
        let err = IOError::FileOperation {
            operation: "read".to_string(),
            path: PathBuf::from("/path/to/file.txt"),
            reason: "File not found".to_string(),
            io_error_kind: Some(std::io::ErrorKind::NotFound),
            context: OperationContext::default(),
        };
        let suggestions = err.recovery_suggestions();
        assert!(suggestions.iter().any(|s| s.contains("Check if file exists")));
    }

    #[test]
    fn test_invalid_format_error() {
        let err = IOError::invalid_format("NPY", "Invalid magic bytes");
        assert_eq!(err.severity(), ErrorSeverity::Medium);
        
        if let IOError::InvalidFormat { format, details, .. } = &err {
            assert_eq!(*format, "NPY");
            assert_eq!(*details, "Invalid magic bytes");
        } else {
            panic!("Expected InvalidFormat variant");
        }
    }

    #[test]
    fn test_validation_error() {
        let err = IOError::Validation {
            validation_type: "checksum".to_string(),
            details: "Checksum mismatch".to_string(),
            expected_value: Some("abc123".to_string()),
            actual_value: Some("def456".to_string()),
            context: OperationContext::default(),
        };

        let suggestions = err.recovery_suggestions();
        assert!(suggestions.iter().any(|s| s.contains("Expected: abc123, Got: def456")));
    }

    #[test]
    fn test_from_std_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let numrs_err = IOError::from(io_err);
        
        if let IOError::FileOperation { io_error_kind, .. } = &numrs_err {
            assert_eq!(*io_error_kind, Some(std::io::ErrorKind::NotFound));
        } else {
            panic!("Expected FileOperation variant");
        }
    }
}