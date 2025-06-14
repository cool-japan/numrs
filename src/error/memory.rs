//! Memory management errors - allocation failures, out of memory, corruption

use super::context::{ErrorSeverity, OperationContext};
use thiserror::Error;

/// Memory management errors
#[derive(Error, Debug, Clone)]
pub enum MemoryError {
    /// Memory allocation failed
    #[error("Memory allocation failed: {reason} (requested {requested_bytes} bytes)")]
    AllocationFailed {
        reason: String,
        requested_bytes: usize,
        available_bytes: Option<usize>,
        allocator_type: Option<String>,
        context: OperationContext,
    },

    /// Out of memory condition
    #[error("Out of memory: {details}")]
    OutOfMemory {
        details: String,
        total_requested: usize,
        system_available: Option<usize>,
        process_limit: Option<usize>,
        context: OperationContext,
    },

    /// Memory alignment error
    #[error("Memory alignment error: {reason} (required alignment: {required_alignment} bytes)")]
    AlignmentError {
        reason: String,
        required_alignment: usize,
        actual_alignment: Option<usize>,
        context: OperationContext,
    },

    /// Memory corruption detected
    #[error("Memory corruption detected: {details}")]
    MemoryCorruption {
        details: String,
        corruption_type: CorruptionType,
        affected_range: Option<(usize, usize)>, // (start, end) addresses
        context: OperationContext,
    },

    /// Invalid memory access
    #[error("Invalid memory access: {reason}")]
    InvalidAccess {
        reason: String,
        access_type: AccessType,
        address: Option<usize>,
        valid_range: Option<(usize, usize)>,
        context: OperationContext,
    },

    /// Memory leak detected
    #[error("Memory leak detected: {details}")]
    MemoryLeak {
        details: String,
        leaked_bytes: usize,
        allocation_source: Option<String>,
        context: OperationContext,
    },

    /// Memory fragmentation issue
    #[error("Memory fragmentation: {reason}")]
    Fragmentation {
        reason: String,
        fragmentation_level: f64, // 0.0 = no fragmentation, 1.0 = highly fragmented
        largest_free_block: Option<usize>,
        context: OperationContext,
    },

    /// Memory pool exhausted
    #[error("Memory pool exhausted: {pool_type} pool has no available blocks")]
    PoolExhausted {
        pool_type: String,
        pool_size: usize,
        block_size: usize,
        allocated_blocks: usize,
        context: OperationContext,
    },

    /// Arena allocator overflow
    #[error("Arena overflow: cannot allocate {requested_bytes} bytes (remaining: {remaining_bytes})")]
    ArenaOverflow {
        requested_bytes: usize,
        remaining_bytes: usize,
        arena_size: usize,
        context: OperationContext,
    },

    /// Memory mapping error
    #[error("Memory mapping failed: {reason}")]
    MappingError {
        reason: String,
        file_path: Option<String>,
        mapping_size: Option<usize>,
        context: OperationContext,
    },

    /// NUMA (Non-Uniform Memory Access) related error
    #[error("NUMA error: {reason}")]
    NumaError {
        reason: String,
        numa_node: Option<usize>,
        preferred_node: Option<usize>,
        context: OperationContext,
    },

    /// GPU memory error
    #[error("GPU memory error: {reason}")]
    GpuMemoryError {
        reason: String,
        device_id: Option<usize>,
        requested_bytes: Option<usize>,
        available_bytes: Option<usize>,
        context: OperationContext,
    },
}

/// Types of memory corruption
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorruptionType {
    /// Buffer overflow
    BufferOverflow,
    /// Buffer underflow
    BufferUnderflow,
    /// Use after free
    UseAfterFree,
    /// Double free
    DoubleFree,
    /// Invalid pointer
    InvalidPointer,
    /// Stack overflow
    StackOverflow,
    /// Heap corruption
    HeapCorruption,
}

/// Types of memory access
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessType {
    /// Read access
    Read,
    /// Write access
    Write,
    /// Execute access
    Execute,
    /// Mixed read/write
    ReadWrite,
}

impl MemoryError {
    /// Get the severity level of this error
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            MemoryError::AllocationFailed { .. } => ErrorSeverity::Critical,
            MemoryError::OutOfMemory { .. } => ErrorSeverity::Critical,
            MemoryError::AlignmentError { .. } => ErrorSeverity::High,
            MemoryError::MemoryCorruption { corruption_type, .. } => {
                match corruption_type {
                    CorruptionType::UseAfterFree | 
                    CorruptionType::DoubleFree | 
                    CorruptionType::StackOverflow => ErrorSeverity::Critical,
                    _ => ErrorSeverity::High,
                }
            },
            MemoryError::InvalidAccess { .. } => ErrorSeverity::High,
            MemoryError::MemoryLeak { leaked_bytes, .. } => {
                // Large leaks are more critical
                if *leaked_bytes > 1_000_000 { // > 1MB
                    ErrorSeverity::High
                } else {
                    ErrorSeverity::Medium
                }
            },
            MemoryError::Fragmentation { fragmentation_level, .. } => {
                if *fragmentation_level > 0.8 {
                    ErrorSeverity::Medium
                } else {
                    ErrorSeverity::Low
                }
            },
            MemoryError::PoolExhausted { .. } => ErrorSeverity::High,
            MemoryError::ArenaOverflow { .. } => ErrorSeverity::High,
            MemoryError::MappingError { .. } => ErrorSeverity::High,
            MemoryError::NumaError { .. } => ErrorSeverity::Medium,
            MemoryError::GpuMemoryError { .. } => ErrorSeverity::High,
        }
    }

    /// Get the operation context
    pub fn context(&self) -> &OperationContext {
        match self {
            MemoryError::AllocationFailed { context, .. } => context,
            MemoryError::OutOfMemory { context, .. } => context,
            MemoryError::AlignmentError { context, .. } => context,
            MemoryError::MemoryCorruption { context, .. } => context,
            MemoryError::InvalidAccess { context, .. } => context,
            MemoryError::MemoryLeak { context, .. } => context,
            MemoryError::Fragmentation { context, .. } => context,
            MemoryError::PoolExhausted { context, .. } => context,
            MemoryError::ArenaOverflow { context, .. } => context,
            MemoryError::MappingError { context, .. } => context,
            MemoryError::NumaError { context, .. } => context,
            MemoryError::GpuMemoryError { context, .. } => context,
        }
    }

    /// Check if this error is likely to be transient (might succeed on retry)
    pub fn is_transient(&self) -> bool {
        matches!(self,
            MemoryError::AllocationFailed { .. } |
            MemoryError::Fragmentation { .. } |
            MemoryError::PoolExhausted { .. }
        )
    }

    /// Get suggested recovery actions
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            MemoryError::AllocationFailed { requested_bytes, available_bytes, .. } => {
                let mut suggestions = vec![
                    "Reduce memory usage by processing data in smaller chunks".to_string(),
                    "Consider using streaming or out-of-core algorithms".to_string(),
                    "Free unused memory before attempting allocation".to_string(),
                ];
                
                if let Some(available) = available_bytes {
                    suggestions.push(format!(
                        "Requested {} bytes, but only {} bytes available", 
                        requested_bytes, available
                    ));
                }
                suggestions
            },
            MemoryError::OutOfMemory { total_requested, system_available, .. } => {
                vec![
                    format!("Reduce memory usage (requested: {} bytes)", total_requested),
                    if let Some(available) = system_available {
                        format!("System has {} bytes available", available)
                    } else {
                        "Check system memory availability".to_string()
                    },
                    "Use memory-mapped files for large datasets".to_string(),
                    "Consider distributed computing for very large problems".to_string(),
                ]
            },
            MemoryError::AlignmentError { required_alignment, .. } => {
                vec![
                    format!("Use properly aligned memory (required: {} bytes)", required_alignment),
                    "Use aligned allocation functions".to_string(),
                    "Check data structure packing and alignment".to_string(),
                ]
            },
            MemoryError::MemoryCorruption { corruption_type, .. } => {
                match corruption_type {
                    CorruptionType::BufferOverflow => vec![
                        "Check array bounds before access".to_string(),
                        "Use safe indexing methods".to_string(),
                        "Validate input sizes".to_string(),
                    ],
                    CorruptionType::UseAfterFree => vec![
                        "Avoid using freed memory".to_string(),
                        "Use smart pointers or RAII".to_string(),
                        "Check object lifetimes".to_string(),
                    ],
                    CorruptionType::DoubleFree => vec![
                        "Avoid freeing the same memory twice".to_string(),
                        "Set pointers to null after freeing".to_string(),
                        "Use automatic memory management".to_string(),
                    ],
                    _ => vec!["Review memory management code".to_string()],
                }
            },
            MemoryError::Fragmentation { fragmentation_level, .. } => {
                vec![
                    format!("Memory is {:.1}% fragmented", fragmentation_level * 100.0),
                    "Use memory pools for frequent allocations".to_string(),
                    "Consider memory defragmentation".to_string(),
                    "Use arena allocators for temporary memory".to_string(),
                ]
            },
            MemoryError::PoolExhausted { pool_type, block_size, .. } => {
                vec![
                    format!("Increase {} pool size", pool_type),
                    format!("Current block size: {} bytes", block_size),
                    "Consider using different allocation strategy".to_string(),
                    "Free unused blocks before allocating new ones".to_string(),
                ]
            },
            MemoryError::ArenaOverflow { remaining_bytes, arena_size, .. } => {
                vec![
                    format!("Increase arena size (current: {} bytes)", arena_size),
                    format!("Only {} bytes remaining in arena", remaining_bytes),
                    "Reset arena to reclaim space".to_string(),
                    "Use multiple arenas for large allocations".to_string(),
                ]
            },
            MemoryError::GpuMemoryError { device_id, available_bytes, .. } => {
                let mut suggestions = vec![
                    "Reduce GPU memory usage".to_string(),
                    "Transfer data in smaller batches".to_string(),
                ];
                if let Some(device) = device_id {
                    suggestions.push(format!("Error on GPU device {}", device));
                }
                if let Some(available) = available_bytes {
                    suggestions.push(format!("GPU has {} bytes available", available));
                }
                suggestions
            },
            _ => vec!["Review memory usage patterns".to_string()],
        }
    }
}

/// Convenience constructors for common memory errors
impl MemoryError {
    /// Create an allocation failed error
    pub fn allocation_failed(reason: &str, requested_bytes: usize) -> Self {
        MemoryError::AllocationFailed {
            reason: reason.to_string(),
            requested_bytes,
            available_bytes: None,
            allocator_type: None,
            context: OperationContext::default(),
        }
    }

    /// Create an out of memory error
    pub fn out_of_memory(details: &str, total_requested: usize) -> Self {
        MemoryError::OutOfMemory {
            details: details.to_string(),
            total_requested,
            system_available: None,
            process_limit: None,
            context: OperationContext::default(),
        }
    }

    /// Create an alignment error
    pub fn alignment_error(reason: &str, required_alignment: usize) -> Self {
        MemoryError::AlignmentError {
            reason: reason.to_string(),
            required_alignment,
            actual_alignment: None,
            context: OperationContext::default(),
        }
    }

    /// Create a memory corruption error
    pub fn memory_corruption(details: &str, corruption_type: CorruptionType) -> Self {
        MemoryError::MemoryCorruption {
            details: details.to_string(),
            corruption_type,
            affected_range: None,
            context: OperationContext::default(),
        }
    }

    /// Create a pool exhausted error
    pub fn pool_exhausted(pool_type: &str, pool_size: usize, block_size: usize) -> Self {
        MemoryError::PoolExhausted {
            pool_type: pool_type.to_string(),
            pool_size,
            block_size,
            allocated_blocks: pool_size,
            context: OperationContext::default(),
        }
    }

    /// Create an arena overflow error
    pub fn arena_overflow(requested_bytes: usize, remaining_bytes: usize, arena_size: usize) -> Self {
        MemoryError::ArenaOverflow {
            requested_bytes,
            remaining_bytes,
            arena_size,
            context: OperationContext::default(),
        }
    }

    /// Create a GPU memory error
    pub fn gpu_memory_error(reason: &str, device_id: Option<usize>) -> Self {
        MemoryError::GpuMemoryError {
            reason: reason.to_string(),
            device_id,
            requested_bytes: None,
            available_bytes: None,
            context: OperationContext::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_failed_severity() {
        let err = MemoryError::allocation_failed("System out of memory", 1024);
        assert_eq!(err.severity(), ErrorSeverity::Critical);
        assert!(err.is_transient());
    }

    #[test]
    fn test_memory_corruption_severity() {
        let use_after_free = MemoryError::memory_corruption("Used freed pointer", CorruptionType::UseAfterFree);
        assert_eq!(use_after_free.severity(), ErrorSeverity::Critical);

        let buffer_overflow = MemoryError::memory_corruption("Array access out of bounds", CorruptionType::BufferOverflow);
        assert_eq!(buffer_overflow.severity(), ErrorSeverity::High);
    }

    #[test]
    fn test_memory_leak_severity() {
        let small_leak = MemoryError::MemoryLeak {
            details: "Small leak".to_string(),
            leaked_bytes: 1024,
            allocation_source: None,
            context: OperationContext::default(),
        };
        assert_eq!(small_leak.severity(), ErrorSeverity::Medium);

        let large_leak = MemoryError::MemoryLeak {
            details: "Large leak".to_string(),
            leaked_bytes: 10_000_000,
            allocation_source: None,
            context: OperationContext::default(),
        };
        assert_eq!(large_leak.severity(), ErrorSeverity::High);
    }

    #[test]
    fn test_fragmentation_suggestions() {
        let err = MemoryError::Fragmentation {
            reason: "High fragmentation".to_string(),
            fragmentation_level: 0.9,
            largest_free_block: Some(1024),
            context: OperationContext::default(),
        };

        let suggestions = err.recovery_suggestions();
        assert!(suggestions.iter().any(|s| s.contains("90.0% fragmented")));
        assert!(suggestions.iter().any(|s| s.contains("memory pools")));
    }

    #[test]
    fn test_arena_overflow() {
        let err = MemoryError::arena_overflow(2048, 512, 4096);
        let suggestions = err.recovery_suggestions();
        assert!(suggestions.iter().any(|s| s.contains("4096 bytes")));
        assert!(suggestions.iter().any(|s| s.contains("512 bytes remaining")));
    }

    #[test]
    fn test_gpu_memory_error() {
        let err = MemoryError::gpu_memory_error("Insufficient GPU memory", Some(0));
        assert_eq!(err.severity(), ErrorSeverity::High);
        
        let suggestions = err.recovery_suggestions();
        assert!(suggestions.iter().any(|s| s.contains("GPU device 0")));
    }
}