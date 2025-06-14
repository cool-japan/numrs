//! Array operations and data structures
//!
//! This module provides core array functionality including basic operations,
//! linear algebra, statistical functions, and advanced NumPy-style operations.

// Advanced array operations (Phase 3)
pub mod advanced_ops;
pub mod broadcasting;
pub mod fancy_indexing;
pub mod shape_manipulation;
pub mod enhanced_indexing;
pub mod stride_optimization;

// Re-export advanced operations
pub use advanced_ops::*;
pub use broadcasting::*;
pub use fancy_indexing::*;
pub use shape_manipulation::*;
pub use enhanced_indexing::*;
pub use stride_optimization::*;