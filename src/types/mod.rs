//! Advanced data types and type system for NumRS
//!
//! This module provides various data types beyond the basic numeric types,
//! including datetime, timedelta, structured arrays, and record arrays.

pub mod datetime;
pub mod structured;
pub mod custom;

// Re-export the most commonly used types
pub use datetime::{DateTime64, TimeDelta64, DateUnit, DateTimeUnit};
pub use structured::{StructuredArray, RecordArray, DType, Field};
pub use custom::CustomDType;
