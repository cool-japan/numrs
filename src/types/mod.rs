//! Advanced data types and type system for NumRS
//!
//! This module provides various data types beyond the basic numeric types,
//! including datetime, timedelta, structured arrays, and record arrays.

pub mod custom;
pub mod datetime;
pub mod structured;

// Re-export the most commonly used types
pub use custom::CustomDType;
pub use datetime::{DateTime64, DateTimeUnit, DateUnit, TimeDelta64};
pub use structured::{DType, Field, RecordArray, StructuredArray};
