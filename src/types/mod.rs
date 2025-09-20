//! Advanced data types and type system for NumRS
//!
//! This module provides various data types beyond the basic numeric types,
//! including datetime, timedelta, structured arrays, and record arrays.

pub mod custom;
pub mod datetime;
pub mod structured;

// Re-export the most commonly used types
pub use custom::CustomDType;
pub use datetime::{
    business_days,
    // NumPy-compatible API functions
    datetime64,
    datetime_array,
    datetime_as_string,
    datetime_data,
    timedelta64,
    DateTime64,
    DateTimeUnit,
    DateUnit,
    TimeDelta64,
    Timezone,
    TimezoneDateTime,
};
pub use structured::{DType, Field, RecordArray, StructuredArray};
