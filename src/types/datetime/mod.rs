//! Date and time functionality for NumRS
//!
//! This module provides data types for working with dates and times,
//! similar to NumPy's datetime64 and timedelta64.
//!
//! # Module Organization
//!
//! - [`units`] - Date and time unit enums (DateUnit, DateTimeUnit)
//! - [`mod@datetime64`] - DateTime64 type and date calculation functions
//! - [`mod@timedelta64`] - TimeDelta64 type and arithmetic operations
//! - [`timezone`] - Timezone support and business day calculations
//! - [`api`] - NumPy-compatible API functions and array operations
//!
//! # Examples
//!
//! ## Creating DateTime64 values
//!
//! ```
//! use numrs2::types::datetime::{DateTime64, DateTimeUnit, datetime64};
//!
//! // Using the constructor
//! let dt = DateTime64::new(100, DateTimeUnit::Second);
//!
//! // Using NumPy-compatible API
//! let dt2 = datetime64("2023-01-01", Some("D")).expect("should parse");
//! ```
//!
//! ## Working with timedeltas
//!
//! ```
//! use numrs2::types::datetime::{TimeDelta64, DateTimeUnit, timedelta64};
//!
//! // Using the constructor
//! let td = TimeDelta64::new(5, DateTimeUnit::Day);
//!
//! // Using NumPy-compatible API
//! let td2 = timedelta64(24, "h").expect("should create");
//! ```
//!
//! ## Timezone support
//!
//! ```
//! use numrs2::types::datetime::{DateTime64, DateTimeUnit, Timezone, TimezoneDateTime};
//!
//! let utc_dt = DateTime64::new(0, DateTimeUnit::Second);
//! let tz_dt = TimezoneDateTime::new(utc_dt, Timezone::est());
//! ```

// Submodules
pub mod api;
pub mod datetime64;
pub mod timedelta64;
pub mod timezone;
pub mod units;

// Re-export main types from units
pub use units::{parse_unit_string, DateTimeUnit, DateUnit};

// Re-export main types from datetime64
pub use datetime64::{
    date_from_days_since_epoch, days_in_month, days_since_epoch, days_to_date, is_leap_year,
    DateTime64,
};

// Re-export main types from timedelta64
pub use timedelta64::TimeDelta64;

// Re-export timezone types
pub use timezone::{business_days, Timezone, TimezoneDateTime};

// Re-export API functions
pub use api::{
    array_ops, datetime64, datetime_array, datetime_as_string, datetime_data, timedelta64,
};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_module_reexports() {
        // Verify all main types are accessible
        let _dt = DateTime64::new(0, DateTimeUnit::Second);
        let _td = TimeDelta64::new(1, DateTimeUnit::Day);
        let _tz = Timezone::utc();
    }

    #[test]
    fn test_api_functions() {
        // Verify API functions are accessible
        let dt = datetime64("2023-01-01", Some("D")).expect("should parse");
        let td = timedelta64(5, "D").expect("should create");
        let _s = datetime_as_string(&dt, None, None).expect("should convert");
        let (_unit, _count) = datetime_data(&dt);

        // Verify timedelta operations
        let dt2 = dt + td;
        assert!(dt2.value() > dt.value());
    }

    #[test]
    fn test_timezone_accessible() {
        let utc_dt = DateTime64::from_iso_string("2023-01-01T12:00:00", DateTimeUnit::Second)
            .expect("should parse");
        let tz_dt = TimezoneDateTime::new(utc_dt, Timezone::jst());
        assert_eq!(tz_dt.timezone.name, "JST");
    }

    #[test]
    fn test_business_days_accessible() {
        let dt =
            DateTime64::from_iso_string("2023-01-02", DateTimeUnit::Day).expect("should parse");
        let is_busday = business_days::is_busday(&dt).expect("should check");
        assert!(is_busday);
    }
}
