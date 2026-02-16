//! Date and time unit types for NumRS
//!
//! This module defines the unit types used for date and datetime operations.

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::error::{NumRs2Error, Result};

/// Represents the unit of time for date operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DateUnit {
    /// Year unit
    Year,
    /// Month unit
    Month,
    /// Week unit
    Week,
    /// Day unit
    Day,
}

/// Represents the unit of time for datetime and timedelta operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DateTimeUnit {
    /// Year unit
    Year,
    /// Month unit
    Month,
    /// Week unit
    Week,
    /// Day unit
    Day,
    /// Hour unit
    Hour,
    /// Minute unit
    Minute,
    /// Second unit
    Second,
    /// Millisecond unit
    Millisecond,
    /// Microsecond unit
    Microsecond,
    /// Nanosecond unit
    Nanosecond,
}

impl fmt::Display for DateTimeUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DateTimeUnit::Year => write!(f, "Y"),
            DateTimeUnit::Month => write!(f, "M"),
            DateTimeUnit::Week => write!(f, "W"),
            DateTimeUnit::Day => write!(f, "D"),
            DateTimeUnit::Hour => write!(f, "h"),
            DateTimeUnit::Minute => write!(f, "m"),
            DateTimeUnit::Second => write!(f, "s"),
            DateTimeUnit::Millisecond => write!(f, "ms"),
            DateTimeUnit::Microsecond => write!(f, "us"),
            DateTimeUnit::Nanosecond => write!(f, "ns"),
        }
    }
}

/// Helper function to parse unit strings in NumPy format
pub fn parse_unit_string(unit: &str) -> Result<DateTimeUnit> {
    match unit.to_lowercase().as_str() {
        "y" | "year" | "years" => Ok(DateTimeUnit::Year),
        "m" | "month" | "months" => Ok(DateTimeUnit::Month),
        "w" | "week" | "weeks" => Ok(DateTimeUnit::Week),
        "d" | "day" | "days" => Ok(DateTimeUnit::Day),
        "h" | "hour" | "hours" => Ok(DateTimeUnit::Hour),
        "min" | "minute" | "minutes" => Ok(DateTimeUnit::Minute),
        "s" | "sec" | "second" | "seconds" => Ok(DateTimeUnit::Second),
        "ms" | "millisecond" | "milliseconds" => Ok(DateTimeUnit::Millisecond),
        "us" | "microsecond" | "microseconds" => Ok(DateTimeUnit::Microsecond),
        "ns" | "nanosecond" | "nanoseconds" => Ok(DateTimeUnit::Nanosecond),
        _ => Err(NumRs2Error::ValueError(format!("Unknown unit: {}", unit))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datetime_unit_display() {
        assert_eq!(format!("{}", DateTimeUnit::Year), "Y");
        assert_eq!(format!("{}", DateTimeUnit::Month), "M");
        assert_eq!(format!("{}", DateTimeUnit::Week), "W");
        assert_eq!(format!("{}", DateTimeUnit::Day), "D");
        assert_eq!(format!("{}", DateTimeUnit::Hour), "h");
        assert_eq!(format!("{}", DateTimeUnit::Minute), "m");
        assert_eq!(format!("{}", DateTimeUnit::Second), "s");
        assert_eq!(format!("{}", DateTimeUnit::Millisecond), "ms");
        assert_eq!(format!("{}", DateTimeUnit::Microsecond), "us");
        assert_eq!(format!("{}", DateTimeUnit::Nanosecond), "ns");
    }

    #[test]
    fn test_parse_unit_string() {
        assert_eq!(
            parse_unit_string("D").expect("should parse day unit"),
            DateTimeUnit::Day
        );
        assert_eq!(
            parse_unit_string("s").expect("should parse second unit"),
            DateTimeUnit::Second
        );
        assert_eq!(
            parse_unit_string("ms").expect("should parse millisecond unit"),
            DateTimeUnit::Millisecond
        );
        assert!(parse_unit_string("invalid").is_err());
    }
}
