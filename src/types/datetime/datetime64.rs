//! DateTime64 type for NumRS
//!
//! This module provides the DateTime64 type, similar to NumPy's datetime64,
//! storing a date and time as a 64-bit integer representing the number of
//! units since the Unix epoch.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;
use std::time::{Duration, SystemTime};

use crate::error::{NumRs2Error, Result};

use super::units::DateTimeUnit;
use super::TimeDelta64;

/// Represents a date and time value with a specified unit
///
/// This type is similar to NumPy's datetime64 type, storing a date and time
/// as a 64-bit integer representing the number of units since the Unix epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DateTime64 {
    /// Number of time units since the epoch
    value: i64,
    /// The unit of time
    unit: DateTimeUnit,
}

impl DateTime64 {
    /// Create a new DateTime64 with the specified value and unit
    pub fn new(value: i64, unit: DateTimeUnit) -> Self {
        Self { value, unit }
    }

    /// Create a DateTime64 from a SystemTime
    pub fn from_system_time(time: SystemTime, unit: DateTimeUnit) -> Result<Self> {
        use std::time::UNIX_EPOCH;
        let duration = time
            .duration_since(UNIX_EPOCH)
            .map_err(|e| NumRs2Error::ValueError(format!("Error converting SystemTime: {}", e)))?;

        let value = match unit {
            DateTimeUnit::Year => {
                // Approximate years since epoch
                (duration.as_secs() / (365 * 24 * 60 * 60)) as i64
            }
            DateTimeUnit::Month => {
                // Approximate months since epoch
                (duration.as_secs() / (30 * 24 * 60 * 60)) as i64
            }
            DateTimeUnit::Week => {
                // Weeks since epoch
                (duration.as_secs() / (7 * 24 * 60 * 60)) as i64
            }
            DateTimeUnit::Day => {
                // Days since epoch
                (duration.as_secs() / (24 * 60 * 60)) as i64
            }
            DateTimeUnit::Hour => {
                // Hours since epoch
                (duration.as_secs() / (60 * 60)) as i64
            }
            DateTimeUnit::Minute => {
                // Minutes since epoch
                (duration.as_secs() / 60) as i64
            }
            DateTimeUnit::Second => {
                // Seconds since epoch
                duration.as_secs() as i64
            }
            DateTimeUnit::Millisecond => {
                // Milliseconds since epoch
                (duration.as_secs() * 1000 + duration.subsec_millis() as u64) as i64
            }
            DateTimeUnit::Microsecond => {
                // Microseconds since epoch
                (duration.as_secs() * 1_000_000 + duration.subsec_micros() as u64) as i64
            }
            DateTimeUnit::Nanosecond => {
                // Nanoseconds since epoch
                (duration.as_secs() * 1_000_000_000 + duration.subsec_nanos() as u64) as i64
            }
        };

        Ok(Self { value, unit })
    }

    /// Get the raw value
    pub fn value(&self) -> i64 {
        self.value
    }

    /// Get the unit
    pub fn unit(&self) -> DateTimeUnit {
        self.unit
    }

    /// Convert to a different unit
    pub fn to_unit(&self, unit: DateTimeUnit) -> Self {
        if self.unit == unit {
            return *self;
        }

        // First convert to nanoseconds as a common intermediate format
        let ns_value = match self.unit {
            DateTimeUnit::Year => self.value * 365 * 24 * 60 * 60 * 1_000_000_000,
            DateTimeUnit::Month => self.value * 30 * 24 * 60 * 60 * 1_000_000_000,
            DateTimeUnit::Week => self.value * 7 * 24 * 60 * 60 * 1_000_000_000,
            DateTimeUnit::Day => self.value * 24 * 60 * 60 * 1_000_000_000,
            DateTimeUnit::Hour => self.value * 60 * 60 * 1_000_000_000,
            DateTimeUnit::Minute => self.value * 60 * 1_000_000_000,
            DateTimeUnit::Second => self.value * 1_000_000_000,
            DateTimeUnit::Millisecond => self.value * 1_000_000,
            DateTimeUnit::Microsecond => self.value * 1_000,
            DateTimeUnit::Nanosecond => self.value,
        };

        // Then convert from nanoseconds to the target unit
        let target_value = match unit {
            DateTimeUnit::Year => ns_value / (365 * 24 * 60 * 60 * 1_000_000_000),
            DateTimeUnit::Month => ns_value / (30 * 24 * 60 * 60 * 1_000_000_000),
            DateTimeUnit::Week => ns_value / (7 * 24 * 60 * 60 * 1_000_000_000),
            DateTimeUnit::Day => ns_value / (24 * 60 * 60 * 1_000_000_000),
            DateTimeUnit::Hour => ns_value / (60 * 60 * 1_000_000_000),
            DateTimeUnit::Minute => ns_value / (60 * 1_000_000_000),
            DateTimeUnit::Second => ns_value / 1_000_000_000,
            DateTimeUnit::Millisecond => ns_value / 1_000_000,
            DateTimeUnit::Microsecond => ns_value / 1_000,
            DateTimeUnit::Nanosecond => ns_value,
        };

        Self {
            value: target_value,
            unit,
        }
    }

    /// Convert to a SystemTime
    pub fn to_system_time(&self) -> SystemTime {
        use std::time::UNIX_EPOCH;
        // Convert to seconds and nanoseconds
        let (secs, nanos) = match self.unit {
            DateTimeUnit::Year => {
                let secs = self.value * 365 * 24 * 60 * 60;
                (secs, 0)
            }
            DateTimeUnit::Month => {
                let secs = self.value * 30 * 24 * 60 * 60;
                (secs, 0)
            }
            DateTimeUnit::Week => {
                let secs = self.value * 7 * 24 * 60 * 60;
                (secs, 0)
            }
            DateTimeUnit::Day => {
                let secs = self.value * 24 * 60 * 60;
                (secs, 0)
            }
            DateTimeUnit::Hour => {
                let secs = self.value * 60 * 60;
                (secs, 0)
            }
            DateTimeUnit::Minute => {
                let secs = self.value * 60;
                (secs, 0)
            }
            DateTimeUnit::Second => (self.value, 0),
            DateTimeUnit::Millisecond => {
                let secs = self.value / 1000;
                let nanos = (self.value % 1000) * 1_000_000;
                (secs, nanos as u32)
            }
            DateTimeUnit::Microsecond => {
                let secs = self.value / 1_000_000;
                let nanos = (self.value % 1_000_000) * 1_000;
                (secs, nanos as u32)
            }
            DateTimeUnit::Nanosecond => {
                let secs = self.value / 1_000_000_000;
                let nanos = (self.value % 1_000_000_000) as u32;
                (secs, nanos)
            }
        };

        UNIX_EPOCH + Duration::new(secs as u64, nanos)
    }

    /// Parse a DateTime64 from an ISO 8601 string
    ///
    /// Supports formats like "2023-12-25T15:30:45" or "2023-12-25"
    pub fn from_iso_string(s: &str, unit: DateTimeUnit) -> Result<Self> {
        // Basic ISO 8601 parsing - simplified implementation
        let parts: Vec<&str> = s.split('T').collect();
        if parts.is_empty() {
            return Err(NumRs2Error::ValueError("Invalid date format".to_string()));
        }

        let date_part = parts[0];
        let time_part = parts.get(1).unwrap_or(&"00:00:00");

        // Parse date (YYYY-MM-DD)
        let date_components: Vec<&str> = date_part.split('-').collect();
        if date_components.len() != 3 {
            return Err(NumRs2Error::ValueError(
                "Invalid date format, expected YYYY-MM-DD".to_string(),
            ));
        }

        let year: i32 = date_components[0]
            .parse()
            .map_err(|_| NumRs2Error::ValueError("Invalid year".to_string()))?;
        let month: u32 = date_components[1]
            .parse()
            .map_err(|_| NumRs2Error::ValueError("Invalid month".to_string()))?;
        let day: u32 = date_components[2]
            .parse()
            .map_err(|_| NumRs2Error::ValueError("Invalid day".to_string()))?;

        // Parse time (HH:MM:SS)
        let time_components: Vec<&str> = time_part.split(':').collect();
        let hour: u32 = if !time_components.is_empty() {
            time_components[0].parse().unwrap_or(0)
        } else {
            0
        };
        let minute: u32 = if time_components.len() >= 2 {
            time_components[1].parse().unwrap_or(0)
        } else {
            0
        };
        let second: u32 = if time_components.len() >= 3 {
            time_components[2].parse().unwrap_or(0)
        } else {
            0
        };

        // Calculate days since Unix epoch (1970-01-01)
        let days_since_epoch = days_since_epoch(year, month, day)?;
        let seconds_in_day = hour * 3600 + minute * 60 + second;
        let total_seconds = days_since_epoch * 86400 + seconds_in_day as i64;

        // Convert to the requested unit
        let value = match unit {
            DateTimeUnit::Year => days_since_epoch / 365,
            DateTimeUnit::Month => days_since_epoch / 30,
            DateTimeUnit::Week => days_since_epoch / 7,
            DateTimeUnit::Day => days_since_epoch,
            DateTimeUnit::Hour => total_seconds / 3600,
            DateTimeUnit::Minute => total_seconds / 60,
            DateTimeUnit::Second => total_seconds,
            DateTimeUnit::Millisecond => total_seconds * 1000,
            DateTimeUnit::Microsecond => total_seconds * 1_000_000,
            DateTimeUnit::Nanosecond => total_seconds * 1_000_000_000,
        };

        Ok(Self { value, unit })
    }

    /// Format as ISO 8601 string
    pub fn to_iso_string(&self) -> Result<String> {
        let dt_seconds = self.to_unit(DateTimeUnit::Second);
        let total_seconds = dt_seconds.value;

        // Convert to date components
        let days_since_epoch = total_seconds / 86400;
        let seconds_in_day = (total_seconds % 86400) as u32;

        let (year, month, day) = date_from_days_since_epoch(days_since_epoch)?;
        let hour = seconds_in_day / 3600;
        let minute = (seconds_in_day % 3600) / 60;
        let second = seconds_in_day % 60;

        Ok(format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            year, month, day, hour, minute, second
        ))
    }
}

// Arithmetic operations are implemented in the timedelta64 module

impl fmt::Display for DateTime64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.value, self.unit)
    }
}

impl FromStr for DateTime64 {
    type Err = NumRs2Error;

    fn from_str(s: &str) -> Result<Self> {
        // Default to second precision for parsing
        Self::from_iso_string(s, DateTimeUnit::Second)
    }
}

// ============================================================================
// Date calculation helper functions
// ============================================================================

/// Calculate days since Unix epoch (1970-01-01)
pub fn days_since_epoch(year: i32, month: u32, day: u32) -> Result<i64> {
    if !(1..=12).contains(&month) {
        return Err(NumRs2Error::ValueError("Invalid month".to_string()));
    }
    if day < 1 || day > days_in_month(year, month) {
        return Err(NumRs2Error::ValueError("Invalid day".to_string()));
    }

    // Days in each month (non-leap year)
    const DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    // Calculate total days
    let mut total_days = 0i64;

    // Add days for complete years
    for y in 1970..year {
        total_days += if is_leap_year(y) { 366 } else { 365 };
    }

    // Add days for complete months in the current year
    for m in 1..month {
        total_days += DAYS_IN_MONTH[(m - 1) as usize] as i64;
        if m == 2 && is_leap_year(year) {
            total_days += 1; // Leap day
        }
    }

    // Add remaining days
    total_days += (day - 1) as i64;

    Ok(total_days)
}

/// Convert days since epoch to date components
pub fn date_from_days_since_epoch(days: i64) -> Result<(i32, u32, u32)> {
    let mut remaining_days = days;
    let mut year = 1970i32;

    // Find the year
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days >= days_in_year {
            remaining_days -= days_in_year;
            year += 1;
        } else {
            break;
        }
    }

    // Find the month and day
    const DAYS_IN_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1u32;

    for m in 1..=12 {
        let mut days_in_month = DAYS_IN_MONTH[(m - 1) as usize] as i64;
        if m == 2 && is_leap_year(year) {
            days_in_month += 1; // Leap day
        }

        if remaining_days >= days_in_month {
            remaining_days -= days_in_month;
            month += 1;
        } else {
            break;
        }
    }

    let day = (remaining_days + 1) as u32;

    Ok((year, month, day))
}

/// Check if a year is a leap year
pub fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Get number of days in a month
pub fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Helper function to convert days since Unix epoch to (year, month, day)
pub fn days_to_date(days: i64) -> (i32, u32, u32) {
    // Simple algorithm for converting days since epoch to date
    let mut year = 1970i32;
    let mut remaining_days = days;

    // Handle years
    while remaining_days >= 365 {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining_days >= days_in_year {
            remaining_days -= days_in_year;
            year += 1;
        } else {
            break;
        }
    }

    // Handle months
    let days_in_months = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1u32;
    for &days_in_month in &days_in_months {
        if remaining_days >= days_in_month as i64 {
            remaining_days -= days_in_month as i64;
            month += 1;
        } else {
            break;
        }
    }

    let day = (remaining_days + 1) as u32;
    (year, month, day)
}

// ============================================================================
// Operator implementations
// ============================================================================

impl std::ops::Add<TimeDelta64> for DateTime64 {
    type Output = DateTime64;

    fn add(self, rhs: TimeDelta64) -> Self::Output {
        // Convert the timedelta to the same unit as the datetime
        let td = rhs.to_unit(self.unit);

        // Add the values
        DateTime64 {
            value: self.value + td.value(),
            unit: self.unit,
        }
    }
}

impl std::ops::Sub<TimeDelta64> for DateTime64 {
    type Output = DateTime64;

    fn sub(self, rhs: TimeDelta64) -> Self::Output {
        // Convert the timedelta to the same unit as the datetime
        let td = rhs.to_unit(self.unit);

        // Subtract the values
        DateTime64 {
            value: self.value - td.value(),
            unit: self.unit,
        }
    }
}

impl std::ops::Sub<DateTime64> for DateTime64 {
    type Output = TimeDelta64;

    fn sub(self, rhs: DateTime64) -> Self::Output {
        // Convert the other datetime to the same unit as this one
        let dt = rhs.to_unit(self.unit);

        // Subtract the values
        TimeDelta64::new(self.value - dt.value, self.unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_datetime64_creation() {
        let dt = DateTime64::new(100, DateTimeUnit::Second);
        assert_eq!(dt.value(), 100);
        assert_eq!(dt.unit(), DateTimeUnit::Second);
    }

    #[test]
    fn test_datetime64_conversion() {
        let dt = DateTime64::new(60, DateTimeUnit::Second);

        // Convert to minutes
        let dt_min = dt.to_unit(DateTimeUnit::Minute);
        assert_eq!(dt_min.value(), 1);
        assert_eq!(dt_min.unit(), DateTimeUnit::Minute);

        // Convert to milliseconds
        let dt_ms = dt.to_unit(DateTimeUnit::Millisecond);
        assert_eq!(dt_ms.value(), 60_000);
        assert_eq!(dt_ms.unit(), DateTimeUnit::Millisecond);
    }

    #[test]
    fn test_datetime64_system_time() {
        let now = SystemTime::now();
        let dt = DateTime64::from_system_time(now, DateTimeUnit::Second)
            .expect("should convert SystemTime to DateTime64");
        let time = dt.to_system_time();

        // Allow for some rounding errors in the conversion
        let diff = now
            .duration_since(time)
            .unwrap_or(Duration::from_secs(0))
            .max(time.duration_since(now).unwrap_or(Duration::from_secs(0)));
        assert!(
            diff.as_secs() < 1,
            "Difference should be less than 1 second"
        );
    }

    #[test]
    fn test_datetime_parsing() {
        // Test ISO string parsing
        let dt = DateTime64::from_iso_string("2023-12-25T15:30:45", DateTimeUnit::Second)
            .expect("should parse ISO datetime string");
        let iso_str = dt.to_iso_string().expect("should convert to ISO string");
        assert!(iso_str.starts_with("2023-12-25T15:30:45"));

        // Test date-only parsing
        let dt2 = DateTime64::from_iso_string("2023-01-01", DateTimeUnit::Day)
            .expect("should parse ISO date string");
        let iso_str2 = dt2.to_iso_string().expect("should convert to ISO string");
        assert!(iso_str2.starts_with("2023-01-01T00:00:00"));
    }

    #[test]
    fn test_leap_year_calculations() {
        // Test leap year function
        assert!(is_leap_year(2000)); // Divisible by 400
        assert!(is_leap_year(2004)); // Divisible by 4, not by 100
        assert!(!is_leap_year(1900)); // Divisible by 100, not by 400
        assert!(!is_leap_year(2001)); // Not divisible by 4

        // Test February days
        assert_eq!(days_in_month(2000, 2), 29); // Leap year
        assert_eq!(days_in_month(2001, 2), 28); // Non-leap year
        assert_eq!(days_in_month(2023, 1), 31); // January
        assert_eq!(days_in_month(2023, 4), 30); // April
    }

    #[test]
    fn test_datetime_timedelta_operations() {
        let dt1 = DateTime64::new(100, DateTimeUnit::Second);
        let td = TimeDelta64::new(50, DateTimeUnit::Second);

        // Add timedelta to datetime
        let dt2 = dt1 + td;
        assert_eq!(dt2.value(), 150);
        assert_eq!(dt2.unit(), DateTimeUnit::Second);

        // Subtract timedelta from datetime
        let dt3 = dt1 - td;
        assert_eq!(dt3.value(), 50);
        assert_eq!(dt3.unit(), DateTimeUnit::Second);

        // Subtract datetime from datetime
        let td2 = dt2 - dt1;
        assert_eq!(td2.value(), 50);
        assert_eq!(td2.unit(), DateTimeUnit::Second);
    }

    #[test]
    fn test_different_units() {
        let dt = DateTime64::new(1, DateTimeUnit::Minute);
        let td = TimeDelta64::new(30, DateTimeUnit::Second);

        // Add timedelta with different unit to datetime
        let dt2 = dt + td;
        assert_eq!(dt2.value(), 1);
        assert_eq!(dt2.unit(), DateTimeUnit::Minute);

        // Convert to seconds to see the actual value
        let dt2_sec = dt2.to_unit(DateTimeUnit::Second);
        assert_eq!(dt2_sec.value(), 60); // 1 minute = 60 seconds
    }
}
