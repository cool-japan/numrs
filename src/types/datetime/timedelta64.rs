//! TimeDelta64 type for NumRS
//!
//! This module provides the TimeDelta64 type, similar to NumPy's timedelta64,
//! storing a time duration as a 64-bit integer representing the number of units.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::time::Duration;

use super::units::DateTimeUnit;

/// Represents a time difference with a specified unit
///
/// This type is similar to NumPy's timedelta64 type, storing a time duration
/// as a 64-bit integer representing the number of units.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimeDelta64 {
    /// Number of time units
    value: i64,
    /// The unit of time
    unit: DateTimeUnit,
}

impl TimeDelta64 {
    /// Create a new TimeDelta64 with the specified value and unit
    pub fn new(value: i64, unit: DateTimeUnit) -> Self {
        Self { value, unit }
    }

    /// Create a TimeDelta64 from a Duration
    pub fn from_duration(duration: Duration, unit: DateTimeUnit) -> Self {
        let value = match unit {
            DateTimeUnit::Year => {
                // Approximate years
                (duration.as_secs() / (365 * 24 * 60 * 60)) as i64
            }
            DateTimeUnit::Month => {
                // Approximate months
                (duration.as_secs() / (30 * 24 * 60 * 60)) as i64
            }
            DateTimeUnit::Week => {
                // Weeks
                (duration.as_secs() / (7 * 24 * 60 * 60)) as i64
            }
            DateTimeUnit::Day => {
                // Days
                (duration.as_secs() / (24 * 60 * 60)) as i64
            }
            DateTimeUnit::Hour => {
                // Hours
                (duration.as_secs() / (60 * 60)) as i64
            }
            DateTimeUnit::Minute => {
                // Minutes
                (duration.as_secs() / 60) as i64
            }
            DateTimeUnit::Second => {
                // Seconds
                duration.as_secs() as i64
            }
            DateTimeUnit::Millisecond => {
                // Milliseconds
                (duration.as_secs() * 1000 + duration.subsec_millis() as u64) as i64
            }
            DateTimeUnit::Microsecond => {
                // Microseconds
                (duration.as_secs() * 1_000_000 + duration.subsec_micros() as u64) as i64
            }
            DateTimeUnit::Nanosecond => {
                // Nanoseconds
                (duration.as_secs() * 1_000_000_000 + duration.subsec_nanos() as u64) as i64
            }
        };

        Self { value, unit }
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

    /// Convert to a Duration
    pub fn to_duration(&self) -> Duration {
        match self.unit {
            DateTimeUnit::Year => Duration::from_secs((self.value * 365 * 24 * 60 * 60) as u64),
            DateTimeUnit::Month => Duration::from_secs((self.value * 30 * 24 * 60 * 60) as u64),
            DateTimeUnit::Week => Duration::from_secs((self.value * 7 * 24 * 60 * 60) as u64),
            DateTimeUnit::Day => Duration::from_secs((self.value * 24 * 60 * 60) as u64),
            DateTimeUnit::Hour => Duration::from_secs((self.value * 60 * 60) as u64),
            DateTimeUnit::Minute => Duration::from_secs((self.value * 60) as u64),
            DateTimeUnit::Second => Duration::from_secs(self.value as u64),
            DateTimeUnit::Millisecond => {
                let secs = self.value / 1000;
                let nanos = (self.value % 1000) * 1_000_000;
                Duration::new(secs as u64, nanos as u32)
            }
            DateTimeUnit::Microsecond => {
                let secs = self.value / 1_000_000;
                let nanos = (self.value % 1_000_000) * 1_000;
                Duration::new(secs as u64, nanos as u32)
            }
            DateTimeUnit::Nanosecond => {
                let secs = self.value / 1_000_000_000;
                let nanos = (self.value % 1_000_000_000) as u32;
                Duration::new(secs as u64, nanos)
            }
        }
    }

    /// Get absolute value of the timedelta
    pub fn abs(&self) -> Self {
        Self {
            value: self.value.abs(),
            unit: self.unit,
        }
    }

    /// Get negative of the timedelta
    pub fn neg(&self) -> Self {
        Self {
            value: -self.value,
            unit: self.unit,
        }
    }
}

impl fmt::Display for TimeDelta64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.value, self.unit)
    }
}

// ============================================================================
// Operator implementations for TimeDelta64
// ============================================================================

impl Add for TimeDelta64 {
    type Output = TimeDelta64;

    fn add(self, rhs: TimeDelta64) -> Self::Output {
        // Convert the other timedelta to the same unit as this one
        let td = rhs.to_unit(self.unit);

        // Add the values
        TimeDelta64 {
            value: self.value + td.value,
            unit: self.unit,
        }
    }
}

impl Sub for TimeDelta64 {
    type Output = TimeDelta64;

    fn sub(self, rhs: TimeDelta64) -> Self::Output {
        // Convert the other timedelta to the same unit as this one
        let td = rhs.to_unit(self.unit);

        // Subtract the values
        TimeDelta64 {
            value: self.value - td.value,
            unit: self.unit,
        }
    }
}

impl Mul<i64> for TimeDelta64 {
    type Output = TimeDelta64;

    fn mul(self, rhs: i64) -> Self::Output {
        TimeDelta64 {
            value: self.value * rhs,
            unit: self.unit,
        }
    }
}

impl Mul<TimeDelta64> for i64 {
    type Output = TimeDelta64;

    fn mul(self, rhs: TimeDelta64) -> Self::Output {
        TimeDelta64 {
            value: self * rhs.value,
            unit: rhs.unit,
        }
    }
}

impl Div<i64> for TimeDelta64 {
    type Output = TimeDelta64;

    fn div(self, rhs: i64) -> Self::Output {
        TimeDelta64 {
            value: self.value / rhs,
            unit: self.unit,
        }
    }
}

impl Div for TimeDelta64 {
    type Output = f64;

    fn div(self, rhs: TimeDelta64) -> Self::Output {
        // Convert to same unit and divide
        let rhs_converted = rhs.to_unit(self.unit);
        self.value as f64 / rhs_converted.value as f64
    }
}

impl Neg for TimeDelta64 {
    type Output = TimeDelta64;

    fn neg(self) -> Self::Output {
        TimeDelta64 {
            value: -self.value,
            unit: self.unit,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timedelta64_creation() {
        let td = TimeDelta64::new(100, DateTimeUnit::Second);
        assert_eq!(td.value(), 100);
        assert_eq!(td.unit(), DateTimeUnit::Second);
    }

    #[test]
    fn test_timedelta64_conversion() {
        let td = TimeDelta64::new(60, DateTimeUnit::Second);

        // Convert to minutes
        let td_min = td.to_unit(DateTimeUnit::Minute);
        assert_eq!(td_min.value(), 1);
        assert_eq!(td_min.unit(), DateTimeUnit::Minute);

        // Convert to milliseconds
        let td_ms = td.to_unit(DateTimeUnit::Millisecond);
        assert_eq!(td_ms.value(), 60_000);
        assert_eq!(td_ms.unit(), DateTimeUnit::Millisecond);
    }

    #[test]
    fn test_timedelta_operations() {
        let td1 = TimeDelta64::new(100, DateTimeUnit::Second);
        let td2 = TimeDelta64::new(50, DateTimeUnit::Second);

        // Add timedeltas
        let td3 = td1 + td2;
        assert_eq!(td3.value(), 150);
        assert_eq!(td3.unit(), DateTimeUnit::Second);

        // Subtract timedeltas
        let td4 = td1 - td2;
        assert_eq!(td4.value(), 50);
        assert_eq!(td4.unit(), DateTimeUnit::Second);
    }

    #[test]
    fn test_timedelta_arithmetic() {
        let td1 = TimeDelta64::new(100, DateTimeUnit::Second);
        let td2 = TimeDelta64::new(50, DateTimeUnit::Second);

        // Test multiplication
        let td3 = td1 * 2;
        assert_eq!(td3.value(), 200);

        let td4 = 3 * td2;
        assert_eq!(td4.value(), 150);

        // Test division
        let td5 = td1 / 2;
        assert_eq!(td5.value(), 50);

        let ratio = td1 / td2;
        assert_eq!(ratio, 2.0);

        // Test negation
        let td6 = -td1;
        assert_eq!(td6.value(), -100);
    }

    #[test]
    fn test_timedelta_abs_neg() {
        let td = TimeDelta64::new(-50, DateTimeUnit::Second);

        // Test abs
        let td_abs = td.abs();
        assert_eq!(td_abs.value(), 50);

        // Test neg method
        let td_neg = td.neg();
        assert_eq!(td_neg.value(), 50);
    }

    #[test]
    fn test_timedelta_from_duration() {
        let duration = Duration::from_secs(3600); // 1 hour
        let td = TimeDelta64::from_duration(duration, DateTimeUnit::Hour);
        assert_eq!(td.value(), 1);

        let td_min = TimeDelta64::from_duration(duration, DateTimeUnit::Minute);
        assert_eq!(td_min.value(), 60);
    }

    #[test]
    fn test_timedelta_to_duration() {
        let td = TimeDelta64::new(60, DateTimeUnit::Second);
        let duration = td.to_duration();
        assert_eq!(duration.as_secs(), 60);

        let td_ms = TimeDelta64::new(1500, DateTimeUnit::Millisecond);
        let duration_ms = td_ms.to_duration();
        assert_eq!(duration_ms.as_secs(), 1);
        assert_eq!(duration_ms.subsec_millis(), 500);
    }
}
