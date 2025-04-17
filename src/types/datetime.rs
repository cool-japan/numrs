//! Date and time functionality for NumRS
//!
//! This module provides data types for working with dates and times,
//! similar to NumPy's datetime64 and timedelta64.

use std::fmt;
use std::ops::{Add, Sub};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use crate::error::{NumRs2Error, Result};

/// Represents the unit of time for date and datetime operations
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
        let duration = time.duration_since(UNIX_EPOCH)
            .map_err(|e| NumRs2Error::ValueError(format!("Error converting SystemTime: {}", e)))?;
        
        let value = match unit {
            DateTimeUnit::Year => {
                // Approximate years since epoch
                (duration.as_secs() / (365 * 24 * 60 * 60)) as i64
            },
            DateTimeUnit::Month => {
                // Approximate months since epoch
                (duration.as_secs() / (30 * 24 * 60 * 60)) as i64
            },
            DateTimeUnit::Week => {
                // Weeks since epoch
                (duration.as_secs() / (7 * 24 * 60 * 60)) as i64
            },
            DateTimeUnit::Day => {
                // Days since epoch
                (duration.as_secs() / (24 * 60 * 60)) as i64
            },
            DateTimeUnit::Hour => {
                // Hours since epoch
                (duration.as_secs() / (60 * 60)) as i64
            },
            DateTimeUnit::Minute => {
                // Minutes since epoch
                (duration.as_secs() / 60) as i64
            },
            DateTimeUnit::Second => {
                // Seconds since epoch
                duration.as_secs() as i64
            },
            DateTimeUnit::Millisecond => {
                // Milliseconds since epoch
                (duration.as_secs() * 1000 + (duration.subsec_nanos() / 1_000_000) as u64) as i64
            },
            DateTimeUnit::Microsecond => {
                // Microseconds since epoch
                (duration.as_secs() * 1_000_000 + (duration.subsec_nanos() / 1_000) as u64) as i64
            },
            DateTimeUnit::Nanosecond => {
                // Nanoseconds since epoch
                (duration.as_secs() * 1_000_000_000 + duration.subsec_nanos() as u64) as i64
            },
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
        
        Self { value: target_value, unit }
    }
    
    /// Convert to a SystemTime
    pub fn to_system_time(&self) -> SystemTime {
        // Convert to seconds and nanoseconds
        let (secs, nanos) = match self.unit {
            DateTimeUnit::Year => {
                let secs = self.value * 365 * 24 * 60 * 60;
                (secs, 0)
            },
            DateTimeUnit::Month => {
                let secs = self.value * 30 * 24 * 60 * 60;
                (secs, 0)
            },
            DateTimeUnit::Week => {
                let secs = self.value * 7 * 24 * 60 * 60;
                (secs, 0)
            },
            DateTimeUnit::Day => {
                let secs = self.value * 24 * 60 * 60;
                (secs, 0)
            },
            DateTimeUnit::Hour => {
                let secs = self.value * 60 * 60;
                (secs, 0)
            },
            DateTimeUnit::Minute => {
                let secs = self.value * 60;
                (secs, 0)
            },
            DateTimeUnit::Second => {
                (self.value, 0)
            },
            DateTimeUnit::Millisecond => {
                let secs = self.value / 1000;
                let nanos = (self.value % 1000) * 1_000_000;
                (secs, nanos as u32)
            },
            DateTimeUnit::Microsecond => {
                let secs = self.value / 1_000_000;
                let nanos = (self.value % 1_000_000) * 1_000;
                (secs, nanos as u32)
            },
            DateTimeUnit::Nanosecond => {
                let secs = self.value / 1_000_000_000;
                let nanos = (self.value % 1_000_000_000) as u32;
                (secs, nanos)
            },
        };
        
        UNIX_EPOCH + Duration::new(secs as u64, nanos)
    }
}

impl fmt::Display for DateTime64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.value, self.unit)
    }
}

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
            },
            DateTimeUnit::Month => {
                // Approximate months
                (duration.as_secs() / (30 * 24 * 60 * 60)) as i64
            },
            DateTimeUnit::Week => {
                // Weeks
                (duration.as_secs() / (7 * 24 * 60 * 60)) as i64
            },
            DateTimeUnit::Day => {
                // Days
                (duration.as_secs() / (24 * 60 * 60)) as i64
            },
            DateTimeUnit::Hour => {
                // Hours
                (duration.as_secs() / (60 * 60)) as i64
            },
            DateTimeUnit::Minute => {
                // Minutes
                (duration.as_secs() / 60) as i64
            },
            DateTimeUnit::Second => {
                // Seconds
                duration.as_secs() as i64
            },
            DateTimeUnit::Millisecond => {
                // Milliseconds
                (duration.as_secs() * 1000 + (duration.subsec_nanos() / 1_000_000) as u64) as i64
            },
            DateTimeUnit::Microsecond => {
                // Microseconds
                (duration.as_secs() * 1_000_000 + (duration.subsec_nanos() / 1_000) as u64) as i64
            },
            DateTimeUnit::Nanosecond => {
                // Nanoseconds
                (duration.as_secs() * 1_000_000_000 + duration.subsec_nanos() as u64) as i64
            },
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
        
        Self { value: target_value, unit }
    }
    
    /// Convert to a Duration
    pub fn to_duration(&self) -> Duration {
        match self.unit {
            DateTimeUnit::Year => {
                Duration::from_secs((self.value * 365 * 24 * 60 * 60) as u64)
            },
            DateTimeUnit::Month => {
                Duration::from_secs((self.value * 30 * 24 * 60 * 60) as u64)
            },
            DateTimeUnit::Week => {
                Duration::from_secs((self.value * 7 * 24 * 60 * 60) as u64)
            },
            DateTimeUnit::Day => {
                Duration::from_secs((self.value * 24 * 60 * 60) as u64)
            },
            DateTimeUnit::Hour => {
                Duration::from_secs((self.value * 60 * 60) as u64)
            },
            DateTimeUnit::Minute => {
                Duration::from_secs((self.value * 60) as u64)
            },
            DateTimeUnit::Second => {
                Duration::from_secs(self.value as u64)
            },
            DateTimeUnit::Millisecond => {
                let secs = self.value / 1000;
                let nanos = (self.value % 1000) * 1_000_000;
                Duration::new(secs as u64, nanos as u32)
            },
            DateTimeUnit::Microsecond => {
                let secs = self.value / 1_000_000;
                let nanos = (self.value % 1_000_000) * 1_000;
                Duration::new(secs as u64, nanos as u32)
            },
            DateTimeUnit::Nanosecond => {
                let secs = self.value / 1_000_000_000;
                let nanos = (self.value % 1_000_000_000) as u32;
                Duration::new(secs as u64, nanos)
            },
        }
    }
}

impl fmt::Display for TimeDelta64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.value, self.unit)
    }
}

// Implementation of operations

impl Add<TimeDelta64> for DateTime64 {
    type Output = DateTime64;
    
    fn add(self, rhs: TimeDelta64) -> Self::Output {
        // Convert the timedelta to the same unit as the datetime
        let td = rhs.to_unit(self.unit);
        
        // Add the values
        DateTime64 {
            value: self.value + td.value,
            unit: self.unit,
        }
    }
}

impl Sub<TimeDelta64> for DateTime64 {
    type Output = DateTime64;
    
    fn sub(self, rhs: TimeDelta64) -> Self::Output {
        // Convert the timedelta to the same unit as the datetime
        let td = rhs.to_unit(self.unit);
        
        // Subtract the values
        DateTime64 {
            value: self.value - td.value,
            unit: self.unit,
        }
    }
}

impl Sub<DateTime64> for DateTime64 {
    type Output = TimeDelta64;
    
    fn sub(self, rhs: DateTime64) -> Self::Output {
        // Convert the other datetime to the same unit as this one
        let dt = rhs.to_unit(self.unit);
        
        // Subtract the values
        TimeDelta64 {
            value: self.value - dt.value,
            unit: self.unit,
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::UNIX_EPOCH;
    
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
        let dt = DateTime64::from_system_time(now, DateTimeUnit::Second).unwrap();
        let time = dt.to_system_time();
        
        // Allow for some rounding errors in the conversion
        let diff = now.duration_since(time).unwrap_or(Duration::from_secs(0))
            .max(time.duration_since(now).unwrap_or(Duration::from_secs(0)));
        assert!(diff.as_secs() < 1, "Difference should be less than 1 second");
    }
    
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
