//! Date and time functionality for NumRS
//!
//! This module provides data types for working with dates and times,
//! similar to NumPy's datetime64 and timedelta64.

use crate::array::Array;
use crate::error::{NumRs2Error, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Sub, Mul, Div};
use std::str::FromStr;
use std::time::{Duration, SystemTime};

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
            return Err(NumRs2Error::ValueError("Invalid date format, expected YYYY-MM-DD".to_string()));
        }
        
        let year: i32 = date_components[0].parse()
            .map_err(|_| NumRs2Error::ValueError("Invalid year".to_string()))?;
        let month: u32 = date_components[1].parse()
            .map_err(|_| NumRs2Error::ValueError("Invalid month".to_string()))?;
        let day: u32 = date_components[2].parse()
            .map_err(|_| NumRs2Error::ValueError("Invalid day".to_string()))?;
        
        // Parse time (HH:MM:SS)
        let time_components: Vec<&str> = time_part.split(':').collect();
        let hour: u32 = if time_components.len() >= 1 {
            time_components[0].parse().unwrap_or(0)
        } else { 0 };
        let minute: u32 = if time_components.len() >= 2 {
            time_components[1].parse().unwrap_or(0)
        } else { 0 };
        let second: u32 = if time_components.len() >= 3 {
            time_components[2].parse().unwrap_or(0)
        } else { 0 };
        
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
        
        Ok(format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", 
                   year, month, day, hour, minute, second))
    }
}

/// Calculate days since Unix epoch (1970-01-01)
fn days_since_epoch(year: i32, month: u32, day: u32) -> Result<i64> {
    if month < 1 || month > 12 {
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
fn date_from_days_since_epoch(days: i64) -> Result<(i32, u32, u32)> {
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
fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

/// Get number of days in a month
fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if is_leap_year(year) { 29 } else { 28 },
        _ => 0,
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

impl std::ops::Neg for TimeDelta64 {
    type Output = TimeDelta64;
    
    fn neg(self) -> Self::Output {
        TimeDelta64 {
            value: -self.value,
            unit: self.unit,
        }
    }
}

impl FromStr for DateTime64 {
    type Err = NumRs2Error;
    
    fn from_str(s: &str) -> Result<Self> {
        // Default to second precision for parsing
        Self::from_iso_string(s, DateTimeUnit::Second)
    }
}

/// Array creation functions for datetime arrays
pub mod datetime_array {
    use super::*;
    
    /// Create an array of datetime values from start to stop with given frequency
    /// 
    /// Similar to NumPy's `pd.date_range()` or `np.arange()` for datetimes
    pub fn date_range(
        start: &str, 
        end: Option<&str>, 
        periods: Option<usize>,
        freq: DateTimeUnit,
        unit: DateTimeUnit
    ) -> Result<Array<DateTime64>> {
        let start_dt = DateTime64::from_iso_string(start, unit)?;
        
        match (end, periods) {
            (Some(end_str), None) => {
                // Generate from start to end
                let end_dt = DateTime64::from_iso_string(end_str, unit)?;
                let _duration = end_dt - start_dt;
                
                // Calculate step size - should be 1 unit of the frequency
                let step = TimeDelta64::new(1, freq);
                let mut result = Vec::new();
                let mut current = start_dt;
                
                while current.value <= end_dt.value {
                    result.push(current);
                    current = current + step;
                }
                
                Ok(Array::from_vec(result))
            }
            (None, Some(num_periods)) => {
                // Generate fixed number of periods
                let step = TimeDelta64::new(1, freq);
                let mut result = Vec::with_capacity(num_periods);
                let mut current = start_dt;
                
                for _ in 0..num_periods {
                    result.push(current);
                    current = current + step;
                }
                
                Ok(Array::from_vec(result))
            }
            (Some(_), Some(_)) => {
                Err(NumRs2Error::ValueError("Cannot specify both end and periods".to_string()))
            }
            (None, None) => {
                Err(NumRs2Error::ValueError("Must specify either end or periods".to_string()))
            }
        }
    }
    
    /// Create an array of timedelta values
    pub fn timedelta_range(
        start: i64,
        end: Option<i64>,
        periods: Option<usize>,
        unit: DateTimeUnit
    ) -> Result<Array<TimeDelta64>> {
        match (end, periods) {
            (Some(end_val), None) => {
                let mut result = Vec::new();
                for val in start..=end_val {
                    result.push(TimeDelta64::new(val, unit));
                }
                Ok(Array::from_vec(result))
            }
            (None, Some(num_periods)) => {
                let mut result = Vec::with_capacity(num_periods);
                for i in 0..num_periods {
                    result.push(TimeDelta64::new(start + i as i64, unit));
                }
                Ok(Array::from_vec(result))
            }
            (Some(_), Some(_)) => {
                Err(NumRs2Error::ValueError("Cannot specify both end and periods".to_string()))
            }
            (None, None) => {
                Err(NumRs2Error::ValueError("Must specify either end or periods".to_string()))
            }
        }
    }
    
    /// Create an array of datetime values from string representations
    pub fn datetime_from_strings(strings: &[&str], unit: DateTimeUnit) -> Result<Array<DateTime64>> {
        let mut result = Vec::with_capacity(strings.len());
        
        for s in strings {
            let dt = DateTime64::from_iso_string(s, unit)?;
            result.push(dt);
        }
        
        Ok(Array::from_vec(result))
    }
    
    /// Create datetime array for today with different time units
    pub fn today(unit: DateTimeUnit) -> Result<DateTime64> {
        let now = SystemTime::now();
        DateTime64::from_system_time(now, unit)
    }
    
    /// Create datetime array for now with different time units  
    pub fn now(unit: DateTimeUnit) -> Result<DateTime64> {
        today(unit)
    }
}

/// Timezone-aware datetime support
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Timezone {
    /// Timezone name (e.g., "UTC", "EST", "PST")
    pub name: String,
    /// Offset from UTC in minutes
    pub offset_minutes: i32,
}

impl Timezone {
    /// Create UTC timezone
    pub fn utc() -> Self {
        Self {
            name: "UTC".to_string(),
            offset_minutes: 0,
        }
    }
    
    /// Create timezone with fixed offset from UTC
    pub fn fixed_offset(name: &str, hours: i32, minutes: i32) -> Self {
        Self {
            name: name.to_string(),
            offset_minutes: hours * 60 + minutes,
        }
    }
    
    /// Common timezone presets
    pub fn est() -> Self {
        Self::fixed_offset("EST", -5, 0)
    }
    
    pub fn pst() -> Self {
        Self::fixed_offset("PST", -8, 0)
    }
    
    pub fn cet() -> Self {
        Self::fixed_offset("CET", 1, 0)
    }
    
    pub fn jst() -> Self {
        Self::fixed_offset("JST", 9, 0)
    }
}

impl fmt::Display for Timezone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.offset_minutes >= 0 { "+" } else { "-" };
        let abs_minutes = self.offset_minutes.abs();
        let hours = abs_minutes / 60;
        let minutes = abs_minutes % 60;
        write!(f, "{} ({}{:02}:{:02})", self.name, sign, hours, minutes)
    }
}

/// Timezone-aware datetime
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimezoneDateTime {
    /// The UTC datetime value
    pub utc_datetime: DateTime64,
    /// The timezone information
    pub timezone: Timezone,
}

impl TimezoneDateTime {
    /// Create a new timezone-aware datetime
    pub fn new(utc_datetime: DateTime64, timezone: Timezone) -> Self {
        Self {
            utc_datetime,
            timezone,
        }
    }
    
    /// Create from local time and timezone
    pub fn from_local(local_datetime: DateTime64, timezone: Timezone) -> Self {
        // Convert local time to UTC
        let offset_delta = TimeDelta64::new(
            timezone.offset_minutes as i64,
            DateTimeUnit::Minute
        );
        let utc_datetime = local_datetime - offset_delta;
        
        Self {
            utc_datetime,
            timezone,
        }
    }
    
    /// Parse timezone-aware datetime from ISO 8601 string with timezone
    /// 
    /// Supports formats like "2023-12-25T15:30:45+05:00" or "2023-12-25T15:30:45Z"
    pub fn from_iso_string_with_tz(s: &str, unit: DateTimeUnit) -> Result<Self> {
        // Handle Z suffix (UTC)
        if s.ends_with('Z') {
            let datetime_part = &s[..s.len()-1];
            let utc_dt = DateTime64::from_iso_string(datetime_part, unit)?;
            return Ok(Self::new(utc_dt, Timezone::utc()));
        }
        
        // Look for timezone offset (+/-HH:MM)
        let tz_patterns = ["+", "-"];
        for &pattern in &tz_patterns {
            if let Some(tz_pos) = s.rfind(pattern) {
                let datetime_part = &s[..tz_pos];
                let tz_part = &s[tz_pos..];
                
                // Parse timezone offset
                let tz_sign = if pattern == "+" { 1 } else { -1 };
                let tz_components: Vec<&str> = tz_part[1..].split(':').collect();
                
                if tz_components.len() >= 2 {
                    let tz_hours: i32 = tz_components[0].parse()
                        .map_err(|_| NumRs2Error::ValueError("Invalid timezone hours".to_string()))?;
                    let tz_minutes: i32 = tz_components[1].parse()
                        .map_err(|_| NumRs2Error::ValueError("Invalid timezone minutes".to_string()))?;
                    
                    let offset_minutes = tz_sign * (tz_hours * 60 + tz_minutes);
                    let timezone = Timezone {
                        name: tz_part.to_string(),
                        offset_minutes,
                    };
                    
                    // Parse datetime as local time and convert
                    let local_dt = DateTime64::from_iso_string(datetime_part, unit)?;
                    return Ok(Self::from_local(local_dt, timezone));
                }
            }
        }
        
        // No timezone found, assume UTC
        let utc_dt = DateTime64::from_iso_string(s, unit)?;
        Ok(Self::new(utc_dt, Timezone::utc()))
    }
    
    /// Get the local datetime in the specified timezone
    pub fn to_local(&self) -> DateTime64 {
        let offset_delta = TimeDelta64::new(
            self.timezone.offset_minutes as i64,
            DateTimeUnit::Minute
        );
        self.utc_datetime + offset_delta
    }
    
    /// Convert to a different timezone
    pub fn to_timezone(&self, new_timezone: Timezone) -> Self {
        Self {
            utc_datetime: self.utc_datetime,
            timezone: new_timezone,
        }
    }
    
    /// Format as ISO 8601 string with timezone
    pub fn to_iso_string_with_tz(&self) -> Result<String> {
        let local_dt = self.to_local();
        let base_str = local_dt.to_iso_string()?;
        
        if self.timezone.name == "UTC" {
            Ok(format!("{}Z", base_str))
        } else {
            let sign = if self.timezone.offset_minutes >= 0 { "+" } else { "-" };
            let abs_minutes = self.timezone.offset_minutes.abs();
            let hours = abs_minutes / 60;
            let minutes = abs_minutes % 60;
            Ok(format!("{}{}{:02}:{:02}", base_str, sign, hours, minutes))
        }
    }
}

impl fmt::Display for TimezoneDateTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.to_iso_string_with_tz() {
            Ok(s) => write!(f, "{}", s),
            Err(_) => write!(f, "<invalid datetime>"),
        }
    }
}

impl Add<TimeDelta64> for TimezoneDateTime {
    type Output = TimezoneDateTime;
    
    fn add(self, rhs: TimeDelta64) -> Self::Output {
        TimezoneDateTime {
            utc_datetime: self.utc_datetime + rhs,
            timezone: self.timezone,
        }
    }
}

impl Sub<TimeDelta64> for TimezoneDateTime {
    type Output = TimezoneDateTime;
    
    fn sub(self, rhs: TimeDelta64) -> Self::Output {
        TimezoneDateTime {
            utc_datetime: self.utc_datetime - rhs,
            timezone: self.timezone,
        }
    }
}

impl Sub for TimezoneDateTime {
    type Output = TimeDelta64;
    
    fn sub(self, rhs: TimezoneDateTime) -> Self::Output {
        self.utc_datetime - rhs.utc_datetime
    }
}

/// Business day and calendar functions
pub mod business_days {
    use super::*;
    
    /// Day of the week (0 = Monday, 6 = Sunday)
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Weekday {
        Monday = 0,
        Tuesday = 1,
        Wednesday = 2,
        Thursday = 3,
        Friday = 4,
        Saturday = 5,
        Sunday = 6,
    }
    
    impl Weekday {
        /// Check if this is a business day (Monday-Friday)
        pub fn is_business_day(&self) -> bool {
            matches!(self, Weekday::Monday | Weekday::Tuesday | Weekday::Wednesday | Weekday::Thursday | Weekday::Friday)
        }
    }
    
    /// Get the weekday for a given date
    pub fn weekday(dt: &DateTime64) -> Result<Weekday> {
        let dt_days = dt.to_unit(DateTimeUnit::Day);
        // Unix epoch (1970-01-01) was a Thursday (index 3)
        let days_since_epoch = dt_days.value;
        let weekday_index = ((days_since_epoch + 3) % 7 + 7) % 7; // Handle negative numbers
        
        match weekday_index {
            0 => Ok(Weekday::Monday),
            1 => Ok(Weekday::Tuesday),
            2 => Ok(Weekday::Wednesday),
            3 => Ok(Weekday::Thursday),
            4 => Ok(Weekday::Friday),
            5 => Ok(Weekday::Saturday),
            6 => Ok(Weekday::Sunday),
            _ => Err(NumRs2Error::ValueError("Invalid weekday calculation".to_string())),
        }
    }
    
    /// Check if a date is a business day
    pub fn is_busday(dt: &DateTime64) -> Result<bool> {
        let wd = weekday(dt)?;
        Ok(wd.is_business_day())
    }
    
    /// Count business days between two dates
    pub fn busday_count(start: &DateTime64, end: &DateTime64) -> Result<i64> {
        let start_days = start.to_unit(DateTimeUnit::Day);
        let end_days = end.to_unit(DateTimeUnit::Day);
        
        if start_days.value > end_days.value {
            return Ok(-busday_count(end, start)?);
        }
        
        let mut count = 0i64;
        let mut current = start_days;
        
        while current.value < end_days.value {
            if is_busday(&current)? {
                count += 1;
            }
            current = current + TimeDelta64::new(1, DateTimeUnit::Day);
        }
        
        Ok(count)
    }
    
    /// Offset a date by a number of business days
    pub fn busday_offset(dt: &DateTime64, offset: i64) -> Result<DateTime64> {
        let mut current = dt.to_unit(DateTimeUnit::Day);
        let mut remaining = offset.abs();
        let direction = if offset >= 0 { 1 } else { -1 };
        
        while remaining > 0 {
            current = current + TimeDelta64::new(direction, DateTimeUnit::Day);
            if is_busday(&current)? {
                remaining -= 1;
            }
        }
        
        Ok(current.to_unit(dt.unit))
    }
    
    /// Holiday calendar - simple implementation with common holidays
    #[derive(Debug, Clone)]
    pub struct HolidayCalendar {
        holidays: Vec<DateTime64>,
    }
    
    impl HolidayCalendar {
        /// Create a new holiday calendar
        pub fn new() -> Self {
            Self {
                holidays: Vec::new(),
            }
        }
        
        /// Add a holiday to the calendar
        pub fn add_holiday(&mut self, date: DateTime64) {
            self.holidays.push(date.to_unit(DateTimeUnit::Day));
        }
        
        /// Check if a date is a holiday
        pub fn is_holiday(&self, dt: &DateTime64) -> bool {
            let dt_days = dt.to_unit(DateTimeUnit::Day);
            self.holidays.iter().any(|h| h.value == dt_days.value)
        }
        
        /// Check if a date is a business day (not weekend and not holiday)
        pub fn is_business_day(&self, dt: &DateTime64) -> Result<bool> {
            Ok(is_busday(dt)? && !self.is_holiday(dt))
        }
        
        /// Count business days between two dates considering holidays
        pub fn business_day_count(&self, start: &DateTime64, end: &DateTime64) -> Result<i64> {
            let start_days = start.to_unit(DateTimeUnit::Day);
            let end_days = end.to_unit(DateTimeUnit::Day);
            
            if start_days.value > end_days.value {
                return Ok(-self.business_day_count(end, start)?);
            }
            
            let mut count = 0i64;
            let mut current = start_days;
            
            while current.value < end_days.value {
                if self.is_business_day(&current)? {
                    count += 1;
                }
                current = current + TimeDelta64::new(1, DateTimeUnit::Day);
            }
            
            Ok(count)
        }
        
        /// Create a calendar with US federal holidays for a given year
        pub fn us_federal(year: i32) -> Result<Self> {
            let mut calendar = Self::new();
            
            // New Year's Day - January 1
            calendar.add_holiday(DateTime64::from_iso_string(&format!("{}-01-01", year), DateTimeUnit::Day)?);
            
            // Independence Day - July 4
            calendar.add_holiday(DateTime64::from_iso_string(&format!("{}-07-04", year), DateTimeUnit::Day)?);
            
            // Christmas Day - December 25
            calendar.add_holiday(DateTime64::from_iso_string(&format!("{}-12-25", year), DateTimeUnit::Day)?);
            
            // TODO: Add more complex holidays (Memorial Day, Labor Day, Thanksgiving, etc.)
            // These require more complex date calculations
            
            Ok(calendar)
        }
    }
    
    impl Default for HolidayCalendar {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    
    #[test]
    fn test_datetime_parsing() {
        // Test ISO string parsing
        let dt = DateTime64::from_iso_string("2023-12-25T15:30:45", DateTimeUnit::Second).unwrap();
        let iso_str = dt.to_iso_string().unwrap();
        assert!(iso_str.starts_with("2023-12-25T15:30:45"));
        
        // Test date-only parsing
        let dt2 = DateTime64::from_iso_string("2023-01-01", DateTimeUnit::Day).unwrap();
        let iso_str2 = dt2.to_iso_string().unwrap();
        assert!(iso_str2.starts_with("2023-01-01T00:00:00"));
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
    fn test_date_range_creation() {
        // Test date range with end date
        let range1 = datetime_array::date_range(
            "2023-01-01", 
            Some("2023-01-05"), 
            None, 
            DateTimeUnit::Day,
            DateTimeUnit::Day
        ).unwrap();
        assert!(range1.size() >= 4);
        
        // Test date range with periods
        let range2 = datetime_array::date_range(
            "2023-01-01", 
            None, 
            Some(5), 
            DateTimeUnit::Day,
            DateTimeUnit::Day
        ).unwrap();
        assert_eq!(range2.size(), 5);
    }
    
    #[test]
    fn test_timedelta_range_creation() {
        // Test timedelta range with end
        let range1 = datetime_array::timedelta_range(
            0, 
            Some(10), 
            None, 
            DateTimeUnit::Second
        ).unwrap();
        assert_eq!(range1.size(), 11); // 0 to 10 inclusive
        
        // Test timedelta range with periods
        let range2 = datetime_array::timedelta_range(
            5, 
            None, 
            Some(3), 
            DateTimeUnit::Minute
        ).unwrap();
        assert_eq!(range2.size(), 3);
    }
    
    #[test]
    fn test_business_days() {
        use business_days::*;
        
        // Test weekday calculation (2023-01-01 was a Sunday)
        let dt = DateTime64::from_iso_string("2023-01-01", DateTimeUnit::Day).unwrap();
        let wd = weekday(&dt).unwrap();
        assert_eq!(wd, Weekday::Sunday);
        assert!(!wd.is_business_day());
        
        // Test business day check (2023-01-02 was a Monday)
        let dt2 = DateTime64::from_iso_string("2023-01-02", DateTimeUnit::Day).unwrap();
        assert!(is_busday(&dt2).unwrap());
        
        // Test business day count
        let start = DateTime64::from_iso_string("2023-01-02", DateTimeUnit::Day).unwrap(); // Monday
        let end = DateTime64::from_iso_string("2023-01-06", DateTimeUnit::Day).unwrap(); // Friday
        let count = busday_count(&start, &end).unwrap();
        assert_eq!(count, 4); // Mon, Tue, Wed, Thu (not including end date)
        
        // Test business day offset
        let offset_dt = busday_offset(&start, 2).unwrap();
        let expected = DateTime64::from_iso_string("2023-01-04", DateTimeUnit::Day).unwrap(); // Wednesday
        assert_eq!(offset_dt.value, expected.value);
    }
    
    #[test]
    fn test_holiday_calendar() {
        use business_days::*;
        
        let mut calendar = HolidayCalendar::new();
        let holiday = DateTime64::from_iso_string("2023-07-04", DateTimeUnit::Day).unwrap();
        calendar.add_holiday(holiday);
        
        // Test holiday detection
        assert!(calendar.is_holiday(&holiday));
        
        let non_holiday = DateTime64::from_iso_string("2023-07-05", DateTimeUnit::Day).unwrap();
        assert!(!calendar.is_holiday(&non_holiday));
        
        // Test US federal calendar creation
        let us_calendar = HolidayCalendar::us_federal(2023).unwrap();
        let new_years = DateTime64::from_iso_string("2023-01-01", DateTimeUnit::Day).unwrap();
        assert!(us_calendar.is_holiday(&new_years));
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
    fn test_timezone_support() {
        // Test timezone creation
        let utc = Timezone::utc();
        assert_eq!(utc.offset_minutes, 0);
        assert_eq!(utc.name, "UTC");
        
        let est = Timezone::est();
        assert_eq!(est.offset_minutes, -300); // -5 hours
        
        let custom_tz = Timezone::fixed_offset("GMT+5:30", 5, 30);
        assert_eq!(custom_tz.offset_minutes, 330); // 5.5 hours
        
        // Test timezone display
        let tz_str = format!("{}", est);
        assert!(tz_str.contains("EST"));
        assert!(tz_str.contains("-05:00"));
    }
    
    #[test]
    fn test_timezone_datetime() {
        // Test creation from UTC
        let utc_dt = DateTime64::from_iso_string("2023-01-01T12:00:00", DateTimeUnit::Second).unwrap();
        let tz_dt = TimezoneDateTime::new(utc_dt, Timezone::est());
        
        // Test local time conversion
        let local_dt = tz_dt.to_local();
        let local_str = local_dt.to_iso_string().unwrap();
        assert!(local_str.starts_with("2023-01-01T07:00:00")); // UTC 12:00 = EST 07:00
        
        // Test timezone conversion
        let pst_dt = tz_dt.to_timezone(Timezone::pst());
        assert_eq!(pst_dt.timezone.name, "PST");
        assert_eq!(pst_dt.utc_datetime, tz_dt.utc_datetime); // UTC time should be same
    }
    
    #[test]
    fn test_timezone_datetime_parsing() {
        // Test UTC parsing
        let utc_dt = TimezoneDateTime::from_iso_string_with_tz(
            "2023-01-01T12:00:00Z", 
            DateTimeUnit::Second
        ).unwrap();
        assert_eq!(utc_dt.timezone.name, "UTC");
        
        // Test positive offset parsing
        let plus_dt = TimezoneDateTime::from_iso_string_with_tz(
            "2023-01-01T12:00:00+05:30", 
            DateTimeUnit::Second
        ).unwrap();
        assert_eq!(plus_dt.timezone.offset_minutes, 330);
        
        // Test negative offset parsing
        let minus_dt = TimezoneDateTime::from_iso_string_with_tz(
            "2023-01-01T12:00:00-08:00", 
            DateTimeUnit::Second
        ).unwrap();
        assert_eq!(minus_dt.timezone.offset_minutes, -480);
        
        // Test round-trip conversion
        let iso_str = plus_dt.to_iso_string_with_tz().unwrap();
        assert!(iso_str.contains("+05:30"));
    }
    
    #[test]
    fn test_timezone_datetime_arithmetic() {
        let utc_dt = DateTime64::from_iso_string("2023-01-01T12:00:00", DateTimeUnit::Second).unwrap();
        let tz_dt = TimezoneDateTime::new(utc_dt, Timezone::est());
        
        // Test addition
        let td = TimeDelta64::new(3600, DateTimeUnit::Second); // 1 hour
        let result = tz_dt.clone() + td;
        assert_eq!(result.timezone.name, "EST");
        
        // Test subtraction with timedelta
        let result2 = tz_dt.clone() - td;
        assert_eq!(result2.timezone.name, "EST");
        
        // Test datetime difference
        let diff = result - tz_dt;
        assert_eq!(diff.value, 3600);
        assert_eq!(diff.unit, DateTimeUnit::Second);
    }
}
