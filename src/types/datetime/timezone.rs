//! Timezone support for NumRS
//!
//! This module provides timezone-aware datetime functionality and business day
//! calculations including holiday calendars.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Sub};
use std::str::FromStr;

use crate::error::{NumRs2Error, Result};

use super::datetime64::{days_to_date, DateTime64};
use super::timedelta64::TimeDelta64;
use super::units::DateTimeUnit;

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

    /// Pacific Standard Time
    pub fn pst() -> Self {
        Self::fixed_offset("PST", -8, 0)
    }

    /// Central European Time
    pub fn cet() -> Self {
        Self::fixed_offset("CET", 1, 0)
    }

    /// Japan Standard Time
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
        let offset_delta = TimeDelta64::new(timezone.offset_minutes as i64, DateTimeUnit::Minute);
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
        if let Some(datetime_part) = s.strip_suffix('Z') {
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
                    let tz_hours: i32 = tz_components[0].parse().map_err(|_| {
                        NumRs2Error::ValueError("Invalid timezone hours".to_string())
                    })?;
                    let tz_minutes: i32 = tz_components[1].parse().map_err(|_| {
                        NumRs2Error::ValueError("Invalid timezone minutes".to_string())
                    })?;

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
        let offset_delta =
            TimeDelta64::new(self.timezone.offset_minutes as i64, DateTimeUnit::Minute);
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
            let sign = if self.timezone.offset_minutes >= 0 {
                "+"
            } else {
                "-"
            };
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

// ============================================================================
// Business day and calendar functions
// ============================================================================

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
            matches!(
                self,
                Weekday::Monday
                    | Weekday::Tuesday
                    | Weekday::Wednesday
                    | Weekday::Thursday
                    | Weekday::Friday
            )
        }
    }

    /// Get the weekday for a given date
    pub fn weekday(dt: &DateTime64) -> Result<Weekday> {
        if dt.is_nat() {
            return Err(NumRs2Error::ValueError(
                "Cannot compute weekday of NaT (Not a Time)".to_string(),
            ));
        }

        let dt_days = dt.to_unit(DateTimeUnit::Day);
        // Unix epoch (1970-01-01) was a Thursday (index 3)
        let days_since_epoch = dt_days.value();
        let weekday_index = ((days_since_epoch + 3) % 7 + 7) % 7; // Handle negative numbers

        match weekday_index {
            0 => Ok(Weekday::Monday),
            1 => Ok(Weekday::Tuesday),
            2 => Ok(Weekday::Wednesday),
            3 => Ok(Weekday::Thursday),
            4 => Ok(Weekday::Friday),
            5 => Ok(Weekday::Saturday),
            6 => Ok(Weekday::Sunday),
            _ => Err(NumRs2Error::ValueError(
                "Invalid weekday calculation".to_string(),
            )),
        }
    }

    /// Check if a date is a business day (Monday-Friday). This crate does
    /// not support a custom weekmask; every roll/offset/count function in
    /// this module treats Saturday and Sunday as the only fixed non-business
    /// weekdays, and layers an optional explicit `holidays` list on top
    /// (see [`busday_offset`] and [`busday_count`]).
    pub fn is_busday(dt: &DateTime64) -> Result<bool> {
        let wd = weekday(dt)?;
        Ok(wd.is_business_day())
    }

    /// `true` if `date` is a business day *and* is not one of `holidays`.
    ///
    /// This is the single predicate shared by [`busday_offset`]'s rolling
    /// and stepping logic and by [`busday_count`], so that an explicit
    /// holiday list affects rolling, offsetting, and counting consistently.
    /// `date` may be in any unit; `holidays` entries are compared by
    /// calendar day regardless of their own unit.
    fn is_valid_business_day(date: &DateTime64, holidays: Option<&[DateTime64]>) -> Result<bool> {
        if !is_busday(date)? {
            return Ok(false);
        }

        if let Some(holiday_list) = holidays {
            let day_value = date.to_unit(DateTimeUnit::Day).value();
            let is_holiday = holiday_list
                .iter()
                .any(|h| h.to_unit(DateTimeUnit::Day).value() == day_value);
            if is_holiday {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Roll convention for [`busday_offset`], matching NumPy's `roll`
    /// parameter (`numpy.busday_offset(..., roll=...)`).
    ///
    /// Parsed from a string via [`FromStr`] (case-insensitive): `"raise"`,
    /// `"nat"`, `"forward"`/`"following"`, `"backward"`/`"preceding"`,
    /// `"modifiedfollowing"`, `"modifiedpreceding"`.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Roll {
        /// Return an error if the date is not a business day (NumPy's default).
        Raise,
        /// Return NaT ([`DateTime64::nat`]) if the date is not a business day.
        Nat,
        /// Take the first business day later in time ("following").
        Forward,
        /// Take the first business day earlier in time ("preceding").
        Backward,
        /// Like [`Roll::Forward`], unless that would cross into the next
        /// calendar month, in which case behave like [`Roll::Backward`].
        ModifiedFollowing,
        /// Like [`Roll::Backward`], unless that would cross into the
        /// previous calendar month, in which case behave like [`Roll::Forward`].
        ModifiedPreceding,
    }

    impl FromStr for Roll {
        type Err = NumRs2Error;

        fn from_str(s: &str) -> Result<Self> {
            match s.to_lowercase().as_str() {
                "raise" => Ok(Roll::Raise),
                "nat" => Ok(Roll::Nat),
                "forward" | "following" => Ok(Roll::Forward),
                "backward" | "preceding" => Ok(Roll::Backward),
                "modifiedfollowing" => Ok(Roll::ModifiedFollowing),
                "modifiedpreceding" => Ok(Roll::ModifiedPreceding),
                _ => Err(NumRs2Error::ValueError(format!(
                    "Unknown roll convention '{}': expected one of 'raise', 'nat', \
                     'forward'/'following', 'backward'/'preceding', 'modifiedfollowing', \
                     'modifiedpreceding'",
                    s
                ))),
            }
        }
    }

    /// Count business days in the half-open interval `[begin, end)`.
    ///
    /// Matches NumPy's `busday_count` semantics: `begin` is counted if it is
    /// a business day, `end` is never counted. If `end` is earlier than
    /// `begin`, the result is negative. `holidays`, when given, lists
    /// additional dates excluded from the count even on a weekday (mirrors
    /// NumPy's `holidays` parameter); this crate has no weekmask concept, so
    /// Saturday/Sunday are always the only implicit non-business days (see
    /// [`is_busday`]).
    pub fn busday_count(
        begin: &DateTime64,
        end: &DateTime64,
        holidays: Option<&[DateTime64]>,
    ) -> Result<i64> {
        if begin.is_nat() || end.is_nat() {
            return Err(NumRs2Error::ValueError(
                "Cannot count business days with a NaT (Not a Time) endpoint".to_string(),
            ));
        }

        let begin_days = begin.to_unit(DateTimeUnit::Day);
        let end_days = end.to_unit(DateTimeUnit::Day);

        if begin_days.value() > end_days.value() {
            return Ok(-busday_count(end, begin, holidays)?);
        }

        let mut count = 0i64;
        let mut current = begin_days;

        while current.value() < end_days.value() {
            if is_valid_business_day(&current, holidays)? {
                count += 1;
            }
            current = current + TimeDelta64::new(1, DateTimeUnit::Day);
        }

        Ok(count)
    }

    /// Offset a date by a number of business days, applying NumPy's roll
    /// convention first.
    ///
    /// NumPy's algorithm (see `numpy.busday_offset`) is two steps:
    /// 1. If `dt` is not already a valid business day, roll it to one
    ///    according to `roll` (see [`Roll`]); if `dt` is already valid, no
    ///    rolling happens regardless of `roll`.
    /// 2. Starting from that business day, step `offset` business days
    ///    forward (positive) or backward (negative), skipping non-business
    ///    days. `offset == 0` leaves the rolled date unchanged, which is
    ///    exactly NumPy's documented behavior for a zero offset.
    ///
    /// `roll` defaults to `"raise"` when `None`, matching NumPy's default.
    /// `holidays`, when given, excludes those dates from both the rolling
    /// and the stepping steps (both go through the same internal
    /// business-day-with-holidays predicate as [`busday_count`]).
    ///
    /// If `dt` is NaT, NaT is returned unconditionally (NaT propagates,
    /// like NaN in floating-point arithmetic) regardless of `roll`.
    pub fn busday_offset(
        dt: &DateTime64,
        offset: i64,
        roll: Option<&str>,
        holidays: Option<&[DateTime64]>,
    ) -> Result<DateTime64> {
        if dt.is_nat() {
            return Ok(DateTime64::nat(dt.unit()));
        }

        let roll_mode = Roll::from_str(roll.unwrap_or("raise"))?;
        let original_unit = dt.unit();
        let date_days = dt.to_unit(DateTimeUnit::Day);

        let rolled = match roll_to_business_day(date_days, roll_mode, holidays)? {
            Some(d) => d,
            None => return Ok(DateTime64::nat(original_unit)),
        };

        let offset_result = step_business_days(rolled, offset, holidays)?;
        Ok(offset_result.to_unit(original_unit))
    }

    /// Roll `date` (already in [`DateTimeUnit::Day`] units) to a business
    /// day per `roll`. Returns `Ok(None)` only for [`Roll::Nat`] when `date`
    /// is not a business day; otherwise the resolved business day, or an
    /// error for [`Roll::Raise`].
    fn roll_to_business_day(
        date: DateTime64,
        roll: Roll,
        holidays: Option<&[DateTime64]>,
    ) -> Result<Option<DateTime64>> {
        if is_valid_business_day(&date, holidays)? {
            return Ok(Some(date));
        }

        match roll {
            Roll::Raise => {
                let (y, m, d) = days_to_date(date.value());
                Err(NumRs2Error::ValueError(format!(
                    "Non-business day date {:04}-{:02}-{:02} with roll='raise'",
                    y, m, d
                )))
            }
            Roll::Nat => Ok(None),
            Roll::Forward => Ok(Some(nearest_business_day(date, 1, holidays)?)),
            Roll::Backward => Ok(Some(nearest_business_day(date, -1, holidays)?)),
            Roll::ModifiedFollowing => {
                let forward = nearest_business_day(date, 1, holidays)?;
                if same_month(date, forward) {
                    Ok(Some(forward))
                } else {
                    Ok(Some(nearest_business_day(date, -1, holidays)?))
                }
            }
            Roll::ModifiedPreceding => {
                let backward = nearest_business_day(date, -1, holidays)?;
                if same_month(date, backward) {
                    Ok(Some(backward))
                } else {
                    Ok(Some(nearest_business_day(date, 1, holidays)?))
                }
            }
        }
    }

    /// Scan one day at a time in `direction` (`1` or `-1`) from `date`
    /// (Day units) until a valid business day is found.
    fn nearest_business_day(
        mut date: DateTime64,
        direction: i64,
        holidays: Option<&[DateTime64]>,
    ) -> Result<DateTime64> {
        loop {
            date = date + TimeDelta64::new(direction, DateTimeUnit::Day);
            if is_valid_business_day(&date, holidays)? {
                return Ok(date);
            }
        }
    }

    /// `true` if `a` and `b` (both Day units) fall in the same calendar
    /// year and month, used by the `modifiedfollowing`/`modifiedpreceding`
    /// roll conventions to detect a month-boundary crossing.
    fn same_month(a: DateTime64, b: DateTime64) -> bool {
        let (ay, am, _) = days_to_date(a.value());
        let (by, bm, _) = days_to_date(b.value());
        ay == by && am == bm
    }

    /// Step `offset` business days from `start` (already a valid business
    /// day, Day units), skipping non-business days.
    fn step_business_days(
        start: DateTime64,
        offset: i64,
        holidays: Option<&[DateTime64]>,
    ) -> Result<DateTime64> {
        let mut current = start;
        let mut remaining = offset.abs();
        let direction = if offset >= 0 { 1 } else { -1 };

        while remaining > 0 {
            current = current + TimeDelta64::new(direction, DateTimeUnit::Day);
            if is_valid_business_day(&current, holidays)? {
                remaining -= 1;
            }
        }

        Ok(current)
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
            self.holidays.iter().any(|h| h.value() == dt_days.value())
        }

        /// Check if a date is a business day (not weekend and not holiday)
        pub fn is_business_day(&self, dt: &DateTime64) -> Result<bool> {
            Ok(is_busday(dt)? && !self.is_holiday(dt))
        }

        /// Count business days between two dates considering holidays
        ///
        /// Delegates to the free [`busday_count`] function with this
        /// calendar's holidays, so both share one implementation of the
        /// `[begin, end)` / negative-when-reversed semantics.
        pub fn business_day_count(&self, start: &DateTime64, end: &DateTime64) -> Result<i64> {
            busday_count(start, end, Some(self.holidays.as_slice()))
        }

        /// Create a calendar with US federal holidays for a given year
        pub fn us_federal(year: i32) -> Result<Self> {
            let mut calendar = Self::new();

            // New Year's Day - January 1
            calendar.add_holiday(DateTime64::from_iso_string(
                &format!("{}-01-01", year),
                DateTimeUnit::Day,
            )?);

            // Independence Day - July 4
            calendar.add_holiday(DateTime64::from_iso_string(
                &format!("{}-07-04", year),
                DateTimeUnit::Day,
            )?);

            // Christmas Day - December 25
            calendar.add_holiday(DateTime64::from_iso_string(
                &format!("{}-12-25", year),
                DateTimeUnit::Day,
            )?);

            // Martin Luther King Jr. Day - Third Monday in January
            if let Some(mlk_day) = Self::nth_weekday_of_month(year, 1, Weekday::Monday, 3) {
                calendar.add_holiday(mlk_day);
            }

            // Presidents' Day - Third Monday in February
            if let Some(presidents_day) = Self::nth_weekday_of_month(year, 2, Weekday::Monday, 3) {
                calendar.add_holiday(presidents_day);
            }

            // Memorial Day - Last Monday in May
            if let Some(memorial_day) = Self::last_weekday_of_month(year, 5, Weekday::Monday) {
                calendar.add_holiday(memorial_day);
            }

            // Labor Day - First Monday in September
            if let Some(labor_day) = Self::nth_weekday_of_month(year, 9, Weekday::Monday, 1) {
                calendar.add_holiday(labor_day);
            }

            // Columbus Day - Second Monday in October
            if let Some(columbus_day) = Self::nth_weekday_of_month(year, 10, Weekday::Monday, 2) {
                calendar.add_holiday(columbus_day);
            }

            // Veterans Day - November 11
            calendar.add_holiday(DateTime64::from_iso_string(
                &format!("{}-11-11", year),
                DateTimeUnit::Day,
            )?);

            // Thanksgiving - Fourth Thursday in November
            if let Some(thanksgiving) = Self::nth_weekday_of_month(year, 11, Weekday::Thursday, 4) {
                calendar.add_holiday(thanksgiving);
            }

            Ok(calendar)
        }

        /// Calculate the nth occurrence of a weekday in a given month
        /// target_weekday: The target Weekday enum value
        /// occurrence: 1=first, 2=second, 3=third, 4=fourth
        fn nth_weekday_of_month(
            year: i32,
            month: u8,
            target_weekday: Weekday,
            occurrence: u8,
        ) -> Option<DateTime64> {
            if month == 0 || month > 12 || occurrence == 0 || occurrence > 5 {
                return None;
            }

            // Get the first day of the month
            let first_day = DateTime64::from_iso_string(
                &format!("{}-{:02}-01", year, month),
                DateTimeUnit::Day,
            )
            .ok()?;
            let first_weekday = weekday(&first_day).ok()?;

            // Calculate days to add to get to the first occurrence of target weekday
            let target_weekday_num = target_weekday as u8;
            let first_weekday_num = first_weekday as u8;

            let days_to_first = if target_weekday_num >= first_weekday_num {
                target_weekday_num - first_weekday_num
            } else {
                7 - (first_weekday_num - target_weekday_num)
            };

            // Calculate the target date
            let target_day = 1 + days_to_first + (occurrence - 1) * 7;

            // Check if the target day is valid for this month
            let days_in_month = Self::days_in_month(year, month);
            if target_day > days_in_month {
                return None;
            }

            DateTime64::from_iso_string(
                &format!("{}-{:02}-{:02}", year, month, target_day),
                DateTimeUnit::Day,
            )
            .ok()
        }

        /// Calculate the last occurrence of a weekday in a given month
        /// target_weekday: The target Weekday enum value
        fn last_weekday_of_month(
            year: i32,
            month: u8,
            target_weekday: Weekday,
        ) -> Option<DateTime64> {
            if month == 0 || month > 12 {
                return None;
            }

            let days_in_month = Self::days_in_month(year, month);

            // Start from the last day and work backwards
            for day in (1..=days_in_month).rev() {
                if let Ok(date) = DateTime64::from_iso_string(
                    &format!("{}-{:02}-{:02}", year, month, day),
                    DateTimeUnit::Day,
                ) {
                    if let Ok(day_weekday) = weekday(&date) {
                        if day_weekday as u8 == target_weekday as u8 {
                            return Some(date);
                        }
                    }
                }
            }

            None
        }

        /// Get the number of days in a given month
        fn days_in_month(year: i32, month: u8) -> u8 {
            match month {
                1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                4 | 6 | 9 | 11 => 30,
                2 => {
                    // Check for leap year
                    if (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0) {
                        29
                    } else {
                        28
                    }
                }
                _ => 0,
            }
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
    use business_days::*;

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
        let utc_dt = DateTime64::from_iso_string("2023-01-01T12:00:00", DateTimeUnit::Second)
            .expect("should parse UTC datetime");
        let tz_dt = TimezoneDateTime::new(utc_dt, Timezone::est());

        // Test local time conversion
        let local_dt = tz_dt.to_local();
        let local_str = local_dt
            .to_iso_string()
            .expect("should convert to ISO string");
        assert!(local_str.starts_with("2023-01-01T07:00:00")); // UTC 12:00 = EST 07:00

        // Test timezone conversion
        let pst_dt = tz_dt.to_timezone(Timezone::pst());
        assert_eq!(pst_dt.timezone.name, "PST");
        assert_eq!(pst_dt.utc_datetime, tz_dt.utc_datetime); // UTC time should be same
    }

    #[test]
    fn test_timezone_datetime_parsing() {
        // Test UTC parsing
        let utc_dt =
            TimezoneDateTime::from_iso_string_with_tz("2023-01-01T12:00:00Z", DateTimeUnit::Second)
                .expect("should parse UTC datetime with Z suffix");
        assert_eq!(utc_dt.timezone.name, "UTC");

        // Test positive offset parsing
        let plus_dt = TimezoneDateTime::from_iso_string_with_tz(
            "2023-01-01T12:00:00+05:30",
            DateTimeUnit::Second,
        )
        .expect("should parse datetime with positive offset");
        assert_eq!(plus_dt.timezone.offset_minutes, 330);

        // Test negative offset parsing
        let minus_dt = TimezoneDateTime::from_iso_string_with_tz(
            "2023-01-01T12:00:00-08:00",
            DateTimeUnit::Second,
        )
        .expect("should parse datetime with negative offset");
        assert_eq!(minus_dt.timezone.offset_minutes, -480);

        // Test round-trip conversion
        let iso_str = plus_dt
            .to_iso_string_with_tz()
            .expect("should convert to ISO string with timezone");
        assert!(iso_str.contains("+05:30"));
    }

    #[test]
    fn test_timezone_datetime_arithmetic() {
        let utc_dt = DateTime64::from_iso_string("2023-01-01T12:00:00", DateTimeUnit::Second)
            .expect("should parse datetime");
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
        assert_eq!(diff.value(), 3600);
        assert_eq!(diff.unit(), DateTimeUnit::Second);
    }

    #[test]
    fn test_business_days() {
        // Test weekday calculation (2023-01-01 was a Sunday)
        let dt = DateTime64::from_iso_string("2023-01-01", DateTimeUnit::Day)
            .expect("should parse date");
        let wd = weekday(&dt).expect("should calculate weekday");
        assert_eq!(wd, Weekday::Sunday);
        assert!(!wd.is_business_day());

        // Test business day check (2023-01-02 was a Monday)
        let dt2 = DateTime64::from_iso_string("2023-01-02", DateTimeUnit::Day)
            .expect("should parse date");
        assert!(is_busday(&dt2).expect("should check business day"));

        // Test business day count
        let start = DateTime64::from_iso_string("2023-01-02", DateTimeUnit::Day)
            .expect("should parse start date"); // Monday
        let end = DateTime64::from_iso_string("2023-01-06", DateTimeUnit::Day)
            .expect("should parse end date"); // Friday
        let count = busday_count(&start, &end, None).expect("should count business days");
        assert_eq!(count, 4); // Mon, Tue, Wed, Thu (not including end date)

        // Test business day offset (roll=None defaults to "raise"; start is
        // already a business day so no rolling is needed here)
        let offset_dt =
            busday_offset(&start, 2, None, None).expect("should calculate business day offset");
        let expected = DateTime64::from_iso_string("2023-01-04", DateTimeUnit::Day)
            .expect("should parse expected date"); // Wednesday
        assert_eq!(offset_dt.value(), expected.value());
    }

    #[test]
    fn test_holiday_calendar() {
        let mut calendar = HolidayCalendar::new();
        let holiday = DateTime64::from_iso_string("2023-07-04", DateTimeUnit::Day)
            .expect("should parse holiday date");
        calendar.add_holiday(holiday);

        // Test holiday detection
        assert!(calendar.is_holiday(&holiday));

        let non_holiday = DateTime64::from_iso_string("2023-07-05", DateTimeUnit::Day)
            .expect("should parse non-holiday date");
        assert!(!calendar.is_holiday(&non_holiday));

        // Test US federal calendar creation
        let us_calendar =
            HolidayCalendar::us_federal(2023).expect("should create US federal calendar");
        let new_years = DateTime64::from_iso_string("2023-01-01", DateTimeUnit::Day)
            .expect("should parse New Year's date");
        assert!(us_calendar.is_holiday(&new_years));
    }

    #[test]
    fn test_holiday_calendar_business_day_count() {
        // Mon 2023-01-02 .. Fri 2023-01-06 is 4 business days (Mon-Thu) with
        // no holidays. Marking the Wednesday (2023-01-04) as a holiday
        // should exclude it, leaving 3 -- this exercises
        // HolidayCalendar::business_day_count, which now delegates to the
        // free `busday_count` function.
        let start = DateTime64::from_iso_string("2023-01-02", DateTimeUnit::Day)
            .expect("should parse start date"); // Monday
        let end = DateTime64::from_iso_string("2023-01-06", DateTimeUnit::Day)
            .expect("should parse end date"); // Friday

        let empty_calendar = HolidayCalendar::new();
        let count_no_holidays = empty_calendar
            .business_day_count(&start, &end)
            .expect("should count business days with no holidays");
        assert_eq!(count_no_holidays, 4);

        let mut calendar = HolidayCalendar::new();
        let wednesday = DateTime64::from_iso_string("2023-01-04", DateTimeUnit::Day)
            .expect("should parse Wednesday date");
        calendar.add_holiday(wednesday);
        let count_with_holiday = calendar
            .business_day_count(&start, &end)
            .expect("should count business days excluding the holiday");
        assert_eq!(count_with_holiday, 3);

        // Reversed range still negates through the delegated implementation.
        let reversed = calendar
            .business_day_count(&end, &start)
            .expect("should count reversed range");
        assert_eq!(reversed, -3);
    }
}
