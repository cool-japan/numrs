//! NumPy-compatible API for datetime operations
//!
//! This module provides NumPy-compatible functions for creating and manipulating
//! datetime values, including array operations.

use std::str::FromStr;
use std::time::SystemTime;

use chrono::{DateTime as ChronoDateTime, Local, Utc};

use crate::array::Array;
use crate::error::{NumRs2Error, Result};

use super::datetime64::{days_to_date, DateTime64};
use super::timedelta64::TimeDelta64;
use super::timezone::business_days;
use super::units::{parse_unit_string, DateTimeUnit};

// ============================================================================
// NumPy-compatible API functions
// ============================================================================

/// NumPy-compatible constructor for datetime64 values
///
/// Creates a DateTime64 from a string representation, similar to np.datetime64()
///
/// # Arguments
/// * `value` - String representation of datetime (e.g., "2023-01-01", "2023-01-01T12:00:00")
/// * `unit` - Optional unit specification (e.g., "D", "s", "ms")
///
/// # Examples
/// ```
/// use numrs2::types::datetime::datetime64;
///
/// let dt = datetime64("2023-01-01", Some("D")).expect("should parse datetime64");
/// let dt_auto = datetime64("2023-01-01T12:00:00", None).expect("should auto-detect unit");
/// ```
pub fn datetime64(value: &str, unit: Option<&str>) -> Result<DateTime64> {
    let parsed_unit = if let Some(u) = unit {
        parse_unit_string(u)?
    } else {
        // Auto-detect unit based on string format
        if value.contains('T') || value.contains(' ') {
            if value.contains('.') {
                DateTimeUnit::Microsecond
            } else {
                DateTimeUnit::Second
            }
        } else {
            DateTimeUnit::Day
        }
    };

    DateTime64::from_str(value).map(|dt| dt.to_unit(parsed_unit))
}

/// NumPy-compatible constructor for timedelta64 values
///
/// Creates a TimeDelta64 from a numeric value and unit, similar to np.timedelta64()
///
/// # Arguments
/// * `value` - Numeric value for the timedelta
/// * `unit` - Unit specification (e.g., "D", "s", "ms")
///
/// # Examples
/// ```
/// use numrs2::types::datetime::timedelta64;
///
/// let td = timedelta64(5, "D").expect("should create 5 day timedelta");  // 5 days
/// let td_hours = timedelta64(24, "h").expect("should create 24 hour timedelta");  // 24 hours
/// ```
pub fn timedelta64(value: i64, unit: &str) -> Result<TimeDelta64> {
    let parsed_unit = parse_unit_string(unit)?;
    Ok(TimeDelta64::new(value, parsed_unit))
}

/// Convert datetime64 to string representation
///
/// Similar to np.datetime_as_string(), converts a DateTime64 to its string
/// representation.
///
/// # Arguments
/// * `dt` - DateTime64 value to convert. If `dt.is_nat()`, the result is
///   always the literal string `"NaT"`, regardless of `unit`/`timezone`
///   (matching `np.datetime_as_string(np.datetime64('NaT'))`).
/// * `unit` - Optional unit for output format (defaults to `dt`'s own unit)
/// * `timezone` - Timezone mode, matching NumPy's `timezone` argument:
///   - `None` or `Some("naive")` (case-insensitive): no timezone suffix.
///     This is the crate's pre-existing formatting, unchanged.
///   - `Some("UTC")`: append a `Z` suffix (NumPy's UTC marker).
///   - `Some("local")`: shift the displayed instant by this machine's local
///     UTC offset (via `chrono::Local`, looked up for `dt`'s own instant so
///     DST is handled correctly) and append a `+HHMM`/`-HHMM` suffix.
///   - Any other value is a `ValueError`.
///
/// # NumPy deviations
/// * For `timezone = "local"`, real NumPy raises for a Day-or-coarser unit
///   (there is no time-of-day to shift). This crate instead always shifts
///   the underlying instant by the local offset before re-deriving the
///   requested unit's components, which can change the displayed calendar
///   date near a timezone boundary rather than erroring.
/// * The `Z` / `+HHMM`/`-HHMM` suffix is appended uniformly for every unit
///   this function supports (including `Y`/`M`), rather than replicating
///   NumPy's own per-unit formatting rules exactly.
///
/// # Examples
/// ```
/// use numrs2::types::datetime::{datetime64, datetime_as_string};
///
/// let dt = datetime64("2023-01-01", Some("D")).expect("should parse datetime64");
/// let naive = datetime_as_string(&dt, None, None).expect("should convert to string");
/// let utc = datetime_as_string(&dt, None, Some("UTC")).expect("should format as UTC");
/// assert_eq!(naive, "2023-01-01");
/// assert_eq!(utc, "2023-01-01Z");
/// ```
pub fn datetime_as_string(
    dt: &DateTime64,
    unit: Option<&str>,
    timezone: Option<&str>,
) -> Result<String> {
    if dt.is_nat() {
        return Ok("NaT".to_string());
    }

    let target_unit = if let Some(u) = unit {
        parse_unit_string(u)?
    } else {
        dt.unit()
    };

    let tz_mode = timezone.unwrap_or("naive").to_lowercase();

    match tz_mode.as_str() {
        "naive" => {
            let converted_dt = dt.to_unit(target_unit);
            format_naive_string(converted_dt, target_unit)
        }
        "utc" => {
            let converted_dt = dt.to_unit(target_unit);
            let core = format_naive_string(converted_dt, target_unit)?;
            if core.ends_with('Z') {
                Ok(core)
            } else {
                Ok(format!("{}Z", core))
            }
        }
        "local" => {
            let offset_seconds = local_utc_offset_seconds(dt)?;

            // Shift at nanosecond precision (the finest unit) so no
            // sub-second precision is lost before re-deriving `target_unit`.
            let dt_ns = dt.to_unit(DateTimeUnit::Nanosecond);
            let shifted_ns = DateTime64::new(
                dt_ns.value() + offset_seconds as i64 * 1_000_000_000,
                DateTimeUnit::Nanosecond,
            );
            let converted_dt = shifted_ns.to_unit(target_unit);
            let core = format_naive_string(converted_dt, target_unit)?;
            // format_naive_string hardcodes a trailing `Z` for sub-second
            // units; strip it before appending the numeric local offset.
            let stripped = core.strip_suffix('Z').unwrap_or(&core);

            let sign = if offset_seconds >= 0 { '+' } else { '-' };
            let abs_secs = offset_seconds.unsigned_abs();
            let offset_hours = abs_secs / 3600;
            let offset_minutes = (abs_secs % 3600) / 60;
            Ok(format!(
                "{stripped}{sign}{offset_hours:02}{offset_minutes:02}"
            ))
        }
        other => Err(NumRs2Error::ValueError(format!(
            "Unknown timezone mode '{}' for datetime_as_string: expected 'naive', 'UTC', or 'local'",
            other
        ))),
    }
}

/// Core NumPy-style ISO-8601 formatting with no timezone-suffix logic of its
/// own. `dt` must already be converted to `unit`. This is byte-for-byte the
/// pre-existing (pre-timezone-support) formatting, including its existing
/// quirk of hardcoding a trailing `Z` for sub-second units even though it
/// takes no timezone parameter -- callers needing a timezone-aware string
/// should go through [`datetime_as_string`] instead, which layers
/// naive/UTC/local handling on top of this.
fn format_naive_string(dt: DateTime64, unit: DateTimeUnit) -> Result<String> {
    match unit {
        DateTimeUnit::Year => Ok(format!("{:04}", 1970 + dt.value())),
        DateTimeUnit::Month => {
            let years = dt.value() / 12;
            let months = dt.value() % 12;
            Ok(format!("{:04}-{:02}", 1970 + years, months + 1))
        }
        DateTimeUnit::Day => {
            // Convert days since epoch to date
            let days = dt.value();
            let (year, month, day) = days_to_date(days);
            Ok(format!("{:04}-{:02}-{:02}", year, month, day))
        }
        DateTimeUnit::Second => {
            let secs = dt.value();
            let days = secs / 86400;
            let remaining_secs = secs % 86400;
            let hours = remaining_secs / 3600;
            let minutes = (remaining_secs % 3600) / 60;
            let seconds = remaining_secs % 60;

            let (year, month, day) = days_to_date(days);
            Ok(format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
                year, month, day, hours, minutes, seconds
            ))
        }
        DateTimeUnit::Millisecond => {
            let millis = dt.value();
            let secs = millis / 1000;
            let subsec_millis = millis % 1000;
            let days = secs / 86400;
            let remaining_secs = secs % 86400;
            let hours = remaining_secs / 3600;
            let minutes = (remaining_secs % 3600) / 60;
            let seconds = remaining_secs % 60;

            let (year, month, day) = days_to_date(days);
            Ok(format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
                year, month, day, hours, minutes, seconds, subsec_millis
            ))
        }
        DateTimeUnit::Microsecond => {
            let micros = dt.value();
            let secs = micros / 1_000_000;
            let subsec_micros = micros % 1_000_000;
            let days = secs / 86400;
            let remaining_secs = secs % 86400;
            let hours = remaining_secs / 3600;
            let minutes = (remaining_secs % 3600) / 60;
            let seconds = remaining_secs % 60;

            let (year, month, day) = days_to_date(days);
            Ok(format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}Z",
                year, month, day, hours, minutes, seconds, subsec_micros
            ))
        }
        DateTimeUnit::Nanosecond => {
            let nanos = dt.value();
            let secs = nanos / 1_000_000_000;
            let subsec_nanos = nanos % 1_000_000_000;
            let days = secs / 86400;
            let remaining_secs = secs % 86400;
            let hours = remaining_secs / 3600;
            let minutes = (remaining_secs % 3600) / 60;
            let seconds = remaining_secs % 60;

            let (year, month, day) = days_to_date(days);
            Ok(format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:09}Z",
                year, month, day, hours, minutes, seconds, subsec_nanos
            ))
        }
        _ => {
            // For other units (Week, Hour, Minute), convert to day and format.
            let as_days = dt.to_unit(DateTimeUnit::Day);
            format_naive_string(as_days, DateTimeUnit::Day)
        }
    }
}

/// This machine's local UTC offset (in seconds) for the instant `dt`
/// represents, via `chrono::Local` (looked up for that specific instant, so
/// DST transitions are handled the same way `chrono` handles them).
fn local_utc_offset_seconds(dt: &DateTime64) -> Result<i32> {
    let secs = dt.to_unit(DateTimeUnit::Second).value();
    let utc_dt: ChronoDateTime<Utc> = ChronoDateTime::from_timestamp(secs, 0).ok_or_else(|| {
        NumRs2Error::ValueError(format!(
            "Timestamp {} seconds since epoch is out of range for timezone conversion",
            secs
        ))
    })?;
    let local_dt = utc_dt.with_timezone(&Local);
    Ok(local_dt.offset().local_minus_utc())
}

/// Get unit and count information from datetime64 dtype
///
/// Similar to np.datetime_data(), returns the unit and count for a DateTime64
///
/// # Arguments
/// * `dt` - DateTime64 value to analyze
///
/// # Returns
/// Tuple of (unit_string, count) where count is always 1 for DateTime64
///
/// # Examples
/// ```
/// use numrs2::types::datetime::{datetime64, datetime_data};
///
/// let dt = datetime64("2023-01-01", Some("D")).expect("should parse datetime64");
/// let (unit, count) = datetime_data(&dt);
/// assert_eq!(unit, "D");
/// assert_eq!(count, 1);
/// ```
pub fn datetime_data(dt: &DateTime64) -> (String, i64) {
    let unit_str = match dt.unit() {
        DateTimeUnit::Year => "Y",
        DateTimeUnit::Month => "M",
        DateTimeUnit::Week => "W",
        DateTimeUnit::Day => "D",
        DateTimeUnit::Hour => "h",
        DateTimeUnit::Minute => "m",
        DateTimeUnit::Second => "s",
        DateTimeUnit::Millisecond => "ms",
        DateTimeUnit::Microsecond => "us",
        DateTimeUnit::Nanosecond => "ns",
    };
    (unit_str.to_string(), 1)
}

// ============================================================================
// Array creation functions for datetime arrays
// ============================================================================

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
        unit: DateTimeUnit,
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

                while current.value() <= end_dt.value() {
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
            (Some(_), Some(_)) => Err(NumRs2Error::ValueError(
                "Cannot specify both end and periods".to_string(),
            )),
            (None, None) => Err(NumRs2Error::ValueError(
                "Must specify either end or periods".to_string(),
            )),
        }
    }

    /// Create an array of timedelta values
    pub fn timedelta_range(
        start: i64,
        end: Option<i64>,
        periods: Option<usize>,
        unit: DateTimeUnit,
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
            (Some(_), Some(_)) => Err(NumRs2Error::ValueError(
                "Cannot specify both end and periods".to_string(),
            )),
            (None, None) => Err(NumRs2Error::ValueError(
                "Must specify either end or periods".to_string(),
            )),
        }
    }

    /// Create an array of datetime values from string representations
    pub fn datetime_from_strings(
        strings: &[&str],
        unit: DateTimeUnit,
    ) -> Result<Array<DateTime64>> {
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

// ============================================================================
// Array-based datetime operations
// ============================================================================

/// Array-based datetime operations
pub mod array_ops {
    use super::*;

    /// Convert an array of datetime64 values to string representations
    pub fn datetime_as_string_array(
        dts: &Array<DateTime64>,
        unit: Option<&str>,
        timezone: Option<&str>,
    ) -> Result<Array<String>> {
        let data = dts.to_vec();
        let strings: Result<Vec<_>> = data
            .iter()
            .map(|dt| super::datetime_as_string(dt, unit, timezone))
            .collect();

        let string_vec = strings?;
        let shape = dts.shape();
        Array::from_vec_shape(string_vec, &shape)
    }

    /// Check if datetime64 values are business days
    pub fn is_busday_array(dts: &Array<DateTime64>) -> Result<Array<bool>> {
        let data = dts.to_vec();
        let results: Result<Vec<bool>> = data.iter().map(business_days::is_busday).collect();

        let result_vec = results?;
        let shape = dts.shape();
        Array::from_vec_shape(result_vec, &shape)
    }

    /// Apply business day offset to a datetime64 array, element-wise.
    ///
    /// `roll` is NumPy's roll convention (`"raise"` when `None`; see
    /// [`business_days::Roll`] for the full list) applied to each element
    /// before its offset -- see [`business_days::busday_offset`] for the
    /// exact per-element algorithm. This array-level entry point does not
    /// accept a `holidays` list (only the fixed Saturday/Sunday weekend is
    /// applied); use [`business_days::busday_offset`] directly for
    /// holiday-aware rolling of a single date.
    pub fn busday_offset_array(
        dts: &Array<DateTime64>,
        offsets: &Array<i32>,
        roll: Option<&str>,
    ) -> Result<Array<DateTime64>> {
        if dts.len() != offsets.len() {
            return Err(NumRs2Error::ValueError(
                "Arrays must have same length".to_string(),
            ));
        }

        let dt_data = dts.to_vec();
        let offset_data = offsets.to_vec();
        let results: Result<Vec<_>> = dt_data
            .iter()
            .zip(offset_data.iter())
            .map(|(dt, &offset)| business_days::busday_offset(dt, offset as i64, roll, None))
            .collect();

        let result_vec = results?;
        let shape = dts.shape();
        Array::from_vec_shape(result_vec, &shape)
    }

    /// Count business days between corresponding elements of two datetime64
    /// arrays, element-wise.
    ///
    /// Mirrors NumPy's `busday_count(begindates, enddates)`: each result
    /// element counts business days in the half-open interval
    /// `[begin, end)` (see [`business_days::busday_count`]), negative when
    /// `end < begin`. Does not accept a `holidays` list at the array level
    /// (only the fixed Saturday/Sunday weekend applies); use
    /// [`business_days::busday_count`] directly for holiday-aware counting
    /// of a single pair of dates.
    pub fn busday_count_array(
        begin_dts: &Array<DateTime64>,
        end_dts: &Array<DateTime64>,
    ) -> Result<Array<i64>> {
        if begin_dts.len() != end_dts.len() {
            return Err(NumRs2Error::ValueError(
                "Arrays must have same length".to_string(),
            ));
        }

        let begin_data = begin_dts.to_vec();
        let end_data = end_dts.to_vec();
        let results: Result<Vec<i64>> = begin_data
            .iter()
            .zip(end_data.iter())
            .map(|(begin, end)| business_days::busday_count(begin, end, None))
            .collect();

        let result_vec = results?;
        let shape = begin_dts.shape();
        Array::from_vec_shape(result_vec, &shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datetime64_constructor() {
        let dt =
            datetime64("2023-01-01", Some("D")).expect("should parse datetime64 with day unit");
        assert_eq!(dt.unit(), DateTimeUnit::Day);

        let dt_auto =
            datetime64("2023-01-01T12:00:00", None).expect("should parse datetime64 auto-detect");
        assert_eq!(dt_auto.unit(), DateTimeUnit::Second);
    }

    #[test]
    fn test_timedelta64_constructor() {
        let td = timedelta64(5, "D").expect("should create timedelta64 with day unit");
        assert_eq!(td.unit(), DateTimeUnit::Day);
        assert_eq!(td.value(), 5);

        let td_hours = timedelta64(24, "h").expect("should create timedelta64 with hour unit");
        assert_eq!(td_hours.unit(), DateTimeUnit::Hour);
        assert_eq!(td_hours.value(), 24);
    }

    #[test]
    fn test_datetime_as_string() {
        let dt = datetime64("2023-01-01", Some("D")).expect("should parse datetime64");
        let s = datetime_as_string(&dt, None, None).expect("should convert to string");
        assert!(s.contains("2023"));
    }

    #[test]
    fn test_datetime_as_string_naive_matches_current_behavior() {
        // timezone=None and timezone=Some("naive") must both match the
        // crate's pre-existing (pre-timezone-support) formatting exactly:
        // no suffix for Day/Second, but the pre-existing trailing `Z` quirk
        // preserved for sub-second units.
        let day = datetime64("2023-01-01", Some("D")).expect("should parse day");
        assert_eq!(
            datetime_as_string(&day, None, None).expect("naive via None"),
            "2023-01-01"
        );
        assert_eq!(
            datetime_as_string(&day, None, Some("naive")).expect("naive via Some"),
            "2023-01-01"
        );

        let secs = datetime64("2023-01-01T12:30:45", Some("s")).expect("should parse seconds");
        assert_eq!(
            datetime_as_string(&secs, None, None).expect("naive seconds"),
            "2023-01-01T12:30:45"
        );
    }

    #[test]
    fn test_datetime_as_string_utc_appends_z() {
        let day = datetime64("2023-01-01", Some("D")).expect("should parse day");
        assert_eq!(
            datetime_as_string(&day, None, Some("UTC")).expect("utc day"),
            "2023-01-01Z"
        );
        // Case-insensitive.
        assert_eq!(
            datetime_as_string(&day, None, Some("utc")).expect("utc lowercase"),
            "2023-01-01Z"
        );

        let secs = datetime64("2023-01-01T12:30:45", Some("s")).expect("should parse seconds");
        assert_eq!(
            datetime_as_string(&secs, None, Some("UTC")).expect("utc seconds"),
            "2023-01-01T12:30:45Z"
        );
    }

    #[test]
    fn test_datetime_as_string_unknown_timezone_errors() {
        let dt = datetime64("2023-01-01", Some("D")).expect("should parse datetime64");
        assert!(datetime_as_string(&dt, None, Some("Mars/Olympus_Mons")).is_err());
    }

    #[test]
    fn test_datetime_as_string_nat() {
        let nat = DateTime64::nat(DateTimeUnit::Day);
        assert_eq!(
            datetime_as_string(&nat, None, None).expect("NaT should format"),
            "NaT"
        );
        assert_eq!(
            datetime_as_string(&nat, None, Some("UTC")).expect("NaT should format under UTC too"),
            "NaT"
        );
    }

    #[test]
    fn test_datetime_data() {
        let dt = datetime64("2023-01-01", Some("D")).expect("should parse datetime64");
        let (unit, count) = datetime_data(&dt);
        assert_eq!(unit, "D");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_date_range_creation() {
        // Test date range with end date
        let range1 = datetime_array::date_range(
            "2023-01-01",
            Some("2023-01-05"),
            None,
            DateTimeUnit::Day,
            DateTimeUnit::Day,
        )
        .expect("should create date range with end date");
        assert!(range1.size() >= 4);

        // Test date range with periods
        let range2 = datetime_array::date_range(
            "2023-01-01",
            None,
            Some(5),
            DateTimeUnit::Day,
            DateTimeUnit::Day,
        )
        .expect("should create date range with periods");
        assert_eq!(range2.size(), 5);
    }

    #[test]
    fn test_timedelta_range_creation() {
        // Test timedelta range with end
        let range1 = datetime_array::timedelta_range(0, Some(10), None, DateTimeUnit::Second)
            .expect("should create timedelta range with end");
        assert_eq!(range1.size(), 11); // 0 to 10 inclusive

        // Test timedelta range with periods
        let range2 = datetime_array::timedelta_range(5, None, Some(3), DateTimeUnit::Minute)
            .expect("should create timedelta range with periods");
        assert_eq!(range2.size(), 3);
    }
}
