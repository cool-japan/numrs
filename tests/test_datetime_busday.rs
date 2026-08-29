//! Regression tests for datetime business-day roll/count semantics and
//! `datetime_as_string` timezone handling (LANE W1-H).
//!
//! All calendar facts below (which date falls on which weekday) are
//! independently verified against the crate's own `weekday()` formula --
//! `days_since_epoch` + `((days+3)%7+7)%7` (Unix epoch 1970-01-01 was a
//! Thursday) -- which matches NumPy's proleptic-Gregorian weekday
//! convention. In particular: 2026-08-14 = Fri, 08-15 = Sat, 08-17 = Mon,
//! 08-18 = Tue, 08-21 = Fri, 08-24 = Mon, 10-30 = Fri, 10-31 = Sat,
//! 11-01 = Sun, 11-02 = Mon.
//!
//! Methodology note: two of these tests (`busday_offset_array` ignoring
//! `roll`, and `datetime_as_string` ignoring `timezone`) were run against
//! the pre-fix code first and observed to fail for exactly the reason
//! described (see the lane report). The `busday_count`-with-holidays and
//! 4-argument `busday_offset` cases are new API surface added by this fix,
//! so their "before" state is "does not compile" rather than a runtime
//! failure -- there is no meaningful red/green split to demonstrate for
//! brand-new parameters.

use numrs2::prelude::*;
use numrs2::types::datetime::array_ops;
use numrs2::types::datetime::business_days;

fn day(s: &str) -> DateTime64 {
    DateTime64::from_iso_string(s, DateTimeUnit::Day).expect("should parse test date")
}

// ===========================================================================
// busday_offset: roll conventions (item 1)
// ===========================================================================

#[test]
fn test_busday_offset_roll_forward_snaps_saturday_to_monday() {
    // 2026-08-15 is a Saturday; roll='forward' must snap to the next
    // business day, Monday 2026-08-17, before applying the (zero) offset.
    let result = business_days::busday_offset(&day("2026-08-15"), 0, Some("forward"), None)
        .expect("should roll forward");
    assert_eq!(result.value(), day("2026-08-17").value());

    // "following" is a documented alias for "forward".
    let alias = business_days::busday_offset(&day("2026-08-15"), 0, Some("following"), None)
        .expect("should roll via alias");
    assert_eq!(alias.value(), day("2026-08-17").value());
}

#[test]
fn test_busday_offset_roll_backward_snaps_saturday_to_friday() {
    let result = business_days::busday_offset(&day("2026-08-15"), 0, Some("backward"), None)
        .expect("should roll backward");
    assert_eq!(result.value(), day("2026-08-14").value());

    // "preceding" is a documented alias for "backward".
    let alias = business_days::busday_offset(&day("2026-08-15"), 0, Some("preceding"), None)
        .expect("should roll via alias");
    assert_eq!(alias.value(), day("2026-08-14").value());
}

#[test]
fn test_busday_offset_modifiedfollowing_crosses_month_rolls_backward() {
    // 2026-10-31 is a Saturday at month end. Rolling forward would land on
    // Monday 2026-11-02, crossing into November, so modifiedfollowing must
    // instead roll backward to Friday 2026-10-30.
    let result =
        business_days::busday_offset(&day("2026-10-31"), 0, Some("modifiedfollowing"), None)
            .expect("should roll modifiedfollowing");
    assert_eq!(result.value(), day("2026-10-30").value());
}

#[test]
fn test_busday_offset_modifiedfollowing_without_crossing_behaves_like_forward() {
    // 2026-08-15 (Saturday) rolling forward lands on Monday 2026-08-17,
    // still within August, so modifiedfollowing should behave exactly like
    // plain forward here (no crossing to correct for).
    let result =
        business_days::busday_offset(&day("2026-08-15"), 0, Some("modifiedfollowing"), None)
            .expect("should roll modifiedfollowing");
    assert_eq!(result.value(), day("2026-08-17").value());
}

#[test]
fn test_busday_offset_modifiedpreceding_crosses_month_rolls_forward() {
    // 2026-11-01 is a Sunday at month start. Rolling backward would land on
    // Friday 2026-10-30, crossing into October, so modifiedpreceding must
    // instead roll forward to Monday 2026-11-02.
    let result =
        business_days::busday_offset(&day("2026-11-01"), 0, Some("modifiedpreceding"), None)
            .expect("should roll modifiedpreceding");
    assert_eq!(result.value(), day("2026-11-02").value());
}

#[test]
fn test_busday_offset_modifiedpreceding_without_crossing_behaves_like_backward() {
    let result =
        business_days::busday_offset(&day("2026-08-15"), 0, Some("modifiedpreceding"), None)
            .expect("should roll modifiedpreceding");
    assert_eq!(result.value(), day("2026-08-14").value());
}

#[test]
fn test_busday_offset_roll_raise_errors_on_non_business_day() {
    // Default roll ('raise' when None) must error rather than silently
    // returning the invalid Saturday unchanged (the original bug).
    assert!(business_days::busday_offset(&day("2026-08-15"), 0, None, None).is_err());
    assert!(business_days::busday_offset(&day("2026-08-15"), 0, Some("raise"), None).is_err());

    // An already-valid business day never errors under 'raise'.
    assert!(business_days::busday_offset(&day("2026-08-17"), 0, Some("raise"), None).is_ok());
}

#[test]
fn test_busday_offset_roll_nat_returns_nat_sentinel() {
    let result = business_days::busday_offset(&day("2026-08-15"), 0, Some("nat"), None)
        .expect("roll='nat' must not error");
    assert!(result.is_nat());

    // Formatting a roll='nat' result end-to-end must produce NumPy's "NaT",
    // not garbage or a panic (DateTime64::to_iso_string / to_unit both
    // guard against the i64::MIN sentinel).
    let formatted = datetime_as_string(&result, None, None).expect("NaT should format, not error");
    assert_eq!(formatted, "NaT");

    // An already-valid business day is never NaT, regardless of roll.
    let valid = business_days::busday_offset(&day("2026-08-17"), 0, Some("nat"), None)
        .expect("Monday is already valid");
    assert!(!valid.is_nat());
}

#[test]
fn test_busday_offset_hops_over_weekends() {
    // Friday + 1 business day skips the weekend to the following Monday.
    let forward = business_days::busday_offset(&day("2026-08-14"), 1, Some("raise"), None)
        .expect("should offset");
    assert_eq!(forward.value(), day("2026-08-17").value());

    // Monday - 1 business day skips the weekend back to the prior Friday.
    let backward = business_days::busday_offset(&day("2026-08-17"), -1, Some("raise"), None)
        .expect("should offset");
    assert_eq!(backward.value(), day("2026-08-14").value());

    // Monday + 5 business days: Tue,Wed,Thu,Fri, then skip Sat/Sun, then
    // the following Monday = 2026-08-24.
    let five = business_days::busday_offset(&day("2026-08-17"), 5, Some("raise"), None)
        .expect("should offset");
    assert_eq!(five.value(), day("2026-08-24").value());

    // offset=0 on an already-valid business day is a no-op.
    let zero = business_days::busday_offset(&day("2026-08-17"), 0, Some("raise"), None)
        .expect("should offset");
    assert_eq!(zero.value(), day("2026-08-17").value());
}

#[test]
fn test_busday_offset_holidays_affect_both_rolling_and_stepping() {
    // 2026-08-21 is a Friday but is marked a holiday: rolling forward from
    // it must skip past the weekend to Monday 2026-08-24, not stop on the
    // (holiday) Friday itself.
    let holidays = [day("2026-08-21")];
    let rolled =
        business_days::busday_offset(&day("2026-08-21"), 0, Some("forward"), Some(&holidays))
            .expect("should roll past the holiday");
    assert_eq!(rolled.value(), day("2026-08-24").value());

    // Stepping: Monday 2026-08-17 + 2 business days would normally land on
    // Wednesday 2026-08-19, but marking Tuesday 2026-08-18 a holiday pushes
    // it to Thursday 2026-08-20.
    let tuesday_holiday = [day("2026-08-18")];
    let stepped =
        business_days::busday_offset(&day("2026-08-17"), 2, Some("raise"), Some(&tuesday_holiday))
            .expect("should step past the holiday");
    assert_eq!(stepped.value(), day("2026-08-20").value());
}

#[test]
fn test_busday_offset_unknown_roll_errors() {
    assert!(business_days::busday_offset(&day("2026-08-17"), 0, Some("sideways"), None).is_err());
}

// ===========================================================================
// busday_offset_array: array-level roll plumbing (item 1, array API)
// ===========================================================================

#[test]
fn test_busday_offset_array_applies_roll_per_element() {
    let dts = Array::from_vec(vec![
        day("2026-08-15"),
        day("2026-08-15"),
        day("2026-08-17"),
    ]);
    let offsets = Array::from_vec(vec![0i32, 0i32, 1i32]);

    let forward =
        array_ops::busday_offset_array(&dts, &offsets, Some("forward")).expect("should not error");
    let forward_vals: Vec<i64> = forward.to_vec().iter().map(|d| d.value()).collect();
    assert_eq!(
        forward_vals,
        vec![
            day("2026-08-17").value(), // Sat rolled forward -> Mon
            day("2026-08-17").value(), // Sat rolled forward -> Mon
            day("2026-08-18").value(), // Mon (valid) + 1 business day -> Tue
        ]
    );

    let backward = array_ops::busday_offset_array(
        &Array::from_vec(vec![day("2026-08-15")]),
        &Array::from_vec(vec![0i32]),
        Some("backward"),
    )
    .expect("should not error");
    assert_eq!(backward.to_vec()[0].value(), day("2026-08-14").value());
}

#[test]
fn test_busday_offset_array_length_mismatch_errors() {
    let dts = Array::from_vec(vec![day("2026-08-17")]);
    let offsets = Array::from_vec(vec![0i32, 1i32]);
    assert!(array_ops::busday_offset_array(&dts, &offsets, Some("raise")).is_err());
}

// ===========================================================================
// busday_count (item 3)
// ===========================================================================

#[test]
fn test_busday_count_monday_to_friday_same_week_is_four() {
    // [Mon 2026-08-17, Fri 2026-08-21) = Mon,Tue,Wed,Thu = 4 business days;
    // Friday itself is excluded (half-open interval).
    let count = business_days::busday_count(&day("2026-08-17"), &day("2026-08-21"), None)
        .expect("should count");
    assert_eq!(count, 4);
}

#[test]
fn test_busday_count_over_a_weekend() {
    // [Fri 2026-08-14, Tue 2026-08-18) = Fri,Sat,Sun,Mon; weekend excluded,
    // so only Fri and Mon count = 2.
    let count = business_days::busday_count(&day("2026-08-14"), &day("2026-08-18"), None)
        .expect("should count");
    assert_eq!(count, 2);
}

#[test]
fn test_busday_count_reversed_is_negative() {
    let forward = business_days::busday_count(&day("2026-08-17"), &day("2026-08-21"), None)
        .expect("should count");
    let reversed = business_days::busday_count(&day("2026-08-21"), &day("2026-08-17"), None)
        .expect("should count reversed");
    assert_eq!(reversed, -forward);
    assert_eq!(reversed, -4);
}

#[test]
fn test_busday_count_with_holiday_excluded() {
    // Same Mon->Fri range as above (4 business days), but with Wednesday
    // 2026-08-19 marked a holiday, it must be excluded, leaving 3.
    let holidays = [day("2026-08-19")];
    let count =
        business_days::busday_count(&day("2026-08-17"), &day("2026-08-21"), Some(&holidays))
            .expect("should count excluding holiday");
    assert_eq!(count, 3);

    // A holiday that falls on a weekend changes nothing (it was already
    // excluded).
    let weekend_holiday = [day("2026-08-15")]; // Saturday
    let count_weekend_holiday = business_days::busday_count(
        &day("2026-08-17"),
        &day("2026-08-21"),
        Some(&weekend_holiday),
    )
    .expect("should count");
    assert_eq!(count_weekend_holiday, 4);
}

#[test]
fn test_busday_count_nat_endpoint_errors() {
    let nat = DateTime64::nat(DateTimeUnit::Day);
    assert!(business_days::busday_count(&nat, &day("2026-08-21"), None).is_err());
    assert!(business_days::busday_count(&day("2026-08-17"), &nat, None).is_err());
}

#[test]
fn test_busday_count_array_matches_scalar_per_element() {
    let begins = Array::from_vec(vec![day("2026-08-17"), day("2026-08-21")]);
    let ends = Array::from_vec(vec![day("2026-08-21"), day("2026-08-17")]);
    let counts = array_ops::busday_count_array(&begins, &ends).expect("should count array");
    assert_eq!(counts.to_vec(), vec![4i64, -4i64]);
}

// ===========================================================================
// datetime_as_string: timezone handling (item 2)
// ===========================================================================
// (Naive/UTC/NaT are also covered as unit tests colocated in api.rs; these
// integration tests focus on cross-module and portability concerns.)

#[test]
fn test_datetime_as_string_naive_default_matches_explicit_naive() {
    let dt = datetime64("2023-06-15T08:09:10", Some("s")).expect("should parse");
    let via_none = datetime_as_string(&dt, None, None).expect("naive via None");
    let via_explicit = datetime_as_string(&dt, None, Some("naive")).expect("naive explicit");
    assert_eq!(via_none, via_explicit);
    assert_eq!(via_none, "2023-06-15T08:09:10");
}

#[test]
fn test_datetime_as_string_utc_vs_naive_differ_only_by_suffix() {
    let dt = datetime64("2023-06-15T08:09:10", Some("s")).expect("should parse");
    let naive = datetime_as_string(&dt, None, None).expect("naive");
    let utc = datetime_as_string(&dt, None, Some("UTC")).expect("utc");
    assert_eq!(format!("{naive}Z"), utc);
}

#[test]
fn test_datetime_as_string_local_matches_independently_computed_chrono_offset() {
    // Portable across whatever timezone the host machine/CI runner is in
    // (including UTC itself, where the offset is legitimately +0000):
    // recompute the expected local offset independently via chrono and
    // check the function's output carries the matching suffix, without
    // asserting on the (offset-dependent, possibly day-shifted) date/time
    // portion.
    use chrono::{DateTime as ChronoDateTime, Local, Utc};

    let dt = datetime64("2023-06-15T12:00:00", Some("s")).expect("should parse");
    let s = datetime_as_string(&dt, None, Some("local")).expect("should format local");

    let utc_dt: ChronoDateTime<Utc> = ChronoDateTime::from_timestamp(
        DateTime64::from_iso_string("2023-06-15T12:00:00", DateTimeUnit::Second)
            .expect("should parse reference instant")
            .value(),
        0,
    )
    .expect("valid timestamp");
    let local_dt = utc_dt.with_timezone(&Local);
    let offset_seconds = local_dt.offset().local_minus_utc();

    let sign = if offset_seconds >= 0 { '+' } else { '-' };
    let abs_secs = offset_seconds.unsigned_abs();
    let expected_suffix = format!("{sign}{:02}{:02}", abs_secs / 3600, (abs_secs % 3600) / 60);

    assert!(
        s.ends_with(&expected_suffix),
        "expected suffix '{expected_suffix}' at the end of '{s}'"
    );
    assert!(
        !s.contains('Z'),
        "local-formatted string must not contain Z: {s}"
    );
}

#[test]
fn test_datetime_as_string_unknown_timezone_is_an_error() {
    let dt = datetime64("2023-01-01", Some("D")).expect("should parse");
    assert!(datetime_as_string(&dt, None, Some("not-a-timezone")).is_err());
}
