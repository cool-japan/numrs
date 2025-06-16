//! Integration test for enhanced datetime functionality

use numrs2::prelude::*;

#[test]
fn test_datetime_basic_functionality() {
    // Test basic DateTime64 creation
    let dt = DateTime64::new(1000, DateTimeUnit::Second);
    assert_eq!(dt.value(), 1000);
    assert_eq!(dt.unit(), DateTimeUnit::Second);
}

#[test]
fn test_datetime_parsing() {
    // Test ISO string parsing
    let dt = DateTime64::from_iso_string("2023-12-25T15:30:45", DateTimeUnit::Second).unwrap();
    let iso_str = dt.to_iso_string().unwrap();
    assert!(iso_str.contains("2023-12-25T15:30:45"));
}

#[test]
fn test_timezone_functionality() {
    // Test timezone creation
    let utc = Timezone::utc();
    assert_eq!(utc.offset_minutes, 0);
    
    let est = Timezone::est();
    assert_eq!(est.offset_minutes, -300);
    
    // Test timezone-aware datetime
    let utc_dt = DateTime64::from_iso_string("2023-01-01T12:00:00", DateTimeUnit::Second).unwrap();
    let tz_dt = TimezoneDateTime::new(utc_dt, est);
    
    let local_dt = tz_dt.to_local();
    let local_str = local_dt.to_iso_string().unwrap();
    assert!(local_str.starts_with("2023-01-01T07:00:00"));
}

#[test]
fn test_business_days() {
    // Test business day functionality
    let dt = DateTime64::from_iso_string("2023-01-02", DateTimeUnit::Day).unwrap(); // Monday
    assert!(business_days::is_busday(&dt).unwrap());
    
    let weekend = DateTime64::from_iso_string("2023-01-01", DateTimeUnit::Day).unwrap(); // Sunday
    assert!(!business_days::is_busday(&weekend).unwrap());
}

#[test]
fn test_date_range() {
    // Test date range generation
    let range = datetime_array::date_range(
        "2023-01-01", 
        Some("2023-01-05"), 
        None, 
        DateTimeUnit::Day,
        DateTimeUnit::Day
    ).unwrap();
    
    assert!(range.size() >= 4);
}