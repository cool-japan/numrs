//! Integration tests for `CustomDType` / `TypeDescriptor` serialization.
//!
//! `src/types/custom.rs` used to have `to_bytes`/`parse_bytes`
//! implementations that were pure placeholders: `to_bytes` always returned
//! `vec![0; size]` and `parse_bytes` always returned `T::default()`,
//! silently discarding every value passed through a round trip. These
//! tests exercise the real `CustomDTypeCodec`-backed implementation (via
//! `oxicode`, the COOLJAPAN pure-Rust replacement for `bincode`) from
//! outside the crate, the way a downstream consumer would use it.

use numrs2::prelude::*;
use numrs2::types::custom::{CustomDTypeCodec, TypeDescriptor};
use serde::{Deserialize, Serialize};
use std::any::Any;

/// A user-defined struct with heterogeneous fields, standing in for an
/// arbitrary custom dtype a downstream crate might register.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Point3D {
    x: f64,
    y: f64,
    z: i32,
}

/// A second, differently-shaped custom type, to make sure the codec isn't
/// accidentally hardcoded to one struct's layout.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Label {
    tag: String,
    active: bool,
}

#[test]
fn test_custom_dtype_round_trip_preserves_struct_value() {
    let dtype = CustomDType::<Point3D>::new("Point3D", 20, false);
    let original = Point3D {
        x: 1.5,
        y: -2.25,
        z: 42,
    };

    let boxed: Box<dyn Any> = Box::new(original.clone());
    let bytes = dtype
        .to_bytes(boxed.as_ref())
        .expect("to_bytes should encode a matching value");
    let parsed = dtype
        .parse_bytes(&bytes)
        .expect("parse_bytes should decode well-formed bytes");
    let parsed: &Point3D = parsed
        .downcast_ref()
        .expect("parsed value should downcast back to Point3D");

    assert_eq!(parsed, &original);
}

#[test]
fn test_custom_dtype_round_trip_preserves_string_and_bool() {
    let dtype = CustomDType::<Label>::new("Label", 24, false);
    let original = Label {
        tag: "release-candidate".to_string(),
        active: true,
    };

    let boxed: Box<dyn Any> = Box::new(original.clone());
    let bytes = dtype
        .to_bytes(boxed.as_ref())
        .expect("to_bytes should encode a matching value");
    let parsed = dtype
        .parse_bytes(&bytes)
        .expect("parse_bytes should decode well-formed bytes");
    let parsed: &Label = parsed
        .downcast_ref()
        .expect("parsed value should downcast back to Label");

    assert_eq!(parsed, &original);
}

#[test]
fn test_custom_dtype_codec_trait_direct_round_trip() {
    // Exercise `CustomDTypeCodec` directly, without going through the
    // `dyn Any`-erased `TypeDescriptor` interface.
    let original = Point3D {
        x: 0.0,
        y: 12.34567,
        z: -7,
    };
    let bytes = original.to_bytes().expect("encode should succeed");
    let decoded = Point3D::from_bytes(&bytes).expect("decode should succeed");
    assert_eq!(original, decoded);
}

#[test]
fn test_custom_dtype_distinct_values_do_not_collide() {
    // Guards against a to_bytes/parse_bytes implementation that ignores its
    // input: two different values must decode to two different results.
    let dtype = CustomDType::<Point3D>::new("Point3D", 20, false);
    let a = Point3D {
        x: 1.0,
        y: 2.0,
        z: 3,
    };
    let b = Point3D {
        x: 4.0,
        y: 5.0,
        z: 6,
    };
    assert_ne!(a, b);

    let a_boxed: Box<dyn Any> = Box::new(a.clone());
    let b_boxed: Box<dyn Any> = Box::new(b.clone());
    let a_bytes = dtype.to_bytes(a_boxed.as_ref()).unwrap();
    let b_bytes = dtype.to_bytes(b_boxed.as_ref()).unwrap();
    assert_ne!(a_bytes, b_bytes);

    let a_back = dtype.parse_bytes(&a_bytes).unwrap();
    let b_back = dtype.parse_bytes(&b_bytes).unwrap();
    assert_eq!(a_back.downcast_ref::<Point3D>().unwrap(), &a);
    assert_eq!(b_back.downcast_ref::<Point3D>().unwrap(), &b);
}

#[test]
fn test_custom_dtype_wrong_size_bytes_errors_not_default() {
    let dtype = CustomDType::<Point3D>::new("Point3D", 20, false);
    let original = Point3D {
        x: 9.0,
        y: 9.0,
        z: 9,
    };

    // A truncated buffer must error, never silently decode to
    // `Point3D::default()`.
    let mut too_short = original.to_bytes().expect("encode should succeed");
    too_short.pop();
    let result = dtype.parse_bytes(&too_short);
    assert!(
        result.is_err(),
        "truncated bytes must not decode into a default value"
    );

    // Extra trailing bytes must also error, not be silently ignored.
    let mut too_long = original.to_bytes().expect("encode should succeed");
    too_long.extend_from_slice(&[0xAB, 0xCD, 0xEF]);
    let result = dtype.parse_bytes(&too_long);
    assert!(
        result.is_err(),
        "trailing garbage bytes must not decode silently"
    );

    // Completely empty input must error too.
    let result = dtype.parse_bytes(&[]);
    assert!(result.is_err(), "empty bytes must not decode to a default");
}

#[test]
fn test_custom_dtype_to_bytes_rejects_mismatched_any_value() {
    let dtype = CustomDType::<Point3D>::new("Point3D", 20, false);
    let wrong: Box<dyn Any> = Box::new(String::from("not a Point3D"));
    let result = dtype.to_bytes(wrong.as_ref());
    assert!(
        result.is_err(),
        "to_bytes must reject a value of the wrong underlying type"
    );
}

#[test]
fn test_custom_dtype_metadata_and_default_value() {
    let dtype = CustomDType::<Point3D>::new("Point3D", 20, false);
    assert_eq!(dtype.name(), "Point3D");
    assert_eq!(dtype.size_in_bytes(), 20);
    assert!(!dtype.is_numeric());

    let default = dtype.default_value();
    let default: &Point3D = default
        .downcast_ref()
        .expect("default_value should downcast to Point3D");
    assert_eq!(default, &Point3D::default());
}
