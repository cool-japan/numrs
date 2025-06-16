//! Integration tests for enhanced structured array functionality

use numrs2::prelude::*;
use std::collections::HashMap;

#[test]
fn test_structured_array_creation() {
    // Test basic structured array creation
    let fields = vec![
        Field::new("x", DType::Float64),
        Field::new("y", DType::Int32),
        Field::new("name", DType::String(20)),
    ];
    let dtype = DType::Struct(fields);
    let shape = [3, 2];
    let arr = StructuredArray::new(&shape, dtype);

    assert_eq!(arr.shape(), &[3, 2]);
    assert_eq!(arr.size(), 6);
    assert_eq!(arr.ndim(), 2);
}

#[test]
fn test_structured_array_field_access() {
    // Create a structured array with numeric fields
    let fields = vec![
        Field::new("x", DType::Float64),
        Field::new("y", DType::Int32),
    ];
    let dtype = DType::Struct(fields);
    let shape = [2, 2];
    let mut arr = StructuredArray::new(&shape, dtype);

    // Set some values
    arr.set_field(&[0, 0], "x", 1.5f64).unwrap();
    arr.set_field(&[0, 0], "y", 42i32).unwrap();
    arr.set_field(&[1, 1], "x", 2.7f64).unwrap();
    arr.set_field(&[1, 1], "y", 84i32).unwrap();

    // Get field arrays
    let x_field: Array<f64> = arr.field("x").unwrap();
    let y_field: Array<i32> = arr.field("y").unwrap();

    assert_eq!(x_field.shape(), &[2, 2]);
    assert_eq!(y_field.shape(), &[2, 2]);

    // Check values by accessing the underlying ndarray
    assert_eq!(x_field.array()[[0, 0]], 1.5f64);
    assert_eq!(y_field.array()[[0, 0]], 42i32);
    assert_eq!(x_field.array()[[1, 1]], 2.7f64);
    assert_eq!(y_field.array()[[1, 1]], 84i32);
}

#[test]
fn test_structured_array_string_fields() {
    // Test structured array with string fields
    let fields = vec![
        Field::new("name", DType::String(10)),
        Field::new("value", DType::Float64),
    ];
    let dtype = DType::Struct(fields);
    let shape = [2];
    let mut arr = StructuredArray::new(&shape, dtype);

    // Set string and numeric values
    arr.set_field(&[0], "name", "Alice".to_string()).unwrap();
    arr.set_field(&[0], "value", 3.14f64).unwrap();
    arr.set_field(&[1], "name", "Bob".to_string()).unwrap();
    arr.set_field(&[1], "value", 2.71f64).unwrap();

    // Get field arrays
    let name_field: Array<String> = arr.field("name").unwrap();
    let value_field: Array<f64> = arr.field("value").unwrap();

    // Check values by accessing the underlying ndarray
    assert_eq!(name_field.array()[[0]], "Alice");
    assert_eq!(value_field.array()[[0]], 3.14f64);
    assert_eq!(name_field.array()[[1]], "Bob");
    assert_eq!(value_field.array()[[1]], 2.71f64);
}

#[test]
fn test_structured_array_complex_fields() {
    // Test structured array with complex number fields
    let fields = vec![
        Field::new("complex32", DType::Complex32),
        Field::new("complex64", DType::Complex64),
    ];
    let dtype = DType::Struct(fields);
    let shape = [2];
    let mut arr = StructuredArray::new(&shape, dtype);

    // Set complex values
    let c32 = Complex::new(1.0f32, 2.0f32);
    let c64 = Complex::new(3.0f64, 4.0f64);

    arr.set_field(&[0], "complex32", c32).unwrap();
    arr.set_field(&[0], "complex64", c64).unwrap();

    // Get field arrays
    let c32_field: Array<Complex<f32>> = arr.field("complex32").unwrap();
    let c64_field: Array<Complex<f64>> = arr.field("complex64").unwrap();

    // Check values by accessing the underlying ndarray
    assert_eq!(c32_field.array()[[0]], c32);
    assert_eq!(c64_field.array()[[0]], c64);
}

#[test]
fn test_record_array_creation() {
    // Test record array creation and field access
    let fields = vec![
        Field::new("x", DType::Float64),
        Field::new("y", DType::Float64),
        Field::new("z", DType::Float64),
    ];
    let shape = [3];
    let mut record = RecordArray::new(&shape, fields);

    assert_eq!(record.shape(), &[3]);
    assert_eq!(record.size(), 3);
    assert_eq!(record.ndim(), 1);

    // Set field values
    record.set_field(&[0], "x", 1.0).unwrap();
    record.set_field(&[0], "y", 2.0).unwrap();
    record.set_field(&[0], "z", 3.0).unwrap();

    record.set_field(&[1], "x", 4.0).unwrap();
    record.set_field(&[1], "y", 5.0).unwrap();
    record.set_field(&[1], "z", 6.0).unwrap();

    // Check field names
    let field_names = record.field_names();
    assert!(field_names.contains(&"x".to_string()));
    assert!(field_names.contains(&"y".to_string()));
    assert!(field_names.contains(&"z".to_string()));
}

#[test]
fn test_record_array_from_arrays() {
    // Create individual arrays
    let x_data = vec![1.0, 2.0, 3.0];
    let y_data = vec![4.0, 5.0, 6.0];
    let z_data = vec![7.0, 8.0, 9.0];

    let x_array = Array::from_vec(x_data);
    let y_array = Array::from_vec(y_data);
    let z_array = Array::from_vec(z_data);

    // Create arrays map
    let mut arrays = HashMap::new();
    arrays.insert("x".to_string(), x_array);
    arrays.insert("y".to_string(), y_array);
    arrays.insert("z".to_string(), z_array);

    // Create record array from individual arrays
    let shape = [3];
    let record = RecordArray::from_arrays(&arrays, &shape).unwrap();

    // Check field access
    let x_field = record.field("x").unwrap();
    let y_field = record.field("y").unwrap();
    let z_field = record.field("z").unwrap();

    assert_eq!(x_field.array()[[0]], 1.0);
    assert_eq!(y_field.array()[[1]], 5.0);
    assert_eq!(z_field.array()[[2]], 9.0);
}

#[test]
fn test_record_array_field_management() {
    // Test adding and removing fields from record arrays
    let fields = vec![
        Field::new("x", DType::Float64),
        Field::new("y", DType::Float64),
    ];
    let shape = [2];
    let mut record = RecordArray::new(&shape, fields);

    // Set initial values
    record.set_field(&[0], "x", 1.0).unwrap();
    record.set_field(&[0], "y", 2.0).unwrap();
    record.set_field(&[1], "x", 3.0).unwrap();
    record.set_field(&[1], "y", 4.0).unwrap();

    // Add a new field
    let z_data = vec![5.0, 6.0];
    let z_array = Array::from_vec(z_data);
    record.add_field("z", z_array).unwrap();

    // Check that the new field was added
    let field_names = record.field_names();
    assert!(field_names.contains(&"z".to_string()));

    let z_field = record.field("z").unwrap();
    assert_eq!(z_field.array()[[0]], 5.0);
    assert_eq!(z_field.array()[[1]], 6.0);

    // Remove a field
    let removed_field = record.remove_field("y").unwrap();
    assert_eq!(removed_field.array()[[0]], 2.0);
    assert_eq!(removed_field.array()[[1]], 4.0);

    // Check that the field was removed
    let field_names = record.field_names();
    assert!(!field_names.contains(&"y".to_string()));
    assert!(field_names.contains(&"x".to_string()));
    assert!(field_names.contains(&"z".to_string()));
}

#[test]
fn test_dtype_properties() {
    // Test DType utility methods
    assert!(DType::Float64.is_numeric());
    assert!(DType::Float64.is_floating_point());
    assert!(DType::Complex64.is_complex());
    assert!(DType::String(10).is_string());

    let fields = vec![Field::new("x", DType::Int32)];
    let struct_dtype = DType::Struct(fields);
    assert!(struct_dtype.is_struct());
    assert!(!struct_dtype.is_numeric());

    // Test size calculations
    assert_eq!(DType::Float64.size_in_bytes(), 8);
    assert_eq!(DType::Int32.size_in_bytes(), 4);
    assert_eq!(DType::String(20).size_in_bytes(), 20);
    assert_eq!(DType::Complex64.size_in_bytes(), 16);
}

#[test]
fn test_error_handling() {
    // Test error conditions for structured arrays
    let fields = vec![Field::new("x", DType::Float64)];
    let dtype = DType::Struct(fields);
    let shape = [2, 2];
    let mut arr = StructuredArray::new(&shape, dtype);

    // Test invalid field name
    let result = arr.field::<f64>("nonexistent");
    assert!(result.is_err());

    // Test invalid index
    let result = arr.set_field(&[2, 2], "x", 1.0f64);
    assert!(result.is_err());

    // Test dimension mismatch
    let result = arr.set_field(&[0], "x", 1.0f64);
    assert!(result.is_err());
}

#[test]
fn test_multiple_data_types() {
    // Test structured array with various data types
    let fields = vec![
        Field::new("bool_field", DType::Bool),
        Field::new("int8_field", DType::Int8),
        Field::new("int16_field", DType::Int16),
        Field::new("int32_field", DType::Int32),
        Field::new("int64_field", DType::Int64),
        Field::new("uint8_field", DType::UInt8),
        Field::new("uint16_field", DType::UInt16),
        Field::new("uint32_field", DType::UInt32),
        Field::new("uint64_field", DType::UInt64),
        Field::new("float32_field", DType::Float32),
        Field::new("float64_field", DType::Float64),
    ];
    let dtype = DType::Struct(fields);
    let shape = [1];
    let mut arr = StructuredArray::new(&shape, dtype);

    // Set values for each type
    arr.set_field(&[0], "bool_field", true).unwrap();
    arr.set_field(&[0], "int8_field", 42i8).unwrap();
    arr.set_field(&[0], "int16_field", 1234i16).unwrap();
    arr.set_field(&[0], "int32_field", 123456i32).unwrap();
    arr.set_field(&[0], "int64_field", 123456789i64).unwrap();
    arr.set_field(&[0], "uint8_field", 255u8).unwrap();
    arr.set_field(&[0], "uint16_field", 65535u16).unwrap();
    arr.set_field(&[0], "uint32_field", 4294967295u32).unwrap();
    arr.set_field(&[0], "uint64_field", 18446744073709551615u64)
        .unwrap();
    arr.set_field(&[0], "float32_field", 3.14f32).unwrap();
    arr.set_field(&[0], "float64_field", 2.71828f64).unwrap();

    // Get field arrays and verify values
    let bool_field: Array<bool> = arr.field("bool_field").unwrap();
    assert_eq!(bool_field.array()[[0]], true);

    let int8_field: Array<i8> = arr.field("int8_field").unwrap();
    assert_eq!(int8_field.array()[[0]], 42i8);

    let float64_field: Array<f64> = arr.field("float64_field").unwrap();
    assert_eq!(float64_field.array()[[0]], 2.71828f64);
}
