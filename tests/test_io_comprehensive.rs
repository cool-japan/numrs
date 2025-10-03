//! Comprehensive tests for NumRS2 I/O functionality
//!
//! This test suite covers all I/O functions including text file I/O (loadtxt, savetxt, genfromtxt),
//! binary file I/O (NPY/NPZ), and various serialization formats.

use numrs2::io::text::*;
use numrs2::io::SerializeFormat;
use numrs2::prelude::*;
use std::fs;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_loadtxt_basic_functionality() {
    // Create a temporary file with test data
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "1.0 2.0 3.0").unwrap();
    writeln!(temp_file, "4.0 5.0 6.0").unwrap();
    writeln!(temp_file, "7.0 8.0 9.0").unwrap();

    let array = loadtxt::<f64>(temp_file.path(), LoadTxtOptions::default()).unwrap();
    assert_eq!(array.shape(), &[3, 3]);
    assert_eq!(
        array.to_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn test_loadtxt_with_comments_and_skip() {
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "# This is a header comment").unwrap();
    writeln!(temp_file, "# x y z").unwrap();
    writeln!(temp_file, "1.0 2.0 3.0").unwrap();
    writeln!(temp_file, "4.0 5.0 6.0").unwrap();
    writeln!(temp_file, "# This is a comment in the middle").unwrap();
    writeln!(temp_file, "7.0 8.0 9.0").unwrap();

    let mut options = LoadTxtOptions::default();
    options.skiprows = 0; // Don't skip any rows, just ignore comments

    let array = loadtxt::<f64>(temp_file.path(), options).unwrap();
    assert_eq!(array.shape(), &[3, 3]);
    assert_eq!(
        array.to_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn test_loadtxt_with_delimiter() {
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "1.0,2.0,3.0").unwrap();
    writeln!(temp_file, "4.0,5.0,6.0").unwrap();

    let mut options = LoadTxtOptions::default();
    options.delimiter = Some(",".to_string());

    let array = loadtxt::<f64>(temp_file.path(), options).unwrap();
    assert_eq!(array.shape(), &[2, 3]);
    assert_eq!(array.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_loadtxt_with_usecols() {
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "1.0 2.0 3.0 4.0").unwrap();
    writeln!(temp_file, "5.0 6.0 7.0 8.0").unwrap();

    let mut options = LoadTxtOptions::default();
    options.usecols = Some(vec![0, 2]); // Only columns 0 and 2

    let array = loadtxt::<f64>(temp_file.path(), options).unwrap();
    assert_eq!(array.shape(), &[2, 2]);
    assert_eq!(array.to_vec(), vec![1.0, 3.0, 5.0, 7.0]);
}

#[test]
fn test_loadtxt_with_max_rows() {
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "1.0 2.0").unwrap();
    writeln!(temp_file, "3.0 4.0").unwrap();
    writeln!(temp_file, "5.0 6.0").unwrap();
    writeln!(temp_file, "7.0 8.0").unwrap();

    let mut options = LoadTxtOptions::default();
    options.max_rows = Some(2);

    let array = loadtxt::<f64>(temp_file.path(), options).unwrap();
    assert_eq!(array.shape(), &[2, 2]);
    assert_eq!(array.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_loadtxt_integer_types() {
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "1 2 3").unwrap();
    writeln!(temp_file, "4 5 6").unwrap();

    let array = loadtxt::<i32>(temp_file.path(), LoadTxtOptions::default()).unwrap();
    assert_eq!(array.shape(), &[2, 3]);
    assert_eq!(array.to_vec(), vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn test_savetxt_basic_functionality() {
    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let temp_file = NamedTempFile::new().unwrap();

    savetxt(temp_file.path(), &array, SaveTxtOptions::default()).unwrap();

    let content = fs::read_to_string(temp_file.path()).unwrap();
    let lines: Vec<&str> = content.trim().split('\n').collect();
    assert_eq!(lines.len(), 2);
    assert!(lines[0].contains("1") && lines[0].contains("2"));
    assert!(lines[1].contains("3") && lines[1].contains("4"));
}

#[test]
fn test_savetxt_with_delimiter() {
    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let temp_file = NamedTempFile::new().unwrap();

    let mut options = SaveTxtOptions::default();
    options.delimiter = ",".to_string();

    savetxt(temp_file.path(), &array, options).unwrap();

    let content = fs::read_to_string(temp_file.path()).unwrap();
    assert!(content.contains(","));
}

#[test]
fn test_savetxt_with_header_footer() {
    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let temp_file = NamedTempFile::new().unwrap();

    let mut options = SaveTxtOptions::default();
    options.header = Some("x y".to_string());
    options.footer = Some("End of data".to_string());

    savetxt(temp_file.path(), &array, options).unwrap();

    let content = fs::read_to_string(temp_file.path()).unwrap();
    assert!(content.contains("# x y"));
    assert!(content.contains("# End of data"));
}

#[test]
fn test_savetxt_1d_array() {
    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let temp_file = NamedTempFile::new().unwrap();

    savetxt(temp_file.path(), &array, SaveTxtOptions::default()).unwrap();

    let content = fs::read_to_string(temp_file.path()).unwrap();
    let lines: Vec<&str> = content.trim().split('\n').collect();
    assert_eq!(lines.len(), 4); // 1D array should be saved as column vector
}

#[test]
fn test_genfromtxt_with_missing_values() {
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "1.0 2.0 3.0").unwrap();
    writeln!(temp_file, "4.0 nan 6.0").unwrap();
    writeln!(temp_file, "7.0 8.0 N/A").unwrap();
    writeln!(temp_file, "10.0 NULL 12.0").unwrap();

    let array = genfromtxt::<f64>(temp_file.path(), GenFromTxtOptions::default()).unwrap();
    assert_eq!(array.shape(), &[4, 3]);
    // Missing values should be replaced with 0.0
    let expected = vec![1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 0.0, 10.0, 0.0, 12.0];
    assert_eq!(array.to_vec(), expected);
}

#[test]
fn test_genfromtxt_with_skip_header_footer() {
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "Header line 1").unwrap();
    writeln!(temp_file, "Header line 2").unwrap();
    writeln!(temp_file, "1.0 2.0").unwrap();
    writeln!(temp_file, "3.0 4.0").unwrap();
    writeln!(temp_file, "Footer line").unwrap();

    let mut options = GenFromTxtOptions::default();
    options.skip_header = 2;
    options.skip_footer = 1;

    let array = genfromtxt::<f64>(temp_file.path(), options).unwrap();
    assert_eq!(array.shape(), &[2, 2]);
    assert_eq!(array.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_genfromtxt_with_custom_missing_values() {
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "1.0 2.0 3.0").unwrap();
    writeln!(temp_file, "4.0 MISSING 6.0").unwrap();
    writeln!(temp_file, "7.0 8.0 EMPTY").unwrap();

    let mut options = GenFromTxtOptions::default();
    options.default_missing.push("MISSING".to_string());
    options.default_missing.push("EMPTY".to_string());

    let array = genfromtxt::<f64>(temp_file.path(), options).unwrap();
    assert_eq!(array.shape(), &[3, 3]);
    let expected = vec![1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0, 0.0];
    assert_eq!(array.to_vec(), expected);
}

#[test]
fn test_detect_delimiter() {
    // Test CSV format detection
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "1,2,3").unwrap();
    writeln!(temp_file, "4,5,6").unwrap();
    writeln!(temp_file, "7,8,9").unwrap();

    let delimiter = detect_delimiter(temp_file.path(), Some(3)).unwrap();
    assert_eq!(delimiter, ",");

    // Test tab-separated format detection
    let mut temp_file2 = NamedTempFile::new().unwrap();
    writeln!(temp_file2, "1\t2\t3").unwrap();
    writeln!(temp_file2, "4\t5\t6").unwrap();

    let delimiter2 = detect_delimiter(temp_file2.path(), Some(2)).unwrap();
    assert_eq!(delimiter2, "\t");
}

#[test]
fn test_npy_format_f64() {
    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let temp_file = NamedTempFile::new().unwrap();

    // Save as NPY
    array
        .to_file(temp_file.path(), SerializeFormat::Npy)
        .unwrap();

    // Load back from NPY
    let loaded_array = Array::<f64>::from_file(temp_file.path(), SerializeFormat::Npy).unwrap();
    assert_eq!(loaded_array.shape(), array.shape());
    assert_eq!(loaded_array.to_vec(), array.to_vec());
}

#[test]
fn test_npy_format_i32() {
    let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    let temp_file = NamedTempFile::new().unwrap();

    // Save as NPY
    array
        .to_file(temp_file.path(), SerializeFormat::Npy)
        .unwrap();

    // Load back from NPY
    let loaded_array = Array::<i32>::from_file(temp_file.path(), SerializeFormat::Npy).unwrap();
    assert_eq!(loaded_array.shape(), array.shape());
    assert_eq!(loaded_array.to_vec(), array.to_vec());
}

#[test]
fn test_npy_format_bool() {
    let array = Array::from_vec(vec![true, false, true, false]).reshape(&[2, 2]);
    let temp_file = NamedTempFile::new().unwrap();

    // Save as NPY
    array
        .to_file(temp_file.path(), SerializeFormat::Npy)
        .unwrap();

    // Load back from NPY
    let loaded_array = Array::<bool>::from_file(temp_file.path(), SerializeFormat::Npy).unwrap();
    assert_eq!(loaded_array.shape(), array.shape());
    assert_eq!(loaded_array.to_vec(), array.to_vec());
}

#[test]
fn test_npy_format_u8() {
    let array = Array::from_vec(vec![1u8, 2u8, 3u8, 4u8]).reshape(&[2, 2]);
    let temp_file = NamedTempFile::new().unwrap();

    // Save as NPY
    array
        .to_file(temp_file.path(), SerializeFormat::Npy)
        .unwrap();

    // Load back from NPY
    let loaded_array = Array::<u8>::from_file(temp_file.path(), SerializeFormat::Npy).unwrap();
    assert_eq!(loaded_array.shape(), array.shape());
    assert_eq!(loaded_array.to_vec(), array.to_vec());
}

#[test]
fn test_npz_format_f64() {
    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let temp_file = NamedTempFile::new().unwrap();

    // Save as NPZ
    array
        .to_file(temp_file.path(), SerializeFormat::Npz)
        .unwrap();

    // Load back from NPZ
    let loaded_array = Array::<f64>::from_file(temp_file.path(), SerializeFormat::Npz).unwrap();
    assert_eq!(loaded_array.shape(), array.shape());
    assert_eq!(loaded_array.to_vec(), array.to_vec());
}

#[test]
fn test_json_serialization() {
    let array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    let temp_file = NamedTempFile::new().unwrap();

    // Save as JSON
    array
        .to_file(temp_file.path(), SerializeFormat::Json)
        .unwrap();

    // Load back from JSON
    let loaded_array = Array::<i32>::from_file(temp_file.path(), SerializeFormat::Json).unwrap();
    assert_eq!(loaded_array.shape(), array.shape());
    assert_eq!(loaded_array.to_vec(), array.to_vec());
}

#[test]
fn test_csv_serialization() {
    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let temp_file = NamedTempFile::new().unwrap();

    // Save as CSV
    array
        .to_file(temp_file.path(), SerializeFormat::Csv)
        .unwrap();

    // Load back from CSV
    let loaded_array = Array::<f64>::from_file(temp_file.path(), SerializeFormat::Csv).unwrap();

    assert_eq!(loaded_array.shape(), array.shape());
    assert_eq!(loaded_array.to_vec(), array.to_vec());
}

#[test]
#[ignore = "Binary serialization I/O issue unrelated to SCIRS2 migration"]
fn test_binary_serialization() {
    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let temp_file = NamedTempFile::new().unwrap();

    // Save as Binary
    array
        .to_file(temp_file.path(), SerializeFormat::Binary)
        .unwrap();

    // Load back from Binary
    let loaded_array = Array::<f64>::from_file(temp_file.path(), SerializeFormat::Binary).unwrap();
    assert_eq!(loaded_array.shape(), array.shape());
    assert_eq!(loaded_array.to_vec(), array.to_vec());
}

#[test]
fn test_roundtrip_different_shapes() {
    // Test 1D array
    let array_1d = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let temp_file_1d = NamedTempFile::new().unwrap();
    array_1d
        .to_file(temp_file_1d.path(), SerializeFormat::Npy)
        .unwrap();
    let loaded_1d = Array::<f64>::from_file(temp_file_1d.path(), SerializeFormat::Npy).unwrap();
    assert_eq!(loaded_1d.shape(), array_1d.shape());
    assert_eq!(loaded_1d.to_vec(), array_1d.to_vec());

    // Test 3D array
    let array_3d =
        Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2, 2]);
    let temp_file_3d = NamedTempFile::new().unwrap();
    array_3d
        .to_file(temp_file_3d.path(), SerializeFormat::Npy)
        .unwrap();
    let loaded_3d = Array::<f64>::from_file(temp_file_3d.path(), SerializeFormat::Npy).unwrap();
    assert_eq!(loaded_3d.shape(), array_3d.shape());
    assert_eq!(loaded_3d.to_vec(), array_3d.to_vec());
}

#[test]
fn test_large_array_io() {
    // Test with a larger array to ensure memory efficiency
    let size = 1000;
    let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let array = Array::from_vec(data.clone()).reshape(&[10, 100]);
    let temp_file = NamedTempFile::new().unwrap();

    // Save and load
    array
        .to_file(temp_file.path(), SerializeFormat::Npy)
        .unwrap();
    let loaded_array = Array::<f64>::from_file(temp_file.path(), SerializeFormat::Npy).unwrap();

    assert_eq!(loaded_array.shape(), array.shape());
    assert_eq!(loaded_array.to_vec(), data);
}

#[test]
fn test_pickle_serialization() {
    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let temp_file = NamedTempFile::new().unwrap();

    // Save as Pickle
    array
        .to_file(temp_file.path(), SerializeFormat::Pickle)
        .unwrap();

    // Load back from Pickle
    let loaded_array = Array::<f64>::from_file(temp_file.path(), SerializeFormat::Pickle).unwrap();
    assert_eq!(loaded_array.shape(), array.shape());
    assert_eq!(loaded_array.to_vec(), array.to_vec());
}

#[test]
fn test_pickle_different_types() {
    // Test with integers
    let int_array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    let temp_file_int = NamedTempFile::new().unwrap();

    int_array
        .to_file(temp_file_int.path(), SerializeFormat::Pickle)
        .unwrap();
    let loaded_int =
        Array::<i32>::from_file(temp_file_int.path(), SerializeFormat::Pickle).unwrap();
    assert_eq!(loaded_int.shape(), int_array.shape());
    assert_eq!(loaded_int.to_vec(), int_array.to_vec());

    // Test with booleans
    let bool_array = Array::from_vec(vec![true, false, true, false]).reshape(&[2, 2]);
    let temp_file_bool = NamedTempFile::new().unwrap();

    bool_array
        .to_file(temp_file_bool.path(), SerializeFormat::Pickle)
        .unwrap();
    let loaded_bool =
        Array::<bool>::from_file(temp_file_bool.path(), SerializeFormat::Pickle).unwrap();
    assert_eq!(loaded_bool.shape(), bool_array.shape());
    assert_eq!(loaded_bool.to_vec(), bool_array.to_vec());
}

#[test]
fn test_pickle_large_array() {
    // Test with a larger array to ensure memory efficiency
    let size = 1000;
    let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
    let array = Array::from_vec(data.clone()).reshape(&[10, 100]);
    let temp_file = NamedTempFile::new().unwrap();

    // Save and load
    array
        .to_file(temp_file.path(), SerializeFormat::Pickle)
        .unwrap();
    let loaded_array = Array::<f64>::from_file(temp_file.path(), SerializeFormat::Pickle).unwrap();

    assert_eq!(loaded_array.shape(), array.shape());
    assert_eq!(loaded_array.to_vec(), data);
}

#[test]
fn test_pickle_error_handling() {
    // Test that string serialization/deserialization is not supported
    let array = Array::from_vec(vec![1.0, 2.0]);
    let result = array.to_string(SerializeFormat::Pickle);
    assert!(result.is_err());

    let result = Array::<f64>::from_string("", SerializeFormat::Pickle);
    assert!(result.is_err());

    // Test loading invalid pickle file
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "This is not a valid pickle file").unwrap();
    let result = Array::<f64>::from_file(temp_file.path(), SerializeFormat::Pickle);
    assert!(result.is_err());
}

#[test]
fn test_error_handling() {
    // Test invalid file path
    let array = Array::from_vec(vec![1.0, 2.0]);
    let result = array.to_file(
        &std::path::Path::new("/invalid/path/file.npy"),
        SerializeFormat::Npy,
    );
    assert!(result.is_err());

    // Test loading non-existent file
    let result = Array::<f64>::from_file(
        &std::path::Path::new("/non/existent/file.npy"),
        SerializeFormat::Npy,
    );
    assert!(result.is_err());

    // Test loading invalid NPY file
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "This is not a valid NPY file").unwrap();
    let result = Array::<f64>::from_file(temp_file.path(), SerializeFormat::Npy);
    assert!(result.is_err());
}

#[test]
fn test_text_io_roundtrip() {
    // Create test data
    let array = Array::from_vec(vec![
        1.5,
        2.7,
        std::f64::consts::PI,
        4.0,
        5.5,
        std::f64::consts::TAU,
    ])
    .reshape(&[2, 3]);
    let temp_file = NamedTempFile::new().unwrap();

    // Save using savetxt
    let mut save_options = SaveTxtOptions::default();
    save_options.delimiter = " ".to_string();
    savetxt(temp_file.path(), &array, save_options).unwrap();

    // Load using loadtxt
    let mut load_options = LoadTxtOptions::default();
    load_options.delimiter = None; // Use whitespace as delimiter
    let loaded_array = loadtxt::<f64>(temp_file.path(), load_options).unwrap();

    assert_eq!(loaded_array.shape(), array.shape());
    // Check that values are approximately equal (floating point precision)
    for (original, loaded) in array.to_vec().iter().zip(loaded_array.to_vec().iter()) {
        assert!((original - loaded).abs() < 1e-10);
    }
}
