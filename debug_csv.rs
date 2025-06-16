use numrs2::prelude::*;
use numrs2::io::*;
use std::fs;
use tempfile::NamedTempFile;

fn main() {
    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    println!("Original array shape: {:?}", array.shape());
    println!("Original array data: {:?}", array.to_vec());
    
    let temp_file = NamedTempFile::new().unwrap();
    
    // Save as CSV
    array.to_file(temp_file.path(), SerializeFormat::Csv).unwrap();
    
    // Read the CSV file as text to see what was actually written
    let csv_content = fs::read_to_string(temp_file.path()).unwrap();
    println!("CSV file content:");
    println!("{}", csv_content);
    
    // Load back from CSV
    let loaded_array = Array::<f64>::from_file(temp_file.path(), SerializeFormat::Csv).unwrap();
    println!("Loaded array shape: {:?}", loaded_array.shape());
    println!("Loaded array data: {:?}", loaded_array.to_vec());
}