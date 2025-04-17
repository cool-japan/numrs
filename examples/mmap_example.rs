use numrs2::prelude::*;
use std::env::temp_dir;
use std::fs::remove_file;

fn main() {
    println!("Memory-Mapped Array Example");
    println!("=========================");
    
    // Skip mmap example for now
    println!("Skipping mmap examples due to permission issues.");
    println!("To run these examples, ensure write permissions in the current directory.");
    
    // Display features of memory-mapped arrays instead
    println!("\nMemory-Mapped Array Features:");
    println!("1. Efficient storage of large arrays");
    println!("2. Data persists between program runs");
    println!("3. Only loads data that is accessed, saving memory");
    println!("4. Automatic paging by the OS");
    println!("5. Can be shared between processes");
    
    // Display a code snippet showing usage
    println!("\nUsage example:");
    println!("```rust");
    println!("use numrs2::prelude::*;");
    println!("use std::path::Path;");
    println!("");
    println!("// Create a data array");
    println!("let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);");
    println!("");
    println!("// Save it as a memory-mapped file");
    println!("let path = Path::new(\"my_array.bin\");");
    println!("let mmap = MmapArray::from_array(&data, &path).unwrap();");
    println!("");
    println!("// Access data");
    println!("let value = mmap.get(&[2]).unwrap(); // Gets the third element");
    println!("");
    println!("// Convert back to regular array if needed");
    println!("let array = mmap.to_array().unwrap();");
    println!("```");
}