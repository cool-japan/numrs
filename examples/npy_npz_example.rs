#![allow(deprecated)]

use numrs2::io::SerializeFormat;
use numrs2::prelude::*;
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    println!("NumRS Array NPY/NPZ Format Example");
    println!("=================================\n");

    // Create a 2D array for our examples
    let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    println!("Original array (shape {:?}):", array.shape());
    println!("{}", array);

    // Create a directory for our output files if it doesn't exist
    let output_dir = Path::new("./examples/output");
    if !output_dir.exists() {
        fs::create_dir_all(output_dir).unwrap_or_else(|_| {
            println!("Note: Could not create output directory, using current directory instead");
        });
    }

    // 1. Save array in NPY format
    println!("\n1. NPY Format");
    let npy_path = output_dir.join("array.npy");
    println!("Saving array to NPY file: {:?}", npy_path);

    // Write to NPY file
    array.to_file(&npy_path, SerializeFormat::Npy)?;
    println!("Array would be saved to NPY file if uncommented");

    // Reading back the NPY file
    println!("Reading back from NPY file:");
    let array_from_npy = Array::<f64>::from_file(&npy_path, SerializeFormat::Npy)?;
    println!(
        "Array loaded from NPY (shape {:?}):",
        array_from_npy.shape()
    );
    println!("{}", array_from_npy);

    // 2. Save array in NPZ format (zipped NPY)
    println!("\n2. NPZ Format");
    let npz_path = output_dir.join("array.npz");
    println!("Saving array to NPZ file: {:?}", npz_path);

    // Write to NPZ file
    array.to_file(&npz_path, SerializeFormat::Npz)?;
    println!("Array would be saved to NPZ file if uncommented");

    // Reading back the NPZ file
    println!("Reading back from NPZ file:");
    let array_from_npz = Array::<f64>::from_file(&npz_path, SerializeFormat::Npz)?;
    println!(
        "Array loaded from NPZ (shape {:?}):",
        array_from_npz.shape()
    );
    println!("{}", array_from_npz);

    // 3. Show that NPZ files can be read by NumPy in Python
    println!("\n3. Interoperability with NumPy");
    println!("NPY and NPZ files created by NumRS can be read directly by NumPy in Python:");
    println!("```python");
    println!("import numpy as np");
    println!("# Load array from NPY file");
    println!("arr_from_npy = np.load('array.npy')");
    println!("print(arr_from_npy)");
    println!("");
    println!("# Load array from NPZ file");
    println!("npz_file = np.load('array.npz')");
    println!("arr_from_npz = npz_file['arr_0']");
    println!("print(arr_from_npz)");
    println!("```");

    // 4. Show that NumPy-created NPY/NPZ files can be read by NumRS
    println!("\n4. Reading NumPy-created files in NumRS");
    println!("```rust");
    println!("// Read NPY file created by NumPy");
    println!("let array_from_numpy = Array::<f64>::from_file(\"numpy_created.npy\", SerializeFormat::Npy)?;");
    println!("");
    println!("// Read NPZ file created by NumPy");
    println!("let array_from_numpy_npz = Array::<f64>::from_file(\"numpy_created.npz\", SerializeFormat::Npz)?;");
    println!("```");

    Ok(())
}
