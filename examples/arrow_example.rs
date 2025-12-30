//! Apache Arrow Integration Example
//!
//! This example demonstrates NumRS2's Apache Arrow integration capabilities,
//! including zero-copy conversions, IPC streaming, and Feather format support.
//!
//! Run with: cargo run --example arrow_example --features arrow

use numrs2::prelude::*;

#[cfg(feature = "arrow")]
use numrs2::arrow::*;
#[cfg(feature = "arrow")]
use numrs2::math;
#[cfg(feature = "arrow")]
use std::fs::File;
#[cfg(feature = "arrow")]
use std::io::{BufReader, BufWriter};

#[cfg(not(feature = "arrow"))]
fn main() {
    println!("This example requires the 'arrow' feature.");
    println!("Run with: cargo run --example arrow_example --features arrow");
}

#[cfg(feature = "arrow")]
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== NumRS2 Apache Arrow Integration Examples ===\n");

    // Example 1: Basic NumRS to Arrow Conversion
    println!("1. Basic NumRS to Arrow Conversion");
    println!("-----------------------------------");

    let data = Array::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]).reshape(&[5]);
    println!("NumRS Array: {:?}", data.to_vec());

    let arrow_array = to_arrow(&data)?;
    println!("Arrow Array created with {} elements\n", arrow_array.len());

    // Example 2: Arrow to NumRS Conversion
    println!("2. Arrow to NumRS Conversion");
    println!("----------------------------");

    let reconstructed: Array<f64> = from_arrow(&*arrow_array)?;
    println!("Reconstructed NumRS Array: {:?}", reconstructed.to_vec());
    println!(
        "Original == Reconstructed: {}\n",
        data.to_vec() == reconstructed.to_vec()
    );

    // Example 3: Multiple Data Types
    println!("3. Multiple Data Types Support");
    println!("------------------------------");

    // f32
    let data_f32 = Array::from_vec(vec![1.5_f32, 2.5, 3.5]);
    let arrow_f32 = to_arrow(&data_f32)?;
    println!("f32: {:?} -> Arrow -> NumRS", data_f32.to_vec());

    // i32
    let data_i32 = Array::from_vec(vec![10_i32, 20, 30, 40]);
    let arrow_i32 = to_arrow(&data_i32)?;
    println!("i32: {:?} -> Arrow -> NumRS", data_i32.to_vec());

    // u64
    let data_u64 = Array::from_vec(vec![100_u64, 200, 300]);
    let arrow_u64 = to_arrow(&data_u64)?;
    println!("u64: {:?} -> Arrow -> NumRS\n", data_u64.to_vec());

    // Example 4: Feather Format - Write
    println!("4. Writing to Feather Format");
    println!("----------------------------");

    let temp_dir = std::env::temp_dir();
    let feather_path = temp_dir.join("numrs_example.feather");

    let column1 = Array::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0]);
    let column2 = Array::from_vec(vec![10.0_f64, 20.0, 30.0, 40.0, 50.0]);
    let column3 = Array::from_vec(vec![100.0_f64, 200.0, 300.0, 400.0, 500.0]);

    write_feather(
        &feather_path,
        &[("col1", &column1), ("col2", &column2), ("col3", &column3)],
    )?;

    println!("Written 3 columns to: {:?}", feather_path);
    println!("  col1: {:?}", column1.to_vec());
    println!("  col2: {:?}", column2.to_vec());
    println!("  col3: {:?}\n", column3.to_vec());

    // Example 5: Feather Format - Read All Columns
    println!("5. Reading All Columns from Feather");
    println!("------------------------------------");

    let columns: Vec<(String, Array<f64>)> = read_feather_all(&feather_path)?;
    println!("Read {} columns:", columns.len());
    for (name, data) in &columns {
        println!("  {}: {:?}", name, data.to_vec());
    }
    println!();

    // Example 6: Feather Format - Read Single Column
    println!("6. Reading Single Column from Feather");
    println!("--------------------------------------");

    let col2_numrs: Array<f64> = read_feather(&feather_path, "col2")?;
    println!("Read column 'col2': {:?}\n", col2_numrs.to_vec());

    // Example 7: IPC Stream Writer
    println!("7. IPC Stream Writer");
    println!("--------------------");

    let stream_path = temp_dir.join("numrs_stream.ipc");
    let file = File::create(&stream_path)?;
    let writer = BufWriter::new(file);

    let batch1_data = Array::from_vec(vec![1.0_f64, 2.0, 3.0]);
    let batch2_data = Array::from_vec(vec![4.0_f64, 5.0, 6.0]);
    let batch3_data = Array::from_vec(vec![7.0_f64, 8.0, 9.0]);

    // Create schema
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    let schema = Schema::new(vec![Field::new("data", DataType::Float64, false)]);

    let mut stream_writer = IpcStreamWriter::new(writer, &schema)?;
    stream_writer.write_batch(&[("data", &batch1_data)])?;
    stream_writer.write_batch(&[("data", &batch2_data)])?;
    stream_writer.write_batch(&[("data", &batch3_data)])?;
    stream_writer.finish()?;

    println!("Written 3 batches to IPC stream: {:?}", stream_path);
    println!("  batch1: {:?}", batch1_data.to_vec());
    println!("  batch2: {:?}", batch2_data.to_vec());
    println!("  batch3: {:?}\n", batch3_data.to_vec());

    // Example 8: IPC Stream Reader
    println!("8. IPC Stream Reader");
    println!("--------------------");

    let file = File::open(&stream_path)?;
    let reader = BufReader::new(file);
    let mut stream_reader = IpcStreamReader::new(reader)?;

    let mut batch_num = 1;
    while let Some(batches) = stream_reader.read_batch::<f64>()? {
        for batch in batches {
            println!("  Read batch {}: {:?}", batch_num, batch.to_vec());
            batch_num += 1;
        }
    }
    println!();

    // Example 9: Matrix Operations with Arrow Round-trip
    println!("9. Matrix Operations with Arrow Round-trip");
    println!("-------------------------------------------");

    let matrix =
        Array::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).reshape(&[3, 3]);

    println!("Original matrix (3x3):");
    for i in 0..3 {
        print!("  [");
        for j in 0..3 {
            print!("{:4.1}", matrix.get(&[i, j]).unwrap());
        }
        println!(" ]");
    }

    // Convert to Arrow and back
    let matrix_flat = matrix.flatten(Some("C"));
    let arrow_matrix = to_arrow(&matrix_flat)?;
    let reconstructed_flat: Array<f64> = from_arrow(&*arrow_matrix)?;
    let reconstructed = reconstructed_flat.reshape(&[3, 3]);

    println!("\nReconstructed matrix (after Arrow round-trip):");
    for i in 0..3 {
        print!("  [");
        for j in 0..3 {
            print!("{:4.1}", reconstructed.get(&[i, j]).unwrap());
        }
        println!(" ]");
    }

    // Verify equality
    let original_vec = matrix.to_vec();
    let reconstructed_vec = reconstructed.to_vec();
    println!("\nData preserved: {}\n", original_vec == reconstructed_vec);

    // Example 10: Large Array Performance
    println!("10. Large Array Performance");
    println!("---------------------------");

    let large_array = math::linspace(0.0, 1000000.0, 1000000);
    println!("Created large array with {} elements", large_array.size());

    let start = std::time::Instant::now();
    let arrow_large = to_arrow(&large_array)?;
    let to_arrow_time = start.elapsed();

    let start = std::time::Instant::now();
    let reconstructed_large: Array<f64> = from_arrow(&*arrow_large)?;
    let from_arrow_time = start.elapsed();

    println!("  to_arrow time: {:?}", to_arrow_time);
    println!("  from_arrow time: {:?}", from_arrow_time);
    println!(
        "  First 5 elements: {:?}",
        &reconstructed_large.to_vec()[0..5]
    );
    println!(
        "  Last 5 elements: {:?}\n",
        &reconstructed_large.to_vec()[999995..1000000]
    );

    // Cleanup
    std::fs::remove_file(&feather_path).ok();
    std::fs::remove_file(&stream_path).ok();

    println!("=== Examples Complete ===");
    Ok(())
}
