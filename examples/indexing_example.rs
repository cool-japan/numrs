#![allow(deprecated)]

use numrs2::indexing::IndexSpec;
use numrs2::prelude::*;

fn main() -> Result<()> {
    println!("NumRS Indexing Examples");
    println!("======================\n");

    // Boolean Indexing
    println!("Boolean Indexing:");
    println!("-----------------");

    // Create a simple 1D array
    let arr = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    println!("Original array: {:?}", arr);

    // Create a boolean mask
    let mask_vec = vec![true, false, true, false, true];
    let mask = Array::<bool>::from_vec(mask_vec);
    println!("Boolean mask: {:?}", mask);

    // Apply the boolean mask
    let result = arr.index(&[IndexSpec::Mask(mask.to_vec())])?;
    println!("Result of boolean indexing: {:?}", result);

    // Create a 2D array
    println!("\n2D Boolean Indexing:");
    let arr_2d = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    println!("2D array (shape {:?}):", arr_2d.shape());
    println!("{:?}", arr_2d);

    // Select specific columns
    let col_mask = vec![true, false, true];
    println!("Column mask: {:?}", col_mask);

    let result_2d = arr_2d.index(&[IndexSpec::All, IndexSpec::Mask(col_mask)])?;
    println!("Result of column mask (shape {:?}):", result_2d.shape());
    println!("{:?}", result_2d);

    // Modifying values with a mask
    println!("\nSetting values with a mask:");
    let mut arr_copy = arr.clone();
    println!("Original array: {:?}", arr_copy);

    let values = Array::<f64>::from_vec(vec![10.0, 30.0, 50.0]);
    println!("Values to set: {:?}", values);

    arr_copy.set_mask(&mask, &values)?;
    println!("Array after set_mask: {:?}", arr_copy);

    // Fancy Indexing
    println!("\nFancy Indexing:");
    println!("--------------");

    // Create an array
    let arr = Array::<f64>::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    println!("Original array: {:?}", arr);

    // Use fancy indexing with explicit indices
    let indices = vec![2, 0, 4];
    println!("Indices: {:?}", indices);

    let result = arr.index(&[IndexSpec::Indices(indices)])?;
    println!("Result of fancy indexing: {:?}", result);

    // 2D fancy indexing
    println!("\n2D Fancy Indexing:");
    let arr_2d =
        Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).reshape(&[3, 3]);
    println!("2D array (shape {:?}):", arr_2d.shape());
    println!("{:?}", arr_2d);

    // Select specific rows
    let row_indices = vec![0, 2];
    println!("Row indices: {:?}", row_indices);

    let result_rows = arr_2d.index(&[IndexSpec::Indices(row_indices), IndexSpec::All])?;
    println!("Result of row selection (shape {:?}):", result_rows.shape());
    println!("{:?}", result_rows);

    // Advanced indexing with both dimensions
    println!("\nAdvanced indexing - diagonal elements in reverse:");
    let row_indices = vec![0, 1, 2];
    let col_indices = vec![2, 1, 0];

    let diag_reverse = arr_2d.index(&[
        IndexSpec::Indices(row_indices),
        IndexSpec::Indices(col_indices),
    ])?;
    println!(
        "Result (shape {:?}): {:?}",
        diag_reverse.shape(),
        diag_reverse
    );

    // Slice indexing
    println!("\nSlice Indexing:");
    println!("--------------");

    let slice_result = arr_2d.index(&[IndexSpec::Slice(0, Some(2), None), IndexSpec::All])?;
    println!("First two rows (shape {:?}):", slice_result.shape());
    println!("{:?}", slice_result);

    // Combine slicing with regular indexing
    println!("\nMixed indexing (slice + specific index):");
    let mixed_result = arr_2d.index(&[IndexSpec::Slice(0, Some(2), None), IndexSpec::Index(1)])?;
    println!(
        "Result (shape {:?}): {:?}",
        mixed_result.shape(),
        mixed_result
    );

    Ok(())
}
