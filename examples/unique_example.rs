#![allow(deprecated)]
#![allow(clippy::result_large_err)]

// Example demonstrating the `unique` function in NumRS2
//
// This example shows how to use the `unique` function to find unique elements in arrays,
// including different parameters such as axis, return_index, return_inverse, and return_counts.
// The example also demonstrates how to work with the `UniqueResult` struct.

use numrs2::error::Result;
use numrs2::prelude::*;

fn main() -> Result<()> {
    println!("NumRS2 Unique Function Example");
    println!("==============================");

    // Part 1: Basic usage with 1D arrays
    println!("\nPart 1: Basic Usage with 1D Arrays");
    println!("--------------------------------");

    // Create a 1D array with some repeated elements
    let a = Array::from_vec(vec![1, 2, 3, 2, 1, 4]).reshape(&[6]);
    println!("Original array: {:?}", a);

    // Find unique elements
    let result = unique(&a, None, None, None, None)?;
    println!("Unique values: {:?}", result.values);

    // Part 2: Getting additional information
    println!("\nPart 2: Getting Additional Information");
    println!("------------------------------------");

    // Find unique elements and the indices of their first occurrences
    let result = unique(&a, None, Some(true), None, None)?;
    let (values, indices) = result.values_indices()?;
    println!("Unique values: {:?}", values);
    println!("Indices of first occurrences: {:?}", indices);

    // Find unique elements and their counts
    let result = unique(&a, None, None, None, Some(true))?;
    let (values, counts) = result.values_counts()?;
    println!("Unique values: {:?}", values);
    println!("Counts of each value: {:?}", counts);

    // Find unique elements and the indices to reconstruct the original array
    let result = unique(&a, None, None, Some(true), None)?;
    let (values, inverse) = result.values_inverse()?;
    println!("Unique values: {:?}", values);
    println!("Inverse indices: {:?}", inverse);

    // Reconstruct the original array using inverse indices
    let values_vec = values.to_vec();
    let inverse_vec = inverse.to_vec();
    let mut reconstructed = Vec::with_capacity(a.size());
    for &idx in &inverse_vec {
        reconstructed.push(values_vec[idx]);
    }
    println!("Reconstructed array: {:?}", reconstructed);
    println!("Original array: {:?}", a.to_vec());
    println!(
        "Reconstruction matches original: {}",
        reconstructed == a.to_vec()
    );

    // Part 3: Working with multiple return values
    println!("\nPart 3: Working with Multiple Return Values");
    println!("----------------------------------------");

    // Get all return values
    let result = unique(&a, None, Some(true), Some(true), Some(true))?;
    let (values, indices, inverse, counts) = result.values_indices_inverse_counts()?;
    println!("Unique values: {:?}", values);
    println!("Indices of first occurrences: {:?}", indices);
    println!("Inverse indices: {:?}", inverse);
    println!("Counts of each value: {:?}", counts);

    // Part 4: Using axis parameter with 2D arrays
    println!("\nPart 4: Using Axis Parameter with 2D Arrays");
    println!("----------------------------------------");

    // Create a 2D array
    let b = Array::from_vec(vec![1, 2, 3, 1, 2, 3, 7, 8, 9]).reshape(&[3, 3]);
    println!("Original 2D array:");
    println!("[1, 2, 3]");
    println!("[1, 2, 3]");
    println!("[7, 8, 9]");

    // Find unique rows (axis=0)
    let result = unique(&b, Some(0), None, None, None)?;
    println!("\nUnique rows (axis=0):");
    println!("Shape: {:?}", result.values.shape());
    println!("Values: {:?}", result.values);

    // Find unique rows with counts
    let result = unique(&b, Some(0), None, None, Some(true))?;
    let (_values, counts) = result.values_counts()?;
    println!("Counts of each unique row: {:?}", counts);

    // Find unique columns (axis=1)
    let bt = b.transpose(); // Transpose so we can find unique columns
    let result = unique(&bt, Some(0), None, None, None)?;
    println!("\nUnique columns (axis=0 of transposed array):");
    println!("Shape: {:?}", result.values.shape());
    println!("Values: {:?}", result.values);

    // Part 5: Edge cases
    println!("\nPart 5: Edge Cases");
    println!("----------------");

    // Empty array
    let empty = Array::<i32>::from_vec(vec![]).reshape(&[0]);
    let result = unique(&empty, None, None, None, None)?;
    println!("Unique values of empty array: {:?}", result.values.to_vec());

    // Array with single value
    let single = Array::from_vec(vec![5]);
    let result = unique(&single, None, None, None, None)?;
    println!("Unique values of single-element array: {:?}", result.values);

    // Array with all identical elements
    let all_same = Array::from_vec(vec![7, 7, 7, 7]);
    let result = unique(&all_same, None, None, None, None)?;
    println!(
        "Unique values of array with all identical elements: {:?}",
        result.values
    );

    println!("\nExample complete. NumRS2 unique function successfully demonstrated!");

    Ok(())
}
