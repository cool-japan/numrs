#![allow(deprecated)]

use numrs2::indexing::IndexSpec;
use numrs2::prelude::*;

fn main() -> Result<()> {
    println!("NumRS Shape Manipulation Operations");
    println!("===================================\n");

    // Create a 2D array for our examples
    let array = Array::from_vec(vec![1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
    println!("Original array (shape {:?}):", array.shape());
    println!("{}", array);

    // 1. Ravel - flattens the array (returns a view when possible)
    println!("\n1. Ravel - flatten an array (C-order by default)");
    let flat_c = ravel(&array, Some('C'))?;
    println!("C-order ravel (shape {:?}):", flat_c.shape());
    println!("{}", flat_c);

    let flat_f = ravel(&array, Some('F'))?;
    println!("F-order (column-major) ravel (shape {:?}):", flat_f.shape());
    println!("{}", flat_f);

    // 2. Flatten - always returns a copy
    println!("\n2. Flatten - returns a copy of the flattened array");
    let flat_copy = flatten(&array, None)?;
    println!("Flattened array (shape {:?}):", flat_copy.shape());
    println!("{}", flat_copy);

    // 3. Swapaxes - interchange two axes
    println!("\n3. Swapaxes - interchange two axes");
    let swapped = swapaxes(&array, 0, 1)?;
    println!("Swapped axes 0 and 1 (shape {:?}):", swapped.shape());
    println!("{}", swapped);

    // 4. Create a 3D array for more complex examples
    let array3d = Array::from_vec(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).reshape(&[2, 2, 3]);
    println!("\n3D array (shape {:?}):", array3d.shape());
    println!("First slice (array3d[0]):");
    println!(
        "{}",
        array3d.index(&[IndexSpec::Index(0), IndexSpec::All, IndexSpec::All])?
    );
    println!("Second slice (array3d[1]):");
    println!(
        "{}",
        array3d.index(&[IndexSpec::Index(1), IndexSpec::All, IndexSpec::All])?
    );

    // 5. Moveaxis - move axes to new positions
    println!("\n5. Moveaxis - move axes to new positions");
    let moved = moveaxis(&array3d, &[0], &[2])?;
    println!("Moved axis 0 to position 2 (shape {:?}):", moved.shape());
    println!("First slice (moved[0]):");
    println!(
        "{}",
        moved.index(&[IndexSpec::Index(0), IndexSpec::All, IndexSpec::All])?
    );
    println!("Second slice (moved[1]):");
    println!(
        "{}",
        moved.index(&[IndexSpec::Index(1), IndexSpec::All, IndexSpec::All])?
    );

    // 6. Atleast functions
    println!("\n6. Atleast functions - ensure arrays have at least n dimensions");

    // Create a 1D array
    let array1d = Array::from_vec(vec![1, 2, 3]);
    println!("1D array (shape {:?}):", array1d.shape());
    println!("{}", array1d);

    // Atleast 1d - no change for 1D array
    let atleast1d_result = atleast_1d(&[&array1d])?[0].clone();
    println!(
        "\natleast_1d result (shape {:?}):",
        atleast1d_result.shape()
    );
    println!("{}", atleast1d_result);

    // Atleast 2d - adds a dimension
    let atleast2d_result = atleast_2d(&[&array1d])?[0].clone();
    println!(
        "\natleast_2d result (shape {:?}):",
        atleast2d_result.shape()
    );
    println!("{}", atleast2d_result);

    // Atleast 3d - adds two dimensions
    let atleast3d_result = atleast_3d(&[&array1d])?[0].clone();
    println!(
        "\natleast_3d result (shape {:?}):",
        atleast3d_result.shape()
    );
    println!("First slice (atleast3d_result[0]):");
    println!(
        "{}",
        atleast3d_result.index(&[IndexSpec::Index(0), IndexSpec::All, IndexSpec::All])?
    );

    // 7. Multiple arrays at once
    println!("\n7. Processing multiple arrays at once with atleast functions");
    let array1 = Array::from_vec(vec![1, 2, 3]);
    let array2 = Array::from_vec(vec![4, 5, 6, 7, 8, 9]).reshape(&[2, 3]);

    let results = atleast_3d(&[&array1, &array2])?;
    println!("First array result (shape {:?}):", results[0].shape());
    println!("Second array result (shape {:?}):", results[1].shape());

    Ok(())
}
