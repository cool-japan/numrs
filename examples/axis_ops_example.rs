#![allow(deprecated)]
#![allow(clippy::result_large_err)]

use numrs2::prelude::*;

fn main() -> Result<()> {
    println!("NumRS2 Axis-Based Operations Examples");
    println!("===================================\n");

    // Create a 2D array for our examples
    let arr = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    println!("Original 2D array (shape {:?}):", arr.shape());
    println!("{:?}", arr);

    // Sum operations
    println!("\nSum Operations:");
    println!("--------------");

    // Sum entire array
    println!("Sum of entire array: {}", arr.sum());

    // Sum along axis 0 (columns)
    let sum_axis0 = arr.sum_axis(0)?;
    println!(
        "Sum along axis 0 (columns) (shape {:?}):",
        sum_axis0.shape()
    );
    println!("{:?}", sum_axis0);

    // Sum along axis 1 (rows)
    let sum_axis1 = arr.sum_axis(1)?;
    println!("Sum along axis 1 (rows) (shape {:?}):", sum_axis1.shape());
    println!("{:?}", sum_axis1);

    // Mean operations
    println!("\nMean Operations:");
    println!("---------------");

    // Mean of entire array
    println!("Mean of entire array: {}", arr.mean());

    // Mean along axis 0
    let mean_axis0 = arr.mean_axis(Some(0))?;
    println!("Mean along axis 0 (shape {:?}):", mean_axis0.shape());
    println!("{:?}", mean_axis0);

    // Mean along axis 1
    let mean_axis1 = arr.mean_axis(Some(1))?;
    println!("Mean along axis 1 (shape {:?}):", mean_axis1.shape());
    println!("{:?}", mean_axis1);

    // Min/Max operations
    println!("\nMin/Max Operations:");
    println!("------------------");

    println!("Min of entire array: {}", arr.min());
    println!("Max of entire array: {}", arr.max());

    // Min/Max along axes
    let min_axis0 = arr.min_axis(Some(0))?;
    println!("Min along axis 0: {:?}", min_axis0);

    let max_axis1 = arr.max_axis(Some(1))?;
    println!("Max along axis 1: {:?}", max_axis1);

    // Product operations
    println!("\nProduct Operations:");
    println!("------------------");

    // Product of entire array
    println!("Product of entire array: {}", arr.product());

    // Product along axes
    let prod_axis0 = arr.prod_axis(Some(0))?;
    println!("Product along axis 0: {:?}", prod_axis0);

    let prod_axis1 = arr.prod_axis(Some(1))?;
    println!("Product along axis 1: {:?}", prod_axis1);

    // Cumulative operations
    println!("\nCumulative Operations:");
    println!("---------------------");

    // Cumulative sum along axes
    let cumsum_axis0 = arr.cumsum_axis(0)?;
    println!(
        "Cumulative sum along axis 0 (shape {:?}):",
        cumsum_axis0.shape()
    );
    println!("{:?}", cumsum_axis0);

    let cumsum_axis1 = arr.cumsum_axis(1)?;
    println!(
        "Cumulative sum along axis 1 (shape {:?}):",
        cumsum_axis1.shape()
    );
    println!("{:?}", cumsum_axis1);

    // Cumulative product along axes
    let cumprod_axis1 = arr.cumprod_axis(1)?;
    println!(
        "Cumulative product along axis 1 (shape {:?}):",
        cumprod_axis1.shape()
    );
    println!("{:?}", cumprod_axis1);

    // Argmin/Argmax operations
    println!("\nArgmin/Argmax Operations:");
    println!("------------------------");

    let argmin_axis1 = arr.argmin_axis(1)?;
    println!(
        "Argmin along axis 1 (indices of minimum values): {:?}",
        argmin_axis1
    );

    let argmax_axis0 = arr.argmax_axis(0)?;
    println!(
        "Argmax along axis 0 (indices of maximum values): {:?}",
        argmax_axis0
    );

    // Statistical operations
    println!("\nStatistical Operations:");
    println!("----------------------");

    // Variance and standard deviation
    println!("Variance of entire array: {}", arr.var());
    println!("Standard deviation of entire array: {}", arr.std());

    // Variance along axes
    let var_axis0 = arr.var_axis(Some(0))?;
    println!("Variance along axis 0: {:?}", var_axis0);

    let std_axis1 = arr.std_axis(Some(1))?;
    println!("Standard deviation along axis 1: {:?}", std_axis1);

    // Example with a 3D array
    println!("\n3D Array Example:");
    println!("----------------");

    let arr_3d = Array::<f64>::from_vec(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ])
    .reshape(&[3, 2, 2]);

    println!("3D array shape: {:?}", arr_3d.shape());

    // Sum along different axes
    let sum_axis0_3d = arr_3d.sum_axis(0)?;
    println!("Sum along axis 0 (shape {:?}):", sum_axis0_3d.shape());
    println!("{:?}", sum_axis0_3d);

    let sum_axis1_3d = arr_3d.sum_axis(1)?;
    println!("Sum along axis 1 (shape {:?}):", sum_axis1_3d.shape());
    println!("{:?}", sum_axis1_3d);

    let sum_axis2_3d = arr_3d.sum_axis(2)?;
    println!("Sum along axis 2 (shape {:?}):", sum_axis2_3d.shape());
    println!("{:?}", sum_axis2_3d);

    Ok(())
}
