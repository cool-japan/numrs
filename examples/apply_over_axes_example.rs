#![allow(deprecated)]
#![allow(clippy::result_large_err)]

use numrs2::axis_ops::AxisOps;
use numrs2::prelude::*;

fn main() {
    println!("NumRS2 apply_over_axes Example");
    println!("==============================\n");

    // Create a sample 3D array
    let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2, 2]);
    println!("3D Array Shape: {:?}", arr.shape());
    println!("3D Array Data: {:?}", arr.to_vec());

    // For apply_over_axes, we need to use the AxisOps trait methods
    // because of the mismatch between the Array class and the AxisOps trait

    // Apply sum function over axes 0 and 1
    let result = apply_over_axes(&arr, &[0, 1], |a, ax| {
        // The AxisOps trait provides sum_axis that takes an Option<usize>
        AxisOps::sum_axis(a, Some(ax))
    })
    .unwrap();

    println!("\nSum over axes 0 and 1:");
    println!("Shape: {:?}", result.shape());
    println!("Data: {:?}", result.to_vec());

    // Apply min function over axes 1 and 2
    let result = apply_over_axes(&arr, &[1, 2], |a, ax| {
        // The AxisOps trait provides min_axis that takes an Option<usize>
        AxisOps::min_axis(a, Some(ax))
    })
    .unwrap();

    println!("\nMin over axes 1 and 2:");
    println!("Shape: {:?}", result.shape());
    println!("Data: {:?}", result.to_vec());
}
