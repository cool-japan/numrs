// NumRS2 Array Views Example
//
// This example demonstrates the powerful view functionality in NumRS2,
// which provides zero-copy operations on arrays. Views are lightweight
// references to array data that avoid unnecessary copying, making operations
// more memory-efficient.

use numrs2::prelude::*;
use ndarray::{Axis, Slice};
use num_complex::Complex64;

fn main() -> Result<()> {
    println!("NumRS2 Array Views Example");
    println!("========================\n");

    // SECTION 1: Basic Views
    println!("1. Basic Array Views");
    println!("-------------------");
    
    // Create a 3x3 array for our examples
    let mut array = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        .reshape(&[3, 3]);
    
    println!("Original array (shape {:?}):", array.shape());
    println!("{}", array);
    
    // Create a read-only view
    let view = array.view();
    println!("\nCreated a read-only view (shape {:?})", view.shape());
    println!("First element through view: {}", view.get(&[0, 0])?);
    println!("Elements accessed through view are from the original array");
    
    // Create a mutable view and modify the array
    let mut mutable_view = array.view_mut();
    println!("\nCreated a mutable view (shape {:?})", mutable_view.shape());
    
    // Modify the array through the view
    mutable_view.set(&[0, 0], 10.0)?;
    println!("Modified first element to 10.0 through the mutable view");
    
    // Verify the change in the original array
    println!("Original array after modification:");
    println!("{}", array);
    println!("Changes through the view are reflected in the original array");
    
    // Reset for next examples
    array.set(&[0, 0], 1.0)?;
    
    // SECTION 2: Lifetime Semantics and Memory Efficiency
    println!("\n2. Lifetime Semantics and Memory Efficiency");
    println!("------------------------------------------");
    
    println!("Views maintain a reference to the original data without copying");
    println!("This allows efficient operations on subsets of data");
    
    {
        // Demonstrate zero-copy nature with a simple benchmark
        use std::time::Instant;
        
        let large_array = Array::<f64>::from_vec((0..1_000_000).map(|i| i as f64).collect())
            .reshape(&[1000, 1000]);
        
        // Time to create a view vs a copy
        let start = Instant::now();
        let _view = large_array.view();
        let view_time = start.elapsed();
        
        let start = Instant::now();
        let _copy = large_array.clone();
        let copy_time = start.elapsed();
        
        println!("\nPerformance comparison:");
        println!("Time to create a view: {:?}", view_time);
        println!("Time to create a copy: {:?}", copy_time);
        println!("Views are significantly faster as they don't copy data");
    }
    
    // SECTION 3: Slicing with Views
    println!("\n3. Slicing with Views");
    println!("-------------------");
    
    // Reset our array to the original state
    let array = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        .reshape(&[3, 3]);
    
    println!("Original array:");
    println!("{}", array);
    
    // Direct slice view of a specific row
    let row_view = array.slice_view(0, 1)?;
    println!("\nView of the second row (shape {:?}):", row_view.shape());
    println!("Elements: {:?}", row_view.to_vec());
    
    // Slice with explicit slicing API
    let array_view = array.view();
    let slice_axis_view = array_view.slice_axis(Axis(0), Slice::from(0..2));
    println!("\nView of the first two rows using slice_axis (shape {:?}):", 
             slice_axis_view.shape());
    println!("Elements: {:?}", slice_axis_view.to_vec());
    
    // Using SliceOrIndex for advanced slicing
    let slices = vec![
        SliceOrIndex::Slice(0, Some(2), None),  // First two rows
        SliceOrIndex::Slice(1, None, None)      // Columns starting from index 1
    ];
    
    let advanced_slice = array.sliced_view(&slices)?;
    println!("\nAdvanced slicing view (first two rows, columns from index 1) (shape {:?}):", 
             advanced_slice.shape());
    println!("Elements: {:?}", advanced_slice.to_vec());
    
    // SECTION 4: Strided Views
    println!("\n4. Strided Views");
    println!("--------------");
    
    println!("Original array:");
    println!("{}", array);
    
    // Create a strided view to get every other element in both dimensions
    let strided = array.strided_view(&[2, 2])?;
    println!("\nStrided view with strides [2, 2] (shape {:?}):", strided.shape());
    println!("Elements:");
    for i in 0..strided.shape()[0] {
        for j in 0..strided.shape()[1] {
            print!("{} ", strided.get(&[i, j])?);
        }
        println!();
    }
    
    // Create a negatively strided view (reversed order)
    let reversed = array.strided_view(&[-1, 1])?;
    println!("\nNegatively strided view (rows in reverse) (shape {:?}):", reversed.shape());
    println!("Elements:");
    for i in 0..reversed.shape()[0] {
        for j in 0..reversed.shape()[1] {
            print!("{} ", reversed.get(&[i, j])?);
        }
        println!();
    }
    
    // SECTION 5: View Transformations
    println!("\n5. View Transformations");
    println!("---------------------");
    
    // Transposed view
    let transposed = array.transposed_view();
    println!("Transposed view (shape {:?}):", transposed.shape());
    println!("Original:");
    println!("{}", array);
    println!("Transposed:");
    println!("Elements:");
    for i in 0..transposed.shape()[0] {
        for j in 0..transposed.shape()[1] {
            print!("{} ", transposed.get(&[i, j])?);
        }
        println!();
    }
    
    // Broadcasting view
    let vector = Array::<f64>::from_vec(vec![10.0, 20.0, 30.0]);
    println!("\nOriginal vector (shape {:?}):", vector.shape());
    println!("{}", vector);
    
    let broadcast = vector.broadcast_view(&[3, 3])?;
    println!("\nBroadcast to shape [3, 3]:");
    println!("Elements:");
    for i in 0..broadcast.shape()[0] {
        for j in 0..broadcast.shape()[1] {
            print!("{} ", broadcast.get(&[i, j])?);
        }
        println!();
    }
    
    // SECTION 6: Operations on Views
    println!("\n6. Operations on Views");
    println!("--------------------");
    
    // Arithmetic operations between views
    let a = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array::<f64>::from_vec(vec![4.0, 5.0, 6.0]);
    
    let view_a = a.view();
    let view_b = b.view();
    
    // Add two views
    let sum_result = &view_a + &view_b;
    println!("Addition of two views: {:?}", sum_result.to_vec());
    
    // Subtract two views
    let sub_result = &view_a - &view_b;
    println!("Subtraction of two views: {:?}", sub_result.to_vec());
    
    // Multiply two views (element-wise)
    let mul_result = &view_a * &view_b;
    println!("Multiplication of two views: {:?}", mul_result.to_vec());
    
    // Divide two views (element-wise)
    let div_result = &view_a / &view_b;
    println!("Division of two views: {:?}", div_result.to_vec());
    
    // Mapping operations on views
    let squared = view_a.map(|x| x * x);
    println!("Mapping square function over view: {:?}", squared.to_vec());
    
    // More complex mapping
    let complex_map = view_a.map(|x| Complex64::new(*x, *x / 2.0));
    println!("Mapping to complex numbers: {:?}", complex_map.to_vec());
    
    // SECTION 7: Demonstrating Zero-Copy Nature
    println!("\n7. Zero-Copy Nature of Views");
    println!("--------------------------");
    
    // Create a fresh array
    let mut array = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        .reshape(&[3, 3]);
    
    println!("Original array:");
    println!("{}", array);
    
    {
        // Create a view and print some elements
        let view = array.view();
        println!("\nView of the array:");
        println!("Element at [0,0]: {}", view.get(&[0, 0])?);
        println!("Element at [1,1]: {}", view.get(&[1, 1])?);
    }
    
    // Modify the original array after the view is dropped
    array.set(&[0, 0], 100.0)?;
    array.set(&[1, 1], 500.0)?;
    
    println!("\nAfter modifying original array:");
    println!("{}", array);
    
    // Create a new view to see the changes
    {
        let view = array.view();
        println!("\nNew view reflects changes:");
        println!("Element at [0,0]: {}", view.get(&[0, 0])?);
        println!("Element at [1,1]: {}", view.get(&[1, 1])?);
    }
    
    // SECTION 8: Type Conversion from Views
    println!("\n8. Type Conversion from Views");
    println!("--------------------------");
    
    // Create an integer array and view
    let int_array = Array::from_vec(vec![1, 2, 3, 4]).reshape(&[2, 2]);
    let int_view = int_array.view();
    
    // Convert view to float array
    let float_array = int_view.astype::<f64>()?;
    println!("Integer view: {:?}", int_view.to_vec());
    println!("Converted to float: {:?}", float_array.to_vec());
    
    // Convert view to complex
    let complex_array = int_view.to_complex::<f64>()?;
    println!("Converted to complex: {:?}", complex_array.to_vec());
    
    println!("\nType conversions from views create new owned arrays");
    println!("This is necessary because the element type changes");
    
    Ok(())
}