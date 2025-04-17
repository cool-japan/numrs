use numrs2::prelude::*;

fn main() {
    println!("NumRS2 Array Manipulation Functions Example");
    println!("==========================================\n");

    // Create a sample 3D array
    let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        .reshape(&[2, 2, 2]);
    println!("3D Array Shape: {:?}", arr.shape());
    println!("3D Array Data: {:?}", arr.to_vec());

    // Demonstrate apply_along_axis
    println!("\n--- apply_along_axis Examples ---");
    
    // Example 1: Apply sum function along axis 0
    let result = apply_along_axis(&arr, 0, |slice| {
        // Sum elements in each slice
        slice.to_vec().iter().sum::<f64>()
    }).unwrap();
    println!("Sum along axis 0: shape={:?}, data={:?}", result.shape(), result.to_vec());
    
    // Example 2: Apply custom function along axis 1
    let result = apply_along_axis(&arr, 1, |slice| {
        // Calculate product of elements in each slice
        slice.to_vec().iter().fold(1.0, |acc, &x| acc * x)
    }).unwrap();
    println!("Product along axis 1: shape={:?}, data={:?}", result.shape(), result.to_vec());
    
    // Example 3: Apply function that returns a different type
    let result = apply_along_axis(&arr, 2, |slice| {
        // Return a boolean based on the sum
        slice.to_vec().iter().sum::<f64>() > 5.0
    }).unwrap();
    println!("Boolean based on sum along axis 2: shape={:?}, data={:?}", 
             result.shape(), result.to_vec());

    // Demonstrate vectorize
    println!("\n--- vectorize Examples ---");
    
    // Create a 1D array
    let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    
    // Create a function that squares a number
    let square = |x: f64| x * x;
    
    // Vectorize the function
    let vec_square = vectorize(square);
    let result = vec_square(&x);
    println!("Square function vectorized: {:?}", result.to_vec());
    
    // Create a function that converts to a boolean
    let is_even = |x: f64| (x as i32) % 2 == 0;
    
    // Vectorize the function
    let vec_is_even = vectorize(is_even);
    let result = vec_is_even(&x);
    println!("is_even function vectorized: {:?}", result.to_vec());

    // More complex example: multi-dimensional array with a string-returning function
    let arr_2d = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    
    // Vectorize a function that returns a string description
    let describe = |x: f64| {
        if x < 2.0 {
            "small"
        } else if x < 4.0 {
            "medium"
        } else {
            "large"
        }
    };
    
    let vec_describe = vectorize(describe);
    let result = vec_describe(&arr_2d);
    println!("Descriptive function vectorized: {:?}", result.to_vec());
}