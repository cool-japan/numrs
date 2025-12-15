#![allow(clippy::result_large_err)]

use numrs2::array::Array;
use numrs2::error::Result;
use numrs2::ufuncs::{
    abs, add, cos, divide, exp, log, maximum, minimum, multiply, power, sin, sqrt, square, subtract,
};

fn main() -> Result<()> {
    println!("NumRS Universal Functions (ufuncs) Examples");
    println!("=========================================\n");

    // Create arrays for our examples
    let a = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let b = Array::<f64>::from_vec(vec![5.0, 6.0, 7.0, 8.0]);

    println!("Array a: {:?}", a);
    println!("Array b: {:?}", b);

    // Binary ufuncs
    println!("\nBinary UFuncs:");
    println!("-------------");

    // Addition
    let result = add(&a, &b)?;
    println!("a + b = {:?}", result);

    // Subtraction
    let result = subtract(&a, &b)?;
    println!("a - b = {:?}", result);

    // Multiplication
    let result = multiply(&a, &b)?;
    println!("a * b = {:?}", result);

    // Division
    let result = divide(&a, &b)?;
    println!("a / b = {:?}", result);

    // Power
    let result = power(&a, &b)?;
    println!("a ** b = {:?}", result);

    // Maximum and minimum
    let result = maximum(&a, &b)?;
    println!("maximum(a, b) = {:?}", result);

    let result = minimum(&a, &b)?;
    println!("minimum(a, b) = {:?}", result);

    // Unary ufuncs
    println!("\nUnary UFuncs:");
    println!("------------");

    // Square
    let result = square(&a);
    println!("square(a) = {:?}", result);

    // Square root
    let result = sqrt(&a);
    println!("sqrt(a) = {:?}", result);

    // Exponential
    let result = exp(&a);
    println!("exp(a) = {:?}", result);

    // Natural logarithm
    let result = log(&a);
    println!("log(a) = {:?}", result);

    // Sine and cosine
    let result = sin(&a);
    println!("sin(a) = {:?}", result);

    let result = cos(&a);
    println!("cos(a) = {:?}", result);

    // Absolute value
    let neg_a = multiply(&a, &Array::<f64>::from_vec(vec![-1.0]))?;
    println!("neg_a = {:?}", neg_a);

    let result = abs(&neg_a);
    println!("abs(neg_a) = {:?}", result);

    // Broadcasting with ufuncs
    println!("\nBroadcasting with UFuncs:");
    println!("------------------------");

    // Scalar operations
    let scalar = Array::<f64>::from_vec(vec![2.0]);
    println!("scalar = {:?}", scalar);

    // Scalar multiplication
    let result = multiply(&a, &scalar)?;
    println!("a * 2 = {:?}", result);

    // 2D arrays with broadcasting
    let row = Array::<f64>::from_vec(vec![10.0, 20.0]).reshape(&[1, 2]);
    let col = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[3, 1]);

    println!("row (shape {:?}): {:?}", row.shape(), row);
    println!("col (shape {:?}): {:?}", col.shape(), col);

    // Add row and column to create a grid
    let result = add(&row, &col)?;
    println!("row + col (shape {:?}):", result.shape());
    println!("{:?}", result);

    // Chain ufuncs
    println!("\nChaining UFuncs:");
    println!("--------------");

    // Compute (a + b)²
    let sum = add(&a, &b)?;
    let squared_sum = square(&sum);
    println!("(a + b)² = {:?}", squared_sum);

    // Compute sin(a) * cos(b)
    let sin_a = sin(&a);
    let cos_b = cos(&b);
    let sin_cos_prod = multiply(&sin_a, &cos_b)?;
    println!("sin(a) * cos(b) = {:?}", sin_cos_prod);

    // Complex example: compute sqrt(a² + b²)
    let a_squared = square(&a);
    let b_squared = square(&b);
    let sum_squares = add(&a_squared, &b_squared)?;
    let result = sqrt(&sum_squares);
    println!("sqrt(a² + b²) = {:?}", result);

    // Different shapes with broadcasting
    println!("\nBroadcasting with Different Shapes:");
    println!("----------------------------------");

    // Create a 2x3 array
    let arr_2d = Array::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
    println!("2D array (shape {:?}):", arr_2d.shape());
    println!("{:?}", arr_2d);

    // Add a 1D array
    let arr_1d = Array::<f64>::from_vec(vec![10.0, 20.0, 30.0]);
    println!("1D array: {:?}", arr_1d);

    let result = add(&arr_2d, &arr_1d)?;
    println!("2D + 1D = (shape {:?}):", result.shape());
    println!("{:?}", result);

    // Add a scalar
    let result = add(&arr_2d, &scalar)?;
    println!("2D + scalar = (shape {:?}):", result.shape());
    println!("{:?}", result);

    Ok(())
}
