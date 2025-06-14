#![allow(deprecated)]

use numrs2::prelude::*;

fn main() -> Result<()> {
    println!("NumRS2 Array Creation and Manipulation Example");
    println!("===========================================");

    // Array Creation Functions
    println!("Array Creation Functions");
    println!("----------------------");

    // Create a range of values from 0 to 9
    let a = arange(0.0, 10.0, 1.0);
    println!("arange(0, 10, 1) = {}", a);

    // Create evenly spaced values from 0 to 10
    let b = linspace(0.0, 10.0, 6);
    println!("linspace(0, 10, 6) = {}", b);

    // Create logarithmically spaced values
    let c = logspace(0.0, 3.0, 4, None);
    println!("logspace(0, 3, 4) = {}", c);

    // Create geometrically spaced values
    let d = geomspace(1.0, 1000.0, 4);
    println!("geomspace(1, 1000, 4) = {}", d);

    // Create a meshgrid
    let x = linspace(-5.0, 5.0, 5);
    let y = linspace(-5.0, 5.0, 5);
    let grid_result = meshgrid(&[&x, &y], None)?;
    let xx = grid_result[0].clone();
    let yy = grid_result[1].clone();
    println!("\nMeshgrid coordinates:");
    println!("X coordinates shape: {:?}", xx.shape());
    println!("X coordinates = {}", xx);
    println!("Y coordinates shape: {:?}", yy.shape());
    println!("Y coordinates = {}", yy);

    // Array Manipulation Functions
    println!("\nArray Manipulation Functions");
    println!("--------------------------");

    // Tile an array
    let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let tiled = tile(&a, &[2])?;
    println!("tile([1, 2, 3], 2) = {}", tiled);

    // Repeat elements
    let repeated = repeat(&a, 3, None)?;
    println!("repeat([1, 2, 3], 3) = {}", repeated);

    // Repeat along axis
    let a_2d = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    println!("\nOriginal 2D array:");
    println!("{}", a_2d);

    let repeated_axis0 = repeat(&a_2d, 2, Some(0))?;
    println!("\nRepeat along axis 0:");
    println!("{}", repeated_axis0);

    let repeated_axis1 = repeat(&a_2d, 2, Some(1))?;
    println!("\nRepeat along axis 1:");
    println!("{}", repeated_axis1);

    // Concatenate arrays
    let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
    let c = concatenate(&[&a, &b], 0)?;
    println!("\nConcatenate 1D arrays:");
    println!("array1 = {}", a);
    println!("array2 = {}", b);
    println!("concatenate(array1, array2) = {}", c);

    // Stack arrays
    let a_2d = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
    let b_2d = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0]).reshape(&[2, 2]);

    println!("\nStack 2D arrays:");
    println!("array1 =\n{}", a_2d);
    println!("array2 =\n{}", b_2d);

    let stacked0 = stack(&[&a_2d, &b_2d], 0)?;
    println!("\nstack(arrays, axis=0) shape = {:?}", stacked0.shape());
    println!("{}", stacked0);

    let stacked1 = stack(&[&a_2d, &b_2d], 1)?;
    println!("\nstack(arrays, axis=1) shape = {:?}", stacked1.shape());
    println!("{}", stacked1);

    // Split arrays
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let splits = split(&a, &[2, 4], 0)?;

    println!("\nSplit array:");
    println!("original = {}", a);
    println!("split at indices [2, 4]:");
    for (i, split) in splits.iter().enumerate() {
        println!("  split {} = {}", i, split);
    }

    // Expand dimensions
    let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
    let expanded0 = expand_dims(&a, 0)?;
    let expanded1 = expand_dims(&a, 1)?;

    println!("\nExpand dimensions:");
    println!("original = {} (shape = {:?})", a, a.shape());
    println!(
        "expand_dims(a, 0) = {} (shape = {:?})",
        expanded0,
        expanded0.shape()
    );
    println!(
        "expand_dims(a, 1) = {} (shape = {:?})",
        expanded1,
        expanded1.shape()
    );

    // Squeeze dimensions
    let a = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3, 1]);
    let squeezed = squeeze(&a, None)?;

    println!("\nSqueeze dimensions:");
    println!("original shape = {:?}", a.shape());
    println!("squeezed shape = {:?}", squeezed.shape());

    // Element-wise Math Functions
    println!("\nElement-wise Math Functions");
    println!("-------------------------");

    let angles = linspace(0.0, std::f64::consts::PI, 5);
    println!("Angles = {}", angles);

    println!("\nTrigonometric functions:");
    println!("sin(angles) = {}", angles.sin());
    println!("cos(angles) = {}", angles.cos());
    println!("tan(angles) = {}", angles.tan());

    let values = Array::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    println!("\nValues = {}", values);

    println!("\nBasic math functions:");
    println!("abs(values) = {}", values.abs());
    println!("exp(values) = {}", values.exp());
    println!("Square root of abs(values) = {}", values.abs().sqrt());

    println!("\nRounding functions:");
    let floats = Array::from_vec(vec![1.2, 1.5, 1.8, 2.1, 2.5, 2.9]);
    println!("Original = {}", floats);
    println!("floor(floats) = {}", floats.floor());
    println!("ceil(floats) = {}", floats.ceil());
    println!("round(floats) = {}", floats.round());

    Ok(())
}
