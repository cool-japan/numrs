#![allow(deprecated)]

use numrs2::prelude::*;

fn main() {
    println!("MaskedArray Examples");
    println!("===================\n");

    basic_masked_array_example();
    masked_operations_example();
    missing_data_example();
    statistical_operations_example();
}

fn basic_masked_array_example() {
    println!("1. Basic MaskedArray Usage");
    println!("-------------------------");

    // Create a data array
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    println!("Original data: {:?}", data);

    // Create a mask (true = masked/invalid)
    let mask = Array::from_vec(vec![false, true, false, true, false]);
    println!("Mask (true = masked): {:?}", mask);

    // Create a masked array
    let masked_array = MaskedArray::new(data, Some(mask), Some(0.0)).unwrap();
    println!("Masked array: {}", masked_array);

    // Get the data with masked values replaced by fill value
    let filled_data = masked_array.filled(None);
    println!(
        "Filled data (masked values replaced by fill value): {:?}",
        filled_data
    );

    // Get only the valid (unmasked) data
    let valid_data = masked_array.compressed();
    println!("Valid data (compressed): {:?}", valid_data);

    // Count masked and valid values
    println!("Masked count: {}", masked_array.count_masked());
    println!("Valid count: {}", masked_array.count_valid());

    println!();
}

fn masked_operations_example() {
    println!("2. Operations with MaskedArrays");
    println!("------------------------------");

    // Create masked arrays
    let data1 = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let mask1 = Array::from_vec(vec![false, true, false, false, false]);
    let ma1 = MaskedArray::new(data1, Some(mask1), Some(0.0)).unwrap();

    let data2 = Array::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    let mask2 = Array::from_vec(vec![false, false, false, true, false]);
    let ma2 = MaskedArray::new(data2, Some(mask2), Some(0.0)).unwrap();

    println!("Masked array 1: {}", ma1);
    println!("Masked array 2: {}", ma2);

    use std::ops::{Add, Div, Mul, Sub};

    // Arithmetic operations
    let addition = Add::add(&ma1, &ma2);
    println!("Addition result: {}", addition);
    println!("Addition filled: {:?}", addition.filled(None));

    let subtraction = Sub::sub(&ma1, &ma2);
    println!("Subtraction result: {}", subtraction);
    println!("Subtraction filled: {:?}", subtraction.filled(None));

    let multiplication = Mul::mul(&ma1, &ma2);
    println!("Multiplication result: {}", multiplication);
    println!("Multiplication filled: {:?}", multiplication.filled(None));

    let division = Div::div(&ma1, &ma2);
    println!("Division result: {}", division);
    println!("Division filled: {:?}", division.filled(None));

    // Division by zero
    let data3 = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0]);
    let data4 = Array::from_vec(vec![2.0, 0.0, 3.0, 0.0, 5.0]);

    let ma3 = MaskedArray::new(
        data3.clone(),
        Some(Array::from_vec(vec![false; 5])),
        Some(0.0),
    )
    .unwrap();
    let ma4 = MaskedArray::new(
        data4.clone(),
        Some(Array::from_vec(vec![false; 5])),
        Some(0.0),
    )
    .unwrap();

    println!("\nDivision by zero example:");
    println!("Dividend: {}", ma3);
    println!("Divisor: {}", ma4);

    let div_result = Div::div(&ma3, &ma4);
    println!("Division result: {}", div_result);
    println!("Division filled: {:?}", div_result.filled(None));
    println!(
        "Masked count: {} (zeros were automatically masked)",
        div_result.count_masked()
    );

    println!();
}

fn missing_data_example() {
    println!("3. Handling Missing Data");
    println!("-----------------------");

    // Common scenario: data with missing values (represented as a special value like -999)
    let data = Array::from_vec(vec![1.0, 2.0, -999.0, 4.0, -999.0, 6.0, 7.0]);
    println!("Data with missing values (represented as -999): {:?}", data);

    // Create masked array by masking specific values
    let masked = MaskedArray::masked_values(data, -999.0, Some(0.0)).unwrap();
    println!("Masked array: {}", masked);

    // Create masked array from condition
    let data2 = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let condition = data2.map(|x| x < 3.0 || x > 6.0); // Mask values < 3 or > 6
    let masked2 = MaskedArray::masked_where(data2, condition, Some(0.0)).unwrap();
    println!(
        "\nData with values masked by condition (x < 3 || x > 6): {}",
        masked2
    );

    // Dealing with NaN and Infinity values
    let data3 = Array::from_vec(vec![
        1.0,
        f64::NAN,
        3.0,
        f64::INFINITY,
        5.0,
        f64::NEG_INFINITY,
        7.0,
    ]);
    println!("\nData with NaN and Infinity values: {:?}", data3);

    let masked3 = MaskedArray::<f64>::masked_invalid(data3, Some(0.0)).unwrap();
    println!("Masked array (NaN and Infinity masked): {}", masked3);
    println!("Filled data: {:?}", masked3.filled(None));

    println!();
}

fn statistical_operations_example() {
    println!("4. Statistical Operations with MaskedArrays");
    println!("------------------------------------------");

    // Create data with masked values
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    let mask = Array::from_vec(vec![
        false, false, true, false, true, false, true, false, false, true,
    ]);
    let masked = MaskedArray::new(data.clone(), Some(mask), Some(0.0)).unwrap();

    println!("Masked array: {}", masked);
    println!("Masked values: 3.0, 5.0, 7.0, 10.0");

    // Statistical operations ignore masked values
    if let Some(mean) = masked.mean() {
        println!("Mean of unmasked values: {}", mean);
    }

    if let Some(sum) = masked.sum() {
        println!("Sum of unmasked values: {}", sum);
    }

    if let Some(min) = masked.min() {
        println!("Min of unmasked values: {}", min);
    }

    if let Some(max) = masked.max() {
        println!("Max of unmasked values: {}", max);
    }

    // Compare with operations on all values
    println!("\nFor comparison, operations on all values:");
    println!("Mean of all values: {}", data.mean());
    println!("Sum of all values: {}", data.sum());
    println!("Min of all values: {}", data.min());
    println!("Max of all values: {}", data.max());

    // Case where all values are masked
    let all_masked = MaskedArray::masked_all(data, Some(0.0)).unwrap();
    println!("\nAll values masked: {}", all_masked);
    println!("Mean of all masked values: {:?}", all_masked.mean());
    println!("Sum of all masked values: {:?}", all_masked.sum());

    println!();
}
