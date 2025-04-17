use numrs2::prelude::*;
use approx::assert_relative_eq;

#[test]
fn test_masked_array_creation() {
    // Create a data array
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // Create mask (true = masked)
    let mask = Array::from_vec(vec![false, true, false, true, false]);
    
    // Create masked array
    let masked = MaskedArray::new(data.clone(), Some(mask.clone()), Some(0.0)).unwrap();
    
    // Check shape and size
    assert_eq!(masked.shape(), vec![5]);
    assert_eq!(masked.size(), 5);
    assert_eq!(masked.count_masked(), 2);
    assert_eq!(masked.count_valid(), 3);
    
    // Create from values
    let data2 = Array::from_vec(vec![1.0, 2.0, -999.0, 4.0, -999.0]);
    let masked2 = MaskedArray::masked_values(data2, -999.0, Some(0.0)).unwrap();
    
    // Should have same mask pattern as the first array
    assert_eq!(masked2.count_masked(), 2);
    
    // Test creation with no explicit mask
    let masked3 = MaskedArray::new(data.clone(), None, Some(0.0)).unwrap();
    assert_eq!(masked3.count_masked(), 0);
    
    // Test masked_where
    let data_copy = data.clone();
    let condition = data_copy.map(|x| x > 3.0);
    let masked4 = MaskedArray::masked_where(data.clone(), condition, Some(0.0)).unwrap();
    assert_eq!(masked4.count_masked(), 2);
}

#[test]
fn test_masked_array_get_set() {
    // Create a data array
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // Create mask (true = masked)
    let mask = Array::from_vec(vec![false, true, false, true, false]);
    
    // Create masked array
    let mut masked = MaskedArray::new(data.clone(), Some(mask), Some(0.0)).unwrap();
    
    // Get values - masked elements should return fill value
    assert_relative_eq!(masked.get(&[0]).unwrap(), 1.0);
    assert_relative_eq!(masked.get(&[1]).unwrap(), 0.0); // masked -> fill value
    assert_relative_eq!(masked.get(&[2]).unwrap(), 3.0);
    assert_relative_eq!(masked.get(&[3]).unwrap(), 0.0); // masked -> fill value
    assert_relative_eq!(masked.get(&[4]).unwrap(), 5.0);
    
    // Set values
    masked.set(&[0], 10.0, None).unwrap();
    masked.set(&[1], 20.0, Some(false)).unwrap(); // unmask
    masked.set(&[2], 30.0, Some(true)).unwrap(); // mask
    
    // Check updated values
    assert_relative_eq!(masked.get(&[0]).unwrap(), 10.0);
    assert_relative_eq!(masked.get(&[1]).unwrap(), 20.0); // now unmasked
    assert_relative_eq!(masked.get(&[2]).unwrap(), 0.0); // now masked -> fill value
    
    // Check mask count
    assert_eq!(masked.count_masked(), 2); // still 2 masked elements
}

#[test]
fn test_masked_array_reshape_transpose() {
    // Create a data array
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    
    // Create mask (true = masked)
    let mask = Array::from_vec(vec![false, true, false, true, false, true]);
    
    // Create masked array
    let masked = MaskedArray::new(data.clone(), Some(mask), Some(0.0)).unwrap();
    
    // Reshape
    let reshaped = masked.reshape(&[2, 3]);
    assert_eq!(reshaped.shape(), vec![2, 3]);
    assert_eq!(reshaped.count_masked(), 3);
    
    // Transpose
    let transposed = reshaped.transpose();
    assert_eq!(transposed.shape(), vec![3, 2]);
    assert_eq!(transposed.count_masked(), 3);
}

#[test]
fn test_masked_array_filled_compressed() {
    // Create a data array
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // Create mask (true = masked)
    let mask = Array::from_vec(vec![false, true, false, true, false]);
    
    // Create masked array
    let masked = MaskedArray::new(data.clone(), Some(mask), Some(0.0)).unwrap();
    
    // Get filled array (masked values replaced with fill value)
    let filled = masked.filled(None);
    assert_eq!(filled.shape(), vec![5]);
    assert_eq!(filled.to_vec(), vec![1.0, 0.0, 3.0, 0.0, 5.0]);
    
    // Get filled array with custom fill value
    let filled2 = masked.filled(Some(-1.0));
    assert_eq!(filled2.to_vec(), vec![1.0, -1.0, 3.0, -1.0, 5.0]);
    
    // Get compressed array (only unmasked values)
    let compressed = masked.compressed();
    assert_eq!(compressed.shape(), vec![3]);
    assert_eq!(compressed.to_vec(), vec![1.0, 3.0, 5.0]);
}

#[test]
fn test_masked_array_arithmetic() {
    // Create data arrays
    let data1 = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let data2 = Array::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    
    // Create masks
    let mask1 = Array::from_vec(vec![false, true, false, false, false]);
    let mask2 = Array::from_vec(vec![false, false, false, true, false]);
    
    // Create masked arrays
    let ma1 = MaskedArray::new(data1.clone(), Some(mask1), Some(0.0)).unwrap();
    let ma2 = MaskedArray::new(data2.clone(), Some(mask2), Some(0.0)).unwrap();
    
    // Addition
    let sum = &ma1 + &ma2;
    assert_eq!(sum.count_masked(), 2); // Both positions mask1 and mask2 are true
    
    // Should be masked at positions 1 and 3
    let sum_filled = sum.filled(None);
    assert_eq!(sum_filled.to_vec(), vec![6.0, 0.0, 6.0, 0.0, 6.0]);
    
    // Subtraction
    let diff = &ma1 - &ma2;
    let diff_filled = diff.filled(None);
    assert_eq!(diff_filled.to_vec(), vec![-4.0, 0.0, 0.0, 0.0, 4.0]);
    
    // Multiplication
    let prod = &ma1 * &ma2;
    let prod_filled = prod.filled(None);
    assert_eq!(prod_filled.to_vec(), vec![5.0, 0.0, 9.0, 0.0, 5.0]);
    
    // Division
    let quot = &ma1 / &ma2;
    let quot_filled = quot.filled(None);
    assert_relative_eq!(quot_filled.to_vec()[0], 0.2, epsilon = 1e-10);
    assert_relative_eq!(quot_filled.to_vec()[2], 1.0, epsilon = 1e-10);
    assert_relative_eq!(quot_filled.to_vec()[4], 5.0, epsilon = 1e-10);
    
    // Division by zero should mask the result
    let data3 = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let data4 = Array::from_vec(vec![5.0, 4.0, 0.0, 2.0, 0.0]);
    let ma3 = MaskedArray::new(data3.clone(), Some(Array::from_vec(vec![false; 5])), Some(0.0)).unwrap();
    let ma4 = MaskedArray::new(data4.clone(), Some(Array::from_vec(vec![false; 5])), Some(0.0)).unwrap();
    
    let div_result = &ma3 / &ma4;
    assert_eq!(div_result.count_masked(), 2); // Positions where divisor is 0
}

#[test]
fn test_masked_array_statistics() {
    // Create data array
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    
    // Create mask (true = masked)
    let mask = Array::from_vec(vec![false, true, false, true, false]);
    
    // Create masked array
    let masked = MaskedArray::new(data.clone(), Some(mask), Some(0.0)).unwrap();
    
    // Statistical operations should ignore masked values
    let mean = masked.mean().unwrap();
    assert_relative_eq!(mean, 3.0, epsilon = 1e-10); // Mean of [1, 3, 5]
    
    let sum = masked.sum().unwrap();
    assert_relative_eq!(sum, 9.0, epsilon = 1e-10); // Sum of [1, 3, 5]
    
    let min = masked.min().unwrap();
    assert_relative_eq!(min, 1.0, epsilon = 1e-10);
    
    let max = masked.max().unwrap();
    assert_relative_eq!(max, 5.0, epsilon = 1e-10);
    
    // Test with all values masked
    let all_masked = MaskedArray::masked_all(data.clone(), Some(0.0)).unwrap();
    assert!(all_masked.mean().is_none());
    assert!(all_masked.sum().is_none());
    assert!(all_masked.min().is_none());
    assert!(all_masked.max().is_none());
}

#[test]
fn test_masked_invalid() {
    // Create array with some NaN and Inf values
    let data = Array::from_vec(vec![1.0, f64::NAN, 3.0, f64::INFINITY, 5.0]);
    
    // Create masked array with NaN and Inf values masked
    let masked = MaskedArray::<f64>::masked_invalid(data.clone(), Some(0.0)).unwrap();
    
    // Should have 2 masked values (NaN and Inf)
    assert_eq!(masked.count_masked(), 2);
    
    // Get data with fill value
    let filled = masked.filled(None);
    assert_eq!(filled.to_vec(), vec![1.0, 0.0, 3.0, 0.0, 5.0]);
    
    // Statistical operations should work normally
    let mean = masked.mean().unwrap();
    assert_relative_eq!(mean, 3.0, epsilon = 1e-10); // Mean of [1, 3, 5]
}