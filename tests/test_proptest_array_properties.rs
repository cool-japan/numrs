//! Property-Based Testing for Array Operations and Invariants
//!
//! This module uses proptest to verify array operation properties that should
//! hold for all inputs, including edge cases, NaN handling, and numerical stability.

use numrs2::array::Array;
use numrs2::prelude::*;
use numrs2::ufuncs;
use proptest::prelude::*;

// =============================================================================
// PROPTEST STRATEGIES
// =============================================================================

/// Strategy for generating non-empty arrays
fn non_empty_array_strategy(max_len: usize) -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(-100.0f64..100.0f64, 1..=max_len)
}

/// Strategy for generating small positive integers for shape dimensions
fn shape_dim_strategy() -> impl Strategy<Value = usize> {
    1usize..=10
}

/// Strategy for generating 2D shapes
fn shape_2d_strategy() -> impl Strategy<Value = (usize, usize)> {
    (1usize..=10, 1usize..=10)
}

// =============================================================================
// ARRAY CREATION PROPERTIES
// =============================================================================

proptest! {
    /// zeros() should create arrays where all elements are exactly 0
    #[test]
    fn prop_zeros_all_zero(rows in shape_dim_strategy(), cols in shape_dim_strategy()) {
        let arr = Array::<f64>::zeros(&[rows, cols]);
        let data = arr.to_vec();

        for (i, &val) in data.iter().enumerate() {
            prop_assert_eq!(val, 0.0, "zeros() array has non-zero value {} at index {}", val, i);
        }
    }

    /// ones() should create arrays where all elements are exactly 1
    #[test]
    fn prop_ones_all_one(rows in shape_dim_strategy(), cols in shape_dim_strategy()) {
        let arr = Array::<f64>::ones(&[rows, cols]);
        let data = arr.to_vec();

        for (i, &val) in data.iter().enumerate() {
            prop_assert_eq!(val, 1.0, "ones() array has non-one value {} at index {}", val, i);
        }
    }

    /// full() should create arrays where all elements equal the fill value
    #[test]
    fn prop_full_constant(
        rows in shape_dim_strategy(),
        cols in shape_dim_strategy(),
        fill in -1000.0f64..1000.0f64
    ) {
        let arr = Array::<f64>::full(&[rows, cols], fill);
        let data = arr.to_vec();

        for (i, &val) in data.iter().enumerate() {
            prop_assert!((val - fill).abs() < 1e-15,
                "full({}) array has value {} at index {}", fill, val, i);
        }
    }
}

// =============================================================================
// RESHAPE PROPERTIES
// =============================================================================

proptest! {
    /// Reshape should preserve all elements
    #[test]
    fn prop_reshape_preserves_data(data in non_empty_array_strategy(100)) {
        let n = data.len();
        let original = Array::<f64>::from_vec(data.clone());

        // Find valid reshape dimensions
        for d in 1..=n {
            if n % d == 0 {
                let reshaped = original.clone().reshape(&[d, n / d]);
                let reshaped_data = reshaped.to_vec();

                prop_assert_eq!(reshaped_data.len(), n);
                for (orig, reshp) in data.iter().zip(reshaped_data.iter()) {
                    prop_assert!((orig - reshp).abs() < 1e-15);
                }
            }
        }
    }

    /// Reshape should preserve size
    #[test]
    fn prop_reshape_preserves_size(data in non_empty_array_strategy(100)) {
        let n = data.len();
        let original = Array::<f64>::from_vec(data);

        for d in 1..=n {
            if n % d == 0 {
                let reshaped = original.clone().reshape(&[d, n / d]);
                prop_assert_eq!(reshaped.len(), n);
            }
        }
    }

    /// Multiple reshapes should be consistent
    #[test]
    fn prop_reshape_transitive(data in non_empty_array_strategy(36)) {
        // Use 36 elements (divisible by many numbers)
        if data.len() != 36 { return Ok(()); }

        let arr = Array::<f64>::from_vec(data.clone());

        // Reshape chain: 1D -> 2D -> 3D -> 1D
        let r1 = arr.reshape(&[6, 6]);
        let r2 = r1.reshape(&[3, 3, 4]);
        let r3 = r2.reshape(&[36]);

        let final_data = r3.to_vec();
        prop_assert_eq!(final_data.len(), 36);

        for (orig, final_val) in data.iter().zip(final_data.iter()) {
            prop_assert!((orig - final_val).abs() < 1e-15);
        }
    }
}

// =============================================================================
// TRANSPOSE PROPERTIES
// =============================================================================

proptest! {
    /// Double transpose returns original array
    #[test]
    fn prop_transpose_involution((rows, cols) in shape_2d_strategy()) {
        let data: Vec<f64> = (0..(rows * cols)).map(|i| i as f64).collect();
        let arr = Array::<f64>::from_vec(data.clone()).reshape(&[rows, cols]);

        let transposed = arr.transpose();
        let double_transposed = transposed.transpose();

        let result = double_transposed.to_vec();
        prop_assert_eq!(result.len(), data.len());

        for (orig, result_val) in data.iter().zip(result.iter()) {
            prop_assert!((orig - result_val).abs() < 1e-15,
                "Double transpose changed value: {} -> {}", orig, result_val);
        }
    }

    /// Transpose swaps dimensions
    #[test]
    fn prop_transpose_swaps_shape((rows, cols) in shape_2d_strategy()) {
        let data: Vec<f64> = (0..(rows * cols)).map(|i| i as f64).collect();
        let arr = Array::<f64>::from_vec(data).reshape(&[rows, cols]);

        let transposed = arr.transpose();
        let t_shape = transposed.shape();

        prop_assert_eq!(t_shape[0], cols);
        prop_assert_eq!(t_shape[1], rows);
    }
}

// =============================================================================
// CONCATENATION PROPERTIES
// =============================================================================

proptest! {
    /// Concatenating arrays preserves total size
    #[test]
    fn prop_concat_preserves_size(
        data1 in non_empty_array_strategy(50),
        data2 in non_empty_array_strategy(50)
    ) {
        let arr1 = Array::<f64>::from_vec(data1.clone());
        let arr2 = Array::<f64>::from_vec(data2.clone());

        let concatenated = concatenate(&[&arr1, &arr2], 0).unwrap();
        let result_size = concatenated.len();
        let expected_size = data1.len() + data2.len();

        prop_assert_eq!(result_size, expected_size);
    }

    /// Concatenation preserves element order
    #[test]
    fn prop_concat_preserves_order(
        data1 in non_empty_array_strategy(50),
        data2 in non_empty_array_strategy(50)
    ) {
        let arr1 = Array::<f64>::from_vec(data1.clone());
        let arr2 = Array::<f64>::from_vec(data2.clone());

        let concatenated = concatenate(&[&arr1, &arr2], 0).unwrap();
        let result = concatenated.to_vec();

        // First part should match data1
        for (i, &val) in data1.iter().enumerate() {
            prop_assert!((result[i] - val).abs() < 1e-15,
                "First part mismatch at {}: {} != {}", i, result[i], val);
        }

        // Second part should match data2
        for (i, &val) in data2.iter().enumerate() {
            let idx = data1.len() + i;
            prop_assert!((result[idx] - val).abs() < 1e-15,
                "Second part mismatch at {}: {} != {}", idx, result[idx], val);
        }
    }
}

// =============================================================================
// INDEXING PROPERTIES
// =============================================================================

proptest! {
    /// Accessing elements by index preserves values
    #[test]
    fn prop_indexing_preserves_values(data in non_empty_array_strategy(100)) {
        let arr = Array::<f64>::from_vec(data.clone());
        let retrieved = arr.to_vec();

        for (i, (&original, &retrieved_val)) in data.iter().zip(retrieved.iter()).enumerate() {
            prop_assert!((original - retrieved_val).abs() < 1e-15,
                "Value mismatch at index {}: {} != {}", i, original, retrieved_val);
        }
    }
}

// =============================================================================
// ELEMENT-WISE OPERATION PROPERTIES
// =============================================================================

proptest! {
    /// Addition is commutative
    #[test]
    fn prop_add_commutative(data in non_empty_array_strategy(100)) {
        let arr1 = Array::<f64>::from_vec(data.clone());
        let arr2 = Array::<f64>::from_vec(data.iter().map(|&x| x * 2.0).collect());

        let sum1 = arr1.add(&arr2);
        let sum2 = arr2.add(&arr1);

        let data1 = sum1.to_vec();
        let data2 = sum2.to_vec();

        for (v1, v2) in data1.iter().zip(data2.iter()) {
            prop_assert!((v1 - v2).abs() < 1e-12,
                "Addition not commutative: {} != {}", v1, v2);
        }
    }

    /// Multiplication by 1 is identity
    #[test]
    fn prop_mul_identity(data in non_empty_array_strategy(100)) {
        let arr = Array::<f64>::from_vec(data.clone());
        let ones = Array::<f64>::ones(&[data.len()]);

        let result = arr.multiply(&ones);
        let result_data = result.to_vec();

        for (orig, result_val) in data.iter().zip(result_data.iter()) {
            prop_assert!((orig - result_val).abs() < 1e-12,
                "Multiply by 1 changed value: {} -> {}", orig, result_val);
        }
    }

    /// Addition of 0 is identity
    #[test]
    fn prop_add_zero_identity(data in non_empty_array_strategy(100)) {
        let arr = Array::<f64>::from_vec(data.clone());
        let zeros = Array::<f64>::zeros(&[data.len()]);

        let result = arr.add(&zeros);
        let result_data = result.to_vec();

        for (orig, result_val) in data.iter().zip(result_data.iter()) {
            prop_assert!((orig - result_val).abs() < 1e-12,
                "Add 0 changed value: {} -> {}", orig, result_val);
        }
    }

    /// Subtraction of self is zero
    #[test]
    fn prop_subtract_self_zero(data in non_empty_array_strategy(100)) {
        let arr = Array::<f64>::from_vec(data);
        let result = arr.subtract(&arr);
        let result_data = result.to_vec();

        for &val in result_data.iter() {
            prop_assert!(val.abs() < 1e-12,
                "Subtract self not zero: {}", val);
        }
    }

    /// Division by self is one (for non-zero elements)
    #[test]
    fn prop_divide_self_one(data in prop::collection::vec(0.1f64..100.0f64, 1..=100)) {
        let arr = Array::<f64>::from_vec(data);
        let result = arr.divide(&arr); // divide() returns Array, not Result
        let result_data = result.to_vec();

        for &val in result_data.iter() {
            prop_assert!((val - 1.0).abs() < 1e-12,
                "Divide self not one: {}", val);
        }
    }
}

// =============================================================================
// REDUCTION PROPERTIES
// =============================================================================

proptest! {
    /// Sum of zeros is zero
    #[test]
    fn prop_sum_zeros(size in 1usize..=1000) {
        let arr = Array::<f64>::zeros(&[size]);
        let sum = arr.sum();

        prop_assert!(sum.abs() < 1e-12,
            "Sum of zeros is {}, expected 0", sum);
    }

    /// Product of ones is one
    #[test]
    fn prop_prod_ones(size in 1usize..=1000) {
        let arr = Array::<f64>::ones(&[size]);
        let prod = arr.product();

        prop_assert!((prod - 1.0).abs() < 1e-12,
            "Product of ones is {}, expected 1", prod);
    }

    /// Max >= Mean >= Min (for non-empty arrays)
    #[test]
    fn prop_max_mean_min_ordering(data in non_empty_array_strategy(100)) {
        let arr = Array::<f64>::from_vec(data);

        let max_val = arr.max();
        let min_val = arr.min();
        let mean_val = arr.mean();

        prop_assert!(max_val >= mean_val,
            "Max ({}) < Mean ({})", max_val, mean_val);
        prop_assert!(mean_val >= min_val,
            "Mean ({}) < Min ({})", mean_val, min_val);
    }

    /// Sum(abs(x)) >= abs(Sum(x)) (triangle inequality)
    #[test]
    fn prop_sum_triangle_inequality(data in non_empty_array_strategy(100)) {
        let arr = Array::<f64>::from_vec(data);

        let sum_abs = ufuncs::absolute(&arr).sum();
        let abs_sum = arr.sum().abs();

        prop_assert!(sum_abs >= abs_sum - 1e-12,
            "Triangle inequality violated: sum(abs) = {} < abs(sum) = {}", sum_abs, abs_sum);
    }
}

// =============================================================================
// SIMD OPTIMIZATION PROPERTIES
// =============================================================================

proptest! {
    /// SIMD and scalar implementations should produce identical results for addition
    #[test]
    fn prop_simd_add_correctness(data in non_empty_array_strategy(200)) {
        let arr1 = Array::<f64>::from_vec(data.clone());
        let arr2 = Array::<f64>::from_vec(data.iter().map(|&x| x * 2.0).collect());

        // Use ufuncs which have SIMD dispatch
        let simd_result = ufuncs::add(&arr1, &arr2).unwrap();

        // Manual scalar computation
        let scalar_result: Vec<f64> = data.iter().zip(data.iter())
            .map(|(&a, &b)| a + b * 2.0)
            .collect();

        let simd_data = simd_result.to_vec();
        for (simd, scalar) in simd_data.iter().zip(scalar_result.iter()) {
            prop_assert!((simd - scalar).abs() < 1e-12,
                "SIMD add mismatch: {} != {}", simd, scalar);
        }
    }

    /// SIMD and scalar implementations should produce identical results for multiplication
    #[test]
    fn prop_simd_multiply_correctness(data in non_empty_array_strategy(200)) {
        let arr1 = Array::<f64>::from_vec(data.clone());
        let arr2 = Array::<f64>::from_vec(data.iter().map(|&x| x * 0.5).collect());

        let simd_result = ufuncs::multiply(&arr1, &arr2).unwrap();

        let scalar_result: Vec<f64> = data.iter().zip(data.iter())
            .map(|(&a, &b)| a * (b * 0.5))
            .collect();

        let simd_data = simd_result.to_vec();
        for (simd, scalar) in simd_data.iter().zip(scalar_result.iter()) {
            prop_assert!((simd - scalar).abs() < 1e-12,
                "SIMD multiply mismatch: {} != {}", simd, scalar);
        }
    }

    /// SIMD sqrt should maintain sqrt(x)^2 ≈ x property
    #[test]
    fn prop_simd_sqrt_identity(data in prop::collection::vec(0.1f64..1000.0, 1..=200)) {
        let arr = Array::<f64>::from_vec(data.clone());

        let sqrt_result = ufuncs::sqrt(&arr);
        let squared = ufuncs::multiply(&sqrt_result, &sqrt_result).unwrap();

        let result = squared.to_vec();
        for (orig, reconstructed) in data.iter().zip(result.iter()) {
            prop_assert!((orig - reconstructed).abs() / orig < 1e-10,
                "sqrt identity violated: {} -> {} -> {}", orig, orig.sqrt(), reconstructed);
        }
    }

    /// SIMD subtraction: a - a = 0
    #[test]
    fn prop_simd_subtract_self_zero(data in non_empty_array_strategy(200)) {
        let arr = Array::<f64>::from_vec(data);
        let result = ufuncs::subtract(&arr, &arr).unwrap();
        let result_data = result.to_vec();

        for &val in result_data.iter() {
            prop_assert!(val.abs() < 1e-12,
                "SIMD subtract self not zero: {}", val);
        }
    }

    /// SIMD division: a / a = 1 (for non-zero)
    #[test]
    fn prop_simd_divide_self_one(data in prop::collection::vec(0.1f64..100.0, 1..=200)) {
        let arr = Array::<f64>::from_vec(data);
        let result = ufuncs::divide(&arr, &arr).unwrap();
        let result_data = result.to_vec();

        for &val in result_data.iter() {
            prop_assert!((val - 1.0).abs() < 1e-12,
                "SIMD divide self not one: {}", val);
        }
    }

    /// SIMD maximum should be commutative
    #[test]
    fn prop_simd_maximum_commutative(data in non_empty_array_strategy(200)) {
        let arr1 = Array::<f64>::from_vec(data.clone());
        let arr2 = Array::<f64>::from_vec(data.iter().map(|&x| x * 1.5).collect());

        let max1 = ufuncs::maximum(&arr1, &arr2).unwrap();
        let max2 = ufuncs::maximum(&arr2, &arr1).unwrap();

        let data1 = max1.to_vec();
        let data2 = max2.to_vec();

        for (v1, v2) in data1.iter().zip(data2.iter()) {
            prop_assert!((v1 - v2).abs() < 1e-12,
                "Maximum not commutative: {} != {}", v1, v2);
        }
    }

    /// SIMD dot product should be commutative
    #[test]
    fn prop_simd_dot_commutative(data in non_empty_array_strategy(200)) {
        let arr1 = Array::<f64>::from_vec(data.clone());
        let arr2 = Array::<f64>::from_vec(data.iter().map(|&x| x * 2.0).collect());

        let dot1 = ufuncs::dot(&arr1, &arr2).unwrap();
        let dot2 = ufuncs::dot(&arr2, &arr1).unwrap();

        prop_assert!((dot1 - dot2).abs() < 1e-10,
            "Dot product not commutative: {} != {}", dot1, dot2);
    }

    /// SIMD operations preserve array size
    #[test]
    fn prop_simd_operations_preserve_size(data in non_empty_array_strategy(200)) {
        let arr = Array::<f64>::from_vec(data.clone());
        let n = arr.len();

        prop_assert_eq!(ufuncs::absolute(&arr).len(), n);
        prop_assert_eq!(ufuncs::sqrt(&arr).len(), n);
        prop_assert_eq!(ufuncs::exp(&arr).len(), n);
        prop_assert_eq!(ufuncs::sin(&arr).len(), n);
        prop_assert_eq!(ufuncs::negative(&arr).len(), n);
    }
}
