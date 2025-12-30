#[cfg(test)]
mod unique_tests {
    use numrs2::prelude::*;

    #[test]
    fn test_unique_basic() {
        // Test basic unique functionality (no axis)
        let a = Array::from_vec(vec![1, 2, 3, 2, 1, 4]).reshape(&[6]);
        let result = unique(&a, None, None, None, None).unwrap();
        assert_eq!(result.values.to_vec(), vec![1, 2, 3, 4]);

        // Test with return_index
        let result = unique(&a, None, Some(true), None, None).unwrap();
        let (values, indices) = result.values_indices().unwrap();
        assert_eq!(values.to_vec(), vec![1, 2, 3, 4]);
        assert_eq!(indices.to_vec(), vec![0, 1, 2, 5]); // Positions where each unique value first appears

        // Test with return_counts
        let result = unique(&a, None, None, None, Some(true)).unwrap();
        let (values, counts) = result.values_counts().unwrap();
        assert_eq!(values.to_vec(), vec![1, 2, 3, 4]);
        assert_eq!(counts.to_vec(), vec![2, 2, 1, 1]); // 1 appears twice, 2 appears twice, etc.

        // Test with return_inverse
        let result = unique(&a, None, None, Some(true), None).unwrap();
        let (values, inverse) = result.values_inverse().unwrap();
        assert_eq!(values.to_vec(), vec![1, 2, 3, 4]);
        assert_eq!(inverse.to_vec(), vec![0, 1, 2, 1, 0, 3]); // Indices to reconstruct the original array

        // Verify we can reconstruct the original array
        let mut reconstructed = Vec::with_capacity(a.size());
        let values_vec = values.to_vec();
        for &idx in &inverse.to_vec() {
            reconstructed.push(values_vec[idx]);
        }
        assert_eq!(reconstructed, a.to_vec());
    }

    #[test]
    fn test_unique_2d_array() {
        // Test unique along axis 0 for a 2D array
        let a = Array::from_vec(vec![1, 2, 3, 1, 2, 3, 7, 8, 9]).reshape(&[3, 3]);

        // The array looks like:
        // [1, 2, 3]
        // [1, 2, 3]
        // [7, 8, 9]
        // Unique rows should be [1, 2, 3] and [7, 8, 9]

        let result = unique(&a, Some(0), None, None, None).unwrap();
        assert_eq!(result.values.shape(), vec![2, 3]);

        // Test with return_index for axis 0
        let result = unique(&a, Some(0), Some(true), None, None).unwrap();
        let (_values, indices) = result.values_indices().unwrap();
        assert_eq!(indices.to_vec(), vec![0, 2]); // First and third rows are unique

        // Test with return_counts for axis 0
        let result = unique(&a, Some(0), None, None, Some(true)).unwrap();
        let (_values, counts) = result.values_counts().unwrap();
        assert_eq!(counts.to_vec(), vec![2, 1]); // First row appears twice, third row once
    }

    #[test]
    fn test_unique_edge_cases() {
        // Test empty array
        let empty = Array::<i32>::from_vec(vec![]).reshape(&[0]);
        let result = unique(&empty, None, None, None, None).unwrap();
        let empty_vec: Vec<i32> = vec![];
        assert_eq!(result.values.to_vec(), empty_vec);

        // Test array with single element
        let single = Array::from_vec(vec![5]);
        let result = unique(&single, None, None, None, None).unwrap();
        assert_eq!(result.values.to_vec(), vec![5]);

        // Test array with all same elements
        let all_same = Array::from_vec(vec![7, 7, 7, 7]);
        let result = unique(&all_same, None, None, None, None).unwrap();
        assert_eq!(result.values.to_vec(), vec![7]);

        // Test with all return options
        let a = Array::from_vec(vec![3, 1, 2, 1, 3]);
        let result = unique(&a, None, Some(true), Some(true), Some(true)).unwrap();
        let (values, indices, inverse, counts) = result.values_indices_inverse_counts().unwrap();
        assert_eq!(values.to_vec(), vec![3, 1, 2]);
        assert_eq!(indices.to_vec(), vec![0, 1, 2]);
        assert_eq!(inverse.to_vec(), vec![0, 1, 2, 1, 0]);
        assert_eq!(counts.to_vec(), vec![2, 2, 1]);
    }
}
