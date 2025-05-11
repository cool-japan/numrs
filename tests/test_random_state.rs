use numrs2::array::Array;
use numrs2::random::state::RandomState;
use std::sync::{Arc, Mutex};

/// Tests for the RandomState struct, which forms the core of
/// the random number generation capabilities in NumRS2

#[test]
fn test_random_state_with_seed() {
    // Test that RandomState with the same seed produces the same output

    let rng1 = RandomState::with_seed(42);
    let rng2 = RandomState::with_seed(42);

    let arr1 = rng1.random::<f64>(&[10]).unwrap();
    let arr2 = rng2.random::<f64>(&[10]).unwrap();

    assert_eq!(
        arr1.to_vec(),
        arr2.to_vec(),
        "RandomState instances with same seed should produce identical outputs"
    );

    // Different seeds should produce different output
    let rng3 = RandomState::with_seed(43);
    let arr3 = rng3.random::<f64>(&[10]).unwrap();

    assert_ne!(
        arr1.to_vec(),
        arr3.to_vec(),
        "Different seeds should produce different outputs"
    );
}

#[test]
fn test_random_state_thread_safety() {
    // Test that RandomState can be used safely from multiple threads

    let rng = Arc::new(Mutex::new(RandomState::with_seed(42)));
    let threads = 4;
    let samples_per_thread = 1000;

    let mut handles = Vec::with_capacity(threads);

    for _ in 0..threads {
        let rng_clone = Arc::clone(&rng);

        let handle = std::thread::spawn(move || {
            let samples = samples_per_thread;
            let mut results = Vec::with_capacity(samples);

            for _ in 0..samples {
                let arr = rng_clone.lock().unwrap().random::<f64>(&[1]).unwrap();
                results.push(arr.to_vec()[0]);
            }

            results
        });

        handles.push(handle);
    }

    // Collect results
    let mut all_results = Vec::new();
    for handle in handles {
        let thread_results = handle.join().unwrap();
        all_results.extend(thread_results);
    }

    // We expect to have unique values across threads
    // as each thread acquires the lock, generates a value,
    // and releases the lock, advancing the state
    let unique_count = {
        // Use a vector-based approach instead of HashSet, since f64 doesn't implement Hash
        let mut rounded: Vec<i64> = all_results
            .iter()
            .map(|&val| (val * 1000000000.0).round() as i64)
            .collect();

        rounded.sort_unstable();

        // Count unique values
        let mut count = 0;
        let mut prev = None;

        for &val in &rounded {
            if prev != Some(val) {
                count += 1;
                prev = Some(val);
            }
        }

        count
    };

    // We should have close to threads * samples_per_thread unique values
    // allowing for a very small chance of collisions
    assert!(
        unique_count > all_results.len() * 99 / 100,
        "Expected most values to be unique, got {}/{}",
        unique_count,
        all_results.len()
    );
}

#[test]
fn test_error_handling() {
    // Test that RandomState properly validates parameters and returns appropriate errors

    let rng = RandomState::new();

    // Test normal distribution with invalid standard deviation
    let result = rng.normal(0.0, -1.0, &[10]);
    assert!(
        result.is_err(),
        "Normal distribution with negative std should fail"
    );

    if let Err(e) = result {
        assert!(
            e.to_string()
                .contains("Standard deviation must be positive"),
            "Error message should mention standard deviation"
        );
    } else {
        panic!("Expected error was not returned");
    }

    // Test gamma distribution with invalid parameters
    let result = rng.gamma(0.0, 1.0, &[10]);
    assert!(
        result.is_err(),
        "Gamma distribution with shape=0 should fail"
    );

    let result = rng.gamma(1.0, -1.0, &[10]);
    assert!(
        result.is_err(),
        "Gamma distribution with negative scale should fail"
    );

    // Test beta distribution with invalid parameters
    let result = rng.beta(-1.0, 1.0, &[10]);
    assert!(
        result.is_err(),
        "Beta distribution with negative alpha should fail"
    );

    let result = rng.beta(1.0, 0.0, &[10]);
    assert!(result.is_err(), "Beta distribution with beta=0 should fail");
}

#[test]
fn test_distribution_shape_handling() {
    // Test that RandomState correctly handles different array shapes

    let rng = RandomState::with_seed(42);

    // Test with 1D array
    let arr1d = rng.random::<f64>(&[10]).unwrap();
    assert_eq!(arr1d.shape(), vec![10], "1D shape should be preserved");

    // Test with 2D array
    let arr2d = rng.random::<f64>(&[3, 4]).unwrap();
    assert_eq!(arr2d.shape(), vec![3, 4], "2D shape should be preserved");

    // Test with 3D array
    let arr3d = rng.random::<f64>(&[2, 3, 4]).unwrap();
    assert_eq!(arr3d.shape(), vec![2, 3, 4], "3D shape should be preserved");

    // Test with empty shape (should create scalar)
    let arr0d = rng.random::<f64>(&[]).unwrap();
    assert_eq!(
        arr0d.shape(),
        Vec::<usize>::new(),
        "Scalar shape should be preserved"
    );

    // Test with zero-size dimension
    // Note: In this implementation, an array with a zero dimension
    // is allowed and returns an empty array
    let result = rng.random::<f64>(&[0, 5]);
    assert!(result.is_ok(), "Zero-sized dimension should be allowed");

    // The result should be an empty array with shape [0, 5]
    let arr_empty = result.unwrap();
    assert_eq!(
        arr_empty.shape(),
        vec![0, 5],
        "Shape should be preserved for empty arrays"
    );
    assert_eq!(
        arr_empty.to_vec().len(),
        0,
        "Empty array should have no elements"
    );
}

#[test]
fn test_choice_and_shuffle() {
    // Test RandomState's choice and shuffle operations

    let rng = RandomState::with_seed(42);

    // Create a test array
    let arr = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    // Test shuffle
    let mut arr_copy = arr.clone();
    let shuffle_result = rng.shuffle(&mut arr_copy);
    assert!(shuffle_result.is_ok(), "Shuffle operation should succeed");

    // Shuffled array should have same elements but in different order
    assert_eq!(
        arr_copy.shape(),
        arr.shape(),
        "Shape should be preserved after shuffle"
    );

    let orig_sum: f64 = arr.to_vec().iter().sum();
    let shuffled_sum: f64 = arr_copy.to_vec().iter().sum();
    assert_eq!(
        orig_sum, shuffled_sum,
        "Sum should be preserved after shuffle"
    );

    // Very unlikely that shuffling would give the same arrangement
    assert_ne!(arr.to_vec(), arr_copy.to_vec(), "Array should be shuffled");

    // Test choice with replacement
    let choice_result = rng.choice(&arr, Some(100), Some(true));
    assert!(
        choice_result.is_ok(),
        "Choice with replacement should succeed"
    );

    let choices = choice_result.unwrap();
    assert_eq!(
        choices.shape(),
        vec![100],
        "Choice shape should be as requested"
    );

    // All choices should be values from the original array
    for &val in choices.to_vec().iter() {
        assert!(
            arr.to_vec().contains(&val),
            "Choice should only return values from the original array"
        );
    }

    // Test choice without replacement
    let choice_result = rng.choice(&arr, Some(5), Some(false));
    assert!(
        choice_result.is_ok(),
        "Choice without replacement should succeed"
    );

    let choices = choice_result.unwrap();
    assert_eq!(
        choices.shape(),
        vec![5],
        "Choice shape should be as requested"
    );

    // Should have all 5 elements from original array
    let mut choices_vec = choices.to_vec();
    choices_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut orig_vec = arr.to_vec();
    orig_vec.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_eq!(
        choices_vec, orig_vec,
        "Choice without replacement should return all original elements"
    );

    // Test choice without replacement with size > array size
    let choice_result = rng.choice(&arr, Some(10), Some(false));
    assert!(
        choice_result.is_err(),
        "Choice without replacement with size > array size should error"
    );
}

#[test]
fn test_permutation() {
    // Test RandomState's permutation function

    let rng = RandomState::with_seed(42);

    // Generate a permutation of integers from 0 to 9
    let perm_result = rng.permutation::<usize>(10);
    assert!(perm_result.is_ok(), "Permutation should succeed");

    let perm = perm_result.unwrap();
    assert_eq!(
        perm.shape(),
        vec![10],
        "Permutation should have requested size"
    );

    // Permutation should contain each integer from 0 to 9 exactly once
    let mut perm_vec = perm.to_vec();
    perm_vec.sort();

    for i in 0..10 {
        assert_eq!(
            perm_vec[i], i,
            "Permutation should contain each value from 0 to n-1"
        );
    }
}

#[test]
fn test_dirichlet_distribution() {
    // Test properties of the Dirichlet distribution

    let rng = RandomState::with_seed(42);

    // Create alpha parameters
    let alpha = vec![1.0, 2.0, 3.0];

    // Generate samples
    let samples_result = rng.dirichlet::<f64>(&alpha, &[1000]);
    assert!(samples_result.is_ok(), "Dirichlet sampling should succeed");

    let samples = samples_result.unwrap();
    assert_eq!(
        samples.shape(),
        vec![1000, 3],
        "Dirichlet should produce samples with shape [size, k]"
    );

    // Each sample should sum to approximately 1
    let data = samples.to_vec();
    for i in 0..1000 {
        let sum = data[i * 3] + data[i * 3 + 1] + data[i * 3 + 2];
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Each Dirichlet sample should sum to 1, got {}",
            sum
        );
    }

    // Calculate means
    let mut means = vec![0.0; 3];
    for i in 0..1000 {
        means[0] += data[i * 3] / 1000.0;
        means[1] += data[i * 3 + 1] / 1000.0;
        means[2] += data[i * 3 + 2] / 1000.0;
    }

    // Expected means for Dirichlet are alpha_i / sum(alpha)
    let alpha_sum: f64 = alpha.iter().sum();
    let expected_means: Vec<f64> = alpha.iter().map(|&a| a / alpha_sum).collect();

    for i in 0..3 {
        assert!(
            (means[i] - expected_means[i]).abs() < 0.05,
            "Component {} mean should be close to {}, got {}",
            i,
            expected_means[i],
            means[i]
        );
    }
}

#[test]
fn test_multivariate_normal_distribution() {
    // Test properties of the multivariate normal distribution

    let rng = RandomState::with_seed(42);

    // Define mean and covariance matrix
    let mean = vec![1.0, 2.0];
    let cov_data_orig = vec![1.0, 0.5, 0.5, 2.0]; // 2x2 covariance matrix
    let cov = Array::from_vec(cov_data_orig.clone()).reshape(&[2, 2]);

    // Generate samples
    let samples_result = rng.multivariate_normal::<f64>(&mean, &cov, Some(&[1000]));
    assert!(
        samples_result.is_ok(),
        "Multivariate normal sampling should succeed"
    );

    let samples = samples_result.unwrap();
    assert_eq!(
        samples.shape(),
        vec![1000, 2],
        "Multivariate normal should produce samples with shape [size, d]"
    );

    // Calculate sample mean
    let data = samples.to_vec();
    let mut sample_mean = vec![0.0; 2];
    for i in 0..1000 {
        sample_mean[0] += data[i * 2] / 1000.0;
        sample_mean[1] += data[i * 2 + 1] / 1000.0;
    }

    // Calculate sample covariance
    let mut sample_cov = vec![0.0; 4];
    for i in 0..1000 {
        let diff0 = data[i * 2] - sample_mean[0];
        let diff1 = data[i * 2 + 1] - sample_mean[1];

        sample_cov[0] += diff0 * diff0 / 999.0; // Var(X)
        sample_cov[1] += diff0 * diff1 / 999.0; // Cov(X,Y)
        sample_cov[2] += diff1 * diff0 / 999.0; // Cov(Y,X)
        sample_cov[3] += diff1 * diff1 / 999.0; // Var(Y)
    }

    // Verify mean
    for i in 0..2 {
        assert!(
            (sample_mean[i] - mean[i]).abs() < 0.1,
            "Component {} mean should be close to {}, got {}",
            i,
            mean[i],
            sample_mean[i]
        );
    }

    // Verify covariance
    for i in 0..4 {
        assert!(
            (sample_cov[i] - cov_data_orig[i]).abs() < 0.3,
            "Covariance element [{}] should be close to {}, got {}",
            i,
            cov_data_orig[i],
            sample_cov[i]
        );
    }
}
