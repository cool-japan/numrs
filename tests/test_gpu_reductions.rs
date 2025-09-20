//! Tests for GPU reduction operations

#[cfg(feature = "gpu")]
mod gpu_reduction_tests {
    use numrs2::array::Array;
    use numrs2::error::Result;
    use numrs2::gpu::{self, GpuArray, GpuContext};

    fn create_gpu_context() -> Result<GpuContext> {
        // Create a simple runtime for the async GPU context creation
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| numrs2::error::NumRs2Error::RuntimeError(e.to_string()))?;

        runtime.block_on(GpuContext::new())
    }

    #[test]
    #[ignore = "GPU tests require proper GPU support"]
    fn test_sum_f32() -> Result<()> {
        let _context = create_gpu_context()?;

        // Test data
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cpu_array = Array::from_vec(data.clone());
        let gpu_array = GpuArray::from_array(&cpu_array)?;

        // Compute sum on GPU
        let gpu_sum = gpu::sum_f32(&gpu_array)?;

        // Verify result
        let expected_sum: f32 = data.iter().sum();
        assert!(
            (gpu_sum - expected_sum).abs() < 1e-6,
            "Sum mismatch: expected {}, got {}",
            expected_sum,
            gpu_sum
        );

        Ok(())
    }

    #[test]
    #[ignore = "F64 GPU operations not supported on all hardware"]
    fn test_sum_f64() -> Result<()> {
        let _context = create_gpu_context()?;

        // Test data
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cpu_array = Array::from_vec(data.clone());
        let gpu_array = GpuArray::from_array(&cpu_array)?;

        // Compute sum on GPU
        let gpu_sum = gpu::sum_f64(&gpu_array)?;

        // Verify result
        let expected_sum: f64 = data.iter().sum();
        assert!(
            (gpu_sum - expected_sum).abs() < 1e-12,
            "Sum mismatch: expected {}, got {}",
            expected_sum,
            gpu_sum
        );

        Ok(())
    }

    #[test]
    #[ignore = "GPU tests require proper GPU support"]
    fn test_mean_f32() -> Result<()> {
        let _context = create_gpu_context()?;

        // Test data
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cpu_array = Array::from_vec(data.clone());
        let gpu_array = GpuArray::from_array(&cpu_array)?;

        // Compute mean on GPU
        let gpu_mean = gpu::mean_f32(&gpu_array)?;

        // Verify result
        let expected_mean = data.iter().sum::<f32>() / data.len() as f32;
        assert!(
            (gpu_mean - expected_mean).abs() < 1e-6,
            "Mean mismatch: expected {}, got {}",
            expected_mean,
            gpu_mean
        );

        Ok(())
    }

    #[test]
    #[ignore = "F64 GPU operations not supported on all hardware"]
    fn test_mean_f64() -> Result<()> {
        let _context = create_gpu_context()?;

        // Test data
        let data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cpu_array = Array::from_vec(data.clone());
        let gpu_array = GpuArray::from_array(&cpu_array)?;

        // Compute mean on GPU
        let gpu_mean = gpu::mean_f64(&gpu_array)?;

        // Verify result
        let expected_mean = data.iter().sum::<f64>() / data.len() as f64;
        assert!(
            (gpu_mean - expected_mean).abs() < 1e-12,
            "Mean mismatch: expected {}, got {}",
            expected_mean,
            gpu_mean
        );

        Ok(())
    }

    #[test]
    #[ignore = "GPU tests require proper GPU support"]
    fn test_max_f32() -> Result<()> {
        let _context = create_gpu_context()?;

        // Test data with various values including negative
        let data = vec![3.5f32, -2.0, 8.1, 1.0, -5.5, 7.9, 0.0, 4.2];
        let cpu_array = Array::from_vec(data.clone());
        let gpu_array = GpuArray::from_array(&cpu_array)?;

        // Compute max on GPU
        let gpu_max = gpu::max_f32(&gpu_array)?;

        // Verify result
        let expected_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            (gpu_max - expected_max).abs() < 1e-6,
            "Max mismatch: expected {}, got {}",
            expected_max,
            gpu_max
        );

        Ok(())
    }

    #[test]
    #[ignore = "F64 GPU operations not supported on all hardware"]
    fn test_max_f64() -> Result<()> {
        let _context = create_gpu_context()?;

        // Test data with various values including negative
        let data = vec![3.5f64, -2.0, 8.1, 1.0, -5.5, 7.9, 0.0, 4.2];
        let cpu_array = Array::from_vec(data.clone());
        let gpu_array = GpuArray::from_array(&cpu_array)?;

        // Compute max on GPU
        let gpu_max = gpu::max_f64(&gpu_array)?;

        // Verify result
        let expected_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (gpu_max - expected_max).abs() < 1e-12,
            "Max mismatch: expected {}, got {}",
            expected_max,
            gpu_max
        );

        Ok(())
    }

    #[test]
    #[ignore = "GPU tests require proper GPU support"]
    fn test_min_f32() -> Result<()> {
        let _context = create_gpu_context()?;

        // Test data with various values including negative
        let data = vec![3.5f32, -2.0, 8.1, 1.0, -5.5, 7.9, 0.0, 4.2];
        let cpu_array = Array::from_vec(data.clone());
        let gpu_array = GpuArray::from_array(&cpu_array)?;

        // Compute min on GPU
        let gpu_min = gpu::min_f32(&gpu_array)?;

        // Verify result
        let expected_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(
            (gpu_min - expected_min).abs() < 1e-6,
            "Min mismatch: expected {}, got {}",
            expected_min,
            gpu_min
        );

        Ok(())
    }

    #[test]
    #[ignore = "F64 GPU operations not supported on all hardware"]
    fn test_min_f64() -> Result<()> {
        let _context = create_gpu_context()?;

        // Test data with various values including negative
        let data = vec![3.5f64, -2.0, 8.1, 1.0, -5.5, 7.9, 0.0, 4.2];
        let cpu_array = Array::from_vec(data.clone());
        let gpu_array = GpuArray::from_array(&cpu_array)?;

        // Compute min on GPU
        let gpu_min = gpu::min_f64(&gpu_array)?;

        // Verify result
        let expected_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            (gpu_min - expected_min).abs() < 1e-12,
            "Min mismatch: expected {}, got {}",
            expected_min,
            gpu_min
        );

        Ok(())
    }

    #[test]
    #[ignore = "GPU tests require proper GPU support"]
    fn test_large_array_reductions() -> Result<()> {
        let _context = create_gpu_context()?;

        // Test with a large array to ensure workgroup handling is correct
        let size = 100_000;
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let cpu_array = Array::from_vec(data.clone());
        let gpu_array = GpuArray::from_array(&cpu_array)?;

        // Test sum
        let gpu_sum = gpu::sum_f32(&gpu_array)?;
        let expected_sum = (size * (size - 1)) as f32 / 2.0;
        assert!(
            (gpu_sum - expected_sum).abs() / expected_sum < 1e-5,
            "Large array sum mismatch: expected {}, got {}",
            expected_sum,
            gpu_sum
        );

        // Test mean
        let gpu_mean = gpu::mean_f32(&gpu_array)?;
        let expected_mean = (size - 1) as f32 / 2.0;
        assert!(
            (gpu_mean - expected_mean).abs() < 1e-3,
            "Large array mean mismatch: expected {}, got {}",
            expected_mean,
            gpu_mean
        );

        // Test max
        let gpu_max = gpu::max_f32(&gpu_array)?;
        let expected_max = (size - 1) as f32;
        assert!(
            (gpu_max - expected_max).abs() < 1e-6,
            "Large array max mismatch: expected {}, got {}",
            expected_max,
            gpu_max
        );

        // Test min
        let gpu_min = gpu::min_f32(&gpu_array)?;
        let expected_min = 0.0f32;
        assert!(
            (gpu_min - expected_min).abs() < 1e-6,
            "Large array min mismatch: expected {}, got {}",
            expected_min,
            gpu_min
        );

        Ok(())
    }
}
