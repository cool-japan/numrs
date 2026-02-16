//! Cache Alignment Validation Tests
//!
//! This module tests that hot data structures are properly aligned to cache line
//! boundaries (64 bytes) to prevent false sharing and optimize cache performance.

#[cfg(test)]
mod alignment_tests {
    use std::mem;

    #[test]
    fn test_parallel_config_alignment() {
        use numrs2::parallel_optimize::ParallelConfig;

        // Verify ParallelConfig is aligned to 64 bytes
        let alignment = mem::align_of::<ParallelConfig>();
        assert_eq!(
            alignment, 64,
            "ParallelConfig should be aligned to 64 bytes (cache line size), got {}",
            alignment
        );

        // Verify at runtime
        let config = ParallelConfig::default();
        let addr = &config as *const _ as usize;
        assert_eq!(
            addr % 64,
            0,
            "ParallelConfig instance should be aligned to 64-byte boundary"
        );
    }

    #[test]
    fn test_broadcast_engine_alignment() {
        use numrs2::arrays::broadcasting::{BroadcastConfig, BroadcastEngine};

        let alignment = mem::align_of::<BroadcastEngine>();
        assert_eq!(
            alignment, 64,
            "BroadcastEngine should be aligned to 64 bytes, got {}",
            alignment
        );

        let engine = BroadcastEngine::new(BroadcastConfig::default());
        let addr = &engine as *const _ as usize;
        assert_eq!(
            addr % 64,
            0,
            "BroadcastEngine instance should be aligned to 64-byte boundary"
        );
    }

    #[test]
    fn test_stride_calculator_alignment() {
        use numrs2::arrays::stride_optimization::StrideCalculator;

        let alignment = mem::align_of::<StrideCalculator>();
        assert_eq!(
            alignment, 64,
            "StrideCalculator should be aligned to 64 bytes, got {}",
            alignment
        );

        let calculator = StrideCalculator::default();
        let addr = &calculator as *const _ as usize;
        assert_eq!(
            addr % 64,
            0,
            "StrideCalculator instance should be aligned to 64-byte boundary"
        );
    }

    #[test]
    fn test_fancy_index_engine_alignment() {
        use numrs2::arrays::fancy_indexing::FancyIndexEngine;

        let alignment = mem::align_of::<FancyIndexEngine>();
        assert_eq!(
            alignment, 64,
            "FancyIndexEngine should be aligned to 64 bytes, got {}",
            alignment
        );

        let engine = FancyIndexEngine::default();
        let addr = &engine as *const _ as usize;
        assert_eq!(
            addr % 64,
            0,
            "FancyIndexEngine instance should be aligned to 64-byte boundary"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_memory_pool_alignment() {
        use numrs2::gpu::memory::GpuMemoryPool;

        let alignment = mem::align_of::<GpuMemoryPool>();
        assert_eq!(
            alignment, 64,
            "GpuMemoryPool should be aligned to 64 bytes, got {}",
            alignment
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_transfer_optimizer_alignment() {
        use numrs2::gpu::memory::TransferOptimizer;

        let alignment = mem::align_of::<TransferOptimizer>();
        assert_eq!(
            alignment, 64,
            "TransferOptimizer should be aligned to 64 bytes, got {}",
            alignment
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_context_alignment() {
        use numrs2::gpu::GpuContext;

        let alignment = mem::align_of::<GpuContext>();
        assert_eq!(
            alignment, 64,
            "GpuContext should be aligned to 64 bytes, got {}",
            alignment
        );
    }

    #[test]
    fn test_parallel_config_false_sharing_prevention() {
        use numrs2::parallel_optimize::ParallelConfig;

        // Create array of configs to simulate multi-threaded access
        let configs = [
            ParallelConfig::default(),
            ParallelConfig::default(),
            ParallelConfig::default(),
            ParallelConfig::default(),
        ];

        // Verify each config occupies a separate cache line
        for i in 0..configs.len() - 1 {
            let addr1 = &configs[i] as *const _ as usize;
            let addr2 = &configs[i + 1] as *const _ as usize;
            let distance = addr2.saturating_sub(addr1);

            // Distance should be at least the size of ParallelConfig,
            // and aligned to cache line boundaries
            assert!(
                distance >= mem::size_of::<ParallelConfig>(),
                "Adjacent ParallelConfig instances should not overlap"
            );

            // Verify they're in different cache lines (if size permits)
            let cache_line1 = addr1 / 64;
            let cache_line2 = addr2 / 64;
            if mem::size_of::<ParallelConfig>() <= 64 {
                assert_ne!(
                    cache_line1, cache_line2,
                    "Adjacent ParallelConfig instances should occupy different cache lines"
                );
            }
        }
    }

    #[test]
    fn test_aligned_box_helper() {
        use numrs2::memory_alloc::aligned_helpers::AlignedBox;
        use numrs2::parallel_optimize::ParallelConfig;

        // Test that AlignedBox maintains alignment
        let boxed_config =
            AlignedBox::new(ParallelConfig::default()).expect("Failed to create AlignedBox");

        assert!(
            boxed_config.verify_alignment(),
            "AlignedBox should maintain cache line alignment"
        );

        let addr = boxed_config.as_ptr() as usize;
        assert_eq!(
            addr % 64,
            0,
            "AlignedBox pointer should be aligned to 64 bytes"
        );
    }

    #[test]
    fn test_aligned_vec_helper() {
        use numrs2::memory_alloc::aligned_helpers::AlignedVec;
        use numrs2::parallel_optimize::ParallelConfig;

        // Test that AlignedVec maintains alignment
        let mut vec = AlignedVec::with_capacity(10).expect("Failed to create AlignedVec");

        vec.push(ParallelConfig::default())
            .expect("Failed to push to AlignedVec");

        assert!(
            vec.verify_alignment(),
            "AlignedVec should maintain cache line alignment"
        );

        let addr = vec.as_ptr() as usize;
        assert_eq!(
            addr % 64,
            0,
            "AlignedVec data pointer should be aligned to 64 bytes"
        );
    }

    #[test]
    fn test_cache_line_size_assumption() {
        use numrs2::memory_alloc::aligned_helpers::cache_line_size;

        // Verify our cache line size assumption
        let detected_size = cache_line_size();
        assert_eq!(
            detected_size, 64,
            "Cache line size should be 64 bytes on modern CPUs"
        );
    }

    #[test]
    fn test_alignment_does_not_break_functionality() {
        use numrs2::arrays::broadcasting::{BroadcastConfig, BroadcastEngine};
        use numrs2::parallel_optimize::ParallelConfig;

        // Verify that alignment doesn't break basic functionality

        // Test ParallelConfig
        let config = ParallelConfig::optimized(10000, 1.0);
        assert!(config.should_parallelize(10000));

        // Test BroadcastEngine
        let _engine = BroadcastEngine::new(BroadcastConfig::default());
        // Just verify it constructs correctly
    }

    #[test]
    fn test_struct_sizes_reasonable() {
        use numrs2::arrays::broadcasting::BroadcastEngine;
        use numrs2::arrays::stride_optimization::StrideCalculator;
        use numrs2::parallel_optimize::ParallelConfig;

        // Verify structures haven't become unreasonably large due to alignment
        let config_size = mem::size_of::<ParallelConfig>();
        let engine_size = mem::size_of::<BroadcastEngine>();
        let calc_size = mem::size_of::<StrideCalculator>();

        // ParallelConfig should fit in one or two cache lines
        assert!(
            config_size <= 128,
            "ParallelConfig size {} exceeds 2 cache lines",
            config_size
        );

        // Verify sizes are multiples of alignment (if alignment is enforced)
        assert_eq!(
            config_size % mem::align_of::<ParallelConfig>(),
            0,
            "ParallelConfig size should be multiple of alignment"
        );

        println!(
            "ParallelConfig: size={}, align={}",
            config_size,
            mem::align_of::<ParallelConfig>()
        );
        println!(
            "BroadcastEngine: size={}, align={}",
            engine_size,
            mem::align_of::<BroadcastEngine>()
        );
        println!(
            "StrideCalculator: size={}, align={}",
            calc_size,
            mem::align_of::<StrideCalculator>()
        );
    }
}
