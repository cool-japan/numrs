use numrs2::memory_alloc::aligned::AlignedBox;
use numrs2::memory_alloc::arena::ArenaVec;
use numrs2::memory_alloc::{
    get_default_allocator, get_global_allocator_strategy, init_global_allocator,
    reset_global_allocator, AlignedAllocator, AlignmentConfig, AllocStrategy, ArenaAllocator,
    ArenaConfig, PoolAllocator, PoolConfig,
};
use std::alloc::Layout;
use std::ptr::NonNull;
use std::time::{Duration, Instant};

fn main() {
    println!("NumRS Memory Allocator Optimization Example");
    println!("===========================================\n");

    // SECTION 1: Pool Allocator
    println!("1. Pool Allocator Performance");
    println!("--------------------------");

    // Configure a memory pool
    let pool_config = PoolConfig {
        block_size: 1024,       // 1KB blocks
        initial_blocks: 128,    // 128KB pre-allocated
        max_blocks: Some(1024), // Max 1MB
        auto_resize: true,
        growth_factor: 2.0,
    };

    let pool = PoolAllocator::new(pool_config);

    // Benchmark allocation/deallocation with pool
    let (pool_alloc_time, pool_ptrs) = benchmark_alloc(&pool, 1000);
    let pool_dealloc_time = benchmark_dealloc(&pool, pool_ptrs);

    // Standard allocator benchmark for comparison
    println!("\nComparing with standard allocator:");
    let std_alloc_time = benchmark_standard_alloc(1024, 1000);
    let std_dealloc_time = benchmark_standard_dealloc(1024, 1000);

    println!(
        "  Pool allocator: {} allocations in {:?} ({:?} per allocation)",
        1000,
        pool_alloc_time,
        pool_alloc_time / 1000
    );
    println!(
        "  Standard allocator: {} allocations in {:?} ({:?} per allocation)",
        1000,
        std_alloc_time,
        std_alloc_time / 1000
    );
    println!(
        "  Speed improvement: {:.2}x for allocation",
        std_alloc_time.as_nanos() as f64 / pool_alloc_time.as_nanos() as f64
    );

    println!(
        "\n  Pool deallocator: {} deallocations in {:?} ({:?} per deallocation)",
        1000,
        pool_dealloc_time,
        pool_dealloc_time / 1000
    );
    println!(
        "  Standard deallocator: {} deallocations in {:?} ({:?} per deallocation)",
        1000,
        std_dealloc_time,
        std_dealloc_time / 1000
    );
    println!(
        "  Speed improvement: {:.2}x for deallocation",
        std_dealloc_time.as_nanos() as f64 / pool_dealloc_time.as_nanos() as f64
    );

    // SECTION 2: Arena Allocator
    println!("\n2. Arena Allocator Performance");
    println!("----------------------------");

    // Configure an arena allocator
    let arena_config = ArenaConfig {
        initial_size: 1024 * 1024, // 1MB initial size
        allow_growth: true,
        growth_factor: 2.0,
        alignment: 8,
    };

    let arena = ArenaAllocator::new(arena_config);

    // Benchmark many small allocations with arena
    println!("Arena allocator performance for many small allocations:");
    let start = Instant::now();
    let mut arena_ptrs = Vec::with_capacity(10000);

    for _ in 0..10000 {
        if let Some(ptr) = arena.allocate(32) {
            arena_ptrs.push(ptr);
        }
    }

    let arena_alloc_time = start.elapsed();
    println!(
        "  Arena: 10,000 small allocations (32 bytes) in {:?} ({:?} per allocation)",
        arena_alloc_time,
        arena_alloc_time / 10000
    );

    // Standard allocator for comparison
    let start = Instant::now();
    let mut std_ptrs = Vec::with_capacity(10000);
    unsafe {
        for _ in 0..10000 {
            let layout = Layout::from_size_align(32, 8).unwrap();
            let ptr = std::alloc::alloc(layout);
            if !ptr.is_null() {
                std_ptrs.push((NonNull::new_unchecked(ptr), layout));
            }
        }
    }
    let std_alloc_time = start.elapsed();
    println!(
        "  Standard: 10,000 small allocations (32 bytes) in {:?} ({:?} per allocation)",
        std_alloc_time,
        std_alloc_time / 10000
    );

    println!(
        "  Speed improvement: {:.2}x for many small allocations",
        std_alloc_time.as_nanos() as f64 / arena_alloc_time.as_nanos() as f64
    );

    // Reset arena - very fast deallocation of all memory at once
    let start = Instant::now();
    arena.reset();
    let arena_reset_time = start.elapsed();
    println!(
        "\n  Arena reset (freeing all 10,000 allocations): {:?}",
        arena_reset_time
    );

    // Standard deallocation for comparison
    let start = Instant::now();
    unsafe {
        for (ptr, layout) in std_ptrs {
            std::alloc::dealloc(ptr.as_ptr(), layout);
        }
    }
    let std_dealloc_time = start.elapsed();
    println!(
        "  Standard deallocation (freeing 10,000 allocations): {:?}",
        std_dealloc_time
    );

    println!(
        "  Speed improvement: {:.2}x for bulk deallocation",
        std_dealloc_time.as_nanos() as f64 / arena_reset_time.as_nanos() as f64
    );

    // Demo arena vector
    println!("\nArena Vector demo:");
    let arena = ArenaAllocator::new(ArenaConfig::default());
    let mut vec: ArenaVec<i32> = ArenaVec::with_capacity(100, &arena);

    // Push elements
    for i in 0..100 {
        vec.push(i);
    }

    println!("  Created arena vector with 100 elements");
    println!("  Sum of elements: {}", vec.as_slice().iter().sum::<i32>());

    // SECTION 3: Aligned Allocator
    println!("\n3. Aligned Allocator Performance for SIMD");
    println!("---------------------------------------");

    // Compare aligned vs unaligned allocation for SIMD operations
    let aligned_config = AlignmentConfig::simd_256(); // 32-byte alignment for AVX
    let aligned = AlignedAllocator::new(aligned_config);

    // Create aligned and unaligned arrays of f32 values
    let size = 1024 * 256; // 1MB of f32 values

    let aligned_ptr = aligned
        .allocate_array::<f32>(size)
        .expect("Aligned allocation failed");
    let unaligned_data = vec![1.0f32; size];

    // Initialize aligned data
    unsafe {
        let slice = std::slice::from_raw_parts_mut(aligned_ptr.as_ptr(), size);
        for i in 0..size {
            slice[i] = 1.0;
        }
    }

    // Benchmark summing the arrays (simulating SIMD operation)
    println!("Summing arrays (simulating SIMD operations):");

    // Sum aligned data
    let start = Instant::now();
    let mut aligned_sum = 0.0f32;
    for _ in 0..100 {
        aligned_sum = sum_array(aligned_ptr.as_ptr(), size);
    }
    let aligned_time = start.elapsed();

    // Sum unaligned data
    let start = Instant::now();
    let mut unaligned_sum = 0.0f32;
    for _ in 0..100 {
        unaligned_sum = sum_array(unaligned_data.as_ptr(), size);
    }
    let unaligned_time = start.elapsed();

    println!("  Aligned sum: {} in {:?}", aligned_sum, aligned_time);
    println!("  Unaligned sum: {} in {:?}", unaligned_sum, unaligned_time);
    println!(
        "  Performance improvement: {:.2}x",
        unaligned_time.as_nanos() as f64 / aligned_time.as_nanos() as f64
    );

    // Cleanup
    unsafe {
        aligned.deallocate_array(aligned_ptr, size);
    }

    // Demo aligned box
    println!("\nAligned Box demo:");
    let aligned_box =
        AlignedBox::new(Matrix4x4::identity(), 32).expect("Aligned box creation failed");

    println!("  Created aligned matrix: {:?}", aligned_box.get());
    println!("  Alignment: {} bytes", aligned_box.alignment());

    // SECTION 4: Global Allocator Strategy
    println!("\n4. Global Allocator Strategy");
    println!("---------------------------");

    println!("Default strategy: {:?}", get_global_allocator_strategy());

    // Change the global strategy to Pool
    init_global_allocator(AllocStrategy::Pool);
    println!("After change: {:?}", get_global_allocator_strategy());

    // Get the default allocator (now Pool)
    let default_allocator = get_default_allocator();
    let ptr = default_allocator.allocate(100).expect("Allocation failed");
    println!("Allocated memory with the global default allocator (Pool)");

    // Deallocate
    unsafe {
        default_allocator.deallocate(ptr, Layout::from_size_align(100, 8).unwrap());
    }

    // Reset to Standard
    reset_global_allocator();
    println!("Reset to default: {:?}", get_global_allocator_strategy());

    // SECTION 5: Workload-based Strategy Selection
    println!("\n5. Workload-Based Strategy Selection");
    println!("----------------------------------");

    // Benchmark different allocation patterns to see which strategy works best
    let test_sizes = [32, 4096, 65536, 1048576];
    let iterations = 1000;

    println!("Comparing different allocation strategies:");
    println!("  Allocation size | Standard  | Pool      | Arena     | Aligned   | Best Strategy");
    println!("  ---------------|-----------|-----------|-----------|-----------|-------------");

    for &size in &test_sizes {
        // Benchmark each allocator type
        let std_time = benchmark_allocator_type(AllocStrategy::Standard, size, iterations);
        let pool_time = benchmark_allocator_type(AllocStrategy::Pool, size, iterations);
        let arena_time = benchmark_allocator_type(AllocStrategy::Arena, size, iterations);
        let aligned_time = benchmark_allocator_type(AllocStrategy::Aligned, size, iterations);

        // Determine the best strategy
        let best_time = std_time.min(pool_time).min(arena_time).min(aligned_time);
        let best_strategy = if best_time == std_time {
            "Standard"
        } else if best_time == pool_time {
            "Pool"
        } else if best_time == arena_time {
            "Arena"
        } else {
            "Aligned"
        };

        println!(
            "  {:13} | {:9?} | {:9?} | {:9?} | {:9?} | {}",
            size, std_time, pool_time, arena_time, aligned_time, best_strategy
        );
    }
}

// Benchmarks pool allocator allocations
fn benchmark_alloc(pool: &PoolAllocator, count: usize) -> (Duration, Vec<NonNull<u8>>) {
    let start = Instant::now();
    let mut ptrs = Vec::with_capacity(count);

    for _ in 0..count {
        if let Some(ptr) = pool.allocate() {
            ptrs.push(ptr);
        }
    }

    let elapsed = start.elapsed();
    (elapsed, ptrs)
}

// Benchmarks pool allocator deallocations
fn benchmark_dealloc(pool: &PoolAllocator, ptrs: Vec<NonNull<u8>>) -> Duration {
    let start = Instant::now();

    for ptr in ptrs {
        unsafe {
            pool.deallocate(ptr);
        }
    }

    start.elapsed()
}

// Benchmarks standard allocator allocations
fn benchmark_standard_alloc(size: usize, count: usize) -> Duration {
    let start = Instant::now();
    let mut ptrs = Vec::with_capacity(count);

    unsafe {
        for _ in 0..count {
            let layout = Layout::from_size_align(size, 8).unwrap();
            let ptr = std::alloc::alloc(layout);
            if !ptr.is_null() {
                ptrs.push((NonNull::new_unchecked(ptr), layout));
            }
        }
    }

    // To make this fair, we need to keep the allocations around
    // until we're done (just like the pool benchmark)
    let elapsed = start.elapsed();

    // Deallocate to avoid leaks
    for (ptr, layout) in ptrs {
        unsafe {
            std::alloc::dealloc(ptr.as_ptr(), layout);
        }
    }

    elapsed
}

// Benchmarks standard allocator deallocations
fn benchmark_standard_dealloc(size: usize, count: usize) -> Duration {
    let mut ptrs = Vec::with_capacity(count);

    unsafe {
        for _ in 0..count {
            let layout = Layout::from_size_align(size, 8).unwrap();
            let ptr = std::alloc::alloc(layout);
            if !ptr.is_null() {
                ptrs.push((NonNull::new_unchecked(ptr), layout));
            }
        }
    }

    let start = Instant::now();

    for (ptr, layout) in ptrs {
        unsafe {
            std::alloc::dealloc(ptr.as_ptr(), layout);
        }
    }

    start.elapsed()
}

// Function to benchmark different allocator types
fn benchmark_allocator_type(strategy: AllocStrategy, size: usize, count: usize) -> Duration {
    match strategy {
        AllocStrategy::Standard => benchmark_standard_alloc(size, count),
        AllocStrategy::Pool => {
            // Only use pool for sizes that fit
            if size > 4096 {
                Duration::from_secs(999) // Signal "not suitable"
            } else {
                let config = PoolConfig {
                    block_size: size,
                    initial_blocks: count,
                    max_blocks: Some(count),
                    auto_resize: false,
                    growth_factor: 1.0,
                };
                let pool = PoolAllocator::new(config);
                let (time, ptrs) = benchmark_alloc(&pool, count);
                // Clean up
                for ptr in ptrs {
                    unsafe {
                        pool.deallocate(ptr);
                    }
                }
                time
            }
        }
        AllocStrategy::Arena => {
            let config = ArenaConfig {
                initial_size: size * count,
                allow_growth: false,
                growth_factor: 1.0,
                alignment: 8,
            };
            let arena = ArenaAllocator::new(config);

            let start = Instant::now();
            let mut ptrs = Vec::with_capacity(count);
            for _ in 0..count {
                if let Some(ptr) = arena.allocate(size) {
                    ptrs.push(ptr);
                }
            }
            let elapsed = start.elapsed();

            // No need to manually clean up - arena will deallocate everything on drop
            elapsed
        }
        AllocStrategy::Aligned => {
            let config = AlignmentConfig::simd_256();
            let aligned = AlignedAllocator::new(config);

            let start = Instant::now();
            let mut ptrs = Vec::with_capacity(count);
            for _ in 0..count {
                if let Some(ptr) = aligned.allocate(size) {
                    ptrs.push((ptr, size));
                }
            }
            let elapsed = start.elapsed();

            // Clean up
            for (ptr, size) in ptrs {
                unsafe {
                    aligned.deallocate(ptr, size);
                }
            }

            elapsed
        }
        AllocStrategy::Auto => {
            // Auto strategy selects the best strategy for the workload
            // Just default to standard for simplicity
            benchmark_standard_alloc(size, count)
        }
    }
}

// Function to sum an array (for SIMD benchmarking)
#[inline(never)]
fn sum_array(data: *const f32, len: usize) -> f32 {
    let mut sum = 0.0;
    unsafe {
        for i in 0..len {
            sum += *data.add(i);
        }
    }
    sum
}

// Simple 4x4 matrix for aligned box demo
#[derive(Debug, Clone, Copy)]
struct Matrix4x4 {
    data: [[f32; 4]; 4],
}

impl Matrix4x4 {
    fn identity() -> Self {
        let mut mat = Self {
            data: [[0.0; 4]; 4],
        };

        for i in 0..4 {
            mat.data[i][i] = 1.0;
        }

        mat
    }
}
