# NumRS2 Memory Management Guide

## Overview

NumRS2 provides a sophisticated memory management system designed for high-performance numerical computing. This guide covers the memory allocator hierarchy, allocation strategies, and optimization techniques.

## Memory Management Architecture

### Allocator Hierarchy

```
MemoryAllocator (Base Trait)
├── SpecializedAllocator (Type-specific optimizations)
├── ArrayAllocator (Array-specific memory management)
└── AllocationStrategy (Pluggable allocation strategies)
```

### Core Components

#### 1. MemoryAllocator Trait
```rust
pub trait MemoryAllocator: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn allocate(&self, size: usize, alignment: usize) -> Result<*mut u8, Self::Error>;
    fn deallocate(&self, ptr: *mut u8, size: usize, alignment: usize) -> Result<(), Self::Error>;
    fn reallocate(&self, ptr: *mut u8, old_size: usize, new_size: usize, alignment: usize) 
        -> Result<*mut u8, Self::Error>;
    fn allocated_size(&self) -> usize;
    fn peak_allocated(&self) -> usize;
    fn reset_peak(&self);
}
```

#### 2. AllocationStrategy Trait
```rust
pub trait AllocationStrategy: Send + Sync {
    fn should_use_arena(&self, size: usize) -> bool;
    fn should_use_pool(&self, size: usize) -> bool;
    fn get_alignment(&self, dtype: &str) -> usize;
    fn estimate_fragmentation(&self) -> f64;
    fn can_allocate(&self, size: usize) -> bool;
    fn preferred_block_size(&self, element_size: usize) -> usize;
    fn memory_pressure_level(&self) -> MemoryPressure;
}
```

## Built-in Allocators

### 1. Arena Allocator

Fast allocation for temporary data with automatic cleanup.

```rust
use numrs::memory_alloc::ArenaAllocator;

fn arena_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create arena with 10MB capacity
    let arena = ArenaAllocator::new(10 * 1024 * 1024);
    
    // Fast allocation within arena
    let ptr1 = arena.allocate(1024, 8)?;
    let ptr2 = arena.allocate(2048, 16)?;
    
    // All memory automatically freed when arena is dropped
    println!("Arena allocated: {} bytes", arena.allocated_size());
    println!("Arena peak usage: {} bytes", arena.peak_allocated());
    
    Ok(())
    // Arena automatically cleans up here
}
```

#### Use Cases for Arena Allocator
- Temporary calculations
- Function-local arrays
- Recursive algorithms
- Batch processing

```rust
use numrs::{Array, memory_alloc::ArenaAllocator};

fn matrix_operations_with_arena() -> Result<Array<f64>, Box<dyn std::error::Error>> {
    let arena = ArenaAllocator::new(100 * 1024 * 1024); // 100MB
    
    // Create temporary arrays in arena
    let temp1 = Array::with_allocator(&arena).zeros([1000, 1000])?;
    let temp2 = Array::with_allocator(&arena).ones([1000, 1000])?;
    let temp3 = temp1.add(&temp2)?;
    
    // Final result uses system allocator
    let result = Array::from(&temp3);
    
    Ok(result)
    // All temporary arrays freed when arena drops
}
```

### 2. Pool Allocator

Efficient allocation for fixed-size blocks with reuse.

```rust
use numrs::memory_alloc::PoolAllocator;

fn pool_example() -> Result<(), Box<dyn std::error::Error>> {
    // Pool for 1KB blocks, capacity for 100 blocks
    let pool = PoolAllocator::new(1024, 100)?;
    
    // Fast allocation from pool
    let block1 = pool.allocate_block()?;
    let block2 = pool.allocate_block()?;
    
    // Return blocks to pool for reuse
    pool.deallocate_block(block1)?;
    pool.deallocate_block(block2)?;
    
    // Pool automatically manages memory fragmentation
    println!("Pool fragmentation: {:.2}%", pool.fragmentation_level() * 100.0);
    
    Ok(())
}
```

#### Use Cases for Pool Allocator
- Fixed-size array operations
- Streaming data processing
- Real-time systems
- Memory fragmentation control

### 3. System Allocator

Standard system memory allocator with tracking.

```rust
use numrs::memory_alloc::SystemAllocator;

fn system_allocator_example() {
    let allocator = SystemAllocator::new();
    
    // Standard allocation with tracking
    let ptr = allocator.allocate(4096, 16).expect("Allocation failed");
    
    println!("System allocator stats:");
    println!("  Allocated: {} bytes", allocator.allocated_size());
    println!("  Peak: {} bytes", allocator.peak_allocated());
    
    allocator.deallocate(ptr, 4096, 16).expect("Deallocation failed");
}
```

## Allocation Strategies

### 1. Intelligent Allocation Strategy

Automatically chooses the best allocator based on context.

```rust
use numrs::memory_alloc::IntelligentAllocationStrategy;

fn intelligent_strategy_example() {
    let strategy = IntelligentAllocationStrategy::new()
        .with_arena_threshold(1024 * 1024)     // Use arena for allocations > 1MB
        .with_pool_threshold(4096)             // Use pool for allocations < 4KB
        .with_cache_awareness(true)            // Enable cache-friendly allocation
        .with_numa_awareness(true);            // Enable NUMA-aware allocation
    
    // Strategy automatically selects appropriate allocator
    let small_allocation = 1024;   // Uses pool
    let large_allocation = 10 * 1024 * 1024;  // Uses arena
    let medium_allocation = 128 * 1024;       // Uses system allocator
    
    assert!(strategy.should_use_pool(small_allocation));
    assert!(strategy.should_use_arena(large_allocation));
    assert!(!strategy.should_use_arena(medium_allocation));
}
```

### 2. Cache-Aware Strategy

Optimizes allocation for processor cache hierarchy.

```rust
use numrs::memory_alloc::CacheAwareStrategy;

fn cache_aware_example() {
    let strategy = CacheAwareStrategy::new()
        .with_l1_cache_size(32 * 1024)        // 32KB L1 cache
        .with_l2_cache_size(256 * 1024)       // 256KB L2 cache
        .with_l3_cache_size(8 * 1024 * 1024); // 8MB L3 cache
    
    // Automatic alignment for optimal cache performance
    let alignment_f64 = strategy.get_alignment("f64");    // Returns 32 for AVX2
    let alignment_f32 = strategy.get_alignment("f32");    // Returns 32 for AVX2
    
    // Block size optimization for cache efficiency
    let optimal_block = strategy.preferred_block_size(8); // For f64 elements
    
    println!("Optimal f64 alignment: {} bytes", alignment_f64);
    println!("Optimal block size: {} elements", optimal_block / 8);
}
```

### 3. NUMA-Aware Strategy

Optimizes allocation for Non-Uniform Memory Access systems.

```rust
use numrs::memory_alloc::NumaAwareStrategy;

fn numa_aware_example() {
    let strategy = NumaAwareStrategy::new()
        .with_local_node_preference(true)
        .with_interleaving(false);
    
    // Check NUMA topology
    let node_count = strategy.numa_node_count();
    let current_node = strategy.current_numa_node();
    
    println!("NUMA nodes: {}", node_count);
    println!("Current node: {}", current_node);
    
    // Allocate on specific NUMA node
    let node_allocator = strategy.allocator_for_node(0);
}
```

## Memory Monitoring and Profiling

### Memory Pressure Detection

```rust
use numrs::memory_alloc::{MemoryMonitor, MemoryPressure};

fn memory_monitoring_example() -> Result<(), Box<dyn std::error::Error>> {
    let monitor = MemoryMonitor::new();
    
    // Check current memory pressure
    let pressure = monitor.current_pressure_level();
    
    match pressure {
        MemoryPressure::Low => {
            println!("Memory pressure is low - safe to allocate");
        },
        MemoryPressure::Medium => {
            println!("Memory pressure is medium - consider optimization");
        },
        MemoryPressure::High => {
            println!("Memory pressure is high - reduce allocations");
            cleanup_temporary_data();
        },
        MemoryPressure::Critical => {
            println!("Memory pressure is critical - emergency cleanup");
            emergency_memory_cleanup();
            return Err("Critical memory pressure".into());
        },
    }
    
    // Set up pressure callbacks
    monitor.on_pressure_change(|old_level, new_level| {
        println!("Memory pressure changed: {:?} -> {:?}", old_level, new_level);
    });
    
    Ok(())
}

fn cleanup_temporary_data() {
    // Implementation for cleaning up temporary data
}

fn emergency_memory_cleanup() {
    // Implementation for emergency memory cleanup
}
```

### Memory Statistics and Profiling

```rust
use numrs::memory_alloc::{MemoryProfiler, AllocationStats};

fn memory_profiling_example() {
    let profiler = MemoryProfiler::new();
    
    // Profile a function's memory usage
    let stats = profiler.profile(|| {
        // Some memory-intensive operation
        let arrays: Vec<Array<f64>> = (0..10)
            .map(|_| Array::zeros([1000, 1000]))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        
        // Process arrays
        arrays.iter().map(|a| a.sum()).collect::<Result<Vec<_>, _>>()
    });
    
    match stats {
        Ok((result, allocation_stats)) => {
            println!("Operation completed successfully");
            println!("Memory stats:");
            println!("  Total allocated: {} bytes", allocation_stats.total_allocated);
            println!("  Peak usage: {} bytes", allocation_stats.peak_usage);
            println!("  Allocation count: {}", allocation_stats.allocation_count);
            println!("  Average allocation size: {} bytes", 
                allocation_stats.total_allocated / allocation_stats.allocation_count);
        },
        Err(e) => {
            eprintln!("Operation failed due to memory issues: {}", e);
        }
    }
}
```

## Advanced Memory Management

### Custom Allocators

```rust
use numrs::memory_alloc::{MemoryAllocator, AllocationStats};
use std::sync::{Arc, Mutex};

struct LoggingAllocator<T: MemoryAllocator> {
    inner: T,
    stats: Arc<Mutex<AllocationStats>>,
}

impl<T: MemoryAllocator> LoggingAllocator<T> {
    pub fn new(inner: T) -> Self {
        Self {
            inner,
            stats: Arc::new(Mutex::new(AllocationStats::default())),
        }
    }
}

impl<T: MemoryAllocator> MemoryAllocator for LoggingAllocator<T> {
    type Error = T::Error;

    fn allocate(&self, size: usize, alignment: usize) -> Result<*mut u8, Self::Error> {
        println!("Allocating {} bytes with alignment {}", size, alignment);
        
        let result = self.inner.allocate(size, alignment);
        
        if result.is_ok() {
            let mut stats = self.stats.lock().unwrap();
            stats.total_allocated += size;
            stats.allocation_count += 1;
        }
        
        result
    }

    fn deallocate(&self, ptr: *mut u8, size: usize, alignment: usize) -> Result<(), Self::Error> {
        println!("Deallocating {} bytes", size);
        
        let result = self.inner.deallocate(ptr, size, alignment);
        
        if result.is_ok() {
            let mut stats = self.stats.lock().unwrap();
            stats.total_allocated -= size;
        }
        
        result
    }

    fn reallocate(&self, ptr: *mut u8, old_size: usize, new_size: usize, alignment: usize) 
        -> Result<*mut u8, Self::Error> {
        println!("Reallocating from {} to {} bytes", old_size, new_size);
        self.inner.reallocate(ptr, old_size, new_size, alignment)
    }

    fn allocated_size(&self) -> usize {
        self.inner.allocated_size()
    }

    fn peak_allocated(&self) -> usize {
        self.inner.peak_allocated()
    }

    fn reset_peak(&self) {
        self.inner.reset_peak()
    }
}
```

### Memory-Mapped Arrays

```rust
use numrs::{Array, memory_alloc::MemoryMappedAllocator};
use std::path::Path;

fn memory_mapped_example() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "/tmp/large_array.dat";
    let array_size = [10000, 10000]; // 100M f64 elements
    
    // Create memory-mapped allocator
    let mmap_allocator = MemoryMappedAllocator::new(file_path, true)?; // read-write
    
    // Create array backed by memory-mapped file
    let mut array = Array::with_allocator(&mmap_allocator)
        .zeros(array_size)?;
    
    // Modifications are automatically written to disk
    array[[1000, 1000]] = 42.0;
    
    // Explicit sync to disk
    mmap_allocator.sync()?;
    
    println!("Array backed by file: {}", file_path);
    println!("Array size: {} bytes", array.memory_usage());
    
    Ok(())
}
```

### Thread-Safe Allocation

```rust
use numrs::memory_alloc::ThreadSafeAllocator;
use std::sync::Arc;
use std::thread;

fn thread_safe_allocation_example() -> Result<(), Box<dyn std::error::Error>> {
    let allocator = Arc::new(ThreadSafeAllocator::new());
    let mut handles = Vec::new();
    
    // Spawn multiple threads that share the same allocator
    for i in 0..4 {
        let allocator_clone = Arc::clone(&allocator);
        
        let handle = thread::spawn(move || {
            let thread_id = i;
            
            // Each thread can safely allocate
            for j in 0..100 {
                let size = (j + 1) * 1024;
                let ptr = allocator_clone.allocate(size, 8)
                    .expect("Allocation failed");
                
                // Simulate work
                std::thread::sleep(std::time::Duration::from_millis(1));
                
                allocator_clone.deallocate(ptr, size, 8)
                    .expect("Deallocation failed");
            }
            
            println!("Thread {} completed", thread_id);
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("Final allocator stats:");
    println!("  Current allocated: {} bytes", allocator.allocated_size());
    println!("  Peak allocated: {} bytes", allocator.peak_allocated());
    
    Ok(())
}
```

## Performance Optimization

### Memory Layout Optimization

```rust
use numrs::{Array, memory_alloc::CacheOptimizedAllocator};

fn cache_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    let allocator = CacheOptimizedAllocator::new()
        .with_prefetch_distance(64)   // Prefetch 64 bytes ahead
        .with_alignment(32);          // 32-byte alignment for AVX2
    
    // Create cache-friendly array
    let array = Array::with_allocator(&allocator)
        .zeros([1000, 1000])?;
    
    // Check memory layout
    assert!(array.is_contiguous());
    assert_eq!(array.memory_layout(), MemoryLayout::RowMajor);
    
    // Verify alignment
    let ptr = array.as_ptr() as usize;
    assert_eq!(ptr % 32, 0); // 32-byte aligned
    
    Ok(())
}
```

### NUMA Optimization

```rust
use numrs::memory_alloc::{NumaAllocator, NumaPolicy};

fn numa_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    // Allocate on local NUMA node
    let local_allocator = NumaAllocator::new(NumaPolicy::Local)?;
    
    // Allocate with interleaving across all nodes
    let interleaved_allocator = NumaAllocator::new(NumaPolicy::Interleave)?;
    
    // Allocate on specific node
    let node_allocator = NumaAllocator::new(NumaPolicy::Bind(0))?;
    
    // Create arrays with different NUMA policies
    let local_array = Array::with_allocator(&local_allocator)
        .zeros([5000, 5000])?;
    
    let interleaved_array = Array::with_allocator(&interleaved_allocator)
        .zeros([5000, 5000])?;
    
    // Check NUMA placement
    println!("Local array NUMA node: {:?}", local_array.numa_node());
    println!("Interleaved array NUMA distribution: {:?}", 
        interleaved_array.numa_distribution());
    
    Ok(())
}
```

## Memory Management Best Practices

### 1. Choose the Right Allocator

```rust
use numrs::memory_alloc::*;

fn choose_allocator_example(operation_type: &str, data_size: usize) -> Box<dyn MemoryAllocator> {
    match operation_type {
        "temporary" => {
            // Use arena for temporary data
            Box::new(ArenaAllocator::new(data_size * 2))
        },
        "streaming" => {
            // Use pool for streaming data
            Box::new(PoolAllocator::new(4096, 1000).unwrap())
        },
        "persistent" => {
            // Use system allocator for persistent data
            Box::new(SystemAllocator::new())
        },
        "large_file" => {
            // Use memory mapping for large files
            Box::new(MemoryMappedAllocator::new("/tmp/data.bin", true).unwrap())
        },
        _ => {
            // Use intelligent strategy as default
            Box::new(IntelligentAllocationStrategy::new())
        }
    }
}
```

### 2. Monitor Memory Usage

```rust
use numrs::memory_alloc::MemoryMonitor;

fn memory_aware_operation() -> Result<(), Box<dyn std::error::Error>> {
    let monitor = MemoryMonitor::new();
    
    // Check memory before large allocation
    if monitor.available_memory() < 1024 * 1024 * 1024 { // 1GB
        return Err("Insufficient memory for operation".into());
    }
    
    // Perform operation with monitoring
    let _large_array = Array::zeros([10000, 10000]);
    
    // Check memory pressure after allocation
    if monitor.current_pressure_level() >= MemoryPressure::High {
        // Clean up if pressure is high
        drop(_large_array);
        monitor.trigger_garbage_collection();
    }
    
    Ok(())
}
```

### 3. Use RAII for Cleanup

```rust
use numrs::memory_alloc::ArenaAllocator;

struct TemporaryWorkspace {
    arena: ArenaAllocator,
    arrays: Vec<Array<f64>>,
}

impl TemporaryWorkspace {
    fn new(capacity: usize) -> Self {
        Self {
            arena: ArenaAllocator::new(capacity),
            arrays: Vec::new(),
        }
    }
    
    fn create_array(&mut self, shape: &[usize]) -> Result<&Array<f64>, Box<dyn std::error::Error>> {
        let array = Array::with_allocator(&self.arena).zeros(shape)?;
        self.arrays.push(array);
        Ok(self.arrays.last().unwrap())
    }
}

impl Drop for TemporaryWorkspace {
    fn drop(&mut self) {
        println!("Cleaning up workspace with {} arrays", self.arrays.len());
        // Arena automatically cleans up all allocations
    }
}

fn workspace_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut workspace = TemporaryWorkspace::new(100 * 1024 * 1024); // 100MB
    
    // Create temporary arrays
    let temp1 = workspace.create_array(&[1000, 1000])?;
    let temp2 = workspace.create_array(&[500, 2000])?;
    
    // Perform calculations
    let result = temp1.add(temp2)?;
    
    Ok(())
    // Workspace automatically cleans up here
}
```

### 4. Profile Memory Usage

```rust
use numrs::memory_alloc::MemoryProfiler;

fn profile_algorithm() -> Result<(), Box<dyn std::error::Error>> {
    let profiler = MemoryProfiler::new();
    
    let (result, stats) = profiler.profile(|| {
        // Algorithm to profile
        matrix_multiplication_algorithm()
    })?;
    
    println!("Algorithm profiling results:");
    println!("  Peak memory: {} MB", stats.peak_usage / (1024 * 1024));
    println!("  Total allocations: {}", stats.allocation_count);
    println!("  Average allocation: {} KB", 
        stats.total_allocated / stats.allocation_count / 1024);
    println!("  Memory efficiency: {:.1}%", 
        (stats.useful_bytes as f64 / stats.total_allocated as f64) * 100.0);
    
    // Suggest optimizations
    if stats.fragmentation_ratio > 0.3 {
        println!("High fragmentation detected - consider using pool allocator");
    }
    
    if stats.allocation_count > 1000 {
        println!("Many allocations - consider using arena allocator");
    }
    
    Ok(())
}

fn matrix_multiplication_algorithm() -> Result<Array<f64>, Box<dyn std::error::Error>> {
    // Example algorithm implementation
    let a = Array::random([1000, 1000])?;
    let b = Array::random([1000, 1000])?;
    Ok(a.matmul(&b)?)
}
```

This memory management system provides fine-grained control over allocation behavior while maintaining safety and performance. Choose the appropriate allocator and strategy based on your specific use case and performance requirements.