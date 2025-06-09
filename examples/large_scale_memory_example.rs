//! Example demonstrating large-scale memory management capabilities
//!
//! This example shows how to use NumRS2's large-scale memory management features
//! for handling datasets that are too large to fit entirely in memory.

use numrs::memory_alloc::{
    LargeScaleManager, LargeScaleConfig, OutOfCoreArray, OutOfCoreConfig, CacheStrategy,
    init_global_manager, get_global_memory_stats, get_global_spill_stats, should_spill_globally
};
use numrs::array::Array;
use numrs::error::Result;
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("NumRS2 Large-Scale Memory Management Example");
    println!("============================================\n");

    // Example 1: Large-Scale Memory Manager
    example_large_scale_manager()?;
    
    // Example 2: Out-of-Core Arrays
    example_out_of_core_arrays()?;
    
    // Example 3: Memory Usage Monitoring
    example_memory_monitoring()?;
    
    // Example 4: Data Spilling
    example_data_spilling()?;

    println!("\nAll large-scale memory management examples completed successfully!");
    Ok(())
}

/// Demonstrate the large-scale memory manager
fn example_large_scale_manager() -> Result<()> {
    println!("1. Large-Scale Memory Manager");
    println!("----------------------------");
    
    // Configure the large-scale manager
    let config = LargeScaleConfig {
        max_memory_usage: 100 * 1024 * 1024, // 100MB limit
        spill_threshold: 0.7,                // Spill at 70% usage
        chunk_size: 1024 * 1024,            // 1MB chunks
        temp_dir: std::env::temp_dir().join("numrs_large_scale_example"),
        background_cleanup: true,
        monitor_interval_ms: 500,
        enable_stats: true,
    };
    
    // Initialize the global manager
    init_global_manager(config)?;
    
    println!("✓ Large-scale memory manager initialized");
    println!("  - Memory limit: 100MB");
    println!("  - Spill threshold: 70%");
    println!("  - Chunk size: 1MB");
    
    // Create some large data to test spilling
    let _large_data = vec![1.0f64; 50_000_000]; // ~400MB of data
    
    // This should trigger spilling due to size
    if should_spill_globally() {
        println!("  - Spilling is recommended");
    }
    
    println!();
    Ok(())
}

/// Demonstrate out-of-core arrays
fn example_out_of_core_arrays() -> Result<()> {
    println!("2. Out-of-Core Arrays");
    println!("--------------------");
    
    // Configure out-of-core array
    let config = OutOfCoreConfig {
        max_chunks_in_memory: 4,             // Keep max 4 chunks in memory
        chunk_size: 10_000,                  // 10K elements per chunk
        storage_path: std::env::temp_dir().join("numrs_ooc_example"),
        use_compression: false,
        cache_strategy: CacheStrategy::LRU,
        enable_prefetch: true,
        prefetch_count: 2,
    };
    
    // Create a large dataset
    let data_size = 100_000; // 100K elements
    let data: Vec<f64> = (0..data_size).map(|i| i as f64 * 0.5).collect();
    let shape = vec![data_size];
    
    println!("✓ Creating out-of-core array with {} elements", data_size);
    let mut ooc_array = OutOfCoreArray::from_data(data, shape, config)?;
    
    // Demonstrate access patterns that will trigger chunk loading/eviction
    println!("  - Accessing elements across different chunks...");
    
    // Access elements from different chunks
    let test_indices = [1000, 25000, 50000, 75000, 99000];
    for &index in &test_indices {
        let value = ooc_array.get(&[index])?;
        println!("    Element [{}] = {:.1}", index, value);
    }
    
    // Modify some elements
    println!("  - Modifying elements...");
    ooc_array.set(&[5000], 42.0)?;
    ooc_array.set(&[30000], 84.0)?;
    
    let modified_1 = ooc_array.get(&[5000])?;
    let modified_2 = ooc_array.get(&[30000])?;
    println!("    Modified [5000] = {:.1}", modified_1);
    println!("    Modified [30000] = {:.1}", modified_2);
    
    // Display cache statistics
    let cache_stats = ooc_array.get_cache_stats();
    println!("  - Cache statistics:");
    println!("    Total chunks: {}", cache_stats.total_chunks);
    println!("    Chunks in memory: {}", cache_stats.chunks_in_memory);
    println!("    Chunks on disk: {}", cache_stats.chunks_on_disk);
    println!("    Dirty chunks: {}", cache_stats.dirty_chunks);
    println!("    Cache limit: {}", cache_stats.cache_limit);
    
    // Sync all changes to disk
    ooc_array.sync_all()?;
    println!("  ✓ All changes synced to disk");
    
    println!();
    Ok(())
}

/// Demonstrate memory usage monitoring
fn example_memory_monitoring() -> Result<()> {
    println!("3. Memory Usage Monitoring");
    println!("-------------------------");
    
    // Get initial memory statistics
    let initial_stats = get_global_memory_stats()?;
    println!("✓ Initial memory statistics:");
    println!("  - Current usage: {} bytes", initial_stats.current_usage);
    println!("  - Peak usage: {} bytes", initial_stats.peak_usage);
    println!("  - Active allocations: {}", initial_stats.active_allocations);
    println!("  - Total allocations: {}", initial_stats.total_allocations);
    
    // Create some arrays to allocate memory
    let arrays: Vec<Array<f64>> = (0..5)
        .map(|i| {
            let size = (i + 1) * 10_000;
            let data: Vec<f64> = (0..size).map(|x| x as f64).collect();
            Array::from_vec(data)
        })
        .collect();
    
    println!("  - Created {} arrays with varying sizes", arrays.len());
    
    // Get updated memory statistics
    let updated_stats = get_global_memory_stats()?;
    println!("  ✓ Updated memory statistics:");
    println!("    Current usage: {} bytes", updated_stats.current_usage);
    println!("    Peak usage: {} bytes", updated_stats.peak_usage);
    println!("    Active allocations: {}", updated_stats.active_allocations);
    
    let usage_increase = updated_stats.current_usage.saturating_sub(initial_stats.current_usage);
    println!("    Memory usage increase: {} bytes", usage_increase);
    
    println!();
    Ok(())
}

/// Demonstrate data spilling functionality
fn example_data_spilling() -> Result<()> {
    println!("4. Data Spilling");
    println!("---------------");
    
    // Get spill statistics
    let initial_spill_stats = get_global_spill_stats()?;
    println!("✓ Initial spill statistics:");
    println!("  - Total spilled size: {} bytes", initial_spill_stats.total_spilled_size);
    println!("  - Spilled data count: {}", initial_spill_stats.spilled_count);
    println!("  - Temp directory: {}", initial_spill_stats.temp_dir.display());
    
    // Create chunked data processing example
    let large_dataset_size = 1_000_000;
    let large_dataset: Vec<f64> = (0..large_dataset_size)
        .map(|i| (i as f64).sin())
        .collect();
    
    println!("  - Created large dataset with {} elements", large_dataset_size);
    
    // Process data in chunks to demonstrate memory-efficient processing
    let chunk_size = 50_000;
    let mut sum = 0.0;
    let mut processed_elements = 0;
    
    println!("  - Processing dataset in chunks of {} elements...", chunk_size);
    
    for chunk in large_dataset.chunks(chunk_size) {
        // Simulate processing that might cause memory pressure
        let chunk_sum: f64 = chunk.iter().sum();
        sum += chunk_sum;
        processed_elements += chunk.len();
        
        // Periodically check if we should spill
        if processed_elements % (chunk_size * 4) == 0 {
            if should_spill_globally() {
                println!("    Warning: Memory usage high, spilling recommended");
            }
        }
    }
    
    println!("  ✓ Processed {} elements", processed_elements);
    println!("    Total sum: {:.2}", sum);
    println!("    Average: {:.6}", sum / processed_elements as f64);
    
    // Demonstrate chunked iterator
    println!("  - Using chunked iterator for memory-efficient processing...");
    
    // Create a large array and process it in chunks
    let test_array = Array::from_vec((0..100_000).map(|i| i as f32).collect());
    let mut chunk_count = 0;
    let mut total_processed = 0;
    
    // Process in chunks of 10,000 elements
    for i in (0..test_array.len()).step_by(10_000) {
        let end = std::cmp::min(i + 10_000, test_array.len());
        let chunk_size = end - i;
        chunk_count += 1;
        total_processed += chunk_size;
        
        if chunk_count <= 3 {
            println!("    Chunk {}: {} elements (indices {}-{})", 
                     chunk_count, chunk_size, i, end - 1);
        }
    }
    
    if chunk_count > 3 {
        println!("    ... and {} more chunks", chunk_count - 3);
    }
    
    println!("  ✓ Processed {} elements in {} chunks", total_processed, chunk_count);
    
    // Get final spill statistics
    let final_spill_stats = get_global_spill_stats()?;
    println!("  ✓ Final spill statistics:");
    println!("    Total spilled size: {} bytes", final_spill_stats.total_spilled_size);
    println!("    Spilled data count: {}", final_spill_stats.spilled_count);
    
    println!();
    Ok(())
}