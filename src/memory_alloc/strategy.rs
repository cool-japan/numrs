//! Memory allocation strategy selection and configuration
//!
//! This module defines the various memory allocation strategies and
//! provides the ability to select the appropriate strategy for
//! different numerical workloads.

use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::marker::{Send, Sync};

use super::aligned::{AlignedAllocator, AlignmentConfig};
use super::arena::{ArenaAllocator, ArenaConfig};
use super::pool::{PoolAllocator, PoolConfig};

/// Memory allocation strategies for numerical computing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocStrategy {
    /// Standard system allocator
    Standard,
    /// Memory pool for fixed-size allocations
    Pool,
    /// Arena allocator for bulk temporary allocations
    Arena,
    /// Aligned allocator for SIMD operations
    Aligned,
    /// Automatic selection based on workload characteristics
    Auto,
}

// Global allocation strategy
static GLOBAL_STRATEGY: AtomicUsize = AtomicUsize::new(0); // 0 = Standard

/// Set the global allocation strategy
pub fn set_global_allocator(strategy: AllocStrategy) {
    let value = match strategy {
        AllocStrategy::Standard => 0,
        AllocStrategy::Pool => 1,
        AllocStrategy::Arena => 2,
        AllocStrategy::Aligned => 3,
        AllocStrategy::Auto => 4,
    };
    GLOBAL_STRATEGY.store(value, Ordering::SeqCst);
}

/// Get the current global allocation strategy
pub fn get_global_allocator_strategy() -> AllocStrategy {
    match GLOBAL_STRATEGY.load(Ordering::SeqCst) {
        0 => AllocStrategy::Standard,
        1 => AllocStrategy::Pool,
        2 => AllocStrategy::Arena,
        3 => AllocStrategy::Aligned,
        4 => AllocStrategy::Auto,
        _ => AllocStrategy::Standard, // Fallback
    }
}

/// Reset the global allocator to the default strategy (Standard)
pub fn reset_global_allocator() {
    GLOBAL_STRATEGY.store(0, Ordering::SeqCst);
}

/// Get the default memory allocator based on global strategy
pub fn get_default_allocator() -> Box<dyn MemoryAllocator> {
    match get_global_allocator_strategy() {
        AllocStrategy::Standard => Box::new(StandardAllocator),
        AllocStrategy::Pool => Box::new(PoolAllocator::new(PoolConfig::default())),
        AllocStrategy::Arena => Box::new(ArenaAllocator::new(ArenaConfig::default())),
        AllocStrategy::Aligned => Box::new(AlignedAllocator::new(AlignmentConfig::default())),
        AllocStrategy::Auto => Box::new(AutoAllocator::new()),
    }
}

/// Get the recommended allocation strategy based on workload characteristics
pub fn recommend_strategy(alloc_size: usize, alloc_frequency: AllocFrequency, simd_usage: bool) -> AllocStrategy {
    match (alloc_size, alloc_frequency, simd_usage) {
        // Large allocations are best with the standard allocator
        (size, _, _) if size > 1_000_000 => AllocStrategy::Standard,
        
        // Very small, frequent allocations work well with a pool
        (size, AllocFrequency::VeryHigh, _) if size < 8192 => AllocStrategy::Pool,
        
        // Medium, frequent allocations work well with an arena
        (size, AllocFrequency::High, _) if size < 65536 => AllocStrategy::Arena,
        
        // SIMD operations benefit from aligned memory
        (_, _, true) => AllocStrategy::Aligned,
        
        // Default to standard allocator for other cases
        _ => AllocStrategy::Standard,
    }
}

/// Allocation frequency classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocFrequency {
    /// Very infrequent allocations (once per operation)
    Low,
    /// Moderate allocation frequency
    Medium,
    /// High allocation frequency (many per operation)
    High,
    /// Very high allocation frequency (thousands per operation)
    VeryHigh,
}

/// Trait for memory allocators
pub trait MemoryAllocator: Send + Sync {
    /// Allocate memory of the given size
    fn allocate(&self, size: usize) -> Option<NonNull<u8>>;
    
    /// Allocate memory with the given layout
    fn allocate_layout(&self, layout: Layout) -> Option<NonNull<u8>>;
    
    /// Deallocate previously allocated memory
    ///
    /// # Safety
    ///
    /// - The pointer must have been allocated by this allocator
    /// - The layout must match what was used for allocation
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
}

/// Standard system allocator
pub struct StandardAllocator;

impl MemoryAllocator for StandardAllocator {
    fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        if size == 0 {
            return None;
        }
        let layout = Layout::from_size_align(size, 8).ok()?;
        self.allocate_layout(layout)
    }
    
    fn allocate_layout(&self, layout: Layout) -> Option<NonNull<u8>> {
        unsafe {
            NonNull::new(alloc(layout))
        }
    }
    
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        dealloc(ptr.as_ptr(), layout);
    }
}

/// Auto-selecting allocator based on workload characteristics
pub struct AutoAllocator {
    standard: StandardAllocator,
    pool: PoolAllocator,
    arena: ArenaAllocator,
    aligned: AlignedAllocator,
}

impl AutoAllocator {
    /// Create a new auto-selecting allocator
    pub fn new() -> Self {
        Self {
            standard: StandardAllocator,
            pool: PoolAllocator::new(PoolConfig::default()),
            arena: ArenaAllocator::new(ArenaConfig::default()),
            aligned: AlignedAllocator::new(AlignmentConfig::default()),
        }
    }
    
    /// Select the appropriate allocator for the given allocation size
    fn select_allocator(&self, size: usize) -> &dyn MemoryAllocator {
        // Simple selection based just on size
        // A more sophisticated implementation would consider more factors
        match size {
            0..=4096 => &self.pool as &dyn MemoryAllocator,
            4097..=65536 => &self.arena as &dyn MemoryAllocator,
            _ => &self.standard as &dyn MemoryAllocator,
        }
    }
}

impl MemoryAllocator for AutoAllocator {
    fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        if size == 0 {
            return None;
        }
        
        // For SIMD operations, we typically want aligned memory
        // This is a simplification - in a real implementation, we would detect
        // if the allocation is for SIMD usage
        if size % 16 == 0 && size >= 16 {
            return self.aligned.allocate(size);
        }
        
        self.select_allocator(size).allocate(size)
    }
    
    fn allocate_layout(&self, layout: Layout) -> Option<NonNull<u8>> {
        // If highly aligned, use aligned allocator
        if layout.align() >= 16 {
            return self.aligned.allocate(layout.size());
        }
        
        self.select_allocator(layout.size()).allocate_layout(layout)
    }
    
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // Select the same allocator that would have been used for the allocation
        if layout.align() >= 16 {
            self.aligned.deallocate(ptr, layout.size());
            return;
        }
        
        self.select_allocator(layout.size()).deallocate(ptr, layout);
    }
}

// For Arena allocator
impl MemoryAllocator for ArenaAllocator {
    fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        ArenaAllocator::allocate(self, size)
    }
    
    fn allocate_layout(&self, layout: Layout) -> Option<NonNull<u8>> {
        self.allocate_aligned(layout.size(), layout.align())
    }
    
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // ArenaAllocator doesn't deallocate individual allocations
        // They're freed when the arena is reset
    }
}

// For Pool allocator
impl MemoryAllocator for PoolAllocator {
    fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        if size <= self.block_size() {
            PoolAllocator::allocate(self)
        } else {
            None // Block size too small
        }
    }
    
    fn allocate_layout(&self, layout: Layout) -> Option<NonNull<u8>> {
        if layout.size() <= self.block_size() {
            PoolAllocator::allocate(self)
        } else {
            None // Block size too small
        }
    }
    
    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        PoolAllocator::deallocate(self, ptr);
    }
}

// For Aligned allocator
impl MemoryAllocator for AlignedAllocator {
    fn allocate(&self, size: usize) -> Option<NonNull<u8>> {
        AlignedAllocator::allocate(self, size)
    }
    
    fn allocate_layout(&self, layout: Layout) -> Option<NonNull<u8>> {
        // Use the regular allocate method but ensure alignment is respected
        self.allocate(layout.size())
    }
    
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        AlignedAllocator::deallocate(self, ptr, layout.size());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_global_allocator_settings() {
        // Test setting and getting global allocator strategy
        set_global_allocator(AllocStrategy::Pool);
        assert_eq!(get_global_allocator_strategy(), AllocStrategy::Pool);

        set_global_allocator(AllocStrategy::Arena);
        assert_eq!(get_global_allocator_strategy(), AllocStrategy::Arena);

        reset_global_allocator();
        assert_eq!(get_global_allocator_strategy(), AllocStrategy::Standard);
    }

    #[test]
    fn test_recommended_strategies() {
        // Test strategy recommendations for different workloads
        
        // Large allocation should use standard allocator
        let large_alloc_strategy = recommend_strategy(2_000_000, AllocFrequency::Low, false);
        assert_eq!(large_alloc_strategy, AllocStrategy::Standard);
        
        // Small, very frequent allocations should use pool
        let small_freq_strategy = recommend_strategy(100, AllocFrequency::VeryHigh, false);
        assert_eq!(small_freq_strategy, AllocStrategy::Pool);
        
        // Medium, high frequency allocations should use arena
        let medium_freq_strategy = recommend_strategy(10_000, AllocFrequency::High, false);
        assert_eq!(medium_freq_strategy, AllocStrategy::Arena);
        
        // SIMD operations should use aligned allocator
        let simd_strategy = recommend_strategy(1024, AllocFrequency::Medium, true);
        assert_eq!(simd_strategy, AllocStrategy::Aligned);
    }

    #[test]
    fn test_standard_allocator() {
        let allocator = StandardAllocator;
        
        // Allocate some memory
        let layout = Layout::from_size_align(100, 8).unwrap();
        let ptr = allocator.allocate_layout(layout).expect("Allocation should succeed");
        
        // Write to the memory to ensure it's valid
        unsafe {
            let slice = std::slice::from_raw_parts_mut(ptr.as_ptr(), 100);
            for i in 0..100 {
                slice[i] = i as u8;
            }
            
            // Read back and verify
            for i in 0..100 {
                assert_eq!(slice[i], i as u8);
            }
            
            // Deallocate
            allocator.deallocate(ptr, layout);
        }
    }

    #[test]
    fn test_auto_allocator() {
        let allocator = AutoAllocator::new();
        
        // Test small allocation (should use pool)
        let small_ptr = allocator.allocate(100).expect("Small allocation should succeed");
        
        // Test medium allocation (should use arena)
        let medium_ptr = allocator.allocate(10_000).expect("Medium allocation should succeed");
        
        // Test large allocation (should use standard)
        let large_ptr = allocator.allocate(100_000).expect("Large allocation should succeed");
        
        // Test aligned allocation (should use aligned)
        let layout = Layout::from_size_align(64, 64).unwrap();
        let aligned_ptr = allocator.allocate_layout(layout).expect("Aligned allocation should succeed");
        assert_eq!(aligned_ptr.as_ptr() as usize % 64, 0, "Should be 64-byte aligned");
        
        // Deallocate all
        unsafe {
            allocator.deallocate(small_ptr, Layout::from_size_align(100, 8).unwrap());
            allocator.deallocate(medium_ptr, Layout::from_size_align(10_000, 8).unwrap());
            allocator.deallocate(large_ptr, Layout::from_size_align(100_000, 8).unwrap());
            allocator.deallocate(aligned_ptr, layout);
        }
    }

    #[test]
    fn test_get_default_allocator() {
        // Test getting the default allocator with different strategies
        
        // Standard
        set_global_allocator(AllocStrategy::Standard);
        let allocator = get_default_allocator();
        let ptr = allocator.allocate(100).expect("Allocation should succeed");
        unsafe {
            allocator.deallocate(ptr, Layout::from_size_align(100, 8).unwrap());
        }
        
        // Pool
        set_global_allocator(AllocStrategy::Pool);
        let allocator = get_default_allocator();
        let ptr = allocator.allocate(100).expect("Allocation should succeed");
        unsafe {
            allocator.deallocate(ptr, Layout::from_size_align(100, 8).unwrap());
        }
        
        // Reset to standard
        reset_global_allocator();
    }
}