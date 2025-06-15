//! Enhanced trait implementations for memory allocators
//!
//! This module provides implementations of the new memory management trait hierarchy,
//! integrating with the existing allocator infrastructure while providing enhanced
//! functionality for the refactored NumRS2 architecture.

use crate::error::{NumRs2Error, Result};
use crate::traits::{
    AllocationFrequency, AllocationLifetime, AllocationRequirements, AllocationStats,
    AllocationStrategy, ArrayAllocator, MemoryAllocator as NewMemoryAllocator,
    SpecializedAllocator, StrategyStats, ThreadingRequirements,
};
use std::alloc::Layout;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex};

// Bridge the old and new allocator traits
use super::strategy::{MemoryAllocator as OldMemoryAllocator, StandardAllocator};
use super::{AlignedAllocator, ArenaAllocator, PoolAllocator};

// =============================================================================
// BRIDGE IMPLEMENTATIONS FOR EXISTING ALLOCATORS
// =============================================================================

/// Wrapper to bridge old MemoryAllocator trait to new enhanced trait system
#[derive(Debug, Clone)]
pub struct EnhancedAllocatorBridge<T: OldMemoryAllocator> {
    inner: T,
    stats: Arc<Mutex<AllocationStats>>,
}

impl<T: OldMemoryAllocator + std::fmt::Debug> EnhancedAllocatorBridge<T> {
    pub fn new(allocator: T) -> Self {
        Self {
            inner: allocator,
            stats: Arc::new(Mutex::new(AllocationStats::default())),
        }
    }
}

impl<T: OldMemoryAllocator + std::fmt::Debug> NewMemoryAllocator for EnhancedAllocatorBridge<T> {
    type Error = NumRs2Error;

    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>> {
        let ptr = self.inner.allocate_layout(layout).ok_or_else(|| {
            NumRs2Error::AllocationFailed(format!("Failed to allocate {} bytes", layout.size()))
        })?;

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.bytes_allocated += layout.size();
            stats.active_allocations += 1;
            stats.allocation_count += 1;
            if stats.bytes_allocated - stats.bytes_deallocated > stats.peak_usage {
                stats.peak_usage = stats.bytes_allocated - stats.bytes_deallocated;
            }
        }

        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) -> Result<()> {
        self.inner.deallocate(ptr, layout);

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            stats.bytes_deallocated += layout.size();
            stats.active_allocations = stats.active_allocations.saturating_sub(1);
            stats.deallocation_count += 1;
        }

        Ok(())
    }

    unsafe fn reallocate(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<u8>> {
        // Simple implementation: allocate new, copy, deallocate old
        let new_ptr = self.allocate(new_layout)?;

        let copy_size = std::cmp::min(old_layout.size(), new_layout.size());
        std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), copy_size);

        self.deallocate(ptr, old_layout)?;

        Ok(new_ptr)
    }

    fn statistics(&self) -> Option<AllocationStats> {
        self.stats.lock().ok().map(|stats| stats.clone())
    }

    fn supports_layout(&self, _layout: Layout) -> bool {
        true // Most allocators support arbitrary layouts
    }

    fn preferred_alignment(&self) -> usize {
        std::mem::align_of::<usize>()
    }
}

impl<T: OldMemoryAllocator + std::fmt::Debug> SpecializedAllocator for EnhancedAllocatorBridge<T> {
    fn allocation_error(&self, msg: &str) -> Self::Error {
        NumRs2Error::AllocationFailed(msg.to_string())
    }
}

// =============================================================================
// SPECIALIZED ALLOCATOR IMPLEMENTATIONS
// =============================================================================

/// High-performance allocator specifically designed for numerical arrays
#[derive(Debug, Clone)]
pub struct NumericalArrayAllocator {
    inner: EnhancedAllocatorBridge<StandardAllocator>,
    alignment_preference: usize,
}

impl Default for NumericalArrayAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl NumericalArrayAllocator {
    pub fn new() -> Self {
        Self {
            inner: EnhancedAllocatorBridge::new(StandardAllocator),
            alignment_preference: 32, // 256-bit SIMD alignment
        }
    }

    pub fn with_alignment(alignment: usize) -> Self {
        Self {
            inner: EnhancedAllocatorBridge::new(StandardAllocator),
            alignment_preference: alignment,
        }
    }
}

impl NewMemoryAllocator for NumericalArrayAllocator {
    type Error = NumRs2Error;

    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>> {
        // For numerical data, prefer aligned allocation
        let aligned_layout = Layout::from_size_align(
            layout.size(),
            std::cmp::max(layout.align(), self.alignment_preference),
        )
        .map_err(|_| {
            NumRs2Error::AllocationFailed("Invalid layout for numerical array".to_string())
        })?;

        self.inner.allocate(aligned_layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) -> Result<()> {
        self.inner.deallocate(ptr, layout)
    }

    unsafe fn reallocate(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<u8>> {
        self.inner.reallocate(ptr, old_layout, new_layout)
    }

    fn statistics(&self) -> Option<AllocationStats> {
        self.inner.statistics()
    }

    fn supports_layout(&self, layout: Layout) -> bool {
        // We support layouts that can be aligned to our preference
        layout.align() <= self.alignment_preference
    }

    fn preferred_alignment(&self) -> usize {
        self.alignment_preference
    }
}

impl SpecializedAllocator for NumericalArrayAllocator {
    fn allocation_error(&self, msg: &str) -> Self::Error {
        NumRs2Error::AllocationFailed(msg.to_string())
    }
}

impl ArrayAllocator for NumericalArrayAllocator {
    type Error = NumRs2Error;

    fn allocate_array<T>(&self, len: usize) -> std::result::Result<NonNull<T>, Self::Error> {
        let size = len * std::mem::size_of::<T>();
        let alignment = std::cmp::max(std::mem::align_of::<T>(), self.alignment_preference);
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| NumRs2Error::AllocationFailed("Invalid array layout".to_string()))?;

        self.allocate(layout).map(|ptr| ptr.cast::<T>())
    }

    fn allocate_simd_aligned<T>(
        &self,
        len: usize,
        alignment: usize,
    ) -> std::result::Result<NonNull<T>, Self::Error> {
        let size = len * std::mem::size_of::<T>();
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| NumRs2Error::AllocationFailed("Invalid SIMD layout".to_string()))?;

        self.allocate(layout).map(|ptr| ptr.cast::<T>())
    }
}

// =============================================================================
// INTELLIGENT ALLOCATION STRATEGY
// =============================================================================

/// Intelligent allocation strategy that selects the best allocator based on requirements
#[derive(Debug)]
pub struct IntelligentAllocationStrategy {
    stats: Arc<Mutex<StrategyStats>>,
    allocator_cache:
        Arc<Mutex<HashMap<String, Box<dyn SpecializedAllocator<Error = NumRs2Error>>>>>,
}

impl Default for IntelligentAllocationStrategy {
    fn default() -> Self {
        Self::new()
    }
}

impl IntelligentAllocationStrategy {
    pub fn new() -> Self {
        Self {
            stats: Arc::new(Mutex::new(StrategyStats::default())),
            allocator_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn select_allocator_type(&self, requirements: &AllocationRequirements) -> String {
        // Intelligent selection based on requirements
        match (
            requirements.size,
            requirements.frequency,
            requirements.simd_usage,
            requirements.lifetime,
        ) {
            // Large allocations with standard system allocator
            (size, _, _, _) if size > 1_000_000 => "standard".to_string(),

            // Very small, frequent allocations work well with pool
            (size, AllocationFrequency::VeryHigh, _, AllocationLifetime::Temporary)
                if size < 8192 =>
            {
                "pool".to_string()
            }

            // Medium, frequent allocations work well with arena
            (size, AllocationFrequency::High, _, lifetime)
                if size < 65536
                    && matches!(
                        lifetime,
                        AllocationLifetime::Temporary | AllocationLifetime::ShortTerm
                    ) =>
            {
                "arena".to_string()
            }

            // SIMD operations need aligned memory
            (_, _, true, _) => "aligned".to_string(),

            // Numerical arrays get specialized treatment
            (size, freq, _, _) if size > 1024 && !matches!(freq, AllocationFrequency::VeryHigh) => {
                "numerical".to_string()
            }

            // Default to standard allocator
            _ => "standard".to_string(),
        }
    }

    fn create_allocator(
        &self,
        allocator_type: &str,
    ) -> Box<dyn SpecializedAllocator<Error = NumRs2Error>> {
        match allocator_type {
            "standard" => Box::new(EnhancedAllocatorBridge::new(StandardAllocator)),
            "pool" => Box::new(EnhancedAllocatorBridge::new(PoolAllocator::new(
                super::pool::PoolConfig::default(),
            ))),
            "arena" => Box::new(EnhancedAllocatorBridge::new(ArenaAllocator::new(
                super::arena::ArenaConfig::default(),
            ))),
            "aligned" => Box::new(EnhancedAllocatorBridge::new(AlignedAllocator::new(
                super::aligned::AlignmentConfig::default(),
            ))),
            "numerical" => Box::new(NumericalArrayAllocator::new()),
            _ => Box::new(EnhancedAllocatorBridge::new(StandardAllocator)),
        }
    }
}

impl AllocationStrategy for IntelligentAllocationStrategy {
    fn select_allocator(
        &self,
        requirements: &AllocationRequirements,
    ) -> Box<dyn SpecializedAllocator<Error = NumRs2Error>> {
        let allocator_type = self.select_allocator_type(requirements);

        // Update statistics
        if let Ok(mut stats) = self.stats.lock() {
            *stats
                .allocator_selections
                .entry(allocator_type.clone())
                .or_insert(0) += 1;
            stats.total_requests += 1;
        }

        // Check cache first
        if let Ok(mut cache) = self.allocator_cache.lock() {
            if let Some(allocator) = cache.remove(&allocator_type) {
                return allocator;
            }
        }

        // Create new allocator
        self.create_allocator(&allocator_type)
    }

    fn strategy_stats(&self) -> StrategyStats {
        self.stats
            .lock()
            .map(|stats| stats.clone())
            .unwrap_or_default()
    }
}

// =============================================================================
// HELPER FUNCTIONS FOR REQUIREMENTS DETECTION
// =============================================================================

impl AllocationRequirements {
    /// Create requirements for a typical numerical array allocation
    pub fn for_array<T>(len: usize) -> Self {
        let size = len * std::mem::size_of::<T>();
        Self {
            size,
            alignment: std::mem::align_of::<T>(),
            frequency: if size < 1024 {
                AllocationFrequency::High
            } else {
                AllocationFrequency::Medium
            },
            simd_usage: std::mem::align_of::<T>() >= 16, // Assume SIMD for well-aligned types
            lifetime: AllocationLifetime::MediumTerm,
            threading: ThreadingRequirements::MultiThreadedRead,
        }
    }

    /// Create requirements for temporary computation buffers
    pub fn for_temporary_buffer(size: usize) -> Self {
        Self {
            size,
            alignment: 8,
            frequency: AllocationFrequency::High,
            simd_usage: false,
            lifetime: AllocationLifetime::Temporary,
            threading: ThreadingRequirements::SingleThreaded,
        }
    }

    /// Create requirements for SIMD-optimized operations
    pub fn for_simd_operation<T>(len: usize, alignment: usize) -> Self {
        let size = len * std::mem::size_of::<T>();
        Self {
            size,
            alignment,
            frequency: AllocationFrequency::Medium,
            simd_usage: true,
            lifetime: AllocationLifetime::ShortTerm,
            threading: ThreadingRequirements::MultiThreadedRead,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::Layout;

    #[test]
    fn test_enhanced_allocator_bridge() {
        let allocator = EnhancedAllocatorBridge::new(StandardAllocator);

        let layout = Layout::from_size_align(1024, 8).unwrap();
        let ptr = allocator
            .allocate(layout)
            .expect("Allocation should succeed");

        // Check statistics
        let stats = allocator.statistics().unwrap();
        assert_eq!(stats.bytes_allocated, 1024);
        assert_eq!(stats.active_allocations, 1);

        unsafe {
            allocator
                .deallocate(ptr, layout)
                .expect("Deallocation should succeed");
        }

        let stats = allocator.statistics().unwrap();
        assert_eq!(stats.bytes_deallocated, 1024);
        assert_eq!(stats.active_allocations, 0);
    }

    #[test]
    fn test_numerical_array_allocator() {
        let allocator = NumericalArrayAllocator::new();

        // Test array allocation
        let ptr = allocator
            .allocate_array::<f64>(100)
            .expect("Array allocation should succeed");

        // Verify alignment
        assert_eq!(ptr.as_ptr() as usize % 32, 0, "Should be 32-byte aligned");

        let layout = Layout::array::<f64>(100).unwrap();
        unsafe {
            allocator
                .deallocate(ptr.cast(), layout)
                .expect("Deallocation should succeed");
        }
    }

    #[test]
    fn test_intelligent_allocation_strategy() {
        let strategy = IntelligentAllocationStrategy::new();

        // Test different requirement scenarios
        let array_req = AllocationRequirements::for_array::<f64>(1000);
        let allocator = strategy.select_allocator(&array_req);
        assert!(allocator.supports_layout(Layout::from_size_align(8000, 8).unwrap()));

        let simd_req = AllocationRequirements::for_simd_operation::<f32>(256, 32);
        let allocator = strategy.select_allocator(&simd_req);
        assert!(allocator.preferred_alignment() >= 8);

        // Check statistics
        let stats = strategy.strategy_stats();
        assert!(stats.total_requests >= 2);
    }

    #[test]
    fn test_allocation_requirements_creation() {
        let array_req = AllocationRequirements::for_array::<f64>(1000);
        assert_eq!(array_req.size, 8000);
        assert_eq!(array_req.alignment, 8);

        let simd_req = AllocationRequirements::for_simd_operation::<f32>(64, 32);
        assert_eq!(simd_req.size, 256);
        assert_eq!(simd_req.alignment, 32);
        assert!(simd_req.simd_usage);

        let temp_req = AllocationRequirements::for_temporary_buffer(512);
        assert_eq!(temp_req.size, 512);
        assert_eq!(temp_req.lifetime, AllocationLifetime::Temporary);
    }
}
