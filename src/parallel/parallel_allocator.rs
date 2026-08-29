//! Parallel memory allocation strategies
//!
//! This module provides thread-safe memory allocators optimized for
//! parallel numerical computations with minimal contention.

use crate::error::{NumRs2Error, Result};
use crate::traits::{AllocationStats, MemoryAllocator, SpecializedAllocator};
use std::alloc::Layout;
use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, ThreadId};
use std::time::{Duration, Instant};

thread_local! {
    /// Thread-local allocator for reduced contention
    static LOCAL_ALLOCATOR: RefCell<Option<ThreadLocalState>> = const { RefCell::new(None) };
}

/// Thread-local allocation state
#[derive(Debug)]
struct ThreadLocalState {
    allocator: Box<dyn MemoryAllocator<Error = NumRs2Error> + Send>,
    stats: AllocationStats,
    last_gc: Instant,
    cache_size_limit: usize,
    cached_blocks: Vec<CachedBlock>,
}

#[derive(Debug)]
struct CachedBlock {
    ptr: NonNull<u8>,
    layout: Layout,
    allocated_at: Instant,
}

// SAFETY: `CachedBlock` holds `ptr: NonNull<u8>`, `layout: Layout`, and
// `allocated_at: Instant`; only `NonNull<u8>` prevents the auto-derived
// Send/Sync (raw pointers are `!Send`/`!Sync` by default), and `Layout`/
// `Instant` are already `Send + Sync` so they impose no extra constraint.
//
// Send: `ptr` addresses a raw, uninterpreted byte allocation (`u8`, not a
// generic `T` that could itself be `!Send`, e.g. an `Rc<_>`), so moving a
// `CachedBlock` to another thread carries no thread-affine state -- only an
// address and a size/alignment description. `CachedBlock` is not `Clone` or
// `Copy`, so ordinary Rust ownership rules guarantee the pointer has a
// single owner at a time as it moves between the per-thread cache
// (`ThreadLocalState::cached_blocks`) and the cross-thread `global_pool:
// Arc<Mutex<Vec<CachedBlock>>>` below; there is no path that duplicates the
// pointer while a `CachedBlock` is aliased.
//
// Sync: a shared `&CachedBlock` only exposes `Copy` field reads
// (`NonNull<u8>`, `Layout`, `Instant`); no method dereferences `ptr` through
// a shared reference, so concurrent readers cannot race on the pointee.
// (Any *dereference* of the allocation itself happens through the owning
// allocator's own synchronization -- e.g. the `Arc<Mutex<_>>` wrappers this
// module uses -- not through `CachedBlock` directly.)
unsafe impl Send for CachedBlock {}
unsafe impl Sync for CachedBlock {}

impl ThreadLocalState {
    fn new<A>(allocator: A, cache_size_limit: usize) -> Self
    where
        A: MemoryAllocator<Error = NumRs2Error> + Send + 'static,
    {
        Self {
            allocator: Box::new(allocator),
            stats: AllocationStats::default(),
            last_gc: Instant::now(),
            cache_size_limit,
            cached_blocks: Vec::new(),
        }
    }

    /// Take a cached block back out of the per-thread cache.
    ///
    /// Matching is *exact* on both size and alignment, not "big enough".
    /// A cached block carries the layout it will eventually be freed with,
    /// and handing a 128-byte block out for a 64-byte request means the next
    /// `deallocate` re-caches that same pointer as a 64-byte block -- so the
    /// final `dealloc` would name a layout that never matched the real
    /// allocation, which is undefined behaviour. Exact matching keeps every
    /// `CachedBlock::layout` equal to the layout its pointer was actually
    /// allocated with, which is what makes the `Drop` impls below sound.
    /// Requests that differ only in alignment below the base allocator's
    /// preference simply miss the cache; that costs a fresh allocation, and
    /// nothing more.
    fn try_allocate_from_cache(&mut self, layout: Layout) -> Option<NonNull<u8>> {
        let index = self
            .cached_blocks
            .iter()
            .position(|block| block.layout == layout)?;
        Some(self.cached_blocks.remove(index).ptr)
    }

    fn cache_block(&mut self, ptr: NonNull<u8>, layout: Layout) {
        if self.cached_blocks.len() < self.cache_size_limit {
            self.cached_blocks.push(CachedBlock {
                ptr,
                layout,
                allocated_at: Instant::now(),
            });
        } else {
            // Cache is full, actually deallocate
            unsafe {
                let _ = self.allocator.deallocate(ptr, layout);
            }
        }
    }

    fn garbage_collect(&mut self, max_age: Duration) {
        let now = Instant::now();
        self.cached_blocks.retain(|block| {
            if now.duration_since(block.allocated_at) > max_age {
                // Block is too old, deallocate it
                unsafe {
                    let _ = self.allocator.deallocate(block.ptr, block.layout);
                }
                false
            } else {
                true
            }
        });
        self.last_gc = now;
    }

    fn should_gc(&self, gc_interval: Duration) -> bool {
        self.last_gc.elapsed() > gc_interval
    }
}

impl Drop for ThreadLocalState {
    /// Return every still-cached block to the base allocator.
    ///
    /// `cached_blocks` holds raw allocations that the cache deliberately did
    /// *not* free when the owner deallocated them, so that a later request of
    /// the same layout could reuse them. Nothing else owns those pointers:
    /// once this state goes away -- with the owning [`ParallelAllocator`], or
    /// at thread exit for the `LOCAL_ALLOCATOR` thread-local -- the addresses
    /// are gone and the memory is unreachable. Without this impl every block
    /// left warm in a cache at teardown is a genuine leak, which is what Miri
    /// reported for all nine `parallel_allocator` tests.
    ///
    /// Frees through `self.allocator`, the state's own allocator handle, and
    /// not through some outer allocator. That is required, not stylistic: in
    /// `ParallelAllocator` the `base_allocator` field is declared *before*
    /// `thread_allocators`, so it is already dropped by the time the map
    /// releases the last `Arc` to a state and this impl runs. It is also the
    /// only correct handle for the `LOCAL_ALLOCATOR` thread-local, whose
    /// blocks come from `state.allocator` in the first place.
    fn drop(&mut self) {
        for block in self.cached_blocks.drain(..) {
            // SAFETY: `block.ptr` came from `self.allocator` (the cache is
            // only ever filled by `cache_block`, whose callers just received
            // the pointer from this same allocator), `block.layout` is the
            // exact layout it was allocated with -- `try_allocate_from_cache`
            // matches layouts exactly, so a cached layout can never drift
            // away from its allocation -- and draining the vector means no
            // other copy of the pointer survives to be freed twice.
            unsafe {
                let _ = self.allocator.deallocate(block.ptr, block.layout);
            }
        }
    }
}

/// Configuration for parallel allocator
#[derive(Debug, Clone)]
pub struct ParallelAllocatorConfig {
    /// Enable thread-local caching
    pub enable_thread_local_cache: bool,
    /// Maximum cached blocks per thread
    pub max_cached_blocks_per_thread: usize,
    /// Garbage collection interval
    pub gc_interval: Duration,
    /// Maximum age for cached blocks
    pub max_block_age: Duration,
    /// Enable NUMA-aware allocation
    pub numa_aware: bool,
    /// Global pool size for shared allocations
    pub global_pool_size: usize,
    /// Enable allocation tracking
    pub enable_tracking: bool,
}

impl Default for ParallelAllocatorConfig {
    fn default() -> Self {
        Self {
            enable_thread_local_cache: true,
            max_cached_blocks_per_thread: 100,
            gc_interval: Duration::from_secs(30),
            max_block_age: Duration::from_secs(300),
            numa_aware: false,
            global_pool_size: 1024 * 1024, // 1MB
            enable_tracking: true,
        }
    }
}

/// Parallel memory allocator with thread-local optimization
pub struct ParallelAllocator<A>
where
    A: MemoryAllocator<Error = NumRs2Error> + Send + Sync + Clone,
{
    base_allocator: A,
    config: ParallelAllocatorConfig,
    global_stats: Arc<RwLock<AllocationStats>>,
    thread_allocators: Arc<Mutex<HashMap<ThreadId, Arc<Mutex<ThreadLocalState>>>>>,
    global_pool: Arc<Mutex<Vec<CachedBlock>>>,
}

impl<A> std::fmt::Debug for ParallelAllocator<A>
where
    A: MemoryAllocator<Error = NumRs2Error> + Send + Sync + Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelAllocator")
            .field("base_allocator", &"<allocator>")
            .field("config", &self.config)
            .field("global_stats", &"<mutex>")
            .field("thread_allocators", &"<mutex>")
            .field("global_pool", &"<mutex>")
            .finish()
    }
}

impl<A> ParallelAllocator<A>
where
    A: MemoryAllocator<Error = NumRs2Error> + Send + Sync + Clone + 'static,
{
    /// Create a new parallel allocator
    pub fn new(base_allocator: A, config: ParallelAllocatorConfig) -> Self {
        Self {
            base_allocator,
            config,
            global_stats: Arc::new(RwLock::new(AllocationStats::default())),
            thread_allocators: Arc::new(Mutex::new(HashMap::new())),
            global_pool: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Get or create thread-local state
    fn get_thread_local_state(&self) -> Result<Arc<Mutex<ThreadLocalState>>> {
        let thread_id = thread::current().id();
        let mut allocators = self
            .thread_allocators
            .lock()
            .expect("lock should not be poisoned");

        if let Some(state) = allocators.get(&thread_id) {
            Ok(Arc::clone(state))
        } else {
            let local_state = ThreadLocalState::new(
                self.base_allocator.clone(),
                self.config.max_cached_blocks_per_thread,
            );
            let state = Arc::new(Mutex::new(local_state));
            allocators.insert(thread_id, Arc::clone(&state));
            Ok(state)
        }
    }

    /// Try to allocate from global pool
    ///
    /// Matches layouts exactly, for the same reason
    /// [`ThreadLocalState::try_allocate_from_cache`] does: a pooled block is
    /// eventually freed with the layout recorded alongside it, so handing an
    /// oversized block to a smaller request would let that recorded layout
    /// shrink below the real allocation and turn the eventual `dealloc` into
    /// undefined behaviour.
    fn try_allocate_from_global_pool(&self, layout: Layout) -> Option<NonNull<u8>> {
        if !self.config.enable_thread_local_cache {
            return None;
        }

        let mut pool = self
            .global_pool
            .lock()
            .expect("lock should not be poisoned");

        let index = pool.iter().position(|block| block.layout == layout)?;
        Some(pool.remove(index).ptr)
    }

    /// Return block to global pool
    fn return_to_global_pool(&self, ptr: NonNull<u8>, layout: Layout) {
        if !self.config.enable_thread_local_cache {
            unsafe {
                let _ = self.base_allocator.deallocate(ptr, layout);
            }
            return;
        }

        let mut pool = self
            .global_pool
            .lock()
            .expect("lock should not be poisoned");

        if pool.len() < self.config.global_pool_size / std::mem::size_of::<CachedBlock>() {
            pool.push(CachedBlock {
                ptr,
                layout,
                allocated_at: Instant::now(),
            });
        } else {
            // Pool is full, actually deallocate
            unsafe {
                let _ = self.base_allocator.deallocate(ptr, layout);
            }
        }
    }

    /// Trigger garbage collection for all thread-local caches
    pub fn garbage_collect_all(&self) -> Result<()> {
        let allocators = self
            .thread_allocators
            .lock()
            .expect("lock should not be poisoned");

        for state in allocators.values() {
            if let Ok(mut local_state) = state.try_lock() {
                local_state.garbage_collect(self.config.max_block_age);
            }
        }

        // Also clean global pool
        {
            let mut pool = self
                .global_pool
                .lock()
                .expect("lock should not be poisoned");
            let now = Instant::now();
            pool.retain(|block| {
                if now.duration_since(block.allocated_at) > self.config.max_block_age {
                    unsafe {
                        let _ = self.base_allocator.deallocate(block.ptr, block.layout);
                    }
                    false
                } else {
                    true
                }
            });
        }

        Ok(())
    }

    /// Get aggregate statistics from all threads
    pub fn aggregate_statistics(&self) -> AllocationStats {
        let global_stats = self
            .global_stats
            .read()
            .expect("lock should not be poisoned")
            .clone();
        let allocators = self
            .thread_allocators
            .lock()
            .expect("lock should not be poisoned");

        let mut aggregate = global_stats;

        for state in allocators.values() {
            if let Ok(local_state) = state.try_lock() {
                aggregate.bytes_allocated += local_state.stats.bytes_allocated;
                aggregate.bytes_deallocated += local_state.stats.bytes_deallocated;
                aggregate.allocation_count += local_state.stats.allocation_count;
                aggregate.deallocation_count += local_state.stats.deallocation_count;
                aggregate.active_allocations += local_state.stats.active_allocations;
                aggregate.peak_usage = aggregate.peak_usage.max(local_state.stats.peak_usage);
            }
        }

        aggregate
    }

    /// Get number of cached blocks across all threads
    pub fn total_cached_blocks(&self) -> usize {
        let allocators = self
            .thread_allocators
            .lock()
            .expect("lock should not be poisoned");
        let mut total = 0;

        for state in allocators.values() {
            if let Ok(local_state) = state.try_lock() {
                total += local_state.cached_blocks.len();
            }
        }

        total += self
            .global_pool
            .lock()
            .expect("lock should not be poisoned")
            .len();
        total
    }

    /// Give every block in `blocks` back to the base allocator, keeping any
    /// block whose deallocation failed and reporting the first failure
    /// through `first_error`.
    ///
    /// Deliberately not `for block in blocks.drain(..) { ...? }`. Returning
    /// early out of a `Drain` loop still *completes* the drain when the
    /// iterator is dropped, so every block the loop had not reached yet would
    /// be removed from the vector and never freed -- and unreachable
    /// afterwards, because the `CachedBlock` records the teardown `Drop`
    /// impls would have used are gone with it. `retain` drops a record only
    /// once its memory is genuinely back with the allocator, so a failure
    /// leaves the remaining blocks cached and still owned.
    fn release_blocks(&self, blocks: &mut Vec<CachedBlock>, first_error: &mut Option<NumRs2Error>) {
        blocks.retain(|block| {
            // SAFETY: every cached block was handed to the cache straight
            // from this allocator's `allocate`, paired with the layout it was
            // allocated under; layout matching in `try_allocate_from_cache`
            // and `try_allocate_from_global_pool` is exact, so a cached
            // layout can never drift away from its allocation. A block is
            // removed from the vector only on success, so no pointer is
            // offered for freeing twice.
            match unsafe { self.base_allocator.deallocate(block.ptr, block.layout) } {
                Ok(()) => false,
                Err(err) => {
                    if first_error.is_none() {
                        *first_error = Some(err);
                    }
                    true
                }
            }
        });
    }

    /// Force cleanup of all cached memory
    pub fn force_cleanup(&self) -> Result<()> {
        let mut first_error: Option<NumRs2Error> = None;

        // Clean thread-local caches
        {
            let allocators = self
                .thread_allocators
                .lock()
                .expect("lock should not be poisoned");

            for state in allocators.values() {
                if let Ok(mut local_state) = state.try_lock() {
                    self.release_blocks(&mut local_state.cached_blocks, &mut first_error);
                }
            }
        }

        // Clean global pool
        {
            let mut pool = self
                .global_pool
                .lock()
                .expect("lock should not be poisoned");
            self.release_blocks(&mut pool, &mut first_error);
        }

        match first_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }
}

impl<A> Drop for ParallelAllocator<A>
where
    A: MemoryAllocator<Error = NumRs2Error> + Send + Sync + Clone,
{
    /// Return the shared pool's still-cached blocks to the base allocator.
    ///
    /// Like the per-thread caches, `global_pool` holds raw allocations that
    /// `return_to_global_pool` deliberately kept alive for reuse; dropping
    /// the `Vec` frees the `CachedBlock` records but not the memory they
    /// point at. The per-thread caches need no handling here: dropping
    /// `thread_allocators` drops the last `Arc` to each `ThreadLocalState`,
    /// and that type's own `Drop` frees its blocks through its own clone of
    /// the base allocator.
    ///
    /// Using `self.base_allocator` is safe here specifically because
    /// `Drop::drop` runs before *any* field is dropped; the per-thread states
    /// cannot rely on it (see `ThreadLocalState`'s own `Drop`) because they
    /// are released later, after `base_allocator` is gone.
    ///
    /// A block that fails to deallocate is dropped from the pool and leaks
    /// exactly itself -- there is no `?` here, so one failure cannot strand
    /// the blocks behind it, which is the hazard
    /// `ParallelAllocator::release_blocks` exists to avoid on the
    /// `force_cleanup` path.
    ///
    /// The bounds are written to match the struct declaration exactly, as
    /// `Drop` impls must (no `'static`, unlike the inherent impls below).
    fn drop(&mut self) {
        let Ok(mut pool) = self.global_pool.lock() else {
            // A poisoned pool lock means some thread panicked mid-update and
            // the `Vec` may not describe the live blocks any more. Freeing
            // from it could double-free, which is far worse than leaking, so
            // leave the memory alone.
            return;
        };

        for block in pool.drain(..) {
            // SAFETY: every block in the pool was placed there by
            // `return_to_global_pool` with the pointer and layout it had just
            // received from `self.base_allocator`, and layout matching in
            // `try_allocate_from_global_pool` is exact, so `block.layout` is
            // still the allocation's true layout. Draining consumes each
            // record, so no surviving copy can free the same pointer again.
            unsafe {
                let _ = self.base_allocator.deallocate(block.ptr, block.layout);
            }
        }
    }
}

impl<A> MemoryAllocator for ParallelAllocator<A>
where
    A: MemoryAllocator<Error = NumRs2Error> + Send + Sync + Clone + 'static,
{
    type Error = NumRs2Error;

    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>> {
        // Try thread-local cache first
        if self.config.enable_thread_local_cache {
            let state = self.get_thread_local_state()?;
            let mut local_state = state.lock().expect("lock should not be poisoned");

            // Check if we should do garbage collection
            if local_state.should_gc(self.config.gc_interval) {
                local_state.garbage_collect(self.config.max_block_age);
            }

            // Try to allocate from thread-local cache
            if let Some(ptr) = local_state.try_allocate_from_cache(layout) {
                local_state.stats.allocation_count += 1;
                local_state.stats.active_allocations += 1;
                return Ok(ptr);
            }
        }

        // Try global pool
        if let Some(ptr) = self.try_allocate_from_global_pool(layout) {
            if self.config.enable_tracking {
                let mut stats = self
                    .global_stats
                    .write()
                    .expect("lock should not be poisoned");
                stats.allocation_count += 1;
                stats.active_allocations += 1;
            }
            return Ok(ptr);
        }

        // Allocate new memory from base allocator
        let ptr = self.base_allocator.allocate(layout)?;

        if self.config.enable_tracking {
            let mut stats = self
                .global_stats
                .write()
                .expect("lock should not be poisoned");
            stats.bytes_allocated += layout.size();
            stats.allocation_count += 1;
            stats.active_allocations += 1;
            stats.peak_usage = stats
                .peak_usage
                .max(stats.bytes_allocated - stats.bytes_deallocated);
        }

        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) -> Result<()> {
        // Try to cache the block for reuse
        if self.config.enable_thread_local_cache {
            if let Ok(state) = self.get_thread_local_state() {
                let mut local_state = state.lock().expect("lock should not be poisoned");
                if local_state.cached_blocks.len() < self.config.max_cached_blocks_per_thread {
                    local_state.cache_block(ptr, layout);
                    local_state.stats.deallocation_count += 1;
                    local_state.stats.active_allocations =
                        local_state.stats.active_allocations.saturating_sub(1);
                    return Ok(());
                }
            }
        }

        // Try global pool
        self.return_to_global_pool(ptr, layout);

        if self.config.enable_tracking {
            let mut stats = self
                .global_stats
                .write()
                .expect("lock should not be poisoned");
            stats.bytes_deallocated += layout.size();
            stats.deallocation_count += 1;
            stats.active_allocations = stats.active_allocations.saturating_sub(1);
        }

        Ok(())
    }

    unsafe fn reallocate(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<u8>> {
        // For simplicity, always allocate new and copy
        let new_ptr = self.allocate(new_layout)?;

        let copy_size = old_layout.size().min(new_layout.size());
        std::ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_ptr(), copy_size);

        self.deallocate(ptr, old_layout)?;

        Ok(new_ptr)
    }

    fn supports_layout(&self, layout: Layout) -> bool {
        self.base_allocator.supports_layout(layout)
    }

    fn preferred_alignment(&self) -> usize {
        self.base_allocator.preferred_alignment()
    }

    fn statistics(&self) -> Option<AllocationStats> {
        if self.config.enable_tracking {
            Some(self.aggregate_statistics())
        } else {
            None
        }
    }
}

impl<A> SpecializedAllocator for ParallelAllocator<A>
where
    A: MemoryAllocator<Error = NumRs2Error> + Send + Sync + Clone + 'static,
{
    fn allocation_error(&self, msg: &str) -> Self::Error {
        NumRs2Error::AllocationFailed(msg.to_string())
    }
}

/// Thread-local allocator that provides zero-contention allocation
pub struct ThreadLocalAllocator {
    config: ParallelAllocatorConfig,
}

impl ThreadLocalAllocator {
    /// Create a new thread-local allocator
    pub fn new(config: ParallelAllocatorConfig) -> Self {
        Self { config }
    }

    /// Initialize thread-local storage for current thread
    pub fn initialize_current_thread<A>(&self, allocator: A) -> Result<()>
    where
        A: MemoryAllocator<Error = NumRs2Error> + Send + 'static,
    {
        LOCAL_ALLOCATOR.with(|local| {
            let mut local_ref = local.borrow_mut();
            if local_ref.is_none() {
                *local_ref = Some(ThreadLocalState::new(
                    allocator,
                    self.config.max_cached_blocks_per_thread,
                ));
            }
        });
        Ok(())
    }

    /// Allocate using thread-local allocator
    pub fn allocate(&self, layout: Layout) -> Result<NonNull<u8>> {
        LOCAL_ALLOCATOR.with(|local| {
            let mut local_ref = local.borrow_mut();

            if let Some(ref mut state) = *local_ref {
                // Check cache first
                if let Some(ptr) = state.try_allocate_from_cache(layout) {
                    state.stats.allocation_count += 1;
                    state.stats.active_allocations += 1;
                    return Ok(ptr);
                }

                // Allocate new
                let ptr = state.allocator.allocate(layout)?;
                state.stats.bytes_allocated += layout.size();
                state.stats.allocation_count += 1;
                state.stats.active_allocations += 1;

                Ok(ptr)
            } else {
                Err(NumRs2Error::RuntimeError(
                    "Thread-local allocator not initialized".to_string(),
                ))
            }
        })
    }

    /// Deallocate using thread-local allocator
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - `ptr` was allocated with this allocator using the same `layout`
    /// - `ptr` is not used after this call
    /// - The memory region pointed to by `ptr` is not accessed concurrently
    /// - The layout matches exactly the layout used during allocation
    pub unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) -> Result<()> {
        LOCAL_ALLOCATOR.with(|local| {
            let mut local_ref = local.borrow_mut();

            if let Some(ref mut state) = *local_ref {
                // Try to cache the block
                if state.cached_blocks.len() < self.config.max_cached_blocks_per_thread {
                    state.cache_block(ptr, layout);
                } else {
                    // Cache full, actually deallocate
                    state.allocator.deallocate(ptr, layout)?;
                }

                state.stats.bytes_deallocated += layout.size();
                state.stats.deallocation_count += 1;
                state.stats.active_allocations = state.stats.active_allocations.saturating_sub(1);

                Ok(())
            } else {
                Err(NumRs2Error::RuntimeError(
                    "Thread-local allocator not initialized".to_string(),
                ))
            }
        })
    }

    /// Get statistics for current thread
    pub fn current_thread_statistics(&self) -> Option<AllocationStats> {
        LOCAL_ALLOCATOR.with(|local| local.borrow().as_ref().map(|state| state.stats.clone()))
    }

    /// Trigger garbage collection for current thread
    pub fn garbage_collect_current_thread(&self) -> Result<()> {
        LOCAL_ALLOCATOR.with(|local| {
            let mut local_ref = local.borrow_mut();

            if let Some(ref mut state) = *local_ref {
                state.garbage_collect(self.config.max_block_age);
            }
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_alloc::NumericalArrayAllocator;
    use std::time::Duration;

    #[test]
    fn test_parallel_allocator_creation() {
        let base = NumericalArrayAllocator::new();
        let config = ParallelAllocatorConfig::default();
        let allocator = ParallelAllocator::new(base, config);

        assert!(allocator.config.enable_thread_local_cache);
        assert_eq!(allocator.total_cached_blocks(), 0);
    }

    #[test]
    fn test_basic_allocation() {
        let base = NumericalArrayAllocator::new();
        let config = ParallelAllocatorConfig::default();
        let allocator = ParallelAllocator::new(base, config);

        let layout =
            Layout::from_size_align(1024, 8).expect("layout with size 1024 and align 8 is valid");
        let ptr = allocator
            .allocate(layout)
            .expect("allocation should succeed");

        unsafe {
            allocator
                .deallocate(ptr, layout)
                .expect("deallocation should succeed");
        }
    }

    #[test]
    fn test_thread_local_caching() {
        let base = NumericalArrayAllocator::new();
        let config = ParallelAllocatorConfig {
            max_cached_blocks_per_thread: 5,
            ..Default::default()
        };
        let allocator = ParallelAllocator::new(base, config);

        let layout =
            Layout::from_size_align(64, 8).expect("layout with size 64 and align 8 is valid");

        // Allocate and deallocate several blocks
        for _ in 0..3 {
            let ptr = allocator
                .allocate(layout)
                .expect("allocation should succeed");
            unsafe {
                allocator
                    .deallocate(ptr, layout)
                    .expect("deallocation should succeed");
            }
        }

        // Should have cached blocks
        assert!(allocator.total_cached_blocks() > 0);
    }

    #[test]
    fn test_statistics_aggregation() {
        let base = NumericalArrayAllocator::new();
        let config = ParallelAllocatorConfig::default();
        let allocator = ParallelAllocator::new(base, config);

        let layout =
            Layout::from_size_align(128, 8).expect("layout with size 128 and align 8 is valid");

        // Make some allocations
        let mut ptrs = Vec::new();
        for _ in 0..5 {
            ptrs.push(
                allocator
                    .allocate(layout)
                    .expect("allocation should succeed"),
            );
        }

        let stats = allocator.aggregate_statistics();
        assert!(stats.allocation_count >= 5);
        assert!(stats.bytes_allocated >= 128 * 5);

        // Clean up
        for ptr in ptrs {
            unsafe {
                allocator
                    .deallocate(ptr, layout)
                    .expect("deallocation should succeed");
            }
        }
    }

    #[test]
    fn test_garbage_collection() {
        let base = NumericalArrayAllocator::new();
        let config = ParallelAllocatorConfig {
            max_block_age: Duration::from_millis(1), // Very short for testing
            ..Default::default()
        };
        let allocator = ParallelAllocator::new(base, config);

        let layout =
            Layout::from_size_align(64, 8).expect("layout with size 64 and align 8 is valid");

        // Allocate and deallocate to create cached blocks
        for _ in 0..3 {
            let ptr = allocator
                .allocate(layout)
                .expect("allocation should succeed");
            unsafe {
                allocator
                    .deallocate(ptr, layout)
                    .expect("deallocation should succeed");
            }
        }

        let initial_cached = allocator.total_cached_blocks();
        assert!(initial_cached > 0);

        // Wait for blocks to age
        std::thread::sleep(Duration::from_millis(10));

        // Trigger garbage collection
        allocator
            .garbage_collect_all()
            .expect("garbage collection should succeed");

        // Should have fewer cached blocks
        let final_cached = allocator.total_cached_blocks();
        assert!(final_cached <= initial_cached);
    }

    #[test]
    fn test_thread_local_allocator() {
        let config = ParallelAllocatorConfig::default();
        let tl_allocator = ThreadLocalAllocator::new(config);

        // Initialize for current thread
        let base = NumericalArrayAllocator::new();
        tl_allocator
            .initialize_current_thread(base)
            .expect("thread-local initialization should succeed");

        let layout =
            Layout::from_size_align(256, 8).expect("layout with size 256 and align 8 is valid");
        let ptr = tl_allocator
            .allocate(layout)
            .expect("allocation should succeed");

        unsafe {
            tl_allocator
                .deallocate(ptr, layout)
                .expect("deallocation should succeed");
        }

        let stats = tl_allocator
            .current_thread_statistics()
            .expect("thread-local stats should be available");
        assert_eq!(stats.allocation_count, 1);
        assert_eq!(stats.deallocation_count, 1);
    }

    #[test]
    fn test_force_cleanup() {
        let base = NumericalArrayAllocator::new();
        let config = ParallelAllocatorConfig::default();
        let allocator = ParallelAllocator::new(base, config);

        let layout =
            Layout::from_size_align(64, 8).expect("layout with size 64 and align 8 is valid");

        // Create some cached blocks
        for _ in 0..3 {
            let ptr = allocator
                .allocate(layout)
                .expect("allocation should succeed");
            unsafe {
                allocator
                    .deallocate(ptr, layout)
                    .expect("deallocation should succeed");
            }
        }

        assert!(allocator.total_cached_blocks() > 0);

        // Force cleanup
        allocator
            .force_cleanup()
            .expect("force cleanup should succeed");

        // Should have no cached blocks
        assert_eq!(allocator.total_cached_blocks(), 0);
    }

    #[test]
    fn test_reallocation() {
        let base = NumericalArrayAllocator::new();
        let config = ParallelAllocatorConfig::default();
        let allocator = ParallelAllocator::new(base, config);

        let old_layout =
            Layout::from_size_align(64, 8).expect("layout with size 64 and align 8 is valid");
        let new_layout =
            Layout::from_size_align(128, 8).expect("layout with size 128 and align 8 is valid");

        let ptr = allocator
            .allocate(old_layout)
            .expect("allocation should succeed");

        unsafe {
            let new_ptr = allocator
                .reallocate(ptr, old_layout, new_layout)
                .expect("reallocation should succeed");
            allocator
                .deallocate(new_ptr, new_layout)
                .expect("deallocation should succeed");
        }
    }

    #[test]
    fn test_multithreaded_allocation() {
        let base = NumericalArrayAllocator::new();
        let config = ParallelAllocatorConfig::default();
        let allocator = Arc::new(ParallelAllocator::new(base, config));

        let mut handles = Vec::new();

        for _ in 0..4 {
            let allocator_clone = Arc::clone(&allocator);
            let handle = std::thread::spawn(move || {
                let layout = Layout::from_size_align(128, 8)
                    .expect("layout with size 128 and align 8 is valid");

                for _ in 0..10 {
                    let ptr = allocator_clone
                        .allocate(layout)
                        .expect("allocation should succeed");
                    unsafe {
                        allocator_clone
                            .deallocate(ptr, layout)
                            .expect("deallocation should succeed");
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().expect("thread should join successfully");
        }

        let stats = allocator.aggregate_statistics();
        assert!(stats.allocation_count >= 40);
    }
}
