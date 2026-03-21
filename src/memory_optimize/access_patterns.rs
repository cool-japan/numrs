//! Memory Access Pattern Optimization
//!
//! This module provides tools for analyzing and optimizing memory access patterns
//! in array operations. Proper memory access patterns are crucial for performance,
//! especially for large arrays where cache efficiency becomes the limiting factor.
//!
//! # Key Concepts
//!
//! ## Memory Layout Detection
//!
//! Arrays can have different memory layouts:
//! - **C-contiguous (row-major)**: Elements in each row are contiguous
//! - **F-contiguous (column-major)**: Elements in each column are contiguous
//! - **Strided**: Non-contiguous with arbitrary strides
//!
//! ## Cache-Aware Iteration
//!
//! This module provides blocked/tiled iterators that process data in cache-friendly
//! chunks, improving temporal and spatial locality.
//!
//! # Examples
//!
//! ```
//! use numrs2::memory_optimize::access_patterns::{
//!     MemoryLayout, CacheConfig, BlockedIterator, TiledIterator2D
//! };
//!
//! // Create a cache configuration for L1 cache
//! let cache = CacheConfig::l1_default();
//!
//! // Create a blocked iterator for a 1000-element array
//! let blocked = BlockedIterator::new(1000, cache.elements_per_block::<f64>());
//!
//! // Iterate in cache-friendly blocks
//! for block in blocked {
//!     println!("Block: {}..{}", block.start, block.end);
//! }
//! ```

/// Memory layout classification for arrays
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryLayout {
    /// C-contiguous (row-major) - last index varies fastest
    #[default]
    CContiguous,
    /// F-contiguous (column-major) - first index varies fastest
    FContiguous,
    /// Non-contiguous with arbitrary strides
    Strided,
    /// Single element or empty array
    Scalar,
}

impl MemoryLayout {
    /// Check if the layout is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        matches!(self, MemoryLayout::CContiguous | MemoryLayout::FContiguous)
    }

    /// Check if this is the optimal layout for C-style iteration (row by row)
    pub fn is_c_optimal(&self) -> bool {
        matches!(self, MemoryLayout::CContiguous | MemoryLayout::Scalar)
    }

    /// Check if this is the optimal layout for Fortran-style iteration (column by column)
    pub fn is_f_optimal(&self) -> bool {
        matches!(self, MemoryLayout::FContiguous | MemoryLayout::Scalar)
    }
}

/// Detect memory layout from shape and strides
///
/// # Arguments
///
/// * `shape` - Array shape (dimensions)
/// * `strides` - Array strides in elements (not bytes)
///
/// # Returns
///
/// The detected memory layout
///
/// # Example
///
/// ```
/// use numrs2::memory_optimize::access_patterns::{detect_layout, MemoryLayout};
///
/// // C-contiguous 3x4 array: strides [4, 1]
/// let layout = detect_layout(&[3, 4], &[4, 1]);
/// assert_eq!(layout, MemoryLayout::CContiguous);
///
/// // F-contiguous 3x4 array: strides [1, 3]
/// let layout = detect_layout(&[3, 4], &[1, 3]);
/// assert_eq!(layout, MemoryLayout::FContiguous);
/// ```
pub fn detect_layout(shape: &[usize], strides: &[usize]) -> MemoryLayout {
    if shape.is_empty() || shape.iter().product::<usize>() <= 1 {
        return MemoryLayout::Scalar;
    }

    // Check C-contiguous: strides should be [prod(shape[1:]), prod(shape[2:]), ..., 1]
    let mut expected_c_stride = 1;
    let mut is_c_contiguous = true;
    for i in (0..shape.len()).rev() {
        if strides[i] != expected_c_stride {
            is_c_contiguous = false;
            break;
        }
        expected_c_stride *= shape[i];
    }

    if is_c_contiguous {
        return MemoryLayout::CContiguous;
    }

    // Check F-contiguous: strides should be [1, shape[0], shape[0]*shape[1], ...]
    let mut expected_f_stride = 1;
    let mut is_f_contiguous = true;
    for i in 0..shape.len() {
        if strides[i] != expected_f_stride {
            is_f_contiguous = false;
            break;
        }
        expected_f_stride *= shape[i];
    }

    if is_f_contiguous {
        return MemoryLayout::FContiguous;
    }

    MemoryLayout::Strided
}

/// Cache level configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLevel {
    /// L1 cache (smallest, fastest)
    L1,
    /// L2 cache (medium)
    L2,
    /// L3 cache (largest, slowest)
    L3,
}

/// Cache configuration parameters
///
/// Typical cache sizes:
/// - L1: 32KB - 64KB per core
/// - L2: 256KB - 512KB per core
/// - L3: 4MB - 32MB shared
#[derive(Debug, Clone, Copy)]
pub struct CacheConfig {
    /// Cache level
    pub level: CacheLevel,
    /// Cache size in bytes
    pub size_bytes: usize,
    /// Cache line size in bytes (typically 64)
    pub line_size: usize,
    /// Associativity (number of ways)
    pub associativity: usize,
}

impl CacheConfig {
    /// Create a new cache configuration
    pub fn new(
        level: CacheLevel,
        size_bytes: usize,
        line_size: usize,
        associativity: usize,
    ) -> Self {
        Self {
            level,
            size_bytes,
            line_size,
            associativity,
        }
    }

    /// Default L1 cache configuration (32KB, 64B lines, 8-way)
    pub fn l1_default() -> Self {
        Self::new(CacheLevel::L1, 32 * 1024, 64, 8)
    }

    /// Default L2 cache configuration (256KB, 64B lines, 8-way)
    pub fn l2_default() -> Self {
        Self::new(CacheLevel::L2, 256 * 1024, 64, 8)
    }

    /// Default L3 cache configuration (8MB, 64B lines, 16-way)
    pub fn l3_default() -> Self {
        Self::new(CacheLevel::L3, 8 * 1024 * 1024, 64, 16)
    }

    /// Calculate the number of elements that fit in one cache line
    pub fn elements_per_line<T>(&self) -> usize {
        let elem_size = std::mem::size_of::<T>();
        self.line_size.checked_div(elem_size).unwrap_or(0)
    }

    /// Calculate the optimal block size (number of elements) for this cache level
    ///
    /// Uses approximately 75% of cache to leave room for other data
    pub fn elements_per_block<T>(&self) -> usize {
        let elem_size = std::mem::size_of::<T>();
        if elem_size == 0 {
            return 0;
        }

        // Use 75% of cache for data
        let usable_bytes = (self.size_bytes * 3) / 4;
        usable_bytes / elem_size
    }

    /// Calculate optimal 2D tile size for matrix operations
    ///
    /// Returns (rows, cols) for a square-ish tile that fits in cache
    pub fn tile_size_2d<T>(&self) -> (usize, usize) {
        let block_elements = self.elements_per_block::<T>();
        let tile_dim = (block_elements as f64).sqrt() as usize;
        let tile_dim = tile_dim.max(1);

        // Round to cache line boundary for better efficiency
        let elements_per_line = self.elements_per_line::<T>().max(1);
        let aligned_dim = tile_dim.div_ceil(elements_per_line) * elements_per_line;

        (aligned_dim, aligned_dim)
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self::l2_default()
    }
}

/// A block/range for blocked iteration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    /// Start index (inclusive)
    pub start: usize,
    /// End index (exclusive)
    pub end: usize,
}

impl Block {
    /// Create a new block
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Get the number of elements in this block
    pub fn len(&self) -> usize {
        self.end.saturating_sub(self.start)
    }

    /// Check if the block is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over indices in this block
    pub fn iter(&self) -> std::ops::Range<usize> {
        self.start..self.end
    }
}

/// Iterator that yields cache-friendly blocks of indices
///
/// # Example
///
/// ```
/// use numrs2::memory_optimize::access_patterns::BlockedIterator;
///
/// let iter = BlockedIterator::new(1000, 256);
/// let blocks: Vec<_> = iter.collect();
///
/// // Should have 4 blocks: [0..256), [256..512), [512..768), [768..1000)
/// assert_eq!(blocks.len(), 4);
/// assert_eq!(blocks[0].start, 0);
/// assert_eq!(blocks[0].end, 256);
/// assert_eq!(blocks[3].end, 1000);
/// ```
pub struct BlockedIterator {
    total: usize,
    block_size: usize,
    current: usize,
}

impl BlockedIterator {
    /// Create a new blocked iterator
    ///
    /// # Arguments
    ///
    /// * `total` - Total number of elements
    /// * `block_size` - Size of each block
    pub fn new(total: usize, block_size: usize) -> Self {
        Self {
            total,
            block_size: block_size.max(1),
            current: 0,
        }
    }

    /// Create a blocked iterator optimized for a specific type and cache level
    pub fn for_type<T>(total: usize, cache: CacheConfig) -> Self {
        Self::new(total, cache.elements_per_block::<T>())
    }
}

impl Iterator for BlockedIterator {
    type Item = Block;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.total {
            return None;
        }

        let start = self.current;
        let end = (start + self.block_size).min(self.total);
        self.current = end;

        Some(Block::new(start, end))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total.saturating_sub(self.current);
        let count = remaining.div_ceil(self.block_size);
        (count, Some(count))
    }
}

impl ExactSizeIterator for BlockedIterator {}

/// 2D tile for tiled/blocked matrix operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile2D {
    /// Row start (inclusive)
    pub row_start: usize,
    /// Row end (exclusive)
    pub row_end: usize,
    /// Column start (inclusive)
    pub col_start: usize,
    /// Column end (exclusive)
    pub col_end: usize,
}

impl Tile2D {
    /// Create a new 2D tile
    pub fn new(row_start: usize, row_end: usize, col_start: usize, col_end: usize) -> Self {
        Self {
            row_start,
            row_end,
            col_start,
            col_end,
        }
    }

    /// Get the number of rows in this tile
    pub fn rows(&self) -> usize {
        self.row_end.saturating_sub(self.row_start)
    }

    /// Get the number of columns in this tile
    pub fn cols(&self) -> usize {
        self.col_end.saturating_sub(self.col_start)
    }

    /// Get the total number of elements in this tile
    pub fn len(&self) -> usize {
        self.rows() * self.cols()
    }

    /// Check if the tile is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Iterator that yields cache-friendly 2D tiles for matrix operations
///
/// # Example
///
/// ```
/// use numrs2::memory_optimize::access_patterns::TiledIterator2D;
///
/// // Create iterator for 100x100 matrix with 32x32 tiles
/// let iter = TiledIterator2D::new(100, 100, 32, 32);
/// let tiles: Vec<_> = iter.collect();
///
/// // Should have ceil(100/32) * ceil(100/32) = 4 * 4 = 16 tiles
/// assert_eq!(tiles.len(), 16);
/// ```
pub struct TiledIterator2D {
    total_rows: usize,
    total_cols: usize,
    tile_rows: usize,
    tile_cols: usize,
    current_row: usize,
    current_col: usize,
}

impl TiledIterator2D {
    /// Create a new 2D tiled iterator
    ///
    /// # Arguments
    ///
    /// * `total_rows` - Total number of rows
    /// * `total_cols` - Total number of columns
    /// * `tile_rows` - Rows per tile
    /// * `tile_cols` - Columns per tile
    pub fn new(total_rows: usize, total_cols: usize, tile_rows: usize, tile_cols: usize) -> Self {
        Self {
            total_rows,
            total_cols,
            tile_rows: tile_rows.max(1),
            tile_cols: tile_cols.max(1),
            current_row: 0,
            current_col: 0,
        }
    }

    /// Create a tiled iterator optimized for a specific type and cache level
    pub fn for_type<T>(total_rows: usize, total_cols: usize, cache: CacheConfig) -> Self {
        let (tile_rows, tile_cols) = cache.tile_size_2d::<T>();
        Self::new(total_rows, total_cols, tile_rows, tile_cols)
    }
}

impl Iterator for TiledIterator2D {
    type Item = Tile2D;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_row >= self.total_rows {
            return None;
        }

        let row_start = self.current_row;
        let row_end = (row_start + self.tile_rows).min(self.total_rows);
        let col_start = self.current_col;
        let col_end = (col_start + self.tile_cols).min(self.total_cols);

        // Advance to next tile
        self.current_col += self.tile_cols;
        if self.current_col >= self.total_cols {
            self.current_col = 0;
            self.current_row += self.tile_rows;
        }

        Some(Tile2D::new(row_start, row_end, col_start, col_end))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let row_tiles = self.total_rows.div_ceil(self.tile_rows);
        let col_tiles = self.total_cols.div_ceil(self.tile_cols);
        let total_tiles = row_tiles * col_tiles;

        let current_row_tile = self.current_row / self.tile_rows;
        let current_col_tile = self.current_col / self.tile_cols;
        let current_tile = current_row_tile * col_tiles + current_col_tile;

        let remaining = total_tiles.saturating_sub(current_tile);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for TiledIterator2D {}

/// Memory access pattern hints for optimizers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Sequential forward access
    Sequential,
    /// Sequential backward access
    Reverse,
    /// Random/unpredictable access
    Random,
    /// Strided access (e.g., every Nth element)
    Strided(usize),
    /// Blocked/tiled access
    Blocked,
}

impl AccessPattern {
    /// Get prefetch distance hint for this access pattern
    ///
    /// Returns the number of cache lines to prefetch ahead
    pub fn prefetch_distance(&self) -> usize {
        match self {
            AccessPattern::Sequential => 4,
            AccessPattern::Reverse => 2,
            AccessPattern::Random => 0,
            AccessPattern::Strided(stride) => {
                if *stride <= 8 {
                    2
                } else {
                    1
                }
            }
            AccessPattern::Blocked => 2,
        }
    }

    /// Check if this access pattern benefits from software prefetching
    pub fn benefits_from_prefetch(&self) -> bool {
        !matches!(self, AccessPattern::Random)
    }
}

/// Memory optimization hints for an array operation
#[derive(Debug, Clone)]
pub struct OptimizationHints {
    /// Detected memory layout
    pub layout: MemoryLayout,
    /// Detected access pattern
    pub access_pattern: AccessPattern,
    /// Recommended block size (number of elements)
    pub block_size: usize,
    /// Recommended tile size for 2D operations
    pub tile_size: Option<(usize, usize)>,
    /// Whether to use parallel processing
    pub use_parallel: bool,
    /// Estimated cache efficiency (0.0 to 1.0)
    pub cache_efficiency: f64,
}

impl OptimizationHints {
    /// Create optimization hints for a given array shape and operation
    ///
    /// # Arguments
    ///
    /// * `shape` - Array shape
    /// * `strides` - Array strides
    pub fn analyze<T>(shape: &[usize], strides: &[usize]) -> Self {
        let layout = detect_layout(shape, strides);
        let total_elements: usize = shape.iter().product();
        let total_bytes = total_elements * std::mem::size_of::<T>();

        // Determine cache level to target
        let cache = if total_bytes <= 32 * 1024 {
            CacheConfig::l1_default()
        } else if total_bytes <= 256 * 1024 {
            CacheConfig::l2_default()
        } else {
            CacheConfig::l3_default()
        };

        let block_size = cache.elements_per_block::<T>();

        let tile_size = if shape.len() >= 2 {
            Some(cache.tile_size_2d::<T>())
        } else {
            None
        };

        // Estimate cache efficiency based on layout
        let cache_efficiency = match layout {
            MemoryLayout::CContiguous | MemoryLayout::FContiguous => 0.95,
            MemoryLayout::Strided => 0.5,
            MemoryLayout::Scalar => 1.0,
        };

        // Determine access pattern from layout and shape
        let access_pattern = if layout.is_contiguous() {
            AccessPattern::Sequential
        } else if !strides.is_empty() {
            AccessPattern::Strided(strides.iter().min().copied().unwrap_or(1))
        } else {
            AccessPattern::Random
        };

        // Use parallel processing for large arrays
        let use_parallel = total_elements > 10_000;

        Self {
            layout,
            access_pattern,
            block_size,
            tile_size,
            use_parallel,
            cache_efficiency,
        }
    }

    /// Create default optimization hints
    pub fn default_for<T>(total_elements: usize) -> Self {
        let cache = CacheConfig::l2_default();
        Self {
            layout: MemoryLayout::CContiguous,
            access_pattern: AccessPattern::Sequential,
            block_size: cache.elements_per_block::<T>(),
            tile_size: Some(cache.tile_size_2d::<T>()),
            use_parallel: total_elements > 10_000,
            cache_efficiency: 0.95,
        }
    }
}

impl Default for OptimizationHints {
    fn default() -> Self {
        Self {
            layout: MemoryLayout::CContiguous,
            access_pattern: AccessPattern::Sequential,
            block_size: 4096,
            tile_size: Some((64, 64)),
            use_parallel: false,
            cache_efficiency: 0.95,
        }
    }
}

/// Stride optimizer for non-contiguous array access
///
/// Provides utilities for optimizing strided memory access patterns.
pub struct StrideOptimizer {
    /// Original strides
    strides: Vec<usize>,
    /// Shape of the array
    shape: Vec<usize>,
    /// Optimal iteration order (dimension indices)
    iteration_order: Vec<usize>,
}

impl StrideOptimizer {
    /// Create a new stride optimizer
    pub fn new(shape: &[usize], strides: &[usize]) -> Self {
        let mut iteration_order: Vec<usize> = (0..shape.len()).collect();

        // Sort dimensions by stride (ascending) for cache-optimal iteration
        // Iterating dimensions with smaller strides first is more cache-friendly
        iteration_order.sort_by_key(|&i| strides.get(i).copied().unwrap_or(0));

        Self {
            strides: strides.to_vec(),
            shape: shape.to_vec(),
            iteration_order,
        }
    }

    /// Get the optimal iteration order for this array
    ///
    /// Returns dimension indices in the order they should be iterated
    pub fn optimal_iteration_order(&self) -> &[usize] {
        &self.iteration_order
    }

    /// Check if the array should be copied before processing
    ///
    /// Returns true if copying would be more efficient than strided access
    pub fn should_copy(&self) -> bool {
        let layout = detect_layout(&self.shape, &self.strides);
        if layout.is_contiguous() {
            return false;
        }

        // Heuristic: copy if the smallest stride > 4 (poor locality)
        let min_stride = self.strides.iter().min().copied().unwrap_or(1);
        min_stride > 4
    }

    /// Calculate the memory bandwidth efficiency
    ///
    /// Returns a value between 0 and 1, where 1 is fully contiguous
    pub fn bandwidth_efficiency(&self) -> f64 {
        if self.strides.is_empty() {
            return 1.0;
        }

        // Ideal case: smallest stride is 1 (contiguous in innermost dim)
        let min_stride = self.strides.iter().min().copied().unwrap_or(1) as f64;
        (1.0 / min_stride).min(1.0)
    }
}

/// Cache-aware copy function that processes data in optimal chunks
///
/// # Example
///
/// ```
/// use numrs2::memory_optimize::access_patterns::cache_aware_copy;
///
/// let src = vec![1.0f64; 10000];
/// let mut dst = vec![0.0f64; 10000];
///
/// cache_aware_copy(&src, &mut dst);
/// assert_eq!(dst, src);
/// ```
pub fn cache_aware_copy<T: Copy>(src: &[T], dst: &mut [T]) {
    let len = src.len().min(dst.len());
    if len == 0 {
        return;
    }

    // Use L1 cache block size
    let cache = CacheConfig::l1_default();
    let block_size = cache.elements_per_block::<T>();

    for block in BlockedIterator::new(len, block_size) {
        dst[block.start..block.end].copy_from_slice(&src[block.start..block.end]);
    }
}

/// Cache-aware transformation that processes data in optimal chunks
///
/// # Example
///
/// ```
/// use numrs2::memory_optimize::access_patterns::cache_aware_transform;
///
/// let src = vec![1.0f64, 2.0, 3.0, 4.0];
/// let mut dst = vec![0.0f64; 4];
///
/// cache_aware_transform(&src, &mut dst, |x| x * 2.0);
/// assert_eq!(dst, vec![2.0, 4.0, 6.0, 8.0]);
/// ```
pub fn cache_aware_transform<T, U, F>(src: &[T], dst: &mut [U], f: F)
where
    T: Copy,
    F: Fn(T) -> U,
{
    let len = src.len().min(dst.len());
    if len == 0 {
        return;
    }

    // Use L1 cache block size (based on larger type)
    let cache = CacheConfig::l1_default();
    let elem_size = std::mem::size_of::<T>().max(std::mem::size_of::<U>());
    let block_size = (cache.size_bytes * 3 / 4)
        .checked_div(elem_size)
        .unwrap_or(len);

    for block in BlockedIterator::new(len, block_size) {
        for i in block.start..block.end {
            dst[i] = f(src[i]);
        }
    }
}

/// Cache-aware binary operation that processes data in optimal chunks
///
/// # Example
///
/// ```
/// use numrs2::memory_optimize::access_patterns::cache_aware_binary_op;
///
/// let a = vec![1.0f64, 2.0, 3.0, 4.0];
/// let b = vec![10.0f64, 20.0, 30.0, 40.0];
/// let mut result = vec![0.0f64; 4];
///
/// cache_aware_binary_op(&a, &b, &mut result, |x, y| x + y);
/// assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
/// ```
pub fn cache_aware_binary_op<T, U, V, F>(a: &[T], b: &[U], result: &mut [V], f: F)
where
    T: Copy,
    U: Copy,
    F: Fn(T, U) -> V,
{
    let len = a.len().min(b.len()).min(result.len());
    if len == 0 {
        return;
    }

    // Use L1 cache block size
    let cache = CacheConfig::l1_default();
    let elem_size = std::mem::size_of::<T>()
        .max(std::mem::size_of::<U>())
        .max(std::mem::size_of::<V>());
    let block_size = if elem_size > 0 {
        (cache.size_bytes * 3 / 4) / (elem_size * 3) // 3 arrays
    } else {
        len
    };

    for block in BlockedIterator::new(len, block_size) {
        for i in block.start..block.end {
            result[i] = f(a[i], b[i]);
        }
    }
}

/// Memory access statistics for profiling
#[derive(Debug, Clone, Default)]
pub struct AccessStats {
    /// Total number of memory accesses
    pub total_accesses: u64,
    /// Number of sequential accesses
    pub sequential_accesses: u64,
    /// Number of strided accesses
    pub strided_accesses: u64,
    /// Number of random accesses
    pub random_accesses: u64,
    /// Estimated cache miss rate
    pub estimated_miss_rate: f64,
}

impl AccessStats {
    /// Create new empty statistics
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a sequential access
    pub fn record_sequential(&mut self) {
        self.total_accesses += 1;
        self.sequential_accesses += 1;
    }

    /// Record a strided access
    pub fn record_strided(&mut self) {
        self.total_accesses += 1;
        self.strided_accesses += 1;
    }

    /// Record a random access
    pub fn record_random(&mut self) {
        self.total_accesses += 1;
        self.random_accesses += 1;
    }

    /// Calculate estimated cache efficiency
    pub fn cache_efficiency(&self) -> f64 {
        if self.total_accesses == 0 {
            return 1.0;
        }

        let seq_weight = 1.0;
        let strided_weight = 0.5;
        let random_weight = 0.1;

        let weighted_sum = (self.sequential_accesses as f64 * seq_weight)
            + (self.strided_accesses as f64 * strided_weight)
            + (self.random_accesses as f64 * random_weight);

        weighted_sum / self.total_accesses as f64
    }

    /// Update estimated miss rate based on recorded accesses
    pub fn update_miss_rate(&mut self) {
        // Simple model: random accesses have ~90% miss rate,
        // strided have ~30%, sequential have ~5%
        if self.total_accesses == 0 {
            self.estimated_miss_rate = 0.0;
            return;
        }

        let seq_miss = 0.05;
        let strided_miss = 0.30;
        let random_miss = 0.90;

        self.estimated_miss_rate = ((self.sequential_accesses as f64 * seq_miss)
            + (self.strided_accesses as f64 * strided_miss)
            + (self.random_accesses as f64 * random_miss))
            / self.total_accesses as f64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_layout_c_contiguous() {
        // C-contiguous 3x4 array
        let layout = detect_layout(&[3, 4], &[4, 1]);
        assert_eq!(layout, MemoryLayout::CContiguous);
    }

    #[test]
    fn test_detect_layout_f_contiguous() {
        // F-contiguous 3x4 array
        let layout = detect_layout(&[3, 4], &[1, 3]);
        assert_eq!(layout, MemoryLayout::FContiguous);
    }

    #[test]
    fn test_detect_layout_strided() {
        // Non-contiguous (e.g., every other element)
        let layout = detect_layout(&[3, 4], &[8, 2]);
        assert_eq!(layout, MemoryLayout::Strided);
    }

    #[test]
    fn test_detect_layout_scalar() {
        let layout = detect_layout(&[], &[]);
        assert_eq!(layout, MemoryLayout::Scalar);

        let layout = detect_layout(&[1], &[1]);
        assert_eq!(layout, MemoryLayout::Scalar);
    }

    #[test]
    fn test_cache_config_elements() {
        let cache = CacheConfig::l1_default();

        // f64 is 8 bytes, cache line is 64 bytes
        assert_eq!(cache.elements_per_line::<f64>(), 8);

        // f32 is 4 bytes
        assert_eq!(cache.elements_per_line::<f32>(), 16);
    }

    #[test]
    fn test_blocked_iterator() {
        let iter = BlockedIterator::new(100, 30);
        let blocks: Vec<_> = iter.collect();

        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[0], Block::new(0, 30));
        assert_eq!(blocks[1], Block::new(30, 60));
        assert_eq!(blocks[2], Block::new(60, 90));
        assert_eq!(blocks[3], Block::new(90, 100));
    }

    #[test]
    fn test_blocked_iterator_exact_division() {
        let iter = BlockedIterator::new(100, 25);
        let blocks: Vec<_> = iter.collect();

        assert_eq!(blocks.len(), 4);
        assert_eq!(blocks[3], Block::new(75, 100));
    }

    #[test]
    fn test_tiled_iterator_2d() {
        let iter = TiledIterator2D::new(10, 10, 4, 4);
        let tiles: Vec<_> = iter.collect();

        // 3x3 = 9 tiles (10/4 = 2.5 -> 3 tiles per dimension)
        assert_eq!(tiles.len(), 9);

        // First tile
        assert_eq!(tiles[0].row_start, 0);
        assert_eq!(tiles[0].row_end, 4);
        assert_eq!(tiles[0].col_start, 0);
        assert_eq!(tiles[0].col_end, 4);

        // Last tile
        let last = tiles
            .last()
            .expect("tiles should have at least one element");
        assert_eq!(last.row_start, 8);
        assert_eq!(last.row_end, 10);
        assert_eq!(last.col_start, 8);
        assert_eq!(last.col_end, 10);
    }

    #[test]
    fn test_block_len() {
        let block = Block::new(10, 25);
        assert_eq!(block.len(), 15);
        assert!(!block.is_empty());

        let empty = Block::new(10, 10);
        assert_eq!(empty.len(), 0);
        assert!(empty.is_empty());
    }

    #[test]
    fn test_tile_2d_len() {
        let tile = Tile2D::new(0, 4, 0, 5);
        assert_eq!(tile.rows(), 4);
        assert_eq!(tile.cols(), 5);
        assert_eq!(tile.len(), 20);
    }

    #[test]
    fn test_optimization_hints() {
        // Contiguous array
        let hints = OptimizationHints::analyze::<f64>(&[100, 100], &[100, 1]);
        assert_eq!(hints.layout, MemoryLayout::CContiguous);
        assert_eq!(hints.access_pattern, AccessPattern::Sequential);
        assert!(hints.cache_efficiency > 0.9);
    }

    #[test]
    fn test_stride_optimizer() {
        // C-contiguous array: strides [4, 1] for 3x4
        let optimizer = StrideOptimizer::new(&[3, 4], &[4, 1]);

        // Innermost dimension (stride 1) should come first
        let order = optimizer.optimal_iteration_order();
        assert_eq!(order[0], 1); // col first (stride 1)
        assert_eq!(order[1], 0); // row second (stride 4)

        assert!(!optimizer.should_copy());
        assert!(optimizer.bandwidth_efficiency() > 0.9);
    }

    #[test]
    fn test_cache_aware_copy() {
        let src = vec![1.0f64; 1000];
        let mut dst = vec![0.0f64; 1000];

        cache_aware_copy(&src, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_cache_aware_transform() {
        let src = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut dst = vec![0.0; 5];

        cache_aware_transform(&src, &mut dst, |x| x * x);
        assert_eq!(dst, vec![1.0, 4.0, 9.0, 16.0, 25.0]);
    }

    #[test]
    fn test_cache_aware_binary_op() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let mut result = vec![0.0; 4];

        cache_aware_binary_op(&a, &b, &mut result, |x, y| x + y);
        assert_eq!(result, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_access_stats() {
        let mut stats = AccessStats::new();

        stats.record_sequential();
        stats.record_sequential();
        stats.record_strided();
        stats.record_random();

        assert_eq!(stats.total_accesses, 4);
        assert_eq!(stats.sequential_accesses, 2);
        assert_eq!(stats.strided_accesses, 1);
        assert_eq!(stats.random_accesses, 1);

        // Sequential is most common, so efficiency should be > 0.5
        assert!(stats.cache_efficiency() > 0.5);

        stats.update_miss_rate();
        assert!(stats.estimated_miss_rate > 0.0);
        assert!(stats.estimated_miss_rate < 1.0);
    }

    #[test]
    fn test_access_pattern_prefetch() {
        assert_eq!(AccessPattern::Sequential.prefetch_distance(), 4);
        assert_eq!(AccessPattern::Random.prefetch_distance(), 0);
        assert!(AccessPattern::Sequential.benefits_from_prefetch());
        assert!(!AccessPattern::Random.benefits_from_prefetch());
    }
}
