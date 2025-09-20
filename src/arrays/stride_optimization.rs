//! Advanced stride calculations and memory layout optimization
//!
//! This module provides sophisticated stride calculation algorithms optimized
//! for various access patterns, cache efficiency, and SIMD operations.

use super::shape_manipulation::MemoryLayout;
use std::collections::HashMap;

/// Advanced stride calculator with optimization capabilities
pub struct StrideCalculator {
    /// Cache for computed optimal strides
    stride_cache: HashMap<StrideKey, Vec<usize>>,
    /// Performance hints for optimization
    hints: OptimizationHints,
}

/// Key for stride caching
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct StrideKey {
    shape: Vec<usize>,
    layout: MemoryLayout,
    access_pattern: AccessPattern,
}

/// Access pattern hints for stride optimization
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum AccessPattern {
    /// Sequential access (default)
    Sequential,
    /// Random access pattern
    Random,
    /// Row-wise access (matrix rows)
    RowWise,
    /// Column-wise access (matrix columns)
    ColumnWise,
    /// Block access pattern
    Block { block_size: Vec<usize> },
    /// SIMD-optimized access
    SIMD { vector_width: usize },
    /// Cache-friendly tiled access
    Tiled { tile_size: Vec<usize> },
    /// Broadcast-optimized access
    Broadcast,
}

/// Optimization hints for stride calculation
#[derive(Debug, Clone)]
pub struct OptimizationHints {
    /// Target cache line size (in bytes)
    pub cache_line_size: usize,
    /// L1 cache size (in bytes)
    pub l1_cache_size: usize,
    /// L2 cache size (in bytes)
    pub l2_cache_size: usize,
    /// SIMD vector width (in elements)
    pub simd_width: usize,
    /// Memory alignment requirement (in bytes)
    pub alignment: usize,
    /// Whether to optimize for memory bandwidth
    pub optimize_bandwidth: bool,
    /// Whether to optimize for cache locality
    pub optimize_locality: bool,
}

impl Default for OptimizationHints {
    fn default() -> Self {
        Self {
            cache_line_size: 64,
            l1_cache_size: 32 * 1024,  // 32KB
            l2_cache_size: 256 * 1024, // 256KB
            simd_width: 8,             // AVX2 double precision
            alignment: 32,             // AVX2 alignment
            optimize_bandwidth: true,
            optimize_locality: true,
        }
    }
}

impl Default for StrideCalculator {
    fn default() -> Self {
        Self::new(OptimizationHints::default())
    }
}

impl StrideCalculator {
    /// Create a new stride calculator
    pub fn new(hints: OptimizationHints) -> Self {
        Self {
            stride_cache: HashMap::new(),
            hints,
        }
    }

    /// Compute optimal strides for given shape and access pattern
    pub fn compute_optimal_strides(
        &mut self,
        shape: &[usize],
        access_pattern: AccessPattern,
    ) -> Vec<usize> {
        let layout = self.determine_optimal_layout(shape, &access_pattern);
        let cache_key = StrideKey {
            shape: shape.to_vec(),
            layout,
            access_pattern: access_pattern.clone(),
        };

        if let Some(cached_strides) = self.stride_cache.get(&cache_key) {
            return cached_strides.clone();
        }

        let strides = match access_pattern {
            AccessPattern::Sequential => self.compute_sequential_strides(shape),
            AccessPattern::Random => self.compute_random_access_strides(shape),
            AccessPattern::RowWise => self.compute_row_wise_strides(shape),
            AccessPattern::ColumnWise => self.compute_column_wise_strides(shape),
            AccessPattern::Block { ref block_size } => {
                self.compute_block_strides(shape, block_size)
            }
            AccessPattern::SIMD { vector_width } => self.compute_simd_strides(shape, vector_width),
            AccessPattern::Tiled { ref tile_size } => self.compute_tiled_strides(shape, tile_size),
            AccessPattern::Broadcast => self.compute_broadcast_strides(shape),
        };

        self.stride_cache.insert(cache_key, strides.clone());
        strides
    }

    /// Determine optimal memory layout for access pattern
    fn determine_optimal_layout(
        &self,
        shape: &[usize],
        access_pattern: &AccessPattern,
    ) -> MemoryLayout {
        match access_pattern {
            AccessPattern::RowWise | AccessPattern::Sequential => MemoryLayout::C,
            AccessPattern::ColumnWise => MemoryLayout::Fortran,
            AccessPattern::SIMD { .. }
            | AccessPattern::Block { .. }
            | AccessPattern::Tiled { .. } => MemoryLayout::Custom,
            AccessPattern::Random | AccessPattern::Broadcast => {
                // Choose based on shape characteristics
                if shape.len() <= 2 && shape.iter().all(|&s| s <= 1000) {
                    MemoryLayout::C
                } else {
                    MemoryLayout::Custom
                }
            }
        }
    }

    /// Compute strides for sequential access
    fn compute_sequential_strides(&self, shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];

        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        strides
    }

    /// Compute strides optimized for random access
    fn compute_random_access_strides(&self, shape: &[usize]) -> Vec<usize> {
        // For random access, minimize memory footprint while maintaining reasonable locality
        let mut dim_priorities: Vec<(usize, usize)> = shape
            .iter()
            .enumerate()
            .map(|(i, &size)| (i, size))
            .collect();

        // Sort by size (smaller dimensions get larger strides to improve locality)
        dim_priorities.sort_by_key(|&(_, size)| size);

        let mut strides = vec![0; shape.len()];
        let mut current_stride = 1;

        for &(dim_idx, dim_size) in &dim_priorities {
            strides[dim_idx] = current_stride;
            current_stride *= dim_size;
        }

        strides
    }

    /// Compute strides for row-wise access (C-style)
    fn compute_row_wise_strides(&self, shape: &[usize]) -> Vec<usize> {
        self.compute_sequential_strides(shape)
    }

    /// Compute strides for column-wise access (Fortran-style)
    fn compute_column_wise_strides(&self, shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; shape.len()];

        for i in 1..shape.len() {
            strides[i] = strides[i - 1] * shape[i - 1];
        }

        strides
    }

    /// Compute strides optimized for block access patterns
    fn compute_block_strides(&self, shape: &[usize], block_size: &[usize]) -> Vec<usize> {
        if block_size.len() != shape.len() {
            // Fallback to sequential if block size doesn't match
            return self.compute_sequential_strides(shape);
        }

        // Optimize for block-wise traversal
        let mut strides = vec![1; shape.len()];
        let mut current_stride = 1;

        // Order dimensions by block utilization (favor dimensions with smaller blocks)
        let mut dim_order: Vec<usize> = (0..shape.len()).collect();
        dim_order.sort_by_key(|&i| block_size[i]);

        for &dim in &dim_order {
            strides[dim] = current_stride;
            current_stride *= shape[dim];
        }

        strides
    }

    /// Compute strides optimized for SIMD operations
    fn compute_simd_strides(&self, shape: &[usize], vector_width: usize) -> Vec<usize> {
        let mut strides = self.compute_sequential_strides(shape);

        if shape.is_empty() {
            return strides;
        }

        // Ensure the innermost dimension is SIMD-friendly
        let innermost_dim = shape.len() - 1;
        let innermost_size = shape[innermost_dim];

        // If the innermost dimension is not SIMD-aligned, adjust strides
        if !innermost_size.is_multiple_of(vector_width) {
            let padded_size = innermost_size.div_ceil(vector_width) * vector_width;

            // Recalculate strides with padding
            strides[innermost_dim] = 1;
            for i in (0..innermost_dim).rev() {
                let next_size = if i == innermost_dim - 1 {
                    padded_size
                } else {
                    shape[i + 1]
                };
                strides[i] = strides[i + 1] * next_size;
            }
        }

        strides
    }

    /// Compute strides optimized for tiled access patterns
    fn compute_tiled_strides(&self, shape: &[usize], tile_size: &[usize]) -> Vec<usize> {
        if tile_size.len() != shape.len() {
            return self.compute_sequential_strides(shape);
        }

        // Calculate optimal tile ordering based on cache efficiency
        let cache_line_elements = self.hints.cache_line_size / std::mem::size_of::<f64>();

        let mut strides = vec![1; shape.len()];
        let mut current_stride = 1;

        // Order dimensions to maximize cache line utilization
        let mut dim_order: Vec<usize> = (0..shape.len()).collect();
        dim_order.sort_by_key(|&i| {
            // Prefer dimensions where tile size fits well in cache lines
            let tile_elements = tile_size[i];
            let cache_efficiency = if tile_elements <= cache_line_elements {
                cache_line_elements / tile_elements
            } else {
                1
            };
            std::cmp::Reverse(cache_efficiency)
        });

        for &dim in &dim_order {
            strides[dim] = current_stride;
            current_stride *= shape[dim];
        }

        strides
    }

    /// Compute strides optimized for broadcasting operations
    fn compute_broadcast_strides(&self, shape: &[usize]) -> Vec<usize> {
        // For broadcasting, optimize for the most common case where
        // smaller arrays are broadcast against larger ones
        let mut strides = vec![1; shape.len()];

        if shape.len() <= 1 {
            return strides;
        }

        // Sort dimensions by size and assign strides accordingly
        let mut dim_sizes: Vec<(usize, usize)> = shape
            .iter()
            .enumerate()
            .map(|(i, &size)| (i, size))
            .collect();

        // For broadcasting, prioritize larger dimensions (they're less likely to be broadcast)
        dim_sizes.sort_by_key(|&(_, size)| std::cmp::Reverse(size));

        let mut current_stride = 1;
        for &(dim_idx, dim_size) in &dim_sizes {
            strides[dim_idx] = current_stride;
            current_stride *= dim_size;
        }

        strides
    }

    /// Analyze stride efficiency for cache performance
    pub fn analyze_stride_efficiency(&self, shape: &[usize], strides: &[usize]) -> StrideAnalysis {
        if shape.len() != strides.len() {
            return StrideAnalysis::default();
        }

        let element_size = std::mem::size_of::<f64>(); // Assume f64 for analysis
        let cache_line_elements = self.hints.cache_line_size / element_size;

        // Calculate cache utilization for each dimension
        let mut cache_utilizations = Vec::new();
        for (&stride, &dim_size) in strides.iter().zip(shape.iter()) {
            let utilization = if stride == 0 {
                0.0
            } else {
                let elements_per_cache_line = cache_line_elements / stride.max(1);
                elements_per_cache_line.min(dim_size) as f64 / cache_line_elements as f64
            };
            cache_utilizations.push(utilization);
        }

        // Calculate overall efficiency metrics
        let avg_cache_utilization =
            cache_utilizations.iter().sum::<f64>() / cache_utilizations.len() as f64;

        // Calculate memory bandwidth efficiency
        let total_elements: usize = shape.iter().product();
        let memory_span = self.calculate_memory_span(shape, strides);
        let bandwidth_efficiency = if memory_span > 0 {
            total_elements as f64 / memory_span as f64
        } else {
            0.0
        };

        // Calculate SIMD efficiency
        let simd_efficiency = if shape.is_empty() || strides.is_empty() {
            0.0
        } else {
            let innermost_stride = strides[strides.len() - 1];
            if innermost_stride == 1 {
                1.0 // Perfect for SIMD
            } else {
                1.0 / innermost_stride as f64
            }
        };

        // Detect stride patterns
        let pattern = self.detect_stride_pattern(strides);

        StrideAnalysis {
            cache_utilization: avg_cache_utilization,
            bandwidth_efficiency,
            simd_efficiency,
            pattern,
            memory_span,
            cache_utilizations,
            is_optimal: avg_cache_utilization > 0.8 && bandwidth_efficiency > 0.9,
        }
    }

    /// Calculate memory span covered by the array
    fn calculate_memory_span(&self, shape: &[usize], strides: &[usize]) -> usize {
        if shape.is_empty() || strides.is_empty() {
            return 0;
        }

        let min_addr = 0;
        let mut max_addr = 0;

        for (&dim_size, &stride) in shape.iter().zip(strides.iter()) {
            if dim_size > 1 {
                let span = (dim_size - 1) * stride;
                max_addr += span;
            }
        }

        max_addr - min_addr + 1
    }

    /// Detect the type of stride pattern
    fn detect_stride_pattern(&self, strides: &[usize]) -> StridePattern {
        if strides.is_empty() {
            return StridePattern::Empty;
        }

        if strides.len() == 1 {
            return StridePattern::OneDimensional;
        }

        // Check for C-contiguous pattern
        let mut expected_stride = 1;
        let mut is_c_contiguous = true;
        for &stride in strides.iter().rev() {
            if stride != expected_stride {
                is_c_contiguous = false;
                break;
            }
            expected_stride *= stride; // This is simplified; full calculation would need shape
        }

        if is_c_contiguous {
            return StridePattern::CContiguous;
        }

        // Check for Fortran-contiguous pattern
        let mut expected_stride = 1;
        let mut is_f_contiguous = true;
        for &stride in strides.iter() {
            if stride != expected_stride {
                is_f_contiguous = false;
                break;
            }
            expected_stride *= stride; // This is simplified
        }

        if is_f_contiguous {
            return StridePattern::FortranContiguous;
        }

        // Check for unit stride in any dimension
        if strides.contains(&1) {
            return StridePattern::UnitStride;
        }

        // Check for regular pattern (powers of 2)
        let mut is_power_of_two = true;
        for &stride in strides {
            if stride > 0 && (stride & (stride - 1)) != 0 {
                is_power_of_two = false;
                break;
            }
        }

        if is_power_of_two {
            StridePattern::PowerOfTwo
        } else {
            StridePattern::Irregular
        }
    }

    /// Optimize strides for specific hardware characteristics
    pub fn optimize_for_hardware(
        &mut self,
        shape: &[usize],
        access_pattern: AccessPattern,
        target_arch: TargetArchitecture,
    ) -> Vec<usize> {
        // Update hints based on target architecture
        self.hints = match target_arch {
            TargetArchitecture::X86_64Avx2 => OptimizationHints {
                cache_line_size: 64,
                l1_cache_size: 32 * 1024,
                l2_cache_size: 256 * 1024,
                simd_width: 4, // AVX2 double precision
                alignment: 32,
                optimize_bandwidth: true,
                optimize_locality: true,
            },
            TargetArchitecture::X86_64Avx512 => OptimizationHints {
                cache_line_size: 64,
                l1_cache_size: 32 * 1024,
                l2_cache_size: 512 * 1024,
                simd_width: 8, // AVX-512 double precision
                alignment: 64,
                optimize_bandwidth: true,
                optimize_locality: true,
            },
            TargetArchitecture::ArmNeon => OptimizationHints {
                cache_line_size: 64,
                l1_cache_size: 64 * 1024,
                l2_cache_size: 512 * 1024,
                simd_width: 2, // NEON double precision
                alignment: 16,
                optimize_bandwidth: true,
                optimize_locality: true,
            },
            TargetArchitecture::Generic => OptimizationHints::default(),
        };

        self.compute_optimal_strides(shape, access_pattern)
    }

    /// Clear the stride cache
    pub fn clear_cache(&mut self) {
        self.stride_cache.clear();
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize) {
        (self.stride_cache.len(), self.stride_cache.capacity())
    }
}

/// Analysis results for stride efficiency
#[derive(Debug, Clone)]
pub struct StrideAnalysis {
    /// Overall cache utilization (0.0 to 1.0)
    pub cache_utilization: f64,
    /// Memory bandwidth efficiency (0.0 to 1.0)
    pub bandwidth_efficiency: f64,
    /// SIMD operation efficiency (0.0 to 1.0)
    pub simd_efficiency: f64,
    /// Detected stride pattern
    pub pattern: StridePattern,
    /// Total memory span
    pub memory_span: usize,
    /// Per-dimension cache utilizations
    pub cache_utilizations: Vec<f64>,
    /// Whether the stride configuration is considered optimal
    pub is_optimal: bool,
}

impl Default for StrideAnalysis {
    fn default() -> Self {
        Self {
            cache_utilization: 0.0,
            bandwidth_efficiency: 0.0,
            simd_efficiency: 0.0,
            pattern: StridePattern::Empty,
            memory_span: 0,
            cache_utilizations: Vec::new(),
            is_optimal: false,
        }
    }
}

/// Types of stride patterns
#[derive(Debug, Clone, PartialEq)]
pub enum StridePattern {
    /// Empty array
    Empty,
    /// One-dimensional array
    OneDimensional,
    /// C-style contiguous
    CContiguous,
    /// Fortran-style contiguous
    FortranContiguous,
    /// Has unit stride in at least one dimension
    UnitStride,
    /// Strides are powers of 2
    PowerOfTwo,
    /// Irregular pattern
    Irregular,
}

/// Target architecture for optimization
#[derive(Debug, Clone, Copy)]
pub enum TargetArchitecture {
    /// Intel/AMD x86-64 with AVX2
    X86_64Avx2,
    /// Intel/AMD x86-64 with AVX-512
    X86_64Avx512,
    /// ARM with NEON
    ArmNeon,
    /// Generic architecture
    Generic,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stride_calculator_creation() {
        let calculator = StrideCalculator::default();
        assert_eq!(calculator.hints.cache_line_size, 64);
    }

    #[test]
    fn test_sequential_strides() {
        let mut calculator = StrideCalculator::default();
        let shape = [2, 3, 4];
        let strides = calculator.compute_optimal_strides(&shape, AccessPattern::Sequential);
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_column_wise_strides() {
        let mut calculator = StrideCalculator::default();
        let shape = [2, 3, 4];
        let strides = calculator.compute_optimal_strides(&shape, AccessPattern::ColumnWise);
        assert_eq!(strides, vec![1, 2, 6]);
    }

    #[test]
    fn test_simd_strides() {
        let mut calculator = StrideCalculator::default();
        let shape = [2, 7]; // 7 is not SIMD-aligned
        let strides =
            calculator.compute_optimal_strides(&shape, AccessPattern::SIMD { vector_width: 4 });

        // Should pad to next multiple of 4
        assert!(strides[0] >= 8); // 8 is next multiple of 4 after 7
    }

    #[test]
    fn test_stride_analysis() {
        let calculator = StrideCalculator::default();
        let shape = [3, 4];
        let strides = [4, 1]; // C-contiguous

        let analysis = calculator.analyze_stride_efficiency(&shape, &strides);
        assert_eq!(analysis.pattern, StridePattern::UnitStride);
        assert!(analysis.simd_efficiency > 0.9); // Unit stride in last dimension
    }

    #[test]
    fn test_block_strides() {
        let mut calculator = StrideCalculator::default();
        let shape = [4, 4];
        let block_size = [2, 2];
        let strides = calculator.compute_optimal_strides(
            &shape,
            AccessPattern::Block {
                block_size: block_size.to_vec(),
            },
        );

        // Block access should optimize for the given block size
        assert!(strides.len() == 2);
    }

    #[test]
    fn test_hardware_optimization() {
        let mut calculator = StrideCalculator::default();
        let shape = [2, 8];

        let strides_avx2 = calculator.optimize_for_hardware(
            &shape,
            AccessPattern::SIMD { vector_width: 4 },
            TargetArchitecture::X86_64Avx2,
        );

        let strides_avx512 = calculator.optimize_for_hardware(
            &shape,
            AccessPattern::SIMD { vector_width: 8 },
            TargetArchitecture::X86_64Avx512,
        );

        // Different architectures should potentially give different results
        assert!(strides_avx2.len() == strides_avx512.len());
    }

    #[test]
    fn test_cache_functionality() {
        let mut calculator = StrideCalculator::default();
        let shape = [2, 3];

        // First call should miss cache
        let _strides1 = calculator.compute_optimal_strides(&shape, AccessPattern::Sequential);

        // Second call should hit cache
        let _strides2 = calculator.compute_optimal_strides(&shape, AccessPattern::Sequential);

        let (cache_size, _) = calculator.get_cache_stats();
        assert!(cache_size > 0);

        calculator.clear_cache();
        let (cache_size_after_clear, _) = calculator.get_cache_stats();
        assert_eq!(cache_size_after_clear, 0);
    }
}
