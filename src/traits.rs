//! Core Trait System for NumRS2
//!
//! This module provides the foundational trait hierarchy that enables extensible,
//! type-safe, and efficient numerical computing operations across different array
//! types and backends.
//!
//! ## Trait Hierarchy Overview
//!
//! ```text
//! NumericElement (base)
//! ├── FloatingPoint
//! ├── IntegerElement  
//! └── ComplexElement
//!
//! ArrayOps<T> (core operations)
//! ├── ArrayReduction<T>
//! ├── ArrayIndexing<T>
//! └── ArrayMath<T>
//!
//! LinearAlgebra<T> (matrix operations)
//! └── MatrixDecomposition<T>
//! ```

use crate::error::Result;
use crate::indexing::IndexSpec;
use num_complex::Complex;
use num_traits::Float;
use std::collections::HashMap;
use std::fmt::Debug;

// =============================================================================
// NUMERIC ELEMENT TRAITS
// =============================================================================

/// Base trait for all numeric types that can be used in NumRS2 arrays
pub trait NumericElement: Clone + Send + Sync + Debug + 'static {
    /// Zero value for this type
    fn zero() -> Self;

    /// One value for this type  
    fn one() -> Self;

    /// Check if this value is approximately zero
    fn is_zero(&self) -> bool;

    /// Convert to f64 for generic computations
    fn to_f64(&self) -> Option<f64>;

    /// Convert from f64, may fail for out-of-range values
    fn from_f64(val: f64) -> Option<Self>;
}

/// Trait for types that support floating-point operations
pub trait FloatingPoint: NumericElement + Float {
    /// Machine epsilon for this type
    fn epsilon() -> Self;

    /// Infinity value
    fn infinity() -> Self;

    /// Negative infinity value
    fn neg_infinity() -> Self;

    /// Not-a-number value
    fn nan() -> Self;

    /// Check if value is finite
    fn is_finite(&self) -> bool;

    /// Check if value is NaN
    fn is_nan(&self) -> bool;
}

/// Trait for integer types
pub trait IntegerElement: NumericElement + num_traits::PrimInt {
    /// Maximum value for this type
    fn max_value() -> Self;

    /// Minimum value for this type
    fn min_value() -> Self;

    /// Saturating addition
    fn saturating_add(&self, other: &Self) -> Self;

    /// Saturating multiplication
    fn saturating_mul(&self, other: &Self) -> Self;
}

/// Trait for complex number types
pub trait ComplexElement: NumericElement {
    type Real: FloatingPoint;

    /// Create from real and imaginary parts
    fn new(real: Self::Real, imag: Self::Real) -> Self;

    /// Get real part
    fn real(&self) -> Self::Real;

    /// Get imaginary part
    fn imag(&self) -> Self::Real;

    /// Get magnitude
    fn magnitude(&self) -> Self::Real;

    /// Get phase/argument
    fn phase(&self) -> Self::Real;

    /// Complex conjugate
    fn conj(&self) -> Self;
}

// =============================================================================
// ARRAY OPERATION TRAITS
// =============================================================================

/// Core array operations that all array types must support
pub trait ArrayOps<T: NumericElement> {
    type Output: ArrayOps<T>;
    type Error: std::error::Error + Send + Sync + 'static;

    /// Element-wise addition
    fn add(&self, other: &Self) -> Result<Self::Output>;

    /// Element-wise subtraction
    fn sub(&self, other: &Self) -> Result<Self::Output>;

    /// Element-wise multiplication
    fn mul(&self, other: &Self) -> Result<Self::Output>;

    /// Element-wise division
    fn div(&self, other: &Self) -> Result<Self::Output>;

    /// Scalar operations
    fn add_scalar(&self, scalar: T) -> Self::Output;
    fn mul_scalar(&self, scalar: T) -> Self::Output;
    fn div_scalar(&self, scalar: T) -> Result<Self::Output>;

    /// Broadcasting operations
    fn add_broadcast(&self, other: &Self) -> Result<Self::Output>;
    fn mul_broadcast(&self, other: &Self) -> Result<Self::Output>;
}

/// Array reduction operations
pub trait ArrayReduction<T: NumericElement>: Sized {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Sum all elements
    fn sum(&self) -> T;

    /// Sum along specific axis
    fn sum_axis(&self, axis: usize) -> Result<Self>;

    /// Mean of all elements
    fn mean(&self) -> T
    where
        T: std::ops::Div<Output = T> + From<usize>;

    /// Mean along specific axis
    fn mean_axis(&self, axis: Option<usize>) -> Result<Self>;

    /// Standard deviation
    fn std(&self) -> T
    where
        T: FloatingPoint;

    /// Standard deviation along axis
    fn std_axis(&self, axis: Option<usize>) -> Result<Self>;

    /// Minimum value
    fn min(&self) -> T
    where
        T: PartialOrd;

    /// Maximum value
    fn max(&self) -> T
    where
        T: PartialOrd;

    /// Argument of minimum
    fn argmin(&self) -> usize
    where
        T: PartialOrd;

    /// Argument of maximum
    fn argmax(&self) -> usize
    where
        T: PartialOrd;
}

/// Array indexing and slicing operations
pub trait ArrayIndexing<T: NumericElement> {
    type IndexResult;
    type Error: std::error::Error;

    /// Basic indexing with multi-dimensional indices
    fn get(&self, indices: &[usize]) -> Result<T>;

    /// Set value at indices
    fn set(&mut self, indices: &[usize], value: T) -> Result<()>;

    /// Advanced indexing with IndexSpec
    fn index(&self, specs: &[IndexSpec]) -> Result<Self::IndexResult>;

    /// Fancy indexing with integer arrays
    fn fancy_index(&self, indices: &[&[usize]]) -> Result<Self::IndexResult>;

    /// Boolean indexing
    fn bool_index(&self, mask: &[bool]) -> Result<Self::IndexResult>;

    /// Slice along axis
    fn slice(&self, axis: usize, start: usize, end: Option<usize>) -> Result<Self::IndexResult>;
}

/// Mathematical function operations
pub trait ArrayMath<T: NumericElement>: ArrayOps<T> {
    /// Element-wise absolute value
    fn abs(&self) -> Self::Output
    where
        T: num_traits::Signed;

    /// Element-wise square root
    fn sqrt(&self) -> Self::Output
    where
        T: FloatingPoint;

    /// Element-wise exponential
    fn exp(&self) -> Self::Output
    where
        T: FloatingPoint;

    /// Element-wise natural logarithm
    fn ln(&self) -> Self::Output
    where
        T: FloatingPoint;

    /// Element-wise sine
    fn sin(&self) -> Self::Output
    where
        T: FloatingPoint;

    /// Element-wise cosine
    fn cos(&self) -> Self::Output
    where
        T: FloatingPoint;

    /// Element-wise tangent
    fn tan(&self) -> Self::Output
    where
        T: FloatingPoint;

    /// Element-wise power
    fn pow(&self, exponent: T) -> Self::Output
    where
        T: FloatingPoint;

    /// Element-wise power with another array
    fn pow_array(&self, exponents: &Self) -> Result<Self::Output>
    where
        T: FloatingPoint;
}

// =============================================================================
// LINEAR ALGEBRA TRAITS
// =============================================================================

/// Linear algebra operations for 2D arrays (matrices)
pub trait LinearAlgebra<T: FloatingPoint>: Sized {
    type Error: std::error::Error;

    /// Matrix multiplication
    fn matmul(&self, other: &Self) -> Result<Self>;

    /// Matrix transpose
    fn transpose(&self) -> Self;

    /// Matrix determinant (for square matrices)
    fn det(&self) -> Result<T>;

    /// Matrix inverse
    fn inv(&self) -> Result<Self>;

    /// Solve linear system Ax = b
    fn solve(&self, b: &Self) -> Result<Self>;

    /// Matrix rank
    fn rank(&self) -> Result<usize>;

    /// Matrix condition number
    fn cond(&self) -> Result<T>;

    /// Matrix norms
    fn norm(&self, ord: Option<T>) -> Result<T>;
}

/// Matrix decomposition operations
pub trait MatrixDecomposition<T: FloatingPoint>: Sized {
    type DecompositionResult;
    type Error: std::error::Error;

    /// LU decomposition
    fn lu(&self) -> Result<Self::DecompositionResult>;

    /// QR decomposition  
    fn qr(&self) -> Result<Self::DecompositionResult>;

    /// SVD decomposition
    fn svd(&self) -> Result<Self::DecompositionResult>;

    /// Cholesky decomposition
    fn cholesky(&self) -> Result<Self>;

    /// Eigenvalue decomposition
    fn eig(&self) -> Result<Self::DecompositionResult>;

    /// Schur decomposition
    fn schur(&self) -> Result<Self::DecompositionResult>;
}

// =============================================================================
// MEMORY MANAGEMENT TRAITS
// =============================================================================

/// Trait for memory allocation strategies with performance optimization
pub trait MemoryAllocator: Send + Sync + std::fmt::Debug {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Allocate memory with specified size and alignment
    fn allocate(&self, layout: std::alloc::Layout) -> Result<std::ptr::NonNull<u8>>;

    /// Deallocate previously allocated memory
    ///
    /// # Safety
    /// The pointer must have been allocated by this allocator
    unsafe fn deallocate(
        &self,
        ptr: std::ptr::NonNull<u8>,
        layout: std::alloc::Layout,
    ) -> Result<()>;

    /// Reallocate memory to a new size
    ///
    /// # Safety  
    /// The pointer must have been allocated by this allocator
    unsafe fn reallocate(
        &self,
        ptr: std::ptr::NonNull<u8>,
        old_layout: std::alloc::Layout,
        new_layout: std::alloc::Layout,
    ) -> Result<std::ptr::NonNull<u8>>;

    /// Get allocation statistics if supported
    fn statistics(&self) -> Option<AllocationStats> {
        None
    }

    /// Check if this allocator supports the given layout efficiently
    fn supports_layout(&self, layout: std::alloc::Layout) -> bool;

    /// Get the preferred alignment for this allocator
    fn preferred_alignment(&self) -> usize {
        std::mem::align_of::<usize>()
    }
}

/// Memory allocation statistics
#[derive(Debug, Clone, Default)]
pub struct AllocationStats {
    /// Total bytes allocated
    pub bytes_allocated: usize,
    /// Total bytes deallocated  
    pub bytes_deallocated: usize,
    /// Number of active allocations
    pub active_allocations: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Number of allocation requests
    pub allocation_count: usize,
    /// Number of deallocation requests
    pub deallocation_count: usize,
}

/// Trait for specialized memory allocators with domain-specific optimizations
pub trait SpecializedAllocator: MemoryAllocator {
    /// Helper to create allocation errors
    fn allocation_error(&self, msg: &str) -> Self::Error;
}

/// Extension trait for type-safe array allocations (separate to maintain object safety)
pub trait ArrayAllocator {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Allocate memory optimized for numerical arrays
    fn allocate_array<T>(
        &self,
        len: usize,
    ) -> std::result::Result<std::ptr::NonNull<T>, Self::Error>;

    /// Allocate SIMD-aligned memory for vectorized operations
    fn allocate_simd_aligned<T>(
        &self,
        len: usize,
        alignment: usize,
    ) -> std::result::Result<std::ptr::NonNull<T>, Self::Error>;

    /// Allocate memory with cache-line alignment
    fn allocate_cache_aligned<T>(
        &self,
        len: usize,
    ) -> std::result::Result<std::ptr::NonNull<T>, Self::Error> {
        self.allocate_simd_aligned::<T>(len, 64) // 64-byte cache line
    }
}

/// Strategy pattern for selecting allocators based on workload characteristics  
pub trait AllocationStrategy: Send + Sync + std::fmt::Debug {
    /// Select the best allocator for the given requirements
    fn select_allocator(
        &self,
        requirements: &AllocationRequirements,
    ) -> Box<dyn SpecializedAllocator<Error = crate::error::NumRs2Error>>;

    /// Get strategy statistics
    fn strategy_stats(&self) -> StrategyStats {
        StrategyStats::default()
    }
}

/// Requirements for memory allocation to guide strategy selection
#[derive(Debug, Clone)]
pub struct AllocationRequirements {
    /// Expected allocation size
    pub size: usize,
    /// Required alignment
    pub alignment: usize,
    /// Expected allocation frequency
    pub frequency: AllocationFrequency,
    /// Whether SIMD operations will be used
    pub simd_usage: bool,
    /// Expected lifetime of the allocation
    pub lifetime: AllocationLifetime,
    /// Thread safety requirements
    pub threading: ThreadingRequirements,
}

/// Allocation frequency classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationFrequency {
    /// Very infrequent (once per operation)
    Low,
    /// Moderate frequency
    Medium,
    /// High frequency (many per operation)
    High,
    /// Very high frequency (thousands per operation)
    VeryHigh,
}

/// Expected allocation lifetime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationLifetime {
    /// Very short-lived (microseconds)
    Temporary,
    /// Short-lived (milliseconds)
    ShortTerm,
    /// Medium-lived (seconds)
    MediumTerm,
    /// Long-lived (minutes or more)
    LongTerm,
    /// Permanent for the duration of the program
    Permanent,
}

/// Threading requirements for allocations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadingRequirements {
    /// Single-threaded access only
    SingleThreaded,
    /// Multi-threaded read access
    MultiThreadedRead,
    /// Multi-threaded read/write access
    MultiThreadedReadWrite,
    /// Lock-free operations required
    LockFree,
}

/// Statistics for allocation strategies
#[derive(Debug, Clone, Default)]
pub struct StrategyStats {
    /// Number of times each allocator type was selected
    pub allocator_selections: HashMap<String, usize>,
    /// Total allocation requests handled
    pub total_requests: usize,
    /// Strategy switch count
    pub strategy_switches: usize,
}

/// Trait for memory-aware containers that can optimize based on allocation patterns
pub trait MemoryAware {
    /// Set the preferred allocator for this container
    fn set_allocator(
        &mut self,
        allocator: Box<dyn SpecializedAllocator<Error = crate::error::NumRs2Error>>,
    );

    /// Get memory usage information
    fn memory_usage(&self) -> MemoryUsage;

    /// Optimize memory layout for the current usage pattern
    fn optimize_memory_layout(&mut self) -> Result<()>;

    /// Suggest memory optimizations
    fn suggest_optimizations(&self) -> Vec<MemoryOptimization>;
}

/// Memory usage information
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Total bytes used
    pub total_bytes: usize,
    /// Number of separate allocations
    pub allocation_count: usize,
    /// Memory fragmentation level (0.0 = no fragmentation, 1.0 = highly fragmented)
    pub fragmentation: f64,
    /// Memory efficiency (bytes used / bytes allocated)
    pub efficiency: f64,
}

/// Suggested memory optimization
#[derive(Debug, Clone)]
pub struct MemoryOptimization {
    /// Type of optimization
    pub optimization_type: OptimizationType,
    /// Estimated benefit description
    pub description: String,
    /// Estimated memory savings in bytes
    pub estimated_savings: usize,
    /// Implementation complexity (1-5, 1 = easy, 5 = complex)
    pub complexity: u8,
}

/// Types of memory optimizations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationType {
    /// Switch to a more suitable allocator
    AllocatorSwitch,
    /// Combine multiple small allocations
    AllocationCoalescing,
    /// Improve memory alignment
    AlignmentOptimization,
    /// Reduce memory fragmentation
    DefragmentationCompaction,
    /// Pool frequently allocated sizes
    PoolingOptimization,
    /// Use arena allocation for bulk operations
    ArenaOptimization,
}

// =============================================================================
// IMPLEMENTATIONS FOR STANDARD TYPES
// =============================================================================

impl NumericElement for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn is_zero(&self) -> bool {
        *self == 0.0
    }
    fn to_f64(&self) -> Option<f64> {
        Some(*self as f64)
    }
    fn from_f64(val: f64) -> Option<Self> {
        Some(val as f32)
    }
}

impl NumericElement for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn is_zero(&self) -> bool {
        *self == 0.0
    }
    fn to_f64(&self) -> Option<f64> {
        Some(*self)
    }
    fn from_f64(val: f64) -> Option<Self> {
        Some(val)
    }
}

impl NumericElement for i32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn is_zero(&self) -> bool {
        *self == 0
    }
    fn to_f64(&self) -> Option<f64> {
        Some(*self as f64)
    }
    fn from_f64(val: f64) -> Option<Self> {
        Some(val as i32)
    }
}

impl NumericElement for i64 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
    fn is_zero(&self) -> bool {
        *self == 0
    }
    fn to_f64(&self) -> Option<f64> {
        Some(*self as f64)
    }
    fn from_f64(val: f64) -> Option<Self> {
        Some(val as i64)
    }
}

impl FloatingPoint for f32 {
    fn epsilon() -> Self {
        f32::EPSILON
    }
    fn infinity() -> Self {
        f32::INFINITY
    }
    fn neg_infinity() -> Self {
        f32::NEG_INFINITY
    }
    fn nan() -> Self {
        f32::NAN
    }
    fn is_finite(&self) -> bool {
        f32::is_finite(*self)
    }
    fn is_nan(&self) -> bool {
        f32::is_nan(*self)
    }
}

impl FloatingPoint for f64 {
    fn epsilon() -> Self {
        f64::EPSILON
    }
    fn infinity() -> Self {
        f64::INFINITY
    }
    fn neg_infinity() -> Self {
        f64::NEG_INFINITY
    }
    fn nan() -> Self {
        f64::NAN
    }
    fn is_finite(&self) -> bool {
        f64::is_finite(*self)
    }
    fn is_nan(&self) -> bool {
        f64::is_nan(*self)
    }
}

impl IntegerElement for i32 {
    fn max_value() -> Self {
        i32::MAX
    }
    fn min_value() -> Self {
        i32::MIN
    }
    fn saturating_add(&self, other: &Self) -> Self {
        (*self).saturating_add(*other)
    }
    fn saturating_mul(&self, other: &Self) -> Self {
        (*self).saturating_mul(*other)
    }
}

impl IntegerElement for i64 {
    fn max_value() -> Self {
        i64::MAX
    }
    fn min_value() -> Self {
        i64::MIN
    }
    fn saturating_add(&self, other: &Self) -> Self {
        (*self).saturating_add(*other)
    }
    fn saturating_mul(&self, other: &Self) -> Self {
        (*self).saturating_mul(*other)
    }
}

impl<T: FloatingPoint> ComplexElement for Complex<T> {
    type Real = T;

    fn new(real: Self::Real, imag: Self::Real) -> Self {
        Complex::new(real, imag)
    }

    fn real(&self) -> Self::Real {
        self.re
    }
    fn imag(&self) -> Self::Real {
        self.im
    }

    fn magnitude(&self) -> Self::Real {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    fn phase(&self) -> Self::Real {
        self.im.atan2(self.re)
    }

    fn conj(&self) -> Self {
        Complex::new(self.re, -self.im)
    }
}

impl<T: FloatingPoint> NumericElement for Complex<T> {
    fn zero() -> Self {
        Complex::new(<T as NumericElement>::zero(), <T as NumericElement>::zero())
    }
    fn one() -> Self {
        Complex::new(<T as NumericElement>::one(), <T as NumericElement>::zero())
    }
    fn is_zero(&self) -> bool {
        NumericElement::is_zero(&self.re) && NumericElement::is_zero(&self.im)
    }
    fn to_f64(&self) -> Option<f64> {
        NumericElement::to_f64(&self.re)
    }
    fn from_f64(val: f64) -> Option<Self> {
        T::from_f64(val).map(|r| Complex::new(r, <T as NumericElement>::zero()))
    }
}
