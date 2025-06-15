# NumRS2 Trait System Guide

## Overview

NumRS2's trait system provides a foundation for generic programming, extensibility, and type safety. This guide covers all traits, their purposes, and how to use them effectively.

## Core Numeric Traits

### NumericElement

The foundation trait for all numeric types in NumRS2.

```rust
pub trait NumericElement: Clone + Send + Sync + Debug + 'static {
    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;
    fn to_f64(&self) -> Option<f64>;
    fn from_f64(val: f64) -> Option<Self>;
}
```

#### Usage
```rust
use numrs::traits::NumericElement;

fn generic_accumulate<T: NumericElement>(values: &[T]) -> T {
    values.iter().fold(T::zero(), |acc, val| {
        // This would work in a real implementation with proper Add trait
        acc + val.clone()
    })
}

// Works with any numeric type
let float_sum = generic_accumulate(&[1.0, 2.0, 3.0]);
let int_sum = generic_accumulate(&[1, 2, 3]);
```

#### Implementing for Custom Types
```rust
#[derive(Debug, Clone)]
struct Money(f64); // Currency with two decimal places

impl NumericElement for Money {
    fn zero() -> Self { Money(0.0) }
    fn one() -> Self { Money(1.0) }
    fn is_zero(&self) -> bool { self.0.abs() < 0.01 }
    fn to_f64(&self) -> Option<f64> { Some(self.0) }
    fn from_f64(val: f64) -> Option<Self> { 
        Some(Money((val * 100.0).round() / 100.0)) // Round to cents
    }
}
```

### Specialized Numeric Traits

#### FloatingPoint
```rust
pub trait FloatingPoint: NumericElement {
    fn nan() -> Self;
    fn infinity() -> Self;
    fn neg_infinity() -> Self;
    fn is_nan(&self) -> bool;
    fn is_infinite(&self) -> bool;
    fn is_finite(&self) -> bool;
    fn abs(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn exp(&self) -> Self;
    fn ln(&self) -> Self;
}
```

#### IntegerElement
```rust
pub trait IntegerElement: NumericElement {
    fn min_value() -> Self;
    fn max_value() -> Self;
    fn wrapping_add(&self, other: &Self) -> Self;
    fn wrapping_sub(&self, other: &Self) -> Self;
    fn wrapping_mul(&self, other: &Self) -> Self;
    fn saturating_add(&self, other: &Self) -> Self;
    fn saturating_sub(&self, other: &Self) -> Self;
    fn saturating_mul(&self, other: &Self) -> Self;
}
```

#### ComplexElement
```rust
pub trait ComplexElement: NumericElement {
    type Real: FloatingPoint;
    fn real(&self) -> Self::Real;
    fn imag(&self) -> Self::Real;
    fn conj(&self) -> Self;
    fn norm(&self) -> Self::Real;
    fn arg(&self) -> Self::Real;
    fn from_polar(r: Self::Real, theta: Self::Real) -> Self;
}
```

## Array Operation Traits

### ArrayOps - Basic Operations

```rust
pub trait ArrayOps<T: NumericElement> {
    type Output: ArrayOps<T>;
    type Error: std::error::Error + Send + Sync + 'static;

    fn add(&self, other: &Self) -> Result<Self::Output, Self::Error>;
    fn sub(&self, other: &Self) -> Result<Self::Output, Self::Error>;
    fn mul(&self, other: &Self) -> Result<Self::Output, Self::Error>;
    fn div(&self, other: &Self) -> Result<Self::Output, Self::Error>;
    fn scalar_add(&self, scalar: T) -> Result<Self::Output, Self::Error>;
    fn scalar_mul(&self, scalar: T) -> Result<Self::Output, Self::Error>;
    fn negate(&self) -> Result<Self::Output, Self::Error>;
}
```

#### Usage Examples
```rust
use numrs::{Array, traits::ArrayOps};

fn array_operations() -> Result<(), Box<dyn std::error::Error>> {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0], [3]);
    let b = Array::from_vec(vec![4.0, 5.0, 6.0], [3]);

    // Element-wise operations
    let sum = a.add(&b)?;
    let difference = a.sub(&b)?;
    let product = a.mul(&b)?;
    let quotient = a.div(&b)?;

    // Scalar operations
    let scaled = a.scalar_mul(2.0)?;
    let shifted = a.scalar_add(10.0)?;

    // Negation
    let negated = a.negate()?;

    Ok(())
}

// Generic function using ArrayOps
fn compute_variance<T, A>(data: &A, mean: T) -> Result<A::Output, A::Error>
where
    A: ArrayOps<T>,
    T: NumericElement,
{
    let centered = data.scalar_add(-mean)?;
    let squared = centered.mul(&centered)?;
    // In a real implementation, you'd sum and divide by n-1
    Ok(squared)
}
```

### ArrayReduction - Aggregate Operations

```rust
pub trait ArrayReduction<T: NumericElement> {
    type Error: std::error::Error + Send + Sync + 'static;

    fn sum(&self) -> Result<T, Self::Error>;
    fn product(&self) -> Result<T, Self::Error>;
    fn mean(&self) -> Result<T, Self::Error>;
    fn min(&self) -> Result<T, Self::Error>;
    fn max(&self) -> Result<T, Self::Error>;
    fn sum_axis(&self, axis: usize) -> Result<Self, Self::Error>
    where
        Self: Sized;
    fn mean_axis(&self, axis: usize) -> Result<Self, Self::Error>
    where
        Self: Sized;
}
```

#### Usage Examples
```rust
use numrs::{Array, traits::ArrayReduction};

fn statistical_analysis() -> Result<(), Box<dyn std::error::Error>> {
    let data = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], [5]);

    // Basic reductions
    let total = data.sum()?;
    let average = data.mean()?;
    let minimum = data.min()?;
    let maximum = data.max()?;

    println!("Sum: {}, Mean: {}, Min: {}, Max: {}", total, average, minimum, maximum);

    // Multi-dimensional reductions
    let matrix = Array::zeros([3, 4]);
    let row_sums = matrix.sum_axis(1)?; // Sum along columns
    let col_means = matrix.mean_axis(0)?; // Mean along rows

    Ok(())
}

// Generic statistical function
fn compute_statistics<T, A>(data: &A) -> Result<(T, T, T, T), A::Error>
where
    A: ArrayReduction<T>,
    T: NumericElement,
{
    let sum = data.sum()?;
    let mean = data.mean()?;
    let min = data.min()?;
    let max = data.max()?;
    Ok((sum, mean, min, max))
}
```

### ArrayIndexing - Access and Slicing

```rust
pub trait ArrayIndexing<T: NumericElement> {
    type Output;
    type Error: std::error::Error + Send + Sync + 'static;

    fn get(&self, index: &[usize]) -> Result<&T, Self::Error>;
    fn get_mut(&mut self, index: &[usize]) -> Result<&mut T, Self::Error>;
    fn slice(&self, ranges: &[Option<std::ops::Range<usize>>]) -> Result<Self::Output, Self::Error>;
    fn select(&self, indices: &[&[usize]]) -> Result<Self::Output, Self::Error>;
}
```

#### Usage Examples
```rust
use numrs::{Array, traits::ArrayIndexing};

fn array_access() -> Result<(), Box<dyn std::error::Error>> {
    let mut matrix = Array::zeros([3, 3]);

    // Single element access
    let element = matrix.get(&[1, 2])?;
    *matrix.get_mut(&[1, 2])? = 42.0;

    // Slicing
    let top_row = matrix.slice(&[Some(0..1), None])?; // First row, all columns
    let middle_col = matrix.slice(&[None, Some(1..2)])?; // All rows, middle column

    // Advanced indexing
    let corners = matrix.select(&[
        &[0, 0], &[0, 2], &[2, 0], &[2, 2]
    ])?;

    Ok(())
}
```

### ArrayMath - Mathematical Functions

```rust
pub trait ArrayMath<T: NumericElement> {
    type Output: ArrayMath<T>;
    type Error: std::error::Error + Send + Sync + 'static;

    fn abs(&self) -> Result<Self::Output, Self::Error>;
    fn sqrt(&self) -> Result<Self::Output, Self::Error>;
    fn exp(&self) -> Result<Self::Output, Self::Error>;
    fn ln(&self) -> Result<Self::Output, Self::Error>;
    fn sin(&self) -> Result<Self::Output, Self::Error>;
    fn cos(&self) -> Result<Self::Output, Self::Error>;
    fn tan(&self) -> Result<Self::Output, Self::Error>;
    fn pow(&self, exponent: T) -> Result<Self::Output, Self::Error>;
}
```

## Linear Algebra Traits

### LinearAlgebra

```rust
pub trait LinearAlgebra<T: NumericElement> {
    type Output: LinearAlgebra<T>;
    type Error: std::error::Error + Send + Sync + 'static;

    fn dot(&self, other: &Self) -> Result<Self::Output, Self::Error>;
    fn matmul(&self, other: &Self) -> Result<Self::Output, Self::Error>;
    fn transpose(&self) -> Result<Self::Output, Self::Error>;
    fn inverse(&self) -> Result<Self::Output, Self::Error>;
    fn determinant(&self) -> Result<T, Self::Error>;
    fn eigenvalues(&self) -> Result<Vec<T>, Self::Error>;
    fn norm(&self, ord: Option<NormType>) -> Result<T, Self::Error>;
}
```

#### Usage Examples
```rust
use numrs::{Array, traits::LinearAlgebra};

fn linear_algebra_operations() -> Result<(), Box<dyn std::error::Error>> {
    let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
    let b = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], [2, 2]);

    // Matrix operations
    let product = a.matmul(&b)?;
    let transposed = a.transpose()?;
    let inverse = a.inverse()?;

    // Vector operations
    let vector_a = Array::from_vec(vec![1.0, 2.0, 3.0], [3]);
    let vector_b = Array::from_vec(vec![4.0, 5.0, 6.0], [3]);
    let dot_product = vector_a.dot(&vector_b)?;

    // Matrix properties
    let det = a.determinant()?;
    let eigenvals = a.eigenvalues()?;

    Ok(())
}
```

### MatrixDecomposition

```rust
pub trait MatrixDecomposition<T: FloatingPoint> {
    type Error: std::error::Error + Send + Sync + 'static;

    fn lu_decomposition(&self) -> Result<(Self, Self, Vec<usize>), Self::Error>
    where
        Self: Sized;
    fn qr_decomposition(&self) -> Result<(Self, Self), Self::Error>
    where
        Self: Sized;
    fn svd(&self) -> Result<(Self, Vec<T>, Self), Self::Error>
    where
        Self: Sized;
    fn cholesky_decomposition(&self) -> Result<Self, Self::Error>
    where
        Self: Sized;
    fn eigendecomposition(&self) -> Result<(Vec<T>, Self), Self::Error>
    where
        Self: Sized;
}
```

## Memory Management Traits

### MemoryAllocator

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

### SpecializedAllocator

```rust
pub trait SpecializedAllocator: MemoryAllocator {
    fn allocate_for_dtype(&self, dtype: &str, count: usize) -> Result<*mut u8, Self::Error>;
    fn can_allocate(&self, size: usize) -> bool;
    fn preferred_alignment(&self, dtype: &str) -> usize;
    fn supports_realloc(&self) -> bool;
    fn fragmentation_level(&self) -> f64;
}
```

### Usage Examples
```rust
use numrs::memory_alloc::{ArenaAllocator, SpecializedAllocator};

fn custom_allocation() -> Result<(), Box<dyn std::error::Error>> {
    let arena = ArenaAllocator::new(1024 * 1024); // 1MB arena

    // Check if allocation is possible
    if arena.can_allocate(1024) {
        let ptr = arena.allocate_for_dtype("f64", 128)?; // 128 f64 values
        
        // Use the allocated memory
        // ... 
        
        // Arena automatically cleans up when dropped
    }

    // Monitor fragmentation
    let frag_level = arena.fragmentation_level();
    if frag_level > 0.5 {
        println!("High fragmentation: {:.2}%", frag_level * 100.0);
    }

    Ok(())
}
```

## Memory Awareness Traits

### MemoryAware

```rust
pub trait MemoryAware {
    fn memory_usage(&self) -> usize;
    fn memory_layout(&self) -> MemoryLayout;
    fn is_contiguous(&self) -> bool;
    fn preferred_allocator(&self) -> AllocatorHint;
}

#[derive(Debug, Clone)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
    Strided { strides: Vec<usize> },
}

#[derive(Debug, Clone)]
pub enum AllocatorHint {
    Arena,
    Pool,
    System,
    Custom(String),
}
```

## Trait Implementation Guidelines

### Performance Considerations

```rust
// Good: Zero-cost abstractions
impl<T: NumericElement> ArrayOps<T> for Array<T> {
    type Output = Array<T>;
    type Error = NumRs2Error;

    #[inline] // Always inline simple operations
    fn add(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        // Direct delegation to optimized implementation
        self.add_impl(other)
    }
}

// Good: SIMD-friendly implementations
impl ArrayMath<f64> for Array<f64> {
    fn sqrt(&self) -> Result<Self::Output, Self::Error> {
        let mut result = self.clone();
        
        // Use SIMD when available
        #[cfg(target_feature = "avx2")]
        {
            simd_sqrt_f64(result.as_mut_slice());
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            for val in result.as_mut_slice() {
                *val = val.sqrt();
            }
        }
        
        Ok(result)
    }
}
```

### Error Handling Best Practices

```rust
use numrs::error::{CoreError, OperationContext};

impl<T: NumericElement> ArrayOps<T> for Array<T> {
    fn add(&self, other: &Self) -> Result<Self::Output, Self::Error> {
        // Validate shapes
        if self.shape() != other.shape() {
            let context = OperationContext::new("array_add")
                .with_shape(self.shape().to_vec())
                .with_shape(other.shape().to_vec());
                
            return Err(CoreError::shape_mismatch(
                self.shape().to_vec(),
                other.shape().to_vec()
            ).with_context(context).into());
        }

        // Perform operation
        Ok(self.add_impl(other))
    }
}
```

### Generic Programming Patterns

```rust
// Pattern 1: Trait bounds for generic functions
fn generic_matrix_operations<T, M>(matrix: &M) -> Result<T, M::Error>
where
    T: FloatingPoint,
    M: LinearAlgebra<T> + ArrayReduction<T>,
{
    let det = matrix.determinant()?;
    let trace = matrix.sum_axis(0)?.sum()?; // Diagonal sum approximation
    Ok(det + trace)
}

// Pattern 2: Associated types for complex relationships
trait DataProcessor<T> {
    type Input: ArrayOps<T>;
    type Output: ArrayOps<T>;
    type Config;
    
    fn process(&self, input: &Self::Input, config: &Self::Config) 
        -> Result<Self::Output, Box<dyn std::error::Error>>;
}

// Pattern 3: Trait objects for dynamic dispatch
fn apply_operation(
    arrays: &[Box<dyn ArrayOps<f64, Output = Array<f64>, Error = NumRs2Error>>]
) -> Result<Array<f64>, NumRs2Error> {
    let mut result = arrays[0].clone();
    for array in &arrays[1..] {
        result = result.add(array.as_ref())?;
    }
    Ok(result)
}
```

## Advanced Trait Patterns

### Conditional Trait Implementation

```rust
// Implement only for floating-point types
impl<T> ArrayMath<T> for Array<T>
where
    T: FloatingPoint,
{
    fn sqrt(&self) -> Result<Self::Output, Self::Error> {
        let mut result = self.clone();
        for val in result.as_mut_slice() {
            *val = val.sqrt();
        }
        Ok(result)
    }
}

// Implement only for complex types
impl<T> ArrayMath<Complex<T>> for Array<Complex<T>>
where
    T: FloatingPoint,
{
    fn abs(&self) -> Result<Self::Output, Self::Error> {
        let mut result = Array::zeros(self.shape());
        for (src, dst) in self.iter().zip(result.iter_mut()) {
            *dst = Complex::new(src.norm(), T::zero());
        }
        Ok(result)
    }
}
```

### Trait Composition

```rust
// Composite trait for complete array functionality
pub trait FullArray<T>: 
    ArrayOps<T> + 
    ArrayReduction<T> + 
    ArrayIndexing<T> + 
    ArrayMath<T> + 
    LinearAlgebra<T> +
    MemoryAware
where
    T: NumericElement,
{
    // Additional methods that require all functionality
    fn standardize(&self) -> Result<Self::Output, Self::Error>
    where
        Self: Sized,
        Self::Output: FullArray<T>,
    {
        let mean = self.mean()?;
        let centered = self.scalar_add(-mean)?;
        let variance = centered.mul(&centered)?.mean()?;
        let std_dev = variance.sqrt();
        centered.scalar_div(std_dev)
    }
}

// Blanket implementation
impl<T, A> FullArray<T> for A
where
    T: NumericElement,
    A: ArrayOps<T> + ArrayReduction<T> + ArrayIndexing<T> + ArrayMath<T> + LinearAlgebra<T> + MemoryAware,
{
    // Default implementations provided above
}
```

## Testing Trait Implementations

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use numrs::Array;

    #[test]
    fn test_trait_composition() {
        let array = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], [2, 2]);
        
        // Test that all traits work together
        let sum = array.sum().unwrap();
        let product = array.matmul(&array).unwrap();
        let abs_array = array.abs().unwrap();
        
        assert_eq!(sum, 10.0);
        assert_eq!(product.shape(), [2, 2]);
    }

    #[test]
    fn test_generic_function() {
        fn process_numeric_array<T, A>(arr: &A) -> Result<T, A::Error>
        where
            T: NumericElement,
            A: ArrayReduction<T>,
        {
            arr.sum()
        }

        let int_array = Array::from_vec(vec![1, 2, 3], [3]);
        let float_array = Array::from_vec(vec![1.0, 2.0, 3.0], [3]);
        
        assert_eq!(process_numeric_array(&int_array).unwrap(), 6);
        assert_eq!(process_numeric_array(&float_array).unwrap(), 6.0);
    }
}
```

This trait system provides a solid foundation for generic programming while maintaining performance and type safety. Use traits to write reusable, testable code that works across different numeric types and array implementations.