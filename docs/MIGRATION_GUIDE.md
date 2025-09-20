# NumRS2 Migration Guide

## Overview

This guide helps developers migrate from the legacy NumRS2 systems to the new enhanced architecture introduced in version 0.1.0-beta.1. The migration maintains 100% backward compatibility while providing opt-in access to enhanced features.

## Error System Migration

### Legacy to Hierarchical Errors

#### Before (Legacy System)
```rust
use numrs::error::NumRs2Error;

fn legacy_function() -> Result<(), NumRs2Error> {
    // Old error handling
    Err(NumRs2Error::ShapeMismatch {
        expected: vec![2, 3],
        actual: vec![3, 2],
    })
}
```

#### After (Enhanced System)
```rust
use numrs::error::{CoreError, ErrorContext, OperationContext};

fn enhanced_function() -> Result<(), ErrorContext<CoreError>> {
    // Enhanced error with context
    let context = OperationContext::new("matrix_multiply")
        .with_shape(vec![2, 3])
        .with_shape(vec![3, 2])
        .with_parameter("method", "BLAS");
    
    Err(CoreError::shape_mismatch(vec![2, 3], vec![3, 2])
        .with_context(context)
        .with_suggestion("Ensure matrix dimensions are compatible for multiplication"))
}
```

#### Gradual Migration Strategy
```rust
use numrs::error::{NumRs2Error, CoreError};

// Option 1: Keep using legacy, auto-conversion happens
fn mixed_approach() -> Result<(), NumRs2Error> {
    let core_error = CoreError::shape_mismatch(vec![2, 3], vec![3, 2]);
    Err(NumRs2Error::Core(core_error)) // Automatic conversion
}

// Option 2: Use both systems during transition
fn transition_function() -> Result<(), NumRs2Error> {
    match some_operation() {
        Ok(result) => Ok(result),
        Err(legacy_err) => {
            // Convert to enhanced for logging/debugging
            let enhanced = legacy_err.with_context(
                OperationContext::new("transition_function")
            );
            eprintln!("Enhanced error info: {}", enhanced);
            Err(enhanced.into_inner()) // Convert back to legacy
        }
    }
}
```

### Error Categories and Severity

#### Understanding Error Categories
```rust
use numrs::error::{ErrorCategory, ErrorSeverity, NumRs2Error};

fn handle_error_by_category(err: &NumRs2Error) {
    match err.category() {
        ErrorCategory::Core => {
            // Handle shape mismatches, indexing errors
            eprintln!("Core library error: {}", err);
        },
        ErrorCategory::Computation => {
            // Handle numerical instability, convergence issues
            eprintln!("Computation error - may need different algorithm: {}", err);
        },
        ErrorCategory::Memory => {
            // Handle allocation failures, memory pressure
            eprintln!("Memory error - consider reducing data size: {}", err);
        },
        ErrorCategory::IO => {
            // Handle file operations, serialization issues
            eprintln!("I/O error - check file permissions and format: {}", err);
        },
    }
    
    // Handle by severity
    match err.severity() {
        ErrorSeverity::Critical => {
            // Immediately abort operation
            panic!("Critical error: {}", err);
        },
        ErrorSeverity::High => {
            // Log and try alternative approach
            log::error!("High severity error: {}", err);
        },
        ErrorSeverity::Medium | ErrorSeverity::Low => {
            // Log and continue with fallback
            log::warn!("Recoverable error: {}", err);
        },
    }
}
```

## Trait System Migration

### Legacy Array Operations
```rust
// Before - direct method calls
use numrs::Array;

fn legacy_operations() {
    let a = Array::zeros([3, 3]);
    let b = Array::ones([3, 3]);
    let result = a.add(&b); // Direct method call
}
```

### Enhanced Trait-Based Operations
```rust
// After - trait-based operations (backward compatible)
use numrs::{Array, traits::ArrayOps};

fn enhanced_operations() {
    let a = Array::zeros([3, 3]);
    let b = Array::ones([3, 3]);
    
    // Option 1: Still works (legacy)
    let result1 = a.add(&b);
    
    // Option 2: Trait-based (new)
    let result2 = ArrayOps::add(&a, &b).expect("Addition failed");
    
    // Option 3: Generic programming (new capability)
    fn generic_add<T, A>(x: &A, y: &A) -> A::Output 
    where 
        A: ArrayOps<T>,
        T: numrs::traits::NumericElement,
    {
        x.add(y).expect("Addition failed")
    }
}
```

### Implementing Custom Types
```rust
use numrs::traits::{NumericElement, ArrayOps};

// Custom numeric type
#[derive(Debug, Clone)]
struct MyNumber(f64);

impl NumericElement for MyNumber {
    fn zero() -> Self { MyNumber(0.0) }
    fn one() -> Self { MyNumber(1.0) }
    fn is_zero(&self) -> bool { self.0 == 0.0 }
    fn to_f64(&self) -> Option<f64> { Some(self.0) }
    fn from_f64(val: f64) -> Option<Self> { Some(MyNumber(val)) }
}

// Now MyNumber works with all NumRS2 generic operations
```

## Memory Management Migration

### Legacy Memory Usage
```rust
// Before - implicit memory management
let large_array = Array::zeros([10000, 10000]);
// Memory allocation happens behind the scenes
```

### Enhanced Memory Control
```rust
use numrs::memory_alloc::{ArenaAllocator, AllocationStrategy, IntelligentAllocationStrategy};

// Option 1: Use intelligent allocation strategy
let strategy = IntelligentAllocationStrategy::new()
    .with_arena_threshold(1024 * 1024) // Use arena for large allocations
    .with_cache_awareness(true);

// Option 2: Custom allocator for specific use cases
fn with_arena_allocator() {
    let arena = ArenaAllocator::new(1024 * 1024 * 100); // 100MB arena
    // Use arena for temporary calculations
    let temp_arrays = create_temporary_arrays_in_arena(&arena);
    // Arena automatically cleans up when dropped
}

// Option 3: Memory pressure monitoring
use numrs::error::{MemoryInfo, MemoryPressure};

fn monitor_memory_usage() {
    let memory_info = MemoryInfo {
        total_allocated: get_total_allocated(),
        peak_usage: get_peak_usage(),
        available_memory: Some(get_available_memory()),
        pressure_level: MemoryPressure::Medium,
    };
    
    match memory_info.pressure_level {
        MemoryPressure::High | MemoryPressure::Critical => {
            // Reduce memory usage
            cleanup_temporary_arrays();
        },
        _ => {
            // Continue normal operation
        }
    }
}
```

## Performance Migration

### SIMD Optimization Migration
```rust
// Before - manual SIMD usage
#[cfg(target_feature = "avx2")]
fn manual_simd_operation(data: &[f64]) -> Vec<f64> {
    // Manual SIMD implementation
    unimplemented!()
}

// After - automatic SIMD dispatch
use numrs::Array;

fn automatic_simd_operation(data: &Array<f64>) -> Array<f64> {
    // SIMD automatically used when available and beneficial
    data.map(|x| x * 2.0) // Automatically vectorized
}
```

### Memory Layout Optimization
```rust
use numrs::{Array, memory_alloc::AllocationStrategy};

// Configure memory layout for optimal performance
fn optimize_memory_layout() {
    let strategy = AllocationStrategy::cache_aware()
        .with_alignment(32) // AVX2 alignment
        .with_prefetch_hint(true);
    
    let array = Array::with_allocator(strategy)
        .zeros([1000, 1000]);
    
    // Array is automatically laid out for optimal cache performance
}
```

## API Migration Examples

### Random Number Generation
```rust
// Before
use numrs::random::RandomState;

fn legacy_random() {
    let mut rng = RandomState::new();
    let value = rng.random_range(0.0..1.0); // Old API
}

// After (backward compatible)
use numrs::random::RandomState;

fn enhanced_random() {
    let mut rng = RandomState::new();
    
    // Option 1: Legacy API still works
    let value1 = rng.random_range(0.0..1.0);
    
    // Option 2: Enhanced API with error handling
    let value2 = rng.try_random_range(0.0..1.0)
        .expect("Random generation failed");
    
    // Option 3: Seed management
    let seeded_rng = RandomState::with_seed(12345);
}
```

### Matrix Operations
```rust
// Before
use numrs::Array;

fn legacy_matrix_ops() {
    let a = Array::zeros([100, 100]);
    let b = Array::ones([100, 100]);
    let result = a.dot(&b);
}

// After - enhanced with error handling and context
use numrs::{Array, traits::LinearAlgebra, error::OperationContext};

fn enhanced_matrix_ops() -> Result<Array<f64>, numrs::error::NumRs2Error> {
    let a = Array::zeros([100, 100]);
    let b = Array::ones([100, 100]);
    
    // Enhanced operation with context
    let context = OperationContext::new("matrix_multiply")
        .with_shape(vec![100, 100])
        .with_shape(vec![100, 100])
        .with_parameter("algorithm", "BLAS");
    
    LinearAlgebra::dot(&a, &b)
        .map_err(|e| e.with_context(context))
}
```

## Testing Migration

### Enhanced Test Utilities
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use numrs::error::{ErrorCategory, ErrorSeverity};
    
    #[test]
    fn test_error_handling() {
        let result = some_operation_that_fails();
        
        match result {
            Err(e) => {
                // Test error category
                assert_eq!(e.category(), ErrorCategory::Core);
                
                // Test severity
                assert_eq!(e.severity(), ErrorSeverity::High);
                
                // Test recovery suggestions
                assert!(!e.recovery_suggestions().is_empty());
            },
            Ok(_) => panic!("Expected error"),
        }
    }
    
    #[test]
    fn test_memory_pressure_handling() {
        // Test memory pressure scenarios
        let large_operation = || {
            // Operation that might cause memory pressure
            Array::zeros([10000, 10000])
        };
        
        // Monitor memory usage during test
        let result = large_operation();
        assert!(result.is_ok());
    }
}
```

## Migration Checklist

### Phase 1: Immediate (No Code Changes Required)
- [ ] Update to NumRS2 0.1.0-beta.1
- [ ] Run existing tests - all should pass
- [ ] No breaking changes in public API

### Phase 2: Enhanced Error Handling (Optional)
- [ ] Replace manual error handling with error categories
- [ ] Add error severity checking
- [ ] Implement recovery suggestions
- [ ] Use error context for debugging

### Phase 3: Trait-Based Operations (Optional)
- [ ] Migrate to trait-based operations for new code
- [ ] Implement custom numeric types if needed
- [ ] Use generic programming for reusable components

### Phase 4: Memory Optimization (Performance Improvement)
- [ ] Configure allocation strategies for your use case
- [ ] Monitor memory pressure in critical paths
- [ ] Use arena allocators for temporary calculations

### Phase 5: Full Migration (Long-term)
- [ ] Fully adopt hierarchical error system
- [ ] Use enhanced memory management throughout
- [ ] Leverage trait system for extensibility
- [ ] Optimize performance with new features

## Common Migration Issues

### Issue 1: Import Path Changes
```rust
// Old
use numrs::error::NumRs2Error;

// New (both work)
use numrs::error::NumRs2Error; // Still works
use numrs::error::prelude::*;  // Recommended for new code
```

### Issue 2: Result Type Compatibility
```rust
// Functions returning old Result<T> work with new error types
fn legacy_result() -> numrs::error::Result<i32> {
    Ok(42)
}

fn new_code() {
    match legacy_result() {
        Ok(value) => println!("Success: {}", value),
        Err(e) => {
            // Can use new error features
            println!("Error category: {}", e.category());
            println!("Severity: {}", e.severity());
        }
    }
}
```

### Issue 3: Performance Regression Concerns
```rust
// New trait system has zero runtime cost
fn performance_test() {
    // Legacy call
    let result1 = array1.add(&array2);
    
    // Trait-based call - identical performance
    let result2 = ArrayOps::add(&array1, &array2).unwrap();
    
    // Both compile to identical assembly code
}
```

## Getting Help

### Documentation Resources
- `docs/ARCHITECTURE.md` - System architecture overview
- `docs/TRAIT_GUIDE.md` - Comprehensive trait system guide
- `docs/ERROR_HANDLING.md` - Error system documentation
- `docs/MEMORY_MANAGEMENT.md` - Memory management guide

### Community Support
- GitHub Issues: Report migration problems
- Documentation: In-code documentation for all public APIs
- Examples: See `examples/` directory for migration examples

### Performance Profiling
```rust
// Use built-in profiling for migration validation
use numrs::profiling::profile_operation;

let result = profile_operation("matrix_multiply", || {
    your_migrated_operation()
});

println!("Operation took: {:?}", result.duration);
println!("Memory used: {} bytes", result.peak_memory);
```