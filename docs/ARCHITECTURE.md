# NumRS2 Architecture Documentation

## Overview

NumRS2 is a high-performance numerical computing library for Rust, designed as a Rust-native alternative to NumPy. This document describes the architecture, design patterns, and core systems.

## Core Architecture

### Module Structure

```
src/
├── lib.rs                  # Main library entry point
├── traits.rs              # Core trait definitions
├── traits/
│   └── implementations.rs  # Trait implementations for Array<T>
├── array/                  # Array data structures
├── memory_alloc/           # Memory management systems
│   ├── arena.rs           # Arena allocator
│   ├── strategy.rs        # Allocation strategies
│   └── enhanced_traits.rs # Enhanced allocator traits
├── error/                  # Hierarchical error system
│   ├── mod.rs             # Main error module
│   ├── context.rs         # Error context and metadata
│   ├── core.rs            # Core library errors
│   ├── computation.rs     # Numerical computation errors
│   ├── memory.rs          # Memory management errors
│   ├── io.rs              # I/O and serialization errors
│   ├── legacy.rs          # Backward compatibility
│   └── hierarchical.rs    # Unified hierarchical system
├── random/                 # Random number generation
├── matrix_decomp/          # Matrix decompositions
├── fft/                    # Fast Fourier Transform
├── comparisons.rs          # Array comparison operations
├── interop/                # Interoperability layer
└── tests/                  # Test modules
```

## Core Design Principles

### 1. Trait-Based Architecture

NumRS2 uses a comprehensive trait system to enable generic programming and extensibility:

#### Numeric Element Traits
```rust
pub trait NumericElement: Clone + Send + Sync + Debug + 'static {
    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;
    fn to_f64(&self) -> Option<f64>;
    fn from_f64(val: f64) -> Option<Self>;
}
```

#### Array Operation Traits
```rust
pub trait ArrayOps<T: NumericElement> {
    type Output: ArrayOps<T>;
    type Error: std::error::Error + Send + Sync + 'static;
    
    fn add(&self, other: &Self) -> Result<Self::Output, Self::Error>;
    fn sub(&self, other: &Self) -> Result<Self::Output, Self::Error>;
    fn mul(&self, other: &Self) -> Result<Self::Output, Self::Error>;
    fn div(&self, other: &Self) -> Result<Self::Output, Self::Error>;
}
```

### 2. Memory Management System

#### Allocation Strategy Hierarchy
```rust
pub trait AllocationStrategy: Send + Sync {
    fn should_use_arena(&self, size: usize) -> bool;
    fn should_use_pool(&self, size: usize) -> bool;
    fn get_alignment(&self, dtype: &str) -> usize;
    fn estimate_fragmentation(&self) -> f64;
}
```

#### Memory Allocator Traits
- `MemoryAllocator`: Core allocation interface
- `SpecializedAllocator`: Type-specific optimizations
- `ArrayAllocator`: Array-specific memory management
- `AllocationStrategy`: Pluggable allocation strategies

### 3. Hierarchical Error System

#### Error Categories
- **Core**: Shape mismatches, indexing, basic operations
- **Computation**: Numerical instability, convergence failures
- **Memory**: Allocation failures, memory corruption
- **I/O**: File operations, serialization, network errors

#### Error Context System
```rust
pub struct ErrorContext<E> {
    error: E,
    context: OperationContext,
    location: Option<ErrorLocation>,
    chain: Vec<Box<dyn std::error::Error + Send + Sync>>,
    recovery_suggestions: Vec<String>,
}
```

## Performance Optimizations

### 1. SIMD Integration
- Vectorized operations for supported data types
- Automatic SIMD dispatch based on runtime detection
- Fallback implementations for unsupported architectures

### 2. Memory Layout Optimization
- Cache-friendly data structures
- Memory alignment for SIMD operations
- Zero-copy operations where possible

### 3. Allocation Strategies
- **Arena Allocator**: Fast allocation for temporary arrays
- **Pool Allocator**: Reuse of fixed-size blocks
- **Cache-Aware**: Optimized for processor cache hierarchy

## Backward Compatibility

### Legacy Error System
The original `NumRs2Error` enum is preserved and automatically converts to the new hierarchical system:

```rust
pub enum NumRs2Error {
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    DimensionMismatch(String),
    // ... other legacy variants
    
    // New hierarchical integration
    Core(#[from] super::hierarchical::CoreError),
    Computation(#[from] super::hierarchical::ComputationError),
    Memory(#[from] super::hierarchical::MemoryError),
    IO(#[from] super::hierarchical::IOError),
}
```

### Migration Path
1. Existing code continues to work unchanged
2. New code can opt into enhanced error system
3. Gradual migration using compatibility traits
4. No breaking changes to public APIs

## Thread Safety

### Concurrent Operations
- All core types implement `Send + Sync`
- Thread-safe memory allocators
- Lock-free operations where possible
- Parallel processing support via Rayon

### Memory Safety
- Rust's ownership system prevents data races
- Reference counting for shared arrays
- Atomic operations for statistics tracking

## Extensibility

### Custom Types
- Implement `NumericElement` for custom numeric types
- Plugin architecture for specialized operations
- Custom allocators via trait implementation

### Interoperability
- C/C++ bindings for existing libraries
- Python integration layer
- BLAS/LAPACK compatibility

## Testing Strategy

### Test Categories
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Cross-component interactions
3. **Property Tests**: Mathematical property verification
4. **Benchmark Tests**: Performance regression detection

### Quality Assurance
- Comprehensive test coverage (>95%)
- Continuous integration testing
- Memory safety verification
- Performance benchmarking

## Future Roadmap

### Phase 2: Performance & Memory Optimization (Weeks 4-7)
- SIMD optimization expansion
- Memory allocator performance tuning
- Cache-aware algorithms
- Parallel processing enhancements

### Phase 3: API Enhancement & Future-Proofing (Weeks 8-10)
- Generic programming improvements
- Enhanced type safety
- API consistency improvements
- Documentation standardization

### Phase 4: Ecosystem Integration (Weeks 11-12)
- Python bindings finalization
- Jupyter notebook integration
- Package ecosystem coordination
- Community feedback integration

## Contributing

### Development Guidelines
1. Follow the trait-based architecture
2. Maintain backward compatibility
3. Include comprehensive tests
4. Document public APIs
5. Optimize for performance and safety

### Code Review Process
1. Automated testing must pass
2. Performance benchmarks must not regress
3. Memory safety verification required
4. Documentation updates for public APIs
5. Peer review for architectural changes