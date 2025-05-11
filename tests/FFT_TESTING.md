# FFT Testing Guide

This document provides an overview of the testing approach for Fast Fourier Transform (FFT) operations in NumRS2.

## Testing Strategy

The FFT operations in NumRS2 are tested using a multi-faceted approach:

1. **Property-based testing**: Testing mathematical properties and relationships
2. **Reference testing**: Testing against known reference values
3. **Performance benchmarking**: Measuring and optimizing performance

### Property-Based Testing

Property-based testing focuses on verifying the mathematical properties and relationships that FFT operations should satisfy. These include:

- **Fundamental FFT properties**:
  - FFT followed by IFFT should recover the original signal
  - For real input, the FFT has complex conjugate symmetry
  - Linearity: FFT(a*x + b*y) = a*FFT(x) + b*FFT(y)
  - Shift property: If y[n] = x[n-m], then Y[k] = X[k] * e^(-j*2π*k*m/N)
  - Convolution property: FFT(x * y) = FFT(x) . FFT(y) (element-wise product)
  - Parseval's theorem: Sum of |x[n]|² = (1/N) * Sum of |X[k]|²

- **Special case behaviors**:
  - FFT of a delta function is a constant
  - FFT of a complex exponential is a delta function
  - Modulation property: If y[n] = x[n] * cos(2π*f₀*n/N), then Y[k] = 0.5 * (X[k-f₀] + X[k+f₀])

- **Real FFT properties**:
  - RFFT should contain n/2+1 values
  - IRFFT should recover the original signal

- **2D FFT properties**:
  - 2D FFT followed by 2D IFFT should recover the original array
  - 2D FFT should have conjugate symmetry for real input

- **Window functions**:
  - Window functions should be symmetric
  - Window functions should have specific values at endpoints
  - Window functions should effectively reduce spectral leakage

### Reference Testing

Reference tests compare the computed values against known reference implementations or analytical solutions. These tests:

- Verify that the FFT of specific signals (e.g., sinusoids, delta functions) matches expected results
- Check spectral peak detection
- Verify frequency axis generation
- Test that FFT shift operations work correctly

### Performance Benchmarking

Performance benchmarks measure the execution time of FFT operations with various input sizes and types. These benchmarks:

- Track performance as a function of input size
- Compare different FFT implementations (e.g., FFT vs. RFFT)
- Evaluate window function performance
- Assess end-to-end FFT workflows

## Testing Strategy for Specific Functions

### 1D FFT

1D FFT functions (fft, ifft, rfft, irfft) are tested for:

- Basic properties like FFT followed by IFFT
- Linearity
- Shift property
- Convolution property
- Conjugate symmetry
- Parseval's theorem
- Spectral peak detection

### 2D FFT

2D FFT functions (fft2, ifft2, rfft2, irfft2) are tested for:

- 2D FFT followed by 2D IFFT should recover the original array
- Conjugate symmetry
- Handling of different input shapes

### Window Functions

Window functions (apply_window) are tested for:

- Symmetry
- Endpoint values (e.g., Hann window is zero at endpoints)
- Spectral leakage reduction
- Effect on spectral resolution

### FFT Shift Operations

FFT shift operations (fftshift, ifftshift) are tested for:

- Correctly moving frequency components
- Reversibility (fftshift followed by ifftshift)
- Handling of even and odd-sized arrays

### Frequency Axis Generation

Frequency axis generation (fftfreq, rfftfreq) are tested for:

- Correct frequency values
- Handling of Nyquist frequency
- Positive and negative frequency mapping

## Running the Tests

To run tests for FFT operations, use the following commands:

```bash
# Run property-based tests for FFT operations
cargo test --test test_fft_properties

# Run performance benchmarks for FFT operations
cargo bench --bench fft_benchmark
```

## Adding New Tests

When adding new FFT-related functions or modifying existing ones, please also add corresponding tests:

1. **Add property tests** in `tests/test_fft_properties.rs`
2. **Add benchmarks** in `benches/fft_benchmark.rs`

Focus on testing:
- Mathematical properties and identities
- Edge cases and numerical stability
- Performance characteristics
- Compatibility with different input types and sizes

## Common Testing Patterns

### Testing FFT Followed by IFFT

```rust
// Apply FFT
let fft_x = FFT::fft(&x).unwrap();

// Apply IFFT
let ifft_x = FFT::ifft(&fft_x).unwrap();

// Check that the original signal is recovered
for i in 0..n {
    assert_abs_diff_eq!(
        ifft_x.to_vec()[i].re, 
        x.to_vec()[i], 
        epsilon = TOLERANCE,
        "FFT followed by IFFT should recover the original signal"
    );
}
```

### Testing Spectral Peak Detection

```rust
// Create a sinusoidal signal with known frequency
let frequency = 10.0; // Hz
let signal = generate_sinusoidal(n, frequency, sample_rate);

// Compute FFT and magnitude spectrum
let fft_result = FFT::fft(&signal).unwrap();
let mut magnitude = Vec::with_capacity(n);
for val in fft_result.to_vec() {
    magnitude.push(val.norm());
}

// Find spectral peak
let mut max_idx = 0;
for i in 1..n/2 {
    if magnitude[i] > magnitude[max_idx] {
        max_idx = i;
    }
}

// Compute frequency of the peak
let freq_axis = FFT::fftfreq(n, 1.0 / sample_rate).unwrap();
let detected_freq = freq_axis.to_vec()[max_idx];

// Check that the detected frequency is close to the input frequency
assert_abs_diff_eq!(
    detected_freq.abs(), 
    frequency, 
    epsilon = sample_rate / n as f64,
    "Detected frequency should match input frequency"
);
```

### Testing Window Functions

```rust
// Apply window function
let windowed = FFT::apply_window(&signal, "hann").unwrap();
let windowed_data = windowed.to_vec();

// Check symmetry
for i in 0..n/2 {
    assert_abs_diff_eq!(
        windowed_data[i], 
        windowed_data[n-1-i], 
        epsilon = TOLERANCE,
        "Window function should be symmetric"
    );
}

// Check endpoints
assert_abs_diff_eq!(
    windowed_data[0], 
    0.0, 
    epsilon = TOLERANCE,
    "Hann window should be zero at endpoints"
);
assert_abs_diff_eq!(
    windowed_data[n-1], 
    0.0, 
    epsilon = TOLERANCE,
    "Hann window should be zero at endpoints"
);
```

## References

For verifying FFT properties and behavior, we use the following references:

1. Oppenheim, A.V. and Schafer, R.W. (2009), "Discrete-Time Signal Processing"
2. Brigham, E.O. (1988), "The Fast Fourier Transform and Its Applications"
3. NumPy's FFT implementation
4. FFTW (Fastest Fourier Transform in the West) library