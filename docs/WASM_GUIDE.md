# NumRS2 WebAssembly Guide

This comprehensive guide covers how to use NumRS2 in WebAssembly environments, including browsers and Node.js.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Build Instructions](#build-instructions)
- [Deployment](#deployment)
- [JavaScript API](#javascript-api)
- [Usage Examples](#usage-examples)
- [Performance Optimization](#performance-optimization)
- [Memory Management](#memory-management)
- [Troubleshooting](#troubleshooting)
- [Browser Compatibility](#browser-compatibility)
- [Advanced Topics](#advanced-topics)

## Introduction

NumRS2 provides full WebAssembly support, allowing you to run high-performance numerical computing directly in web browsers and Node.js environments. The WebAssembly build is:

- **Pure Rust**: 100% Rust code with no C/C++ dependencies
- **High Performance**: SIMD-accelerated operations where available
- **Small Bundle**: Optimized builds under 500KB (gzipped)
- **Type Safe**: Robust error handling with no `unwrap()` calls
- **SCIRS2 Integrated**: Built on the SciRS2 ecosystem for scientific computing

### Features

- **Array Operations**: Create and manipulate N-dimensional arrays
- **Linear Algebra**: Matrix operations, decompositions, SVD, eigenvalues
- **Statistics**: Mean, median, variance, correlation, distributions
- **Random Numbers**: Uniform and normal distributions
- **Broadcasting**: NumPy-style broadcasting for element-wise operations
- **Memory Efficient**: Optimized memory layout and allocation

## Prerequisites

Before building NumRS2 for WebAssembly, you need:

### 1. Rust Toolchain

Install the latest stable Rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Add the WASM target:

```bash
rustup target add wasm32-unknown-unknown
```

### 2. wasm-pack

Install `wasm-pack`, the build tool for Rust-generated WebAssembly:

```bash
cargo install wasm-pack
```

Or use npm:

```bash
npm install -g wasm-pack
```

### 3. Node.js (for development)

Install Node.js v18 or later:

```bash
# Download from https://nodejs.org/
# Or use nvm:
nvm install 18
```

### 4. Verify Installation

```bash
rustc --version
wasm-pack --version
node --version
```

## Installation

### From Source

Clone the NumRS2 repository:

```bash
git clone https://github.com/cool-japan/numrs.git
cd numrs
```

## Build Instructions

### Build for Web

Build the WebAssembly module for use in web browsers:

```bash
# Development build (faster compilation, larger size)
wasm-pack build --target web --features wasm

# Release build (optimized, smaller size)
wasm-pack build --target web --features wasm --release

# With LAPACK features (matrix decompositions, eigenvalues)
wasm-pack build --target web --features wasm,lapack --release
```

Output will be in `pkg/`:
- `numrs2.js` - JavaScript bindings
- `numrs2_bg.wasm` - WebAssembly binary
- `numrs2.d.ts` - TypeScript definitions
- `package.json` - NPM package metadata

### Build for Node.js

Build for Node.js environment:

```bash
wasm-pack build --target nodejs --features wasm --release
```

### Build for Bundlers

Build for webpack, rollup, or other bundlers:

```bash
wasm-pack build --target bundler --features wasm --release
```

### Custom Output Directory

Specify a custom output directory:

```bash
wasm-pack build --target web --features wasm --out-dir my-wasm-pkg
```

### Build Optimization

For production builds, enable additional optimizations in `Cargo.toml`:

```toml
[profile.release]
opt-level = "z"     # Optimize for size
lto = true          # Link-time optimization
codegen-units = 1   # Better optimization, slower compile
panic = "abort"     # Smaller binary
strip = true        # Remove debug symbols
```

Build with optimizations:

```bash
wasm-pack build --target web --features wasm --release
```

This can reduce bundle size from ~1MB to ~300-500KB (before gzip).

## Deployment

### CDN Deployment

1. Build the release version:
   ```bash
   wasm-pack build --target web --features wasm --release
   ```

2. Upload `pkg/` directory to your CDN

3. Load in your HTML:
   ```html
   <script type="module">
     import init from 'https://cdn.example.com/numrs2/pkg/numrs2.js';

     await init();
     // Use NumRS2 functions
   </script>
   ```

### NPM Package

1. Build for bundlers:
   ```bash
   wasm-pack build --target bundler --features wasm --release
   ```

2. Publish to npm:
   ```bash
   cd pkg
   npm publish
   ```

3. Install in your project:
   ```bash
   npm install numrs2-wasm
   ```

### Static File Server

For local development or simple deployment:

```bash
# Python 3
python3 -m http.server 8080

# Node.js (with http-server)
npx http-server -p 8080
```

Ensure your server sends correct MIME types:
- `.wasm` files: `application/wasm`
- `.js` files: `application/javascript`

### Vite/Webpack Configuration

See `examples/wasm/vite.config.js` for a complete Vite setup with WASM support.

## JavaScript API

### Initialization

Always initialize the WASM module before using NumRS2:

```javascript
import init, { WasmArray } from './pkg/numrs2.js';

async function main() {
    // Initialize WASM module
    await init();

    // Now you can use NumRS2 functions
    const arr = WasmArray.zeros([3, 4]);
}

main();
```

### Array Creation

```javascript
// Create zeros array
const zeros = WasmArray.zeros([2, 3]);

// Create ones array
const ones = WasmArray.ones([3, 4]);

// Create array filled with value
const fives = WasmArray.full([2, 2], 5.0);

// Create from JavaScript array
const arr = WasmArray.from_vec([1, 2, 3, 4, 5, 6], [2, 3]);

// Create random array (uniform [0, 1))
const rand = WasmArray.random([3, 3]);

// Create random array (standard normal)
const randn = WasmArray.randn([100, 100]);
```

### Array Properties

```javascript
const arr = WasmArray.zeros([2, 3, 4]);

// Get shape
console.log(arr.shape());  // [2, 3, 4]

// Get number of dimensions
console.log(arr.ndim());   // 3

// Get total number of elements
console.log(arr.size());   // 24
```

### Array Manipulation

```javascript
// Reshape
const reshaped = arr.reshape([4, 6]);

// Transpose
const transposed = arr.transpose();

// Get element
const value = arr.get([0, 1]);

// Set element
arr.set([0, 1], 5.0);

// Convert to JavaScript array
const jsArray = arr.to_vec();
```

### Array Operations

```javascript
const a = WasmArray.arange(0, 6, 1).reshape([2, 3]);
const b = WasmArray.ones([2, 3]);

// Element-wise operations
const sum = a.add(b);
const diff = a.subtract(b);
const prod = a.multiply(b);
const quot = a.divide(b);

// Scalar operations
const scaled = a.multiply_scalar(2.0);
const shifted = a.add_scalar(10.0);

// Matrix multiplication
const c = WasmArray.arange(0, 6, 1).reshape([3, 2]);
const matmul = a.matmul(c);  // [2,3] @ [3,2] = [2,2]
```

### Statistics

```javascript
import { mean, median, std_dev, variance } from './pkg/numrs2.js';

const arr = WasmArray.from_vec([1, 2, 3, 4, 5], [5]);

// Basic statistics
console.log(mean(arr));      // 3.0
console.log(median(arr));    // 3.0
console.log(std_dev(arr));   // ~1.414
console.log(variance(arr));  // 2.0

// Min/max
console.log(arr.min());      // 1.0
console.log(arr.max());      // 5.0

// Sum and product
console.log(arr.sum());      // 15.0
console.log(arr.prod());     // 120.0
```

### Linear Algebra

```javascript
import { matmul, dot_product, compute_norm, trace } from './pkg/numrs2.js';

// Matrix multiplication
const a = WasmArray.ones([2, 3]);
const b = WasmArray.ones([3, 2]);
const c = matmul(a, b);

// Dot product (1D arrays)
const v1 = WasmArray.from_vec([1, 2, 3], [3]);
const v2 = WasmArray.from_vec([4, 5, 6], [3]);
const dot = dot_product(v1, v2);  // 32.0

// Vector norm
const norm = compute_norm(v1, 2.0);  // L2 norm

// Matrix trace
const matrix = WasmArray.from_vec([1, 2, 3, 4], [2, 2]);
const tr = trace(matrix);  // 5.0
```

With `lapack` feature:

```javascript
import { determinant, matrix_inverse } from './pkg/numrs2.js';

const matrix = WasmArray.from_vec([1, 2, 3, 4], [2, 2]);

// Determinant
const det = determinant(matrix);  // -2.0

// Matrix inverse
const inv = matrix_inverse(matrix);
```

## Usage Examples

### Basic Array Operations

```javascript
import init, { WasmArray } from './pkg/numrs2.js';

async function arrayExample() {
    await init();

    // Create arrays
    const a = WasmArray.arange(0, 12, 1).reshape([3, 4]);
    const b = WasmArray.ones([3, 4]);

    // Arithmetic
    const sum = a.add(b);
    const scaled = sum.multiply_scalar(2.0);

    // Statistics
    console.log('Mean:', scaled.mean());
    console.log('Std:', scaled.std());

    // Convert to JS array
    console.log('Data:', scaled.to_vec());
}

arrayExample();
```

### Machine Learning Data Preprocessing

```javascript
async function normalizeData() {
    await init();

    // Load data (100 samples, 5 features)
    const data = WasmArray.random([100, 5]);

    // Compute statistics
    const mean = data.mean();
    const std = data.std();

    // Standardize: (x - mean) / std
    const centered = data.add_scalar(-mean);
    const normalized = centered.divide_scalar(std);

    return normalized;
}
```

### Linear Regression

```javascript
import { matmul, matrix_inverse } from './pkg/numrs2.js';

async function linearRegression(X, y) {
    await init();

    // Add bias column: X_bias = [1, X]
    const ones = WasmArray.ones([X.shape()[0], 1]);
    const X_bias = ones.concatenate(X, 1);

    // Compute: w = (X^T @ X)^(-1) @ X^T @ y
    const Xt = X_bias.transpose();
    const XtX = matmul(Xt, X_bias);
    const XtX_inv = matrix_inverse(XtX);
    const Xty = matmul(Xt, y);
    const weights = matmul(XtX_inv, Xty);

    return weights;
}
```

### Matrix Decomposition

```javascript
import { singular_value_decomposition } from './pkg/numrs2.js';

async function imageCompression(image, k) {
    await init();

    // SVD: A = U @ S @ V^T
    const [U, S, Vt] = singular_value_decomposition(image);

    // Keep only top k singular values
    const U_k = U.slice_columns([0, k]);
    const S_k = S.slice([0], [k]);
    const Vt_k = Vt.slice_rows([0, k]);

    // Reconstruct: A_k = U_k @ S_k @ Vt_k
    const S_diag = WasmArray.diag(S_k);
    const temp = matmul(U_k, S_diag);
    const compressed = matmul(temp, Vt_k);

    return compressed;
}
```

### Time Series Analysis

```javascript
async function movingAverage(data, window) {
    await init();

    const n = data.size();
    const result = WasmArray.zeros([n - window + 1]);

    for (let i = 0; i <= n - window; i++) {
        const slice = data.slice([i], [i + window]);
        result.set([i], slice.mean());
    }

    return result;
}
```

## Performance Optimization

### Bundle Size Optimization

1. **Use release builds** with size optimization:
   ```bash
   wasm-pack build --target web --features wasm --release
   ```

2. **Enable LTO** in `Cargo.toml`:
   ```toml
   [profile.release]
   lto = true
   opt-level = "z"
   ```

3. **Compress with gzip/brotli**:
   - ~1MB uncompressed → ~300KB gzipped

4. **Lazy loading**:
   ```javascript
   // Load WASM only when needed
   button.onclick = async () => {
       const { init, WasmArray } = await import('./pkg/numrs2.js');
       await init();
       // Use NumRS2
   };
   ```

### Runtime Performance

1. **Use SIMD when available**:
   ```javascript
   import { is_simd_available } from './pkg/numrs2.js';

   if (is_simd_available()) {
       console.log('SIMD acceleration enabled!');
   }
   ```

2. **Minimize JS ↔ WASM boundary crossings**:
   ```javascript
   // Bad: Multiple calls
   for (let i = 0; i < 1000; i++) {
       const arr = WasmArray.zeros([10]);
   }

   // Good: Single call
   const arr = WasmArray.zeros([1000, 10]);
   ```

3. **Reuse arrays**:
   ```javascript
   // Bad: Create new array each time
   function process() {
       const temp = WasmArray.zeros([100, 100]);
       // ... use temp
   }

   // Good: Reuse array
   const temp = WasmArray.zeros([100, 100]);
   function process() {
       // ... use temp
       temp.fill(0.0);  // Reset if needed
   }
   ```

4. **Use Web Workers** for heavy computation:
   ```javascript
   // worker.js
   importScripts('./pkg/numrs2.js');

   self.addEventListener('message', async (e) => {
       const { init, WasmArray } = wasm_bindgen;
       await init('./pkg/numrs2_bg.wasm');

       const result = WasmArray.zeros(e.data.shape);
       self.postMessage({ result: result.to_vec() });
   });
   ```

### SIMD Support

NumRS2 automatically uses WASM SIMD when available:

- **Chrome**: Version 91+ (enable at `chrome://flags/#enable-webassembly-simd`)
- **Firefox**: Version 89+
- **Safari**: Version 16.4+
- **Node.js**: Version 16+

Check SIMD availability:

```javascript
if (is_simd_available()) {
    console.log('SIMD enabled - expect 2-4x speedup');
} else {
    console.log('SIMD not available - using scalar fallback');
}
```

## Memory Management

### Automatic Memory Management

NumRS2 uses WebAssembly's linear memory model. Memory is automatically managed:

```javascript
{
    const arr = WasmArray.zeros([1000, 1000]);
    // ... use arr
} // Memory freed when arr goes out of scope
```

### Manual Cleanup

For explicit cleanup (rarely needed):

```javascript
const arr = WasmArray.zeros([1000, 1000]);
// ... use arr
arr.free();  // Explicit cleanup
```

### Memory Limits

WebAssembly has a maximum memory size:

- **32-bit WASM**: 4GB maximum
- **Initial allocation**: 17MB (configurable)
- **Growth**: Automatic up to limit

For large arrays:

```javascript
try {
    const huge = WasmArray.zeros([10000, 10000]);  // 800MB
} catch (e) {
    console.error('Out of memory:', e);
    // Handle error
}
```

### Memory Profiling

Monitor memory usage:

```javascript
if (performance.memory) {
    console.log('Used:', performance.memory.usedJSHeapSize / 1024 / 1024, 'MB');
    console.log('Total:', performance.memory.totalJSHeapSize / 1024 / 1024, 'MB');
}
```

## Troubleshooting

### Common Issues

#### 1. Module Not Found

**Error**: `Cannot find module './pkg/numrs2.js'`

**Solution**: Build the WASM module first:
```bash
wasm-pack build --target web --features wasm
```

#### 2. MIME Type Error

**Error**: `Failed to execute 'compile' on 'WebAssembly': Incorrect response MIME type`

**Solution**: Configure server to serve `.wasm` files with `application/wasm` MIME type.

For nginx:
```nginx
types {
    application/wasm wasm;
}
```

For Apache (`.htaccess`):
```apache
AddType application/wasm .wasm
```

#### 3. CORS Errors

**Error**: `CORS policy: No 'Access-Control-Allow-Origin' header`

**Solution**: Enable CORS on your server:

```javascript
// Node.js (Express)
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    next();
});
```

#### 4. SharedArrayBuffer Not Defined

**Error**: `SharedArrayBuffer is not defined`

**Solution**: Add required headers (for multi-threading support):

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

#### 5. Out of Memory

**Error**: `RuntimeError: memory access out of bounds`

**Solutions**:
- Reduce array sizes
- Process data in batches
- Use streaming operations
- Increase WASM memory limit (if possible)

#### 6. Slow Performance

**Symptoms**: Operations slower than expected

**Solutions**:
- Use release build (`--release`)
- Enable SIMD in browser
- Check for excessive JS ↔ WASM calls
- Use Web Workers for parallelism
- Profile with browser DevTools

### Debugging

Enable verbose logging:

```javascript
console.log('WASM module loaded');
console.log('Version:', get_version());
console.log('SIMD:', is_simd_available());
```

Use browser DevTools:
- Chrome: `chrome://inspect`
- Firefox: `about:debugging`

Check WASM memory:
```javascript
// In Chrome DevTools Console
performance.memory
```

## Browser Compatibility

### Support Matrix

| Browser | Version | WASM | SIMD | Notes |
|---------|---------|------|------|-------|
| **Chrome** | 57+ | ✅ | 91+ | Recommended |
| **Firefox** | 52+ | ✅ | 89+ | Good support |
| **Safari** | 11+ | ✅ | 16.4+ | Limited SIMD |
| **Edge** | 79+ | ✅ | 91+ | Chrome-based |
| **Opera** | 44+ | ✅ | 77+ | Chrome-based |
| **Samsung Internet** | 7.2+ | ✅ | ❌ | No SIMD |
| **Node.js** | 16+ | ✅ | ✅ | Full support |

### Feature Detection

```javascript
// Check WASM support
if (typeof WebAssembly !== 'undefined') {
    console.log('WebAssembly supported');
} else {
    console.error('WebAssembly not supported - use fallback');
}

// Check SIMD support
const simd = await WebAssembly.validate(
    new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0])
);
console.log('SIMD supported:', simd);
```

### Polyfills

No polyfills are needed for modern browsers (2020+). For older browsers, consider:

- Providing a JavaScript fallback
- Showing an upgrade message
- Using progressive enhancement

## Advanced Topics

### Web Workers

Run NumRS2 in a background thread:

```javascript
// worker.js
importScripts('./pkg/numrs2.js');

let initialized = false;

self.addEventListener('message', async (e) => {
    if (!initialized) {
        const { init } = wasm_bindgen;
        await init('./pkg/numrs2_bg.wasm');
        initialized = true;
    }

    const { WasmArray } = wasm_bindgen;
    const result = WasmArray.zeros(e.data.shape);

    // Send result back
    self.postMessage({
        shape: result.shape(),
        data: result.to_vec()
    });
});

// main.js
const worker = new Worker('worker.js');
worker.postMessage({ shape: [1000, 1000] });
worker.onmessage = (e) => {
    console.log('Result:', e.data);
};
```

### Custom Memory Allocator

NumRS2 uses `wee_alloc` for smaller binary size. You can customize in `Cargo.toml`:

```toml
[dependencies]
wee_alloc = { version = "0.4.5", optional = true }

[features]
default = []
wasm = ["dep:wasm-bindgen", "dep:wee_alloc"]
```

### TypeScript Integration

Use TypeScript definitions:

```typescript
import init, { WasmArray } from './pkg/numrs2';

async function typed() {
    await init();

    const arr: WasmArray = WasmArray.zeros([2, 3]);
    const shape: number[] = arr.shape();
    const mean: number = arr.mean();
}
```

### Streaming Large Data

Process large datasets in chunks:

```javascript
async function processLargeDataset(data) {
    await init();

    const chunkSize = 10000;
    const results = [];

    for (let i = 0; i < data.length; i += chunkSize) {
        const chunk = data.slice(i, i + chunkSize);
        const arr = WasmArray.from_vec(chunk, [chunk.length]);
        results.push(arr.mean());
    }

    return results;
}
```

## Documentation Links

- [NumRS2 Main Documentation](https://docs.rs/numrs2)
- [Examples](../examples/wasm/)
- [GitHub Repository](https://github.com/cool-japan/numrs)
- [SciRS2 Ecosystem](https://github.com/cool-japan/scirs2-core)
- [wasm-pack Documentation](https://rustwasm.github.io/wasm-pack/)

## Support

For questions and issues:

- Open an issue on [GitHub](https://github.com/cool-japan/numrs/issues)
- Check [discussions](https://github.com/cool-japan/numrs/discussions)

---

**NumRS2 WebAssembly** - Pure Rust Numerical Computing for the Web

Part of the **COOLJAPAN Ecosystem** - 100% Pure Rust, Zero C Dependencies

Copyright © 2025 COOLJAPAN OU (Team KitaSan) | Apache License 2.0
