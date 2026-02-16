# NumRS2 WebAssembly Demo

This directory contains a complete web application demonstrating NumRS2's WebAssembly capabilities. The demo showcases array operations, linear algebra, statistics, and performance benchmarks running directly in your browser.

## Features

- **Array Operations**: Create and manipulate N-dimensional arrays
- **Linear Algebra**: Matrix operations, decompositions, and solvers
- **Statistics**: Statistical functions and probability distributions
- **Performance Benchmarks**: Compare WASM vs native JavaScript performance
- **Interactive UI**: Real-time examples with visual feedback

## Prerequisites

Before running this demo, you need:

1. **Rust** (latest stable version)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **wasm-pack** (WASM build tool)
   ```bash
   cargo install wasm-pack
   ```

3. **Node.js** (v18 or later)
   ```bash
   # Download from https://nodejs.org/
   # Or use nvm:
   nvm install 18
   ```

4. **WASM target** (for Rust)
   ```bash
   rustup target add wasm32-unknown-unknown
   ```

## Quick Start

### 1. Build the WASM Module

From the project root (`numrs/`):

```bash
# Development build (faster compilation)
wasm-pack build --target web --features wasm --out-dir examples/wasm/pkg

# Release build (optimized, smaller bundle)
wasm-pack build --target web --features wasm --release --out-dir examples/wasm/pkg
```

### 2. Install Dependencies

From this directory (`examples/wasm/`):

```bash
npm install
```

### 3. Run the Development Server

```bash
npm run dev
```

This will start a Vite development server at `http://localhost:3000` and open it in your browser.

## Build Scripts

The `package.json` includes several useful scripts:

| Script | Description |
|--------|-------------|
| `npm run build` | Build WASM module for web (development) |
| `npm run build:release` | Build WASM module for web (optimized) |
| `npm run build:nodejs` | Build WASM module for Node.js |
| `npm run build:bundler` | Build WASM module for webpack/rollup |
| `npm run dev` | Start development server |
| `npm run preview` | Preview production build |
| `npm run serve` | Simple Python HTTP server |
| `npm run clean` | Clean all build artifacts |

## Project Structure

```
examples/wasm/
├── index.html           # Main HTML page with UI
├── app.js              # JavaScript application logic
├── package.json        # NPM configuration
├── vite.config.js      # Vite build configuration
├── README.md           # This file
└── pkg/                # WASM build output (generated)
    ├── numrs2.js       # JavaScript bindings
    ├── numrs2_bg.wasm  # WASM binary
    └── ...
```

## Usage Examples

### Basic Array Creation

```javascript
import init, { WasmArray } from './pkg/numrs2.js';

await init();

// Create arrays
const zeros = WasmArray.zeros([3, 4]);
const ones = WasmArray.ones([2, 5]);
const range = WasmArray.arange(0, 10, 1);
const random = WasmArray.random([3, 3]);

console.log(zeros.shape());  // [3, 4]
console.log(ones.toString());
```

### Linear Algebra

```javascript
// Matrix multiplication
const a = WasmArray.arange(0, 6, 1).reshape([2, 3]);
const b = WasmArray.arange(0, 6, 1).reshape([3, 2]);
const c = a.matmul(b);

// Matrix operations
const matrix = WasmArray.from_array([[1, 2], [3, 4]]);
const det = matrix.det();
const inv = matrix.inv();
const transpose = matrix.transpose();
```

### Statistics

```javascript
// Basic statistics
const data = WasmArray.arange(1, 101, 1);
const mean = data.mean();
const std = data.std();
const min = data.min();
const max = data.max();

// Random distributions
const normal = WasmArray.randn([1000]);
const uniform = WasmArray.random([1000]);
```

## Performance Considerations

### Bundle Size

The WASM binary size depends on the build configuration:

- **Development**: ~2-3 MB (unoptimized, with debug symbols)
- **Release**: ~500 KB - 1 MB (optimized, stripped)
- **Release + gzip**: ~200-400 KB (served with compression)

To minimize bundle size:

```bash
# Use release build
wasm-pack build --target web --features wasm --release

# Enable link-time optimization (LTO) in Cargo.toml
# [profile.release]
# lto = true
# opt-level = 'z'  # Optimize for size
```

### SIMD Acceleration

NumRS2 supports WebAssembly SIMD when available:

```javascript
import { is_simd_available } from './pkg/numrs2.js';

if (is_simd_available()) {
    console.log('SIMD acceleration enabled!');
} else {
    console.log('SIMD not available - using scalar fallback');
}
```

### Memory Management

WASM memory is managed automatically, but for large arrays:

```javascript
// Arrays are automatically freed when they go out of scope
{
    const large = WasmArray.zeros([1000, 1000]);
    // ... use large array
} // Memory freed here

// Explicit cleanup (if needed)
const arr = WasmArray.zeros([100, 100]);
// ... use arr
arr.free();  // Explicit cleanup
```

## Deployment

### Building for Production

1. Build the optimized WASM module:
   ```bash
   wasm-pack build --target web --features wasm --release --out-dir examples/wasm/pkg
   ```

2. Build the web assets:
   ```bash
   npm run preview
   ```

3. Deploy the `dist/` directory to your web server.

### Serving Requirements

Your web server must:

1. **Serve WASM with correct MIME type**:
   ```
   Content-Type: application/wasm
   ```

2. **Enable CORS headers** (if loading from different origin):
   ```
   Access-Control-Allow-Origin: *
   ```

3. **Enable compression** (gzip or brotli) for smaller transfers

### CDN Deployment

You can host the WASM module on a CDN:

```javascript
import init from 'https://cdn.example.com/numrs2/pkg/numrs2.js';

// Initialize with custom WASM URL
await init('https://cdn.example.com/numrs2/pkg/numrs2_bg.wasm');
```

## Troubleshooting

### Common Issues

#### 1. WASM Module Not Found

**Error**: `Cannot find module './pkg/numrs2.js'`

**Solution**: Build the WASM module first:
```bash
wasm-pack build --target web --features wasm --out-dir examples/wasm/pkg
```

#### 2. MIME Type Error

**Error**: `Failed to execute 'compile' on 'WebAssembly': Incorrect response MIME type`

**Solution**: Configure your server to serve `.wasm` files with `application/wasm` MIME type.

For Python's http.server:
```bash
# The serve script handles this automatically
npm run serve
```

#### 3. SharedArrayBuffer Not Available

**Error**: `SharedArrayBuffer is not defined`

**Solution**: Ensure your server sends the required headers (handled by Vite config):
```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

#### 4. Out of Memory

**Error**: `RuntimeError: memory access out of bounds`

**Solution**:
- Reduce array sizes
- Use streaming operations for large data
- Check for memory leaks

#### 5. Performance Issues

**Symptoms**: Slow operations, browser freezing

**Solutions**:
- Use release build instead of development build
- Enable SIMD in browser settings (Chrome: `chrome://flags/#enable-webassembly-simd`)
- Use Web Workers for heavy computations
- Check browser console for warnings

### Browser Compatibility

| Browser | Version | WASM | SIMD |
|---------|---------|------|------|
| Chrome  | 57+     | ✅   | 91+  |
| Firefox | 52+     | ✅   | 89+  |
| Safari  | 11+     | ✅   | 16.4+|
| Edge    | 79+     | ✅   | 91+  |

### Debugging

Enable verbose logging:

```javascript
// In app.js, add:
console.log('WASM initialized:', await init());

// Check WASM memory usage
performance.memory && console.log('Memory:', performance.memory);
```

## Advanced Usage

### Using with Web Workers

```javascript
// worker.js
importScripts('./pkg/numrs2.js');

self.addEventListener('message', async (e) => {
    const { init, WasmArray } = wasm_bindgen;
    await init('./pkg/numrs2_bg.wasm');

    const arr = WasmArray.zeros(e.data.shape);
    self.postMessage({ result: arr.to_array() });
});

// main.js
const worker = new Worker('worker.js');
worker.postMessage({ shape: [1000, 1000] });
```

### Custom Memory Allocator

NumRS2 uses `wee_alloc` for smaller WASM binary size. You can customize this in `Cargo.toml`.

## Documentation

- [NumRS2 Documentation](https://docs.rs/numrs2)
- [WASM Guide](../../docs/WASM_GUIDE.md)
- [API Reference](https://docs.rs/numrs2/latest/numrs2/wasm/)
- [GitHub Repository](https://github.com/cool-japan/numrs)

## Support

For questions and support:

- Open an issue on [GitHub](https://github.com/cool-japan/numrs/issues)
- Check existing [discussions](https://github.com/cool-japan/numrs/discussions)

## License

NumRS2 is licensed under the Apache License 2.0.

Copyright (c) 2025 COOLJAPAN OU (Team KitaSan)

---

**NumRS2** - Pure Rust Numerical Computing | Part of the COOLJAPAN Ecosystem
