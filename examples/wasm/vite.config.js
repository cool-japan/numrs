/**
 * Vite configuration for NumRS2 WASM demo
 *
 * This configuration enables proper WASM module loading and top-level await
 * support for the NumRS2 WebAssembly demo application.
 */

import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';
import topLevelAwait from 'vite-plugin-top-level-await';

export default defineConfig({
  plugins: [
    // Enable WebAssembly support
    wasm(),
    // Enable top-level await (required for WASM initialization)
    topLevelAwait(),
  ],

  // Server configuration for development
  server: {
    port: 3000,
    open: true,
    headers: {
      // Required headers for WASM and SharedArrayBuffer
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },

  // Preview server configuration
  preview: {
    port: 3000,
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },

  // Build configuration
  build: {
    target: 'esnext',
    outDir: 'dist',
    assetsDir: 'assets',
    sourcemap: true,
    minify: 'esbuild',
    rollupOptions: {
      output: {
        // Optimize chunk splitting for better caching
        manualChunks: {
          'wasm-core': ['./pkg/numrs2.js'],
        },
      },
    },
  },

  // Optimize dependencies
  optimizeDeps: {
    exclude: ['./pkg/numrs2.js'],
  },

  // Worker configuration (if using Web Workers with WASM)
  worker: {
    format: 'es',
    plugins: [
      wasm(),
      topLevelAwait(),
    ],
  },
});
