/**
 * NumRS2 WebAssembly Demo Application
 *
 * This file demonstrates how to use NumRS2 in a web browser environment.
 * It provides examples of array operations, linear algebra, and statistics.
 */

// Import NumRS2 WASM module
// Note: The path will be './pkg/numrs2.js' after building with wasm-pack
import init, {
    get_version,
    is_simd_available,
    WasmArray
} from './pkg/numrs2.js';

let wasmInitialized = false;

/**
 * Initialize the WASM module
 */
async function initializeWasm() {
    try {
        await init();
        wasmInitialized = true;

        // Hide loading screen
        document.getElementById('loading').style.display = 'none';
        document.getElementById('main-content').style.display = 'block';

        // Display version and SIMD info
        try {
            const version = get_version();
            const simdAvailable = is_simd_available();

            document.getElementById('version-info').textContent = ` | Version: ${version}`;
            document.getElementById('simd-info').textContent = ` | SIMD: ${simdAvailable ? 'Enabled ⚡' : 'Disabled'}`;
        } catch (error) {
            console.error('Error getting version info:', error);
        }

        console.log('NumRS2 WASM module initialized successfully');
    } catch (error) {
        console.error('Failed to initialize WASM module:', error);
        document.getElementById('loading').innerHTML = `
            <div class="output error">
                Failed to load NumRS2 WASM module: ${error.message}
                <br><br>
                Please ensure you have built the WASM module using:
                <br>
                <code>wasm-pack build --target web --features wasm</code>
            </div>
        `;
    }
}

/**
 * Utility function to display output
 */
function displayOutput(elementId, content, isError = false) {
    const element = document.getElementById(elementId);
    if (!element) return;

    element.textContent = content;
    element.className = 'output';
    if (isError) {
        element.classList.add('error');
    } else {
        element.classList.add('success');
    }
}

/**
 * Utility function to check WASM initialization
 */
function checkWasmReady() {
    if (!wasmInitialized) {
        alert('WASM module is still loading. Please wait...');
        return false;
    }
    return true;
}

/**
 * Format array for display
 */
function formatArray(array, name = 'Array') {
    try {
        return `${name}:\n${array.toString()}`;
    } catch (error) {
        return `Error formatting array: ${error.message}`;
    }
}

/**
 * Measure execution time
 */
function measureTime(fn) {
    const start = performance.now();
    const result = fn();
    const end = performance.now();
    return { result, time: (end - start).toFixed(3) };
}

// ============================================================================
// Array Operations
// ============================================================================

/**
 * Create a zeros array
 */
window.createZerosArray = function() {
    if (!checkWasmReady()) return;

    try {
        const { result: arr, time } = measureTime(() =>
            WasmArray.zeros([3, 4])
        );

        displayOutput('array-output',
            `Created 3×4 zeros array in ${time}ms:\n\n${formatArray(arr)}`
        );
    } catch (error) {
        displayOutput('array-output',
            `Error creating zeros array: ${error.message}`,
            true
        );
    }
};

/**
 * Create a ones array
 */
window.createOnesArray = function() {
    if (!checkWasmReady()) return;

    try {
        const { result: arr, time } = measureTime(() =>
            WasmArray.ones([2, 5])
        );

        displayOutput('array-output',
            `Created 2×5 ones array in ${time}ms:\n\n${formatArray(arr)}`
        );
    } catch (error) {
        displayOutput('array-output',
            `Error creating ones array: ${error.message}`,
            true
        );
    }
};

/**
 * Create an arange array
 */
window.createArangeArray = function() {
    if (!checkWasmReady()) return;

    try {
        const { result: arr, time } = measureTime(() =>
            WasmArray.arange(0, 12, 1).reshape([3, 4])
        );

        displayOutput('array-output',
            `Created arange(0, 12) reshaped to 3×4 in ${time}ms:\n\n${formatArray(arr)}`
        );
    } catch (error) {
        displayOutput('array-output',
            `Error creating arange array: ${error.message}`,
            true
        );
    }
};

/**
 * Create a random array
 */
window.createRandomArray = function() {
    if (!checkWasmReady()) return;

    try {
        const { result: arr, time } = measureTime(() =>
            WasmArray.random([3, 3])
        );

        displayOutput('array-output',
            `Created 3×3 random array in ${time}ms:\n\n${formatArray(arr)}`
        );
    } catch (error) {
        displayOutput('array-output',
            `Error creating random array: ${error.message}`,
            true
        );
    }
};

/**
 * Array addition example
 */
window.arrayAddition = function() {
    if (!checkWasmReady()) return;

    try {
        const { result, time } = measureTime(() => {
            const a = WasmArray.arange(0, 6, 1).reshape([2, 3]);
            const b = WasmArray.ones([2, 3]);
            return a.add(b);
        });

        displayOutput('arithmetic-output',
            `Array addition completed in ${time}ms:\n\n${formatArray(result, 'Result')}`
        );
    } catch (error) {
        displayOutput('arithmetic-output',
            `Error in array addition: ${error.message}`,
            true
        );
    }
};

/**
 * Array multiplication example
 */
window.arrayMultiplication = function() {
    if (!checkWasmReady()) return;

    try {
        const { result, time } = measureTime(() => {
            const a = WasmArray.arange(1, 7, 1).reshape([2, 3]);
            const b = WasmArray.fill([2, 3], 2.0);
            return a.multiply(b);
        });

        displayOutput('arithmetic-output',
            `Element-wise multiplication completed in ${time}ms:\n\n${formatArray(result, 'Result')}`
        );
    } catch (error) {
        displayOutput('arithmetic-output',
            `Error in array multiplication: ${error.message}`,
            true
        );
    }
};

/**
 * Broadcasting example
 */
window.arrayBroadcasting = function() {
    if (!checkWasmReady()) return;

    try {
        const { result, time } = measureTime(() => {
            const a = WasmArray.arange(0, 12, 1).reshape([4, 3]);
            return a.add_scalar(10.0);
        });

        displayOutput('arithmetic-output',
            `Broadcasting (add 10 to all elements) completed in ${time}ms:\n\n${formatArray(result, 'Result')}`
        );
    } catch (error) {
        displayOutput('arithmetic-output',
            `Error in broadcasting: ${error.message}`,
            true
        );
    }
};

// ============================================================================
// Linear Algebra Operations
// ============================================================================

/**
 * Matrix multiplication
 */
window.matrixMultiply = function() {
    if (!checkWasmReady()) return;

    try {
        const { result, time } = measureTime(() => {
            const a = WasmArray.arange(0, 6, 1).reshape([2, 3]);
            const b = WasmArray.arange(0, 6, 1).reshape([3, 2]);
            return a.matmul(b);
        });

        displayOutput('linalg-output',
            `Matrix multiplication (2×3 @ 3×2) completed in ${time}ms:\n\n${formatArray(result, 'Result')}`
        );
    } catch (error) {
        displayOutput('linalg-output',
            `Error in matrix multiplication: ${error.message}`,
            true
        );
    }
};

/**
 * Matrix transpose
 */
window.matrixTranspose = function() {
    if (!checkWasmReady()) return;

    try {
        const { result, time } = measureTime(() => {
            const a = WasmArray.arange(0, 12, 1).reshape([3, 4]);
            return a.transpose();
        });

        displayOutput('linalg-output',
            `Matrix transpose (3×4 → 4×3) completed in ${time}ms:\n\n${formatArray(result, 'Result')}`
        );
    } catch (error) {
        displayOutput('linalg-output',
            `Error in matrix transpose: ${error.message}`,
            true
        );
    }
};

/**
 * Matrix determinant
 */
window.matrixDeterminant = function() {
    if (!checkWasmReady()) return;

    try {
        const { result, time } = measureTime(() => {
            const a = WasmArray.from_array([[1, 2], [3, 4]]);
            return a.det();
        });

        displayOutput('linalg-output',
            `Determinant of [[1,2],[3,4]] computed in ${time}ms:\n\ndet = ${result.toFixed(6)}`
        );
    } catch (error) {
        displayOutput('linalg-output',
            `Error computing determinant: ${error.message}`,
            true
        );
    }
};

/**
 * Matrix inverse
 */
window.matrixInverse = function() {
    if (!checkWasmReady()) return;

    try {
        const { result, time } = measureTime(() => {
            const a = WasmArray.from_array([[4, 7], [2, 6]]);
            return a.inv();
        });

        displayOutput('linalg-output',
            `Matrix inverse of [[4,7],[2,6]] computed in ${time}ms:\n\n${formatArray(result, 'Inverse')}`
        );
    } catch (error) {
        displayOutput('linalg-output',
            `Error computing matrix inverse: ${error.message}`,
            true
        );
    }
};

// ============================================================================
// Statistics Operations
// ============================================================================

/**
 * Compute basic statistics
 */
window.computeStats = function() {
    if (!checkWasmReady()) return;

    try {
        const { result, time } = measureTime(() => {
            const arr = WasmArray.arange(1, 101, 1);
            return {
                mean: arr.mean(),
                std: arr.std(),
                min: arr.min(),
                max: arr.max()
            };
        });

        displayOutput('stats-output',
            `Statistics for array [1, 2, ..., 100] computed in ${time}ms:\n\n` +
            `Mean: ${result.mean.toFixed(2)}\n` +
            `Std Dev: ${result.std.toFixed(2)}\n` +
            `Min: ${result.min.toFixed(2)}\n` +
            `Max: ${result.max.toFixed(2)}`
        );
    } catch (error) {
        displayOutput('stats-output',
            `Error computing statistics: ${error.message}`,
            true
        );
    }
};

/**
 * Compute correlation matrix
 */
window.computeCorrelation = function() {
    if (!checkWasmReady()) return;

    try {
        const { result, time } = measureTime(() => {
            const data = WasmArray.random([100, 3]);
            return data.corrcoef();
        });

        displayOutput('stats-output',
            `Correlation matrix for 100×3 random data computed in ${time}ms:\n\n${formatArray(result, 'Correlation')}`
        );
    } catch (error) {
        displayOutput('stats-output',
            `Error computing correlation: ${error.message}`,
            true
        );
    }
};

/**
 * Generate random distribution
 */
window.generateDistribution = function() {
    if (!checkWasmReady()) return;

    try {
        const { result, time } = measureTime(() => {
            return WasmArray.randn([1000]);
        });

        const stats = {
            mean: result.mean(),
            std: result.std()
        };

        displayOutput('stats-output',
            `Generated 1000 samples from normal distribution in ${time}ms:\n\n` +
            `Mean: ${stats.mean.toFixed(4)} (expected: 0.0)\n` +
            `Std Dev: ${stats.std.toFixed(4)} (expected: 1.0)\n\n` +
            `First 10 samples:\n${result.slice([0], [10]).toString()}`
        );
    } catch (error) {
        displayOutput('stats-output',
            `Error generating distribution: ${error.message}`,
            true
        );
    }
};

// ============================================================================
// Performance Benchmarks
// ============================================================================

/**
 * Benchmark array operations
 */
window.benchmarkArrayOps = function() {
    if (!checkWasmReady()) return;

    try {
        const size = 1000;
        const iterations = 100;

        // WASM benchmark
        const wasmTime = measureTime(() => {
            for (let i = 0; i < iterations; i++) {
                const a = WasmArray.random([size]);
                const b = WasmArray.random([size]);
                const c = a.add(b);
            }
        }).time;

        // JavaScript benchmark
        const jsTime = measureTime(() => {
            for (let i = 0; i < iterations; i++) {
                const a = Array.from({ length: size }, () => Math.random());
                const b = Array.from({ length: size }, () => Math.random());
                const c = a.map((val, idx) => val + b[idx]);
            }
        }).time;

        const speedup = (jsTime / wasmTime).toFixed(2);

        displayOutput('benchmark-output',
            `Array Addition Benchmark (${iterations} iterations, size ${size}):\n\n` +
            `NumRS2 WASM: ${wasmTime}ms\n` +
            `JavaScript: ${jsTime}ms\n\n` +
            `Speedup: ${speedup}x ${speedup > 1 ? '⚡' : ''}`
        );
    } catch (error) {
        displayOutput('benchmark-output',
            `Error in benchmark: ${error.message}`,
            true
        );
    }
};

/**
 * Benchmark matrix operations
 */
window.benchmarkMatrixOps = function() {
    if (!checkWasmReady()) return;

    try {
        const size = 50;
        const iterations = 50;

        // WASM benchmark
        const wasmTime = measureTime(() => {
            for (let i = 0; i < iterations; i++) {
                const a = WasmArray.random([size, size]);
                const b = WasmArray.random([size, size]);
                const c = a.matmul(b);
            }
        }).time;

        // Note: JavaScript implementation would be very slow for matrix multiplication
        // We'll just show WASM performance

        displayOutput('benchmark-output',
            `Matrix Multiplication Benchmark (${iterations} iterations, ${size}×${size}):\n\n` +
            `NumRS2 WASM: ${wasmTime}ms\n` +
            `Average per operation: ${(wasmTime / iterations).toFixed(2)}ms\n\n` +
            `Note: Native JavaScript matrix multiplication would be significantly slower ⚡`
        );
    } catch (error) {
        displayOutput('benchmark-output',
            `Error in benchmark: ${error.message}`,
            true
        );
    }
};

// Initialize WASM when page loads
initializeWasm();
