"""
NumRS2: High-Performance Numerical Computing for Python

A Rust-powered numerical computing library with a NumPy-compatible interface,
providing blazing-fast performance for scientific computing tasks.

Key Features:
- NumPy-compatible API for seamless migration
- Zero-copy data sharing with NumPy arrays
- High-performance Rust implementations
- SIMD and parallel processing support
- GPU acceleration (optional)

Example:
    >>> import numrs2 as nr
    >>> import numpy as np
    >>>
    >>> # Create arrays
    >>> a = nr.array([1.0, 2.0, 3.0, 4.0])
    >>> b = nr.zeros([2, 3])
    >>>
    >>> # Operations
    >>> c = a * 2.0
    >>> d = nr.matmul(b, b.transpose())
    >>>
    >>> # Convert to NumPy (zero-copy when possible)
    >>> numpy_arr = np.array(c)
"""

from ._numrs2 import (
    __version__,
    Array,
    array,
    zeros,
    ones,
    eye,
    linspace,
    arange,
    matmul,
    dot,
)

__all__ = [
    "__version__",
    "Array",
    "array",
    "zeros",
    "ones",
    "eye",
    "linspace",
    "arange",
    "matmul",
    "dot",
]


def __dir__():
    return __all__
