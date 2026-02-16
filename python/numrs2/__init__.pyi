"""
Type stubs for NumRS2 Python bindings

Provides type hints for IDE support and type checking.
"""

from typing import Any, Optional, Sequence, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike, NDArray

__version__: str

class Array:
    """NumRS2 Array class - High-performance n-dimensional array"""

    def __init__(self, data: Union[Sequence[float], NDArray[np.floating], "Array"]) -> None:
        """Create a new array from data"""
        ...

    @property
    def shape(self) -> list[int]:
        """Shape of the array"""
        ...

    @property
    def ndim(self) -> int:
        """Number of dimensions"""
        ...

    @property
    def size(self) -> int:
        """Total number of elements"""
        ...

    @property
    def dtype(self) -> str:
        """Data type of the array"""
        ...

    def reshape(self, shape: Sequence[int]) -> "Array":
        """Reshape array to new shape"""
        ...

    def transpose(self) -> "Array":
        """Transpose array"""
        ...

    def flatten(self) -> "Array":
        """Flatten array to 1D"""
        ...

    def squeeze(self) -> "Array":
        """Remove dimensions of size 1"""
        ...

    def tolist(self) -> list[float]:
        """Convert to Python list"""
        ...

    def to_numpy(self, py: Any) -> NDArray[np.float64]:
        """Convert to NumPy array"""
        ...

    def copy(self) -> "Array":
        """Create a copy"""
        ...

    def sum(self) -> float:
        """Sum of all elements"""
        ...

    def mean(self) -> float:
        """Mean of all elements"""
        ...

    def min(self) -> float:
        """Minimum value"""
        ...

    def max(self) -> float:
        """Maximum value"""
        ...

    def __add__(self, other: "Array") -> "Array": ...
    def __sub__(self, other: "Array") -> "Array": ...
    def __mul__(self, other: "Array") -> "Array": ...
    def __truediv__(self, other: "Array") -> "Array": ...
    def __neg__(self) -> "Array": ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

# Array creation functions
def array(data: ArrayLike) -> Array:
    """Create an array from data"""
    ...

def zeros(shape: Sequence[int]) -> Array:
    """Create an array of zeros"""
    ...

def ones(shape: Sequence[int]) -> Array:
    """Create an array of ones"""
    ...

def eye(n: int, m: Optional[int] = None, k: Optional[int] = None) -> Array:
    """Create a 2D array with ones on the diagonal"""
    ...

def identity(n: int) -> Array:
    """Create an identity matrix"""
    ...

def linspace(
    start: float, stop: float, num: int, endpoint: Optional[bool] = True
) -> Array:
    """Create an array with evenly spaced values"""
    ...

def arange(start: float, stop: float, step: Optional[float] = None) -> Array:
    """Create an array with values in a range"""
    ...

def full(shape: Sequence[int], fill_value: float) -> Array:
    """Create an array filled with a constant value"""
    ...

def zeros_like(a: Array) -> Array:
    """Create an array of zeros with the same shape as a"""
    ...

def ones_like(a: Array) -> Array:
    """Create an array of ones with the same shape as a"""
    ...

def concatenate(arrays: Sequence[Array], axis: Optional[int] = None) -> Array:
    """Concatenate arrays"""
    ...

# Top-level linear algebra
def matmul(a: Array, b: Array) -> Array:
    """Matrix multiplication"""
    ...

def dot(a: Array, b: Array) -> float:
    """Dot product"""
    ...

# Linear algebra submodule
class linalg:
    """Linear algebra operations"""

    @staticmethod
    def matmul(a: Array, b: Array) -> Array: ...
    @staticmethod
    def dot(a: Array, b: Array) -> float: ...
    @staticmethod
    def matvec(a: Array, b: Array) -> Array: ...
    @staticmethod
    def det(a: Array) -> float: ...
    @staticmethod
    def trace(a: Array) -> float: ...
    @staticmethod
    def inv(a: Array) -> Array: ...
    @staticmethod
    def solve(a: Array, b: Array) -> Array: ...
    @staticmethod
    def eigvals(a: Array) -> Array: ...
    @staticmethod
    def eig(a: Array) -> Tuple[Array, Array]: ...
    @staticmethod
    def svd(a: Array, full_matrices: Optional[bool] = True) -> Tuple[Array, Array, Array]: ...
    @staticmethod
    def qr(a: Array) -> Tuple[Array, Array]: ...
    @staticmethod
    def cholesky(a: Array) -> Array: ...
    @staticmethod
    def lu(a: Array) -> Tuple[Array, Array, Array]: ...
    @staticmethod
    def norm(a: Array, ord: Optional[str] = None) -> float: ...
    @staticmethod
    def cond(a: Array) -> float: ...
    @staticmethod
    def matrix_rank(a: Array, tol: Optional[float] = None) -> int: ...

# Statistics submodule
class stats:
    """Statistical operations"""

    @staticmethod
    def mean(a: Array, axis: Optional[int] = None) -> float: ...
    @staticmethod
    def median(a: Array, axis: Optional[int] = None) -> float: ...
    @staticmethod
    def std(a: Array, axis: Optional[int] = None, ddof: Optional[int] = None) -> float: ...
    @staticmethod
    def var(a: Array, axis: Optional[int] = None, ddof: Optional[int] = None) -> float: ...
    @staticmethod
    def corrcoef(x: Array, y: Optional[Array] = None) -> Array: ...
    @staticmethod
    def cov(m: Array, rowvar: Optional[bool] = True) -> Array: ...
    @staticmethod
    def histogram(
        a: Array, bins: Optional[int] = None, range: Optional[Tuple[float, float]] = None
    ) -> Tuple[Array, Array]: ...
    @staticmethod
    def percentile(a: Array, q: float) -> float: ...

# Random submodule
class random:
    """Random number generation"""

    @staticmethod
    def randn(size: Sequence[int]) -> Array: ...
    @staticmethod
    def rand(size: Sequence[int]) -> Array: ...

# Optimization submodule
class optimize:
    """Optimization algorithms"""

    @staticmethod
    def minimize(
        fun: Any, x0: Array, method: Optional[str] = None, tol: Optional[float] = None
    ) -> Any: ...
    @staticmethod
    def root_scalar(fun: Any, bracket: Tuple[float, float], method: Optional[str] = None) -> float: ...

# Neural network submodule
class nn:
    """Neural network primitives"""

    @staticmethod
    def relu(x: Array) -> Array: ...
    @staticmethod
    def sigmoid(x: Array) -> Array: ...
    @staticmethod
    def tanh(x: Array) -> Array: ...
    @staticmethod
    def softmax(x: Array, axis: Optional[int] = None) -> Array: ...
    @staticmethod
    def mse_loss(predictions: Array, targets: Array) -> float: ...
    @staticmethod
    def cross_entropy_loss(predictions: Array, targets: Array) -> float: ...
    @staticmethod
    def dropout(x: Array, p: float) -> Array: ...
    @staticmethod
    def batch_norm(x: Array, eps: Optional[float] = None) -> Array: ...

# Symbolic submodule
class symbolic:
    """Symbolic computation"""

    @staticmethod
    def symbol(name: str) -> Any: ...
    @staticmethod
    def diff(expr: Any, var: Any) -> Any: ...
    @staticmethod
    def simplify(expr: Any) -> Any: ...

# I/O submodule
class io:
    """Data input/output"""

    @staticmethod
    def save_npy(file: str, arr: Array) -> None: ...
    @staticmethod
    def load_npy(file: str) -> Array: ...
    @staticmethod
    def save_csv(file: str, arr: Array) -> None: ...
    @staticmethod
    def load_csv(file: str) -> Array: ...
    @staticmethod
    def save_json(file: str, arr: Array) -> None: ...
    @staticmethod
    def load_json(file: str) -> Array: ...
