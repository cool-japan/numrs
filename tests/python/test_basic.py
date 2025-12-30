"""
Basic tests for NumRS2 Python bindings
"""

import pytest
import numpy as np

try:
    import numrs2 as nr
    NUMRS2_AVAILABLE = True
except ImportError:
    NUMRS2_AVAILABLE = False
    nr = None

pytestmark = pytest.mark.skipif(not NUMRS2_AVAILABLE, reason="numrs2 not built with Python bindings")


def test_import():
    """Test that numrs2 can be imported"""
    assert nr is not None
    assert hasattr(nr, "__version__")


def test_array_creation_from_list():
    """Test creating array from Python list"""
    arr = nr.array([1.0, 2.0, 3.0, 4.0])
    assert arr.size == 4
    assert arr.ndim == 1
    assert arr.shape == [4]


def test_array_creation_from_numpy():
    """Test creating array from NumPy array"""
    np_arr = np.array([1.0, 2.0, 3.0, 4.0])
    nr_arr = nr.array(np_arr)
    assert nr_arr.size == 4
    assert list(nr_arr.tolist()) == [1.0, 2.0, 3.0, 4.0]


def test_zeros():
    """Test zeros creation"""
    arr = nr.zeros([2, 3])
    assert arr.shape == [2, 3]
    assert arr.size == 6
    assert all(x == 0.0 for x in arr.tolist())


def test_ones():
    """Test ones creation"""
    arr = nr.ones([2, 3])
    assert arr.shape == [2, 3]
    assert arr.size == 6
    assert all(x == 1.0 for x in arr.tolist())


def test_eye():
    """Test identity matrix creation"""
    arr = nr.eye(3)
    assert arr.shape == [3, 3]
    assert arr.size == 9

    # Convert to list and check diagonal
    data = arr.tolist()
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            assert data[i * 3 + j] == expected


def test_linspace():
    """Test linspace"""
    arr = nr.linspace(0.0, 1.0, 11)
    assert arr.size == 11
    assert arr.tolist()[0] == 0.0
    assert arr.tolist()[-1] == 1.0


def test_arange():
    """Test arange"""
    arr = nr.arange(0.0, 10.0, 2.0)
    assert list(arr.tolist()) == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_reshape():
    """Test reshape operation"""
    arr = nr.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    reshaped = arr.reshape([2, 3])
    assert reshaped.shape == [2, 3]
    assert reshaped.size == 6


def test_transpose():
    """Test transpose operation"""
    arr = nr.zeros([2, 3])
    transposed = arr.transpose()
    assert transposed.shape == [3, 2]


def test_addition():
    """Test element-wise addition"""
    a = nr.array([1.0, 2.0, 3.0])
    b = nr.array([4.0, 5.0, 6.0])
    c = a + b
    assert list(c.tolist()) == [5.0, 7.0, 9.0]


def test_subtraction():
    """Test element-wise subtraction"""
    a = nr.array([4.0, 5.0, 6.0])
    b = nr.array([1.0, 2.0, 3.0])
    c = a - b
    assert list(c.tolist()) == [3.0, 3.0, 3.0]


def test_multiplication():
    """Test element-wise multiplication"""
    a = nr.array([2.0, 3.0, 4.0])
    b = nr.array([5.0, 6.0, 7.0])
    c = a * b
    assert list(c.tolist()) == [10.0, 18.0, 28.0]


def test_division():
    """Test element-wise division"""
    a = nr.array([10.0, 20.0, 30.0])
    b = nr.array([2.0, 4.0, 5.0])
    c = a / b
    assert list(c.tolist()) == [5.0, 5.0, 6.0]


def test_negation():
    """Test negation"""
    a = nr.array([1.0, -2.0, 3.0])
    b = -a
    assert list(b.tolist()) == [-1.0, 2.0, -3.0]


def test_dot_product():
    """Test dot product"""
    a = nr.array([1.0, 2.0, 3.0])
    b = nr.array([4.0, 5.0, 6.0])
    result = nr.dot(a, b)
    assert result == 32.0  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32


def test_to_numpy():
    """Test conversion to NumPy"""
    nr_arr = nr.array([1.0, 2.0, 3.0, 4.0])
    np_arr = nr_arr.to_numpy(None)

    assert isinstance(np_arr, np.ndarray)
    assert list(np_arr) == [1.0, 2.0, 3.0, 4.0]


def test_numpy_roundtrip():
    """Test NumPy roundtrip conversion"""
    original = np.array([1.0, 2.0, 3.0, 4.0])
    nr_arr = nr.array(original)
    back_to_numpy = np.array(nr_arr.tolist())

    assert np.allclose(original, back_to_numpy)


def test_repr():
    """Test string representation"""
    arr = nr.array([1.0, 2.0, 3.0])
    repr_str = repr(arr)
    assert "Array" in repr_str
    assert "shape" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
