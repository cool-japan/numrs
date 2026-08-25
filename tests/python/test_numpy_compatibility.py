"""
NumPy compatibility tests for NumRS2 Python bindings

Tests that NumRS2 can interoperate with NumPy arrays.
"""

import pytest
import numpy as np

try:
    import numrs2 as nr
    NUMRS2_AVAILABLE = True
except ImportError:
    NUMRS2_AVAILABLE = False
    nr = None

pytestmark = pytest.mark.skipif(
    not NUMRS2_AVAILABLE, reason="numrs2 not built with Python bindings"
)


def test_create_from_numpy_1d():
    """Test creating NumRS2 array from NumPy 1D array"""
    np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    nr_arr = nr.array(np_arr)

    assert nr_arr.size == np_arr.size
    assert nr_arr.ndim == np_arr.ndim


def test_create_from_numpy_2d():
    """Test creating NumRS2 array from NumPy 2D array"""
    np_arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    nr_arr = nr.array(np_arr)

    assert nr_arr.size == np_arr.size
    assert nr_arr.ndim == np_arr.ndim
    assert nr_arr.shape == list(np_arr.shape)


def test_convert_to_numpy():
    """Test converting NumRS2 array to NumPy"""
    nr_arr = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np_arr = nr_arr.to_numpy()

    assert isinstance(np_arr, np.ndarray)
    assert len(np_arr) == nr_arr.size


def test_numpy_roundtrip():
    """Test NumPy roundtrip conversion"""
    original = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    nr_arr = nr.array(original)
    back_to_numpy = nr_arr.to_numpy()

    assert np.allclose(original, back_to_numpy)


def test_operations_with_numpy():
    """Test that operations produce correct results compatible with NumPy"""
    np_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    nr_arr = nr.array(np_arr)

    # Test mean
    np_mean = np.mean(np_arr)
    nr_mean = nr_arr.mean()
    assert abs(np_mean - nr_mean) < 1e-10

    # Test sum
    np_sum = np.sum(np_arr)
    nr_sum = nr_arr.sum()
    assert abs(np_sum - nr_sum) < 1e-10


def test_zeros_like_numpy():
    """Test zeros creation matches NumPy"""
    np_zeros = np.zeros([3, 4])
    nr_zeros = nr.zeros([3, 4])

    assert nr_zeros.shape == list(np_zeros.shape)
    assert nr_zeros.size == np_zeros.size


def test_ones_like_numpy():
    """Test ones creation matches NumPy"""
    np_ones = np.ones([3, 4])
    nr_ones = nr.ones([3, 4])

    assert nr_ones.shape == list(np_ones.shape)
    assert nr_ones.size == np_ones.size


def test_linspace_like_numpy():
    """Test linspace matches NumPy"""
    np_lin = np.linspace(0, 1, 11)
    nr_lin = nr.linspace(0.0, 1.0, 11)

    assert nr_lin.size == len(np_lin)

    # Check endpoint values
    nr_list = nr_lin.tolist()
    assert abs(nr_list[0] - 0.0) < 1e-10
    assert abs(nr_list[-1] - 1.0) < 1e-10


def test_arange_like_numpy():
    """Test arange matches NumPy"""
    np_arr = np.arange(0, 10, 2)
    nr_arr = nr.arange(0.0, 10.0, 2.0)

    assert nr_arr.size == len(np_arr)


def test_eye_like_numpy():
    """Test eye matches NumPy"""
    np_eye = np.eye(3)
    nr_eye = nr.eye(3)

    assert nr_eye.shape == list(np_eye.shape)
    assert nr_eye.size == np_eye.size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
