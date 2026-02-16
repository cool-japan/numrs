"""
Tests for NumRS2 linear algebra Python bindings
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


def test_matmul():
    """Test matrix multiplication"""
    a = nr.array([1.0, 2.0, 3.0, 4.0])
    a_mat = a.reshape([2, 2])

    b = nr.array([5.0, 6.0, 7.0, 8.0])
    b_mat = b.reshape([2, 2])

    c = nr.matmul(a_mat, b_mat)

    assert c.shape == [2, 2]
    # Result should be [[19, 22], [43, 50]]
    result = c.tolist()
    assert len(result) == 4


def test_linalg_det():
    """Test determinant calculation"""
    # 2x2 identity matrix has det = 1
    a = nr.eye(2)
    det = nr.linalg.det(a)
    assert abs(det - 1.0) < 1e-10


def test_linalg_trace():
    """Test trace calculation"""
    a = nr.array([1.0, 2.0, 3.0, 4.0]).reshape([2, 2])
    trace = nr.linalg.trace(a)
    # Trace = 1 + 4 = 5
    assert abs(trace - 5.0) < 1e-10


def test_linalg_inv():
    """Test matrix inversion"""
    # Simple 2x2 matrix
    a = nr.array([4.0, 7.0, 2.0, 6.0]).reshape([2, 2])
    a_inv = nr.linalg.inv(a)

    # Check A * A^-1 = I
    identity = nr.matmul(a, a_inv)
    result = identity.tolist()

    # Diagonal elements should be close to 1
    assert abs(result[0] - 1.0) < 1e-9
    assert abs(result[3] - 1.0) < 1e-9


def test_linalg_eigvals():
    """Test eigenvalue calculation"""
    # 2x2 identity matrix eigenvalues are [1, 1]
    a = nr.eye(2)
    eigvals = nr.linalg.eigvals(a)

    assert eigvals.size == 2
    values = eigvals.tolist()
    # Both eigenvalues should be 1
    for val in values:
        assert abs(val - 1.0) < 1e-10


def test_linalg_eig():
    """Test eigendecomposition"""
    a = nr.eye(2)
    vals, vecs = nr.linalg.eig(a)

    assert vals.size == 2
    assert vecs.shape == [2, 2]


def test_linalg_svd():
    """Test singular value decomposition"""
    a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([2, 3])
    u, s, vt = nr.linalg.svd(a)

    assert u.ndim == 2
    assert s.ndim == 1
    assert vt.ndim == 2


def test_linalg_qr():
    """Test QR decomposition"""
    a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([2, 3])
    q, r = nr.linalg.qr(a)

    assert q.ndim == 2
    assert r.ndim == 2


def test_linalg_norm():
    """Test matrix norm"""
    a = nr.array([3.0, 4.0])
    norm = nr.linalg.norm(a)

    # L2 norm of [3, 4] is 5
    assert abs(norm - 5.0) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
