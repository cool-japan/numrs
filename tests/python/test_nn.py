"""
Tests for NumRS2 neural network Python bindings
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


def test_nn_relu():
    """Test ReLU activation"""
    x = nr.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = nr.nn.relu(x)
    result = y.tolist()

    # ReLU: max(0, x)
    expected = [0.0, 0.0, 0.0, 1.0, 2.0]
    for i, val in enumerate(result):
        assert abs(val - expected[i]) < 1e-10


def test_nn_sigmoid():
    """Test sigmoid activation"""
    x = nr.array([0.0])
    y = nr.nn.sigmoid(x)
    result = y.tolist()

    # sigmoid(0) = 0.5
    assert abs(result[0] - 0.5) < 1e-10


def test_nn_tanh():
    """Test tanh activation"""
    x = nr.array([0.0])
    y = nr.nn.tanh(x)
    result = y.tolist()

    # tanh(0) = 0
    assert abs(result[0] - 0.0) < 1e-10


def test_nn_softmax():
    """Test softmax activation"""
    x = nr.array([1.0, 2.0, 3.0])
    y = nr.nn.softmax(x)
    result = y.tolist()

    # Softmax output should sum to 1
    total = sum(result)
    assert abs(total - 1.0) < 1e-10


def test_nn_mse_loss():
    """Test MSE loss"""
    predictions = nr.array([1.0, 2.0, 3.0])
    targets = nr.array([1.0, 2.0, 3.0])
    loss = nr.nn.mse_loss(predictions, targets)

    # Perfect predictions should have 0 loss
    assert abs(loss - 0.0) < 1e-10


def test_nn_dropout():
    """Test dropout"""
    x = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = nr.nn.dropout(x, 0.5)

    # Output should have same shape
    assert y.size == x.size


def test_nn_batch_norm():
    """Test batch normalization"""
    x = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = nr.nn.batch_norm(x)

    # Normalized output should have mean ≈ 0
    mean = y.mean()
    assert abs(mean) < 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
