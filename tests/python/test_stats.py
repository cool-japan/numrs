"""
Tests for NumRS2 statistics Python bindings
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


def test_stats_mean():
    """Test mean calculation"""
    a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mean = nr.stats.mean(a)
    assert abs(mean - 3.0) < 1e-10


def test_stats_median():
    """Test median calculation"""
    a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
    median = nr.stats.median(a)
    assert abs(median - 3.0) < 1e-10


def test_stats_median_even():
    """Test median with even number of elements"""
    a = nr.array([1.0, 2.0, 3.0, 4.0])
    median = nr.stats.median(a)
    assert abs(median - 2.5) < 1e-10


def test_stats_std():
    """Test standard deviation"""
    a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
    std = nr.stats.std(a)
    # Population std of [1,2,3,4,5] ≈ 1.414
    assert std > 1.0 and std < 2.0


def test_stats_var():
    """Test variance"""
    a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
    var = nr.stats.var(a)
    # Population variance of [1,2,3,4,5] = 2
    assert abs(var - 2.0) < 1e-10


def test_stats_corrcoef():
    """Test correlation coefficient"""
    x = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = nr.array([2.0, 4.0, 6.0, 8.0, 10.0])

    corr_matrix = nr.stats.corrcoef(x, y)
    # Perfect correlation should give 1.0
    result = corr_matrix.tolist()
    # Off-diagonal elements should be 1.0 (perfect correlation)
    assert abs(result[1] - 1.0) < 1e-10


def test_stats_histogram():
    """Test histogram"""
    a = nr.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0])
    counts, edges = nr.stats.histogram(a, bins=5)

    assert counts.size == 5
    assert edges.size == 6  # n_bins + 1 edges


def test_stats_percentile():
    """Test percentile"""
    a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    p50 = nr.stats.percentile(a, 50.0)
    # 50th percentile (median) should be 5.5
    assert abs(p50 - 5.5) < 1e-10


def test_random_randn():
    """Test random normal distribution"""
    a = nr.random.randn([100])
    assert a.size == 100
    # Mean should be close to 0
    mean = a.mean()
    assert abs(mean) < 0.5


def test_random_rand():
    """Test random uniform distribution"""
    a = nr.random.rand([100])
    assert a.size == 100
    # All values should be in [0, 1]
    values = a.tolist()
    assert all(0.0 <= v <= 1.0 for v in values)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
