"""
Tests for NumRS2 I/O Python bindings
"""

import pytest
import tempfile
import os

try:
    import numrs2 as nr
    NUMRS2_AVAILABLE = True
except ImportError:
    NUMRS2_AVAILABLE = False
    nr = None

pytestmark = pytest.mark.skipif(
    not NUMRS2_AVAILABLE, reason="numrs2 not built with Python bindings"
)


def test_io_npy_save_load():
    """Test NPY save/load roundtrip"""
    a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.npy")

        # Save
        nr.io.save_npy(filepath, a)
        assert os.path.exists(filepath)

        # Load
        b = nr.io.load_npy(filepath)
        assert b.size == a.size

        # Check values match
        a_list = a.tolist()
        b_list = b.tolist()
        for i in range(len(a_list)):
            assert abs(a_list[i] - b_list[i]) < 1e-10


def test_io_csv_save_load():
    """Test CSV save/load roundtrip"""
    a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.csv")

        # Save
        nr.io.save_csv(filepath, a)
        assert os.path.exists(filepath)

        # Load
        b = nr.io.load_csv(filepath)
        assert b.size == a.size


def test_io_json_save_load():
    """Test JSON save/load roundtrip"""
    a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0])

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.json")

        # Save
        nr.io.save_json(filepath, a)
        assert os.path.exists(filepath)

        # Load
        b = nr.io.load_json(filepath)
        assert b.size == a.size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
