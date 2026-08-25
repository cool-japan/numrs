"""
End-to-end smoke test for the NumRS2 Python bindings (Wave 5-C).

Exercises array creation, elementwise ops, matmul, the `random.Generator`
API, an fft/rfft/fftn/rfftn round trip, `stats.corrcoef`'s 2-D/`rowvar`
branches, `linalg.svd`'s NumPy-shaped output, and N-D support in `nn.*`
against the actual compiled `_numrs2` extension, comparing against NumPy
directly wherever NumPy provides the equivalent function. Self-skips
(rather than failing) when the extension has not been built into the
active Python environment, matching the convention already used by
`tests/python/*.py` in this repository.

Build + run (from the repository root, where both Cargo.toml and
pyproject.toml live -- maturin discovers pyproject.toml's `[tool.maturin]`
table from there automatically; `-m`/`--manifest-path` takes a *Cargo.toml*
path, not pyproject.toml, so it is omitted here):

    python3 -m venv /path/to/venv && source /path/to/venv/bin/activate
    pip install maturin numpy pytest
    maturin develop --features python
    pytest tests/python_smoke.py -v
"""

import pytest

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - numpy is a declared dependency
    NUMPY_AVAILABLE = False
    np = None

try:
    import numrs2 as nr

    NUMRS2_AVAILABLE = True
except ImportError:
    NUMRS2_AVAILABLE = False
    nr = None

pytestmark = pytest.mark.skipif(
    not NUMRS2_AVAILABLE or not NUMPY_AVAILABLE,
    reason="numrs2 extension not built/importable (run `maturin develop --features python` first)",
)


def test_array_create_and_add():
    a = nr.array([1.0, 2.0, 3.0, 4.0])
    b = nr.array([10.0, 20.0, 30.0, 40.0])
    assert a.shape == [4]
    assert a.ndim == 1
    assert a.size == 4

    c = a + b
    assert list(c.tolist()) == [11.0, 22.0, 33.0, 44.0]

    # NumPy round trip (copies once; see src/python/array.rs doc comments
    # for why NumRS2's Arc-based copy-on-write `Array` cannot safely alias
    # NumPy's buffer without a copy).
    np_arr = a.to_numpy()
    assert isinstance(np_arr, np.ndarray)
    assert np.allclose(np_arr, [1.0, 2.0, 3.0, 4.0])


def test_array_create_from_float32_numpy():
    # Regression coverage for the `PyArray::new` float32 branch: `Array` is
    # f64-only storage, so this is a widening-cast copy, not a zero-copy
    # view -- but it must not raise the misleading "Expected list, tuple,
    # or NumPy array" `TypeError` a `float32` input used to hit (it isn't a
    # list/tuple, and the old code only ever tried `PyReadonlyArrayDyn<f64>`,
    # an exact-dtype match that a `float32` buffer fails).
    np_f32 = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    a = nr.array(np_f32)
    assert a.shape == [3]
    assert a.dtype == "float64"
    assert np.allclose(a.tolist(), [1.5, 2.5, 3.5])


def test_matmul():
    a = nr.array([1.0, 2.0, 3.0, 4.0]).reshape([2, 2])  # [[1, 2], [3, 4]]
    identity = nr.eye(2)

    result = nr.matmul(a, identity)
    assert list(result.tolist()) == list(a.tolist())

    b = nr.array([5.0, 6.0, 7.0, 8.0]).reshape([2, 2])  # [[5, 6], [7, 8]]
    result = nr.linalg.matmul(a, b)
    # [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    assert list(result.tolist()) == [19.0, 22.0, 43.0, 50.0]


def test_random_generator_is_seeded_and_reproducible():
    rng1 = nr.random.default_rng(42)
    rng2 = nr.random.default_rng(42)

    u1 = rng1.uniform(0.0, 1.0, [5])
    u2 = rng2.uniform(0.0, 1.0, [5])
    assert list(u1.tolist()) == list(u2.tolist())
    assert all(0.0 <= v < 1.0 for v in u1.tolist())

    # size=None collapses to a scalar float, matching NumPy's own Generator.
    scalar = rng1.normal()
    assert isinstance(scalar, float)

    ints = rng1.integers(0, 10, [50])
    assert all(0.0 <= v < 10.0 and v == int(v) for v in ints.tolist())

    perm = rng1.permutation(10)
    assert sorted(perm.tolist()) == [float(i) for i in range(10)]


def test_random_legacy_convenience_functions():
    a = nr.random.rand([3, 3])
    assert a.shape == [3, 3]
    assert all(0.0 <= v < 1.0 for v in a.tolist())

    b = nr.random.randn([200])
    assert b.shape == [200]
    # A standard-normal sample of 200 draws should not be degenerate.
    data = b.tolist()
    assert min(data) < 0.0 < max(data)


def test_fft_round_trip():
    signal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    x = nr.array(signal)

    spectrum = nr.fft.fft(x)
    assert isinstance(spectrum, np.ndarray)
    assert spectrum.dtype == np.complex128
    assert spectrum.shape == (8,)

    recovered = nr.fft.ifft(spectrum)
    assert np.allclose(recovered.real, signal, atol=1e-9)
    assert np.allclose(recovered.imag, 0.0, atol=1e-9)


def test_rfft_irfft_round_trip():
    signal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    x = nr.array(signal)

    spectrum = nr.fft.rfft(x)
    assert spectrum.shape == (5,)  # n // 2 + 1

    recovered = nr.fft.irfft(spectrum, n=8)
    assert isinstance(recovered, nr.Array)
    assert np.allclose(recovered.tolist(), signal, atol=1e-9)


def test_fftn_round_trip_2d():
    x = nr.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([2, 3])

    spectrum = nr.fft.fftn(x)
    assert spectrum.shape == (2, 3)
    assert spectrum.dtype == np.complex128

    recovered = nr.fft.ifftn(spectrum)
    assert np.allclose(recovered.real.reshape(-1), x.tolist(), atol=1e-9)


def test_rfftn_irfftn_round_trip():
    # `rfftn`'s core (`crate::fft::numpy_parity::rfftn`) deliberately does
    # *not* delegate to `scirs2_fft::rfftn` (see that function's own doc
    # comment for the two bugs this works around); it hand-rolls the
    # last-axis halving on top of `fftn` instead, so this is worth checking
    # against NumPy directly rather than trusting the 1-D `rfft`/`irfft`
    # coverage above to generalize.
    data = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    x_np = np.array(data)
    x = nr.array(x_np)

    spectrum = nr.fft.rfftn(x)
    assert spectrum.shape == (2, 3)  # last axis halved: 4 // 2 + 1
    assert np.allclose(spectrum, np.fft.rfftn(x_np), atol=1e-9)

    recovered = nr.fft.irfftn(spectrum, s=[2, 4])
    assert isinstance(recovered, nr.Array)
    assert np.allclose(np.array(recovered.tolist()).reshape(recovered.shape), x_np, atol=1e-9)


def test_stats_corrcoef_matches_numpy():
    # Regression coverage for the `corrcoef` rewrite documented in
    # `src/python/stats.rs`: the pre-existing `tests/python/test_stats.py`
    # only ever exercises the old, restricted (x, y both 1-D) path. This
    # checks the branches that replaced it: a 1-D `x` alone (NumPy's
    # scalar-collapse quirk), a 2-D `x` alone (`rowvar=True` default), and
    # `rowvar=False` -- each compared directly against `numpy.corrcoef`.
    scalar = nr.stats.corrcoef(nr.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert isinstance(scalar, float)
    assert abs(scalar - 1.0) < 1e-9

    data = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 1.0, 4.0, 3.0, 5.0],
        [5.0, 4.0, 3.0, 2.0, 1.0],
    ]
    x_np = np.array(data)
    x = nr.array(x_np)

    r = nr.stats.corrcoef(x)
    assert np.allclose(np.array(r.tolist()).reshape(r.shape), np.corrcoef(x_np), atol=1e-9)

    r_t = nr.stats.corrcoef(x, rowvar=False)
    assert np.allclose(
        np.array(r_t.tolist()).reshape(r_t.shape), np.corrcoef(x_np, rowvar=False), atol=1e-9
    )


def test_svd_singular_values_are_1d():
    # Regression test: `crate::linalg::svd` (the Rust core this binds to)
    # returns its singular values embedded in a full (m, n) matrix, for a
    # Rust caller's convenience reconstructing `a` via plain matmul; the
    # `nr.linalg.svd` binding must unpack that into NumPy's own convention
    # (`s` as the 1-D vector of singular values) rather than leak the
    # core's internal representation, matching `numpy.linalg.svd`.
    a = nr.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([2, 3])
    u, s, vt = nr.linalg.svd(a)

    assert u.ndim == 2
    assert s.ndim == 1
    assert vt.ndim == 2
    assert s.shape == [2]  # min(2, 3) singular values
    assert list(s.tolist()) == sorted(s.tolist(), reverse=True)

    # Reconstruct a via u @ S @ vt and compare against the original, where S
    # is the (m, n) rectangular embedding of s -- `svd`'s `u`/`vt` are the
    # "full_matrices" shapes ((m, m) and (n, n)), so `np.diag(s_np)`'s
    # square (k, k) is the wrong shape to multiply back against them; this
    # is NumPy's own documented reconstruction recipe for `full_matrices=True`
    # (see the `numpy.linalg.svd` docstring's example).
    # (`Array.tolist()` always flattens, regardless of ndim -- see
    # `src/python/array.rs` -- so the original is reshaped back for the
    # comparison.)
    u_np, s_np, vt_np = u.to_numpy(), s.to_numpy(), vt.to_numpy()
    s_full = np.zeros((u_np.shape[1], vt_np.shape[0]))
    s_full[: s_np.shape[0], : s_np.shape[0]] = np.diag(s_np)
    reconstructed = u_np @ s_full @ vt_np
    original = np.array(a.tolist()).reshape(a.shape)
    assert np.allclose(reconstructed, original, atol=1e-9)


def test_nn_nd_support():
    # Every `nr.nn` op used to reject ndim > 2 outright; this exercises each
    # one against genuine 3-D input (batch=2, seq=3, features/classes=4) to
    # confirm the N-D generalization actually runs end-to-end, not just that
    # the pre-existing 2-D path still does.
    shape = [2, 3, 4]
    size = 2 * 3 * 4
    x = nr.array([float(i) - size / 2 for i in range(size)]).reshape(shape)

    # Elementwise ops: shape-preserving for any rank.
    assert nr.nn.relu(x).shape == shape
    assert nr.nn.sigmoid(x).shape == shape
    assert nr.nn.tanh(x).shape == shape
    assert nr.nn.dropout(x, 0.0).shape == shape  # p=0 -> nothing dropped

    assert nr.nn.mse_loss(x, x) == 0.0

    # softmax: default axis=-1; every last-axis lane must sum to 1.
    sm = nr.nn.softmax(x)
    assert sm.shape == shape
    assert np.allclose(sm.to_numpy().sum(axis=-1), 1.0, atol=1e-9)

    # batch_norm: last axis is "features", every leading axis is "batch";
    # per-feature mean over the flattened (batch*seq) dimension is ~0.
    bn = nr.nn.batch_norm(x)
    assert bn.shape == shape
    assert np.allclose(bn.to_numpy().reshape(-1, shape[-1]).mean(axis=0), 0.0, atol=1e-7)

    # cross_entropy_loss: last axis as classes, leading axes as batch --
    # must not reject rank-3 input (it used to).
    loss = nr.nn.cross_entropy_loss(sm, sm)
    assert isinstance(loss, float)
    assert loss == loss  # not NaN


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
