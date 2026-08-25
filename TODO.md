# NumRS2 Development Status

This document tracks NumRS2's current, still-open work only. For what has already shipped, see
[CHANGELOG.md](CHANGELOG.md) (per-release, with in-tree verification pointers for every entry).
Historical development narrative that used to live in this file has been retired now that its
content is either superseded by CHANGELOG.md or no longer true against the working tree.

## Current Focus

The `0.4.1` branch is a production-hardening pass: `Array<T>` is now `Arc`-backed copy-on-write, a
shared `kernels` dispatch layer backs matmul/elementwise/reduction hot paths, the `distributed`
feature runs real collectives and linear algebra over a TCP transport instead of returning
fabricated results, and a large batch of NumPy-parity APIs landed. See CHANGELOG.md's `[0.4.1]`
section for the full, verified list, including a **Known Upstream Issues** subsection for the
`scirs2-core`/`scirs2-fft` bugs this release works around.

## Known Deferred Items

Items below are open in the current tree as of this writing. Each was checked against the source
(not carried forward from an old note) before being listed here.

- **`fused!` compile-time fusion macro** — deferred to 0.6.0. The runtime fusion path
  (`IntoExpr::expr()` + `.eval()`) shipped in 0.4.1; a macro that expands an expression at compile
  time instead of building an `ExprNode` tree at runtime has not started. See `src/expr/owned.rs`'s
  module doc.
- **CAQR (wide-case distributed QR)** — `distributed::linalg::tsqr` implements Tall-Skinny QR for
  row-block matrices where every rank's block is at least as tall as the matrix is wide
  (`m_i >= n`); it returns `DistributedLinalgError::UnsupportedShape` rather than approximating
  when that precondition fails. The wide-block regime needs communication-avoiding QR (CAQR,
  factoring each column panel with its own TSQR), which does not exist yet. See
  `src/distributed/linalg/tsqr.rs`'s module doc.
- **Quantized allreduce is not wired end-to-end** — `distributed::compression::QuantizedTensor`
  implements affine (scale + zero-point) 4-/8-bit quantization, but the generic
  `compress_tensor`/collective-reduction path does not perform quantization itself (its
  `(values, indices)` shape can't represent bit-packed data); `CompressionStrategy::Quantization`
  returns a clear error pointing callers at `QuantizedTensor::quantize` directly instead. A
  quantized value integrated into an actual `allreduce` call is still a manual assembly job for
  the caller. See `src/distributed/compression.rs`.
- **GPU has no software-adapter fallback on macOS** — `NUMRS2_GPU_FALLBACK=1` requests a
  software adapter (e.g. lavapipe, WARP) on platforms that ship one; macOS/Metal does not ship a
  software adapter at all, so the fallback is a no-op there and GPU tests/examples need a real
  GPU on macOS. See `src/gpu/context.rs` and `src/gpu/mod.rs`.
- **`new_modules::special::hypergeometric::polylog_scalar` has no `|z| > 1` branch** — direct
  series summation is used for `|z| < 1` and the `z == 1` case reduces to `zeta`; for `|z| > 1`
  the function returns `NaN` (`// Complex extension needed`) rather than an analytic
  continuation. See `src/new_modules/special/hypergeometric.rs`.
- **`tests/nn/test_simd_ops.rs::test_simd_matmul_f64`** — still `#[ignore]`d pending an upstream
  `scirs2-core` SIMD f64 matmul fix. The in-file comment cites `scirs2-core-0.1.5`; the crate is
  now on `scirs2-core` 0.6.5, so that version reference is stale, but the underlying issue has not
  been independently re-verified as fixed, and the test remains ignored either way.

### Intentional NumPy deviations (design decisions, not bugs)

These are documented, deliberate differences from NumPy semantics — tracked here so they stay
visible, not because they are planned to change:

- **`axis=None` flattens** (row-major/C order) before reducing, matching NumPy's own
  `axis=None` convention, but several `axis=None` code paths in this crate return a **shape-`[1]`
  array** where NumPy would return a bare scalar / 0-d array (e.g. `math::statistics::argmin`'s
  and `MaskedArray::argmin`'s `axis: None` branch ignores `keepdims` and always returns `[1]`).
  See `src/masked/search.rs`'s module doc, which cross-references the unmasked cousin.
- **Masked `argmin`/`argmax` return `Err` on a fully-masked lane**, rather than `numpy.ma`'s
  degenerate (and silently ambiguous) index `0` — this crate has no warning channel to flag that
  degenerate case the way NumPy does, so an `Err` replaces a silently-misleading position. Call
  `MaskedArray::count_valid` first to distinguish "no candidates" from a real answer without
  pattern-matching the error. See `src/masked/search.rs`.
- **`Generator::permuted` is not bit-identical to `np.random.Generator.permuted`, on any bit
  generator.** `permuted` always shuffles by seeding a fresh per-call `StdRng` from the bit
  generator's stream rather than drawing raw stream values the way NumPy's own Fisher-Yates does —
  so even `Philox4x64BitGenerator`/`SFC64BitGenerator` (whose *raw* output streams do reproduce
  NumPy's `Philox`/`SFC64` exactly, unlike this crate's `PCG64BitGenerator`, which derives its
  128-bit state differently from NumPy's own default `PCG64`) are not `permuted`-bit-identical
  vs. NumPy. What is guaranteed is the semantic contract (independent per-lane shuffling for
  `axis=Some(k)`, matching whole-array `shuffle` semantics otherwise) and reproducibility against
  another `numrs2` `Generator` seeded the same way. See `src/random/generator.rs`'s `permuted` doc
  (the "Exactness" section).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to NumRS2, and
[scripts/ci-local.sh](scripts/ci-local.sh) for the local checks a change is expected to pass
(there is no hosted CI build — GitHub Actions is restricted by project policy to publish
workflows only).
