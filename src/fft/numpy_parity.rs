//! NumPy-parity parameter wrappers.
//!
//! `scirs2_fft`'s own N-D entry points (`fftn`/`ifftn`/`rfftn`/`irfftn`,
//! re-exported by the parent `fft` module) already have a real,
//! hardware-accelerated (`oxifft` backend) N-dimensional implementation --
//! there is no need to hand-roll a
//! separable axis-by-axis pass on top of them. What they *don't* match is
//! NumPy's exact calling convention:
//!
//! * `axes` there is `Option<Vec<usize>>` (no negative-index support);
//!   NumPy's `axes` accepts negative indices (`-1` = last axis, etc.).
//! * `shape` there must either be `None` or cover *every* dimension of the
//!   input; NumPy's `s` may instead have one entry per axis in `axes`.
//! * A handful of them (verified empirically against `numpy.fft`, see the
//!   pinned tests below) silently treat an unrecognized `norm` string the
//!   same as `None` instead of rejecting it the way NumPy does.
//! * `scirs2_fft::fftn`'s own `norm` handling has two confirmed bugs
//!   (found by probing it directly, then cross-checking the same calls
//!   against `numpy.fft.fftn`): passing the explicit string `"backward"`
//!   applies a spurious `1/N` scale to the *forward* transform (it should
//!   be a complete no-op there, identical to `norm=None` -- "backward"
//!   only means "scale the *inverse* direction"), and its `"ortho"`/
//!   `"forward"` scale factor is computed from the size of the full
//!   *output* array rather than from just the product of the sizes of the
//!   axes actually being transformed -- so it's silently wrong the moment
//!   `axes` names a strict subset of the array's dimensions. Both bugs
//!   live in `fftn`'s own normalization step (`src/fft/algorithms.rs`,
//!   using `outputshape.iter().product()` as the scale basis regardless of
//!   which `axes` were requested); `rfftn` inherits the second one by
//!   forwarding its own `norm` straight into `fftn` internally. `ifftn`
//!   does not share either bug -- it independently computes its scale
//!   basis as `axes.iter().map(|&a| outputshape[a]).product()`, which is
//!   exactly right -- and `irfftn` inherits that correctness the same way
//!   `rfftn` inherits `fftn`'s bug, by delegating its own normalization to
//!   `ifftn` internally. So `fftn`/`rfftn` below always request a *raw*
//!   (`norm: None`) transform from `scirs2_fft` and apply the requested
//!   scale themselves, using the correct axes-restricted basis; `ifftn`/
//!   `irfftn` forward `norm` straight through unmodified, since delegating
//!   is already correct there.

use num_traits::NumCast;
use scirs2_core::ndarray::{ArrayD, Slice};
use scirs2_core::Complex64;
use scirs2_fft::{FFTError, FFTResult};
use std::fmt::Debug;

/// Normalize a NumPy-style, possibly-negative axis list against `ndim`.
///
/// `None` passes through unchanged (scirs2_fft's own N-D functions already
/// default a missing axis list to "every axis, in order"). Each negative
/// axis `a` is remapped to `a + ndim`, matching NumPy's `axis`/`axes`
/// conventions; order is preserved (not sorted), since some callers (e.g.
/// `rfftn`) treat the *last given* axis specially.
fn normalize_axes(axes: Option<&[isize]>, ndim: usize) -> FFTResult<Option<Vec<usize>>> {
    let Some(axes) = axes else {
        return Ok(None);
    };
    let mut out = Vec::with_capacity(axes.len());
    for &axis in axes {
        let normalized = if axis < 0 { axis + ndim as isize } else { axis };
        if normalized < 0 || normalized as usize >= ndim {
            return Err(FFTError::ValueError(format!(
                "axis {axis} is out of bounds for array of dimension {ndim}"
            )));
        }
        out.push(normalized as usize);
    }
    Ok(Some(out))
}

/// Expand a NumPy-style `s` (sizes for just the transformed axes) into the
/// full-`ndim` output shape that `fftn`/`ifftn`/`rfftn`/`irfftn` require,
/// falling back to `input_shape` on every axis `s` doesn't cover.
///
/// Mirrors NumPy's `s`/`axes` pairing: when `axes` is `None`, `s` (if
/// given) must cover every axis; when `axes` is `Some`, `s` (if given)
/// must have exactly one entry per axis, applied positionally.
fn expand_shape(
    input_shape: &[usize],
    s: Option<&[usize]>,
    axes: &Option<Vec<usize>>,
) -> FFTResult<Option<Vec<usize>>> {
    let Some(sizes) = s else {
        return Ok(None);
    };
    let mut full = input_shape.to_vec();
    match axes {
        Some(ax) => {
            if sizes.len() != ax.len() {
                return Err(FFTError::ValueError(format!(
                    "s must have the same length as axes ({} vs {})",
                    sizes.len(),
                    ax.len()
                )));
            }
            for (&axis, &size) in ax.iter().zip(sizes.iter()) {
                full[axis] = size;
            }
        }
        None => {
            if sizes.len() != input_shape.len() {
                return Err(FFTError::ValueError(format!(
                    "s must have the same length as the input's number of dimensions ({}) when axes is not given, got {}",
                    input_shape.len(),
                    sizes.len()
                )));
            }
            full.copy_from_slice(sizes);
        }
    }
    Ok(Some(full))
}

/// Validate a NumPy-style `norm` string ourselves: `scirs2_fft::fftn`/
/// `ifftn` silently treat any unrecognized string the same as `None`
/// (no normalization) rather than rejecting it, so without this check an
/// invalid `norm` would be ignored instead of reported.
fn validate_norm(norm: Option<&str>) -> FFTResult<()> {
    match norm {
        None | Some("backward") | Some("ortho") | Some("forward") => Ok(()),
        Some(other) => Err(FFTError::ValueError(format!(
            "Invalid norm value '{other}': expected \"backward\", \"forward\", or \"ortho\""
        ))),
    }
}

/// Product of the output shape's sizes along exactly the axes that get
/// transformed -- the correct `norm="ortho"`/`"forward"` scale basis for
/// the N-D transforms (matching NumPy's own `functools.reduce(operator.mul,
/// (s[a] for a in axes))`), as opposed to `scirs2_fft::fftn`'s buggy use of
/// the full output array size regardless of `axes` (see the module-level
/// note above).
fn transformed_axes_size(outshape: &[usize], axes: &Option<Vec<usize>>) -> usize {
    match axes {
        Some(ax) => ax.iter().map(|&a| outshape[a]).product(),
        None => outshape.iter().product(),
    }
}

/// N-dimensional FFT with NumPy's exact `fftn(a, s=None, axes=None,
/// norm=None)` parameter conventions, shadowing [`scirs2_fft::fftn`] (see
/// the module-level note above for why, including the two `norm` bugs
/// this works around by always requesting a raw transform and applying
/// the correct scale itself).
pub fn fftn<T>(
    input: &ArrayD<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: Option<&str>,
) -> FFTResult<ArrayD<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    validate_norm(norm)?;
    let normalized_axes = normalize_axes(axes, input.ndim())?;
    let outshape = expand_shape(input.shape(), s, &normalized_axes)?;
    let mut result = scirs2_fft::fftn(
        input,
        outshape.clone(),
        normalized_axes.clone(),
        None,
        None,
        None,
    )?;
    let scale = forward_norm_scale(
        norm,
        transformed_axes_size(result.shape(), &normalized_axes),
    )?;
    if scale != 1.0 {
        result.mapv_inplace(|c| c * scale);
    }
    Ok(result)
}

/// N-dimensional inverse FFT with NumPy's exact `ifftn(a, s=None,
/// axes=None, norm=None)` parameter conventions, shadowing
/// [`scirs2_fft::ifftn`].
pub fn ifftn<T>(
    input: &ArrayD<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: Option<&str>,
) -> FFTResult<ArrayD<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    validate_norm(norm)?;
    let normalized_axes = normalize_axes(axes, input.ndim())?;
    let outshape = expand_shape(input.shape(), s, &normalized_axes)?;
    scirs2_fft::ifftn(input, outshape, normalized_axes, norm, None, None)
}

/// N-dimensional real-input FFT with NumPy's exact `rfftn(a, s=None,
/// axes=None, norm=None)` parameter conventions, shadowing
/// [`scirs2_fft::rfftn`].
///
/// `scirs2_fft::rfftn` only halves its last transformed axis (keeping the
/// non-redundant half of the Hermitian-symmetric spectrum) when its own
/// `shape` parameter is `None`; passing it any explicit shape -- which is
/// unavoidable once `s` zero-pads or truncates an axis -- disables that
/// halving. It also forwards `norm` straight into (buggy) `scirs2_fft::
/// fftn` internally, inheriting both bugs described in the module-level
/// note above. So instead of delegating to it at all, this computes the
/// full complex N-D FFT via this module's own [`fftn`] (whose zero-pad/
/// truncate semantics are unconditional and whose `norm` handling is
/// already correct) and then keeps only the non-redundant half of the
/// last transformed axis itself, unconditionally -- the same relationship
/// NumPy documents between `rfftn` and `fftn`.
pub fn rfftn<T>(
    input: &ArrayD<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: Option<&str>,
) -> FFTResult<ArrayD<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    let ndim = input.ndim();
    let normalized_axes = normalize_axes(axes, ndim)?;

    let last_axis = match &normalized_axes {
        Some(ax) => *ax.last().ok_or_else(|| {
            FFTError::ValueError("axes must contain at least one axis".to_string())
        })?,
        None => ndim.checked_sub(1).ok_or_else(|| {
            FFTError::ValueError("rfftn requires an array with at least one dimension".to_string())
        })?,
    };

    let full = fftn(input, s, axes, norm)?;
    let half_len = full.shape()[last_axis] / 2 + 1;
    let sliced = full
        .slice_each_axis(|ax| {
            if ax.axis.index() == last_axis {
                Slice::new(0, Some(half_len as isize), 1)
            } else {
                Slice::new(0, None, 1)
            }
        })
        .to_owned();
    Ok(sliced)
}

/// N-dimensional inverse real FFT with NumPy's exact `irfftn(a, s=None,
/// axes=None, norm=None)` parameter conventions, shadowing
/// [`scirs2_fft::irfftn`].
///
/// Unlike `rfftn`, `scirs2_fft::irfftn` already defaults a missing `shape`
/// to NumPy's own convention (`2 * (m - 1)` on the last transformed axis),
/// so an absent `s` can be passed straight through as `None`; it also
/// delegates its own normalization to `scirs2_fft::ifftn` internally,
/// which -- unlike `fftn` -- already computes its scale from the correct,
/// axes-restricted basis (see the module-level note above), so `norm` is
/// likewise safe to forward through unmodified here.
pub fn irfftn<T>(
    input: &ArrayD<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
    norm: Option<&str>,
) -> FFTResult<ArrayD<f64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    validate_norm(norm)?;
    let normalized_axes = normalize_axes(axes, input.ndim())?;
    let outshape = expand_shape(input.shape(), s, &normalized_axes)?;
    scirs2_fft::irfftn(&input.view(), outshape, normalized_axes, norm, None, None)
}

/// Validate a NumPy-style `axis` for a transform that operates on a flat,
/// one-dimensional slice: the only legal values are `0` and `-1` (both
/// name the slice's single axis). Multi-axis selection is only meaningful
/// for the N-D family ([`fftn`], [`ifftn`], [`rfftn`], [`irfftn`]).
fn validate_flat_axis(axis: Option<isize>) -> FFTResult<()> {
    match axis {
        None | Some(0) | Some(-1) => Ok(()),
        Some(other) => Err(FFTError::ValueError(format!(
            "axis {other} is out of bounds for a 1-D transform (expected 0 or -1)"
        ))),
    }
}

/// Absolute scale factor for a *raw, completely unnormalized* forward
/// transform of size `n`, matching NumPy's `norm` semantics for
/// `fft`/`rfft` (`"backward"`, the default, applies no scaling at all to
/// the forward direction).
fn forward_norm_scale(norm: Option<&str>, n: usize) -> FFTResult<f64> {
    match norm {
        None | Some("backward") => Ok(1.0),
        Some("forward") => Ok(1.0 / n as f64),
        Some("ortho") => Ok(1.0 / (n as f64).sqrt()),
        Some(other) => Err(FFTError::ValueError(format!(
            "Invalid norm value '{other}': expected \"backward\", \"forward\", or \"ortho\""
        ))),
    }
}

/// Correction factor to apply on top of `scirs2_fft`'s `ifft`/`irfft`,
/// which always apply the `"backward"`-mode `1/n` scaling internally, to
/// realize whatever `norm` was actually requested.
fn inverse_norm_correction(norm: Option<&str>, n: usize) -> FFTResult<f64> {
    match norm {
        None | Some("backward") => Ok(1.0),
        Some("forward") => Ok(n as f64),
        Some("ortho") => Ok((n as f64).sqrt()),
        Some(other) => Err(FFTError::ValueError(format!(
            "Invalid norm value '{other}': expected \"backward\", \"forward\", or \"ortho\""
        ))),
    }
}

/// 1-D FFT with NumPy's exact `fft(a, n=None, axis=-1, norm=None)`
/// parameters. The plain [`fft`] (re-exported from `scirs2_fft` above)
/// keeps its original 2-argument signature (`x`, `n`) unchanged; this adds
/// the full parameter set as a separate, `_with`-suffixed function rather
/// than changing `fft`'s arity, matching the calling convention
/// `scirs2_fft` itself already uses for its multi-parameter transforms
/// (`fft2`, `fftn`, `dct`, ...).
pub fn fft_with<T>(
    x: &[T],
    n: Option<usize>,
    axis: Option<isize>,
    norm: Option<&str>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    validate_flat_axis(axis)?;
    let size = n.unwrap_or(x.len());
    let scale = forward_norm_scale(norm, size)?;
    let mut result = scirs2_fft::fft(x, Some(size))?;
    result.iter_mut().for_each(|c| *c *= scale);
    Ok(result)
}

/// 1-D inverse FFT with NumPy's exact `ifft(a, n=None, axis=-1,
/// norm=None)` parameters. See [`fft_with`] for why this is a separate
/// function rather than a change to [`ifft`]'s signature.
pub fn ifft_with<T>(
    x: &[T],
    n: Option<usize>,
    axis: Option<isize>,
    norm: Option<&str>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    validate_flat_axis(axis)?;
    let size = n.unwrap_or(x.len());
    let correction = inverse_norm_correction(norm, size)?;
    let mut result = scirs2_fft::ifft(x, Some(size))?;
    result.iter_mut().for_each(|c| *c *= correction);
    Ok(result)
}

/// 1-D real-input FFT with NumPy's exact `rfft(a, n=None, axis=-1,
/// norm=None)` parameters. See [`fft_with`] for why this is a separate
/// function rather than a change to [`rfft`]'s signature.
pub fn rfft_with<T>(
    x: &[T],
    n: Option<usize>,
    axis: Option<isize>,
    norm: Option<&str>,
) -> FFTResult<Vec<Complex64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    validate_flat_axis(axis)?;
    let size = n.unwrap_or(x.len());
    let scale = forward_norm_scale(norm, size)?;
    let mut result = scirs2_fft::rfft(x, Some(size))?;
    result.iter_mut().for_each(|c| *c *= scale);
    Ok(result)
}

/// 1-D inverse real FFT with NumPy's exact `irfft(a, n=None, axis=-1,
/// norm=None)` parameters. See [`fft_with`] for why this is a separate
/// function rather than a change to [`irfft`]'s signature.
pub fn irfft_with<T>(
    x: &[T],
    n: Option<usize>,
    axis: Option<isize>,
    norm: Option<&str>,
) -> FFTResult<Vec<f64>>
where
    T: NumCast + Copy + Debug + 'static,
{
    validate_flat_axis(axis)?;
    // NumPy's own default, mirrored by `scirs2_fft::irfft`: n = 2*(m-1)
    // where m is the input (spectrum) length.
    let size = n.unwrap_or_else(|| 2 * x.len().saturating_sub(1));
    let correction = inverse_norm_correction(norm, size)?;
    let mut result = scirs2_fft::irfft(x, Some(size))?;
    result.iter_mut().for_each(|v| *v *= correction);
    Ok(result)
}

// Additional NumRS2-specific convenience functions and aliases can be added here
