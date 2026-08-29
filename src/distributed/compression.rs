//! Gradient/Tensor Compression for Distributed Communication
//!
//! This module implements the actual compression strategies referenced by
//! [`CompressionStrategy`]: top-k sparsification (by magnitude), random-k
//! sparsification (uniform sampling without replacement, seed-deterministic),
//! threshold-based sparsification, and affine bit-packed quantization via
//! [`QuantizedTensor`].
//!
//! Carved out of `communication.rs` (which still re-exports
//! [`CompressionStrategy`], [`compress_tensor`], and [`decompress_tensor`]
//! for backward compatibility with existing call sites).
//!
//! # Example
//!
//! ```rust
//! use numrs2::distributed::compression::{compress_tensor, decompress_tensor, CompressionStrategy};
//!
//! let data = vec![1.0_f64, -9.0, 3.0, 8.0];
//! let (values, indices) = compress_tensor(&data, &CompressionStrategy::TopK { k: 2 })
//!     .expect("compression should succeed");
//! // Top-2 by absolute value: -9.0 and 8.0 (order follows ascending index).
//! assert_eq!(values, vec![-9.0, 8.0]);
//!
//! let restored = decompress_tensor(&values, indices.as_deref(), data.len())
//!     .expect("decompression should succeed");
//! assert_eq!(restored, vec![0.0, -9.0, 0.0, 8.0]);
//! ```

use super::communication::CommunicationError;
use num_traits::Float;
use oxicode::{Decode, Encode};
use scirs2_core::random::{SeedableRng, StdRng};
use std::cmp::Ordering;

/// Compression strategy for bandwidth optimization.
///
/// Moved here from `communication.rs` (which re-exports this type under its
/// own namespace for backward compatibility); this is where compression
/// behavior for each variant is actually implemented, in [`compress_tensor`].
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub enum CompressionStrategy {
    /// No compression
    None,

    /// Top-k sparsification: keep only the k largest elements by absolute
    /// value.
    TopK {
        /// Number of elements to keep.
        k: usize,
    },

    /// Random-k: uniformly sample k distinct elements without replacement,
    /// using Floyd's algorithm seeded by `seed` (same seed always produces
    /// the same set of indices for a given `(k, data.len())`).
    RandomK {
        /// Number of elements to sample.
        k: usize,
        /// Seed controlling which indices are sampled.
        seed: u64,
    },

    /// Quantization to reduce precision. `compress_tensor` does not perform
    /// this itself (it has a different return shape than bit-packed
    /// quantization needs) — see [`QuantizedTensor::quantize`].
    Quantization {
        /// Bits per quantized value (4 or 8).
        bits: u8,
    },

    /// Threshold-based sparsification: keep elements whose absolute value is
    /// at least `threshold`.
    Threshold {
        /// Minimum absolute value to keep.
        threshold: f64,
    },
}

/// Compress tensor data using the specified strategy.
///
/// Returns `(values, indices)`: `indices` is `None` when every element was
/// kept (nothing to reconstruct), `Some(_)` otherwise, pairing each kept
/// value with its original position for [`decompress_tensor`].
///
/// Safe on empty input and on `k == 0` for every k-based strategy (yields an
/// empty selection rather than panicking or dividing by zero).
pub fn compress_tensor<T>(
    data: &[T],
    strategy: &CompressionStrategy,
) -> Result<(Vec<T>, Option<Vec<usize>>), CommunicationError>
where
    T: Float,
{
    match strategy {
        CompressionStrategy::None => Ok((data.to_vec(), None)),

        CompressionStrategy::TopK { k } => {
            if *k >= data.len() {
                return Ok((data.to_vec(), None));
            }

            let mut indexed: Vec<(usize, T)> =
                data.iter().enumerate().map(|(i, v)| (i, *v)).collect();

            // Partition so the k largest-by-absolute-value elements land in
            // positions `0..k` (descending order of |value|).
            indexed.select_nth_unstable_by(*k, |a, b| {
                b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(Ordering::Equal)
            });

            let mut selected: Vec<(usize, T)> = indexed.into_iter().take(*k).collect();
            selected.sort_unstable_by_key(|(idx, _)| *idx);

            let mut values = Vec::with_capacity(*k);
            let mut indices = Vec::with_capacity(*k);
            for (idx, val) in selected {
                indices.push(idx);
                values.push(val);
            }

            Ok((values, Some(indices)))
        }

        CompressionStrategy::RandomK { k, seed } => {
            if *k >= data.len() {
                return Ok((data.to_vec(), None));
            }

            let indices = floyd_sample(data.len(), *k, *seed);
            let values: Vec<T> = indices.iter().map(|&i| data[i]).collect();

            Ok((values, Some(indices)))
        }

        CompressionStrategy::Quantization { .. } => Err(CommunicationError::Compression(
            "Quantization is not performed by compress_tensor (its (values, indices) shape \
             cannot represent bit-packed data); use QuantizedTensor::quantize instead"
                .to_string(),
        )),

        CompressionStrategy::Threshold { threshold } => {
            let threshold_t = T::from(*threshold).ok_or_else(|| {
                CommunicationError::Compression(format!(
                    "threshold {threshold} is out of range for the element type"
                ))
            })?;

            let mut values = Vec::new();
            let mut indices = Vec::new();
            for (i, &v) in data.iter().enumerate() {
                if v.abs() >= threshold_t {
                    indices.push(i);
                    values.push(v);
                }
            }

            Ok((values, Some(indices)))
        }
    }
}

/// Sample `k` distinct indices from `0..n` uniformly at random, without
/// replacement, using Floyd's algorithm seeded by `seed`. The same
/// `(n, k, seed)` always produces the same set of indices (returned sorted
/// ascending). Requires `k <= n`; returns an empty vector when `k == 0`.
fn floyd_sample(n: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut selected: Vec<usize> = Vec::with_capacity(k);

    for j in (n - k)..n {
        let t: usize = rng.gen_range(0..=j);
        if selected.contains(&t) {
            selected.push(j);
        } else {
            selected.push(t);
        }
    }

    selected.sort_unstable();
    selected
}

/// Decompress tensor data.
///
/// Scatters `compressed` values back to their original positions using
/// `indices` (`None` means `compressed` already holds every element, e.g.
/// produced by `CompressionStrategy::None` or the `k >= len` fast path).
pub fn decompress_tensor<T>(
    compressed: &[T],
    indices: Option<&[usize]>,
    original_size: usize,
) -> Result<Vec<T>, CommunicationError>
where
    T: Clone + Default,
{
    match indices {
        None => Ok(compressed.to_vec()),
        Some(idx) => {
            let mut result = vec![T::default(); original_size];
            for (i, &pos) in idx.iter().enumerate() {
                if pos < original_size && i < compressed.len() {
                    result[pos] = compressed[i].clone();
                }
            }
            Ok(result)
        }
    }
}

/// A tensor quantized to `bits`-per-element (4 or 8) via affine
/// (scale + zero-point) quantization, with values bit-packed into `data`.
///
/// Dequantization recovers `x ≈ (code - zero_point) * scale`; the maximum
/// error per element is bounded by `scale / 2`.
#[derive(Debug, Clone, PartialEq, Encode, Decode)]
pub struct QuantizedTensor {
    /// Bits per quantized element: 4 or 8.
    pub bits: u8,
    /// Number of original (unpacked) elements.
    pub len: usize,
    /// Quantization scale (real-value units per quantized step).
    pub scale: f64,
    /// The quantized code that represents real value `0.0`.
    pub zero_point: i32,
    /// Bit-packed quantized codes: one byte per element when `bits == 8`,
    /// two elements (low nibble first) per byte when `bits == 4`.
    pub data: Vec<u8>,
}

impl QuantizedTensor {
    /// Quantize `data` to `bits` bits per element (4 or 8).
    ///
    /// Safe on empty input (returns a zero-length `QuantizedTensor`) and on
    /// constant input (`scale` falls back to `1.0` rather than dividing by
    /// zero).
    pub fn quantize<T: Float>(data: &[T], bits: u8) -> Result<Self, CommunicationError> {
        if bits != 4 && bits != 8 {
            return Err(CommunicationError::Compression(format!(
                "unsupported quantization width: {bits} bits (only 4 or 8 are supported)"
            )));
        }

        let len = data.len();
        if len == 0 {
            return Ok(Self {
                bits,
                len: 0,
                scale: 1.0,
                zero_point: 0,
                data: Vec::new(),
            });
        }

        let qmax = ((1u32 << bits) - 1) as f64;

        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for v in data {
            let f = v.to_f64().unwrap_or(0.0);
            min_v = min_v.min(f);
            max_v = max_v.max(f);
        }

        let scale = if max_v > min_v {
            (max_v - min_v) / qmax
        } else {
            1.0
        };
        let zero_point = (-min_v / scale).round().clamp(0.0, qmax) as i32;

        let codes: Vec<u32> = data
            .iter()
            .map(|v| {
                let f = v.to_f64().unwrap_or(0.0);
                let q = (f / scale).round() + zero_point as f64;
                q.clamp(0.0, qmax) as u32
            })
            .collect();

        Ok(Self {
            bits,
            len,
            scale,
            zero_point,
            data: pack_codes(&codes, bits),
        })
    }

    /// Dequantize back to `T`, approximately recovering the original values
    /// (bounded error of `scale / 2` per element from the quantization
    /// step).
    pub fn dequantize<T: Float>(&self) -> Result<Vec<T>, CommunicationError> {
        if self.len == 0 {
            return Ok(Vec::new());
        }

        unpack_codes(&self.data, self.bits, self.len)
            .into_iter()
            .map(|code| {
                let x = (code as f64 - self.zero_point as f64) * self.scale;
                T::from(x).ok_or_else(|| {
                    CommunicationError::Compression(format!(
                        "dequantized value {x} is out of range for the target type"
                    ))
                })
            })
            .collect()
    }
}

/// Pack `codes` (each `< 2^bits`) into bytes: one code per byte for 8 bits,
/// two codes per byte (low nibble = even index) for 4 bits.
fn pack_codes(codes: &[u32], bits: u8) -> Vec<u8> {
    if bits == 8 {
        return codes.iter().map(|&c| c as u8).collect();
    }

    // bits == 4
    let mut packed = Vec::with_capacity(codes.len().div_ceil(2));
    for pair in codes.chunks(2) {
        let low = (pair[0] & 0x0F) as u8;
        let high = pair.get(1).map(|&c| (c & 0x0F) as u8).unwrap_or(0);
        packed.push(low | (high << 4));
    }
    packed
}

/// Inverse of [`pack_codes`]: unpack `len` codes from `packed`.
fn unpack_codes(packed: &[u8], bits: u8, len: usize) -> Vec<u32> {
    if bits == 8 {
        return packed.iter().take(len).map(|&b| b as u32).collect();
    }

    // bits == 4
    let mut codes = Vec::with_capacity(len);
    for i in 0..len {
        let byte = packed.get(i / 2).copied().unwrap_or(0);
        let code = if i % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        };
        codes.push(code as u32);
    }
    codes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_none() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = compress_tensor(&data, &CompressionStrategy::None);
        assert!(result.is_ok());

        let (compressed, indices) = result.expect("compression failed");
        assert_eq!(compressed, data);
        assert!(indices.is_none());
    }

    #[test]
    fn test_compression_topk_selects_by_absolute_value() {
        // Reference case from the task: TopK(2) of [1,-9,3,8] keeps {-9, 8}.
        let data = vec![1.0, -9.0, 3.0, 8.0];
        let (values, indices) = compress_tensor(&data, &CompressionStrategy::TopK { k: 2 })
            .expect("compression failed");
        assert_eq!(values, vec![-9.0, 8.0]);
        assert_eq!(indices, Some(vec![1, 3]));
    }

    #[test]
    fn test_compression_topk_len() {
        let data = vec![5.0, 1.0, 8.0, 3.0, 9.0, 2.0];
        let (compressed, indices) = compress_tensor(&data, &CompressionStrategy::TopK { k: 3 })
            .expect("compression failed");
        assert_eq!(compressed.len(), 3);
        assert_eq!(indices.expect("indices missing").len(), 3);
    }

    #[test]
    fn test_compression_topk_zero_k_is_safe() {
        let data = vec![1.0, 2.0, 3.0];
        let (values, indices) =
            compress_tensor(&data, &CompressionStrategy::TopK { k: 0 }).expect("should not panic");
        assert!(values.is_empty());
        assert_eq!(indices, Some(Vec::new()));
    }

    #[test]
    fn test_compression_topk_empty_data_is_safe() {
        let data: Vec<f64> = vec![];
        let (values, indices) =
            compress_tensor(&data, &CompressionStrategy::TopK { k: 5 }).expect("should not panic");
        assert!(values.is_empty());
        assert!(indices.is_none());
    }

    #[test]
    fn test_compression_topk_full_data() {
        let data = vec![1.0, 2.0, 3.0];
        let result = compress_tensor(&data, &CompressionStrategy::TopK { k: 10 });
        assert!(result.is_ok());

        let (compressed, indices) = result.expect("compression failed");
        assert_eq!(compressed, data);
        assert!(indices.is_none());
    }

    #[test]
    fn test_compression_randomk_len_and_distinct() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (compressed, indices) =
            compress_tensor(&data, &CompressionStrategy::RandomK { k: 3, seed: 7 })
                .expect("compression failed");
        assert_eq!(compressed.len(), 3);
        let indices = indices.expect("indices missing");
        assert_eq!(indices.len(), 3);

        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "indices must be distinct: {indices:?}");
        for &i in &indices {
            assert!(i < data.len());
        }
    }

    #[test]
    fn test_compression_randomk_seed_determinism() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let (values_a, indices_a) =
            compress_tensor(&data, &CompressionStrategy::RandomK { k: 5, seed: 123 })
                .expect("compression failed");
        let (values_b, indices_b) =
            compress_tensor(&data, &CompressionStrategy::RandomK { k: 5, seed: 123 })
                .expect("compression failed");
        assert_eq!(indices_a, indices_b, "same seed must give same indices");
        assert_eq!(values_a, values_b);

        let (indices_c, _) = {
            let (v, i) = compress_tensor(&data, &CompressionStrategy::RandomK { k: 5, seed: 456 })
                .expect("compression failed");
            (i, v)
        };
        assert_ne!(
            indices_a, indices_c,
            "different seeds should (almost certainly) give different indices"
        );
    }

    #[test]
    fn test_compression_randomk_zero_k_is_safe() {
        let data = vec![1.0, 2.0, 3.0];
        let (values, indices) =
            compress_tensor(&data, &CompressionStrategy::RandomK { k: 0, seed: 1 })
                .expect("should not panic");
        assert!(values.is_empty());
        assert_eq!(indices, Some(Vec::new()));
    }

    #[test]
    fn test_compression_randomk_full_data() {
        let data = vec![1.0, 2.0, 3.0];
        let result = compress_tensor(&data, &CompressionStrategy::RandomK { k: 10, seed: 0 });
        assert!(result.is_ok());

        let (compressed, indices) = result.expect("compression failed");
        assert_eq!(compressed, data);
        assert!(indices.is_none());
    }

    #[test]
    fn test_compression_threshold_actually_thresholds() {
        let data = vec![0.1, -5.0, 0.05, 3.0, -0.2];
        let (values, indices) =
            compress_tensor(&data, &CompressionStrategy::Threshold { threshold: 1.0 })
                .expect("compression failed");
        assert_eq!(values, vec![-5.0, 3.0]);
        assert_eq!(indices, Some(vec![1, 3]));
    }

    #[test]
    fn test_compression_threshold_none_pass() {
        let data = vec![0.1, 0.2, 0.05];
        let (values, indices) =
            compress_tensor(&data, &CompressionStrategy::Threshold { threshold: 10.0 })
                .expect("compression failed");
        assert!(values.is_empty());
        assert_eq!(indices, Some(Vec::new()));
    }

    #[test]
    fn test_compression_quantization_returns_clear_error() {
        let data = vec![1.0, 2.0, 3.0];
        let result = compress_tensor(&data, &CompressionStrategy::Quantization { bits: 8 });
        let err = result.expect_err("Quantization must not silently no-op in compress_tensor");
        assert!(
            err.to_string().contains("QuantizedTensor"),
            "error should point at QuantizedTensor, got: {err}"
        );
    }

    #[test]
    fn test_decompress_none() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = decompress_tensor(&data, None, data.len());
        assert!(result.is_ok());

        let decompressed = result.expect("decompression failed");
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_decompress_with_indices() {
        let compressed = vec![5.0, 8.0, 9.0];
        let indices = vec![0, 2, 4];
        let original_size = 6;

        let result = decompress_tensor(&compressed, Some(&indices), original_size);
        assert!(result.is_ok());

        let decompressed = result.expect("decompression failed");
        assert_eq!(decompressed.len(), original_size);
        assert_eq!(decompressed[0], 5.0);
        assert_eq!(decompressed[2], 8.0);
        assert_eq!(decompressed[4], 9.0);
    }

    #[test]
    fn test_compress_decompress_roundtrip_topk() {
        let data = vec![1.0, -9.0, 3.0, 8.0, 0.5, -0.2];
        let (values, indices) = compress_tensor(&data, &CompressionStrategy::TopK { k: 3 })
            .expect("compression failed");
        let restored = decompress_tensor(&values, indices.as_deref(), data.len())
            .expect("decompression failed");
        // Every kept position must match exactly; everything else is zero.
        for (i, &orig) in data.iter().enumerate() {
            if indices.as_ref().expect("indices missing").contains(&i) {
                assert_eq!(restored[i], orig);
            } else {
                assert_eq!(restored[i], 0.0);
            }
        }
    }

    #[test]
    fn test_compression_strategy_serialization() {
        let strategies = vec![
            CompressionStrategy::None,
            CompressionStrategy::TopK { k: 10 },
            CompressionStrategy::RandomK { k: 5, seed: 99 },
            CompressionStrategy::Quantization { bits: 8 },
            CompressionStrategy::Threshold { threshold: 0.01 },
        ];

        for strategy in strategies {
            let serialized = oxicode::encode_to_vec(&strategy);
            assert!(serialized.is_ok());

            let bytes = serialized.expect("serialization failed");
            let deserialized: Result<(CompressionStrategy, usize), _> =
                oxicode::decode_from_slice(&bytes);
            assert!(deserialized.is_ok());
            let (restored, _) = deserialized.expect("deserialization failed");
            assert_eq!(restored, strategy);
        }
    }

    // --- QuantizedTensor ---

    #[test]
    fn test_quantize_dequantize_roundtrip_8bit() {
        let data = vec![-10.0_f64, -5.0, 0.0, 2.5, 10.0];
        let q = QuantizedTensor::quantize(&data, 8).expect("quantize failed");
        assert_eq!(q.bits, 8);
        assert_eq!(q.len, data.len());
        assert_eq!(q.data.len(), data.len());

        let restored: Vec<f64> = q.dequantize().expect("dequantize failed");
        assert_eq!(restored.len(), data.len());
        for (orig, got) in data.iter().zip(restored.iter()) {
            assert!(
                (orig - got).abs() <= q.scale,
                "orig={orig} got={got} scale={}",
                q.scale
            );
        }
    }

    #[test]
    fn test_quantize_dequantize_roundtrip_4bit() {
        let data: Vec<f32> = (0..17).map(|i| (i as f32 - 8.0) * 0.5).collect();
        let q = QuantizedTensor::quantize(&data, 4).expect("quantize failed");
        assert_eq!(q.bits, 4);
        assert_eq!(q.len, data.len());
        // Two 4-bit codes per byte.
        assert_eq!(q.data.len(), data.len().div_ceil(2));

        let restored: Vec<f32> = q.dequantize().expect("dequantize failed");
        for (orig, got) in data.iter().zip(restored.iter()) {
            assert!(
                (orig - got).abs() <= q.scale as f32 + 1e-4,
                "orig={orig} got={got} scale={}",
                q.scale
            );
        }
    }

    #[test]
    fn test_quantize_empty_is_safe() {
        let data: Vec<f64> = vec![];
        let q = QuantizedTensor::quantize(&data, 8).expect("quantize failed");
        assert_eq!(q.len, 0);
        assert!(q.data.is_empty());
        let restored: Vec<f64> = q.dequantize().expect("dequantize failed");
        assert!(restored.is_empty());
    }

    #[test]
    fn test_quantize_constant_data_is_safe() {
        // max == min: scale must fall back rather than dividing by zero.
        let data = vec![3.0_f64; 10];
        let q = QuantizedTensor::quantize(&data, 8).expect("quantize failed");
        assert_eq!(q.scale, 1.0);
        let restored: Vec<f64> = q.dequantize().expect("dequantize failed");
        for got in restored {
            assert!((got - 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_quantize_rejects_unsupported_bit_width() {
        let data = vec![1.0, 2.0];
        let result = QuantizedTensor::quantize(&data, 6);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_tensor_serialization() {
        let data = vec![1.0_f64, -2.0, 3.5];
        let q = QuantizedTensor::quantize(&data, 8).expect("quantize failed");
        let bytes = oxicode::encode_to_vec(&q).expect("serialization failed");
        let (restored, _): (QuantizedTensor, usize) =
            oxicode::decode_from_slice(&bytes).expect("deserialization failed");
        assert_eq!(restored, q);
    }
}
