//! Regression tests for LANE W1-J: scattered silent-wrong fixes.
//!
//! Each test asserts the CORRECT (independently verified) behavior for a
//! bug that previously either (a) panicked, (b) silently produced a wrong
//! numeric result, or (c) silently ignored a parameter. Every test here
//! failed against the pre-fix code and passes against the fixed code.

use numrs2::new_modules::wavelets::mra::sureshrink_threshold;
use numrs2::prelude::*;

// ---------------------------------------------------------------------
// Item 9: Array::to_vec() must return elements in LOGICAL (row-major) order,
// not raw-memory order, for non-standard-layout arrays.
// ---------------------------------------------------------------------
mod to_vec_logical_order {
    use super::*;

    #[test]
    fn standard_layout_fast_path_unchanged() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
        assert!(a.array().is_standard_layout());
        assert_eq!(a.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn to_f_layout_to_vec_is_logical_row_major_order() {
        // A fresh C-contiguous (2,3) array is not F-contiguous, so
        // to_f_layout() takes the `reversed_axes()` branch: the physical
        // buffer is untouched ([1,2,3,4,5,6]) but shape becomes (3,2) and
        // strides become (1,3) (F-contiguous for that shape).
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
        let f = a.to_f_layout();
        assert_eq!(f.shape(), vec![3, 2]);
        // Logical row-major traversal of shape (3,2) with strides (1,3)
        // over buffer [1,2,3,4,5,6]:
        //   [0,0]=buf[0]=1  [0,1]=buf[3]=4
        //   [1,0]=buf[1]=2  [1,1]=buf[4]=5
        //   [2,0]=buf[2]=3  [2,1]=buf[5]=6
        // => [1,4,2,5,3,6]
        assert_eq!(f.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        // Cross-check against a direct logical iteration (ndarray's `iter`
        // walks via strides, independent of the raw buffer order), so this
        // isn't just re-asserting a hardcoded literal.
        let expected: Vec<f64> = f.array().iter().cloned().collect();
        assert_eq!(f.to_vec(), expected);
        // The buggy implementation returned the raw memory buffer, which
        // differs from the correct logical order for this array.
        let (raw_buf, _) = f.array().clone().into_raw_vec_and_offset();
        assert_ne!(
            f.to_vec(),
            raw_buf,
            "to_vec() must not equal the raw memory buffer for a non-standard-layout array"
        );
    }

    #[test]
    fn permuted_axes_to_vec_is_logical_row_major_order() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
        let permuted_nd = a.array().clone().permuted_axes(vec![1, 0]);
        let permuted = Array::from_ndarray(permuted_nd);
        assert_eq!(permuted.shape(), vec![3, 2]);
        assert_eq!(permuted.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let expected: Vec<f64> = permuted.array().iter().cloned().collect();
        assert_eq!(permuted.to_vec(), expected);
    }
}

// ---------------------------------------------------------------------
// Item 8: sureshrink_threshold must minimize the EXACT SURE risk
//   SURE(t) = n - 2*#{|x_i| <= t} + sum_i min(x_i^2, t^2)
// over the candidate thresholds t = |x_(i)|, not a formula that drops the
// capped-magnitude term for coefficients above threshold.
// ---------------------------------------------------------------------
mod sureshrink_exact_sure {
    use super::*;

    /// Independent, deliberately-unoptimized reference implementation of
    /// the textbook SURE formula, used to cross-check the crate's O(n log n)
    /// implementation. Ties are broken the same way as the implementation
    /// under test: scan candidates in ascending |x| order, keep the first
    /// strictly-smaller risk (`<`, not `<=`).
    fn brute_force_sure_threshold(data: &[f64]) -> f64 {
        let n = data.len() as f64;
        let mut candidates: Vec<f64> = data.iter().map(|x| x.abs()).collect();
        candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut best_t = 0.0;
        let mut min_risk = f64::INFINITY;
        for &t in &candidates {
            let count_le = data.iter().filter(|&&x| x.abs() <= t).count() as f64;
            let sum_min_sq: f64 = data.iter().map(|&x| (x * x).min(t * t)).sum();
            let risk = n - 2.0 * count_le + sum_min_sq;
            if risk < min_risk {
                min_risk = risk;
                best_t = t;
            }
        }
        best_t
    }

    #[test]
    fn matches_brute_force_on_mixed_signal_and_spike_vector() {
        // A handful of small "noise-like" coefficients plus two large
        // "signal" spikes. The true SURE-optimal threshold shrinks the
        // noise while preserving the spikes, so it does NOT sit at the
        // minimum magnitude (0.05) -- this vector is chosen specifically
        // to distinguish the correct minimizer from the old buggy formula.
        let data = vec![0.1, -0.2, 0.15, -0.05, 5.0, 0.3, -4.8, 0.25];

        let expected = brute_force_sure_threshold(&data);
        let actual = sureshrink_threshold(&data);

        assert!(
            (actual - expected).abs() < 1e-9,
            "sureshrink_threshold({data:?}) = {actual}, brute-force SURE minimizer = {expected}"
        );

        // The pre-fix formula's risk is strictly increasing in the
        // candidate index (it is missing the capped term for coefficients
        // above threshold), so it always returns the *minimum* absolute
        // coefficient magnitude regardless of the data. Confirm the fixed
        // function does not exhibit that degenerate signature here.
        let min_abs = data.iter().fold(f64::INFINITY, |acc, &x| acc.min(x.abs()));
        assert!(
            (actual - min_abs).abs() > 1e-6,
            "sureshrink_threshold returned the trivial min-magnitude threshold ({min_abs}); \
             the exact SURE minimizer for this vector should differ"
        );
    }

    #[test]
    fn empty_input_returns_zero() {
        assert_eq!(sureshrink_threshold(&[]), 0.0);
    }

    #[test]
    fn single_element_returns_its_magnitude() {
        // With n=1, SURE(t=|x|) = 1 - 2*1 + x^2 = x^2 - 1, the only candidate.
        assert!((sureshrink_threshold(&[-3.5]) - 3.5).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------
// Item 6: eigh/eigvalsh must honor `uplo`, symmetrizing from the named
// triangle instead of silently reading the full (possibly asymmetric)
// matrix.
// ---------------------------------------------------------------------
#[cfg(feature = "lapack")]
mod eigh_honors_uplo {
    use super::*;

    fn clean_symmetric() -> Array<f64> {
        // [[4, 1, 0],
        //  [1, 3, 2],
        //  [0, 2, 5]]
        Array::from_vec(vec![4.0, 1.0, 0.0, 1.0, 3.0, 2.0, 0.0, 2.0, 5.0]).reshape(&[3, 3])
    }

    fn sorted_eigenvalues(a: &Array<f64>, uplo: &str) -> Vec<f64> {
        let mut v = a.eigvalsh(uplo).expect("eigvalsh should succeed").to_vec();
        v.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    #[test]
    fn garbage_in_unused_upper_triangle_is_ignored_when_uplo_is_lower() {
        let clean = clean_symmetric();
        let mut garbage_upper = clean.clone();
        // Corrupt only the strictly-upper triangle; lower + diagonal stay clean.
        garbage_upper.set(&[0, 1], 999.0).expect("set");
        garbage_upper.set(&[0, 2], -777.0).expect("set");
        garbage_upper.set(&[1, 2], 555.0).expect("set");

        let clean_eigs = sorted_eigenvalues(&clean, "lower");
        let garbage_eigs = sorted_eigenvalues(&garbage_upper, "lower");

        for (c, g) in clean_eigs.iter().zip(garbage_eigs.iter()) {
            assert!(
                (c - g).abs() < 1e-8,
                "garbage in the unused (upper) triangle changed the eigenvalues under uplo=\"lower\": {:?} vs {:?}",
                clean_eigs, garbage_eigs
            );
        }
    }

    #[test]
    fn garbage_in_upper_triangle_changes_result_when_uplo_is_upper() {
        // The converse check: this is what actually proves `uplo` is read
        // rather than always defaulting to one triangle internally.
        let clean = clean_symmetric();
        let mut garbage_upper = clean.clone();
        garbage_upper.set(&[0, 1], 999.0).expect("set");
        garbage_upper.set(&[0, 2], -777.0).expect("set");
        garbage_upper.set(&[1, 2], 555.0).expect("set");

        let clean_eigs = sorted_eigenvalues(&clean, "upper");
        let garbage_eigs = sorted_eigenvalues(&garbage_upper, "upper");

        let max_diff = clean_eigs
            .iter()
            .zip(garbage_eigs.iter())
            .fold(0.0_f64, |acc, (c, g)| acc.max((c - g).abs()));
        assert!(
            max_diff > 1.0,
            "reading uplo=\"upper\" from a matrix with corrupted upper triangle should change \
             the spectrum substantially, got max diff {max_diff}"
        );
    }

    #[test]
    fn garbage_in_unused_lower_triangle_is_ignored_when_uplo_is_upper() {
        let clean = clean_symmetric();
        let mut garbage_lower = clean.clone();
        garbage_lower.set(&[1, 0], 321.0).expect("set");
        garbage_lower.set(&[2, 0], -654.0).expect("set");
        garbage_lower.set(&[2, 1], 111.0).expect("set");

        let clean_eigs = sorted_eigenvalues(&clean, "upper");
        let garbage_eigs = sorted_eigenvalues(&garbage_lower, "upper");
        for (c, g) in clean_eigs.iter().zip(garbage_eigs.iter()) {
            assert!(
                (c - g).abs() < 1e-8,
                "{:?} vs {:?}",
                clean_eigs,
                garbage_eigs
            );
        }
    }

    #[test]
    fn eigh_matrix_and_uplo_are_consistent() {
        // eigh (eigenvectors + eigenvalues) must show the same behavior as
        // eigvalsh, since it shares the same symmetrization helper.
        let clean = clean_symmetric();
        let mut garbage_upper = clean.clone();
        garbage_upper.set(&[0, 1], 999.0).expect("set");
        garbage_upper.set(&[0, 2], -777.0).expect("set");
        garbage_upper.set(&[1, 2], 555.0).expect("set");

        let (clean_eigs, _) = clean.eigh("lower").expect("eigh should succeed");
        let (garbage_eigs, _) = garbage_upper.eigh("lower").expect("eigh should succeed");
        let mut clean_v = clean_eigs.to_vec();
        let mut garbage_v = garbage_eigs.to_vec();
        clean_v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        garbage_v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        for (c, g) in clean_v.iter().zip(garbage_v.iter()) {
            assert!((c - g).abs() < 1e-8);
        }
    }

    #[test]
    fn invalid_uplo_is_an_error_not_a_panic() {
        let clean = clean_symmetric();
        assert!(clean.eigvalsh("sideways").is_err());
        assert!(clean.eigvalsh("").is_err());
    }
}

// ---------------------------------------------------------------------
// Item 4: einsum's general path must not panic on (a) an output index
// absent from every input operand, or (b) a candidate output axis whose
// size happens to be 1.
// ---------------------------------------------------------------------
mod einsum_general_path_panics {
    use super::*;
    use numrs2::linalg::tensor_ops::einsum;

    #[test]
    fn output_index_missing_from_inputs_is_an_error_not_a_panic() {
        // 'k' never appears in the input spec "ij" -- its size is
        // undefined, so this must be a clear error, not a HashMap-index
        // panic.
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]).reshape(&[2, 2]);
        let result = einsum("ij->ik", &[&a]);
        assert!(result.is_err(), "expected an error, got {:?}", result);
    }

    #[test]
    fn three_operand_contraction_with_output_dim_one_does_not_panic() {
        // Bypasses every fast-path special case (they only match 1 or 2
        // operand specs), forcing the general path. All three inputs are
        // length-1 vectors, so the sole output axis has size 1 -- this is
        // exactly the case the buggy `output_shape[0] != 1` guard mishandled.
        let a = Array::from_vec(vec![2.0]);
        let b = Array::from_vec(vec![3.0]);
        let c = Array::from_vec(vec![5.0]);
        let result = einsum("i,i,i->i", &[&a, &b, &c]).expect("einsum should not panic or error");
        assert_eq!(result.shape(), vec![1]);
        assert_eq!(result.to_vec(), vec![30.0]);
    }

    #[test]
    fn matmul_with_output_dim_one_via_general_path() {
        // A 3-operand elementwise-style contraction that still produces an
        // output axis of size 1, exercising the general path's index-value
        // population with a non-trivial (non-scalar) output shape.
        // "ij,jk,k->i" : contract j and k, output only i (size 1).
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]).reshape(&[1, 3]); // i=1,j=3
        let b = Array::from_vec(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).reshape(&[3, 3]); // j=3,k=3 (identity)
        let c = Array::from_vec(vec![10.0, 20.0, 30.0]); // k=3
        let result = einsum("ij,jk,k->i", &[&a, &b, &c]).expect("einsum should not panic");
        assert_eq!(result.shape(), vec![1]);
        // b is the identity, so this reduces to sum_j a[0,j] * c[j]
        // = 1*10 + 2*20 + 3*30 = 140
        assert_eq!(result.to_vec(), vec![140.0]);
    }
}

// ---------------------------------------------------------------------
// Item 7: incomplete_lu(a, fill_factor) must actually use fill_factor
// (ILUT-style fill control), instead of silently always computing ILU(0).
// ---------------------------------------------------------------------
mod incomplete_lu_honors_fill_factor {
    use super::*;

    /// A 5x5 "arrowhead" SPD matrix: a dense diagonal plus a dense first
    /// row/column. Gaussian elimination of the first pivot introduces
    /// genuine fill-in at every (i,j), i,j in 1..5, i != j position (all
    /// zero in the original matrix) -- exactly the scenario where ILU(0)
    /// (pattern-restricted) and ILUT (magnitude-and-fill-controlled) must
    /// diverge.
    fn arrowhead_spd(n: usize) -> SparseMatrix<f64> {
        let mut a = SparseMatrix::new(&[n, n]).expect("sparse matrix creation");
        for i in 0..n {
            a.set(i, i, 10.0).expect("set diag");
        }
        for i in 1..n {
            a.set(0, i, 2.0).expect("set arrow row");
            a.set(i, 0, 2.0).expect("set arrow col");
        }
        a
    }

    /// Dense Frobenius-norm residual ||L*U - A||_F for an n x n system.
    fn lu_residual_norm(
        l: &SparseMatrix<f64>,
        u: &SparseMatrix<f64>,
        a: &SparseMatrix<f64>,
        n: usize,
    ) -> f64 {
        let mut sum_sq = 0.0;
        for i in 0..n {
            for j in 0..n {
                let mut lu_ij = 0.0;
                for k in 0..=i.min(j) {
                    lu_ij += l.get(i, k).expect("get l") * u.get(k, j).expect("get u");
                }
                let diff = lu_ij - a.get(i, j).expect("get a");
                sum_sq += diff * diff;
            }
        }
        sum_sq.sqrt()
    }

    #[test]
    fn fill_factor_changes_the_factorization() {
        // The direct regression for "the parameter was ignored": two
        // different fill_factor values must produce different L/U.
        let a = arrowhead_spd(5);
        let (l_ilu0, u_ilu0) =
            SparseOpsAdvanced::incomplete_lu(&a, 1.0).expect("ILU(0) decomposition");
        let (l_ilut, u_ilut) =
            SparseOpsAdvanced::incomplete_lu(&a, 50.0).expect("ILUT decomposition");

        let mut any_l_diff = false;
        let mut any_u_diff = false;
        for i in 0..5 {
            for j in 0..5 {
                if (l_ilu0.get(i, j).expect("get") - l_ilut.get(i, j).expect("get")).abs() > 1e-12 {
                    any_l_diff = true;
                }
                if (u_ilu0.get(i, j).expect("get") - u_ilut.get(i, j).expect("get")).abs() > 1e-12 {
                    any_u_diff = true;
                }
            }
        }
        assert!(
            any_l_diff || any_u_diff,
            "incomplete_lu(a, 1.0) and incomplete_lu(a, 50.0) produced identical L/U; \
             fill_factor is being ignored"
        );
    }

    #[test]
    fn generous_fill_factor_reproduces_dense_lu_better_than_ilu0() {
        let a = arrowhead_spd(5);
        let (l_ilu0, u_ilu0) =
            SparseOpsAdvanced::incomplete_lu(&a, 1.0).expect("ILU(0) decomposition");
        let (l_ilut, u_ilut) =
            SparseOpsAdvanced::incomplete_lu(&a, 50.0).expect("ILUT decomposition");

        let residual_ilu0 = lu_residual_norm(&l_ilu0, &u_ilu0, &a, 5);
        let residual_ilut = lu_residual_norm(&l_ilut, &u_ilut, &a, 5);

        assert!(
            residual_ilut < residual_ilu0 - 1e-8,
            "ILUT with generous fill (residual {residual_ilut}) should reproduce A more \
             closely than ILU(0) (residual {residual_ilu0})"
        );
        // With a tiny drop tolerance and a very generous fill cap, ILUT on
        // this small 5x5 system should essentially recover exact LU.
        assert!(
            residual_ilut < 1e-8,
            "generous-fill ILUT residual {residual_ilut} should be ~exact on this small system"
        );
        // ILU(0) genuinely drops real fill-in on this matrix, so its
        // residual should be clearly nonzero.
        assert!(
            residual_ilu0 > 1e-3,
            "ILU(0) residual {residual_ilu0} should be well above zero on an arrowhead matrix"
        );
    }

    #[test]
    fn structural_triangularity_preserved_regardless_of_fill_factor() {
        let a = arrowhead_spd(5);
        for fill in [1.0, 5.0, 100.0] {
            let (l, u) = SparseOpsAdvanced::incomplete_lu(&a, fill).expect("ILU decomposition");
            for i in 0..5 {
                assert!((l.get(i, i).expect("get") - 1.0).abs() < 1e-12);
                for j in (i + 1)..5 {
                    assert_eq!(l.get(i, j).expect("get"), 0.0, "L not lower-triangular");
                    assert_eq!(u.get(j, i).expect("get"), 0.0, "U not upper-triangular");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Item 5: PostProcessor::evaluate_at_point must actually locate/interpolate
// Quad4 elements instead of silently skipping them, and its final error
// must not misattribute the failure to "triangular" elements.
// ---------------------------------------------------------------------
mod fem_quad4_point_location {
    use numrs2::new_modules::fem::{Mesh, PostProcessor};

    #[test]
    fn quad4_bilinear_interpolation_reproduces_an_affine_field_exactly() {
        // Bilinear shape functions are linearly complete: for any affine
        // field u = a + b*x + c*y, interpolating its NODAL values with the
        // Quad4 shape functions reproduces u exactly everywhere inside the
        // element (not just at nodes) -- this is what actually proves the
        // bilinear inverse map + interpolation is implemented correctly,
        // not merely that it returns `Ok(_)`.
        let mesh = Mesh::generate_2d_rectangular(0.0, 2.0, 0.0, 1.0, 2, 1)
            .expect("mesh generation should succeed");

        let (a, b, c) = (1.5, 2.0, -3.0);
        let field = |x: f64, y: f64| a + b * x + c * y;
        let solution: Vec<f64> = (0..mesh.num_nodes())
            .map(|n| {
                let coords = mesh.node_coords(n).expect("node coords");
                field(coords[0], coords[1])
            })
            .collect();

        // Interior point of the right-hand quad element, [1,2]x[0,1].
        let px = 1.7;
        let py = 0.3;
        let value = PostProcessor::evaluate_at_point(&mesh, &solution, &[px, py])
            .expect("point should be located inside a Quad4 element");
        let expected = field(px, py);
        assert!(
            (value - expected).abs() < 1e-9,
            "evaluate_at_point({px},{py}) = {value}, expected exact affine value {expected}"
        );

        // Also check a point in the left-hand quad element, [0,1]x[0,1].
        let (px2, py2) = (0.4, 0.8);
        let value2 = PostProcessor::evaluate_at_point(&mesh, &solution, &[px2, py2])
            .expect("point should be located inside a Quad4 element");
        assert!((value2 - field(px2, py2)).abs() < 1e-9);
    }

    #[test]
    fn point_outside_the_mesh_is_still_an_error_with_an_accurate_message() {
        let mesh = Mesh::generate_2d_rectangular(0.0, 1.0, 0.0, 1.0, 1, 1)
            .expect("mesh generation should succeed");
        let solution = vec![0.0; mesh.num_nodes()];

        let result = PostProcessor::evaluate_at_point(&mesh, &solution, &[100.0, 100.0]);
        let err = result.expect_err("point far outside the mesh must be an error");
        let msg = err.to_string();
        assert!(
            !msg.to_lowercase().contains("triangular"),
            "error message should not blame 'triangular' elements when the mesh is Quad4-only: {msg}"
        );
    }
}

// ---------------------------------------------------------------------
// Item 2: may_share_memory/shares_memory must reflect actual pointer-range
// overlap (self-share is true; distinct arrays -- including clones, since
// Array is deep-copy-only today -- are false); iscontiguous must reflect
// actual memory layout (to_f_layout produces a non-standard layout).
// ---------------------------------------------------------------------
mod contiguous_and_memory_sharing {
    use numrs2::array_ops::creation::{
        ascontiguousarray, iscontiguous, may_share_memory, shares_memory,
    };
    use numrs2::prelude::*;

    #[test]
    fn self_share_is_true_for_both_functions() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        assert!(
            may_share_memory(&a, &a),
            "an array must share memory with itself"
        );
        assert!(
            shares_memory(&a, &a),
            "an array must share memory with itself"
        );
    }

    #[test]
    fn distinct_arrays_never_share_memory() {
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array::from_vec(vec![4.0, 5.0, 6.0]);
        assert!(!may_share_memory(&a, &b));
        assert!(!shares_memory(&a, &b));
    }

    #[test]
    fn a_deep_clone_does_not_share_memory_with_the_original() {
        // Array::clone() is a deep copy: a fresh buffer is allocated, so
        // (unlike NumPy views) a clone must NOT be reported as sharing
        // memory with its source.
        let a = Array::from_vec(vec![1.0, 2.0, 3.0]);
        let c = a.clone();
        assert!(!may_share_memory(&a, &c));
        assert!(!shares_memory(&a, &c));
    }

    #[test]
    fn iscontiguous_reflects_actual_layout() {
        let c_layout = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(&[2, 3]);
        assert!(iscontiguous(&c_layout));

        // to_f_layout takes the reversed_axes() branch for a fresh
        // C-contiguous array, producing genuinely non-standard-layout
        // strides -- iscontiguous must now report false instead of the
        // old hardcoded true.
        let f_layout = c_layout.to_f_layout();
        assert!(
            !f_layout.array().is_standard_layout(),
            "sanity: to_f_layout should be non-standard here"
        );
        assert!(!iscontiguous(&f_layout));

        // ascontiguousarray must materialize an actually-contiguous copy,
        // not silently preserve the non-standard layout via a bare clone.
        let recontiguized = ascontiguousarray(&f_layout).expect("ascontiguousarray should succeed");
        assert!(iscontiguous(&recontiguized));
    }
}

// ---------------------------------------------------------------------
// Item 1: Var::fit's log-likelihood must propagate a Sigma-inversion
// failure as an error instead of silently substituting a
// diag(1/sigma_ii) approximation for Sigma^-1.
// ---------------------------------------------------------------------
mod var_log_likelihood_propagates_errors {
    use numrs2::new_modules::timeseries::Var;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn singular_residual_covariance_is_an_error_not_a_silent_fallback() {
        // A perfectly linear bivariate series: VAR(1) (with an intercept)
        // fits it essentially exactly, driving the OLS residuals -- and
        // hence the residual covariance Sigma = resid^T*resid/n -- to a
        // numerically singular matrix. Silently falling back to
        // diag(1/sigma_ii) here would report a log-likelihood/AIC/BIC
        // computed from a fabricated Sigma^-1 with no indication anything
        // was wrong; the fix must surface this as an error instead.
        let data = arr2(&[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]]);
        let var = Var::new(1);
        let result = var.fit(&data.view());
        assert!(
            result.is_err(),
            "fitting a VAR model on perfectly-collinear data (singular Sigma) should error, \
             not silently return a fabricated log-likelihood"
        );
    }

    #[test]
    fn well_conditioned_data_still_fits_successfully() {
        // Sanity check: the fix must not make ordinary, well-conditioned
        // inputs fail.
        let data = arr2(&[
            [1.0, 2.0],
            [1.7, 2.1],
            [1.9, 2.8],
            [2.8, 2.6],
            [2.6, 3.4],
            [3.5, 3.0],
            [3.4, 3.9],
            [4.3, 3.5],
            [4.0, 4.3],
            [4.9, 3.9],
        ]);
        let var = Var::new(1);
        let params = var
            .fit(&data.view())
            .expect("well-conditioned VAR fit should succeed");
        assert!(params.log_likelihood.is_finite());
        assert!(params.aic.is_finite());
        assert!(params.bic.is_finite());
    }
}
