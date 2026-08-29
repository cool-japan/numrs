//! Right-looking block Cholesky over column-block-cyclic matrices, and the
//! distributed triangular solves that go with it.
//!
//! # Why column-block-*cyclic*
//!
//! A right-looking factorization only ever touches the trailing submatrix,
//! which shrinks from the left. Under a plain contiguous block-column split
//! the rank owning the leftmost columns finishes first and then idles, and
//! by the last panel exactly one rank is doing all the work — the classic
//! way a "distributed" factorization ends up slower than a local one.
//! Dealing the `panel_width`-wide panels round robin ([`Layout::ColBlockCyclic`])
//! keeps every rank holding columns near the right edge for as long as
//! possible, so the trailing update stays spread out.
//!
//! # One panel step
//!
//! For panel `k` spanning global columns `[c0, c1)`:
//!
//! 1. its owner factors the `w x w` diagonal block `A[c0..c1, c0..c1]` in
//!    place, then runs a local `trsm` down its own column strip —
//!    `L[c1.., c0..c1] := A[c1.., c0..c1] * L_kk^-T`, one forward
//!    substitution per row. Both stay local because a rank owning a panel
//!    owns *every row* of it;
//! 2. the owner broadcasts rows `c0..n` of the finished panel — the
//!    sub-diagonal strip **and the diagonal block**. Carrying the diagonal
//!    block costs one `w x w` block per panel and buys two things: the
//!    message is a complete panel of `L` rather than a fragment, and every
//!    receiver indexes it as `global row - c0` with no special case at the
//!    top;
//! 3. every rank updates the panels it owns to the right of `k`:
//!    `A[d0.., d0..d1] -= L[d0.., c0..c1] * L[d0..d1, c0..c1]^T`. Rows above
//!    `d0` are strictly upper-triangular in the global matrix and are never
//!    read again, so they are skipped rather than computed and discarded.
//!
//! Nothing above ever reads the strict upper triangle, so it is simply
//! zeroed once at the end.
//!
//! # Failure is collective
//!
//! Only the owner can discover that a diagonal block is not positive
//! definite. Returning that error locally would strand every peer in the
//! panel broadcast, so the panel travels through
//! [`super::bcast_fallible_bytes`] and every rank raises the same
//! [`DistributedLinalgError::NotPositiveDefinite`] with the same pivot
//! index.

use super::matrix::{
    decode_matrix, decode_vec, encode_matrix, encode_slice, panel_columns, panel_count, DistFloat,
    DistributedMatrix, Layout,
};
use super::{bcast_fallible_bytes, DistTransport, DistributedLinalgError};
use scirs2_core::ndarray::{s, Array2};

/// Tag base for the panel broadcast, one tag per panel.
const TAG_PANEL: u64 = 0x2000;
/// Tag base for the forward-substitution broadcast, one tag per panel.
const TAG_FORWARD: u64 = 0x3000;
/// Tag base for the back-substitution broadcast, one tag per panel.
const TAG_BACK: u64 = 0x4000;

/// One rank's stake in the column-block-cyclic layout: the panels it owns,
/// their global column range, and where they sit in its local block.
#[derive(Debug, Clone, Copy)]
struct OwnedPanel {
    panel: usize,
    start: usize,
    end: usize,
    local_offset: usize,
}

/// Everything a Cholesky pass needs to know about its operand, validated
/// once. Each check reads only replicated facts (the global shape, the
/// layout, the world size), so it rejects on every rank at once — a
/// rank-local early return would deadlock the panel broadcasts.
struct Plan {
    n: usize,
    panel_width: usize,
    panels: usize,
    owned: Vec<OwnedPanel>,
}

fn plan<T: DistFloat, C: DistTransport + ?Sized>(
    a: &DistributedMatrix<T>,
    comm: &C,
    what: &str,
) -> Result<Plan, DistributedLinalgError> {
    let Layout::ColBlockCyclic { panel_width } = a.layout() else {
        return Err(DistributedLinalgError::UnsupportedShape(format!(
            "{what} requires Layout::ColBlockCyclic, got {:?}",
            a.layout()
        )));
    };
    if a.world_size() != comm.world_size() || a.rank() != comm.rank() {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "matrix was partitioned as rank {}/{} but the communicator is rank {}/{}",
            a.rank(),
            a.world_size(),
            comm.rank(),
            comm.world_size()
        )));
    }
    let (n, cols) = a.global_shape();
    if n != cols {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "{what} requires a square matrix, got {n}x{cols}"
        )));
    }
    if n == 0 || panel_width == 0 {
        return Err(DistributedLinalgError::InvalidDimensions {
            rows: n,
            cols: panel_width,
        });
    }

    let mut owned = Vec::new();
    for panel in a.owned_panels() {
        let (start, end) = panel_columns(n, panel_width, panel);
        let local_offset = a.local_panel_offset(panel).ok_or_else(|| {
            DistributedLinalgError::LinalgError(format!(
                "panel {panel} is listed as owned by rank {} but has no local offset",
                a.rank()
            ))
        })?;
        owned.push(OwnedPanel {
            panel,
            start,
            end,
            local_offset,
        });
    }

    Ok(Plan {
        n,
        panel_width,
        panels: panel_count(n, panel_width),
        owned,
    })
}

/// Factor the diagonal block of panel `[c0, c1)` in place, run the local
/// `trsm` down the rest of the owner's column strip, and return rows
/// `c0..n` of the finished panel encoded for the broadcast.
fn factor_panel<T: DistFloat>(
    local: &mut Array2<T>,
    n: usize,
    c0: usize,
    c1: usize,
    offset: usize,
) -> Result<Vec<u8>, DistributedLinalgError> {
    let width = c1 - c0;

    // Unblocked Cholesky of the w x w diagonal block, lower triangle only.
    for j in 0..width {
        let mut diagonal = local[[c0 + j, offset + j]];
        for t in 0..j {
            let value = local[[c0 + j, offset + t]];
            diagonal -= value * value;
        }
        // A NaN pivot fails both tests, so it is reported here rather than
        // propagating silently through the rest of the factorization.
        if diagonal <= T::zero() || !diagonal.is_finite() {
            return Err(DistributedLinalgError::NotPositiveDefinite { index: c0 + j });
        }
        let pivot = diagonal.sqrt();
        local[[c0 + j, offset + j]] = pivot;
        for i in (j + 1)..width {
            let mut acc = local[[c0 + i, offset + j]];
            for t in 0..j {
                acc -= local[[c0 + i, offset + t]] * local[[c0 + j, offset + t]];
            }
            local[[c0 + i, offset + j]] = acc / pivot;
        }
    }

    // trsm: solve `L_kk z = a_i` for each sub-diagonal row `a_i`, which is
    // the row-wise form of `L_sub L_kk^T = A_sub`. Forward substitution,
    // in place, one row at a time.
    for i in c1..n {
        for t in 0..width {
            let mut acc = local[[i, offset + t]];
            for u in 0..t {
                acc -= local[[c0 + t, offset + u]] * local[[i, offset + u]];
            }
            local[[i, offset + t]] = acc / local[[c0 + t, offset + t]];
        }
    }

    // Ship rows c0..n, the diagonal block included (see the module docs).
    // Its strict upper triangle is zeroed here so the message is a clean
    // panel of L rather than a mix of L and leftover A.
    let mut panel = Array2::<T>::zeros((n - c0, width));
    for i in c0..n {
        for t in 0..width {
            if i >= c0 + t {
                panel[[i - c0, t]] = local[[i, offset + t]];
            }
        }
    }
    Ok(encode_matrix(&panel.view()))
}

/// Factor `a = L L^T` with a right-looking block algorithm over
/// [`Layout::ColBlockCyclic`] panels.
///
/// `a` must be square, symmetric and positive definite; only its lower
/// triangle is read. The result carries `L` in the same layout, with the
/// strict upper triangle zeroed.
///
/// # Errors
///
/// - [`DistributedLinalgError::UnsupportedShape`] when the matrix is not
///   column-block-cyclic;
/// - [`DistributedLinalgError::NotPositiveDefinite`], on *every* rank, when
///   a diagonal block turns out not to be positive definite.
pub async fn block_cholesky<T: DistFloat, C: DistTransport + ?Sized>(
    a: &DistributedMatrix<T>,
    comm: &C,
) -> Result<DistributedMatrix<T>, DistributedLinalgError> {
    let Plan {
        n,
        panel_width,
        panels,
        owned,
    } = plan(a, comm, "block Cholesky")?;
    let ctx = comm.next_ctx();
    let rank = comm.rank();
    let mut l = a.clone();

    for k in 0..panels {
        let (c0, c1) = panel_columns(n, panel_width, k);
        let width = c1 - c0;
        let owner = l.panel_owner(k);
        let tag = TAG_PANEL + k as u64;

        let produced = if rank == owner {
            let offset = owned
                .iter()
                .find(|p| p.panel == k)
                .map(|p| p.local_offset)
                .ok_or_else(|| {
                    DistributedLinalgError::LinalgError(format!(
                        "rank {rank} owns panel {k} but cannot locate it locally"
                    ))
                })?;
            factor_panel(l.local_mut(), n, c0, c1, offset)
        } else {
            Ok(Vec::new())
        };

        let panel =
            decode_matrix::<T>(&bcast_fallible_bytes(comm, owner, ctx, tag, produced).await?)?;
        if panel.dim() != (n - c0, width) {
            return Err(DistributedLinalgError::DimensionMismatch(format!(
                "panel {k} arrived as {:?}, expected {:?}",
                panel.dim(),
                (n - c0, width)
            )));
        }

        // Trailing update, owned panels only. Rows above `start` are in the
        // strict upper triangle and nothing reads them again, so the update
        // begins at the diagonal block of each panel.
        for target in owned.iter().filter(|p| p.panel > k) {
            let below = panel.slice(s![(target.start - c0).., ..]);
            let diagonal_rows = panel.slice(s![(target.start - c0)..(target.end - c0), ..]);
            let update = below.dot(&diagonal_rows.t());
            let mut destination = l.local_mut().slice_mut(s![
                target.start..,
                target.local_offset..target.local_offset + (target.end - target.start)
            ]);
            destination -= &update;
        }
    }

    {
        let local = l.local_mut();
        for target in &owned {
            for t in 0..(target.end - target.start) {
                let column = target.start + t;
                for row in 0..column.min(n) {
                    local[[row, target.local_offset + t]] = T::zero();
                }
            }
        }
    }

    Ok(l)
}

/// Solve `L y = b` for a lower triangular column-block-cyclic `L`.
///
/// Right-looking, mirroring the factorization: panel `k`'s owner solves the
/// `w` unknowns its diagonal block covers and then applies that segment's
/// contribution to the rest of the right-hand side, because it is the only
/// rank holding those columns of `L`. The updated vector — `O(n)`, next to
/// nothing against the `O(n^2/p)` panel traffic of the factorization — is
/// broadcast so the next panel's owner starts from it.
pub async fn forward_substitution<T: DistFloat, C: DistTransport + ?Sized>(
    l: &DistributedMatrix<T>,
    b: &[T],
    comm: &C,
) -> Result<Vec<T>, DistributedLinalgError> {
    let Plan {
        n,
        panel_width,
        panels,
        owned,
    } = plan(l, comm, "forward substitution")?;
    if b.len() != n {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "right-hand side has {} entries, expected {n}",
            b.len()
        )));
    }
    let ctx = comm.next_ctx();
    let rank = comm.rank();
    let mut x = b.to_vec();

    for k in 0..panels {
        let (c0, c1) = panel_columns(n, panel_width, k);
        let owner = l.panel_owner(k);
        let tag = TAG_FORWARD + k as u64;

        let produced = if rank == owner {
            let offset = local_offset_of(&owned, k, rank)?;
            let local = l.local_view();
            let mut work = x.clone();
            let mut failure = None;
            for t in 0..(c1 - c0) {
                let mut acc = work[c0 + t];
                for u in 0..t {
                    acc -= local[[c0 + t, offset + u]] * work[c0 + u];
                }
                let pivot = local[[c0 + t, offset + t]];
                if pivot == T::zero() {
                    failure = Some(DistributedLinalgError::SingularMatrix);
                    break;
                }
                work[c0 + t] = acc / pivot;
            }
            for i in c1..n {
                let mut acc = work[i];
                for t in 0..(c1 - c0) {
                    acc -= local[[i, offset + t]] * work[c0 + t];
                }
                work[i] = acc;
            }
            match failure {
                Some(error) => Err(error),
                None => Ok(encode_slice(&work)),
            }
        } else {
            Ok(Vec::new())
        };

        x = decode_vec::<T>(&bcast_fallible_bytes(comm, owner, ctx, tag, produced).await?)?;
        if x.len() != n {
            return Err(DistributedLinalgError::DimensionMismatch(format!(
                "forward substitution step {k} produced {} entries, expected {n}",
                x.len()
            )));
        }
    }
    Ok(x)
}

/// Solve `L^T x = y` for a lower triangular column-block-cyclic `L`.
///
/// Left-looking and cheaper than its forward counterpart: panel `k`'s owner
/// already holds every `L[i, c]` the rows below need, so it folds them in
/// locally and broadcasts only the `w` freshly solved entries.
pub async fn back_substitution<T: DistFloat, C: DistTransport + ?Sized>(
    l: &DistributedMatrix<T>,
    y: &[T],
    comm: &C,
) -> Result<Vec<T>, DistributedLinalgError> {
    let Plan {
        n,
        panel_width,
        panels,
        owned,
    } = plan(l, comm, "back substitution")?;
    if y.len() != n {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "right-hand side has {} entries, expected {n}",
            y.len()
        )));
    }
    let ctx = comm.next_ctx();
    let rank = comm.rank();
    let mut x = vec![T::zero(); n];

    for k in (0..panels).rev() {
        let (c0, c1) = panel_columns(n, panel_width, k);
        let width = c1 - c0;
        let owner = l.panel_owner(k);
        let tag = TAG_BACK + k as u64;

        let produced = if rank == owner {
            let offset = local_offset_of(&owned, k, rank)?;
            let local = l.local_view();
            // Fold in the already-solved tail: row `c0 + t` of L^T is
            // column `c0 + t` of L, which this rank owns in full.
            let mut segment = vec![T::zero(); width];
            for (t, slot) in segment.iter_mut().enumerate() {
                let mut acc = y[c0 + t];
                for (i, xi) in x.iter().enumerate().take(n).skip(c1) {
                    acc -= local[[i, offset + t]] * *xi;
                }
                *slot = acc;
            }
            // Back substitution against L_kk^T, whose (t, s) entry is
            // L_kk[s, t] for s >= t.
            let mut failure = None;
            for t in (0..width).rev() {
                let mut acc = segment[t];
                for s in (t + 1)..width {
                    acc -= local[[c0 + s, offset + t]] * segment[s];
                }
                let pivot = local[[c0 + t, offset + t]];
                if pivot == T::zero() {
                    failure = Some(DistributedLinalgError::SingularMatrix);
                    break;
                }
                segment[t] = acc / pivot;
            }
            match failure {
                Some(error) => Err(error),
                None => Ok(encode_slice(&segment)),
            }
        } else {
            Ok(Vec::new())
        };

        let segment =
            decode_vec::<T>(&bcast_fallible_bytes(comm, owner, ctx, tag, produced).await?)?;
        if segment.len() != width {
            return Err(DistributedLinalgError::DimensionMismatch(format!(
                "back substitution step {k} produced {} entries, expected {width}",
                segment.len()
            )));
        }
        x.get_mut(c0..c1)
            .ok_or_else(|| {
                DistributedLinalgError::DimensionMismatch(format!(
                    "panel {k} covers columns {c0}..{c1} of an {n}-entry solution"
                ))
            })?
            .copy_from_slice(&segment);
    }
    Ok(x)
}

fn local_offset_of(
    owned: &[OwnedPanel],
    panel: usize,
    rank: u32,
) -> Result<usize, DistributedLinalgError> {
    owned
        .iter()
        .find(|p| p.panel == panel)
        .map(|p| p.local_offset)
        .ok_or_else(|| {
            DistributedLinalgError::LinalgError(format!(
                "rank {rank} owns panel {panel} but cannot locate it locally"
            ))
        })
}

/// Solve `A x = b` for a symmetric positive definite `A`: Cholesky, then
/// the two triangular solves.
///
/// `b` and the returned solution are replicated `n`-vectors — `O(n)` next to
/// the matrix's `O(n^2)`, so nothing is gained by splitting them.
pub async fn solve_spd<T: DistFloat, C: DistTransport + ?Sized>(
    a: &DistributedMatrix<T>,
    b: &[T],
    comm: &C,
) -> Result<Vec<T>, DistributedLinalgError> {
    let l = block_cholesky(a, comm).await?;
    let y = forward_substitution(&l, b, comm).await?;
    back_substitution(&l, &y, comm).await
}

#[cfg(test)]
mod tests {
    use super::super::matrix::testutil::{frobenius, spd_matrix};
    use super::*;
    use crate::distributed::linalg::LocalFabric;
    use crate::distributed::testing::{LocalCluster, RankContext};
    use std::sync::Arc;

    /// The whole point of the layout under test: with `n = 48` and
    /// `panel_width = 8` there are six panels over four ranks, so ranks 0
    /// and 1 hold two panels each and ranks 2 and 3 hold one — a
    /// deliberately uneven deal that a contiguous split would not produce.
    /// The ragged rows (`n = 50`) exercise a final panel narrower than
    /// `panel_width`, and `(12, 5, 4)` a world size larger than the panel
    /// count, where two ranks own nothing at all.
    const CASES: &[(usize, usize, u32)] = &[
        (48, 8, 4),
        (48, 8, 1),
        (48, 8, 2),
        (48, 8, 3),
        (50, 8, 4),
        (50, 7, 3),
        (12, 5, 4),
    ];

    fn case_matrix(n: usize) -> Array2<f64> {
        spd_matrix(n, 4242 + n as u64)
    }

    fn lower_triangular_only(l: &Array2<f64>) -> bool {
        let n = l.nrows();
        (0..n).all(|i| (i + 1..n).all(|j| l[[i, j]].abs() < 1e-15))
    }

    async fn gather_factor(panel_width: usize, world_size: u32, a: Array2<f64>) -> Array2<f64> {
        let fabric = LocalFabric::new(world_size);
        let results = LocalCluster::run(world_size, move |ctx: RankContext| {
            let fabric = Arc::clone(&fabric);
            let a = a.clone();
            async move {
                let comm = fabric.transport(ctx.rank)?;
                let da = DistributedMatrix::from_global(
                    Layout::ColBlockCyclic { panel_width },
                    &a.view(),
                    ctx.rank,
                    ctx.world_size,
                )?;
                let l = block_cholesky(&da, &comm).await?;
                Ok(l.gather_to_root(&comm, 0).await?)
            }
        })
        .await
        .expect("cluster run should succeed");
        results
            .first()
            .cloned()
            .flatten()
            .expect("root gathers the factor")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn block_cholesky_reconstructs_the_spd_matrix() {
        for &(n, panel_width, world_size) in CASES {
            let a = case_matrix(n);
            let l = gather_factor(panel_width, world_size, a.clone()).await;

            assert!(
                lower_triangular_only(&l),
                "n={n} w={panel_width} p={world_size}: factor is not lower triangular"
            );
            let diff = &l.dot(&l.t()) - &a;
            assert!(
                frobenius(&diff.view()) < 1e-9,
                "n={n} w={panel_width} p={world_size}: ||LL^T - A||_F = {}",
                frobenius(&diff.view())
            );
        }
    }

    /// The distributed factor must match the sequential one entry for entry
    /// (Cholesky has no sign freedom), which pins the panel ordering as well
    /// as the arithmetic.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distributed_factor_matches_scirs2_cholesky() {
        let a = case_matrix(48);
        let reference = scirs2_linalg::cholesky(&a.view(), None).expect("local cholesky");
        for world_size in 1..=4u32 {
            let l = gather_factor(8, world_size, a.clone()).await;
            let diff = &l - &reference;
            assert!(
                frobenius(&diff.view()) < 1e-9,
                "p={world_size}: ||L_dist - L_local||_F = {}",
                frobenius(&diff.view())
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn solve_spd_recovers_the_right_hand_side() {
        for &(n, panel_width, world_size) in CASES {
            let a = case_matrix(n);
            let x_true: Vec<f64> = (0..n).map(|i| 0.25 + (i % 7) as f64).collect();
            let b: Vec<f64> = a
                .rows()
                .into_iter()
                .map(|row| row.iter().zip(x_true.iter()).map(|(v, x)| v * x).sum())
                .collect();

            let fabric = LocalFabric::new(world_size);
            let (a_ref, b_ref) = (a.clone(), b.clone());
            let results = LocalCluster::run(world_size, move |ctx: RankContext| {
                let fabric = Arc::clone(&fabric);
                let (a, b) = (a_ref.clone(), b_ref.clone());
                async move {
                    let comm = fabric.transport(ctx.rank)?;
                    let da = DistributedMatrix::from_global(
                        Layout::ColBlockCyclic { panel_width },
                        &a.view(),
                        ctx.rank,
                        ctx.world_size,
                    )?;
                    Ok(solve_spd(&da, &b, &comm).await?)
                }
            })
            .await
            .expect("cluster run should succeed");

            for (rank, x) in results.iter().enumerate() {
                assert_eq!(x.len(), n);
                let error = x
                    .iter()
                    .zip(x_true.iter())
                    .map(|(got, want)| (got - want).abs())
                    .fold(0.0_f64, f64::max);
                assert!(
                    error < 1e-9,
                    "n={n} w={panel_width} p={world_size} rank={rank}: max |x - x_true| = {error}"
                );
            }
        }
    }

    /// A non-positive-definite pivot must reach every rank, not just the one
    /// that found it — otherwise the peers sit in the panel broadcast until
    /// the cluster timeout fires.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn indefinite_matrix_fails_on_every_rank() {
        let mut a = spd_matrix(16, 5);
        // Drive one late diagonal entry negative; the leading blocks stay
        // fine, so the failure surfaces mid-factorization on a rank that is
        // not rank 0.
        a[[11, 11]] = -1.0;
        let fabric = LocalFabric::new(4);
        let results = LocalCluster::run(4, move |ctx: RankContext| {
            let fabric = Arc::clone(&fabric);
            let a = a.clone();
            async move {
                let comm = fabric.transport(ctx.rank)?;
                let da = DistributedMatrix::from_global(
                    Layout::ColBlockCyclic { panel_width: 4 },
                    &a.view(),
                    ctx.rank,
                    ctx.world_size,
                )?;
                Ok(match block_cholesky(&da, &comm).await {
                    Err(DistributedLinalgError::NotPositiveDefinite { index }) => Some(index),
                    _ => None,
                })
            }
        })
        .await
        .expect("cluster run should succeed");

        let first = results.first().copied().flatten();
        assert!(first.is_some(), "rank 0 did not report a bad pivot");
        for (rank, index) in results.iter().enumerate() {
            assert_eq!(*index, first, "rank {rank} disagrees about the pivot");
        }
    }

    /// The factorization over the real network stack rather than the
    /// in-process fabric: same panels, same broadcasts, real TCP links and
    /// framing underneath. Nothing else in this module crosses a socket.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn block_cholesky_runs_over_the_endpoint_transport() {
        use crate::distributed::linalg::EndpointTransport;
        use crate::distributed::testing::ClusterNode;

        let a = spd_matrix(20, 1313);
        let reference = a.clone();
        let results = LocalCluster::run_connected(2, move |node: ClusterNode| {
            let a = reference.clone();
            async move {
                let comm = EndpointTransport::new(node.endpoint);
                let da = DistributedMatrix::from_global(
                    Layout::ColBlockCyclic { panel_width: 4 },
                    &a.view(),
                    node.rank,
                    node.world_size,
                )?;
                let l = block_cholesky(&da, &comm).await?;
                Ok(l.gather_to_root(&comm, 0).await?)
            }
        })
        .await
        .expect("connected cluster run should succeed");

        let l = results
            .first()
            .cloned()
            .flatten()
            .expect("root gathers the factor");
        assert!(lower_triangular_only(&l), "over TCP: not lower triangular");
        let diff = &l.dot(&l.t()) - &a;
        assert!(
            frobenius(&diff.view()) < 1e-9,
            "over TCP: ||LL^T - A||_F = {}",
            frobenius(&diff.view())
        );
    }

    /// The `f32` instantiation of [`DistFloat`], whose wire encoding is four
    /// bytes per element instead of eight.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn block_cholesky_factors_an_f32_matrix() {
        let wide = spd_matrix(16, 1717);
        let a = wide.mapv(|v| v as f32);
        let fabric = LocalFabric::new(3);
        let reference = a.clone();
        let results = LocalCluster::run(3, move |ctx: RankContext| {
            let fabric = Arc::clone(&fabric);
            let a = reference.clone();
            async move {
                let comm = fabric.transport(ctx.rank)?;
                let da = DistributedMatrix::from_global(
                    Layout::ColBlockCyclic { panel_width: 4 },
                    &a.view(),
                    ctx.rank,
                    ctx.world_size,
                )?;
                let l = block_cholesky(&da, &comm).await?;
                Ok(l.gather_to_root(&comm, 0).await?)
            }
        })
        .await
        .expect("cluster run should succeed");

        let l = results
            .first()
            .cloned()
            .flatten()
            .expect("root gathers the factor");
        let diff = (&l.dot(&l.t()) - &a).mapv(f64::from);
        assert!(
            frobenius(&diff.view()) < 1e-3,
            "f32: ||LL^T - A||_F = {}",
            frobenius(&diff.view())
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn row_block_input_is_refused() {
        let fabric = LocalFabric::new(1);
        let results = LocalCluster::run(1, move |ctx: RankContext| {
            let fabric = Arc::clone(&fabric);
            async move {
                let comm = fabric.transport(ctx.rank)?;
                let a = spd_matrix(8, 3);
                let da = DistributedMatrix::from_global(
                    Layout::RowBlock,
                    &a.view(),
                    ctx.rank,
                    ctx.world_size,
                )?;
                Ok(matches!(
                    block_cholesky(&da, &comm).await,
                    Err(DistributedLinalgError::UnsupportedShape(_))
                ))
            }
        })
        .await
        .expect("cluster run should succeed");
        assert_eq!(results.first(), Some(&true));
    }
}
