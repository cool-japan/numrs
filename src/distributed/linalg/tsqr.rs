//! Tall-Skinny QR (TSQR): a communication-avoiding QR factorization for
//! matrices distributed by row blocks — many rows, comparatively few
//! columns.
//!
//! # The tree
//!
//! Each rank QR-factors its own row block with [`super::householder`]
//! reflectors, producing a small `n x n` `R`. Those `R`s are then combined
//! pairwise up a binary reduction tree: at level `l` a rank whose id is a
//! multiple of `2^(l+1)` receives its partner's `R`, stacks it under its own
//! into a `2n x n` block, and factors *that*. Rank 0 ends up holding the
//! global `R`, after `ceil(log2(p))` messages of `n^2` elements each —
//! instead of one round trip per Householder step across the whole matrix,
//! which is what makes the algorithm communication-*avoiding*.
//!
//! [`TsqrLevel`] records what each rank did at each level, and every factor
//! along the way is kept. That is the whole point: `Q` is never formed, it
//! is *replayed*.
//!
//! # Why every factor has to be kept
//!
//! Writing `A` as row blocks `A_i = Q_i R_i` and stacking, the factorization
//! telescopes into
//!
//! ```text
//! A = D_0 D_1 ... D_{L-1} R,   D_l = blkdiag(the level-l factors)
//! ```
//!
//! so `Q = D_0 D_1 ... D_{L-1}` and therefore
//!
//! ```text
//! Q^T B = D_{L-1}^T ... D_1^T D_0^T B      (leaves first, then up the tree)
//! Q C   = D_0 D_1 ... D_{L-1} C            (root first, then down the tree)
//! ```
//!
//! [`TsqrFactorization::apply_qt`] walks the stored tree bottom-up and
//! [`TsqrFactorization::apply_q`] mirrors it top-down. **Neither is the sum
//! of the per-rank `Q_i^T B_i`** — that is the classic TSQR trap. The leaf
//! factors alone are `D_0`; dropping `D_1 ... D_{L-1}` silently returns
//! something that is not orthogonal to anything, and it *looks* plausible
//! because the shapes all line up. For `p = 1` the two happen to coincide,
//! which is exactly how such a bug survives its first test.
//!
//! # Wide row blocks
//!
//! Every leaf needs `m_i >= n` for a thin QR to exist. When some rank's
//! block is wider than it is tall, this module returns
//! [`DistributedLinalgError::UnsupportedShape`] rather than approximating:
//! the correct algorithm for that regime is communication-avoiding QR
//! (CAQR), which factors each *column panel* with its own TSQR instead of
//! treating a rank's whole block as one leaf. The precondition is checked
//! from the global shape alone, so it fails on every rank at once and never
//! deadlocks the tree.

use super::householder::HouseholderQr;
use super::matrix::{
    block_len, broadcast_matrix, decode_matrix, encode_matrix, DistFloat, DistributedMatrix, Layout,
};
use super::{DistTransport, DistributedLinalgError};
use scirs2_core::ndarray::{s, Array2, ArrayView2};

/// The rank the reduction tree converges on.
///
/// The pairing rule makes rank 0 a receiver at every level, so it is the
/// root by construction rather than by choice.
pub const ROOT: u32 = 0;

/// Tag base for the `R` reduction, one tag per tree level.
const TAG_R_REDUCE: u64 = 0x1000;
/// Tag for replicating the final `R`.
const TAG_R_BCAST: u64 = 0x1100;
/// Tag base for [`TsqrFactorization::apply_qt`]'s upward leg.
const TAG_APPLY_QT: u64 = 0x1200;
/// Tag base for [`TsqrFactorization::apply_q`]'s downward leg.
const TAG_APPLY_Q: u64 = 0x1300;

fn inconsistent(what: &str) -> DistributedLinalgError {
    DistributedLinalgError::LinalgError(format!(
        "TSQR tree state is inconsistent: {what} (this is a bug in the tree walk, not in the input)"
    ))
}

/// Number of levels in the reduction tree over `world_size` ranks: the
/// smallest `L` with `2^L >= world_size` (so a single rank has none).
pub fn tree_levels(world_size: u32) -> u32 {
    let mut levels = 0u32;
    while (1u64 << levels) < u64::from(world_size) {
        levels += 1;
    }
    levels
}

/// What `rank` does at `level`: `Some((true, partner))` to receive from
/// `partner`, `Some((false, partner))` to send to it, `None` to sit the
/// level out.
///
/// A rank is a receiver exactly while its low `level + 1` bits are zero, so
/// it sends exactly once — at the level of its lowest set bit — and is
/// inactive from then on. Arithmetic is in `u64` because a `u32` world size
/// reaches level 31, where `1u32 << 32` would overflow.
fn level_pairing(rank: u32, world_size: u32, level: u32) -> Option<(bool, u32)> {
    let distance = 1u64 << level;
    let span = distance << 1;
    let rank = u64::from(rank);
    if rank % span == 0 {
        let partner = rank + distance;
        (partner < u64::from(world_size)).then_some((true, partner as u32))
    } else if rank % span == distance {
        Some((false, (rank - distance) as u32))
    } else {
        None
    }
}

/// One rank's part in one level of the reduction tree.
///
/// The variant also fixes what the rank does on the way back *down*: a
/// `Received` node splits its block and returns the lower half to the
/// partner it took it from, and a `Sent` node collects that half back.
#[derive(Debug, Clone)]
pub enum TsqrLevel<T> {
    /// This rank took `partner`'s `R` and factored the stacked
    /// `[R_self; R_partner]`. The `2n x n` factor is kept so both
    /// directions of `Q` can be replayed through it.
    Received {
        /// The rank whose `R` was stacked underneath this one's.
        partner: u32,
        /// The factorization of the stacked pair.
        factor: HouseholderQr<T>,
    },
    /// This rank handed its `R` to `partner` and takes no further part in
    /// the reduction.
    Sent {
        /// The rank that adopted this one's `R`.
        partner: u32,
    },
    /// This rank passes through untouched: either it had no partner at this
    /// level (an odd world size), or it has already sent.
    Idle,
}

/// A distributed QR factorization stored as a tree of implicit reflectors.
///
/// Produced by [`tsqr()`]. `R` is replicated on every rank; `Q` exists only as
/// the stored factors and is applied with [`Self::apply_q`] /
/// [`Self::apply_qt`].
#[derive(Debug, Clone)]
pub struct TsqrFactorization<T> {
    leaf: HouseholderQr<T>,
    levels: Vec<TsqrLevel<T>>,
    r: Array2<T>,
    global_rows: usize,
    cols: usize,
    rank: u32,
    world_size: u32,
}

impl<T: DistFloat> TsqrFactorization<T> {
    /// The upper triangular `n x n` factor, identical on every rank.
    pub fn r(&self) -> ArrayView2<'_, T> {
        self.r.view()
    }

    /// Global `(rows, cols)` of the factored matrix.
    pub fn global_shape(&self) -> (usize, usize) {
        (self.global_rows, self.cols)
    }

    /// This rank's leaf factorization of its own row block.
    pub fn leaf(&self) -> &HouseholderQr<T> {
        &self.leaf
    }

    /// This rank's part in each tree level, lowest level first.
    pub fn levels(&self) -> &[TsqrLevel<T>] {
        &self.levels
    }

    /// Rows of this rank's row block.
    pub fn local_rows(&self) -> usize {
        self.leaf.nrows()
    }

    fn check_comm<C: DistTransport + ?Sized>(
        &self,
        comm: &C,
    ) -> Result<(), DistributedLinalgError> {
        if comm.rank() != self.rank || comm.world_size() != self.world_size {
            return Err(DistributedLinalgError::DimensionMismatch(format!(
                "factorization belongs to rank {}/{} but was applied with rank {}/{}",
                self.rank,
                self.world_size,
                comm.rank(),
                comm.world_size()
            )));
        }
        Ok(())
    }

    /// `Q_thin^T B` for a row-block distributed `B`.
    ///
    /// Each rank supplies its own `m_i x k` block; the `n x k` result lands
    /// on [`ROOT`] and every other rank gets `None`.
    ///
    /// The walk is bottom-up. The leaf contributes the top `n` rows of the
    /// full `Q_i^T B_i` (the thin-`Q` identity in [`super::householder`]),
    /// and each subsequent level stacks the partner's `n x k` carry under
    /// its own and pushes the pair through that level's stored factor. See
    /// the module docs for why the leaf step alone is not the answer.
    ///
    /// Every rank must pass the same number of columns. A disagreement is
    /// caught where the mismatched carries meet, as a
    /// [`DistributedLinalgError::DimensionMismatch`] on the receiving rank —
    /// so it is reported rather than silently mis-stacked, but it is not
    /// detected collectively and the sending rank may be left waiting.
    pub async fn apply_qt<C: DistTransport + ?Sized>(
        &self,
        comm: &C,
        b_local: ArrayView2<'_, T>,
    ) -> Result<Option<Array2<T>>, DistributedLinalgError> {
        self.check_comm(comm)?;
        let ctx = comm.next_ctx();
        let n = self.cols;
        let k = b_local.ncols();
        if b_local.nrows() != self.leaf.nrows() {
            return Err(DistributedLinalgError::DimensionMismatch(format!(
                "rank {} supplied a {:?} block of B but holds {} rows of A",
                self.rank,
                b_local.dim(),
                self.leaf.nrows()
            )));
        }

        let mut work = b_local.to_owned();
        self.leaf.apply_qt_in_place(&mut work)?;
        let mut carried = Some(work.slice(s![..n, ..]).to_owned());

        for (level, state) in self.levels.iter().enumerate() {
            let tag = TAG_APPLY_QT + level as u64;
            match state {
                TsqrLevel::Received { partner, factor } => {
                    let own = carried
                        .take()
                        .ok_or_else(|| inconsistent("receiver has no carry"))?;
                    let received = decode_matrix::<T>(&comm.recv_bytes(*partner, ctx, tag).await?)?;
                    if received.dim() != (n, k) {
                        return Err(DistributedLinalgError::DimensionMismatch(format!(
                            "rank {partner} sent a {:?} carry at level {level}, expected {:?}",
                            received.dim(),
                            (n, k)
                        )));
                    }
                    let mut stacked = Array2::<T>::zeros((2 * n, k));
                    stacked.slice_mut(s![..n, ..]).assign(&own);
                    stacked.slice_mut(s![n.., ..]).assign(&received);
                    factor.apply_qt_in_place(&mut stacked)?;
                    carried = Some(stacked.slice(s![..n, ..]).to_owned());
                }
                TsqrLevel::Sent { partner } => {
                    let own = carried
                        .take()
                        .ok_or_else(|| inconsistent("sender has no carry"))?;
                    comm.send_bytes(*partner, ctx, tag, &encode_matrix(&own.view()))
                        .await?;
                }
                TsqrLevel::Idle => {}
            }
        }
        Ok(carried)
    }

    /// `Q_thin * C` for an `n x k` `C`, returning this rank's `m_i x k` row
    /// block of the product.
    ///
    /// Only [`ROOT`]'s `c` is read, so callers holding a replicated `C` may
    /// pass `Some` everywhere. The walk mirrors [`Self::apply_qt`]: from the
    /// top level down, each stored factor expands one `n x k` carry into
    /// `2n x k`, keeps the top half and returns the bottom half to the
    /// partner that contributed it, until every rank has a carry to push
    /// through its leaf. `k` is discovered from the first block a rank
    /// receives, so no extra broadcast is needed to agree on it.
    ///
    /// # Panics and hangs
    ///
    /// Never panics. The shape check on `c` is, unavoidably, *rank-local* —
    /// only the root holds `c` — so supplying `None` or a wrong-height `c`
    /// on the root returns an error there while the peers stay blocked
    /// waiting for their share of the descent. That is a caller-contract
    /// violation, not a data-dependent failure: every entry point in
    /// [`super::decomp`] passes a factor whose height it derived from this
    /// factorization, so the condition cannot arise from input values.
    pub async fn apply_q<C: DistTransport + ?Sized>(
        &self,
        comm: &C,
        c: Option<ArrayView2<'_, T>>,
    ) -> Result<Array2<T>, DistributedLinalgError> {
        self.check_comm(comm)?;
        let ctx = comm.next_ctx();
        let n = self.cols;

        let mut carried = if self.rank == ROOT {
            let c = c.ok_or_else(|| {
                DistributedLinalgError::DimensionMismatch(
                    "apply_q root must supply the n x k factor to rotate".to_string(),
                )
            })?;
            if c.nrows() != n {
                return Err(DistributedLinalgError::DimensionMismatch(format!(
                    "apply_q was given a {:?} factor, expected {n} rows",
                    c.dim()
                )));
            }
            Some(c.to_owned())
        } else {
            None
        };

        for level in (0..self.levels.len()).rev() {
            let tag = TAG_APPLY_Q + level as u64;
            let state = self
                .levels
                .get(level)
                .ok_or_else(|| inconsistent("level index out of range"))?;
            match state {
                TsqrLevel::Received { partner, factor } => {
                    let own = carried
                        .take()
                        .ok_or_else(|| inconsistent("receiver has no carry"))?;
                    let k = own.ncols();
                    let mut stacked = Array2::<T>::zeros((2 * n, k));
                    stacked.slice_mut(s![..n, ..]).assign(&own);
                    factor.apply_q_in_place(&mut stacked)?;
                    let lower = stacked.slice(s![n.., ..]).to_owned();
                    comm.send_bytes(*partner, ctx, tag, &encode_matrix(&lower.view()))
                        .await?;
                    carried = Some(stacked.slice(s![..n, ..]).to_owned());
                }
                TsqrLevel::Sent { partner } => {
                    carried = Some(decode_matrix::<T>(
                        &comm.recv_bytes(*partner, ctx, tag).await?,
                    )?);
                }
                TsqrLevel::Idle => {}
            }
        }

        let top = carried.ok_or_else(|| inconsistent("no carry reached the leaf"))?;
        if top.nrows() != n {
            return Err(DistributedLinalgError::DimensionMismatch(format!(
                "carry reaching the leaf has {} rows, expected {n}",
                top.nrows()
            )));
        }
        let mut out = Array2::<T>::zeros((self.leaf.nrows(), top.ncols()));
        out.slice_mut(s![..n, ..]).assign(&top);
        self.leaf.apply_q_in_place(&mut out)?;
        Ok(out)
    }
}

/// Factor a row-block distributed `A` as `Q R` with TSQR.
///
/// `R` comes back replicated on every rank; `Q` stays implicit inside the
/// returned [`TsqrFactorization`].
///
/// # Errors
///
/// - [`DistributedLinalgError::UnsupportedShape`] when the matrix is not in
///   [`Layout::RowBlock`], or when any rank's row block is shorter than the
///   matrix is wide (see the module docs on CAQR).
/// - [`DistributedLinalgError::DimensionMismatch`] when the matrix was
///   partitioned for a different rank or world size than `comm` has.
pub async fn tsqr<T: DistFloat, C: DistTransport + ?Sized>(
    a: &DistributedMatrix<T>,
    comm: &C,
) -> Result<TsqrFactorization<T>, DistributedLinalgError> {
    let size = comm.world_size();
    let rank = comm.rank();
    if a.layout() != Layout::RowBlock {
        return Err(DistributedLinalgError::UnsupportedShape(format!(
            "TSQR requires Layout::RowBlock, got {:?}",
            a.layout()
        )));
    }
    if a.world_size() != size || a.rank() != rank {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "matrix was partitioned as rank {}/{} but the communicator is rank {rank}/{size}",
            a.rank(),
            a.world_size()
        )));
    }
    let (m, n) = a.global_shape();
    if m == 0 || n == 0 {
        return Err(DistributedLinalgError::InvalidDimensions { rows: m, cols: n });
    }

    // Every precondition below is evaluated from replicated data — the
    // global shape and the world size — so it either rejects on every rank
    // or on none. Never make one of these rank-local: a rank returning early
    // while its peers wait in `recv_bytes` deadlocks the tree *and*
    // desynchronizes the `next_ctx()` lockstep the whole module rides on.
    for peer in 0..size {
        let rows = block_len(m, size, peer);
        if rows < n {
            return Err(DistributedLinalgError::UnsupportedShape(format!(
                "TSQR needs every row block at least as tall as the matrix is wide, but rank \
                 {peer} of {size} holds {rows} rows of a {m}x{n} matrix; that regime needs \
                 communication-avoiding QR (CAQR), which panels the columns, not a tree of thin \
                 local QRs"
            )));
        }
    }

    let ctx = comm.next_ctx();
    let leaf = HouseholderQr::factor(a.local_view().to_owned())?;
    let mut carried = Some(leaf.r());
    let level_count = tree_levels(size);
    let mut levels = Vec::with_capacity(level_count as usize);

    for level in 0..level_count {
        let tag = TAG_R_REDUCE + u64::from(level);
        match level_pairing(rank, size, level) {
            Some((true, partner)) => {
                let own = carried
                    .take()
                    .ok_or_else(|| inconsistent("receiver has no R"))?;
                let received = decode_matrix::<T>(&comm.recv_bytes(partner, ctx, tag).await?)?;
                if received.dim() != (n, n) {
                    return Err(DistributedLinalgError::DimensionMismatch(format!(
                        "rank {partner} sent a {:?} R at level {level}, expected {:?}",
                        received.dim(),
                        (n, n)
                    )));
                }
                let mut stacked = Array2::<T>::zeros((2 * n, n));
                stacked.slice_mut(s![..n, ..]).assign(&own);
                stacked.slice_mut(s![n.., ..]).assign(&received);
                let factor = HouseholderQr::factor(stacked)?;
                carried = Some(factor.r());
                levels.push(TsqrLevel::Received { partner, factor });
            }
            Some((false, partner)) => {
                let own = carried
                    .take()
                    .ok_or_else(|| inconsistent("sender has no R"))?;
                comm.send_bytes(partner, ctx, tag, &encode_matrix(&own.view()))
                    .await?;
                levels.push(TsqrLevel::Sent { partner });
            }
            None => levels.push(TsqrLevel::Idle),
        }
    }

    // Replicate R: it is `n x n`, and having it everywhere lets a caller run
    // the triangular solve behind a least-squares fit without a round trip.
    let r = broadcast_matrix(comm, ROOT, ctx, TAG_R_BCAST, carried.as_ref()).await?;

    Ok(TsqrFactorization {
        leaf,
        levels,
        r,
        global_rows: m,
        cols: n,
        rank,
        world_size: size,
    })
}

#[cfg(test)]
mod tests {
    use super::super::matrix::testutil::{deterministic_matrix, frobenius};
    use super::*;
    use crate::distributed::linalg::LocalFabric;
    use crate::distributed::testing::{LocalCluster, RankContext};
    use std::sync::Arc;

    fn identity(n: usize) -> Array2<f64> {
        let mut eye = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            eye[[i, i]] = 1.0;
        }
        eye
    }

    #[test]
    fn tree_shape_matches_the_world_size() {
        assert_eq!(tree_levels(1), 0);
        assert_eq!(tree_levels(2), 1);
        assert_eq!(tree_levels(3), 2);
        assert_eq!(tree_levels(4), 2);
        assert_eq!(tree_levels(5), 3);
        assert_eq!(tree_levels(8), 3);
        assert_eq!(tree_levels(9), 4);
    }

    #[test]
    fn every_rank_sends_exactly_once_and_rank_zero_never_does() {
        for world_size in 1..=9u32 {
            let levels = tree_levels(world_size);
            for rank in 0..world_size {
                let sends = (0..levels)
                    .filter(|&level| {
                        matches!(level_pairing(rank, world_size, level), Some((false, _)))
                    })
                    .count();
                if rank == ROOT {
                    assert_eq!(sends, 0, "world_size={world_size}: root must never send");
                } else {
                    assert_eq!(sends, 1, "world_size={world_size} rank={rank}");
                }
            }
            // Every send is matched by exactly one receive.
            for level in 0..levels {
                for rank in 0..world_size {
                    if let Some((false, partner)) = level_pairing(rank, world_size, level) {
                        assert_eq!(
                            level_pairing(partner, world_size, level),
                            Some((true, rank)),
                            "world_size={world_size} level={level} rank={rank}"
                        );
                    }
                }
            }
        }
    }

    /// `A = Q R` with `Q` orthonormal, across every world size up to 8.
    ///
    /// The range matters more than it looks. Powers of two give a perfect
    /// binary tree, where a mistake in the pairing rule can cancel out;
    /// 3, 5, 6 and 7 are the shapes with `Idle` levels and partners that run
    /// off the end of the world, and 5 and 7 are the first sizes needing a
    /// third tree level. The splits are ragged too (32 over 3 is 11, 11, 10;
    /// over 7 it is 5, 5, 4, 4, 4, 4, 4).
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tsqr_reconstructs_a_tall_matrix_and_is_orthonormal() {
        for world_size in 1..=8u32 {
            let a = deterministic_matrix(32, 4, 4242);
            let fabric = LocalFabric::new(world_size);
            let reference = a.clone();
            let results = LocalCluster::run(world_size, move |ctx: RankContext| {
                let fabric = Arc::clone(&fabric);
                let a = reference.clone();
                async move {
                    let comm = fabric.transport(ctx.rank)?;
                    let da = DistributedMatrix::from_global(
                        Layout::RowBlock,
                        &a.view(),
                        ctx.rank,
                        ctx.world_size,
                    )?;
                    let f = tsqr(&da, &comm).await?;
                    let eye = identity(4);
                    let q_local = f.apply_q(&comm, Some(eye.view())).await?;
                    let dq = DistributedMatrix::from_local(
                        Layout::RowBlock,
                        32,
                        4,
                        ctx.rank,
                        ctx.world_size,
                        q_local,
                    )?;
                    let gathered = dq.gather_to_root(&comm, ROOT).await?;
                    Ok((gathered, f.r().to_owned()))
                }
            })
            .await
            .expect("cluster run should succeed");

            let (gathered, r) = results.first().cloned().expect("root result");
            let q = gathered.expect("root gathers Q");

            let diff = &q.dot(&r) - &a;
            assert!(
                frobenius(&diff.view()) < 1e-10,
                "world_size={world_size}: ||QR - A||_F = {}",
                frobenius(&diff.view())
            );

            let ortho = &q.t().dot(&q) - &identity(4);
            assert!(
                frobenius(&ortho.view()) < 1e-10,
                "world_size={world_size}: ||Q^T Q - I||_F = {}",
                frobenius(&ortho.view())
            );

            // R is upper triangular and replicated bit-for-bit.
            for i in 0..4 {
                for j in 0..i {
                    assert!(r[[i, j]].abs() < 1e-14);
                }
            }
            for (rank, (_, other)) in results.iter().enumerate() {
                assert_eq!(other, &r, "rank {rank} disagrees about R");
            }
        }
    }

    /// `Q^T B` computed through the tree must equal `Q_gathered^T B`. This is
    /// the assertion the "sum of `Q_i^T B_i`" shortcut fails for `p > 1`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn apply_qt_matches_the_gathered_thin_q() {
        for world_size in 1..=8u32 {
            let a = deterministic_matrix(24, 3, 8181);
            let b = deterministic_matrix(24, 2, 9191);
            let fabric = LocalFabric::new(world_size);
            let (a_ref, b_ref) = (a.clone(), b.clone());
            let results = LocalCluster::run(world_size, move |ctx: RankContext| {
                let fabric = Arc::clone(&fabric);
                let (a, b) = (a_ref.clone(), b_ref.clone());
                async move {
                    let comm = fabric.transport(ctx.rank)?;
                    let da = DistributedMatrix::from_global(
                        Layout::RowBlock,
                        &a.view(),
                        ctx.rank,
                        ctx.world_size,
                    )?;
                    let db = DistributedMatrix::from_global(
                        Layout::RowBlock,
                        &b.view(),
                        ctx.rank,
                        ctx.world_size,
                    )?;
                    let f = tsqr(&da, &comm).await?;
                    let qt_b = f.apply_qt(&comm, db.local_view()).await?;
                    let q_local = f.apply_q(&comm, Some(identity(3).view())).await?;
                    let dq = DistributedMatrix::from_local(
                        Layout::RowBlock,
                        24,
                        3,
                        ctx.rank,
                        ctx.world_size,
                        q_local,
                    )?;
                    Ok((qt_b, dq.gather_to_root(&comm, ROOT).await?))
                }
            })
            .await
            .expect("cluster run should succeed");

            let (qt_b, gathered) = results.first().cloned().expect("root result");
            let qt_b = qt_b.expect("root holds Q^T B");
            let q = gathered.expect("root gathers Q");
            let expected = q.t().dot(&b);
            let diff = &qt_b - &expected;
            assert!(
                frobenius(&diff.view()) < 1e-10,
                "world_size={world_size}: ||Q^T B - Q_gathered^T B||_F = {}",
                frobenius(&diff.view())
            );
            for (rank, (carry, _)) in results.iter().enumerate().skip(1) {
                assert!(carry.is_none(), "rank {rank} should not hold Q^T B");
            }
        }
    }

    /// A world size of 1 must exercise the same code path, not a shortcut:
    /// the tree is simply empty and the leaf factor is the whole answer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_rank_tree_has_no_levels() {
        let a = deterministic_matrix(10, 3, 606);
        let fabric = LocalFabric::new(1);
        let reference = a.clone();
        let results = LocalCluster::run(1, move |ctx: RankContext| {
            let fabric = Arc::clone(&fabric);
            let a = reference.clone();
            async move {
                let comm = fabric.transport(ctx.rank)?;
                let da = DistributedMatrix::from_global(
                    Layout::RowBlock,
                    &a.view(),
                    ctx.rank,
                    ctx.world_size,
                )?;
                let f = tsqr(&da, &comm).await?;
                Ok((f.levels().len(), f.r().to_owned()))
            }
        })
        .await
        .expect("cluster run should succeed");

        let (levels, r) = results.first().cloned().expect("root result");
        assert_eq!(levels, 0);
        let local = HouseholderQr::factor(a).expect("local factorization");
        assert_eq!(r, local.r());
    }

    /// A block wider than it is tall is refused on *every* rank, so nobody is
    /// left waiting in the tree for a peer that bailed out.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn wide_row_blocks_are_refused_on_every_rank() {
        // 6 rows over 4 ranks gives 2, 2, 1, 1 rows against 4 columns.
        let fabric = LocalFabric::new(4);
        let results = LocalCluster::run(4, move |ctx: RankContext| {
            let fabric = Arc::clone(&fabric);
            async move {
                let comm = fabric.transport(ctx.rank)?;
                let a = deterministic_matrix(6, 4, 909);
                let da = DistributedMatrix::from_global(
                    Layout::RowBlock,
                    &a.view(),
                    ctx.rank,
                    ctx.world_size,
                )?;
                let message = match tsqr(&da, &comm).await {
                    Err(DistributedLinalgError::UnsupportedShape(message)) => Some(message),
                    _ => None,
                };
                Ok(message)
            }
        })
        .await
        .expect("cluster run should succeed");

        for (rank, message) in results.iter().enumerate() {
            match message {
                Some(message) => assert!(message.contains("CAQR"), "rank {rank}: {message}"),
                None => panic!("rank {rank} did not report UnsupportedShape"),
            }
        }
    }

    /// The same factorization over the real network stack: TCP links,
    /// framing and [`EndpointTransport`] instead of the in-process fabric.
    /// Everything else in this module's tests runs on [`LocalFabric`], which
    /// would never catch a wire-format or endpoint-adapter defect — the two
    /// transports meet only here.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tsqr_runs_over_the_endpoint_transport() {
        use crate::distributed::linalg::EndpointTransport;
        use crate::distributed::testing::ClusterNode;

        let a = deterministic_matrix(20, 3, 6161);
        let reference = a.clone();
        let results = LocalCluster::run_connected(2, move |node: ClusterNode| {
            let a = reference.clone();
            async move {
                let comm = EndpointTransport::new(node.endpoint);
                let da = DistributedMatrix::from_global(
                    Layout::RowBlock,
                    &a.view(),
                    node.rank,
                    node.world_size,
                )?;
                let f = tsqr(&da, &comm).await?;
                let q_local = f.apply_q(&comm, Some(identity(3).view())).await?;
                let dq = DistributedMatrix::from_local(
                    Layout::RowBlock,
                    20,
                    3,
                    node.rank,
                    node.world_size,
                    q_local,
                )?;
                Ok((dq.gather_to_root(&comm, ROOT).await?, f.r().to_owned()))
            }
        })
        .await
        .expect("connected cluster run should succeed");

        let (gathered, r) = results.first().cloned().expect("root result");
        let q = gathered.expect("root gathers Q");
        let diff = &q.dot(&r) - &a;
        assert!(
            frobenius(&diff.view()) < 1e-10,
            "over TCP: ||QR - A||_F = {}",
            frobenius(&diff.view())
        );
        let ortho = &q.t().dot(&q) - &identity(3);
        assert!(
            frobenius(&ortho.view()) < 1e-10,
            "over TCP: Q not orthonormal"
        );
    }

    /// `DistFloat` is implemented for `f32` as well, and its wire encoding is
    /// four bytes per element rather than eight. Nothing else exercises that
    /// branch end to end, so a swapped width would otherwise only surface in
    /// a user's `f32` job.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tsqr_factors_an_f32_matrix() {
        let wide = deterministic_matrix(16, 3, 2727);
        let a = wide.mapv(|v| v as f32);
        let fabric = LocalFabric::new(3);
        let reference = a.clone();
        let results = LocalCluster::run(3, move |ctx: RankContext| {
            let fabric = Arc::clone(&fabric);
            let a = reference.clone();
            async move {
                let comm = fabric.transport(ctx.rank)?;
                let da = DistributedMatrix::from_global(
                    Layout::RowBlock,
                    &a.view(),
                    ctx.rank,
                    ctx.world_size,
                )?;
                let f = tsqr(&da, &comm).await?;
                let mut eye = Array2::<f32>::zeros((3, 3));
                for i in 0..3 {
                    eye[[i, i]] = 1.0;
                }
                let q_local = f.apply_q(&comm, Some(eye.view())).await?;
                let dq = DistributedMatrix::from_local(
                    Layout::RowBlock,
                    16,
                    3,
                    ctx.rank,
                    ctx.world_size,
                    q_local,
                )?;
                Ok((dq.gather_to_root(&comm, ROOT).await?, f.r().to_owned()))
            }
        })
        .await
        .expect("cluster run should succeed");

        let (gathered, r) = results.first().cloned().expect("root result");
        let q = gathered.expect("root gathers Q");
        // f32 carries ~7 decimal digits, so the tolerance is the single
        // precision counterpart of the 1e-10 used everywhere else.
        let diff = (&q.dot(&r) - &a).mapv(f64::from);
        assert!(
            frobenius(&diff.view()) < 1e-5,
            "f32: ||QR - A||_F = {}",
            frobenius(&diff.view())
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn column_block_cyclic_input_is_refused() {
        let fabric = LocalFabric::new(1);
        let results = LocalCluster::run(1, move |ctx: RankContext| {
            let fabric = Arc::clone(&fabric);
            async move {
                let comm = fabric.transport(ctx.rank)?;
                let a = deterministic_matrix(8, 4, 12);
                let da = DistributedMatrix::from_global(
                    Layout::ColBlockCyclic { panel_width: 2 },
                    &a.view(),
                    ctx.rank,
                    ctx.world_size,
                )?;
                Ok(matches!(
                    tsqr(&da, &comm).await,
                    Err(DistributedLinalgError::UnsupportedShape(_))
                ))
            }
        })
        .await
        .expect("cluster run should succeed");
        assert_eq!(results.first(), Some(&true));
    }
}
