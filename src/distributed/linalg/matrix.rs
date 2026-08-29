//! Distributed dense matrices: layouts, wire encoding, and the two dense
//! kernels that ride directly on the point-to-point transport
//! ([`matmul`], [`matvec`]).
//!
//! # Layouts
//!
//! [`Layout::RowBlock`] cuts the global matrix into contiguous row blocks
//! by the Block rule (`rank`s `0..(rows % p)` get one extra row each), the
//! layout every tall-skinny algorithm in [`mod@super::tsqr`] and
//! [`super::decomp`] wants.
//!
//! [`Layout::ColBlockCyclic`] cuts it into `panel_width`-wide column
//! panels dealt round-robin to ranks, the layout
//! [`super::cholesky`] wants: a right-looking factorization would leave
//! every rank but one idle under a plain block-column split, because the
//! trailing submatrix shrinks from the left.
//!
//! # Wire format
//!
//! Everything crossing the transport is plain little-endian bytes, no
//! external serialization crate involved: an 8-byte row count, an 8-byte
//! column count, then the elements row-major. [`DistFloat`] provides the
//! per-element half of that, which is why it is bounded on nothing more
//! exotic than `to_le_bytes`/`from_le_bytes` (`bytemuck` is gpu-gated in
//! this crate, so it is not available here).

use super::{
    allgather_bytes, allreduce_sum_f64, bcast_bytes, gather_bytes, DistTransport,
    DistributedLinalgError,
};
use num_traits::{Float, NumAssign};
use scirs2_core::ndarray::{s, Array2, ArrayView2, LinalgScalar, ScalarOperand};
use std::fmt::Debug;
use std::iter::Sum;

/// Tag space for [`matmul`]'s ring rotation.
const TAG_MATMUL_RING: u64 = 0x100;
/// Tag space for [`matvec`]'s allgather.
const TAG_MATVEC_ALLGATHER: u64 = 0x200;
/// Tag space for [`DistributedMatrix::gather_to_root`].
const TAG_GATHER: u64 = 0x300;
/// Tag space for [`DistributedMatrix::frobenius_norm`].
const TAG_NORM: u64 = 0x400;

/// Element types the distributed linear algebra kernels support.
///
/// The bounds are the union of what `scirs2_linalg`'s local kernels
/// require (`Float + NumAssign + Sum + ScalarOperand`), what `ndarray`'s
/// `dot` requires (`LinalgScalar`), what crossing threads requires
/// (`Send + Sync + 'static`), and the little-endian wire conversion this
/// trait adds.
pub trait DistFloat:
    Float + NumAssign + Sum + ScalarOperand + LinalgScalar + Send + Sync + Debug + 'static
{
    /// Width of one element on the wire, in bytes.
    const ELEM_BYTES: usize;

    /// Append this value to `out` in little-endian order.
    fn write_le(self, out: &mut Vec<u8>);

    /// Read one value from exactly [`Self::ELEM_BYTES`] little-endian bytes.
    fn read_le(chunk: &[u8]) -> Option<Self>;
}

impl DistFloat for f32 {
    const ELEM_BYTES: usize = 4;

    fn write_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }

    fn read_le(chunk: &[u8]) -> Option<Self> {
        <[u8; 4]>::try_from(chunk).ok().map(f32::from_le_bytes)
    }
}

impl DistFloat for f64 {
    const ELEM_BYTES: usize = 8;

    fn write_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }

    fn read_le(chunk: &[u8]) -> Option<Self> {
        <[u8; 8]>::try_from(chunk).ok().map(f64::from_le_bytes)
    }
}

/// Encode a slice as little-endian bytes.
pub fn encode_slice<T: DistFloat>(values: &[T]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * T::ELEM_BYTES);
    for value in values {
        value.write_le(&mut out);
    }
    out
}

/// Decode a little-endian byte run produced by [`encode_slice`].
pub fn decode_vec<T: DistFloat>(bytes: &[u8]) -> Result<Vec<T>, DistributedLinalgError> {
    if !bytes.len().is_multiple_of(T::ELEM_BYTES) {
        return Err(DistributedLinalgError::Transport(format!(
            "payload of {} bytes is not a whole number of {}-byte elements",
            bytes.len(),
            T::ELEM_BYTES
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / T::ELEM_BYTES);
    for chunk in bytes.chunks_exact(T::ELEM_BYTES) {
        out.push(T::read_le(chunk).ok_or_else(|| {
            DistributedLinalgError::Transport("malformed element in payload".to_string())
        })?);
    }
    Ok(out)
}

/// Encode a matrix as `[rows: u64][cols: u64][row-major elements]`.
pub fn encode_matrix<T: DistFloat>(a: &ArrayView2<T>) -> Vec<u8> {
    let (rows, cols) = a.dim();
    let mut out = Vec::with_capacity(16 + rows * cols * T::ELEM_BYTES);
    out.extend_from_slice(&(rows as u64).to_le_bytes());
    out.extend_from_slice(&(cols as u64).to_le_bytes());
    for row in a.rows() {
        for value in row {
            value.write_le(&mut out);
        }
    }
    out
}

/// Decode a payload produced by [`encode_matrix`].
pub fn decode_matrix<T: DistFloat>(bytes: &[u8]) -> Result<Array2<T>, DistributedLinalgError> {
    let malformed = || DistributedLinalgError::Transport("truncated matrix payload".to_string());
    let rows_bytes: [u8; 8] = bytes
        .get(0..8)
        .ok_or_else(malformed)?
        .try_into()
        .map_err(|_| malformed())?;
    let cols_bytes: [u8; 8] = bytes
        .get(8..16)
        .ok_or_else(malformed)?
        .try_into()
        .map_err(|_| malformed())?;
    let rows = u64::from_le_bytes(rows_bytes) as usize;
    let cols = u64::from_le_bytes(cols_bytes) as usize;
    let body = bytes.get(16..).ok_or_else(malformed)?;
    let values = decode_vec::<T>(body)?;
    if values.len() != rows * cols {
        return Err(DistributedLinalgError::Transport(format!(
            "matrix payload declares {rows}x{cols} but carries {} elements",
            values.len()
        )));
    }
    Array2::from_shape_vec((rows, cols), values).map_err(|e| {
        DistributedLinalgError::Transport(format!("matrix payload has an invalid shape: {e}"))
    })
}

/// Length of block `index` when `total` items are split over `parts` ranks
/// by the Block rule: the first `total % parts` blocks get one extra item.
pub fn block_len(total: usize, parts: u32, index: u32) -> usize {
    if parts == 0 || index >= parts {
        return 0;
    }
    let parts = parts as usize;
    let index = index as usize;
    total / parts + usize::from(index < total % parts)
}

/// Offset of block `index` under the same rule as [`block_len`].
pub fn block_offset(total: usize, parts: u32, index: u32) -> usize {
    if parts == 0 {
        return 0;
    }
    let parts_usize = parts as usize;
    let index = (index as usize).min(parts_usize);
    index * (total / parts_usize) + index.min(total % parts_usize)
}

/// How a [`DistributedMatrix`]'s global extent maps onto ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Contiguous row blocks, Block rule (see [`block_offset`]).
    RowBlock,
    /// `panel_width`-wide column panels dealt round-robin: panel `j` lives
    /// on rank `j % world_size`.
    ColBlockCyclic {
        /// Number of columns per panel (the last panel may be narrower).
        panel_width: usize,
    },
}

/// A dense matrix split across ranks according to a [`Layout`].
///
/// The struct itself carries no transport: every collective takes the
/// [`DistTransport`] explicitly, so one matrix can be used with whichever
/// communicator the caller is holding.
#[derive(Debug, Clone)]
pub struct DistributedMatrix<T> {
    layout: Layout,
    global_rows: usize,
    global_cols: usize,
    rank: u32,
    world_size: u32,
    local: Array2<T>,
}

impl<T: DistFloat> DistributedMatrix<T> {
    /// Local shape rank `rank` must hold under `layout`.
    pub fn expected_local_shape(
        layout: Layout,
        global_rows: usize,
        global_cols: usize,
        rank: u32,
        world_size: u32,
    ) -> (usize, usize) {
        match layout {
            Layout::RowBlock => (block_len(global_rows, world_size, rank), global_cols),
            Layout::ColBlockCyclic { panel_width } => {
                let owned = owned_panel_widths(global_cols, panel_width, rank, world_size)
                    .into_iter()
                    .sum();
                (global_rows, owned)
            }
        }
    }

    /// Wrap an already-partitioned local block.
    ///
    /// The local shape is validated against `layout`, so a mis-sliced block
    /// fails here instead of producing a silently wrong factorization.
    pub fn from_local(
        layout: Layout,
        global_rows: usize,
        global_cols: usize,
        rank: u32,
        world_size: u32,
        local: Array2<T>,
    ) -> Result<Self, DistributedLinalgError> {
        if world_size == 0 || rank >= world_size {
            return Err(DistributedLinalgError::InvalidRank { rank, world_size });
        }
        if let Layout::ColBlockCyclic { panel_width } = layout {
            if panel_width == 0 {
                return Err(DistributedLinalgError::InvalidDimensions {
                    rows: global_rows,
                    cols: 0,
                });
            }
        }
        let expected =
            Self::expected_local_shape(layout, global_rows, global_cols, rank, world_size);
        if local.dim() != expected {
            return Err(DistributedLinalgError::DimensionMismatch(format!(
                "rank {rank} holds a {:?} block but {layout:?} over {global_rows}x{global_cols} \
                 on {world_size} ranks prescribes {expected:?}",
                local.dim()
            )));
        }
        Ok(Self {
            layout,
            global_rows,
            global_cols,
            rank,
            world_size,
            local,
        })
    }

    /// Slice this rank's block out of a globally replicated matrix.
    ///
    /// Every rank passes the same `global` and keeps only its own part; no
    /// communication happens.
    pub fn from_global(
        layout: Layout,
        global: &ArrayView2<T>,
        rank: u32,
        world_size: u32,
    ) -> Result<Self, DistributedLinalgError> {
        let (global_rows, global_cols) = global.dim();
        let local = match layout {
            Layout::RowBlock => {
                let offset = block_offset(global_rows, world_size, rank);
                let len = block_len(global_rows, world_size, rank);
                global.slice(s![offset..offset + len, ..]).to_owned()
            }
            Layout::ColBlockCyclic { panel_width } => {
                if panel_width == 0 {
                    return Err(DistributedLinalgError::InvalidDimensions {
                        rows: global_rows,
                        cols: 0,
                    });
                }
                let width = owned_panel_widths(global_cols, panel_width, rank, world_size)
                    .into_iter()
                    .sum();
                let mut local = Array2::<T>::zeros((global_rows, width));
                let mut cursor = 0usize;
                for panel in owned_panels(global_cols, panel_width, rank, world_size) {
                    let (start, end) = panel_columns(global_cols, panel_width, panel);
                    let span = end - start;
                    local
                        .slice_mut(s![.., cursor..cursor + span])
                        .assign(&global.slice(s![.., start..end]));
                    cursor += span;
                }
                local
            }
        };
        Self::from_local(layout, global_rows, global_cols, rank, world_size, local)
    }

    /// An all-zero matrix in the given layout.
    pub fn zeros(
        layout: Layout,
        global_rows: usize,
        global_cols: usize,
        rank: u32,
        world_size: u32,
    ) -> Result<Self, DistributedLinalgError> {
        let shape = Self::expected_local_shape(layout, global_rows, global_cols, rank, world_size);
        Self::from_local(
            layout,
            global_rows,
            global_cols,
            rank,
            world_size,
            Array2::zeros(shape),
        )
    }

    /// This matrix's layout.
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Global `(rows, cols)`.
    pub fn global_shape(&self) -> (usize, usize) {
        (self.global_rows, self.global_cols)
    }

    /// The rank holding this block.
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// The world size this matrix was partitioned for.
    pub fn world_size(&self) -> u32 {
        self.world_size
    }

    /// Read-only view of this rank's block.
    pub fn local_view(&self) -> ArrayView2<'_, T> {
        self.local.view()
    }

    /// Mutable access to this rank's block.
    pub fn local_mut(&mut self) -> &mut Array2<T> {
        &mut self.local
    }

    /// Take ownership of this rank's block.
    pub fn into_local(self) -> Array2<T> {
        self.local
    }

    /// Global row index of this block's first row (row-block layouts only).
    pub fn row_offset(&self) -> usize {
        match self.layout {
            Layout::RowBlock => block_offset(self.global_rows, self.world_size, self.rank),
            Layout::ColBlockCyclic { .. } => 0,
        }
    }

    /// Number of column panels (column-block-cyclic layouts only).
    pub fn panel_count(&self) -> usize {
        match self.layout {
            Layout::RowBlock => 0,
            Layout::ColBlockCyclic { panel_width } => panel_count(self.global_cols, panel_width),
        }
    }

    /// Global `[start, end)` column range of panel `panel`.
    pub fn panel_columns(&self, panel: usize) -> (usize, usize) {
        match self.layout {
            Layout::RowBlock => (0, self.global_cols),
            Layout::ColBlockCyclic { panel_width } => {
                panel_columns(self.global_cols, panel_width, panel)
            }
        }
    }

    /// Which rank owns panel `panel`.
    pub fn panel_owner(&self, panel: usize) -> u32 {
        if self.world_size == 0 {
            return 0;
        }
        (panel % self.world_size as usize) as u32
    }

    /// Offset of panel `panel` inside this rank's local block, or `None`
    /// when this rank does not own it.
    pub fn local_panel_offset(&self, panel: usize) -> Option<usize> {
        match self.layout {
            Layout::RowBlock => None,
            Layout::ColBlockCyclic { panel_width } => {
                if self.panel_owner(panel) != self.rank {
                    return None;
                }
                let mut cursor = 0usize;
                for owned in owned_panels(self.global_cols, panel_width, self.rank, self.world_size)
                {
                    if owned == panel {
                        return Some(cursor);
                    }
                    let (start, end) = panel_columns(self.global_cols, panel_width, owned);
                    cursor += end - start;
                }
                None
            }
        }
    }

    /// Every panel index this rank owns, ascending.
    pub fn owned_panels(&self) -> Vec<usize> {
        match self.layout {
            Layout::RowBlock => Vec::new(),
            Layout::ColBlockCyclic { panel_width } => {
                owned_panels(self.global_cols, panel_width, self.rank, self.world_size)
            }
        }
    }

    /// Reassemble the whole matrix at `root`.
    ///
    /// Returns `Some(global)` on `root` and `None` elsewhere. This
    /// materializes the entire matrix on one rank, so it is meant for
    /// small results, diagnostics and tests — not for the working data of
    /// a large factorization.
    pub async fn gather_to_root<C: DistTransport + ?Sized>(
        &self,
        comm: &C,
        root: u32,
    ) -> Result<Option<Array2<T>>, DistributedLinalgError> {
        let ctx = comm.next_ctx();
        let payload = encode_matrix(&self.local.view());
        let gathered = gather_bytes(comm, root, ctx, TAG_GATHER, &payload).await?;
        let Some(blocks) = gathered else {
            return Ok(None);
        };

        let mut global = Array2::<T>::zeros((self.global_rows, self.global_cols));
        for (rank, block) in blocks.iter().enumerate() {
            let rank = rank as u32;
            let decoded = decode_matrix::<T>(block)?;
            match self.layout {
                Layout::RowBlock => {
                    let offset = block_offset(self.global_rows, self.world_size, rank);
                    let len = block_len(self.global_rows, self.world_size, rank);
                    if decoded.dim() != (len, self.global_cols) {
                        return Err(DistributedLinalgError::DimensionMismatch(format!(
                            "rank {rank} contributed a {:?} block, expected {:?}",
                            decoded.dim(),
                            (len, self.global_cols)
                        )));
                    }
                    global
                        .slice_mut(s![offset..offset + len, ..])
                        .assign(&decoded);
                }
                Layout::ColBlockCyclic { panel_width } => {
                    let mut cursor = 0usize;
                    for panel in owned_panels(self.global_cols, panel_width, rank, self.world_size)
                    {
                        let (start, end) = panel_columns(self.global_cols, panel_width, panel);
                        let span = end - start;
                        if cursor + span > decoded.ncols() {
                            return Err(DistributedLinalgError::DimensionMismatch(format!(
                                "rank {rank} contributed {} columns, too few for panel {panel}",
                                decoded.ncols()
                            )));
                        }
                        global
                            .slice_mut(s![.., start..end])
                            .assign(&decoded.slice(s![.., cursor..cursor + span]));
                        cursor += span;
                    }
                }
            }
        }
        Ok(Some(global))
    }

    /// Frobenius norm of the whole matrix, identical on every rank.
    ///
    /// Both layouts partition the elements disjointly, so the local sums of
    /// squares add up without any double counting.
    pub async fn frobenius_norm<C: DistTransport + ?Sized>(
        &self,
        comm: &C,
    ) -> Result<f64, DistributedLinalgError> {
        let ctx = comm.next_ctx();
        let mut local_sum = 0.0_f64;
        for value in self.local.iter() {
            let as_f64 = value.to_f64().ok_or_else(|| {
                DistributedLinalgError::LinalgError(
                    "element is not representable as f64".to_string(),
                )
            })?;
            local_sum += as_f64 * as_f64;
        }
        let total = allreduce_sum_f64(comm, ctx, TAG_NORM, local_sum).await?;
        Ok(total.sqrt())
    }
}

/// Number of column panels `global_cols` splits into at `panel_width`.
pub fn panel_count(global_cols: usize, panel_width: usize) -> usize {
    if panel_width == 0 {
        return 0;
    }
    global_cols.div_ceil(panel_width)
}

/// Global `[start, end)` columns of panel `panel`. The final panel is
/// narrower than `panel_width` whenever `panel_width` does not divide
/// `global_cols` — every consumer must use `end - start`, never
/// `panel_width`.
pub fn panel_columns(global_cols: usize, panel_width: usize, panel: usize) -> (usize, usize) {
    if panel_width == 0 {
        return (0, 0);
    }
    let start = (panel * panel_width).min(global_cols);
    let end = (start + panel_width).min(global_cols);
    (start, end)
}

/// Panel indices owned by `rank`, ascending.
pub fn owned_panels(
    global_cols: usize,
    panel_width: usize,
    rank: u32,
    world_size: u32,
) -> Vec<usize> {
    if world_size == 0 {
        return Vec::new();
    }
    (0..panel_count(global_cols, panel_width))
        .filter(|panel| panel % world_size as usize == rank as usize)
        .collect()
}

fn owned_panel_widths(
    global_cols: usize,
    panel_width: usize,
    rank: u32,
    world_size: u32,
) -> Vec<usize> {
    owned_panels(global_cols, panel_width, rank, world_size)
        .into_iter()
        .map(|panel| {
            let (start, end) = panel_columns(global_cols, panel_width, panel);
            end - start
        })
        .collect()
}

/// `C = A * B` for row-block distributed `A` and `B`, by rotating `B`'s
/// blocks around a ring.
///
/// Rank `r` starts holding `B_r` and, at step `s`, holds `B_{(r + s) % p}`;
/// it multiplies that against the matching column strip of its own `A_r`
/// and accumulates. Only one remote block is resident at a time, so peak
/// extra memory is `O(k * q / p)` rather than the `O(k * q)` a broadcast of
/// the whole `B` would cost.
///
/// `A` is `m x k` row-blocked, `B` is `k x q` row-blocked, and the result
/// is `m x q` row-blocked exactly like `A`.
pub async fn matmul<T: DistFloat, C: DistTransport + ?Sized>(
    a: &DistributedMatrix<T>,
    b: &DistributedMatrix<T>,
    comm: &C,
) -> Result<DistributedMatrix<T>, DistributedLinalgError> {
    let ctx = comm.next_ctx();
    if a.layout != Layout::RowBlock || b.layout != Layout::RowBlock {
        return Err(DistributedLinalgError::UnsupportedShape(
            "ring matmul requires both operands in Layout::RowBlock".to_string(),
        ));
    }
    let (m, k) = a.global_shape();
    let (b_rows, q) = b.global_shape();
    if k != b_rows {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "cannot multiply {m}x{k} by {b_rows}x{q}"
        )));
    }
    let size = comm.world_size();
    if a.world_size != size || b.world_size != size {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "operands were partitioned for {} and {} ranks but the communicator has {size}",
            a.world_size, b.world_size
        )));
    }
    let rank = comm.rank();
    if a.rank != rank || b.rank != rank {
        return Err(DistributedLinalgError::InvalidRank {
            rank: a.rank,
            world_size: size,
        });
    }

    let local_rows = a.local.nrows();
    let mut c = Array2::<T>::zeros((local_rows, q));
    let mut current = b.local.clone();
    let mut current_index = rank;
    let left = (rank + size - 1) % size;
    let right = (rank + 1) % size;

    for step in 0..size {
        let offset = block_offset(k, size, current_index);
        let len = block_len(k, size, current_index);
        if len != current.nrows() {
            return Err(DistributedLinalgError::DimensionMismatch(format!(
                "block {current_index} of B has {} rows, expected {len}",
                current.nrows()
            )));
        }
        if len > 0 && local_rows > 0 {
            let strip = a.local.slice(s![.., offset..offset + len]);
            c += &strip.dot(&current);
        }
        if step + 1 < size {
            let tag = TAG_MATMUL_RING + u64::from(step);
            let payload = encode_matrix(&current.view());
            comm.send_bytes(left, ctx, tag, &payload).await?;
            let received = comm.recv_bytes(right, ctx, tag).await?;
            current = decode_matrix::<T>(&received)?;
            current_index = (current_index + 1) % size;
        }
    }

    DistributedMatrix::from_local(Layout::RowBlock, m, q, rank, size, c)
}

/// `y = A * x` for a row-block distributed `A` and a block-distributed `x`.
///
/// `x` is gathered in full first (it is `O(n)`, negligible against `A`'s
/// `O(m * n)`), then each rank runs one local `gemv`. The returned chunk is
/// this rank's slice of `y` under the same row distribution as `A`.
pub async fn matvec<T: DistFloat, C: DistTransport + ?Sized>(
    a: &DistributedMatrix<T>,
    x_local: &[T],
    comm: &C,
) -> Result<Vec<T>, DistributedLinalgError> {
    let ctx = comm.next_ctx();
    if a.layout != Layout::RowBlock {
        return Err(DistributedLinalgError::UnsupportedShape(
            "matvec requires Layout::RowBlock".to_string(),
        ));
    }
    let size = comm.world_size();
    let rank = comm.rank();
    let (_, n) = a.global_shape();
    let expected = block_len(n, size, rank);
    if x_local.len() != expected {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "rank {rank} supplied {} elements of x, expected {expected}",
            x_local.len()
        )));
    }

    let chunks = allgather_bytes(comm, ctx, TAG_MATVEC_ALLGATHER, encode_slice(x_local)).await?;
    let mut x = Vec::with_capacity(n);
    for chunk in &chunks {
        x.extend(decode_vec::<T>(chunk)?);
    }
    if x.len() != n {
        return Err(DistributedLinalgError::DimensionMismatch(format!(
            "allgathered x has {} elements, expected {n}",
            x.len()
        )));
    }

    let local = a.local_view();
    let mut y = Vec::with_capacity(local.nrows());
    for row in local.rows() {
        let mut acc = T::zero();
        for (value, xi) in row.iter().zip(x.iter()) {
            acc += *value * *xi;
        }
        y.push(acc);
    }
    Ok(y)
}

/// Broadcast a small replicated matrix from `root` to every rank.
///
/// Used for the `n x n` factors (`R`, `V^T`, panels) that every rank needs
/// in full; `tag` separates it from any other broadcast in flight under
/// the same `ctx`.
pub async fn broadcast_matrix<T: DistFloat, C: DistTransport + ?Sized>(
    comm: &C,
    root: u32,
    ctx: u64,
    tag: u64,
    matrix: Option<&Array2<T>>,
) -> Result<Array2<T>, DistributedLinalgError> {
    let payload = if comm.rank() == root {
        let matrix = matrix.ok_or_else(|| {
            DistributedLinalgError::DimensionMismatch(
                "broadcast root must supply the matrix".to_string(),
            )
        })?;
        Some(encode_matrix(&matrix.view()))
    } else {
        None
    };
    let bytes = bcast_bytes(comm, root, ctx, tag, payload).await?;
    decode_matrix::<T>(&bytes)
}

/// Broadcast a small replicated vector from `root` to every rank.
pub async fn broadcast_vec<T: DistFloat, C: DistTransport + ?Sized>(
    comm: &C,
    root: u32,
    ctx: u64,
    tag: u64,
    values: Option<&[T]>,
) -> Result<Vec<T>, DistributedLinalgError> {
    let payload = if comm.rank() == root {
        let values = values.ok_or_else(|| {
            DistributedLinalgError::DimensionMismatch(
                "broadcast root must supply the vector".to_string(),
            )
        })?;
        Some(encode_slice(values))
    } else {
        None
    };
    let bytes = bcast_bytes(comm, root, ctx, tag, payload).await?;
    decode_vec::<T>(&bytes)
}

#[cfg(test)]
pub(crate) mod testutil {
    //! Deterministic fixtures shared by every test module in this lane.
    //!
    //! No RNG crate is involved: a fixed 64-bit LCG makes each test's
    //! matrix reproducible across runs, ranks and platforms, which matters
    //! because several tests compare a distributed result against a local
    //! one computed from "the same" matrix on a different rank.

    use scirs2_core::ndarray::{Array2, ArrayView2};

    /// Deterministic pseudo-random matrix in `[-1, 1)`.
    pub fn deterministic_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let mut values = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            values.push(unit * 2.0 - 1.0);
        }
        Array2::from_shape_vec((rows, cols), values).unwrap_or_else(|_| Array2::zeros((rows, cols)))
    }

    /// Deterministic symmetric positive definite matrix: `B^T B + n I`.
    pub fn spd_matrix(n: usize, seed: u64) -> Array2<f64> {
        let b = deterministic_matrix(n, n, seed);
        let mut a = b.t().dot(&b);
        for i in 0..n {
            a[[i, i]] += n as f64;
        }
        // Force exact symmetry so tests measure the algorithm, not the
        // rounding of B^T B.
        for i in 0..n {
            for j in 0..i {
                let mean = (a[[i, j]] + a[[j, i]]) * 0.5;
                a[[i, j]] = mean;
                a[[j, i]] = mean;
            }
        }
        a
    }

    /// Frobenius norm of a local matrix.
    pub fn frobenius(a: &ArrayView2<f64>) -> f64 {
        a.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::testutil::{deterministic_matrix, frobenius};
    use super::*;
    use crate::distributed::linalg::LocalFabric;
    use crate::distributed::testing::{LocalCluster, RankContext};
    use scirs2_core::ndarray::Array2;
    use std::sync::Arc;

    #[test]
    fn block_rule_partitions_without_gaps_or_overlap() {
        for total in 0..12usize {
            for parts in 1..6u32 {
                let mut covered = 0usize;
                for index in 0..parts {
                    assert_eq!(block_offset(total, parts, index), covered);
                    covered += block_len(total, parts, index);
                }
                assert_eq!(covered, total);
            }
        }
    }

    #[test]
    fn ragged_panels_report_their_actual_width() {
        // 50 columns at width 8: panels 0..5 are 8 wide, panel 6 is 2 wide.
        assert_eq!(panel_count(50, 8), 7);
        assert_eq!(panel_columns(50, 8, 5), (40, 48));
        assert_eq!(panel_columns(50, 8, 6), (48, 50));
        assert_eq!(owned_panels(50, 8, 2, 4), vec![2, 6]);
    }

    #[test]
    fn round_trips_a_matrix_through_the_wire_format() {
        let a = deterministic_matrix(5, 3, 11);
        let bytes = encode_matrix(&a.view());
        let back = decode_matrix::<f64>(&bytes).expect("decodes");
        assert_eq!(back, a);
    }

    #[test]
    fn rejects_a_truncated_matrix_payload() {
        let a = deterministic_matrix(2, 2, 3);
        let mut bytes = encode_matrix(&a.view());
        bytes.truncate(20);
        assert!(matches!(
            decode_matrix::<f64>(&bytes),
            Err(DistributedLinalgError::Transport(_))
        ));
    }

    #[test]
    fn rejects_a_mis_sliced_local_block() {
        let bad = Array2::<f64>::zeros((3, 4));
        let err = DistributedMatrix::from_local(Layout::RowBlock, 8, 4, 0, 4, bad);
        assert!(matches!(
            err,
            Err(DistributedLinalgError::DimensionMismatch(_))
        ));
    }

    #[test]
    fn column_cyclic_local_block_concatenates_owned_panels() {
        let global = deterministic_matrix(6, 10, 5);
        let layout = Layout::ColBlockCyclic { panel_width: 3 };
        // 10 columns at width 3 -> panels [0,3) [3,6) [6,9) [9,10)
        // rank 0 of 2 owns panels 0 and 2 -> 3 + 3 = 6 columns.
        let dm = DistributedMatrix::from_global(layout, &global.view(), 0, 2).expect("partitions");
        assert_eq!(dm.local_view().dim(), (6, 6));
        assert_eq!(dm.owned_panels(), vec![0, 2]);
        assert_eq!(dm.local_panel_offset(2), Some(3));
        assert_eq!(dm.local_panel_offset(1), None);
        assert_eq!(
            dm.local_view().slice(s![.., 3..6]),
            global.slice(s![.., 6..9])
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn gather_to_root_reassembles_both_layouts() {
        for world_size in 1..=4u32 {
            for layout in [Layout::RowBlock, Layout::ColBlockCyclic { panel_width: 3 }] {
                let global = deterministic_matrix(9, 10, 21);
                let fabric = LocalFabric::new(world_size);
                let reference = global.clone();
                let results = LocalCluster::run(world_size, move |ctx: RankContext| {
                    let fabric = Arc::clone(&fabric);
                    let global = reference.clone();
                    async move {
                        let comm = fabric.transport(ctx.rank)?;
                        let dm = DistributedMatrix::from_global(
                            layout,
                            &global.view(),
                            ctx.rank,
                            ctx.world_size,
                        )?;
                        let gathered = dm.gather_to_root(&comm, 0).await?;
                        Ok(gathered)
                    }
                })
                .await
                .expect("cluster run should succeed");

                let root = results[0].clone().expect("root gathers the matrix");
                assert_eq!(root, global, "world_size={world_size} layout={layout:?}");
                for other in results.iter().skip(1) {
                    assert!(other.is_none());
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn frobenius_norm_matches_the_local_value() {
        let global = deterministic_matrix(8, 5, 33);
        let expected = frobenius(&global.view());
        let fabric = LocalFabric::new(3);
        let reference = global.clone();
        let results = LocalCluster::run(3, move |ctx: RankContext| {
            let fabric = Arc::clone(&fabric);
            let global = reference.clone();
            async move {
                let comm = fabric.transport(ctx.rank)?;
                let dm = DistributedMatrix::from_global(
                    Layout::RowBlock,
                    &global.view(),
                    ctx.rank,
                    ctx.world_size,
                )?;
                let norm = dm.frobenius_norm(&comm).await?;
                Ok(norm)
            }
        })
        .await
        .expect("cluster run should succeed");

        for norm in &results {
            assert!((norm - expected).abs() < 1e-12, "{norm} vs {expected}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ring_matmul_matches_the_local_product() {
        for world_size in 1..=4u32 {
            let a = deterministic_matrix(11, 7, 101);
            let b = deterministic_matrix(7, 5, 202);
            let expected = a.dot(&b);
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
                    let dc = matmul(&da, &db, &comm).await?;
                    let gathered = dc.gather_to_root(&comm, 0).await?;
                    Ok(gathered)
                }
            })
            .await
            .expect("cluster run should succeed");

            let got = results[0].clone().expect("root gathers C");
            let diff = &got - &expected;
            assert!(
                frobenius(&diff.view()) < 1e-12,
                "world_size={world_size}: ||C - A*B||_F = {}",
                frobenius(&diff.view())
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn allgather_matvec_matches_the_local_product() {
        for world_size in 1..=4u32 {
            let a = deterministic_matrix(10, 6, 303);
            let x: Vec<f64> = (0..6).map(|i| 0.5 + i as f64).collect();
            let expected: Vec<f64> = a
                .rows()
                .into_iter()
                .map(|row| row.iter().zip(x.iter()).map(|(r, v)| r * v).sum::<f64>())
                .collect();
            let fabric = LocalFabric::new(world_size);
            let (a_ref, x_ref) = (a.clone(), x.clone());
            let results = LocalCluster::run(world_size, move |ctx: RankContext| {
                let fabric = Arc::clone(&fabric);
                let (a, x) = (a_ref.clone(), x_ref.clone());
                async move {
                    let comm = fabric.transport(ctx.rank)?;
                    let da = DistributedMatrix::from_global(
                        Layout::RowBlock,
                        &a.view(),
                        ctx.rank,
                        ctx.world_size,
                    )?;
                    let offset = block_offset(x.len(), ctx.world_size, ctx.rank);
                    let len = block_len(x.len(), ctx.world_size, ctx.rank);
                    let y = matvec(&da, &x[offset..offset + len], &comm).await?;
                    Ok(y)
                }
            })
            .await
            .expect("cluster run should succeed");

            let mut assembled = Vec::new();
            for chunk in &results {
                assembled.extend_from_slice(chunk);
            }
            assert_eq!(assembled.len(), expected.len());
            for (got, want) in assembled.iter().zip(expected.iter()) {
                assert!((got - want).abs() < 1e-12, "{got} vs {want}");
            }
        }
    }
}
