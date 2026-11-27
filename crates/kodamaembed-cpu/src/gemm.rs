//! C = A W^T + bias in FP32; all tiles reduce depth in one order, so bits ignore batch and workers.

use crate::{math, workers::POOL};
use rayon::prelude::*;

/// A strip's rows and a panel's columns; a 6 x 16 tile holds twelve 8-lane accumulators.
pub(crate) const MR: usize = 6;
pub(crate) const NR: usize = 16;
/// K block: a strip of A and a panel slice of W stay in L1 and L2 cache.
pub(crate) const KC: usize = 256;
/// Rows per task in long inputs, so a band of A stays in L2 across its panel tiles.
const MC: usize = 16 * MR;
/// Up to this many rows one band holds them all, so each panel tile is read once per K block.
pub(crate) const SHORT_ROWS: usize = 256;
/// Up to this many rows a band's K block of A fits a 32 KB L1 once gathered.
const GATHER_ROWS: usize = 32 * 1024 / (KC * size_of::<f32>());
/// Panels a band must feed before rereading gathered rows repays the copy.
const GATHER_PANELS: usize = 12;
/// AVX2 pair path: a depth block's two panel slices (16 KB) serve a chunk of strips from L1.
#[cfg(target_arch = "x86_64")]
const DEPTH: usize = 128;
/// Strips whose sums wait on the stack between the AVX2 pair path's depth blocks.
#[cfg(target_arch = "x86_64")]
const CHUNK: usize = 8;
/// Depth steps an AVX2 loop iteration: Gracemont mispredicts a 128-trip exit, not a 32-trip one.
#[cfg(target_arch = "x86_64")]
const STEPS: usize = 4;
/// Depth steps ahead the AVX2 pair loop prefetches its panel lines, so their L2 hits do not stall.
#[cfg(target_arch = "x86_64")]
const AHEAD: usize = 8;

/// Which microkernel runs the tiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// Eight-lane FMA (x86-64 with AVX2 and FMA).
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// Avx2's operations in its order on four-lane vectors (64-bit ARM), so results keep its bits.
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    Neon,
    /// Plain Rust, for any processor.
    Portable,
}

/// The instruction set as its maker spells it, for device names and messages.
impl std::fmt::Display for Kernel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => "AVX2",
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            Self::Neon => "NEON",
            Self::Portable => "portable",
        })
    }
}

impl Kernel {
    /// Whether this processor supports the explicitly selected kernel.
    pub fn is_supported(self) -> bool {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            Self::Neon => true,
            Self::Portable => true,
        }
    }

    /// The fastest kernel this processor runs.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Self::Avx2;
        }
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            Self::Neon
        }
        #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
        {
            Self::Portable
        }
    }
}

/// A weight W [n, k] (output by input) packed for the microkernel.
pub struct PackedWeight {
    n: usize,
    k: usize,
    panels: Vec<f32>,
}

impl PackedWeight {
    pub fn new(w: &[f32], n: usize, k: usize) -> Self {
        assert!(n > 0 && k > 0, "positive dimensions");
        let elements = n.checked_mul(k).expect("dimensions overflow");
        assert_eq!(w.len(), elements);
        let mut panels = vec![0.0; panels_len(n, k)];
        pack_rows(w, k, n, k, &mut panels);
        Self { n, k, panels }
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn k(&self) -> usize {
        self.k
    }

    #[cfg(target_arch = "x86_64")]
    pub(crate) fn rows(&self) -> Vec<f32> {
        let mut rows = vec![0.0; self.n * self.k];
        for (j, row) in rows.chunks_exact_mut(self.k).enumerate() {
            let panel = &self.panels[j / NR * self.k * NR..];
            for (i, value) in row.iter_mut().enumerate() {
                *value = panel[i * NR + j % NR];
            }
        }
        rows
    }

    fn columns(&self, first: usize, end: usize) -> Panels<'_> {
        Panels {
            values: &self.panels[first * self.k * NR..end * self.k * NR],
            n: (end * NR).min(self.n) - first * NR,
            k: self.k,
        }
    }
}

/// Columns in NR-wide panels laid out [k][NR], zero past n.
#[derive(Clone, Copy)]
pub(crate) struct Panels<'a> {
    pub(crate) values: &'a [f32],
    pub(crate) n: usize,
    pub(crate) k: usize,
}

impl<'a> Panels<'a> {
    /// K block `k0..k0 + kc` of panels `p..p + width`.
    fn slices(&self, p: usize, width: usize, k0: usize, kc: usize) -> Slices<'a> {
        let k = self.k;
        Slices {
            values: &self.values[(p * k + k0) * NR..((p + width - 1) * k + k0 + kc) * NR],
            next: k * NR,
            width,
        }
    }
}

/// One K block of `width` adjacent panels, `next` floats apart.
#[derive(Clone, Copy)]
struct Slices<'a> {
    values: &'a [f32],
    next: usize,
    width: usize,
}

/// Runs of `span` whole panels, then each panel left over, as (first, width).
fn panel_groups(panels: usize, whole: usize, span: usize) -> impl Iterator<Item = (usize, usize)> {
    let grouped = whole - whole % span;
    (0..grouped)
        .step_by(span)
        .map(move |p| (p, span))
        .chain((grouped..panels).map(|p| (p, 1)))
}

pub(crate) fn panels_len(n: usize, k: usize) -> usize {
    let elements = n
        .div_ceil(NR)
        .checked_mul(k)
        .and_then(|size| size.checked_mul(NR))
        .expect("dimensions overflow");
    assert!(
        elements <= isize::MAX as usize / size_of::<f32>(),
        "packed dimensions exceed addressable memory"
    );
    elements
}

pub(crate) fn pack_rows(source: &[f32], stride: usize, n: usize, k: usize, out: &mut [f32]) {
    for (p, panel) in out[..panels_len(n, k)].chunks_exact_mut(k * NR).enumerate() {
        let cols = NR.min(n - p * NR);
        if cols < NR {
            panel.fill(0.0);
        }
        for col in 0..cols {
            let row = &source[(p * NR + col) * stride..][..k];
            for (i, &value) in row.iter().enumerate() {
                panel[i * NR + col] = value;
            }
        }
    }
}

pub(crate) fn pack_columns(source: &[f32], stride: usize, n: usize, k: usize, out: &mut [f32]) {
    for (p, panel) in out[..panels_len(n, k)].chunks_exact_mut(k * NR).enumerate() {
        let cols = NR.min(n - p * NR);
        for (i, row) in panel.as_chunks_mut::<NR>().0.iter_mut().enumerate() {
            row[..cols].copy_from_slice(&source[i * stride + p * NR..][..cols]);
            row[cols..].fill(0.0);
        }
    }
}

/// What is applied to A W^T before it is stored in C [m, n].
#[derive(Clone, Copy)]
pub enum Epilogue<'a> {
    /// + bias (n values)
    Bias(&'a [f32]),
    /// gelu(. + bias)
    BiasGelu(&'a [f32]),
    /// . + bias + residual [m, n]
    BiasResidual(&'a [f32], &'a [f32]),
}

impl<'a> Epilogue<'a> {
    fn bias(&self) -> &[f32] {
        match self {
            Self::Bias(bias) | Self::BiasGelu(bias) | Self::BiasResidual(bias, _) => bias,
        }
    }

    fn columns_from(self, col: usize) -> Epilogue<'a> {
        match self {
            Self::Bias(bias) => Self::Bias(&bias[col..]),
            Self::BiasGelu(bias) => Self::BiasGelu(&bias[col..]),
            Self::BiasResidual(bias, residual) => {
                Self::BiasResidual(&bias[col..], &residual[col..])
            }
        }
    }
}

/// c [m, n] = epilogue(a [m, k] W^T) on KodamaEmbed's worker threads.
pub fn gemm(
    kernel: Kernel,
    a: &[f32],
    m: usize,
    w: &PackedWeight,
    epilogue: Epilogue,
    c: &mut [f32],
) {
    assert!(
        kernel.is_supported(),
        "kernel {kernel} is unavailable on this processor"
    );
    let (n, k) = (w.n, w.k);
    let a_len = m.checked_mul(k).expect("dimensions overflow");
    let c_len = m.checked_mul(n).expect("dimensions overflow");
    assert!(
        a.len() >= a_len && c.len() >= c_len,
        "a is m x k and c m x n"
    );
    assert!(epilogue.bias().len() >= n, "bias per column");
    if let Epilogue::BiasResidual(_, residual) = epilogue {
        assert!(residual.len() >= c_len, "residual too short");
    }
    if m == 0 {
        return;
    }
    POOL.install(|| gemm_current_pool(kernel, a, m, w, epilogue, c));
}

/// Executes in the caller-selected Rayon pool; encoder callers install it once.
pub(crate) fn gemm_current_pool(
    kernel: Kernel,
    a: &[f32],
    m: usize,
    w: &PackedWeight,
    epilogue: Epilogue,
    c: &mut [f32],
) {
    assert!(
        kernel.is_supported(),
        "kernel {kernel} is unavailable on this processor"
    );
    let (n, k) = (w.n, w.k);
    let a_len = m.checked_mul(k).expect("dimensions overflow");
    let c_len = m.checked_mul(n).expect("dimensions overflow");
    assert!(
        a.len() >= a_len && c.len() >= c_len,
        "a is m x k and c m x n"
    );
    assert!(epilogue.bias().len() >= n, "bias per column");
    if let Epilogue::BiasResidual(_, residual) = epilogue {
        assert!(residual.len() >= c_len, "residual too short");
    }
    if m == 0 {
        return;
    }
    let panels = n.div_ceil(NR);
    // Short inputs: one band, an equal run of panels per worker; long inputs: bands by runs of
    // `span` panels. A job per task, so a worker that runs out takes any task left.
    let (height, runs) = if m <= SHORT_ROWS {
        (m, rayon::current_num_threads().min(panels))
    } else {
        (MC, panels.div_ceil(kernel.span()))
    };
    let out = SharedOut(c.as_mut_ptr());
    (0..m.div_ceil(height) * runs)
        .into_par_iter()
        .with_max_len(1)
        .for_each(|task| {
            let (row0, run) = (task / runs * height, task % runs);
            let (first, end) = (run * panels / runs, (run + 1) * panels / runs);
            let band = Band {
                kernel,
                a: &a[row0 * k..],
                lda: k,
                rows: height.min(m - row0),
                panels: w.columns(first, end),
                ldc: n,
                epilogue: Some(Placed {
                    epilogue: epilogue.columns_from(first * NR),
                    row0,
                    stride: n,
                }),
            };
            // SAFETY: C holds m rows of n values, and a task writes only its band of its panels.
            unsafe { band.run(out.at(row0 * n + first * NR)) };
        });
}

/// Tasks' tiles interleave in memory, which disjoint slices cannot express.
#[derive(Clone, Copy)]
struct SharedOut(*mut f32);

// SAFETY: each task writes only its own cells.
unsafe impl Send for SharedOut {}
// SAFETY: as above.
unsafe impl Sync for SharedOut {}

impl SharedOut {
    /// A method, so closures capture the Sync wrapper, not its raw pointer.
    ///
    /// # Safety
    /// `offset` is inside C.
    unsafe fn at(self, offset: usize) -> *mut f32 {
        // SAFETY: the caller vouches for offset.
        unsafe { self.0.add(offset) }
    }
}

/// c [rows, ldc] = a [rows, lda] W^T on the calling thread, without an epilogue.
pub(crate) fn product(
    kernel: Kernel,
    a: &[f32],
    lda: usize,
    rows: usize,
    panels: Panels,
    c: &mut [f32],
    ldc: usize,
) {
    if rows == 0 {
        return;
    }
    assert!(c.len() >= (rows - 1) * ldc + panels.n, "c holds the rows");
    let band = Band {
        kernel,
        a,
        lda,
        rows,
        panels,
        ldc,
        epilogue: None,
    };
    // SAFETY: c holds the rows (asserted) and is this call's alone.
    unsafe { band.run(c.as_mut_ptr()) };
}

/// # Safety
/// As for [`Kernel::tile`], on a processor with AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_rows_avx2<const R: usize>(
    kc: usize,
    a: *const f32,
    lda: usize,
    panel: *const f32,
    c: *mut f32,
    ldc: usize,
    first: bool,
    finish: Option<Finish>,
) {
    use std::arch::x86_64::*;
    // SAFETY: the caller vouches for every address read and written.
    unsafe {
        let mut acc = [[_mm256_setzero_ps(); 2]; R];
        steps(0..kc, |p| {
            let b0 = _mm256_loadu_ps(panel.add(p * NR));
            let b1 = _mm256_loadu_ps(panel.add(p * NR + 8));
            for (r, row) in acc.iter_mut().enumerate() {
                let value = _mm256_broadcast_ss(&*a.add(r * lda + p));
                row[0] = _mm256_fmadd_ps(value, b0, row[0]);
                row[1] = _mm256_fmadd_ps(value, b1, row[1]);
            }
        });
        for (r, &row) in acc.iter().enumerate() {
            store_avx2(row, c, ldc, r, first, finish);
        }
    }
}

/// tile_rows_avx2's multiply-adds in its order, four lanes at a time.
///
/// # Safety
/// As for [`Kernel::tile`], on a processor with NEON.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[target_feature(enable = "neon")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_rows_neon<const R: usize>(
    kc: usize,
    a: *const f32,
    lda: usize,
    panel: *const f32,
    c: *mut f32,
    ldc: usize,
    first: bool,
    finish: Option<Finish>,
) {
    use std::arch::aarch64::*;
    // SAFETY: the caller vouches for every address read and written.
    unsafe {
        let mut acc = [[vdupq_n_f32(0.0); NR / 4]; R];
        for p in 0..kc {
            let columns: [float32x4_t; NR / 4] =
                std::array::from_fn(|q| vld1q_f32(panel.add(p * NR + 4 * q)));
            for (r, row) in acc.iter_mut().enumerate() {
                let value = *a.add(r * lda + p);
                for (sum, &column) in row.iter_mut().zip(&columns) {
                    *sum = vfmaq_n_f32(*sum, column, value);
                }
            }
        }
        for (r, row) in acc.iter().enumerate() {
            for (q, &sum) in row.iter().enumerate() {
                let at = c.add(r * ldc + 4 * q);
                let mut value = if first {
                    sum
                } else {
                    vaddq_f32(vld1q_f32(at), sum)
                };
                if let Some(finish) = finish {
                    value = vaddq_f32(value, vld1q_f32(finish.bias.add(4 * q)));
                    if finish.gelu {
                        value = math::neon::gelu_unbounded(value);
                    }
                    if !finish.residual.is_null() {
                        let residual = finish.residual.add(r * finish.stride + 4 * q);
                        value = vaddq_f32(value, vld1q_f32(residual));
                    }
                }
                vst1q_f32(at, value);
            }
        }
    }
}

/// # Safety
/// As for [`Kernel::tile`].
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn tile_rows_portable<const R: usize>(
    kc: usize,
    a: *const f32,
    lda: usize,
    panel: *const f32,
    c: *mut f32,
    ldc: usize,
    first: bool,
    finish: Option<Finish>,
) {
    let mut acc = [[0.0f32; NR]; R];
    for p in 0..kc {
        // SAFETY: the caller vouches for every address read.
        let b = unsafe { std::slice::from_raw_parts(panel.add(p * NR), NR) };
        for (r, row) in acc.iter_mut().enumerate() {
            // SAFETY: as above.
            let a = unsafe { *a.add(r * lda + p) };
            for (sum, &value) in row.iter_mut().zip(b) {
                *sum += a * value;
            }
        }
    }
    for (r, row) in acc.iter().enumerate() {
        for (j, &sum) in row.iter().enumerate() {
            // SAFETY: the caller vouches for the tile at c and for finish.
            unsafe {
                let at = c.add(r * ldc + j);
                let mut value = if first { sum } else { *at + sum };
                if let Some(finish) = finish {
                    value += *finish.bias.add(j);
                    if finish.gelu {
                        value = math::gelu_unbounded(value);
                    }
                    if !finish.residual.is_null() {
                        value += *finish.residual.add(r * finish.stride + j);
                    }
                }
                *at = value;
            }
        }
    }
}

/// Rows of A by a run of panels: one task's share of a GEMM.
struct Band<'a> {
    kernel: Kernel,
    a: &'a [f32],
    lda: usize,
    rows: usize,
    panels: Panels<'a>,
    ldc: usize,
    epilogue: Option<Placed<'a>>,
}

#[derive(Clone, Copy)]
struct Placed<'a> {
    epilogue: Epilogue<'a>,
    row0: usize,
    stride: usize,
}

#[derive(Clone, Copy)]
struct Tile {
    row: usize,
    rows: usize,
    col: usize,
    cols: usize,
}

#[derive(Clone, Copy)]
struct Finish {
    bias: *const f32,
    gelu: bool,
    /// Null without a residual.
    residual: *const f32,
    stride: usize,
}

impl Band<'_> {
    /// # Safety
    /// `c` covers `rows` rows of `panels.n` values `ldc` apart that no other thread touches.
    unsafe fn run(&self, c: *mut f32) {
        let k = self.panels.k;
        assert!(
            self.a.len() >= (self.rows - 1) * self.lda + k,
            "a holds the band's rows"
        );
        // Rows reread by many panels come from one contiguous copy in L1, not scattered rows.
        let gather = self.rows <= GATHER_ROWS && self.panels.n >= GATHER_PANELS * NR;
        let mut gathered = Vec::with_capacity(if gather { self.rows * KC } else { 0 });
        for k0 in (0..k).step_by(KC) {
            let kc = KC.min(k - k0);
            let (a, lda) = if gather {
                gathered.clear();
                for r in 0..self.rows {
                    gathered.extend_from_slice(&self.a[r * self.lda + k0..][..kc]);
                }
                (gathered.as_ptr(), kc)
            } else {
                (self.a[k0..].as_ptr(), self.lda)
            };
            // SAFETY: the caller vouches for c, and a holds the band's K block, rows lda apart.
            unsafe { self.block(k0, kc, a, lda, c) };
        }
    }

    /// One K block: whole panels' strips and row tails straight into C, a partial panel padded.
    ///
    /// # Safety
    /// As for [`Band::run`], with `a` at this K block's first depth, rows `lda` apart.
    unsafe fn block(&self, k0: usize, kc: usize, a: *const f32, lda: usize, c: *mut f32) {
        let (n, k, ldc) = (self.panels.n, self.panels.k, self.ldc);
        let (full, rest, whole) = (self.rows / MR, self.rows % MR, n / NR);
        let (first, epilogue) = (k0 == 0, if k0 + kc == k { self.epilogue } else { None });
        for (p, width) in panel_groups(n.div_ceil(NR), whole, self.kernel.span()) {
            let slices = self.panels.slices(p, width, k0, kc);
            if p == whole {
                for row in (0..self.rows).step_by(MR) {
                    let (rows, col) = (MR.min(self.rows - row), p * NR);
                    let tile = Tile {
                        row,
                        rows,
                        col,
                        cols: n - col,
                    };
                    // SAFETY: the caller vouches for c and a.
                    unsafe { self.edge(kc, a, lda, slices.values, c, tile, first, epilogue) };
                }
                continue;
            }
            if full > 0 {
                let finish = self.finish(epilogue, 0, full * MR, p * NR, width * NR);
                // SAFETY: as above.
                unsafe {
                    let c = c.add(p * NR);
                    self.kernel
                        .run(kc, a, lda, full, slices, c, ldc, first, finish);
                }
            }
            if rest > 0 {
                for q in 0..width {
                    let (row, col) = (full * MR, (p + q) * NR);
                    let finish = self.finish(epilogue, row, rest, col, NR);
                    let panel = slices.values[q * slices.next..].as_ptr();
                    // SAFETY: as above.
                    unsafe {
                        let (a, c) = (a.add(row * lda), c.add(row * ldc + col));
                        self.kernel
                            .tile(rest, kc, a, lda, panel, c, ldc, first, finish);
                    }
                }
            }
        }
    }

    fn finish(
        &self,
        epilogue: Option<Placed>,
        row: usize,
        rows: usize,
        col: usize,
        cols: usize,
    ) -> Option<Finish> {
        epilogue.map(
            |Placed {
                 epilogue,
                 row0,
                 stride,
             }| Finish {
                bias: epilogue.bias()[col..col + cols].as_ptr(),
                gelu: matches!(epilogue, Epilogue::BiasGelu(_)),
                residual: match epilogue {
                    Epilogue::BiasResidual(_, residual) => {
                        let at = (row0 + row) * stride + col;
                        residual[at..at + (rows - 1) * stride + cols].as_ptr()
                    }
                    _ => std::ptr::null(),
                },
                stride,
            },
        )
    }

    /// Padded storage, same reduction order: edge tiles match full ones.
    ///
    /// # Safety
    /// As for [`Band::block`].
    #[allow(clippy::too_many_arguments)]
    unsafe fn edge(
        &self,
        kc: usize,
        a: *const f32,
        lda: usize,
        panel: &[f32],
        c: *mut f32,
        tile: Tile,
        first: bool,
        epilogue: Option<Placed>,
    ) {
        let ldc = self.ldc;
        let mut buffer = [0.0f32; MR * NR];
        let mut bias = [0.0f32; NR];
        let mut residual = [0.0f32; MR * NR];
        for r in 0..tile.rows {
            let at = (tile.row + r) * ldc + tile.col;
            // SAFETY: the caller vouches for the tile's cells at c.
            let row = unsafe { std::slice::from_raw_parts(c.add(at), tile.cols) };
            buffer[r * NR..r * NR + tile.cols].copy_from_slice(row);
        }
        let finish = epilogue.map(
            |Placed {
                 epilogue,
                 row0,
                 stride,
             }| {
                bias[..tile.cols].copy_from_slice(&epilogue.bias()[tile.col..tile.col + tile.cols]);
                let with_residual = if let Epilogue::BiasResidual(_, source) = epilogue {
                    for r in 0..tile.rows {
                        let at = (row0 + tile.row + r) * stride + tile.col;
                        residual[r * NR..r * NR + tile.cols]
                            .copy_from_slice(&source[at..at + tile.cols]);
                    }
                    true
                } else {
                    false
                };
                Finish {
                    bias: bias.as_ptr(),
                    gelu: matches!(epilogue, Epilogue::BiasGelu(_)),
                    residual: if with_residual {
                        residual.as_ptr()
                    } else {
                        std::ptr::null()
                    },
                    stride: NR,
                }
            },
        );
        // SAFETY: the buffers hold a tile, and the caller vouches for a.
        unsafe {
            let (a, panel, out) = (a.add(tile.row * lda), panel.as_ptr(), buffer.as_mut_ptr());
            self.kernel
                .tile(tile.rows, kc, a, lda, panel, out, NR, first, finish);
        }
        for r in 0..tile.rows {
            let at = (tile.row + r) * ldc + tile.col;
            // SAFETY: as above, and no other thread touches these cells.
            let row = unsafe { std::slice::from_raw_parts_mut(c.add(at), tile.cols) };
            row.copy_from_slice(&buffer[r * NR..r * NR + tile.cols]);
        }
    }
}

impl Finish {
    /// # Safety
    /// A residual must extend that far.
    unsafe fn down(self, rows: usize) -> Self {
        Self {
            residual: if self.residual.is_null() {
                self.residual
            } else {
                // SAFETY: the caller vouches for the residual's extent.
                unsafe { self.residual.add(rows * self.stride) }
            },
            ..self
        }
    }
}

impl Kernel {
    /// Whole panels one [`Kernel::run`] call covers: AVX2 pairs them for its 3 x 32 tiles.
    fn span(self) -> usize {
        match self {
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => 2,
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            Self::Neon => 1,
            Self::Portable => 1,
        }
    }

    /// # Safety
    /// `a` holds `count` * MR rows of `kc` values `lda` apart, and `c` and `finish` those rows of
    /// `slices.width` * NR values.
    #[allow(clippy::too_many_arguments)]
    unsafe fn run(
        self,
        kc: usize,
        a: *const f32,
        lda: usize,
        count: usize,
        slices: Slices,
        c: *mut f32,
        ldc: usize,
        first: bool,
        finish: Option<Finish>,
    ) {
        assert!(
            (1..=self.span()).contains(&slices.width)
                && slices.values.len() >= (slices.width - 1) * slices.next + kc * NR
        );
        #[cfg(target_arch = "x86_64")]
        if self == Self::Avx2 && slices.width == 2 {
            // SAFETY: Avx2 implies AVX2 and FMA; the caller vouches for every address.
            unsafe { pairs_avx2(kc, count, a, lda, slices, c, ldc, first, finish) };
            return;
        }
        for s in 0..count {
            // SAFETY: the caller vouches for every strip's rows at a, c and in finish.
            unsafe {
                let (a, c) = (a.add(s * MR * lda), c.add(s * MR * ldc));
                let finish = finish.map(|finish| finish.down(s * MR));
                self.tile(
                    MR,
                    kc,
                    a,
                    lda,
                    slices.values.as_ptr(),
                    c,
                    ldc,
                    first,
                    finish,
                );
            }
        }
    }

    /// # Safety
    /// `a` holds `rows` (1 to MR) rows of `kc` values `lda` apart, `panel` kc * NR values, and `c`
    /// and `finish` those rows of NR values.
    #[allow(clippy::too_many_arguments)]
    unsafe fn tile(
        self,
        rows: usize,
        kc: usize,
        a: *const f32,
        lda: usize,
        panel: *const f32,
        c: *mut f32,
        ldc: usize,
        first: bool,
        finish: Option<Finish>,
    ) {
        // Specializing row tails avoids padding work without changing reduction order.
        macro_rules! tile {
            ($kernel:ident) => {
                match rows {
                    1 => $kernel::<1>(kc, a, lda, panel, c, ldc, first, finish),
                    2 => $kernel::<2>(kc, a, lda, panel, c, ldc, first, finish),
                    3 => $kernel::<3>(kc, a, lda, panel, c, ldc, first, finish),
                    4 => $kernel::<4>(kc, a, lda, panel, c, ldc, first, finish),
                    5 => $kernel::<5>(kc, a, lda, panel, c, ldc, first, finish),
                    _ => $kernel::<MR>(kc, a, lda, panel, c, ldc, first, finish),
                }
            };
        }
        // SAFETY: the caller vouches for every address; the entry points check the features.
        unsafe {
            match self {
                #[cfg(target_arch = "x86_64")]
                Self::Avx2 => tile!(tile_rows_avx2),
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                Self::Neon => tile!(tile_rows_neon),
                Self::Portable => tile!(tile_rows_portable),
            }
        }
    }
}

/// Strips by a panel pair as 3 x 32 tiles: half 6 x 16's broadcasts, each an extra uop on E-cores.
///
/// # Safety
/// AVX2 and FMA; `a` holds `count` * MR rows of `kc` values `lda` apart, `slices` two panels, and
/// `c` and `finish` those rows of 2NR values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn pairs_avx2(
    kc: usize,
    count: usize,
    a: *const f32,
    lda: usize,
    slices: Slices,
    c: *mut f32,
    ldc: usize,
    first: bool,
    finish: Option<Finish>,
) {
    use std::arch::x86_64::*;
    use std::mem::MaybeUninit;
    let (b, next) = (slices.values.as_ptr(), slices.next);
    let zero: Triple = [[_mm256_setzero_ps(); 4]; 3];
    for chunk in (0..count).step_by(CHUNK) {
        let strips = CHUNK.min(count - chunk);
        let mut carried: [[Carry; 2]; CHUNK] = [[[MaybeUninit::uninit(); 12]; 2]; CHUNK];
        for start in (0..kc).step_by(DEPTH) {
            let depths = start..kc.min(start + DEPTH);
            for (s, carries) in carried[..strips].iter_mut().enumerate() {
                for (t, carry) in carries.iter_mut().enumerate() {
                    // SAFETY: the caller vouches for the rows and panel slices read, and a carry
                    // is read only after the previous depth block wrote it.
                    let sums = unsafe {
                        let a = a.add(((chunk + s) * MR + 3 * t) * lda);
                        let from = if start == 0 {
                            zero
                        } else {
                            std::array::from_fn(|r| {
                                std::array::from_fn(|h| carry[4 * r + h].assume_init())
                            })
                        };
                        triple_avx2(a, lda, b, next, depths.clone(), from)
                    };
                    for (slot, &sum) in carry.iter_mut().zip(sums.as_flattened()) {
                        slot.write(sum);
                    }
                    if depths.end == kc {
                        let row = (chunk + s) * MR + 3 * t;
                        // SAFETY: the caller vouches for these rows at c and in finish, and every
                        // slot was just written.
                        unsafe { finish_triple_avx2(carry, c, ldc, row, first, finish) };
                    }
                }
            }
        }
    }
}

/// Stores a 3 x 32 tile's sums at row `row`: out of line, so the pair loop keeps its registers.
///
/// # Safety
/// AVX2 and FMA; every slot written, and rows `row..row + 3` of 2NR values at `c` and in `finish`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn finish_triple_avx2(
    sums: &Carry,
    c: *mut f32,
    ldc: usize,
    row: usize,
    first: bool,
    finish: Option<Finish>,
) {
    // SAFETY: the caller vouches for every slot and address.
    unsafe {
        let (c, finish) = (c.add(row * ldc), finish.map(|finish| finish.down(row)));
        for (r, slots) in sums.as_chunks::<4>().0.iter().enumerate() {
            let values = slots.map(|slot| slot.assume_init());
            store_avx2(values, c, ldc, r, first, finish);
        }
    }
}

#[cfg(target_arch = "x86_64")]
type Triple = [[std::arch::x86_64::__m256; 4]; 3];

/// A 3 x 32 tile's sums between depth blocks, a slot per vector: a whole-Triple copy is a memmove.
#[cfg(target_arch = "x86_64")]
type Carry = [std::mem::MaybeUninit<std::arch::x86_64::__m256>; 12];

/// Three rows by two panels over `depths`, from `acc`: 12 sums, 3 held A values, 1 B vector.
///
/// # Safety
/// As for [`pairs_avx2`], for these three rows and depths.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn triple_avx2(
    a: *const f32,
    lda: usize,
    b: *const f32,
    next: usize,
    depths: std::ops::Range<usize>,
    mut acc: Triple,
) -> Triple {
    use std::arch::x86_64::*;
    steps(depths, |p| {
        // SAFETY: the caller vouches for every address read; a prefetch never faults.
        unsafe {
            _mm_prefetch::<_MM_HINT_T0>(b.wrapping_add((p + AHEAD) * NR).cast());
            _mm_prefetch::<_MM_HINT_T0>(b.wrapping_add(next + (p + AHEAD) * NR).cast());
            let values: [__m256; 3] =
                std::array::from_fn(|r| _mm256_broadcast_ss(&*a.add(r * lda + p)));
            for j in 0..4 {
                let column = _mm256_loadu_ps(b.add(j / 2 * next + p * NR + j % 2 * 8));
                for (row, &value) in acc.iter_mut().zip(&values) {
                    row[j] = _mm256_fmadd_ps(value, column, row[j]);
                }
            }
        }
    });
    acc
}

/// `step(p)` for every depth in `depths`, STEPS an iteration.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn steps(depths: std::ops::Range<usize>, mut step: impl FnMut(usize)) {
    let (mut p, end) = (depths.start, depths.end);
    while p + STEPS <= end {
        for u in 0..STEPS {
            step(p + u);
        }
        p += STEPS;
    }
    for p in p..end {
        step(p);
    }
}

/// A row's vectors take GELU together, so their dependent chains overlap.
///
/// # Safety
/// AVX2 and FMA; row `r`'s N vectors at `c` and in `finish`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn store_avx2<const N: usize>(
    sums: [std::arch::x86_64::__m256; N],
    c: *mut f32,
    ldc: usize,
    r: usize,
    first: bool,
    finish: Option<Finish>,
) {
    use std::arch::x86_64::*;
    // SAFETY: the caller vouches for every address.
    unsafe {
        let at = c.add(r * ldc);
        let mut values = sums;
        if !first {
            for (i, value) in values.iter_mut().enumerate() {
                *value = _mm256_add_ps(_mm256_loadu_ps(at.add(8 * i)), *value);
            }
        }
        if let Some(finish) = finish {
            for (i, value) in values.iter_mut().enumerate() {
                *value = _mm256_add_ps(*value, _mm256_loadu_ps(finish.bias.add(8 * i)));
            }
            if finish.gelu {
                values = math::avx2::gelu_unbounded_n(values);
            }
            if !finish.residual.is_null() {
                let residual = finish.residual.add(r * finish.stride);
                for (i, value) in values.iter_mut().enumerate() {
                    *value = _mm256_add_ps(*value, _mm256_loadu_ps(residual.add(8 * i)));
                }
            }
        }
        for (i, value) in values.into_iter().enumerate() {
            _mm256_storeu_ps(at.add(8 * i), value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_display_their_instruction_sets() {
        assert_eq!(Kernel::Portable.to_string(), "portable");
        #[cfg(target_arch = "x86_64")]
        assert_eq!(Kernel::Avx2.to_string(), "AVX2");
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        assert_eq!(Kernel::Neon.to_string(), "NEON");
    }

    #[test]
    fn short_inputs_match_batch_rows_in_every_pool() {
        // A short last panel and K block, so the edge tiles run.
        let (n, k) = (7 * NR + 5, 3 * KC + 11);
        let w: Vec<f32> = (0..n * k)
            .map(|at| ((at * 37 % 101) as f32 - 50.0) / 64.0)
            .collect();
        let weight = PackedWeight::new(&w, n, k);
        let bias: Vec<f32> = (0..n).map(|j| (j % 7) as f32 / 8.0 - 0.4).collect();
        let batch = SHORT_ROWS + 2 * MC;
        let a: Vec<f32> = (0..batch * k)
            .map(|at| ((at * 23 % 97) as f32 - 48.0) / 50.0)
            .collect();
        let residual: Vec<f32> = (0..batch * n)
            .map(|at| ((at % 13) as f32 - 6.0) / 8.0)
            .collect();
        let kernel = Kernel::detect();
        for epilogue in [
            Epilogue::Bias(&bias),
            Epilogue::BiasGelu(&bias),
            Epilogue::BiasResidual(&bias, &residual),
        ] {
            let mut all = vec![0.0; batch * n];
            crate::worker_pool("gemm-test-batch", 1)
                .unwrap()
                .install(|| gemm_current_pool(kernel, &a, batch, &weight, epilogue, &mut all));
            for threads in [2, 3, 8] {
                let pool = crate::worker_pool("gemm-test-short", threads).unwrap();
                for (start, rows) in [(0, 1), (5, 13), (40, SHORT_ROWS)] {
                    let part = match epilogue {
                        Epilogue::BiasResidual(bias, residual) => {
                            Epilogue::BiasResidual(bias, &residual[start * n..])
                        }
                        other => other,
                    };
                    let mut c = vec![f32::NAN; rows * n];
                    pool.install(|| {
                        let a = &a[start * k..(start + rows) * k];
                        gemm_current_pool(kernel, a, rows, &weight, part, &mut c)
                    });
                    assert!(
                        c.iter()
                            .zip(&all[start * n..(start + rows) * n])
                            .all(|(x, y)| x.to_bits() == y.to_bits()),
                        "{threads} threads: rows {start}..{} differ",
                        start + rows
                    );
                }
            }
        }
    }

    /// One output by a kernel's recipe: per K block a chain from zero, added to earlier blocks.
    fn recipe_total(kernel: Kernel, a: &[f32], w: &[f32]) -> f32 {
        let mut total = 0.0;
        for k0 in (0..a.len()).step_by(KC) {
            let terms = a[k0..].iter().zip(&w[k0..]).take(KC);
            let sum = if kernel == Kernel::Portable {
                terms.fold(0.0_f32, |sum, (&x, &y)| sum + x * y)
            } else {
                terms.fold(0.0_f32, |sum, (&x, &y)| x.mul_add(y, sum))
            };
            total = if k0 == 0 { sum } else { total + sum };
        }
        total
    }

    fn recipe(kernel: Kernel, a: &[f32], w: &[f32], m: usize, epilogue: Epilogue) -> Vec<f32> {
        let k = a.len() / m;
        let n = w.len() / k;
        let mut c = vec![0.0; m * n];
        for (i, row) in c.chunks_exact_mut(n).enumerate() {
            for (j, out) in row.iter_mut().enumerate() {
                let total = recipe_total(kernel, &a[i * k..][..k], &w[j * k..][..k]);
                let mut value = total + epilogue.bias()[j];
                if let Epilogue::BiasGelu(_) = epilogue {
                    value = math::gelu_unbounded(value);
                }
                if let Epilogue::BiasResidual(_, residual) = epilogue {
                    value += residual[i * n + j];
                }
                *out = value;
            }
        }
        c
    }

    /// Kernels on different processors that repeat one recipe give the same bits.
    #[test]
    fn simd_kernels_match_the_fused_recipe_bit_for_bit() {
        let kernel = Kernel::detect();
        if kernel == Kernel::Portable {
            return;
        }
        // Panel pairs, a lone whole panel, a partial one, and K blocks of two depths.
        let (n, k) = ((GATHER_PANELS + 1) * NR + 5, 2 * KC + 37);
        let w: Vec<f32> = (0..n * k)
            .map(|at| ((at * 37 % 101) as f32 - 50.0) / 64.0)
            .collect();
        let weight = PackedWeight::new(&w, n, k);
        let bias: Vec<f32> = (0..n).map(|j| (j % 7) as f32 / 8.0 - 0.4).collect();
        // One worker's run feeds every panel, so short bands gather their rows; two read in place.
        let pools = [1, 2].map(|threads| crate::worker_pool("gemm-test-recipe", threads).unwrap());
        for m in [1, 13, SHORT_ROWS + MR + 1] {
            let a: Vec<f32> = (0..m * k)
                .map(|at| ((at * 23 % 97) as f32 - 48.0) / 50.0)
                .collect();
            let residual: Vec<f32> = (0..m * n)
                .map(|at| ((at % 13) as f32 - 6.0) / 8.0)
                .collect();
            for epilogue in [
                Epilogue::Bias(&bias),
                Epilogue::BiasGelu(&bias),
                Epilogue::BiasResidual(&bias, &residual),
            ] {
                let want = recipe(kernel, &a, &w, m, epilogue);
                for pool in &pools {
                    let mut c = vec![f32::NAN; m * n];
                    pool.install(|| gemm_current_pool(kernel, &a, m, &weight, epilogue, &mut c));
                    let threads = pool.current_num_threads();
                    for (at, (x, y)) in c.iter().zip(&want).enumerate() {
                        assert_eq!(
                            x.to_bits(),
                            y.to_bits(),
                            "{kernel:?}, {threads} threads: {m} rows, at {at}: {x} vs {y}"
                        );
                    }
                }
            }
        }
    }

    /// Products of strided rows keep their kernel's recipe for every row tail and a partial panel.
    #[test]
    fn products_match_their_kernels_recipe_bit_for_bit() {
        // Panel pairs, a lone whole panel and a partial one, two K blocks, wide strides; rows read
        // in place, then gathered for enough panels.
        let (k, lda) = (KC + 37, KC + 41);
        let a: Vec<f32> = (0..13 * lda)
            .map(|at| ((at * 17 % 83) as f32 - 41.0) / 40.0)
            .collect();
        for n in [3 * NR + 5, (GATHER_PANELS + 1) * NR + 5] {
            let ldc = n + 3;
            let w: Vec<f32> = (0..n * k)
                .map(|at| ((at * 29 % 89) as f32 - 44.0) / 32.0)
                .collect();
            let weight = PackedWeight::new(&w, n, k);
            let panels = weight.columns(0, n.div_ceil(NR));
            for kernel in [Kernel::detect(), Kernel::Portable] {
                for rows in 1..=13 {
                    let mut c = vec![f32::NAN; rows * ldc];
                    product(kernel, &a, lda, rows, panels, &mut c, ldc);
                    for i in 0..rows {
                        for j in 0..n {
                            let want = recipe_total(kernel, &a[i * lda..][..k], &w[j * k..][..k]);
                            let got = c[i * ldc + j];
                            assert_eq!(
                                got.to_bits(),
                                want.to_bits(),
                                "{kernel:?}, {n} columns: {rows} rows, at ({i}, {j})"
                            );
                        }
                    }
                }
            }
        }
    }
}
