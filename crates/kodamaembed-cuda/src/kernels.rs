//! Typed launches of the embedded kernels; the caller binds their context first.

use crate::driver::{Args, Context, CudaError, DeviceBuffer, Function, Module, Stream};
use half::f16;
use kodamaembed_model::{Activation, Geometry, Pooling, Rotary, relative::RelativeBias};
use std::ops::RangeInclusive;

const NORM_PTX: &str = include_str!("../ptx/norm.ptx");
const GEMM_PTX: &str = include_str!("../ptx/gemm.ptx");
const GEMM_PAIRED_PTX: &str = include_str!("../ptx/gemm_paired.ptx");
const ATTENTION_PTX: &str = include_str!("../ptx/attention.ptx");
const POOL_PTX: &str = include_str!("../ptx/pool.ptx");
const PREDICTION_HEAD_PTX: &str = include_str!("../ptx/prediction_head.ptx");

/// Every tiling's K step divides this.
const GEMM_TILE_K: usize = 64;
pub(crate) const ATTENTION_BLOCK: usize = 64;
/// Both attention kernels stage six tiles of 64 rows of one head.
const ATTENTION_TILES: usize = 6;
/// Sequences this long or longer skip the attention correction, whose extra score work grows with
/// the square of their length.
pub const CORRECTED_BELOW: usize = 256;
/// Longest sequences whose pipelined attention takes 32-row blocks, which pad less and fill more SMs.
const ROWS32: RangeInclusive<usize> = 49..=96;
const _: () = assert!(
    *ROWS32.end() < CORRECTED_BELOW,
    "32-row blocks serve corrected sequences only"
);

#[derive(Clone, Copy)]
pub(crate) struct Tiling {
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) threads: u32,
    pub(crate) shared_bytes: u32,
}

/// The launch shape of gemm.cuh's `Tiling<rows, columns, k, warp_rows, warp_columns, stages>`.
const fn tiling(
    rows: usize,
    columns: usize,
    k: usize,
    warp_rows: usize,
    warp_columns: usize,
    stages: usize,
) -> Tiling {
    Tiling {
        rows,
        columns,
        threads: ((rows / warp_rows) * (columns / warp_columns) * 32) as u32,
        shared_bytes: (stages * (rows + columns) * k * size_of::<f16>()) as u32,
    }
}

/// gemm.cuh's r tilings, named after their output tile's rows and columns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tile {
    R16C32,
    R16C64,
    R32C32,
    R48C32,
    R48C64,
    R48C128,
    R80C64,
    R80C128,
    R96C128,
    R64C64S3,
    R64C64S4,
    R64C128,
    R64C128K64,
    R128C64,
    R128C128,
    R128C128W8,
    R128C128W8K64,
    R256C128,
    R128C88,
    R128C176,
}

impl Tile {
    /// In declaration order, which indexes [`Tiled`]'s functions.
    const ALL: [Self; 20] = [
        Self::R16C32,
        Self::R16C64,
        Self::R32C32,
        Self::R48C32,
        Self::R48C64,
        Self::R48C128,
        Self::R80C64,
        Self::R80C128,
        Self::R96C128,
        Self::R64C64S3,
        Self::R64C64S4,
        Self::R64C128,
        Self::R64C128K64,
        Self::R128C64,
        Self::R128C128,
        Self::R128C128W8,
        Self::R128C128W8K64,
        Self::R256C128,
        Self::R128C88,
        Self::R128C176,
    ];

    /// Only the non-power-of-two kernels mask a partial column tile.
    fn for_columns(self, n: usize) -> Self {
        let columns = self.spec().1.columns;
        if columns.is_power_of_two() && !n.is_multiple_of(columns) {
            Self::R128C88
        } else {
            self
        }
    }

    /// The kernel name suffix and the launch shape, as gemm.cuh declares them.
    const fn spec(self) -> (&'static str, Tiling) {
        match self {
            Self::R16C32 => ("r16c32", tiling(16, 32, 64, 16, 16, 6)),
            Self::R16C64 => ("r16c64", tiling(16, 64, 32, 16, 32, 6)),
            Self::R32C32 => ("r32c32", tiling(32, 32, 32, 16, 16, 6)),
            Self::R48C32 => ("r48c32", tiling(48, 32, 64, 48, 16, 4)),
            Self::R48C64 => ("r48c64", tiling(48, 64, 64, 48, 16, 4)),
            Self::R48C128 => ("r48c128", tiling(48, 128, 32, 48, 32, 4)),
            Self::R80C64 => ("r80c64", tiling(80, 64, 64, 80, 16, 5)),
            Self::R80C128 => ("r80c128", tiling(80, 128, 32, 80, 32, 7)),
            Self::R96C128 => ("r96c128", tiling(96, 128, 32, 96, 32, 4)),
            Self::R64C64S3 => ("r64c64s3", tiling(64, 64, 64, 32, 32, 3)),
            Self::R64C64S4 => ("r64c64s4", tiling(64, 64, 64, 32, 32, 4)),
            Self::R64C128 => ("r64c128", tiling(64, 128, 32, 32, 64, 4)),
            Self::R64C128K64 => ("r64c128k64", tiling(64, 128, 64, 32, 64, 3)),
            Self::R128C64 => ("r128c64", tiling(128, 64, 32, 64, 32, 4)),
            Self::R128C128 => ("r128c128", tiling(128, 128, 32, 64, 64, 3)),
            Self::R128C128W8 => ("r128c128w8", tiling(128, 128, 32, 64, 32, 4)),
            Self::R128C128W8K64 => ("r128c128w8k64", tiling(128, 128, 64, 64, 32, 3)),
            Self::R256C128 => ("r256c128", tiling(256, 128, 32, 64, 64, 4)),
            Self::R128C88 => ("r128c88", tiling(128, 88, 64, 32, 88, 3)),
            Self::R128C176 => ("r128c176", tiling(128, 176, 32, 32, 88, 3)),
        }
    }
}

/// The tiling that ran fastest for `m` rows of an `n` x `k` projection no grid tiling takes, told by
/// its shape: QKV is 3k wide, FFN up 4k wide, FFN down 4n deep and the output projection square.
fn pick(m: usize, n: usize, k: usize) -> Tile {
    use Tile::{
        R16C32, R16C64, R32C32, R48C32, R48C64, R48C128, R64C64S3, R64C64S4, R64C128, R64C128K64,
        R80C64, R80C128, R96C128, R128C64, R128C88, R128C128, R128C128W8, R128C128W8K64, R128C176,
        R256C128,
    };
    let (qkv, up, down) = (n == 3 * k, n == 4 * k, k == 4 * n);
    if n >= 1024 && k >= 1024 {
        // Row-exact tiles keep 17-96 rows in one row band.
        return match m {
            0..=16 => R16C32,
            17..=48 if qkv => R48C64,
            17..=48 if up => R48C128,
            17..=48 => R48C32,
            // Two row tiles of 48 give the contraction 32 blocks: 16 leave a card whose tensor
            // pipes accumulate FP32 at half rate waiting on them.
            49..=96 if down => R48C64,
            49..=64 if qkv || up => R64C64S3,
            49..=64 => R32C32,
            65..=80 if up => R80C128,
            65..=80 => R80C64,
            81..=96 if qkv || up => R96C128,
            81..=96 => R64C64S4,
            // The expansion covers 192 rows in three tiles instead of rounding to 256.
            129..=192 if up && n.is_multiple_of(128) => R64C128K64,
            97..=128 if qkv => R128C64,
            97..=192 if up => R128C128W8K64,
            97..=192 => R64C64S4,
            193..=384 => R64C128K64,
            _ if qkv => R256C128,
            _ if up => R128C176,
            _ => R128C88,
        };
    }
    match m {
        0..=16 => R16C32,
        17..=48 if qkv => R16C64,
        17..=48 if up => R64C64S3,
        17..=48 => R16C32,
        49..=64 if up => R64C64S3,
        49..=64 => R32C32,
        65..=96 if qkv => R64C64S3,
        65..=96 if up => R64C128,
        65..=96 => R32C32,
        97..=192 if qkv || up => R64C64S3,
        97..=192 => R64C64S4,
        193..=384 if qkv => R128C128W8,
        193..=384 if up => R128C64,
        193..=384 => R64C128,
        _ if qkv => R256C128,
        _ if up => R128C128,
        _ => R64C128,
    }
}

/// A gemm.cu grid tiling for a row band of one projection of a 1024-wide layer, one block a tile.
struct GridTile {
    rows: RangeInclusive<usize>,
    columns: usize,
    depth: usize,
    suffix: &'static str,
    tiling: Tiling,
}

impl GridTile {
    const fn new(
        rows: RangeInclusive<usize>,
        (columns, depth): (usize, usize),
        suffix: &'static str,
        tiling: Tiling,
    ) -> Self {
        Self {
            rows,
            columns,
            depth,
            suffix,
            tiling,
        }
    }

    fn fits(&self, m: usize, n: usize, k: usize) -> bool {
        k == self.depth && n == self.columns && self.rows.contains(&m)
    }
}

/// The columns and depth of a 1024-wide layer's projections.
const QKV: (usize, usize) = (3072, 1024);
const UP: (usize, usize) = (4096, 1024);
const OUT: (usize, usize) = (1024, 1024);
const DOWN: (usize, usize) = (1024, 4096);

/// QKV at 129-512 rows: two row tiles by 24 column tiles, one wave of 48 blocks.
const QKV_GRID: &[GridTile] = &[
    GridTile::new(129..=160, QKV, "g80c128", tiling(80, 128, 32, 80, 32, 7)),
    GridTile::new(161..=192, QKV, "g96c128", tiling(96, 128, 32, 96, 32, 4)),
    GridTile::new(193..=224, QKV, "g112c128", tiling(112, 128, 64, 112, 16, 3)),
    GridTile::new(225..=256, QKV, "g128c128", tiling(128, 128, 64, 64, 32, 3)),
    GridTile::new(257..=288, QKV, "g144c128", tiling(144, 128, 32, 48, 32, 4)),
    GridTile::new(289..=320, QKV, "g160c128", tiling(160, 128, 32, 80, 32, 4)),
    GridTile::new(321..=384, QKV, "g192c128", tiling(192, 128, 32, 96, 32, 4)),
    GridTile::new(385..=512, QKV, "g256c128", tiling(256, 128, 32, 64, 64, 4)),
];

/// The feed-forward expansion at 193-512 rows.
const UP_GRID: &[GridTile] = &[
    GridTile::new(193..=240, UP, "g80c256", tiling(80, 256, 32, 80, 32, 4)),
    GridTile::new(241..=256, UP, "g128c192", tiling(128, 192, 32, 64, 48, 4)),
    GridTile::new(257..=288, UP, "g96c256", tiling(96, 256, 32, 48, 64, 4)),
    GridTile::new(289..=320, UP, "g112c256", tiling(112, 256, 32, 112, 32, 4)),
    GridTile::new(321..=384, UP, "g192c176", tiling(192, 176, 32, 48, 88, 4)),
    GridTile::new(385..=512, UP, "g128c176", tiling(128, 176, 32, 32, 88, 3)),
];

/// The output projection at 193-320 rows and the feed-forward contraction at 193-288.
const RESIDUAL_GRID: &[GridTile] = &[
    GridTile::new(193..=224, OUT, "g112c64", tiling(112, 64, 64, 112, 16, 3)),
    GridTile::new(225..=288, OUT, "g96c64", tiling(96, 64, 32, 48, 32, 4)),
    GridTile::new(289..=320, OUT, "g64c128", tiling(64, 128, 64, 32, 64, 3)),
    GridTile::new(193..=224, DOWN, "g112c64", tiling(112, 64, 64, 112, 16, 3)),
    GridTile::new(225..=240, DOWN, "g80c96", tiling(80, 96, 64, 80, 24, 4)),
    GridTile::new(241..=256, DOWN, "g64c128", tiling(64, 128, 64, 32, 64, 3)),
    GridTile::new(257..=288, DOWN, "g96c80", tiling(96, 80, 64, 48, 40, 4)),
];

/// One GEMM epilogue in every tiling, with the blocks a multiprocessor holds, and its grid tilings.
pub(crate) struct Tiled {
    functions: Vec<(Function, u32)>,
    grid: Vec<(&'static GridTile, Function)>,
    sm_count: u32,
}

impl Tiled {
    fn load(
        context: &Context,
        module: &Module,
        name: &str,
        grid_tiles: &'static [GridTile],
    ) -> Result<Self, CudaError> {
        let load = |suffix: &str, tiling: Tiling| -> Result<Function, CudaError> {
            let function = module.function(&format!("{name}_{suffix}"))?;
            if tiling.shared_bytes > 48 * 1024 {
                function.set_max_dynamic_shared(tiling.shared_bytes)?;
            }
            Ok(function)
        };
        let mut functions = Vec::with_capacity(Tile::ALL.len());
        for tile in Tile::ALL {
            let (suffix, tiling) = tile.spec();
            let function = load(suffix, tiling)?;
            let resident = function.resident_blocks(tiling.threads, tiling.shared_bytes)?;
            assert!(resident > 0, "{name}_{suffix} fits a multiprocessor");
            functions.push((function, resident));
        }
        let grid = grid_tiles
            .iter()
            .map(|tile| Ok((tile, load(tile.suffix, tile.tiling)?)))
            .collect::<Result<_, CudaError>>()?;
        Ok(Self {
            functions,
            grid,
            sm_count: context.sm_count,
        })
    }

    /// The tiling and grid for `gemm`: a grid tiling's tiles, else at most the resident blocks.
    fn launch_shape(&self, gemm: &Gemm) -> (&Function, Tiling, [u32; 3]) {
        if let Some((tile, function)) = self
            .grid
            .iter()
            .find(|(tile, _)| tile.fits(gemm.m, gemm.n, gemm.k))
        {
            let shape = tile.tiling;
            let grid = [blocks(gemm.m, shape.rows), blocks(gemm.n, shape.columns), 1];
            return (function, shape, grid);
        }
        let tile = pick(gemm.m, gemm.n, gemm.k).for_columns(gemm.n);
        let (function, resident) = &self.functions[tile as usize];
        let tiling = tile.spec().1;
        let tiles = gemm.m.div_ceil(tiling.rows) * gemm.n.div_ceil(tiling.columns);
        let slots = *resident as usize * self.sm_count as usize;
        let grid = u32::try_from(tiles.min(slots)).expect("grid fits u32");
        (function, tiling, [grid, 1, 1])
    }

    /// # Safety
    /// `args` must match the epilogue's kernels, and its buffers must outlive the work.
    unsafe fn launch(
        &self,
        stream: &Stream,
        gemm: &Gemm,
        args: &mut Args,
    ) -> Result<(), CudaError> {
        let (function, tiling, grid) = self.launch_shape(gemm);
        // SAFETY: forwarded from the caller; every tiling takes the same arguments.
        unsafe {
            function.launch(
                stream,
                grid,
                [tiling.threads, 1, 1],
                tiling.shared_bytes,
                args,
            )
        }
    }
}

/// An attention kernel with the correction and its plain twin for long sequences.
struct Pair {
    corrected: Function,
    plain: Function,
}

impl Pair {
    /// `{prefix}{suffix}` and `{prefix}_plain{suffix}`.
    fn load(module: &Module, prefix: &str, suffix: &str) -> Result<Self, CudaError> {
        Ok(Self {
            corrected: module.function(&format!("{prefix}{suffix}"))?,
            plain: module.function(&format!("{prefix}_plain{suffix}"))?,
        })
    }

    fn kinds(&self) -> (&Function, Option<&Function>) {
        (&self.corrected, Some(&self.plain))
    }
}

/// The attention kernels for a geometry's heads.
enum Attention {
    /// Blocks of 64 query rows.
    Blocks(Pair),
    /// Sixteen heads of 64: blocks of 32 or 64 query rows that issue the next key block's scores
    /// beside this block's softmax.
    Pipelined { rows32: Function, rows64: Pair },
}

pub struct Kernels {
    geometry: Geometry,
    layernorm: Function,
    embed_layernorm: Function,
    gemm_bias_f16: Tiled,
    gemm_bias_gelu_f16: Tiled,
    gemm_bias_residual_f32: Tiled,
    attention: Attention,
    relative_attention: Pair,
    local_attention: Function,
    column_means: Function,
    normalize_means: Function,
    cls_pool_normalize: Function,
    rerank_head: Function,
}

/// A batch of whole sequences packed back to back without padding.
pub struct Packed<'a> {
    /// Cumulative lengths: the first `sequences + 1` values run from 0 to `tokens`.
    pub cu_seqlens: &'a DeviceBuffer<i32>,
    pub sequences: usize,
    pub tokens: usize,
    pub max_len: usize,
    pub min_len: usize,
}

/// MPNet's relative-position bias on the device, as [`Kernels::attention`] reads it.
pub struct RelativeTable {
    values: DeviceBuffer<f32>,
    longest: i32,
}

impl RelativeTable {
    pub fn upload(context: &Context, bias: &RelativeBias) -> Result<Self, CudaError> {
        Ok(Self {
            values: DeviceBuffer::from_slice(context, bias.values())?,
            longest: i32::try_from(bias.longest()).expect("fits i32"),
        })
    }
}

/// Rotary positions on the device, as [`PairedGemms::gemm_bias_turned`] reads them:
/// [`Rotary::paired_turns`].
pub struct Turns {
    pairs: DeviceBuffer<f32>,
    positions: usize,
    head: usize,
}

impl Turns {
    /// The turns of base `theta` at the first `positions` places of a sequence, for heads `head`
    /// columns wide, a power of two.
    pub fn upload(
        context: &Context,
        head: usize,
        theta: f32,
        positions: usize,
    ) -> Result<Self, CudaError> {
        assert!(head.is_power_of_two() && head >= 2);
        let pairs = Rotary::paired_turns(theta, head, positions);
        Ok(Self {
            pairs: DeviceBuffer::from_slice(context, &pairs)?,
            positions,
            head,
        })
    }
}

/// What [`PairedGemms::gemm_bias_turned`] turns: the first `columns` columns of each row, by
/// the row's position.
pub struct Turned<'a> {
    /// Each row's place in its sequence.
    pub positions: &'a DeviceBuffer<i32>,
    pub turns: &'a Turns,
    pub columns: usize,
    /// The longest sequence, which `turns` must reach.
    pub longest: usize,
}

impl Packed<'_> {
    pub(crate) fn check(&self) {
        assert!(self.sequences > 0 && self.min_len > 0 && self.min_len <= self.max_len);
        assert!(self.tokens >= self.max_len);
        assert!(self.cu_seqlens.len() > self.sequences);
    }
}

/// Embedding tables and their LayerNorm, all FP32.
pub struct Embeddings<'a> {
    pub word: &'a DeviceBuffer<f32>,
    pub position: &'a DeviceBuffer<f32>,
    pub token_type: &'a DeviceBuffer<f32>,
    pub gamma: &'a DeviceBuffer<f32>,
    pub beta: &'a DeviceBuffer<f32>,
}

/// One packed batch's tokens: vocabulary ids, segments and positions.
pub struct Tokens<'a> {
    pub ids: &'a DeviceBuffer<i32>,
    pub types: &'a DeviceBuffer<i32>,
    pub positions: &'a DeviceBuffer<i32>,
    pub count: usize,
}

/// A cross-encoder's pooler head, all FP32: a biased dense layer over the first row, tanh and
/// the classifier.
pub struct PoolerHead<'a> {
    pub pooler_w: &'a DeviceBuffer<f32>,
    pub pooler_b: &'a DeviceBuffer<f32>,
    pub classifier_w: &'a DeviceBuffer<f32>,
    pub classifier_b: &'a DeviceBuffer<f32>,
}

/// A cross-encoder's prediction head, all FP32: a dense layer without bias over the pooled rows,
/// an activation, a norm without shift and the classifier.
pub struct PredictionHead<'a> {
    pub pooling: Pooling,
    pub activation: Activation,
    pub dense_w: &'a DeviceBuffer<f32>,
    pub norm_g: &'a DeviceBuffer<f32>,
    pub classifier_w: &'a DeviceBuffer<f32>,
    pub classifier_b: &'a DeviceBuffer<f32>,
    pub eps: f32,
}

/// Where a GEMM writes what FP16 rounding lost from its first `columns`
/// columns, scaled by 2^11: `out ~= high + low * 2^-11`. `out` is [m, columns].
pub struct Low<'a> {
    pub out: &'a DeviceBuffer<f16>,
    pub columns: usize,
}

/// Operands of `a * w^T + bias` with `a` [m, k], `w` [n, k] (both K-contiguous)
/// and `bias` (n values).
pub struct Gemm<'a> {
    pub a: &'a DeviceBuffer<f16>,
    pub w: &'a DeviceBuffer<f16>,
    pub bias: &'a DeviceBuffer<f32>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl Gemm<'_> {
    fn args(&self) -> Args {
        assert!(self.m > 0);
        assert!(self.n.is_multiple_of(8) && self.k.is_multiple_of(GEMM_TILE_K));
        assert!(self.a.len() >= self.m * self.k && self.w.len() >= self.n * self.k);
        assert!(self.bias.len() >= self.n);
        Args::new().ptr(self.a).ptr(self.w).ptr(self.bias)
    }

    /// The operands, then `out` and where `low` keeps what FP16 lost.
    fn split(&self, out: &DeviceBuffer<f16>, low: Option<&Low>) -> Args {
        assert!(out.len() >= self.m * self.n);
        let (low_ptr, low_columns) = match low {
            Some(low) => {
                // The epilogue stores the low parts in 16-byte chunks.
                assert!(low.columns <= self.n && low.columns.is_multiple_of(8));
                assert!(low.out.len() >= self.m * low.columns);
                (low.out.ptr(), low.columns)
            }
            None => (0, 0),
        };
        let columns = i32::try_from(low_columns).expect("fits i32");
        self.args().ptr(out).raw(low_ptr).i32(columns)
    }

    fn dims(&self, args: Args) -> Args {
        let dim = |value: usize| i32::try_from(value).expect("fits i32");
        args.i32(dim(self.m)).i32(dim(self.n)).i32(dim(self.k))
    }
}

pub(crate) fn blocks(count: usize, per_block: usize) -> u32 {
    u32::try_from(count.div_ceil(per_block)).expect("grid fits u32")
}

pub(crate) fn attention_shared_bytes(geometry: Geometry) -> u32 {
    let halfs = ATTENTION_TILES * ATTENTION_BLOCK * geometry.head_dim();
    u32::try_from(halfs * size_of::<f16>()).expect("fits u32")
}

/// The GEMMs over pairs of columns, in a module of their own: only an encoder that launches
/// them pays to load it.
pub struct PairedGemms {
    geglu: Tiled,
    turned: Tiled,
}

impl PairedGemms {
    pub fn load(context: &Context) -> Result<Self, CudaError> {
        let module = Module::load(context, GEMM_PAIRED_PTX)?;
        Ok(Self {
            geglu: Tiled::load(context, &module, "gemm_bias_geglu_f16", &[])?,
            turned: Tiled::load(context, &module, "gemm_bias_turned_f16", &[])?,
        })
    }

    /// [`Kernels::gemm_bias`] with each pair among `turned`'s columns turned by its row's
    /// position before FP16 rounds it: `gemm.w`'s rows and `gemm.bias` as
    /// [`kodamaembed_model::pair_turns`] orders them.
    ///
    /// # Safety
    /// Buffers must outlive `stream`'s work, and `turned`'s positions must hold each row's place
    /// in a sequence of at most its longest.
    pub unsafe fn gemm_bias_turned(
        &self,
        stream: &Stream,
        gemm: &Gemm,
        out: &DeviceBuffer<f16>,
        low: Option<&Low>,
        turned: &Turned,
    ) -> Result<(), CudaError> {
        let turns = turned.turns;
        assert!(turned.positions.len() >= gemm.m);
        assert!(turned.columns <= gemm.n && turned.columns.is_multiple_of(turns.head));
        assert!(turns.pairs.len() == turns.positions * turns.head);
        assert!(turned.longest <= turns.positions, "sequences fit table");
        let dim = |value: usize| i32::try_from(value).expect("fits i32");
        let mut args = gemm.dims(
            gemm.split(out, low)
                .ptr(turned.positions)
                .ptr(&turns.pairs)
                .i32(dim(turns.head))
                .i32(dim(turned.columns)),
        );
        // SAFETY: the arguments match `gemm_bias_turned_f16`; the caller vouches for the rest.
        unsafe { self.turned.launch(stream, gemm, &mut args) }
    }

    /// `out = gelu(input) * gate` in FP16 with the exact (erf) GELU, `gemm.n / 2` columns a row:
    /// columns 2j and 2j + 1 of `a * w^T + bias` are product j's input and gate, `gemm.w`'s rows
    /// and `gemm.bias` as [`kodamaembed_model::pair_gates`] orders them.
    ///
    /// # Safety
    /// Every buffer must stay alive until `stream` has finished this work.
    pub unsafe fn gemm_bias_geglu(
        &self,
        stream: &Stream,
        gemm: &Gemm,
        out: &DeviceBuffer<f16>,
    ) -> Result<(), CudaError> {
        assert!(out.len() >= gemm.m * gemm.n / 2);
        let mut args = gemm.dims(gemm.args().ptr(out));
        // SAFETY: the arguments match `gemm_bias_geglu_f16` and the caller keeps the buffers alive.
        unsafe { self.geglu.launch(stream, gemm, &mut args) }
    }
}

/// A prediction head's kernels, in a module of their own: only an encoder that scores through
/// one pays to load it.
pub struct PredictionKernels {
    hidden: usize,
    cls: Function,
    mean: Function,
}

impl PredictionKernels {
    pub fn load(context: &Context, geometry: Geometry) -> Result<Self, CudaError> {
        let module = Module::load(context, PREDICTION_HEAD_PTX)?;
        let hidden = geometry.hidden();
        Ok(Self {
            hidden,
            cls: module.function(&format!("prediction_head_cls_{hidden}"))?,
            mean: module.function(&format!("prediction_head_mean_{hidden}"))?,
        })
    }

    /// One cross-encoder relevance score per packed sequence through a prediction head, from the
    /// sequence's first row or from its rows' mean, which `kernels`, of this head's geometry,
    /// keeps in `scratch` as [`Kernels::pool_normalize`] keeps it.
    ///
    /// # Safety
    /// As for [`Kernels::attention`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn prediction_head(
        &self,
        kernels: &Kernels,
        stream: &Stream,
        hidden: &DeviceBuffer<f32>,
        packed: &Packed,
        head: &PredictionHead,
        scores: &DeviceBuffer<f32>,
        scratch: &DeviceBuffer<f16>,
    ) -> Result<(), CudaError> {
        packed.check();
        let h = self.hidden;
        assert!(
            kernels.geometry.hidden() == h,
            "the means are as wide as the head"
        );
        assert!(hidden.len() >= packed.tokens * h && scores.len() >= packed.sequences);
        assert!(head.dense_w.len() == h * h && head.norm_g.len() == h);
        assert!(head.classifier_w.len() == h && head.classifier_b.len() == 1);
        let (grid, block) = ([blocks(packed.sequences, 1), 1, 1], [h as u32, 1, 1]);
        let head_args = |pooled: Args| {
            pooled
                .ptr(head.dense_w)
                .ptr(head.norm_g)
                .ptr(head.classifier_w)
                .ptr(head.classifier_b)
                .f32(head.eps)
                .i32(i32::from(head.activation == Activation::Silu))
                .ptr(scores)
        };
        // SAFETY: the arguments match the kernels, the stream orders the means' writes before
        // their reads, and the caller vouches for the rest.
        unsafe {
            match head.pooling {
                Pooling::Cls => {
                    let mut args = head_args(Args::new().ptr(hidden).ptr(packed.cu_seqlens));
                    self.cls.launch(stream, grid, block, 0, &mut args)
                }
                Pooling::Mean => {
                    kernels.column_means(stream, hidden, packed, scratch)?;
                    let mut args = head_args(Args::new().raw(scratch.ptr()));
                    self.mean.launch(stream, grid, block, 0, &mut args)
                }
            }
        }
    }
}

impl Kernels {
    pub fn load(context: &Context, geometry: Geometry) -> Result<Self, CudaError> {
        let norm = Module::load(context, NORM_PTX)?;
        let gemm = Module::load(context, GEMM_PTX)?;
        let attention = Module::load(context, ATTENTION_PTX)?;
        let pool = Module::load(context, POOL_PTX)?;
        let width = geometry.hidden();
        let (heads, head_dim) = (geometry.heads(), geometry.head_dim());
        let suffix = format!("_h{heads}_d{head_dim}");
        let relative_attention = Pair::load(&attention, "attention_varlen_bias", &suffix)?;
        let local_attention = attention.function(&format!("attention_varlen_local{suffix}"))?;
        let attention = if (heads, head_dim) == (16, 64) {
            Attention::Pipelined {
                rows32: attention.function("attention_pipelined_h16_d64_rows32")?,
                rows64: Pair::load(&attention, "attention_pipelined", "_h16_d64_rows64")?,
            }
        } else {
            Attention::Blocks(Pair::load(&attention, "attention_varlen", &suffix)?)
        };
        Ok(Self {
            geometry,
            layernorm: norm.function(&format!("layernorm_{width}"))?,
            embed_layernorm: norm.function(&format!("embed_layernorm_{width}"))?,
            gemm_bias_f16: Tiled::load(context, &gemm, "gemm_bias_f16", QKV_GRID)?,
            gemm_bias_gelu_f16: Tiled::load(context, &gemm, "gemm_bias_gelu_f16", UP_GRID)?,
            gemm_bias_residual_f32: Tiled::load(
                context,
                &gemm,
                "gemm_bias_residual_f32",
                RESIDUAL_GRID,
            )?,
            attention,
            relative_attention,
            local_attention,
            column_means: pool.function(&format!("column_means_{width}"))?,
            normalize_means: pool.function(&format!("normalize_means_{width}"))?,
            cls_pool_normalize: pool.function(&format!("cls_pool_normalize_{width}"))?,
            rerank_head: pool.function(&format!("rerank_head_{width}"))?,
        })
    }

    pub fn geometry(&self) -> Geometry {
        self.geometry
    }

    /// `hidden = LayerNorm((word + token_type) + position)` into FP16 and FP32.
    ///
    /// # Safety
    /// Buffers must outlive `stream`'s work, and ids, types and positions must
    /// index their tables.
    pub unsafe fn embed_layernorm(
        &self,
        stream: &Stream,
        embeddings: &Embeddings,
        tokens: &Tokens,
        out16: &DeviceBuffer<f16>,
        out32: &DeviceBuffer<f32>,
        eps: f32,
    ) -> Result<(), CudaError> {
        let (h, count) = (self.geometry.hidden(), tokens.count);
        assert!(tokens.ids.len() >= count && tokens.types.len() >= count);
        assert!(tokens.positions.len() >= count);
        assert!(out16.len() >= count * h && out32.len() >= count * h);
        assert!(embeddings.token_type.len() >= h);
        assert!(embeddings.gamma.len() == h && embeddings.beta.len() == h);
        let mut args = Args::new()
            .ptr(tokens.ids)
            .ptr(tokens.types)
            .ptr(tokens.positions)
            .ptr(embeddings.word)
            .ptr(embeddings.position)
            .ptr(embeddings.token_type)
            .ptr(embeddings.gamma)
            .ptr(embeddings.beta)
            .ptr(out16)
            .ptr(out32)
            .f32(eps);
        // SAFETY: the arguments match the kernel; the caller vouches for the rest.
        unsafe {
            self.embed_layernorm
                .launch(stream, [blocks(count, 1), 1, 1], [32, 1, 1], 0, &mut args)
        }
    }

    /// LayerNorm over `rows` rows into FP16, and into FP32 when `out32` is given.
    ///
    /// # Safety
    /// Every buffer must stay alive until `stream` has finished this work.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn layernorm(
        &self,
        stream: &Stream,
        x: &DeviceBuffer<f32>,
        gamma: &DeviceBuffer<f32>,
        beta: &DeviceBuffer<f32>,
        out16: &DeviceBuffer<f16>,
        out32: Option<&DeviceBuffer<f32>>,
        rows: usize,
        eps: f32,
    ) -> Result<(), CudaError> {
        let h = self.geometry.hidden();
        assert!(x.len() >= rows * h && out16.len() >= rows * h);
        assert!(gamma.len() == h && beta.len() == h);
        assert!(out32.is_none_or(|out| out.len() >= rows * h));
        let mut args = Args::new()
            .ptr(x)
            .ptr(gamma)
            .ptr(beta)
            .ptr(out16)
            .raw(out32.map_or(0, DeviceBuffer::ptr))
            .f32(eps);
        // SAFETY: the arguments match the kernel and the lengths were checked.
        unsafe {
            self.layernorm
                .launch(stream, [blocks(rows, 1), 1, 1], [32, 1, 1], 0, &mut args)
        }
    }

    /// `out = a * w^T + bias` in FP16, and with `low` what that rounding lost.
    ///
    /// # Safety
    /// Every buffer must stay alive until `stream` has finished this work.
    pub unsafe fn gemm_bias(
        &self,
        stream: &Stream,
        gemm: &Gemm,
        out: &DeviceBuffer<f16>,
        low: Option<&Low>,
    ) -> Result<(), CudaError> {
        let mut args = gemm.dims(gemm.split(out, low));
        // SAFETY: the arguments match `gemm_bias_f16` and the caller keeps the buffers alive.
        unsafe { self.gemm_bias_f16.launch(stream, gemm, &mut args) }
    }

    /// `out = gelu(a * w^T + bias)` in FP16, with the exact (erf) GELU.
    ///
    /// # Safety
    /// Every buffer must stay alive until `stream` has finished this work.
    pub unsafe fn gemm_bias_gelu(
        &self,
        stream: &Stream,
        gemm: &Gemm,
        out: &DeviceBuffer<f16>,
    ) -> Result<(), CudaError> {
        assert!(out.len() >= gemm.m * gemm.n);
        let mut args = gemm.dims(gemm.args().ptr(out));
        // SAFETY: the arguments match `gemm_bias_gelu_f16` and the caller keeps the buffers alive.
        unsafe { self.gemm_bias_gelu_f16.launch(stream, gemm, &mut args) }
    }

    /// `out = a * w^T + bias + residual` in FP32.
    ///
    /// # Safety
    /// Every buffer must stay alive until `stream` has finished this work.
    pub unsafe fn gemm_bias_residual(
        &self,
        stream: &Stream,
        gemm: &Gemm,
        residual: &DeviceBuffer<f32>,
        out: &DeviceBuffer<f32>,
    ) -> Result<(), CudaError> {
        assert!(residual.len() >= gemm.m * gemm.n && out.len() >= gemm.m * gemm.n);
        let mut args = gemm.dims(gemm.args().ptr(residual).ptr(out));
        // SAFETY: the arguments match `gemm_bias_residual_f32` and the caller keeps the buffers alive.
        unsafe { self.gemm_bias_residual_f32.launch(stream, gemm, &mut args) }
    }

    /// Scores leave the kernels in the log2 domain, scaled by the root of the head's width.
    fn score_scale(&self) -> f32 {
        std::f32::consts::LOG2_E / (self.geometry.head_dim() as f32).sqrt()
    }

    /// A block per `rows` query rows of the longest sequence, per head and per sequence.
    fn attention_grid(&self, packed: &Packed, rows: usize) -> [u32; 3] {
        [
            blocks(packed.max_len, rows),
            self.geometry.heads() as u32,
            u32::try_from(packed.sequences).expect("grid fits u32"),
        ]
    }

    /// [`Kernels::attention`] over the keys within `reach` rows of each query, with the
    /// correction at every length: a window's work does not grow with the sequence.
    ///
    /// # Safety
    /// As for [`Kernels::attention`].
    pub unsafe fn local_attention(
        &self,
        stream: &Stream,
        qkv: &DeviceBuffer<f16>,
        low: &DeviceBuffer<f16>,
        packed: &Packed,
        out: &DeviceBuffer<f16>,
        reach: usize,
    ) -> Result<(), CudaError> {
        packed.check();
        let h = self.geometry.hidden();
        assert!(qkv.len() >= packed.tokens * 3 * h && out.len() >= packed.tokens * h);
        assert!(low.len() >= packed.tokens * 2 * h);
        // No key lies farther from a query than the longest sequence.
        let reach = i32::try_from(reach.min(packed.max_len)).expect("fits i32");
        let mut args = Args::new()
            .ptr(qkv)
            .ptr(low)
            .ptr(packed.cu_seqlens)
            .ptr(out)
            .f32(self.score_scale())
            .i32(reach);
        // SAFETY: the arguments match the kernel, whose `shared` bytes stay within the 48 KiB
        // any launch may use; the caller vouches for the rest.
        unsafe {
            self.local_attention.launch(
                stream,
                self.attention_grid(packed, ATTENTION_BLOCK),
                [(ATTENTION_BLOCK / 16 * 32) as u32, 1, 1],
                attention_shared_bytes(self.geometry),
                &mut args,
            )
        }
    }

    /// `low` holds what FP16 lost from Q and K, `2 * hidden` columns of [`Kernels::gemm_bias`], read
    /// only for sequences below [`CORRECTED_BELOW`] tokens; `bias`, when given, joins every score.
    ///
    /// # Safety
    /// Buffers must outlive `stream`'s work, and `packed.cu_seqlens` must hold
    /// the lengths the other fields describe.
    pub unsafe fn attention(
        &self,
        stream: &Stream,
        qkv: &DeviceBuffer<f16>,
        low: &DeviceBuffer<f16>,
        packed: &Packed,
        out: &DeviceBuffer<f16>,
        bias: Option<&RelativeTable>,
    ) -> Result<(), CudaError> {
        packed.check();
        let h = self.geometry.hidden();
        assert!(qkv.len() >= packed.tokens * 3 * h && out.len() >= packed.tokens * h);
        assert!(low.len() >= packed.tokens * 2 * h);
        let scale = self.score_scale();
        let ((corrected, plain), rows) = match (bias, &self.attention) {
            (Some(table), _) => {
                assert!(
                    packed.max_len <= table.longest as usize,
                    "sequences fit table"
                );
                (self.relative_attention.kinds(), ATTENTION_BLOCK)
            }
            (None, Attention::Pipelined { rows32, .. }) if ROWS32.contains(&packed.max_len) => {
                ((rows32, None), 32)
            }
            (None, Attention::Pipelined { rows64, .. }) => (rows64.kinds(), 64),
            (None, Attention::Blocks(pair)) => (pair.kinds(), ATTENTION_BLOCK),
        };
        let grid = self.attention_grid(packed, rows);
        // Each warp holds 16 query rows.
        let threads = (rows / 16 * 32) as u32;
        let shared = attention_shared_bytes(self.geometry);
        let launch = |function: &Function| {
            let args = Args::new()
                .ptr(qkv)
                .ptr(low)
                .ptr(packed.cu_seqlens)
                .ptr(out)
                .f32(scale);
            let mut args = match bias {
                Some(table) => args.ptr(&table.values).i32(table.longest),
                None => args,
            }
            .i32(i32::try_from(CORRECTED_BELOW).expect("fits i32"));
            // SAFETY: the arguments match the kernel, whose `shared` bytes stay within
            // the 48 KiB any launch may use; the caller vouches for the rest.
            unsafe { function.launch(stream, grid, [threads, 1, 1], shared, &mut args) }
        };
        // Each kernel skips the sequences of the other kind.
        if packed.min_len < CORRECTED_BELOW {
            launch(corrected)?;
        }
        if packed.max_len >= CORRECTED_BELOW {
            launch(plain.expect("32-row blocks serve corrected sequences only"))?;
        }
        Ok(())
    }

    /// One L2-normalized mean of the rows `pooling` names per packed sequence. Mean pooling keeps
    /// its FP64 column means in `scratch`, which holds at least `4 * hidden` halfs a sequence.
    ///
    /// # Safety
    /// As for [`Kernels::attention`].
    pub unsafe fn pool_normalize(
        &self,
        stream: &Stream,
        pooling: Pooling,
        hidden: &DeviceBuffer<f32>,
        packed: &Packed,
        out: &DeviceBuffer<f32>,
        scratch: &DeviceBuffer<f16>,
    ) -> Result<(), CudaError> {
        packed.check();
        let h = self.geometry.hidden();
        assert!(hidden.len() >= packed.tokens * h && out.len() >= packed.sequences * h);
        let (sequences, block) = (blocks(packed.sequences, 1), (h / 4) as u32);
        match pooling {
            Pooling::Mean => {
                let mut finish = Args::new().raw(scratch.ptr()).ptr(out);
                // SAFETY: the arguments match the kernel, the stream orders the means' writes
                // before their reads, and the caller vouches for the rest.
                unsafe {
                    self.column_means(stream, hidden, packed, scratch)?;
                    self.normalize_means.launch(
                        stream,
                        [sequences, 1, 1],
                        [block, 1, 1],
                        0,
                        &mut finish,
                    )
                }
            }
            Pooling::Cls => {
                let mut args = Args::new().ptr(hidden).ptr(packed.cu_seqlens).ptr(out);
                // SAFETY: the arguments match the kernel; the caller vouches for the rest.
                unsafe {
                    self.cls_pool_normalize.launch(
                        stream,
                        [sequences, 1, 1],
                        [block, 1, 1],
                        0,
                        &mut args,
                    )
                }
            }
        }
    }

    /// Each packed sequence's FP64 column means into `scratch`, at least `4 * hidden` halfs a
    /// sequence.
    ///
    /// # Safety
    /// As for [`Kernels::attention`].
    unsafe fn column_means(
        &self,
        stream: &Stream,
        hidden: &DeviceBuffer<f32>,
        packed: &Packed,
        scratch: &DeviceBuffer<f16>,
    ) -> Result<(), CudaError> {
        let h = self.geometry.hidden();
        assert!(scratch.len() * size_of::<f16>() >= packed.sequences * h * size_of::<f64>());
        let mut args = Args::new()
            .ptr(hidden)
            .ptr(packed.cu_seqlens)
            .raw(scratch.ptr());
        // SAFETY: the arguments match the kernel; the caller vouches for the rest.
        unsafe {
            self.column_means.launch(
                stream,
                [(h / 32) as u32, blocks(packed.sequences, 1), 1],
                [32, 1, 1],
                0,
                &mut args,
            )
        }
    }

    /// One cross-encoder relevance score per packed sequence, from its first row.
    ///
    /// # Safety
    /// As for [`Kernels::attention`].
    pub unsafe fn rerank_head(
        &self,
        stream: &Stream,
        hidden: &DeviceBuffer<f32>,
        packed: &Packed,
        head: &PoolerHead,
        scores: &DeviceBuffer<f32>,
    ) -> Result<(), CudaError> {
        packed.check();
        let h = self.geometry.hidden();
        assert!(hidden.len() >= packed.tokens * h && scores.len() >= packed.sequences);
        assert!(head.pooler_w.len() == h * h && head.pooler_b.len() == h);
        assert!(head.classifier_w.len() == h && head.classifier_b.len() == 1);
        let mut args = Args::new()
            .ptr(hidden)
            .ptr(packed.cu_seqlens)
            .ptr(head.pooler_w)
            .ptr(head.pooler_b)
            .ptr(head.classifier_w)
            .ptr(head.classifier_b)
            .ptr(scores);
        // SAFETY: the arguments match the kernel; the caller vouches for the rest.
        unsafe {
            self.rerank_head.launch(
                stream,
                [blocks(packed.sequences, 1), 1, 1],
                [h as u32, 1, 1],
                0,
                &mut args,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DOWN, GridTile, OUT, QKV, QKV_GRID, RESIDUAL_GRID, Tile, UP, UP_GRID, pick};

    /// The suffix of the grid tiling that takes `m` rows of a projection.
    fn grid(tiles: &[GridTile], m: usize, (n, k): (usize, usize)) -> Option<&'static str> {
        let mut fitting = tiles.iter().filter(|tile| tile.fits(m, n, k));
        let suffix = fitting.next().map(|tile| tile.suffix);
        assert!(fitting.next().is_none(), "two tilings for {m} rows");
        suffix
    }

    /// Both edges of each band take its tiling, and the rows around the bands no grid tiling.
    fn assert_bands(tiles: &[GridTile], shape: (usize, usize), bands: &[(usize, usize, &str)]) {
        for &(first, last, suffix) in bands {
            assert_eq!(grid(tiles, first, shape), Some(suffix));
            assert_eq!(grid(tiles, last, shape), Some(suffix));
        }
        let (first, last) = (bands[0].0, bands[bands.len() - 1].1);
        assert_eq!(grid(tiles, first - 1, shape), None);
        assert_eq!(grid(tiles, last + 1, shape), None);
    }

    #[test]
    fn tiles_index_their_kernels_in_declaration_order() {
        for (index, tile) in Tile::ALL.into_iter().enumerate() {
            assert_eq!(tile as usize, index, "{tile:?}");
        }
    }

    #[test]
    fn a_grid_tiling_takes_only_its_projection_and_rows() {
        for tile in QKV_GRID.iter().chain(UP_GRID).chain(RESIDUAL_GRID) {
            let (start, end) = (*tile.rows.start(), *tile.rows.end());
            for m in [start, end] {
                assert!(tile.fits(m, tile.columns, tile.depth));
                assert!(!tile.fits(m, tile.columns, 768));
                assert!(!tile.fits(m, 768, tile.depth));
            }
            assert!(!tile.fits(start - 1, tile.columns, tile.depth));
            assert!(!tile.fits(end + 1, tile.columns, tile.depth));
        }
    }

    /// The GPU suite compares every tiling's bits; these pin the row bands' edges.
    #[test]
    fn projections_take_the_tiling_of_their_row_band() {
        let tile = |m, (n, k)| pick(m, n, k);
        assert_eq!(tile(16, QKV), Tile::R16C32);
        assert_eq!(tile(17, QKV), Tile::R48C64);
        assert_eq!(tile(48, UP), Tile::R48C128);
        assert_eq!(tile(33, OUT), Tile::R48C32);
        assert_eq!(tile(49, DOWN), Tile::R48C64);
        assert_eq!(tile(64, OUT), Tile::R32C32);
        assert_eq!(tile(65, UP), Tile::R80C128);
        assert_eq!(tile(80, OUT), Tile::R80C64);
        assert_eq!(tile(96, DOWN), Tile::R48C64);
        assert_eq!(tile(96, QKV), Tile::R96C128);
        assert_eq!(tile(97, DOWN), Tile::R64C64S4);
        assert_eq!(tile(97, QKV), Tile::R128C64);
        assert_eq!(tile(128, QKV), Tile::R128C64);
        assert_eq!(tile(128, UP), Tile::R128C128W8K64);
        assert_eq!(tile(129, UP), Tile::R64C128K64);
        assert_eq!(tile(192, UP), Tile::R64C128K64);
        assert_eq!(tile(192, OUT), Tile::R64C64S4);
        assert_eq!(tile(192, DOWN), Tile::R64C64S4);
        assert_eq!(tile(321, OUT), Tile::R64C128K64);
        assert_eq!(tile(289, DOWN), Tile::R64C128K64);
        assert_eq!(tile(384, DOWN), Tile::R64C128K64);
        assert_eq!(tile(385, OUT), Tile::R128C88);
        assert_eq!(tile(512, DOWN), Tile::R128C88);
        assert_eq!(tile(513, QKV), Tile::R256C128);
        assert_eq!(tile(513, UP), Tile::R128C176);
        assert_bands(
            QKV_GRID,
            QKV,
            &[
                (129, 160, "g80c128"),
                (161, 192, "g96c128"),
                (193, 224, "g112c128"),
                (225, 256, "g128c128"),
                (257, 288, "g144c128"),
                (289, 320, "g160c128"),
                (321, 384, "g192c128"),
                (385, 512, "g256c128"),
            ],
        );
        assert_bands(
            UP_GRID,
            UP,
            &[
                (193, 240, "g80c256"),
                (241, 256, "g128c192"),
                (257, 288, "g96c256"),
                (289, 320, "g112c256"),
                (321, 384, "g192c176"),
                (385, 512, "g128c176"),
            ],
        );
        assert_bands(
            RESIDUAL_GRID,
            OUT,
            &[
                (193, 224, "g112c64"),
                (225, 288, "g96c64"),
                (289, 320, "g64c128"),
            ],
        );
        assert_bands(
            RESIDUAL_GRID,
            DOWN,
            &[
                (193, 224, "g112c64"),
                (225, 240, "g80c96"),
                (241, 256, "g64c128"),
                (257, 288, "g96c80"),
            ],
        );
        // all-MiniLM-L6-v2's projections: 384 wide, 1536 in the FFN.
        assert_eq!(pick(33, 1152, 384), Tile::R16C64);
        assert_eq!(pick(65, 1536, 384), Tile::R64C128);
        assert_eq!(pick(256, 1152, 384), Tile::R128C128W8);
        assert_eq!(pick(256, 1536, 384), Tile::R128C64);
        assert_eq!(pick(512, 1152, 384), Tile::R256C128);
        assert_eq!(pick(512, 384, 1536), Tile::R64C128);
    }
}
