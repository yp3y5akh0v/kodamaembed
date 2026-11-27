//! The host side of `kernels/repro.cu`, built from the CPU encoder's export.

use crate::driver::{
    Args, Context, CudaError, DeviceBuffer, DevicePtr, DeviceRepr, Event, Function, Graph,
    HostBuffer, HostMarked, Module, Stream,
};
use half::f16;
use kodamaembed_model::{
    Geometry, Positions, Shape,
    repro::{DeferredExport, EncoderExport, NormExport, ProjectionExport, STATIC_CLASSES},
};
use std::ops::Range;

mod layout;
use layout::{ArenaLayout, BatchLengths, InputLayout, InputSection, InputSections, Region};

const REPRO_PTX: &str = include_str!("../ptx/repro.ptx");
const THREADS: u32 = 128;
/// The CPU's integer attention bound (its weight-value sums fit INT32).
const MAX_KEYS: usize = 512;
/// repro.cu ATT_THREADS, ATT_ROWS, ATT_ROW_THREADS, ATT_CHUNK and PREP_ROWS.
const ATT_THREADS: u32 = 256;
const ATT_ROWS: usize = 64;
const ATT_ROW_THREADS: u32 = 128;
const ATT_CHUNK: usize = 64;
/// A key split takes at least one chunk.
const MAX_KEY_SPLITS: usize = MAX_KEYS / ATT_CHUNK;
const PREP_ROWS: usize = 16;
/// repro.cu POOL_ROWS, POOL_JOINED_ROWS and POOL_THREADS.
const POOL_ROWS: usize = 4;
const POOL_JOINED_ROWS: usize = 2;
const POOL_THREADS: u32 = 256;
/// repro.cu ATT_WEIGHT_SPAN and ATT_WEIGHT_ENTRIES.
const ATTENTION_SPAN: usize = 3840;
const ATTENTION_WEIGHTS: usize = 3848;
/// repro.cu ATT_PAIR_THREADS, BLOCK and PAIR_CENTER_ROWS.
const ATT_PAIR_THREADS: u32 = 384;
const ROTATION_BLOCK: usize = 128;
const PAIR_CENTER_ROWS: usize = 4;
const _: () = assert!(
    STATIC_CLASSES <= PAIR_CENTER_ROWS,
    "repro_attention_pairs stages every static class's center"
);
/// The first bucket holding the longest narrow sequence; past 32 keys, operands pack once a layer.
const PAIR_LENGTHS: [usize; 5] = [16, 32, ATT_CHUNK, ROW_PACK_KEYS, PACKED_QKV_ROWS];
const PLAIN_PAIR_BUCKETS: usize = 2;

fn packed_bucket(lengths: &[usize]) -> bool {
    lengths
        .iter()
        .any(|&len| len > PAIR_LENGTHS[PLAIN_PAIR_BUCKETS - 1])
}
const ROW_PACK_KEYS: usize = 2 * ATT_CHUNK;
const PACKED_QKV_ROWS: usize = 160;
/// repro.cu REPRO_ATTENTION_PAIRS' rows and PACK_PART_ROWS.
const PAIR_TILE_ROWS: usize = 16;

/// repro.cu REPRO_ATTENTION_PAIRS_PACKED. The 128 keys also get 32-row tiles
/// for grids whose 16-row tiles would need two waves.
struct PackedBucket {
    d: usize,
    keys: usize,
    tile_rows: usize,
    threads: u32,
}

/// Whether the grid fits the multiprocessors at once, a block each.
fn packed_fits(
    bucket: &PackedBucket,
    sequences: usize,
    columns: usize,
    longest: usize,
    sms: usize,
) -> bool {
    sequences * columns * longest.div_ceil(bucket.tile_rows) <= sms
}

/// Smallest tiles first: attend_pairs takes the first whose grid fits.
static PACKED_BUCKETS: [PackedBucket; 4] = [
    PackedBucket {
        d: 64,
        keys: ATT_CHUNK,
        tile_rows: PAIR_TILE_ROWS,
        threads: ATT_PAIR_THREADS,
    },
    PackedBucket {
        d: 64,
        keys: ROW_PACK_KEYS,
        tile_rows: PAIR_TILE_ROWS,
        threads: 512,
    },
    PackedBucket {
        d: 64,
        keys: ROW_PACK_KEYS,
        tile_rows: 2 * PAIR_TILE_ROWS,
        threads: 512,
    },
    PackedBucket {
        d: 64,
        keys: PACKED_QKV_ROWS,
        tile_rows: 2 * PAIR_TILE_ROWS,
        threads: 512,
    },
];

#[cfg(test)]
mod packed_bucket_tests {
    use super::{PACKED_BUCKETS, PackedBucket, ROW_PACK_KEYS, packed_fits};

    fn e5_tile(rows: usize) -> &'static PackedBucket {
        PACKED_BUCKETS
            .iter()
            .find(|b| b.d == 64 && b.keys == ROW_PACK_KEYS && b.tile_rows == rows)
            .expect("E5's 128-key tile")
    }

    #[test]
    fn e5_takes_16_row_query_tiles_to_112_keys_on_58_sms() {
        for longest in 65..=ROW_PACK_KEYS {
            let fits = packed_fits(e5_tile(16), 1, 8, longest, 58);
            assert_eq!(fits, longest <= 112, "{longest} keys");
            assert!(packed_fits(e5_tile(32), 1, 8, longest, 58));
        }
        assert!(!packed_fits(e5_tile(16), 2, 8, 65, 58));
        assert!(packed_fits(e5_tile(32), 2, 8, 65, 58));
    }

    #[test]
    fn each_buckets_tiles_run_from_the_smallest() {
        for (i, a) in PACKED_BUCKETS.iter().enumerate() {
            for b in &PACKED_BUCKETS[i + 1..] {
                if (a.d, a.keys) == (b.d, b.keys) {
                    assert!(a.tile_rows < b.tile_rows, "d {} keys {}", a.d, a.keys);
                }
            }
        }
    }
}

/// repro.cu PairLayout<d, keys, query_rows, threads>::BYTES.
const fn pairs_bytes(d: usize, keys: usize, query_rows: usize, threads: u32) -> u32 {
    let heads = ROTATION_BLOCK / d;
    let (kp, cp) = (d / 4 + 4, keys / 4 + 4);
    let warps = threads as usize / 32;
    let bytes = heads * (query_rows + keys) * kp * 4
        + heads * d * cp * 4
        + heads * (query_rows + keys) * 4
        + 3 * heads * d * 4
        + ATTENTION_WEIGHTS * 2
        + 16 * (warps * 4).div_ceil(16)
        + (PAIR_CENTER_ROWS + 1) * ROTATION_BLOCK * 4
        + query_rows * (ROTATION_BLOCK + 8) * 2;
    bytes as u32
}

/// Up to ATT_CHUNK keys a block holds every row, past it its tile.
const fn packed_query_rows(bucket: &PackedBucket) -> usize {
    if bucket.keys > ATT_CHUNK {
        bucket.tile_rows
    } else {
        ATT_CHUNK
    }
}

/// repro.cu PhaseLayout<d, keys, tile_rows>::BYTES, past pairs_bytes.
const fn phase_bytes(bucket: &PackedBucket) -> u32 {
    let units = ROTATION_BLOCK / bucket.d * (bucket.tile_rows / 16);
    (4 * (4 * units * 16 + units * bucket.keys.div_ceil(32) * 2 * 4 * 32)) as u32
}

const fn packed_pairs_bytes(bucket: &PackedBucket) -> u32 {
    pairs_bytes(
        bucket.d,
        bucket.keys,
        packed_query_rows(bucket),
        bucket.threads,
    ) + phase_bytes(bucket)
}
/// repro.cu NORM_ROWS and NORM_BATCH_ROWS: multiples of four keep gamma and beta 16-byte aligned.
const NORM_ROWS: usize = 4;
const NORM_BATCH_ROWS: usize = 8;
/// Up to one longest sequence, few rows would leave the GPU idle, so each row
/// takes blocks of its own (repro_layernorm_quantize).
const SPLIT_NORM_ROWS: usize = MAX_KEYS;
/// repro.cu NORM_SPLIT_WARPS: a block's 128-blocks of a row.
const NORM_SPLIT_WARPS: usize = 8;
/// repro_layernorm_quantize_joined_{warps}'s width, in 128-blocks.
const JOINED_NORM_WARPS: usize = 8;
/// Batches of at most this many tokens use graphs to reduce launch overhead.
const GRAPH_TOKENS: usize = 1024;
const GRAPH_SHAPES: usize = 64;
/// Batches of at most this many rows read each layer's digits a layer ahead on a second
/// stream where the L2 holds them; repro.cu READ_AHEAD_WARPS warps a block.
const READ_AHEAD_ROWS: usize = 128;
const READ_AHEAD_THREADS: u32 = 64;
const _: () = assert!(
    READ_AHEAD_ROWS <= GRAPH_TOKENS,
    "the reads ahead are recorded into the layers' graphs"
);
const _: () = assert!(
    READ_AHEAD_ROWS <= UP_BLOCKS_ROWS,
    "the half-blocks quantizer keeps the driver's shared memory: it never runs beside a read ahead"
);

/// An L2 holds a layer read ahead beside the layer in use; a smaller one drops it before use.
fn l2_holds_a_layer_ahead(layer_bytes: usize, l2_bytes: u32) -> bool {
    2 * layer_bytes <= l2_bytes as usize
}

#[cfg(test)]
mod read_ahead_tests {
    use super::l2_holds_a_layer_ahead;

    /// E5-large's twelve 1024 by 1024 blocks of digits a layer.
    const E5_LAYER: usize = 12 << 20;

    #[test]
    fn e5_layers_are_read_ahead_from_24_mib_of_l2() {
        assert!(!l2_holds_a_layer_ahead(E5_LAYER, 4 << 20));
        assert!(!l2_holds_a_layer_ahead(E5_LAYER, (24 << 20) - 1));
        assert!(l2_holds_a_layer_ahead(E5_LAYER, 24 << 20));
        assert!(l2_holds_a_layer_ahead(E5_LAYER, 40 << 20));
    }
}

/// Past this many rows, tall tiles let a sequence's rows share each tile's weights.
const FUSED_NARROW_ROWS: usize = 16;
const TALL_ROWS: usize = 64;
/// CUDA's limit on a launch grid's y and z.
const GRID_YZ: usize = 65_535;
/// Tall projections put TALL_ROWS-row tiles on a grid's y, wide attention its
/// sequences on z.
pub const MAX_BATCH_TOKENS: usize = GRID_YZ * TALL_ROWS;
pub const MAX_BATCH_SEQUENCES: usize = GRID_YZ;
/// The CPU arithmetic's f32 constants (EncoderExport): repro.cu FC_* values.
const CONSTANTS_F32: usize = 9;
/// The warp groups of repro.cu repro_restored_*_g*'s kernels.
const GROUPINGS: [usize; 3] = [1, 2, 4];

/// The most tiles in flight on the busiest multiprocessor with every tile
/// resident at once, since a second wave adds a latency-bound pass.
fn fused_groups(n: usize, k: usize, sms: usize) -> usize {
    let tiles = n / 32;
    if tiles <= sms && (k / 64).is_multiple_of(4) {
        4
    } else if tiles <= 2 * sms {
        2
    } else {
        1
    }
}

/// repro.cu HALF: the inputs of one promoted dot, which share a step.
const STEP_INPUTS: usize = 64;
/// The deepest projection the narrow and tall kernels stage steps for.
const STAGED_STEPS_DEPTH: usize = 4_096;

/// repro.cu repro_restored_bytes with the steps of a `k`-deep projection, which lie last.
fn restored_bytes(groups: usize, k: usize) -> u32 {
    assert!(
        k <= STAGED_STEPS_DEPTH && k.is_multiple_of(STEP_INPUTS),
        "the narrow tile's steps fit the loaded kernels' shared bytes"
    );
    let ring = if groups == 1 { 20_480 } else { 15_360 };
    let steps = 16 * (k / STEP_INPUTS) * 4;
    let bytes = groups * ring + 960 + 1_024 + (groups - 1) * 2_048 + steps;
    u32::try_from(bytes).expect("fits u32")
}

/// repro_restored_gelu_next_g*: the next input's centers and signs behind the deepest steps.
fn restored_next_bytes(groups: usize) -> u32 {
    restored_bytes(groups, STAGED_STEPS_DEPTH) + 2_048
}
const _: () = assert!(
    STATIC_CLASSES == 3,
    "NextStaged holds three classes' centers"
);

/// (groups, stages): one, two and three blocks fit a multiprocessor.
const TALL_CONFIGS: [(usize, usize); 3] = [(2, 4), (1, 4), (1, 3)];

/// The most tall tiles resident a multiprocessor (TALL_CONFIGS' last).
const TALL_MOST_RESIDENT: usize = 3;

/// Prefer more warp groups and pipeline stages while the tiles fit in one wave.
fn tall_config(tiles: usize, sms: usize) -> usize {
    if tiles <= sms {
        0
    } else if tiles <= 2 * sms {
        1
    } else {
        2
    }
}

/// repro.cu's stream engine: STREAM_THREADS and the plan bounds it traps past.
const STREAM_THREADS: u32 = 384;
const STREAM_WORKERS: usize = 8;
const STREAM_STASH_UNITS: usize = 4;
const STREAM_MAX_LAYERS: usize = 32;
/// repro_stream.cuh's chunks: 512 inputs of a projection.
const STREAM_CHUNK_INPUTS: usize = 512;
/// repro.cu StreamHandoff; even for repro_clear.
const STREAM_HANDOFFS: usize = 10;
/// repro_stream.cuh STREAM_TILE_ROWS: the rows the persistent plan takes.
const STREAM_TILE_ROWS: usize = 16;
/// The stream's workers take a sequence's 128-blocks of attention one each; past this
/// many rounds of them the per-layer kernels are faster.
const STREAM_ATTENTION_ROUNDS: usize = 4;

/// repro_stream.cuh StreamPlan's BYTES: as many 4 KiB ring slots as sm_86's 99 KiB of
/// opt-in shared memory holds beside the rest.
const STREAM_BYTES: u32 = {
    let rows = STREAM_TILE_ROWS;
    let stash = rows * STREAM_STASH_UNITS * 8 * 2;
    let fixed = rows * 1024
        + rows * 64 * 4
        + stash
        + rows * 4
        + 3 * STREAM_MAX_LAYERS * 4 * 8
        + 2 * STREAM_MAX_LAYERS * 2 * 8
        + 16;
    let slot = 4096 + 2 * 8;
    ((99 * 1024 - fixed) / slot * slot + fixed) as u32
};
const _: () = assert!(STREAM_BYTES == 99_696, "repro_stream.cuh's StreamPlan");

/// Every layer in one cooperative launch, when the widths take whole chunks and
/// the GPU holds the plan; `slots` are zeroed at each encode's start.
struct StreamEngine {
    function: Function,
    layers: DeviceBuffer<u64>,
    slots: DeviceBuffer<i64>,
    moments: DeviceBuffer<u64>,
}

fn load_optional_stream<T>(
    required_shared_bytes: u32,
    max_shared_per_block: u32,
    cooperative: impl FnOnce() -> Result<bool, CudaError>,
    load: impl FnOnce() -> Result<Option<T>, CudaError>,
) -> Result<Option<T>, CudaError> {
    if required_shared_bytes > max_shared_per_block || !cooperative()? {
        return Ok(None);
    }
    load()
}

#[cfg(test)]
mod stream_plan_tests;
#[cfg(all(test, target_arch = "x86_64"))]
mod stream_tests;

impl StreamEngine {
    fn load(
        module: &Module,
        context: &Context,
        shape: Shape,
        layers: &[Layer],
    ) -> Result<Option<Self>, CudaError> {
        let (h, inner, sms) = (shape.hidden, shape.intermediate, context.sm_count as usize);
        let d = h / shape.heads;
        let streamers = sms.saturating_sub(STREAM_WORKERS);
        let most = |n: usize, among: usize| (n / 8).div_ceil(among.max(1));
        // The down projection's four quarters need four streamers. A streamer's
        // other units take as many rounds as its warps need.
        if streamers < 4
            || h > 1024
            || !h.is_multiple_of(STREAM_CHUNK_INPUTS)
            || inner != 4 * h
            || layers.len() > STREAM_MAX_LAYERS
            || most(h, streamers) > STREAM_STASH_UNITS
            || !deferred_norms(layers)
        {
            return Ok(None);
        }
        load_optional_stream(
            STREAM_BYTES,
            context.max_shared_per_block,
            || context.cooperative_launch_supported(),
            || {
                let name = format!("repro_stream_layers_{d}_{}", STREAM_CHUNK_INPUTS / 64);
                let function = module.function(&name)?;
                function.set_max_dynamic_shared(STREAM_BYTES)?;
                if function.resident_blocks(STREAM_THREADS, STREAM_BYTES)? < 1 {
                    return Ok(None);
                }
                // The statistics from the 256-byte boundary past the counters.
                let counters = (layers.len() * STREAM_HANDOFFS / 2).next_multiple_of(32);
                let moments = layers.len() * 2 * STREAM_TILE_ROWS;
                let slots = DeviceBuffer::zeroed(context, counters + moments)?;
                // SAFETY: the engine owns `slots` with the view, past the counters.
                let moments = unsafe { DeviceBuffer::alias(&slots, counters * 8, moments) };
                Ok(Some(Self {
                    function,
                    layers: DeviceBuffer::from_slice(context, &stream_layer_words(layers))?,
                    slots,
                    moments,
                }))
            },
        )
    }
}

/// repro.cu StreamLayer's 34 words.
fn stream_layer_words(layers: &[Layer]) -> Vec<u64> {
    layers
        .iter()
        .flat_map(|l| {
            let projections = [&l.qkv, &l.out, &l.up, &l.down];
            let mut words = Vec::with_capacity(34);
            words.extend(projections.map(|p| p.q.ptr()));
            words.extend(projections.map(|p| p.restored_scales.ptr()));
            words.extend(projections.map(|p| p.static_shared.ptr()));
            words.extend(projections.map(|p| p.centers.ptr()));
            words.extend(projections.map(|p| p.signs.ptr()));
            words.extend(projections.map(|p| p.deferred.as_ref().map_or(0, |d| d.sums.ptr())));
            words.extend(projections.map(|p| p.deferred.as_ref().map_or(0, |d| d.constants.ptr())));
            words.extend([
                l.norm1.gamma.ptr(),
                l.norm1.beta.ptr(),
                l.norm2.gamma.ptr(),
                l.norm2.beta.ptr(),
                l.key_centers.ptr(),
                u64::from(l.norm1.eps.to_bits()) | u64::from(l.norm2.eps.to_bits()) << 32,
            ]);
            words
        })
        .collect()
}

/// repro_stream.cuh's two_qkv and two_up: bit l is set where layer l's operand has two digits.
fn two_digit_layers(layers: &[Layer], projection: impl Fn(&Layer) -> &Projection) -> i32 {
    let mask = layers
        .iter()
        .enumerate()
        .filter(|(_, layer)| projection(layer).digits == 2)
        .fold(0_u32, |mask, (index, _)| mask | 1 << index);
    i32::from_ne_bytes(mask.to_ne_bytes())
}

/// The stream engine needs every LayerNorm deferred to its next projection.
fn deferred_norms(layers: &[Layer]) -> bool {
    layers
        .iter()
        .enumerate()
        .all(|(i, l)| l.up.deferred.is_some() && l.qkv.deferred.is_some() == (i > 0))
}

/// repro.cu repro_tall_bytes.
fn tall_bytes((groups, stages): (usize, usize)) -> u32 {
    let bytes = groups * stages * 6_144 + 1_536 + (groups - 1) * 8_192;
    u32::try_from(bytes).expect("fits u32")
}

/// The down projection runs a group per restarted segment (repro_restored_residual_tall_seg4s3).
const TALL_SEGMENTS: usize = 4;
const TALL_SEGMENT_STAGES: usize = 3;
fn tall_segment_bytes() -> u32 {
    u32::try_from(TALL_SEGMENTS * TALL_SEGMENT_STAGES * 6_144 + 1_536).expect("fits u32")
}

/// Tiles twice as tall (repro_tall_mainloop) run in one wave where tall tiles would need two.
const TALLER_ROWS: usize = 2 * TALL_ROWS;
const TALL_W8_CONFIGS: [usize; 2] = [4, 2];
const TALL_M2_SEGMENT_STAGES: usize = 2;
fn tall_w8_bytes(stages: usize) -> u32 {
    u32::try_from(stages * 10_240 + 2_304).expect("fits u32")
}
fn tall_m2_segment_bytes() -> u32 {
    u32::try_from(TALL_SEGMENTS * TALL_M2_SEGMENT_STAGES * 10_240 + 2_304).expect("fits u32")
}

/// Every tile resident at once, with the most stages.
fn tall_w8_config(tiles: usize, sms: usize) -> usize {
    if tiles <= 2 * sms { 0 } else { 1 }
}

/// repro.cu RestoredWide: 128 x 64 tiles, compute-bound where tall tiles are
/// intake-bound, for one-segment projections past TALLER_ROWS rows.
const RESTORED_WIDE_ROWS: usize = 128;
const RESTORED_WIDE_COLUMNS: usize = 64;
const RESTORED_WIDE_BYTES: u32 = 49_152;
/// repro.cu RestoredWideDown: 64 x 64 tiles for the down projection, and for
/// wide projections whose 128 x 64 grid would leave lone tiles.
const RESTORED_WIDE_DOWN_ROWS: usize = 64;
const RESTORED_WIDE_DOWN_COLUMNS: usize = 64;
const RESTORED_WIDE_DOWN_BYTES: u32 = 32_768;
/// repro.cu RestoredWide64x128: 64 x 128 tiles, which draw the least power
/// for their products, for a layer's four projections together.
const RESTORED_WIDE_64X128_ROWS: usize = 64;
const RESTORED_WIDE_64X128_COLUMNS: usize = 128;
const RESTORED_WIDE_64X128_BYTES: u32 = 49_152;

/// An odd number of 64-row tiles saves half a 128-row tile after the hop. Then each
/// projection can fill the card on its own grid, even when out and down leave SMs idle.
fn wide_64x128_fills(hidden: usize, columns: usize, rows: usize, sms: usize) -> bool {
    let row_tiles = rows.div_ceil(RESTORED_WIDE_64X128_ROWS);
    let width = if rows > HOP_ROWS && !row_tiles.is_multiple_of(2) {
        hidden.max(columns)
    } else {
        hidden
    };
    width.is_multiple_of(RESTORED_WIDE_64X128_COLUMNS)
        && width / RESTORED_WIDE_64X128_COLUMNS * row_tiles >= sms
}

#[cfg(test)]
mod wide_64x128_tests {
    use super::wide_64x128_fills;

    #[test]
    fn e5_layers_take_64_by_128_tiles_once_eight_columns_of_them_reach_every_multiprocessor() {
        for rows in 1..=1024 {
            assert_eq!(
                wide_64x128_fills(1024, 1024, rows, 48),
                rows > 320,
                "{rows} rows"
            );
            assert_eq!(
                wide_64x128_fills(1024, 1024, rows, 58),
                rows > 448,
                "{rows} rows"
            );
        }
    }

    #[test]
    fn wide_projections_use_their_own_grid_when_a_row_tile_is_saved() {
        for columns in [3072, 4096] {
            for (rows, fills) in [
                (176, false),
                (177, true),
                (192, true),
                (193, false),
                (256, false),
                (257, true),
                (320, true),
                (321, true),
            ] {
                assert_eq!(wide_64x128_fills(1024, columns, rows, 48), fills);
            }
        }
    }

    #[test]
    fn a_width_past_whole_tiles_keeps_the_other_tiles() {
        assert!(!wide_64x128_fills(1088, 1088, 1024, 48));
        assert!(wide_64x128_fills(384, 384, 1024, 48));
    }
}

/// repro.cu RestoredSmall: E5's out projection on 32 x 32 tiles, a warp chaining its own outputs.
const RESTORED_SMALL_ROWS: usize = 32;
const RESTORED_SMALL_COLUMNS: usize = 32;
const RESTORED_SMALL_BYTES: u32 = 16_384;
/// repro.cu RestoredBlocks: the up projection on tiles of 32 rows by one
/// 128-block, making the down projection's input while its grid fits a wave.
const RESTORED_BLOCKS_ROWS: usize = 32;
const RESTORED_BLOCKS_THREADS: u32 = 256;
const RESTORED_BLOCKS_DEPTHS: [(&str, u32, usize); 2] =
    [("gelu_blocks", 40_960, 2), ("gelu_blocks3", 30_720, 3)];
const UP_BLOCKS_ROWS: usize = 128;

/// repro_hop.cuh's down projection to HOP_COLUMNS outputs by HOP_DEPTH-deep segment: a kernel for
/// each count of 8-column units a block may take, a warp a HOP_TILE_ROWS-row tile and one more.
const HOP_DEPTH: usize = 1024;
const HOP_COLUMNS: usize = 1024;
/// No count of blocks shares a segment's units in fourteens.
const HOP_UNITS: [usize; 12] = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16];
const HOP_TILE_ROWS: usize = 16;
/// repro_hop.cuh HOP_MAX_TILES.
const HOP_TILES: usize = 11;
const HOP_ROWS: usize = HOP_TILES * HOP_TILE_ROWS;
/// repro_hop.cuh HOP_SLOTS.
const HOP_SLOTS: usize = 4;
/// repro_hop.cuh GELU_PASS_WARPS and GELU_PASS_ROWS.
const GELU_PASS_WARPS: usize = 8;
const GELU_PASS_ROWS: usize = 4;

/// repro_hop.cuh HopPlan<k, units>::BYTES.
const fn hop_bytes(k: usize, units: usize) -> u32 {
    let (slots, slot) = (HOP_SLOTS, units * 512);
    let fixed = HOP_TILES * 4 * 1024 + HOP_TILES * 16 * (k / STEP_INPUTS) * 4;
    (slots * slot + ((slots + 1) * 8).next_multiple_of(128) + fixed) as u32
}
const _: () = assert!(
    hop_bytes(HOP_DEPTH, 10) == 76_928,
    "repro_hop.cuh's HopPlan"
);

fn hop_threads(rows: usize) -> u32 {
    grid(32 * (rows.div_ceil(HOP_TILE_ROWS) + 1))
}

/// The grid, its blocks split over the segments: a block a multiprocessor, in the fewest waves
/// that keep every block's share within the widest kernel's.
fn hop_blocks(sms: usize) -> usize {
    let units = HOP_COLUMNS / 8;
    let fewest = TALL_SEGMENTS * units.div_ceil(HOP_UNITS[HOP_UNITS.len() - 1]);
    let most = TALL_SEGMENTS * (units / HOP_UNITS[0]);
    (fewest.div_ceil(sms) * sms).min(most)
}

fn hop_partials(n: usize, k: usize, rows: usize) -> bool {
    rows > FUSED_NARROW_ROWS
        && rows <= HOP_ROWS
        && k == TALL_SEGMENTS * HOP_DEPTH
        && n == HOP_COLUMNS
}

/// The narrowest kernel, in HOP_UNITS' order, holding a block's share of `units` over `blocks`.
fn hop_units(units: usize, blocks: usize) -> usize {
    let share = units.div_ceil(blocks);
    HOP_UNITS
        .into_iter()
        .find(|&held| held >= share)
        .expect("a kernel for every share (hop_blocks)")
}

/// Past 128 rows the blocks kernel wins while its whole grid fits one wave.
fn up_blocks(n: usize, rows: usize, sms: usize) -> bool {
    let tiles = n / ROTATION_BLOCK * rows.div_ceil(RESTORED_BLOCKS_ROWS);
    rows > FUSED_NARROW_ROWS
        && (rows <= UP_BLOCKS_ROWS || tiles <= RESTORED_BLOCKS_DEPTHS[1].2 * sms)
}

#[cfg(test)]
mod hop_partials_tests {
    use super::{
        FUSED_NARROW_ROWS, HOP_COLUMNS, HOP_ROWS, HOP_UNITS, TALL_SEGMENTS, hop_blocks,
        hop_partials, hop_units, up_blocks,
    };

    #[test]
    fn the_hop_takes_e5_down_projections_of_17_to_176_rows() {
        for rows in 0..=512 {
            let taken = rows > FUSED_NARROW_ROWS && rows <= HOP_ROWS;
            assert_eq!(hop_partials(1024, 4096, rows), taken, "E5, {rows} rows");
            assert!(!hop_partials(1024, 2048, rows), "2048 deep, {rows} rows");
            assert!(!hop_partials(768, 4096, rows), "768 wide, {rows} rows");
        }
    }

    #[test]
    fn every_multiprocessor_takes_a_block() {
        for sms in [32, 33, 47, 48, 58, 84, 127] {
            assert_eq!(hop_blocks(sms), sms, "{sms} multiprocessors");
        }
        // No block takes fewer units than the narrowest kernel's.
        assert_eq!(hop_blocks(128), 128);
        assert_eq!(hop_blocks(170), 128);
    }

    #[test]
    fn fewer_multiprocessors_than_the_widest_shares_fill_take_whole_waves() {
        for (sms, waves) in [(31, 2), (24, 2), (16, 2), (12, 3), (9, 4), (1, 32)] {
            assert_eq!(hop_blocks(sms), waves * sms, "{sms} multiprocessors");
        }
    }

    #[test]
    fn every_multiprocessor_count_has_a_kernel() {
        for sms in 1..=256 {
            let units = (HOP_COLUMNS / 8).div_ceil(hop_blocks(sms) / TALL_SEGMENTS);
            assert!(
                HOP_UNITS.contains(&units),
                "{sms} multiprocessors, {units} units"
            );
        }
    }

    #[test]
    fn a_one_segment_share_takes_the_narrowest_kernel_that_holds_it() {
        let units = TALL_SEGMENTS * HOP_COLUMNS / 8;
        for sms in 1..=256 {
            let blocks = hop_blocks(sms);
            let (share, held) = (units.div_ceil(blocks), hop_units(units, blocks));
            if !up_blocks(TALL_SEGMENTS * HOP_COLUMNS, HOP_ROWS, sms) {
                assert!(held >= 9, "no strip kernel below nine units");
            }
            // Only a share of fourteen units has no kernel of its own.
            assert!(
                held == share || (share, held) == (14, 15),
                "{sms} multiprocessors: {share} units in a kernel of {held}"
            );
        }
        assert_eq!(hop_units(units, hop_blocks(48)), 11);
        assert_eq!(hop_units(units, hop_blocks(38)), 15);
    }
}

/// repro.cu repro_restored_qkv_packed_{name}: one tile row holds the batch.
struct QkvPackedTile {
    name: &'static str,
    rows: usize,
    columns: usize,
    threads: u32,
    bytes: u32,
}

/// repro.cu QkvPacked64(Tall); the first that holds the batch is taken.
const QKV_PACKED_TILES: [QkvPackedTile; 3] = [
    QkvPackedTile {
        name: "64",
        rows: 64,
        columns: 64,
        threads: 256,
        bytes: 32_768,
    },
    QkvPackedTile {
        name: "64_tall",
        rows: 128,
        columns: 64,
        threads: 512,
        bytes: 49_152,
    },
    QkvPackedTile {
        name: "64_160",
        rows: PACKED_QKV_ROWS,
        columns: 64,
        threads: 640,
        bytes: 57_344,
    },
];

/// The steps a tall tile of TALL_ROWS rows stages (repro_project_restored_tall); taller
/// tiles and the segments' read theirs from the table.
fn tall_steps_bytes(k: usize) -> u32 {
    assert!(
        k <= STAGED_STEPS_DEPTH && k.is_multiple_of(STEP_INPUTS),
        "the tall tile's steps fit the loaded kernels' shared bytes"
    );
    u32::try_from(TALL_ROWS * (k / STEP_INPUTS) * 4).expect("fits u32")
}

/// repro.cu REPRO_POOL's columns a thread: rows of up to 1024.
const POOL_COLUMNS: usize = 4;

/// repro.cu pool_norm_shared.
fn pool_norm_bytes(h: usize, rows: usize, staged: bool) -> u32 {
    let stage = if staged { rows * h * 4 } else { 0 };
    u32::try_from(2 * h * 4 + rows * h * 8 + h * 4 + stage).expect("fits u32")
}

/// repro.cu norm_shared; a warp holds a row, at most 32 values a lane.
fn norm_bytes(h: usize, rows: usize) -> u32 {
    assert!(
        h.is_multiple_of(4) && h <= 1024,
        "a LayerNorm row fits a warp's registers"
    );
    u32::try_from(rows * h * 2 + 2 * h * 4).expect("fits u32")
}

fn invalid(reason: &str) -> CudaError {
    CudaError {
        call: "reproducible encoder",
        code: -1,
        name: reason.into(),
    }
}

fn validate_inputs(
    shape: Shape,
    ids: &[i32],
    types: &[i32],
    lengths: &[usize],
) -> Result<(), CudaError> {
    let batch = BatchLengths::new(lengths, MAX_KEYS.min(shape.max_sequence_len()));
    if batch.map(|batch| batch.tokens) != Some(ids.len())
        || types.len() != ids.len()
        || ids.iter().any(|&id| !(0..int(shape.vocab)).contains(&id))
        || types
            .iter()
            .any(|&t| !(0..int(shape.type_vocab)).contains(&t))
    {
        return Err(invalid(
            "inputs outside the model or the integer attention bound",
        ));
    }
    batch
        .expect("validated batch lengths")
        .check_grids()
        .map_err(invalid)
}

#[cfg(test)]
mod batch_input_tests {
    use super::{MAX_BATCH_SEQUENCES, Shape, validate_inputs};
    use kodamaembed_model::Positions;

    fn shape() -> Shape {
        Shape {
            layers: 1,
            hidden: 1024,
            heads: 16,
            intermediate: 4096,
            vocab: 2,
            max_positions: 512,
            type_vocab: 1,
            eps: 1e-12,
            positions: Positions::Sequential,
        }
    }

    fn mixed_lengths(sequences: usize) -> Vec<usize> {
        let mut lengths = vec![1; sequences];
        lengths[0] = 129;
        lengths
    }

    #[test]
    fn sequence_grid_boundary_is_valid_before_allocation() {
        let lengths = mixed_lengths(MAX_BATCH_SEQUENCES);
        let tokens = lengths.iter().sum();
        validate_inputs(shape(), &vec![1; tokens], &vec![0; tokens], &lengths).unwrap();
    }

    #[test]
    fn sequence_grid_overflow_is_rejected_before_allocation() {
        let lengths = mixed_lengths(MAX_BATCH_SEQUENCES + 1);
        let tokens = lengths.iter().sum();
        let error = validate_inputs(shape(), &vec![1; tokens], &vec![0; tokens], &lengths)
            .expect_err("too many sequences must fail before allocating GPU storage");
        assert!(error.to_string().contains("launch grids"), "{error}");
    }

    #[test]
    fn invalid_model_inputs_keep_priority_over_grid_limits() {
        let lengths = mixed_lengths(MAX_BATCH_SEQUENCES + 1);
        let tokens = lengths.iter().sum();
        let mut ids = vec![1; tokens];
        ids[0] = 2;
        let error = validate_inputs(shape(), &ids, &vec![0; tokens], &lengths).unwrap_err();
        assert_eq!(
            error.name,
            "inputs outside the model or the integer attention bound"
        );
    }
}

fn grid(count: usize) -> u32 {
    u32::try_from(count).expect("fits u32")
}

fn blocks(count: usize) -> u32 {
    grid(count.div_ceil(THREADS as usize))
}

fn int(value: usize) -> i32 {
    i32::try_from(value).expect("value fits i32")
}

/// The card's SM count fixes the hop grid and the share that each block holds.
struct HopKernel {
    function: Function,
    blocks: u32,
    bytes: u32,
}

impl HopKernel {
    fn load(
        module: &Module,
        form: &str,
        n: usize,
        segments: usize,
        blocks: usize,
    ) -> Result<Self, CudaError> {
        let units = hop_units(n / 8, blocks / segments);
        let function = module.function(&format!("repro_hop_{form}_{HOP_DEPTH}_{units}"))?;
        let bytes = hop_bytes(HOP_DEPTH, units);
        function.set_max_dynamic_shared(bytes)?;
        Ok(Self {
            function,
            blocks: grid(blocks),
            bytes,
        })
    }

    /// # Safety
    /// Input and output must hold this projection's rows and segment sums through completion.
    unsafe fn launch(
        &self,
        s: &Stream,
        p: &Projection,
        input: Quantized,
        rows: usize,
        batch: &Batch,
    ) -> Result<(), CudaError> {
        // SAFETY: the caller vouches for the buffers.
        unsafe {
            self.function.launch(
                s,
                [self.blocks, 1, 1],
                [hop_threads(rows), 1, 1],
                self.bytes,
                &mut Args::new()
                    .ptr(input.rows)
                    .ptr(input.steps)
                    .ptr(&p.q)
                    .i32(int(rows))
                    .i32(int(p.n))
                    .ptr(&batch.qkv),
            )
        }
    }
}

/// The repro_restored_* kernels by tile shape, each in Restored::index order.
struct RestoredKernels {
    narrow: [[Function; 3]; GROUPINGS.len()],
    tall: [[Function; 3]; TALL_CONFIGS.len()],
    tall_w8: [[Function; 3]; TALL_W8_CONFIGS.len()],
    wide: [Function; 3],
    wide_qkv: Function,
    wide_qkv64: Function,
    wide_residual64: Function,
    residual_small: Function,
    wide_gelu64: Function,
    wide_segments: Function,
    wide_residual_64x128: Function,
    wide_gelu_64x128: Function,
    wide_qkv_64x128: Function,
    wide_segments_64x128: Function,
    tall_segments: Function,
    tall_segments_m2: Function,
    gelu_next: [Function; GROUPINGS.len()],
    gelu_blocks: [Function; RESTORED_BLOCKS_DEPTHS.len()],
    hop_partials: HopKernel,
    hop_sums: Option<HopKernel>,
    qkv_packed: [Function; QKV_PACKED_TILES.len()],
    two: TwoDigitKernels,
}

/// repro.cu's `_two` kernels: the wide tiles' loop walks the weights twice for a two-digit operand.
struct TwoDigitKernels {
    plain_wide: Function,
    gelu_wide: Function,
    gelu_wide_64x128: Function,
    plain_wide64: Function,
    gelu_wide64: Function,
    qkv_wide: Function,
    qkv_wide_64x128: Function,
    qkv_wide64: Function,
    gelu_blocks: [Function; RESTORED_BLOCKS_DEPTHS.len()],
    qkv_packed: [Function; QKV_PACKED_TILES.len()],
}

impl TwoDigitKernels {
    fn load(module: &Module) -> Result<Self, CudaError> {
        let kernel = |name: &str, bytes: u32| -> Result<Function, CudaError> {
            let kernel = module.function(&format!("repro_restored_{name}_two"))?;
            kernel.set_max_dynamic_shared(bytes)?;
            Ok(kernel)
        };
        let blocks = |(name, bytes, _): (&str, u32, usize)| kernel(name, bytes);
        let packed =
            |tile: &QkvPackedTile| kernel(&format!("qkv_packed_{}", tile.name), tile.bytes);
        Ok(Self {
            plain_wide: kernel("plain_wide", RESTORED_WIDE_BYTES)?,
            gelu_wide: kernel("gelu_wide", RESTORED_WIDE_BYTES)?,
            gelu_wide_64x128: kernel("gelu_wide64x128", RESTORED_WIDE_64X128_BYTES)?,
            plain_wide64: kernel("plain_wide64", RESTORED_WIDE_DOWN_BYTES)?,
            gelu_wide64: kernel("gelu_wide64", RESTORED_WIDE_DOWN_BYTES)?,
            qkv_wide: kernel("qkv_wide", RESTORED_WIDE_BYTES)?,
            qkv_wide_64x128: kernel("qkv_wide64x128", RESTORED_WIDE_64X128_BYTES)?,
            qkv_wide64: kernel("qkv_wide64", RESTORED_WIDE_DOWN_BYTES)?,
            gelu_blocks: [
                blocks(RESTORED_BLOCKS_DEPTHS[0])?,
                blocks(RESTORED_BLOCKS_DEPTHS[1])?,
            ],
            qkv_packed: [
                packed(&QKV_PACKED_TILES[0])?,
                packed(&QKV_PACKED_TILES[1])?,
                packed(&QKV_PACKED_TILES[2])?,
            ],
        })
    }
}

impl RestoredKernels {
    fn load(module: &Module, sms: usize) -> Result<Self, CudaError> {
        let kernels = |suffix: &str, bytes: u32| -> Result<[Function; 3], CudaError> {
            let kernels = [
                module.function(&format!("repro_restored_plain{suffix}"))?,
                module.function(&format!("repro_restored_residual{suffix}"))?,
                module.function(&format!("repro_restored_gelu{suffix}"))?,
            ];
            for kernel in &kernels {
                kernel.set_max_dynamic_shared(bytes)?;
            }
            Ok(kernels)
        };
        let narrow = |groups: usize| {
            kernels(
                &format!("_g{groups}"),
                restored_bytes(groups, STAGED_STEPS_DEPTH),
            )
        };
        let tall = |config @ (groups, stages): (usize, usize)| {
            kernels(
                &format!("_tall_g{groups}s{stages}"),
                tall_bytes(config) + tall_steps_bytes(STAGED_STEPS_DEPTH),
            )
        };
        let tall_w8 = |stages: usize| kernels(&format!("_tall_w8s{stages}"), tall_w8_bytes(stages));
        let gelu_next = |groups: usize| -> Result<Function, CudaError> {
            let kernel = module.function(&format!("repro_restored_gelu_next_g{groups}"))?;
            kernel.set_max_dynamic_shared(restored_next_bytes(groups))?;
            Ok(kernel)
        };
        let tall_segments = module.function(&format!(
            "repro_restored_residual_tall_seg{TALL_SEGMENTS}s{TALL_SEGMENT_STAGES}"
        ))?;
        tall_segments.set_max_dynamic_shared(tall_segment_bytes())?;
        let tall_segments_m2 = module.function(&format!(
            "repro_restored_residual_tall_seg{TALL_SEGMENTS}m2s{TALL_M2_SEGMENT_STAGES}"
        ))?;
        tall_segments_m2.set_max_dynamic_shared(tall_m2_segment_bytes())?;
        let wide_segments = module.function("repro_restored_residual_wide_segments")?;
        wide_segments.set_max_dynamic_shared(RESTORED_WIDE_DOWN_BYTES)?;
        let wide_64x128 = |name: &str| -> Result<Function, CudaError> {
            let kernel = module.function(&format!("repro_restored_{name}"))?;
            kernel.set_max_dynamic_shared(RESTORED_WIDE_64X128_BYTES)?;
            Ok(kernel)
        };
        let wide_qkv = module.function("repro_restored_qkv_wide")?;
        wide_qkv.set_max_dynamic_shared(RESTORED_WIDE_BYTES)?;
        let wide_qkv64 = module.function("repro_restored_qkv_wide64")?;
        wide_qkv64.set_max_dynamic_shared(RESTORED_WIDE_DOWN_BYTES)?;
        let wide_residual64 = module.function("repro_restored_residual_wide64")?;
        wide_residual64.set_max_dynamic_shared(RESTORED_WIDE_DOWN_BYTES)?;
        let residual_small = module.function("repro_restored_residual_small")?;
        residual_small.set_max_dynamic_shared(RESTORED_SMALL_BYTES)?;
        let wide_gelu64 = module.function("repro_restored_gelu_wide64")?;
        wide_gelu64.set_max_dynamic_shared(RESTORED_WIDE_DOWN_BYTES)?;
        let gelu_blocks = |(name, bytes, _): (&str, u32, usize)| -> Result<Function, CudaError> {
            let kernel = module.function(&format!("repro_restored_{name}"))?;
            kernel.set_max_dynamic_shared(bytes)?;
            Ok(kernel)
        };
        let qkv_packed = |tile: &QkvPackedTile| -> Result<Function, CudaError> {
            let kernel = module.function(&format!("repro_restored_qkv_packed_{}", tile.name))?;
            kernel.set_max_dynamic_shared(tile.bytes)?;
            Ok(kernel)
        };
        let blocks = hop_blocks(sms);
        let up = TALL_SEGMENTS * HOP_COLUMNS;
        let hop_sums = (!up_blocks(up, HOP_ROWS, sms))
            .then(|| HopKernel::load(module, "sums", up, 1, blocks))
            .transpose()?;
        Ok(Self {
            narrow: [
                narrow(GROUPINGS[0])?,
                narrow(GROUPINGS[1])?,
                narrow(GROUPINGS[2])?,
            ],
            tall: [
                tall(TALL_CONFIGS[0])?,
                tall(TALL_CONFIGS[1])?,
                tall(TALL_CONFIGS[2])?,
            ],
            tall_w8: [tall_w8(TALL_W8_CONFIGS[0])?, tall_w8(TALL_W8_CONFIGS[1])?],
            wide: kernels("_wide", RESTORED_WIDE_BYTES)?,
            wide_qkv,
            wide_qkv64,
            wide_residual64,
            residual_small,
            wide_gelu64,
            wide_segments,
            wide_residual_64x128: wide_64x128("residual_wide64x128")?,
            wide_gelu_64x128: wide_64x128("gelu_wide64x128")?,
            wide_qkv_64x128: wide_64x128("qkv_wide64x128")?,
            wide_segments_64x128: wide_64x128("residual_wide64x128_segments")?,
            tall_segments,
            tall_segments_m2,
            gelu_next: [
                gelu_next(GROUPINGS[0])?,
                gelu_next(GROUPINGS[1])?,
                gelu_next(GROUPINGS[2])?,
            ],
            gelu_blocks: [
                gelu_blocks(RESTORED_BLOCKS_DEPTHS[0])?,
                gelu_blocks(RESTORED_BLOCKS_DEPTHS[1])?,
            ],
            hop_partials: HopKernel::load(module, "partials", HOP_COLUMNS, TALL_SEGMENTS, blocks)?,
            hop_sums,
            qkv_packed: [
                qkv_packed(&QKV_PACKED_TILES[0])?,
                qkv_packed(&QKV_PACKED_TILES[1])?,
                qkv_packed(&QKV_PACKED_TILES[2])?,
            ],
            two: TwoDigitKernels::load(module)?,
        })
    }
}

/// The attention kernel consuming operands packed by QKV or by the row packer.
struct PackedAttention {
    bucket: &'static PackedBucket,
    pairs: Function,
}

struct Functions {
    embed: Function,
    layernorm_pool: Function,
    layernorm_pool_joined: Function,
    layernorm_quantize: Function,
    layernorm_quantize_batch: Function,
    layernorm_quantize_joined: Function,
    clear: Function,
    read_ahead: Function,
    stage_inputs: Function,
    quantize_blocks: Function,
    gelu_half_blocks: Function,
    restored: RestoredKernels,
    /// For layers whose QKV epilogue cannot make attention's operands itself.
    attention_prep: Function,
    attention_values_max: Function,
    attention_rows: Function,
    attention_pairs: [Function; PLAIN_PAIR_BUCKETS],
    attention_pack: [Function; 2],
    attention_packed: Vec<PackedAttention>,
    pool_sums: Function,
    gelus: Function,
    weight_indices: Function,
    reciprocal_check: Function,
    scale_check: Function,
    units_check: Function,
    quotient_check: Function,
    square_sum_check: Function,
    pool_mean_check: Function,
    pool_quotient_check: Function,
}

struct Projection {
    n: usize,
    k: usize,
    /// The operand's INT8 digits (ProjectionExport): with two, a row holds its second digits
    /// behind its first and the product walks the weights twice in one chain.
    digits: usize,
    q: DeviceBuffer<i8>,
    /// Each column's scale / 14, divided as the CPU divides.
    restored_scales: DeviceBuffer<f32>,
    signs: DeviceBuffer<f32>,
    /// Of gamma z when the LayerNorm is deferred.
    centers: DeviceBuffer<f32>,
    /// Shared component plus bias in the CPU's one f32 addition; the component
    /// alone when deferred, as W beta + b holds the bias.
    static_shared: DeviceBuffer<f32>,
    deferred: Option<Deferred>,
}

/// W gamma (`sums`) and W beta + b (`constants`).
struct Deferred {
    sums: DeviceBuffer<f32>,
    constants: DeviceBuffer<f32>,
}

impl Deferred {
    fn upload(context: &Context, n: usize, d: &DeferredExport) -> Result<Self, CudaError> {
        if d.sums.len() != n || d.constants.len() != n {
            return Err(invalid("a deferred norm's terms for each column"));
        }
        Ok(Self {
            sums: DeviceBuffer::from_slice(context, &d.sums)?,
            constants: DeviceBuffer::from_slice(context, &d.constants)?,
        })
    }
}

impl Projection {
    fn upload(
        context: &Context,
        p: &ProjectionExport,
        n: usize,
        k: usize,
    ) -> Result<Self, CudaError> {
        if p.n != n
            || p.k != k
            || p.q.len() != n * k
            || p.scales.len() != n
            || p.signs.len() != k
            || p.bias.len() != n
        {
            return Err(invalid(
                "a projection of its layer's width and depth, its digits, scales, signs and bias",
            ));
        }
        if p.centers.len() != STATIC_CLASSES * k || p.center_shared.len() != STATIC_CLASSES * n {
            return Err(invalid("a center and its shared row for each static class"));
        }
        if !(1..=2).contains(&p.operand_digits) {
            return Err(invalid("a projection's operand in one or two digits"));
        }
        let static_shared: Vec<f32> = match &p.deferred {
            Some(_) => p.center_shared.clone(),
            None => p
                .center_shared
                .chunks_exact(p.n)
                .flat_map(|row| row.iter().zip(&p.bias).map(|(&c, &b)| c + b))
                .collect(),
        };
        let restored_scales: Vec<f32> = p.scales.iter().map(|&scale| scale / 14.0).collect();
        Ok(Self {
            n: p.n,
            k: p.k,
            digits: p.operand_digits,
            q: DeviceBuffer::from_slice(context, &p.q)?,
            restored_scales: DeviceBuffer::from_slice(context, &restored_scales)?,
            signs: DeviceBuffer::from_slice(context, &p.signs)?,
            centers: DeviceBuffer::from_slice(context, &p.centers)?,
            static_shared: DeviceBuffer::from_slice(context, &static_shared)?,
            deferred: p
                .deferred
                .as_ref()
                .map(|d| Deferred::upload(context, p.n, d))
                .transpose()?,
        })
    }
}

struct Norm {
    gamma: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    eps: f32,
}

impl Norm {
    fn upload(context: &Context, n: &NormExport, h: usize) -> Result<Self, CudaError> {
        if n.gamma.len() != h || n.beta.len() != h || !n.eps.is_finite() || n.eps <= 0.0 {
            return Err(invalid(
                "a LayerNorm's gain and bias a value each, its epsilon finite and positive",
            ));
        }
        Ok(Self {
            gamma: DeviceBuffer::from_slice(context, &n.gamma)?,
            beta: DeviceBuffer::from_slice(context, &n.beta)?,
            eps: n.eps,
        })
    }
}

struct Layer {
    qkv: Projection,
    key_centers: DeviceBuffer<f32>,
    out: Projection,
    norm1: Norm,
    up: Projection,
    down: Projection,
    norm2: Norm,
}

/// The CPU's static classes: a sequence's first token on 1, its interior on 3, its last token on 2.
struct GroupLayout {
    classes: Vec<i32>,
    of_row: Vec<i32>,
    row_shared: Vec<i32>,
}

impl GroupLayout {
    fn new(lengths: &[usize]) -> Self {
        let mut layout = Self {
            classes: vec![],
            of_row: vec![],
            row_shared: vec![],
        };
        let push = |layout: &mut Self, size: usize, class: i32| {
            layout
                .of_row
                .extend(std::iter::repeat_n(int(layout.classes.len()), size));
            layout.row_shared.extend(std::iter::repeat_n(-class, size));
            layout.classes.push(class);
        };
        for &len in lengths {
            push(&mut layout, 1, 1);
            if len > 2 {
                push(&mut layout, len - 2, 3);
            }
            if len > 1 {
                push(&mut layout, 1, 2);
            }
        }
        layout
    }
}

struct Groups {
    classes: DeviceBuffer<i32>,
    of_row: DeviceBuffer<i32>,
    row_shared: DeviceBuffer<i32>,
}

struct Work {
    quantized: DeviceBuffer<i8>,
    block_steps: DeviceBuffer<f32>,
    /// Apart from `quantized`, which the up projection still reads while it
    /// makes this; its per-block counts of tiles done are zero between launches.
    next_quantized: DeviceBuffer<i8>,
    next_steps: DeviceBuffer<f32>,
    next_done: DeviceBuffer<i32>,
    /// Each row's (mean, rstd) for the deferred projection.
    moments: DeviceBuffer<f32>,
}

#[derive(Clone, Copy)]
struct Quantized<'a> {
    rows: &'a DeviceBuffer<i8>,
    steps: &'a DeviceBuffer<f32>,
}

impl Work {
    fn input(&self) -> Quantized<'_> {
        Quantized {
            rows: &self.quantized,
            steps: &self.block_steps,
        }
    }

    fn next_input(&self) -> Quantized<'_> {
        Quantized {
            rows: &self.next_quantized,
            steps: &self.next_steps,
        }
    }
}

#[derive(Clone, Copy)]
struct Capacity {
    tokens: usize,
    sequences: usize,
    groups: usize,
    padded: usize,
}

impl Capacity {
    fn covers(self, need: Self) -> bool {
        self.tokens >= need.tokens
            && self.sequences >= need.sequences
            && self.groups >= need.groups
            && self.padded >= need.padded
    }

    fn grow_to(self, need: Self) -> Self {
        Self {
            // Reserve at most one row tile to avoid reallocating on each longer query.
            tokens: if self.tokens < need.tokens && need.tokens <= MAX_KEYS {
                need.tokens.next_multiple_of(PAIR_TILE_ROWS)
            } else {
                self.tokens.max(need.tokens)
            },
            sequences: self.sequences.max(need.sequences),
            groups: self.groups.max(need.groups),
            padded: self.padded.max(need.padded),
        }
    }
}

/// Kept across encodes uncleared: every value is written before it is read,
/// except the sums and counts kernels add into, which their last reader zeroes.
struct Batch {
    capacity: Capacity,
    ids: DeviceBuffer<i32>,
    types: DeviceBuffer<i32>,
    positions: DeviceBuffer<i32>,
    seq_starts: DeviceBuffer<i32>,
    seq_lengths: DeviceBuffer<i32>,
    pstarts: DeviceBuffer<i32>,
    groups: Groups,
    /// The FP16 residual stream; the first LayerNorm overwrites it only after
    /// its last reader, the attention output's residual, has run.
    hidden: DeviceBuffer<f16>,
    /// The context, then the residual projections' outputs: the out projection
    /// quantizes the context before its products land here.
    pre: DeviceBuffer<f16>,
    /// For attention where QKV's epilogue does not make its operands.
    qkv: DeviceBuffer<f32>,
    /// The GELU outputs, rounded to FP16 as every reader takes them.
    intermediate: DeviceBuffer<f16>,
    work: Work,
    attention: Attention,
    /// Row storage, including the temporary views shared by attention and GELU.
    _arena: DeviceBuffer<u8>,
    /// repro.cu ContextTail counts; each one's last block clears its own.
    tail_counts: DeviceBuffer<u32>,
    /// Pooling totals as integers in 2^-24 units, FP16's grid; zero between encodes.
    pool_units: DeviceBuffer<i64>,
    pool_kinds: DeviceBuffer<u32>,
    pool_counts: DeviceBuffer<i32>,
    /// The last read ahead's sums of the bytes each of its threads asked for.
    read_sums: DeviceBuffer<u32>,
    /// The host reads them after the stream, or on the graph path once all are marked.
    vectors: HostMarked,
    vectors_ptr: DevicePtr,
    inputs: DeviceBuffer<i32>,
    /// Packed inputs the embedding copies into `inputs` as each encode starts.
    staging: HostBuffer<i32>,
    staging_ptr: DevicePtr,
    staged: usize,
    input_layout: InputLayout,
}

/// What a repro_restored_* epilogue does with the products.
#[derive(Clone, Copy)]
enum Restored<'a> {
    /// The QKV rows, FP32, narrow attention's input.
    Rows(&'a DeviceBuffer<f32>),
    /// Attention's operands in place of QKV's rows, keys less `key_centers`.
    Attention { key_centers: &'a DeviceBuffer<f32> },
    /// The packed narrow attention's operands, as repro_attention_pack makes them.
    Packed {
        tile_index: usize,
        key_centers: &'a DeviceBuffer<f32>,
        sequences: usize,
        ptokens: usize,
    },
    /// Plus the FP16 residual, into `out` for the next LayerNorm.
    Residual {
        residual: &'a DeviceBuffer<f16>,
        out: &'a DeviceBuffer<f16>,
    },
    /// GELU into FP16 `out` (the up projection).
    Gelu(&'a DeviceBuffer<f16>),
    /// GELU, the tiles also making the down projection's input.
    GeluNext {
        out: &'a DeviceBuffer<f16>,
        next: &'a Projection,
    },
    /// GELU making only the down projection's input, on one-128-block tiles.
    GeluBlocks { next: &'a Projection },
}

impl Region {
    /// # Safety
    /// The owner outlives this view and orders access to overlapping regions.
    unsafe fn view<T: DeviceRepr>(self, arena: &DeviceBuffer<u8>) -> DeviceBuffer<T> {
        assert!(self.bytes.is_multiple_of(size_of::<T>()));
        // SAFETY: the caller retains the arena and orders overlapping accesses.
        unsafe { DeviceBuffer::alias(arena, self.offset, self.bytes / size_of::<T>()) }
    }
}

impl Batch {
    fn new(context: &Context, shape: Shape, mut capacity: Capacity) -> Result<Self, CudaError> {
        // The packed inputs already reserve a full aligned section for the classes.
        capacity.groups = InputLayout::aligned_words(capacity.groups);
        let (h, inner, heads) = (shape.hidden, shape.intermediate, shape.heads);
        let Capacity {
            tokens,
            sequences,
            padded,
            ..
        } = capacity;
        let input_layout = InputLayout::for_capacity(capacity);
        let inputs = DeviceBuffer::<i32>::zeroed(context, input_layout.words())?;
        let staging = HostBuffer::<i32>::zeroed(context, input_layout.words())?;
        let staging_ptr = staging.device_ptr()?;
        let vectors = HostMarked::new(context, sequences * h)?;
        let vectors_ptr = vectors.device_ptr()?;
        // SAFETY: Batch owns `inputs` with its views, each inside it.
        let input = |section: InputSection| unsafe {
            DeviceBuffer::alias(
                &inputs,
                input_layout.byte_offset(section),
                input_layout.len(section),
            )
        };
        // An operand's bytes a row: the down projection's depth, or two digits of an h-deep one.
        let deepest = inner.max(2 * h);
        // GELU cannot overwrite the stream's normalization readers' prefix
        // of `pre`, whose lifetime extends into the up projection.
        let layout = ArenaLayout::new(tokens, h, inner, padded);
        let mut bytes = layout.bytes;
        // Each independent row buffer occupies its own aligned range after the reused views.
        let hidden = Region::append_values::<f16>(&mut bytes, tokens * h);
        let qkv_len = (tokens.max(1) * 3 * h).max(TALL_SEGMENTS * h * tokens.min(HOP_ROWS));
        let qkv = Region::append_values::<f32>(&mut bytes, qkv_len);
        let operand = tokens * deepest;
        let next = tokens.max(STREAM_TILE_ROWS) * inner;
        let quantized = Region::append_values::<i8>(&mut bytes, operand);
        let block_steps = Region::append_values::<f32>(&mut bytes, operand / STEP_INPUTS);
        let next_quantized = Region::append_values::<i8>(&mut bytes, next);
        let next_steps = Region::append_values::<f32>(&mut bytes, next / STEP_INPUTS);
        let moments = Region::append_values::<f32>(&mut bytes, 2 * tokens);
        let key_scales = Region::append_values::<f32>(&mut bytes, tokens * heads);
        let query_scales = Region::append_values::<f32>(&mut bytes, tokens * heads);
        let row_max = Region::append_values::<f32>(&mut bytes, MAX_KEY_SPLITS * tokens * heads);
        let arena = DeviceBuffer::<u8>::zeroed(context, bytes)?;
        // SAFETY: Batch retains the arena. The appended regions are disjoint and
        // attention has finished before GELU reuses the temporary views.
        let (hidden, pre, qkv, intermediate, work, attention) = unsafe {
            (
                hidden.view(&arena),
                layout.pre.view(&arena),
                qkv.view(&arena),
                layout.intermediate.view(&arena),
                Work {
                    quantized: quantized.view(&arena),
                    block_steps: block_steps.view(&arena),
                    next_quantized: next_quantized.view(&arena),
                    next_steps: next_steps.view(&arena),
                    next_done: DeviceBuffer::zeroed(context, inner / 128)?,
                    moments: moments.view(&arena),
                },
                Attention {
                    value_scales: DeviceBuffer::zeroed(context, sequences * h)?,
                    key_scales: key_scales.view(&arena),
                    query_scales: query_scales.view(&arena),
                    row_max: row_max.view(&arena),
                    keys: layout.keys.view(&arena),
                    value_planes: layout.value_planes.view(&arena),
                    queries: layout.queries.view(&arena),
                    values16: layout.values16.view(&arena),
                    value_magnitudes: DeviceBuffer::zeroed(context, sequences * h)?,
                    failed: DeviceBuffer::zeroed(context, sequences * heads)?,
                    sequence_codes: input(InputSection::SequenceCodes),
                },
            )
        };
        Ok(Self {
            capacity,
            ids: input(InputSection::Ids),
            types: input(InputSection::Types),
            positions: input(InputSection::Positions),
            seq_starts: input(InputSection::SequenceStarts),
            seq_lengths: input(InputSection::SequenceLengths),
            pstarts: input(InputSection::PaddedStarts),
            groups: Groups {
                classes: input(InputSection::Classes),
                of_row: input(InputSection::RowGroups),
                row_shared: input(InputSection::SharedRows),
            },
            _arena: arena,
            hidden,
            pre,
            qkv,
            intermediate,
            work,
            attention,
            // Sequences of at most MAX_KEYS.
            tail_counts: DeviceBuffer::zeroed(
                context,
                2 * sequences * MAX_KEYS.div_ceil(ATT_ROWS),
            )?,
            pool_units: DeviceBuffer::zeroed(context, sequences * h)?,
            pool_kinds: DeviceBuffer::zeroed(context, sequences * h)?,
            pool_counts: DeviceBuffer::zeroed(context, sequences)?,
            // Allocated last: the driver packs small buffers in allocation order, and the
            // batch's other small buffers moved by this one's size run slower.
            read_sums: DeviceBuffer::zeroed(
                context,
                context.sm_count as usize * READ_AHEAD_THREADS as usize,
            )?,
            vectors,
            vectors_ptr,
            inputs,
            staging,
            staging_ptr,
            staged: 0,
            input_layout,
        })
    }

    /// The embedding fills `inputs` from the staging, so no transfer launch runs.
    fn upload(&mut self, sections: InputSections<'_>) {
        let sections = sections.as_array();
        let layout = InputLayout::new(sections.map(<[i32]>::len));
        layout.pack(self.staging.as_mut_slice(), sections);
        self.staged = layout.words();
        self.input_layout = layout;
        // SAFETY: Batch owns `inputs` with its views, each inside its capacity.
        let view = |section: InputSection| unsafe {
            DeviceBuffer::alias(
                &self.inputs,
                layout.byte_offset(section),
                layout.len(section),
            )
        };
        self.ids = view(InputSection::Ids);
        self.types = view(InputSection::Types);
        self.positions = view(InputSection::Positions);
        self.attention.sequence_codes = view(InputSection::SequenceCodes);
        self.groups.of_row = view(InputSection::RowGroups);
        self.groups.row_shared = view(InputSection::SharedRows);
        self.seq_starts = view(InputSection::SequenceStarts);
        self.seq_lengths = view(InputSection::SequenceLengths);
        self.pstarts = view(InputSection::PaddedStarts);
        self.groups.classes = view(InputSection::Classes);
    }
}

/// The CPU's calibrated centered INT8 preset on CUDA, bit for bit.
pub struct ReproEncoder {
    context: Context,
    /// Cached graphs must drop before the module that owns their kernels.
    graphs: Vec<(Vec<usize>, Vec<Graph>)>,
    // These handles can no longer be launched. One cached shape retires per encode.
    retired_graphs: Vec<Vec<Graph>>,
    stream: Stream,
    /// Records small batches' launches into `graphs` (non-blocking).
    recording: Stream,
    /// A separate stream lets weight prefetch overlap the current layer.
    reading: Stream,
    /// Whether the card's L2 holds the layers' digits a layer ahead.
    reads_ahead: bool,
    /// Orders `reading` after a layer's start and a capture's end after its reads.
    order: Event,
    done: Event,
    functions: Functions,
    shape: Shape,
    word: DeviceBuffer<f32>,
    position: DeviceBuffer<f32>,
    token_type: DeviceBuffer<f32>,
    embedding_norm: Norm,
    layers: Vec<Layer>,
    constants_f32: DeviceBuffer<f32>,
    /// Zero-padded to repro.cu ATT_WEIGHT_ENTRIES.
    attention_weights: DeviceBuffer<u16>,
    stream_engine: Option<StreamEngine>,
    /// The largest batch's storage so far.
    batch: Option<Batch>,
}

impl ReproEncoder {
    /// From kodamaembed-cpu's `Encoder::export_reproducible` of a calibrated preset.
    pub fn new(context: &Context, export: &EncoderExport) -> Result<Self, CudaError> {
        let shape = export.shape;
        shape.validate().map_err(|reason| invalid(&reason))?;
        // The kernels take e5-large's widths: 16 heads of 64 values, whole
        // 128-column blocks, LayerNorm rows a warp holds.
        if shape.geometry() != Some(Geometry::Large) {
            return Err(invalid("e5-large's widths and heads"));
        }
        if shape.positions == Positions::Mpnet {
            return Err(invalid(
                "MPNet's relative attention bias, which centered attention lacks",
            ));
        }
        let (h, inner) = (shape.hidden, shape.intermediate);
        let table = |values: usize, rows: usize| rows.checked_mul(h) == Some(values);
        if export.layers.len() != shape.layers
            || !table(export.word.len(), shape.vocab)
            || !table(export.position.len(), shape.max_positions)
            || !table(export.token_type.len(), shape.type_vocab)
            || export.constants_f32.len() != CONSTANTS_F32
        {
            return Err(invalid(
                "a layer each, embedding tables of the shape and the arithmetic's constants",
            ));
        }
        let _current = context.enter()?;
        let module = Module::load_side_by_side(context, REPRO_PTX)?;
        let d = h / shape.heads;
        let pack = |len: usize| module.function(&format!("repro_attention_pack_{d}_{len}"));
        let pairs = |len: usize| module.function(&format!("repro_attention_pairs_{d}_{len}"));
        let layernorm_pool = module.function(&format!("repro_layernorm_pool_{POOL_COLUMNS}"))?;
        layernorm_pool.set_max_dynamic_shared(pool_norm_bytes(h, NORM_BATCH_ROWS, false))?;
        let functions = Functions {
            embed: module.function("repro_embed")?,
            layernorm_pool,
            layernorm_pool_joined: module
                .function(&format!("repro_layernorm_pool_joined_{POOL_COLUMNS}"))?,
            layernorm_quantize: module.function("repro_layernorm_quantize")?,
            layernorm_quantize_batch: module.function("repro_layernorm_quantize_batch")?,
            layernorm_quantize_joined: module.function(&format!(
                "repro_layernorm_quantize_joined_{JOINED_NORM_WARPS}"
            ))?,
            clear: module.function("repro_clear")?,
            read_ahead: module.function("repro_read_ahead")?,
            stage_inputs: module.function("repro_stage_inputs")?,
            // Launched past UP_BLOCKS_ROWS rows alone, where nothing reads ahead; most of its
            // loads hit L1.
            quantize_blocks: module.function_alone("repro_quantize_half_blocks")?,
            gelu_half_blocks: module.function("repro_gelu_half_blocks")?,
            restored: RestoredKernels::load(&module, context.sm_count as usize)?,
            attention_prep: module.function("repro_attention_prep")?,
            attention_values_max: module.function(&format!("repro_attention_values_max_{d}"))?,
            attention_rows: module.function(&format!("repro_attention_rows_{d}"))?,
            attention_pairs: [pairs(PAIR_LENGTHS[0])?, pairs(PAIR_LENGTHS[1])?],
            attention_pack: [pack(ATT_CHUNK)?, pack(ROW_PACK_KEYS)?],
            attention_packed: PACKED_BUCKETS
                .iter()
                .filter(|bucket| bucket.d == d)
                .map(|bucket| -> Result<PackedAttention, CudaError> {
                    let (keys, tile) = (bucket.keys, bucket.tile_rows);
                    let pairs = module
                        .function(&format!("repro_attention_pairs_packed_{d}_{keys}_{tile}"))?;
                    pairs.set_max_dynamic_shared(packed_pairs_bytes(bucket))?;
                    Ok(PackedAttention { bucket, pairs })
                })
                .collect::<Result<_, _>>()?,
            pool_sums: module.function(&format!("repro_pool_sums_{POOL_COLUMNS}"))?,
            gelus: module.function("repro_gelus")?,
            weight_indices: module.function("repro_weight_indices")?,
            reciprocal_check: module.function("repro_reciprocal_check")?,
            scale_check: module.function("repro_scale_check")?,
            units_check: module.function("repro_units_check")?,
            quotient_check: module.function("repro_quotient_check")?,
            square_sum_check: module.function("repro_square_sum_check")?,
            pool_mean_check: module.function("repro_pool_mean_check")?,
            pool_quotient_check: module.function("repro_pool_quotient_check")?,
        };
        for f in &functions.attention_pairs {
            f.set_max_dynamic_shared(pairs_bytes(d, ATT_CHUNK, ATT_CHUNK, ATT_PAIR_THREADS))?;
        }
        let layers = export
            .layers
            .iter()
            .map(|l| {
                if l.key_centers.len() != h {
                    return Err(invalid("one static key center per hidden column"));
                }
                Ok(Layer {
                    qkv: Projection::upload(context, &l.qkv, 3 * h, h)?,
                    key_centers: DeviceBuffer::from_slice(context, &l.key_centers)?,
                    out: Projection::upload(context, &l.out, h, h)?,
                    norm1: Norm::upload(context, &l.norm1, h)?,
                    up: Projection::upload(context, &l.up, inner, h)?,
                    down: Projection::upload(context, &l.down, h, inner)?,
                    norm2: Norm::upload(context, &l.norm2, h)?,
                })
            })
            .collect::<Result<Vec<_>, CudaError>>()?;
        // A second digit is made where a LayerNorm quantizes for its deferred projection.
        let two_digits_fit = |l: &Layer| {
            l.out.digits == 1
                && l.down.digits == 1
                && [&l.qkv, &l.up]
                    .iter()
                    .all(|p| p.digits == 1 || p.deferred.is_some())
        };
        if !layers.iter().all(two_digits_fit) {
            return Err(invalid(
                "two operand digits only at a deferred QKV or up projection",
            ));
        }
        if export.attention_weights.len() != ATTENTION_SPAN + 1 {
            return Err(invalid(
                "an attention weight per exponent of the table's span",
            ));
        }
        // The wide attention sends a key past its sequence and a row without a finite score
        // to the span's last entry, which must weigh nothing.
        if export.attention_weights[ATTENTION_SPAN] != 0 {
            return Err(invalid("a zero weight at the end of the table's span"));
        }
        let mut attention_weights = export.attention_weights.clone();
        attention_weights.resize(ATTENTION_WEIGHTS, 0);
        let stream_engine = StreamEngine::load(&module, context, shape, &layers)?;
        // What read_ahead asks for: the four projections' digits of the widest layer.
        let layer_bytes = layers
            .iter()
            .map(|l| {
                [&l.qkv, &l.out, &l.up, &l.down]
                    .iter()
                    .map(|p| p.n * p.k)
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0);
        Ok(Self {
            context: context.clone(),
            stream: Stream::new(context)?,
            recording: Stream::new_non_blocking(context)?,
            reading: Stream::new_non_blocking(context)?,
            reads_ahead: l2_holds_a_layer_ahead(layer_bytes, context.l2_cache_bytes),
            order: Event::untimed(context)?,
            done: Event::untimed(context)?,
            functions,
            shape,
            word: DeviceBuffer::from_slice(context, &export.word)?,
            position: DeviceBuffer::from_slice(context, &export.position)?,
            token_type: DeviceBuffer::from_slice(context, &export.token_type)?,
            embedding_norm: Norm::upload(context, &export.embedding_norm, h)?,
            layers,
            constants_f32: DeviceBuffer::from_slice(context, &export.constants_f32)?,
            attention_weights: DeviceBuffer::from_slice(context, &attention_weights)?,
            stream_engine,
            batch: None,
            graphs: Vec::new(),
            retired_graphs: Vec::new(),
        })
    }

    /// Forces per-layer launches so tests cover them on GPUs that support the persistent kernel.
    #[doc(hidden)]
    pub fn set_aside_stream_engine(&mut self) {
        self.stream_engine = None;
        self.graphs.clear();
    }

    /// The UUID of this encoder's device; needs no NVML.
    pub fn device_uuid(&self) -> Result<[u8; 16], CudaError> {
        self.context.device_uuid()
    }

    /// Mean-pooled, L2-normalized vectors of sequences packed by `lengths`.
    pub fn encode(
        &mut self,
        ids: &[i32],
        types: &[i32],
        lengths: &[usize],
    ) -> Result<Vec<f32>, CudaError> {
        validate_inputs(self.shape, ids, types, lengths)?;
        let _current = self.context.enter()?;
        let ptokens = self.prepare(lengths, ids, types)?;
        // Earlier launches may still use these buffers when a later launch fails.
        let values = self.run(lengths, ptokens).inspect_err(|_| {
            let _ = self.stream.synchronize();
            self.retired_graphs.clear();
        })?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("non-finite outputs"));
        }
        Ok(values)
    }

    fn run(&mut self, lengths: &[usize], ptokens: usize) -> Result<Vec<f32>, CudaError> {
        let h = self.shape.hidden;
        let tokens: usize = lengths.iter().sum();
        let b = self.batch.as_ref().expect("prepared");
        let (layers, count) = (self.layers.len(), lengths.len() * h);
        // A mark from the previous encode would make an unwritten output look ready.
        b.vectors.clear(count);
        Ok(if tokens <= GRAPH_TOKENS {
            if let Some((_, graphs)) = self.graphs.iter().find(|(seen, _)| seen == lengths) {
                for graph in graphs {
                    // SAFETY: the cached graph names the current batch storage.
                    unsafe { graph.launch(&self.stream)? };
                }
            } else {
                if self.graphs.len() == GRAPH_SHAPES {
                    let (_, parts) = self.graphs.remove(0);
                    self.retired_graphs.push(parts);
                }
                let graphs = self.capture_and_launch(b, lengths, ptokens)?;
                self.graphs.push((lengths.to_vec(), graphs));
            }
            self.done.record(&self.stream)?;
            // Retire one shape per encode while the GPU runs the current shape.
            drop(self.retired_graphs.pop());
            // Outputs can arrive before the event completes. Querying it also flushes buffered launches.
            while !b.vectors.all_marked(count) && !self.done.query()? {
                std::hint::spin_loop();
            }
            b.vectors.values(count)
        } else {
            // SAFETY: the storage outlives the work (synchronized below).
            unsafe { self.launch_layers(&self.stream, b, lengths, ptokens, 0..layers + 1)? };
            self.retired_graphs.clear();
            self.stream.synchronize()?;
            b.vectors.values(count)
        })
    }

    /// Launches each part in stream order so the GPU can work during the next capture.
    fn capture_and_launch(
        &self,
        b: &Batch,
        lengths: &[usize],
        ptokens: usize,
    ) -> Result<Vec<Graph>, CudaError> {
        let layers = self.layers.len();
        let streamed = self.stream_engine_for(lengths).is_some();
        let mut graphs = Vec::new();
        let mut start = 0;
        while start < layers {
            // Cutting before the second LayerNorm gives each graph a single root and
            // doubling the part length overlaps host submission with GPU work.
            let next = 2 * start + 1;
            let end = if streamed || next >= layers {
                layers + 1
            } else {
                next
            };
            self.recording.begin_capture()?;
            // SAFETY: prepare laid out this batch, and the capture records the launches.
            let recorded =
                unsafe { self.launch_layers(&self.recording, b, lengths, ptokens, start..end) };
            let graph = self.recording.end_capture();
            recorded?;
            let graph = graph?;
            graph.upload(&self.stream)?;
            // SAFETY: encode retains the batch and drains the stream on error.
            unsafe { graph.launch(&self.stream)? };
            graphs.push(graph);
            start = end;
        }
        Ok(graphs)
    }

    /// Step `i` launches the embedding (step 0) or layer `i - 1`'s second LayerNorm, then
    /// layer `i` up to its own; the step past the last layer launches that LayerNorm alone.
    ///
    /// # Safety
    /// `b` is `prepare`'s layout for `lengths` and outlives the work; earlier
    /// steps have run or precede on the stream.
    unsafe fn launch_layers(
        &self,
        s: &Stream,
        b: &Batch,
        lengths: &[usize],
        ptokens: usize,
        steps: Range<usize>,
    ) -> Result<(), CudaError> {
        let tokens: usize = lengths.iter().sum();
        let layers = self.layers.len();
        // SAFETY: the caller vouches for the storage; each launch matches its kernel.
        unsafe {
            if let Some(engine) = self.stream_engine_for(lengths) {
                assert_eq!(
                    steps,
                    0..layers + 1,
                    "the stream engine runs every layer in one launch"
                );
                self.launch_embedding(s, b, tokens, lengths.len())?;
                self.launch_stream(s, b, lengths, engine)?;
                return self.pool(s, b, lengths);
            }
            let ahead = self.reads_ahead && tokens <= READ_AHEAD_ROWS;
            for i in steps {
                if i == 0 {
                    self.launch_embedding(s, b, tokens, lengths.len())?;
                } else {
                    self.launch_second_norm(s, b, i - 1, lengths)?;
                }
                if i < layers {
                    if ahead {
                        // Wrap to layer zero to warm its weights for the next encode.
                        let next = self.layers.get(i + 1).unwrap_or(&self.layers[0]);
                        self.order.record(s)?;
                        self.reading.wait(&self.order)?;
                        self.read_ahead(b, next)?;
                    }
                    self.launch_restored_layer(s, b, i, lengths, ptokens)?;
                }
            }
            if ahead {
                // Join prefetch work here so stream completion also covers those reads.
                self.order.record(&self.reading)?;
                s.wait(&self.order)?;
            }
        }
        Ok(())
    }

    /// # Safety
    /// As [`Self::launch_layers`].
    unsafe fn read_ahead(&self, b: &Batch, layer: &Layer) -> Result<(), CudaError> {
        let mut args = Args::new();
        for p in [&layer.qkv, &layer.out, &layer.up, &layer.down] {
            args = args.ptr(&p.q).i32(int(p.n * p.k));
        }
        // SAFETY: the digits and the sums outlive the work.
        unsafe {
            self.functions.read_ahead.launch(
                &self.reading,
                [self.context.sm_count, 1, 1],
                [READ_AHEAD_THREADS, 1, 1],
                0,
                &mut args.ptr(&b.read_sums),
            )
        }
    }

    /// Layer `i` up to its second LayerNorm.
    ///
    /// # Safety
    /// As [`Self::launch_layers`].
    unsafe fn launch_restored_layer(
        &self,
        s: &Stream,
        b: &Batch,
        i: usize,
        lengths: &[usize],
        ptokens: usize,
    ) -> Result<(), CudaError> {
        let tokens: usize = lengths.iter().sum();
        let (hidden, qkv, intermediate, pre) = (&b.hidden, &b.qkv, &b.intermediate, &b.pre);
        let layer = &self.layers[i];
        let residual = Restored::Residual {
            residual: hidden,
            out: pre,
        };
        let input = b.work.input();
        // SAFETY: as launch_layers.
        unsafe {
            // Attention's operands come from QKV's epilogue where its tiles allow, else from
            // FP32 rows; failures land in scratch the previous LayerNorm or staging cleared.
            let row_pairs = lengths.iter().all(|&len| len <= ROW_PACK_KEYS);
            let packed_tile = self
                .qkv_packed_tile(tokens)
                .filter(|_| (!row_pairs || tokens <= ROW_PACK_KEYS) && packed_bucket(lengths));
            let narrow = row_pairs || packed_tile.is_some();
            let operands = !narrow && self.qkv_operands(layer.qkv.n, layer.qkv.k, tokens);
            let made = if operands {
                Restored::Attention {
                    key_centers: &layer.key_centers,
                }
            } else if let Some(tile_index) = packed_tile {
                Restored::Packed {
                    tile_index,
                    key_centers: &layer.key_centers,
                    sequences: lengths.len(),
                    ptokens,
                }
            } else {
                Restored::Rows(qkv)
            };
            // The embedding or the previous second LayerNorm quantized the input.
            self.restored_products(s, &layer.qkv, input, tokens, b, made)?;
            if narrow {
                let rows = packed_tile.is_none().then_some(qkv);
                self.attend_pairs(s, b, layer, lengths, ptokens, rows, None)?;
            } else {
                let rows = (!operands).then_some(qkv);
                self.attend_wide(s, b, layer, lengths, ptokens, rows)?;
            }
            self.restored_products(s, &layer.out, input, tokens, b, residual)?;
            self.layernorm_quantize(s, &layer.norm1, pre, hidden, tokens, b, &layer.up, None)?;
            // Down by segment, joined in the second LayerNorm. A narrow up projection of two
            // digits has no kernel that also makes the down projection's input.
            let up_blocks = up_blocks(layer.up.n, tokens, self.context.sm_count as usize);
            if tokens <= FUSED_NARROW_ROWS && layer.up.digits == 1 {
                let up = Restored::GeluNext {
                    out: intermediate,
                    next: &layer.down,
                };
                self.restored_products(s, &layer.up, input, tokens, b, up)?;
                self.restored_products(s, &layer.down, b.work.next_input(), tokens, b, residual)?;
            } else if up_blocks {
                let up = Restored::GeluBlocks { next: &layer.down };
                self.restored_products(s, &layer.up, input, tokens, b, up)?;
                if self.down_partials(&layer.down, tokens) {
                    self.segment_partials(s, &layer.down, b.work.next_input(), tokens, b)?;
                } else {
                    self.restored_products(
                        s,
                        &layer.down,
                        b.work.next_input(),
                        tokens,
                        b,
                        residual,
                    )?;
                }
            } else if let Some(sums) = self.up_strips(layer, tokens) {
                sums.launch(s, &layer.up, input, tokens, b)?;
                self.gelu_half_blocks(s, &layer.up, &layer.down, tokens, b)?;
                self.segment_partials(s, &layer.down, b.work.next_input(), tokens, b)?;
            } else {
                self.restored_products(
                    s,
                    &layer.up,
                    input,
                    tokens,
                    b,
                    Restored::Gelu(intermediate),
                )?;
                if self.down_partials(&layer.down, tokens) {
                    self.quantize(s, &layer.down, intermediate, tokens, b)?;
                    self.segment_partials(s, &layer.down, input, tokens, b)?;
                } else {
                    self.project_restored(s, &layer.down, intermediate, tokens, b, residual)?;
                }
            }
        }
        Ok(())
    }

    /// Layer `i`'s second LayerNorm, which quantizes the next layer's input; the last pools.
    ///
    /// # Safety
    /// As [`Self::launch_layers`], behind the layer's [`Self::launch_restored_layer`].
    unsafe fn launch_second_norm(
        &self,
        s: &Stream,
        b: &Batch,
        i: usize,
        lengths: &[usize],
    ) -> Result<(), CudaError> {
        let tokens: usize = lengths.iter().sum();
        let (hidden, pre) = (&b.hidden, &b.pre);
        let layer = &self.layers[i];
        let joined = self.down_partials(&layer.down, tokens);
        // SAFETY: as launch_layers.
        unsafe {
            match self.layers.get(i + 1) {
                Some(next) => {
                    let clear = packed_bucket(lengths).then_some(lengths.len());
                    if joined {
                        self.layernorm_quantize_joined(
                            s,
                            &layer.norm2,
                            &layer.down,
                            hidden,
                            tokens,
                            b,
                            &next.qkv,
                            clear,
                        )
                    } else {
                        self.layernorm_quantize(
                            s,
                            &layer.norm2,
                            pre,
                            hidden,
                            tokens,
                            b,
                            &next.qkv,
                            clear,
                        )
                    }
                }
                None if joined => {
                    self.layernorm_pool_joined(s, &layer.norm2, &layer.down, b, lengths, tokens)
                }
                None => self.layernorm_pool(s, &layer.norm2, b, lengths),
            }
        }
    }

    /// The stream engine, for a batch of one row tile whose attention takes its workers few rounds.
    fn stream_engine_for(&self, lengths: &[usize]) -> Option<&StreamEngine> {
        let engine = self.stream_engine.as_ref()?;
        let tokens: usize = lengths.iter().sum();
        let units = lengths.len() * (self.shape.hidden / ROTATION_BLOCK);
        ((1..=STREAM_TILE_ROWS).contains(&tokens)
            && units <= STREAM_WORKERS * STREAM_ATTENTION_ROUNDS)
            .then_some(engine)
    }

    /// # Safety
    /// As [`Self::launch_layers`], with the embedding launched.
    unsafe fn launch_stream(
        &self,
        s: &Stream,
        b: &Batch,
        lengths: &[usize],
        engine: &StreamEngine,
    ) -> Result<(), CudaError> {
        let tokens: usize = lengths.iter().sum();
        let (shape, work) = (self.shape, &b.work);
        assert!(tokens <= STREAM_TILE_ROWS);
        let made = work.next_input();
        // SAFETY: the caller vouches for the batch; Work, the QKV buffer and the
        // slots were sized for what repro_stream_layers_* indexes.
        unsafe {
            let slots = &engine.slots;
            self.functions.clear.launch(
                s,
                [1, 1, 1],
                [THREADS, 1, 1],
                0,
                &mut Args::new().ptr(slots).i32(int(slots.len())),
            )?;
            engine.function.launch_cooperative(
                s,
                [self.context.sm_count, 1, 1],
                [STREAM_THREADS, 1, 1],
                STREAM_BYTES,
                &mut Args::new()
                    .ptr(&engine.layers)
                    .i32(int(self.layers.len()))
                    .i32(int(tokens))
                    .i32(int(shape.hidden))
                    .i32(int(shape.intermediate))
                    .i32(int(shape.heads))
                    .i32(int(lengths.len()))
                    .ptr(&b.seq_starts)
                    .ptr(&b.seq_lengths)
                    .ptr(&b.groups.row_shared)
                    .ptr(&b.groups.classes)
                    .ptr(&b.groups.of_row)
                    .ptr(&self.constants_f32)
                    .ptr(&self.attention_weights)
                    .ptr(&b.hidden)
                    .ptr(&b.pre)
                    .ptr(&work.quantized)
                    .ptr(&work.block_steps)
                    .ptr(&b.qkv)
                    .ptr(&b.attention.value_scales)
                    .ptr(&b.intermediate)
                    .ptr(made.rows)
                    .ptr(made.steps)
                    .ptr(slots)
                    .ptr(&engine.moments)
                    .i32(two_digit_layers(&self.layers, |l| &l.qkv))
                    .i32(two_digit_layers(&self.layers, |l| &l.up)),
            )
        }
    }

    /// For checks that run no embedding, which otherwise stages the batch.
    ///
    /// # Safety
    /// `b` holds an upload of at least `sequences` sequences and outlives the work.
    unsafe fn stage_inputs(
        &self,
        s: &Stream,
        b: &Batch,
        sequences: usize,
    ) -> Result<(), CudaError> {
        // Sections end on 64-word boundaries: whole vectors.
        let vectors = b.staged / 4;
        let scratch = &b.attention;
        // SAFETY: the caller vouches for the storage, sized for these words.
        unsafe {
            self.functions.stage_inputs.launch(
                s,
                [blocks(vectors).min(64), 1, 1],
                [THREADS, 1, 1],
                0,
                &mut Args::new()
                    .raw(b.staging_ptr)
                    .ptr(&b.inputs)
                    .i32(int(vectors))
                    .ptr(&scratch.value_magnitudes)
                    .ptr(&scratch.failed)
                    .i32(int(sequences))
                    .i32(int(self.shape.hidden))
                    .i32(int(self.shape.heads)),
            )
        }
    }

    /// # Safety
    /// `b` is `prepare`'s layout for this batch and outlives the work.
    unsafe fn launch_embedding(
        &self,
        s: &Stream,
        b: &Batch,
        tokens: usize,
        sequences: usize,
    ) -> Result<(), CudaError> {
        let h = self.shape.hidden;
        let n = &self.embedding_norm;
        let qkv = &self.layers.first().expect("has layers").qkv;
        assert!(
            qkv.deferred.is_none(),
            "the first QKV projection takes the embedding's normalized rows"
        );
        let staged = |section: InputSection| {
            b.staging_ptr + b.input_layout.byte_offset(section) as DevicePtr
        };
        let scratch = &b.attention;
        // SAFETY: the caller vouches for the storage, sized for this batch.
        unsafe {
            self.functions.embed.launch(
                s,
                [grid(tokens.div_ceil(NORM_ROWS)), 1, 1],
                [grid(32 * NORM_ROWS), 1, 1],
                norm_bytes(h, NORM_ROWS),
                &mut Args::new()
                    .raw(b.staging_ptr)
                    .ptr(&b.inputs)
                    .i32(int(b.staged / 4))
                    .ptr(&scratch.value_magnitudes)
                    .ptr(&scratch.failed)
                    .i32(int(sequences))
                    .i32(int(self.shape.heads))
                    .raw(staged(InputSection::Ids))
                    .raw(staged(InputSection::Types))
                    .raw(staged(InputSection::Positions))
                    .ptr(&self.word)
                    .ptr(&self.position)
                    .ptr(&self.token_type)
                    .ptr(&n.gamma)
                    .ptr(&n.beta)
                    .f32(n.eps)
                    .i32(int(h))
                    .i32(int(tokens))
                    .ptr(&b.hidden)
                    .raw(staged(InputSection::SharedRows))
                    .ptr(&qkv.centers)
                    .ptr(&qkv.signs)
                    .ptr(&self.constants_f32)
                    .ptr(&b.work.quantized)
                    .ptr(&b.work.block_steps),
            )
        }
    }

    /// Uploads a batch in one copy, growing the storage as needed; returns the padded token count.
    fn prepare(
        &mut self,
        lengths: &[usize],
        ids: &[i32],
        types: &[i32],
    ) -> Result<usize, CudaError> {
        let _current = self.context.enter()?;
        let starts: Vec<i32> = lengths
            .iter()
            .scan(0, |at, &len| {
                *at += len;
                Some(int(*at - len))
            })
            .collect();
        // Each sequence's value plane starts at a whole key chunk, so every
        // staged chunk lies inside its sequence's padding.
        let pstarts: Vec<i32> = lengths
            .iter()
            .scan(0, |at, &len| {
                *at += len.next_multiple_of(ATT_CHUNK);
                Some(int(*at - len.next_multiple_of(ATT_CHUNK)))
            })
            .collect();
        let ptokens: usize = lengths
            .iter()
            .map(|len| len.next_multiple_of(ATT_CHUNK))
            .sum();
        let tokens: usize = lengths.iter().sum();
        let layout = GroupLayout::new(lengths);
        let need = Capacity {
            tokens,
            sequences: lengths.len(),
            groups: layout.classes.len(),
            padded: ptokens,
        };
        if !self.batch.as_ref().is_some_and(|b| b.capacity.covers(need)) {
            let grown = self
                .batch
                .as_ref()
                .map_or(need, |b| b.capacity.grow_to(need));
            // Only the new storage may be named by a later launch.
            self.retired_graphs
                .extend(self.graphs.drain(..).map(|(_, parts)| parts));
            self.batch = None;
            let batch = Batch::new(&self.context, self.shape, grown)
                .inspect_err(|_| self.retired_graphs.clear())?;
            self.batch = Some(batch);
        }
        let b = self.batch.as_mut().expect("storage allocated above");
        let positions = self.shape.position_ids(ids, lengths);
        // Twice the row's sequence, plus one for its sinks.
        let sequence_codes: Vec<i32> = lengths
            .iter()
            .enumerate()
            .flat_map(|(s, &len)| {
                (0..len).map(move |j| int(2 * s) + i32::from(j == 0 || j + 1 == len))
            })
            .collect();
        let seq_lengths: Vec<i32> = lengths.iter().map(|&len| int(len)).collect();
        b.upload(InputSections {
            ids,
            types,
            positions: &positions,
            sequence_codes: &sequence_codes,
            row_groups: &layout.of_row,
            shared_rows: &layout.row_shared,
            sequence_starts: &starts,
            sequence_lengths: &seq_lengths,
            padded_starts: &pstarts,
            classes: &layout.classes,
        });
        Ok(ptokens)
    }

    /// Every intermediate stays on chip: a block holds all of its sequence's keys.
    ///
    /// # Safety
    /// `b` holds `prepare`'s batch of narrow sequences, `rows` its QKV rows
    /// unless QKV packed the operands, and a packed bucket's flags are cleared.
    #[allow(clippy::too_many_arguments)]
    unsafe fn attend_pairs(
        &self,
        s: &Stream,
        b: &Batch,
        layer: &Layer,
        lengths: &[usize],
        ptokens: usize,
        rows: Option<&DeviceBuffer<f32>>,
        context: Option<&DeviceBuffer<f16>>,
    ) -> Result<(), CudaError> {
        let (key_centers, out) = (&layer.key_centers, &layer.out);
        let (h, heads) = (self.shape.hidden, self.shape.heads);
        let tokens: usize = lengths.iter().sum();
        let longest = lengths.iter().copied().max().unwrap_or(0);
        let bucket = PAIR_LENGTHS
            .iter()
            .position(|&len| longest <= len)
            .expect("within widest bucket");
        assert_eq!(out.k, h);
        let tiles = |rows: usize| {
            [
                grid(lengths.len()),
                grid(h / ROTATION_BLOCK),
                grid(longest.div_ceil(rows)),
            ]
        };
        let scratch = &b.attention;
        let f = &self.functions;
        // SAFETY: the caller vouches for the batch; every sequence fits the bucket.
        unsafe {
            if let Some(pairs) = f.attention_pairs.get(bucket) {
                let rows = rows.expect("plain bucket rows");
                return pairs.launch(
                    s,
                    tiles(PAIR_TILE_ROWS),
                    [ATT_PAIR_THREADS, 1, 1],
                    pairs_bytes(h / heads, ATT_CHUNK, ATT_CHUNK, ATT_PAIR_THREADS),
                    &mut Args::new()
                        .ptr(rows)
                        .ptr(&b.seq_starts)
                        .ptr(&b.seq_lengths)
                        .i32(int(h))
                        .i32(int(heads))
                        .ptr(key_centers)
                        .ptr(&self.attention_weights)
                        .ptr(&scratch.value_scales)
                        .ptr(&out.centers)
                        .i32(int(STATIC_CLASSES))
                        .ptr(&b.groups.classes)
                        .ptr(&b.groups.of_row)
                        .ptr(&out.signs)
                        .ptr(&self.constants_f32)
                        .ptr(&b.work.quantized)
                        .ptr(&b.work.block_steps)
                        .raw(context.map_or(0, |c| c.ptr()))
                        .i32(int(tokens)),
                );
            }
            let sms = self.context.sm_count as usize;
            let keys = |p: &&PackedAttention| p.bucket.keys == PAIR_LENGTHS[bucket];
            let packed = f
                .attention_packed
                .iter()
                .filter(keys)
                .find(|p| packed_fits(p.bucket, lengths.len(), h / ROTATION_BLOCK, longest, sms))
                .or_else(|| f.attention_packed.iter().rfind(keys))
                .expect("packed bucket");
            if let Some(rows) = rows {
                let pack = &f.attention_pack[bucket - PLAIN_PAIR_BUCKETS];
                pack.launch(
                    s,
                    [
                        grid(lengths.len()),
                        grid(h / ROTATION_BLOCK),
                        grid(packed.bucket.keys / PAIR_TILE_ROWS),
                    ],
                    [ATT_PAIR_THREADS, 1, 1],
                    0,
                    &mut Args::new()
                        .ptr(rows)
                        .ptr(&b.seq_starts)
                        .ptr(&b.seq_lengths)
                        .i32(int(h))
                        .ptr(key_centers)
                        .ptr(&scratch.queries)
                        .ptr(&scratch.query_scales)
                        .ptr(&scratch.keys)
                        .ptr(&scratch.key_scales)
                        .ptr(&scratch.value_planes)
                        .ptr(&scratch.value_scales)
                        .ptr(&scratch.values16)
                        .ptr(&scratch.failed)
                        .ptr(&b.pstarts)
                        .i32(int(tokens))
                        .i32(int(ptokens)),
                )?;
            }
            packed.pairs.launch(
                s,
                tiles(packed.bucket.tile_rows),
                [packed.bucket.threads, 1, 1],
                packed_pairs_bytes(packed.bucket),
                &mut Args::new()
                    .ptr(&b.seq_starts)
                    .ptr(&b.seq_lengths)
                    .i32(int(h))
                    .i32(int(heads))
                    .ptr(&self.attention_weights)
                    .ptr(&scratch.value_scales)
                    .ptr(&out.centers)
                    .i32(int(STATIC_CLASSES))
                    .ptr(&b.groups.classes)
                    .ptr(&b.groups.of_row)
                    .ptr(&out.signs)
                    .ptr(&self.constants_f32)
                    .ptr(&b.work.quantized)
                    .ptr(&b.work.block_steps)
                    .raw(context.map_or(0, |c| c.ptr()))
                    .ptr(&scratch.queries)
                    .ptr(&scratch.query_scales)
                    .ptr(&scratch.keys)
                    .ptr(&scratch.key_scales)
                    .ptr(&scratch.value_planes)
                    .ptr(&scratch.values16)
                    .ptr(&scratch.failed)
                    .ptr(&b.pstarts)
                    .i32(int(tokens))
                    .i32(int(ptokens)),
            )
        }
    }

    /// # Safety
    /// `b` holds `prepare`'s batch, its value magnitudes and flags cleared before
    /// the QKV output took any.
    unsafe fn attend_wide(
        &self,
        s: &Stream,
        b: &Batch,
        layer: &Layer,
        lengths: &[usize],
        ptokens: usize,
        fp32_rows: Option<&DeviceBuffer<f32>>,
    ) -> Result<(), CudaError> {
        let (key_centers, out) = (&layer.key_centers, &layer.out);
        let (h, heads) = (self.shape.hidden, self.shape.heads);
        let (tokens, d) = (lengths.iter().sum::<usize>(), h / heads);
        let longest = lengths.iter().copied().max().unwrap_or(0);
        let f = &self.functions;
        let scratch = &b.attention;
        // SAFETY: the caller vouches for the batch; the arguments match the kernels.
        unsafe {
            if let Some(rows) = fp32_rows {
                // Its row tiles take maxima and flags into the cleared ones.
                f.attention_prep.launch(
                    s,
                    [
                        grid(lengths.len()),
                        grid(heads),
                        grid(longest.div_ceil(PREP_ROWS)),
                    ],
                    [ATT_THREADS, 1, 1],
                    0,
                    &mut Args::new()
                        .ptr(rows)
                        .ptr(&b.seq_starts)
                        .ptr(&b.seq_lengths)
                        .i32(int(tokens))
                        .i32(int(h))
                        .i32(int(heads))
                        .i32(int(d))
                        .ptr(key_centers)
                        .ptr(&scratch.query_scales)
                        .ptr(&scratch.queries)
                        .ptr(&scratch.key_scales)
                        .ptr(&scratch.keys)
                        .ptr(&scratch.values16)
                        .ptr(&scratch.value_magnitudes)
                        .ptr(&scratch.failed),
                )?;
            }
            let (value_tiles, row_tiles) =
                (longest.div_ceil(ATT_CHUNK), longest.div_ceil(ATT_ROWS));
            // Values and row maxima share a launch: each row tile splits its keys
            // over the blocks the value tiles leave resident, at most its chunks.
            let resident = self.context.sm_count as usize
                * f.attention_values_max.resident_blocks(ATT_ROW_THREADS, 0)? as usize;
            let groups = heads * lengths.len();
            let splits = (resident.saturating_sub(value_tiles * groups) / (row_tiles * groups))
                .clamp(1, longest.div_ceil(ATT_CHUNK));
            f.attention_values_max.launch(
                s,
                [
                    grid(value_tiles + row_tiles * splits),
                    grid(heads),
                    grid(lengths.len()),
                ],
                [ATT_ROW_THREADS, 1, 1],
                0,
                &mut Args::new()
                    .ptr(&scratch.values16)
                    .ptr(&scratch.value_magnitudes)
                    .ptr(&scratch.query_scales)
                    .ptr(&scratch.key_scales)
                    .ptr(&b.seq_starts)
                    .ptr(&b.pstarts)
                    .ptr(&b.seq_lengths)
                    .i32(int(tokens))
                    .i32(int(ptokens))
                    .i32(int(h))
                    .i32(int(heads))
                    .ptr(&scratch.value_scales)
                    .ptr(&scratch.value_planes)
                    .ptr(&scratch.failed)
                    .ptr(&scratch.queries)
                    .ptr(&scratch.keys)
                    .i32(int(splits))
                    .ptr(&scratch.row_max)
                    .i32(int(value_tiles)),
            )?;
            // Each quantization group finishes when its last head publishes context.
            f.attention_rows.launch(
                s,
                [grid(heads), grid(row_tiles), grid(lengths.len())],
                [ATT_ROW_THREADS, 1, 1],
                0,
                &mut Args::new()
                    .ptr(&scratch.queries)
                    .ptr(&scratch.query_scales)
                    .ptr(&scratch.keys)
                    .ptr(&scratch.key_scales)
                    .ptr(&scratch.value_planes)
                    .ptr(&scratch.value_scales)
                    .ptr(&scratch.values16)
                    .ptr(&self.attention_weights)
                    .ptr(&scratch.failed)
                    .ptr(&b.seq_starts)
                    .ptr(&b.pstarts)
                    .ptr(&b.seq_lengths)
                    .i32(int(tokens))
                    .i32(int(ptokens))
                    .i32(int(h))
                    .i32(int(heads))
                    .i32(int(splits))
                    .ptr(&scratch.row_max)
                    .ptr(&b.pre)
                    .ptr(&b.tail_counts)
                    .ptr(&b.groups.classes)
                    .ptr(&b.groups.of_row)
                    .ptr(&out.centers)
                    .ptr(&out.signs)
                    .ptr(&self.constants_f32)
                    .ptr(&b.work.quantized)
                    .ptr(&b.work.block_steps),
            )
        }
    }

    /// Attention on crafted inputs, as the CPU's `integer_attention` takes them.
    #[doc(hidden)]
    pub fn crafted_attention(
        &mut self,
        qkv: &[f32],
        lengths: &[usize],
    ) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
        let h = self.shape.hidden;
        let batch = BatchLengths::new(lengths, MAX_KEYS);
        if batch.and_then(|batch| batch.tokens.checked_mul(3 * h)) != Some(qkv.len())
            || batch.is_none_or(|batch| batch.check_grids().is_err())
        {
            return Err(invalid(
                "attention inputs outside the integer attention bound or the launch grids",
            ));
        }
        let tokens = qkv.len() / (3 * h);
        // No embedding runs: the ids and types are zeros.
        let zeros = vec![0; tokens];
        let ptokens = self.prepare(lengths, &zeros, &zeros)?;
        let rows = DeviceBuffer::from_slice(&self.context, qkv)?;
        let b = self.batch.as_ref().expect("prepared");
        let layer = self
            .layers
            .first()
            .ok_or_else(|| invalid("an encoder with layers"))?;
        // SAFETY: prepare laid out this batch and its QKV rows were uploaded;
        // the staging clears the flags before the kernels take theirs.
        let launched = unsafe {
            self.stage_inputs(&self.stream, b, lengths.len())
                .and_then(|()| {
                    if lengths.iter().all(|&len| len <= ROW_PACK_KEYS) {
                        self.attend_pairs(
                            &self.stream,
                            b,
                            layer,
                            lengths,
                            ptokens,
                            Some(&rows),
                            Some(&b.pre),
                        )
                    } else {
                        self.attend_wide(&self.stream, b, layer, lengths, ptokens, Some(&rows))
                    }
                })
        };
        // Kernels queued before a failed launch still read the crafted rows.
        let synchronized = self.stream.synchronize();
        launched?;
        synchronized?;
        let context = b.pre.read_first(tokens * h)?;
        Ok((
            context.iter().map(|v| v.to_f32()).collect(),
            b.attention.value_scales.read_first(lengths.len() * h)?,
        ))
    }

    /// # Safety
    /// `b` holds `prepare`'s batch of at most MAX_KEYS rows a sequence (the
    /// sums' exactness bound), with zeroed pool totals.
    unsafe fn pool(&self, s: &Stream, b: &Batch, lengths: &[usize]) -> Result<(), CudaError> {
        let h = self.shape.hidden;
        let f = &self.functions;
        let longest = lengths.iter().copied().max().unwrap_or(0);
        // SAFETY: the caller vouches for the batch; the arguments match the kernel.
        unsafe {
            f.pool_sums.launch(
                s,
                [grid(lengths.len()), grid(longest.div_ceil(POOL_ROWS)), 1],
                [POOL_THREADS, 1, 1],
                0,
                &mut Args::new()
                    .ptr(&b.hidden)
                    .ptr(&b.seq_starts)
                    .ptr(&b.seq_lengths)
                    .i32(int(h))
                    .ptr(&b.pool_units)
                    .ptr(&b.pool_kinds)
                    .ptr(&b.pool_counts)
                    .raw(b.vectors_ptr),
            )
        }
    }

    /// Pooling on crafted FP16 rows, as the CPU's `pool_rows` takes them.
    #[doc(hidden)]
    pub fn crafted_pool(
        &mut self,
        hidden: &[f16],
        lengths: &[usize],
    ) -> Result<Vec<f32>, CudaError> {
        let h = self.shape.hidden;
        let batch = BatchLengths::new(lengths, MAX_KEYS);
        if batch.and_then(|batch| batch.tokens.checked_mul(h)) != Some(hidden.len())
            || batch.is_none_or(|batch| batch.check_grids().is_err())
        {
            return Err(invalid(
                "pooling inputs of empty sequences or past the key bound, the wrong size or past \
                 the launch grids",
            ));
        }
        let tokens = hidden.len() / h;
        // No embedding runs: the ids and types are zeros.
        let zeros = vec![0; tokens];
        self.prepare(lengths, &zeros, &zeros)?;
        self.batch
            .as_mut()
            .expect("prepared")
            .hidden
            .write(hidden)?;
        let b = self.batch.as_ref().expect("prepared");
        // SAFETY: prepare laid out this batch and its hidden rows were written.
        let launched = unsafe {
            self.stage_inputs(&self.stream, b, lengths.len())
                .and_then(|()| self.pool(&self.stream, b, lengths))
        };
        // Kernels queued before a failed launch still read the batch.
        let synchronized = self.stream.synchronize();
        launched?;
        synchronized?;
        Ok(b.vectors.values(lengths.len() * h))
    }

    /// The sum of the bytes each thread of the last read ahead asked for.
    #[doc(hidden)]
    pub fn read_ahead_sums(&self) -> Result<Vec<u32>, CudaError> {
        self.stream.synchronize()?;
        self.batch.as_ref().expect("prepared").read_sums.read()
    }

    /// Each recorded graph's roots, in the order recorded.
    #[doc(hidden)]
    pub fn graph_roots(&self) -> Vec<usize> {
        self.graphs
            .iter()
            .flat_map(|(_, graphs)| graphs.iter().map(Graph::roots))
            .collect()
    }

    /// GELU of the `n` floats whose bits run from `first`.
    #[doc(hidden)]
    pub fn gelus(&self, first: u32, n: usize) -> Result<Vec<f32>, CudaError> {
        let _current = self.context.enter()?;
        let out = DeviceBuffer::<f32>::zeroed(&self.context, n)?;
        // SAFETY: `out` holds n values and outlives the synchronized launch.
        unsafe {
            self.functions.gelus.launch(
                &self.stream,
                [blocks(n), 1, 1],
                [THREADS, 1, 1],
                0,
                &mut Args::new()
                    .i32(i32::from_ne_bytes(first.to_ne_bytes()))
                    .i32(int(n))
                    .ptr(&self.constants_f32)
                    .ptr(&out),
            )?;
        }
        self.stream.synchronize()?;
        out.read()
    }

    /// The weight-table index of the `n` exponents whose bits run from `first`.
    #[doc(hidden)]
    pub fn weight_indices(&self, first: u32, n: usize) -> Result<Vec<u16>, CudaError> {
        let _current = self.context.enter()?;
        let out = DeviceBuffer::<u16>::zeroed(&self.context, n)?;
        // SAFETY: `out` holds n indices and outlives the synchronized launch.
        unsafe {
            self.functions.weight_indices.launch(
                &self.stream,
                [blocks(n), 1, 1],
                [THREADS, 1, 1],
                0,
                &mut Args::new()
                    .i32(i32::from_ne_bytes(first.to_ne_bytes()))
                    .i32(int(n))
                    .ptr(&out),
            )?;
        }
        self.stream.synchronize()?;
        out.read()
    }

    /// How many reciprocals of `n` floats from `first` differ from division.
    #[doc(hidden)]
    pub fn reciprocal_check(&self, first: u32, n: usize) -> Result<u64, CudaError> {
        self.count_mismatches(&self.functions.reciprocal_check, first, n)
    }

    /// How many query and key steps differ from scale_for's division by 127.
    #[doc(hidden)]
    pub fn scale_check(&self, first: u32, n: usize) -> Result<u64, CudaError> {
        self.count_mismatches(&self.functions.scale_check, first, n)
    }

    /// How many finite FP16 values convert to units unlike their product by 2^24.
    #[doc(hidden)]
    pub fn units_check(&self, first: u32, n: usize) -> Result<u64, CudaError> {
        self.count_mismatches(&self.functions.units_check, first, n)
    }

    /// Each vector's squared norm, as repro_square_sum sums it.
    #[doc(hidden)]
    pub fn square_sum_check(&self, values: &[f64], h: usize) -> Result<Vec<f64>, CudaError> {
        if h == 0 || !values.len().is_multiple_of(h) {
            return Err(invalid("whole vectors"));
        }
        let count = values.len() / h;
        let _current = self.context.enter()?;
        let input = DeviceBuffer::from_slice(&self.context, values)?;
        let sums = DeviceBuffer::<f64>::zeroed(&self.context, count)?;
        // SAFETY: both buffers outlive the synchronized launch.
        unsafe {
            self.functions.square_sum_check.launch(
                &self.stream,
                [grid(count), 1, 1],
                [32, 1, 1],
                0,
                &mut Args::new()
                    .ptr(&input)
                    .i32(int(h))
                    .i32(int(count))
                    .ptr(&sums),
            )?;
        }
        self.stream.synchronize()?;
        sums.read()
    }

    /// Each pooled mean, as repro_pool_mean forms it from 2^-24 units.
    #[doc(hidden)]
    pub fn pool_mean_check(&self, units: &[i64], lens: &[i32]) -> Result<Vec<f64>, CudaError> {
        if units.len() != lens.len()
            || units.iter().any(|u| u.unsigned_abs() >= 1 << 49)
            || lens.iter().any(|&len| !(1..=int(MAX_KEYS)).contains(&len))
        {
            return Err(invalid("totals below 2^49 units over 1 to MAX_KEYS rows"));
        }
        let _current = self.context.enter()?;
        let totals = DeviceBuffer::from_slice(&self.context, units)?;
        let rows = DeviceBuffer::from_slice(&self.context, lens)?;
        let means = DeviceBuffer::<f64>::zeroed(&self.context, units.len())?;
        // SAFETY: the buffers outlive the synchronized launch.
        unsafe {
            self.functions.pool_mean_check.launch(
                &self.stream,
                [blocks(units.len()), 1, 1],
                [THREADS, 1, 1],
                0,
                &mut Args::new()
                    .ptr(&totals)
                    .ptr(&rows)
                    .i32(int(units.len()))
                    .ptr(&means),
            )?;
        }
        self.stream.synchronize()?;
        means.read()
    }

    /// Each pooled mean over its norm, as repro_pool_quotient forms it.
    #[doc(hidden)]
    pub fn pool_quotient_check(&self, means: &[f64], norms: &[f64]) -> Result<Vec<f32>, CudaError> {
        if means.len() != norms.len() {
            return Err(invalid("a norm for each mean"));
        }
        let _current = self.context.enter()?;
        let numerators = DeviceBuffer::from_slice(&self.context, means)?;
        let denominators = DeviceBuffer::from_slice(&self.context, norms)?;
        let quotients = DeviceBuffer::<f32>::zeroed(&self.context, means.len())?;
        // SAFETY: the buffers outlive the synchronized launch.
        unsafe {
            self.functions.pool_quotient_check.launch(
                &self.stream,
                [blocks(means.len()), 1, 1],
                [THREADS, 1, 1],
                0,
                &mut Args::new()
                    .ptr(&numerators)
                    .ptr(&denominators)
                    .i32(int(means.len()))
                    .ptr(&quotients),
            )?;
        }
        self.stream.synchronize()?;
        quotients.read()
    }

    /// How many a / b, a in [1, 2), b from `first`, differ from correct rounding.
    #[doc(hidden)]
    pub fn quotient_check(&self, first: i32, count: usize) -> Result<u64, CudaError> {
        let _current = self.context.enter()?;
        let mismatches = DeviceBuffer::<u64>::zeroed(&self.context, 1)?;
        // SAFETY: the counter outlives the synchronized launch.
        unsafe {
            self.functions.quotient_check.launch(
                &self.stream,
                [blocks(1 << 23), 1, 1],
                [THREADS, 1, 1],
                0,
                &mut Args::new().i32(first).i32(int(count)).ptr(&mismatches),
            )?;
        }
        self.stream.synchronize()?;
        Ok(mismatches.read()?[0])
    }

    fn count_mismatches(&self, check: &Function, first: u32, n: usize) -> Result<u64, CudaError> {
        let _current = self.context.enter()?;
        let mismatches = DeviceBuffer::<u64>::zeroed(&self.context, 1)?;
        // SAFETY: the counter outlives the synchronized launch.
        unsafe {
            check.launch(
                &self.stream,
                [blocks(n), 1, 1],
                [THREADS, 1, 1],
                0,
                &mut Args::new()
                    .i32(i32::from_ne_bytes(first.to_ne_bytes()))
                    .i32(int(n))
                    .ptr(&mismatches),
            )?;
        }
        self.stream.synchronize()?;
        Ok(mismatches.read()?[0])
    }

    /// # Safety
    /// As [`Self::pool`], with the layer's rows in `pre`.
    unsafe fn layernorm_pool(
        &self,
        s: &Stream,
        norm: &Norm,
        b: &Batch,
        lengths: &[usize],
    ) -> Result<(), CudaError> {
        let h = self.shape.hidden;
        let longest = lengths.iter().copied().max().unwrap_or(0);
        // SAFETY: the caller vouches for the batch; the arguments match the kernel.
        unsafe {
            self.functions.layernorm_pool.launch(
                s,
                [
                    grid(lengths.len()),
                    grid(longest.div_ceil(NORM_BATCH_ROWS)),
                    1,
                ],
                [POOL_THREADS, 1, 1],
                pool_norm_bytes(h, NORM_BATCH_ROWS, false),
                &mut Args::new()
                    .ptr(&b.pre)
                    .ptr(&b.hidden)
                    .ptr(&norm.gamma)
                    .ptr(&norm.beta)
                    .f32(norm.eps)
                    .i32(int(h))
                    .ptr(&b.seq_starts)
                    .ptr(&b.seq_lengths)
                    .ptr(&b.pool_units)
                    .ptr(&b.pool_kinds)
                    .ptr(&b.pool_counts)
                    .raw(b.vectors_ptr),
            )
        }
    }

    /// # Safety
    /// As [`Self::layernorm_pool`], with the segment sums in place of `pre`.
    unsafe fn layernorm_pool_joined(
        &self,
        s: &Stream,
        norm: &Norm,
        down: &Projection,
        b: &Batch,
        lengths: &[usize],
        tokens: usize,
    ) -> Result<(), CudaError> {
        let h = self.shape.hidden;
        let longest = lengths.iter().copied().max().unwrap_or(0);
        // SAFETY: the caller vouches for the batch and the sums.
        unsafe {
            self.functions.layernorm_pool_joined.launch(
                s,
                [
                    grid(lengths.len()),
                    grid(longest.div_ceil(POOL_JOINED_ROWS)),
                    1,
                ],
                [POOL_THREADS, 1, 1],
                pool_norm_bytes(h, POOL_JOINED_ROWS, true),
                &mut Args::new()
                    .ptr(&b.qkv)
                    .i32(int(tokens))
                    .ptr(&down.restored_scales)
                    .ptr(&down.static_shared)
                    .ptr(&b.groups.row_shared)
                    .ptr(&b.hidden)
                    .ptr(&norm.gamma)
                    .ptr(&norm.beta)
                    .f32(norm.eps)
                    .i32(int(h))
                    .ptr(&b.seq_starts)
                    .ptr(&b.seq_lengths)
                    .ptr(&b.pool_units)
                    .ptr(&b.pool_kinds)
                    .ptr(&b.pool_counts)
                    .raw(b.vectors_ptr),
            )
        }
    }

    /// A deferred `next` takes the raw rows times gamma; `clear` also zeroes the next
    /// layer's attention scratch.
    ///
    /// # Safety
    /// `rows` and `out` hold `tokens` rows and outlive the work; `batch` lays
    /// them out on static classes with at least `clear` sequences.
    #[allow(clippy::too_many_arguments)]
    unsafe fn layernorm_quantize(
        &self,
        s: &Stream,
        norm: &Norm,
        rows: &DeviceBuffer<f16>,
        out: &DeviceBuffer<f16>,
        tokens: usize,
        batch: &Batch,
        next: &Projection,
        clear: Option<usize>,
    ) -> Result<(), CudaError> {
        let h = self.shape.hidden;
        assert_eq!(next.k, h);
        let args = Args::new()
            .ptr(rows)
            .ptr(out)
            .ptr(&norm.gamma)
            .ptr(&norm.beta)
            .f32(norm.eps)
            .i32(int(h));
        let moments = next
            .deferred
            .as_ref()
            .map_or(0, |_| batch.work.moments.ptr());
        let tail = |args: Args| {
            args.ptr(&batch.groups.row_shared)
                .ptr(&next.centers)
                .ptr(&next.signs)
                .ptr(&self.constants_f32)
                .ptr(&batch.work.quantized)
                .ptr(&batch.work.block_steps)
                .raw(moments)
                .ptr(&batch.attention.value_magnitudes)
                .ptr(&batch.attention.failed)
                .i32(int(clear.unwrap_or(0)))
                .i32(int(self.shape.heads))
                .i32(int(next.digits))
        };
        // SAFETY: the caller vouches for the rows and the batch.
        unsafe {
            if tokens <= SPLIT_NORM_ROWS {
                self.functions.layernorm_quantize.launch(
                    s,
                    [
                        grid(tokens),
                        grid((h / ROTATION_BLOCK).div_ceil(NORM_SPLIT_WARPS)),
                        1,
                    ],
                    [grid(32 * NORM_SPLIT_WARPS), 1, 1],
                    0,
                    &mut tail(args),
                )
            } else {
                self.functions.layernorm_quantize_batch.launch(
                    s,
                    [grid(tokens.div_ceil(NORM_BATCH_ROWS)), 1, 1],
                    [grid(32 * NORM_BATCH_ROWS), 1, 1],
                    norm_bytes(h, NORM_BATCH_ROWS),
                    &mut tail(args.i32(int(tokens))),
                )
            }
        }
    }

    /// # Safety
    /// `input` holds `rows` rows of `p.k` values; `batch` lays out their groups.
    unsafe fn quantize(
        &self,
        s: &Stream,
        p: &Projection,
        input: &DeviceBuffer<f16>,
        rows: usize,
        batch: &Batch,
    ) -> Result<(), CudaError> {
        let quantizer = &self.functions.quantize_blocks;
        let k = p.k;
        // A row takes k / 32 lanes: whole warps, at least 128 threads.
        assert!(k.is_multiple_of(128) && k <= 4096, "whole rotation blocks");
        let lanes = k / 32;
        let mut threads = lanes;
        while !threads.is_multiple_of(32) {
            threads += lanes;
        }
        while threads < 128 {
            threads *= 2;
        }
        let slots = threads / lanes;
        let threads = grid(threads);
        let resident = self.context.sm_count * quantizer.resident_blocks(threads, 0)?;
        // SAFETY: the caller vouches for the buffers.
        unsafe {
            quantizer.launch(
                s,
                [grid(rows.div_ceil(slots)).min(resident), 1, 1],
                [threads, 1, 1],
                0,
                &mut Args::new()
                    .ptr(input)
                    .ptr(&batch.groups.classes)
                    .ptr(&p.centers)
                    .ptr(&batch.groups.of_row)
                    .ptr(&p.signs)
                    .ptr(&self.constants_f32)
                    .i32(int(k))
                    .i32(int(rows))
                    .ptr(&batch.work.quantized)
                    .ptr(&batch.work.block_steps),
            )
        }
    }

    /// # Safety
    /// The buffers outlive the work and `batch` lays out the rows. Only the
    /// quantizer reads `input`, before the products, so the epilogue may write it.
    unsafe fn project_restored(
        &self,
        s: &Stream,
        p: &Projection,
        input: &DeviceBuffer<f16>,
        rows: usize,
        batch: &Batch,
        finish: Restored,
    ) -> Result<(), CudaError> {
        assert!(
            p.deferred.is_none(),
            "a deferred projection's input stage is its LayerNorm's (layernorm_quantize)"
        );
        // SAFETY: as the caller vouches; the quantizer fills what the products read.
        unsafe {
            self.quantize(s, p, input, rows, batch)?;
            self.restored_products(s, p, batch.work.input(), rows, batch, finish)
        }
    }

    /// Up to HOP_ROWS rows, down runs by segment on the hop, joined in the second LayerNorm.
    fn down_partials(&self, p: &Projection, rows: usize) -> bool {
        p.k == TALL_SEGMENTS * self.shape.hidden && hop_partials(p.n, p.k, rows)
    }

    /// # Safety
    /// `input` holds `rows` rows of `p.k`; `batch.qkv` holds TALL_SEGMENTS
    /// planes of rows by n.
    unsafe fn segment_partials(
        &self,
        s: &Stream,
        p: &Projection,
        input: Quantized,
        rows: usize,
        batch: &Batch,
    ) -> Result<(), CudaError> {
        assert!(
            hop_partials(p.n, p.k, rows),
            "the hop takes the segments (down_partials)"
        );
        let kernel = &self.functions.restored.hop_partials;
        // SAFETY: the caller vouches for the buffers.
        unsafe { kernel.launch(s, p, input, rows, batch) }
    }

    /// Past the blocks kernel's row range, strips share the down projection's hop layout.
    fn up_strips(&self, layer: &Layer, rows: usize) -> Option<&HopKernel> {
        let up = &layer.up;
        self.functions.restored.hop_sums.as_ref().filter(|_| {
            up.digits == 1
                && up.k == HOP_DEPTH
                && up.n == TALL_SEGMENTS * HOP_COLUMNS
                && self.down_partials(&layer.down, rows)
        })
    }

    /// The up projection's sums through GELU into the down projection's operand.
    ///
    /// # Safety
    /// `batch.qkv` holds the up projection's strip sums; the buffers outlive the work.
    unsafe fn gelu_half_blocks(
        &self,
        s: &Stream,
        up: &Projection,
        down: &Projection,
        rows: usize,
        batch: &Batch,
    ) -> Result<(), CudaError> {
        // A block a pass: GELU_PASS_WARPS 128-blocks of GELU_PASS_ROWS rows.
        let columns = up.n.div_ceil(ROTATION_BLOCK * GELU_PASS_WARPS);
        let passes = rows.div_ceil(GELU_PASS_ROWS);
        let (work, made) = (&batch.work, batch.work.next_input());
        let args = Args::new()
            .ptr(&batch.qkv)
            .ptr(&up.restored_scales)
            .ptr(&up.static_shared)
            .ptr(&batch.groups.row_shared);
        let args = match &up.deferred {
            Some(d) => args.ptr(&work.moments).ptr(&d.sums).ptr(&d.constants),
            None => args.raw(0).raw(0).raw(0),
        };
        // SAFETY: the caller vouches for the buffers; Work fits the rows.
        unsafe {
            self.functions.gelu_half_blocks.launch(
                s,
                [grid(columns), grid(passes), 1],
                [grid(32 * GELU_PASS_WARPS), 1, 1],
                0,
                &mut args
                    .ptr(&self.constants_f32)
                    .i32(int(rows))
                    .i32(int(up.n))
                    .ptr(&down.centers)
                    .ptr(&down.signs)
                    .ptr(made.rows)
                    .ptr(made.steps),
            )
        }
    }

    /// Joins the segment sums in order as down's residual epilogue would, then normalizes.
    ///
    /// # Safety
    /// As layernorm_quantize, with segment_partials' sums in `batch.qkv`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn layernorm_quantize_joined(
        &self,
        s: &Stream,
        norm: &Norm,
        down: &Projection,
        stream: &DeviceBuffer<f16>,
        tokens: usize,
        batch: &Batch,
        next: &Projection,
        clear: Option<usize>,
    ) -> Result<(), CudaError> {
        let h = self.shape.hidden;
        assert!(
            h <= JOINED_NORM_WARPS * ROTATION_BLOCK && next.k == h,
            "row in one block"
        );
        let moments = next
            .deferred
            .as_ref()
            .map_or(0, |_| batch.work.moments.ptr());
        // SAFETY: the caller vouches for the sums, the rows and the batch.
        unsafe {
            self.functions.layernorm_quantize_joined.launch(
                s,
                [grid(tokens), 1, 1],
                [grid(32 * JOINED_NORM_WARPS), 1, 1],
                0,
                &mut Args::new()
                    .ptr(&batch.qkv)
                    .i32(int(tokens))
                    .ptr(&down.restored_scales)
                    .ptr(&down.static_shared)
                    .ptr(stream)
                    .ptr(&norm.gamma)
                    .ptr(&norm.beta)
                    .f32(norm.eps)
                    .i32(int(h))
                    .ptr(&batch.groups.row_shared)
                    .ptr(&next.centers)
                    .ptr(&next.signs)
                    .ptr(&self.constants_f32)
                    .ptr(&batch.work.quantized)
                    .ptr(&batch.work.block_steps)
                    .raw(moments)
                    .ptr(&batch.attention.value_magnitudes)
                    .ptr(&batch.attention.failed)
                    .i32(int(clear.unwrap_or(0)))
                    .i32(int(self.shape.heads))
                    .i32(int(next.digits)),
            )
        }
    }

    /// Past TALLER_ROWS rows with every multiprocessor filled: an idle one loses to tall tiles.
    fn restored_wide(&self, n: usize, k: usize, rows: usize) -> bool {
        self.restored_operands(n, k)
            && rows > TALLER_ROWS
            && n / RESTORED_WIDE_COLUMNS * rows.div_ceil(RESTORED_WIDE_ROWS)
                >= self.context.sm_count as usize
    }

    /// Up to TALLER_ROWS rows, where tall tiles' groups would hand one chain across barriers.
    fn restored_small(&self, n: usize, k: usize, rows: usize) -> bool {
        k == self.shape.hidden && rows <= TALLER_ROWS && n.is_multiple_of(RESTORED_SMALL_COLUMNS)
    }

    /// Three stages where they take fewer rounds than four, which keep one more stage in flight.
    fn gelu_blocks_depth(&self, n: usize, rows: usize) -> usize {
        let sms = self.context.sm_count as usize;
        let busiest = (n / ROTATION_BLOCK * rows.div_ceil(RESTORED_BLOCKS_ROWS)).div_ceil(sms);
        let rounds = |(_, _, resident): (&str, u32, usize)| busiest.div_ceil(resident);
        usize::from(rounds(RESTORED_BLOCKS_DEPTHS[1]) < rounds(RESTORED_BLOCKS_DEPTHS[0]))
    }

    /// A QKV tile must hold whole heads within one of Q, K and V.
    fn wide_64x128(&self, n: usize, rows: usize) -> bool {
        let (hidden, heads) = (self.shape.hidden, self.shape.heads);
        rows > TALLER_ROWS
            && n.is_multiple_of(RESTORED_WIDE_64X128_COLUMNS)
            && RESTORED_WIDE_64X128_COLUMNS.is_multiple_of(hidden / heads)
            && wide_64x128_fills(hidden, n, rows, self.context.sm_count as usize)
    }

    /// When the 128 x 64 grid leaves a multiprocessor a lone tile and the 64 x 64 grid fits a wave.
    fn restored_wide64(&self, n: usize, rows: usize) -> bool {
        let sms = self.context.sm_count as usize;
        n / RESTORED_WIDE_COLUMNS * rows.div_ceil(RESTORED_WIDE_ROWS) < 2 * sms
            && n / RESTORED_WIDE_DOWN_COLUMNS * rows.div_ceil(RESTORED_WIDE_DOWN_ROWS) <= 3 * sms
    }

    /// When the busiest multiprocessor's rounds take fewer units of work.
    fn restored_gelu_wide64(&self, n: usize, rows: usize) -> bool {
        let sms = self.context.sm_count as usize;
        let busiest = |tiles: usize| tiles.div_ceil(sms);
        let wide = busiest(n / RESTORED_WIDE_COLUMNS * rows.div_ceil(RESTORED_WIDE_ROWS));
        let narrow =
            busiest(n / RESTORED_WIDE_DOWN_COLUMNS * rows.div_ceil(RESTORED_WIDE_DOWN_ROWS));
        narrow.div_ceil(3) * 3 < wide.div_ceil(2) * 4
    }

    /// The prep kernel QKV replaces costs more than wide tiles lose.
    fn restored_operands(&self, n: usize, k: usize) -> bool {
        k == self.shape.hidden && n.is_multiple_of(RESTORED_WIDE_COLUMNS)
    }

    fn qkv_packed_tile(&self, rows: usize) -> Option<usize> {
        let d = self.shape.hidden / self.shape.heads;
        QKV_PACKED_TILES
            .iter()
            .position(|tile| tile.columns == d && rows <= tile.rows)
    }

    /// Also past ROW_PACK_KEYS rows, where the prep kernel outgrows the wide tiles' cost.
    fn qkv_operands(&self, n: usize, k: usize, rows: usize) -> bool {
        self.restored_operands(n, k)
            || (k == self.shape.hidden
                && rows > ROW_PACK_KEYS
                && n.is_multiple_of(RESTORED_WIDE_COLUMNS))
    }

    /// # Safety
    /// `input` holds `rows` rows of `p.k`, with their statistics when `p` is
    /// deferred; the buffers outlive the work.
    unsafe fn restored_products(
        &self,
        s: &Stream,
        p: &Projection,
        input: Quantized,
        rows: usize,
        batch: &Batch,
        finish: Restored,
    ) -> Result<(), CudaError> {
        let (n, k, work) = (p.n, p.k, &batch.work);
        // Restored chains restart every h inputs: one for a projection of depth h, four for the
        // down projection's 4h; a two-digit operand's one chain is twice as deep (`_two` kernels).
        let two = p.digits == 2;
        let segment = p.digits * self.shape.hidden;
        let (kernels, sms) = (&self.functions.restored, self.context.sm_count as usize);
        assert!(n.is_multiple_of(32));
        let groups = fused_groups(n, k, sms);
        assert!(
            segment.is_multiple_of(64)
                && segment / 64 >= groups.max(2)
                && if two {
                    segment == 2 * k && n.is_multiple_of(RESTORED_WIDE_64X128_COLUMNS)
                } else {
                    k.is_multiple_of(segment)
                },
            "a restored chain restarts on whole 64-deep tiles dividing the depth, at least a round's; \
             two digits chain through one on the wide tiles"
        );
        let narrow = GROUPINGS
            .iter()
            .position(|&g| g == groups)
            .expect("has kernels");
        let leading = Args::new()
            .ptr(input.rows)
            .ptr(input.steps)
            .ptr(&p.q)
            .ptr(&p.restored_scales)
            .ptr(&p.static_shared)
            .ptr(&batch.groups.row_shared);
        // The kernels' last arguments (repro.cu DeferredTables).
        let deferred = |args: Args| match &p.deferred {
            Some(d) => args.ptr(&work.moments).ptr(&d.sums).ptr(&d.constants),
            None => args.raw(0).raw(0).raw(0),
        };
        let shape = |args: Args| {
            args.ptr(&self.constants_f32)
                .i32(int(rows))
                .i32(int(n))
                .i32(int(k))
                .i32(int(segment))
        };
        // Indices into repro.cu's tables: REPRO_PLAIN, REPRO_RESIDUAL, REPRO_GELU.
        let (epilogue, residual, out) = match finish {
            Restored::Rows(out) => (0, 0, out.ptr()),
            Restored::Residual { residual, out } => (1, residual.ptr(), out.ptr()),
            Restored::Gelu(out) => (2, 0, out.ptr()),
            Restored::Attention { key_centers } => {
                assert!(
                    self.qkv_operands(n, k, rows),
                    "QKV makes attention's operands on the wide tiles"
                );
                let (scratch, heads) = (&batch.attention, self.shape.heads);
                let (one, twice, tile_rows, tile_columns, bytes) = if self.wide_64x128(n, rows) {
                    let (r, c) = (RESTORED_WIDE_64X128_ROWS, RESTORED_WIDE_64X128_COLUMNS);
                    let (one, twice) = (&kernels.wide_qkv_64x128, &kernels.two.qkv_wide_64x128);
                    (one, twice, r, c, RESTORED_WIDE_64X128_BYTES)
                } else if self.restored_wide64(n, rows) {
                    let (r, c) = (RESTORED_WIDE_DOWN_ROWS, RESTORED_WIDE_DOWN_COLUMNS);
                    let (one, twice) = (&kernels.wide_qkv64, &kernels.two.qkv_wide64);
                    (one, twice, r, c, RESTORED_WIDE_DOWN_BYTES)
                } else {
                    let (r, c) = (RESTORED_WIDE_ROWS, RESTORED_WIDE_COLUMNS);
                    let (one, twice) = (&kernels.wide_qkv, &kernels.two.qkv_wide);
                    (one, twice, r, c, RESTORED_WIDE_BYTES)
                };
                let kernel = if two { twice } else { one };
                // SAFETY: the caller vouches for the buffers and cleared the flags.
                return unsafe {
                    kernel.launch(
                        s,
                        [grid(n / tile_columns), grid(rows.div_ceil(tile_rows)), 1],
                        [128, 1, 1],
                        bytes,
                        &mut deferred(shape(leading.raw(0)).raw(0))
                            .ptr(&scratch.queries)
                            .ptr(&scratch.query_scales)
                            .ptr(key_centers)
                            .ptr(&scratch.keys)
                            .ptr(&scratch.key_scales)
                            .ptr(&scratch.values16)
                            .ptr(&scratch.value_magnitudes)
                            .ptr(&scratch.failed)
                            .ptr(&scratch.sequence_codes)
                            .i32(int(heads))
                            .i32(int(self.shape.hidden / heads)),
                    )
                };
            }
            Restored::Packed {
                tile_index,
                key_centers,
                sequences,
                ptokens,
            } => {
                let tile = &QKV_PACKED_TILES[tile_index];
                let (scratch, heads) = (&batch.attention, self.shape.heads);
                let kernel = if two {
                    &kernels.two.qkv_packed[tile_index]
                } else {
                    &kernels.qkv_packed[tile_index]
                };
                // SAFETY: the caller vouches for the buffers and cleared the
                // flags; one tile row holds every row.
                return unsafe {
                    kernel.launch(
                        s,
                        [grid(n / tile.columns), 1, 1],
                        [tile.threads, 1, 1],
                        tile.bytes,
                        &mut deferred(shape(leading.raw(0)).raw(0))
                            .ptr(&scratch.queries)
                            .ptr(&scratch.query_scales)
                            .ptr(key_centers)
                            .ptr(&scratch.keys)
                            .ptr(&scratch.key_scales)
                            .ptr(&scratch.values16)
                            .ptr(&scratch.failed)
                            .ptr(&scratch.sequence_codes)
                            .i32(int(heads))
                            .i32(int(self.shape.hidden / heads))
                            .ptr(&scratch.value_planes)
                            .ptr(&scratch.value_scales)
                            .ptr(&batch.seq_starts)
                            .ptr(&batch.seq_lengths)
                            .ptr(&batch.pstarts)
                            .i32(int(sequences))
                            .i32(int(ptokens)),
                    )
                };
            }
            Restored::GeluNext { out, next } => {
                assert!(
                    !two && rows <= FUSED_NARROW_ROWS
                        && n.is_multiple_of(ROTATION_BLOCK)
                        && next.k == n,
                    "a narrow up projection of one digit makes its down projection's input by whole 128-blocks"
                );
                let made = work.next_input();
                // SAFETY: the caller vouches for the buffers; Work fits the rows.
                return unsafe {
                    kernels.gelu_next[narrow].launch(
                        s,
                        [grid(n / 32), 1, 1],
                        [grid(128 * groups), 1, 1],
                        restored_next_bytes(groups),
                        &mut deferred(
                            shape(leading)
                                .ptr(out)
                                .ptr(&next.centers)
                                .ptr(&next.signs)
                                .ptr(made.rows)
                                .ptr(made.steps)
                                .ptr(&work.next_done),
                        ),
                    )
                };
            }
            Restored::GeluBlocks { next } => {
                assert!(
                    segment == p.digits * k && n.is_multiple_of(ROTATION_BLOCK) && next.k == n,
                    "a one-segment up projection makes its down projection's input by whole 128-blocks"
                );
                let made = work.next_input();
                let depth = self.gelu_blocks_depth(n, rows);
                let kernel = if two {
                    &kernels.two.gelu_blocks[depth]
                } else {
                    &kernels.gelu_blocks[depth]
                };
                // SAFETY: the caller vouches for the buffers; Work fits the rows.
                return unsafe {
                    kernel.launch(
                        s,
                        [
                            grid(n / ROTATION_BLOCK),
                            grid(rows.div_ceil(RESTORED_BLOCKS_ROWS)),
                            1,
                        ],
                        [RESTORED_BLOCKS_THREADS, 1, 1],
                        RESTORED_BLOCKS_DEPTHS[depth].1,
                        &mut deferred(shape(leading.raw(0)).raw(0))
                            .ptr(&next.centers)
                            .ptr(&next.signs)
                            .ptr(made.rows)
                            .ptr(made.steps),
                    )
                };
            }
        };
        let mut args = deferred(shape(leading.raw(residual)).raw(out));
        let tall_tiles = n / 32 * rows.div_ceil(TALL_ROWS);
        // SAFETY: the caller vouches for the buffers; `Work` holds every row.
        unsafe {
            if two {
                // QKV's rows or the up projection's GELU: past TALLER_ROWS rows the one-digit
                // choice among the wide tiles, else 64 x 64 tiles, the smallest that walk twice.
                let many = rows > TALLER_ROWS;
                let wide = (
                    RESTORED_WIDE_ROWS,
                    RESTORED_WIDE_COLUMNS,
                    RESTORED_WIDE_BYTES,
                );
                let wide64 = (
                    RESTORED_WIDE_DOWN_ROWS,
                    RESTORED_WIDE_DOWN_COLUMNS,
                    RESTORED_WIDE_DOWN_BYTES,
                );
                let (kernel, (tile_rows, tile_columns, bytes)) = match epilogue {
                    0 if many => (&kernels.two.plain_wide, wide),
                    0 => (&kernels.two.plain_wide64, wide64),
                    2 if many && self.wide_64x128(n, rows) => (
                        &kernels.two.gelu_wide_64x128,
                        (
                            RESTORED_WIDE_64X128_ROWS,
                            RESTORED_WIDE_64X128_COLUMNS,
                            RESTORED_WIDE_64X128_BYTES,
                        ),
                    ),
                    2 if many && !self.restored_gelu_wide64(n, rows) => {
                        (&kernels.two.gelu_wide, wide)
                    }
                    2 => (&kernels.two.gelu_wide64, wide64),
                    _ => unreachable!("two digits are QKV's or the up projection's operand"),
                };
                kernel.launch(
                    s,
                    [grid(n / tile_columns), grid(rows.div_ceil(tile_rows)), 1],
                    [128, 1, 1],
                    bytes,
                    &mut args,
                )
            } else if rows <= FUSED_NARROW_ROWS {
                kernels.narrow[narrow][epilogue].launch(
                    s,
                    [grid(n / 32), 1, 1],
                    [grid(128 * groups), 1, 1],
                    restored_bytes(groups, k),
                    &mut args,
                )
            } else if k == TALL_SEGMENTS * segment {
                // A group per restarted segment; taller tiles when tall ones
                // would not all be resident at once.
                assert_eq!(
                    epilogue, 1,
                    "the restarted projection takes the residual epilogue"
                );
                let wide_down_tiles =
                    n / RESTORED_WIDE_DOWN_COLUMNS * rows.div_ceil(RESTORED_WIDE_DOWN_ROWS);
                if self.wide_64x128(n, rows) {
                    kernels.wide_segments_64x128.launch(
                        s,
                        [
                            grid(n / RESTORED_WIDE_64X128_COLUMNS),
                            grid(rows.div_ceil(RESTORED_WIDE_64X128_ROWS)),
                            1,
                        ],
                        [128, 1, 1],
                        RESTORED_WIDE_64X128_BYTES,
                        &mut args,
                    )
                } else if rows > TALLER_ROWS
                    && n.is_multiple_of(RESTORED_WIDE_DOWN_COLUMNS)
                    && wide_down_tiles >= sms
                {
                    // Many rows: 64 x 64 tiles, the segments in sequence.
                    kernels.wide_segments.launch(
                        s,
                        [
                            grid(n / RESTORED_WIDE_DOWN_COLUMNS),
                            grid(rows.div_ceil(RESTORED_WIDE_DOWN_ROWS)),
                            1,
                        ],
                        [128, 1, 1],
                        RESTORED_WIDE_DOWN_BYTES,
                        &mut args,
                    )
                } else if rows > TALL_ROWS && rows <= TALLER_ROWS && tall_tiles > sms {
                    kernels.tall_segments_m2.launch(
                        s,
                        [grid(n / 32), 1, 1],
                        [grid(128 * TALL_SEGMENTS), 1, 1],
                        tall_m2_segment_bytes(),
                        &mut args,
                    )
                } else {
                    kernels.tall_segments.launch(
                        s,
                        [grid(n / 32), grid(rows.div_ceil(TALL_ROWS)), 1],
                        [grid(128 * TALL_SEGMENTS), 1, 1],
                        tall_segment_bytes(),
                        &mut args,
                    )
                }
            } else if epilogue == 1 && self.restored_small(n, k, rows) {
                // Few rows of the out projection: 32 x 32 tiles, a chain a warp.
                kernels.residual_small.launch(
                    s,
                    [
                        grid(n / RESTORED_SMALL_COLUMNS),
                        grid(rows.div_ceil(RESTORED_SMALL_ROWS)),
                        1,
                    ],
                    [128, 1, 1],
                    RESTORED_SMALL_BYTES,
                    &mut args,
                )
            } else if self.restored_wide(n, k, rows) && epilogue != 0 && self.wide_64x128(n, rows) {
                let kernel = if epilogue == 1 {
                    &kernels.wide_residual_64x128
                } else {
                    &kernels.wide_gelu_64x128
                };
                kernel.launch(
                    s,
                    [
                        grid(n / RESTORED_WIDE_64X128_COLUMNS),
                        grid(rows.div_ceil(RESTORED_WIDE_64X128_ROWS)),
                        1,
                    ],
                    [128, 1, 1],
                    RESTORED_WIDE_64X128_BYTES,
                    &mut args,
                )
            } else if self.restored_wide(n, k, rows) {
                // Many rows: 128 x 64 tiles, or 64 x 64 where those would leave
                // lone tiles or take more rounds.
                let narrow64 = match epilogue {
                    1 => self
                        .restored_wide64(n, rows)
                        .then_some(&kernels.wide_residual64),
                    2 => self
                        .restored_gelu_wide64(n, rows)
                        .then_some(&kernels.wide_gelu64),
                    _ => None,
                };
                if let Some(kernel) = narrow64 {
                    kernel.launch(
                        s,
                        [
                            grid(n / RESTORED_WIDE_DOWN_COLUMNS),
                            grid(rows.div_ceil(RESTORED_WIDE_DOWN_ROWS)),
                            1,
                        ],
                        [128, 1, 1],
                        RESTORED_WIDE_DOWN_BYTES,
                        &mut args,
                    )
                } else {
                    kernels.wide[epilogue].launch(
                        s,
                        [
                            grid(n / RESTORED_WIDE_COLUMNS),
                            grid(rows.div_ceil(RESTORED_WIDE_ROWS)),
                            1,
                        ],
                        [128, 1, 1],
                        RESTORED_WIDE_BYTES,
                        &mut args,
                    )
                }
            } else if rows > TALL_ROWS
                && rows <= TALLER_ROWS
                && tall_tiles > TALL_MOST_RESIDENT * sms
            {
                // Taller tiles when tall ones would not all be resident at once.
                let at = tall_w8_config(n / 32, sms);
                kernels.tall_w8[at][epilogue].launch(
                    s,
                    [grid(n / 32), 1, 1],
                    [256, 1, 1],
                    tall_w8_bytes(TALL_W8_CONFIGS[at]),
                    &mut args,
                )
            } else {
                let row_tiles = rows.div_ceil(TALL_ROWS);
                let at = tall_config(tall_tiles, sms);
                let config @ (groups, _) = TALL_CONFIGS[at];
                kernels.tall[at][epilogue].launch(
                    s,
                    [grid(n / 32), grid(row_tiles), 1],
                    [grid(128 * groups), 1, 1],
                    tall_bytes(config) + tall_steps_bytes(k),
                    &mut args,
                )
            }
        }
    }
}

/// Integer attention's scratch for one batch (repro.cu ReproAttentionPrep).
struct Attention {
    value_scales: DeviceBuffer<f32>,
    key_scales: DeviceBuffer<f32>,
    query_scales: DeviceBuffer<f32>,
    /// Each query row's largest score per key split.
    row_max: DeviceBuffer<f32>,
    keys: DeviceBuffer<i8>,
    value_planes: DeviceBuffer<i8>,
    queries: DeviceBuffer<i8>,
    values16: DeviceBuffer<f16>,
    value_magnitudes: DeviceBuffer<u32>,
    failed: DeviceBuffer<i32>,
    sequence_codes: DeviceBuffer<i32>,
}
