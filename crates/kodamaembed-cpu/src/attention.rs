//! FP32 and centered INT8 attention; a task holds one sequence and head, so rows never mix.

use crate::{
    gemm::{self, Kernel, NR, Panels},
    math,
    workers::POOL,
};
use kodamaembed_model::relative::RelativeBias;
use rayon::prelude::*;
use std::cell::RefCell;

#[cfg(target_arch = "x86_64")]
mod integer;
#[cfg(target_arch = "x86_64")]
pub(crate) use integer::{weight_indices_of, weight_table};

/// Query rows per block, a multiple of the microkernel's six.
const QUERY_BLOCK: usize = 48;

/// One head's packed keys and values, shared by its query blocks.
#[derive(Default)]
struct Head {
    keys: Vec<f32>,
    values: Vec<f32>,
}

/// One query block's buffers, kept across blocks and batches.
#[derive(Default)]
struct Block {
    scores: Vec<f32>,
    block: Vec<f32>,
    /// The values a windowed block's rows reach, packed for it alone.
    values: Vec<f32>,
}

thread_local! {
    /// Taken during use: a thread waiting on its head's blocks may run another head.
    static HEADS: RefCell<Head> = RefCell::default();
    static BLOCKS: RefCell<Block> = RefCell::default();
}

#[cfg(target_arch = "x86_64")]
thread_local! {
    /// Taken during use, as above.
    static INTEGER_HEADS: RefCell<integer::Head> = RefCell::default();
    static INTEGER_BLOCKS: RefCell<integer::Block> = RefCell::default();
}

struct Task {
    start: usize,
    len: usize,
    head: usize,
}

/// Tasks write disjoint heads and rows of the context at once.
#[derive(Clone, Copy)]
struct Context {
    values: *mut f32,
    len: usize,
    hidden: usize,
}

// SAFETY: tasks write disjoint elements, and nothing else reads until they end.
unsafe impl Sync for Context {}

impl Context {
    /// # Safety
    /// No other thread may access those elements meanwhile.
    unsafe fn write(self, token: usize, column: usize, values: &[f32], sum: f32) {
        let at = token * self.hidden + column;
        assert!(column + values.len() <= self.hidden && at + values.len() <= self.len);
        // SAFETY: in bounds (asserted) and the caller's alone.
        let row = unsafe { std::slice::from_raw_parts_mut(self.values.add(at), values.len()) };
        for (x, &value) in row.iter_mut().zip(values) {
            *x = value / sum;
        }
    }

    /// # Safety
    /// No other thread may access those elements meanwhile.
    #[cfg(target_arch = "x86_64")]
    unsafe fn store(self, token: usize, column: usize, values: &[f32]) {
        let at = token * self.hidden + column;
        assert!(column + values.len() <= self.hidden && at + values.len() <= self.len);
        // SAFETY: in bounds (asserted) and the caller's alone.
        unsafe { std::slice::from_raw_parts_mut(self.values.add(at), values.len()) }
            .copy_from_slice(values);
    }
}

/// softmax(Q K^T / sqrt(d) + bias) V per sequence and head, from qkv rows holding
/// Q, K and V side by side.
pub fn attention(
    kernel: Kernel,
    qkv: &[f32],
    lengths: &[usize],
    hidden: usize,
    heads: usize,
    bias: Option<&RelativeBias>,
    context: &mut [f32],
) {
    assert!(
        kernel.is_supported(),
        "kernel {kernel} is unavailable on this processor"
    );
    assert!(
        hidden > 0 && heads > 0 && hidden.is_multiple_of(heads),
        "hidden must be positive and split evenly into positive heads"
    );
    let tokens = lengths
        .iter()
        .try_fold(0usize, |total, &len| total.checked_add(len))
        .expect("lengths overflow");
    let context_len = tokens.checked_mul(hidden).expect("dimensions overflow");
    let qkv_len = context_len.checked_mul(3).expect("dimensions overflow");
    assert!(qkv.len() >= qkv_len && context.len() >= context_len);
    if tokens == 0 {
        return;
    }
    POOL.install(|| {
        attention_current_pool(
            kernel, qkv, lengths, hidden, heads, false, None, bias, None, context,
        )
    });
}

/// Needs AVX-VNNI, AVX2, FMA and F16C, at most 512 tokens a sequence and head
/// widths a multiple of 16.
#[doc(hidden)]
#[cfg(target_arch = "x86_64")]
pub fn integer_attention(
    qkv: &[f32],
    lengths: &[usize],
    hidden: usize,
    heads: usize,
    key_centers: Option<&[f32]>,
    context: &mut [f32],
) {
    POOL.install(|| {
        attention_current_pool(
            Kernel::Avx2,
            qkv,
            lengths,
            hidden,
            heads,
            true,
            key_centers,
            None,
            None,
            context,
        )
    });
}

/// Runs in the caller's Rayon pool; encoder callers install it once. With `reach`, a query
/// attends only that many keys each side of itself.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_current_pool(
    kernel: Kernel,
    qkv: &[f32],
    lengths: &[usize],
    hidden: usize,
    heads: usize,
    integer: bool,
    key_centers: Option<&[f32]>,
    bias: Option<&RelativeBias>,
    reach: Option<usize>,
    context: &mut [f32],
) {
    assert!(
        kernel.is_supported(),
        "kernel {kernel} is unavailable on this processor"
    );
    assert!(
        hidden > 0 && heads > 0 && hidden.is_multiple_of(heads),
        "hidden must be positive and split evenly into positive heads"
    );
    assert!(
        key_centers.is_none_or(|centers| integer && centers.len() == hidden),
        "static key centers belong to integer attention, one per column"
    );
    assert!(
        !(integer && (bias.is_some() || reach.is_some())),
        "unbiased integer attention over whole sequences"
    );
    let tokens = lengths
        .iter()
        .try_fold(0usize, |total, &len| total.checked_add(len))
        .expect("lengths overflow");
    let context_len = tokens.checked_mul(hidden).expect("dimensions overflow");
    let qkv_len = context_len.checked_mul(3).expect("dimensions overflow");
    assert!(qkv.len() >= qkv_len && context.len() >= context_len);
    if tokens == 0 {
        return;
    }
    let mut tasks = Vec::with_capacity(lengths.len() * heads);
    let mut start = 0;
    for &len in lengths {
        if len > 0 {
            tasks.extend((0..heads).map(|head| Task { start, len, head }));
        }
        start += len;
    }
    let context = Context {
        values: context.as_mut_ptr(),
        len: context_len,
        hidden,
    };
    // A partial round of heads idles workers unless its heads' query blocks spread over them.
    // Past two rounds that idle share is small, and in whole rounds there is none.
    let threads = rayon::current_num_threads();
    let spread = tasks.len() < 2 * threads && !tasks.len().is_multiple_of(threads);

    if integer {
        #[cfg(target_arch = "x86_64")]
        {
            assert!(
                crate::int8::is_supported(),
                "integer attention requires AVX-VNNI, AVX2, FMA and F16C"
            );
            let (d, stride) = (hidden / heads, 3 * hidden);
            assert!(
                crate::int8::validate_depth(d).is_ok(),
                "integer attention head width exceeds INT32 range"
            );
            tasks.into_par_iter().for_each(|Task { start, len, head }| {
                let rows = &qkv[start * stride..(start + len) * stride];
                let column = head * d;
                let center = key_centers.map(|centers| &centers[column..][..d]);
                let mut prepared = INTEGER_HEADS.take();
                // SAFETY: AVX-VNNI, AVX2, FMA and F16C are asserted above.
                let ready =
                    unsafe { prepared.prepare(rows, stride, hidden, column, len, d, center) };
                let attend_block = |block| {
                    INTEGER_BLOCKS.with_borrow_mut(|scratch| {
                        // SAFETY: as above; only this block writes its rows of the head.
                        unsafe {
                            prepared.attend(
                                &rows[column..],
                                stride,
                                block,
                                scratch,
                                |row, values| context.store(start + row, column, values),
                            )
                        }
                    })
                };
                if !ready {
                    let nan = vec![f32::NAN; d];
                    for row in 0..len {
                        // SAFETY: only this task writes its head and rows.
                        unsafe { context.store(start + row, column, &nan) };
                    }
                } else if spread {
                    (0..prepared.blocks())
                        .into_par_iter()
                        .for_each(attend_block);
                } else {
                    (0..prepared.blocks()).for_each(attend_block);
                }
                INTEGER_HEADS.set(prepared);
            });
            return;
        }
        #[cfg(not(target_arch = "x86_64"))]
        panic!("integer attention requires x86-64");
    }
    tasks
        .into_par_iter()
        .for_each(|task| attend(kernel, qkv, heads, bias, reach, task, context, spread));
}

#[allow(clippy::too_many_arguments)]
fn attend(
    kernel: Kernel,
    qkv: &[f32],
    heads: usize,
    bias: Option<&RelativeBias>,
    reach: Option<usize>,
    task: Task,
    out: Context,
    spread: bool,
) {
    let Task { start, len, head } = task;
    let hidden = out.hidden;
    let (d, stride) = (hidden / heads, 3 * hidden);
    let column = head * d;
    let sequence = &qkv[start * stride..(start + len) * stride];
    let mut packed = HEADS.take();
    packed.keys.resize(gemm::panels_len(len, d), 0.0);
    gemm::pack_rows(
        &sequence[hidden + column..],
        stride,
        len,
        d,
        &mut packed.keys,
    );
    // Without a window every block reads the sequence's values, packed once.
    let windowed = reach.is_some();
    if !windowed {
        packed.values.resize(gemm::panels_len(d, len), 0.0);
        gemm::pack_columns(
            &sequence[2 * hidden + column..],
            stride,
            d,
            len,
            &mut packed.values,
        );
    }
    // No window is one that reaches every key.
    let reach = reach.unwrap_or(len);
    let scale = 1.0 / (d as f32).sqrt();
    let attend_block = |block: usize| {
        BLOCKS.with_borrow_mut(|s| {
            let first = block * QUERY_BLOCK;
            let rows = QUERY_BLOCK.min(len - first);
            let mut sums = [0.0f32; QUERY_BLOCK];
            // The keys the block's rows reach, in whole panels, so a window's work does not
            // grow with the sequence; the panels' padding keys are masked out of the softmax.
            let from = first.saturating_sub(reach) / NR * NR;
            let to = (first + rows + reach).min(len).next_multiple_of(NR);
            let span = to - from;
            let keys = Panels {
                values: &packed.keys[from * d..to * d],
                n: span,
                k: d,
            };
            s.scores.resize(rows * span, 0.0);
            s.block.resize(rows * d, 0.0);
            let queries = &sequence[first * stride + column..];
            gemm::product(kernel, queries, stride, rows, keys, &mut s.scores, span);
            let scored = s.scores.chunks_exact_mut(span).zip(&mut sums[..rows]);
            for (query, (row, sum)) in (first..).zip(scored) {
                let seen = query.saturating_sub(reach)..(query + reach + 1).min(len);
                row[..seen.start - from].fill(f32::NEG_INFINITY);
                row[seen.end - from..].fill(f32::NEG_INFINITY);
                // The bias joins scaled scores, so a biased row is scaled before softmax.
                let softmax_scale = match bias {
                    Some(bias) => {
                        for (score, &term) in row.iter_mut().zip(bias.keys(head, query, len)) {
                            *score = *score * scale + term;
                        }
                        1.0
                    }
                    None => scale,
                };
                *sum = match kernel {
                    #[cfg(target_arch = "x86_64")]
                    // SAFETY: Avx2 implies AVX2 and FMA.
                    Kernel::Avx2 => unsafe { math::avx2::softmax(row, softmax_scale) },
                    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                    // SAFETY: Neon implies NEON.
                    Kernel::Neon => unsafe { math::neon::softmax(row, softmax_scale) },
                    Kernel::Portable => math::softmax(row, softmax_scale),
                };
            }
            let count = to.min(len) - from;
            let values = if windowed {
                s.values.resize(gemm::panels_len(d, count), 0.0);
                gemm::pack_columns(
                    &sequence[from * stride + 2 * hidden + column..],
                    stride,
                    d,
                    count,
                    &mut s.values,
                );
                &s.values
            } else {
                &packed.values
            };
            let values = Panels {
                values,
                n: d,
                k: count,
            };
            gemm::product(kernel, &s.scores, span, rows, values, &mut s.block, d);
            for (r, (row, &sum)) in s.block.chunks_exact(d).zip(&sums[..rows]).enumerate() {
                // SAFETY: only this block writes its rows of the head.
                unsafe { out.write(start + first + r, column, row, sum) };
            }
        })
    };
    let blocks = len.div_ceil(QUERY_BLOCK);
    if spread {
        (0..blocks).into_par_iter().for_each(attend_block);
    } else {
        (0..blocks).for_each(attend_block);
    }
    HEADS.set(packed);
}
