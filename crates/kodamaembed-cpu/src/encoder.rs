//! CPU transformer encoder for packed sequences.

#[cfg(target_arch = "x86_64")]
use crate::int8::{DeferredNorm, Operand};
use crate::{
    CpuPrecision,
    attention::attention_current_pool,
    gemm::{Epilogue, Kernel, NR, PackedWeight, gemm_current_pool},
    math,
    projection::Projection,
    workers::{POOL, SHORT_POOL, Workers},
};
#[cfg(target_arch = "x86_64")]
use kodamaembed_model::repro::Kind;
use kodamaembed_model::{
    Activation, Head, Output, Pooling, Positions, Profile, Rotary, STAGES, Shape, Stage, Tensors,
    fused_qkv, relative::RelativeBias, repro::restored_class, tensor, tensor_len,
};
use rayon::prelude::*;
use std::{fmt, num::NonZeroUsize, time::Instant};

/// Token rows per task in the row-wise kernels.
const ROWS: usize = 64;

/// Short inputs take fewer rows a task, so every worker gets some.
fn rows_per_task(tokens: usize) -> usize {
    tokens.div_ceil(rayon::current_num_threads()).clamp(1, ROWS)
}

/// Token limit for the smaller pool, which reduces short-batch scheduling overhead.
const SHORT_TOKENS: usize = 256;

/// One sequence's token ids and segment types.
pub type TokenSequence = (Vec<i32>, Vec<i32>);

/// The mean of the sequences' key means: the center nearest them in squared distance.
#[cfg(target_arch = "x86_64")]
fn key_means(qkv: &[f32], hidden: usize, firsts: &[usize], lengths: &[usize]) -> Vec<f64> {
    let mut total = vec![0.0f64; hidden];
    let mut sequence = vec![0.0f64; hidden];
    for (&first, &len) in firsts.iter().zip(lengths) {
        sequence.fill(0.0);
        for row in first..first + len {
            for (sum, &x) in sequence
                .iter_mut()
                .zip(&qkv[row * 3 * hidden + hidden..][..hidden])
            {
                *sum += f64::from(x);
            }
        }
        for (all, &sum) in total.iter_mut().zip(&sequence) {
            *all += sum / len as f64;
        }
    }
    total.iter().map(|&sum| sum / firsts.len() as f64).collect()
}

#[cfg(target_arch = "x86_64")]
fn class_means(rows: &[f32], width: usize, firsts: &[usize], lasts: &[usize]) -> Vec<f64> {
    let mut means = vec![0.0; 2 * width];
    for (mean, picks) in means.chunks_exact_mut(width).zip([firsts, lasts]) {
        for &row in picks {
            for (sum, &x) in mean.iter_mut().zip(&rows[row * width..(row + 1) * width]) {
                *sum += f64::from(x);
            }
        }
        for sum in mean {
            *sum /= picks.len() as f64;
        }
    }
    means
}

#[cfg(target_arch = "x86_64")]
fn middle_mean(
    rows: &[f32],
    width: usize,
    firsts: &[usize],
    lengths: &[usize],
) -> Option<Vec<f64>> {
    let mut mean = vec![0.0; width];
    let mut sequences = 0;
    for (&first, &len) in firsts.iter().zip(lengths).filter(|(_, len)| **len > 2) {
        let mut own = vec![0.0; width];
        for row in rows[(first + 1) * width..(first + len - 1) * width].chunks_exact(width) {
            for (sum, &x) in own.iter_mut().zip(row) {
                *sum += f64::from(x);
            }
        }
        for (total, sum) in mean.iter_mut().zip(own) {
            *total += sum / (len - 2) as f64;
        }
        sequences += 1;
    }
    (sequences > 0).then(|| {
        mean.into_iter()
            .map(|total| total / f64::from(sequences))
            .collect()
    })
}

/// Worker counts change only partitioning, never reduction order or bits.
fn pool_for(own: Option<&Workers>, tokens: usize) -> &Workers {
    match own {
        Some(pool) => pool,
        None if tokens <= SHORT_TOKENS => SHORT_POOL.as_ref().unwrap_or(&POOL),
        None => &POOL,
    }
}

struct Timer {
    last: Instant,
    milliseconds: [f64; STAGES.len()],
}

fn mark(timer: &mut Option<Timer>, stage: Stage) {
    if let Some(timer) = timer {
        let now = Instant::now();
        timer.milliseconds[stage as usize] += (now - timer.last).as_secs_f64() * 1e3;
        timer.last = now;
    }
}

#[derive(Debug)]
pub enum EncoderError {
    Unsupported(String),
    Weights(String),
    Input(String),
}

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(reason) => write!(f, "unsupported model: {reason}"),
            Self::Weights(reason) => write!(f, "weights: {reason}"),
            Self::Input(reason) => write!(f, "input: {reason}"),
        }
    }
}

impl std::error::Error for EncoderError {}

struct Norm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    eps: f32,
}

impl Norm {
    /// None when the mean or variance + eps is not finite.
    fn moments(&self, row: &[f32]) -> Option<(f32, f32)> {
        let n = row.len() as f32;
        let mean = lane_sum(row, |x| x) / n;
        let variance = lane_sum(row, |x| (x - mean) * (x - mean)) / n;
        let denominator = variance + self.eps;
        (mean.is_finite() && denominator.is_finite()).then(|| (mean, 1.0 / denominator.sqrt()))
    }

    fn apply(&self, row: &mut [f32]) {
        self.normalize(row, self.moments(row));
    }

    /// A row whose variance overflows becomes NaN, not zeros: the output check rejects it.
    fn normalize(&self, row: &mut [f32], moments: Option<(f32, f32)>) {
        let Some((mean, rstd)) = moments else {
            row.fill(f32::NAN);
            return;
        };
        for ((x, &gamma), &beta) in row.iter_mut().zip(&self.gamma).zip(&self.beta) {
            *x = (*x - mean) * rstd * gamma + beta;
        }
    }

    fn apply_rows(&self, rows: &mut [f32], half: bool) {
        let width = self.gamma.len();
        let per = rows_per_task(rows.len() / width);
        rows.par_chunks_mut(per * width).for_each(|chunk| {
            round_half(chunk, half);
            for row in chunk.chunks_exact_mut(width) {
                self.apply(row);
            }
            round_half(chunk, half);
        });
    }

    /// Also keeps the rows and statistics the deferred projection reads.
    fn apply_rows_keeping(&self, rows: &mut [f32], half: bool, input: &mut NormInput) {
        let width = self.gamma.len();
        let tokens = rows.len() / width;
        input.resize(tokens, width);
        let per = rows_per_task(tokens);
        rows.par_chunks_mut(per * width)
            .zip(input.rows.par_chunks_mut(per * width))
            .zip(
                input
                    .mean
                    .par_chunks_mut(per)
                    .zip(input.rstd.par_chunks_mut(per)),
            )
            .for_each(|((chunk, kept), (means, rstds))| {
                round_half(chunk, half);
                kept.copy_from_slice(chunk);
                for ((row, mean), rstd) in chunk.chunks_exact_mut(width).zip(means).zip(rstds) {
                    let moments = self.moments(row);
                    (*mean, *rstd) = moments.unwrap_or((f32::NAN, f32::NAN));
                    self.normalize(row, moments);
                }
                round_half(chunk, half);
            });
    }
}

/// A deferred projection's input: the norm's rows as read and their statistics.
#[derive(Default)]
struct NormInput {
    rows: Vec<f32>,
    mean: Vec<f32>,
    rstd: Vec<f32>,
}

impl NormInput {
    fn resize(&mut self, tokens: usize, width: usize) {
        self.rows.resize(tokens * width, 0.0);
        self.mean.resize(tokens, 0.0);
        self.rstd.resize(tokens, 0.0);
    }
}

/// Integer attention rounds residuals and norms to FP16, as projection inputs are.
fn round_half(rows: &mut [f32], half: bool) {
    if !half {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    // SAFETY: only integer-attention encoders round, and they require F16C.
    unsafe {
        crate::math::avx2::round_to_fp16(rows)
    };
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = rows;
        panic!("FP16 residual rows belong to integer attention, which requires x86-64");
    }
}

/// 32 lane sums halved by partners, in a GPU warp's xor-shuffle order: the same bits on both.
fn lane_sum<T: Copy + Default + std::ops::AddAssign>(values: &[T], f: impl Fn(T) -> T) -> T {
    let (chunks, tail) = values.as_chunks::<32>();
    let mut lanes = [T::default(); 32];
    for chunk in chunks {
        for (lane, &x) in lanes.iter_mut().zip(chunk) {
            *lane += f(x);
        }
    }
    for (lane, &x) in lanes.iter_mut().zip(tail) {
        *lane += f(x);
    }
    let mut width = 16;
    while width > 0 {
        for j in 0..width {
            lanes[j] += lanes[j + width];
        }
        width /= 2;
    }
    lanes[0]
}

struct Embeddings {
    word: Vec<f32>,
    position: Vec<f32>,
    token_type: Vec<f32>,
    norm: Norm,
}

impl Embeddings {
    fn apply(&self, ids: &[i32], types: &[i32], positions: &[i32], out: &mut [f32], half: bool) {
        let h = self.norm.gamma.len();
        let per = rows_per_task(positions.len());
        out.par_chunks_mut(per * h)
            .enumerate()
            .for_each(|(chunk, out)| {
                for (r, row) in out.chunks_exact_mut(h).enumerate() {
                    let token = chunk * per + r;
                    let word = &self.word[ids[token] as usize * h..][..h];
                    let segment = &self.token_type[types[token] as usize * h..][..h];
                    let position = &self.position[positions[token] as usize * h..][..h];
                    for (((x, &w), &s), &p) in row.iter_mut().zip(word).zip(segment).zip(position) {
                        *x = (w + s) + p;
                    }
                    self.norm.apply(row);
                }
                round_half(out, half);
            });
    }
}

/// A BERT layer: a norm after attention and after the feed-forward.
struct Layer {
    qkv: Projection,
    qkv_bias: Vec<f32>,
    key_centers: Option<Vec<f32>>,
    out: Projection,
    out_bias: Vec<f32>,
    norm1: Norm,
    up: Projection,
    up_bias: Vec<f32>,
    down: Projection,
    down_bias: Vec<f32>,
    norm2: Norm,
}

/// A ModernBERT layer: a norm before attention and before the gated feed-forward, no biases.
struct ModernLayer {
    /// None in the first layer, which attends over the embedding norm's rows as they are.
    attention_norm: Option<Norm>,
    qkv: PackedWeight,
    out: PackedWeight,
    feed_forward_norm: Norm,
    /// The up projection's rows by half: the activated inputs, then their gates.
    inputs: PackedWeight,
    gates: PackedWeight,
    down: PackedWeight,
}

/// Rotary positions: [`Rotary::turns`] for one rotation base.
struct Turns(Vec<f32>);

impl Turns {
    /// Turns each head's query and key in every qkv row by its token's position.
    fn apply(&self, qkv: &mut [f32], hidden: usize, head: usize, positions: &[i32]) {
        let pairs = head / 2;
        let per = rows_per_task(positions.len());
        qkv.par_chunks_mut(per * 3 * hidden)
            .zip(positions.par_chunks(per))
            .for_each(|(rows, positions)| {
                for (row, &position) in rows.chunks_exact_mut(3 * hidden).zip(positions) {
                    let turns = &self.0[position as usize * head..][..head];
                    let (cos, sin) = turns.split_at(pairs);
                    for vector in row[..2 * hidden].chunks_exact_mut(head) {
                        let (first, second) = vector.split_at_mut(pairs);
                        for (((x, y), &c), &s) in first.iter_mut().zip(second).zip(cos).zip(sin) {
                            (*x, *y) = (*x * c - *y * s, *y * c + *x * s);
                        }
                    }
                }
            });
    }
}

/// BERT, RoBERTa and MPNet: tables for positions and segments, and layers that norm afterwards.
struct Bert {
    embeddings: Embeddings,
    layers: Vec<Layer>,
    relative: Option<RelativeBias>,
}

/// ModernBERT: positions turn the queries and keys, and layers norm beforehand.
struct Modern {
    rotary: Rotary,
    word: Vec<f32>,
    embedding_norm: Norm,
    layers: Vec<ModernLayer>,
    /// The turns of the layers over whole sequences, and of the windowed ones.
    global_turns: Turns,
    local_turns: Turns,
    final_norm: Norm,
    /// What every projection adds as its bias: the family has none.
    no_bias: Vec<f32>,
    /// A sublayer's normed input, and the feed-forward's gates.
    normed: Vec<f32>,
    gates: Vec<f32>,
}

/// A model family's weights.
enum Family {
    Bert(Box<Bert>),
    Modern(Box<Modern>),
}

#[cfg(target_arch = "x86_64")]
struct Calibrated {
    projections: [Projection; 4],
    keys: Option<Vec<f32>>,
}

struct Workspace {
    hidden: Vec<f32>,
    qkv: Vec<f32>,
    context: Vec<f32>,
    attended: Vec<f32>,
    intermediate: Vec<f32>,
    norm1_input: NormInput,
    norm2_input: NormInput,
}

impl Workspace {
    fn new(shape: &Shape, tokens: usize) -> Result<Self, EncoderError> {
        let (h, i) = (shape.hidden, shape.intermediate);
        let hidden = tensor_len(&[tokens, h]).map_err(EncoderError::Unsupported)?;
        let qkv = tensor_len(&[tokens, 3, h]).map_err(EncoderError::Unsupported)?;
        let intermediate = tensor_len(&[tokens, i]).map_err(EncoderError::Unsupported)?;
        Ok(Self {
            hidden: zeros(hidden)?,
            qkv: zeros(qkv)?,
            context: zeros(hidden)?,
            attended: zeros(hidden)?,
            intermediate: zeros(intermediate)?,
            norm1_input: NormInput::default(),
            norm2_input: NormInput::default(),
        })
    }
}

/// Allocation failure is an error, not an abort.
fn zeros(len: usize) -> Result<Vec<f32>, EncoderError> {
    let layout = std::alloc::Layout::array::<f32>(len)
        .map_err(|_| EncoderError::Unsupported(format!("{len} activations overflow a layout")))?;
    if layout.size() == 0 {
        return Ok(Vec::new());
    }
    // SAFETY: the layout has a nonzero size.
    let at = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<f32>();
    if at.is_null() {
        return Err(EncoderError::Unsupported(format!(
            "{len} activations exceed the memory available"
        )));
    }
    // SAFETY: `at` holds `len` zeroed f32 values from this layout.
    Ok(unsafe { Vec::from_raw_parts(at, len, len) })
}

/// A batch's token count, once it fits `max_tokens` and the model.
fn checked_tokens(
    shape: &Shape,
    max_tokens: usize,
    ids: &[i32],
    types: &[i32],
    lengths: &[usize],
) -> Result<usize, EncoderError> {
    let tokens = checked_input(shape, ids, types, lengths)?;
    if tokens > max_tokens {
        return Err(EncoderError::Input(format!(
            "{tokens} tokens (limit {max_tokens})"
        )));
    }
    Ok(tokens)
}

fn checked_input(
    shape: &Shape,
    ids: &[i32],
    types: &[i32],
    lengths: &[usize],
) -> Result<usize, EncoderError> {
    let tokens = lengths
        .iter()
        .try_fold(0usize, |total, &len| total.checked_add(len))
        .ok_or_else(|| EncoderError::Input("sequence lengths overflow usize".into()))?;
    let max_len = lengths.iter().copied().max().unwrap_or(0);
    if lengths.is_empty() || lengths.contains(&0) || tokens != ids.len() {
        return Err(EncoderError::Input(format!(
            "{} ids for lengths summing to {tokens}, with no empty sequence allowed",
            ids.len()
        )));
    }
    if types.len() != ids.len() {
        return Err(EncoderError::Input(format!(
            "{} segment types for {} ids",
            types.len(),
            ids.len()
        )));
    }
    if max_len > shape.max_sequence_len() {
        return Err(EncoderError::Input(format!(
            "a sequence of {max_len} tokens (limit {})",
            shape.max_sequence_len()
        )));
    }
    let vocab = shape.vocab as i32;
    if let Some(bad) = ids.iter().find(|&&id| !(0..vocab).contains(&id)) {
        return Err(EncoderError::Input(format!(
            "token id {bad} outside the vocabulary"
        )));
    }
    let segments = shape.type_vocab as i32;
    if let Some(bad) = types.iter().find(|&&kind| !(0..segments).contains(&kind)) {
        return Err(EncoderError::Input(format!(
            "segment type {bad} outside the {segments} the model has"
        )));
    }
    Ok(tokens)
}

#[derive(Clone, Copy)]
struct ExecutionOptions {
    kernel: Kernel,
    integer_attention: bool,
    half_residual: bool,
    /// For calibration's means; centered layers keep them anyway: deferred projections read them.
    keep_norm_inputs: bool,
    /// Calibration's own pass centers keys on their sequences' means instead.
    static_keys: bool,
}

impl Layer {
    /// Both norms defer: QKV reads the previous layer's kept input, the up projection this layer's.
    #[allow(clippy::too_many_arguments)]
    fn apply(
        &self,
        execution: ExecutionOptions,
        heads: usize,
        relative: Option<&RelativeBias>,
        lengths: &[usize],
        restored: Option<&[u32]>,
        tokens: usize,
        w: &mut Workspace,
        timer: &mut Option<Timer>,
    ) -> Result<(), EncoderError> {
        let ExecutionOptions {
            kernel,
            integer_attention,
            half_residual,
            keep_norm_inputs,
            static_keys,
        } = execution;
        let keep = keep_norm_inputs || restored.is_some();
        // Restart restored chains at hidden-width boundaries for batch-invariant rounding.
        let h = self.out.n();
        let project =
            |projection: &Projection, input: &[f32], epilogue: Epilogue<'_>, output: &mut [f32]| {
                match restored {
                    Some(classes) => projection
                        .apply_restored(kernel, input, tokens, classes, h, epilogue, output),
                    None => projection.apply(kernel, input, tokens, epilogue, output),
                }
            };
        let Workspace {
            hidden,
            qkv,
            context,
            attended,
            intermediate,
            norm1_input,
            norm2_input,
        } = w;
        let hidden = &mut hidden[..tokens * h];
        let qkv = &mut qkv[..tokens * 3 * h];
        let context = &mut context[..tokens * h];
        let attended = &mut attended[..tokens * h];
        let intermediate = &mut intermediate[..tokens * self.up.n()];
        match restored.filter(|_| self.qkv.defers()) {
            Some(classes) => self.qkv.apply_deferred(
                kernel,
                &norm2_input.rows,
                tokens,
                classes,
                &norm2_input.mean,
                &norm2_input.rstd,
                false,
                h,
                qkv,
            )?,
            None => project(&self.qkv, hidden, Epilogue::Bias(&self.qkv_bias), qkv)?,
        }
        mark(timer, Stage::Qkv);
        attention_current_pool(
            kernel,
            qkv,
            lengths,
            h,
            heads,
            integer_attention,
            self.key_centers.as_deref().filter(|_| static_keys),
            relative,
            None,
            context,
        );
        mark(timer, Stage::Attention);
        project(
            &self.out,
            context,
            Epilogue::BiasResidual(&self.out_bias, hidden),
            attended,
        )?;
        mark(timer, Stage::Out);
        if keep {
            self.norm1
                .apply_rows_keeping(attended, half_residual, norm1_input);
        } else {
            self.norm1.apply_rows(attended, half_residual);
        }
        mark(timer, Stage::Norm1);
        match restored.filter(|_| self.up.defers()) {
            Some(classes) => self.up.apply_deferred(
                kernel,
                &norm1_input.rows,
                tokens,
                classes,
                &norm1_input.mean,
                &norm1_input.rstd,
                true,
                h,
                intermediate,
            )?,
            None => project(
                &self.up,
                attended,
                Epilogue::BiasGelu(&self.up_bias),
                intermediate,
            )?,
        }
        mark(timer, Stage::Ffn1);
        project(
            &self.down,
            intermediate,
            Epilogue::BiasResidual(&self.down_bias, attended),
            hidden,
        )?;
        mark(timer, Stage::Ffn2);
        if keep {
            self.norm2
                .apply_rows_keeping(hidden, half_residual, norm2_input);
        } else {
            self.norm2.apply_rows(hidden, half_residual);
        }
        mark(timer, Stage::Norm2);
        Ok(())
    }
}

impl Bert {
    fn load(shape: &Shape, tensors: &dyn Tensors) -> Result<Self, EncoderError> {
        let (h, i) = (shape.hidden, shape.intermediate);
        let get =
            |name: &str, len: usize| tensor(tensors, name, len).map_err(EncoderError::Weights);
        let norm = |name: &str| -> Result<Norm, EncoderError> {
            Ok(Norm {
                gamma: get(&format!("{name}.weight"), h)?,
                beta: get(&format!("{name}.bias"), h)?,
                eps: shape.eps,
            })
        };
        let embeddings = Embeddings {
            word: get("embeddings.word_embeddings.weight", shape.vocab * h)?,
            position: get(
                "embeddings.position_embeddings.weight",
                shape.max_positions * h,
            )?,
            token_type: get(
                "embeddings.token_type_embeddings.weight",
                shape.type_vocab * h,
            )?,
            norm: norm("embeddings.LayerNorm")?,
        };
        let mut layers: Vec<Layer> = Vec::with_capacity(shape.layers);
        for layer in 0..shape.layers {
            let at = |suffix: &str| format!("encoder.layer.{layer}.{suffix}");
            let (qkv_weights, qkv_bias) =
                fused_qkv(tensors, shape, layer).map_err(EncoderError::Weights)?;
            layers.push(Layer {
                qkv: Projection::new(&qkv_weights, 3 * h, h),
                qkv_bias,
                key_centers: None,
                out: Projection::new(&get(&at("attention.output.dense.weight"), h * h)?, h, h),
                out_bias: get(&at("attention.output.dense.bias"), h)?,
                norm1: norm(&at("attention.output.LayerNorm"))?,
                up: Projection::new(&get(&at("intermediate.dense.weight"), i * h)?, i, h),
                up_bias: get(&at("intermediate.dense.bias"), i)?,
                down: Projection::new(&get(&at("output.dense.weight"), h * i)?, h, i),
                down_bias: get(&at("output.dense.bias"), h)?,
                norm2: norm(&at("output.LayerNorm"))?,
            });
        }
        Ok(Self {
            embeddings,
            layers,
            relative: RelativeBias::load(shape, tensors).map_err(EncoderError::Weights)?,
        })
    }

    /// The calibrated projections and key centers in place of the FP32 ones.
    #[cfg(target_arch = "x86_64")]
    fn install(&mut self, built: Vec<Calibrated>) {
        for (layer, built) in self.layers.iter_mut().zip(built) {
            let [qkv, out, up, down] = built.projections;
            (layer.qkv, layer.out, layer.up, layer.down) = (qkv, out, up, down);
            layer.key_centers = built.keys;
        }
    }
}

impl Modern {
    fn load(
        shape: &Shape,
        rotary: Rotary,
        max_tokens: usize,
        tensors: &dyn Tensors,
    ) -> Result<Self, EncoderError> {
        let (h, inner) = (shape.hidden, shape.intermediate);
        let rows = |width: usize| -> Result<Vec<f32>, EncoderError> {
            zeros(tensor_len(&[max_tokens, width]).map_err(EncoderError::Unsupported)?)
        };
        let (normed, gates) = (rows(h)?, rows(inner)?);
        let get =
            |name: &str, len: usize| tensor(tensors, name, len).map_err(EncoderError::Weights);
        let norm = |name: &str| -> Result<Norm, EncoderError> {
            Ok(Norm {
                gamma: get(&format!("{name}.weight"), h)?,
                beta: vec![0.0; h],
                eps: shape.eps,
            })
        };
        let word = get("embeddings.tok_embeddings.weight", shape.vocab * h)?;
        let embedding_norm = norm("embeddings.norm")?;
        let mut layers: Vec<ModernLayer> = Vec::with_capacity(shape.layers);
        for layer in 0..shape.layers {
            let at = |suffix: &str| format!("layers.{layer}.{suffix}");
            let attention_norm = match layer {
                0 => None,
                _ => Some(norm(&at("attn_norm"))?),
            };
            let qkv = PackedWeight::new(&get(&at("attn.Wqkv.weight"), 3 * h * h)?, 3 * h, h);
            let out = PackedWeight::new(&get(&at("attn.Wo.weight"), h * h)?, h, h);
            let feed_forward_norm = norm(&at("mlp_norm"))?;
            let up = get(&at("mlp.Wi.weight"), 2 * inner * h)?;
            let (inputs, gates) = up.split_at(inner * h);
            layers.push(ModernLayer {
                attention_norm,
                qkv,
                out,
                feed_forward_norm,
                inputs: PackedWeight::new(inputs, inner, h),
                gates: PackedWeight::new(gates, inner, h),
                down: PackedWeight::new(&get(&at("mlp.Wo.weight"), h * inner)?, h, inner),
            });
        }
        let final_norm = norm("final_norm")?;
        let (head, positions) = (h / shape.heads, max_tokens.min(shape.max_sequence_len()));
        let turns = |theta: f32| Turns(Rotary::turns(theta, head, positions));
        Ok(Self {
            rotary,
            word,
            embedding_norm,
            layers,
            global_turns: turns(rotary.global_theta),
            local_turns: turns(rotary.local_theta),
            final_norm,
            no_bias: vec![0.0; (3 * h).max(inner)],
            normed,
            gates,
        })
    }

    /// Embeds the batch and runs every layer and the final norm, which leaves the encoded rows
    /// in `w.hidden`.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        kernel: Kernel,
        shape: &Shape,
        ids: &[i32],
        positions: &[i32],
        lengths: &[usize],
        w: &mut Workspace,
        timer: &mut Option<Timer>,
    ) {
        let (h, inner, heads) = (shape.hidden, shape.intermediate, shape.heads);
        let tokens = ids.len();
        let Self {
            rotary,
            word,
            embedding_norm,
            layers,
            global_turns,
            local_turns,
            final_norm,
            no_bias,
            normed,
            gates,
        } = self;
        let (qkv_bias, row_bias, inner_bias) =
            (&no_bias[..3 * h], &no_bias[..h], &no_bias[..inner]);
        let hidden = &mut w.hidden[..tokens * h];
        let qkv = &mut w.qkv[..tokens * 3 * h];
        let context = &mut w.context[..tokens * h];
        let attended = &mut w.attended[..tokens * h];
        let inputs = &mut w.intermediate[..tokens * inner];
        let normed = &mut normed[..tokens * h];
        let gates = &mut gates[..tokens * inner];
        let per = rows_per_task(tokens);
        hidden
            .par_chunks_mut(per * h)
            .zip(ids.par_chunks(per))
            .for_each(|(rows, ids)| {
                for (row, &id) in rows.chunks_exact_mut(h).zip(ids) {
                    row.copy_from_slice(&word[id as usize * h..][..h]);
                    embedding_norm.apply(row);
                }
            });
        mark(timer, Stage::Embed);
        for (index, layer) in layers.iter().enumerate() {
            let input: &[f32] = match &layer.attention_norm {
                Some(norm) => {
                    normed.copy_from_slice(hidden);
                    norm.apply_rows(normed, false);
                    normed
                }
                None => hidden,
            };
            mark(timer, Stage::Norm1);
            gemm_current_pool(
                kernel,
                input,
                tokens,
                &layer.qkv,
                Epilogue::Bias(qkv_bias),
                qkv,
            );
            let reach = rotary.reach(index);
            let turns: &Turns = match reach {
                Some(_) => local_turns,
                None => global_turns,
            };
            turns.apply(qkv, h, h / heads, positions);
            mark(timer, Stage::Qkv);
            attention_current_pool(
                kernel, qkv, lengths, h, heads, false, None, None, reach, context,
            );
            mark(timer, Stage::Attention);
            gemm_current_pool(
                kernel,
                context,
                tokens,
                &layer.out,
                Epilogue::BiasResidual(row_bias, hidden),
                attended,
            );
            mark(timer, Stage::Out);
            normed.copy_from_slice(attended);
            layer.feed_forward_norm.apply_rows(normed, false);
            mark(timer, Stage::Norm2);
            gemm_current_pool(
                kernel,
                normed,
                tokens,
                &layer.inputs,
                Epilogue::BiasGelu(inner_bias),
                inputs,
            );
            gemm_current_pool(
                kernel,
                normed,
                tokens,
                &layer.gates,
                Epilogue::Bias(inner_bias),
                gates,
            );
            inputs
                .par_chunks_mut(per * inner)
                .zip(gates.par_chunks(per * inner))
                .for_each(|(inputs, gates)| {
                    for (input, &gate) in inputs.iter_mut().zip(gates) {
                        *input *= gate;
                    }
                });
            mark(timer, Stage::Ffn1);
            gemm_current_pool(
                kernel,
                inputs,
                tokens,
                &layer.down,
                Epilogue::BiasResidual(row_bias, attended),
                hidden,
            );
            mark(timer, Stage::Ffn2);
        }
        final_norm.apply_rows(hidden, false);
    }
}

/// Mean pooling and L2 normalization, one output row per sequence.
#[doc(hidden)]
pub fn pool_rows(hidden: &[f32], h: usize, lengths: &[usize], out: &mut [f32]) {
    Finish::Pool(Pooling::Mean).apply(Kernel::Portable, hidden, h, lengths, out);
}

/// How a cross-encoder's head turns its dense layer's outputs into the classifier's features.
enum Features {
    /// BERT's pooler: tanh.
    Tanh,
    /// ModernBERT's prediction head: an activation, then a norm.
    Normed(Activation, Norm),
}

enum Finish {
    /// L2-normalized pooled rows, `hidden` values per sequence.
    Pool(Pooling),
    /// A cross-encoder's head and classifier, one score per sequence.
    Score {
        /// The rows the head's dense layer reads: the first, or their mean.
        pooling: Pooling,
        dense: PackedWeight,
        /// Zeros in a prediction head, whose dense layer has none.
        dense_bias: Vec<f32>,
        features: Features,
        classifier: Vec<f32>,
        classifier_bias: f32,
    },
}

/// The mean of `rows`, each `h` wide, summed in their order.
fn mean_row(rows: &[f32], h: usize) -> Vec<f64> {
    let mut mean = vec![0.0f64; h];
    for row in rows.chunks_exact(h) {
        for (sum, &x) in mean.iter_mut().zip(row) {
            *sum += f64::from(x);
        }
    }
    let count = (rows.len() / h) as f64;
    for m in &mut mean {
        *m /= count;
    }
    mean
}

impl Finish {
    fn apply(&self, kernel: Kernel, hidden: &[f32], h: usize, lengths: &[usize], out: &mut [f32]) {
        let starts: Vec<usize> = lengths
            .iter()
            .scan(0, |at, &len| {
                let start = *at;
                *at += len;
                Some(start)
            })
            .collect();
        match self {
            Self::Pool(pooling) => {
                out.par_chunks_mut(h)
                    .zip(starts.par_iter().zip(lengths))
                    .for_each(|(out, (&start, &len))| {
                        let mean = mean_row(&hidden[start * h..(start + pooling.rows(len)) * h], h);
                        let norm = lane_sum(&mean, |m| m * m).sqrt().max(1e-12);
                        for (o, m) in out.iter_mut().zip(&mean) {
                            *o = (m / norm) as f32;
                        }
                    });
            }
            Self::Score {
                pooling,
                dense,
                dense_bias,
                features,
                classifier,
                classifier_bias,
            } => {
                let rows: Vec<f32> = match pooling {
                    Pooling::Cls => starts
                        .iter()
                        .flat_map(|&start| &hidden[start * h..(start + 1) * h])
                        .copied()
                        .collect(),
                    Pooling::Mean => starts
                        .iter()
                        .zip(lengths)
                        .flat_map(|(&start, &len)| {
                            mean_row(&hidden[start * h..(start + len) * h], h)
                        })
                        .map(|mean| mean as f32)
                        .collect(),
                };
                let mut projected = vec![0.0; lengths.len() * h];
                gemm_current_pool(
                    kernel,
                    &rows,
                    lengths.len(),
                    dense,
                    Epilogue::Bias(dense_bias),
                    &mut projected,
                );
                for (score, row) in out.iter_mut().zip(projected.chunks_exact_mut(h)) {
                    // tanh maps infinity to a finite value, hiding overflow in the dense layer's
                    // matrix product or bias addition.
                    if row.iter().any(|value| !value.is_finite()) {
                        *score = f32::NAN;
                        continue;
                    }
                    match features {
                        Features::Tanh => row.iter_mut().for_each(|x| *x = x.tanh()),
                        Features::Normed(activation, norm) => {
                            let activate: fn(f32) -> f32 = match activation {
                                Activation::Gelu => math::gelu_unbounded,
                                Activation::Silu => math::silu,
                            };
                            row.iter_mut().for_each(|x| *x = activate(*x));
                            norm.apply(row);
                        }
                    }
                    let dot: f64 = row
                        .iter()
                        .zip(classifier)
                        .map(|(&feature, &w)| f64::from(feature) * f64::from(w))
                        .sum();
                    *score = (dot + f64::from(*classifier_bias)) as f32;
                }
            }
        }
    }
}

pub struct Encoder {
    // Dedicated pools isolate explicit worker budgets from the shared pools.
    pool: Option<Workers>,
    precision: CpuPrecision,
    statics: bool,
    half_residual: bool,
    kernel: Kernel,
    shape: Shape,
    max_tokens: usize,
    family: Family,
    finish: Finish,
    work: Workspace,
    profile: Option<Profile>,
}

impl Encoder {
    pub fn new(
        kernel: Kernel,
        shape: Shape,
        output: Output,
        max_tokens: usize,
        tensors: &dyn Tensors,
    ) -> Result<Self, EncoderError> {
        Self::new_with_precision(
            kernel,
            shape,
            output,
            max_tokens,
            tensors,
            CpuPrecision::Fp32,
        )
    }

    /// Loads with an explicit projection policy; [`CpuPrecision`] gives each
    /// preset's requirements.
    pub fn new_with_precision(
        kernel: Kernel,
        shape: Shape,
        output: Output,
        max_tokens: usize,
        tensors: &dyn Tensors,
        precision: CpuPrecision,
    ) -> Result<Self, EncoderError> {
        shape.validate().map_err(EncoderError::Unsupported)?;
        precision.validate(kernel, shape, output)?;
        Self::load(kernel, shape, output, max_tokens, tensors, precision)
    }

    /// [`Self::new_with_precision`] without the shape check, so centered tests run fewer layers.
    #[cfg(all(test, target_arch = "x86_64"))]
    pub(crate) fn new_without_shape_check(
        kernel: Kernel,
        shape: Shape,
        output: Output,
        max_tokens: usize,
        tensors: &dyn Tensors,
        precision: CpuPrecision,
    ) -> Result<Self, EncoderError> {
        shape.validate().map_err(EncoderError::Unsupported)?;
        Self::load(kernel, shape, output, max_tokens, tensors, precision)
    }

    fn load(
        kernel: Kernel,
        shape: Shape,
        output: Output,
        max_tokens: usize,
        tensors: &dyn Tensors,
        precision: CpuPrecision,
    ) -> Result<Self, EncoderError> {
        if !kernel.is_supported() {
            return Err(EncoderError::Unsupported(format!(
                "{kernel} is unavailable on this processor"
            )));
        }
        if max_tokens == 0 {
            return Err(EncoderError::Unsupported(
                "max_tokens must be positive".into(),
            ));
        }
        let (h, i) = (shape.hidden, shape.intermediate);
        let rotary = match shape.positions {
            Positions::ModernBert(rotary) => Some(rotary),
            _ => None,
        };
        let layer_bytes = match rotary {
            Some(_) => size_of::<ModernLayer>(),
            None => size_of::<Layer>(),
        };
        if shape
            .layers
            .checked_mul(layer_bytes)
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(EncoderError::Unsupported(
                "layer storage exceeds addressable memory".into(),
            ));
        }
        // ModernBERT packs its up projection by half, each as wide as BERT's.
        for (columns, depth) in [(3 * h, h), (h, h), (i, h), (h, i)] {
            tensor_len(&[columns.div_ceil(NR), NR, depth]).map_err(EncoderError::Unsupported)?;
        }
        let work = Workspace::new(&shape, max_tokens)?;
        let family = match rotary {
            Some(rotary) => {
                Family::Modern(Box::new(Modern::load(&shape, rotary, max_tokens, tensors)?))
            }
            None => Family::Bert(Box::new(Bert::load(&shape, tensors)?)),
        };
        let get =
            |name: &str, len: usize| tensor(tensors, name, len).map_err(EncoderError::Weights);
        let finish = match output {
            Output::Embedding(pooling) => Finish::Pool(pooling),
            Output::Relevance(head) => {
                let (pooling, dense, dense_bias, features) = match head {
                    Head::Pooler => {
                        let dense = PackedWeight::new(&get("pooler.dense.weight", h * h)?, h, h);
                        let bias = get("pooler.dense.bias", h)?;
                        (Pooling::Cls, dense, bias, Features::Tanh)
                    }
                    Head::Prediction(pooling, activation) => {
                        let dense = PackedWeight::new(&get("head.dense.weight", h * h)?, h, h);
                        let norm = Norm {
                            gamma: get("head.norm.weight", h)?,
                            beta: vec![0.0; h],
                            eps: shape.eps,
                        };
                        let features = Features::Normed(activation, norm);
                        (pooling, dense, vec![0.0; h], features)
                    }
                };
                Finish::Score {
                    pooling,
                    dense,
                    dense_bias,
                    features,
                    classifier: get("classifier.weight", h)?,
                    classifier_bias: get("classifier.bias", 1)?[0],
                }
            }
        };
        Ok(Self {
            pool: None,
            precision,
            statics: false,
            half_residual: false,
            kernel,
            shape,
            max_tokens,
            family,
            finish,
            work,
            profile: None,
        })
    }

    /// Loads with a dedicated inference pool, independent of RAYON_NUM_THREADS.
    pub fn new_with_precision_and_workers(
        kernel: Kernel,
        shape: Shape,
        output: Output,
        max_tokens: usize,
        tensors: &dyn Tensors,
        precision: CpuPrecision,
        workers: NonZeroUsize,
    ) -> Result<Self, EncoderError> {
        if workers.get() > rayon::max_num_threads() {
            return Err(EncoderError::Unsupported(format!(
                "CPU worker count exceeds Rayon's limit {}",
                rayon::max_num_threads()
            )));
        }
        let mut encoder =
            Self::new_with_precision(kernel, shape, output, max_tokens, tensors, precision)?;
        encoder.pool = Some(
            crate::worker_pool("kodamaembed-cpu-bounded", workers.get())
                .map_err(|e| EncoderError::Unsupported(format!("CPU worker pool: {e}")))?,
        );
        if encoder.threads() != workers.get() {
            return Err(EncoderError::Unsupported(
                "CPU worker pool did not honor the requested count".into(),
            ));
        }
        Ok(encoder)
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// The explicitly selected CPU projection policy.
    pub fn precision(&self) -> CpuPrecision {
        self.precision
    }

    /// The label a fingerprint hashes with [`crate::NUMERICAL_SOURCES`], the
    /// [`Kernel`] and the model bytes.
    pub fn numerical_identity(&self) -> &'static str {
        self.precision.identity()
    }

    /// Without a dedicated pool, batches of up to 256 tokens may run on a
    /// smaller pool of performance cores.
    pub fn threads(&self) -> usize {
        self.pool
            .as_ref()
            .unwrap_or_else(|| &POOL)
            .current_num_threads()
    }

    /// Times every stage of every later batch; see [`Encoder::profile`].
    pub fn enable_profiling(&mut self) {
        self.profile = Some(Profile::default());
    }

    /// Wall time per stage of the batches since profiling was enabled.
    pub fn profile(&self) -> Option<&Profile> {
        self.profile.as_ref()
    }

    /// Values returned per sequence: `hidden` for sentence vectors, 1 for scores.
    pub fn output_width(&self) -> usize {
        match self.finish {
            Finish::Pool(_) => self.shape.hidden,
            Finish::Score { .. } => 1,
        }
    }

    /// Quantizes a centered preset by GPTQ over the FP32 forward of `inputs`, two or more
    /// tokens each and one with interior tokens; once only, a failure leaves it uncalibrated.
    #[cfg(target_arch = "x86_64")]
    pub fn calibrate(&mut self, inputs: &[TokenSequence]) -> Result<(), EncoderError> {
        if !self.precision.centered() {
            return Err(EncoderError::Unsupported(
                "calibration quantizes a centered INT8 preset".into(),
            ));
        }
        if self.statics {
            return Err(EncoderError::Unsupported(
                "the encoder is calibrated already".into(),
            ));
        }
        if inputs.is_empty()
            || inputs
                .iter()
                .any(|(ids, types)| ids.len() < 2 || types.len() != ids.len())
        {
            return Err(EncoderError::Input(
                "static centers need sequences of at least two tokens, a segment type each".into(),
            ));
        }
        let ids: Vec<i32> = inputs
            .iter()
            .flat_map(|(ids, _)| ids.iter().copied())
            .collect();
        let types: Vec<i32> = inputs
            .iter()
            .flat_map(|(_, types)| types.iter().copied())
            .collect();
        let lengths: Vec<usize> = inputs.iter().map(|(ids, _)| ids.len()).collect();
        let tokens = checked_input(&self.shape, &ids, &types, &lengths)?;
        let mut own = if tokens > self.max_tokens {
            Some(Workspace::new(&self.shape, tokens)?)
        } else {
            None
        };
        let firsts: Vec<usize> = lengths
            .iter()
            .scan(0, |at, &len| {
                *at += len;
                Some(*at - len)
            })
            .collect();
        let lasts: Vec<usize> = firsts
            .iter()
            .zip(&lengths)
            .map(|(&first, &len)| first + len - 1)
            .collect();
        let row_classes: Vec<u32> = lengths
            .iter()
            .flat_map(|&len| (0..len).map(move |t| restored_class(t, len) as u32))
            .collect();
        let execution = ExecutionOptions {
            kernel: self.kernel,
            integer_attention: false,
            half_residual: false,
            keep_norm_inputs: true,
            static_keys: false,
        };
        let (h, i, heads) = (self.shape.hidden, self.shape.intermediate, self.shape.heads);
        let (precision, integer) = (self.precision, self.precision.integer_attention());
        let means = |input: &[f32], width: usize| -> Result<Vec<f64>, EncoderError> {
            let mut means = class_means(input, width, &firsts, &lasts);
            let Some(between) = middle_mean(input, width, &firsts, &lengths) else {
                return Err(EncoderError::Input(
                    "static centers need inputs with tokens between their first and last".into(),
                ));
            };
            means.extend(between);
            Ok(means)
        };
        let gained = |input: &NormInput, norm: &Norm| -> (Vec<f32>, Vec<f32>) {
            (
                input.rows[..tokens * h]
                    .chunks_exact(h)
                    .flat_map(|row| row.iter().zip(&norm.gamma).map(|(&z, &gamma)| z * gamma))
                    .collect(),
                input.rstd[..tokens]
                    .iter()
                    .map(|&rstd| rstd * rstd)
                    .collect(),
            )
        };
        let positions = self.shape.position_ids(&ids, &lengths);
        let Family::Bert(bert) = &mut self.family else {
            return Err(EncoderError::Unsupported(
                "calibration quantizes a centered INT8 preset".into(),
            ));
        };
        let (embeddings, layers, relative) =
            (&bert.embeddings, &bert.layers, bert.relative.as_ref());
        let w = match &mut own {
            Some(work) => work,
            None => &mut self.work,
        };
        let pool = pool_for(self.pool.as_ref(), tokens);
        // Every layer builds before any installs, so a failure installs none.
        let built = pool.install(|| -> Result<Vec<Calibrated>, EncoderError> {
            embeddings.apply(&ids, &types, &positions, &mut w.hidden[..tokens * h], false);
            let mut built = Vec::with_capacity(layers.len());
            for (index, layer) in layers.iter().enumerate() {
                let quantize = |projection: &Projection,
                                kind: Kind,
                                rows: &[f32],
                                weights: Option<&[f32]>,
                                norm: Option<DeferredNorm<'_>>| {
                    let means = means(rows, projection.k())?;
                    let operand = Operand {
                        rows,
                        classes: &row_classes,
                        means: &means,
                        weights,
                    };
                    projection.calibrated(index, kind, precision, operand, norm)
                };
                // Past layer 0, QKV defers the previous second LayerNorm.
                let qkv = match index.checked_sub(1).map(|last| &layers[last].norm2) {
                    None => quantize(&layer.qkv, Kind::Qkv, &w.hidden[..tokens * h], None, None)?,
                    Some(norm) => {
                        let (rows, weights) = gained(&w.norm2_input, norm);
                        let deferred = DeferredNorm {
                            gains: &norm.gamma,
                            beta: &norm.beta,
                            bias: &layer.qkv_bias,
                        };
                        quantize(&layer.qkv, Kind::Qkv, &rows, Some(&weights), Some(deferred))?
                    }
                };
                layer.apply(
                    execution, heads, relative, &lengths, None, tokens, w, &mut None,
                )?;
                let out = quantize(&layer.out, Kind::Out, &w.context[..tokens * h], None, None)?;
                let (rows, weights) = gained(&w.norm1_input, &layer.norm1);
                let deferred = DeferredNorm {
                    gains: &layer.norm1.gamma,
                    beta: &layer.norm1.beta,
                    bias: &layer.up_bias,
                };
                let up = quantize(&layer.up, Kind::Ffn1, &rows, Some(&weights), Some(deferred))?;
                let down = quantize(
                    &layer.down,
                    Kind::Ffn2,
                    &w.intermediate[..tokens * i],
                    None,
                    None,
                )?;
                let keys = integer.then(|| {
                    key_means(&w.qkv[..tokens * 3 * h], h, &firsts, &lengths)
                        .iter()
                        .map(|&mean| mean as f32)
                        .collect()
                });
                built.push(Calibrated {
                    projections: [qkv, out, up, down],
                    keys,
                });
            }
            Ok(built)
        })?;
        bert.install(built);
        self.set_calibrated();
        Ok(())
    }

    /// Calibration quantizes a centered INT8 preset, which needs x86-64.
    #[cfg(not(target_arch = "x86_64"))]
    pub fn calibrate(&mut self, _inputs: &[TokenSequence]) -> Result<(), EncoderError> {
        Err(EncoderError::Unsupported(
            "calibration quantizes a centered INT8 preset, which requires x86-64".into(),
        ))
    }

    /// The calibration as the bytes [`Self::install_calibration`] takes back.
    #[cfg(target_arch = "x86_64")]
    pub fn calibration(&self) -> Result<Vec<u8>, EncoderError> {
        let Some(bert) = self.calibrated() else {
            return Err(EncoderError::Unsupported(
                "calibration tables need a calibrated centered preset".into(),
            ));
        };
        let mut out = Vec::new();
        for layer in &bert.layers {
            for projection in [&layer.qkv, &layer.out, &layer.up, &layer.down] {
                projection.write_calibration(&mut out)?;
            }
            for key in layer.key_centers.iter().flatten() {
                out.extend(key.to_le_bytes());
            }
        }
        Ok(out)
    }

    /// Installs [`Self::calibration`]'s bytes without the forward pass. Malformed
    /// tables are refused and leave the encoder uncalibrated; once only.
    #[cfg(target_arch = "x86_64")]
    pub fn install_calibration(&mut self, tables: &[u8]) -> Result<(), EncoderError> {
        if !self.precision.centered() {
            return Err(EncoderError::Unsupported(
                "calibration quantizes a centered INT8 preset".into(),
            ));
        }
        if self.statics {
            return Err(EncoderError::Unsupported(
                "the encoder is calibrated already".into(),
            ));
        }
        let (precision, h) = (self.precision, self.shape.hidden);
        let Family::Bert(bert) = &mut self.family else {
            return Err(EncoderError::Unsupported(
                "calibration quantizes a centered INT8 preset".into(),
            ));
        };
        let layers = &bert.layers;
        let mut rest = tables;
        let mut built = Vec::with_capacity(layers.len());
        for (index, layer) in layers.iter().enumerate() {
            // Deferred as calibrate defers them.
            let qkv_gains = index
                .checked_sub(1)
                .map(|last| &layers[last].norm2.gamma[..]);
            let up_gains = Some(&layer.norm1.gamma[..]);
            let projections = [
                layer
                    .qkv
                    .installed(index, Kind::Qkv, precision, qkv_gains, &mut rest)?,
                layer
                    .out
                    .installed(index, Kind::Out, precision, None, &mut rest)?,
                layer
                    .up
                    .installed(index, Kind::Ffn1, precision, up_gains, &mut rest)?,
                layer
                    .down
                    .installed(index, Kind::Ffn2, precision, None, &mut rest)?,
            ];
            let keys = precision
                .integer_attention()
                .then(|| crate::int8::take_floats(&mut rest, h))
                .transpose()
                .map_err(|e| EncoderError::Input(format!("layer {index} key centers: {e}")))?;
            built.push(Calibrated { projections, keys });
        }
        if !rest.is_empty() {
            return Err(EncoderError::Input(
                "calibration tables run past this model's".into(),
            ));
        }
        bert.install(built);
        self.set_calibrated();
        Ok(())
    }

    /// From here on, encodes run the installed calibration.
    #[cfg(target_arch = "x86_64")]
    fn set_calibrated(&mut self) {
        self.statics = true;
        self.half_residual = self.precision.integer_attention();
    }

    /// The weights a calibrated centered preset runs: BERT's, the family it quantizes.
    #[cfg(target_arch = "x86_64")]
    fn calibrated(&self) -> Option<&Bert> {
        match &self.family {
            Family::Bert(bert) if self.precision.centered() && self.statics => Some(bert),
            _ => None,
        }
    }

    /// Calibrated parameters for bit-identical execution on another backend.
    #[cfg(target_arch = "x86_64")]
    pub fn export_reproducible(
        &self,
    ) -> Result<kodamaembed_model::repro::EncoderExport, EncoderError> {
        use kodamaembed_model::repro::{EncoderExport, LayerExport, NormExport};
        let mean_pooled = matches!(self.finish, Finish::Pool(Pooling::Mean));
        let Some(bert) = self.calibrated().filter(|_| mean_pooled) else {
            return Err(EncoderError::Unsupported(
                "exports need a calibrated centered mean-pooling encoder".into(),
            ));
        };
        let norm = |n: &Norm| NormExport {
            gamma: n.gamma.clone(),
            beta: n.beta.clone(),
            eps: n.eps,
        };
        let layers = bert
            .layers
            .iter()
            .map(|layer| {
                Ok(LayerExport {
                    qkv: layer.qkv.export(&layer.qkv_bias)?,
                    key_centers: layer.key_centers.clone().ok_or_else(|| {
                        EncoderError::Unsupported("exports need static key centers".into())
                    })?,
                    out: layer.out.export(&layer.out_bias)?,
                    norm1: norm(&layer.norm1),
                    up: layer.up.export(&layer.up_bias)?,
                    down: layer.down.export(&layer.down_bias)?,
                    norm2: norm(&layer.norm2),
                })
            })
            .collect::<Result<Vec<_>, EncoderError>>()?;
        Ok(EncoderExport {
            shape: self.shape,
            word: bert.embeddings.word.clone(),
            position: bert.embeddings.position.clone(),
            token_type: bert.embeddings.token_type.clone(),
            embedding_norm: norm(&bert.embeddings.norm),
            layers,
            constants_f32: crate::int8::repro_constants(),
            attention_weights: crate::attention::weight_table().to_vec(),
        })
    }

    /// Encodes sequences packed back to back by `lengths`, [`Encoder::output_width`]
    /// values each; no sequence depends on another in the batch.
    pub fn encode(
        &mut self,
        ids: &[i32],
        types: &[i32],
        lengths: &[usize],
    ) -> Result<Vec<f32>, EncoderError> {
        let tokens = checked_tokens(&self.shape, self.max_tokens, ids, types, lengths)?;
        let (h, mut out) = (
            self.shape.hidden,
            vec![0.0; lengths.len() * self.output_width()],
        );
        let mut timer = self.profile.is_some().then(|| Timer {
            last: Instant::now(),
            milliseconds: [0.0; STAGES.len()],
        });
        let (kernel, heads) = (self.kernel, self.shape.heads);
        let execution = ExecutionOptions {
            kernel,
            integer_attention: self.precision.integer_attention(),
            half_residual: self.half_residual,
            keep_norm_inputs: false,
            static_keys: true,
        };
        let centered = self.precision.centered();
        if centered && !self.statics {
            return Err(EncoderError::Unsupported(
                "centered INT8 requires calibrated static centers".into(),
            ));
        }
        let row_classes: Option<Vec<u32>> = centered.then(|| {
            lengths
                .iter()
                .flat_map(|&len| (0..len).map(move |t| restored_class(t, len) as u32))
                .collect()
        });
        let positions = self.shape.position_ids(ids, lengths);
        let (shape, family, finish, w) =
            (&self.shape, &mut self.family, &self.finish, &mut self.work);
        pool_for(self.pool.as_ref(), tokens).install(|| -> Result<(), EncoderError> {
            match family {
                Family::Bert(bert) => {
                    bert.embeddings.apply(
                        ids,
                        types,
                        &positions,
                        &mut w.hidden[..tokens * h],
                        execution.half_residual,
                    );
                    mark(&mut timer, Stage::Embed);
                    let (relative, restored) = (bert.relative.as_ref(), row_classes.as_deref());
                    for layer in &bert.layers {
                        layer.apply(
                            execution, heads, relative, lengths, restored, tokens, w, &mut timer,
                        )?;
                    }
                }
                Family::Modern(modern) => {
                    modern.forward(kernel, shape, ids, &positions, lengths, w, &mut timer);
                }
            }
            finish.apply(kernel, &w.hidden[..tokens * h], h, lengths, &mut out);
            mark(&mut timer, Stage::Pool);
            Ok(())
        })?;
        if out.iter().any(|value| !value.is_finite()) {
            return Err(EncoderError::Weights(
                "encoder produced non-finite outputs; checkpoint values exceed backend numerical range".into(),
            ));
        }
        if let (Some(profile), Some(timer)) = (&mut self.profile, timer) {
            for (total, ms) in profile.milliseconds.iter_mut().zip(timer.milliseconds) {
                *total += ms;
            }
            profile.batches += 1;
            profile.tokens += tokens;
        }
        Ok(out)
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod centered_tests;

#[cfg(test)]
mod worker_budget_tests {
    use super::*;
    use kodamaembed_model::Positions;

    struct Weights;
    impl Tensors for Weights {
        fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
            Ok(if name.contains("LayerNorm.weight") {
                vec![1.0; len]
            } else {
                (0..len).map(|i| ((i % 23) as f32 - 11.0) * 0.005).collect()
            })
        }
    }

    fn encoder(output: Output, workers: Option<NonZeroUsize>) -> Encoder {
        let shape = Shape {
            layers: 1,
            hidden: 32,
            heads: 4,
            intermediate: 64,
            vocab: 8,
            max_positions: 16,
            type_vocab: 2,
            eps: 1e-5,
            positions: Positions::Sequential,
        };
        let capacity = 2 * SHORT_TOKENS;
        match workers {
            Some(workers) => Encoder::new_with_precision_and_workers(
                Kernel::Portable,
                shape,
                output,
                capacity,
                &Weights,
                CpuPrecision::Fp32,
                workers,
            ),
            None => Encoder::new(Kernel::Portable, shape, output, capacity, &Weights),
        }
        .unwrap()
    }

    const OUTPUTS: [Output; 3] = [
        Output::Embedding(Pooling::Mean),
        Output::Embedding(Pooling::Cls),
        Output::Relevance(Head::Pooler),
    ];

    #[test]
    fn dedicated_pools_preserve_short_panel_and_multihead_arithmetic_and_reuse() {
        for output in OUTPUTS {
            let mut reference = encoder(output, None);
            for count in [1, 2] {
                let workers = NonZeroUsize::new(count).unwrap();
                let mut actual = encoder(output, Some(workers));
                assert_eq!(actual.threads(), count);
                for lengths in [&[1][..], &[1, 3, 2][..], &[16, 16, 1][..], &[1][..]] {
                    let tokens = lengths.iter().sum::<usize>();
                    let ids: Vec<i32> = (0..tokens).map(|i| (i % 8) as i32).collect();
                    let types = vec![0; tokens];
                    let expected = reference.encode(&ids, &types, lengths).unwrap();
                    let observed = actual.encode(&ids, &types, lengths).unwrap();
                    assert_eq!(
                        observed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    );
                }
                assert!(actual.encode(&[8], &[0], &[1]).is_err());
                assert_eq!(
                    actual
                        .encode(&[1], &[0], &[1])
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    reference
                        .encode(&[1], &[0], &[1])
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                );
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn class_means_average_first_and_last_rows_in_f64() {
        let rows = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(
            class_means(&rows, 2, &[0, 2], &[1, 3]),
            [3.0, 4.0, 5.0, 6.0]
        );
    }

    #[test]
    fn short_batches_run_on_one_thread_per_performance_core() {
        let all = POOL.current_num_threads();
        let short = SHORT_POOL.as_ref().unwrap_or(&POOL).current_num_threads();
        let cores = crate::workers::allowed_cores();
        eprintln!("allowed cores {cores:?}, short-batch threads {short} of {all}");
        let rule = cores.and_then(|cores| crate::workers::short_pool_threads(cores, all));
        assert_eq!(short, rule.unwrap_or(all));
        assert_eq!(pool_for(None, 1).current_num_threads(), short);
        assert_eq!(pool_for(None, SHORT_TOKENS).current_num_threads(), short);
        assert_eq!(pool_for(None, SHORT_TOKENS + 1).current_num_threads(), all);
        let own = crate::worker_pool("own", 3).unwrap();
        assert_eq!(pool_for(Some(&own), 1).current_num_threads(), 3);
        assert_eq!(
            pool_for(Some(&own), SHORT_TOKENS + 1).current_num_threads(),
            3
        );
    }

    #[test]
    fn inputs_alone_match_their_rows_in_a_batch_past_the_short_threshold() {
        let lengths = [16; SHORT_TOKENS / 16 + 1];
        let tokens = lengths.iter().sum::<usize>();
        assert!(tokens > SHORT_TOKENS);
        let ids: Vec<i32> = (0..tokens).map(|i| (i * 5 % 8) as i32).collect();
        let types = vec![0; tokens];
        for output in OUTPUTS {
            let mut encoder = encoder(output, None);
            let batch = encoder.encode(&ids, &types, &lengths).unwrap();
            let width = batch.len() / lengths.len();
            for (s, row) in batch.chunks(width).enumerate() {
                let range = 16 * s..16 * (s + 1);
                let alone = encoder
                    .encode(&ids[range.clone()], &types[range], &[16])
                    .unwrap();
                assert_eq!(
                    alone.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{output:?} sequence {s}"
                );
            }
        }
    }

    #[test]
    fn dedicated_embedding_and_scoring_complete_with_all_legacy_workers_parked() {
        use std::sync::{Arc, Condvar, Mutex, mpsc};
        use std::thread::{self, JoinHandle};
        use std::time::Duration;

        struct Parked {
            state: Arc<(Mutex<(usize, bool)>, Condvar)>,
            job: Option<JoinHandle<()>>,
        }
        impl Parked {
            fn release_and_join(&mut self) -> thread::Result<()> {
                let (lock, wake) = &*self.state;
                lock.lock().unwrap_or_else(|e| e.into_inner()).1 = true;
                wake.notify_all();
                self.job.take().map_or(Ok(()), |job| job.join())
            }
        }
        impl Drop for Parked {
            fn drop(&mut self) {
                // Runs before any assertion, even on a timeout or panic.
                let _ = self.release_and_join();
            }
        }

        let workers = NonZeroUsize::new(1).unwrap();
        let mut embedding = encoder(Output::Embedding(Pooling::Mean), Some(workers));
        let mut scoring = encoder(Output::Relevance(Head::Pooler), Some(workers));
        let expected_embedding = encoder(Output::Embedding(Pooling::Mean), None)
            .encode(&[1, 2, 3], &[0, 0, 0], &[1, 2])
            .unwrap();
        let expected_scoring = encoder(Output::Relevance(Head::Pooler), None)
            .encode(&[1, 2, 3], &[0, 0, 0], &[1, 2])
            .unwrap();
        let count = POOL.current_num_threads();
        let state = Arc::new((Mutex::new((0_usize, false)), Condvar::new()));
        let worker_state = Arc::clone(&state);
        let job = thread::spawn(move || {
            POOL.broadcast(|_| {
                let (lock, wake) = &*worker_state;
                let mut state = lock.lock().unwrap_or_else(|e| e.into_inner());
                state.0 += 1;
                wake.notify_all();
                while !state.1 {
                    state = wake.wait(state).unwrap_or_else(|e| e.into_inner());
                }
            });
        });
        let mut parked = Parked {
            state,
            job: Some(job),
        };
        let all_parked = {
            let (lock, wake) = &*parked.state;
            let state = lock.lock().unwrap_or_else(|e| e.into_inner());
            let (state, _) = wake
                .wait_timeout_while(state, Duration::from_secs(10), |state| state.0 < count)
                .unwrap_or_else(|e| e.into_inner());
            state.0 == count
        };
        if !all_parked {
            let cleanup = parked.release_and_join();
            assert!(cleanup.is_ok(), "park job failed");
            panic!("workers never parked");
        }
        let (send, receive) = mpsc::sync_channel(1);
        let actual = thread::spawn(move || {
            let embeddings = embedding.encode(&[1, 2, 3], &[0, 0, 0], &[1, 2]);
            let scores = scoring.encode(&[1, 2, 3], &[0, 0, 0], &[1, 2]);
            let _ = send.send((embeddings, scores));
        });
        // A deadlock watchdog.
        let completed_while_parked = receive.recv_timeout(Duration::from_secs(10));
        let cleanup = parked.release_and_join();
        let worker_result = actual.join();
        assert!(cleanup.is_ok(), "park job failed");
        assert!(worker_result.is_ok(), "explicit-pool worker panicked");
        let (embeddings, scores) = completed_while_parked.expect("used parked pool");
        for (observed, expected) in [
            (embeddings.unwrap(), expected_embedding),
            (scores.unwrap(), expected_scoring),
        ] {
            assert_eq!(
                observed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            );
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod norm_tests {
    use super::{Norm, NormInput, round_half};

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// Keeping a LayerNorm's input changes nothing it writes.
    #[test]
    fn kept_inputs_are_the_rows_and_statistics_the_norm_used() {
        let (width, tokens) = (128, 70);
        let norm = Norm {
            gamma: (0..width).map(|t| 0.5 + t as f32 / 97.0).collect(),
            beta: (0..width).map(|t| (t % 7) as f32 / 9.0 - 0.3).collect(),
            eps: 1e-12,
        };
        let mut rows: Vec<f32> = (0..tokens * width)
            .map(|at| ((at * 37 % 211) as f32 - 100.0) / 13.0 + 0.001)
            .collect();
        for (t, x) in rows[3 * width..4 * width].iter_mut().enumerate() {
            *x = if t % 2 == 0 { 3e19 } else { -3e19 };
        }
        let halves: &[bool] = if std::is_x86_feature_detected!("f16c") {
            &[false, true]
        } else {
            &[false]
        };
        for &half in halves {
            let mut plain = rows.clone();
            norm.apply_rows(&mut plain, half);
            let mut kept = rows.clone();
            let mut input = NormInput::default();
            norm.apply_rows_keeping(&mut kept, half, &mut input);
            assert_eq!(bits(&kept), bits(&plain), "half {half}");
            let mut read = rows.clone();
            round_half(&mut read, half);
            assert_eq!(bits(&input.rows), bits(&read), "half {half}");
            assert_eq!((input.mean.len(), input.rstd.len()), (tokens, tokens));
            for (r, row) in read.chunks_exact(width).enumerate() {
                let (mean, rstd) = norm.moments(row).unwrap_or((f32::NAN, f32::NAN));
                assert_eq!(
                    input.mean[r].to_bits(),
                    mean.to_bits(),
                    "half {half} row {r}"
                );
                assert_eq!(
                    input.rstd[r].to_bits(),
                    rstd.to_bits(),
                    "half {half} row {r}"
                );
            }
            assert!(
                input.rstd[3].is_nan() && kept[3 * width..4 * width].iter().all(|x| x.is_nan())
            );
        }
    }
}

#[cfg(test)]
mod lane_sum_tests {
    use super::lane_sum;

    /// lane_sum as a warp computes it (repro.cu repro_layernorm_lanes).
    fn warp_sum<T: Copy + Default + std::ops::Add<Output = T>>(values: &[T]) -> [T; 32] {
        let mut lanes = [T::default(); 32];
        for (c, &x) in values.iter().enumerate() {
            lanes[c % 32] = lanes[c % 32] + x;
        }
        let mut width = 16;
        while width > 0 {
            let before = lanes;
            for (j, lane) in lanes.iter_mut().enumerate() {
                *lane = before[j] + before[j ^ width];
            }
            width /= 2;
        }
        lanes
    }

    fn values(seed: u64, n: usize) -> Vec<f64> {
        let mut x = seed;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 11) as f64 / (1_u64 << 53) as f64 - 0.5) * 2_f64.powi((x % 31) as i32 - 15)
            })
            .collect()
    }

    #[test]
    fn a_warps_butterfly_holds_lane_sums_total_in_every_lane() {
        for (seed, n) in [(3, 1024), (5, 384), (7, 1000), (11, 32), (13, 5)] {
            let wide = values(seed, n);
            let narrow: Vec<f32> = wide.iter().map(|&v| v as f32).collect();
            let total = lane_sum(&wide, |v| v);
            assert!(
                warp_sum(&wide)
                    .iter()
                    .all(|l| l.to_bits() == total.to_bits()),
                "f64 n {n}"
            );
            let total = lane_sum(&narrow, |v| v);
            assert!(
                warp_sum(&narrow)
                    .iter()
                    .all(|l| l.to_bits() == total.to_bits()),
                "f32 n {n}"
            );
        }
    }
}
