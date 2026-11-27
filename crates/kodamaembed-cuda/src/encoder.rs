//! The transformer encoder on CUDA: packed, padding-free batches through KodamaEmbed's kernels.

use crate::{
    driver::{Context, CudaError, DeviceBuffer, Event, Graph, HostBuffer, Stream},
    kernels::{
        CORRECTED_BELOW, Embeddings, Gemm, Kernels, Low, Packed, PairedGemms, PoolerHead,
        PredictionHead, PredictionKernels, RelativeTable, Tokens, Turned, Turns,
    },
};
use half::f16;
use kodamaembed_model::{
    Activation, Geometry, Head, Output, Pooling, Positions, Profile, Rotary, Shape, Stage, Tensors,
    fused_qkv, pair_gates, pair_turns, relative::RelativeBias, tensor,
};
use std::{
    fmt,
    time::{Duration, Instant},
};

/// The most sequences a batch holds: attention launches one grid row per
/// sequence along z, where CUDA allows at most this many blocks.
pub const MAX_SEQUENCES: usize = 65_535;

/// The most token rows a batch holds; the kernels count rows and tiles in 32 bits.
pub const MAX_TOKENS: usize = 65_535 * 128;

/// Tokens, longest sequence, sequences and whether one is corrected: a batch's launch parameters.
type Batch = (usize, usize, usize, bool);

/// Batch shapes whose recorded launches are kept. A shape is recorded on its second sight,
/// beside that sight's direct launches: a recording adds to its call and holds GPU memory.
const GRAPHS: usize = 32;

/// Batches poll this long before a blocking wait, which wakes late; single sequences poll to the end.
const SPIN: Duration = Duration::from_millis(5);

/// Input sections start at multiples of 64 words, as views start at multiples of 256 bytes.
const SECTION: usize = 64;

#[derive(Debug)]
pub enum EncoderError {
    Cuda(CudaError),
    Unsupported(String),
    Weights(String),
    Input(String),
}

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(error) => write!(f, "{error}"),
            Self::Unsupported(reason) => write!(f, "unsupported model: {reason}"),
            Self::Weights(reason) => write!(f, "weights: {reason}"),
            Self::Input(reason) => write!(f, "input: {reason}"),
        }
    }
}

impl std::error::Error for EncoderError {}

impl From<CudaError> for EncoderError {
    fn from(error: CudaError) -> Self {
        // No binary for this GPU, no JIT compiler, or a driver too old for the shipped PTX.
        if error.call == "cuModuleLoadData" && matches!(error.code, 209 | 221 | 222) {
            Self::Unsupported(error.to_string())
        } else {
            Self::Cuda(error)
        }
    }
}

/// A BERT layer: a norm after attention and after the feed-forward.
struct Layer {
    qkv: DeviceBuffer<f16>,
    out: DeviceBuffer<f16>,
    ffn1: DeviceBuffer<f16>,
    ffn2: DeviceBuffer<f16>,
    qkv_b: DeviceBuffer<f32>,
    out_b: DeviceBuffer<f32>,
    ln1_g: DeviceBuffer<f32>,
    ln1_b: DeviceBuffer<f32>,
    ffn1_b: DeviceBuffer<f32>,
    ffn2_b: DeviceBuffer<f32>,
    ln2_g: DeviceBuffer<f32>,
    ln2_b: DeviceBuffer<f32>,
}

/// A ModernBERT layer: a norm before attention and before the gated feed-forward, no biases.
struct ModernLayer {
    /// None in the first layer, which attends over the embedding norm's rows as they are.
    attention_norm: Option<DeviceBuffer<f32>>,
    /// The queries' and keys' rows paired as the turn takes them, then the values'.
    qkv: DeviceBuffer<f16>,
    out: DeviceBuffer<f16>,
    feed_forward_norm: DeviceBuffer<f32>,
    /// The up projection: each input's row, then its gate's.
    up: DeviceBuffer<f16>,
    down: DeviceBuffer<f16>,
}

/// BERT, RoBERTa and MPNet: tables for positions and segments, and layers that norm afterwards.
struct Bert {
    word: DeviceBuffer<f32>,
    position: DeviceBuffer<f32>,
    token_type: DeviceBuffer<f32>,
    embed_g: DeviceBuffer<f32>,
    embed_b: DeviceBuffer<f32>,
    relative: Option<RelativeTable>,
    layers: Vec<Layer>,
}

/// ModernBERT: positions turn the queries and keys, and layers norm beforehand.
struct Modern {
    rotary: Rotary,
    /// The GEMMs of the turned and the gated projections, which no other family launches.
    paired: PairedGemms,
    word: DeviceBuffer<f32>,
    embed_g: DeviceBuffer<f32>,
    layers: Vec<ModernLayer>,
    final_g: DeviceBuffer<f32>,
    /// The turns of the layers over whole sequences, and of the windowed ones.
    global_turns: Turns,
    local_turns: Turns,
    /// What every projection adds as its bias: the family has none.
    no_bias: DeviceBuffer<f32>,
    /// What every norm adds as its shift, and every token as its position and segment: the
    /// family has none of them.
    zero_row: DeviceBuffer<f32>,
}

/// A model family's weights on the device, with the kernels only it launches.
enum Family {
    Bert(Box<Bert>),
    Modern(Box<Modern>),
}

/// A checkpoint's tensors on their way to the device.
struct Upload<'a> {
    context: &'a Context,
    tensors: &'a dyn Tensors,
}

impl Upload<'_> {
    fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, EncoderError> {
        tensor(self.tensors, name, len).map_err(EncoderError::Weights)
    }

    fn fp32(&self, name: &str, len: usize) -> Result<DeviceBuffer<f32>, EncoderError> {
        Ok(DeviceBuffer::from_slice(
            self.context,
            &self.get(name, len)?,
        )?)
    }

    /// A matrix in FP16, as the GEMMs read it.
    fn half(&self, matrix: &[f32]) -> Result<DeviceBuffer<f16>, EncoderError> {
        Ok(DeviceBuffer::from_slice(self.context, &to_half(matrix)?)?)
    }

    fn matrix(&self, name: &str, len: usize) -> Result<DeviceBuffer<f16>, EncoderError> {
        self.half(&self.get(name, len)?)
    }
}

/// What one batch's launches share.
struct Pass<'a> {
    kernels: &'a Kernels,
    work: &'a Workspace,
    tokens: &'a Tokens<'a>,
    packed: &'a Packed<'a>,
    /// Whether a sequence is short enough for the corrected attention.
    corrected: bool,
    shape: &'a Shape,
}

/// Records a stage's end on the stream when the encoder profiles.
type Mark<'a> = dyn FnMut(Stage) -> Result<(), CudaError> + 'a;

impl Bert {
    fn load(upload: &Upload, shape: &Shape) -> Result<Self, EncoderError> {
        let (h, i) = (shape.hidden, shape.intermediate);
        let mut layers = Vec::with_capacity(shape.layers);
        for layer in 0..shape.layers {
            let at = |suffix: &str| format!("encoder.layer.{layer}.{suffix}");
            let (qkv_w, qkv_b) =
                fused_qkv(upload.tensors, shape, layer).map_err(EncoderError::Weights)?;
            layers.push(Layer {
                qkv: upload.half(&qkv_w)?,
                out: upload.matrix(&at("attention.output.dense.weight"), h * h)?,
                ffn1: upload.matrix(&at("intermediate.dense.weight"), i * h)?,
                ffn2: upload.matrix(&at("output.dense.weight"), h * i)?,
                qkv_b: DeviceBuffer::from_slice(upload.context, &qkv_b)?,
                out_b: upload.fp32(&at("attention.output.dense.bias"), h)?,
                ln1_g: upload.fp32(&at("attention.output.LayerNorm.weight"), h)?,
                ln1_b: upload.fp32(&at("attention.output.LayerNorm.bias"), h)?,
                ffn1_b: upload.fp32(&at("intermediate.dense.bias"), i)?,
                ffn2_b: upload.fp32(&at("output.dense.bias"), h)?,
                ln2_g: upload.fp32(&at("output.LayerNorm.weight"), h)?,
                ln2_b: upload.fp32(&at("output.LayerNorm.bias"), h)?,
            });
        }
        Ok(Self {
            word: upload.fp32("embeddings.word_embeddings.weight", shape.vocab * h)?,
            position: upload.fp32(
                "embeddings.position_embeddings.weight",
                shape.max_positions * h,
            )?,
            token_type: upload.fp32(
                "embeddings.token_type_embeddings.weight",
                shape.type_vocab * h,
            )?,
            embed_g: upload.fp32("embeddings.LayerNorm.weight", h)?,
            embed_b: upload.fp32("embeddings.LayerNorm.bias", h)?,
            relative: RelativeBias::load(shape, upload.tensors)
                .map_err(EncoderError::Weights)?
                .map(|bias| RelativeTable::upload(upload.context, &bias))
                .transpose()?,
            layers,
        })
    }

    /// Embeds the batch and runs every layer, which leaves the encoded rows in `hidden32`.
    ///
    /// # Safety
    /// As for the kernels: every buffer outlives the stream's work, and the batch's ids, types
    /// and positions index their tables.
    unsafe fn launch(&self, s: &Stream, pass: &Pass, mark: &mut Mark) -> Result<(), CudaError> {
        let (k, work, packed) = (pass.kernels, pass.work, pass.packed);
        let (h, i, eps) = (pass.shape.hidden, pass.shape.intermediate, pass.shape.eps);
        let tokens = packed.tokens;
        let embeddings = Embeddings {
            word: &self.word,
            position: &self.position,
            token_type: &self.token_type,
            gamma: &self.embed_g,
            beta: &self.embed_b,
        };
        // SAFETY: forwarded from the caller.
        unsafe {
            k.embed_layernorm(
                s,
                &embeddings,
                pass.tokens,
                &work.hidden16,
                &work.hidden32,
                eps,
            )?;
            mark(Stage::Embed)?;
            for layer in &self.layers {
                let qkv = Gemm {
                    a: &work.hidden16,
                    w: &layer.qkv,
                    bias: &layer.qkv_b,
                    m: tokens,
                    n: 3 * h,
                    k: h,
                };
                let low = Low {
                    out: &work.qk_low,
                    columns: 2 * h,
                };
                // Only corrected sequences read Q and K's low parts.
                k.gemm_bias(s, &qkv, &work.qkv, pass.corrected.then_some(&low))?;
                mark(Stage::Qkv)?;
                let relative = self.relative.as_ref();
                k.attention(s, &work.qkv, &work.qk_low, packed, &work.context, relative)?;
                mark(Stage::Attention)?;
                let out = Gemm {
                    a: &work.context,
                    w: &layer.out,
                    bias: &layer.out_b,
                    m: tokens,
                    n: h,
                    k: h,
                };
                k.gemm_bias_residual(s, &out, &work.hidden32, &work.sum)?;
                mark(Stage::Out)?;
                k.layernorm(
                    s,
                    &work.sum,
                    &layer.ln1_g,
                    &layer.ln1_b,
                    &work.hidden16,
                    Some(&work.hidden32),
                    tokens,
                    eps,
                )?;
                mark(Stage::Norm1)?;
                let up = Gemm {
                    a: &work.hidden16,
                    w: &layer.ffn1,
                    bias: &layer.ffn1_b,
                    m: tokens,
                    n: i,
                    k: h,
                };
                k.gemm_bias_gelu(s, &up, &work.intermediate)?;
                mark(Stage::Ffn1)?;
                let down = Gemm {
                    a: &work.intermediate,
                    w: &layer.ffn2,
                    bias: &layer.ffn2_b,
                    m: tokens,
                    n: h,
                    k: i,
                };
                k.gemm_bias_residual(s, &down, &work.hidden32, &work.sum)?;
                mark(Stage::Ffn2)?;
                k.layernorm(
                    s,
                    &work.sum,
                    &layer.ln2_g,
                    &layer.ln2_b,
                    &work.hidden16,
                    Some(&work.hidden32),
                    tokens,
                    eps,
                )?;
                mark(Stage::Norm2)?;
            }
        }
        Ok(())
    }
}

impl Modern {
    fn load(
        upload: &Upload,
        shape: &Shape,
        rotary: Rotary,
        geometry: Geometry,
        max_tokens: usize,
    ) -> Result<Self, EncoderError> {
        let (h, i, context) = (shape.hidden, shape.intermediate, upload.context);
        let head = geometry.head_dim();
        let mut layers = Vec::with_capacity(shape.layers);
        for layer in 0..shape.layers {
            let at = |suffix: &str| format!("layers.{layer}.{suffix}");
            let qkv = upload.get(&at("attn.Wqkv.weight"), 3 * h * h)?;
            let up = upload.get(&at("mlp.Wi.weight"), 2 * i * h)?;
            layers.push(ModernLayer {
                attention_norm: match layer {
                    0 => None,
                    _ => Some(upload.fp32(&at("attn_norm.weight"), h)?),
                },
                qkv: upload.half(&pair_turns(&qkv, h, head, 2 * h))?,
                out: upload.matrix(&at("attn.Wo.weight"), h * h)?,
                feed_forward_norm: upload.fp32(&at("mlp_norm.weight"), h)?,
                up: upload.half(&pair_gates(&up, h))?,
                down: upload.matrix(&at("mlp.Wo.weight"), h * i)?,
            });
        }
        let positions = max_tokens.min(shape.max_sequence_len());
        let turns = |theta: f32| Turns::upload(context, head, theta, positions);
        Ok(Self {
            rotary,
            paired: PairedGemms::load(context)?,
            word: upload.fp32("embeddings.tok_embeddings.weight", shape.vocab * h)?,
            embed_g: upload.fp32("embeddings.norm.weight", h)?,
            layers,
            final_g: upload.fp32("final_norm.weight", h)?,
            global_turns: turns(rotary.global_theta)?,
            local_turns: turns(rotary.local_theta)?,
            no_bias: DeviceBuffer::zeroed(context, (3 * h).max(2 * i))?,
            zero_row: DeviceBuffer::zeroed(context, h)?,
        })
    }

    /// `m` rows of `a` through the `n` rows of `w`, `k` values each, with no bias.
    fn projection<'a>(
        &'a self,
        a: &'a DeviceBuffer<f16>,
        w: &'a DeviceBuffer<f16>,
        [m, n, k]: [usize; 3],
    ) -> Gemm<'a> {
        Gemm {
            a,
            w,
            bias: &self.no_bias,
            m,
            n,
            k,
        }
    }

    /// As [`Bert::launch`]. Between layers the residual rows are in `sum`, and between a layer's
    /// attention and its feed-forward in `hidden32`.
    ///
    /// # Safety
    /// As for [`Bert::launch`], with every token in segment 0.
    unsafe fn launch(&self, s: &Stream, pass: &Pass, mark: &mut Mark) -> Result<(), CudaError> {
        let (k, work, packed) = (pass.kernels, pass.work, pass.packed);
        let (h, i, eps) = (pass.shape.hidden, pass.shape.intermediate, pass.shape.eps);
        let tokens = packed.tokens;
        // A token's row takes no position and no segment: each reads the zero row, which the
        // segment ids, all 0, index in both tables.
        let embeddings = Embeddings {
            word: &self.word,
            position: &self.zero_row,
            token_type: &self.zero_row,
            gamma: &self.embed_g,
            beta: &self.zero_row,
        };
        let unplaced = Tokens {
            ids: pass.tokens.ids,
            types: pass.tokens.types,
            positions: pass.tokens.types,
            count: tokens,
        };
        let projection = |a, w, n, k| self.projection(a, w, [tokens, n, k]);
        // SAFETY: forwarded from the caller.
        unsafe {
            k.embed_layernorm(s, &embeddings, &unplaced, &work.hidden16, &work.sum, eps)?;
            mark(Stage::Embed)?;
            for (index, layer) in self.layers.iter().enumerate() {
                if let Some(gamma) = &layer.attention_norm {
                    k.layernorm(
                        s,
                        &work.sum,
                        gamma,
                        &self.zero_row,
                        &work.hidden16,
                        None,
                        tokens,
                        eps,
                    )?;
                }
                mark(Stage::Norm1)?;
                let reach = self.rotary.reach(index);
                let low = Low {
                    out: &work.qk_low,
                    columns: 2 * h,
                };
                // A window's attention corrects at every length, a whole sequence's only below
                // the corrected length.
                let low = (reach.is_some() || pass.corrected).then_some(&low);
                let turned = Turned {
                    positions: pass.tokens.positions,
                    turns: match reach {
                        Some(_) => &self.local_turns,
                        None => &self.global_turns,
                    },
                    columns: 2 * h,
                    longest: packed.max_len,
                };
                let qkv = projection(&work.hidden16, &layer.qkv, 3 * h, h);
                self.paired
                    .gemm_bias_turned(s, &qkv, &work.qkv, low, &turned)?;
                mark(Stage::Qkv)?;
                match reach {
                    Some(reach) => {
                        k.local_attention(s, &work.qkv, &work.qk_low, packed, &work.context, reach)?
                    }
                    None => k.attention(s, &work.qkv, &work.qk_low, packed, &work.context, None)?,
                }
                mark(Stage::Attention)?;
                let out = projection(&work.context, &layer.out, h, h);
                k.gemm_bias_residual(s, &out, &work.sum, &work.hidden32)?;
                mark(Stage::Out)?;
                k.layernorm(
                    s,
                    &work.hidden32,
                    &layer.feed_forward_norm,
                    &self.zero_row,
                    &work.hidden16,
                    None,
                    tokens,
                    eps,
                )?;
                mark(Stage::Norm2)?;
                let up = projection(&work.hidden16, &layer.up, 2 * i, h);
                self.paired.gemm_bias_geglu(s, &up, &work.intermediate)?;
                mark(Stage::Ffn1)?;
                let down = projection(&work.intermediate, &layer.down, h, i);
                k.gemm_bias_residual(s, &down, &work.hidden32, &work.sum)?;
                mark(Stage::Ffn2)?;
            }
            // The final norm's FP16 rows go unread.
            k.layernorm(
                s,
                &work.sum,
                &self.final_g,
                &self.zero_row,
                &work.hidden16,
                Some(&work.hidden32),
                tokens,
                eps,
            )
        }
    }
}

/// A cross-encoder's head and classifier on the device.
enum Scorer {
    /// BERT's pooler: a biased dense layer and tanh over the first row.
    Pooler {
        pooler_w: DeviceBuffer<f32>,
        pooler_b: DeviceBuffer<f32>,
        classifier_w: DeviceBuffer<f32>,
        classifier_b: DeviceBuffer<f32>,
    },
    /// ModernBERT's prediction head: a dense layer, an activation and a norm over the pooled rows.
    Prediction {
        /// The head's kernels, which no other head launches.
        kernels: PredictionKernels,
        pooling: Pooling,
        activation: Activation,
        dense_w: DeviceBuffer<f32>,
        norm_g: DeviceBuffer<f32>,
        classifier_w: DeviceBuffer<f32>,
        classifier_b: DeviceBuffer<f32>,
    },
}

/// What runs after the last layer.
enum Finish {
    /// L2-normalized pooled rows, `hidden` values per sequence.
    Pool(Pooling),
    /// A cross-encoder's head and classifier, one score per sequence.
    Score(Box<Scorer>),
}

/// A batch's first launch copies `staging` into `inputs`; its last copies outputs to `download`.
struct Transfers {
    staging: HostBuffer<i32>,
    inputs: DeviceBuffer<i32>,
    download: HostBuffer<f32>,
}

struct Workspace {
    hidden16: DeviceBuffer<f16>,
    hidden32: DeviceBuffer<f32>,
    qkv: DeviceBuffer<f16>,
    /// Low parts of Q and K, which correct the FP16 attention scores.
    qk_low: DeviceBuffer<f16>,
    context: DeviceBuffer<f16>,
    /// Projection residuals, then compact final outputs after the last norm.
    sum: DeviceBuffer<f32>,
    /// The feed-forward's activations, then mean pooling's scratch: never under `4 * hidden`
    /// halfs a token, which a batch of one-token sequences needs.
    intermediate: DeviceBuffer<f16>,
}

struct Profiler {
    events: Vec<Event>,
    stages: Vec<Stage>,
    profile: Profile,
}

impl Profiler {
    fn new(context: &Context, layers: usize) -> Result<Self, CudaError> {
        let events = (0..3 + 7 * layers)
            .map(|_| Event::new(context))
            .collect::<Result<_, _>>()?;
        Ok(Self {
            events,
            stages: Vec::with_capacity(2 + 7 * layers),
            profile: Profile::default(),
        })
    }

    fn begin(&mut self, stream: &Stream) -> Result<(), CudaError> {
        self.stages.clear();
        self.events[0].record(stream)
    }

    fn mark(&mut self, stream: &Stream, stage: Stage) -> Result<(), CudaError> {
        self.stages.push(stage);
        self.events[self.stages.len()].record(stream)
    }

    /// Adds the finished batch; the stream must have been synchronized.
    fn finish(&mut self, tokens: usize) -> Result<(), CudaError> {
        for (index, &stage) in self.stages.iter().enumerate() {
            let ms = self.events[index].elapsed_ms(&self.events[index + 1])?;
            self.profile.milliseconds[stage as usize] += f64::from(ms);
        }
        self.profile.batches += 1;
        self.profile.tokens += tokens;
        Ok(())
    }
}

/// A model on one GPU. It may move between threads: every call that reaches
/// the GPU first makes the encoder's context current on the calling thread.
pub struct Encoder {
    /// Recorded batches, least recently used first; declared first to drop before what they name.
    graphs: Vec<(Batch, Graph)>,
    /// Shapes seen once and not recorded, oldest first.
    seen: Vec<Batch>,
    shape: Shape,
    max_tokens: usize,
    kernels: Kernels,
    stream: Stream,
    /// Records graphs; unordered with the legacy stream, so no other thread's copy meets a capture.
    recording: Stream,
    /// Polled for a batch's completion.
    done: Event,
    context: Context,
    profiler: Option<Profiler>,
    family: Family,
    finish: Finish,
    transfers: Transfers,
    work: Workspace,
}

fn to_half(values: &[f32]) -> Result<Vec<f16>, EncoderError> {
    values
        .iter()
        .copied()
        .map(|value| {
            let half = f16::from_f32(value);
            if !half.is_finite() {
                Err(EncoderError::Weights(format!(
                    "matrix value {value} is not finite in FP16"
                )))
            } else {
                Ok(half)
            }
        })
        .collect()
}

impl Encoder {
    pub fn new(
        context: &Context,
        shape: Shape,
        output: Output,
        max_tokens: usize,
        tensors: &dyn Tensors,
    ) -> Result<Self, EncoderError> {
        shape.validate().map_err(EncoderError::Unsupported)?;
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
            || shape
                .layers
                .checked_mul(7)
                .and_then(|events| events.checked_add(3))
                .is_none()
        {
            return Err(EncoderError::Unsupported(
                "layer count exceeds addressable storage".into(),
            ));
        }
        if max_tokens == 0 || max_tokens > MAX_TOKENS {
            return Err(EncoderError::Input(format!(
                "token capacity {max_tokens} must be in 1..={MAX_TOKENS}"
            )));
        }
        let geometry = shape.geometry().ok_or_else(|| {
            EncoderError::Unsupported(format!(
                "hidden {}, heads {}, intermediate {}",
                shape.hidden, shape.heads, shape.intermediate
            ))
        })?;
        let _current = context.enter()?;
        let (h, i) = (shape.hidden, shape.intermediate);
        let upload = Upload { context, tensors };
        let family = match rotary {
            Some(rotary) => Family::Modern(Box::new(Modern::load(
                &upload, &shape, rotary, geometry, max_tokens,
            )?)),
            None => Family::Bert(Box::new(Bert::load(&upload, &shape)?)),
        };
        let finish = match output {
            Output::Embedding(pooling) => Finish::Pool(pooling),
            Output::Relevance(Head::Pooler) => Finish::Score(Box::new(Scorer::Pooler {
                pooler_w: upload.fp32("pooler.dense.weight", h * h)?,
                pooler_b: upload.fp32("pooler.dense.bias", h)?,
                classifier_w: upload.fp32("classifier.weight", h)?,
                classifier_b: upload.fp32("classifier.bias", 1)?,
            })),
            Output::Relevance(Head::Prediction(pooling, activation)) => {
                Finish::Score(Box::new(Scorer::Prediction {
                    kernels: PredictionKernels::load(context, geometry)?,
                    pooling,
                    activation,
                    dense_w: upload.fp32("head.dense.weight", h * h)?,
                    norm_g: upload.fp32("head.norm.weight", h)?,
                    classifier_w: upload.fp32("classifier.weight", h)?,
                    classifier_b: upload.fp32("classifier.bias", 1)?,
                }))
            }
        };
        let t = max_tokens;
        let sequences = t.min(MAX_SEQUENCES);
        let words = 3 * t.next_multiple_of(SECTION) + sequences + 1;
        let width = output.width(h);
        Ok(Self {
            graphs: Vec::new(),
            seen: Vec::new(),
            shape,
            max_tokens,
            kernels: Kernels::load(context, geometry)?,
            stream: Stream::new(context)?,
            recording: Stream::new_non_blocking(context)?,
            done: Event::untimed(context)?,
            context: context.clone(),
            profiler: None,
            family,
            finish,
            transfers: Transfers {
                staging: HostBuffer::zeroed(context, words)?,
                inputs: DeviceBuffer::zeroed(context, words)?,
                download: HostBuffer::zeroed(context, sequences * width)?,
            },
            work: Workspace {
                hidden16: DeviceBuffer::zeroed(context, t * h)?,
                hidden32: DeviceBuffer::zeroed(context, t * h)?,
                qkv: DeviceBuffer::zeroed(context, t * 3 * h)?,
                qk_low: DeviceBuffer::zeroed(context, t * 2 * h)?,
                context: DeviceBuffer::zeroed(context, t * h)?,
                sum: DeviceBuffer::zeroed(context, t * h)?,
                intermediate: DeviceBuffer::zeroed(context, t * i.max(4 * h))?,
            },
        })
    }

    /// The UUID of this encoder's device; needs no NVML.
    pub fn device_uuid(&self) -> Result<[u8; 16], CudaError> {
        self.context.device_uuid()
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// The most sequences one batch may hold.
    pub fn max_sequences(&self) -> usize {
        MAX_SEQUENCES
    }

    /// Values returned per sequence: `hidden` for sentence vectors, 1 for scores.
    pub fn output_width(&self) -> usize {
        match self.finish {
            Finish::Pool(_) => self.shape.hidden,
            Finish::Score(_) => 1,
        }
    }

    /// Times every stage of every later batch; see [`Encoder::profile`].
    pub fn enable_profiling(&mut self) -> Result<(), EncoderError> {
        let _current = self.context.enter()?;
        self.profiler = Some(Profiler::new(&self.context, self.shape.layers)?);
        Ok(())
    }

    pub fn profile(&self) -> Option<&Profile> {
        self.profiler.as_ref().map(|profiler| &profiler.profile)
    }

    /// Encodes sequences of token `ids` and segment `types` packed back to back
    /// with the given `lengths`: [`Encoder::output_width`] values per sequence.
    pub fn encode(
        &mut self,
        ids: &[i32],
        types: &[i32],
        lengths: &[usize],
    ) -> Result<Vec<f32>, EncoderError> {
        let tokens = lengths
            .iter()
            .try_fold(0usize, |total, &length| total.checked_add(length))
            .ok_or_else(|| EncoderError::Input("sequence lengths overflow usize".into()))?;
        let max_len = lengths.iter().copied().max().unwrap_or(0);
        let min_len = lengths.iter().copied().min().unwrap_or(0);
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
        if tokens > self.max_tokens || max_len > self.shape.max_sequence_len() {
            return Err(EncoderError::Input(format!(
                "{tokens} tokens (limit {}), longest {max_len} (limit {})",
                self.max_tokens,
                self.shape.max_sequence_len()
            )));
        }
        if lengths.len() > MAX_SEQUENCES {
            return Err(EncoderError::Input(format!(
                "{} sequences in one batch (limit {MAX_SEQUENCES})",
                lengths.len()
            )));
        }
        let vocab = self.shape.vocab as i32;
        if let Some(bad) = ids.iter().find(|&&id| !(0..vocab).contains(&id)) {
            return Err(EncoderError::Input(format!(
                "token id {bad} outside the vocabulary"
            )));
        }
        let segments = self.shape.type_vocab as i32;
        if let Some(bad) = types.iter().find(|&&kind| !(0..segments).contains(&kind)) {
            return Err(EncoderError::Input(format!(
                "segment type {bad} outside the {segments} the model has"
            )));
        }
        let (sequences, outputs) = (lengths.len(), lengths.len() * self.output_width());
        let positions = self.shape.position_ids(ids, lengths);
        let cu: Vec<i32> = std::iter::once(0)
            .chain(lengths.iter().scan(0i32, |total, &len| {
                *total += len as i32;
                Some(*total)
            }))
            .collect();
        let _current = self.context.enter()?;
        let sections = tokens.next_multiple_of(SECTION);
        let staged = 3 * sections + sequences + 1;
        let staging = self.transfers.staging.as_mut_slice();
        staging[..tokens].copy_from_slice(ids);
        staging[sections..sections + tokens].copy_from_slice(types);
        staging[2 * sections..2 * sections + tokens].copy_from_slice(&positions);
        staging[3 * sections..staged].copy_from_slice(&cu);
        let transfers = &mut self.transfers;
        let inputs = &transfers.inputs;
        // SAFETY: sized for the capacity, `inputs` holds every batch's sections,
        // each at a multiple of 256 bytes.
        let section = |at: usize, len: usize| unsafe {
            DeviceBuffer::<i32>::alias(inputs, at * sections * size_of::<i32>(), len)
        };
        let cu_seqlens = section(3, sequences + 1);
        let packed = Packed {
            cu_seqlens: &cu_seqlens,
            sequences,
            tokens,
            max_len,
            min_len,
        };
        let corrected = min_len < CORRECTED_BELOW;
        let (token_ids, token_types, token_positions) =
            (section(0, tokens), section(1, tokens), section(2, tokens));
        let batch = Tokens {
            ids: &token_ids,
            types: &token_types,
            positions: &token_positions,
            count: tokens,
        };
        let (k, work) = (&self.kernels, &self.work);
        let pass = Pass {
            kernels: k,
            work,
            tokens: &batch,
            packed: &packed,
            corrected,
            shape: &self.shape,
        };
        let profiling = self.profiler.is_some();
        if let Some(profiler) = self.profiler.as_mut() {
            profiler.begin(&self.stream)?;
        }
        let (family, finish, profiler) = (&self.family, &self.finish, &mut self.profiler);
        let mut launch = |s: &Stream| -> Result<(), CudaError> {
            let mut mark = |stage| match profiler.as_mut() {
                Some(profiler) => profiler.mark(s, stage),
                None => Ok(()),
            };
            // SAFETY: `self` owns every buffer, the work is waited for below even
            // on a failed launch, and ids and types were range-checked.
            unsafe {
                transfers
                    .inputs
                    .write_first_async(staged, &transfers.staging, s)?;
                match family {
                    Family::Bert(bert) => bert.launch(s, &pass, &mut mark)?,
                    Family::Modern(modern) => modern.launch(s, &pass, &mut mark)?,
                }
                // The stream orders the last norm's reads of `sum` before the finish overwrites it;
                // hidden32 keeps the normalized input; `intermediate`, the means' scratch, is free.
                match finish {
                    Finish::Pool(pooling) => k.pool_normalize(
                        s,
                        *pooling,
                        &work.hidden32,
                        &packed,
                        &work.sum,
                        &work.intermediate,
                    )?,
                    Finish::Score(scorer) => match &**scorer {
                        Scorer::Pooler {
                            pooler_w,
                            pooler_b,
                            classifier_w,
                            classifier_b,
                        } => {
                            let head = PoolerHead {
                                pooler_w,
                                pooler_b,
                                classifier_w,
                                classifier_b,
                            };
                            k.rerank_head(s, &work.hidden32, &packed, &head, &work.sum)?;
                        }
                        Scorer::Prediction {
                            kernels,
                            pooling,
                            activation,
                            dense_w,
                            norm_g,
                            classifier_w,
                            classifier_b,
                        } => {
                            let head = PredictionHead {
                                pooling: *pooling,
                                activation: *activation,
                                dense_w,
                                norm_g,
                                classifier_w,
                                classifier_b,
                                eps: pass.shape.eps,
                            };
                            let (rows, scratch) = (&work.hidden32, &work.intermediate);
                            kernels
                                .prediction_head(k, s, rows, &packed, &head, &work.sum, scratch)?;
                        }
                    },
                }
                mark(Stage::Pool)?;
                work.sum
                    .read_first_async(outputs, &mut transfers.download, s)
            }
        };
        let (stream, recording) = (&self.stream, &self.recording);
        let (graphs, seen) = (&mut self.graphs, &mut self.seen);
        let key = (tokens, max_len, sequences, corrected);
        let launched = if profiling {
            // Stage events need the launches themselves on the stream.
            launch(stream)
        } else if let Some(at) = graphs.iter().position(|(shape, _)| *shape == key) {
            let kept = graphs.remove(at);
            // SAFETY: the graph names this encoder's storage, which outlives the work.
            let replayed = unsafe { kept.1.launch(stream) };
            graphs.push(kept);
            replayed
        } else if let Some(at) = seen.iter().position(|shape| *shape == key) {
            seen.remove(at);
            // The GPU runs the launches while the thread records them for the next sight.
            launch(stream).and_then(|()| {
                recording.begin_capture()?;
                let recorded = launch(recording);
                let captured = recording.end_capture();
                recorded?;
                let graph = captured?;
                graph.upload(stream)?;
                if graphs.len() == GRAPHS {
                    graphs.remove(0);
                }
                graphs.push((key, graph));
                Ok(())
            })
        } else {
            if seen.len() == GRAPHS {
                seen.remove(0);
            }
            seen.push(key);
            launch(stream)
        };
        // Work queued before a failed launch still uses the buffers.
        let waited = self.wait(sequences);
        launched?;
        waited?;
        if let Some(profiler) = self.profiler.as_mut() {
            profiler.finish(tokens)?;
        }
        let output = self.transfers.download.as_slice()[..outputs].to_vec();
        if output.iter().any(|value| !value.is_finite()) {
            return Err(EncoderError::Weights(
                "encoder produced non-finite outputs; checkpoint values exceed the backend's numerical range".into(),
            ));
        }
        Ok(output)
    }

    fn wait(&self, sequences: usize) -> Result<(), CudaError> {
        let polled = self.done.record(&self.stream).and_then(|()| {
            let started = Instant::now();
            while !self.done.query()? {
                if sequences > 1 && started.elapsed() >= SPIN {
                    return self.stream.synchronize();
                }
                std::hint::spin_loop();
            }
            Ok(())
        });
        // After a failed record or query only a synchronized stream shows the work done.
        polled.or_else(|error| self.stream.synchronize().and(Err(error)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_upload_rejects_fp16_overflow() {
        assert!(to_half(&[1.0, f32::MAX]).is_err());
        assert!(to_half(&[f32::INFINITY]).is_err());
        assert!(to_half(&[f32::NAN]).is_err());
        assert_eq!(
            to_half(&[65_504.0, -65_504.0]).unwrap()[0].to_f32(),
            65_504.0
        );
    }
}
