//! The transformer encoder on Vulkan: packed, padding-free batches through KodamaEmbed's shaders.

use crate::{
    device::{Binding, Buffer, Commands, Device, VulkanError},
    pipeline::{Descriptors, Kernel, KernelSpec, Push, Recorder},
    shaders,
};
use ash::vk;
use half::f16;
use kodamaembed_model::{
    Activation, Geometry, Head, Output, Pooling, Positions, Rotary, Shape, Tensors, fused_qkv,
    pair_gates, pair_turns, relative::RelativeBias, tensor, tensor_len,
};
use std::fmt;

/// Output tile edge of the cooperative-matrix and portable GEMMs.
const MATRIX_TILE: usize = 128;
const PORTABLE_TILE: usize = 64;
/// gemm_rows' output tile, rows by columns.
const ROWS_TILE: usize = 16;
const ROWS_COLUMNS: usize = 64;
/// 128-wide K stages halve the round trips that bound few rows; past WIDE_STAGE_ROWS
/// rows, 64-wide stages hold fewer registers.
const ROW_STAGES: [u32; 2] = [128, 64];
const WIDE_STAGE_ROWS: usize = 64;
/// Below this many 128 x 128 tiles gemm_matrix leaves most of the GPU idle, so gemm_rows,
/// writing the same bits, runs instead.
const ROWS_BELOW_TILES: usize = 40;
/// Query rows per attention workgroup.
const ATTENTION_ROWS: u32 = 64;
/// The cooperative-matrix shaders run in subgroups of this size.
const MATRIX_SUBGROUP: u32 = 32;
/// The embedding set's bindings, the widest of any set.
const EMBED_BINDINGS: usize = 10;
const FINISH_BINDINGS: usize = 7;
/// A BERT layer's sets and bindings: four projections, the attention and two norms.
const LAYER_SETS: usize = 7;
const LAYER_BINDINGS: usize = 38;
/// A ModernBERT layer's sets and bindings: two norms, four projections, two of them over pairs
/// of columns, and the attention; the final norm counts as the first layer's absent attention norm.
const MODERN_LAYER_SETS: usize = 7;
const MODERN_LAYER_BINDINGS: usize = 42;

/// Maximum batch capacity for the dispatches that use one workgroup per token.
pub const MAX_TOKENS: usize = u16::MAX as usize;

#[derive(Debug)]
pub enum EncoderError {
    Vulkan(VulkanError),
    Unsupported(String),
    Weights(String),
    Input(String),
}

impl fmt::Display for EncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulkan(error) => write!(f, "{error}"),
            Self::Unsupported(reason) => write!(f, "unsupported model: {reason}"),
            Self::Weights(reason) => write!(f, "weights: {reason}"),
            Self::Input(reason) => write!(f, "input: {reason}"),
        }
    }
}

impl std::error::Error for EncoderError {}

impl From<VulkanError> for EncoderError {
    fn from(error: VulkanError) -> Self {
        Self::Vulkan(error)
    }
}

#[derive(Clone, Copy)]
enum Epilogue {
    Bias = 0,
    Gelu = 1,
    Residual = 2,
}

/// The epilogues of the GEMM shaders' paired builds, which take their columns two by two.
#[derive(Clone, Copy)]
enum Paired {
    /// A pair is an input and its gate: their gated product, one output column a pair.
    Gated = 3,
    /// The bias epilogue after each pair of its low columns turns by the row's position.
    Turned = 4,
}

/// A projection's kernels: gemm_matrix, or gemm_portable without cooperative matrices, and
/// with them gemm_rows by K stage.
struct Gemm {
    tiles: Kernel,
    rows: Option<[Kernel; 2]>,
}

impl Gemm {
    /// `low_columns` are the columns whose low parts the bias epilogue keeps.
    fn load(
        device: &Device,
        matrix: bool,
        epilogue: Epilogue,
        low_columns: u32,
    ) -> Result<Self, VulkanError> {
        let shaders = [
            shaders::GEMM_MATRIX,
            shaders::GEMM_ROWS,
            shaders::GEMM_PORTABLE,
        ];
        let constants = [epilogue as u32, low_columns];
        Self::build(device, matrix, shaders, constants, None)
    }

    /// The paired build: the turned epilogue turns its `low_columns` columns, in heads of
    /// `head`, and keeps their low parts.
    fn paired(
        device: &Device,
        matrix: bool,
        epilogue: Paired,
        low_columns: u32,
        head: u32,
    ) -> Result<Self, VulkanError> {
        let shaders = [
            shaders::GEMM_MATRIX_PAIRED,
            shaders::GEMM_ROWS_PAIRED,
            shaders::GEMM_PORTABLE_PAIRED,
        ];
        let constants = [epilogue as u32, low_columns];
        Self::build(device, matrix, shaders, constants, Some(head))
    }

    /// The matrix, rows and portable shaders with their first two constants; `head` is the
    /// paired builds' own constant, and their two more bindings come with it.
    fn build(
        device: &Device,
        matrix: bool,
        [tiles, rows, portable]: [&[u8]; 3],
        constants: [u32; 2],
        head: Option<u32>,
    ) -> Result<Self, VulkanError> {
        let kernel = |spirv: &[u8], stage: Option<u32>| {
            let mut constants = constants.to_vec();
            constants.extend(stage);
            if let Some(head) = head {
                // The head is constant 3; constant 2 is gemm_rows' stage alone.
                constants.resize(3, 0);
                constants.push(head);
            }
            let spec = KernelSpec {
                spirv,
                bindings: if head.is_some() { 8 } else { 6 },
                push_bytes: 12,
                constants: &constants,
                subgroup_size: matrix.then_some(MATRIX_SUBGROUP),
            };
            // SAFETY: an embedded validated shader; Encoder::new checks the device limits first.
            unsafe { Kernel::new(device, &spec) }
        };
        Ok(if matrix {
            Self {
                tiles: kernel(tiles, None)?,
                rows: Some([
                    kernel(rows, Some(ROW_STAGES[0]))?,
                    kernel(rows, Some(ROW_STAGES[1]))?,
                ]),
            }
        } else {
            Self {
                tiles: kernel(portable, None)?,
                rows: None,
            }
        })
    }

    /// Records the projection of `m` rows to `n` columns over `depth` on the GEMM
    /// that suits `m`; all of them write the same bits.
    ///
    /// # Safety
    /// As [`Recorder::dispatch`], with `set` bound for this projection.
    unsafe fn project(
        &self,
        r: &Recorder,
        set: vk::DescriptorSet,
        m: usize,
        n: usize,
        depth: usize,
    ) {
        let push = Push::new().u32(m as u32).u32(n as u32).u32(depth as u32);
        let (kernel, tile_rows, tile_columns) = match &self.rows {
            Some(stages) if (n / MATRIX_TILE) * m.div_ceil(MATRIX_TILE) < ROWS_BELOW_TILES => {
                // A depth the wide stage does not divide takes the narrow one.
                let wide = ROW_STAGES[0] as usize;
                let narrow = m > WIDE_STAGE_ROWS || !depth.is_multiple_of(wide);
                (&stages[usize::from(narrow)], ROWS_TILE, ROWS_COLUMNS)
            }
            Some(_) => (&self.tiles, MATRIX_TILE, MATRIX_TILE),
            None => (&self.tiles, PORTABLE_TILE, PORTABLE_TILE),
        };
        let groups = [(n / tile_columns) as u32, m.div_ceil(tile_rows) as u32, 1];
        // SAFETY: the caller's.
        unsafe { r.dispatch(kernel, set, &push, groups) }
    }
}

struct Kernels {
    embed: Kernel,
    /// By epilogue.
    gemm: [Gemm; 3],
    attention: Kernel,
    layernorm: Kernel,
    /// Pooling or the reranking head, by the encoder's output.
    finish: Kernel,
}

/// What ModernBERT's layers run beside [`Kernels`].
struct ModernKernels {
    /// Attention inside a window.
    local_attention: Kernel,
    /// The query, key and value projection, which turns the queries and keys by their positions.
    qkv: Gemm,
    /// The up projection, which gates each input.
    up: Gemm,
}

fn spec<'a>(
    spirv: &'a [u8],
    bindings: u32,
    push_bytes: u32,
    constants: &'a [u32],
) -> KernelSpec<'a> {
    KernelSpec {
        spirv,
        bindings,
        push_bytes,
        constants,
        subgroup_size: None,
    }
}

impl Kernels {
    fn load(
        device: &Device,
        matrix: bool,
        geometry: Geometry,
        output: Output,
        relative: bool,
    ) -> Result<Self, VulkanError> {
        // SAFETY: embedded validated shaders, specialized for these geometries;
        // Encoder::new checks the device limits first.
        let quarter = [(geometry.hidden() / 4) as u32];
        let heads = [geometry.head_dim() as u32, geometry.heads() as u32];
        let width = [geometry.hidden() as u32];
        // The bias epilogue also keeps what FP16 lost from Q and K.
        let low_columns = |epilogue: Epilogue| match epilogue {
            Epilogue::Bias => 2 * geometry.hidden() as u32,
            Epilogue::Gelu | Epilogue::Residual => 0,
        };
        let gemm = |epilogue: Epilogue| Gemm::load(device, matrix, epilogue, low_columns(epilogue));
        let finish = match output {
            Output::Embedding(pooling) => {
                // The workgroup width, then whether the shader pools the first row alone.
                let constants = [quarter[0], u32::from(pooling == Pooling::Cls)];
                unsafe { Kernel::new(device, &spec(shaders::POOL, 3, 0, &constants)) }
            }
            Output::Relevance(Head::Pooler) => unsafe {
                Kernel::new(device, &spec(shaders::RERANK, 7, 0, &width))
            },
            Output::Relevance(Head::Prediction(pooling, activation)) => {
                // The width, then whether the rows' mean is pooled and whether SiLU activates.
                let constants = [
                    width[0],
                    u32::from(pooling == Pooling::Mean),
                    u32::from(activation == Activation::Silu),
                ];
                unsafe { Kernel::new(device, &spec(shaders::RERANK_PREDICTION, 7, 4, &constants)) }
            }
        }?;
        let attention = KernelSpec {
            spirv: match (matrix, relative) {
                (true, true) => shaders::ATTENTION_MATRIX_BIAS,
                (true, false) => shaders::ATTENTION_MATRIX,
                (false, true) => shaders::ATTENTION_BIAS,
                (false, false) => shaders::ATTENTION,
            },
            bindings: 4 + u32::from(relative),
            push_bytes: 4 + 4 * u32::from(relative),
            constants: &heads,
            subgroup_size: matrix.then_some(MATRIX_SUBGROUP),
        };
        Ok(Self {
            embed: unsafe { Kernel::new(device, &spec(shaders::EMBED, 10, 8, &quarter)) }?,
            gemm: [
                gemm(Epilogue::Bias)?,
                gemm(Epilogue::Gelu)?,
                gemm(Epilogue::Residual)?,
            ],
            attention: unsafe { Kernel::new(device, &attention) }?,
            layernorm: unsafe { Kernel::new(device, &spec(shaders::LAYERNORM, 5, 12, &quarter)) }?,
            finish,
        })
    }

    fn gemm(&self, epilogue: Epilogue) -> &Gemm {
        &self.gemm[epilogue as usize]
    }
}

impl ModernKernels {
    fn load(device: &Device, matrix: bool, geometry: Geometry) -> Result<Self, VulkanError> {
        let (head, queries_and_keys) = (geometry.head_dim() as u32, 2 * geometry.hidden() as u32);
        let heads = [head, geometry.heads() as u32];
        let local_attention = KernelSpec {
            spirv: if matrix {
                shaders::ATTENTION_MATRIX_LOCAL
            } else {
                shaders::ATTENTION_LOCAL
            },
            bindings: 4,
            push_bytes: 8,
            constants: &heads,
            subgroup_size: matrix.then_some(MATRIX_SUBGROUP),
        };
        Ok(Self {
            // SAFETY: as for Kernels::load.
            local_attention: unsafe { Kernel::new(device, &local_attention) }?,
            qkv: Gemm::paired(device, matrix, Paired::Turned, queries_and_keys, head)?,
            up: Gemm::paired(device, matrix, Paired::Gated, 0, head)?,
        })
    }
}

/// MPNet's relative-position bias on the GPU, with the longest sequence it covers.
struct RelativeTable {
    table: Buffer<f32>,
    longest: u32,
}

/// A BERT layer's bound steps: a norm after attention and after the feed-forward.
struct Layer {
    _weights: Vec<Buffer<f16>>,
    _vectors: Vec<Buffer<f32>>,
    qkv: vk::DescriptorSet,
    attention: vk::DescriptorSet,
    out: vk::DescriptorSet,
    norm1: vk::DescriptorSet,
    ffn1: vk::DescriptorSet,
    ffn2: vk::DescriptorSet,
    norm2: vk::DescriptorSet,
}

/// A ModernBERT layer's bound steps: a norm before attention and before the gated feed-forward.
struct ModernLayer {
    _weights: Vec<Buffer<f16>>,
    _vectors: Vec<Buffer<f32>>,
    /// None in the first layer, which attends over the embedding norm's rows as they are.
    attention_norm: Option<vk::DescriptorSet>,
    qkv: vk::DescriptorSet,
    attention: vk::DescriptorSet,
    /// The keys a query sees each side of itself; None over the whole sequence.
    reach: Option<usize>,
    out: vk::DescriptorSet,
    feed_forward_norm: vk::DescriptorSet,
    up: vk::DescriptorSet,
    down: vk::DescriptorSet,
}

/// ModernBERT's bound steps and what they read beyond the encoder's own storage.
struct Modern {
    kernels: ModernKernels,
    layers: Vec<ModernLayer>,
    final_norm: vk::DescriptorSet,
    /// The turns of the layers over whole sequences and of the windowed ones, and zero rows
    /// for what the family lacks.
    _buffers: Vec<Buffer<f32>>,
}

/// A model family's bound layers.
enum Family {
    Bert(Vec<Layer>),
    Modern(Box<Modern>),
}

struct Workspace {
    ids: Buffer<i32>,
    types: Buffer<i32>,
    positions: Buffer<i32>,
    cu_seqlens: Buffer<i32>,
    hidden16: Buffer<f16>,
    hidden32: Buffer<f32>,
    qkv: Buffer<f16>,
    /// Low parts of Q and K, which correct the FP16 attention scores.
    qk_low: Buffer<f32>,
    context: Buffer<f16>,
    /// Projection residuals, then compact final outputs after the last norm.
    sum: Buffer<f32>,
    intermediate: Buffer<f16>,
    /// Bound to slots a kernel variant does not use.
    unused: Buffer<f32>,
    /// Ids, types, positions and cu_seqlens at 0, t, 2t and 3t (t = max tokens).
    upload: Buffer<i32>,
    /// Host memory the batch's outputs are copied to as it ends.
    download: Buffer<f32>,
}

/// Tokens, longest sequence and sequences: every recorded parameter of an encode.
type Batch = (usize, usize, usize);

/// Recorded encodes kept for batches that repeat a shape, as queries do.
const RECORDED: usize = 32;

/// Up to this many tokens an encode is one command buffer, which the slowest supported GPU
/// finishes well within the operating system's 2 s watchdog.
const ONE_BUFFER_TOKENS: usize = 128;

pub struct Encoder {
    /// Recorded encodes, least recently used first; declared first to drop before what they use.
    recorded: Vec<(Batch, Commands)>,
    device: Device,
    shape: Shape,
    geometry: Geometry,
    output: Output,
    max_tokens: usize,
    matrix: bool,
    kernels: Kernels,
    _tables: Vec<Buffer<f32>>,
    relative: Option<RelativeTable>,
    family: Family,
    work: Workspace,
    embed: vk::DescriptorSet,
    finish: vk::DescriptorSet,
    _descriptors: Descriptors,
}

fn to_half(values: &[f32]) -> Result<Vec<f16>, EncoderError> {
    values
        .iter()
        .copied()
        .map(|value| {
            let rounded = f16::from_f32(value);
            if rounded.is_finite() {
                Ok(rounded)
            } else {
                Err(EncoderError::Weights(format!(
                    "{value} cannot be stored as finite FP16"
                )))
            }
        })
        .collect()
}

fn checked_tokens(lengths: &[usize]) -> Result<usize, EncoderError> {
    lengths
        .iter()
        .try_fold(0usize, |total, &len| total.checked_add(len))
        .ok_or_else(|| EncoderError::Input("sequence lengths overflow the token count".into()))
}

fn validate_limits(
    limits: &vk::PhysicalDeviceLimits,
    geometry: Geometry,
) -> Result<(), EncoderError> {
    // Both GEMMs have 256 invocations; the large embedding/norm shader has
    // 256 along x. The portable GEMM has 8 KiB of shared tiles.
    if limits.max_compute_work_group_invocations < 256
        || limits.max_compute_work_group_size[0] < (geometry.hidden() / 4) as u32
        || limits.max_compute_work_group_size[0] < 128
        || limits.max_compute_work_group_size[1] < 16
        || limits.max_compute_shared_memory_size < 8_192
        || limits.max_per_stage_descriptor_storage_buffers < EMBED_BINDINGS as u32
        || limits.max_descriptor_set_storage_buffers < EMBED_BINDINGS as u32
    {
        return Err(EncoderError::Unsupported(
            "Vulkan device limits are too small for these shaders".into(),
        ));
    }
    Ok(())
}

/// Each table and activation must fit in one storage-buffer binding.
fn validate_buffer_sizes(
    limits: &vk::PhysicalDeviceLimits,
    shape: &Shape,
    tokens: usize,
) -> Result<(), EncoderError> {
    // The largest live activations are the low parts of Q and K (FP32) and the FFN output (FP16).
    let row_bytes =
        (2 * shape.hidden * size_of::<f32>()).max(shape.intermediate * size_of::<f16>());
    if tokens > limits.max_storage_buffer_range as usize / row_bytes {
        return Err(EncoderError::Unsupported(format!(
            "{tokens} tokens exceed this device's {}-byte storage buffers",
            limits.max_storage_buffer_range
        )));
    }
    // ModernBERT embeds its words alone; a rotation table is smaller than the activations above.
    let tables: &[usize] = match shape.positions {
        Positions::ModernBert(_) => &[shape.vocab],
        _ => &[shape.vocab, shape.type_vocab, shape.max_positions],
    };
    for &rows in tables {
        let values = tensor_len(&[rows, shape.hidden]).map_err(EncoderError::Unsupported)?;
        if values / 4 > u32::MAX as usize {
            return Err(EncoderError::Unsupported(
                "embedding table exceeds shader indexing".into(),
            ));
        }
        let bytes = values * size_of::<f32>();
        if bytes > limits.max_storage_buffer_range as usize {
            return Err(EncoderError::Unsupported(format!(
                "a {bytes}-byte embedding table exceeds this device's {}-byte storage buffers",
                limits.max_storage_buffer_range
            )));
        }
    }
    Ok(())
}

/// Uploads a checkpoint's tensors and binds a model's steps to them. Every buffer a set names
/// moves into the [`Encoder`] with the pool, and encode waits for completion before any can drop.
struct Binder<'a> {
    device: &'a Device,
    descriptors: &'a Descriptors,
    kernels: &'a Kernels,
    work: &'a Workspace,
    tensors: &'a dyn Tensors,
}

/// The embedding tables and other FP32 buffers a model keeps, its embedding step and its layers.
type Bound = (Vec<Buffer<f32>>, vk::DescriptorSet, Family);

impl Binder<'_> {
    fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, EncoderError> {
        tensor(self.tensors, name, len).map_err(EncoderError::Weights)
    }

    fn upload(&self, name: &str, len: usize) -> Result<Buffer<f32>, EncoderError> {
        Ok(Buffer::from_slice(self.device, &self.get(name, len)?)?)
    }

    /// A matrix in FP16, as the GEMMs read it.
    fn half(&self, matrix: &[f32]) -> Result<Buffer<f16>, EncoderError> {
        Ok(Buffer::from_slice(self.device, &to_half(matrix)?)?)
    }

    fn matrix(&self, name: &str, len: usize) -> Result<Buffer<f16>, EncoderError> {
        self.half(&self.get(name, len)?)
    }

    fn bind(&self, kernel: &Kernel, buffers: &[Binding]) -> Result<vk::DescriptorSet, VulkanError> {
        // SAFETY: see the type.
        unsafe { self.descriptors.bind(kernel, buffers) }
    }

    /// Input, weight, bias, residual, FP16 output and FP32 output.
    fn gemm(&self, gemm: &Gemm, buffers: [Binding; 6]) -> Result<vk::DescriptorSet, VulkanError> {
        self.bind(&gemm.tiles, &buffers)
    }

    /// [`Self::gemm`]'s buffers, then the rows' positions and the turns, for a paired build.
    fn paired(&self, gemm: &Gemm, buffers: [Binding; 8]) -> Result<vk::DescriptorSet, VulkanError> {
        self.bind(&gemm.tiles, &buffers)
    }

    /// Input, gamma, beta, FP16 output and FP32 output.
    fn norm(&self, buffers: [Binding; 5]) -> Result<vk::DescriptorSet, VulkanError> {
        self.bind(&self.kernels.layernorm, &buffers)
    }

    fn bert(&self, shape: &Shape, relative: Option<&RelativeTable>) -> Result<Bound, EncoderError> {
        let (h, i, w, k) = (shape.hidden, shape.intermediate, self.work, self.kernels);
        let tables = vec![
            self.upload("embeddings.word_embeddings.weight", shape.vocab * h)?,
            self.upload(
                "embeddings.position_embeddings.weight",
                shape.max_positions * h,
            )?,
            self.upload(
                "embeddings.token_type_embeddings.weight",
                shape.type_vocab * h,
            )?,
            self.upload("embeddings.LayerNorm.weight", h)?,
            self.upload("embeddings.LayerNorm.bias", h)?,
        ];
        let embed = self.bind(
            &k.embed,
            &[
                w.ids.binding(),
                w.types.binding(),
                w.positions.binding(),
                tables[0].binding(),
                tables[1].binding(),
                tables[2].binding(),
                tables[3].binding(),
                tables[4].binding(),
                w.hidden16.binding(),
                w.hidden32.binding(),
            ],
        )?;
        let unused = w.unused.binding();
        let attention: Vec<Binding> = [
            w.qkv.binding(),
            w.qk_low.binding(),
            w.cu_seqlens.binding(),
            w.context.binding(),
        ]
        .into_iter()
        .chain(relative.map(|relative| relative.table.binding()))
        .collect();
        let mut layers = Vec::with_capacity(shape.layers);
        for layer in 0..shape.layers {
            let at = |suffix: &str| format!("encoder.layer.{layer}.{suffix}");
            let (qkv_w, qkv_b) =
                fused_qkv(self.tensors, shape, layer).map_err(EncoderError::Weights)?;
            let weights = vec![
                self.half(&qkv_w)?,
                self.matrix(&at("attention.output.dense.weight"), h * h)?,
                self.matrix(&at("intermediate.dense.weight"), i * h)?,
                self.matrix(&at("output.dense.weight"), h * i)?,
            ];
            let vectors = vec![
                Buffer::from_slice(self.device, &qkv_b)?,
                self.upload(&at("attention.output.dense.bias"), h)?,
                self.upload(&at("attention.output.LayerNorm.weight"), h)?,
                self.upload(&at("attention.output.LayerNorm.bias"), h)?,
                self.upload(&at("intermediate.dense.bias"), i)?,
                self.upload(&at("output.dense.bias"), h)?,
                self.upload(&at("output.LayerNorm.weight"), h)?,
                self.upload(&at("output.LayerNorm.bias"), h)?,
            ];
            let (g, b) = (&weights, &vectors);
            layers.push(Layer {
                qkv: self.gemm(
                    k.gemm(Epilogue::Bias),
                    [
                        w.hidden16.binding(),
                        g[0].binding(),
                        b[0].binding(),
                        unused,
                        w.qkv.binding(),
                        w.qk_low.binding(),
                    ],
                )?,
                attention: self.bind(&k.attention, &attention)?,
                out: self.gemm(
                    k.gemm(Epilogue::Residual),
                    [
                        w.context.binding(),
                        g[1].binding(),
                        b[1].binding(),
                        w.hidden32.binding(),
                        unused,
                        w.sum.binding(),
                    ],
                )?,
                // The out projection has read hidden32 before norm1 overwrites it.
                norm1: self.norm([
                    w.sum.binding(),
                    b[2].binding(),
                    b[3].binding(),
                    w.hidden16.binding(),
                    w.hidden32.binding(),
                ])?,
                ffn1: self.gemm(
                    k.gemm(Epilogue::Gelu),
                    [
                        w.hidden16.binding(),
                        g[2].binding(),
                        b[4].binding(),
                        unused,
                        w.intermediate.binding(),
                        unused,
                    ],
                )?,
                ffn2: self.gemm(
                    k.gemm(Epilogue::Residual),
                    [
                        w.intermediate.binding(),
                        g[3].binding(),
                        b[5].binding(),
                        w.hidden32.binding(),
                        unused,
                        w.sum.binding(),
                    ],
                )?,
                norm2: self.norm([
                    w.sum.binding(),
                    b[6].binding(),
                    b[7].binding(),
                    w.hidden16.binding(),
                    w.hidden32.binding(),
                ])?,
                _weights: weights,
                _vectors: vectors,
            });
        }
        Ok((tables, embed, Family::Bert(layers)))
    }

    /// Between layers the residual rows are in `sum`, and between a layer's attention and its
    /// feed-forward in `hidden32`; the final norm leaves the encoded rows in `hidden32`.
    fn modern(
        &self,
        shape: &Shape,
        rotary: Rotary,
        geometry: Geometry,
        max_tokens: usize,
    ) -> Result<Bound, EncoderError> {
        let (h, i, w, k, device) = (
            shape.hidden,
            shape.intermediate,
            self.work,
            self.kernels,
            self.device,
        );
        let kernels = ModernKernels::load(device, device.matrix_enabled(), geometry)?;
        let tables = vec![
            self.upload("embeddings.tok_embeddings.weight", shape.vocab * h)?,
            self.upload("embeddings.norm.weight", h)?,
        ];
        let (head, positions) = (
            geometry.head_dim(),
            max_tokens.min(shape.max_sequence_len()),
        );
        let turns =
            |theta: f32| Buffer::from_slice(device, &Rotary::paired_turns(theta, head, positions));
        let (global_turns, local_turns) = (turns(rotary.global_theta)?, turns(rotary.local_theta)?);
        // What every norm adds as its shift and every projection as its bias, and the row every
        // token embeds as its position and segment: the family has none of them.
        let zero_row = Buffer::<f32>::zeroed(device, h)?;
        let no_bias = Buffer::<f32>::zeroed(device, (3 * h).max(2 * i))?;
        let (unused, zero, bias) = (w.unused.binding(), zero_row.binding(), no_bias.binding());
        // The segment ids, all 0, index the zero row as positions too.
        let embed = self.bind(
            &k.embed,
            &[
                w.ids.binding(),
                w.types.binding(),
                w.types.binding(),
                tables[0].binding(),
                zero,
                zero,
                tables[1].binding(),
                zero,
                w.hidden16.binding(),
                w.sum.binding(),
            ],
        )?;
        let attended = [
            w.qkv.binding(),
            w.qk_low.binding(),
            w.cu_seqlens.binding(),
            w.context.binding(),
        ];
        let mut layers = Vec::with_capacity(shape.layers);
        for layer in 0..shape.layers {
            let at = |suffix: &str| format!("layers.{layer}.{suffix}");
            let qkv = self.get(&at("attn.Wqkv.weight"), 3 * h * h)?;
            let up = self.get(&at("mlp.Wi.weight"), 2 * i * h)?;
            // The paired projections take the queries' and keys' rows as the turn pairs them,
            // and each input's row beside its gate's.
            let weights = vec![
                self.half(&pair_turns(&qkv, h, head, 2 * h))?,
                self.matrix(&at("attn.Wo.weight"), h * h)?,
                self.half(&pair_gates(&up, h))?,
                self.matrix(&at("mlp.Wo.weight"), h * i)?,
            ];
            let mut vectors = vec![self.upload(&at("mlp_norm.weight"), h)?];
            if layer > 0 {
                vectors.push(self.upload(&at("attn_norm.weight"), h)?);
            }
            let reach = rotary.reach(layer);
            let turned = match reach {
                Some(_) => &local_turns,
                None => &global_turns,
            };
            let g = &weights;
            layers.push(ModernLayer {
                attention_norm: match vectors.get(1) {
                    Some(gamma) => Some(self.norm([
                        w.sum.binding(),
                        gamma.binding(),
                        zero,
                        w.hidden16.binding(),
                        unused,
                    ])?),
                    None => None,
                },
                qkv: self.paired(
                    &kernels.qkv,
                    [
                        w.hidden16.binding(),
                        g[0].binding(),
                        bias,
                        unused,
                        w.qkv.binding(),
                        w.qk_low.binding(),
                        w.positions.binding(),
                        turned.binding(),
                    ],
                )?,
                attention: match reach {
                    Some(_) => self.bind(&kernels.local_attention, &attended),
                    None => self.bind(&k.attention, &attended),
                }?,
                reach,
                out: self.gemm(
                    k.gemm(Epilogue::Residual),
                    [
                        w.context.binding(),
                        g[1].binding(),
                        bias,
                        w.sum.binding(),
                        unused,
                        w.hidden32.binding(),
                    ],
                )?,
                feed_forward_norm: self.norm([
                    w.hidden32.binding(),
                    vectors[0].binding(),
                    zero,
                    w.hidden16.binding(),
                    unused,
                ])?,
                up: self.paired(
                    &kernels.up,
                    [
                        w.hidden16.binding(),
                        g[2].binding(),
                        bias,
                        unused,
                        w.intermediate.binding(),
                        unused,
                        unused,
                        unused,
                    ],
                )?,
                down: self.gemm(
                    k.gemm(Epilogue::Residual),
                    [
                        w.intermediate.binding(),
                        g[3].binding(),
                        bias,
                        w.hidden32.binding(),
                        unused,
                        w.sum.binding(),
                    ],
                )?,
                _weights: weights,
                _vectors: vectors,
            });
        }
        let final_gamma = self.upload("final_norm.weight", h)?;
        let final_norm = self.norm([
            w.sum.binding(),
            final_gamma.binding(),
            zero,
            w.hidden16.binding(),
            w.hidden32.binding(),
        ])?;
        let modern = Modern {
            kernels,
            layers,
            final_norm,
            _buffers: vec![global_turns, local_turns, zero_row, no_bias, final_gamma],
        };
        Ok((tables, embed, Family::Modern(Box::new(modern))))
    }
}

impl Encoder {
    pub fn new(
        device: &Device,
        shape: Shape,
        output: Output,
        max_tokens: usize,
        tensors: &dyn Tensors,
    ) -> Result<Self, EncoderError> {
        shape.validate().map_err(EncoderError::Unsupported)?;
        let geometry = shape.geometry().ok_or_else(|| {
            EncoderError::Unsupported(format!(
                "hidden {}, heads {}, intermediate {}",
                shape.hidden, shape.heads, shape.intermediate
            ))
        })?;
        if max_tokens == 0 || max_tokens > MAX_TOKENS {
            return Err(EncoderError::Input(format!(
                "{max_tokens} tokens per batch exceeds one dispatch dimension"
            )));
        }
        let rotary = match shape.positions {
            Positions::ModernBert(rotary) => Some(rotary),
            _ => None,
        };
        let (h, i, t) = (shape.hidden, shape.intermediate, max_tokens);
        validate_limits(&device.shared.limits, geometry)?;
        validate_buffer_sizes(&device.shared.limits, &shape, max_tokens)?;
        let relative = RelativeBias::load(&shape, tensors).map_err(EncoderError::Weights)?;
        let (layer_sets, layer_bindings) = match rotary {
            Some(_) => (MODERN_LAYER_SETS, MODERN_LAYER_BINDINGS),
            None => (LAYER_SETS, LAYER_BINDINGS + usize::from(relative.is_some())),
        };
        let sets = shape
            .layers
            .checked_mul(layer_sets)
            .and_then(|n| n.checked_add(2))
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| EncoderError::Unsupported("too many encoder layers".into()))?;
        let buffers = shape
            .layers
            .checked_mul(layer_bindings)
            .and_then(|n| n.checked_add(EMBED_BINDINGS + FINISH_BINDINGS))
            .and_then(|n| u32::try_from(n).ok())
            .ok_or_else(|| EncoderError::Unsupported("too many encoder bindings".into()))?;
        let matrix = device.matrix_enabled();

        let work = Workspace {
            ids: Buffer::zeroed(device, t)?,
            types: Buffer::zeroed(device, t)?,
            positions: Buffer::zeroed(device, t)?,
            cu_seqlens: Buffer::zeroed(device, t + 1)?,
            hidden16: Buffer::zeroed(device, t * h)?,
            hidden32: Buffer::zeroed(device, t * h)?,
            qkv: Buffer::zeroed(device, t * 3 * h)?,
            qk_low: Buffer::zeroed(device, t * 2 * h)?,
            context: Buffer::zeroed(device, t * h)?,
            sum: Buffer::zeroed(device, t * h)?,
            intermediate: Buffer::zeroed(device, t * i)?,
            unused: Buffer::zeroed(device, 1)?,
            upload: Buffer::host(device, 4 * t + 1)?,
            download: Buffer::readback(device, t * output.width(h))?,
        };
        let kernels = Kernels::load(device, matrix, geometry, output, relative.is_some())?;
        let descriptors = Descriptors::new(device, sets, buffers)?;
        let relative = match relative {
            Some(bias) => Some(RelativeTable {
                table: Buffer::from_slice(device, bias.values())?,
                longest: u32::try_from(bias.longest()).expect("fits u32"),
            }),
            None => None,
        };
        let binder = Binder {
            device,
            descriptors: &descriptors,
            kernels: &kernels,
            work: &work,
            tensors,
        };
        let (mut tables, embed, family) = match rotary {
            Some(rotary) => binder.modern(&shape, rotary, geometry, max_tokens)?,
            None => binder.bert(&shape, relative.as_ref())?,
        };
        let w = &work;
        let finish = match output {
            Output::Embedding(_) => binder.bind(
                &kernels.finish,
                &[
                    w.hidden32.binding(),
                    w.cu_seqlens.binding(),
                    w.sum.binding(),
                ],
            )?,
            Output::Relevance(head) => {
                // A prediction head's norm scales where a pooler's bias adds.
                let (dense, vector) = match head {
                    Head::Pooler => ("pooler.dense.weight", "pooler.dense.bias"),
                    Head::Prediction(..) => ("head.dense.weight", "head.norm.weight"),
                };
                let head = [
                    binder.upload(dense, h * h)?,
                    binder.upload(vector, h)?,
                    binder.upload("classifier.weight", h)?,
                    binder.upload("classifier.bias", 1)?,
                ];
                let set = binder.bind(
                    &kernels.finish,
                    &[
                        w.hidden32.binding(),
                        w.cu_seqlens.binding(),
                        head[0].binding(),
                        head[1].binding(),
                        head[2].binding(),
                        head[3].binding(),
                        w.sum.binding(),
                    ],
                )?;
                tables.extend(head);
                set
            }
        };
        Ok(Self {
            recorded: Vec::new(),
            device: device.clone(),
            shape,
            geometry,
            output,
            max_tokens,
            matrix,
            kernels,
            _tables: tables,
            relative,
            family,
            work,
            embed,
            finish,
            _descriptors: descriptors,
        })
    }

    /// Vendor and UUID of the device this encoder uses.
    pub fn device_identity(&self) -> (u32, [u8; 16]) {
        self.device.device_identity()
    }

    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Whether GEMMs and attention run on cooperative matrices (tensor units).
    pub fn uses_matrix_units(&self) -> bool {
        self.matrix
    }

    /// Values returned per sequence: `hidden` for sentence vectors, 1 for scores.
    pub fn output_width(&self) -> usize {
        self.output.width(self.shape.hidden)
    }

    /// Encodes sequences of token `ids` and segment `types` packed back to back
    /// with the given `lengths`: [`Encoder::output_width`] values per sequence.
    pub fn encode(
        &mut self,
        ids: &[i32],
        types: &[i32],
        lengths: &[usize],
    ) -> Result<Vec<f32>, EncoderError> {
        self.device.shared.check_usable()?;
        let tokens = checked_tokens(lengths)?;
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
        if tokens > self.max_tokens || max_len > self.shape.max_sequence_len() {
            return Err(EncoderError::Input(format!(
                "{tokens} tokens (limit {}), longest {max_len} (limit {})",
                self.max_tokens,
                self.shape.max_sequence_len()
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
        let positions = self.shape.position_ids(ids, lengths);
        let cu: Vec<i32> = std::iter::once(0)
            .chain(lengths.iter().scan(0i32, |total, &len| {
                *total += len as i32;
                Some(*total)
            }))
            .collect();
        let (t, outputs) = (self.max_tokens, lengths.len() * self.output_width());
        let upload = self.work.upload.contents_mut();
        upload[..tokens].copy_from_slice(ids);
        upload[t..t + tokens].copy_from_slice(types);
        upload[2 * t..2 * t + tokens].copy_from_slice(&positions);
        upload[3 * t..3 * t + cu.len()].copy_from_slice(&cu);

        let batch = (tokens, max_len, lengths.len());
        match self.recorded.iter().position(|(kept, _)| *kept == batch) {
            Some(at) => {
                let kept = self.recorded.remove(at);
                kept.1.run()?;
                self.recorded.push(kept);
            }
            None => {
                let commands = self.record(batch)?;
                if self.recorded.len() == RECORDED {
                    self.recorded.remove(0);
                }
                self.recorded.push((batch, commands));
            }
        }
        let result = &self.work.download.contents()[..outputs];
        if result.iter().any(|value| !value.is_finite()) {
            return Err(EncoderError::Weights(
                "inference produced non-finite output".into(),
            ));
        }
        Ok(result.to_vec())
    }

    /// Records and runs an encode, keeping its commands for later batches of this shape.
    fn record(&self, (tokens, max_len, sequences): Batch) -> Result<Commands, EncoderError> {
        let (device, w) = (&self.device, &self.work);
        let (t, outputs) = (self.max_tokens, sequences * self.output_width());
        let (h, i) = (self.shape.hidden, self.shape.intermediate);
        let m = tokens as u32;
        let k = &self.kernels;
        let norm = Push::new().u32(m).u32(1).f32(self.shape.eps);
        let scale = 1.0 / (self.geometry.head_dim() as f32).sqrt();
        let attention = match &self.relative {
            Some(relative) => Push::new().f32(scale).u32(relative.longest),
            None => Push::new().f32(scale),
        };
        let attention_groups = [
            (max_len as u32).div_ceil(ATTENTION_ROWS),
            self.geometry.heads() as u32,
            sequences as u32,
        ];
        // SAFETY (all three): encode's checks bound every access, and the kept
        // commands drop before the encoder's resources they use.
        let embedding = |command: vk::CommandBuffer| unsafe {
            let r = Recorder::new(device, command);
            r.copy(&w.upload, 0, &w.ids, 0, tokens);
            r.copy(&w.upload, t, &w.types, 0, tokens);
            r.copy(&w.upload, 2 * t, &w.positions, 0, tokens);
            r.copy(&w.upload, 3 * t, &w.cu_seqlens, 0, sequences + 1);
            r.copies_to_shaders();
            r.dispatch(
                &k.embed,
                self.embed,
                &Push::new().u32(m).f32(self.shape.eps),
                [m, 1, 1],
            );
        };
        let bert_layer = |command: vk::CommandBuffer, layer: &Layer| unsafe {
            let r = Recorder::new(device, command);
            r.barrier();
            k.gemm(Epilogue::Bias)
                .project(&r, layer.qkv, tokens, 3 * h, h);
            r.barrier();
            r.dispatch(&k.attention, layer.attention, &attention, attention_groups);
            r.barrier();
            k.gemm(Epilogue::Residual)
                .project(&r, layer.out, tokens, h, h);
            r.barrier();
            r.dispatch(&k.layernorm, layer.norm1, &norm, [m, 1, 1]);
            r.barrier();
            k.gemm(Epilogue::Gelu).project(&r, layer.ffn1, tokens, i, h);
            r.barrier();
            k.gemm(Epilogue::Residual)
                .project(&r, layer.ffn2, tokens, h, i);
            r.barrier();
            r.dispatch(&k.layernorm, layer.norm2, &norm, [m, 1, 1]);
        };
        // A norm before a projection writes its FP16 rows alone.
        let pre_norm = Push::new().u32(m).u32(0).f32(self.shape.eps);
        let modern_layer = |command: vk::CommandBuffer, modern: &Modern, layer: &ModernLayer| unsafe {
            let (r, own) = (Recorder::new(device, command), &modern.kernels);
            if let Some(attention_norm) = layer.attention_norm {
                r.barrier();
                r.dispatch(&k.layernorm, attention_norm, &pre_norm, [m, 1, 1]);
            }
            r.barrier();
            own.qkv.project(&r, layer.qkv, tokens, 3 * h, h);
            r.barrier();
            match layer.reach {
                Some(reach) => {
                    // No key lies farther from a query than the longest sequence.
                    let window = Push::new().f32(scale).u32(reach.min(max_len) as u32);
                    let kernel = &own.local_attention;
                    r.dispatch(kernel, layer.attention, &window, attention_groups);
                }
                None => r.dispatch(&k.attention, layer.attention, &attention, attention_groups),
            }
            r.barrier();
            k.gemm(Epilogue::Residual)
                .project(&r, layer.out, tokens, h, h);
            r.barrier();
            r.dispatch(&k.layernorm, layer.feed_forward_norm, &pre_norm, [m, 1, 1]);
            r.barrier();
            own.up.project(&r, layer.up, tokens, 2 * i, h);
            r.barrier();
            k.gemm(Epilogue::Residual)
                .project(&r, layer.down, tokens, h, i);
        };
        let encoding = |command: vk::CommandBuffer, layer: usize| match &self.family {
            Family::Bert(layers) => bert_layer(command, &layers[layer]),
            Family::Modern(modern) => modern_layer(command, modern, &modern.layers[layer]),
        };
        let finishing = |command: vk::CommandBuffer| unsafe {
            let r = Recorder::new(device, command);
            if let Family::Modern(modern) = &self.family {
                r.barrier();
                r.dispatch(&k.layernorm, modern.final_norm, &norm, [m, 1, 1]);
            }
            // The finish shader writes outputs over `sum`, after the last norm reads it.
            r.barrier();
            let push = match self.output {
                Output::Relevance(Head::Prediction(..)) => Push::new().f32(self.shape.eps),
                _ => Push::new(),
            };
            r.dispatch(&k.finish, self.finish, &push, [sequences as u32, 1, 1]);
            r.shaders_to_copies();
            r.copy(&w.sum, 0, &w.download, 0, outputs);
            r.copies_to_host();
        };
        let mut submission = device.reusable_submission();
        if tokens <= ONE_BUFFER_TOKENS {
            submission.record(|command| {
                embedding(command);
                for layer in 0..self.shape.layers {
                    encoding(command, layer);
                }
                finishing(command);
                Ok(())
            })?;
        } else {
            // One command buffer per layer limits watchdog exposure. Each later
            // buffer opens with a barrier ordering it after earlier queue work.
            submission.record(|command| {
                embedding(command);
                Ok(())
            })?;
            for layer in 0..self.shape.layers {
                submission.record(|command| {
                    encoding(command, layer);
                    Ok(())
                })?;
            }
            submission.record(|command| {
                finishing(command);
                Ok(())
            })?;
        }
        Ok(submission.run_kept()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kodamaembed_model::Positions;

    #[test]
    fn embedding_tables_past_the_storage_buffer_range_are_unsupported() {
        let xlm_roberta_large = Shape {
            layers: 24,
            hidden: 1024,
            heads: 16,
            intermediate: 4096,
            vocab: 250_002,
            max_positions: 514,
            type_vocab: 1,
            eps: 1e-5,
            positions: Positions::Roberta { pad: 1 },
        };
        let mut limits = vk::PhysicalDeviceLimits {
            max_storage_buffer_range: 1_000_000_000,
            ..Default::default()
        };
        assert!(matches!(
            validate_buffer_sizes(&limits, &xlm_roberta_large, 512),
            Err(EncoderError::Unsupported(_))
        ));
        limits.max_storage_buffer_range = u32::MAX;
        assert!(validate_buffer_sizes(&limits, &xlm_roberta_large, 512).is_ok());
        let small_vocab = Shape {
            vocab: 1,
            ..xlm_roberta_large
        };
        limits.max_storage_buffer_range = 8 * 1024 * 1024;
        assert!(validate_buffer_sizes(&limits, &small_vocab, 1024).is_ok());
        assert!(matches!(
            validate_buffer_sizes(&limits, &small_vocab, 1025),
            Err(EncoderError::Unsupported(_))
        ));
    }

    /// ModernBERT rotates by position and embeds no position table; the low parts of its queries
    /// and keys, 2 x 768 FP32 values a token, are its largest rows.
    #[test]
    fn modernbert_is_sized_by_its_queries_and_keys_and_not_by_a_position_table() {
        let modernbert_base = Shape {
            layers: 22,
            hidden: 768,
            heads: 12,
            intermediate: 1152,
            vocab: 1,
            max_positions: 8192,
            type_vocab: 1,
            eps: 1e-5,
            positions: Positions::ModernBert(Rotary {
                global_theta: 160_000.0,
                local_theta: 10_000.0,
                local_layers: 0,
                window: 128,
            }),
        };
        let limits = vk::PhysicalDeviceLimits {
            max_storage_buffer_range: 6_144 * 1000,
            ..Default::default()
        };
        assert!(validate_buffer_sizes(&limits, &modernbert_base, 1000).is_ok());
        assert!(matches!(
            validate_buffer_sizes(&limits, &modernbert_base, 1001),
            Err(EncoderError::Unsupported(_))
        ));
    }

    #[test]
    fn overflowing_sequence_lengths_are_rejected() {
        assert!(matches!(
            checked_tokens(&[usize::MAX, 1]),
            Err(EncoderError::Input(_))
        ));
        assert_eq!(checked_tokens(&[3, 5, 2]).unwrap(), 10);
    }

    #[test]
    fn weights_that_overflow_half_precision_are_rejected() {
        assert!(to_half(&[1.0, 65_504.0, -65_504.0]).is_ok());
        for value in [f32::MAX, -f32::MAX, f32::NAN, f32::INFINITY] {
            assert!(matches!(to_half(&[value]), Err(EncoderError::Weights(_))));
        }
    }

    #[test]
    fn limits_are_checked_for_every_model_geometry() {
        let mut limits = vk::PhysicalDeviceLimits {
            max_compute_work_group_invocations: 256,
            max_compute_work_group_size: [128, 128, 64],
            max_compute_shared_memory_size: 32_768,
            max_per_stage_descriptor_storage_buffers: 10,
            max_descriptor_set_storage_buffers: 10,
            ..Default::default()
        };
        assert!(validate_limits(&limits, Geometry::Small).is_ok());
        assert!(validate_limits(&limits, Geometry::Base).is_err());
        limits.max_compute_work_group_size[0] = 192;
        assert!(validate_limits(&limits, Geometry::Base).is_ok());
        assert!(validate_limits(&limits, Geometry::Large).is_err());
        limits.max_compute_work_group_size[0] = 256;
        assert!(validate_limits(&limits, Geometry::Large).is_ok());
        limits.max_per_stage_descriptor_storage_buffers = 4;
        assert!(validate_limits(&limits, Geometry::Small).is_err());
    }
}
