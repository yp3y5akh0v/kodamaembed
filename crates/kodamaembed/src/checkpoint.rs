//! Model files: `config.json` and an F32, F16 or BF16 `model.safetensors`.

use kodamaembed_model::{
    Activation, Head, Output, Pooling, Positions, Rotary, Shape, Tensors, relative,
};
use rustc_hash::FxHashMap;
use serde::Deserialize;
use std::borrow::Cow;

/// The model families KodamaEmbed runs.
#[derive(Clone, Copy)]
enum Family {
    /// BERT's layers, as one of the families that share them numbers positions.
    Bert(BertLike),
    ModernBert,
}

#[derive(Clone, Copy)]
enum BertLike {
    Bert,
    XlmRoberta,
    Mpnet,
}

impl Family {
    /// Read from `model_type` alone: a family's config lacks the fields of the others.
    fn from_config(bytes: &[u8]) -> Result<Self, String> {
        #[derive(Deserialize)]
        struct Named {
            model_type: String,
        }
        let Named { model_type } = serde_json::from_slice(bytes).map_err(malformed)?;
        match model_type.as_str() {
            "bert" => Ok(Self::Bert(BertLike::Bert)),
            "xlm-roberta" => Ok(Self::Bert(BertLike::XlmRoberta)),
            "mpnet" => Ok(Self::Bert(BertLike::Mpnet)),
            "modernbert" => Ok(Self::ModernBert),
            other => Err(format!(
                "a {other} checkpoint, where KodamaEmbed runs bert, xlm-roberta, mpnet and modernbert"
            )),
        }
    }
}

fn malformed(error: serde_json::Error) -> String {
    format!("config.json: {error}")
}

#[derive(Deserialize)]
struct Config {
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    type_vocab_size: Option<usize>,
    layer_norm_eps: f32,
    hidden_act: String,
    #[serde(default = "absolute")]
    position_embedding_type: String,
    #[serde(default)]
    is_decoder: bool,
    #[serde(default)]
    add_cross_attention: bool,
    pad_token_id: Option<i32>,
    relative_attention_num_buckets: Option<usize>,
}

fn absolute() -> String {
    "absolute".into()
}

#[derive(Deserialize)]
struct ModernBertConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    intermediate_size: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    norm_eps: f32,
    hidden_activation: String,
    local_attention: usize,
    /// Each layer's attention, as transformers 5 writes it; older files give the period.
    layer_types: Option<Vec<String>>,
    #[serde(default = "every_third")]
    global_attn_every_n_layers: usize,
    /// Each attention kind's rotation, as transformers 5 writes it; older files give the bases.
    rope_parameters: Option<RopeParameters>,
    global_rope_theta: Option<f32>,
    local_rope_theta: Option<f32>,
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    norm_bias: bool,
}

fn every_third() -> usize {
    3
}

#[derive(Deserialize)]
struct RopeParameters {
    full_attention: Option<Rope>,
    sliding_attention: Option<Rope>,
}

#[derive(Deserialize)]
struct Rope {
    rope_theta: Option<f32>,
    rope_type: Option<String>,
}

/// transformers' rotation bases where a file gives none.
const GLOBAL_THETA: f32 = 160_000.0;
const LOCAL_THETA: f32 = 10_000.0;

/// An attention kind's rotation base: its own, else the older field's, else transformers'.
fn rope_theta(
    kind: &str,
    rope: Option<&Rope>,
    older: Option<f32>,
    default: f32,
) -> Result<f32, String> {
    let scaled = rope
        .and_then(|rope| rope.rope_type.as_deref())
        .filter(|rope_type| *rope_type != "default");
    if let Some(scaled) = scaled {
        return Err(format!("{kind} with {scaled} rotary positions"));
    }
    Ok(rope
        .and_then(|rope| rope.rope_theta)
        .or(older)
        .unwrap_or(default))
}

/// The layers that attend inside the window, as [`Rotary::local_layers`] names them.
fn local_layers(config: &ModernBertConfig) -> Result<u64, String> {
    let layers = config.num_hidden_layers;
    let period = config.global_attn_every_n_layers;
    match &config.layer_types {
        Some(kinds) if kinds.len() != layers => {
            return Err(format!(
                "layer_types names {} layers of {layers}",
                kinds.len()
            ));
        }
        None if period == 0 => return Err("global_attn_every_n_layers must be positive".into()),
        _ => {}
    }
    // Shape::validate refuses a model deeper than the set names.
    (0..layers.min(Rotary::MAX_LAYERS)).try_fold(0, |local, layer| {
        let windowed = match &config.layer_types {
            Some(kinds) => match kinds[layer].as_str() {
                "full_attention" => false,
                "sliding_attention" => true,
                other => return Err(format!("layer {layer} with {other}")),
            },
            // Older files: every n-th layer, from the first, attends over the whole sequence.
            None => !layer.is_multiple_of(period),
        };
        Ok(local | u64::from(windowed) << layer)
    })
}

/// A ModernBERT cross-encoder's head, with transformers' defaults where a file names none.
#[derive(Deserialize)]
struct PredictionHead {
    #[serde(default = "first_row")]
    classifier_pooling: String,
    #[serde(default = "gelu")]
    classifier_activation: String,
    #[serde(default)]
    classifier_bias: bool,
}

fn first_row() -> String {
    "cls".into()
}

fn gelu() -> String {
    "gelu".into()
}

/// The head a cross-encoder's `config.json` describes: BERT's families have the pooler.
pub fn head_from_bytes(bytes: &[u8]) -> Result<Head, String> {
    if let Family::Bert(_) = Family::from_config(bytes)? {
        return Ok(Head::Pooler);
    }
    let head: PredictionHead = serde_json::from_slice(bytes).map_err(malformed)?;
    if head.classifier_bias {
        return Err(
            "a ModernBERT head with classifier_bias, where KodamaEmbed runs its bias-free dense layer"
                .into(),
        );
    }
    let pooling = match head.classifier_pooling.as_str() {
        "cls" => Pooling::Cls,
        "mean" => Pooling::Mean,
        other => return Err(format!("classifier pooling {other}")),
    };
    let activation = match head.classifier_activation.as_str() {
        "gelu" => Activation::Gelu,
        "silu" => Activation::Silu,
        other => return Err(format!("classifier activation {other}")),
    };
    Ok(Head::Prediction(pooling, activation))
}

pub fn shape_from_bytes(bytes: &[u8]) -> Result<Shape, String> {
    let shape = match Family::from_config(bytes)? {
        Family::Bert(family) => {
            bert_shape(family, serde_json::from_slice(bytes).map_err(malformed)?)?
        }
        Family::ModernBert => modernbert_shape(serde_json::from_slice(bytes).map_err(malformed)?)?,
    };
    shape.validate()?;
    Ok(shape)
}

fn modernbert_shape(config: ModernBertConfig) -> Result<Shape, String> {
    for (flag, set) in [
        ("attention_bias", config.attention_bias),
        ("mlp_bias", config.mlp_bias),
        ("norm_bias", config.norm_bias),
    ] {
        if set {
            return Err(format!(
                "a ModernBERT checkpoint with {flag}, where KodamaEmbed runs its bias-free layers"
            ));
        }
    }
    if config.hidden_activation != "gelu" {
        return Err(format!("activation {}", config.hidden_activation));
    }
    if config.rope_scaling.is_some() {
        return Err("rope_scaling, where KodamaEmbed runs unscaled rotary positions".into());
    }
    let rope = config.rope_parameters.as_ref();
    let global_theta = rope_theta(
        "full_attention",
        rope.and_then(|rope| rope.full_attention.as_ref()),
        config.global_rope_theta,
        GLOBAL_THETA,
    )?;
    let local_theta = rope_theta(
        "sliding_attention",
        rope.and_then(|rope| rope.sliding_attention.as_ref()),
        config.local_rope_theta,
        LOCAL_THETA,
    )?;
    Ok(Shape {
        layers: config.num_hidden_layers,
        hidden: config.hidden_size,
        heads: config.num_attention_heads,
        intermediate: config.intermediate_size,
        vocab: config.vocab_size,
        max_positions: config.max_position_embeddings,
        // No segment table: every token is in the one segment.
        type_vocab: 1,
        eps: config.norm_eps,
        positions: Positions::ModernBert(Rotary {
            global_theta,
            local_theta,
            local_layers: local_layers(&config)?,
            window: config.local_attention,
        }),
    })
}

fn bert_shape(family: BertLike, config: Config) -> Result<Shape, String> {
    let segments = || {
        config
            .type_vocab_size
            .ok_or_else(|| "config.json without type_vocab_size".to_owned())
    };
    let (positions, type_vocab) = match (family, config.pad_token_id) {
        (BertLike::Bert, _) => (Positions::Sequential, segments()?),
        (BertLike::XlmRoberta, Some(pad)) => (Positions::Roberta { pad }, segments()?),
        (BertLike::XlmRoberta, None) => {
            return Err("an xlm-roberta config.json without pad_token_id".into());
        }
        // MPNet has no segment embeddings: its one segment's row is served as zeros.
        (BertLike::Mpnet, _) => match config.relative_attention_num_buckets {
            None | Some(relative::BUCKETS) => (Positions::Mpnet, 1),
            Some(buckets) => {
                return Err(format!(
                    "a {buckets}-bucket MPNet, where transformers' forward uses {}",
                    relative::BUCKETS
                ));
            }
        },
    };
    if config.is_decoder || config.add_cross_attention {
        return Err("decoder and cross-attention configurations are unsupported".into());
    }
    if config.hidden_act != "gelu" || config.position_embedding_type != "absolute" {
        return Err(format!(
            "activation {} with {} positions",
            config.hidden_act, config.position_embedding_type
        ));
    }
    Ok(Shape {
        layers: config.num_hidden_layers,
        hidden: config.hidden_size,
        heads: config.num_attention_heads,
        intermediate: config.intermediate_size,
        vocab: config.vocab_size,
        max_positions: config.max_position_embeddings,
        type_vocab,
        eps: config.layer_norm_eps,
        positions,
    })
}

#[derive(Deserialize)]
struct Entry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// Where a checkpoint keeps the tensors KodamaEmbed reads by BERT's names, or by ModernBERT's.
#[derive(Clone, Copy)]
enum Layout {
    /// The encoder at the root.
    Encoder,
    /// The encoder and pooler under `bert.`, the classifier at the root.
    Bert,
    /// The encoder under `roberta.`, its head's dense and out_proj as pooler and classifier.
    Roberta,
    /// MPNet's attention names, the encoder under `mpnet.` with a RoBERTa head when nested.
    Mpnet { nested: bool },
    /// ModernBERT's encoder under `model.`, as its masked-language-model and classification
    /// checkpoints keep it, with the head and the classifier at the root.
    ModernBert,
}

pub struct SafeTensors<'a> {
    bytes: Cow<'a, [u8]>,
    data_start: usize,
    entries: FxHashMap<String, Entry>,
    layout: Layout,
}

impl<'a> SafeTensors<'a> {
    pub fn from_bytes(bytes: Cow<'a, [u8]>) -> Result<Self, String> {
        let header_len = bytes
            .get(..8)
            .map(|b| u64::from_le_bytes(b.try_into().expect("eight bytes")))
            .ok_or("safetensors file shorter than its header length")?;
        let header_len = usize::try_from(header_len)
            .map_err(|_| "safetensors header exceeds the address space")?;
        let data_start = 8usize
            .checked_add(header_len)
            .filter(|&end| end <= bytes.len())
            .ok_or("safetensors header runs past the file")?;
        let mut header: FxHashMap<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..data_start]).map_err(|e| format!("header: {e}"))?;
        header.remove("__metadata__");
        let entries = header
            .into_iter()
            .map(|(name, value)| {
                serde_json::from_value(value)
                    .map(|entry| (name.clone(), entry))
                    .map_err(|e| format!("{name}: {e}"))
            })
            .collect::<Result<FxHashMap<String, Entry>, String>>()?;
        let layout = if entries.contains_key("bert.embeddings.word_embeddings.weight") {
            Layout::Bert
        } else if entries.contains_key("roberta.embeddings.word_embeddings.weight") {
            Layout::Roberta
        } else if entries.contains_key(&format!("mpnet.{}", relative::TENSOR)) {
            Layout::Mpnet { nested: true }
        } else if entries.contains_key(relative::TENSOR) {
            Layout::Mpnet { nested: false }
        } else if entries.contains_key("model.embeddings.tok_embeddings.weight") {
            Layout::ModernBert
        } else {
            Layout::Encoder
        };
        Ok(Self {
            bytes,
            data_start,
            entries,
            layout,
        })
    }

    /// Element counts cannot tell a transposed matrix from the expected one.
    pub fn validate_shapes(&self, shape: Shape, output: Output) -> Result<(), String> {
        if let Positions::ModernBert(_) = shape.positions {
            self.validate_modernbert(shape)?;
            return self.validate_head(shape.hidden, output);
        }
        // Mapped names and zero segments would let another family's config run MPNet's weights.
        if shape.positions == Positions::Mpnet {
            self.check_shape(relative::TENSOR, &[relative::BUCKETS, shape.heads])?;
        } else if matches!(self.layout, Layout::Mpnet { .. }) {
            return Err("an MPNet checkpoint with another model's config.json".into());
        }
        let h = shape.hidden;
        for (name, dimensions) in [
            ("embeddings.word_embeddings.weight", vec![shape.vocab, h]),
            (
                "embeddings.position_embeddings.weight",
                vec![shape.max_positions, h],
            ),
            (
                "embeddings.token_type_embeddings.weight",
                vec![shape.type_vocab, h],
            ),
            ("embeddings.LayerNorm.weight", vec![h]),
            ("embeddings.LayerNorm.bias", vec![h]),
        ] {
            self.check_shape(name, &dimensions)?;
        }
        for layer in 0..shape.layers {
            let root = format!("encoder.layer.{layer}");
            for projection in ["query", "key", "value"] {
                self.check_shape(
                    &format!("{root}.attention.self.{projection}.weight"),
                    &[h, h],
                )?;
                self.check_shape(&format!("{root}.attention.self.{projection}.bias"), &[h])?;
            }
            for (part, rows, columns) in [
                ("attention.output.dense", h, h),
                ("intermediate.dense", shape.intermediate, h),
                ("output.dense", h, shape.intermediate),
            ] {
                self.check_shape(&format!("{root}.{part}.weight"), &[rows, columns])?;
                self.check_shape(&format!("{root}.{part}.bias"), &[rows])?;
            }
            for part in ["attention.output.LayerNorm", "output.LayerNorm"] {
                self.check_shape(&format!("{root}.{part}.weight"), &[h])?;
                self.check_shape(&format!("{root}.{part}.bias"), &[h])?;
            }
        }
        self.validate_head(h, output)
    }

    /// A cross-encoder's head and its one-label classifier.
    fn validate_head(&self, h: usize, output: Output) -> Result<(), String> {
        match output {
            Output::Embedding(_) => return Ok(()),
            Output::Relevance(Head::Pooler) => {
                self.check_shape("pooler.dense.weight", &[h, h])?;
                self.check_shape("pooler.dense.bias", &[h])?;
            }
            Output::Relevance(Head::Prediction(..)) => {
                self.check_shape("head.dense.weight", &[h, h])?;
                self.check_shape("head.norm.weight", &[h])?;
            }
        }
        self.check_shape("classifier.weight", &[1, h])?;
        self.check_shape("classifier.bias", &[1])
    }

    fn validate_modernbert(&self, shape: Shape) -> Result<(), String> {
        let (h, inner) = (shape.hidden, shape.intermediate);
        self.check_shape("embeddings.tok_embeddings.weight", &[shape.vocab, h])?;
        self.check_shape("embeddings.norm.weight", &[h])?;
        for layer in 0..shape.layers {
            let at = |part: &str| format!("layers.{layer}.{part}.weight");
            // The first layer attends over the embedding norm's rows.
            if layer > 0 {
                self.check_shape(&at("attn_norm"), &[h])?;
            }
            self.check_shape(&at("attn.Wqkv"), &[3 * h, h])?;
            self.check_shape(&at("attn.Wo"), &[h, h])?;
            self.check_shape(&at("mlp_norm"), &[h])?;
            self.check_shape(&at("mlp.Wi"), &[2 * inner, h])?;
            self.check_shape(&at("mlp.Wo"), &[h, inner])?;
        }
        self.check_shape("final_norm.weight", &[h])
    }

    fn check_shape(&self, name: &str, dimensions: &[usize]) -> Result<(), String> {
        if self.serves_zeros(name) {
            return Ok(());
        }
        let key = self.key(name);
        let entry = self
            .entries
            .get(&key)
            .ok_or_else(|| format!("{key} missing"))?;
        if entry.shape != dimensions {
            return Err(format!(
                "{key} has shape {:?}, expected {dimensions:?}",
                entry.shape
            ));
        }
        Ok(())
    }

    fn key(&self, name: &str) -> String {
        match self.layout {
            Layout::Encoder => name.to_owned(),
            Layout::Bert if name.starts_with("classifier.") => name.to_owned(),
            Layout::Bert => format!("bert.{name}"),
            Layout::Roberta => {
                classification_head(name).unwrap_or_else(|| format!("roberta.{name}"))
            }
            Layout::Mpnet { nested: false } => mpnet_name(name),
            Layout::Mpnet { nested: true } => {
                classification_head(name).unwrap_or_else(|| format!("mpnet.{}", mpnet_name(name)))
            }
            Layout::ModernBert if name.starts_with("head.") || name.starts_with("classifier.") => {
                name.to_owned()
            }
            Layout::ModernBert => format!("model.{name}"),
        }
    }

    /// Whether `name` is the segment table MPNet lacks, read as zeros.
    fn serves_zeros(&self, name: &str) -> bool {
        matches!(self.layout, Layout::Mpnet { .. })
            && name == "embeddings.token_type_embeddings.weight"
    }
}

/// A RoBERTa-style head's dense and out_proj, served as BERT's pooler and classifier.
fn classification_head(name: &str) -> Option<String> {
    if let Some(part) = name.strip_prefix("pooler.dense.") {
        return Some(format!("classifier.dense.{part}"));
    }
    name.strip_prefix("classifier.")
        .map(|part| format!("classifier.out_proj.{part}"))
}

/// MPNet's name for the tensor BERT calls `name`.
fn mpnet_name(name: &str) -> String {
    for (bert, mpnet) in [
        (".attention.self.query.", ".attention.attn.q."),
        (".attention.self.key.", ".attention.attn.k."),
        (".attention.self.value.", ".attention.attn.v."),
        (".attention.output.dense.", ".attention.attn.o."),
        (".attention.output.LayerNorm.", ".attention.LayerNorm."),
    ] {
        if let Some((layer, part)) = name.split_once(bert) {
            return format!("{layer}{mpnet}{part}");
        }
    }
    name.to_owned()
}

/// Element types a checkpoint may store, each widening exactly to FP32.
#[derive(Clone, Copy)]
enum Dtype {
    F32,
    F16,
    Bf16,
}

impl Dtype {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "F32" => Some(Self::F32),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::Bf16),
            _ => None,
        }
    }

    fn bytes(self) -> usize {
        match self {
            Self::F32 => size_of::<f32>(),
            Self::F16 => size_of::<half::f16>(),
            Self::Bf16 => size_of::<half::bf16>(),
        }
    }

    fn widen(self, data: &[u8]) -> Vec<f32> {
        match self {
            Self::F32 => data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|value| f32::from_le_bytes(*value))
                .collect(),
            Self::F16 => data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|value| half::f16::from_le_bytes(*value).to_f32())
                .collect(),
            Self::Bf16 => data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|value| half::bf16::from_le_bytes(*value).to_f32())
                .collect(),
        }
    }
}

impl Tensors for SafeTensors<'_> {
    fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        if self.serves_zeros(name) {
            return Ok(vec![0.0; len]);
        }
        let key = self.key(name);
        let entry = self
            .entries
            .get(&key)
            .ok_or_else(|| format!("{key} missing"))?;
        let count = entry
            .shape
            .iter()
            .try_fold(1usize, |count, &size| count.checked_mul(size));
        let dtype = Dtype::parse(&entry.dtype)
            .filter(|_| count == Some(len))
            .ok_or_else(|| {
                format!(
                    "{key} is {} {:?}, expected F32, F16 or BF16 with {len} values",
                    entry.dtype, entry.shape
                )
            })?;
        let [start, end] = entry
            .data_offsets
            .map(|offset| self.data_start.checked_add(offset));
        let byte_len = len
            .checked_mul(dtype.bytes())
            .ok_or_else(|| format!("{key} byte length overflows"))?;
        let data = start
            .zip(end)
            .and_then(|(start, end)| self.bytes.get(start..end))
            .filter(|data| data.len() == byte_len)
            .ok_or_else(|| format!("{key} data lies outside the file"))?;
        Ok(dtype.widen(data))
    }
}

#[cfg(test)]
#[path = "checkpoint_tests.rs"]
mod tests;
