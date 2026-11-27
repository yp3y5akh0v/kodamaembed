//! Encoder shapes, checkpoint tensors and numerical contracts shared by backends.

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/yp3y5akh0v/kodamaembed/master/docs/assets/crest.svg",
    html_favicon_url = "https://raw.githubusercontent.com/yp3y5akh0v/kodamaembed/master/docs/assets/favicon.ico"
)]

pub mod reference;
pub mod relative;
pub mod repro;

/// Fingerprinted sources: transforms like [`fused_qkv`] and the [`repro`] spec
/// change outputs without changing checkpoint bytes.
pub const NUMERICAL_SOURCES: &[&[u8]] = &[
    include_bytes!("lib.rs"),
    include_bytes!("relative.rs"),
    include_bytes!("repro.rs"),
];

/// The gated feed-forward's contraction reads its depth, the intermediate width, in the GPU
/// GEMMs' steps of this many values.
const GATE_TILE: usize = 64;

/// The checkpoint shape.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub max_positions: usize,
    pub type_vocab: usize,
    pub eps: f32,
    pub positions: Positions,
}

impl Shape {
    /// Run before a backend allocates or indexes checkpoint tensors.
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("layers", self.layers),
            ("hidden", self.hidden),
            ("heads", self.heads),
            ("intermediate", self.intermediate),
            ("vocab", self.vocab),
            ("max_positions", self.max_positions),
            ("type_vocab", self.type_vocab),
        ] {
            if value == 0 {
                return Err(format!("{name} must be positive"));
            }
        }
        if !self.hidden.is_multiple_of(self.heads) {
            return Err(format!(
                "hidden {} does not split into {} heads",
                self.hidden, self.heads
            ));
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return Err("layer norm epsilon must be finite and positive".into());
        }
        if self.max_sequence_len() == 0 {
            return Err(format!(
                "{:?} positions leave no room in a table of {}",
                self.positions, self.max_positions
            ));
        }
        // Token, segment and position IDs are signed 32-bit values in every backend.
        if [self.vocab, self.type_vocab, self.max_positions]
            .iter()
            .any(|&rows| rows > i32::MAX as usize)
        {
            return Err("vocabulary or position dimensions exceed signed 32-bit IDs".into());
        }
        for dimensions in [
            &[3, self.hidden, self.hidden][..],
            &[self.intermediate, self.hidden],
            &[self.vocab, self.hidden],
            &[self.max_positions, self.hidden],
            &[self.type_vocab, self.hidden],
        ] {
            tensor_len(dimensions)?;
        }
        if let Positions::ModernBert(rotary) = self.positions {
            if self.type_vocab != 1 {
                return Err(format!(
                    "ModernBERT embeds no segment: {} segment types",
                    self.type_vocab
                ));
            }
            if self.layers > Rotary::MAX_LAYERS {
                return Err(format!(
                    "ModernBERT with {} layers, where KodamaEmbed runs up to {} layers",
                    self.layers,
                    Rotary::MAX_LAYERS
                ));
            }
            let head = self.hidden / self.heads;
            if !head.is_multiple_of(2) {
                return Err(format!("a head {head} wide does not split into pairs"));
            }
            if rotary.window == 0 {
                return Err("the attention window must be positive".into());
            }
            if [rotary.global_theta, rotary.local_theta]
                .iter()
                .any(|theta| !theta.is_finite() || *theta <= 0.0)
            {
                return Err("rotation bases must be finite and positive".into());
            }
            // The gated feed-forward's up projection holds both halves.
            tensor_len(&[2, self.intermediate, self.hidden])?;
        }
        Ok(())
    }

    /// The kernel geometry this shape runs on, when KodamaEmbed's GPU kernels take its
    /// feed-forward: BERT's four times wider, ModernBERT's gate in whole GEMM tiles.
    pub fn geometry(&self) -> Option<Geometry> {
        let feed_forward = match self.positions {
            Positions::ModernBert(_) => self.intermediate.is_multiple_of(GATE_TILE),
            _ => self.hidden.checked_mul(4) == Some(self.intermediate),
        };
        Geometry::of(self.hidden, self.heads).filter(|_| feed_forward)
    }

    /// The most tokens a sequence holds before its positions leave the table.
    pub fn max_sequence_len(&self) -> usize {
        match self.positions.padding() {
            None => self.max_positions,
            Some(pad) => usize::try_from(pad)
                .ok()
                .and_then(|pad| self.max_positions.checked_sub(pad + 1))
                .unwrap_or(0),
        }
    }

    /// Each token's position in sequences packed back to back; `lengths` sum to `ids.len()`.
    pub fn position_ids(&self, ids: &[i32], lengths: &[usize]) -> Vec<i32> {
        let mut positions = Vec::with_capacity(ids.len());
        let mut rest = ids;
        for &len in lengths {
            let (sequence, after) = rest.split_at(len);
            rest = after;
            match self.positions.padding() {
                None => positions.extend(0..len as i32),
                Some(pad) => {
                    let mut last = pad;
                    positions.extend(sequence.iter().map(|&id| {
                        if id == pad {
                            return pad;
                        }
                        last += 1;
                        last
                    }));
                }
            }
        }
        positions
    }
}

/// How a model encodes its tokens' positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Positions {
    /// From 0 in each sequence, as BERT numbers them.
    Sequential,
    /// RoBERTa's: the padding id sits at `pad`, every other token at `pad + 1` plus the
    /// non-padding tokens before it.
    Roberta { pad: i32 },
    /// MPNet's: RoBERTa's numbering past padding id 1, plus a learned bias per head for each
    /// relative-position bucket ([`relative`]) in every attention layer.
    Mpnet,
    /// ModernBERT's: no position table; every attention layer rotates queries and keys by
    /// position ([`Rotary`]). The family's layers also normalize before each sublayer, gate
    /// the feed-forward and carry no biases.
    ModernBert(Rotary),
}

impl Positions {
    /// The padding id RoBERTa's numbering starts past, if the model uses it.
    fn padding(self) -> Option<i32> {
        match self {
            Self::Sequential | Self::ModernBert(_) => None,
            Self::Roberta { pad } => Some(pad),
            // transformers hard-codes MPNet's padding index whatever pad_token_id says.
            Self::Mpnet => Some(1),
        }
    }
}

/// ModernBERT's rotary positions: the bases its layers rotate by and the layers that attend
/// inside a window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rotary {
    /// The base where a layer attends over the whole sequence.
    pub global_theta: f32,
    /// The base where a layer attends inside the window.
    pub local_theta: f32,
    /// The layers that attend inside the window: bit `layer` is set for each.
    pub local_layers: u64,
    /// The tokens a query attends over in those layers, half each side of itself.
    pub window: usize,
}

impl Rotary {
    /// The most layers `local_layers` names.
    pub const MAX_LAYERS: usize = u64::BITS as usize;

    /// The keys a query of `layer` sees each side of itself; None over the whole sequence.
    pub fn reach(&self, layer: usize) -> Option<usize> {
        let local = u32::try_from(layer)
            .ok()
            .and_then(|layer| self.local_layers.checked_shr(layer))
            .is_some_and(|layers| layers & 1 == 1);
        local.then_some(self.window / 2)
    }

    /// The rotation base of `layer`.
    pub fn theta(&self, layer: usize) -> f32 {
        match self.reach(layer) {
            Some(_) => self.local_theta,
            None => self.global_theta,
        }
    }

    /// The angle pair `pair` of a `head_dim`-wide head turns at `position`, in transformers'
    /// float32 steps: the power, its inverse, the product.
    pub fn angle(theta: f32, head_dim: usize, pair: usize, position: usize) -> f32 {
        let exponent = (2 * pair) as f32 / head_dim as f32;
        let frequency = 1.0 / (f64::from(theta).powf(f64::from(exponent)) as f32);
        position as f32 * frequency
    }

    /// The cosine and sine of [`Self::angle`], rounded from float64: every backend turns by
    /// these values.
    pub fn turn(theta: f32, head_dim: usize, pair: usize, position: usize) -> (f32, f32) {
        let angle = f64::from(Self::angle(theta, head_dim, pair, position));
        (angle.cos() as f32, angle.sin() as f32)
    }

    /// Every pair's [`Self::turn`] at the first `positions` places of a sequence: `head_dim`
    /// values a position, its cosines and then its sines.
    pub fn turns(theta: f32, head_dim: usize, positions: usize) -> Vec<f32> {
        let mut table = Vec::with_capacity(positions * head_dim);
        for position in 0..positions {
            let turns = (0..head_dim / 2).map(|pair| Self::turn(theta, head_dim, pair, position));
            let (cosines, sines): (Vec<f32>, Vec<f32>) = turns.unzip();
            table.extend(cosines);
            table.extend(sines);
        }
        table
    }

    /// [`Self::turns`] with each pair's cosine beside its sine, as a projection that turns its
    /// own column pairs reads them.
    pub fn paired_turns(theta: f32, head_dim: usize, positions: usize) -> Vec<f32> {
        Self::turns(theta, head_dim, positions)
            .chunks_exact(head_dim)
            .flat_map(|turn| {
                let (cosines, sines) = turn.split_at(head_dim / 2);
                cosines.iter().zip(sines).flat_map(|(&c, &s)| [c, s])
            })
            .collect()
    }
}

/// Which final hidden rows of a sequence its sentence vector or its prediction head averages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Every row.
    Mean,
    /// The first row, the `[CLS]` token's.
    Cls,
}

impl Pooling {
    /// The leading rows of a `len`-row sequence that it averages.
    pub fn rows(self, len: usize) -> usize {
        match self {
            Self::Mean => len,
            Self::Cls => 1,
        }
    }
}

/// What an encoder returns per sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Output {
    /// An L2-normalized sentence vector, `hidden` values.
    Embedding(Pooling),
    /// A cross-encoder's relevance score through its head, one value.
    Relevance(Head),
}

impl Output {
    /// Values per sequence of a model `hidden` wide.
    pub fn width(self, hidden: usize) -> usize {
        match self {
            Self::Embedding(_) => hidden,
            Self::Relevance(_) => 1,
        }
    }
}

/// How a cross-encoder scores a sequence from its final hidden rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Head {
    /// BERT's pooler: classifier . tanh(dense . first row + bias) + bias.
    Pooler,
    /// ModernBERT's prediction head: classifier . norm(activation(dense . pooled rows)) + bias.
    Prediction(Pooling, Activation),
}

/// What a prediction head applies to its dense layer's outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// The exact GELU, x times the normal distribution's share below x.
    Gelu,
    /// x * sigmoid(x).
    Silu,
}

/// The model widths and head layouts KodamaEmbed's kernels are specialized to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Geometry {
    /// 384 wide, 12 heads of 32 (MiniLM).
    Small,
    /// 768 wide, 12 heads of 64 (BERT-base, XLM-R base).
    Base,
    /// 1024 wide, 16 heads of 64 (BERT-large, e5-large).
    Large,
}

impl Geometry {
    pub fn of(hidden: usize, heads: usize) -> Option<Self> {
        match (hidden, heads) {
            (384, 12) => Some(Self::Small),
            (768, 12) => Some(Self::Base),
            (1024, 16) => Some(Self::Large),
            _ => None,
        }
    }

    pub fn hidden(self) -> usize {
        match self {
            Self::Small => 384,
            Self::Base => 768,
            Self::Large => 1024,
        }
    }

    pub fn heads(self) -> usize {
        match self {
            Self::Small | Self::Base => 12,
            Self::Large => 16,
        }
    }

    pub fn head_dim(self) -> usize {
        self.hidden() / self.heads()
    }
}

/// The encoder's stages, in the order they run within a layer.
#[derive(Debug, Clone, Copy)]
pub enum Stage {
    /// Word, segment and position embeddings, summed and normalized.
    Embed,
    /// The fused query, key and value projection.
    Qkv,
    /// Softmax attention within each sequence.
    Attention,
    /// The attention output projection and its residual.
    Out,
    /// The layer norm after attention.
    Norm1,
    /// The feed-forward up projection and its GELU.
    Ffn1,
    /// The feed-forward down projection and its residual.
    Ffn2,
    /// The layer norm after the feed-forward block.
    Norm2,
    /// Pooling and normalization, or a cross-encoder's scoring head.
    Pool,
}

/// Every stage, at the index `Stage as usize` gives it.
pub const STAGES: [Stage; 9] = [
    Stage::Embed,
    Stage::Qkv,
    Stage::Attention,
    Stage::Out,
    Stage::Norm1,
    Stage::Ffn1,
    Stage::Ffn2,
    Stage::Norm2,
    Stage::Pool,
];

/// Time per stage over the profiled batches: the GPU timeline on CUDA, where a
/// gap counts toward the next stage, and wall time on the CPU.
#[derive(Debug, Clone, Default)]
pub struct Profile {
    /// Batches profiled.
    pub batches: usize,
    /// Tokens in those batches.
    pub tokens: usize,
    /// Milliseconds per stage, indexed like [`STAGES`].
    pub milliseconds: [f64; STAGES.len()],
}

/// FP32 checkpoint tensors by name, with the expected element count.
pub trait Tensors {
    fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String>;
}

/// Element count, checked against overflow and Rust's largest allocation.
pub fn tensor_len(dimensions: &[usize]) -> Result<usize, String> {
    let elements = dimensions
        .iter()
        .try_fold(1usize, |size, &dimension| size.checked_mul(dimension));
    match elements {
        Some(elements) if elements <= isize::MAX as usize / size_of::<f32>() => Ok(elements),
        _ => Err(format!(
            "tensor dimensions {dimensions:?} exceed addressable memory"
        )),
    }
}

/// Checks length and finiteness, which a provider need not enforce.
pub fn tensor(tensors: &dyn Tensors, name: &str, len: usize) -> Result<Vec<f32>, String> {
    tensor_len(&[len])?;
    let values = tensors.get(name, len)?;
    if values.len() != len {
        return Err(format!(
            "{name} has {} values, expected {len}",
            values.len()
        ));
    }
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(format!(
            "{name} contains a non-finite value at element {index}"
        ));
    }
    Ok(values)
}

/// Layer `layer`'s query, key and value projections stacked as one
/// `[3 * hidden, hidden]` weight and `[3 * hidden]` bias.
pub fn fused_qkv(
    tensors: &dyn Tensors,
    shape: &Shape,
    layer: usize,
) -> Result<(Vec<f32>, Vec<f32>), String> {
    let h = shape.hidden;
    let at = |part: &str, kind: &str| format!("encoder.layer.{layer}.attention.self.{part}.{kind}");
    let mut weight = Vec::with_capacity(tensor_len(&[3, h, h])?);
    let mut bias = Vec::with_capacity(tensor_len(&[3, h])?);
    for part in ["query", "key", "value"] {
        weight.extend(tensor(tensors, &at(part, "weight"), tensor_len(&[h, h])?)?);
        bias.extend(tensor(tensors, &at(part, "bias"), h)?);
    }
    Ok((weight, bias))
}

/// A gated feed-forward's up projection, `k` values a row, with each input's row before its
/// gate's: columns 2j and 2j + 1 of its output are then input j and its gate.
pub fn pair_gates(weights: &[f32], k: usize) -> Vec<f32> {
    let (inputs, gates) = weights.split_at(weights.len() / 2);
    inputs
        .chunks_exact(k)
        .zip(gates.chunks_exact(k))
        .flat_map(|(input, gate)| input.iter().chain(gate))
        .copied()
        .collect()
}

/// A projection, `k` values a row, with rows m and m + head / 2 of each head among its first
/// `rows` side by side, the pairs rotary positions turn. A head's dot products ignore the order.
pub fn pair_turns(weights: &[f32], k: usize, head: usize, rows: usize) -> Vec<f32> {
    let mut paired = weights.to_vec();
    let heads = weights[..rows * k].chunks_exact(head * k);
    for (target, source) in paired.chunks_exact_mut(head * k).zip(heads) {
        let (first, second) = source.split_at(head / 2 * k);
        let pairs = first.chunks_exact(k).zip(second.chunks_exact(k));
        for (pair, (x, y)) in target.chunks_exact_mut(2 * k).zip(pairs) {
            pair[..k].copy_from_slice(x);
            pair[k..].copy_from_slice(y);
        }
    }
    paired
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Named(Vec<(String, Vec<f32>)>);

    impl Tensors for Named {
        fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
            let (_, values) = self
                .0
                .iter()
                .find(|(key, _)| key == name)
                .ok_or(format!("{name} missing"))?;
            assert_eq!(values.len(), len);
            Ok(values.clone())
        }
    }

    #[test]
    fn projections_stack_in_query_key_value_order() {
        let shape = Shape {
            layers: 1,
            hidden: 2,
            heads: 1,
            intermediate: 8,
            vocab: 1,
            max_positions: 1,
            type_vocab: 1,
            eps: 1e-12,
            positions: Positions::Sequential,
        };
        let name = |part: &str, kind: &str| format!("encoder.layer.0.attention.self.{part}.{kind}");
        let mut named = Vec::new();
        for (index, part) in ["query", "key", "value"].iter().enumerate() {
            let base = index as f32 * 10.0;
            named.push((
                name(part, "weight"),
                vec![base, base + 1.0, base + 2.0, base + 3.0],
            ));
            named.push((name(part, "bias"), vec![base + 5.0, base + 6.0]));
        }
        let (weight, bias) = fused_qkv(&Named(named), &shape, 0).unwrap();
        assert_eq!(
            weight,
            [
                0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0
            ]
        );
        assert_eq!(bias, [5.0, 6.0, 15.0, 16.0, 25.0, 26.0]);
    }

    #[test]
    fn gates_sit_beside_their_inputs() {
        // Two inputs' rows, then their gates', three values a row.
        let weights: Vec<f32> = (0..12u8).map(f32::from).collect();
        assert_eq!(
            pair_gates(&weights, 3),
            [0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 3.0, 4.0, 5.0, 9.0, 10.0, 11.0]
        );
    }

    #[test]
    fn a_heads_rows_sit_beside_the_rows_half_a_head_away() {
        // Two heads of four rows turn, two values a row; the last four rows stay.
        let weights: Vec<f32> = (0..24u8).map(f32::from).collect();
        let rows = |order: [u8; 12]| -> Vec<f32> {
            order
                .into_iter()
                .flat_map(|row| [f32::from(2 * row), f32::from(2 * row + 1)])
                .collect()
        };
        assert_eq!(
            pair_turns(&weights, 2, 4, 8),
            rows([0, 2, 1, 3, 4, 6, 5, 7, 8, 9, 10, 11])
        );
    }

    #[test]
    fn paired_turns_hold_each_pairs_cosine_beside_its_sine() {
        let (theta, head, positions) = (10_000.0, 8, 3);
        let table = Rotary::paired_turns(theta, head, positions);
        assert_eq!(table.len(), positions * head);
        for position in 0..positions {
            for pair in 0..head / 2 {
                let at = position * head + 2 * pair;
                assert_eq!(
                    (table[at], table[at + 1]),
                    Rotary::turn(theta, head, pair, position),
                    "position {position}, pair {pair}"
                );
            }
        }
    }

    struct Unchecked(Vec<f32>);

    impl Tensors for Unchecked {
        fn get(&self, _: &str, _: usize) -> Result<Vec<f32>, String> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn tensor_providers_cannot_supply_wrong_lengths_or_nonfinite_weights() {
        for values in [vec![], vec![1.0, 2.0], vec![f32::NAN], vec![f32::INFINITY]] {
            assert!(tensor(&Unchecked(values), "test", 1).is_err());
        }
        assert_eq!(tensor(&Unchecked(vec![2.0]), "test", 1).unwrap(), [2.0]);
    }

    #[test]
    fn mean_pooling_averages_every_row_and_cls_pooling_the_first() {
        for len in [1, 2, 512] {
            assert_eq!(Pooling::Mean.rows(len), len);
            assert_eq!(Pooling::Cls.rows(len), 1);
        }
    }

    #[test]
    fn embeddings_are_hidden_wide_and_scores_one_value() {
        for pooling in [Pooling::Mean, Pooling::Cls] {
            assert_eq!(Output::Embedding(pooling).width(384), 384);
        }
        for head in [
            Head::Pooler,
            Head::Prediction(Pooling::Mean, Activation::Silu),
        ] {
            assert_eq!(Output::Relevance(head).width(384), 1);
        }
    }

    #[test]
    fn dimension_products_are_checked_before_allocation() {
        assert_eq!(tensor_len(&[3, 5, 7]).unwrap(), 105);
        assert!(tensor_len(&[usize::MAX, 2]).is_err());
        assert!(tensor_len(&[isize::MAX as usize]).is_err());
    }

    fn shape(max_positions: usize, positions: Positions) -> Shape {
        Shape {
            layers: 1,
            hidden: 2,
            heads: 1,
            intermediate: 8,
            vocab: 16,
            max_positions,
            type_vocab: 1,
            eps: 1e-12,
            positions,
        }
    }

    #[test]
    fn base_models_take_the_768_wide_geometry() {
        assert_eq!(Geometry::of(768, 12), Some(Geometry::Base));
        assert_eq!(Geometry::Base.head_dim(), 64);
    }

    #[test]
    fn bert_counts_positions_from_zero_in_each_sequence() {
        let shape = shape(512, Positions::Sequential);
        assert_eq!(
            shape.position_ids(&[7, 8, 9, 7, 8], &[3, 2]),
            [0, 1, 2, 0, 1]
        );
    }

    #[test]
    fn roberta_counts_past_the_padding_id_which_keeps_its_own_position() {
        let shape = shape(514, Positions::Roberta { pad: 1 });
        let ids = [0, 5, 1, 7, 2, 0, 9, 2];
        assert_eq!(shape.position_ids(&ids, &[5, 3]), [2, 3, 1, 4, 5, 2, 3, 4]);
    }

    #[test]
    fn mpnet_positions_are_robertas_from_padding_id_one() {
        let ids = [0, 5, 1, 7, 2, 0, 9, 2];
        assert_eq!(
            shape(514, Positions::Mpnet).position_ids(&ids, &[5, 3]),
            shape(514, Positions::Roberta { pad: 1 }).position_ids(&ids, &[5, 3])
        );
        assert_eq!(shape(514, Positions::Mpnet).max_sequence_len(), 512);
    }

    #[test]
    fn the_padding_rows_shorten_the_longest_sequence() {
        assert_eq!(shape(514, Positions::Sequential).max_sequence_len(), 514);
        assert_eq!(
            shape(514, Positions::Roberta { pad: 1 }).max_sequence_len(),
            512
        );
    }

    /// Seven layers, every third from the first over the whole sequence.
    const ROTARY: Rotary = Rotary {
        global_theta: 160_000.0,
        local_theta: 10_000.0,
        local_layers: 0b011_0110,
        window: 128,
    };

    fn modernbert(rotary: Rotary) -> Shape {
        Shape {
            layers: 7,
            hidden: 8,
            max_positions: 8192,
            ..shape(8192, Positions::ModernBert(rotary))
        }
    }

    #[test]
    fn modernbert_names_each_layer_that_attends_inside_the_window() {
        let listed = Rotary {
            local_layers: 0b1001,
            ..ROTARY
        };
        let layers: Vec<Option<usize>> = (0..5).map(|layer| listed.reach(layer)).collect();
        assert_eq!(layers, [Some(64), None, None, Some(64), None]);
        assert_eq!([listed.theta(0), listed.theta(1)], [10_000.0, 160_000.0]);
        // A layer past the set attends over the whole sequence.
        assert_eq!(
            Rotary {
                local_layers: u64::MAX,
                ..ROTARY
            }
            .reach(64),
            None
        );
        let deep = |layers| Shape {
            layers,
            ..modernbert(ROTARY)
        };
        assert!(deep(Rotary::MAX_LAYERS).validate().is_ok());
        assert!(
            deep(Rotary::MAX_LAYERS + 1)
                .validate()
                .unwrap_err()
                .contains("64 layers")
        );
    }

    #[test]
    fn modernbert_rotates_pairs_inside_a_window_on_its_local_layers() {
        assert!(modernbert(ROTARY).validate().is_ok());
        let odd_heads = Shape {
            hidden: 6,
            heads: 2,
            ..modernbert(ROTARY)
        };
        assert!(odd_heads.validate().is_err());
        let segments = Shape {
            type_vocab: 2,
            ..modernbert(ROTARY)
        };
        assert!(segments.validate().unwrap_err().contains("segment"));
        for broken in [
            Rotary {
                window: 0,
                ..ROTARY
            },
            Rotary {
                global_theta: 0.0,
                ..ROTARY
            },
            Rotary {
                local_theta: f32::NAN,
                ..ROTARY
            },
            Rotary {
                local_theta: f32::INFINITY,
                ..ROTARY
            },
        ] {
            assert!(modernbert(broken).validate().is_err(), "{broken:?}");
        }
        let layers: Vec<Option<usize>> = (0..7).map(|layer| ROTARY.reach(layer)).collect();
        assert_eq!(
            layers,
            [None, Some(64), Some(64), None, Some(64), Some(64), None]
        );
        assert_eq!(
            [ROTARY.theta(0), ROTARY.theta(1), ROTARY.theta(3)],
            [160_000.0, 10_000.0, 160_000.0]
        );
    }

    #[test]
    fn rotation_angles_are_positions_scaled_by_falling_powers_of_the_base() {
        // The first pair turns one radian a position.
        assert_eq!(Rotary::angle(10_000.0, 64, 0, 7), 7.0);
        assert_eq!(Rotary::angle(10_000.0, 64, 31, 0), 0.0);
        // 10000^(-2/64) and 10000^(-62/64), within float32's spacing at each angle.
        let second = f64::from(Rotary::angle(10_000.0, 64, 1, 511));
        assert!(
            (second - 511.0 * 0.749_894_209_332_455_8).abs() < 5e-5,
            "{second}"
        );
        let last = f64::from(Rotary::angle(10_000.0, 64, 31, 8191));
        assert!(
            (last - 8191.0 * 1.333_521_432_163_324e-4).abs() < 1e-6,
            "{last}"
        );
    }

    #[test]
    fn a_turn_is_the_cosine_and_sine_of_its_angle() {
        assert_eq!(Rotary::turn(10_000.0, 64, 31, 0), (1.0, 0.0));
        let (cos, sin) = Rotary::turn(10_000.0, 64, 0, 7);
        assert!((f64::from(cos) - 7f64.cos()).abs() < 1e-7, "{cos}");
        assert!((f64::from(sin) - 7f64.sin()).abs() < 1e-7, "{sin}");
    }

    #[test]
    fn the_turns_table_holds_each_positions_cosines_then_its_sines() {
        let table = Rotary::turns(10_000.0, 4, 3);
        assert_eq!(table.len(), 3 * 4);
        assert_eq!(table[..4], [1.0, 1.0, 0.0, 0.0]);
        for (position, row) in table.as_chunks::<4>().0.iter().enumerate() {
            for pair in 0..2 {
                let (cos, sin) = Rotary::turn(10_000.0, 4, pair, position);
                assert_eq!((row[pair], row[2 + pair]), (cos, sin), "{position}, {pair}");
            }
        }
    }

    #[test]
    fn modernbert_counts_positions_from_zero() {
        let shape = modernbert(ROTARY);
        assert_eq!(
            shape.position_ids(&[7, 8, 9, 7, 8], &[3, 2]),
            [0, 1, 2, 0, 1]
        );
        assert_eq!(shape.max_sequence_len(), 8192);
    }

    #[test]
    fn modernbert_takes_a_gpu_geometry_when_its_gate_is_whole_gemm_tiles() {
        // granite-small's width, heads and feed-forward are MiniLM's.
        let small = Shape {
            hidden: 384,
            heads: 12,
            intermediate: 1536,
            ..modernbert(ROTARY)
        };
        assert_eq!(small.geometry(), Some(Geometry::Small));
        let base = Shape {
            hidden: 768,
            intermediate: 1152,
            ..small
        };
        assert_eq!(base.geometry(), Some(Geometry::Base));
        let large = Shape {
            hidden: 1024,
            heads: 16,
            intermediate: 2624,
            ..small
        };
        assert_eq!(large.geometry(), Some(Geometry::Large));
        for unfit in [
            Shape {
                intermediate: 1160,
                ..base
            },
            Shape { heads: 8, ..base },
            // BERT's layers keep their four-times-wider feed-forward.
            Shape {
                positions: Positions::Sequential,
                ..base
            },
        ] {
            assert_eq!(unfit.geometry(), None, "{unfit:?}");
        }
        assert_eq!(
            Shape {
                positions: Positions::Sequential,
                ..small
            }
            .geometry(),
            Some(Geometry::Small)
        );
    }

    #[test]
    fn a_padding_id_that_leaves_no_positions_is_rejected() {
        for pad in [-1, 513] {
            assert!(shape(514, Positions::Roberta { pad }).validate().is_err());
        }
        assert!(
            shape(514, Positions::Roberta { pad: 512 })
                .validate()
                .is_ok()
        );
    }
}
