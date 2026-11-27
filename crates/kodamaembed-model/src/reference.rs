//! Double-precision encoder reference implementation for backend validation.

use crate::{
    Activation, Head, Pooling, Positions, Rotary, Shape, Tensors, relative, tensor, tensor_len,
};

struct Linear {
    weight: Vec<f64>,
    bias: Vec<f64>,
    inputs: usize,
}

impl Linear {
    fn load(
        tensors: &dyn Tensors,
        name: &str,
        outputs: usize,
        inputs: usize,
    ) -> Result<Self, String> {
        let mut linear = Self::unbiased(tensors, name, outputs, inputs)?;
        linear.bias = widen(tensor(tensors, &format!("{name}.bias"), outputs)?);
        Ok(linear)
    }

    fn unbiased(
        tensors: &dyn Tensors,
        name: &str,
        outputs: usize,
        inputs: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            weight: widen(tensor(
                tensors,
                &format!("{name}.weight"),
                tensor_len(&[outputs, inputs])?,
            )?),
            bias: vec![0.0; outputs],
            inputs,
        })
    }

    fn apply(&self, x: &[f64]) -> Vec<f64> {
        self.weight
            .chunks(self.inputs)
            .zip(&self.bias)
            .map(|(row, bias)| row.iter().zip(x).map(|(w, v)| w * v).sum::<f64>() + bias)
            .collect()
    }
}

struct Norm {
    gamma: Vec<f64>,
    beta: Vec<f64>,
    eps: f64,
}

impl Norm {
    fn load(tensors: &dyn Tensors, name: &str, width: usize, eps: f32) -> Result<Self, String> {
        let mut norm = Self::unbiased(tensors, name, width, eps)?;
        norm.beta = widen(tensor(tensors, &format!("{name}.bias"), width)?);
        Ok(norm)
    }

    fn unbiased(tensors: &dyn Tensors, name: &str, width: usize, eps: f32) -> Result<Self, String> {
        Ok(Self {
            gamma: widen(tensor(tensors, &format!("{name}.weight"), width)?),
            beta: vec![0.0; width],
            eps: f64::from(eps),
        })
    }

    fn apply(&self, x: &[f64]) -> Vec<f64> {
        let n = x.len() as f64;
        let mean = x.iter().sum::<f64>() / n;
        let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let rstd = 1.0 / (variance + self.eps).sqrt();
        x.iter()
            .zip(self.gamma.iter().zip(&self.beta))
            .map(|(v, (g, b))| (v - mean) * rstd * g + b)
            .collect()
    }
}

fn widen(values: Vec<f32>) -> Vec<f64> {
    values.into_iter().map(f64::from).collect()
}

fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Taylor series below 2.5, the erfc continued fraction above: about 1e-14.
fn erf(x: f64) -> f64 {
    if x.abs() < 2.5 {
        let (mut term, mut sum, mut n) = (x, x, 0.0);
        while term.abs() > 1e-17 * sum.abs().max(1e-300) {
            n += 1.0;
            term *= -x * x / n;
            sum += term / (2.0 * n + 1.0);
        }
        2.0 / std::f64::consts::PI.sqrt() * sum
    } else {
        let z = x.abs();
        let mut fraction = 0.0;
        for n in (1..60).rev() {
            fraction = f64::from(n) / 2.0 / (z + fraction);
        }
        let erfc = (-z * z).exp() / std::f64::consts::PI.sqrt() / (z + fraction);
        (1.0 - erfc).copysign(x)
    }
}

/// L2-normalized sentence vectors, pooled as `pooling` says, for sequences of
/// token `ids` packed back to back with the given `lengths` (all in segment 0).
pub fn encode(
    shape: &Shape,
    tensors: &dyn Tensors,
    pooling: Pooling,
    ids: &[i32],
    lengths: &[usize],
) -> Result<Vec<Vec<f64>>, String> {
    let hidden = hidden_states(shape, tensors, ids, &vec![0; ids.len()], lengths)?;
    let mut vectors = Vec::with_capacity(lengths.len());
    let mut start = 0;
    for &len in lengths {
        let mean = pooled(pooling, &hidden[start..start + len]);
        let norm = mean.iter().map(|m| m * m).sum::<f64>().sqrt().max(1e-12);
        vectors.push(mean.iter().map(|m| m / norm).collect());
        start += len;
    }
    Ok(vectors)
}

/// The mean of the rows of one sequence that `pooling` names.
fn pooled(pooling: Pooling, sequence: &[Vec<f64>]) -> Vec<f64> {
    let rows = &sequence[..pooling.rows(sequence.len())];
    (0..rows[0].len())
        .map(|d| rows.iter().map(|row| row[d]).sum::<f64>() / rows.len() as f64)
        .collect()
}

/// A head's features of one sequence's rows, which its classifier scores.
type Features = Box<dyn Fn(&[Vec<f64>]) -> Vec<f64>>;

/// A cross-encoder's score per sequence, through `head`.
pub fn rerank(
    shape: &Shape,
    tensors: &dyn Tensors,
    head: Head,
    ids: &[i32],
    types: &[i32],
    lengths: &[usize],
) -> Result<Vec<f64>, String> {
    let hidden = hidden_states(shape, tensors, ids, types, lengths)?;
    scores(shape, tensors, head, &hidden, lengths)
}

/// `head`'s score per sequence, from the final `hidden` rows of sequences packed back to back
/// with the given `lengths`.
pub fn scores(
    shape: &Shape,
    tensors: &dyn Tensors,
    head: Head,
    hidden: &[Vec<f64>],
    lengths: &[usize],
) -> Result<Vec<f64>, String> {
    let h = shape.hidden;
    let classifier = Linear::load(tensors, "classifier", 1, h)?;
    let features: Features = match head {
        Head::Pooler => {
            let pooler = Linear::load(tensors, "pooler.dense", h, h)?;
            Box::new(move |sequence| {
                let dense = pooler.apply(&sequence[0]);
                dense.into_iter().map(f64::tanh).collect()
            })
        }
        Head::Prediction(pooling, activation) => {
            let dense = Linear::unbiased(tensors, "head.dense", h, h)?;
            let norm = Norm::unbiased(tensors, "head.norm", h, shape.eps)?;
            let activate: fn(f64) -> f64 = match activation {
                Activation::Gelu => gelu,
                Activation::Silu => |x| x / (1.0 + (-x).exp()),
            };
            Box::new(move |sequence| {
                let dense = dense.apply(&pooled(pooling, sequence));
                norm.apply(&dense.into_iter().map(activate).collect::<Vec<f64>>())
            })
        }
    };
    let mut scores = Vec::with_capacity(lengths.len());
    let mut start = 0;
    for &len in lengths {
        scores.push(classifier.apply(&features(&hidden[start..start + len]))[0]);
        start += len;
    }
    Ok(scores)
}

fn hidden_states(
    shape: &Shape,
    tensors: &dyn Tensors,
    ids: &[i32],
    types: &[i32],
    lengths: &[usize],
) -> Result<Vec<Vec<f64>>, String> {
    shape.validate()?;
    let tokens = lengths
        .iter()
        .try_fold(0usize, |total, &len| total.checked_add(len))
        .ok_or("sequence lengths overflow usize")?;
    if lengths.is_empty() || lengths.contains(&0) || tokens != ids.len() || types.len() != ids.len()
    {
        return Err("IDs, segment types and nonempty sequence lengths must agree".into());
    }
    if lengths.iter().any(|&len| len > shape.max_sequence_len()) {
        return Err("a sequence exceeds the model's position embeddings".into());
    }
    if ids.iter().any(|&id| id < 0 || id as usize >= shape.vocab) {
        return Err("a token ID is outside the model's vocabulary".into());
    }
    if types
        .iter()
        .any(|&kind| kind < 0 || kind as usize >= shape.type_vocab)
    {
        return Err("a segment type is outside the model's vocabulary".into());
    }
    if let Positions::ModernBert(rotary) = shape.positions {
        return modernbert_states(shape, rotary, tensors, ids, lengths);
    }
    let (h, heads) = (shape.hidden, shape.heads);
    let head_dim = h / heads;
    let word = widen(tensor(
        tensors,
        "embeddings.word_embeddings.weight",
        shape.vocab * h,
    )?);
    let position = widen(tensor(
        tensors,
        "embeddings.position_embeddings.weight",
        shape.max_positions * h,
    )?);
    let token_type = widen(tensor(
        tensors,
        "embeddings.token_type_embeddings.weight",
        shape.type_vocab * h,
    )?);
    let embed_norm = Norm::load(tensors, "embeddings.LayerNorm", h, shape.eps)?;
    let relative = match shape.positions {
        Positions::Mpnet => Some(widen(tensor(
            tensors,
            relative::TENSOR,
            relative::BUCKETS * heads,
        )?)),
        Positions::Sequential | Positions::Roberta { .. } | Positions::ModernBert(_) => None,
    };

    let mut hidden: Vec<Vec<f64>> = Vec::with_capacity(ids.len());
    let positions = shape.position_ids(ids, lengths);
    for ((&id, &segment), &place) in ids.iter().zip(types).zip(&positions) {
        let (id, segment, place) = (id as usize, segment as usize, place as usize);
        let row: Vec<f64> = (0..h)
            .map(|d| (word[id * h + d] + token_type[segment * h + d]) + position[place * h + d])
            .collect();
        hidden.push(embed_norm.apply(&row));
    }

    for layer in 0..shape.layers {
        let at = |suffix: &str| format!("encoder.layer.{layer}.{suffix}");
        let query = Linear::load(tensors, &at("attention.self.query"), h, h)?;
        let key = Linear::load(tensors, &at("attention.self.key"), h, h)?;
        let value = Linear::load(tensors, &at("attention.self.value"), h, h)?;
        let output = Linear::load(tensors, &at("attention.output.dense"), h, h)?;
        let norm1 = Norm::load(tensors, &at("attention.output.LayerNorm"), h, shape.eps)?;
        let up = Linear::load(tensors, &at("intermediate.dense"), shape.intermediate, h)?;
        let down = Linear::load(tensors, &at("output.dense"), h, shape.intermediate)?;
        let norm2 = Norm::load(tensors, &at("output.LayerNorm"), h, shape.eps)?;

        let q: Vec<Vec<f64>> = hidden.iter().map(|x| query.apply(x)).collect();
        let k: Vec<Vec<f64>> = hidden.iter().map(|x| key.apply(x)).collect();
        let v: Vec<Vec<f64>> = hidden.iter().map(|x| value.apply(x)).collect();
        let mut context = vec![vec![0.0; h]; hidden.len()];
        let mut start = 0;
        for &len in lengths {
            for head in 0..heads {
                let span = head * head_dim..(head + 1) * head_dim;
                for i in start..start + len {
                    let scores: Vec<f64> = (start..start + len)
                        .map(|j| {
                            let score = q[i][span.clone()]
                                .iter()
                                .zip(&k[j][span.clone()])
                                .map(|(a, b)| a * b)
                                .sum::<f64>()
                                / (head_dim as f64).sqrt();
                            match &relative {
                                Some(weights) => {
                                    let offset = j as isize - i as isize;
                                    score + weights[relative::bucket(offset) * heads + head]
                                }
                                None => score,
                            }
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                    let total: f64 = weights.iter().sum();
                    for d in span.clone() {
                        context[i][d] = weights
                            .iter()
                            .zip(start..start + len)
                            .map(|(w, j)| w * v[j][d])
                            .sum::<f64>()
                            / total;
                    }
                }
            }
            start += len;
        }
        hidden = hidden
            .iter()
            .zip(&context)
            .map(|(x, c)| {
                let attended: Vec<f64> =
                    output.apply(c).iter().zip(x).map(|(o, r)| o + r).collect();
                let attended = norm1.apply(&attended);
                let inner: Vec<f64> = up.apply(&attended).into_iter().map(gelu).collect();
                let out: Vec<f64> = down
                    .apply(&inner)
                    .iter()
                    .zip(&attended)
                    .map(|(o, r)| o + r)
                    .collect();
                norm2.apply(&out)
            })
            .collect();
    }
    Ok(hidden)
}

/// A head's vector with each pair (m, m + half) turned by `angles[m]`.
fn rotated(vector: &[f64], angles: &[f32]) -> Vec<f64> {
    let (first, second) = vector.split_at(angles.len());
    let turn = |sign: f64, x: &[f64], y: &[f64]| -> Vec<f64> {
        x.iter()
            .zip(y)
            .zip(angles)
            .map(|((x, y), &angle)| {
                let angle = f64::from(angle);
                x * angle.cos() + sign * y * angle.sin()
            })
            .collect()
    };
    [turn(-1.0, first, second), turn(1.0, second, first)].concat()
}

/// ModernBERT's layers: a norm before each sublayer, queries and keys rotated by position,
/// a window on the local layers, a gated feed-forward, a final norm and no biases.
fn modernbert_states(
    shape: &Shape,
    rotary: Rotary,
    tensors: &dyn Tensors,
    ids: &[i32],
    lengths: &[usize],
) -> Result<Vec<Vec<f64>>, String> {
    let (h, heads, inner) = (shape.hidden, shape.heads, shape.intermediate);
    let head_dim = h / heads;
    let word = widen(tensor(
        tensors,
        "embeddings.tok_embeddings.weight",
        shape.vocab * h,
    )?);
    let embed_norm = Norm::unbiased(tensors, "embeddings.norm", h, shape.eps)?;
    let mut hidden: Vec<Vec<f64>> = ids
        .iter()
        .map(|&id| embed_norm.apply(&word[id as usize * h..][..h]))
        .collect();
    let places = shape.position_ids(ids, lengths);

    for layer in 0..shape.layers {
        let at = |suffix: &str| format!("layers.{layer}.{suffix}");
        let qkv = Linear::unbiased(tensors, &at("attn.Wqkv"), 3 * h, h)?;
        let output = Linear::unbiased(tensors, &at("attn.Wo"), h, h)?;
        let mlp_norm = Norm::unbiased(tensors, &at("mlp_norm"), h, shape.eps)?;
        let up = Linear::unbiased(tensors, &at("mlp.Wi"), 2 * inner, h)?;
        let down = Linear::unbiased(tensors, &at("mlp.Wo"), h, inner)?;
        // The first layer attends over the embedding norm's rows as they are.
        let attn_norm = match layer {
            0 => None,
            _ => Some(Norm::unbiased(tensors, &at("attn_norm"), h, shape.eps)?),
        };
        let theta = rotary.theta(layer);
        // Per token: q, k and v side by side, q and k turned head by head.
        let projected: Vec<Vec<f64>> = hidden
            .iter()
            .zip(&places)
            .map(|(x, &place)| {
                let angles: Vec<f32> = (0..head_dim / 2)
                    .map(|pair| Rotary::angle(theta, head_dim, pair, place as usize))
                    .collect();
                let mut row = match &attn_norm {
                    Some(norm) => qkv.apply(&norm.apply(x)),
                    None => qkv.apply(x),
                };
                for head in row[..2 * h].chunks_exact_mut(head_dim) {
                    head.copy_from_slice(&rotated(head, &angles));
                }
                row
            })
            .collect();
        let mut context = vec![vec![0.0; h]; hidden.len()];
        let mut start = 0;
        for &len in lengths {
            for head in 0..heads {
                let span = head * head_dim..(head + 1) * head_dim;
                for i in start..start + len {
                    let keys = match rotary.reach(layer) {
                        Some(reach) => {
                            i.saturating_sub(reach).max(start)..(i + reach + 1).min(start + len)
                        }
                        None => start..start + len,
                    };
                    let scores: Vec<f64> = keys
                        .clone()
                        .map(|j| {
                            projected[i][span.clone()]
                                .iter()
                                .zip(&projected[j][h..][span.clone()])
                                .map(|(q, k)| q * k)
                                .sum::<f64>()
                                / (head_dim as f64).sqrt()
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                    let total: f64 = weights.iter().sum();
                    for d in span.clone() {
                        context[i][d] = weights
                            .iter()
                            .zip(keys.clone())
                            .map(|(w, j)| w * projected[j][2 * h + d])
                            .sum::<f64>()
                            / total;
                    }
                }
            }
            start += len;
        }
        hidden = hidden
            .iter()
            .zip(&context)
            .map(|(x, c)| {
                let attended: Vec<f64> =
                    output.apply(c).iter().zip(x).map(|(o, r)| o + r).collect();
                let both = up.apply(&mlp_norm.apply(&attended));
                let (inputs, gates) = both.split_at(inner);
                let gated: Vec<f64> = inputs
                    .iter()
                    .zip(gates)
                    .map(|(&input, gate)| gelu(input) * gate)
                    .collect();
                down.apply(&gated)
                    .iter()
                    .zip(&attended)
                    .map(|(o, r)| o + r)
                    .collect()
            })
            .collect();
    }
    let final_norm = Norm::unbiased(tensors, "final_norm", h, shape.eps)?;
    Ok(hidden.iter().map(|x| final_norm.apply(x)).collect())
}

/// Reproducible uniform draws for [`Synthetic`]'s tensors.
struct Draws(u64);

impl Draws {
    fn draw(&mut self, len: usize, center: f32, spread: f32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = (self.0 >> 40) as f32 / (1u64 << 24) as f32;
                center + (unit * 2.0 - 1.0) * spread
            })
            .collect()
    }
}

/// A random checkpoint with its family's tensor names, for tests without real weights.
pub struct Synthetic {
    tensors: std::collections::BTreeMap<String, Vec<f32>>,
}

impl Synthetic {
    pub fn new(shape: &Shape, seed: u64) -> Self {
        shape.validate().expect("valid shape");
        let mut draws = Draws(seed);
        if let Positions::ModernBert(_) = shape.positions {
            return Self::modernbert(shape, &mut draws);
        }
        let mut draw = |len: usize, center: f32, spread: f32| draws.draw(len, center, spread);
        let (h, i) = (shape.hidden, shape.intermediate);
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert(
            "embeddings.word_embeddings.weight".into(),
            draw(shape.vocab * h, 0.0, 0.1),
        );
        tensors.insert(
            "embeddings.position_embeddings.weight".into(),
            draw(shape.max_positions * h, 0.0, 0.05),
        );
        tensors.insert(
            "embeddings.token_type_embeddings.weight".into(),
            draw(shape.type_vocab * h, 0.0, 0.02),
        );
        tensors.insert("embeddings.LayerNorm.weight".into(), draw(h, 1.0, 0.2));
        tensors.insert("embeddings.LayerNorm.bias".into(), draw(h, 0.0, 0.1));
        for layer in 0..shape.layers {
            let mut put = |suffix: &str, values: Vec<f32>| {
                tensors.insert(format!("encoder.layer.{layer}.{suffix}"), values);
            };
            for part in ["query", "key", "value"] {
                put(
                    &format!("attention.self.{part}.weight"),
                    draw(h * h, 0.0, 0.06),
                );
                put(&format!("attention.self.{part}.bias"), draw(h, 0.0, 0.1));
            }
            put("attention.output.dense.weight", draw(h * h, 0.0, 0.03));
            put("attention.output.dense.bias", draw(h, 0.0, 0.1));
            put("attention.output.LayerNorm.weight", draw(h, 1.0, 0.2));
            put("attention.output.LayerNorm.bias", draw(h, 0.0, 0.1));
            put("intermediate.dense.weight", draw(i * h, 0.0, 0.03));
            put("intermediate.dense.bias", draw(i, 0.0, 0.1));
            put("output.dense.weight", draw(h * i, 0.0, 0.015));
            put("output.dense.bias", draw(h, 0.0, 0.1));
            put("output.LayerNorm.weight", draw(h, 1.0, 0.2));
            put("output.LayerNorm.bias", draw(h, 0.0, 0.1));
        }
        if shape.positions == Positions::Mpnet {
            tensors.insert(
                relative::TENSOR.into(),
                draw(relative::BUCKETS * shape.heads, 0.0, 0.5),
            );
        }
        tensors.insert("pooler.dense.weight".into(), draw(h * h, 0.0, 0.05));
        tensors.insert("pooler.dense.bias".into(), draw(h, 0.0, 0.1));
        tensors.insert("classifier.weight".into(), draw(h, 0.0, 0.2));
        tensors.insert("classifier.bias".into(), draw(1, 0.0, 0.1));
        Self { tensors }
    }

    fn modernbert(shape: &Shape, draws: &mut Draws) -> Self {
        let (h, i) = (shape.hidden, shape.intermediate);
        let mut tensors = std::collections::BTreeMap::new();
        let mut put = |name: String, values: Vec<f32>| {
            tensors.insert(name, values);
        };
        put(
            "embeddings.tok_embeddings.weight".into(),
            draws.draw(shape.vocab * h, 0.0, 0.1),
        );
        put("embeddings.norm.weight".into(), draws.draw(h, 1.0, 0.2));
        for layer in 0..shape.layers {
            let at = |suffix: &str| format!("layers.{layer}.{suffix}");
            if layer > 0 {
                put(at("attn_norm.weight"), draws.draw(h, 1.0, 0.2));
            }
            put(at("attn.Wqkv.weight"), draws.draw(3 * h * h, 0.0, 0.06));
            put(at("attn.Wo.weight"), draws.draw(h * h, 0.0, 0.03));
            put(at("mlp_norm.weight"), draws.draw(h, 1.0, 0.2));
            put(at("mlp.Wi.weight"), draws.draw(2 * i * h, 0.0, 0.03));
            put(at("mlp.Wo.weight"), draws.draw(h * i, 0.0, 0.015));
        }
        put("final_norm.weight".into(), draws.draw(h, 1.0, 0.2));
        put("head.dense.weight".into(), draws.draw(h * h, 0.0, 0.05));
        put("head.norm.weight".into(), draws.draw(h, 1.0, 0.2));
        put("classifier.weight".into(), draws.draw(h, 0.0, 0.2));
        put("classifier.bias".into(), draws.draw(1, 0.0, 0.1));
        Self { tensors }
    }
}

impl Tensors for Synthetic {
    fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
        let values = self
            .tensors
            .get(name)
            .ok_or_else(|| format!("{name} missing"))?;
        if values.len() != len {
            return Err(format!(
                "{name} has {} values, expected {len}",
                values.len()
            ));
        }
        Ok(values.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Activation, Head, Positions, Rotary, relative};

    fn unit(row: &[f64]) -> Vec<f64> {
        let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        row.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn each_pooling_normalizes_the_rows_it_names() {
        let shape = Shape {
            layers: 1,
            hidden: 8,
            heads: 2,
            intermediate: 32,
            vocab: 10,
            max_positions: 8,
            type_vocab: 2,
            eps: 1e-12,
            positions: Positions::Sequential,
        };
        let model = Synthetic::new(&shape, 7);
        let (ids, lengths) = ([3, 1, 4, 1, 5], [2, 3]);
        let hidden = hidden_states(&shape, &model, &ids, &[0; 5], &lengths).unwrap();
        let mean = |rows: &[Vec<f64>]| -> Vec<f64> {
            (0..shape.hidden)
                .map(|d| rows.iter().map(|row| row[d]).sum::<f64>() / rows.len() as f64)
                .collect()
        };
        let cases = [
            (
                Pooling::Mean,
                [unit(&mean(&hidden[..2])), unit(&mean(&hidden[2..]))],
            ),
            (Pooling::Cls, [unit(&hidden[0]), unit(&hidden[2])]),
        ];
        for (pooling, expected) in cases {
            let vectors = encode(&shape, &model, pooling, &ids, &lengths).unwrap();
            for (vector, expected) in vectors.iter().zip(&expected) {
                for (got, want) in vector.iter().zip(expected) {
                    assert!((got - want).abs() < 1e-15, "{pooling:?}: {got} vs {want}");
                }
            }
        }
    }

    /// `model` with the first `skipped` values of its position table left out.
    struct Shifted<'a> {
        model: &'a Synthetic,
        skipped: usize,
    }

    impl Tensors for Shifted<'_> {
        fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
            if name != "embeddings.position_embeddings.weight" {
                return self.model.get(name, len);
            }
            Ok(self.model.get(name, len + self.skipped)?[self.skipped..].to_vec())
        }
    }

    /// Without the padding id, RoBERTa's positions are BERT's past the table's first pad + 1 rows.
    #[test]
    fn roberta_positions_read_the_position_table_past_its_padding_rows() {
        let roberta = Shape {
            layers: 1,
            hidden: 8,
            heads: 2,
            intermediate: 32,
            vocab: 10,
            max_positions: 10,
            type_vocab: 1,
            eps: 1e-12,
            positions: Positions::Roberta { pad: 1 },
        };
        let bert = Shape {
            max_positions: 8,
            positions: Positions::Sequential,
            ..roberta
        };
        let model = Synthetic::new(&roberta, 3);
        let shifted = Shifted {
            model: &model,
            skipped: 2 * roberta.hidden,
        };
        let (ids, lengths) = ([2, 5, 9, 3, 4, 6, 7, 8], [3, 5]);
        for pooling in [Pooling::Mean, Pooling::Cls] {
            assert_eq!(
                encode(&roberta, &model, pooling, &ids, &lengths).unwrap(),
                encode(&bert, &shifted, pooling, &ids, &lengths).unwrap()
            );
        }
    }

    /// `model` with every relative bias set to `value`.
    struct Uniform<'a> {
        model: &'a Synthetic,
        value: f32,
    }

    impl Tensors for Uniform<'_> {
        fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
            if name == relative::TENSOR {
                return Ok(vec![self.value; len]);
            }
            self.model.get(name, len)
        }
    }

    /// An MPNet shape, and the same shape without the relative bias.
    fn mpnet_and_roberta() -> (Shape, Shape) {
        let mpnet = Shape {
            layers: 1,
            hidden: 8,
            heads: 2,
            intermediate: 32,
            vocab: 10,
            max_positions: 12,
            type_vocab: 1,
            eps: 1e-12,
            positions: Positions::Mpnet,
        };
        let roberta = Shape {
            positions: Positions::Roberta { pad: 1 },
            ..mpnet
        };
        (mpnet, roberta)
    }

    fn largest_difference(a: &[Vec<f64>], b: &[Vec<f64>]) -> f64 {
        a.iter()
            .flatten()
            .zip(b.iter().flatten())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn a_bias_the_same_for_every_bucket_leaves_attention_unchanged() {
        let (mpnet, roberta) = mpnet_and_roberta();
        let model = Synthetic::new(&mpnet, 5);
        let uniform = Uniform {
            model: &model,
            value: 0.5,
        };
        let (ids, lengths) = ([2, 5, 9, 3, 4, 6, 7, 8], [3, 5]);
        for pooling in [Pooling::Mean, Pooling::Cls] {
            let plain = encode(&roberta, &model, pooling, &ids, &lengths).unwrap();
            let biased = encode(&mpnet, &uniform, pooling, &ids, &lengths).unwrap();
            assert!(largest_difference(&biased, &plain) < 1e-12);
        }
    }

    #[test]
    fn relative_bias_moves_the_vectors() {
        let (mpnet, roberta) = mpnet_and_roberta();
        let model = Synthetic::new(&mpnet, 5);
        let (ids, lengths) = ([2, 5, 9, 3, 4, 6, 7, 8], [3, 5]);
        for pooling in [Pooling::Mean, Pooling::Cls] {
            let plain = encode(&roberta, &model, pooling, &ids, &lengths).unwrap();
            let biased = encode(&mpnet, &model, pooling, &ids, &lengths).unwrap();
            assert!(largest_difference(&biased, &plain) > 1e-6);
        }
    }

    const ROTARY: Rotary = Rotary {
        global_theta: 160_000.0,
        local_theta: 10_000.0,
        local_layers: 0b0110,
        window: 16,
    };

    /// Four layers: the first and last attend over the whole sequence.
    fn modernbert(rotary: Rotary) -> Shape {
        Shape {
            layers: 4,
            hidden: 8,
            heads: 2,
            intermediate: 6,
            vocab: 10,
            max_positions: 64,
            type_vocab: 1,
            eps: 1e-5,
            positions: Positions::ModernBert(rotary),
        }
    }

    /// Each pooling and activation by its definition, from the reference's own final rows.
    #[test]
    fn a_prediction_head_scores_its_pooled_rows_through_a_dense_layer_an_activation_and_a_norm() {
        let shape = modernbert(ROTARY);
        let model = Synthetic::new(&shape, 9);
        let (ids, lengths) = ([2, 5, 9, 3, 4, 6, 7, 8], [3, 5]);
        let hidden = hidden_states(&shape, &model, &ids, &[0; 8], &lengths).unwrap();
        let h = shape.hidden;
        let wide = |name: &str, len: usize| widen(model.get(name, len).unwrap());
        let dense = wide("head.dense.weight", h * h);
        let gamma = wide("head.norm.weight", h);
        let classifier = wide("classifier.weight", h);
        let bias = wide("classifier.bias", 1)[0];
        let heads = [Pooling::Cls, Pooling::Mean]
            .into_iter()
            .flat_map(|pooling| [Activation::Gelu, Activation::Silu].map(|act| (pooling, act)));
        for (pooling, activation) in heads {
            let act = |x: f64| match activation {
                Activation::Gelu => gelu(x),
                Activation::Silu => x / (1.0 + (-x).exp()),
            };
            let head = Head::Prediction(pooling, activation);
            let scores = rerank(&shape, &model, head, &ids, &[0; 8], &lengths).unwrap();
            let mut start = 0;
            for (&len, &score) in lengths.iter().zip(&scores) {
                let rows = &hidden[start..start + pooling.rows(len)];
                let pooled: Vec<f64> = (0..h)
                    .map(|d| rows.iter().map(|row| row[d]).sum::<f64>() / rows.len() as f64)
                    .collect();
                let activated: Vec<f64> = dense
                    .chunks(h)
                    .map(|row| act(row.iter().zip(&pooled).map(|(w, x)| w * x).sum()))
                    .collect();
                let mean = activated.iter().sum::<f64>() / h as f64;
                let variance = activated.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / h as f64;
                let rstd = 1.0 / (variance + f64::from(shape.eps)).sqrt();
                let expected = activated
                    .iter()
                    .zip(gamma.iter().zip(&classifier))
                    .map(|(a, (g, c))| (a - mean) * rstd * g * c)
                    .sum::<f64>()
                    + bias;
                assert!(
                    (score - expected).abs() < 1e-12,
                    "{head:?}: {score} vs {expected}"
                );
                start += len;
            }
        }
    }

    #[test]
    fn rotation_turns_each_pair_of_a_head_by_its_angle() {
        let (a, b) = (0.5f32, 2.0f32);
        let turned = rotated(&[1.0, 0.0, 0.0, 1.0], &[a, b]);
        let (a, b) = (f64::from(a), f64::from(b));
        assert_eq!(turned, [a.cos(), -b.sin(), a.sin(), b.cos()]);
    }

    #[test]
    fn a_window_past_the_sequence_is_attention_over_all_of_it() {
        let one_base = Rotary {
            local_theta: ROTARY.global_theta,
            ..ROTARY
        };
        let windowed = modernbert(one_base);
        let global = modernbert(Rotary {
            local_layers: 0,
            ..one_base
        });
        let model = Synthetic::new(&windowed, 5);
        let (ids, lengths) = ([2, 5, 9, 3, 4, 6, 7, 8], [3, 5]);
        for pooling in [Pooling::Mean, Pooling::Cls] {
            assert_eq!(
                encode(&windowed, &model, pooling, &ids, &lengths).unwrap(),
                encode(&global, &model, pooling, &ids, &lengths).unwrap()
            );
        }
    }

    #[test]
    fn a_narrower_window_and_another_base_each_move_the_vectors() {
        let model = Synthetic::new(&modernbert(ROTARY), 5);
        let (ids, lengths) = ([2, 5, 9, 3, 4, 6, 7, 8], [8]);
        let vectors = |rotary: Rotary| {
            encode(&modernbert(rotary), &model, Pooling::Mean, &ids, &lengths).unwrap()
        };
        let whole = vectors(ROTARY);
        let narrow = vectors(Rotary {
            window: 2,
            ..ROTARY
        });
        let turned = vectors(Rotary {
            local_theta: 100.0,
            ..ROTARY
        });
        assert!(largest_difference(&narrow, &whole) > 1e-6);
        assert!(largest_difference(&turned, &whole) > 1e-6);
    }
}
