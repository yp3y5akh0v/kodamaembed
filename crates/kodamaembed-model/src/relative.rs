//! MPNet's relative-position attention bias.

use crate::{Positions, Shape, Tensors, tensor};

/// Relative-position buckets MPNet's forward sorts every key offset into.
pub const BUCKETS: usize = 32;

/// The checkpoint's bias per bucket and head, `[bucket][head]`.
pub const TENSOR: &str = "encoder.relative_attention_bias.weight";

/// Distances below this have a bucket each.
const EXACT: usize = 8;

/// The bucket of a key at `offset = key - query`; later keys take the upper half.
pub fn bucket(offset: isize) -> usize {
    let side = if offset > 0 { BUCKETS / 2 } else { 0 };
    let distance = offset.unsigned_abs();
    if distance < EXACT {
        return side + distance;
    }
    // transformers' 8 + floor(8 ln(n / 8) / ln(128 / 8)) is 8 + floor(log2((n / 8)^2)): integers
    // give it exactly, where float logarithms could round across a boundary.
    let log = distance.saturating_mul(distance).ilog2() - (EXACT * EXACT).ilog2();
    side + (EXACT + log as usize).min(BUCKETS / 2 - 1)
}

/// Each head's bias for every key-minus-query offset in a sequence, `[head][offset + longest - 1]`.
pub struct RelativeBias {
    values: Vec<f32>,
    longest: usize,
}

impl RelativeBias {
    /// MPNet's table for a valid `shape`'s longest sequence; `None` for models without one.
    pub fn load(shape: &Shape, tensors: &dyn Tensors) -> Result<Option<Self>, String> {
        if shape.positions != Positions::Mpnet {
            return Ok(None);
        }
        let weights = tensor(tensors, TENSOR, BUCKETS * shape.heads)?;
        Ok(Some(Self::new(
            &weights,
            shape.heads,
            shape.max_sequence_len(),
        )))
    }

    /// The table for sequences of at most `longest` tokens from `[bucket][head]` weights.
    pub fn new(weights: &[f32], heads: usize, longest: usize) -> Self {
        assert_eq!(weights.len(), BUCKETS * heads, "bucket weights");
        assert!(longest > 0, "positive longest");
        let span = 2 * longest - 1;
        let first = 1 - longest as isize;
        let values = (0..heads)
            .flat_map(|head| {
                (first..first + span as isize)
                    .map(move |offset| weights[bucket(offset) * heads + head])
            })
            .collect();
        Self { values, longest }
    }

    /// The bias of `query` over keys `0..len` for one head, `len` at most [`Self::longest`].
    pub fn keys(&self, head: usize, query: usize, len: usize) -> &[f32] {
        let span = 2 * self.longest - 1;
        &self.values[head * span + self.longest - 1 - query..][..len]
    }

    /// The whole table, as GPUs upload it.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// The longest sequence the table covers.
    pub fn longest(&self) -> usize {
        self.longest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_match_transformers_at_every_boundary() {
        // transformers 5.15.1 on torch 2.13.0+cpu.
        let expected = [
            (-128, 15),
            (-91, 15),
            (-90, 14),
            (-64, 14),
            (-63, 13),
            (-32, 12),
            (-31, 11),
            (-16, 10),
            (-15, 9),
            (-8, 8),
            (-7, 7),
            (0, 0),
            (7, 23),
            (8, 24),
            (15, 25),
            (16, 26),
            (32, 28),
            (64, 30),
            (90, 30),
            (91, 31),
            (511, 31),
        ];
        for (offset, expected) in expected {
            assert_eq!(bucket(offset), expected, "offset {offset}");
        }
    }

    #[test]
    fn each_query_reads_its_bucket_of_every_key() {
        let weights: Vec<f32> = (0..BUCKETS * 2)
            .map(|at| (at / 2 * 10 + at % 2) as f32)
            .collect();
        let table = RelativeBias::new(&weights, 2, 40);
        for (head, query) in [(0, 0), (1, 39), (1, 7)] {
            for (key, &value) in table.keys(head, query, 40).iter().enumerate() {
                let offset = key as isize - query as isize;
                assert_eq!(value, (bucket(offset) * 10 + head) as f32);
            }
        }
    }
}
