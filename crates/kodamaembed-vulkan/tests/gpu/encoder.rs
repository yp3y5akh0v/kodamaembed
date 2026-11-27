use crate::support::{GEOMETRIES, Rng, devices};
use kodamaembed_model::{
    Activation, Head, Output, Pooling, Positions, Rotary, Shape, Tensors,
    reference::{self, Synthetic},
};
use kodamaembed_vulkan::{Encoder, EncoderError};

/// Each GPU geometry's width and heads.
fn widths() -> impl Iterator<Item = (usize, usize)> {
    GEOMETRIES
        .into_iter()
        .map(|geometry| (geometry.hidden(), geometry.heads()))
}

const OUTPUTS: [Output; 3] = [
    Output::Embedding(Pooling::Mean),
    Output::Embedding(Pooling::Cls),
    Output::Relevance(Head::Pooler),
];

fn shape(hidden: usize, heads: usize) -> Shape {
    Shape {
        layers: 2,
        hidden,
        heads,
        intermediate: 4 * hidden,
        vocab: 50,
        max_positions: 512,
        type_vocab: 2,
        eps: 1e-12,
        positions: Positions::Sequential,
    }
}

fn random_ids(rng: &mut Rng, count: usize, vocab: usize) -> Vec<i32> {
    (0..count).map(|_| rng.below(vocab as u64) as i32).collect()
}

/// granite-embedding-small's, ModernBERT-base's and ModernBERT-large's width, heads and gate.
const MODERN: [(usize, usize, usize); 3] = [(384, 12, 1536), (768, 12, 1152), (1024, 16, 2624)];

/// ModernBERT's layers: of four, the first and last attend over the whole sequence and the
/// others inside `window`.
fn modern_shape((hidden, heads, intermediate): (usize, usize, usize), window: usize) -> Shape {
    Shape {
        layers: 4,
        hidden,
        heads,
        intermediate,
        vocab: 50,
        max_positions: 1024,
        type_vocab: 1,
        eps: 1e-5,
        positions: Positions::ModernBert(Rotary {
            global_theta: 160_000.0,
            local_theta: 10_000.0,
            local_layers: 0b0110,
            window,
        }),
    }
}

/// Each of `got`'s unit vectors beside the reference's for its sequence.
fn assert_cosines(got: &[f32], expected: &[Vec<f64>], case: &str) {
    let hidden = expected[0].len();
    assert_eq!(got.len(), expected.len() * hidden);
    for (sequence, (have, want)) in got.chunks_exact(hidden).zip(expected).enumerate() {
        let cosine: f64 = have.iter().zip(want).map(|(&a, &b)| f64::from(a) * b).sum();
        assert!(
            cosine > 0.9999,
            "{case}, sequence {sequence}: cosine {cosine}"
        );
    }
}

#[test]
fn the_encoder_matches_the_double_precision_reference_on_every_gpu() {
    let lengths = [3usize, 20, 1];
    for (hidden, heads) in widths() {
        for positions in [
            Positions::Sequential,
            Positions::Roberta { pad: 1 },
            Positions::Mpnet,
        ] {
            let shape = Shape {
                positions,
                ..shape(hidden, heads)
            };
            let model = Synthetic::new(&shape, 5);
            let mut ids = random_ids(&mut Rng::new(9), lengths.iter().sum(), shape.vocab);
            // RoBERTa positions the padding id, 1, apart from the other tokens.
            ids[1] = 1;
            for pooling in [Pooling::Mean, Pooling::Cls] {
                let expected = reference::encode(&shape, &model, pooling, &ids, &lengths).unwrap();
                for device in devices() {
                    let output = Output::Embedding(pooling);
                    let mut encoder = Encoder::new(&device, shape, output, 512, &model).unwrap();
                    let got = encoder.encode(&ids, &vec![0; ids.len()], &lengths).unwrap();
                    let name = &device.name;
                    let case = format!("{name} {pooling:?} {positions:?} hidden {hidden}");
                    assert_cosines(&got, &expected, &case);
                }
            }
        }
    }
}

/// A window of 16 leaves keys out for every sequence past 9 tokens, and from its 73rd row on a
/// query sees none of the first keys its workgroup reads.
#[test]
fn the_modernbert_encoder_matches_the_double_precision_reference_on_every_gpu() {
    let lengths = [3usize, 100, 1, 20];
    for geometry in MODERN {
        let shape = modern_shape(geometry, 16);
        let model = Synthetic::new(&shape, 5);
        let ids = random_ids(&mut Rng::new(9), lengths.iter().sum(), shape.vocab);
        for pooling in [Pooling::Mean, Pooling::Cls] {
            let expected = reference::encode(&shape, &model, pooling, &ids, &lengths).unwrap();
            for device in devices() {
                let output = Output::Embedding(pooling);
                let mut encoder = Encoder::new(&device, shape, output, 512, &model).unwrap();
                let got = encoder.encode(&ids, &vec![0; ids.len()], &lengths).unwrap();
                let case = format!("{} {pooling:?} {geometry:?}", device.name);
                assert_cosines(&got, &expected, &case);
            }
        }
    }
}

/// ModernBERT's window of 128 over 700 tokens: key blocks past a sequence's first and positions
/// far into the rotation tables.
#[test]
fn modernbert_reads_sequences_far_past_its_window_on_every_gpu() {
    let lengths = [700usize, 5];
    let shape = modern_shape(MODERN[0], 128);
    let model = Synthetic::new(&shape, 7);
    let ids = random_ids(&mut Rng::new(11), lengths.iter().sum(), shape.vocab);
    let expected = reference::encode(&shape, &model, Pooling::Mean, &ids, &lengths).unwrap();
    for device in devices() {
        let output = Output::Embedding(Pooling::Mean);
        let mut encoder = Encoder::new(&device, shape, output, 1024, &model).unwrap();
        let got = encoder.encode(&ids, &vec![0; ids.len()], &lengths).unwrap();
        assert_cosines(&got, &expected, &format!("{} 700 tokens", device.name));
    }
}

/// 600 tokens take other GEMMs than a sequence of 37 alone; ModernBERT-large's gate, 2,624
/// wide, is no whole number of the row GEMM's wide K stages.
#[test]
fn a_modernbert_sequence_keeps_its_bits_whichever_gemm_its_batch_takes() {
    let lengths = [37usize, 200, 200, 163];
    for geometry in MODERN {
        let shape = modern_shape(geometry, 128);
        let model = Synthetic::new(&shape, 5);
        let ids = random_ids(&mut Rng::new(21), lengths.iter().sum(), shape.vocab);
        let types = vec![0; ids.len()];
        for device in devices() {
            let output = Output::Embedding(Pooling::Mean);
            let mut encoder = Encoder::new(&device, shape, output, 600, &model).unwrap();
            let batch = encoder.encode(&ids, &types, &lengths).unwrap();
            let alone = encoder.encode(&ids[..37], &types[..37], &[37]).unwrap();
            assert_same_bits(&alone, &batch[..shape.hidden]);
        }
    }
}

/// Each of `got`'s scores beside the reference's for its sequence.
fn assert_scores(got: &[f32], expected: &[f64], case: &str) {
    assert_eq!(got.len(), expected.len());
    for (sequence, (&have, &want)) in got.iter().zip(expected).enumerate() {
        let error = (f64::from(have) - want).abs();
        assert!(
            error <= 1e-2 * (1.0 + want.abs()),
            "{case}, sequence {sequence}: {have} vs {want}"
        );
    }
}

/// The prediction head from the first row and from the mean, through each activation.
#[test]
fn the_modernbert_cross_encoder_matches_the_double_precision_reference_on_every_gpu() {
    let lengths = [9usize, 30, 2, 72];
    for geometry in MODERN {
        let shape = modern_shape(geometry, 16);
        let model = Synthetic::new(&shape, 13);
        let ids = random_ids(&mut Rng::new(17), lengths.iter().sum(), shape.vocab);
        let types = vec![0; ids.len()];
        for pooling in [Pooling::Mean, Pooling::Cls] {
            for activation in [Activation::Gelu, Activation::Silu] {
                let head = Head::Prediction(pooling, activation);
                let expected =
                    reference::rerank(&shape, &model, head, &ids, &types, &lengths).unwrap();
                for device in devices() {
                    let output = Output::Relevance(head);
                    let mut encoder = Encoder::new(&device, shape, output, 512, &model).unwrap();
                    let got = encoder.encode(&ids, &types, &lengths).unwrap();
                    let case = format!("{} {head:?} {geometry:?}", device.name);
                    assert_scores(&got, &expected, &case);
                }
            }
        }
    }
}

#[test]
fn the_cross_encoder_matches_the_double_precision_reference_on_every_gpu() {
    // (query, passage) token counts: segment 0 then segment 1, as a reranker packs them.
    let pairs = [(4usize, 5usize), (10, 20), (1, 1)];
    let lengths: Vec<usize> = pairs
        .iter()
        .map(|(query, passage)| query + passage)
        .collect();
    let types: Vec<i32> = pairs
        .iter()
        .flat_map(|&(query, passage)| {
            std::iter::repeat_n(0, query).chain(std::iter::repeat_n(1, passage))
        })
        .collect();
    for (hidden, heads) in widths() {
        let shape = shape(hidden, heads);
        let model = Synthetic::new(&shape, 13);
        let ids = random_ids(&mut Rng::new(17), types.len(), shape.vocab);
        let head = Head::Pooler;
        let expected = reference::rerank(&shape, &model, head, &ids, &types, &lengths).unwrap();
        for device in devices() {
            let output = Output::Relevance(head);
            let mut encoder = Encoder::new(&device, shape, output, 512, &model).unwrap();
            let got = encoder.encode(&ids, &types, &lengths).unwrap();
            assert_scores(&got, &expected, &format!("{} hidden {hidden}", device.name));
        }
    }
}

fn assert_same_bits(have: &[f32], want: &[f32]) {
    assert_eq!(have.len(), want.len());
    for (index, (have, want)) in have.iter().zip(want).enumerate() {
        assert_eq!(have.to_bits(), want.to_bits(), "output coordinate {index}");
    }
}

#[test]
fn reused_activation_storage_is_batch_independent_and_handles_alternating_inputs() {
    for (hidden, heads) in widths() {
        let shape = shape(hidden, heads);
        let model = Synthetic::new(&shape, 5);
        for device in devices() {
            for output in OUTPUTS {
                const CAPACITY: usize = 129;
                let mut encoder = Encoder::new(&device, shape, output, CAPACITY, &model).unwrap();
                let lengths = [3usize, 65, 1];
                let ids = random_ids(&mut Rng::new(9), lengths.iter().sum(), shape.vocab);
                let types: Vec<i32> = (0..ids.len()).map(|index| (index % 2) as i32).collect();
                let batch = encoder.encode(&ids, &types, &lengths).unwrap();
                let width = encoder.output_width();
                let mut first = 0;
                for (sequence, &len) in lengths.iter().enumerate() {
                    let single = encoder
                        .encode(&ids[first..first + len], &types[first..first + len], &[len])
                        .unwrap();
                    assert_same_bits(&single, &batch[sequence * width..(sequence + 1) * width]);
                    first += len;
                }

                // Most outputs, with a row-tile tail, alternating with one long
                // sequence: storage is reused without recreating the encoder.
                let dense_ids = random_ids(&mut Rng::new(11), CAPACITY, shape.vocab);
                let dense_types: Vec<i32> = (0..CAPACITY).map(|index| (index % 2) as i32).collect();
                let dense_lengths = vec![1; CAPACITY];
                let dense = encoder
                    .encode(&dense_ids, &dense_types, &dense_lengths)
                    .unwrap();
                assert_eq!(dense.len(), CAPACITY * width);
                let reversed_ids: Vec<i32> = dense_ids.iter().copied().rev().collect();
                let reversed_types: Vec<i32> = dense_types.iter().copied().rev().collect();
                let reversed = encoder
                    .encode(&reversed_ids, &reversed_types, &dense_lengths)
                    .unwrap();
                for (have, want) in reversed
                    .chunks_exact(width)
                    .zip(dense.chunks_exact(width).rev())
                {
                    assert_same_bits(have, want);
                }
                encoder
                    .encode(&dense_ids, &dense_types, &[CAPACITY])
                    .unwrap();
                encoder.encode(&[1], &[0], &[1]).unwrap();
                assert_same_bits(
                    &encoder
                        .encode(&dense_ids, &dense_types, &dense_lengths)
                        .unwrap(),
                    &dense,
                );

                // A rejected call must leave reused storage usable for the next batch.
                assert!(matches!(
                    encoder.encode(&[1], &[shape.type_vocab as i32], &[1]),
                    Err(EncoderError::Input(_))
                ));
                assert!(matches!(
                    encoder.encode(&[], &[], &[0]),
                    Err(EncoderError::Input(_))
                ));
                assert_same_bits(&encoder.encode(&ids, &types, &lengths).unwrap(), &batch);
            }
        }
    }
}

#[test]
fn a_sequence_keeps_its_bits_whichever_gemm_its_batch_takes() {
    // 600 tokens put MiniLM's QKV and first FFN projections on gemm_matrix and
    // the other two on the row GEMM; the sequence alone takes the row GEMM.
    let shape = shape(384, 12);
    let model = Synthetic::new(&shape, 5);
    let lengths = [37usize, 200, 200, 163];
    let ids = random_ids(&mut Rng::new(21), lengths.iter().sum(), shape.vocab);
    let types = vec![0; ids.len()];
    for device in devices() {
        let mut encoder = Encoder::new(
            &device,
            shape,
            Output::Embedding(Pooling::Mean),
            600,
            &model,
        )
        .unwrap();
        let batch = encoder.encode(&ids, &types, &lengths).unwrap();
        let alone = encoder.encode(&ids[..37], &types[..37], &[37]).unwrap();
        assert_same_bits(&alone, &batch[..shape.hidden]);
    }
}

#[test]
fn segment_types_outside_the_model_are_rejected_before_any_dispatch() {
    let shape = shape(384, 12);
    let model = Synthetic::new(&shape, 5);
    let ids = [1, 2, 3];
    for device in devices() {
        let mut encoder =
            Encoder::new(&device, shape, Output::Relevance(Head::Pooler), 64, &model).unwrap();
        for types in [&[0, 1, 2][..], &[0, -1, 0], &[0, 0]] {
            assert!(
                matches!(
                    encoder.encode(&ids, types, &[3]),
                    Err(EncoderError::Input(_))
                ),
                "{} types {types:?}",
                device.name
            );
        }
    }
}

#[test]
fn overflowing_sequence_lengths_are_rejected_before_any_dispatch() {
    let shape = shape(384, 12);
    let model = Synthetic::new(&shape, 5);
    for device in devices() {
        let mut encoder =
            Encoder::new(&device, shape, Output::Embedding(Pooling::Mean), 64, &model).unwrap();
        assert!(matches!(
            encoder.encode(&[1], &[0], &[usize::MAX, 2]),
            Err(EncoderError::Input(_))
        ));
        // Invalid inputs must leave the encoder usable.
        assert_eq!(
            encoder.encode(&[1], &[0], &[1]).unwrap().len(),
            shape.hidden
        );
    }
}

#[test]
fn editing_adapter_information_cannot_enable_unsupported_matrix_features() {
    let shape = shape(384, 12);
    let model = Synthetic::new(&shape, 5);
    for mut device in devices() {
        let enabled = device.cooperative_matrix && device.subgroup_size == 32;
        device.cooperative_matrix = !device.cooperative_matrix;
        device.subgroup_size = 32;
        let encoder =
            Encoder::new(&device, shape, Output::Embedding(Pooling::Mean), 64, &model).unwrap();
        assert_eq!(encoder.uses_matrix_units(), enabled);
    }
}

#[test]
fn finite_intermediate_overflow_returns_an_encoder_error() {
    struct Overflow<'a> {
        base: &'a Synthetic,
        case: usize,
        width: usize,
    }
    impl Tensors for Overflow<'_> {
        fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
            let alternating = |scale: f32| {
                (0..len)
                    .map(|index| if index % 2 == 0 { scale } else { -scale })
                    .collect()
            };
            match (self.case, name) {
                (0, "embeddings.word_embeddings.weight") => Ok(alternating(1e20)),
                (1, "embeddings.word_embeddings.weight") => Ok(alternating(1e16)),
                (2, "encoder.layer.0.attention.output.dense.bias") => Ok(alternating(1e20)),
                (3 | 4, "encoder.layer.0.output.LayerNorm.weight") => Ok(vec![0.0; len]),
                (3 | 4, "encoder.layer.0.output.LayerNorm.bias") => Ok(vec![1e20; len]),
                (4, "pooler.dense.weight") => {
                    let mut values = vec![0.0; len];
                    for row in values.chunks_mut(self.width) {
                        row[0] = 1e20;
                        row[1] = -1e20;
                    }
                    Ok(values)
                }
                (4, "pooler.dense.bias" | "classifier.bias") => Ok(vec![0.0; len]),
                (4, "classifier.weight") => Ok(vec![1.0; len]),
                _ => self.base.get(name, len),
            }
        }
    }
    for (hidden, heads) in widths() {
        let mut shape = shape(hidden, heads);
        shape.layers = 1;
        let base = Synthetic::new(&shape, 29);
        for device in devices() {
            for (case, label) in [
                "embedding variance",
                "finite variance plus finite epsilon",
                "attention LayerNorm variance",
                "pooling squared norm",
                "rerank cancelling preactivation",
            ]
            .iter()
            .enumerate()
            {
                let tensors = Overflow {
                    base: &base,
                    case,
                    width: hidden,
                };
                let shape = Shape {
                    eps: if case == 1 { f32::MAX } else { shape.eps },
                    ..shape
                };
                let output = if case == 4 {
                    Output::Relevance(Head::Pooler)
                } else {
                    Output::Embedding(Pooling::Mean)
                };
                let mut encoder = Encoder::new(&device, shape, output, 2, &tensors).unwrap();
                let result = encoder.encode(&[1, 2], &[0, 0], &[2]);
                assert!(
                    matches!(&result, Err(EncoderError::Weights(message)) if message.contains("non-finite")),
                    "{} hidden {hidden}, {label}: {result:?}",
                    device.name
                );
            }
        }
    }
}

#[test]
fn a_prediction_head_that_overflows_returns_an_encoder_error() {
    /// Final rows near 1e20 under a dense layer that scales each by `diagonal`.
    struct LargeHead<'a> {
        base: &'a Synthetic,
        hidden: usize,
        diagonal: f32,
    }
    impl Tensors for LargeHead<'_> {
        fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
            match name {
                "final_norm.weight" => Ok(vec![1e20; len]),
                "head.dense.weight" => {
                    let mut weight = vec![0.0; len];
                    weight
                        .iter_mut()
                        .step_by(self.hidden + 1)
                        .for_each(|w| *w = self.diagonal);
                    Ok(weight)
                }
                _ => self.base.get(name, len),
            }
        }
    }
    for geometry in MODERN {
        let shape = Shape {
            layers: 1,
            ..modern_shape(geometry, 16)
        };
        let base = Synthetic::new(&shape, 29);
        for device in devices() {
            // In FP32 the dense layer's products overflow in the first case, the variance of
            // its finite activations in the second.
            for (diagonal, overflowing) in [(1e20, "dense products"), (1.0, "variance")] {
                let tensors = LargeHead {
                    base: &base,
                    hidden: shape.hidden,
                    diagonal,
                };
                for pooling in [Pooling::Mean, Pooling::Cls] {
                    for activation in [Activation::Gelu, Activation::Silu] {
                        let head = Head::Prediction(pooling, activation);
                        let output = Output::Relevance(head);
                        let mut encoder =
                            Encoder::new(&device, shape, output, 2, &tensors).unwrap();
                        let result = encoder.encode(&[1, 2], &[0, 0], &[2]);
                        assert!(
                            matches!(&result, Err(EncoderError::Weights(message)) if message.contains("non-finite")),
                            "{} {geometry:?} {head:?}, {overflowing}: {result:?}",
                            device.name
                        );
                    }
                }
            }
        }
    }
}
