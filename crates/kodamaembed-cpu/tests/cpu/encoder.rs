use crate::support::{Rng, kernels};
use kodamaembed_cpu::{CpuPrecision, Encoder, EncoderError, Kernel};
use kodamaembed_model::{
    Activation, Head, Output, Pooling, Positions, Rotary, STAGES, Shape,
    reference::{self, Synthetic},
};
use std::num::NonZeroUsize;

/// MiniLM's and e5-large's widths and head counts.
const WIDTHS: [(usize, usize); 2] = [(384, 12), (1024, 16)];

/// ModernBERT's layers: of four, the first and last attend over the whole sequence; a window of
/// 16 leaves keys out of the other two for every sequence past 9 tokens.
fn modern_shape(hidden: usize, heads: usize, intermediate: usize) -> Shape {
    Shape {
        layers: 4,
        hidden,
        heads,
        intermediate,
        vocab: 50,
        max_positions: 512,
        type_vocab: 1,
        eps: 1e-5,
        positions: Positions::ModernBert(Rotary {
            global_theta: 160_000.0,
            local_theta: 10_000.0,
            local_layers: 0b0110,
            window: 16,
        }),
    }
}

/// Each of `got`'s vectors beside the reference's for its sequence.
fn assert_vectors(got: &[f32], expected: &[Vec<f64>], case: &str) {
    let hidden = expected[0].len();
    assert_eq!(got.len(), expected.len() * hidden);
    for (sequence, (have, want)) in got.chunks_exact(hidden).zip(expected).enumerate() {
        let error = have
            .iter()
            .zip(want)
            .map(|(&a, &b)| (f64::from(a) - b).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            error <= 1e-6,
            "{case}, sequence {sequence}: L2 error {error:e}"
        );
    }
}

/// Each sequence's vector alone, bit for bit, beside its vector in the batch.
fn assert_batch_independent(encoder: &mut Encoder, ids: &[i32], lengths: &[usize], case: &str) {
    let together = encoder.encode(ids, &vec![0; ids.len()], lengths).unwrap();
    let width = together.len() / lengths.len();
    let mut start = 0;
    for (sequence, &len) in lengths.iter().enumerate() {
        let alone = encoder
            .encode(&ids[start..start + len], &vec![0; len], &[len])
            .unwrap();
        assert!(
            alone
                .iter()
                .zip(&together[sequence * width..(sequence + 1) * width])
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{case}: sequence {sequence} differs"
        );
        start += len;
    }
}

const POOLINGS: [Pooling; 2] = [Pooling::Mean, Pooling::Cls];

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

#[test]
fn the_encoder_matches_the_double_precision_reference() {
    let lengths = [3usize, 20, 1, 70];
    for (hidden, heads) in WIDTHS {
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
            for pooling in POOLINGS {
                let expected = reference::encode(&shape, &model, pooling, &ids, &lengths).unwrap();
                for kernel in kernels() {
                    let output = Output::Embedding(pooling);
                    let mut encoder = Encoder::new(kernel, shape, output, 512, &model).unwrap();
                    let got = encoder.encode(&ids, &vec![0; ids.len()], &lengths).unwrap();
                    let case = format!("{kernel:?} {pooling:?} {positions:?} hidden {hidden}");
                    assert_vectors(&got, &expected, &case);
                }
            }
        }
    }
}

/// granite-embedding-small's and ModernBERT-base's widths, heads and gates.
#[test]
fn the_modernbert_encoder_matches_the_double_precision_reference() {
    let lengths = [3usize, 20, 1, 70];
    for (hidden, heads, intermediate) in [(384, 12, 1536), (768, 12, 1152)] {
        let shape = modern_shape(hidden, heads, intermediate);
        let model = Synthetic::new(&shape, 5);
        let ids = random_ids(&mut Rng::new(9), lengths.iter().sum(), shape.vocab);
        for pooling in POOLINGS {
            let expected = reference::encode(&shape, &model, pooling, &ids, &lengths).unwrap();
            for kernel in kernels() {
                let output = Output::Embedding(pooling);
                let mut encoder = Encoder::new(kernel, shape, output, 512, &model).unwrap();
                let got = encoder.encode(&ids, &vec![0; ids.len()], &lengths).unwrap();
                let case = format!("{kernel:?} {pooling:?} hidden {hidden}");
                assert_vectors(&got, &expected, &case);
            }
        }
    }
}

#[test]
fn the_cross_encoder_matches_the_double_precision_reference() {
    // (query, passage) token counts: segment 0 then segment 1, as a reranker packs them.
    let pairs = [(4usize, 5usize), (10, 20), (1, 1), (12, 60)];
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
    for (hidden, heads) in WIDTHS {
        let shape = shape(hidden, heads);
        let model = Synthetic::new(&shape, 13);
        let ids = random_ids(&mut Rng::new(17), types.len(), shape.vocab);
        let head = Head::Pooler;
        let expected = reference::rerank(&shape, &model, head, &ids, &types, &lengths).unwrap();
        for kernel in kernels() {
            let output = Output::Relevance(head);
            let mut encoder = Encoder::new(kernel, shape, output, 512, &model).unwrap();
            let got = encoder.encode(&ids, &types, &lengths).unwrap();
            assert_scores(&got, &expected, &format!("{kernel:?} hidden {hidden}"));
        }
    }
}

/// Each of `got`'s scores beside the reference's for its sequence.
fn assert_scores(got: &[f32], expected: &[f64], case: &str) {
    assert_eq!(got.len(), expected.len());
    for (sequence, (&have, &want)) in got.iter().zip(expected).enumerate() {
        let error = (f64::from(have) - want).abs();
        assert!(
            error <= 3e-6 * (1.0 + want.abs()),
            "{case}, sequence {sequence}: {have} vs {want}"
        );
    }
}

/// The prediction head from the first row and from the mean, through each activation.
#[test]
fn the_modernbert_cross_encoder_matches_the_double_precision_reference() {
    let lengths = [9usize, 30, 2, 72];
    let shape = modern_shape(384, 12, 1536);
    let model = Synthetic::new(&shape, 13);
    let ids = random_ids(&mut Rng::new(17), lengths.iter().sum(), shape.vocab);
    let types = vec![0; ids.len()];
    for pooling in POOLINGS {
        for activation in [Activation::Gelu, Activation::Silu] {
            let head = Head::Prediction(pooling, activation);
            let expected = reference::rerank(&shape, &model, head, &ids, &types, &lengths).unwrap();
            for kernel in kernels() {
                let output = Output::Relevance(head);
                let mut encoder = Encoder::new(kernel, shape, output, 512, &model).unwrap();
                let got = encoder.encode(&ids, &types, &lengths).unwrap();
                assert_scores(&got, &expected, &format!("{kernel:?} {head:?}"));
            }
        }
    }
}

/// Seven query blocks, the later ones reading keys that start far into the sequence.
#[test]
fn modernbert_reads_sequences_far_past_its_window() {
    let lengths = [300usize, 5, 130];
    let shape = modern_shape(32, 4, 48);
    let model = Synthetic::new(&shape, 7);
    let ids = random_ids(&mut Rng::new(11), lengths.iter().sum(), shape.vocab);
    for pooling in POOLINGS {
        let expected = reference::encode(&shape, &model, pooling, &ids, &lengths).unwrap();
        for kernel in kernels() {
            let output = Output::Embedding(pooling);
            let mut encoder = Encoder::new(kernel, shape, output, 512, &model).unwrap();
            let got = encoder.encode(&ids, &vec![0; ids.len()], &lengths).unwrap();
            assert_vectors(&got, &expected, &format!("{kernel:?} {pooling:?}"));
            assert_batch_independent(&mut encoder, &ids, &lengths, &format!("{kernel:?}"));
        }
    }
}

/// A sequence's vector is the same bits alone or among others, in BERT's layers and ModernBERT's.
#[test]
fn vectors_do_not_depend_on_the_batch() {
    let lengths = [3usize, 49, 1, 20];
    for shape in [shape(384, 12), modern_shape(384, 12, 1536)] {
        let model = Synthetic::new(&shape, 5);
        let ids = random_ids(&mut Rng::new(21), lengths.iter().sum(), shape.vocab);
        for kernel in kernels() {
            for pooling in POOLINGS {
                let output = Output::Embedding(pooling);
                let mut encoder = Encoder::new(kernel, shape, output, 512, &model).unwrap();
                let case = format!("{kernel:?} {pooling:?} {:?}", shape.positions);
                assert_batch_independent(&mut encoder, &ids, &lengths, &case);
            }
        }
    }
}

#[test]
fn modernbert_worker_counts_do_not_change_the_bits() {
    let lengths = [30usize, 2, 17];
    let shape = modern_shape(384, 12, 1536);
    let model = Synthetic::new(&shape, 7);
    let ids = random_ids(&mut Rng::new(3), lengths.iter().sum(), shape.vocab);
    let output = Output::Embedding(Pooling::Mean);
    let kernel = Kernel::detect();
    let expected = Encoder::new(kernel, shape, output, 64, &model)
        .unwrap()
        .encode(&ids, &vec![0; ids.len()], &lengths)
        .unwrap();
    for workers in [1, 3] {
        let workers = NonZeroUsize::new(workers).unwrap();
        let mut bounded = Encoder::new_with_precision_and_workers(
            kernel,
            shape,
            output,
            64,
            &model,
            CpuPrecision::Fp32,
            workers,
        )
        .unwrap();
        assert_eq!(bounded.threads(), workers.get());
        let got = bounded.encode(&ids, &vec![0; ids.len()], &lengths).unwrap();
        assert!(
            got.iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{workers} workers"
        );
    }
}

#[test]
fn a_centered_modernbert_is_unsupported() {
    let shape = modern_shape(8, 2, 6);
    let model = Synthetic::new(&shape, 1);
    assert!(matches!(
        Encoder::new_with_precision(
            Kernel::detect(),
            shape,
            Output::Embedding(Pooling::Mean),
            8,
            &model,
            CpuPrecision::E5CenteredInt8,
        ),
        Err(EncoderError::Unsupported(_))
    ));
}

/// ModernBERT has one segment, and its positions end at the model's longest sequence.
#[test]
fn modernbert_inputs_outside_the_model_are_rejected() {
    let shape = Shape {
        max_positions: 40,
        ..modern_shape(8, 2, 6)
    };
    let model = Synthetic::new(&shape, 5);
    let output = Output::Embedding(Pooling::Mean);
    let mut encoder = Encoder::new(Kernel::Portable, shape, output, 64, &model).unwrap();
    let mut rejected = |ids: &[i32], types: &[i32], lengths: &[usize]| {
        matches!(
            encoder.encode(ids, types, lengths),
            Err(EncoderError::Input(_))
        )
    };
    assert!(rejected(&[1, 2, 3], &[0, 1, 0], &[3]), "a second segment");
    assert!(
        rejected(&[1; 41], &[0; 41], &[41]),
        "a sequence past the model's positions"
    );
    assert!(
        rejected(&[1; 65], &[0; 65], &[40, 25]),
        "a batch past max_tokens"
    );
}

#[test]
fn modernbert_weights_that_are_not_finite_or_overflow_the_activations_are_errors() {
    struct Faulty(Synthetic, f32);
    impl kodamaembed_model::Tensors for Faulty {
        fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
            if name == "layers.1.mlp.Wi.weight" {
                return Ok(vec![self.1; len]);
            }
            self.0.get(name, len)
        }
    }
    let shape = modern_shape(8, 2, 6);
    let output = Output::Embedding(Pooling::Mean);
    let load = |value: f32| {
        Encoder::new(
            Kernel::Portable,
            shape,
            output,
            8,
            &Faulty(Synthetic::new(&shape, 1), value),
        )
    };
    assert!(matches!(load(f32::INFINITY), Err(EncoderError::Weights(_))));
    let mut overflowing = load(f32::MAX).unwrap();
    assert!(matches!(
        overflowing.encode(&[1, 2], &[0, 0], &[2]),
        Err(EncoderError::Weights(_))
    ));
}

#[test]
fn profiling_times_every_stage_of_the_modernbert_batches_after_it() {
    let shape = modern_shape(8, 2, 6);
    let model = Synthetic::new(&shape, 5);
    let output = Output::Embedding(Pooling::Cls);
    let mut encoder = Encoder::new(Kernel::Portable, shape, output, 8, &model).unwrap();
    encoder.encode(&[1, 2], &[0, 0], &[2]).unwrap();
    assert!(encoder.profile().is_none());
    encoder.enable_profiling();
    encoder.encode(&[1, 2, 3], &[0; 3], &[1, 2]).unwrap();
    let profile = encoder.profile().unwrap();
    assert_eq!((profile.batches, profile.tokens), (1, 3));
    assert_eq!(profile.milliseconds.len(), STAGES.len());
    assert!(profile.milliseconds.iter().all(|ms| *ms >= 0.0));
}

#[test]
fn inputs_outside_the_model_or_the_batch_are_rejected() {
    let shape = shape(384, 12);
    let model = Synthetic::new(&shape, 5);
    let mut encoder = Encoder::new(
        kernels()[0],
        shape,
        Output::Relevance(Head::Pooler),
        64,
        &model,
    )
    .unwrap();
    let rejected = |encoder: &mut Encoder, ids: &[i32], types: &[i32], lengths: &[usize]| {
        matches!(
            encoder.encode(ids, types, lengths),
            Err(EncoderError::Input(_))
        )
    };
    for types in [&[0, 1, 2][..], &[0, -1, 0], &[0, 0]] {
        assert!(
            rejected(&mut encoder, &[1, 2, 3], types, &[3]),
            "types {types:?}"
        );
    }
    assert!(
        rejected(&mut encoder, &[1, 50, 3], &[0; 3], &[3]),
        "an id past the vocabulary"
    );
    assert!(
        rejected(&mut encoder, &[1, 2, 3], &[0; 3], &[2, 0, 1]),
        "an empty sequence"
    );
    assert!(
        rejected(&mut encoder, &[1, 2, 3], &[0; 3], &[2]),
        "lengths short of the ids"
    );
    assert!(
        rejected(&mut encoder, &[1; 65], &[0; 65], &[65]),
        "a batch past max_tokens"
    );
    assert!(
        rejected(&mut encoder, &[], &[], &[usize::MAX, 1]),
        "overflowing sequence lengths"
    );
}

#[test]
fn invalid_shapes_and_workspaces_are_rejected_before_reading_weights() {
    struct Unread;
    impl kodamaembed_model::Tensors for Unread {
        fn get(&self, _: &str, _: usize) -> Result<Vec<f32>, String> {
            panic!("invalid dimensions must fail before loading tensors")
        }
    }
    let valid = shape(8, 2);
    for invalid in [
        Shape { hidden: 0, ..valid },
        Shape { heads: 0, ..valid },
        Shape { heads: 3, ..valid },
        Shape {
            intermediate: 0,
            ..valid
        },
        Shape { vocab: 0, ..valid },
        Shape {
            max_positions: 0,
            ..valid
        },
        Shape {
            type_vocab: 0,
            ..valid
        },
        Shape {
            eps: f32::NAN,
            ..valid
        },
        Shape { eps: -1.0, ..valid },
        Shape {
            hidden: usize::MAX - 1,
            ..valid
        },
        Shape {
            layers: usize::MAX,
            ..valid
        },
    ] {
        assert!(matches!(
            Encoder::new(
                kodamaembed_cpu::Kernel::Portable,
                invalid,
                Output::Embedding(Pooling::Mean),
                8,
                &Unread
            ),
            Err(EncoderError::Unsupported(_))
        ));
    }
    // 1 << 50 tokens are addressable activations no memory holds (32 PiB of
    // hidden states): an error, not an aborted allocation.
    for tokens in [0, 1 << 50, usize::MAX] {
        assert!(matches!(
            Encoder::new(
                kodamaembed_cpu::Kernel::Portable,
                valid,
                Output::Embedding(Pooling::Mean),
                tokens,
                &Unread
            ),
            Err(EncoderError::Unsupported(_))
        ));
    }
}

#[test]
fn an_unchecked_tensor_provider_is_validated_at_the_backend_boundary() {
    struct Faulty {
        model: Synthetic,
        name: &'static str,
        nonfinite: bool,
    }
    impl kodamaembed_model::Tensors for Faulty {
        fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
            if name == self.name {
                return Ok(if self.nonfinite {
                    vec![f32::INFINITY; len]
                } else {
                    vec![]
                });
            }
            self.model.get(name, len)
        }
    }
    let valid = shape(8, 2);
    for name in [
        "embeddings.LayerNorm.weight",
        "encoder.layer.0.attention.self.query.bias",
        "classifier.bias",
    ] {
        for nonfinite in [false, true] {
            let tensors = Faulty {
                model: Synthetic::new(&valid, 1),
                name,
                nonfinite,
            };
            assert!(matches!(
                Encoder::new(
                    kodamaembed_cpu::Kernel::Portable,
                    valid,
                    Output::Relevance(Head::Pooler),
                    8,
                    &tensors
                ),
                Err(EncoderError::Weights(_))
            ));
        }
    }
}

#[test]
fn finite_checkpoint_values_that_overflow_activations_return_an_error() {
    struct Extreme;
    impl kodamaembed_model::Tensors for Extreme {
        fn get(&self, _: &str, len: usize) -> Result<Vec<f32>, String> {
            Ok(vec![f32::MAX; len])
        }
    }
    let valid = shape(8, 2);
    for output in OUTPUTS {
        let mut encoder = Encoder::new(
            kodamaembed_cpu::Kernel::Portable,
            valid,
            output,
            8,
            &Extreme,
        )
        .unwrap();
        assert!(matches!(
            encoder.encode(&[1], &[0], &[1]),
            Err(EncoderError::Weights(_))
        ));
    }
}

#[test]
fn overflowing_layernorm_statistics_do_not_become_finite_outputs() {
    struct LargeEmbeddings(Synthetic, f32);
    impl kodamaembed_model::Tensors for LargeEmbeddings {
        fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
            if name == "embeddings.word_embeddings.weight" {
                return Ok((0..len)
                    .map(|i| if i % 2 == 0 { self.1 } else { -self.1 })
                    .collect());
            }
            self.0.get(name, len)
        }
    }
    // The second case has finite variance, but variance + epsilon overflows.
    for (magnitude, eps) in [(1e20, 1e-12), (1e18, f32::MAX)] {
        let shape = Shape { eps, ..shape(8, 2) };
        let model = LargeEmbeddings(Synthetic::new(&shape, 41), magnitude);
        let reference = reference::encode(&shape, &model, Pooling::Mean, &[1], &[1]).unwrap();
        assert!(reference.iter().flatten().all(|value| value.is_finite()));
        for kernel in kernels() {
            for output in OUTPUTS {
                let mut encoder = Encoder::new(kernel, shape, output, 8, &model).unwrap();
                assert!(
                    matches!(
                        encoder.encode(&[1], &[0], &[1]),
                        Err(EncoderError::Weights(_))
                    ),
                    "{kernel:?} {output:?} magnitude={magnitude}: non-finite statistics were hidden"
                );
            }
        }
    }
}

#[test]
fn overflowing_pooler_sums_do_not_become_saturated_finite_scores() {
    struct CancellingPooler(Synthetic);
    impl kodamaembed_model::Tensors for CancellingPooler {
        fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
            match name {
                "encoder.layer.1.output.LayerNorm.weight" => Ok(vec![0.0; len]),
                "encoder.layer.1.output.LayerNorm.bias" => Ok(vec![1e20; len]),
                "pooler.dense.weight" => Ok((0..len)
                    .map(|i| if i % 8 < 4 { 1e18 } else { -1e18 })
                    .collect()),
                "pooler.dense.bias" | "classifier.bias" => Ok(vec![0.0; len]),
                "classifier.weight" => Ok(vec![1.0; len]),
                _ => self.0.get(name, len),
            }
        }
    }
    let shape = shape(8, 2);
    let model = CancellingPooler(Synthetic::new(&shape, 43));
    // Every product fits FP32. Four positive terms overflow the partial sum,
    // then four negative terms would cancel exactly in real arithmetic.
    let expected = reference::rerank(&shape, &model, Head::Pooler, &[1], &[0], &[1]).unwrap();
    assert_eq!(expected, vec![0.0]);
    for kernel in kernels() {
        let mut encoder =
            Encoder::new(kernel, shape, Output::Relevance(Head::Pooler), 8, &model).unwrap();
        assert!(
            matches!(
                encoder.encode(&[1], &[0], &[1]),
                Err(EncoderError::Weights(_))
            ),
            "{kernel:?}: tanh hid an overflowing pooler sum"
        );
    }
}

#[test]
fn an_overflowing_prediction_head_does_not_become_a_finite_score() {
    /// Final rows near 1e20 under a dense layer that scales each by `diagonal`.
    struct LargeHead {
        model: Synthetic,
        hidden: usize,
        diagonal: f32,
    }
    impl kodamaembed_model::Tensors for LargeHead {
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
                _ => self.model.get(name, len),
            }
        }
    }
    let shape = modern_shape(8, 2, 6);
    // Every weight is finite and so is the reference's score. In FP32 the dense layer's products
    // overflow in the first case, the variance of its finite activations in the second.
    for (diagonal, overflowing) in [(1e20, "dense products"), (1.0, "variance")] {
        let model = LargeHead {
            model: Synthetic::new(&shape, 43),
            hidden: shape.hidden,
            diagonal,
        };
        for pooling in POOLINGS {
            for activation in [Activation::Gelu, Activation::Silu] {
                let head = Head::Prediction(pooling, activation);
                let expected =
                    reference::rerank(&shape, &model, head, &[1, 2], &[0, 0], &[2]).unwrap();
                assert!(expected[0].is_finite(), "{head:?} {overflowing}");
                for kernel in kernels() {
                    let output = Output::Relevance(head);
                    let mut encoder = Encoder::new(kernel, shape, output, 8, &model).unwrap();
                    assert!(
                        matches!(
                            encoder.encode(&[1, 2], &[0, 0], &[2]),
                            Err(EncoderError::Weights(_))
                        ),
                        "{kernel:?} {head:?}: a finite score hid an overflowing {overflowing}"
                    );
                }
            }
        }
    }
}
