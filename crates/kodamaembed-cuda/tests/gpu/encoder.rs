use crate::support::{GEOMETRIES, Rng};
use kodamaembed_cuda::{Context, Encoder, EncoderError, kernels::CORRECTED_BELOW};
use kodamaembed_model::{
    Activation, Head, Output, Pooling, Positions, Rotary, Shape,
    reference::{self, Synthetic},
};

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
fn the_encoder_matches_the_double_precision_reference() {
    let lengths = [3usize, 20, 1];
    let context = Context::new(0).unwrap();
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
                let output = Output::Embedding(pooling);
                let mut encoder = Encoder::new(&context, shape, output, 512, &model).unwrap();
                let got = encoder.encode(&ids, &vec![0; ids.len()], &lengths).unwrap();
                let case = format!("{pooling:?} {positions:?} hidden {hidden}");
                assert_cosines(&got, &expected, &case);
            }
        }
    }
}

/// A window of 16 leaves keys out for every sequence past 9 tokens, and from its 73rd row on a
/// query sees none of the first 64 keys its block reads.
#[test]
fn the_modernbert_encoder_matches_the_double_precision_reference() {
    let lengths = [3usize, 100, 1, 20];
    let context = Context::new(0).unwrap();
    for geometry in MODERN {
        let shape = modern_shape(geometry, 16);
        let model = Synthetic::new(&shape, 5);
        let ids = random_ids(&mut Rng::new(9), lengths.iter().sum(), shape.vocab);
        for pooling in [Pooling::Mean, Pooling::Cls] {
            let expected = reference::encode(&shape, &model, pooling, &ids, &lengths).unwrap();
            let output = Output::Embedding(pooling);
            let mut encoder = Encoder::new(&context, shape, output, 512, &model).unwrap();
            let got = encoder.encode(&ids, &vec![0; ids.len()], &lengths).unwrap();
            assert_cosines(&got, &expected, &format!("{pooling:?} {geometry:?}"));
        }
    }
}

/// ModernBERT's window of 128 over 700 tokens: key blocks past a sequence's first, positions past
/// the correction limit and the uncorrected attention over whole sequences.
#[test]
fn modernbert_reads_sequences_far_past_its_window() {
    let lengths = [700usize, 5];
    let shape = modern_shape(MODERN[0], 128);
    let model = Synthetic::new(&shape, 7);
    let ids = random_ids(&mut Rng::new(11), lengths.iter().sum(), shape.vocab);
    let expected = reference::encode(&shape, &model, Pooling::Mean, &ids, &lengths).unwrap();
    let context = Context::new(0).unwrap();
    let output = Output::Embedding(Pooling::Mean);
    let mut encoder = Encoder::new(&context, shape, output, 1024, &model).unwrap();
    let got = encoder.encode(&ids, &vec![0; ids.len()], &lengths).unwrap();
    assert_cosines(&got, &expected, "700 tokens");
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
fn the_modernbert_cross_encoder_matches_the_double_precision_reference() {
    let lengths = [9usize, 30, 2, 72];
    let context = Context::new(0).unwrap();
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
                let output = Output::Relevance(head);
                let mut encoder = Encoder::new(&context, shape, output, 512, &model).unwrap();
                let got = encoder.encode(&ids, &types, &lengths).unwrap();
                assert_scores(&got, &expected, &format!("{head:?} {geometry:?}"));
            }
        }
    }
}

/// ModernBERT-base's gate is narrower than the means that pooling keeps for a batch of
/// one-token sequences.
#[test]
fn a_modernbert_batch_of_single_tokens_fills_its_capacity() {
    const CAPACITY: usize = 129;
    let shape = modern_shape(MODERN[1], 128);
    let model = Synthetic::new(&shape, 5);
    let context = Context::new(0).unwrap();
    let output = Output::Embedding(Pooling::Mean);
    let mut encoder = Encoder::new(&context, shape, output, CAPACITY, &model).unwrap();
    let ids = random_ids(&mut Rng::new(13), CAPACITY, shape.vocab);
    let batch = encoder
        .encode(&ids, &[0; CAPACITY], &[1; CAPACITY])
        .unwrap();
    for (&id, together) in ids.iter().zip(batch.chunks_exact(shape.hidden)) {
        assert_same_bits(&encoder.encode(&[id], &[0], &[1]).unwrap(), together);
    }
}

#[test]
fn the_cross_encoder_matches_the_double_precision_reference() {
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
    let context = Context::new(0).unwrap();
    for (hidden, heads) in widths() {
        let shape = shape(hidden, heads);
        let model = Synthetic::new(&shape, 13);
        let ids = random_ids(&mut Rng::new(17), types.len(), shape.vocab);
        let head = Head::Pooler;
        let expected = reference::rerank(&shape, &model, head, &ids, &types, &lengths).unwrap();
        let output = Output::Relevance(head);
        let mut encoder = Encoder::new(&context, shape, output, 512, &model).unwrap();
        let got = encoder.encode(&ids, &types, &lengths).unwrap();
        assert_scores(&got, &expected, &format!("hidden {hidden}"));
    }
}

/// Each sequence alone, then beside the other attention kind and in a batch shaped like one
/// recorded with that kind alone.
fn assert_bits_in_any_batch(encoder: &mut Encoder, shape: &Shape, case: &str) {
    let (limit, hidden) = (CORRECTED_BELOW, shape.hidden);
    let mut rng = Rng::new(23);
    // Equal tokens, longest sequence and sequences; only the second holds a short one.
    for lengths in [
        [limit + 44, limit + 4, limit],
        [limit + 44, limit + 44, limit - 40],
    ] {
        let sequences: Vec<Vec<i32>> = lengths
            .iter()
            .map(|&len| random_ids(&mut rng, len, shape.vocab))
            .collect();
        let alone: Vec<Vec<f32>> = sequences
            .iter()
            .map(|ids| {
                encoder
                    .encode(ids, &vec![0; ids.len()], &[ids.len()])
                    .unwrap()
            })
            .collect();
        let ids = sequences.concat();
        // The second sight records the batch's graph and the third replays it.
        for sight in 1..=3 {
            let got = encoder.encode(&ids, &vec![0; ids.len()], &lengths).unwrap();
            for (sequence, want) in alone.iter().enumerate() {
                let have = &got[sequence * hidden..(sequence + 1) * hidden];
                assert!(
                    have.iter()
                        .zip(want)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{case}, {lengths:?}, sight {sight}, sequence {sequence}"
                );
            }
        }
    }
}

#[test]
fn a_sequence_keeps_its_bits_in_any_batch() {
    let context = Context::new(0).unwrap();
    for (hidden, heads) in widths() {
        for positions in [Positions::Sequential, Positions::Mpnet] {
            let shape = Shape {
                positions,
                ..shape(hidden, heads)
            };
            let model = Synthetic::new(&shape, 21);
            let output = Output::Embedding(Pooling::Mean);
            let mut encoder = Encoder::new(&context, shape, output, 1024, &model).unwrap();
            let case = format!("hidden {hidden}, {positions:?}");
            assert_bits_in_any_batch(&mut encoder, &shape, &case);
        }
    }
}

#[test]
fn a_modernbert_sequence_keeps_its_bits_in_any_batch() {
    let context = Context::new(0).unwrap();
    for geometry in MODERN {
        let shape = modern_shape(geometry, 128);
        let model = Synthetic::new(&shape, 21);
        let output = Output::Embedding(Pooling::Mean);
        let mut encoder = Encoder::new(&context, shape, output, 1024, &model).unwrap();
        assert_bits_in_any_batch(&mut encoder, &shape, &format!("{geometry:?}"));
    }
}

/// Built on one thread, run and freed on another whose context stack is empty.
#[test]
fn an_encoder_moves_to_another_thread() {
    let shape = shape(384, 12);
    let model = Synthetic::new(&shape, 5);
    let context = Context::new(0).unwrap();
    let mut encoder = Encoder::new(
        &context,
        shape,
        Output::Embedding(Pooling::Mean),
        64,
        &model,
    )
    .unwrap();
    let lengths = [3usize, 20];
    let ids = random_ids(&mut Rng::new(9), lengths.iter().sum(), shape.vocab);
    let types = vec![0; ids.len()];
    let here = encoder.encode(&ids, &types, &lengths).unwrap();
    let there = std::thread::spawn(move || encoder.encode(&ids, &types, &lengths).unwrap())
        .join()
        .unwrap();
    assert!(
        here.iter()
            .zip(&there)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the other thread's vectors differ"
    );
}

#[test]
fn segment_types_outside_the_model_are_rejected_before_any_launch() {
    let shape = shape(384, 12);
    let model = Synthetic::new(&shape, 5);
    let context = Context::new(0).unwrap();
    let mut encoder =
        Encoder::new(&context, shape, Output::Relevance(Head::Pooler), 64, &model).unwrap();
    let ids = [1, 2, 3];
    for types in [&[0, 1, 2][..], &[0, -1, 0], &[0, 0]] {
        assert!(
            matches!(
                encoder.encode(&ids, types, &[3]),
                Err(EncoderError::Input(_))
            ),
            "types {types:?}"
        );
    }
}

/// A sequence past attention's 65,535 is an input error, not a failed launch.
#[test]
fn a_batch_holds_at_most_65535_sequences() {
    let shape = shape(384, 12);
    let model = Synthetic::new(&shape, 5);
    let context = Context::new(0).unwrap();
    let mut encoder = Encoder::new(
        &context,
        shape,
        Output::Embedding(Pooling::Mean),
        65_536,
        &model,
    )
    .unwrap();
    for (sequences, accepted) in [(65_535usize, true), (65_536, false)] {
        let ids = vec![1; sequences];
        let result = encoder.encode(&ids, &vec![0; sequences], &vec![1; sequences]);
        match result {
            Ok(vectors) => {
                assert!(accepted, "{sequences} sequences");
                assert_eq!(vectors.len(), sequences * 384);
            }
            Err(error) => assert!(
                !accepted && matches!(error, EncoderError::Input(_)),
                "{sequences} sequences: {error}"
            ),
        }
    }
}

#[test]
fn invalid_capacities_and_overflowing_lengths_are_input_errors() {
    let context = Context::new(0).unwrap();
    let shape = shape(384, 12);
    let model = Synthetic::new(&shape, 5);
    for capacity in [0, kodamaembed_cuda::MAX_TOKENS + 1, usize::MAX] {
        assert!(matches!(
            Encoder::new(
                &context,
                shape,
                Output::Embedding(Pooling::Mean),
                capacity,
                &model
            ),
            Err(EncoderError::Input(_))
        ));
    }
    let mut encoder = Encoder::new(
        &context,
        shape,
        Output::Embedding(Pooling::Mean),
        64,
        &model,
    )
    .unwrap();
    assert!(matches!(
        encoder.encode(&[1], &[0], &[usize::MAX, 2]),
        Err(EncoderError::Input(_))
    ));
}

fn assert_same_bits(have: &[f32], want: &[f32]) {
    assert_eq!(have.len(), want.len());
    for (index, (have, want)) in have.iter().zip(want).enumerate() {
        assert_eq!(have.to_bits(), want.to_bits(), "output coordinate {index}");
    }
}

#[test]
fn reused_activation_storage_is_batch_independent_and_handles_alternating_inputs() {
    let context = Context::new(0).unwrap();
    for (hidden, heads) in widths() {
        let shape = shape(hidden, heads);
        let model = Synthetic::new(&shape, 5);
        for output in OUTPUTS {
            const CAPACITY: usize = 129;
            let mut encoder = Encoder::new(&context, shape, output, CAPACITY, &model).unwrap();
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

/// Shapes equal in tokens alone never share a recording, and a displaced shape is recorded again.
#[test]
fn recorded_shapes_replay_each_batch_as_launched_directly() {
    let shape = shape(384, 12);
    let model = Synthetic::new(&shape, 5);
    let context = Context::new(0).unwrap();
    let encode = |encoder: &mut Encoder, seed: u64, lengths: &[usize]| {
        let ids = random_ids(&mut Rng::new(seed), lengths.iter().sum(), shape.vocab);
        encoder.encode(&ids, &vec![0; ids.len()], lengths).unwrap()
    };
    let mut recorded = Encoder::new(
        &context,
        shape,
        Output::Embedding(Pooling::Mean),
        128,
        &model,
    )
    .unwrap();
    // Profiling launches every batch directly, as its stage events need.
    let mut direct = Encoder::new(
        &context,
        shape,
        Output::Embedding(Pooling::Mean),
        128,
        &model,
    )
    .unwrap();
    direct.enable_profiling().unwrap();
    // 110 tokens each; attention spans one 64-row block, then two.
    for lengths in [&[55usize, 55][..], &[55, 30, 25], &[10, 100]] {
        for seed in 0..3 {
            assert_same_bits(
                &encode(&mut recorded, seed, lengths),
                &encode(&mut direct, seed, lengths),
            );
        }
    }
    // An encoder keeps 32 recordings: lengths 1 to 40, each seen twice, displace the three above
    // and 1 to 8, so 40 and 9 replay and 1 is recorded again before it does.
    let twice = (1..=40).flat_map(|len| [len, len]);
    for len in twice.chain([40, 9, 1, 1, 1]) {
        assert_same_bits(
            &encode(&mut recorded, len as u64, &[len]),
            &encode(&mut direct, len as u64, &[len]),
        );
    }
}

#[test]
fn nonfinite_encoder_outputs_are_reported_as_errors() {
    struct ExcessiveNorm(Synthetic);
    impl kodamaembed_model::Tensors for ExcessiveNorm {
        fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
            if name == "embeddings.LayerNorm.weight" {
                return Ok(vec![f32::MAX; len]);
            }
            self.0.get(name, len)
        }
    }
    let context = Context::new(0).unwrap();
    let shape = shape(384, 12);
    let model = ExcessiveNorm(Synthetic::new(&shape, 7));
    let mut encoder = Encoder::new(
        &context,
        shape,
        Output::Embedding(Pooling::Mean),
        16,
        &model,
    )
    .unwrap();
    for _ in 0..2 {
        assert!(matches!(
            encoder.encode(&[1, 2, 3], &[0, 0, 0], &[3]),
            Err(EncoderError::Weights(_))
        ));
    }
}

#[test]
fn finite_embedding_weights_with_overflowing_statistics_are_rejected() {
    struct LargeEmbeddings {
        model: Synthetic,
        magnitude: f32,
    }
    impl kodamaembed_model::Tensors for LargeEmbeddings {
        fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
            match name {
                "embeddings.word_embeddings.weight" => Ok((0..len)
                    .map(|column| {
                        if column % 2 == 0 {
                            self.magnitude
                        } else {
                            -self.magnitude
                        }
                    })
                    .collect()),
                "embeddings.position_embeddings.weight"
                | "embeddings.token_type_embeddings.weight" => Ok(vec![0.0; len]),
                _ => self.model.get(name, len),
            }
        }
    }
    let context = Context::new(0).unwrap();
    for (hidden, heads) in widths() {
        for (magnitude, eps) in [(1e20, 1e-12), (1e17, f32::MAX)] {
            let mut shape = shape(hidden, heads);
            shape.layers = 1;
            shape.eps = eps;
            let model = LargeEmbeddings {
                model: Synthetic::new(&shape, 7),
                magnitude,
            };
            let mut encoder =
                Encoder::new(&context, shape, Output::Embedding(Pooling::Mean), 8, &model).unwrap();
            for _ in 0..2 {
                assert!(
                    matches!(
                        encoder.encode(&[1, 2], &[0, 0], &[2]),
                        Err(EncoderError::Weights(_))
                    ),
                    "hidden {hidden}, magnitude {magnitude}, eps {eps}"
                );
            }
        }
    }
}

#[test]
fn finite_pooler_weights_with_overflowing_activations_are_rejected() {
    struct LargePooler(Synthetic);
    impl kodamaembed_model::Tensors for LargePooler {
        fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
            match name {
                // Large finite final states without overflowing earlier layers.
                "encoder.layer.0.output.LayerNorm.weight" => Ok(vec![0.0; len]),
                "encoder.layer.0.output.LayerNorm.bias" => {
                    let mut bias = vec![0.0; len];
                    bias[..2].fill(1e20);
                    Ok(bias)
                }
                "pooler.dense.weight" => {
                    let mut weight = vec![0.0; len];
                    weight[0] = 1e20;
                    weight[1] = -1e20;
                    Ok(weight)
                }
                "pooler.dense.bias" => Ok(vec![0.0; len]),
                "classifier.weight" => {
                    let mut weight = vec![0.0; len];
                    weight[0] = 1.0;
                    Ok(weight)
                }
                "classifier.bias" => Ok(vec![0.25; len]),
                _ => self.0.get(name, len),
            }
        }
    }
    let context = Context::new(0).unwrap();
    for (hidden, heads) in widths() {
        let mut shape = shape(hidden, heads);
        shape.layers = 1;
        let model = LargePooler(Synthetic::new(&shape, 7));
        let mut encoder =
            Encoder::new(&context, shape, Output::Relevance(Head::Pooler), 8, &model).unwrap();
        for _ in 0..2 {
            assert!(
                matches!(
                    encoder.encode(&[1, 2], &[0, 0], &[2]),
                    Err(EncoderError::Weights(_))
                ),
                "hidden {hidden}"
            );
        }
    }
}

#[test]
fn finite_prediction_head_weights_that_overflow_are_rejected() {
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
    let context = Context::new(0).unwrap();
    for geometry in MODERN {
        let shape = Shape {
            layers: 1,
            ..modern_shape(geometry, 16)
        };
        // In FP32 the dense layer's products overflow in the first case, the variance of its
        // finite activations in the second.
        for (diagonal, overflowing) in [(1e20, "dense products"), (1.0, "variance")] {
            let model = LargeHead {
                model: Synthetic::new(&shape, 7),
                hidden: shape.hidden,
                diagonal,
            };
            for pooling in [Pooling::Mean, Pooling::Cls] {
                for activation in [Activation::Gelu, Activation::Silu] {
                    let head = Head::Prediction(pooling, activation);
                    let output = Output::Relevance(head);
                    let mut encoder = Encoder::new(&context, shape, output, 8, &model).unwrap();
                    for _ in 0..2 {
                        assert!(
                            matches!(
                                encoder.encode(&[1, 2], &[0, 0], &[2]),
                                Err(EncoderError::Weights(_))
                            ),
                            "{geometry:?} {head:?}: a finite score hid an overflowing {overflowing}"
                        );
                    }
                }
            }
        }
    }
}
