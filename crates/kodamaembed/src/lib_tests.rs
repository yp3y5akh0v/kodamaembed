use super::*;
use crate::test_support::{
    TempDir, tiny_bert_checkpoint, tiny_bert_config, tiny_modernbert_checkpoint,
    tiny_modernbert_config, tiny_modernbert_cross_encoder,
};
use kodamaembed_model::{Activation, Head, Positions};

fn assert_same_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "value {index} differs"
        );
    }
}

#[test]
fn auto_runs_shapes_the_gpu_kernels_do_not_cover_on_the_processor() {
    let shape = Shape {
        layers: 8,
        hidden: 512,
        heads: 8,
        intermediate: 2048,
        vocab: 30_522,
        max_positions: 512,
        type_vocab: 2,
        eps: 1e-12,
        positions: Positions::Sequential,
    };
    assert_eq!(
        auto_gpu(&shape, 256, |_| unreachable!()).unwrap(),
        None::<Device>
    );
    let covered = Shape {
        hidden: 384,
        heads: 12,
        intermediate: 1536,
        ..shape
    };
    assert_eq!(
        auto_gpu(
            &covered,
            kodamaembed_cuda::MAX_TOKENS + 1,
            |_| unreachable!()
        )
        .unwrap(),
        None::<Device>
    );
}

#[test]
fn auto_passes_over_devices_that_lack_something_and_returns_any_other_failure() {
    use kodamaembed_cuda::EncoderError as CudaEncoderError;
    use kodamaembed_vulkan::EncoderError as VulkanEncoderError;
    let devices = [
        Device::Cuda(0),
        Device::Vulkan(0),
        Device::Vulkan(1),
        Device::Vulkan(2),
    ];
    let lacking = |device: Device| match device {
        Device::Cuda(_) => Error::Cuda(CudaEncoderError::Unsupported("compute".into())),
        Device::Vulkan(0) => Error::Unsupported("shapes".into()),
        _ => Error::Vulkan(VulkanEncoderError::Unsupported("limits".into())),
    };

    let mut tried = Vec::new();
    let usable = first_usable(devices, |device| {
        tried.push(device);
        if device == Device::Vulkan(2) {
            Ok(device)
        } else {
            Err(lacking(device))
        }
    });
    assert!(matches!(usable, Ok(Some(Device::Vulkan(2)))), "{usable:?}");
    assert_eq!(tried, devices);

    let usable = first_usable(devices, |device| Err::<Device, _>(lacking(device)));
    assert!(matches!(usable, Ok(None)), "{usable:?}");

    let mut tried = Vec::new();
    let usable = first_usable(devices, |device| {
        tried.push(device);
        Err::<Device, _>(Error::Cuda(CudaEncoderError::Weights(
            "out of memory".into(),
        )))
    });
    assert!(
        matches!(usable, Err(Error::Cuda(CudaEncoderError::Weights(_)))),
        "{usable:?}"
    );
    assert_eq!(tried, [Device::Cuda(0)]);
}

/// CUDA's attention grid holds 65,535 sequences.
#[test]
fn a_batch_runs_before_it_outgrows_the_backend() {
    let limits = Limits {
        tokens: 10,
        sequences: 3,
    };
    assert!(!limits.full(0, 0, 10));
    assert!(!limits.full(6, 2, 4));
    assert!(limits.full(6, 2, 5), "11 tokens");
    assert!(limits.full(3, 3, 1), "4 sequences");
}

#[test]
fn models_are_shared_between_threads() {
    fn shared<T: Send + Sync>() {}
    shared::<Embedder>();
    shared::<Reranker>();
}

/// The long call's sink holds its first batch until the short call has run whole.
#[test]
fn a_short_call_runs_between_two_batches_of_a_long_one() {
    use std::{sync::mpsc, time::Duration};
    let embedder = embedder(Output::Embedding(Pooling::Mean));
    let long = vec!["a b"; CHUNK * 3];
    let expected = embedder.embed_passages(&long).unwrap();
    let alone = embedder.embed_queries(&["c a"]).unwrap();
    let (delivered, first_batch) = mpsc::channel();
    let (answered, answer) = mpsc::channel();
    let mut rows = Vec::new();
    std::thread::scope(|scope| {
        let shared = &embedder;
        scope.spawn(move || {
            first_batch.recv().unwrap();
            let between = shared.embed_queries(&["c a"]).unwrap();
            answered.send(between).unwrap();
        });
        embedder
            .embed_passages_stream(long.iter(), |batch| {
                if batch.start == 0 {
                    delivered.send(()).unwrap();
                    let between = answer
                        .recv_timeout(Duration::from_secs(60))
                        .expect("the short call waited for the long one");
                    assert_same_bits(&between[0], &alone[0]);
                }
                rows.extend(batch.values.chunks_exact(batch.dim).map(<[f32]>::to_vec));
                Ok::<_, std::convert::Infallible>(())
            })
            .unwrap();
    });
    assert_eq!(rows.len(), expected.len());
    for (row, expected) in rows.iter().zip(&expected) {
        assert_same_bits(row, expected);
    }
}

/// The holder asks again at once, as a long call's next batch does, and goes behind both waiters.
#[test]
fn callers_take_the_device_in_arrival_order() {
    use std::sync::Mutex;
    let embedder = embedder(Output::Embedding(Pooling::Mean));
    let device = &embedder.model.device;
    let order = Mutex::new(Vec::new());
    let record = |caller| order.lock().unwrap().push(caller);
    std::thread::scope(|scope| {
        let held = device.turn();
        for waiter in ["first", "second"] {
            let queued = device.places().taken;
            scope.spawn(move || {
                let _turn = device.turn();
                record(waiter);
            });
            while device.places().taken == queued {
                std::thread::yield_now();
            }
        }
        assert!(order.lock().unwrap().is_empty());
        drop(held);
        let _next = device.turn();
        record("holder");
    });
    assert_eq!(order.into_inner().unwrap(), ["first", "second", "holder"]);
}

/// On its turn, a `stop` that waits for another call would hold the device from that call.
#[test]
fn stop_is_asked_before_the_callers_turn() {
    use std::{sync::mpsc, time::Duration};
    let embedder = embedder(Output::Embedding(Pooling::Mean));
    let alone = embedder.embed_queries(&["c a"]).unwrap();
    std::thread::scope(|scope| {
        let shared = &embedder;
        let embedded = embedder.embed_passages_until(
            ["a b"],
            || {
                let (answered, answer) = mpsc::channel();
                scope.spawn(move || answered.send(shared.embed_queries(&["c a"])));
                let meanwhile = answer
                    .recv_timeout(Duration::from_secs(60))
                    .expect("the other call waited for this call's turn");
                assert_same_bits(&meanwhile.unwrap()[0], &alone[0]);
                false
            },
            |_| Ok::<_, std::convert::Infallible>(()),
        );
        assert_eq!(embedded.unwrap(), 1);
    });
}

/// A target without threads tokenizes each chunk when inference reaches it.
#[test]
fn chunks_tokenized_on_demand_give_the_producer_threads_rows() {
    let embedder = embedder(Output::Embedding(Pooling::Mean));
    let texts: Vec<&str> = (0..CHUNK * 2 + 7)
        .map(|i| ["a", "b c", "a b c"][i % 3])
        .collect();
    let expected = embedder.embed_passages(&texts).unwrap();
    let model = &embedder.model;
    let mut inputs = texts.iter().copied();
    let first: Vec<&str> = inputs.by_ref().take(CHUNK).collect();
    let tokenized = std::cell::Cell::new(0);
    let (mut seen, mut rows) = (Vec::new(), Vec::new());
    let count = Model::run_encoded(
        &model.device,
        tokenized_on_demand(first, inputs, |chunk| {
            tokenized.set(tokenized.get() + 1);
            tokenize_texts(&model.tokenizer, &model.tokenizing, "", chunk)
        }),
        || false,
        |batch: Batch<'_>| {
            seen.push(tokenized.get());
            rows.extend(batch.values.chunks_exact(batch.dim).map(<[f32]>::to_vec));
            Ok::<_, std::convert::Infallible>(())
        },
    )
    .unwrap();
    assert_eq!(count, texts.len());
    assert_eq!((seen[0], tokenized.get()), (1, 3), "one chunk at a time");
    for (row, expected) in rows.iter().zip(&expected) {
        assert_same_bits(row, expected);
    }
}

#[test]
fn a_panic_inside_a_batch_stops_later_calls_and_one_outside_does_not() {
    let embedder = embedder(Output::Embedding(Pooling::Mean));
    let callers = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        embedder.embed_passages_until(
            ["a"],
            || panic!("the caller's own"),
            |_| Ok::<_, std::convert::Infallible>(()),
        )
    }));
    assert!(callers.is_err());
    assert_eq!(embedder.embed_passages(&["a"]).unwrap().len(), 1);
    std::thread::scope(|scope| {
        let holder = scope.spawn(|| {
            let _backend = embedder.model.device.backend.lock().unwrap();
            panic!("inside a batch");
        });
        assert!(holder.join().is_err());
    });
    assert!(matches!(
        embedder.embed_passages(&["a"]),
        Err(Error::Poisoned)
    ));
}

#[test]
fn max_length_leaves_room_for_text_and_fits_the_positions() {
    // BERT pairs add [CLS] and two [SEP]: five tokens hold one of each side's.
    assert!(check_max_length(4, 3, 512).is_err());
    assert!(check_max_length(5, 3, 512).is_ok());
    assert!(check_max_length(512, 3, 512).is_ok());
    assert!(check_max_length(513, 3, 512).is_err());
    assert!(check_max_length(512, usize::MAX, 512).is_err());
}

/// A tiny BERT `config.json` and no weights: a load that passes every check stops at the weights.
struct TinyModel {
    dir: TempDir,
}

impl TinyModel {
    fn new(vocab_size: usize) -> Self {
        let dir = TempDir::new("tiny-model");
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&tiny_bert_config(vocab_size)).unwrap(),
        )
        .unwrap();
        Self { dir }
    }

    fn write_tokenizer(&self, json: impl AsRef<[u8]>) {
        std::fs::write(self.dir.path().join("tokenizer.json"), json).unwrap();
    }

    /// An embedder and a reranker share every check before the checkpoint.
    fn loads(&self, max_length: usize) -> [Result<(), Error>; 2] {
        let spec = ModelSpec {
            passage_prefix: "",
            query_prefix: "",
            max_length,
            pooling: Pooling::Mean,
        };
        [
            Embedder::builder()
                .device(Device::Cpu)
                .batch_tokens(16)
                .build_from_dir(self.dir.path(), spec)
                .map(|_| ()),
            Reranker::builder()
                .device(Device::Cpu)
                .batch_tokens(16)
                .build_from_dir(self.dir.path(), max_length)
                .map(|_| ()),
        ]
    }
}

#[test]
fn asymmetric_special_token_counts_reject_before_checkpoint_loading() {
    use tokenizers::{models::wordlevel::WordLevel, processors::template::TemplateProcessing};

    let fixture = TinyModel::new(4);
    let model = WordLevel::builder()
        .vocab(
            [("[UNK]", 0), ("[CLS]", 1), ("[SEP]", 2), ("a", 3)]
                .into_iter()
                .map(|(word, id)| (word.to_owned(), id))
                .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let single_six = "[CLS] [CLS] [CLS] [CLS] [CLS] $A [SEP]";
    let single_two = "[CLS] $A [SEP]";
    let pair_six = "[CLS] [CLS] [CLS] [CLS] $A [SEP] $B:1 [SEP]:1";
    let pair_three = "[CLS] $A [SEP] $B:1 [SEP]:1";
    for (single, pair, max_length, accepted) in [
        (single_six, pair_three, 5, false),
        (single_six, pair_three, 7, false),
        (single_six, pair_three, 8, true),
        (single_two, pair_six, 7, false),
        (single_two, pair_six, 8, true),
        (single_two, pair_three, 5, true),
    ] {
        let mut tokenizer = Tokenizer::new(model.clone());
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single(single)
                .unwrap()
                .try_pair(pair)
                .unwrap()
                .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
                .build()
                .unwrap(),
        ));
        fixture.write_tokenizer(tokenizer.to_string(false).unwrap());
        for result in fixture.loads(max_length) {
            let Err(Error::Model(reason)) = result else {
                panic!("expected model validation result, got {result:?}");
            };
            // A valid budget reaches the deliberately absent checkpoint.
            assert!(
                if accepted {
                    reason.contains("model.safetensors")
                } else {
                    reason.contains("max_length") && reason.contains("special tokens")
                },
                "single={single}, pair={pair}, max_length={max_length}: {reason}"
            );
        }
    }
}

#[test]
fn stochastic_bpe_is_rejected_before_weights_or_backend_loading() {
    let fixture = TinyModel::new(4);
    for dropout in [
        None,
        Some(0.0),
        Some(1.0),
        Some(f32::EPSILON),
        Some(0.5),
        Some(1.0 - f32::EPSILON),
    ] {
        let mut builder = tokenizers::models::bpe::BPE::builder().vocab_and_merges(
            [
                ("a".into(), 0),
                ("b".into(), 1),
                ("ab".into(), 2),
                ("[UNK]".into(), 3),
            ]
            .into_iter()
            .collect::<tokenizers::models::bpe::Vocab>(),
            vec![("a".into(), "b".into())],
        );
        if let Some(value) = dropout {
            builder = builder.dropout(value);
        }
        let tokenizer = Tokenizer::new(builder.build().unwrap());
        let mut json: serde_json::Value =
            serde_json::from_str(&tokenizer.to_string(false).unwrap()).unwrap();
        // Tagged and untagged model JSON must both use the parsed model's policy.
        for tagged in [true, false] {
            if !tagged {
                json["model"].as_object_mut().unwrap().remove("type");
            }
            fixture.write_tokenizer(serde_json::to_vec(&json).unwrap());
            for result in fixture.loads(8) {
                match result {
                    Err(Error::Tokenizer(reason))
                        if dropout.is_some_and(|p| p > 0.0 && p < 1.0) =>
                    {
                        assert!(reason.contains("dropout") && reason.contains("stochastic"));
                    }
                    Err(Error::Model(reason)) if dropout.is_none_or(|p| p == 0.0 || p == 1.0) => {
                        // The fixture has no weights, so loading stops there.
                        assert!(reason.contains("model.safetensors"));
                    }
                    other => panic!("dropout {dropout:?}, tagged {tagged}: {other:?}"),
                }
            }
        }
    }
}

#[test]
fn token_ids_past_the_embedding_table_reject_before_checkpoint_loading() {
    use tokenizers::models::wordlevel::WordLevel;

    // Four rows, IDs 0 to 3; the largest ID the tokenizer holds decides.
    let fixture = TinyModel::new(4);
    for (largest, accepted) in [(3, true), (4, false), (250_001, false)] {
        let vocab = [("[UNK]", 0), ("a", 1), ("b", largest)]
            .into_iter()
            .map(|(word, id)| (word.to_owned(), id))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        fixture.write_tokenizer(Tokenizer::new(model).to_string(false).unwrap());
        for result in fixture.loads(8) {
            let Err(Error::Model(reason)) = result else {
                panic!("largest ID {largest}: expected a model error, got {result:?}");
            };
            if accepted {
                assert!(reason.contains("model.safetensors"), "{reason}");
            } else {
                assert!(
                    reason.contains("tokenizer.json")
                        && reason.contains(&format!("ID {largest}"))
                        && reason.contains("4 rows"),
                    "{reason}"
                );
            }
        }
    }
}

/// Two of XLM-R's positions sit before its first token.
#[test]
fn xlm_roberta_sequences_hold_two_tokens_fewer_than_their_positions() {
    use tokenizers::models::wordlevel::WordLevel;

    let fixture = TinyModel::new(4);
    let mut config = tiny_bert_config(4);
    config["model_type"] = "xlm-roberta".into();
    config["pad_token_id"] = 1.into();
    config["max_position_embeddings"] = 10.into();
    std::fs::write(fixture.dir.path().join("config.json"), config.to_string()).unwrap();
    let vocab = [("[UNK]", 0), ("a", 1)]
        .into_iter()
        .map(|(word, id)| (word.to_owned(), id))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    fixture.write_tokenizer(Tokenizer::new(model).to_string(false).unwrap());
    for (max_length, accepted) in [(8, true), (9, false)] {
        for result in fixture.loads(max_length) {
            let Err(Error::Model(reason)) = result else {
                panic!("max_length {max_length}: expected a model error, got {result:?}");
            };
            let expected = if accepted {
                "model.safetensors"
            } else {
                "8-token"
            };
            assert!(reason.contains(expected), "{reason}");
        }
    }
}

/// Whitespace-separated words of a four-entry vocabulary, with no special tokens.
fn word_tokenizer() -> Tokenizer {
    let words = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(
            [
                ("[UNK]".into(), 0),
                ("a".into(), 1),
                ("b".into(), 2),
                ("c".into(), 3),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(words);
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    tokenizer
}

const MODEL_FILES: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];

/// A complete tiny model's three files, in [`MODEL_FILES`]' order.
fn tiny_model_files(relevance: bool) -> [Vec<u8>; 3] {
    let config = tiny_bert_config(4);
    [
        serde_json::to_vec(&config).unwrap(),
        word_tokenizer().to_string(false).unwrap().into_bytes(),
        tiny_bert_checkpoint(&config, relevance),
    ]
}

fn write_model(files: &[Vec<u8>; 3]) -> TempDir {
    let dir = TempDir::new("model-files");
    for (name, bytes) in MODEL_FILES.into_iter().zip(files) {
        std::fs::write(dir.path().join(name), bytes).unwrap();
    }
    dir
}

#[test]
fn an_embedder_loads_from_memory_as_from_its_directory() {
    let spec = ModelSpec {
        passage_prefix: "a ",
        query_prefix: "b ",
        max_length: 8,
        pooling: Pooling::Mean,
    };
    let files = tiny_model_files(false);
    let dir = write_model(&files);
    let builder = Embedder::builder().device(Device::Cpu).batch_tokens(16);
    let from_dir = builder.build_from_dir(dir.path(), spec).unwrap();
    let [config, tokenizer, weights] = &files;
    let from_memory = builder
        .build_from_bytes(config, tokenizer, weights, spec)
        .unwrap();
    assert_eq!(from_memory.fingerprint(), from_dir.fingerprint());
    let texts = ["a b", "c"];
    for (memory, directory) in [
        (
            from_memory.embed_passages(&texts).unwrap(),
            from_dir.embed_passages(&texts).unwrap(),
        ),
        (
            from_memory.embed_queries(&texts).unwrap(),
            from_dir.embed_queries(&texts).unwrap(),
        ),
    ] {
        assert_eq!(memory.len(), texts.len());
        for (memory, directory) in memory.iter().zip(&directory) {
            assert_same_bits(memory, directory);
        }
    }
}

#[test]
fn a_reranker_loads_from_memory_as_from_its_directory() {
    let files = tiny_model_files(true);
    let dir = write_model(&files);
    let builder = Reranker::builder().device(Device::Cpu).batch_tokens(16);
    let from_dir = builder.build_from_dir(dir.path(), 8).unwrap();
    let [config, tokenizer, weights] = &files;
    let from_memory = builder
        .build_from_bytes(config, tokenizer, weights, 8)
        .unwrap();
    assert_eq!(from_memory.fingerprint(), from_dir.fingerprint());
    let passages = ["b", "c a", "a b c"];
    assert_same_bits(
        &from_memory.score("a", &passages).unwrap(),
        &from_dir.score("a", &passages).unwrap(),
    );
}

#[test]
fn a_broken_file_in_memory_is_refused_as_in_its_directory() {
    let spec = ModelSpec {
        passage_prefix: "",
        query_prefix: "",
        max_length: 8,
        pooling: Pooling::Mean,
    };
    let builder = Embedder::builder().device(Device::Cpu).batch_tokens(16);
    for (broken, name) in MODEL_FILES.into_iter().enumerate() {
        let mut files = tiny_model_files(false);
        files[broken].truncate(1);
        let dir = write_model(&files);
        let [config, tokenizer, weights] = &files;
        let refusals = [
            builder.build_from_bytes(config, tokenizer, weights, spec),
            builder.build_from_dir(dir.path(), spec),
        ]
        .map(|loaded| match loaded {
            Ok(_) => panic!("a one-byte {name} loaded"),
            Err(error) => error.to_string(),
        });
        assert_eq!(refusals[0], refusals[1], "{name}");
    }
}

#[test]
fn prefixes_live_only_until_the_embedder_is_built() {
    let files = tiny_model_files(false);
    let [config, tokenizer, weights] = &files;
    let builder = Embedder::builder().device(Device::Cpu).batch_tokens(16);
    let owned = {
        let prefixes = [String::from("a "), String::from("b ")];
        let spec = ModelSpec {
            passage_prefix: &prefixes[0],
            query_prefix: &prefixes[1],
            max_length: 8,
            pooling: Pooling::Mean,
        };
        builder
            .build_from_bytes(config, tokenizer, weights, spec)
            .unwrap()
    };
    let spec = ModelSpec {
        passage_prefix: "a ",
        query_prefix: "b ",
        max_length: 8,
        pooling: Pooling::Mean,
    };
    let literal = builder
        .build_from_bytes(config, tokenizer, weights, spec)
        .unwrap();
    assert_eq!(owned.fingerprint(), literal.fingerprint());
    let texts = ["c", "a b"];
    let passages = owned.embed_passages(&texts).unwrap();
    let queries = owned.embed_queries(&texts).unwrap();
    assert_ne!(passages, queries);
    for (owned, literal) in [
        (passages, literal.embed_passages(&texts).unwrap()),
        (queries, literal.embed_queries(&texts).unwrap()),
    ] {
        for (owned, literal) in owned.iter().zip(&literal) {
            assert_same_bits(owned, literal);
        }
    }
}

/// A complete tiny ModernBERT's three files, its tensors named under `prefix`.
fn tiny_modernbert_files(prefix: &str) -> [Vec<u8>; 3] {
    let config = tiny_modernbert_config(4);
    [
        serde_json::to_vec(&config).unwrap(),
        word_tokenizer().to_string(false).unwrap().into_bytes(),
        tiny_modernbert_checkpoint(&config, prefix),
    ]
}

/// The six-token text reaches past the local layer's window of four.
#[test]
fn a_modernbert_checkpoint_embeds_as_the_reference_forward_from_disk_and_memory() {
    let builder = Embedder::builder().device(Device::Cpu).batch_tokens(16);
    let (texts, ids, lengths) = (["a b", "c b a c c a"], [1, 2, 3, 2, 1, 3, 3, 1], [2, 6]);
    for prefix in ["", "model."] {
        let files = tiny_modernbert_files(prefix);
        let dir = write_model(&files);
        let [config, tokenizer, weights] = &files;
        let shape = checkpoint::shape_from_bytes(config).unwrap();
        let tensors = checkpoint::SafeTensors::from_bytes(Cow::Borrowed(weights)).unwrap();
        for pooling in [Pooling::Mean, Pooling::Cls] {
            let spec = ModelSpec {
                passage_prefix: "",
                query_prefix: "",
                max_length: 8,
                pooling,
            };
            let from_dir = builder.build_from_dir(dir.path(), spec).unwrap();
            let from_memory = builder
                .build_from_bytes(config, tokenizer, weights, spec)
                .unwrap();
            assert_eq!(from_memory.fingerprint(), from_dir.fingerprint());
            let vectors = from_dir.embed_passages(&texts).unwrap();
            for (memory, directory) in from_memory
                .embed_passages(&texts)
                .unwrap()
                .iter()
                .zip(&vectors)
            {
                assert_same_bits(memory, directory);
            }
            let expected =
                kodamaembed_model::reference::encode(&shape, &tensors, pooling, &ids, &lengths)
                    .unwrap();
            for (vector, expected) in vectors.iter().zip(&expected) {
                let error = vector
                    .iter()
                    .zip(expected)
                    .map(|(&have, want)| (f64::from(have) - want).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!(error <= 1e-6, "{prefix:?} {pooling:?}: L2 error {error:e}");
            }
        }
    }
}

#[test]
fn a_modernbert_too_small_for_the_gpu_kernels_takes_the_processor() {
    let [config, tokenizer, weights] = &tiny_modernbert_files("");
    let spec = ModelSpec {
        passage_prefix: "",
        query_prefix: "",
        max_length: 8,
        pooling: Pooling::Mean,
    };
    let embedders = Embedder::builder().batch_tokens(16);
    let automatic = embedders
        .build_from_bytes(config, tokenizer, weights, spec)
        .unwrap();
    assert!(automatic.device_name().starts_with("CPU"));
    assert!(automatic.cpu_workers().is_some());
}

/// [`word_tokenizer`]'s words between ModernBERT's special tokens, a pair in one segment.
fn framed_tokenizer() -> Tokenizer {
    use tokenizers::{models::wordlevel::WordLevel, processors::template::TemplateProcessing};
    let words = WordLevel::builder()
        .vocab(
            [
                ("[UNK]", 0),
                ("a", 1),
                ("b", 2),
                ("c", 3),
                ("[CLS]", 4),
                ("[SEP]", 5),
            ]
            .into_iter()
            .map(|(word, id)| (word.to_owned(), id))
            .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(words);
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .unwrap()
            .try_pair("[CLS] $A [SEP] $B:0 [SEP]:0")
            .unwrap()
            .special_tokens(vec![("[CLS]", 4), ("[SEP]", 5)])
            .build()
            .unwrap(),
    ));
    tokenizer
}

/// The head each config names, on pairs framed as `[CLS] a [SEP] passage [SEP]`.
#[test]
fn a_modernbert_reranker_scores_as_the_reference_head_from_disk_and_memory() {
    let builder = Reranker::builder().device(Device::Cpu).batch_tokens(16);
    let passages = ["b", "c a", "a b c"];
    let ids = [4, 1, 5, 2, 5, 4, 1, 5, 3, 1, 5, 4, 1, 5, 1, 2, 3, 5];
    let lengths = [5, 6, 7];
    for (pooling, activation, head) in [
        (
            "cls",
            "silu",
            Head::Prediction(Pooling::Cls, Activation::Silu),
        ),
        (
            "mean",
            "gelu",
            Head::Prediction(Pooling::Mean, Activation::Gelu),
        ),
    ] {
        let mut config = tiny_modernbert_config(6);
        config["classifier_pooling"] = pooling.into();
        config["classifier_activation"] = activation.into();
        let files = [
            serde_json::to_vec(&config).unwrap(),
            framed_tokenizer().to_string(false).unwrap().into_bytes(),
            tiny_modernbert_cross_encoder(&config),
        ];
        let dir = write_model(&files);
        let [config, tokenizer, weights] = &files;
        let from_dir = builder.build_from_dir(dir.path(), 8).unwrap();
        let from_memory = builder
            .build_from_bytes(config, tokenizer, weights, 8)
            .unwrap();
        assert_eq!(from_memory.fingerprint(), from_dir.fingerprint());
        let scores = from_dir.score("a", &passages).unwrap();
        assert_same_bits(&from_memory.score("a", &passages).unwrap(), &scores);
        let shape = checkpoint::shape_from_bytes(config).unwrap();
        let tensors = checkpoint::SafeTensors::from_bytes(Cow::Borrowed(weights)).unwrap();
        let expected =
            kodamaembed_model::reference::rerank(&shape, &tensors, head, &ids, &[0; 18], &lengths)
                .unwrap();
        for (&score, expected) in scores.iter().zip(&expected) {
            assert!(
                (f64::from(score) - expected).abs() <= 1e-6 * (1.0 + expected.abs()),
                "{head:?}: {score} vs {expected}"
            );
        }
    }
}

#[test]
fn a_modernbert_embedder_profiles_and_takes_explicit_workers() {
    let [config, tokenizer, weights] = &tiny_modernbert_files("");
    let spec = ModelSpec {
        passage_prefix: "",
        query_prefix: "",
        max_length: 8,
        pooling: Pooling::Mean,
    };
    let workers = explicit_workers(2, 1);
    let builder = Embedder::builder().device(Device::Cpu).batch_tokens(16);
    let mut bounded = builder
        .workers(workers)
        .build_from_bytes(config, tokenizer, weights, spec)
        .unwrap();
    assert_eq!(bounded.cpu_workers(), Some(workers));
    let shared = builder
        .build_from_bytes(config, tokenizer, weights, spec)
        .unwrap();
    assert_ne!(bounded.fingerprint(), shared.fingerprint());
    bounded.enable_profiling().unwrap();
    let texts = ["a b c", "c"];
    let vectors = bounded.embed_passages(&texts).unwrap();
    for (bounded, shared) in vectors.iter().zip(&shared.embed_passages(&texts).unwrap()) {
        assert_same_bits(bounded, shared);
    }
    let profile = bounded.profile().unwrap();
    assert_eq!((profile.batches, profile.tokens), (1, 4));
    assert!(bounded.gpu_identity().is_err());
}

fn embedder(output: Output) -> Embedder {
    test_embedder(output, false)
}

fn test_embedder(output: Output, bert_inputs: bool) -> Embedder {
    test_embedder_with_workers(output, bert_inputs, None)
}

fn test_embedder_with_workers(
    output: Output,
    bert_inputs: bool,
    workers: Option<CpuWorkers>,
) -> Embedder {
    struct Weights;
    impl kodamaembed_model::Tensors for Weights {
        fn get(&self, name: &str, len: usize) -> Result<Vec<f32>, String> {
            Ok(if name.contains("LayerNorm.weight") {
                vec![1.0; len]
            } else {
                (0..len).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect()
            })
        }
    }
    let shape = Shape {
        layers: 1,
        hidden: 8,
        heads: 2,
        intermediate: 16,
        vocab: if bert_inputs { 2003 } else { 4 },
        max_positions: 16,
        type_vocab: 2,
        eps: 1e-5,
        positions: Positions::Sequential,
    };
    let tokenizer = if bert_inputs {
        bert_wordpiece::tests::configured()
    } else {
        word_tokenizer()
    };
    Embedder {
        model: Model {
            device: Shared::new(Backend::Cpu(Box::new(
                match workers {
                    Some(workers) => kodamaembed_cpu::Encoder::new_with_precision_and_workers(
                        kodamaembed_cpu::Kernel::Portable,
                        shape,
                        output,
                        8,
                        &Weights,
                        CpuPrecision::Fp32,
                        workers.inference_workers,
                    ),
                    None => kodamaembed_cpu::Encoder::new(
                        kodamaembed_cpu::Kernel::Portable,
                        shape,
                        output,
                        8,
                        &Weights,
                    ),
                }
                .unwrap(),
            ))),
            device_name: "test CPU".into(),
            tokenizer: tokenizing::InferenceTokenizer::new(tokenizer),
            tokenizing: kodamaembed_cpu::worker_pool(
                "test-tokenize",
                workers.map_or(1, |w| w.tokenizer_workers.get()),
            )
            .unwrap(),
            hidden: shape.hidden,
            fingerprint: "test".into(),
        },
        passage_prefix: "".into(),
        query_prefix: "".into(),
    }
}

#[test]
fn guarded_tokenization_preserves_streamed_embedding_bits() {
    let fast = test_embedder(Output::Embedding(Pooling::Mean), true);
    let mut reference = test_embedder(Output::Embedding(Pooling::Mean), true);
    let mut original = bert_wordpiece::tests::configured();
    // A nonzero stride selects the reference tokenizer without changing the first window.
    original.get_truncation_mut().unwrap().stride = 1;
    reference.model.tokenizer = tokenizing::InferenceTokenizer::new(original);
    let texts: Vec<_> = (0..150)
        .map(|i| {
            [
                "hello",
                "worlds café [MASK]",
                "hel\0lo",
                "hello\u{301} worlds",
                "hello worlds hello worlds hello worlds hello worlds",
            ][i % 5]
        })
        .collect();
    let expected = reference.embed_passages(&texts).unwrap();
    let mut rows = Vec::new();
    let count = fast
        .embed_passages_stream(texts, |batch| {
            assert_eq!(batch.start, rows.len());
            rows.extend(batch.values.chunks_exact(batch.dim).map(<[f32]>::to_vec));
            Ok::<_, std::io::Error>(())
        })
        .unwrap();
    assert_eq!(count, expected.len());
    for (actual, expected) in rows.iter().zip(expected) {
        assert_same_bits(actual, &expected);
    }
}

fn explicit_workers(inference: usize, tokenizer: usize) -> CpuWorkers {
    CpuWorkers {
        inference_workers: NonZeroUsize::new(inference).unwrap(),
        tokenizer_workers: NonZeroUsize::new(tokenizer).unwrap(),
    }
}

#[test]
fn explicit_worker_counts_preserve_embedding_and_reranking_bits() {
    let texts = ["a", "b c", "c a b", "a b c a b", "c"];
    let expected_embeddings = embedder(Output::Embedding(Pooling::Mean))
        .embed_passages(&texts)
        .unwrap();
    let reference = Reranker {
        model: embedder(Output::Relevance(Head::Pooler)).model,
    };
    let expected_scores = reference.score("a", &texts).unwrap();
    for workers in [
        explicit_workers(1, 1),
        explicit_workers(2, 1),
        explicit_workers(1, 2),
        explicit_workers(2, 2),
    ] {
        let bounded =
            test_embedder_with_workers(Output::Embedding(Pooling::Mean), false, Some(workers));
        assert_eq!(bounded.cpu_workers(), Some(workers));
        let actual = bounded.embed_passages(&texts).unwrap();
        for (actual, expected) in actual.iter().zip(&expected_embeddings) {
            assert_same_bits(actual, expected);
        }
        let reranker = Reranker {
            model: test_embedder_with_workers(
                Output::Relevance(Head::Pooler),
                false,
                Some(workers),
            )
            .model,
        };
        assert_eq!(reranker.cpu_workers(), Some(workers));
        assert_same_bits(&reranker.score("a", &texts).unwrap(), &expected_scores);
    }
}

#[test]
fn independent_worker_budgets_can_run_concurrently_without_changing_outputs() {
    let one = test_embedder_with_workers(
        Output::Embedding(Pooling::Mean),
        false,
        Some(explicit_workers(1, 1)),
    );
    let two = test_embedder_with_workers(
        Output::Embedding(Pooling::Mean),
        false,
        Some(explicit_workers(2, 2)),
    );
    let texts = ["a", "a b", "c", "b c a"];
    let expected = embedder(Output::Embedding(Pooling::Mean))
        .embed_passages(&texts)
        .unwrap();
    std::thread::scope(|scope| {
        let first = scope.spawn(|| one.embed_passages(&texts).unwrap());
        let second = scope.spawn(|| two.embed_passages(&texts).unwrap());
        for actual in [first.join().unwrap(), second.join().unwrap()] {
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_same_bits(actual, expected);
            }
        }
    });
    assert_eq!(one.cpu_workers(), Some(explicit_workers(1, 1)));
    assert_eq!(two.cpu_workers(), Some(explicit_workers(2, 2)));
}

#[test]
fn explicit_worker_stream_sink_errors_leave_both_models_reusable() {
    let workers = explicit_workers(1, 1);
    let texts = vec!["a b"; CHUNK * 5];
    let bounded =
        test_embedder_with_workers(Output::Embedding(Pooling::Mean), false, Some(workers));
    let failure = bounded.embed_passages_stream(texts.iter(), |_| Err::<(), _>("disk full"));
    assert!(matches!(failure, Err(StreamError::Sink("disk full"))));
    let expected = embedder(Output::Embedding(Pooling::Mean))
        .embed_queries(&["c a"])
        .unwrap();
    let actual = bounded.embed_queries(&["c a"]).unwrap();
    assert_same_bits(&actual[0], &expected[0]);
    let reranker = Reranker {
        model: test_embedder_with_workers(Output::Relevance(Head::Pooler), false, Some(workers))
            .model,
    };
    let failure = reranker.score_stream("a", texts.iter(), |_| Err::<(), _>("disk full"));
    assert!(matches!(failure, Err(StreamError::Sink("disk full"))));
    let reference = Reranker {
        model: embedder(Output::Relevance(Head::Pooler)).model,
    };
    assert_same_bits(
        &reranker.score("a", &["b c"]).unwrap(),
        &reference.score("a", &["b c"]).unwrap(),
    );
}

#[test]
fn a_stop_request_ends_an_embedding_run_before_its_next_batch() {
    let texts = vec!["a b"; CHUNK * 3];
    let embedder = embedder(Output::Embedding(Pooling::Mean));
    let expected = embedder.embed_passages(&texts).unwrap();
    let mut starts = Vec::new();
    embedder
        .embed_passages_stream(texts.iter(), |batch| {
            starts.push(batch.start);
            Ok::<_, std::convert::Infallible>(())
        })
        .unwrap();
    for batches in [0, 1, starts.len() - 1] {
        let mut asked = 0;
        let mut rows = Vec::new();
        let outcome = embedder.embed_passages_until(
            texts.iter(),
            || {
                asked += 1;
                asked > batches
            },
            |batch| {
                rows.extend(batch.values.chunks_exact(batch.dim).map(<[f32]>::to_vec));
                Ok::<_, std::convert::Infallible>(())
            },
        );
        assert!(matches!(
            outcome,
            Err(StreamError::Inference(Error::Interrupted))
        ));
        assert_eq!((asked, rows.len()), (batches + 1, starts[batches]));
        for (row, expected) in rows.iter().zip(&expected) {
            assert_same_bits(row, expected);
        }
    }
}

#[test]
fn runs_that_are_never_stopped_embed_as_passages_and_as_queries() {
    let mut embedder = embedder(Output::Embedding(Pooling::Mean));
    embedder.passage_prefix = "a ".into();
    embedder.query_prefix = "b ".into();
    let texts = ["c", "a b"];
    let passages = embedder.embed_passages(&texts).unwrap();
    let queries = embedder.embed_queries(&texts).unwrap();
    assert_ne!(passages, queries);
    let mut rows = Vec::new();
    let count = embedder
        .embed_passages_until(
            texts,
            || false,
            |batch| {
                rows.extend(batch.values.chunks_exact(batch.dim).map(<[f32]>::to_vec));
                Ok::<_, std::convert::Infallible>(())
            },
        )
        .unwrap();
    assert_eq!((count, &rows), (texts.len(), &passages));
    rows.clear();
    let count = embedder
        .embed_queries_until(
            texts,
            || false,
            |batch| {
                rows.extend(batch.values.chunks_exact(batch.dim).map(<[f32]>::to_vec));
                Ok::<_, std::convert::Infallible>(())
            },
        )
        .unwrap();
    assert_eq!((count, &rows), (texts.len(), &queries));
}

#[test]
fn a_stop_request_ends_a_scoring_run_before_its_next_batch() {
    let passages = vec!["b c"; CHUNK * 3];
    let reranker = Reranker {
        model: embedder(Output::Relevance(Head::Pooler)).model,
    };
    let expected = reranker.score("a", &passages).unwrap();
    let mut asked = 0;
    let mut scores = Vec::new();
    let outcome = reranker.score_until(
        "a",
        passages.iter(),
        || {
            asked += 1;
            asked > 1
        },
        |batch| {
            scores.extend_from_slice(batch.values);
            Ok::<_, std::convert::Infallible>(())
        },
    );
    assert!(matches!(
        outcome,
        Err(StreamError::Inference(Error::Interrupted))
    ));
    assert!(!scores.is_empty() && scores.len() < expected.len());
    assert_same_bits(&scores, &expected[..scores.len()]);
    scores.clear();
    let count = reranker
        .score_until(
            "a",
            passages.iter(),
            || false,
            |batch| {
                scores.extend_from_slice(batch.values);
                Ok::<_, std::convert::Infallible>(())
            },
        )
        .unwrap();
    assert_eq!(count, expected.len());
    assert_same_bits(&scores, &expected);
}

#[test]
fn explicit_worker_zero_token_errors_leave_embedding_and_reranking_reusable() {
    let workers = explicit_workers(1, 1);
    let bounded =
        test_embedder_with_workers(Output::Embedding(Pooling::Mean), false, Some(workers));
    let mut delivered = false;
    let failure = bounded.embed_passages_stream([""], |_| {
        delivered = true;
        Ok::<(), ()>(())
    });
    assert!(matches!(
        failure,
        Err(StreamError::Inference(Error::Tokenizer(_)))
    ));
    assert!(!delivered);
    let expected = embedder(Output::Embedding(Pooling::Mean))
        .embed_passages(&["a"])
        .unwrap();
    assert_same_bits(&bounded.embed_passages(&["a"]).unwrap()[0], &expected[0]);
    let reranker = Reranker {
        model: test_embedder_with_workers(Output::Relevance(Head::Pooler), false, Some(workers))
            .model,
    };
    assert!(reranker.score("", &[""]).is_err());
    let reference = Reranker {
        model: embedder(Output::Relevance(Head::Pooler)).model,
    };
    assert_same_bits(
        &reranker.score("a", &["b"]).unwrap(),
        &reference.score("a", &["b"]).unwrap(),
    );
}

#[test]
fn excessive_explicit_worker_counts_reject_before_checkpoint_io() {
    let over_limit = rayon::max_num_threads().checked_add(1).unwrap();
    for workers in [
        explicit_workers(over_limit, 1),
        explicit_workers(1, over_limit),
    ] {
        assert!(matches!(
            Embedder::builder()
                .device(Device::Cpu)
                .batch_tokens(256)
                .workers(workers)
                .build_from_dir("missing-model", EmbeddingModel::AllMiniLmL6V2.spec()),
            Err(Error::Unsupported(message)) if message.contains("Rayon")
        ));
        assert!(matches!(
            Reranker::builder()
                .device(Device::Cpu)
                .batch_tokens(256)
                .workers(workers)
                .build_from_dir("missing-model", 256),
            Err(Error::Unsupported(message)) if message.contains("Rayon")
        ));
    }
}

#[test]
fn each_setting_routes_to_a_device_that_can_honor_it() {
    for (device, precision, workers, expected) in [
        (
            Device::Auto,
            Precision::Float,
            false,
            Some(Route::Float(Device::Auto)),
        ),
        (
            Device::Cuda(1),
            Precision::Float,
            false,
            Some(Route::Float(Device::Cuda(1))),
        ),
        (
            Device::Vulkan(2),
            Precision::Float,
            false,
            Some(Route::Float(Device::Vulkan(2))),
        ),
        (
            Device::Cpu,
            Precision::Float,
            true,
            Some(Route::Float(Device::Cpu)),
        ),
        (
            Device::Auto,
            Precision::Float,
            true,
            Some(Route::Float(Device::Cpu)),
        ),
        (Device::Cuda(0), Precision::Float, true, None),
        (Device::Vulkan(0), Precision::Float, true, None),
        (
            Device::Cpu,
            Precision::Quantized,
            false,
            Some(Route::CenteredCpu),
        ),
        (
            Device::Cpu,
            Precision::Quantized,
            true,
            Some(Route::CenteredCpu),
        ),
        (
            Device::Auto,
            Precision::Quantized,
            false,
            Some(Route::CenteredAuto),
        ),
        (
            Device::Auto,
            Precision::Quantized,
            true,
            Some(Route::CenteredCpu),
        ),
        (
            Device::Cuda(3),
            Precision::Quantized,
            false,
            Some(Route::CenteredCuda(3)),
        ),
        (Device::Cuda(3), Precision::Quantized, true, None),
        (Device::Vulkan(0), Precision::Quantized, false, None),
    ] {
        let case = format!("{device:?} {precision:?} workers {workers}");
        match (route(device, precision, workers), expected) {
            (Ok(actual), Some(expected)) => assert_eq!(actual, expected, "{case}"),
            (Err(Error::Unsupported(_)), None) => {}
            (other, _) => panic!("{case}: {other:?}"),
        }
    }
}

#[test]
fn settings_a_device_cannot_honor_fail_before_any_file_is_read() {
    let workers = explicit_workers(1, 1);
    let spec = EmbeddingModel::AllMiniLmL6V2.spec();
    // The directory does not exist, so reading a file would be a model error.
    for builder in [
        Embedder::builder()
            .device(Device::Vulkan(0))
            .precision(Precision::Quantized),
        Embedder::builder()
            .device(Device::Vulkan(0))
            .workers(workers),
        Embedder::builder()
            .device(Device::Cuda(0))
            .precision(Precision::Quantized)
            .workers(workers),
    ] {
        let result = builder.build_from_dir("missing-model", spec).map(|_| ());
        assert!(
            matches!(result, Err(Error::Unsupported(_))),
            "{builder:?}: {result:?}"
        );
    }
    let result = Reranker::builder()
        .device(Device::Cuda(0))
        .workers(workers)
        .build_from_dir("missing-model", 256)
        .map(|_| ());
    assert!(matches!(result, Err(Error::Unsupported(_))), "{result:?}");
}

#[test]
fn cpu_source_changes_invalidate_the_cpu_numerical_policy() {
    let sources = kodamaembed_cpu::NUMERICAL_SOURCES;
    let original = policy("cpu-centered-int8", sources);
    for (component, source) in sources.iter().enumerate() {
        let mut edited = source.to_vec();
        edited.extend_from_slice(b"\n// Simulated arithmetic revision.\n");
        let mut edited_sources = sources.to_vec();
        edited_sources[component] = &edited;
        assert_ne!(
            original,
            policy("cpu-centered-int8", &edited_sources),
            "{component}"
        );
    }
}

#[test]
fn explicit_worker_fingerprint_policy_is_conservative_and_the_default_label_is_unsuffixed() {
    let kernel = kodamaembed_cpu::Kernel::Portable;
    let default = cpu_policy_identity(kernel, "cpu-fp32-v1", None);
    assert_eq!(default, "CPU Portable: cpu-fp32-v1");
    let policy_fingerprint = |workers| {
        let label = cpu_policy_identity(kernel, "cpu-fp32-v1", workers);
        identity::fingerprint(identity::IdentityInputs {
            config_sha256: &[1; 32],
            tokenizer_sha256: &[2; 32],
            checkpoint_sha256: &[3; 32],
            output: Output::Embedding(Pooling::Mean),
            passage_prefix: "",
            query_prefix: "",
            max_length: 16,
            numerical_policy: &policy(&label, kodamaembed_cpu::NUMERICAL_SOURCES),
        })
    };
    let one = policy_fingerprint(Some(explicit_workers(1, 1)));
    assert_eq!(one, policy_fingerprint(Some(explicit_workers(1, 1))));
    assert_ne!(one, policy_fingerprint(None));
    assert_ne!(one, policy_fingerprint(Some(explicit_workers(2, 1))));
    assert_ne!(one, policy_fingerprint(Some(explicit_workers(1, 2))));
}

#[test]
fn chunk_owned_and_borrowed_texts_match_concatenation_for_both_prefix_paths() {
    // Cell makes this input Send but not Sync, as the streaming API permits.
    struct Text(String, std::cell::Cell<usize>);
    impl AsRef<str> for Text {
        fn as_ref(&self) -> &str {
            self.1.set(self.1.get() + 1);
            &self.0
        }
    }
    let texts: Vec<String> = (0..150)
        .map(|i| {
            [
                "hello worlds",
                "café 世界 [MASK]",
                "hel\0lo\u{b} worlds",
                "hello\u{301} worlds",
                "hello worlds hello worlds hello worlds hello worlds",
            ][i % 5]
                .to_owned()
        })
        .collect();
    for prefix in ["", "passage: "] {
        let reference = test_embedder(Output::Embedding(Pooling::Mean), true);
        let mut expected = Vec::new();
        reference
            .model
            .run(
                texts.iter().map(|text| format!("{prefix}{text}")),
                || false,
                |batch: Batch<'_>| {
                    assert_eq!(batch.start, expected.len());
                    let rows = batch.values.as_chunks::<8>().0;
                    expected.extend(rows.iter().map(|row| row.to_vec()));
                    Ok::<_, std::convert::Infallible>(())
                },
            )
            .unwrap();
        let mut actual = test_embedder(Output::Embedding(Pooling::Mean), true);
        actual.passage_prefix = prefix.into();
        actual.query_prefix = prefix.into();
        let mut owned = Vec::new();
        let count = actual
            .embed_passages_stream(
                texts
                    .iter()
                    .cloned()
                    .map(|text| Text(text, std::cell::Cell::new(0))),
                |batch| {
                    assert_eq!(batch.start, owned.len());
                    owned.extend(batch.values.chunks_exact(batch.dim).map(<[f32]>::to_vec));
                    Ok::<_, std::convert::Infallible>(())
                },
            )
            .unwrap();
        assert_eq!(count, texts.len());
        let borrowed = actual
            .embed_queries(&texts.iter().map(String::as_str).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(owned.len(), expected.len());
        assert_eq!(borrowed.len(), expected.len());
        for ((owned, borrowed), expected) in owned.iter().zip(&borrowed).zip(&expected) {
            assert_same_bits(owned, expected);
            assert_same_bits(borrowed, expected);
        }
    }
}

#[test]
fn borrowed_chunk_sink_failure_drops_owned_inputs_and_allows_reuse() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    struct Text {
        value: String,
        drops: Arc<AtomicUsize>,
        _marker: std::cell::Cell<()>,
    }
    impl AsRef<str> for Text {
        fn as_ref(&self) -> &str {
            &self.value
        }
    }
    impl Drop for Text {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    let inputs: Vec<_> = (0..150)
        .map(|_| Text {
            value: "café hello worlds".into(),
            drops: drops.clone(),
            _marker: std::cell::Cell::new(()),
        })
        .collect();
    let actual = test_embedder(Output::Embedding(Pooling::Mean), true);
    let result = actual.embed_passages_stream(inputs, |_| Err("disk full"));
    assert!(matches!(result, Err(StreamError::Sink("disk full"))));
    assert_eq!(drops.load(Ordering::Relaxed), 150);
    let expected = test_embedder(Output::Embedding(Pooling::Mean), true)
        .embed_passages(&["café hello worlds"])
        .unwrap();
    let reused = actual.embed_passages(&["café hello worlds"]).unwrap();
    assert_same_bits(&reused[0], &expected[0]);
}

#[test]
fn streaming_matches_collected_outputs_and_preserves_order() {
    let embedder = embedder(Output::Embedding(Pooling::Mean));
    let texts: Vec<&str> = (0..150).map(|i| ["a", "b c", "a b c"][i % 3]).collect();
    let expected = embedder.embed_passages(&texts).unwrap();
    let mut rows = Vec::new();
    let count = embedder
        .embed_passages_stream(texts.iter().copied(), |batch| {
            assert_eq!(batch.start, rows.len());
            assert_eq!(batch.dim, 8);
            rows.extend(batch.values.chunks_exact(batch.dim).map(<[f32]>::to_vec));
            Ok::<_, std::io::Error>(())
        })
        .unwrap();
    assert_eq!(count, texts.len());
    assert_eq!(rows.len(), expected.len());
    for (actual, expected) in rows.iter().zip(&expected) {
        assert_same_bits(actual, expected);
    }
    let mut called = false;
    assert_eq!(
        embedder
            .embed_queries_stream(std::iter::empty::<&str>(), |_| {
                called = true;
                Ok::<_, std::io::Error>(())
            })
            .unwrap(),
        0
    );
    assert!(!called);
}

#[test]
fn finite_embeddings_match_streaming_across_chunk_boundaries_and_prefixes() {
    for bert_inputs in [false, true] {
        for prefixes in [("", ""), ("b ", "a ")] {
            let mut actual = test_embedder(Output::Embedding(Pooling::Mean), bert_inputs);
            actual.passage_prefix = prefixes.0.into();
            actual.query_prefix = prefixes.1.into();
            let reference = test_embedder(Output::Embedding(Pooling::Mean), bert_inputs);
            for query in [false, true] {
                let prefix = if query { prefixes.1 } else { prefixes.0 };
                for count in [0, 1, CHUNK, CHUNK + 1, 150] {
                    let texts: Vec<&str> = (0..count)
                        .map(|index| ["a", "b c", "c a b", "café a"][index % 4])
                        .collect();
                    let prefixed: Vec<String> =
                        texts.iter().map(|text| format!("{prefix}{text}")).collect();
                    let mut expected = Vec::new();
                    let mut batches = 0;
                    let completed = reference
                        .embed_passages_stream(prefixed.iter(), |batch| {
                            assert_eq!(batch.start, expected.len());
                            batches += 1;
                            expected
                                .extend(batch.values.chunks_exact(batch.dim).map(<[f32]>::to_vec));
                            Ok::<_, ()>(())
                        })
                        .unwrap();
                    assert_eq!(completed, count);
                    if count >= CHUNK {
                        assert!(batches > 1, "call must split");
                    }
                    let collected = if query {
                        actual.embed_queries(&texts)
                    } else {
                        actual.embed_passages(&texts)
                    }
                    .unwrap();
                    assert_eq!(collected.len(), expected.len());
                    for (row, expected) in collected.iter().zip(&expected) {
                        assert_same_bits(row, expected);
                    }
                }
            }
        }
    }
}

#[test]
fn finite_invalid_sequences_reject_and_allow_reuse_on_both_dispatch_paths() {
    let expected = embedder(Output::Embedding(Pooling::Mean))
        .embed_passages(&["b c"])
        .unwrap();
    for count in [1, CHUNK, CHUNK + 1] {
        for (invalid, reason) in [("".to_owned(), "no tokens"), ("a ".repeat(9), "exceeds")] {
            let actual = embedder(Output::Embedding(Pooling::Mean));
            let mut texts = vec!["a"; count];
            texts[count - 1] = &invalid;
            let error = actual.embed_passages(&texts).unwrap_err();
            assert!(matches!(error, Error::Tokenizer(ref text) if text.contains(reason)));
            let reused = actual.embed_passages(&["b c"]).unwrap();
            assert_same_bits(&reused[0], &expected[0]);
        }
    }
}

#[test]
fn invalid_encodings_reject_and_allow_reuse_on_both_dispatch_paths() {
    fn invalid_chunk(kind: usize) -> tokenizers::Result<Vec<tokenizing::InferenceInput>> {
        use tokenizing::InferenceInput::Compact;
        let invalid = match kind {
            0 => Compact {
                ids: vec![],
                types: vec![],
            },
            1 => Compact {
                ids: vec![1; 9],
                types: vec![0; 9],
            },
            2 => Compact {
                ids: vec![u32::MAX],
                types: vec![0],
            },
            3 => Compact {
                ids: vec![1],
                types: vec![u32::MAX],
            },
            _ => return Err(std::io::Error::other("tokenizer failure").into()),
        };
        Ok(vec![
            Compact {
                ids: vec![1],
                types: vec![0],
            },
            invalid,
        ])
    }

    let expected = embedder(Output::Embedding(Pooling::Mean))
        .embed_queries(&["c a"])
        .unwrap();
    // One chunk is tokenized inline; more start the producer thread.
    for count in [1, CHUNK + 1] {
        let actual = embedder(Output::Embedding(Pooling::Mean));
        for (kind, reason) in [
            "no tokens",
            "exceeds",
            "signed 32-bit",
            "signed 32-bit",
            "tokenizer failure",
        ]
        .into_iter()
        .enumerate()
        {
            let mut called = false;
            let sink = |_: Batch<'_>| {
                called = true;
                Ok::<_, ()>(())
            };
            let result = actual.model.run_tokenized(
                vec![(); count],
                |_, _, _| invalid_chunk(kind),
                || false,
                sink,
            );
            assert!(matches!(
                result,
                Err(StreamError::Inference(Error::Tokenizer(ref text))) if text.contains(reason)
            ));
            assert!(!called);
            let reused = actual.embed_queries(&["c a"]).unwrap();
            assert_same_bits(&reused[0], &expected[0]);
        }
    }
}

#[test]
fn sink_failure_bounds_lookahead_and_leaves_the_model_reusable() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let produced = AtomicUsize::new(0);
    let inputs = (0..10_000).map(|_| {
        produced.fetch_add(1, Ordering::Relaxed);
        "a b c"
    });
    let embedder = embedder(Output::Embedding(Pooling::Mean));
    let result = embedder.embed_passages_stream(inputs, |_| Err("disk full"));
    assert!(matches!(result, Err(StreamError::Sink("disk full"))));
    // One consumed chunk, two queued chunks, one blocked producer chunk.
    assert!(produced.load(Ordering::Relaxed) <= CHUNK * 4);
    assert_eq!(embedder.embed_passages(&["a"]).unwrap().len(), 1);
}

#[test]
fn streamed_reranking_matches_collected_scores() {
    let reranker = Reranker {
        model: embedder(Output::Relevance(Head::Pooler)).model,
    };
    let texts = ["a", "b c", "c", "a b"];
    let expected = reranker.score("a", &texts).unwrap();
    let mut scores = Vec::new();
    let count = reranker
        .score_stream("a", texts, |batch| {
            assert_eq!(batch.start, scores.len());
            assert_eq!(batch.dim, 1);
            scores.extend_from_slice(batch.values);
            Ok::<_, std::io::Error>(())
        })
        .unwrap();
    assert_eq!(count, texts.len());
    assert_same_bits(&scores, &expected);
}

#[test]
fn oversized_tokenizer_outputs_return_errors_and_stop() {
    let embedder = embedder(Output::Embedding(Pooling::Mean));
    let result = embedder.embed_passages_stream(["a ".repeat(9)], |_| Ok::<_, ()>(()));
    assert!(matches!(
        result,
        Err(StreamError::Inference(Error::Tokenizer(_)))
    ));
}

#[test]
fn zero_token_streams_stop_before_the_sink_and_allow_reuse() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    for text in ["", " \t\r\n"] {
        let produced = AtomicUsize::new(0);
        let inputs = std::iter::repeat(text).inspect(|_| {
            let count = produced.fetch_add(1, Ordering::Relaxed) + 1;
            assert!(
                count <= CHUNK * 4,
                "zero-token lookahead exceeded its bound"
            );
        });
        // This tokenizer has no special-token postprocessor.
        let actual = embedder(Output::Embedding(Pooling::Mean));
        let mut called = false;
        let result = actual.embed_passages_stream(inputs, |_| {
            called = true;
            Ok::<_, ()>(())
        });
        assert!(matches!(
            result,
            Err(StreamError::Inference(Error::Tokenizer(ref reason)))
                if reason.contains("no tokens")
        ));
        assert!(!called);
        assert!(produced.load(Ordering::Relaxed) <= CHUNK * 4);
        let expected = embedder(Output::Embedding(Pooling::Mean))
            .embed_passages(&["a"])
            .unwrap();
        let reused = actual.embed_passages(&["a"]).unwrap();
        assert_same_bits(&reused[0], &expected[0]);
    }
}

#[test]
fn zero_token_pairs_reject_and_allow_reranker_reuse() {
    let actual = Reranker {
        model: embedder(Output::Relevance(Head::Pooler)).model,
    };
    let mut called = false;
    let result = actual.score_stream("", [" \t\r\n"], |_| {
        called = true;
        Ok::<_, ()>(())
    });
    assert!(matches!(
        result,
        Err(StreamError::Inference(Error::Tokenizer(ref reason)))
            if reason.contains("no tokens")
    ));
    assert!(!called);
    let reference = Reranker {
        model: embedder(Output::Relevance(Head::Pooler)).model,
    };
    assert_same_bits(
        &actual.score("", &["a"]).unwrap(),
        &reference.score("", &["a"]).unwrap(),
    );
}

#[test]
fn empty_text_with_special_tokens_remains_valid() {
    let actual = test_embedder(Output::Embedding(Pooling::Mean), true);
    let expected = actual.embed_queries(&[""]).unwrap();
    let mut values = Vec::new();
    let count = actual
        .embed_queries_stream([""], |batch| {
            values.extend_from_slice(batch.values);
            Ok::<_, ()>(())
        })
        .unwrap();
    assert_eq!(count, 1);
    assert_same_bits(&values, &expected[0]);
}

#[test]
fn settings_that_cannot_run_fail_before_anything_downloads() {
    // An empty, offline cache: reaching the files would be a download error.
    let cache = TempDir::new("settings-first");
    let hub = Hub::new(cache.path()).offline(true);
    let too_many = rayon::max_num_threads().checked_add(1).unwrap();
    for builder in [
        Embedder::builder()
            .device(Device::Vulkan(0))
            .precision(Precision::Quantized),
        Embedder::builder()
            .device(Device::Cpu)
            .workers(explicit_workers(too_many, 1)),
    ] {
        let result = builder
            .build_from_hub(&hub, EmbeddingModel::AllMiniLmL6V2)
            .map(|_| ());
        assert!(
            matches!(result, Err(Error::Unsupported(_))),
            "{builder:?}: {result:?}"
        );
    }
    let result = Reranker::builder()
        .device(Device::Cuda(0))
        .workers(explicit_workers(1, 1))
        .build_from_hub(&hub, RerankerModel::MsMarcoMiniLmL6V2)
        .map(|_| ());
    assert!(matches!(result, Err(Error::Unsupported(_))), "{result:?}");
    // e5-large-v2 has e5-large's settings; only its pinned files reject the quantized preset.
    for &model in EmbeddingModel::ALL
        .iter()
        .filter(|&&model| model != EmbeddingModel::E5Large)
    {
        let result = Embedder::builder()
            .device(Device::Cpu)
            .precision(Precision::Quantized)
            .build_from_hub(&hub, model)
            .map(|_| ());
        assert!(
            matches!(&result, Err(Error::Unsupported(reason)) if reason.contains("quantized precision")),
            "{model:?}: {result:?}"
        );
    }
}

#[test]
fn batches_report_the_tokens_they_encoded() {
    // Three words are three tokens, and a test encoder batch holds eight.
    let embedder = embedder(Output::Embedding(Pooling::Mean));
    let mut tokens = Vec::new();
    embedder
        .embed_passages_stream(["a b c"; 4], |batch| {
            tokens.push(batch.tokens);
            Ok::<_, std::convert::Infallible>(())
        })
        .unwrap();
    assert_eq!(tokens, [6, 6]);
}

#[test]
fn pinned_files_are_checked_before_anything_parses_them() {
    use std::fs;
    let fixture = TinyModel::new(4);
    let dir = fixture.dir.path();
    let words = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab([("[UNK]".into(), 0), ("a".into(), 1)].into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    fixture.write_tokenizer(Tokenizer::new(words).to_string(false).unwrap());
    // Too short for a safetensors header, so a load that reaches it stops there.
    fs::write(dir.join("model.safetensors"), b"weights").unwrap();
    let pins = MODEL_FILES
        .map(|name| identity::hex(&Sha256::digest(fs::read(dir.join(name)).unwrap()).into()));
    let load = |pinned: [&str; 3]| {
        Model::load(Load {
            artifacts: directory(dir, Some(pinned)),
            task: Task::Embed(Pooling::Mean),
            max_length: 8,
            prefixes: ("", ""),
            device: Device::Cpu,
            batch_tokens: 16,
            selection: Selection {
                cpu: CpuPrecision::Fp32,
                workers: None,
            },
        })
        .map(|_| ())
    };
    let matching = pins.each_ref().map(String::as_str);
    let Err(Error::Model(reason)) = load(matching) else {
        panic!("matching pins must reach the weights");
    };
    assert!(
        reason.contains("shorter than its header length"),
        "{reason}"
    );
    let other = "0".repeat(64);
    for (index, name) in MODEL_FILES.into_iter().enumerate() {
        let mut pinned = matching;
        pinned[index] = &other;
        let reason = match load(pinned) {
            Err(Error::Tokenizer(reason)) if index == 1 => reason,
            Err(Error::Model(reason)) if index != 1 => reason,
            result => panic!("{name}: {result:?}"),
        };
        assert!(
            reason.contains(name) && reason.contains(&pins[index]) && reason.contains(&other),
            "{reason}"
        );
    }
}

#[test]
fn models_by_name_check_their_cached_files_against_the_registry() {
    let cache = TempDir::new("registry-cache");
    let hub = Hub::new(cache.path()).offline(true);
    for entry in [
        EmbeddingModel::AllMiniLmL6V2.entry(),
        RerankerModel::MsMarcoMiniLmL6V2.entry(),
    ] {
        let snapshot = hub.snapshot(entry);
        std::fs::create_dir_all(&snapshot).unwrap();
        // Zeros of each pinned size: the files are found but cannot match.
        for file in &entry.files {
            let created = std::fs::File::create(snapshot.join(file.name)).unwrap();
            created.set_len(file.size).unwrap();
        }
    }
    let results = [
        Embedder::builder()
            .device(Device::Cpu)
            .build_from_hub(&hub, EmbeddingModel::AllMiniLmL6V2)
            .map(|_| ()),
        Reranker::builder()
            .device(Device::Cpu)
            .build_from_hub(&hub, RerankerModel::MsMarcoMiniLmL6V2)
            .map(|_| ()),
        EmbeddingModel::AllMiniLmL6V2.download(&hub).map(|_| ()),
        RerankerModel::MsMarcoMiniLmL6V2.download(&hub).map(|_| ()),
    ];
    for result in results {
        let Err(Error::Model(reason)) = result else {
            panic!("zeros must fail the pinned hash: {result:?}");
        };
        assert!(
            reason.contains("config.json") && reason.contains("SHA-256"),
            "{reason}"
        );
    }
}

/// The reference tests read the shared cache; this one starts from an empty cache.
#[test]
#[ignore = "downloads about 180 MB from huggingface.co"]
fn models_download_into_an_empty_cache_and_then_load_offline() {
    let cache = TempDir::new("registry-download");
    let (embedder, reranker) = (
        EmbeddingModel::AllMiniLmL6V2,
        RerankerModel::MsMarcoMiniLmL6V2,
    );
    let hub = Hub::new(cache.path());
    embedder.download(&hub).unwrap();
    reranker.download(&hub).unwrap();
    let offline = Hub::new(cache.path()).offline(true);
    let loaded = Embedder::builder()
        .device(Device::Cpu)
        .build_from_hub(&offline, embedder)
        .unwrap();
    let vectors = loaded.embed_queries(&["capital of Germany"]).unwrap();
    assert_eq!(vectors[0].len(), embedder.dim());
    let scorer = Reranker::builder()
        .device(Device::Cpu)
        .build_from_hub(&offline, reranker)
        .unwrap();
    let passages = ["Berlin is the capital of Germany.", "Bananas are yellow."];
    let scores = scorer.score("capital of Germany", &passages).unwrap();
    assert!(scores[0] > scores[1], "{scores:?}");
}

#[test]
#[ignore = "uses the Hugging Face cache and the network; CUDA when present"]
fn quantized_vectors_do_not_depend_on_the_device_auto_picks() {
    let model = EmbeddingModel::E5Large;
    let quantized = Embedder::builder().precision(Precision::Quantized);
    let cpu = quantized.device(Device::Cpu).build(model).unwrap();
    let auto = quantized.build(model).unwrap();
    eprintln!(
        "Device::Auto runs Precision::Quantized on {}",
        auto.device_name()
    );
    assert_eq!(cpu.fingerprint(), auto.fingerprint());
    let texts = [
        "Berlin is the capital of Germany.",
        "a",
        "Bananas are yellow and rich in potassium.",
    ];
    let expected = cpu.embed_passages(&texts).unwrap();
    for (actual, expected) in auto.embed_passages(&texts).unwrap().iter().zip(&expected) {
        assert_same_bits(actual, expected);
    }
}

/// The manifest stays out of the fingerprint: `cargo publish` rewrites it.
#[test]
fn fingerprint_inputs_exclude_the_manifest() {
    let manifest: &[u8] = include_bytes!("../Cargo.toml");
    assert!(!SOURCES.contains(&manifest));
}

/// Every stored quantized vector depends on the drawn queries, so their recipe is pinned.
#[test]
fn calibration_ends_with_the_same_drawn_queries_inside_the_query_frame() {
    use tokenizers::{
        AddedToken, models::wordlevel::WordLevel, pre_tokenizers::bert::BertPreTokenizer,
        processors::template::TemplateProcessing,
    };

    let words = ["[UNK]", "[CLS]", "[SEP]", "query", ":", "a", "b", "c"];
    let model = WordLevel::builder()
        .vocab(
            words
                .iter()
                .enumerate()
                .map(|(id, word)| ((*word).to_owned(), id as u32))
                .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .unwrap()
            .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")
            .unwrap()
            .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
            .build()
            .unwrap(),
    ));
    tokenizer.add_special_tokens(
        &["[UNK]", "[CLS]", "[SEP]"].map(|token| AddedToken::from(token, true)),
    );

    let prefixes = ("passage: ", "query: ");
    let inputs = calibration_inputs(&tokenizer, prefixes).unwrap();
    let texts = 2 * CALIBRATION_PASSAGES.lines().count();
    assert_eq!(inputs.len(), texts + DRAWN_QUERIES);
    let drawn = &inputs[texts..];
    let mut by_pieces = [0usize; 3];
    for (ids, types) in drawn {
        // [CLS] query : pieces [SEP]; rows 0 to 2 are the special tokens.
        assert_eq!(ids[..3], [1, 3, 4], "{ids:?}");
        assert_eq!(ids[ids.len() - 1], 2, "{ids:?}");
        let pieces = &ids[3..ids.len() - 1];
        assert!((1..=3).contains(&pieces.len()), "{ids:?}");
        assert!(pieces.iter().all(|id| (3..8).contains(id)), "{ids:?}");
        assert_eq!(types, &vec![0; ids.len()]);
        by_pieces[pieces.len() - 1] += 1;
    }
    assert_eq!(by_pieces, [351, 343, 330]);
    assert_eq!(drawn[0].0, [1, 3, 4, 7, 2]);
    assert_eq!(drawn[1].0, [1, 3, 4, 5, 3, 2]);
    assert_eq!(drawn[2].0, [1, 3, 4, 3, 7, 3, 2]);
    assert_eq!(drawn[DRAWN_QUERIES - 1].0, [1, 3, 4, 4, 5, 6, 2]);
    assert_eq!(calibration_inputs(&tokenizer, prefixes).unwrap(), inputs);
}

/// What a program inside a JavaScript engine counts on: models from bytes, on the calling
/// thread. `wasm-bindgen-test-runner` runs these alone; the tests above need threads or files.
#[cfg(target_family = "wasm")]
mod wasm {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    const SPEC: ModelSpec<'static> = ModelSpec {
        passage_prefix: "",
        query_prefix: "",
        max_length: 8,
        pooling: Pooling::Mean,
    };

    fn shape_and_tensors<'a>(
        config: &[u8],
        weights: &'a [u8],
    ) -> (Shape, checkpoint::SafeTensors<'a>) {
        (
            checkpoint::shape_from_bytes(config).unwrap(),
            checkpoint::SafeTensors::from_bytes(Cow::Borrowed(weights)).unwrap(),
        )
    }

    /// BERT's layers and ModernBERT's, on the device the engine picks.
    #[wasm_bindgen_test]
    fn embedders_from_bytes_match_the_reference() {
        let (texts, ids, lengths) = (["a b", "c b a c c a"], [1, 2, 3, 2, 1, 3, 3, 1], [2, 6]);
        for [config, tokenizer, weights] in [tiny_model_files(false), tiny_modernbert_files("")] {
            let embedder = Embedder::builder()
                .batch_tokens(16)
                .build_from_bytes(&config, &tokenizer, &weights, SPEC)
                .unwrap();
            let device = embedder.device_name();
            assert!(
                device.starts_with("CPU") && device.contains("1 threads"),
                "{device}"
            );
            let (shape, tensors) = shape_and_tensors(&config, &weights);
            let expected = kodamaembed_model::reference::encode(
                &shape,
                &tensors,
                Pooling::Mean,
                &ids,
                &lengths,
            )
            .unwrap();
            let vectors = embedder.embed_passages(&texts).unwrap();
            assert_eq!(vectors.len(), expected.len());
            for (vector, expected) in vectors.iter().zip(&expected) {
                let error = vector
                    .iter()
                    .zip(expected)
                    .map(|(&have, want)| (f64::from(have) - want).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!(error <= 1e-6, "{:?}: L2 error {error:e}", shape.positions);
            }
        }
    }

    /// BERT's pooler and ModernBERT's prediction head.
    #[wasm_bindgen_test]
    fn rerankers_from_bytes_match_the_reference() {
        let passages = ["b", "c a", "a b c"];
        let mut modern = tiny_modernbert_config(6);
        modern["classifier_pooling"] = "mean".into();
        let framed = [
            serde_json::to_vec(&modern).unwrap(),
            framed_tokenizer().to_string(false).unwrap().into_bytes(),
            tiny_modernbert_cross_encoder(&modern),
        ];
        let cases: [(_, Head, &[i32], &[usize]); 2] = [
            // The word tokenizer frames nothing: a pair is the query's word, then the passage's.
            (
                tiny_model_files(true),
                Head::Pooler,
                &[1, 2, 1, 3, 1, 1, 1, 2, 3],
                &[2, 3, 4],
            ),
            (
                framed,
                Head::Prediction(Pooling::Mean, Activation::Gelu),
                &[4, 1, 5, 2, 5, 4, 1, 5, 3, 1, 5, 4, 1, 5, 1, 2, 3, 5],
                &[5, 6, 7],
            ),
        ];
        for ([config, tokenizer, weights], head, ids, lengths) in cases {
            let reranker = Reranker::builder()
                .batch_tokens(16)
                .build_from_bytes(&config, &tokenizer, &weights, 8)
                .unwrap();
            assert!(reranker.device_name().starts_with("CPU"));
            let (shape, tensors) = shape_and_tensors(&config, &weights);
            // The second segment starts after the query's one word where the tokenizer marks it.
            let types: Vec<i32> = match head {
                Head::Pooler => lengths
                    .iter()
                    .flat_map(|&len| std::iter::once(0).chain(std::iter::repeat_n(1, len - 1)))
                    .collect(),
                Head::Prediction(..) => vec![0; ids.len()],
            };
            let expected =
                kodamaembed_model::reference::rerank(&shape, &tensors, head, ids, &types, lengths)
                    .unwrap();
            let scores = reranker.score("a", &passages).unwrap();
            assert_eq!(scores.len(), expected.len());
            for (&score, expected) in scores.iter().zip(&expected) {
                assert!(
                    (f64::from(score) - expected).abs() <= 1e-6 * (1.0 + expected.abs()),
                    "{head:?}: {score} vs {expected}"
                );
            }
        }
    }

    /// With no thread to tokenize ahead, chunks tokenize as inference asks for them.
    #[wasm_bindgen_test]
    fn long_runs_keep_each_texts_bits_and_stop_when_asked() {
        let [config, tokenizer, weights] = tiny_model_files(false);
        let embedder = Embedder::builder()
            .batch_tokens(16)
            .build_from_bytes(&config, &tokenizer, &weights, SPEC)
            .unwrap();
        let kinds = ["a b", "c", "b a c"];
        let alone = embedder.embed_passages(&kinds).unwrap();
        let texts: Vec<&str> = kinds.into_iter().cycle().take(CHUNK * 3).collect();
        let together = embedder.embed_passages(&texts).unwrap();
        assert_eq!(together.len(), texts.len());
        for (row, vector) in together.iter().enumerate() {
            assert_same_bits(vector, &alone[row % kinds.len()]);
        }
        let (mut asked, mut rows) = (0, 0);
        let outcome = embedder.embed_passages_until(
            texts.iter(),
            || {
                asked += 1;
                asked > 1
            },
            |batch| {
                rows += batch.values.len() / batch.dim;
                Ok::<_, std::convert::Infallible>(())
            },
        );
        assert!(matches!(
            outcome,
            Err(StreamError::Inference(Error::Interrupted))
        ));
        assert!(rows > 0 && rows < texts.len(), "{rows} rows");
    }

    /// No GPU driver, clock, second thread or file system: each is an error with a reason.
    #[wasm_bindgen_test]
    fn what_the_target_lacks_is_refused_without_a_panic() {
        let [config, tokenizer, weights] = tiny_model_files(false);
        let builder = Embedder::builder().batch_tokens(16);
        let load = |builder: EmbedderBuilder| {
            builder.build_from_bytes(&config, &tokenizer, &weights, SPEC)
        };
        for device in [Device::Cuda(0), Device::Vulkan(0)] {
            assert!(load(builder.device(device)).is_err(), "{device:?}");
        }
        let workers = |count: usize| CpuWorkers {
            inference_workers: NonZeroUsize::new(count).unwrap(),
            tokenizer_workers: NonZeroUsize::new(count).unwrap(),
        };
        let mut embedder = load(builder.workers(workers(1))).unwrap();
        assert!(load(builder.workers(workers(4))).is_err());
        assert!(matches!(
            embedder.enable_profiling(),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(embedder.embed_passages(&["a b"]).unwrap().len(), 1);
        assert!(Embedder::new(EmbeddingModel::AllMiniLmL6V2).is_err());
    }
}
