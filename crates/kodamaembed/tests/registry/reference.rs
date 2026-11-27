//! Registered models against sentence-transformers' FP32 output (`scripts/reference_vectors.py`).

use kodamaembed::{Device, Embedder, EmbeddingModel, Pooling, Reranker, RerankerModel};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    query: String,
    passages: Vec<String>,
    embedders: Vec<EmbedderCase>,
    rerankers: Vec<RerankerCase>,
}

#[derive(Deserialize)]
struct EmbedderCase {
    repository: String,
    revision: String,
    pooling: String,
    max_length: usize,
    query_prefix: String,
    passage_prefix: String,
    query_vectors: Vec<Vec<f32>>,
    passage_vectors: Vec<Vec<f32>>,
}

#[derive(Deserialize)]
struct RerankerCase {
    repository: String,
    revision: String,
    max_length: usize,
    scores: Vec<f32>,
}

fn fixture() -> Fixture {
    serde_json::from_str(include_str!("vectors.json")).expect("vectors.json parses")
}

fn pooling(name: &str) -> Pooling {
    match name {
        "mean" => Pooling::Mean,
        "cls" => Pooling::Cls,
        other => panic!("unknown pooling {other}"),
    }
}

fn distance(actual: &[f32], expected: &[f32]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn the_reference_covers_every_registered_model_with_its_settings() {
    let fixture = fixture();
    let embedders: Vec<EmbeddingModel> = fixture
        .embedders
        .iter()
        .map(|case| case.repository.parse().unwrap())
        .collect();
    assert_eq!(embedders, EmbeddingModel::ALL);
    for (case, &model) in fixture.embedders.iter().zip(EmbeddingModel::ALL) {
        let spec = model.spec();
        assert_eq!(case.revision, model.revision(), "{model:?}");
        assert_eq!(pooling(&case.pooling), spec.pooling, "{model:?}");
        assert_eq!(case.max_length, spec.max_length, "{model:?}");
        assert_eq!(case.query_prefix, spec.query_prefix, "{model:?}");
        assert_eq!(case.passage_prefix, spec.passage_prefix, "{model:?}");
        assert_eq!(case.query_vectors.len(), 1, "{model:?}");
        assert_eq!(
            case.passage_vectors.len(),
            fixture.passages.len(),
            "{model:?}"
        );
        for vector in case.query_vectors.iter().chain(&case.passage_vectors) {
            assert_eq!(vector.len(), model.dim(), "{model:?}");
        }
    }
    let rerankers: Vec<RerankerModel> = fixture
        .rerankers
        .iter()
        .map(|case| case.repository.parse().unwrap())
        .collect();
    assert_eq!(rerankers, RerankerModel::ALL);
    for (case, &model) in fixture.rerankers.iter().zip(RerankerModel::ALL) {
        assert_eq!(case.revision, model.revision(), "{model:?}");
        assert_eq!(case.max_length, model.max_length(), "{model:?}");
        assert_eq!(case.scores.len(), fixture.passages.len(), "{model:?}");
    }
}

/// Each model's largest error on `device`: embedders' L2 distances, then rerankers' score errors.
fn errors(device: Device) -> [Vec<(String, f64)>; 2] {
    let fixture = fixture();
    let passages: Vec<&str> = fixture.passages.iter().map(String::as_str).collect();
    let (mut vectors, mut scored) = (Vec::new(), Vec::new());
    for (case, &model) in fixture.embedders.iter().zip(EmbeddingModel::ALL) {
        let embedder = Embedder::builder().device(device).build(model).unwrap();
        let query_vectors = embedder.embed_queries(&[&fixture.query]).unwrap();
        let passage_vectors = embedder.embed_passages(&passages).unwrap();
        let actual = query_vectors.iter().chain(&passage_vectors);
        let expected = case.query_vectors.iter().chain(&case.passage_vectors);
        let worst = actual
            .zip(expected)
            .map(|(actual, expected)| distance(actual, expected))
            .fold(0.0, f64::max);
        vectors.push((case.repository.clone(), worst));
    }
    for (case, &model) in fixture.rerankers.iter().zip(RerankerModel::ALL) {
        let reranker = Reranker::builder().device(device).build(model).unwrap();
        let scores = reranker.score(&fixture.query, &passages).unwrap();
        let worst = scores
            .iter()
            .zip(&case.scores)
            .map(|(&actual, &expected)| (f64::from(actual) - f64::from(expected)).abs())
            .fold(0.0, f64::max);
        scored.push((case.repository.clone(), worst));
    }
    [vectors, scored]
}

/// About twice the largest errors measured. Processor (AVX2): 4.2e-6 for vectors and 1.04e-5 for
/// scores, on a pair whose score in double precision sentence-transformers' FP32 misses by 4.2e-6
/// the other way. GPUs: 2.2e-3 (CUDA) and 2.5e-3 (Vulkan), both for scores.
const PROCESSOR_VECTORS: f64 = 1e-5;
const PROCESSOR_SCORES: f64 = 2e-5;
const GPU_TOLERANCE: f64 = 5e-3;

fn assert_within(device: Device, tolerances: [f64; 2]) {
    let errors = errors(device);
    // With --nocapture, the margins the tolerances above rest on.
    for (repository, error) in errors.iter().flatten() {
        eprintln!("{device:?} {repository}: {error:.2e}");
    }
    for (kind, tolerance) in errors.iter().zip(tolerances) {
        assert!(
            kind.iter().all(|(_, error)| *error <= tolerance),
            "{device:?} exceeds {tolerance}: {kind:#?}"
        );
    }
}

#[test]
#[ignore = "downloads every registered model into the Hugging Face cache"]
fn every_registered_model_matches_sentence_transformers_on_the_processor() {
    assert_within(Device::Cpu, [PROCESSOR_VECTORS, PROCESSOR_SCORES]);
}

#[test]
#[ignore = "needs CUDA and every registered model"]
fn every_registered_model_matches_sentence_transformers_on_cuda() {
    assert_within(Device::Cuda(0), [GPU_TOLERANCE; 2]);
}

#[test]
#[ignore = "needs a discrete Vulkan GPU and every registered model"]
fn every_registered_model_matches_sentence_transformers_on_vulkan() {
    assert_within(Device::first_discrete_vulkan().unwrap(), [GPU_TOLERANCE; 2]);
}
