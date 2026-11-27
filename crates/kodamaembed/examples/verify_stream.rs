//! Checks that the collected and streaming APIs agree bit for bit on a real checkpoint.
//! Usage: verify_stream MODEL minilm|e5|rerank cpu|cpu-centered|cuda|cuda-centered|vulkan|auto
use kodamaembed::{Device, Embedder, EmbeddingModel, Precision, Reranker};
use std::{convert::Infallible, error::Error, time::Instant};

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

fn assert_vectors_same_bits(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_same_bits(actual, expected);
    }
}

fn requested_precision(family: &str, device: &str) -> Result<Precision, Box<dyn Error>> {
    if !matches!(device, "cpu-centered" | "cuda-centered") {
        return Ok(Precision::Float);
    }
    match family {
        "minilm" | "e5" => Ok(Precision::Quantized),
        "rerank" => {
            Err(format!("{device} supports embeddings only; reranking is unsupported").into())
        }
        _ => {
            Err(format!("{device} requires family minilm or e5 and its verified checkpoint").into())
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: verify_stream MODEL minilm|e5|rerank \
             cpu|cpu-centered|cuda|cuda-centered|vulkan|auto"
            .into());
    }
    let precision = requested_precision(&args[1], &args[2])?;
    let device = match args[2].as_str() {
        "cpu" | "cpu-centered" => Device::Cpu,
        "cuda" | "cuda-centered" => Device::Cuda(0),
        "vulkan" => Device::first_discrete_vulkan()?,
        "auto" => Device::Auto,
        _ => return Err("unknown device".into()),
    };
    // Enough inputs that a batch and a lone input take different kernel routes.
    let texts = [
        "Berlin is the capital of Germany.".to_owned(),
        String::new(),
        "Unicode: café 東京 🦀".to_owned(),
        "word ".repeat(700),
        "Berlin is the capital of Germany.".to_owned(),
        "Where did Melanie go hiking with her two kids last summer?".to_owned(),
        "The mitochondrion is the powerhouse of the cell.".to_owned(),
        "a".to_owned(),
        "Rust guarantees memory safety without a garbage collector.".to_owned(),
        "Paris".to_owned(),
        "How many moons does Jupiter have?".to_owned(),
        "The quick brown fox jumps over the lazy dog.".to_owned(),
        "Water boils at 100 degrees Celsius at sea level.".to_owned(),
        "query: what causes rainbows".to_owned(),
        "passage: Rainbows form when sunlight refracts in raindrops.".to_owned(),
        "1 2 3 4 5 6 7 8 9 10".to_owned(),
        "Ein Satz auf Deutsch.".to_owned(),
        "The meeting moved to Thursday at 3 pm.".to_owned(),
        "GPUs stream weights from memory for every query.".to_owned(),
        "Stockholm is the capital of Sweden.".to_owned(),
    ];
    let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
    let start = Instant::now();
    let (name, dim, count, fingerprint) = if args[1] == "rerank" {
        let model = Reranker::builder()
            .device(device)
            .batch_tokens(1024)
            .build_from_dir(&args[0], 512)?;
        let expected = model.score("capital of Germany", &refs)?;
        let mut got = Vec::new();
        let count = model.score_stream("capital of Germany", texts.iter(), |batch| {
            assert_eq!(batch.start, got.len());
            got.extend_from_slice(batch.values);
            Ok::<_, Infallible>(())
        })?;
        assert_same_bits(&got, &expected);
        for (text, expected) in refs.iter().zip(expected) {
            assert_same_bits(&model.score("capital of Germany", &[*text])?, &[expected]);
        }
        (
            model.device_name().to_owned(),
            1,
            count,
            model.fingerprint().to_owned(),
        )
    } else {
        let spec = match args[1].as_str() {
            "minilm" => EmbeddingModel::AllMiniLmL6V2.spec(),
            "e5" => EmbeddingModel::E5Large.spec(),
            _ => return Err("unknown family".into()),
        };
        let model = Embedder::builder()
            .device(device)
            .precision(precision)
            .batch_tokens(1024)
            .build_from_dir(&args[0], spec)?;
        let expected = model.embed_passages(&refs)?;
        let mut got = Vec::new();
        let count = model.embed_passages_stream(texts.iter(), |batch| {
            assert_eq!(batch.start, got.len());
            got.extend(batch.values.chunks_exact(batch.dim).map(<[f32]>::to_vec));
            Ok::<_, Infallible>(())
        })?;
        assert_vectors_same_bits(&got, &expected);
        assert_same_bits(&got[0], &got[4]);
        for (text, expected) in refs.iter().zip(expected) {
            assert_vectors_same_bits(&model.embed_passages(&[*text])?, &[expected]);
        }
        (
            model.device_name().to_owned(),
            model.dim(),
            count,
            model.fingerprint().to_owned(),
        )
    };
    assert_eq!(count, refs.len());
    println!(
        "{}",
        serde_json::json!({"device":name,"family":args[1],"requested_device":args[2],"dim":dim,
        "inputs":count,"fingerprint":fingerprint,"bit_identical":true,"load_and_checks_seconds":start.elapsed().as_secs_f64()})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantized_devices_select_matching_modes_and_reject_reranking() {
        for family in ["minilm", "e5"] {
            for device in ["cpu-centered", "cuda-centered"] {
                assert_eq!(
                    requested_precision(family, device).unwrap(),
                    Precision::Quantized
                );
            }
        }
        for device in ["cpu-centered", "cuda-centered"] {
            assert!(
                requested_precision("rerank", device)
                    .unwrap_err()
                    .to_string()
                    .contains("reranking is unsupported")
            );
            assert!(requested_precision("unknown", device).is_err());
        }
        for device in ["cpu", "cuda", "vulkan", "auto"] {
            assert_eq!(
                requested_precision("rerank", device).unwrap(),
                Precision::Float
            );
        }
    }
}
