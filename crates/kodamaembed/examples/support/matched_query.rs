use kodamaembed::{CpuWorkers, Device, Embedder, EmbeddingModel, ModelSpec, Precision};
use rustc_hash::FxHashMap;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::{self, File},
    io::{Read, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokenizers::{Tokenizer, TruncationParams};

pub(crate) type Result<T> = std::result::Result<T, Box<dyn Error>>;

const USAGE: &str = "bench_query_api --fixture FILE --model-dir DIR --mode MODE --out JSON [--settle SECONDS] [--warm SECONDS] [--threads N] [--prepare-only]";

fn options(args: &[String]) -> Result<(FxHashMap<String, String>, bool)> {
    let mut values = FxHashMap::default();
    let mut prepare = false;
    let mut index = 0;
    while index < args.len() {
        let name = &args[index];
        if name == "--prepare-only" {
            if prepare {
                return Err("duplicate --prepare-only".into());
            }
            prepare = true;
            index += 1;
            continue;
        }
        if !matches!(
            name.as_str(),
            "--fixture" | "--model-dir" | "--mode" | "--out" | "--settle" | "--warm" | "--threads"
        ) || index + 1 >= args.len()
            || values
                .insert(name.clone(), args[index + 1].clone())
                .is_some()
        {
            return Err(USAGE.into());
        }
        index += 2;
    }
    for required in ["--fixture", "--model-dir", "--mode", "--out"] {
        if !values.contains_key(required) {
            return Err(USAGE.into());
        }
    }
    Ok((values, prepare))
}

/// Equal inference and tokenizer worker counts from `--threads`, which only the CPU modes take.
pub(crate) fn requested_workers(
    options: &FxHashMap<String, String>,
    mode: &str,
) -> Result<Option<CpuWorkers>> {
    let Some(value) = options.get("--threads") else {
        return Ok(None);
    };
    if !mode.starts_with("cpu-") {
        return Err("--threads applies to the CPU modes".into());
    }
    let count: NonZeroUsize = value.parse()?;
    Ok(Some(CpuWorkers {
        inference_workers: count,
        tokenizer_workers: count,
    }))
}

pub(crate) fn file_hash(path: &Path) -> Result<String> {
    let mut source = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let count = source.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

pub(crate) fn unix_ms() -> Result<f64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64() * 1e3)
}

pub(crate) fn validate_vectors(
    vectors: &[Vec<f32>],
    rows: usize,
    dim: usize,
) -> Result<(f64, f64)> {
    if vectors.len() != rows {
        return Err("output row count differs from fixture".into());
    }
    let mut minimum = f64::INFINITY;
    let mut maximum = 0.0_f64;
    for vector in vectors {
        if vector.len() != dim || vector.iter().any(|value| !value.is_finite()) {
            return Err("output width or finiteness check failed".into());
        }
        let norm = vector
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        if (norm - 1.0).abs() > 1e-3 {
            return Err("output is not L2 normalized".into());
        }
        minimum = minimum.min(norm);
        maximum = maximum.max(norm);
    }
    Ok((minimum, maximum))
}

pub(crate) fn same_bits(actual: &[Vec<f32>], expected: &[Vec<f32>]) -> bool {
    actual.len() == expected.len()
        && actual.iter().zip(expected).all(|(left, right)| {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
        })
}

/// The registered model whose text settings a benchmark family's checkpoint takes.
pub(crate) fn family_model(family: &str) -> Result<EmbeddingModel> {
    match family {
        "minilm" => Ok(EmbeddingModel::AllMiniLmL6V2),
        "e5" => Ok(EmbeddingModel::E5Large),
        "modernbert-embed" => Ok(EmbeddingModel::ModernBertEmbedBase),
        _ => Err("family must be minilm|e5|modernbert-embed".into()),
    }
}

/// The text a model's settings put before a text of `role`.
pub(crate) fn role_prefix(spec: &ModelSpec<'static>, role: &str) -> &'static str {
    if role == "query" {
        spec.query_prefix
    } else {
        spec.passage_prefix
    }
}

pub(crate) fn embed(model: &Embedder, texts: &[&str], role: &str) -> Result<Vec<Vec<f32>>> {
    Ok(if role == "query" {
        model.embed_queries(texts)?
    } else {
        model.embed_passages(texts)?
    })
}

pub fn run(args: &[String]) -> Result<()> {
    let (options, prepare_only) = options(args)?;
    let fixture_path = Path::new(&options["--fixture"]);
    let model_dir = Path::new(&options["--model-dir"]);
    let output = PathBuf::from(&options["--out"]);
    let vector_path = output.with_extension("vectors.f32");
    if output.exists() || vector_path.exists() {
        return Err("output artifacts already exist".into());
    }
    let duration = |key| -> Result<Duration> {
        Ok(Duration::try_from_secs_f64(
            options.get(key).map_or("0", String::as_str).parse()?,
        )?)
    };
    let (settle, warm) = (duration("--settle")?, duration("--warm")?);
    let fixture: Value = serde_json::from_slice(&fs::read(fixture_path)?)?;
    let family = fixture["family"]
        .as_str()
        .ok_or("fixture requires family")?;
    let role = fixture["role"].as_str().ok_or("fixture requires role")?;
    if !matches!(role, "query" | "passage") || fixture["label"].as_str().is_none_or(str::is_empty) {
        return Err("fixture requires query|passage role and nonempty label".into());
    }
    let owned_texts: Vec<String> = serde_json::from_value(fixture["texts"].clone())?;
    let texts: Vec<&str> = owned_texts.iter().map(String::as_str).collect();
    if texts.is_empty() {
        return Err("fixture texts must not be empty".into());
    }
    let model = family_model(family)?;
    let (spec, dim) = (model.spec(), model.dim());
    let maximum = spec.max_length;
    let expected_ids: Vec<Vec<u32>> = serde_json::from_value(fixture["expected_ids"].clone())?;
    let expected_types: Vec<Vec<u32>> = serde_json::from_value(fixture["expected_types"].clone())?;
    let lengths: Vec<usize> = serde_json::from_value(fixture["expected_lengths"].clone())?;
    if expected_ids.len() != texts.len()
        || expected_types.len() != texts.len()
        || lengths.len() != texts.len()
    {
        return Err("fixture token row counts differ".into());
    }
    let mut tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|error| error.to_string())?;
    tokenizer.with_padding(None);
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: maximum,
            ..Default::default()
        }))
        .map_err(|error| error.to_string())?;
    let prefix = role_prefix(&spec, role);
    let mut token_digest = Sha256::new();
    for (index, text) in texts.iter().enumerate() {
        let encoding = tokenizer
            .encode(format!("{prefix}{text}"), true)
            .map_err(|error| error.to_string())?;
        if encoding.get_ids() != expected_ids[index]
            || encoding.get_type_ids() != expected_types[index]
            || encoding.len() != lengths[index]
            || lengths[index] == 0
        {
            return Err("fixture IDs, types or lengths differ from tokenization".into());
        }
        token_digest.update((encoding.len() as u64).to_le_bytes());
        for value in encoding.get_ids().iter().chain(encoding.get_type_ids()) {
            token_digest.update(value.to_le_bytes());
        }
    }
    let mode = &options["--mode"];
    let loading = Instant::now();
    let (device, precision) = match mode.as_str() {
        "cuda-centered" => (Device::Cuda(0), Precision::Quantized),
        "cuda-fp16" => (Device::Cuda(0), Precision::Float),
        "cpu-fp32" => (Device::Cpu, Precision::Float),
        "cpu-centered" => (Device::Cpu, Precision::Quantized),
        _ => {
            return Err(
                "matched mode must be cuda-centered|cuda-fp16|cpu-fp32|cpu-centered".into(),
            );
        }
    };
    let workers = requested_workers(&options, mode)?;
    let mut builder = Embedder::builder()
        .device(device)
        .precision(precision)
        .batch_tokens(16_384);
    if let Some(workers) = workers {
        builder = builder.workers(workers);
    }
    let model = builder.build_from_dir(model_dir, spec)?;
    let load_seconds = loading.elapsed().as_secs_f64();
    let reference = embed(&model, &texts, role)?;
    let (norm_min, norm_max) = validate_vectors(&reference, texts.len(), dim)?;
    let mut receipt = json!({
        "schema_version": 1,
        "status": if prepare_only { "prepared" } else { "complete" },
        "metadata": {
            "engine": "kodamaembed", "mode": mode, "scope": "end-to-end",
            "scope_description": "fresh text tokenization through normalized FP32 host vectors",
            "family": family, "role": role, "label": fixture["label"],
            "shape": [texts.len(), lengths.iter().max().ok_or("missing length")?],
            "lengths": lengths, "real_tokens": lengths.iter().sum::<usize>(),
            "hidden": dim, "max_length": maximum, "batch_tokens": 16384,
            "fixture_sha256": file_hash(fixture_path)?,
            "input_ids_types_sha256": format!("{:x}", token_digest.finalize()),
            "token_audit": "reference tokenizer outside timing; production tokenizer parity is covered by library regression tests",
            "checkpoint_sha256": {
                "config.json": file_hash(&model_dir.join("config.json"))?,
                "tokenizer.json": file_hash(&model_dir.join("tokenizer.json"))?,
                "model.safetensors": file_hash(&model_dir.join("model.safetensors"))?
            },
            "fingerprint": model.fingerprint(), "device": model.device_name(),
            "threads": workers.map(|value| value.inference_workers.get()),
            "cpu_workers": model.cpu_workers().map(|value| json!({
                "inference": value.inference_workers.get(),
                "tokenizer": value.tokenizer_workers.get()
            })),
            "load_seconds": load_seconds, "settle_seconds": settle.as_secs_f64(),
            "warm_seconds": warm.as_secs_f64(), "calls_per_sample": 3,
            "caller_retention": "three returned outputs retained until the sample clock stops; validation and release outside timing",
            "output_norm_min": norm_min, "output_norm_max": norm_max
        }
    });
    if !prepare_only {
        std::thread::sleep(settle);
        let started = Instant::now();
        let mut warmed = 0;
        while warmed < 20 || started.elapsed() < warm {
            embed(&model, &texts, role)?;
            warmed += 1;
        }
        let mut samples = Vec::with_capacity(51);
        let mut intervals = Vec::with_capacity(51);
        let timing_start = unix_ms()?;
        for _ in 0..51 {
            let mut results: [Vec<Vec<f32>>; 3] = std::array::from_fn(|_| Vec::new());
            let sample_start = unix_ms()?;
            let start = Instant::now();
            for result in &mut results {
                *result = embed(&model, &texts, role)?;
            }
            samples.push(start.elapsed().as_secs_f64() * 1e3 / 3.0);
            intervals.push([sample_start, unix_ms()?]);
            for result in &results {
                validate_vectors(result, texts.len(), dim)?;
                if !same_bits(result, &reference) {
                    return Err("repeated embedding output bits changed".into());
                }
            }
        }
        receipt["timing_start_unix_ms"] = json!(timing_start);
        receipt["timing_end_unix_ms"] = json!(unix_ms()?);
        receipt["samples_ms"] = json!(samples);
        receipt["sample_intervals_unix_ms"] = json!(intervals);
        samples.sort_by(f64::total_cmp);
        receipt["median_ms"] = json!(samples[25]);
        receipt["min_ms"] = json!(samples[0]);
        receipt["max_ms"] = json!(samples[50]);
        receipt["metadata"]["warm_calls"] = json!(warmed);
    }
    if let Some(parent) = output
        .parent()
        .filter(|value| !value.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut vectors = File::create_new(&vector_path)?;
    for value in reference.iter().flatten() {
        vectors.write_all(&value.to_le_bytes())?;
    }
    drop(vectors);
    receipt["vectors"] = json!({
        "path": vector_path.file_name().ok_or("invalid vector path")?.to_string_lossy(),
        "rows": texts.len(), "dim": dim, "format": "row-major little-endian f32",
        "sha256": file_hash(&vector_path)?
    });
    serde_json::to_writer_pretty(File::create_new(&output)?, &receipt)?;
    println!("{}", receipt["status"]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_checks_reject_shape_nan_and_missing_normalization() {
        assert!(validate_vectors(&[vec![1.0, 0.0]], 1, 2).is_ok());
        assert!(validate_vectors(&[vec![1.0]], 1, 2).is_err());
        assert!(validate_vectors(&[vec![f32::NAN, 0.0]], 1, 2).is_err());
        assert!(validate_vectors(&[vec![0.5, 0.0]], 1, 2).is_err());
        assert!(!same_bits(&[vec![0.0]], &[vec![-0.0]]));
    }

    #[test]
    fn each_benchmark_family_names_its_model_and_takes_its_prefixes() {
        let prefixes = |family: &str| {
            let spec = family_model(family).unwrap().spec();
            (role_prefix(&spec, "query"), role_prefix(&spec, "passage"))
        };
        assert_eq!(
            family_model("minilm").unwrap(),
            EmbeddingModel::AllMiniLmL6V2
        );
        assert_eq!(prefixes("minilm"), ("", ""));
        assert_eq!(family_model("e5").unwrap(), EmbeddingModel::E5Large);
        assert_eq!(prefixes("e5"), ("query: ", "passage: "));
        assert_eq!(
            family_model("modernbert-embed").unwrap(),
            EmbeddingModel::ModernBertEmbedBase
        );
        assert_eq!(
            prefixes("modernbert-embed"),
            ("search_query: ", "search_document: ")
        );
        assert!(family_model("bert").is_err());
    }

    #[test]
    fn options_reject_missing_duplicate_and_unknown_arguments() {
        let base = [
            "--fixture",
            "fixture.json",
            "--model-dir",
            "model",
            "--mode",
            "cuda-centered",
            "--out",
            "out.json",
        ];
        let mut args: Vec<_> = base.into_iter().map(str::to_owned).collect();
        assert!(options(&args).is_ok());
        let mut threaded = args.clone();
        threaded.extend(["--threads".into(), "2".into()]);
        assert!(options(&threaded).is_ok());
        args.extend(["--mode".into(), "cpu-fp32".into()]);
        assert!(options(&args).is_err());
        assert!(options(&["--unknown".into()]).is_err());
    }

    #[test]
    fn threads_set_equal_worker_counts_on_the_cpu_modes_only() {
        let threads =
            |value: &str| FxHashMap::from_iter([("--threads".to_owned(), value.to_owned())]);
        let four = NonZeroUsize::new(4).expect("four is nonzero");
        assert_eq!(
            requested_workers(&threads("4"), "cpu-fp32").unwrap(),
            Some(CpuWorkers {
                inference_workers: four,
                tokenizer_workers: four
            })
        );
        assert_eq!(
            requested_workers(&FxHashMap::default(), "cuda-centered").unwrap(),
            None
        );
        assert!(requested_workers(&threads("4"), "cuda-centered").is_err());
        assert!(requested_workers(&threads("0"), "cpu-centered").is_err());
        assert!(requested_workers(&threads("four"), "cpu-fp32").is_err());
    }
}
