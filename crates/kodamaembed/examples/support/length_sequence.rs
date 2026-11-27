use crate::matched_query::{
    Result, embed, family_model, file_hash, requested_workers, role_prefix, same_bits, unix_ms,
    validate_vectors,
};
use kodamaembed::{Device, Embedder, Precision};
use rustc_hash::FxHashMap;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tokenizers::{Tokenizer, TruncationParams};

const USAGE: &str = "bench_query_api --sequence FILE --model-dir DIR --mode MODE --out JSON [--settle SECONDS] [--threads N]";
/// The first pass meets every shape on a fresh embedder; the later ones repeat it.
const PASSES: usize = 3;

fn options(args: &[String]) -> Result<FxHashMap<String, String>> {
    let mut values = FxHashMap::default();
    for pair in args.chunks(2) {
        let [name, value] = pair else {
            return Err(USAGE.into());
        };
        if !matches!(
            name.as_str(),
            "--sequence" | "--model-dir" | "--mode" | "--out" | "--settle" | "--threads"
        ) || values.insert(name.clone(), value.clone()).is_some()
        {
            return Err(USAGE.into());
        }
    }
    for required in ["--sequence", "--model-dir", "--mode", "--out"] {
        if !values.contains_key(required) {
            return Err(USAGE.into());
        }
    }
    Ok(values)
}

/// The call order: every text of the fixture once.
fn call_order(fixture: &Value, texts: usize) -> Result<Vec<usize>> {
    let order: Vec<usize> = serde_json::from_value(fixture["order"].clone())?;
    let mut seen = vec![false; texts];
    for &index in &order {
        if index >= texts || std::mem::replace(&mut seen[index], true) {
            return Err("fixture order must name every text once".into());
        }
    }
    if order.len() != texts {
        return Err("fixture order must name every text once".into());
    }
    Ok(order)
}

pub fn run(args: &[String]) -> Result<()> {
    let options = options(args)?;
    let fixture_path = Path::new(&options["--sequence"]);
    let model_dir = Path::new(&options["--model-dir"]);
    let output = PathBuf::from(&options["--out"]);
    let vector_path = output.with_extension("vectors.f32");
    if output.exists() || vector_path.exists() {
        return Err("output artifacts already exist".into());
    }
    let settle = Duration::try_from_secs_f64(
        options
            .get("--settle")
            .map_or("0", String::as_str)
            .parse()?,
    )?;
    let fixture: Value = serde_json::from_slice(&fs::read(fixture_path)?)?;
    let family = fixture["family"]
        .as_str()
        .ok_or("fixture requires family")?;
    let role = fixture["role"].as_str().ok_or("fixture requires role")?;
    if !matches!(role, "query" | "passage") || fixture["label"].as_str().is_none_or(str::is_empty) {
        return Err("fixture requires query|passage role and nonempty label".into());
    }
    let texts: Vec<String> = serde_json::from_value(fixture["texts"].clone())?;
    let model = family_model(family)?;
    let (spec, dim) = (model.spec(), model.dim());
    let maximum = spec.max_length;
    let expected_ids: Vec<Vec<u32>> = serde_json::from_value(fixture["expected_ids"].clone())?;
    let expected_types: Vec<Vec<u32>> = serde_json::from_value(fixture["expected_types"].clone())?;
    let lengths: Vec<usize> = serde_json::from_value(fixture["expected_lengths"].clone())?;
    if texts.is_empty()
        || expected_ids.len() != texts.len()
        || expected_types.len() != texts.len()
        || lengths.len() != texts.len()
    {
        return Err("fixture token row counts differ".into());
    }
    let order = call_order(&fixture, texts.len())?;
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
    let (device, precision) = match mode.as_str() {
        "cuda-centered" => (Device::Cuda(0), Precision::Quantized),
        "cuda-fp16" => (Device::Cuda(0), Precision::Float),
        "cpu-fp32" => (Device::Cpu, Precision::Float),
        "cpu-centered" => (Device::Cpu, Precision::Quantized),
        _ => {
            return Err(
                "sequence mode must be cuda-centered|cuda-fp16|cpu-fp32|cpu-centered".into(),
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
    let loading = Instant::now();
    let model = builder.build_from_dir(model_dir, spec)?;
    let load_seconds = loading.elapsed().as_secs_f64();
    std::thread::sleep(settle);
    let mut reference: Vec<Vec<f32>> = vec![Vec::new(); texts.len()];
    let mut passes = Vec::with_capacity(PASSES);
    let mut intervals = Vec::with_capacity(PASSES);
    let (mut norm_min, mut norm_max) = (f64::INFINITY, 0.0_f64);
    for pass in 0..PASSES {
        let mut samples = Vec::with_capacity(order.len());
        let mut results = Vec::with_capacity(order.len());
        let pass_start = unix_ms()?;
        for &index in &order {
            let text = [texts[index].as_str()];
            let start = Instant::now();
            let result = embed(&model, &text, role)?;
            samples.push(start.elapsed().as_secs_f64() * 1e3);
            results.push(result);
        }
        intervals.push([pass_start, unix_ms()?]);
        for (&index, result) in order.iter().zip(results) {
            let (low, high) = validate_vectors(&result, 1, dim)?;
            (norm_min, norm_max) = (norm_min.min(low), norm_max.max(high));
            if pass == 0 {
                reference[index] = result.into_iter().next().ok_or("missing output row")?;
            } else if !same_bits(&result, std::slice::from_ref(&reference[index])) {
                return Err("a later pass changed an embedding's bits".into());
            }
        }
        passes.push(samples);
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
    let receipt = json!({
        "schema_version": 1,
        "status": "complete",
        "passes_ms": passes,
        "pass_intervals_unix_ms": intervals,
        "timing_start_unix_ms": intervals[0][0],
        "timing_end_unix_ms": intervals[PASSES - 1][1],
        "metadata": {
            "engine": "kodamaembed", "mode": mode, "scope": "end-to-end",
            "scope_description": "fresh text tokenization through a normalized FP32 host vector, one text a call",
            "family": family, "role": role, "label": fixture["label"],
            "lengths": lengths, "calls_per_pass": order.len(), "passes": PASSES,
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
            "load_seconds": load_seconds, "settle_seconds": settle.as_secs_f64(), "warm_calls": 0,
            "caller_retention": "a pass's returned outputs retained until its clock stops; validation and release outside timing",
            "output_norm_min": norm_min, "output_norm_max": norm_max
        },
        "vectors": {
            "path": vector_path.file_name().ok_or("invalid vector path")?.to_string_lossy(),
            "rows": texts.len(), "dim": dim, "format": "row-major little-endian f32, in fixture text order",
            "sha256": file_hash(&vector_path)?
        }
    });
    serde_json::to_writer_pretty(File::create_new(&output)?, &receipt)?;
    println!("{}", receipt["status"]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_reject_missing_duplicate_unpaired_and_unknown_arguments() {
        let base = [
            "--sequence",
            "sequence.json",
            "--model-dir",
            "model",
            "--mode",
            "cuda-centered",
            "--out",
            "out.json",
        ];
        let args: Vec<_> = base.into_iter().map(str::to_owned).collect();
        assert!(options(&args).is_ok());
        let mut settled = args.clone();
        settled.extend(["--settle".into(), "1.5".into()]);
        assert!(options(&settled).is_ok());
        let mut duplicate = args.clone();
        duplicate.extend(["--mode".into(), "cpu-fp32".into()]);
        assert!(options(&duplicate).is_err());
        let mut unpaired = args.clone();
        unpaired.push("--threads".into());
        assert!(options(&unpaired).is_err());
        let mut warmed = args.clone();
        warmed.extend(["--warm".into(), "2".into()]);
        assert!(options(&warmed).is_err());
        assert!(options(&args[2..]).is_err());
    }

    #[test]
    fn the_call_order_names_every_text_once() {
        let fixture = |order: &[usize]| json!({ "order": order });
        assert_eq!(call_order(&fixture(&[2, 0, 1]), 3).unwrap(), [2, 0, 1]);
        assert!(call_order(&fixture(&[0, 1]), 3).is_err());
        assert!(call_order(&fixture(&[0, 1, 1]), 3).is_err());
        assert!(call_order(&fixture(&[0, 1, 3]), 3).is_err());
        assert!(call_order(&json!({}), 3).is_err());
    }
}
