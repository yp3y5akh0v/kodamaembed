//! Query latency through the public API, tokenization included: the median of 51 three-call means.
//! Usage: bench_query_api MODEL_DIR minilm|e5|modernbert-embed cpu-fp32|cpu-centered|cuda-fp16|cuda-centered [SETTLE_SECONDS] [WARM_SECONDS] [OUT_CSV]

use kodamaembed::{Device, Embedder, Precision};
use std::{
    error::Error,
    io::Write,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[path = "support/length_sequence.rs"]
mod length_sequence;
#[path = "support/matched_query.rs"]
mod matched_query;

/// 16 tokens with e5's `query: ` prefix and the special tokens.
const QUERY: &str = "Where did Melanie go hiking with her two kids last summer?";
const SAMPLES: usize = 51;
const RUNS: usize = 3;
const BATCH_TOKENS: usize = 16_384;
const USAGE: &str = "usage: bench_query_api MODEL_DIR minilm|e5|modernbert-embed MODE [SETTLE_SECONDS] [WARM_SECONDS] [OUT_CSV]";

fn unix_ms() -> Result<f64, std::time::SystemTimeError> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64() * 1e3)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|value| value == "--fixture") {
        return matched_query::run(&args);
    }
    if args.first().is_some_and(|value| value == "--sequence") {
        return length_sequence::run(&args);
    }
    let [dir, family, mode, timing @ ..] = args.as_slice() else {
        return Err(USAGE.into());
    };
    if timing.len() > 3 {
        return Err(USAGE.into());
    }
    let seconds = |at: usize| -> Result<Duration, Box<dyn Error>> {
        let value = timing.get(at).map_or("0", String::as_str).parse()?;
        Ok(Duration::try_from_secs_f64(value)?)
    };
    let (settle, warm) = (seconds(0)?, seconds(1)?);
    let spec = matched_query::family_model(family)?.spec();
    let (device, precision) = match mode.as_str() {
        "cpu-fp32" => (Device::Cpu, Precision::Float),
        "cpu-centered" => (Device::Cpu, Precision::Quantized),
        "cuda-fp16" => (Device::Cuda(0), Precision::Float),
        "cuda-centered" => (Device::Cuda(0), Precision::Quantized),
        _ => return Err("unknown mode".into()),
    };
    let embedder = Embedder::builder()
        .device(device)
        .precision(precision)
        .batch_tokens(BATCH_TOKENS)
        .build_from_dir(dir, spec)?;
    std::thread::sleep(settle);
    let started = Instant::now();
    let mut warmed = 0;
    while warmed < 20 || started.elapsed() < warm {
        embedder.embed_queries(&[QUERY])?;
        warmed += 1;
    }
    let mut samples = Vec::with_capacity(SAMPLES);
    let mut intervals = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start_unix_ms = unix_ms()?;
        let started = Instant::now();
        for _ in 0..RUNS {
            embedder.embed_queries(&[QUERY])?;
        }
        samples.push(started.elapsed().as_secs_f64() * 1e3 / RUNS as f64);
        intervals.push((start_unix_ms, unix_ms()?));
    }
    if let Some(path) = timing.get(2) {
        let mut file = std::fs::File::create_new(path)?;
        writeln!(file, "sample,latency_ms,start_unix_ms,end_unix_ms")?;
        for (index, (latency, (start, end))) in samples.iter().zip(intervals).enumerate() {
            writeln!(file, "{index},{latency:.9},{start:.6},{end:.6}")?;
        }
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "{family} {mode} ({}): one query median {:.3} ms (min {:.3}, max {:.3})",
        embedder.device_name(),
        samples[SAMPLES / 2],
        samples[0],
        samples[SAMPLES - 1]
    );
    println!("fingerprint: {}", embedder.fingerprint());
    Ok(())
}
