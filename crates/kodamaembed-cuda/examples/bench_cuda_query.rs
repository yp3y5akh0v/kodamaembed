//! e5-large's single-query latency on CUDA device 0: the median of 51 three-encode means.
//! Usage: bench_cuda_query [all|reproducible] [TOKENS] [SETTLE_SECONDS] [WARM_SECONDS] [OUT_CSV]

#[cfg(target_arch = "x86_64")]
use kodamaembed_cpu::{CpuPrecision, Encoder as CpuEncoder, Kernel};
#[cfg(target_arch = "x86_64")]
use kodamaembed_cuda::{Context, Encoder, repro::ReproEncoder};
#[cfg(target_arch = "x86_64")]
use kodamaembed_model::{Output, Pooling, Positions, Shape, reference::Synthetic};
#[cfg(target_arch = "x86_64")]
use std::{
    io::Write,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(target_arch = "x86_64")]
const SAMPLES: usize = 51;
#[cfg(target_arch = "x86_64")]
const RUNS: usize = 3;

#[cfg(target_arch = "x86_64")]
const E5: Shape = Shape {
    layers: 24,
    hidden: 1024,
    heads: 16,
    intermediate: 4096,
    vocab: 30_522,
    max_positions: 512,
    type_vocab: 2,
    eps: 1e-12,
    positions: Positions::Sequential,
};

#[cfg(target_arch = "x86_64")]
struct Sample {
    latency_ms: f64,
    start_unix_ms: f64,
    end_unix_ms: f64,
}

#[cfg(target_arch = "x86_64")]
fn unix_ms() -> Result<f64, std::time::SystemTimeError> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64() * 1e3)
}

#[cfg(target_arch = "x86_64")]
fn time(
    warm: Duration,
    mut encode: impl FnMut() -> Result<(), Box<dyn std::error::Error>>,
) -> Result<Vec<Sample>, Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mut warmed = 0;
    while warmed < 20 || started.elapsed() < warm {
        encode()?;
        warmed += 1;
    }
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start_unix_ms = unix_ms()?;
        let started = Instant::now();
        for _ in 0..RUNS {
            encode()?;
        }
        let latency_ms = started.elapsed().as_secs_f64() * 1e3 / RUNS as f64;
        samples.push(Sample {
            latency_ms,
            start_unix_ms,
            end_unix_ms: unix_ms()?,
        });
    }
    Ok(samples)
}

#[cfg(not(target_arch = "x86_64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("centered CUDA benchmarks require an x86_64 host".into())
}

#[cfg(target_arch = "x86_64")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().len() > 6 {
        return Err("too many arguments".into());
    }
    let all = match std::env::args().nth(1).as_deref() {
        None | Some("all") => true,
        Some("reproducible") => false,
        Some(other) => return Err(format!("unknown encoders {other}").into()),
    };
    let tokens: usize = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "16".into())
        .parse()?;
    if tokens == 0 || tokens > E5.max_positions {
        return Err("TOKENS must be between 1 and the model's position limit".into());
    }
    let seconds = |at: usize| -> Result<Duration, Box<dyn std::error::Error>> {
        Ok(Duration::try_from_secs_f64(
            std::env::args()
                .nth(at)
                .unwrap_or_else(|| "0".into())
                .parse()?,
        )?)
    };
    let settle = seconds(3)?;
    let warm = seconds(4)?;
    let out_csv = std::env::args().nth(5);
    if all && out_csv.is_some() {
        return Err("OUT_CSV requires reproducible mode".into());
    }
    let context = Context::new(0)?;
    let weights = Synthetic::new(&E5, 7);
    let ids: Vec<i32> = (0..tokens).map(|i| (i * 7_919 % E5.vocab) as i32).collect();
    let types = vec![0; tokens];
    let report = |name: &str, samples: &[Sample]| {
        let mut values: Vec<f64> = samples.iter().map(|sample| sample.latency_ms).collect();
        values.sort_by(f64::total_cmp);
        let (median, min, max) = (values[SAMPLES / 2], values[0], values[SAMPLES - 1]);
        println!(
            "{} e5-large {name}: one {tokens}-token sequence median {median:.3} ms (min {min:.3}, max {max:.3})",
            context.name
        );
    };
    if all {
        let mut encoder = Encoder::new(
            &context,
            E5,
            Output::Embedding(Pooling::Mean),
            tokens,
            &weights,
        )?;
        std::thread::sleep(settle);
        let timing = time(warm, || {
            encoder.encode(&ids, &types, &[tokens])?;
            Ok(())
        })?;
        report("FP16", &timing);
    }
    // Calibrated on two synthetic sequences: timing needs no real centers.
    let mut cpu = CpuEncoder::new_with_precision(
        Kernel::Avx2,
        E5,
        Output::Embedding(Pooling::Mean),
        tokens,
        &weights,
        CpuPrecision::E5CenteredInt8,
    )?;
    cpu.calibrate(&[
        (vec![101, 102], vec![0, 0]),
        (vec![101, 2000, 1024, 102], vec![0; 4]),
    ])?;
    let mut repro = ReproEncoder::new(&context, &cpu.export_reproducible()?)?;
    std::thread::sleep(settle);
    let timing = time(warm, || {
        repro.encode(&ids, &types, &[tokens])?;
        Ok(())
    })?;
    report("reproducible", &timing);
    if let Some(path) = out_csv {
        let mut file = std::fs::File::create_new(path)?;
        writeln!(file, "sample,latency_ms,start_unix_ms,end_unix_ms")?;
        for (index, sample) in timing.iter().enumerate() {
            writeln!(
                file,
                "{index},{:.9},{:.6},{:.6}",
                sample.latency_ms, sample.start_unix_ms, sample.end_unix_ms
            )?;
        }
    }
    Ok(())
}
