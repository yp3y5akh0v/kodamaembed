//! Centered e5-large encoder latency by length on CUDA device 0, where a route boundary shows as
//! a step. Usage: bench_cuda_curve LENGTHS [SETTLE_SECONDS] [WARM_SECONDS]

#[cfg(target_arch = "x86_64")]
use kodamaembed_cpu::{CpuPrecision, Encoder as CpuEncoder, Kernel};
#[cfg(target_arch = "x86_64")]
use kodamaembed_cuda::{Context, repro::ReproEncoder};
#[cfg(target_arch = "x86_64")]
use kodamaembed_model::{Output, Pooling, Positions, Shape, reference::Synthetic};
#[cfg(target_arch = "x86_64")]
use std::time::{Duration, Instant};

#[cfg(target_arch = "x86_64")]
const WARMUP: usize = 10;
#[cfg(target_arch = "x86_64")]
const SAMPLES: usize = 15;
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

/// `1-128,136-512/8,300`: ranges with an optional step, and single lengths.
#[cfg(target_arch = "x86_64")]
fn lengths(arg: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    for part in arg.split(',') {
        let (range, step) = part.split_once('/').unwrap_or((part, "1"));
        let step: usize = step.parse()?;
        if step == 0 {
            return Err(format!("zero step in {part}").into());
        }
        match range.split_once('-') {
            Some((first, last)) => {
                let (first, last): (usize, usize) = (first.parse()?, last.parse()?);
                out.extend((first..=last).step_by(step));
            }
            None => out.push(range.parse()?),
        }
    }
    Ok(out)
}

#[cfg(not(target_arch = "x86_64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("centered CUDA benchmarks require an x86_64 host".into())
}

#[cfg(target_arch = "x86_64")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [list, timing @ ..] = args.as_slice() else {
        return Err("usage: bench_cuda_curve LENGTHS [SETTLE_SECONDS] [WARM_SECONDS]".into());
    };
    let lengths = lengths(list)?;
    if lengths.is_empty()
        || lengths
            .iter()
            .any(|&len| len == 0 || len > E5.max_positions)
    {
        return Err("every length must be between 1 and the model's position limit".into());
    }
    let seconds = |at: usize| -> Result<Duration, Box<dyn std::error::Error>> {
        Ok(Duration::try_from_secs_f64(
            timing.get(at).map_or("0", String::as_str).parse()?,
        )?)
    };
    let (settle, warm) = (seconds(0)?, seconds(1)?);
    let longest = lengths.iter().copied().max().unwrap_or(1);
    let context = Context::new(0)?;
    let weights = Synthetic::new(&E5, 7);
    let mut cpu = CpuEncoder::new_with_precision(
        Kernel::Avx2,
        E5,
        Output::Embedding(Pooling::Mean),
        longest,
        &weights,
        CpuPrecision::E5CenteredInt8,
    )?;
    cpu.calibrate(&[
        (vec![101, 102], vec![0, 0]),
        (vec![101, 2000, 1024, 102], vec![0; 4]),
    ])?;
    let mut repro = ReproEncoder::new(&context, &cpu.export_reproducible()?)?;
    std::thread::sleep(settle);
    let first = lengths[0];
    let ids: Vec<i32> = (0..first).map(|i| (i * 7_919 % E5.vocab) as i32).collect();
    let types = vec![0; first];
    let started = Instant::now();
    let mut warmed = 0;
    while warmed < 20 || started.elapsed() < warm {
        repro.encode(&ids, &types, &[first])?;
        warmed += 1;
    }
    println!(
        "{} e5-large reproducible: length, median ms, min, max",
        context.name
    );
    for len in lengths {
        let ids: Vec<i32> = (0..len).map(|i| (i * 7_919 % E5.vocab) as i32).collect();
        let types = vec![0; len];
        for _ in 0..WARMUP {
            repro.encode(&ids, &types, &[len])?;
        }
        let mut samples = Vec::with_capacity(SAMPLES);
        for _ in 0..SAMPLES {
            let started = Instant::now();
            for _ in 0..RUNS {
                repro.encode(&ids, &types, &[len])?;
            }
            samples.push(started.elapsed().as_secs_f64() * 1e3 / RUNS as f64);
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "{len} {:.3} {:.3} {:.3}",
            samples[SAMPLES / 2],
            samples[0],
            samples[SAMPLES - 1]
        );
    }
    Ok(())
}
