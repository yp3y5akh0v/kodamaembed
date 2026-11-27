//! e5-large's FP16 and centered INT8 encoders, alternating on a 16,384-token batch; each batch's
//! CSV row carries its Unix-time window. Usage: bench_cuda_encode [ROUNDS] [IDLE_MS]

#[cfg(target_arch = "x86_64")]
use kodamaembed_cpu::{CpuPrecision, Encoder as CpuEncoder, Kernel};
#[cfg(target_arch = "x86_64")]
use kodamaembed_cuda::{Context, Encoder, repro::ReproEncoder};
#[cfg(target_arch = "x86_64")]
use kodamaembed_model::{Output, Pooling, Positions, Shape, reference::Synthetic};
#[cfg(target_arch = "x86_64")]
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[cfg(target_arch = "x86_64")]
const TOKENS: usize = 16_384;
#[cfg(target_arch = "x86_64")]
const LENGTHS: [usize; 8] = [24, 48, 96, 160, 216, 320, 448, 512];

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
fn unix_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64() * 1e3)
}

#[cfg(not(target_arch = "x86_64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("centered CUDA benchmarks require an x86_64 host".into())
}

#[cfg(target_arch = "x86_64")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let rounds: usize = args.next().map_or(Ok(40), |arg| arg.parse())?;
    let idle = Duration::from_millis(args.next().map_or(Ok(0), |arg| arg.parse())?);
    let context = Context::new(0)?;
    let weights = Synthetic::new(&E5, 7);
    let mut lengths = Vec::new();
    let mut tokens = 0;
    for &len in LENGTHS.iter().cycle() {
        if tokens + len > TOKENS {
            break;
        }
        lengths.push(len);
        tokens += len;
    }
    let ids: Vec<i32> = (0..tokens).map(|i| (i * 7_919 % E5.vocab) as i32).collect();
    let types = vec![0; tokens];
    let (free_before, _) = context.memory()?;
    let mut fp16 = Encoder::new(
        &context,
        E5,
        Output::Embedding(Pooling::Mean),
        TOKENS,
        &weights,
    )?;
    let (free_after, _) = context.memory()?;
    eprintln!(
        "FP16: {:.1} MiB of device memory for {TOKENS} tokens",
        free_before.saturating_sub(free_after) as f64 / (1u64 << 20) as f64
    );
    let mut cpu = CpuEncoder::new_with_precision(
        Kernel::Avx2,
        E5,
        Output::Embedding(Pooling::Mean),
        TOKENS,
        &weights,
        CpuPrecision::E5CenteredInt8,
    )?;
    cpu.calibrate(&[
        (vec![101, 102], vec![0, 0]),
        (vec![101, 2000, 1024, 102], vec![0; 4]),
    ])?;
    let mut centered = ReproEncoder::new(&context, &cpu.export_reproducible()?)?;
    let mut encode = |mode: usize| -> Result<(), Box<dyn std::error::Error>> {
        let _ = match mode {
            0 => fp16.encode(&ids, &types, &lengths)?,
            _ => centered.encode(&ids, &types, &lengths)?,
        };
        Ok(())
    };
    // Untimed first: the centered encoder allocates on its first encode.
    encode(0)?;
    encode(1)?;
    eprintln!(
        "{} e5-large: {} sequences, {tokens} tokens per batch, {rounds} rounds",
        context.name,
        lengths.len()
    );
    println!("round,mode,start_ms,end_ms,batch_ms");
    for round in 0..rounds {
        std::thread::sleep(idle);
        encode(round % 2)?;
        for step in 0..2 {
            let mode = (round + step) % 2;
            let start = unix_ms();
            let started = Instant::now();
            encode(mode)?;
            let batch_ms = started.elapsed().as_secs_f64() * 1e3;
            let end = unix_ms();
            let name = if mode == 0 { "fp16" } else { "centered" };
            println!("{round},{name},{start:.1},{end:.1},{batch_ms:.3}");
        }
    }
    Ok(())
}
