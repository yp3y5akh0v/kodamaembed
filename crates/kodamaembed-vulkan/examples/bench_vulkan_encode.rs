//! Encoder time on synthetic weights for one 16-token sequence and a 16,384-token batch of mixed
//! lengths. Usage: bench_vulkan_encode [ADAPTER, default the first discrete GPU] [small|large]

use kodamaembed_model::{Output, Pooling, Positions, Shape, reference::Synthetic};
use kodamaembed_vulkan::{Device, Encoder, EncoderError};
use std::time::Instant;

const TOKENS: usize = 16_384;
const LENGTHS: [usize; 8] = [24, 48, 96, 160, 216, 320, 448, 512];
const ITERATIONS: usize = 20;

fn shape(layers: usize, hidden: usize, heads: usize) -> Shape {
    Shape {
        layers,
        hidden,
        heads,
        intermediate: 4 * hidden,
        vocab: 30_522,
        max_positions: 512,
        type_vocab: 2,
        eps: 1e-12,
        positions: Positions::Sequential,
    }
}

/// Mean milliseconds per `encode` of sequences with these lengths.
fn time(encoder: &mut Encoder, vocab: usize, lengths: &[usize]) -> Result<f64, EncoderError> {
    let tokens: usize = lengths.iter().sum();
    let ids: Vec<i32> = (0..tokens).map(|i| (i * 7_919 % vocab) as i32).collect();
    let types = vec![0; tokens];
    encoder.encode(&ids, &types, lengths)?;
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        encoder.encode(&ids, &types, lengths)?;
    }
    Ok(started.elapsed().as_secs_f64() * 1e3 / ITERATIONS as f64)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let index = match args.next() {
        Some(index) => index.parse()?,
        None => {
            Device::adapters()?
                .into_iter()
                .find(|adapter| adapter.discrete)
                .ok_or("no discrete Vulkan GPU")?
                .index
        }
    };
    let (label, shape) = match args.next().as_deref() {
        None | Some("small") => ("MiniLM", shape(6, 384, 12)),
        Some("large") => ("e5-large", shape(24, 1024, 16)),
        Some(other) => return Err(format!("unknown shape {other}").into()),
    };
    let device = Device::new(index)?;
    let weights = Synthetic::new(&shape, 7);
    let mut encoder = Encoder::new(
        &device,
        shape,
        Output::Embedding(Pooling::Mean),
        TOKENS,
        &weights,
    )?;
    let mut lengths = Vec::new();
    let mut tokens = 0;
    for &len in LENGTHS.iter().cycle() {
        if tokens + len > TOKENS {
            break;
        }
        lengths.push(len);
        tokens += len;
    }
    let single = time(&mut encoder, shape.vocab, &[16])?;
    let batch = time(&mut encoder, shape.vocab, &lengths)?;
    println!(
        "{} {label}: one 16-token sequence {single:.3} ms; {} sequences, {tokens} tokens {batch:.3} ms, {:.0} tokens/s",
        device.name,
        lengths.len(),
        tokens as f64 / (batch * 1e-3)
    );
    Ok(())
}
