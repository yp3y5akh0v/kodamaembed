//! Encoder time on synthetic weights for one 16-token sequence and a 16,384-token batch, with
//! stage profiles. Usage: bench_cpu_encode [small|large] [fp32|centered]

use kodamaembed_cpu::{CpuPrecision, Encoder, EncoderError, Kernel};
use kodamaembed_model::{Output, Pooling, Positions, STAGES, Shape, reference::Synthetic};
use std::time::Instant;

const TOKENS: usize = 16_384;
const LENGTHS: [usize; 8] = [24, 48, 96, 160, 216, 320, 448, 512];
const ITERATIONS: usize = 3;
const QUERIES: usize = 51;

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

fn load_encoder(
    kernel: Kernel,
    shape: Shape,
    max_tokens: usize,
    weights: &dyn kodamaembed_model::Tensors,
    precision: CpuPrecision,
) -> Result<Encoder, EncoderError> {
    let mut encoder = Encoder::new_with_precision(
        kernel,
        shape,
        Output::Embedding(Pooling::Mean),
        max_tokens,
        weights,
        precision,
    )?;
    if precision == CpuPrecision::E5CenteredInt8 {
        // Timing alone: one sequence of consecutive words calibrates.
        let words: Vec<i32> = [101].into_iter().chain(2000..2062).chain([102]).collect();
        let types = vec![0; words.len()];
        encoder.calibrate(&[(words, types)])?;
    }
    Ok(encoder)
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
    let (label, shape) = match std::env::args().nth(1).as_deref() {
        None | Some("small") => ("MiniLM", shape(6, 384, 12)),
        Some("large") => ("e5-large", shape(24, 1024, 16)),
        Some(other) => return Err(format!("unknown shape {other}").into()),
    };
    let large = shape.layers == 24;
    let precision = match std::env::args().nth(2).as_deref() {
        None | Some("fp32") => CpuPrecision::Fp32,
        Some("centered") if large => CpuPrecision::E5CenteredInt8,
        Some("centered") => return Err("the centered INT8 preset needs the large shape".into()),
        Some(other) => return Err(format!("unknown precision {other}").into()),
    };
    let kernel = Kernel::detect();
    let weights = Synthetic::new(&shape, 7);
    let mut encoder = load_encoder(kernel, shape, TOKENS, &weights, precision)?;
    let mut lengths = Vec::new();
    let mut tokens = 0;
    for &len in LENGTHS.iter().cycle() {
        if tokens + len > TOKENS {
            break;
        }
        lengths.push(len);
        tokens += len;
    }
    let mut queries = Vec::new();
    for _ in 0..QUERIES {
        queries.push(time(&mut encoder, shape.vocab, &[16])?);
    }
    queries.sort_by(f64::total_cmp);
    let single = queries[QUERIES / 2];
    let batch = time(&mut encoder, shape.vocab, &lengths)?;
    println!(
        "{kernel} {precision:?} x {} threads {label}: one 16-token sequence median {single:.3} ms (min {:.3}, max {:.3}); {} sequences, {tokens} tokens {batch:.1} ms, {:.0} tokens/s",
        encoder.threads(),
        queries[0],
        queries[QUERIES - 1],
        lengths.len(),
        tokens as f64 / (batch * 1e-3)
    );
    for (name, lengths) in [("16-token sequence", &[16][..]), ("batch", &lengths[..])] {
        encoder.enable_profiling();
        time(&mut encoder, shape.vocab, lengths)?;
        let profile = encoder.profile().ok_or("profiling is enabled")?;
        let stages: Vec<String> = STAGES
            .iter()
            .zip(profile.milliseconds)
            .map(|(stage, ms)| format!("{stage:?} {:.2}", ms / profile.batches as f64))
            .collect();
        println!("ms per {name}: {}", stages.join(", "));
    }
    Ok(())
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn centered_benchmark_is_calibrated_before_timing() {
        if Kernel::detect() != Kernel::Avx2
            || !std::is_x86_feature_detected!("avxvnni")
            || !std::is_x86_feature_detected!("f16c")
        {
            return;
        }
        let shape = shape(24, 1024, 16);
        let weights = Synthetic::new(&shape, 7);
        let encoder = load_encoder(
            Kernel::Avx2,
            shape,
            16,
            &weights,
            CpuPrecision::E5CenteredInt8,
        )
        .unwrap();
        assert!(encoder.export_reproducible().is_ok());
    }
}
