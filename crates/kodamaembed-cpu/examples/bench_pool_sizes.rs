//! Median encode time by input size on the given pools and the default ones. Usage: bench_pool_sizes
//! [small|large] [fp32|centered] [POOLS, e.g. 8,24] [SIZES, e.g. 2x128,4x64]
use kodamaembed_cpu::{CpuPrecision, Encoder, Kernel};
use kodamaembed_model::{Output, Pooling, Positions, Shape, reference::Synthetic};
use std::{num::NonZeroUsize, time::Instant};

const BLOCK: usize = 10;
/// (sequences, tokens per sequence).
const SIZES: [(usize, usize); 13] = [
    (1, 16),
    (1, 32),
    (1, 48),
    (1, 64),
    (1, 96),
    (1, 128),
    (1, 192),
    (1, 256),
    (1, 512),
    (2, 64),
    (4, 32),
    (4, 128),
    (16, 128),
];

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
    workers: Option<NonZeroUsize>,
) -> Result<Encoder, kodamaembed_cpu::EncoderError> {
    let mut encoder = match workers {
        Some(workers) => Encoder::new_with_precision_and_workers(
            kernel,
            shape,
            Output::Embedding(Pooling::Mean),
            max_tokens,
            weights,
            precision,
            workers,
        ),
        None => Encoder::new_with_precision(
            kernel,
            shape,
            Output::Embedding(Pooling::Mean),
            max_tokens,
            weights,
            precision,
        ),
    }?;
    if precision == CpuPrecision::E5CenteredInt8 {
        // Timing alone: one sequence of consecutive words calibrates.
        let words: Vec<i32> = [101].into_iter().chain(2000..2062).chain([102]).collect();
        let types = vec![0; words.len()];
        encoder.calibrate(&[(words, types)])?;
    }
    Ok(encoder)
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
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
    let pools = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "8,24".into())
        .split(',')
        .map(str::parse::<NonZeroUsize>)
        .collect::<Result<Vec<_>, _>>()?;
    let sizes: Vec<(usize, usize)> = match std::env::args().nth(4) {
        None => SIZES.to_vec(),
        Some(list) => list
            .split(',')
            .map(|size| {
                let (s, l) = size.split_once('x').ok_or("a size is SEQUENCESxLENGTH")?;
                Ok((s.parse()?, l.parse()?))
            })
            .collect::<Result<_, Box<dyn std::error::Error>>>()?,
    };
    let kernel = Kernel::detect();
    let weights = Synthetic::new(&shape, 7);
    let capacity = sizes.iter().map(|&(s, t)| s * t).max().unwrap_or(0);
    let mut encoders = Vec::new();
    for &workers in &pools {
        let encoder = load_encoder(kernel, shape, capacity, &weights, precision, Some(workers))?;
        encoders.push((format!("{workers} thr"), encoder));
    }
    let encoder = load_encoder(kernel, shape, capacity, &weights, precision, None)?;
    encoders.push(("default".into(), encoder));
    println!(
        "{label} {precision:?}: median ms of {} encodes; last column first pool / second",
        2 * BLOCK
    );
    let names: Vec<String> = encoders.iter().map(|(n, _)| format!("{n:>9}")).collect();
    println!("  sequences x length  tokens {}", names.join(""));
    let order: Vec<usize> = (0..encoders.len())
        .chain((0..encoders.len()).rev())
        .collect();
    for (sequences, length) in sizes {
        let lengths = vec![length; sequences];
        let tokens = sequences * length;
        let ids: Vec<i32> = (0..tokens).map(|i| (i * 7_919 % 30_522) as i32).collect();
        let types = vec![0; tokens];
        let mut samples = vec![Vec::new(); encoders.len()];
        for &e in &order {
            let encoder = &mut encoders[e].1;
            encoder.encode(&ids, &types, &lengths)?;
            for _ in 0..BLOCK {
                let started = Instant::now();
                encoder.encode(&ids, &types, &lengths)?;
                samples[e].push(started.elapsed().as_secs_f64() * 1e3);
            }
        }
        let medians: Vec<f64> = samples.iter_mut().map(|s| median(s)).collect();
        let cells: Vec<String> = medians.iter().map(|m| format!("{m:>9.2}")).collect();
        let ratio = if pools.len() >= 2 {
            format!("{:>8.3}", medians[0] / medians[1])
        } else {
            String::new()
        };
        println!(
            "  {sequences:>9} x {length:>6} {tokens:>7} {}{ratio}",
            cells.join("")
        );
    }
    Ok(())
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    #[test]
    fn centered_benchmarks_are_calibrated_with_shared_and_dedicated_pools() {
        if Kernel::detect() != Kernel::Avx2
            || !std::is_x86_feature_detected!("avxvnni")
            || !std::is_x86_feature_detected!("f16c")
        {
            return;
        }
        let shape = shape(24, 1024, 16);
        let weights = Synthetic::new(&shape, 7);
        for workers in [None, NonZeroUsize::new(1)] {
            let encoder = load_encoder(
                Kernel::Avx2,
                shape,
                16,
                &weights,
                CpuPrecision::E5CenteredInt8,
                workers,
            )
            .unwrap();
            assert!(encoder.export_reproducible().is_ok(), "workers {workers:?}");
        }
    }
}
