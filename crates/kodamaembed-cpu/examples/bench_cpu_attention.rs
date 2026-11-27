//! Synthetic attention timing with fixed sequence lengths and a 16,384-token
//! budget, in MiniLM's heads (`small`, default) or E5-large's (`large`).

use kodamaembed_cpu::{Kernel, attention};
use std::time::Instant;

const TOKENS: usize = 16_384;
const LENGTHS: [usize; 8] = [24, 48, 96, 160, 216, 320, 448, 512];
const ITERATIONS: u32 = 5;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (hidden, heads) = match std::env::args().nth(1).as_deref() {
        None | Some("small") => (384, 12),
        Some("large") => (1024, 16),
        Some(other) => return Err(format!("unknown shape {other}").into()),
    };
    let mut lengths = Vec::new();
    let mut tokens = 0;
    for &len in LENGTHS.iter().cycle() {
        if tokens + len > TOKENS {
            break;
        }
        lengths.push(len);
        tokens += len;
    }
    let mut state = 1u64;
    let qkv: Vec<f32> = (0..tokens * 3 * hidden)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    let mut context = vec![0.0; tokens * hidden];
    let kernel = Kernel::detect();
    attention(kernel, &qkv, &lengths, hidden, heads, None, &mut context);
    let started = Instant::now();
    for _ in 0..ITERATIONS {
        attention(kernel, &qkv, &lengths, hidden, heads, None, &mut context);
    }
    let seconds = started.elapsed().as_secs_f64() / f64::from(ITERATIONS);
    // Q K^T and P V: 2 L^2 d multiply-adds per head and sequence.
    let flops: f64 = lengths
        .iter()
        .map(|&len| 4.0 * (len * len * hidden) as f64)
        .sum();
    println!(
        "{kernel} hidden {hidden}, {heads} heads, {} sequences, {tokens} tokens: {:.2} ms, {:.0} GFLOPS",
        lengths.len(),
        seconds * 1e3,
        flops / seconds / 1e9
    );
    Ok(())
}
