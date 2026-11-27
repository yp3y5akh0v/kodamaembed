//! GEMM throughput at MiniLM's (`small`, default) or e5-large's (`large`)
//! projection shapes, 4,096 rows, with every kernel the processor runs.

use kodamaembed_cpu::{Epilogue, Kernel, PackedWeight, gemm};
use std::time::Instant;

fn random(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rows = std::env::args()
        .nth(2)
        .map(|arg| arg.parse::<usize>())
        .transpose()?
        .unwrap_or(4096);
    if rows == 0 {
        return Err("rows must be positive".into());
    }
    let iterations: u32 = if rows <= 96 { 200 } else { 5 };
    let shapes: [(usize, usize); 4] = match std::env::args().nth(1).as_deref() {
        None | Some("small") => [(1152, 384), (384, 384), (1536, 384), (384, 1536)],
        Some("large") => [(3072, 1024), (1024, 1024), (4096, 1024), (1024, 4096)],
        Some(other) => return Err(format!("unknown shape set {other}").into()),
    };
    let mut kernels = vec![Kernel::detect()];
    if kernels[0] != Kernel::Portable {
        kernels.push(Kernel::Portable);
    }
    for kernel in kernels {
        for (n, k) in shapes {
            let a = random(rows * k, 1);
            let w = PackedWeight::new(&random(n * k, 2), n, k);
            let bias = random(n, 3);
            let mut c = vec![0.0; rows * n];
            gemm(kernel, &a, rows, &w, Epilogue::Bias(&bias), &mut c);
            let started = Instant::now();
            for _ in 0..iterations {
                gemm(kernel, &a, rows, &w, Epilogue::Bias(&bias), &mut c);
            }
            let seconds = started.elapsed().as_secs_f64() / f64::from(iterations);
            let flops = 2.0 * (rows * n * k) as f64;
            println!(
                "{kernel} {rows}x{n}x{k}: {:.3} ms, {:.0} GFLOPS",
                seconds * 1e3,
                flops / seconds / 1e9
            );
        }
    }
    Ok(())
}
