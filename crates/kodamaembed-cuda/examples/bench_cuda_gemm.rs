//! GEMM throughput on e5-large's four projection shapes at 16,384 tokens.
//! Inputs are random: tensor-core power depends on the bits that toggle.

use half::f16;
use kodamaembed_cuda::{
    Context, DeviceBuffer, Stream,
    kernels::{Gemm, Kernels},
};
use kodamaembed_model::Geometry;
use std::time::Instant;

fn random(len: usize, seed: u64, scale: f32) -> Vec<f16> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
            f16::from_f32((unit * 2.0 - 1.0) * scale)
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = Context::new(0)?;
    let (major, minor) = context.compute_capability;
    println!(
        "{} ({} SMs, sm_{major}{minor})",
        context.name, context.sm_count
    );
    let kernels = Kernels::load(&context, Geometry::Large)?;
    let stream = Stream::new(&context)?;
    let m = 16_384;
    let mut total_flops = 0.0;
    let mut total_seconds = 0.0;
    for (name, n, k) in [
        ("qkv", 3072, 1024),
        ("out", 1024, 1024),
        ("ffn1", 4096, 1024),
        ("ffn2", 1024, 4096),
    ] {
        let a = DeviceBuffer::from_slice(&context, &random(m * k, 1, 1.0))?;
        let w = DeviceBuffer::from_slice(&context, &random(n * k, 2, 0.05))?;
        let bias = DeviceBuffer::<f32>::zeroed(&context, n)?;
        let out = DeviceBuffer::<f16>::zeroed(&context, m * n)?;
        let gemm = Gemm {
            a: &a,
            w: &w,
            bias: &bias,
            m,
            n,
            k,
        };
        // SAFETY: the buffers outlive every launch; the stream is synchronized below.
        unsafe { kernels.gemm_bias(&stream, &gemm, &out, None)? };
        stream.synchronize()?;
        let iterations = 20;
        let started = Instant::now();
        for _ in 0..iterations {
            // SAFETY: as above.
            unsafe { kernels.gemm_bias(&stream, &gemm, &out, None)? };
        }
        stream.synchronize()?;
        let seconds = started.elapsed().as_secs_f64() / f64::from(iterations);
        let flops = 2.0 * (m * n * k) as f64;
        total_flops += flops;
        total_seconds += seconds;
        println!(
            "{name:>5} {m}x{n}x{k}: {:.3} ms, {:.1} TFLOPS",
            seconds * 1e3,
            flops / seconds / 1e12
        );
    }
    println!(
        "layer mix: {:.1} TFLOPS",
        total_flops / total_seconds / 1e12
    );
    Ok(())
}
