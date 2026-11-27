//! e5-large's projection GEMMs at 16,384 tokens, on cooperative matrices where present.
//! Usage: bench_vulkan_gemm [ADAPTER, default the first discrete GPU]

use half::f16;
use kodamaembed_vulkan::{
    Buffer, Descriptors, Device, Kernel, KernelSpec, Push, Recorder, shaders,
};
use std::time::Instant;

const M: usize = 16_384;
const ITERATIONS: usize = 10;
/// The matrix GEMM places four subgroups of this size per workgroup.
const MATRIX_SUBGROUP: u32 = 32;

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
    let index = match std::env::args().nth(1) {
        Some(index) => index.parse()?,
        None => {
            Device::adapters()?
                .into_iter()
                .find(|adapter| adapter.discrete)
                .ok_or("no discrete Vulkan GPU")?
                .index
        }
    };
    let device = Device::new(index)?;
    let matrix = device.cooperative_matrix && device.subgroup_size == MATRIX_SUBGROUP;
    println!(
        "{} ({})",
        device.name,
        if matrix {
            "cooperative matrices"
        } else {
            "portable shaders"
        }
    );
    let constants = [0u32];
    let kernel = unsafe {
        Kernel::new(
            &device,
            &KernelSpec {
                spirv: if matrix {
                    shaders::GEMM_MATRIX
                } else {
                    shaders::GEMM_PORTABLE
                },
                bindings: 6,
                push_bytes: 12,
                constants: &constants,
                subgroup_size: matrix.then_some(MATRIX_SUBGROUP),
            },
        )
    }?;
    let (mut total_flops, mut total_seconds) = (0.0, 0.0);
    for (name, n, k) in [
        ("qkv", 3072, 1024),
        ("out", 1024, 1024),
        ("ffn1", 4096, 1024),
        ("ffn2", 1024, 4096),
    ] {
        let a = Buffer::from_slice(&device, &random(M * k, 1, 1.0))?;
        let w = Buffer::from_slice(&device, &random(n * k, 2, 0.05))?;
        let bias = Buffer::<f32>::zeroed(&device, n)?;
        let out16 = Buffer::<f16>::zeroed(&device, M * n)?;
        let unused = Buffer::<f32>::zeroed(&device, 1)?;
        let descriptors = Descriptors::new(&device, 1, 6)?;
        let set = unsafe {
            descriptors.bind(
                &kernel,
                &[
                    a.binding(),
                    w.binding(),
                    bias.binding(),
                    unused.binding(),
                    out16.binding(),
                    unused.binding(),
                ],
            )
        }?;
        let push = Push::new().u32(M as u32).u32(n as u32).u32(k as u32);
        let tile = if matrix { 128 } else { 64 };
        let groups = [(n / tile) as u32, M.div_ceil(tile) as u32, 1];
        let run = |count: usize| {
            device.submit(|command| unsafe {
                let recorder = Recorder::new(&device, command);
                for _ in 0..count {
                    recorder.dispatch(&kernel, set, &push, groups);
                    recorder.barrier();
                }
                Ok(())
            })
        };
        run(1)?;
        let started = Instant::now();
        run(ITERATIONS)?;
        let seconds = started.elapsed().as_secs_f64() / ITERATIONS as f64;
        let flops = 2.0 * (M * n * k) as f64;
        total_flops += flops;
        total_seconds += seconds;
        println!(
            "{name:>5} {M}x{n}x{k}: {:.3} ms, {:.1} TFLOPS",
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
