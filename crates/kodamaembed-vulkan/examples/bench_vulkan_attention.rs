//! Attention throughput of the portable and cooperative-matrix shaders on a 16,384-token batch of
//! mixed lengths. Usage: bench_vulkan_attention [ADAPTER, default the first discrete GPU] [large|small]

use half::f16;
use kodamaembed_model::Geometry;
use kodamaembed_vulkan::{
    Buffer, Descriptors, Device, Kernel, KernelSpec, Push, Recorder, shaders,
};
use std::time::Instant;

const TOKENS: usize = 16_384;
const LENGTHS: [usize; 8] = [24, 48, 96, 160, 216, 320, 448, 512];
const ITERATIONS: usize = 10;

fn random(len: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = (state >> 40) as f32 / (1u64 << 24) as f32;
            (unit * 2.0 - 1.0) * scale
        })
        .collect()
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
    let geometry = match args.next().as_deref() {
        None | Some("large") => Geometry::Large,
        Some("small") => Geometry::Small,
        Some(other) => return Err(format!("unknown geometry {other}").into()),
    };
    let device = Device::new(index)?;
    let h = geometry.hidden();
    let mut lengths = Vec::new();
    let mut tokens = 0;
    for &len in LENGTHS.iter().cycle() {
        if tokens + len > TOKENS {
            break;
        }
        lengths.push(len);
        tokens += len;
    }
    let cu: Vec<i32> = std::iter::once(0)
        .chain(lengths.iter().scan(0i32, |total, &len| {
            *total += len as i32;
            Some(*total)
        }))
        .collect();
    let qkv: Vec<f16> = random(tokens * 3 * h, 1, 1.0)
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let qkv = Buffer::from_slice(&device, &qkv)?;
    let low = Buffer::from_slice(&device, &random(tokens * 2 * h, 2, 1e-4))?;
    let cu_seqlens = Buffer::from_slice(&device, &cu)?;
    let out = Buffer::<f16>::zeroed(&device, tokens * h)?;
    let constants = [geometry.head_dim() as u32, geometry.heads() as u32];
    let push = Push::new().f32(1.0 / (geometry.head_dim() as f32).sqrt());
    let max_len = *LENGTHS.iter().max().expect("lengths") as u32;
    let groups = [
        max_len.div_ceil(64),
        geometry.heads() as u32,
        lengths.len() as u32,
    ];
    let flops: f64 = lengths
        .iter()
        .map(|&len| 4.0 * (len * len * h) as f64)
        .sum();
    let mut variants = vec![("portable", shaders::ATTENTION, None)];
    if device.cooperative_matrix {
        variants.push(("matrix", shaders::ATTENTION_MATRIX, Some(32)));
    }
    for (label, spirv, subgroup_size) in variants {
        let kernel = unsafe {
            Kernel::new(
                &device,
                &KernelSpec {
                    spirv,
                    bindings: 4,
                    push_bytes: 4,
                    constants: &constants,
                    subgroup_size,
                },
            )
        }?;
        let descriptors = Descriptors::new(&device, 1, 4)?;
        let set = unsafe {
            descriptors.bind(
                &kernel,
                &[
                    qkv.binding(),
                    low.binding(),
                    cu_seqlens.binding(),
                    out.binding(),
                ],
            )
        }?;
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
        let ms = started.elapsed().as_secs_f64() * 1e3 / ITERATIONS as f64;
        println!(
            "{} {geometry:?} {label}: {} sequences, {tokens} tokens: {ms:.3} ms, {:.2} TFLOPS",
            device.name,
            lengths.len(),
            flops / (ms * 1e-3) / 1e12
        );
    }
    Ok(())
}
