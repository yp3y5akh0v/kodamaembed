//! Attention throughput on a packed 16,384-token batch of mixed lengths.
//! Usage: bench_cuda_attention [large|small]

use half::f16;
use kodamaembed_cuda::{
    Context, DeviceBuffer, Event, Stream,
    kernels::{Kernels, Packed},
};
use kodamaembed_model::Geometry;

const TOKENS: usize = 16_384;
const LENGTHS: [usize; 8] = [24, 48, 96, 160, 216, 320, 448, 512];
const ITERATIONS: u32 = 50;

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
    let geometry = match std::env::args().nth(1).as_deref() {
        None | Some("large") => Geometry::Large,
        Some("small") => Geometry::Small,
        Some(other) => return Err(format!("unknown geometry {other}").into()),
    };
    let context = Context::new(0)?;
    let kernels = Kernels::load(&context, geometry)?;
    let stream = Stream::new(&context)?;
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
    let qkv = DeviceBuffer::from_slice(&context, &random(tokens * 3 * h, 1, 1.0))?;
    let low = DeviceBuffer::from_slice(&context, &random(tokens * 2 * h, 2, 1.0))?;
    let cu_seqlens = DeviceBuffer::from_slice(&context, &cu)?;
    let out = DeviceBuffer::<f16>::zeroed(&context, tokens * h)?;
    let packed = Packed {
        cu_seqlens: &cu_seqlens,
        sequences: lengths.len(),
        tokens,
        max_len: *LENGTHS.iter().max().expect("lengths"),
        min_len: *LENGTHS.iter().min().expect("lengths"),
    };
    let (start, stop) = (Event::new(&context)?, Event::new(&context)?);
    // SAFETY: the buffers outlive every launch; the stream is synchronized below.
    unsafe { kernels.attention(&stream, &qkv, &low, &packed, &out, None)? };
    start.record(&stream)?;
    for _ in 0..ITERATIONS {
        // SAFETY: as above.
        unsafe { kernels.attention(&stream, &qkv, &low, &packed, &out, None)? };
    }
    stop.record(&stream)?;
    stream.synchronize()?;
    let ms = f64::from(start.elapsed_ms(&stop)?) / f64::from(ITERATIONS);
    let flops: f64 = lengths
        .iter()
        .map(|&len| 4.0 * (len * len * h) as f64)
        .sum();
    println!(
        "{geometry:?}: {} sequences, {tokens} tokens: {ms:.3} ms, {:.1} TFLOPS",
        lengths.len(),
        flops / (ms * 1e-3) / 1e12
    );
    Ok(())
}
