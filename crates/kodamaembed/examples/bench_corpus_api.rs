//! Peak host memory of the collected or streaming API on repeated short text, one mode per process.
//! Usage: bench_corpus_api MODEL minilm|e5 cpu|cuda|vulkan collect|stream COUNT

use kodamaembed::{Device, Embedder, EmbeddingModel};
use std::{convert::Infallible, error::Error, time::Instant};

const TEXT: &str = "Berlin is the capital of Germany.";
const BATCH_TOKENS: usize = 16_384;

struct Memory {
    working_set: usize,
    peak_working_set: usize,
}

#[cfg(windows)]
fn memory() -> Option<Memory> {
    use std::ffi::c_void;

    // PROCESS_MEMORY_COUNTERS from psapi.h; SIZE_T fields follow two DWORDs.
    #[repr(C)]
    #[derive(Default)]
    struct Counters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
    }
    #[link(name = "psapi")]
    unsafe extern "system" {
        fn GetProcessMemoryInfo(process: *mut c_void, counters: *mut Counters, size: u32) -> i32;
    }

    let size = size_of::<Counters>() as u32;
    let mut counters = Counters {
        cb: size,
        ..Default::default()
    };
    // SAFETY: the pseudo handle is valid and `counters` is writable at `size`.
    let success = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, size) };
    (success != 0).then_some(Memory {
        working_set: counters.working_set_size,
        peak_working_set: counters.peak_working_set_size,
    })
}

#[cfg(not(windows))]
fn memory() -> Option<Memory> {
    None
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err(
            "usage: bench_corpus_api MODEL minilm|e5 cpu|cuda|vulkan collect|stream COUNT".into(),
        );
    }
    let spec = match args[1].as_str() {
        "minilm" => EmbeddingModel::AllMiniLmL6V2.spec(),
        "e5" => EmbeddingModel::E5Large.spec(),
        _ => return Err("unknown model family".into()),
    };
    let device = match args[2].as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda(0),
        "vulkan" => Device::first_discrete_vulkan()?,
        _ => return Err("unknown device".into()),
    };
    if !matches!(args[3].as_str(), "collect" | "stream") {
        return Err("mode must be collect or stream".into());
    }
    let count: usize = args[4].parse()?;

    let started = Instant::now();
    let embedder = Embedder::builder()
        .device(device)
        .batch_tokens(BATCH_TOKENS)
        .build_from_dir(&args[0], spec)?;
    let load_seconds = started.elapsed().as_secs_f64();
    let load_memory = memory();
    let dim = embedder.dim();

    // Both modes time the same input setup and checksum.
    let started = Instant::now();
    let mut checksum = 0.0f64;
    let completed = if args[3] == "collect" {
        let texts = vec![TEXT; count];
        let vectors = embedder.embed_passages(&texts)?;
        for vector in &vectors {
            assert_eq!(vector.len(), dim);
            for &value in vector {
                checksum += f64::from(value);
            }
        }
        vectors.len()
    } else {
        let mut seen = 0;
        let completed =
            embedder.embed_passages_stream(std::iter::repeat_n(TEXT, count), |batch| {
                assert_eq!(batch.start, seen);
                assert_eq!(batch.dim, dim);
                assert_eq!(batch.values.len() % dim, 0);
                seen += batch.values.len() / dim;
                for &value in batch.values {
                    checksum += f64::from(value);
                }
                Ok::<_, Infallible>(())
            })?;
        assert_eq!(completed, seen);
        completed
    };
    let inference_seconds = started.elapsed().as_secs_f64();
    assert_eq!(completed, count);
    let final_memory = memory();

    println!(
        "{}",
        serde_json::json!({
            "benchmark": "synthetic repeated-text API host memory",
            "real_corpus_throughput": false,
            "memory_scope": "process host working set; includes model loading; excludes GPU memory",
            "model_family": args[1],
            "device": embedder.device_name(),
            "mode": args[3],
            "batch_tokens": BATCH_TOKENS,
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "count": completed,
            "dim": dim,
            "checksum": checksum,
            "load_working_set_bytes": load_memory.as_ref().map(|m| m.working_set),
            "load_peak_working_set_bytes": load_memory.as_ref().map(|m| m.peak_working_set),
            "working_set_bytes": final_memory.as_ref().map(|m| m.working_set),
            "peak_working_set_bytes": final_memory.as_ref().map(|m| m.peak_working_set),
        })
    );
    Ok(())
}
