//! End-to-end embedding benchmark on JSONL `text` and optional `title` fields.
//! Usage: bench_corpus MODEL DATA_JSONL minilm|e5 cpu|cpu-centered|cuda|cuda-centered|vulkan REPEATS OUT_DIR [ROWS [IDLE_MS [BATCH_TOKENS]]]

use kodamaembed::{Device, Embedder, EmbeddingModel, Precision};
use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs,
    path::Path,
    process::Command,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokenizers::{Tokenizer, TruncationParams};

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

fn qos() -> Result<()> {
    #[cfg(windows)]
    {
        #[repr(C)]
        struct State {
            version: u32,
            control_mask: u32,
            state_mask: u32,
        }
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn GetCurrentProcess() -> *mut std::ffi::c_void;
            fn SetProcessInformation(
                process: *mut std::ffi::c_void,
                class: i32,
                state: *const std::ffi::c_void,
                len: u32,
            ) -> i32;
        }
        let state = State {
            version: 1,
            control_mask: 1,
            state_mask: 0,
        };
        // SAFETY: the structure has the documented layout and outlives the call.
        if unsafe {
            SetProcessInformation(
                GetCurrentProcess(),
                4,
                (&raw const state).cast(),
                size_of::<State>() as u32,
            )
        } == 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
    }
    Ok(())
}

fn telemetry(
    identity: &std::result::Result<kodamaembed::GpuIdentity, String>,
) -> serde_json::Value {
    let unavailable =
        |reason: String| serde_json::json!({ "status": "unavailable", "reason": reason });
    let gpu = match identity {
        Ok(gpu) => gpu,
        Err(reason) => return unavailable(reason.clone()),
    };
    if gpu.vendor_id != 0x10de || gpu.uuid == [0; 16] {
        return unavailable("selected GPU has no verified NVIDIA board identity".into());
    }
    let hex: String = gpu.uuid.iter().map(|byte| format!("{byte:02x}")).collect();
    let uuid = format!(
        "GPU-{}-{}-{}-{}-{}",
        &hex[..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..]
    );
    let output = match Command::new("nvidia-smi")
        .arg(format!("--id={uuid}"))
        .args([
            "--query-gpu=uuid,pci.bus_id,timestamp,temperature.gpu,clocks.current.sm,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(output) => output,
        Err(error) => return unavailable(error.to_string()),
    };
    if !output.status.success() {
        return unavailable(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let rows: Vec<_> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect();
    if rows.len() != 1
        || !rows[0]
            .split(',')
            .next()
            .unwrap_or("")
            .trim()
            .eq_ignore_ascii_case(&uuid)
    {
        return unavailable("snapshot UUID does not match the selected physical GPU".into());
    }
    serde_json::json!({ "status": "available", "uuid": uuid,
        "scope": "matched whole-board snapshot including other activity; not energy integration",
        "csv_columns": "uuid,pci.bus_id,timestamp,temperature.gpu,clocks.current.sm,power.draw,power.limit",
        "csv": rows[0].trim() })
}

fn main() -> Result<()> {
    qos()?;
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(6..=9).contains(&args.len()) {
        return Err(
            "usage: bench_corpus MODEL DATA_JSONL minilm|e5 cpu|cpu-centered|cuda|cuda-centered|vulkan REPEATS OUT_DIR [ROWS [IDLE_MS [BATCH_TOKENS]]]"
                .into(),
        );
    }
    let spec = match args[2].as_str() {
        "minilm" => EmbeddingModel::AllMiniLmL6V2.spec(),
        "e5" => EmbeddingModel::E5Large.spec(),
        _ => return Err("unsupported family".into()),
    };
    let device = match args[3].as_str() {
        "cpu" | "cpu-centered" => Device::Cpu,
        "cuda" | "cuda-centered" => Device::Cuda(0),
        "vulkan" => Device::first_discrete_vulkan()?,
        _ => return Err("unsupported device".into()),
    };
    let repeats = args[4].parse::<usize>()?;
    let limit = args
        .get(6)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(usize::MAX);
    let idle_ms = args
        .get(7)
        .map(|value| value.parse::<u64>())
        .transpose()?
        .unwrap_or(0);
    let batch_tokens = args
        .get(8)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(16_384);
    if repeats == 0 || limit == 0 {
        return Err("repeats and rows must be positive".into());
    }
    if idle_ms > 60_000 {
        return Err("IDLE_MS must be at most 60000".into());
    }
    let data = fs::read(&args[1])?;
    let rows = std::str::from_utf8(&data)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .take(limit)
        .map(|line| -> Result<String> {
            let row: serde_json::Value = serde_json::from_str(line)?;
            let text = row["text"].as_str().ok_or("row requires text")?;
            let title = row["title"].as_str().unwrap_or("");
            Ok(if title.is_empty() {
                text.to_owned()
            } else {
                format!("{title} {text}")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if rows.is_empty() {
        return Err("no rows".into());
    }
    let model_dir = Path::new(&args[0]);
    let tokenizer_bytes = fs::read(model_dir.join("tokenizer.json"))?;
    let mut tokenizer = Tokenizer::from_bytes(&tokenizer_bytes)?;
    tokenizer.with_padding(None);
    tokenizer.with_truncation(Some(TruncationParams {
        max_length: spec.max_length,
        ..Default::default()
    }))?;
    // The receipt's input digest, computed outside timing.
    let mut input_hash = Sha256::new();
    let mut tokens = 0;
    for text in &rows {
        let encoding = tokenizer.encode(format!("{}{text}", spec.passage_prefix), true)?;
        input_hash.update((encoding.len() as u64).to_le_bytes());
        for &value in encoding.get_ids().iter().chain(encoding.get_type_ids()) {
            input_hash.update(value.to_le_bytes());
        }
        tokens += encoding.len();
    }
    let load = Instant::now();
    let precision = match args[3].as_str() {
        "cpu-centered" | "cuda-centered" => Precision::Quantized,
        _ => Precision::Float,
    };
    let model = Embedder::builder()
        .device(device)
        .precision(precision)
        .batch_tokens(batch_tokens)
        .build_from_dir(model_dir, spec)?;
    let load_seconds = load.elapsed().as_secs_f64();
    let gpu_identity = model.gpu_identity().map_err(|error| error.to_string());
    model.embed_passages_stream(rows.iter().take(64), |_| {
        Ok::<_, std::convert::Infallible>(())
    })?;
    let mut runs = Vec::new();
    let mut first_hash = None;
    let mut final_values = Vec::new();
    for round in 0..repeats {
        if round > 0 && idle_ms != 0 {
            // With an idle gap the runs are bursts, not continuous throughput.
            std::thread::sleep(Duration::from_millis(idle_ms));
        }
        let before = telemetry(&gpu_identity);
        let mut values = Vec::with_capacity(rows.len() * model.dim());
        let started_unix_seconds = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        let start = Instant::now();
        let count = model.embed_passages_stream(rows.iter(), |batch| {
            assert_eq!(batch.start * batch.dim, values.len());
            values.extend_from_slice(batch.values);
            Ok::<_, std::convert::Infallible>(())
        })?;
        let seconds = start.elapsed().as_secs_f64();
        let finished_unix_seconds = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        let after = telemetry(&gpu_identity);
        assert_eq!(count, rows.len());
        let mut hash = Sha256::new();
        for value in &values {
            hash.update(value.to_le_bytes());
        }
        let hash = format!("{:x}", hash.finalize());
        match &first_hash {
            Some(first) => assert_eq!(&hash, first),
            None => first_hash = Some(hash.clone()),
        }
        runs.push(serde_json::json!({ "round":round,"seconds":seconds,
            "started_unix_seconds":started_unix_seconds,"finished_unix_seconds":finished_unix_seconds,
            "tokens_per_second":tokens as f64/seconds,"output_sha256":hash,
            "telemetry_before":before,"telemetry_after":after }));
        final_values = values;
    }
    let out = Path::new(&args[5]);
    fs::create_dir_all(out)?;
    let mut vectors = Vec::with_capacity(24 + final_values.len() * 4);
    vectors.extend_from_slice(b"KODV");
    vectors.extend_from_slice(&1_u32.to_le_bytes());
    vectors.extend_from_slice(&(rows.len() as u64).to_le_bytes());
    vectors.extend_from_slice(&(model.dim() as u64).to_le_bytes());
    for value in final_values {
        vectors.extend_from_slice(&value.to_le_bytes());
    }
    fs::write(out.join("passages.kodv"), vectors)?;
    let receipt = serde_json::json!({
        "scope":"resident-model tokenization through retained ordered embeddings; excludes load, input file I/O, checksums and output file I/O",
        "backend":args[3],"device":model.device_name(),"fingerprint":model.fingerprint(),
        "gpu_identity":gpu_identity,
        "family":args[2],"max_length":spec.max_length,"batch_tokens":batch_tokens,
        "rows":rows.len(),"dim":model.dim(),"unpadded_tokens":tokens,
        "data_sha256":format!("{:x}",Sha256::digest(&data)),
        "tokenizer_sha256":format!("{:x}",Sha256::digest(&tokenizer_bytes)),
        "input_ids_types_sha256":format!("{:x}",input_hash.finalize()),
        "load_seconds":load_seconds,"warmup_rows":rows.len().min(64),"runs":runs,
        "inter_repeat_idle_ms":idle_ms,
        "process_qos":if cfg!(windows) { "execution speed throttling off before any workers" } else { "not applicable" },
        "harness_sha256":format!("{:x}",Sha256::digest(include_bytes!("bench_corpus.rs"))),
        "binary_sha256":format!("{:x}",Sha256::digest(fs::read(std::env::current_exe()?)?)),
        "cpu_sources_sha256":kodamaembed_cpu::NUMERICAL_SOURCES.iter().map(|s|format!("{:x}",Sha256::digest(s))).collect::<Vec<_>>(),
    });
    fs::write(out.join("run.json"), serde_json::to_vec_pretty(&receipt)?)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
