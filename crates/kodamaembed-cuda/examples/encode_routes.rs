//! Centered e5-large encodes checked against the CPU, for compute-sanitizer. Usage: encode_routes
//! BATCHES (e.g. `64;15,17;8x16`) [layers: no persistent plan, whose spin-waits racecheck serializes]

#[cfg(target_arch = "x86_64")]
use kodamaembed_cpu::{CpuPrecision, Encoder as CpuEncoder, Kernel};
#[cfg(target_arch = "x86_64")]
use kodamaembed_cuda::{Context, repro::ReproEncoder};
#[cfg(target_arch = "x86_64")]
use kodamaembed_model::{Output, Pooling, Positions, Shape, reference::Synthetic};

#[cfg(target_arch = "x86_64")]
const E5: Shape = Shape {
    layers: 24,
    hidden: 1024,
    heads: 16,
    intermediate: 4096,
    vocab: 30_522,
    max_positions: 512,
    type_vocab: 2,
    eps: 1e-12,
    positions: Positions::Sequential,
};

/// `64`, `15,17` or `8x16`.
#[cfg(target_arch = "x86_64")]
fn batch(arg: &str) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    if let Some((count, len)) = arg.split_once('x') {
        return Ok(vec![len.parse()?; count.parse()?]);
    }
    arg.split(',').map(|len| Ok(len.parse()?)).collect()
}

#[cfg(not(target_arch = "x86_64"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("centered CUDA encodes require an x86_64 host".into())
}

#[cfg(target_arch = "x86_64")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (list, layers) = match args.as_slice() {
        [list] => (list, false),
        [list, layers] if layers == "layers" => (list, true),
        _ => return Err("usage: encode_routes BATCHES [layers]".into()),
    };
    let batches = list.split(';').map(batch).collect::<Result<Vec<_>, _>>()?;
    if batches.iter().any(|lengths| {
        lengths.is_empty()
            || lengths
                .iter()
                .any(|&len| len == 0 || len > E5.max_positions)
    }) {
        return Err("every batch needs lengths between 1 and the model's position limit".into());
    }
    let capacity = batches
        .iter()
        .map(|lengths| lengths.iter().sum())
        .max()
        .unwrap_or(1);
    let context = Context::new(0)?;
    let weights = Synthetic::new(&E5, 7);
    let mut cpu = CpuEncoder::new_with_precision(
        Kernel::Avx2,
        E5,
        Output::Embedding(Pooling::Mean),
        capacity,
        &weights,
        CpuPrecision::E5CenteredInt8,
    )?;
    cpu.calibrate(&[
        (vec![101, 102], vec![0, 0]),
        (vec![101, 2000, 1024, 102], vec![0; 4]),
    ])?;
    let mut repro = ReproEncoder::new(&context, &cpu.export_reproducible()?)?;
    if layers {
        repro.set_aside_stream_engine();
    }
    for lengths in &batches {
        let tokens: usize = lengths.iter().sum();
        let ids: Vec<i32> = (0..tokens).map(|i| (i * 7_919 % E5.vocab) as i32).collect();
        let types = vec![0; tokens];
        let expected = cpu.encode(&ids, &types, lengths)?;
        let actual = repro.encode(&ids, &types, lengths)?;
        let differing = expected
            .iter()
            .zip(&actual)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        if differing != 0 || expected.len() != actual.len() {
            return Err(format!("{lengths:?}: {differing} values differ").into());
        }
        println!("{lengths:?}: bit for bit");
    }
    Ok(())
}
