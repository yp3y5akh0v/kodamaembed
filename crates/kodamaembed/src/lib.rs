//! Text embeddings and reranking on CPUs and GPUs.
//!
//! A text's vector or score has the same bits whatever batch it shares.
//!
//! ```no_run
//! use kodamaembed::{Embedder, EmbeddingModel, Reranker, RerankerModel};
//!
//! # fn main() -> Result<(), kodamaembed::Error> {
//! // Downloads on first use, then runs on CUDA, a discrete Vulkan GPU or the processor.
//! let embedder = Embedder::new(EmbeddingModel::E5Large)?;
//! let passages = embedder.embed_passages(&["Berlin is the capital of Germany."])?;
//! let query = embedder.embed_queries(&["capital of Germany"])?;
//! let cosine: f32 = query[0].iter().zip(&passages[0]).map(|(q, p)| q * p).sum();
//!
//! let reranker = Reranker::new(RerankerModel::MsMarcoMiniLmL6V2)?;
//! let scores = reranker.score("capital of Germany", &["Berlin is the capital of Germany."])?;
//! # Ok(())
//! # }
//! ```

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/yp3y5akh0v/kodamaembed/master/docs/assets/crest.svg",
    html_favicon_url = "https://raw.githubusercontent.com/yp3y5akh0v/kodamaembed/master/docs/assets/favicon.ico"
)]
#![warn(missing_docs)]

mod artifact_io;
mod bert_wordpiece;
#[cfg(target_arch = "x86_64")]
mod calibration_cache;
mod checkpoint;
mod cpu_workers;
mod hub;
mod identity;
mod precision_scope;
#[cfg(any(target_arch = "x86_64", feature = "download"))]
mod publish;
mod registry;
#[cfg(test)]
mod test_support;
mod tokenizing;

pub use cpu_workers::CpuWorkers;
pub use hub::Hub;
#[cfg(feature = "download")]
pub use hub::Progress;
pub use registry::{EmbeddingModel, RerankerModel};

use kodamaembed_cpu::CpuPrecision;

/// Numerical mode for [`EmbedderBuilder::precision`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// FP32 on the processor, FP16 projection operands with FP32 sums on
    /// CUDA, and the adapter's shaders on Vulkan.
    #[default]
    Float,
    /// The evaluated centered INT8 preset of E5-large-v1: the same vectors and
    /// fingerprint on the processor and on CUDA.
    Quantized,
}

use kodamaembed_model::{Output, Shape};
use sha2::{Digest, Sha256};
use std::{
    borrow::Cow,
    fmt,
    num::NonZeroUsize,
    path::{Path, PathBuf},
    sync::{Condvar, Mutex, MutexGuard, PoisonError},
    thread,
};
use tokenizers::{EncodeInput, PostProcessor, Tokenizer, TruncationParams};

/// Tokens an encoder batch holds unless a builder sets another count.
const BATCH_TOKENS: usize = 16_384;

pub use kodamaembed_model::{Pooling, Profile, STAGES, Stage};

/// How an embedding model expects its inputs and pools its outputs;
/// [`EmbeddingModel::spec`] gives each registered model's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec<'a> {
    /// Put before every passage.
    pub passage_prefix: &'a str,
    /// Put before every query.
    pub query_prefix: &'a str,
    /// The tokens a text is truncated to, special tokens included.
    pub max_length: usize,
    /// The final hidden rows a vector averages.
    pub pooling: Pooling,
}

/// Where the encoder runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// An NVIDIA GPU by CUDA ordinal.
    Cuda(usize),
    /// Any Vulkan 1.3 GPU by the loader's adapter index.
    Vulkan(usize),
    /// This machine's processor, on every core.
    Cpu,
    /// CUDA at compute capability 8.0 or newer, else a discrete GPU on a conformant Vulkan driver,
    /// else the processor. Only a GPU lacking what the model or settings need is passed over.
    Auto,
}

/// CUDA 0 with compute capability 8.0 or newer.
fn ampere_cuda() -> Result<bool, Error> {
    Ok(kodamaembed_cuda::Context::capability(0)
        .map_err(kodamaembed_cuda::EncoderError::from)?
        .is_some_and(|capability| capability >= (8, 0)))
}

/// Vulkan enumeration can load a second driver, so defer it until CUDA has been tried.
fn auto_gpu<T>(
    shape: &Shape,
    batch_tokens: usize,
    mut open: impl FnMut(Device) -> Result<T, Error>,
) -> Result<Option<T>, Error> {
    if shape.geometry().is_none() {
        return Ok(None);
    }
    if batch_tokens <= kodamaembed_cuda::MAX_TOKENS
        && ampere_cuda()?
        && let Some(gpu) = first_usable([Device::Cuda(0)], &mut open)?
    {
        return Ok(Some(gpu));
    }
    if batch_tokens > kodamaembed_vulkan::encoder::MAX_TOKENS {
        return Ok(None);
    }
    let adapters = kodamaembed_vulkan::Device::available_adapters()
        .map_err(kodamaembed_vulkan::EncoderError::from)?;
    first_usable(
        adapters
            .into_iter()
            .filter(|adapter| adapter.discrete && !adapter.portability)
            .map(|adapter| Device::Vulkan(adapter.index)),
        open,
    )
}

/// The first of `candidates` that opens; only one lacking what the model needs is passed over.
fn first_usable<D, T>(
    candidates: impl IntoIterator<Item = D>,
    mut open: impl FnMut(D) -> Result<T, Error>,
) -> Result<Option<T>, Error> {
    for candidate in candidates {
        match open(candidate) {
            Err(error) if error.is_unsupported() => {}
            result => return result.map(Some),
        }
    }
    Ok(None)
}

impl Device {
    /// The first discrete Vulkan GPU, or an error naming the adapters found.
    pub fn first_discrete_vulkan() -> Result<Self, Error> {
        let adapters = kodamaembed_vulkan::Device::adapters()
            .map_err(kodamaembed_vulkan::EncoderError::from)?;
        adapters
            .iter()
            .find(|adapter| adapter.discrete)
            .map(|adapter| Self::Vulkan(adapter.index))
            .ok_or_else(|| {
                let names: Vec<&str> = adapters.iter().map(|a| a.name.as_str()).collect();
                Error::Unsupported(format!("no discrete Vulkan GPU among {names:?}"))
            })
    }
}

/// The GPU a loaded model selected; not part of the numerical policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct GpuIdentity {
    /// PCI vendor identifier (NVIDIA is 0x10de).
    pub vendor_id: u32,
    /// Device UUID for matching across GPU APIs; a CUDA MIG UUID can name an
    /// instance rather than the board.
    pub uuid: [u8; 16],
}

/// Why loading or running a model failed.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// The model directory, its configuration or its weights.
    Model(String),
    /// Finding or downloading a model that loads by name.
    Download(String),
    /// Loading the tokenizer or tokenizing a text.
    Tokenizer(String),
    /// The CUDA backend.
    Cuda(kodamaembed_cuda::EncoderError),
    /// The Vulkan backend.
    Vulkan(kodamaembed_vulkan::EncoderError),
    /// The CPU backend.
    Cpu(kodamaembed_cpu::EncoderError),
    /// Something the device or backend does not offer.
    Unsupported(String),
    /// The caller's stop request ended the run.
    Interrupted,
    /// An earlier call panicked inside a batch and left the backend half run.
    Poisoned,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(reason) => write!(f, "model: {reason}"),
            Self::Download(reason) => write!(f, "download: {reason}"),
            Self::Tokenizer(reason) => write!(f, "tokenizer: {reason}"),
            Self::Cuda(error) => write!(f, "{error}"),
            Self::Vulkan(error) => write!(f, "{error}"),
            Self::Cpu(error) => write!(f, "{error}"),
            Self::Unsupported(reason) => write!(f, "unsupported: {reason}"),
            Self::Interrupted => f.write_str("interrupted by the caller"),
            Self::Poisoned => f.write_str("an earlier call panicked inside a batch"),
        }
    }
}

impl std::error::Error for Error {}

impl Error {
    /// The device or backend lacks something the model needs; nothing failed.
    fn is_unsupported(&self) -> bool {
        matches!(
            self,
            Self::Unsupported(_)
                | Self::Cuda(kodamaembed_cuda::EncoderError::Unsupported(_))
                | Self::Vulkan(kodamaembed_vulkan::EncoderError::Unsupported(_))
        )
    }
}

/// A completed, ordered batch borrowed by a streaming output callback.
/// `values.chunks_exact(dim)` yields one row per input.
pub struct Batch<'a> {
    /// Index of the first input represented by this batch.
    pub start: usize,
    /// Values per input: the embedding width, or one for relevance scores.
    pub dim: usize,
    /// Tokens the batch encoded, prefixes and special tokens included.
    pub tokens: usize,
    /// Contiguous rows, valid until the callback returns.
    pub values: &'a [f32],
}

/// An inference failure or an error returned by a streaming output callback.
#[derive(Debug)]
pub enum StreamError<E> {
    /// Loading or running the model's inputs failed.
    Inference(Error),
    /// The output callback failed; the batches before it were delivered.
    Sink(E),
}

impl<E: fmt::Display> fmt::Display for StreamError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Inference(error) => error.fmt(f),
            Self::Sink(error) => write!(f, "output: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for StreamError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Inference(error) => error,
            Self::Sink(error) => error,
        })
    }
}

impl<E> From<Error> for StreamError<E> {
    fn from(error: Error) -> Self {
        Self::Inference(error)
    }
}

impl From<kodamaembed_cuda::EncoderError> for Error {
    fn from(error: kodamaembed_cuda::EncoderError) -> Self {
        Self::Cuda(error)
    }
}

impl From<kodamaembed_vulkan::EncoderError> for Error {
    fn from(error: kodamaembed_vulkan::EncoderError) -> Self {
        Self::Vulkan(error)
    }
}

impl From<kodamaembed_cpu::EncoderError> for Error {
    fn from(error: kodamaembed_cpu::EncoderError) -> Self {
        Self::Cpu(error)
    }
}

enum Backend {
    Cuda(Box<kodamaembed_cuda::Encoder>),
    Vulkan(Box<kodamaembed_vulkan::Encoder>),
    Cpu(Box<kodamaembed_cpu::Encoder>),
    #[cfg(target_arch = "x86_64")]
    CudaCentered {
        encoder: Box<kodamaembed_cuda::repro::ReproEncoder>,
        max_tokens: usize,
    },
}

#[derive(Clone, Copy)]
struct Limits {
    tokens: usize,
    sequences: usize,
}

impl Limits {
    /// Whether the batch must run before a sequence of `len` tokens joins it.
    fn full(self, tokens: usize, sequences: usize, len: usize) -> bool {
        sequences > 0 && (len > self.tokens.saturating_sub(tokens) || sequences >= self.sequences)
    }
}

impl Backend {
    /// Inference workers on the processor, None on a GPU.
    fn threads(&self) -> Option<usize> {
        match self {
            Self::Cpu(encoder) => Some(encoder.threads()),
            Self::Cuda(_) | Self::Vulkan(_) => None,
            #[cfg(target_arch = "x86_64")]
            Self::CudaCentered { .. } => None,
        }
    }

    fn limits(&self) -> Limits {
        let (tokens, sequences) = match self {
            Self::Cuda(encoder) => (encoder.max_tokens(), encoder.max_sequences()),
            Self::Vulkan(encoder) => (encoder.max_tokens(), usize::MAX),
            Self::Cpu(encoder) => (encoder.max_tokens(), usize::MAX),
            #[cfg(target_arch = "x86_64")]
            Self::CudaCentered { max_tokens, .. } => (
                (*max_tokens).min(kodamaembed_cuda::repro::MAX_BATCH_TOKENS),
                kodamaembed_cuda::repro::MAX_BATCH_SEQUENCES,
            ),
        };
        Limits { tokens, sequences }
    }

    fn gpu_identity(&self) -> Result<GpuIdentity, Error> {
        let (vendor_id, uuid) = match self {
            Self::Cuda(encoder) => (
                0x10de,
                encoder
                    .device_uuid()
                    .map_err(kodamaembed_cuda::EncoderError::from)?,
            ),
            Self::Vulkan(encoder) => encoder.device_identity(),
            #[cfg(target_arch = "x86_64")]
            Self::CudaCentered { encoder, .. } => (
                0x10de,
                encoder
                    .device_uuid()
                    .map_err(kodamaembed_cuda::EncoderError::from)?,
            ),
            Self::Cpu(_) => {
                return Err(Error::Unsupported(
                    "the CPU backend reports no GPU identity".into(),
                ));
            }
        };
        if uuid == [0; 16] {
            return Err(Error::Unsupported(
                "selected GPU returned an unknown UUID".into(),
            ));
        }
        Ok(GpuIdentity { vendor_id, uuid })
    }

    fn encode(&mut self, ids: &[i32], types: &[i32], lengths: &[usize]) -> Result<Vec<f32>, Error> {
        Ok(match self {
            Self::Cuda(encoder) => encoder.encode(ids, types, lengths)?,
            Self::Vulkan(encoder) => encoder.encode(ids, types, lengths)?,
            Self::Cpu(encoder) => encoder.encode(ids, types, lengths)?,
            #[cfg(target_arch = "x86_64")]
            Self::CudaCentered { encoder, .. } => encoder
                .encode(ids, types, lengths)
                .map_err(kodamaembed_cuda::EncoderError::from)?,
        })
    }
}

/// The backend of a model its callers share: it runs one batch at a time, and callers get it
/// in arrival order, so a short call waits for one batch of a long one and not for all of it.
struct Shared {
    backend: Mutex<Backend>,
    places: Mutex<Places>,
    served: Condvar,
    limits: Limits,
    /// Inference workers on the processor, None on a GPU.
    threads: Option<usize>,
}

/// The queue for the backend: the places handed out and the one whose turn it is.
#[derive(Default)]
struct Places {
    taken: u64,
    serving: u64,
}

/// A caller's turn at the backend; the next place's begins when it drops.
struct Turn<'a>(&'a Shared);

impl Shared {
    fn new(backend: Backend) -> Self {
        Self {
            limits: backend.limits(),
            threads: backend.threads(),
            backend: Mutex::new(backend),
            places: Mutex::default(),
            served: Condvar::new(),
        }
    }

    /// Nothing panics while the places are held: a poisoned lock still counts right.
    fn places(&self) -> MutexGuard<'_, Places> {
        self.places.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Waits for every caller that came before.
    fn turn(&self) -> Turn<'_> {
        let mut places = self.places();
        let place = places.taken;
        places.taken += 1;
        drop(
            self.served
                .wait_while(places, |places| places.serving != place)
                .unwrap_or_else(PoisonError::into_inner),
        );
        Turn(self)
    }

    /// The backend of a model no caller shares yet or any more.
    fn backend_mut(&mut self) -> Result<&mut Backend, Error> {
        self.backend.get_mut().map_err(|_| Error::Poisoned)
    }

    /// The backend: for the caller whose turn it is, and for a reader of its facts, who
    /// waits out a batch in flight.
    fn backend(&self) -> Result<MutexGuard<'_, Backend>, Error> {
        self.backend.lock().map_err(|_| Error::Poisoned)
    }
}

impl Drop for Turn<'_> {
    fn drop(&mut self) {
        let mut places = self.0.places();
        places.serving += 1;
        // A lone caller has nobody to wake, and pays no system call for it.
        if places.taken != places.serving {
            self.0.served.notify_all();
        }
    }
}

// Bounding input chunks limits tokenized lookahead and first-batch preparation.
const CHUNK: usize = 64;

struct Model {
    device: Shared,
    device_name: String,
    tokenizer: tokenizing::InferenceTokenizer,
    /// Separate from inference so later inputs tokenize while a batch runs.
    tokenizing: kodamaembed_cpu::Workers,
    hidden: usize,
    fingerprint: String,
}

#[derive(Clone, Copy)]
struct Selection {
    cpu: CpuPrecision,
    workers: Option<CpuWorkers>,
}

/// The backend path a builder's settings select.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    Float(Device),
    CenteredCpu,
    /// Centered INT8 on this CUDA GPU, with the processor's arithmetic.
    CenteredCuda(usize),
    /// Centered INT8 on CUDA 0 when it can run it, else on the processor: the same vectors.
    CenteredAuto,
}

/// An explicit device that cannot honor a setting is an error; `Device::Auto` skips such devices.
fn route(device: Device, precision: Precision, workers: bool) -> Result<Route, Error> {
    match (precision, device) {
        (Precision::Float, device) => float_device(device, workers).map(Route::Float),
        (Precision::Quantized, Device::Cpu) => Ok(Route::CenteredCpu),
        (Precision::Quantized, Device::Auto) if workers => Ok(Route::CenteredCpu),
        (Precision::Quantized, Device::Auto) => Ok(Route::CenteredAuto),
        (Precision::Quantized, Device::Cuda(_)) if workers => {
            Err(processor_only("a CPU thread count"))
        }
        (Precision::Quantized, Device::Cuda(ordinal)) => Ok(Route::CenteredCuda(ordinal)),
        (Precision::Quantized, Device::Vulkan(_)) => Err(Error::Unsupported(
            "quantized precision runs on the processor or a CUDA GPU, not Vulkan".into(),
        )),
    }
}

/// The floating-point path's device; explicit workers confine it to the processor.
fn float_device(device: Device, workers: bool) -> Result<Device, Error> {
    match device {
        Device::Cuda(_) | Device::Vulkan(_) if workers => Err(processor_only("a CPU thread count")),
        Device::Auto if workers => Ok(Device::Cpu),
        device => Ok(device),
    }
}

/// Worded for every caller: Rust, Python and the command line.
fn processor_only(setting: &str) -> Error {
    Error::Unsupported(format!("{setting} runs on the processor, not a GPU"))
}

fn cpu_policy_identity(
    kernel: kodamaembed_cpu::Kernel,
    numerical_identity: &str,
    workers: Option<CpuWorkers>,
) -> String {
    // The variant's name, not its display form: fingerprints hold this text.
    let mut identity = format!("CPU {kernel:?}: {numerical_identity}");
    if let Some(workers) = workers {
        identity.push_str(&format!(
            ";explicit-workers-v1:inference={};tokenizer={}",
            workers.inference_workers, workers.tokenizer_workers,
        ));
    }
    identity
}

/// The facade's files hashed into fingerprints; a change to any of them invalidates stored vectors.
const SOURCES: &[&[u8]] = &[
    include_bytes!("lib.rs"),
    include_bytes!("checkpoint.rs"),
    include_bytes!("artifact_io.rs"),
    include_bytes!("precision_scope.rs"),
    include_bytes!("cpu_workers.rs"),
    include_bytes!("tokenizing.rs"),
    include_bytes!("bert_wordpiece.rs"),
    include_bytes!("../calibration/passages.txt"),
];

/// Exact versions of the dependencies that touch the numbers; a test keeps them equal to the lock.
const PINNED_DEPENDENCIES: &[u8] = b"tokenizers=0.22.2;half=2.7.1\0";

fn policy(device: &str, sources: &[&[u8]]) -> Vec<u8> {
    let mut hash = Sha256::new();
    hash.update(b"kodamaembed-numerical-policy-v1\0");
    hash.update(PINNED_DEPENDENCIES);
    for source in [
        device.as_bytes(),
        std::env::consts::OS.as_bytes(),
        std::env::consts::ARCH.as_bytes(),
    ]
    .into_iter()
    .chain(SOURCES.iter().copied())
    .chain(kodamaembed_model::NUMERICAL_SOURCES.iter().copied())
    .chain(sources.iter().copied())
    {
        hash.update((source.len() as u64).to_le_bytes());
        hash.update(source);
    }
    hash.finalize().to_vec()
}

const CALIBRATION_PASSAGES: &str = include_str!("../calibration/passages.txt");
/// Words of a calibration passage that make its query.
const OPENING_WORDS: usize = 16;
/// Calibration queries of drawn vocabulary pieces, which reach the rows no passage uses.
const DRAWN_QUERIES: usize = 1024;
/// The most pieces a drawn query holds.
const DRAWN_PIECES: u64 = 3;

/// SplitMix64, so the drawn queries are the same on every platform and build.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }
}

fn signed(values: &[u32]) -> Result<Vec<i32>, Error> {
    values
        .iter()
        .map(|&value| {
            i32::try_from(value).map_err(|_| {
                Error::Tokenizer(format!(
                    "token id or type {value} exceeds signed 32-bit IDs"
                ))
            })
        })
        .collect()
}

/// One to DRAWN_PIECES drawn pieces in the query prefix's frame; no special token is drawn.
fn drawn_queries(
    tokenizer: &Tokenizer,
    query_prefix: &str,
) -> Result<Vec<kodamaembed_cpu::TokenSequence>, Error> {
    let frame = tokenizer
        .encode(query_prefix, true)
        .map_err(|e| Error::Tokenizer(e.to_string()))?;
    let closing = frame
        .get_special_tokens_mask()
        .iter()
        .rev()
        .take_while(|&&special| special == 1)
        .count();
    let (opening, closing) = frame.get_ids().split_at(frame.get_ids().len() - closing);
    let kind = signed(frame.get_type_ids())?
        .first()
        .copied()
        .unwrap_or_default();
    let mut special: Vec<u32> = tokenizer
        .get_added_tokens_decoder()
        .into_iter()
        .filter(|(_, token)| token.special)
        .map(|(id, _)| id)
        .collect();
    special.sort_unstable();
    let size = u32::try_from(tokenizer.get_vocab_size(false))
        .map_err(|_| Error::Tokenizer("the vocabulary exceeds 32-bit IDs".into()))?;
    let rows: Vec<u32> = (0..size)
        .filter(|id| special.binary_search(id).is_err())
        .collect();
    if rows.is_empty() {
        return Err(Error::Tokenizer(
            "the vocabulary holds no piece to draw calibration queries from".into(),
        ));
    }
    let mut generator = SplitMix64(0x9E37_79B9_7F4A_7C15);
    (0..DRAWN_QUERIES)
        .map(|_| {
            let pieces = 1 + generator.next() % DRAWN_PIECES;
            let mut ids = opening.to_vec();
            for _ in 0..pieces {
                ids.push(rows[(generator.next() % rows.len() as u64) as usize]);
            }
            ids.extend_from_slice(closing);
            Ok((signed(&ids)?, vec![kind; ids.len()]))
        })
        .collect()
}

fn calibration_inputs(
    tokenizer: &Tokenizer,
    prefixes: (&str, &str),
) -> Result<Vec<kodamaembed_cpu::TokenSequence>, Error> {
    let passages = CALIBRATION_PASSAGES.lines();
    let openings = passages.clone().map(|passage| {
        let words: Vec<&str> = passage.split_whitespace().take(OPENING_WORDS).collect();
        format!("{}{}", prefixes.1, words.join(" "))
    });
    let mut inputs = passages
        .map(|passage| format!("{}{passage}", prefixes.0))
        .chain(openings)
        .map(|text| {
            let encoding = tokenizer
                .encode(text, true)
                .map_err(|e| Error::Tokenizer(e.to_string()))?;
            Ok((
                signed(encoding.get_ids())?,
                signed(encoding.get_type_ids())?,
            ))
        })
        .collect::<Result<Vec<kodamaembed_cpu::TokenSequence>, Error>>()?;
    inputs.extend(drawn_queries(tokenizer, prefixes.1)?);
    Ok(inputs)
}

/// The tables are cached outside the model directory, which may be a shared snapshot.
#[cfg(target_arch = "x86_64")]
fn calibrate(
    encoder: &mut kodamaembed_cpu::Encoder,
    fingerprint: &str,
    inputs: impl FnOnce() -> Result<Vec<kodamaembed_cpu::TokenSequence>, Error>,
) -> Result<(), Error> {
    let dir = calibration_cache::dir(
        |name| std::env::var_os(name),
        std::env::home_dir(),
        std::env::consts::OS,
    )
    .ok_or_else(|| {
        Error::Model("no cache directory for calibration tables; set KODAMAEMBED_CACHE".into())
    })?;
    let path = calibration_cache::path(&dir, fingerprint);
    let cache_error =
        |e: std::io::Error| Error::Model(format!("calibration cache {}: {e}", path.display()));
    if let Some(tables) = calibration_cache::read(&path, fingerprint).map_err(cache_error)?
        && encoder.install_calibration(&tables).is_ok()
    {
        return Ok(());
    }
    encoder.calibrate(&inputs()?)?;
    std::fs::create_dir_all(&dir).map_err(cache_error)?;
    calibration_cache::write(&path, fingerprint, &encoder.calibration()?).map_err(cache_error)?;
    Ok(())
}

#[cfg(not(target_arch = "x86_64"))]
fn calibrate(
    _encoder: &mut kodamaembed_cpu::Encoder,
    _fingerprint: &str,
    _inputs: impl FnOnce() -> Result<Vec<kodamaembed_cpu::TokenSequence>, Error>,
) -> Result<(), Error> {
    Err(Error::Unsupported(
        "quantized precision calibrates on an x86-64 processor".into(),
    ))
}

/// Every ID the tokenizer can produce must index the embedding table.
fn check_vocabulary(tokenizer: &Tokenizer, rows: usize) -> Result<(), Error> {
    match tokenizer.get_vocab(true).into_values().max() {
        Some(id) if id as usize >= rows => Err(Error::Model(format!(
            "tokenizer.json holds token ID {id}, but the embedding table has {rows} rows"
        ))),
        _ => Ok(()),
    }
}

/// A truncated pair must keep a token of each text and fit the positions.
fn check_max_length(max_length: usize, special: usize, longest: usize) -> Result<(), Error> {
    if special.checked_add(2).is_none_or(|min| max_length < min) || max_length > longest {
        return Err(Error::Model(format!(
            "max_length {max_length} must leave room beside {special} special tokens \
             and fit the model's {longest}-token sequences"
        )));
    }
    Ok(())
}

fn open(
    device: Device,
    shape: Shape,
    output: Output,
    batch_tokens: usize,
    weights: &checkpoint::SafeTensors<'_>,
    selection: Selection,
) -> Result<(Backend, String, Vec<u8>), Error> {
    Ok(match device {
        Device::Cuda(ordinal) => {
            let context = kodamaembed_cuda::Context::new(ordinal)
                .map_err(kodamaembed_cuda::EncoderError::from)?;
            let encoder =
                kodamaembed_cuda::Encoder::new(&context, shape, output, batch_tokens, weights)?;
            let numerical = policy(
                &context.execution_identity(),
                kodamaembed_cuda::NUMERICAL_SOURCES,
            );
            (
                Backend::Cuda(Box::new(encoder)),
                context.name.clone(),
                numerical,
            )
        }
        Device::Vulkan(index) => {
            let gpu = kodamaembed_vulkan::Device::for_encoder(index)?;
            let encoder =
                kodamaembed_vulkan::Encoder::new(&gpu, shape, output, batch_tokens, weights)?;
            let units = if encoder.uses_matrix_units() {
                "cooperative matrices"
            } else {
                "portable shaders"
            };
            (
                Backend::Vulkan(Box::new(encoder)),
                format!("{} (Vulkan, {units})", gpu.name),
                policy(
                    &gpu.execution_identity(),
                    kodamaembed_vulkan::NUMERICAL_SOURCES,
                ),
            )
        }
        Device::Cpu => {
            let kernel = kodamaembed_cpu::Kernel::detect();
            let encoder = match selection.workers {
                Some(workers) => kodamaembed_cpu::Encoder::new_with_precision_and_workers(
                    kernel,
                    shape,
                    output,
                    batch_tokens,
                    weights,
                    selection.cpu,
                    workers.inference_workers,
                )?,
                None => kodamaembed_cpu::Encoder::new_with_precision(
                    kernel,
                    shape,
                    output,
                    batch_tokens,
                    weights,
                    selection.cpu,
                )?,
            };
            let name = match selection.cpu {
                CpuPrecision::Fp32 => {
                    format!("CPU ({kernel} kernels, {} threads)", encoder.threads())
                }
                CpuPrecision::E5CenteredInt8 => format!(
                    "CPU (E5 centered INT8, {kernel} attention, {} threads)",
                    encoder.threads()
                ),
            };
            let cpu_identity =
                cpu_policy_identity(kernel, encoder.numerical_identity(), selection.workers);
            let numerical = policy(&cpu_identity, kodamaembed_cpu::NUMERICAL_SOURCES);
            (Backend::Cpu(Box::new(encoder)), name, numerical)
        }
        Device::Auto => {
            let gpu = auto_gpu(&shape, batch_tokens, |device| {
                open(device, shape, output, batch_tokens, weights, selection)
            })?;
            return match gpu {
                Some(opened) => Ok(opened),
                None => open(Device::Cpu, shape, output, batch_tokens, weights, selection),
            };
        }
    })
}

/// What a loaded model computes.
#[derive(Clone, Copy)]
enum Task {
    /// Sentence vectors, pooled as the model's spec says.
    Embed(Pooling),
    /// Relevance scores, through the head its `config.json` describes.
    Rerank,
}

struct Load<'a> {
    /// `config.json`, `tokenizer.json` and `model.safetensors`, in the order they are read.
    artifacts: [Artifact<'a>; 3],
    task: Task,
    max_length: usize,
    prefixes: (&'a str, &'a str),
    device: Device,
    batch_tokens: usize,
    selection: Selection,
}

/// One of a model's files, read when the loader reaches it.
enum Artifact<'a> {
    /// A file on disk, with a named model's SHA-256 pin.
    File {
        path: PathBuf,
        pinned: Option<&'a str>,
    },
    /// The caller's bytes.
    Bytes(&'a [u8]),
}

impl<'a> Artifact<'a> {
    /// The bytes and their SHA-256, compared with a pinned hash before anything parses them.
    fn read(self) -> Result<(Cow<'a, [u8]>, [u8; 32]), String> {
        match self {
            Self::Bytes(bytes) => Ok((Cow::Borrowed(bytes), Sha256::digest(bytes).into())),
            Self::File { path, pinned } => {
                let bytes =
                    artifact_io::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                let digest: [u8; 32] = Sha256::digest(&bytes).into();
                if let Some(pinned) = pinned {
                    let actual = identity::hex(&digest);
                    if actual != pinned {
                        return Err(registry::mismatch(&path, &actual, pinned));
                    }
                }
                Ok((Cow::Owned(bytes), digest))
            }
        }
    }
}

/// The files of the model in `dir`, each with its pin when the model is a named one.
fn directory<'a>(dir: &Path, pinned: Option<[&'a str; 3]>) -> [Artifact<'a>; 3] {
    let [config, tokenizer, weights] = pinned.map_or([None; 3], |pins| pins.map(Some));
    [
        ("config.json", config),
        ("tokenizer.json", tokenizer),
        ("model.safetensors", weights),
    ]
    .map(|(name, pinned)| Artifact::File {
        path: dir.join(name),
        pinned,
    })
}

impl Model {
    fn load(load: Load<'_>) -> Result<Self, Error> {
        let Load {
            artifacts: [config, tokenizer, weights],
            task,
            max_length,
            prefixes,
            device,
            batch_tokens,
            selection,
        } = load;
        let (config, config_sha256) = config.read().map_err(Error::Model)?;
        let shape = checkpoint::shape_from_bytes(&config).map_err(Error::Model)?;
        let output = match task {
            Task::Embed(pooling) => Output::Embedding(pooling),
            Task::Rerank => {
                Output::Relevance(checkpoint::head_from_bytes(&config).map_err(Error::Model)?)
            }
        };
        if batch_tokens < max_length {
            return Err(Error::Model(format!(
                "a batch of {batch_tokens} tokens cannot hold a {max_length}-token text"
            )));
        }
        let (tokenizer_bytes, tokenizer_sha256) = tokenizer.read().map_err(Error::Tokenizer)?;
        let mut tokenizer =
            Tokenizer::from_bytes(&tokenizer_bytes).map_err(|e| Error::Tokenizer(e.to_string()))?;
        tokenizing::sentencepiece_pipeline(&mut tokenizer);
        if let tokenizers::models::ModelWrapper::BPE(model) = tokenizer.get_model()
            && let Some(dropout) = model.dropout
            && dropout > 0.0
            && dropout < 1.0
        {
            return Err(Error::Tokenizer(format!(
                "BPE dropout {dropout} is stochastic; deterministic inference requires dropout 0, 1, or null"
            )));
        }
        check_vocabulary(&tokenizer, shape.vocab)?;
        let special = tokenizer.get_post_processor().map_or(0, |processor| {
            processor
                .added_tokens(false)
                .max(processor.added_tokens(true))
        });
        check_max_length(max_length, special, shape.max_sequence_len())?;
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length,
                ..Default::default()
            }))
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        let (checkpoint, checkpoint_sha256) = weights.read().map_err(Error::Model)?;
        let weights = checkpoint::SafeTensors::from_bytes(checkpoint).map_err(Error::Model)?;
        let sha256 = [config_sha256, tokenizer_sha256, checkpoint_sha256]
            .map(|digest| identity::hex(&digest));
        precision_scope::check_artifacts(selection.cpu, sha256.each_ref().map(String::as_str))?;
        weights
            .validate_shapes(shape, output)
            .map_err(Error::Model)?;
        let (mut backend, device_name, numerical_policy) =
            open(device, shape, output, batch_tokens, &weights, selection)?;
        let fingerprint = identity::fingerprint(identity::IdentityInputs {
            config_sha256: &config_sha256,
            tokenizer_sha256: &tokenizer_sha256,
            checkpoint_sha256: &checkpoint_sha256,
            output,
            passage_prefix: prefixes.0,
            query_prefix: prefixes.1,
            max_length,
            numerical_policy: &numerical_policy,
        });
        // The engine's own text calibrates, so vectors never depend on the application's text.
        if let Backend::Cpu(encoder) = &mut backend
            && selection.cpu == CpuPrecision::E5CenteredInt8
        {
            calibrate(encoder, &fingerprint, || {
                calibration_inputs(&tokenizer, prefixes)
            })?;
        }
        let cores = thread::available_parallelism().map_or(1, usize::from);
        let tokenizer_workers = selection
            .workers
            .map_or((cores / 4).max(1), |w| w.tokenizer_workers.get());
        let tokenizing = kodamaembed_cpu::worker_pool("kodamaembed-tokenize", tokenizer_workers)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        if selection.workers.is_some() && tokenizing.current_num_threads() != tokenizer_workers {
            return Err(Error::Unsupported(
                "tokenizer worker pool did not honor the requested count".into(),
            ));
        }
        Ok(Self {
            device: Shared::new(backend),
            device_name,
            tokenizer: tokenizing::InferenceTokenizer::new(tokenizer),
            tokenizing,
            hidden: shape.hidden,
            fingerprint,
        })
    }

    fn cpu_workers(&self) -> Option<CpuWorkers> {
        Some(CpuWorkers {
            inference_workers: NonZeroUsize::new(self.device.threads?).expect("has workers"),
            tokenizer_workers: NonZeroUsize::new(self.tokenizing.current_num_threads())
                .expect("has workers"),
        })
    }

    fn run<'s, I, E, P, F, S>(&self, inputs: I, stop: P, sink: F) -> Result<usize, StreamError<S>>
    where
        I: IntoIterator<Item = E>,
        I::IntoIter: Send,
        E: Into<EncodeInput<'s>> + Send,
        P: FnMut() -> bool,
        F: FnMut(Batch<'_>) -> Result<(), S>,
    {
        self.run_tokenized(
            inputs,
            |tokenizer, pool, chunk| tokenizer.encode_batch_in(pool, chunk),
            stop,
            sink,
        )
    }

    /// One vector per text, in order, after `prefix`.
    fn embed(&self, prefix: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
        let mut vectors = Vec::with_capacity(texts.len());
        let collect = |batch: Batch<'_>| {
            vectors.extend(batch.values.chunks_exact(batch.dim).map(<[f32]>::to_vec));
            Ok::<_, std::convert::Infallible>(())
        };
        collect_stream(self.embed_stream(prefix, texts.iter().copied(), || false, collect))?;
        Ok(vectors)
    }

    fn embed_stream<I, T, P, F, E>(
        &self,
        prefix: &str,
        texts: I,
        stop: P,
        sink: F,
    ) -> Result<usize, StreamError<E>>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: Send,
        T: AsRef<str> + Send,
        P: FnMut() -> bool,
        F: FnMut(Batch<'_>) -> Result<(), E>,
    {
        self.run_tokenized(
            texts,
            |tokenizer, pool, chunk| tokenize_texts(tokenizer, pool, prefix, chunk),
            stop,
            sink,
        )
    }

    /// The producer owns each chunk until tokenizing returns IDs: inputs need not be Sync.
    fn run_tokenized<I, E, C, P, F, S>(
        &self,
        inputs: I,
        mut tokenize: C,
        stop: P,
        sink: F,
    ) -> Result<usize, StreamError<S>>
    where
        I: IntoIterator<Item = E>,
        I::IntoIter: Send,
        E: Send,
        C: FnMut(
                &tokenizing::InferenceTokenizer,
                &kodamaembed_cpu::Workers,
                Vec<E>,
            ) -> tokenizers::Result<Vec<tokenizing::InferenceInput>>
            + Send,
        P: FnMut() -> bool,
        F: FnMut(Batch<'_>) -> Result<(), S>,
    {
        let mut inputs = inputs.into_iter().peekable();
        let (tokenizer, tokenizing, device) = (&self.tokenizer, &self.tokenizing, &self.device);
        let first: Vec<E> = inputs.by_ref().take(CHUNK).collect();
        if first.is_empty() {
            return Ok(0);
        }
        if inputs.peek().is_none() {
            // A single chunk has nothing to overlap: a producer thread would only add its startup.
            let encoded = tokenize(tokenizer, tokenizing, first);
            return Self::run_encoded(device, std::iter::once(encoded), stop, sink);
        }
        #[cfg(target_family = "wasm")]
        return Self::run_encoded(
            device,
            tokenized_on_demand(first, inputs, |chunk| {
                tokenize(tokenizer, tokenizing, chunk)
            }),
            stop,
            sink,
        );
        #[cfg(not(target_family = "wasm"))]
        thread::scope(|scope| {
            // Bounded lookahead; a dropped receiver wakes a blocked producer before the join.
            let (chunks, tokenized) = std::sync::mpsc::sync_channel(2);
            scope.spawn(move || {
                let mut chunk = first;
                while !chunk.is_empty() {
                    let encoded = tokenize(tokenizer, tokenizing, chunk);
                    let failed = encoded.is_err();
                    if chunks.send(encoded).is_err() || failed {
                        break;
                    }
                    chunk = inputs.by_ref().take(CHUNK).collect();
                }
            });
            Self::run_encoded(device, tokenized.into_iter(), stop, sink)
        })
    }

    fn run_encoded<I, P, F, S>(
        device: &Shared,
        tokenized: I,
        mut stop: P,
        mut sink: F,
    ) -> Result<usize, StreamError<S>>
    where
        I: Iterator<Item = tokenizers::Result<Vec<tokenizing::InferenceInput>>>,
        P: FnMut() -> bool,
        F: FnMut(Batch<'_>) -> Result<(), S>,
    {
        let mut completed = 0;
        let (mut ids, mut types, mut lengths) = (Vec::new(), Vec::new(), Vec::new());
        let limits = device.limits;
        let mut encode = |completed: usize,
                          ids: &[i32],
                          types: &[i32],
                          lengths: &[usize]|
         -> Result<(), StreamError<S>> {
            // The caller's closures run off its turn: they may wait, or use the model.
            if stop() {
                return Err(Error::Interrupted.into());
            }
            let values = {
                let _turn = device.turn();
                device.backend()?.encode(ids, types, lengths)?
            };
            sink(Batch {
                start: completed,
                dim: values.len() / lengths.len(),
                tokens: ids.len(),
                values: &values,
            })
            .map_err(StreamError::Sink)
        };
        for encodings in tokenized {
            for encoding in encodings.map_err(|e| Error::Tokenizer(e.to_string()))? {
                let len = encoding.get_ids().len();
                if len == 0 {
                    return Err(Error::Tokenizer(
                        "an input produced no tokens; the encoder requires nonempty sequences"
                            .into(),
                    )
                    .into());
                }
                if len > limits.tokens {
                    return Err(Error::Tokenizer(format!(
                        "a {len}-token input exceeds the backend's {}-token batch",
                        limits.tokens
                    ))
                    .into());
                }
                if limits.full(ids.len(), lengths.len(), len) {
                    encode(completed, &ids, &types, &lengths)?;
                    completed += lengths.len();
                    ids.clear();
                    types.clear();
                    lengths.clear();
                }
                let signed = |value: &u32| {
                    i32::try_from(*value).map_err(|_| {
                        Error::Tokenizer(format!(
                            "token id or type {value} exceeds signed 32-bit IDs"
                        ))
                    })
                };
                for value in encoding.get_ids() {
                    ids.push(signed(value)?);
                }
                for value in encoding.get_type_ids() {
                    types.push(signed(value)?);
                }
                lengths.push(len);
            }
        }
        if !lengths.is_empty() {
            encode(completed, &ids, &types, &lengths)?;
            completed += lengths.len();
        }
        Ok(completed)
    }
}

/// Each chunk tokenized when inference asks for it, where no thread can tokenize ahead.
#[cfg(any(test, target_family = "wasm"))]
fn tokenized_on_demand<E>(
    first: Vec<E>,
    mut inputs: impl Iterator<Item = E>,
    mut tokenize: impl FnMut(Vec<E>) -> tokenizers::Result<Vec<tokenizing::InferenceInput>>,
) -> impl Iterator<Item = tokenizers::Result<Vec<tokenizing::InferenceInput>>> {
    let mut first = Some(first);
    std::iter::from_fn(move || {
        let chunk = first
            .take()
            .unwrap_or_else(|| inputs.by_ref().take(CHUNK).collect());
        (!chunk.is_empty()).then(|| tokenize(chunk))
    })
}

fn tokenize_texts<T: AsRef<str>>(
    tokenizer: &tokenizing::InferenceTokenizer,
    pool: &kodamaembed_cpu::Workers,
    prefix: &str,
    chunk: Vec<T>,
) -> tokenizers::Result<Vec<tokenizing::InferenceInput>> {
    if prefix.is_empty() {
        let texts: Vec<&str> = chunk.iter().map(AsRef::as_ref).collect();
        tokenizer.encode_batch_in(pool, texts)
    } else {
        tokenizer.encode_batch_in(
            pool,
            chunk
                .into_iter()
                .map(|text| format!("{prefix}{}", text.as_ref()))
                .collect::<Vec<_>>(),
        )
    }
}

/// A sentence embedder: one L2-normalized vector per text, pooled as its
/// [`ModelSpec`] says. Threads share one: their calls take turns batch by batch.
pub struct Embedder {
    model: Model,
    passage_prefix: Box<str>,
    query_prefix: Box<str>,
}

impl Embedder {
    /// Loads `model` with [`Self::builder`]'s defaults, downloading it into
    /// the Hugging Face cache on first use.
    pub fn new(model: EmbeddingModel) -> Result<Self, Error> {
        Self::builder().build(model)
    }

    /// Options for loading an embedder.
    pub fn builder() -> EmbedderBuilder {
        EmbedderBuilder::default()
    }

    /// The inference and tokenizer worker counts on the CPU, or None on a GPU.
    pub fn cpu_workers(&self) -> Option<CpuWorkers> {
        self.model.cpu_workers()
    }

    /// Values per vector: the model's hidden width.
    pub fn dim(&self) -> usize {
        self.model.hidden
    }

    /// SHA-256 of the model files, prefixes, truncation, output mode and
    /// numerical implementation.
    pub fn fingerprint(&self) -> &str {
        &self.model.fingerprint
    }

    /// The device and backend the encoder runs on.
    pub fn device_name(&self) -> &str {
        &self.model.device_name
    }

    /// The selected GPU; an error on the CPU or when the driver has no UUID.
    pub fn gpu_identity(&self) -> Result<GpuIdentity, Error> {
        self.model.device.backend()?.gpu_identity()
    }

    /// Times every stage of later batches on CUDA and the CPU.
    pub fn enable_profiling(&mut self) -> Result<(), Error> {
        // The standard library has no clock there, and reading it panics.
        #[cfg(all(target_family = "wasm", target_os = "unknown"))]
        return Err(Error::Unsupported(
            "profiling needs a clock, which this target lacks".into(),
        ));
        #[cfg(not(all(target_family = "wasm", target_os = "unknown")))]
        match self.model.device.backend_mut()? {
            Backend::Cuda(encoder) => Ok(encoder.enable_profiling()?),
            Backend::Cpu(encoder) => {
                encoder.enable_profiling();
                Ok(())
            }
            Backend::Vulkan(_) => Err(Error::Unsupported(
                "profiling needs the FP16 CUDA or the CPU backend".into(),
            )),
            #[cfg(target_arch = "x86_64")]
            Backend::CudaCentered { .. } => Err(Error::Unsupported(
                "profiling needs the FP16 CUDA or the CPU backend".into(),
            )),
        }
    }

    /// Time per stage of the batches since profiling was enabled.
    pub fn profile(&self) -> Option<Profile> {
        match &*self.model.device.backend().ok()? {
            Backend::Cuda(encoder) => encoder.profile().cloned(),
            Backend::Cpu(encoder) => encoder.profile().cloned(),
            Backend::Vulkan(_) => None,
            #[cfg(target_arch = "x86_64")]
            Backend::CudaCentered { .. } => None,
        }
    }

    /// One vector per passage, in order, after the spec's passage prefix.
    pub fn embed_passages(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
        self.model.embed(&self.passage_prefix, texts)
    }

    /// One vector per query, in order, after the spec's query prefix.
    pub fn embed_queries(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
        self.model.embed(&self.query_prefix, texts)
    }

    /// Embeds a lazy passage iterator, handing `sink` completed batches in
    /// order; a sink error stops inference and is returned.
    pub fn embed_passages_stream<I, T, F, E>(
        &self,
        texts: I,
        sink: F,
    ) -> Result<usize, StreamError<E>>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: Send,
        T: AsRef<str> + Send,
        F: FnMut(Batch<'_>) -> Result<(), E>,
    {
        self.embed_passages_until(texts, || false, sink)
    }

    /// Like [`Self::embed_passages_stream`], with the model's query prefix.
    pub fn embed_queries_stream<I, T, F, E>(
        &self,
        texts: I,
        sink: F,
    ) -> Result<usize, StreamError<E>>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: Send,
        T: AsRef<str> + Send,
        F: FnMut(Batch<'_>) -> Result<(), E>,
    {
        self.embed_queries_until(texts, || false, sink)
    }

    /// Like [`Self::embed_passages_stream`], asking `stop` before each batch is
    /// encoded: `true` ends the run with [`Error::Interrupted`].
    pub fn embed_passages_until<I, T, P, F, E>(
        &self,
        texts: I,
        stop: P,
        sink: F,
    ) -> Result<usize, StreamError<E>>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: Send,
        T: AsRef<str> + Send,
        P: FnMut() -> bool,
        F: FnMut(Batch<'_>) -> Result<(), E>,
    {
        self.model
            .embed_stream(&self.passage_prefix, texts, stop, sink)
    }

    /// Like [`Self::embed_passages_until`], with the model's query prefix.
    pub fn embed_queries_until<I, T, P, F, E>(
        &self,
        texts: I,
        stop: P,
        sink: F,
    ) -> Result<usize, StreamError<E>>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: Send,
        T: AsRef<str> + Send,
        P: FnMut() -> bool,
        F: FnMut(Batch<'_>) -> Result<(), E>,
    {
        self.model
            .embed_stream(&self.query_prefix, texts, stop, sink)
    }
}

/// Options for an [`Embedder`]; the `build` methods choose where its files
/// come from.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct EmbedderBuilder {
    device: Device,
    precision: Precision,
    batch_tokens: usize,
    workers: Option<CpuWorkers>,
}

impl Default for EmbedderBuilder {
    fn default() -> Self {
        Self {
            device: Device::Auto,
            precision: Precision::Float,
            batch_tokens: BATCH_TOKENS,
            workers: None,
        }
    }
}

impl EmbedderBuilder {
    /// Where the encoder runs; [`Device::Auto`] unless set.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// The numerical mode; [`Precision::Float`] unless set.
    pub fn precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }

    /// Tokens one encoder batch holds, at least the model's maximum length;
    /// 16,384 unless set. Fewer tokens allocate less working memory.
    pub fn batch_tokens(mut self, tokens: usize) -> Self {
        self.batch_tokens = tokens;
        self
    }

    /// Explicit inference and tokenizer worker counts on the processor; the counts enter
    /// [`Embedder::fingerprint`], and `RAYON_NUM_THREADS` does not apply.
    pub fn workers(mut self, workers: CpuWorkers) -> Self {
        self.workers = Some(workers);
        self
    }

    /// Loads `model` from the Hugging Face cache, downloading it on first
    /// use; see [`Hub::from_env`].
    pub fn build(self, model: EmbeddingModel) -> Result<Embedder, Error> {
        self.build_from_hub(&Hub::from_env()?, model)
    }

    /// Loads `model` from `hub`, checking every file against its pinned hash.
    pub fn build_from_hub(self, hub: &Hub, model: EmbeddingModel) -> Result<Embedder, Error> {
        let entry = model.entry();
        let pins = entry.pins();
        let plan = self.plan(model.spec(), Some(pins))?;
        self.load(plan, directory(&hub.locate(entry)?, Some(pins)))
    }

    /// Loads a BERT, XLM-R, MPNet or ModernBERT checkpoint from `dir` (`config.json`,
    /// `tokenizer.json`, F32, F16 or BF16 `model.safetensors`), pooled as `spec` says.
    pub fn build_from_dir(
        self,
        dir: impl AsRef<Path>,
        spec: ModelSpec<'_>,
    ) -> Result<Embedder, Error> {
        let plan = self.plan(spec, None)?;
        self.load(plan, directory(dir.as_ref(), None))
    }

    /// Like [`Self::build_from_dir`], from the contents of the three files: a model held in
    /// memory needs no directory.
    pub fn build_from_bytes(
        self,
        config: &[u8],
        tokenizer: &[u8],
        weights: &[u8],
        spec: ModelSpec<'_>,
    ) -> Result<Embedder, Error> {
        let plan = self.plan(spec, None)?;
        self.load(plan, [config, tokenizer, weights].map(Artifact::Bytes))
    }

    /// Resolves the settings before any file is read: settings that conflict download nothing.
    fn plan<'a>(&self, spec: ModelSpec<'a>, pinned: Option<[&str; 3]>) -> Result<Plan<'a>, Error> {
        if let Some(workers) = self.workers {
            workers.validate()?;
        }
        let route = route(self.device, self.precision, self.workers.is_some())?;
        let (device, cpu) = match route {
            Route::Float(device) => (device, CpuPrecision::Fp32),
            Route::CenteredCpu | Route::CenteredCuda(_) | Route::CenteredAuto => {
                (Device::Cpu, precision_scope::centered_preset(spec)?)
            }
        };
        if let Some(pins) = pinned {
            precision_scope::check_artifacts(cpu, pins)?;
        }
        Ok(Plan {
            route,
            device,
            cpu,
            spec,
        })
    }

    fn load(self, plan: Plan<'_>, artifacts: [Artifact<'_>; 3]) -> Result<Embedder, Error> {
        let Plan {
            route,
            device,
            cpu,
            spec,
        } = plan;
        let mut model = Model::load(Load {
            artifacts,
            task: Task::Embed(spec.pooling),
            max_length: spec.max_length,
            prefixes: (spec.passage_prefix, spec.query_prefix),
            device,
            batch_tokens: self.batch_tokens,
            selection: Selection {
                cpu,
                workers: self.workers,
            },
        })?;
        let prepared = model.device.backend_mut()?;
        let cuda = match route {
            Route::CenteredCuda(ordinal) => {
                Some(centered_cuda(prepared, ordinal, self.batch_tokens)?)
            }
            Route::CenteredAuto => {
                // Without a usable CUDA GPU the processor gives the same arithmetic and vectors.
                let ordinals = ampere_cuda()?.then_some(0);
                first_usable(ordinals, |ordinal| {
                    centered_cuda(prepared, ordinal, self.batch_tokens)
                })?
            }
            Route::Float(_) | Route::CenteredCpu => None,
        };
        if let Some((backend, name)) = cuda {
            (model.device, model.device_name) = (Shared::new(backend), name);
        }
        Ok(Embedder {
            model,
            passage_prefix: spec.passage_prefix.into(),
            query_prefix: spec.query_prefix.into(),
        })
    }
}

struct Plan<'a> {
    route: Route,
    device: Device,
    cpu: CpuPrecision,
    spec: ModelSpec<'a>,
}

/// A CUDA backend reproducing the preset the CPU prepared; vectors and fingerprint stay the CPU's.
#[cfg(target_arch = "x86_64")]
fn centered_cuda(
    prepared: &Backend,
    ordinal: usize,
    batch_tokens: usize,
) -> Result<(Backend, String), Error> {
    let Backend::Cpu(cpu) = prepared else {
        return Err(Error::Unsupported(
            "a centered INT8 preset starts from a CPU encoder".into(),
        ));
    };
    let export = cpu.export_reproducible()?;
    let context =
        kodamaembed_cuda::Context::new(ordinal).map_err(kodamaembed_cuda::EncoderError::from)?;
    let encoder = kodamaembed_cuda::repro::ReproEncoder::new(&context, &export)
        .map_err(kodamaembed_cuda::EncoderError::from)?;
    let backend = Backend::CudaCentered {
        encoder: Box::new(encoder),
        max_tokens: batch_tokens,
    };
    Ok((backend, format!("{} (centered INT8)", context.name)))
}

#[cfg(not(target_arch = "x86_64"))]
fn centered_cuda(
    _prepared: &Backend,
    _ordinal: usize,
    _batch_tokens: usize,
) -> Result<(Backend, String), Error> {
    Err(Error::Unsupported(
        "quantized precision prepares its preset on an x86-64 processor".into(),
    ))
}

/// Options for a [`Reranker`]; the `build` methods choose where its files
/// come from.
#[derive(Debug, Clone, Copy)]
#[must_use]
pub struct RerankerBuilder {
    device: Device,
    batch_tokens: usize,
    workers: Option<CpuWorkers>,
}

impl Default for RerankerBuilder {
    fn default() -> Self {
        Self {
            device: Device::Auto,
            batch_tokens: BATCH_TOKENS,
            workers: None,
        }
    }
}

impl RerankerBuilder {
    /// Where the encoder runs; [`Device::Auto`] unless set.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Tokens one encoder batch holds, at least the pairs' maximum length;
    /// 16,384 unless set.
    pub fn batch_tokens(mut self, tokens: usize) -> Self {
        self.batch_tokens = tokens;
        self
    }

    /// Explicit worker counts on the processor, as [`EmbedderBuilder::workers`].
    pub fn workers(mut self, workers: CpuWorkers) -> Self {
        self.workers = Some(workers);
        self
    }

    /// Loads `model` from the Hugging Face cache, downloading it on first
    /// use; see [`Hub::from_env`].
    pub fn build(self, model: RerankerModel) -> Result<Reranker, Error> {
        self.build_from_hub(&Hub::from_env()?, model)
    }

    /// Loads `model` from `hub`, checking every file against its pinned hash.
    pub fn build_from_hub(self, hub: &Hub, model: RerankerModel) -> Result<Reranker, Error> {
        let device = self.plan()?;
        let entry = model.entry();
        self.load(
            device,
            directory(&hub.locate(entry)?, Some(entry.pins())),
            model.max_length(),
        )
    }

    /// Loads a one-label sequence-classification checkpoint, like
    /// `cross-encoder/ms-marco-MiniLM-L6-v2`, from `dir`; pairs truncate to `max_length`.
    pub fn build_from_dir(
        self,
        dir: impl AsRef<Path>,
        max_length: usize,
    ) -> Result<Reranker, Error> {
        let device = self.plan()?;
        self.load(device, directory(dir.as_ref(), None), max_length)
    }

    /// Like [`Self::build_from_dir`], from the contents of the three files: a model held in
    /// memory needs no directory.
    pub fn build_from_bytes(
        self,
        config: &[u8],
        tokenizer: &[u8],
        weights: &[u8],
        max_length: usize,
    ) -> Result<Reranker, Error> {
        let device = self.plan()?;
        self.load(
            device,
            [config, tokenizer, weights].map(Artifact::Bytes),
            max_length,
        )
    }

    /// The device, resolved before any file is read, as for an embedder.
    fn plan(&self) -> Result<Device, Error> {
        if let Some(workers) = self.workers {
            workers.validate()?;
        }
        float_device(self.device, self.workers.is_some())
    }

    fn load(
        self,
        device: Device,
        artifacts: [Artifact<'_>; 3],
        max_length: usize,
    ) -> Result<Reranker, Error> {
        Ok(Reranker {
            model: Model::load(Load {
                artifacts,
                task: Task::Rerank,
                max_length,
                prefixes: ("", ""),
                device,
                batch_tokens: self.batch_tokens,
                selection: Selection {
                    cpu: CpuPrecision::Fp32,
                    workers: self.workers,
                },
            })?,
        })
    }
}

/// A cross-encoder: one relevance score per (query, passage) pair. Threads share one, as an
/// [`Embedder`].
pub struct Reranker {
    model: Model,
}

impl Reranker {
    /// Loads `model` with [`Self::builder`]'s defaults, downloading it into
    /// the Hugging Face cache on first use.
    pub fn new(model: RerankerModel) -> Result<Self, Error> {
        Self::builder().build(model)
    }

    /// Options for loading a reranker.
    pub fn builder() -> RerankerBuilder {
        RerankerBuilder::default()
    }

    /// The inference and tokenizer worker counts on the CPU, or None on a GPU.
    pub fn cpu_workers(&self) -> Option<CpuWorkers> {
        self.model.cpu_workers()
    }

    /// SHA-256 of the model files, truncation and numerical implementation.
    pub fn fingerprint(&self) -> &str {
        &self.model.fingerprint
    }

    /// The device and backend the encoder runs on.
    pub fn device_name(&self) -> &str {
        &self.model.device_name
    }

    /// The selected GPU; an error on the CPU or when the driver has no UUID.
    pub fn gpu_identity(&self) -> Result<GpuIdentity, Error> {
        self.model.device.backend()?.gpu_identity()
    }

    /// The relevance logit of each passage to `query`, in the passages' order.
    pub fn score(&self, query: &str, passages: &[&str]) -> Result<Vec<f32>, Error> {
        let mut scores = Vec::with_capacity(passages.len());
        collect_stream(self.score_stream(query, passages.iter().copied(), |batch| {
            scores.extend_from_slice(batch.values);
            Ok::<_, std::convert::Infallible>(())
        }))?;
        Ok(scores)
    }

    /// Scores a lazy passage iterator, handing `sink` ordered batches of one
    /// logit a pair; a sink error stops work and is returned.
    pub fn score_stream<I, T, F, E>(
        &self,
        query: &str,
        passages: I,
        sink: F,
    ) -> Result<usize, StreamError<E>>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: Send,
        T: AsRef<str> + Send,
        F: FnMut(Batch<'_>) -> Result<(), E>,
    {
        self.score_until(query, passages, || false, sink)
    }

    /// Like [`Self::score_stream`], asking `stop` before each batch is encoded:
    /// `true` ends the run with [`Error::Interrupted`].
    pub fn score_until<I, T, P, F, E>(
        &self,
        query: &str,
        passages: I,
        stop: P,
        sink: F,
    ) -> Result<usize, StreamError<E>>
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: Send,
        T: AsRef<str> + Send,
        P: FnMut() -> bool,
        F: FnMut(Batch<'_>) -> Result<(), E>,
    {
        // Own each pair only while tokenizing its chunk; no corpus-sized copy.
        self.model.run(
            passages
                .into_iter()
                .map(|passage| (query.to_owned(), passage.as_ref().to_owned())),
            stop,
            sink,
        )
    }
}

fn collect_stream(
    result: Result<usize, StreamError<std::convert::Infallible>>,
) -> Result<usize, Error> {
    result.map_err(|error| match error {
        StreamError::Inference(error) => error,
        StreamError::Sink(never) => match never {},
    })
}

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;
