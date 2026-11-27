//! The `kodamaembed._kodamaembed` extension module, which the `kodamaembed` Python package wraps.

use kodamaembed::{
    Batch, CpuWorkers, Device, Embedder, EmbeddingModel, Error, Hub, Precision, Reranker,
    RerankerModel, StreamError,
};
use pyo3::{
    create_exception,
    exceptions::{PyException, PyValueError},
    prelude::*,
    types::PyByteArray,
};
use std::{convert::Infallible, num::NonZeroUsize, path::PathBuf};

// Named after the package that exports them, so tracebacks show the public name.
create_exception!(
    kodamaembed,
    KodamaEmbedError,
    PyException,
    "Loading or running a model failed."
);
create_exception!(
    kodamaembed,
    DownloadError,
    KodamaEmbedError,
    "Finding or downloading a model failed."
);
create_exception!(
    kodamaembed,
    ModelError,
    KodamaEmbedError,
    "A model file or the tokenizer failed."
);
create_exception!(
    kodamaembed,
    UnsupportedError,
    KodamaEmbedError,
    "The model, device or setting is not supported."
);

/// The exception's class names the category that `Error`'s display prefixes.
fn exception(error: Error) -> PyErr {
    match error {
        Error::Download(reason) => DownloadError::new_err(reason),
        Error::Model(reason) => ModelError::new_err(reason),
        Error::Unsupported(reason) => UnsupportedError::new_err(reason),
        error @ Error::Tokenizer(_) => ModelError::new_err(error.to_string()),
        error => KodamaEmbedError::new_err(error.to_string()),
    }
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct Source {
    cache_dir: Option<PathBuf>,
    offline: Option<bool>,
    progress: Option<Py<PyAny>>,
}

impl Source {
    fn hub(self) -> PyResult<Hub> {
        let mut hub = Hub::from_env().map_err(exception)?;
        if let Some(cache) = self.cache_dir {
            hub = hub.with_cache(cache);
        }
        if let Some(offline) = self.offline {
            hub = hub.offline(offline);
        }
        if let Some(callback) = self.progress {
            hub = hub.on_progress(move |progress| {
                Python::attach(|py| {
                    let args = (
                        progress.repository,
                        progress.file,
                        progress.received,
                        progress.total,
                    );
                    // A failing progress display must not stop the download.
                    if let Err(error) = callback.call1(py, args) {
                        error.write_unraisable(py, Some(callback.bind(py)));
                    }
                });
            });
        }
        Ok(hub)
    }
}

#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct Run {
    device: String,
    batch_tokens: Option<usize>,
    threads: Option<usize>,
}

impl Run {
    fn device(&self) -> PyResult<Device> {
        let invalid = || {
            PyValueError::new_err(format!(
                "device must be auto, cpu, cuda, cuda:N, vulkan or vulkan:N, not {:?}",
                self.device
            ))
        };
        let (kind, index) = match self.device.split_once(':') {
            Some((kind, index)) => (kind, Some(index.parse().map_err(|_| invalid())?)),
            None => (self.device.as_str(), None),
        };
        match (kind, index) {
            ("auto", None) => Ok(Device::Auto),
            ("cpu", None) => Ok(Device::Cpu),
            ("cuda", index) => Ok(Device::Cuda(index.unwrap_or(0))),
            ("vulkan", Some(index)) => Ok(Device::Vulkan(index)),
            ("vulkan", None) => Device::first_discrete_vulkan().map_err(exception),
            _ => Err(invalid()),
        }
    }

    /// The tokenizer pool gets a quarter of the threads, as the default pools do.
    fn workers(&self) -> PyResult<Option<CpuWorkers>> {
        self.threads
            .map(|threads| {
                let inference_workers = NonZeroUsize::new(threads)
                    .ok_or_else(|| PyValueError::new_err("threads must be positive"))?;
                let tokenizer_workers = NonZeroUsize::new(threads / 4).unwrap_or(NonZeroUsize::MIN);
                Ok(CpuWorkers {
                    inference_workers,
                    tokenizer_workers,
                })
            })
            .transpose()
    }
}

fn precision(name: &str) -> PyResult<Precision> {
    match name {
        "float" => Ok(Precision::Float),
        "quantized" => Ok(Precision::Quantized),
        _ => Err(PyValueError::new_err(format!(
            "precision must be float or quantized, not {name:?}"
        ))),
    }
}

/// The rows of every batch in order, and the tokens they encoded.
struct Collected {
    values: Vec<f32>,
    tokens: usize,
}

impl Collected {
    fn add(&mut self, batch: Batch<'_>) -> Result<(), Infallible> {
        self.values.extend_from_slice(batch.values);
        self.tokens += batch.tokens;
        Ok(())
    }

    fn finish(self, result: Result<usize, StreamError<Infallible>>) -> Result<Self, Error> {
        match result {
            Ok(_) => Ok(self),
            Err(StreamError::Inference(error)) => Err(error),
            Err(StreamError::Sink(never)) => match never {},
        }
    }

    /// Native-endian `f32` bytes, which numpy reads without copying.
    fn into_python(self, py: Python<'_>) -> PyResult<(Bound<'_, PyByteArray>, usize)> {
        let bytes = PyByteArray::new_with(py, self.values.len() * 4, |buffer| {
            let (words, _) = buffer.as_chunks_mut::<4>();
            for (word, value) in words.iter_mut().zip(&self.values) {
                *word = value.to_ne_bytes();
            }
            Ok(())
        })?;
        Ok((bytes, self.tokens))
    }
}

enum Role {
    Passage,
    Query,
}

/// A loaded embedder; `kodamaembed.Embedder` wraps it.
#[pyclass(module = "kodamaembed._kodamaembed", name = "Embedder", frozen)]
struct NativeEmbedder {
    inner: Embedder,
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    dim: usize,
    #[pyo3(get)]
    device: String,
    #[pyo3(get)]
    fingerprint: String,
}

#[pymethods]
impl NativeEmbedder {
    #[new]
    fn new(
        py: Python<'_>,
        model: &str,
        precision: &str,
        run: Run,
        source: Source,
    ) -> PyResult<Self> {
        let model: EmbeddingModel = model.parse().map_err(exception)?;
        let mut builder = Embedder::builder()
            .device(run.device()?)
            .precision(self::precision(precision)?);
        if let Some(tokens) = run.batch_tokens {
            builder = builder.batch_tokens(tokens);
        }
        if let Some(workers) = run.workers()? {
            builder = builder.workers(workers);
        }
        let hub = source.hub()?;
        let embedder = py
            .detach(|| builder.build_from_hub(&hub, model))
            .map_err(exception)?;
        Ok(Self {
            name: model.repository().into(),
            dim: embedder.dim(),
            device: embedder.device_name().into(),
            fingerprint: embedder.fingerprint().into(),
            inner: embedder,
        })
    }

    fn embed_passages<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> PyResult<(Bound<'py, PyByteArray>, usize)> {
        self.embed(py, &texts, Role::Passage)
    }

    fn embed_queries<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<String>,
    ) -> PyResult<(Bound<'py, PyByteArray>, usize)> {
        self.embed(py, &texts, Role::Query)
    }
}

impl NativeEmbedder {
    fn embed<'py>(
        &self,
        py: Python<'py>,
        texts: &[String],
        role: Role,
    ) -> PyResult<(Bound<'py, PyByteArray>, usize)> {
        let collected = py.detach(|| {
            let mut collected = Collected {
                values: Vec::with_capacity(texts.len() * self.dim),
                tokens: 0,
            };
            let sink = |batch: Batch<'_>| collected.add(batch);
            let result = match role {
                Role::Passage => self.inner.embed_passages_stream(texts, sink),
                Role::Query => self.inner.embed_queries_stream(texts, sink),
            };
            collected.finish(result).map_err(exception)
        })?;
        collected.into_python(py)
    }
}

/// A loaded cross-encoder; `kodamaembed.Reranker` wraps it.
#[pyclass(module = "kodamaembed._kodamaembed", name = "Reranker", frozen)]
struct NativeReranker {
    inner: Reranker,
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    device: String,
    #[pyo3(get)]
    fingerprint: String,
}

#[pymethods]
impl NativeReranker {
    #[new]
    fn new(py: Python<'_>, model: &str, run: Run, source: Source) -> PyResult<Self> {
        let model: RerankerModel = model.parse().map_err(exception)?;
        let mut builder = Reranker::builder().device(run.device()?);
        if let Some(tokens) = run.batch_tokens {
            builder = builder.batch_tokens(tokens);
        }
        if let Some(workers) = run.workers()? {
            builder = builder.workers(workers);
        }
        let hub = source.hub()?;
        let reranker = py
            .detach(|| builder.build_from_hub(&hub, model))
            .map_err(exception)?;
        Ok(Self {
            name: model.repository().into(),
            device: reranker.device_name().into(),
            fingerprint: reranker.fingerprint().into(),
            inner: reranker,
        })
    }

    fn score<'py>(
        &self,
        py: Python<'py>,
        query: &str,
        documents: Vec<String>,
    ) -> PyResult<(Bound<'py, PyByteArray>, usize)> {
        let collected = py.detach(|| {
            let mut collected = Collected {
                values: Vec::with_capacity(documents.len()),
                tokens: 0,
            };
            let result = self
                .inner
                .score_stream(query, &documents, |batch| collected.add(batch));
            collected.finish(result).map_err(exception)
        })?;
        collected.into_python(py)
    }
}

/// (name, kind, dim, max_length, download_size, revision), in `kodamaembed.ModelInfo`'s order.
type ModelRow = (
    &'static str,
    &'static str,
    Option<usize>,
    usize,
    u64,
    &'static str,
);

/// Every model that loads by name.
#[pyfunction]
fn models() -> Vec<ModelRow> {
    let embedders = EmbeddingModel::ALL.iter().map(|model| {
        (
            model.repository(),
            "embedding",
            Some(model.dim()),
            model.spec().max_length,
            model.download_size(),
            model.revision(),
        )
    });
    let rerankers = RerankerModel::ALL.iter().map(|model| {
        (
            model.repository(),
            "reranker",
            None,
            model.max_length(),
            model.download_size(),
            model.revision(),
        )
    });
    embedders.chain(rerankers).collect()
}

/// Downloads a model by name and returns the directory of its checked files.
#[pyfunction]
fn download(py: Python<'_>, model: &str, source: Source) -> PyResult<PathBuf> {
    let hub = source.hub()?;
    let fetched = match (
        model.parse::<EmbeddingModel>(),
        model.parse::<RerankerModel>(),
    ) {
        (Ok(embedder), _) => py.detach(|| embedder.download(&hub)),
        (_, Ok(reranker)) => py.detach(|| reranker.download(&hub)),
        (Err(_), Err(_)) => {
            let names: Vec<&str> = models().into_iter().map(|row| row.0).collect();
            return Err(UnsupportedError::new_err(format!(
                "no model named {model}; KodamaEmbed loads {}",
                names.join(", ")
            )));
        }
    };
    fetched.map_err(exception)
}

#[pymodule]
fn _kodamaembed(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_class::<NativeEmbedder>()?;
    module.add_class::<NativeReranker>()?;
    module.add_function(wrap_pyfunction!(models, module)?)?;
    module.add_function(wrap_pyfunction!(download, module)?)?;
    module.add("KodamaEmbedError", py.get_type::<KodamaEmbedError>())?;
    module.add("DownloadError", py.get_type::<DownloadError>())?;
    module.add("ModelError", py.get_type::<ModelError>())?;
    module.add("UnsupportedError", py.get_type::<UnsupportedError>())?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
