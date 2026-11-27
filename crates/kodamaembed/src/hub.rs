//! The Hugging Face hub cache that Python tools share, and downloads into it.

use crate::{
    Error, artifact_io, identity,
    registry::{self, Entry, Pinned},
};
use sha2::{Digest, Sha256};
use std::{
    ffi::{OsStr, OsString},
    fs, io,
    path::{Path, PathBuf},
};
#[cfg(feature = "download")]
use std::{fmt, sync::Arc};

#[cfg(feature = "download")]
const ENDPOINT: &str = "https://huggingface.co";

/// How far one file's download has come, reported as its bytes arrive.
#[cfg(feature = "download")]
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Progress<'a> {
    /// The repository, such as `intfloat/e5-large`.
    pub repository: &'a str,
    /// The file being downloaded, such as `model.safetensors`.
    pub file: &'a str,
    /// Bytes received so far.
    pub received: u64,
    /// The file's pinned size in bytes.
    pub total: u64,
}

#[cfg(feature = "download")]
#[derive(Clone)]
struct Observer(Arc<dyn Fn(&Progress<'_>) + Send + Sync>);

#[cfg(feature = "download")]
impl fmt::Debug for Observer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Observer")
    }
}

/// Where models that load by name are cached, and whether missing files may
/// download.
#[derive(Debug, Clone)]
pub struct Hub {
    cache: PathBuf,
    #[cfg(feature = "download")]
    endpoint: String,
    #[cfg(feature = "download")]
    observer: Option<Observer>,
    offline: bool,
}

impl Hub {
    /// The `huggingface_hub` cache: `HF_HUB_CACHE`, `HF_HOME/hub`, `XDG_CACHE_HOME/huggingface/hub`
    /// or `~/.cache/huggingface/hub`; `HF_HUB_OFFLINE` forbids downloads, `HF_ENDPOINT` names a mirror.
    pub fn from_env() -> Result<Self, Error> {
        Self::from_vars(|name| std::env::var_os(name), std::env::home_dir())
    }

    fn from_vars(
        var: impl Fn(&str) -> Option<OsString>,
        home: Option<PathBuf>,
    ) -> Result<Self, Error> {
        let path = |name| {
            var(name)
                .filter(|value| !value.is_empty())
                .map(|value| expand_home(PathBuf::from(value), home.as_deref()))
        };
        let cache = path("HF_HUB_CACHE")
            .or_else(|| path("HUGGINGFACE_HUB_CACHE"))
            .or_else(|| path("HF_HOME").map(|hf| hf.join("hub")))
            .or_else(|| path("XDG_CACHE_HOME").map(|xdg| xdg.join("huggingface").join("hub")))
            .or_else(|| {
                let home = home.as_ref()?;
                Some(home.join(".cache").join("huggingface").join("hub"))
            })
            .ok_or_else(|| {
                Error::Download(
                    "no home directory for the cache; set HF_HUB_CACHE or HF_HOME".into(),
                )
            })?;
        // Match huggingface_hub's precedence when HF_HUB_OFFLINE is empty.
        let offline = var("HF_HUB_OFFLINE")
            .filter(|value| !value.is_empty())
            .or_else(|| var("TRANSFORMERS_OFFLINE"))
            .is_some_and(|value| truthy(&value));
        Ok(Self {
            cache,
            #[cfg(feature = "download")]
            endpoint: endpoint(&var),
            #[cfg(feature = "download")]
            observer: None,
            offline,
        })
    }

    /// A hub cache directory, such as `~/.cache/huggingface/hub`, with
    /// downloads from huggingface.co allowed.
    pub fn new(cache: impl Into<PathBuf>) -> Self {
        Self {
            cache: cache.into(),
            #[cfg(feature = "download")]
            endpoint: ENDPOINT.into(),
            #[cfg(feature = "download")]
            observer: None,
            offline: false,
        }
    }

    /// The same settings with models in `cache`.
    pub fn with_cache(mut self, cache: impl Into<PathBuf>) -> Self {
        self.cache = cache.into();
        self
    }

    /// Whether a missing file is an error instead of a download.
    pub fn offline(mut self, offline: bool) -> Self {
        self.offline = offline;
        self
    }

    /// Calls `observe` as each downloading file's bytes arrive.
    #[cfg(feature = "download")]
    pub fn on_progress(mut self, observe: impl Fn(&Progress<'_>) + Send + Sync + 'static) -> Self {
        self.observer = Some(Observer(Arc::new(observe)));
        self
    }

    /// The cache directory.
    pub fn cache(&self) -> &Path {
        &self.cache
    }

    pub(crate) fn snapshot(&self, entry: &Entry) -> PathBuf {
        self.cache
            .join(format!("models--{}", entry.repository.replace('/', "--")))
            .join("snapshots")
            .join(entry.revision)
    }

    /// The snapshot with every pinned file at its pinned size, fetched if missing and not yet hashed.
    pub(crate) fn locate(&self, entry: &Entry) -> Result<PathBuf, Error> {
        let snapshot = self.snapshot(entry);
        for file in &entry.files {
            let path = snapshot.join(file.name);
            if fs::metadata(&path).is_ok_and(|meta| meta.is_file() && meta.len() == file.size) {
                continue;
            }
            if self.offline {
                return Err(Error::Download(format!(
                    "{} {} is not in the cache as {} while offline",
                    entry.repository,
                    file.name,
                    path.display()
                )));
            }
            self.download(entry, file, &path)?;
        }
        Ok(snapshot)
    }

    /// [`Self::locate`], then every file checked against its pinned SHA-256.
    pub(crate) fn fetch(&self, entry: &Entry) -> Result<PathBuf, Error> {
        let snapshot = self.locate(entry)?;
        for file in &entry.files {
            let path = snapshot.join(file.name);
            let actual =
                sha256(&path).map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
            if actual != file.sha256 {
                return Err(Error::Model(registry::mismatch(
                    &path,
                    &actual,
                    file.sha256,
                )));
            }
        }
        Ok(snapshot)
    }

    #[cfg(not(feature = "download"))]
    fn download(&self, entry: &Entry, file: &Pinned, path: &Path) -> Result<(), Error> {
        Err(Error::Download(format!(
            "{} {} is not in the cache as {}, and this build has no download feature",
            entry.repository,
            file.name,
            path.display()
        )))
    }

    #[cfg(feature = "download")]
    fn download(&self, entry: &Entry, file: &Pinned, path: &Path) -> Result<(), Error> {
        let url = format!(
            "{}/{}/resolve/{}/{}",
            self.endpoint, entry.repository, entry.revision, file.name
        );
        let failed = |reason: String| Error::Download(format!("{url}: {reason}"));
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir).map_err(|e| failed(e.to_string()))?;
        }
        let report = |received| {
            if let Some(Observer(observe)) = &self.observer {
                observe(&Progress {
                    repository: entry.repository,
                    file: file.name,
                    received,
                    total: file.size,
                });
            }
        };
        transfer::fetch(&url, file, path, &report).map_err(failed)
    }
}

/// Read in pieces, so a model file is never held whole in memory.
fn sha256(path: &Path) -> io::Result<String> {
    let mut file = artifact_io::open_regular(path)?;
    let mut hash = Sha256::new();
    io::copy(&mut file, &mut hash)?;
    Ok(identity::hex(&hash.finalize().into()))
}

/// `HF_ENDPOINT` without a trailing slash, else huggingface.co.
#[cfg(feature = "download")]
fn endpoint(var: impl Fn(&str) -> Option<OsString>) -> String {
    var("HF_ENDPOINT")
        .and_then(|value| value.into_string().ok())
        .map_or_else(
            || ENDPOINT.into(),
            |value| value.trim_end_matches('/').into(),
        )
}

fn truthy(value: &OsStr) -> bool {
    value.to_str().is_some_and(|value| {
        ["1", "ON", "YES", "TRUE"]
            .iter()
            .any(|word| value.eq_ignore_ascii_case(word))
    })
}

/// huggingface_hub expands a leading `~` in its path variables.
fn expand_home(path: PathBuf, home: Option<&Path>) -> PathBuf {
    match (path.strip_prefix("~"), home) {
        (Ok(rest), Some(home)) => home.join(rest),
        _ => path,
    }
}

#[cfg(feature = "download")]
mod transfer {
    use crate::{publish::publish, registry::Pinned};
    use sha2::{Digest, Sha256};
    use std::{
        fmt,
        fs::File,
        io::{self, Read, Write},
        path::Path,
        thread,
        time::Duration,
    };

    const ATTEMPTS: u32 = 3;

    /// Bytes that arrived whole yet differ from the pin: a retry cannot help.
    #[derive(Debug)]
    struct Mismatch(String);

    impl fmt::Display for Mismatch {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(&self.0)
        }
    }

    impl std::error::Error for Mismatch {}

    enum Failure {
        Transient(String),
        Final(String),
    }

    /// Downloads `file` to `path`; `report` hears the bytes received so far.
    pub(super) fn fetch(
        url: &str,
        file: &Pinned,
        path: &Path,
        report: &dyn Fn(u64),
    ) -> Result<(), String> {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(30)))
            // A large download may take minutes even while every read makes progress.
            .timeout_per_read(Some(Duration::from_secs(60)))
            .user_agent(concat!("kodamaembed/", env!("CARGO_PKG_VERSION")))
            .build()
            .into();
        let mut attempt = 1;
        loop {
            match attempt_once(&agent, url, file, path, report) {
                Ok(()) => return Ok(()),
                Err(Failure::Transient(_)) if attempt < ATTEMPTS => {
                    thread::sleep(Duration::from_millis(500 * u64::from(attempt)));
                    attempt += 1;
                }
                Err(Failure::Transient(reason) | Failure::Final(reason)) => return Err(reason),
            }
        }
    }

    fn attempt_once(
        agent: &ureq::Agent,
        url: &str,
        file: &Pinned,
        path: &Path,
        report: &dyn Fn(u64),
    ) -> Result<(), Failure> {
        let response = agent.get(url).call().map_err(|error| match error {
            ureq::Error::StatusCode(_) => Failure::Final(error.to_string()),
            other => Failure::Transient(other.to_string()),
        })?;
        let mut body = response.into_body().into_reader();
        let copy = |out| copy_verified(&mut body, out, file, report);
        publish(path, copy, || {}).map_err(|error| {
            if error.get_ref().is_some_and(|inner| inner.is::<Mismatch>()) {
                Failure::Final(error.to_string())
            } else {
                Failure::Transient(error.to_string())
            }
        })
    }

    fn copy_verified(
        body: &mut impl Read,
        mut out: File,
        file: &Pinned,
        report: &dyn Fn(u64),
    ) -> io::Result<()> {
        let mismatch =
            |reason: String| io::Error::other(Mismatch(format!("{}: {reason}", file.name)));
        let mut hash = Sha256::new();
        let mut received = 0_u64;
        let mut buffer = vec![0; 1 << 20];
        loop {
            let read = match body.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => read,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            };
            received += read as u64;
            if received > file.size {
                return Err(mismatch(format!(
                    "more than its pinned {} bytes",
                    file.size
                )));
            }
            hash.update(&buffer[..read]);
            out.write_all(&buffer[..read])?;
            report(received);
        }
        if received != file.size {
            return Err(mismatch(format!(
                "{received} bytes of its pinned {}",
                file.size
            )));
        }
        let digest = format!("{:x}", hash.finalize());
        if digest != file.sha256 {
            return Err(mismatch(format!(
                "SHA-256 {digest} differs from the pinned {}",
                file.sha256
            )));
        }
        out.sync_all()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EmbeddingModel, registry::Pinned, test_support::TempDir};
    use std::fs;

    fn hub(vars: &[(&str, &str)], home: Option<&str>) -> Result<Hub, Error> {
        let lookup = |name: &str| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| OsString::from(value))
        };
        Hub::from_vars(lookup, home.map(PathBuf::from))
    }

    #[test]
    fn the_cache_resolves_as_huggingface_hub_resolves_it() {
        let home = Some("/u");
        for (vars, expected) in [
            (
                vec![("HF_HUB_CACHE", "/hub")],
                Path::new("/hub").to_path_buf(),
            ),
            (
                vec![("HUGGINGFACE_HUB_CACHE", "/legacy")],
                Path::new("/legacy").to_path_buf(),
            ),
            (
                vec![
                    ("HF_HUB_CACHE", "/hub"),
                    ("HUGGINGFACE_HUB_CACHE", "/legacy"),
                ],
                Path::new("/hub").to_path_buf(),
            ),
            (vec![("HF_HOME", "/hf")], Path::new("/hf").join("hub")),
            (
                vec![("XDG_CACHE_HOME", "/xdg")],
                Path::new("/xdg").join("huggingface").join("hub"),
            ),
            (
                vec![("HF_HOME", "/hf"), ("XDG_CACHE_HOME", "/xdg")],
                Path::new("/hf").join("hub"),
            ),
            (
                vec![],
                Path::new("/u")
                    .join(".cache")
                    .join("huggingface")
                    .join("hub"),
            ),
            (
                vec![("HF_HOME", "~/hf")],
                Path::new("/u").join("hf").join("hub"),
            ),
        ] {
            assert_eq!(hub(&vars, home).unwrap().cache(), expected, "{vars:?}");
        }
    }

    #[test]
    fn a_cache_needs_a_variable_or_a_home() {
        assert!(
            matches!(hub(&[], None), Err(Error::Download(reason)) if reason.contains("HF_HUB_CACHE"))
        );
        assert!(hub(&[("HF_HOME", "/hf")], None).is_ok());
    }

    #[test]
    fn offline_takes_the_values_huggingface_hub_treats_as_true() {
        for value in ["1", "ON", "yes", "True"] {
            assert!(
                hub(&[("HF_HUB_OFFLINE", value)], Some("/u"))
                    .unwrap()
                    .offline,
                "{value}"
            );
        }
        for value in ["0", "", "off", "no", "false"] {
            assert!(
                !hub(&[("HF_HUB_OFFLINE", value)], Some("/u"))
                    .unwrap()
                    .offline,
                "{value}"
            );
        }
        assert!(!hub(&[], Some("/u")).unwrap().offline);
        assert!(
            hub(&[("TRANSFORMERS_OFFLINE", "1")], Some("/u"))
                .unwrap()
                .offline
        );
        assert!(!Hub::new("/c").offline);
        assert!(Hub::new("/c").offline(true).offline);
    }

    #[test]
    fn another_cache_keeps_the_other_settings() {
        let hub = hub(&[("HF_HUB_OFFLINE", "1")], Some("/u"))
            .unwrap()
            .with_cache("/other");
        assert_eq!(hub.cache(), Path::new("/other"));
        assert!(hub.offline);
    }

    #[test]
    fn snapshots_follow_the_cache_layout() {
        assert_eq!(
            Hub::new("/c").snapshot(EmbeddingModel::AllMiniLmL6V2.entry()),
            Path::new("/c")
                .join("models--sentence-transformers--all-MiniLM-L6-v2")
                .join("snapshots")
                .join("1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
        );
    }

    const BODIES: [&[u8]; 3] = [b"abc", b"abcd", b"abcde"];

    const ENTRY: Entry = Entry {
        repository: "test-owner/test-model",
        revision: "0123456789abcdef0123456789abcdef01234567",
        files: [
            Pinned {
                name: "config.json",
                size: 3,
                sha256: "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            },
            Pinned {
                name: "tokenizer.json",
                size: 4,
                sha256: "88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589",
            },
            Pinned {
                name: "model.safetensors",
                size: 5,
                sha256: "36bbe50ed96841d10443bcb670d6554f0a34b761be67ec9c4a8ad2c0c44ca42c",
            },
        ],
    };

    struct Cache(TempDir);

    impl Cache {
        fn new() -> Self {
            Self(TempDir::new("hub"))
        }

        fn hub(&self, offline: bool) -> Hub {
            Hub::new(self.0.path()).offline(offline)
        }

        fn write(&self, hub: &Hub, files: &[(&str, &[u8])]) {
            let snapshot = hub.snapshot(&ENTRY);
            fs::create_dir_all(&snapshot).unwrap();
            for (name, bytes) in files {
                fs::write(snapshot.join(name), bytes).unwrap();
            }
        }
    }

    /// Every file of [`ENTRY`] at its pinned size and hash.
    const COMPLETE: [(&str, &[u8]); 3] = [
        ("config.json", BODIES[0]),
        ("tokenizer.json", BODIES[1]),
        ("model.safetensors", BODIES[2]),
    ];

    #[test]
    fn a_complete_snapshot_loads_while_offline() {
        let cache = Cache::new();
        let hub = cache.hub(true);
        cache.write(&hub, &COMPLETE);
        assert_eq!(hub.locate(&ENTRY).unwrap(), hub.snapshot(&ENTRY));
    }

    #[test]
    fn fetching_checks_every_file_against_its_pin() {
        let cache = Cache::new();
        let hub = cache.hub(true);
        cache.write(&hub, &COMPLETE);
        assert_eq!(hub.fetch(&ENTRY).unwrap(), hub.snapshot(&ENTRY));
        // Preserve the file size so only the digest can detect corruption.
        cache.write(&hub, &[("tokenizer.json", b"abcX")]);
        let Err(Error::Model(reason)) = hub.fetch(&ENTRY) else {
            panic!("bytes other than the pinned ones must fail");
        };
        assert!(
            reason.contains("tokenizer.json")
                && reason
                    .contains("88d4266fd4e6338d13b845fcf289579d209c897823b9217da3e161936f031589"),
            "{reason}"
        );
    }

    #[test]
    fn offline_a_missing_or_resized_file_is_an_error_naming_it() {
        let cache = Cache::new();
        let hub = cache.hub(true);
        cache.write(
            &hub,
            &[("config.json", BODIES[0]), ("model.safetensors", BODIES[2])],
        );
        let Err(Error::Download(reason)) = hub.locate(&ENTRY) else {
            panic!("a missing file must be a download error");
        };
        assert!(
            reason.contains("tokenizer.json") && reason.contains("offline"),
            "{reason}"
        );
        cache.write(
            &hub,
            &[
                ("tokenizer.json", BODIES[1]),
                ("model.safetensors", b"abcdef"),
            ],
        );
        let Err(Error::Download(reason)) = hub.locate(&ENTRY) else {
            panic!("a file of another size must be a download error");
        };
        assert!(reason.contains("model.safetensors"), "{reason}");
    }

    #[cfg(feature = "download")]
    mod download {
        use super::*;
        use std::{
            io::{Read, Write},
            net::TcpListener,
            sync::{Arc, Mutex},
        };

        /// Serves one scripted response a connection and records each request head.
        struct Server {
            endpoint: String,
            requests: Arc<Mutex<Vec<String>>>,
        }

        impl Server {
            fn hub(&self, cache: &Cache) -> Hub {
                Hub {
                    endpoint: self.endpoint.clone(),
                    ..cache.hub(false)
                }
            }
        }

        fn snapshot_files(hub: &Hub) -> Vec<String> {
            let mut names: Vec<String> = fs::read_dir(hub.snapshot(&ENTRY))
                .unwrap()
                .map(|entry| entry.unwrap().file_name().into_string().unwrap())
                .collect();
            names.sort();
            names
        }

        fn serve(responses: Vec<Vec<u8>>) -> Server {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let endpoint = format!("http://{}", listener.local_addr().unwrap());
            let requests = Arc::new(Mutex::new(Vec::new()));
            let seen = Arc::clone(&requests);
            // Joining would hang tests that deliberately leave a response unused.
            std::thread::spawn(move || {
                for response in responses {
                    let (mut stream, _) = listener.accept().unwrap();
                    let mut head = Vec::new();
                    let mut byte = [0];
                    while !head.ends_with(b"\r\n\r\n") && stream.read(&mut byte).unwrap() == 1 {
                        head.push(byte[0]);
                    }
                    seen.lock().unwrap().push(String::from_utf8(head).unwrap());
                    // A client that rejects the body may hang up before it is sent.
                    let _ = stream.write_all(&response);
                }
            });
            Server { endpoint, requests }
        }

        fn response(status: &str, length: usize, body: &[u8]) -> Vec<u8> {
            let mut bytes = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n"
            )
            .into_bytes();
            bytes.extend_from_slice(body);
            bytes
        }

        fn ok(body: &[u8]) -> Vec<u8> {
            response("200 OK", body.len(), body)
        }

        #[test]
        fn the_endpoint_defaults_to_the_hub_and_takes_a_mirror() {
            assert_eq!(
                hub(&[], Some("/u")).unwrap().endpoint,
                "https://huggingface.co"
            );
            assert_eq!(Hub::new("/c").endpoint, "https://huggingface.co");
            let mirror = hub(&[("HF_ENDPOINT", "https://hf-mirror.com/")], Some("/u")).unwrap();
            assert_eq!(mirror.endpoint, "https://hf-mirror.com");
            assert_eq!(
                mirror.with_cache("/other").endpoint,
                "https://hf-mirror.com"
            );
        }

        #[test]
        fn progress_reports_each_downloading_file_up_to_its_size() {
            let cache = Cache::new();
            let server = serve(vec![ok(BODIES[1]), ok(BODIES[2])]);
            let events = Arc::new(Mutex::new(Vec::new()));
            let seen = Arc::clone(&events);
            let hub = server.hub(&cache).on_progress(move |progress| {
                seen.lock().unwrap().push((
                    progress.repository.to_owned(),
                    progress.file.to_owned(),
                    progress.received,
                    progress.total,
                ));
            });
            cache.write(&hub, &[("config.json", BODIES[0])]);
            hub.locate(&ENTRY).unwrap();
            let events = events.lock().unwrap();
            let last = |file: &str| events.iter().rev().find(|event| event.1 == file).cloned();
            let repository = String::from("test-owner/test-model");
            assert_eq!(
                last("tokenizer.json"),
                Some((repository.clone(), "tokenizer.json".into(), 4, 4))
            );
            assert_eq!(
                last("model.safetensors"),
                Some((repository, "model.safetensors".into(), 5, 5))
            );
            assert!(
                events
                    .iter()
                    .all(|event| event.1 != "config.json" && event.2 <= event.3)
            );
        }

        #[test]
        fn cached_files_of_the_pinned_size_need_no_request() {
            let cache = Cache::new();
            let server = serve(vec![response("404 Not Found", 0, b"")]);
            let hub = server.hub(&cache);
            cache.write(&hub, &COMPLETE);
            assert_eq!(hub.locate(&ENTRY).unwrap(), hub.snapshot(&ENTRY));
            assert!(server.requests.lock().unwrap().is_empty());
        }

        #[test]
        fn missing_files_download_at_the_pinned_revision_and_publish_whole() {
            let cache = Cache::new();
            let server = serve(vec![ok(BODIES[1])]);
            let hub = server.hub(&cache);
            cache.write(
                &hub,
                &[("config.json", BODIES[0]), ("model.safetensors", BODIES[2])],
            );
            assert_eq!(hub.locate(&ENTRY).unwrap(), hub.snapshot(&ENTRY));
            assert_eq!(
                fs::read(hub.snapshot(&ENTRY).join("tokenizer.json")).unwrap(),
                BODIES[1]
            );
            assert_eq!(
                snapshot_files(&hub),
                ["config.json", "model.safetensors", "tokenizer.json"]
            );
            let requests = server.requests.lock().unwrap();
            assert_eq!(requests.len(), 1);
            assert!(
                requests[0].starts_with(
                    "GET /test-owner/test-model/resolve/0123456789abcdef0123456789abcdef01234567/tokenizer.json HTTP/1.1\r\n"
                ),
                "{}",
                requests[0]
            );
            assert!(
                requests[0]
                    .to_ascii_lowercase()
                    .contains("user-agent: kodamaembed/")
            );
        }

        #[test]
        fn a_file_of_another_size_downloads_again() {
            let cache = Cache::new();
            let server = serve(vec![ok(BODIES[2])]);
            let hub = server.hub(&cache);
            cache.write(
                &hub,
                &[
                    ("config.json", BODIES[0]),
                    ("tokenizer.json", BODIES[1]),
                    ("model.safetensors", b"ab"),
                ],
            );
            hub.locate(&ENTRY).unwrap();
            assert_eq!(
                fs::read(hub.snapshot(&ENTRY).join("model.safetensors")).unwrap(),
                BODIES[2]
            );
        }

        #[test]
        fn an_interrupted_transfer_is_retried() {
            let cache = Cache::new();
            let server = serve(vec![response("200 OK", 5, b"ab"), ok(BODIES[2])]);
            let hub = server.hub(&cache);
            cache.write(
                &hub,
                &[("config.json", BODIES[0]), ("tokenizer.json", BODIES[1])],
            );
            hub.locate(&ENTRY).unwrap();
            assert_eq!(
                fs::read(hub.snapshot(&ENTRY).join("model.safetensors")).unwrap(),
                BODIES[2]
            );
            assert_eq!(server.requests.lock().unwrap().len(), 2);
        }

        #[test]
        fn wrong_bytes_publish_nothing_and_are_not_retried() {
            for wrong in [response("200 OK", 5, b"abcdX"), ok(b"abcdef"), ok(b"abcd")] {
                let cache = Cache::new();
                // A retry would receive the right bytes and succeed.
                let server = serve(vec![wrong, ok(BODIES[2])]);
                let hub = server.hub(&cache);
                cache.write(
                    &hub,
                    &[("config.json", BODIES[0]), ("tokenizer.json", BODIES[1])],
                );
                let Err(Error::Download(reason)) = hub.locate(&ENTRY) else {
                    panic!("wrong bytes must fail");
                };
                assert!(reason.contains("model.safetensors"), "{reason}");
                assert_eq!(snapshot_files(&hub), ["config.json", "tokenizer.json"]);
                assert_eq!(server.requests.lock().unwrap().len(), 1);
            }
        }

        #[test]
        fn an_http_error_is_returned_without_retrying() {
            let cache = Cache::new();
            let server = serve(vec![response("404 Not Found", 0, b""), ok(BODIES[2])]);
            let hub = server.hub(&cache);
            cache.write(
                &hub,
                &[("config.json", BODIES[0]), ("tokenizer.json", BODIES[1])],
            );
            let Err(Error::Download(reason)) = hub.locate(&ENTRY) else {
                panic!("a 404 must fail");
            };
            assert!(reason.contains("404"), "{reason}");
            assert_eq!(server.requests.lock().unwrap().len(), 1);
            assert_eq!(snapshot_files(&hub), ["config.json", "tokenizer.json"]);
        }
    }
}
