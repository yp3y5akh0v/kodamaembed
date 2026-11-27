//! Owned model-artifact bytes, read only from an opened regular file.

use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};

pub(crate) fn read(path: impl AsRef<Path>) -> io::Result<Vec<u8>> {
    let mut file = open_regular(path)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// Checks the opened handle before reading so special files cannot block reads.
pub(crate) fn open_regular(path: impl AsRef<Path>) -> io::Result<File> {
    #[cfg(unix)]
    let opened = {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NONBLOCK)
            .open(path.as_ref())
    };
    #[cfg(not(unix))]
    let opened = File::open(path.as_ref());
    let file = opened?;
    if !file.metadata()?.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "model artifact must be a regular file",
        ));
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;
    use std::fs;

    const ARTIFACTS: [&str; 3] = ["config.json", "tokenizer.json", "model.safetensors"];

    #[test]
    fn regular_artifact_bytes_and_empty_files_are_unchanged() {
        let fixture = TempDir::new("regular-artifacts");
        let payload: Vec<u8> = (0..=255).collect();
        for name in ARTIFACTS {
            let path = fixture.path().join(name);
            fs::write(&path, &payload).unwrap();
            assert_eq!(read(&path).unwrap(), payload);
            fs::write(&path, []).unwrap();
            assert!(read(&path).unwrap().is_empty());
        }
    }

    #[test]
    fn missing_artifact_retains_not_found_error() {
        let fixture = TempDir::new("regular-artifacts");
        assert_eq!(
            read(fixture.path().join("config.json")).unwrap_err().kind(),
            io::ErrorKind::NotFound
        );
    }

    #[test]
    fn a_directory_cannot_be_a_model_artifact() {
        let fixture = TempDir::new("regular-artifacts");
        assert!(read(fixture.path()).is_err());
        #[cfg(unix)]
        {
            let error = read(fixture.path()).unwrap_err();
            assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
            assert_eq!(error.to_string(), "model artifact must be a regular file");
        }
    }

    #[cfg(unix)]
    #[test]
    fn symlinks_to_regular_artifacts_remain_supported() {
        let fixture = TempDir::new("regular-artifacts");
        let bytes = b"owned regular artifact bytes\0\xff";
        fs::write(fixture.path().join("regular-target"), bytes).unwrap();
        for name in ARTIFACTS {
            let path = fixture.path().join(name);
            std::os::unix::fs::symlink("regular-target", &path).unwrap();
            assert_eq!(read(&path).unwrap().as_slice(), bytes);
        }
    }

    #[cfg(unix)]
    fn model_inputs(fixture: &TempDir) {
        fs::write(
            fixture.path().join("config.json"),
            serde_json::to_vec(&crate::test_support::tiny_bert_config(4)).unwrap(),
        )
        .unwrap();
        let model = tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(
                [
                    ("a".into(), 0),
                    ("b".into(), 1),
                    ("ab".into(), 2),
                    ("[UNK]".into(), 3),
                ]
                .into_iter()
                .collect::<tokenizers::models::bpe::Vocab>(),
                vec![("a".into(), "b".into())],
            )
            .build()
            .unwrap();
        let tokenizer = tokenizers::Tokenizer::new(model);
        fs::write(
            fixture.path().join("tokenizer.json"),
            tokenizer.to_string(false).unwrap(),
        )
        .unwrap();
        // No valid weights or backend are needed: each child must reject its FIFO first.
        fs::write(
            fixture.path().join("model.safetensors"),
            b"unused checkpoint",
        )
        .unwrap();
    }

    #[cfg(unix)]
    fn make_fifo(path: &Path) {
        use std::{ffi::CString, os::unix::ffi::OsStrExt};
        let path = CString::new(path.as_os_str().as_bytes()).unwrap();
        // SAFETY: the owned NUL-terminated path remains live through this call.
        assert_eq!(
            unsafe { libc::mkfifo(path.as_ptr(), 0o600) },
            0,
            "mkfifo failed: {}",
            io::Error::last_os_error()
        );
    }

    #[cfg(unix)]
    struct ChildCleanup(Option<std::process::Child>);

    #[cfg(unix)]
    impl Drop for ChildCleanup {
        fn drop(&mut self) {
            if let Some(mut child) = self.0.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    #[cfg(unix)]
    fn fifo_child_rejects(fixture: &TempDir, artifact: &str) {
        use std::{
            process::{Command, Stdio},
            time::{Duration, Instant},
        };
        let marker = fixture.path().join("complete");
        assert!(!marker.exists(), "stale marker");
        let child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "artifact_io::tests::fifo_open_child_probe",
                "--ignored",
                "--test-threads=1",
                "--nocapture",
            ])
            .env("KODAMAEMBED_ARTIFACT_FIFO_DIRECTORY", fixture.path())
            .env("KODAMAEMBED_ARTIFACT_FIFO_NAME", artifact)
            .env("KODAMAEMBED_ARTIFACT_FIFO_MARKER", &marker)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let mut child = ChildCleanup(Some(child));
        let started = Instant::now();
        loop {
            if let Some(status) = child.0.as_mut().unwrap().try_wait().unwrap() {
                let _ = child.0.take(); // try_wait has reaped this child.
                assert!(
                    started.elapsed() <= Duration::from_secs(10),
                    "FIFO child completed after watchdog"
                );
                assert!(status.success(), "FIFO child failed: {status}");
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "FIFO child exceeded watchdog"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            fs::read(marker).expect("helper wrote marker"),
            format!("{artifact}: both model loaders rejected a nonregular artifact\n").as_bytes()
        );
    }

    #[cfg(unix)]
    #[test]
    fn all_three_model_artifacts_reject_fifo_and_fifo_symlink_before_reading() {
        for artifact in ARTIFACTS {
            for symlink in [false, true] {
                let fixture = TempDir::new("regular-artifacts");
                model_inputs(&fixture);
                let path = fixture.path().join(artifact);
                fs::remove_file(&path).unwrap();
                if symlink {
                    make_fifo(&fixture.path().join("target.fifo"));
                    std::os::unix::fs::symlink("target.fifo", &path).unwrap();
                } else {
                    make_fifo(&path);
                }
                fifo_child_rejects(&fixture, artifact);
            }
        }
    }

    #[cfg(unix)]
    #[test]
    #[ignore = "subprocess-only bounded FIFO probe"]
    fn fifo_open_child_probe() {
        use std::io::Write;
        let directory = std::path::PathBuf::from(
            std::env::var_os("KODAMAEMBED_ARTIFACT_FIFO_DIRECTORY")
                .expect("Child-only directory required"),
        );
        let artifact =
            std::env::var("KODAMAEMBED_ARTIFACT_FIFO_NAME").expect("Child-only artifact required");
        assert!(ARTIFACTS.contains(&artifact.as_str()));
        let marker = std::path::PathBuf::from(
            std::env::var_os("KODAMAEMBED_ARTIFACT_FIFO_MARKER")
                .expect("Child-only marker required"),
        );
        let spec = crate::ModelSpec {
            passage_prefix: "",
            query_prefix: "",
            max_length: 8,
            pooling: crate::Pooling::Mean,
        };
        let results = [
            crate::Embedder::builder()
                .device(crate::Device::Cpu)
                .batch_tokens(16)
                .build_from_dir(&directory, spec)
                .map(|_| ()),
            crate::Reranker::builder()
                .device(crate::Device::Cpu)
                .batch_tokens(16)
                .build_from_dir(&directory, 8)
                .map(|_| ()),
        ];
        for result in results {
            let reason = match result {
                Err(crate::Error::Tokenizer(reason)) if artifact == "tokenizer.json" => reason,
                Err(crate::Error::Model(reason)) if artifact != "tokenizer.json" => reason,
                other => panic!("Expected the model-artifact guard for {artifact}, got {other:?}"),
            };
            assert!(
                reason.contains(&artifact)
                    && reason.contains("model artifact must be a regular file"),
                "Wrong rejection: {reason}"
            );
        }
        let mut completion = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(marker)
            .unwrap();
        completion
            .write_all(
                format!("{artifact}: both model loaders rejected a nonregular artifact\n")
                    .as_bytes(),
            )
            .unwrap();
        completion.flush().unwrap();
    }
}
