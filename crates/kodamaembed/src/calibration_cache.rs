//! Calibration tables in KodamaEmbed's cache, one checksummed file a fingerprint.

use crate::artifact_io;
use sha2::{Digest, Sha256};
use std::{
    ffi::OsString,
    fs::File,
    io::{self, Write},
    path::{Path, PathBuf},
};

const MAGIC: &[u8] = b"kodamaembed calibration 1\n";

/// The `calibration` directory of `KODAMAEMBED_CACHE`, else of the platform cache's `kodamaembed`.
pub(crate) fn dir(
    var: impl Fn(&str) -> Option<OsString>,
    home: Option<PathBuf>,
    os: &str,
) -> Option<PathBuf> {
    let root = match var("KODAMAEMBED_CACHE") {
        Some(root) => PathBuf::from(root),
        None => {
            let base = match os {
                "windows" => var("LOCALAPPDATA").map(PathBuf::from),
                "macos" => home.map(|home| home.join("Library").join("Caches")),
                _ => var("XDG_CACHE_HOME")
                    .map(PathBuf::from)
                    .or_else(|| home.map(|home| home.join(".cache"))),
            };
            base?.join("kodamaembed")
        }
    };
    Some(root.join("calibration"))
}

pub(crate) fn path(dir: &Path, fingerprint: &str) -> PathBuf {
    dir.join(format!(
        "kodamaembed-{}.calibration",
        fingerprint.replace(':', "-")
    ))
}

/// A missing or damaged file reads as none, so calibrating again replaces it.
pub(crate) fn read(path: &Path, fingerprint: &str) -> io::Result<Option<Vec<u8>>> {
    let mut bytes = match artifact_io::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    Ok(header_len(&bytes, fingerprint).map(|header| {
        bytes.drain(..header);
        bytes
    }))
}

pub(crate) fn write(path: &Path, fingerprint: &str, payload: &[u8]) -> io::Result<()> {
    write_before_commit(path, fingerprint, payload, || {})
}

fn write_before_commit(
    path: &Path,
    fingerprint: &str,
    payload: &[u8],
    before_commit: impl FnOnce(),
) -> io::Result<()> {
    crate::publish::publish(
        path,
        |file| write_file(file, fingerprint, payload),
        before_commit,
    )
}

fn write_file(mut file: File, fingerprint: &str, payload: &[u8]) -> io::Result<()> {
    let fingerprint_len = u32::try_from(fingerprint.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "fingerprint too long"))?;
    file.write_all(MAGIC)?;
    file.write_all(&fingerprint_len.to_le_bytes())?;
    file.write_all(fingerprint.as_bytes())?;
    file.write_all(&(payload.len() as u64).to_le_bytes())?;
    file.write_all(&Sha256::digest(payload))?;
    file.write_all(payload)?;
    file.sync_all()
}

fn header_len(bytes: &[u8], fingerprint: &str) -> Option<usize> {
    let rest = bytes.strip_prefix(MAGIC)?;
    let (stored_len, rest) = rest.split_first_chunk::<4>()?;
    let stored_len = usize::try_from(u32::from_le_bytes(*stored_len)).ok()?;
    let (stored, rest) = rest.split_at_checked(stored_len)?;
    let (payload_len, rest) = rest.split_first_chunk::<8>()?;
    let (digest, payload) = rest.split_first_chunk::<32>()?;
    (stored == fingerprint.as_bytes()
        && u64::from_le_bytes(*payload_len) == payload.len() as u64
        && Sha256::digest(payload)[..] == digest[..])
        .then_some(bytes.len() - payload.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TempDir;
    use std::{fs, sync::atomic::AtomicU64};

    const FINGERPRINT: &str =
        "sha256:00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff";

    fn cache_dir(vars: &[(&str, &str)], home: Option<&str>, os: &str) -> Option<PathBuf> {
        let lookup = |name: &str| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| OsString::from(value))
        };
        dir(lookup, home.map(PathBuf::from), os)
    }

    #[test]
    fn the_cache_directory_follows_each_platforms_convention() {
        let calibration = |root: &str| Some(Path::new(root).join("calibration"));
        let windows = [("LOCALAPPDATA", "D:/profile/AppData/Local")];
        assert_eq!(
            cache_dir(&windows, Some("D:/profile"), "windows"),
            calibration("D:/profile/AppData/Local/kodamaembed")
        );
        assert_eq!(
            cache_dir(&[], Some("/h"), "macos"),
            calibration("/h/Library/Caches/kodamaembed")
        );
        assert_eq!(
            cache_dir(&[("XDG_CACHE_HOME", "/xdg")], Some("/h"), "linux"),
            calibration("/xdg/kodamaembed")
        );
        assert_eq!(
            cache_dir(&[], Some("/h"), "linux"),
            calibration("/h/.cache/kodamaembed")
        );
    }

    #[test]
    fn kodamaembed_cache_overrides_every_platform_default() {
        for os in ["windows", "macos", "linux"] {
            let vars = [
                ("KODAMAEMBED_CACHE", "/chosen"),
                ("LOCALAPPDATA", "/local"),
                ("XDG_CACHE_HOME", "/xdg"),
            ];
            assert_eq!(
                cache_dir(&vars, Some("/h"), os),
                Some(Path::new("/chosen").join("calibration")),
                "{os}"
            );
        }
    }

    #[test]
    fn no_cache_directory_without_its_variables_or_a_home() {
        assert_eq!(cache_dir(&[], Some("D:/profile"), "windows"), None);
        assert_eq!(cache_dir(&[], None, "macos"), None);
        assert_eq!(cache_dir(&[], None, "linux"), None);
    }

    fn file_names(folder: &TempDir) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(folder.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn a_written_payload_reads_back_whole_and_alone() {
        let folder = TempDir::new("calibration-cache");
        let path = path(folder.path(), FINGERPRINT);
        assert_eq!(
            path.file_name().unwrap(),
            "kodamaembed-sha256-00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff.calibration"
        );
        assert!(read(&path, FINGERPRINT).unwrap().is_none());
        let payload: Vec<u8> = (0..100_000_u32).map(|i| (i * 7 % 251) as u8).collect();
        write(&path, FINGERPRINT, &payload).unwrap();
        assert_eq!(read(&path, FINGERPRINT).unwrap().unwrap(), payload);
        write(&path, FINGERPRINT, &payload[..10]).unwrap();
        assert_eq!(read(&path, FINGERPRINT).unwrap().unwrap(), &payload[..10]);
        assert_eq!(
            file_names(&folder),
            [path.file_name().unwrap().to_str().unwrap()]
        );
        write(&path, FINGERPRINT, &[]).unwrap();
        assert_eq!(read(&path, FINGERPRINT).unwrap().unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn concurrent_writes_of_one_fingerprint_both_publish_complete_files() {
        use std::{sync::mpsc, thread, time::Duration};

        let folder = TempDir::new("calibration-cache");
        let path = path(folder.path(), FINGERPRINT);
        let payload = b"the same calibrated tables";
        let (ready, written) = mpsc::channel();
        let (commit_first, first) = mpsc::channel();
        let (commit_second, second) = mpsc::channel();
        let results = thread::scope(|scope| {
            let (path, ready) = (&path, &ready);
            let first = scope.spawn(move || {
                write_before_commit(path, FINGERPRINT, payload, || {
                    ready.send(()).unwrap();
                    first.recv_timeout(Duration::from_secs(10)).unwrap();
                })
            });
            let second = scope.spawn(move || {
                write_before_commit(path, FINGERPRINT, payload, || {
                    ready.send(()).unwrap();
                    second.recv_timeout(Duration::from_secs(10)).unwrap();
                })
            });
            for _ in 0..2 {
                written.recv_timeout(Duration::from_secs(10)).unwrap();
            }
            commit_first.send(()).unwrap();
            let first = first.join().unwrap();
            commit_second.send(()).unwrap();
            [first, second.join().unwrap()]
        });
        assert!(results.iter().all(Result::is_ok), "{results:?}");
        assert_eq!(read(&path, FINGERPRINT).unwrap().unwrap(), payload);
        assert_eq!(
            file_names(&folder),
            [path.file_name().unwrap().to_str().unwrap()]
        );
    }

    #[test]
    fn stale_partial_names_are_preserved_and_skipped() {
        let folder = TempDir::new("calibration-cache");
        let path = path(folder.path(), FINGERPRINT);
        let stale = path.with_extension(format!("calibration.{}.0.partial", std::process::id()));
        fs::write(&stale, b"a previous owner's unfinished file").unwrap();
        let (partial, file) = crate::publish::create_partial(&path, &AtomicU64::new(0)).unwrap();
        assert_ne!(partial, stale);
        assert_eq!(
            partial,
            path.with_extension(format!("calibration.{}.1.partial", std::process::id()))
        );
        write_file(file, FINGERPRINT, b"complete tables").unwrap();
        fs::rename(&partial, &path).unwrap();
        assert_eq!(
            fs::read(&stale).unwrap(),
            b"a previous owner's unfinished file"
        );
        assert_eq!(
            read(&path, FINGERPRINT).unwrap().unwrap(),
            b"complete tables"
        );
        assert!(!partial.exists());
    }

    #[test]
    fn damaged_or_other_files_are_not_read() {
        let folder = TempDir::new("calibration-cache");
        let path = path(folder.path(), FINGERPRINT);
        let payload: Vec<u8> = (0..4_000_u32).map(|i| (i % 253) as u8).collect();
        write(&path, FINGERPRINT, &payload).unwrap();
        let whole = fs::read(&path).unwrap();
        let header = whole.len() - payload.len();
        let mut damaged = Vec::new();
        for at in [
            0,
            MAGIC.len(),
            MAGIC.len() + 4,
            header - 40,
            header - 1,
            header,
            whole.len() - 1,
        ] {
            let mut bytes = whole.clone();
            bytes[at] ^= 1;
            damaged.push(bytes);
        }
        damaged.push(whole[..whole.len() - 1].to_vec());
        let mut longer = whole.clone();
        longer.push(0);
        damaged.push(longer);
        for (i, bytes) in damaged.iter().enumerate() {
            fs::write(&path, bytes).unwrap();
            assert!(read(&path, FINGERPRINT).unwrap().is_none(), "damage {i}");
        }
        fs::write(&path, &whole).unwrap();
        let other = FINGERPRINT.replace("00112233", "00112234");
        assert!(read(&path, &other).unwrap().is_none());
        assert_eq!(read(&path, FINGERPRINT).unwrap().unwrap(), payload);
    }

    #[test]
    fn a_directory_in_the_files_place_is_an_error() {
        let folder = TempDir::new("calibration-cache");
        let path = path(folder.path(), FINGERPRINT);
        fs::create_dir(&path).unwrap();
        assert!(read(&path, FINGERPRINT).is_err());
        assert!(write(&path, FINGERPRINT, b"tables").is_err());
        assert_eq!(
            file_names(&folder),
            [path.file_name().unwrap().to_str().unwrap()]
        );
    }

    #[test]
    fn publication_leaves_other_files_untouched() {
        let folder = TempDir::new("calibration-cache");
        let other_fingerprint = FINGERPRINT.replace("00112233", "00112234");
        let other = path(folder.path(), &other_fingerprint);
        let path = path(folder.path(), FINGERPRINT);
        write(&other, &other_fingerprint, b"other tables").unwrap();
        write(&path, FINGERPRINT, b"tables").unwrap();
        // Cache files, look-alikes and another process's partial file.
        let neighbors = [
            other.file_name().unwrap().to_str().unwrap(),
            "kodamaembed-explicit.calibration",
            "kodamaembed-notes.calibration",
            "tables.bin",
            "kodamaembed-sha256-00.calibration.7.partial",
        ];
        write(
            &folder.path().join(neighbors[1]),
            "another origin",
            b"explicit tables",
        )
        .unwrap();
        fs::write(folder.path().join(neighbors[2]), b"not a calibration").unwrap();
        fs::copy(&other, folder.path().join(neighbors[3])).unwrap();
        fs::copy(&other, folder.path().join(neighbors[4])).unwrap();
        let directory = "kodamaembed-folder.calibration";
        fs::create_dir(folder.path().join(directory)).unwrap();
        let before = neighbors.map(|name| fs::read(folder.path().join(name)).unwrap());
        write(&path, FINGERPRINT, b"new tables").unwrap();
        assert_eq!(read(&path, FINGERPRINT).unwrap().unwrap(), b"new tables");
        for (name, bytes) in neighbors.iter().zip(&before) {
            assert_eq!(
                &fs::read(folder.path().join(name)).unwrap(),
                bytes,
                "{name}"
            );
        }
        let mut files = neighbors.to_vec();
        files.extend([directory, path.file_name().unwrap().to_str().unwrap()]);
        files.sort();
        assert_eq!(file_names(&folder), files);
    }

    #[test]
    fn concurrent_publications_of_distinct_fingerprints_both_persist() {
        let folder = TempDir::new("calibration-cache");
        let other = FINGERPRINT.replace("00112233", "00112234");
        let start = std::sync::Barrier::new(2);
        std::thread::scope(|scope| {
            let first = scope.spawn(|| {
                start.wait();
                write(
                    &path(folder.path(), FINGERPRINT),
                    FINGERPRINT,
                    b"first policy",
                )
            });
            let second = scope.spawn(|| {
                start.wait();
                write(&path(folder.path(), &other), &other, b"second policy")
            });
            first.join().unwrap().unwrap();
            second.join().unwrap().unwrap();
        });
        assert_eq!(
            read(&path(folder.path(), FINGERPRINT), FINGERPRINT)
                .unwrap()
                .unwrap(),
            b"first policy"
        );
        assert_eq!(
            read(&path(folder.path(), &other), &other).unwrap().unwrap(),
            b"second policy"
        );
        assert_eq!(file_names(&folder).len(), 2);
    }

    #[cfg(unix)]
    struct CleanupChild(Option<std::process::Child>);

    #[cfg(unix)]
    impl Drop for CleanupChild {
        fn drop(&mut self) {
            if let Some(mut child) = self.0.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    #[cfg(unix)]
    fn check_publication_with_fifo(link: bool) {
        use std::{
            ffi::CString,
            os::unix::{ffi::OsStrExt, fs::symlink},
            process::{Command, Stdio},
            time::{Duration, Instant},
        };

        let folder = TempDir::new("calibration-cache");
        let current = path(folder.path(), FINGERPRINT);
        write(&current, FINGERPRINT, b"current tables").unwrap();
        write(
            &folder.path().join("kodamaembed-old.calibration"),
            "old",
            b"old tables",
        )
        .unwrap();
        write(
            &folder.path().join("valid-target.bin"),
            "linked",
            b"linked tables",
        )
        .unwrap();
        symlink(
            "valid-target.bin",
            folder.path().join("kodamaembed-linked.calibration"),
        )
        .unwrap();
        fs::write(
            folder.path().join("kodamaembed-notes.calibration"),
            b"unrelated notes",
        )
        .unwrap();

        let fifo_name = if link {
            "target.fifo"
        } else {
            "kodamaembed-pipe.calibration"
        };
        let fifo = CString::new(folder.path().join(fifo_name).as_os_str().as_bytes()).unwrap();
        // SAFETY: the owned NUL-terminated path remains live through this call.
        assert_eq!(unsafe { libc::mkfifo(fifo.as_ptr(), 0o600) }, 0);
        if link {
            symlink(
                "target.fifo",
                folder.path().join("kodamaembed-pipe.calibration"),
            )
            .unwrap();
        }
        let marker = folder.path().join("complete");
        assert!(!marker.exists());
        let child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "calibration_cache::tests::publication_fifo_child_probe",
                "--ignored",
                "--test-threads=1",
                "--nocapture",
            ])
            .env("KODAMAEMBED_CACHE_FIFO_DIRECTORY", folder.path())
            .env(
                "KODAMAEMBED_CACHE_FIFO_SYMLINK",
                if link { "1" } else { "0" },
            )
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let mut child = CleanupChild(Some(child));
        let started = Instant::now();
        loop {
            if let Some(status) = child.0.as_mut().unwrap().try_wait().unwrap() {
                let _ = child.0.take();
                assert!(started.elapsed() <= Duration::from_secs(10));
                assert!(status.success(), "publication child failed: {status}");
                break;
            }
            assert!(
                started.elapsed() < Duration::from_secs(10),
                "cache publication blocked on a neighboring FIFO"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(
            fs::read(marker).expect("probe finished"),
            b"cache publication completed; neighboring artifacts retained\n"
        );
    }

    #[cfg(unix)]
    #[test]
    fn publication_preserves_fifo_without_waiting_for_a_writer() {
        check_publication_with_fifo(false);
    }

    #[cfg(unix)]
    #[test]
    fn publication_preserves_fifo_symlink_without_waiting_for_a_writer() {
        check_publication_with_fifo(true);
    }

    #[cfg(unix)]
    #[test]
    #[ignore = "subprocess-only bounded FIFO publication probe"]
    fn publication_fifo_child_probe() {
        let directory = PathBuf::from(
            std::env::var_os("KODAMAEMBED_CACHE_FIFO_DIRECTORY")
                .expect("Child-only directory required"),
        );
        let link = match std::env::var("KODAMAEMBED_CACHE_FIFO_SYMLINK")
            .unwrap()
            .as_str()
        {
            "0" => false,
            "1" => true,
            _ => panic!("Child-only case must be explicit"),
        };
        let current = path(&directory, FINGERPRINT);
        write(&current, FINGERPRINT, b"current tables").unwrap();
        assert_eq!(
            read(&current, FINGERPRINT).unwrap().unwrap(),
            b"current tables"
        );
        assert_eq!(
            read(&directory.join("kodamaembed-old.calibration"), "old")
                .unwrap()
                .unwrap(),
            b"old tables"
        );
        assert!(
            fs::symlink_metadata(directory.join("kodamaembed-linked.calibration"))
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(
            read(&directory.join("valid-target.bin"), "linked")
                .unwrap()
                .unwrap(),
            b"linked tables"
        );
        assert_eq!(
            fs::read(directory.join("kodamaembed-notes.calibration")).unwrap(),
            b"unrelated notes"
        );
        assert!(fs::symlink_metadata(directory.join("kodamaembed-pipe.calibration")).is_ok());
        if link {
            assert!(fs::symlink_metadata(directory.join("target.fifo")).is_ok());
        }
        let mut marker = File::options()
            .write(true)
            .create_new(true)
            .open(directory.join("complete"))
            .unwrap();
        marker
            .write_all(b"cache publication completed; neighboring artifacts retained\n")
            .unwrap();
        marker.flush().unwrap();
    }
}
