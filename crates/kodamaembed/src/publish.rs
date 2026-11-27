//! Atomic publication keeps incomplete files out of the cache.

use std::{
    fs::{self, File},
    io,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

/// `before_commit` lets callers observe a complete write before this call makes it visible at `path`.
pub(crate) fn publish(
    path: &Path,
    write: impl FnOnce(File) -> io::Result<()>,
    before_commit: impl FnOnce(),
) -> io::Result<()> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let (partial, file) = create_partial(path, &NEXT)?;
    let written = write(file).and_then(|()| {
        before_commit();
        fs::rename(&partial, path)
    });
    if written.is_err() {
        // Preserve the original write or rename error if cleanup also fails.
        let _ = fs::remove_file(&partial);
    }
    written
}

pub(crate) fn create_partial(path: &Path, next: &AtomicU64) -> io::Result<(PathBuf, File)> {
    let name = path.file_name().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "a published path names a file")
    })?;
    loop {
        let ticket = next.fetch_add(1, Ordering::Relaxed);
        let mut partial = name.to_os_string();
        partial.push(format!(".{}.{ticket}.partial", std::process::id()));
        let partial = path.with_file_name(partial);
        match File::options().write(true).create_new(true).open(&partial) {
            Ok(file) => return Ok((partial, file)),
            // A reused process ID can collide with a partial file left by a crashed writer.
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
}
