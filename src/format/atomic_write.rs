//! Crash-safe file replacement

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Write `path` atomically through `write`
///
/// The function creates `.<file_name>.tmp-<pid>-<n>` in the same directory,
/// so the final rename stays on one filesystem. It flushes and syncs the
/// temp file, renames it over `path`, then syncs the parent directory on unix.
///
/// On any error the function removes the temp file and returns the original
/// error. An existing file at `path` stays unchanged.
pub(crate) fn write_atomic<P, F>(path: P, write: F) -> io::Result<()>
where
    P: AsRef<Path>,
    F: FnOnce(&mut BufWriter<File>) -> io::Result<()>,
{
    let path = path.as_ref();
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no file name"))?;
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };

    let mut temp_name = OsString::from(".");
    temp_name.push(name);
    temp_name.push(format!(
        ".tmp-{}-{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let temp = dir.join(temp_name);

    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    let result = fill_and_rename(file, &temp, path, write);
    if result.is_err() {
        let _ = fs::remove_file(&temp);
        return result;
    }
    sync_dir(dir)
}

fn fill_and_rename<F>(file: File, temp: &Path, target: &Path, write: F) -> io::Result<()>
where
    F: FnOnce(&mut BufWriter<File>) -> io::Result<()>,
{
    let mut writer = BufWriter::new(file);
    write(&mut writer)?;
    writer.flush()?;
    let file = writer.into_inner().map_err(|e| e.into_error())?;
    file.sync_all()?;
    drop(file);
    fs::rename(temp, target)
}

#[cfg(unix)]
fn sync_dir(dir: &Path) -> io::Result<()> {
    match File::open(dir) {
        Ok(d) => d.sync_all(),
        Err(_) => Ok(()),
    }
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir_names(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn writes_and_replaces_target() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.json");
        write_atomic(&path, |w| w.write_all(b"one")).unwrap();
        write_atomic(&path, |w| w.write_all(b"two")).unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"two");
        assert_eq!(dir_names(dir.path()), vec!["a.json".to_string()]);
    }

    #[test]
    fn closure_error_removes_temp_and_keeps_target() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.json");
        fs::write(&path, b"good").unwrap();

        let err = write_atomic(&path, |w| {
            w.write_all(b"partial")?;
            Err(io::Error::other("boom"))
        })
        .unwrap_err();

        assert_eq!(err.to_string(), "boom");
        assert_eq!(fs::read(&path).unwrap(), b"good");
        assert_eq!(dir_names(dir.path()), vec!["a.json".to_string()]);
    }

    #[test]
    fn closure_error_without_target_leaves_directory_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.bin");
        let result = write_atomic(&path, |_| Err(io::Error::other("boom")));
        assert!(result.is_err());
        assert!(dir_names(dir.path()).is_empty());
    }

    #[test]
    fn path_without_file_name_is_an_error() {
        assert!(write_atomic(Path::new("/"), |_| Ok(())).is_err());
    }
}
