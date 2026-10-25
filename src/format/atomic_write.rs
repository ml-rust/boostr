//! Crash-safe file replacement

use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Most symlinks [`resolve_target`] follows, the Linux `MAXSYMLINKS` value.
const MAX_SYMLINK_HOPS: usize = 40;

/// Write `path` atomically through `write`
///
/// The function follows `path` through every symlink to the file it names,
/// so a symlinked target (such as a Hugging Face cache snapshot entry) keeps
/// its link and the file behind the link gets the new content. It creates
/// `.<file_name>.tmp-<pid>-<n>` beside that file, so the rename stays on one
/// filesystem. It writes, flushes and syncs the temp file, gives it the
/// permissions of the file it replaces, renames it over that file, then
/// syncs the directory on unix. A new file gets the default permissions of
/// a created file. Owner and group become those of the writing process.
///
/// Errors:
/// - An error before the rename removes the temp file and returns the
///   original error. An existing file stays unchanged.
/// - An error from the directory sync after the rename returns an error of
///   the same kind that says so. The file then holds the new content, and a
///   crash before the directory reaches disk can bring back the old file.
pub fn write_atomic<P, F>(path: P, write: F) -> io::Result<()>
where
    P: AsRef<Path>,
    F: FnOnce(&mut BufWriter<File>) -> io::Result<()>,
{
    let target = resolve_target(path.as_ref())?;
    let name = target
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "path has no file name"))?;
    let dir = match target.parent() {
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
    let result = fill_and_rename(file, &temp, &target, write);
    if result.is_err() {
        let _ = fs::remove_file(&temp);
        return result;
    }
    sync_dir(dir).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!(
                "{} holds the new content, but syncing its directory {} failed: {e}. A crash \
                 before the directory reaches disk can bring back the old file",
                target.display(),
                dir.display()
            ),
        )
    })
}

/// The file `path` names after every symlink: `path` itself when it is no
/// symlink or does not exist, else the end of its link chain. A relative
/// link resolves against the directory of the link.
///
/// # Errors
///
/// A link that cannot be read, a metadata error other than not-found, or
/// more than [`MAX_SYMLINK_HOPS`] links.
fn resolve_target(path: &Path) -> io::Result<PathBuf> {
    let mut current = path.to_path_buf();
    for _ in 0..MAX_SYMLINK_HOPS {
        match fs::symlink_metadata(&current) {
            Ok(meta) if meta.file_type().is_symlink() => {
                let link = fs::read_link(&current)?;
                current = match current.parent() {
                    Some(dir) if link.is_relative() => dir.join(link),
                    _ => link,
                };
            }
            Ok(_) => return Ok(current),
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(current),
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        format!(
            "{} passes through more than {MAX_SYMLINK_HOPS} symlinks: check it for a link loop",
            path.display()
        ),
    ))
}

fn fill_and_rename<F>(file: File, temp: &Path, target: &Path, write: F) -> io::Result<()>
where
    F: FnOnce(&mut BufWriter<File>) -> io::Result<()>,
{
    let mut writer = BufWriter::new(file);
    write(&mut writer)?;
    writer.flush()?;
    let file = writer.into_inner().map_err(|e| e.into_error())?;
    match fs::metadata(target) {
        Ok(existing) => file.set_permissions(existing.permissions())?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
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

    /// The Hugging Face cache layout: `snapshots/<rev>/<file>` links to
    /// `../../blobs/<hash>`.
    #[cfg(unix)]
    #[test]
    fn a_symlinked_target_keeps_its_link_and_the_blob_gets_the_content() {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let blobs = root.path().join("blobs");
        let snap = root.path().join("snapshots").join("rev");
        fs::create_dir_all(&blobs).unwrap();
        fs::create_dir_all(&snap).unwrap();
        let blob = blobs.join("abc123");
        fs::write(&blob, b"old").unwrap();
        let link = snap.join("model.safetensors");
        symlink(Path::new("../../blobs/abc123"), &link).unwrap();
        // A second link to the first: the chain resolves to the blob.
        let outer = root.path().join("model.safetensors");
        symlink(&link, &outer).unwrap();

        write_atomic(&outer, |w| w.write_all(b"new")).unwrap();

        for l in [&link, &outer] {
            assert!(fs::symlink_metadata(l).unwrap().file_type().is_symlink());
        }
        assert_eq!(fs::read(&blob).unwrap(), b"new");
        assert_eq!(fs::read(&outer).unwrap(), b"new");
        assert_eq!(dir_names(&blobs), ["abc123"]);
        assert_eq!(dir_names(&snap), ["model.safetensors"]);
    }

    #[cfg(unix)]
    #[test]
    fn a_dangling_symlink_creates_the_file_it_names() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let link = dir.path().join("link");
        symlink("target.bin", &link).unwrap();
        write_atomic(&link, |w| w.write_all(b"made")).unwrap();
        assert!(
            fs::symlink_metadata(&link)
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert_eq!(fs::read(dir.path().join("target.bin")).unwrap(), b"made");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_loop_is_an_error() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("a"), dir.path().join("b"));
        symlink(&b, &a).unwrap();
        symlink(&a, &b).unwrap();
        let err = write_atomic(&a, |w| w.write_all(b"x")).unwrap_err();
        assert!(err.to_string().contains("link loop"), "{err}");
    }

    #[cfg(unix)]
    #[test]
    fn the_replaced_file_keeps_its_permissions() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.bin");
        fs::write(&path, b"old").unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
        write_atomic(&path, |w| w.write_all(b"new")).unwrap();
        let mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o640);
        assert_eq!(fs::read(&path).unwrap(), b"new");
    }
}
