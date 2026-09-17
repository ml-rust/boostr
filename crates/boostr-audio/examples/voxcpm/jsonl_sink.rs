//! Append-only JSONL sink shared by `voxcpm_clone` and `voxcpm_quality_gate`.

use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};

/// Append-only JSONL sink, flushed after every line.
///
/// Buffering to the end would lose the whole log when a sweep dies partway,
/// which is exactly the case the log exists for.
pub struct JsonlSink {
    file: File,
    path: PathBuf,
}

impl JsonlSink {
    pub fn create(path: &Path) -> Result<Self, String> {
        let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self {
            file,
            path: path.to_path_buf(),
        })
    }

    /// Write one object and flush it to the OS.
    ///
    /// Serialization is `serde_json`, already a boostr dependency, so the
    /// escaping of a prompt containing a quote or a tab is the library's
    /// problem rather than this file's.
    pub fn write(&mut self, value: &serde_json::Value) -> Result<(), String> {
        let line = serde_json::to_string(value)
            .map_err(|e| format!("{}: serializing record: {e}", self.path.display()))?;
        writeln!(self.file, "{line}").map_err(|e| format!("{}: {e}", self.path.display()))?;
        self.file
            .flush()
            .map_err(|e| format!("{}: {e}", self.path.display()))
    }
}
