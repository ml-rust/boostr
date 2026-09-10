//! The importance-matrix FILE: its layout, its writer, its reader, and the
//! check a consumer runs before trusting it.
//!
//! One definition, in the library, because `compressr` reads this file for
//! BOTH codecs (GGUF K-quants and TCF) and `compressr` depends on `boostr`.
//! A parser written a second time beside the consumer is the exact failure
//! this repo already records for block layouts: a writer checked against a
//! reader beside it proves nothing.
//!
//! # Layout
//!
//! Little-endian throughout. No padding, no alignment requirement: every
//! multi-byte field is read from a byte slice, so a file written on one
//! machine is read identically on another.
//!
//! Header, 32 bytes:
//!
//! | offset | size | field         | value                                    |
//! | ------ | ---- | ------------- | ---------------------------------------- |
//! | 0      | 8    | `magic`       | `BSTRIMTX`                               |
//! | 8      | 4    | `version`     | `1`                                      |
//! | 12     | 4    | `flags`       | `0` — reserved, a reader REJECTS nonzero |
//! | 16     | 8    | `entry_count` | entries that follow                      |
//! | 24     | 8    | `token_count` | tokens the whole run accumulated over    |
//!
//! Then `entry_count` entries, back to back, each:
//!
//! | size            | field         | meaning                                |
//! | --------------- | ------------- | -------------------------------------- |
//! | 4               | `name_len`    | length of `name` in bytes              |
//! | 8               | `in_features` | columns of the weight this describes   |
//! | 8               | `rows`        | activation rows summed into this entry |
//! | `name_len`      | `name`        | UTF-8 tensor name, no trailing NUL     |
//! | `in_features*4` | `sums`        | `f32` per column                       |
//!
//! # What a value means
//!
//! `sums[j]` is `sum over rows of x_j * x_j` — a SUM, never a mean. The
//! divisor is `rows`, carried beside it, so a consumer picks its own
//! normalization and two files collected over different corpus sizes stay
//! combinable. `token_count` is the run-level total, which lets a consumer
//! tell a 10-token collection from a 100k-token one at a glance.
//!
//! # Absent is not zero
//!
//! A tensor the run never exercised has NO entry. It is never written as a
//! zero vector: zero importance and "never measured" mean opposite things to
//! a quantizer, and only one of them is safe to act on.
//!
//! # Entry order
//!
//! Entries are written sorted by name, so two runs that collected the same
//! statistics produce byte-identical files regardless of the order the
//! collector observed the tensors in.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::Path;

use crate::error::{Error, Result};

/// First 8 bytes of every importance file.
pub const IMATRIX_MAGIC: [u8; 8] = *b"BSTRIMTX";
/// Format version this module writes, and the only one it reads.
pub const IMATRIX_VERSION: u32 = 1;
/// Bytes before the first entry.
pub const IMATRIX_HEADER_LEN: usize = 32;
/// Longest tensor name accepted, in bytes. A checkpoint key is far shorter;
/// the cap exists so a corrupt `name_len` fails immediately instead of
/// reserving an absurd allocation.
pub const MAX_NAME_LEN: usize = 1024;

fn invalid(reason: impl Into<String>) -> Error {
    Error::InvalidArgument {
        arg: "imatrix",
        reason: reason.into(),
    }
}

/// One weight's column importance.
#[derive(Clone, Debug, PartialEq)]
pub struct ImportanceEntry {
    /// Checkpoint tensor name, exactly as the quantizer sees it.
    pub name: String,
    /// Activation rows summed into `sums`. The divisor for a mean.
    pub rows: u64,
    /// `sum(x_j^2)` per input column `j`. Length is the weight's `in_features`.
    pub sums: Vec<f32>,
}

impl ImportanceEntry {
    /// Columns this entry describes — the weight's `in_features`.
    pub fn in_features(&self) -> usize {
        self.sums.len()
    }

    /// Per-column mean squared activation, `sums / rows`.
    ///
    /// `None` when `rows` is zero: a mean over nothing has no value, and a
    /// zero vector would read as "measured, and unimportant".
    pub fn mean_square(&self) -> Option<Vec<f32>> {
        if self.rows == 0 {
            return None;
        }
        let divisor = self.rows as f32;
        Some(self.sums.iter().map(|v| v / divisor).collect())
    }
}

/// What a file holds: one entry per MEASURED tensor, plus the run's token count.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImportanceMatrix {
    entries: BTreeMap<String, ImportanceEntry>,
    token_count: u64,
}

impl ImportanceMatrix {
    /// Empty matrix for a run that will accumulate over `token_count` tokens.
    pub fn new(token_count: u64) -> Self {
        Self {
            entries: BTreeMap::new(),
            token_count,
        }
    }

    /// Tokens the whole run accumulated over.
    pub fn token_count(&self) -> u64 {
        self.token_count
    }

    /// Add one measured tensor. A duplicate name is an ERROR: silently
    /// keeping one of two disagreeing vectors picks a quantization at random.
    pub fn insert(&mut self, entry: ImportanceEntry) -> Result<()> {
        if entry.name.is_empty() {
            return Err(invalid("tensor name is empty"));
        }
        if entry.name.len() > MAX_NAME_LEN {
            return Err(invalid(format!(
                "tensor name is {} bytes, over the {MAX_NAME_LEN}-byte cap",
                entry.name.len()
            )));
        }
        if entry.sums.is_empty() {
            return Err(invalid(format!(
                "'{}': zero-length column vector",
                entry.name
            )));
        }
        check_values(&entry.name, &entry.sums)?;
        if self.entries.contains_key(&entry.name) {
            return Err(invalid(format!("'{}': recorded twice", entry.name)));
        }
        self.entries.insert(entry.name.clone(), entry);
        Ok(())
    }

    /// One measured tensor by name, or `None` when the run never exercised it.
    pub fn get(&self, name: &str) -> Option<&ImportanceEntry> {
        self.entries.get(name)
    }

    /// Every entry, in name order — the order the file stores them in.
    pub fn entries(&self) -> impl Iterator<Item = &ImportanceEntry> {
        self.entries.values()
    }

    /// Measured tensors in this matrix.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the run measured nothing at all.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Compare against the weights of the model being quantized.
    ///
    /// `expected` is `(tensor name, in_features)` per quantizable weight.
    /// A name present in BOTH whose `in_features` disagree is an ERROR — that
    /// is the file-does-not-match-this-model case, and importance read at the
    /// wrong width is worse than no importance at all. Everything else is
    /// reported for the caller to act on: an absent name means the run never
    /// exercised that weight, and an unknown name means the file describes a
    /// weight this model does not have.
    pub fn check_against(&self, expected: &[(String, usize)]) -> Result<ImportanceCheck> {
        let mut check = ImportanceCheck::default();
        for (name, in_features) in expected {
            match self.entries.get(name) {
                Some(entry) if entry.in_features() != *in_features => {
                    return Err(invalid(format!(
                        "'{name}': file holds {} column(s), this model's weight has {in_features}",
                        entry.in_features()
                    )));
                }
                Some(_) => check.present.push(name.clone()),
                None => check.absent.push(name.clone()),
            }
        }
        let known: std::collections::HashSet<&str> =
            expected.iter().map(|(name, _)| name.as_str()).collect();
        for name in self.entries.keys() {
            if !known.contains(name.as_str()) {
                check.unknown.push(name.clone());
            }
        }
        Ok(check)
    }

    /// Serialize to bytes in the layout this module's docs state.
    pub fn to_bytes(&self) -> Vec<u8> {
        let payload: usize = self
            .entries
            .values()
            .map(|e| 20 + e.name.len() + e.sums.len() * 4)
            .sum();
        let mut out = Vec::with_capacity(IMATRIX_HEADER_LEN + payload);
        out.extend_from_slice(&IMATRIX_MAGIC);
        out.extend_from_slice(&IMATRIX_VERSION.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.token_count.to_le_bytes());
        // `BTreeMap` iterates in name order, which is what pins the byte
        // layout: two runs holding the same statistics write the same file.
        for entry in self.entries.values() {
            out.extend_from_slice(&(entry.name.len() as u32).to_le_bytes());
            out.extend_from_slice(&(entry.sums.len() as u64).to_le_bytes());
            out.extend_from_slice(&entry.rows.to_le_bytes());
            out.extend_from_slice(entry.name.as_bytes());
            for value in &entry.sums {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        out
    }

    /// Write to `path`, creating or truncating it.
    pub fn write_to_path(&self, path: &Path) -> Result<()> {
        let mut file = std::fs::File::create(path)?;
        file.write_all(&self.to_bytes())?;
        file.flush()?;
        Ok(())
    }

    /// Parse a file's bytes, rejecting anything this module did not write.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < IMATRIX_HEADER_LEN {
            return Err(invalid(format!(
                "file is {} byte(s), shorter than the {IMATRIX_HEADER_LEN}-byte header",
                bytes.len()
            )));
        }
        if bytes[..8] != IMATRIX_MAGIC {
            return Err(invalid(
                "not an importance matrix: magic bytes do not match",
            ));
        }
        let version = read_u32(bytes, 8);
        if version != IMATRIX_VERSION {
            return Err(invalid(format!(
                "version {version}, this build reads version {IMATRIX_VERSION}"
            )));
        }
        let flags = read_u32(bytes, 12);
        if flags != 0 {
            return Err(invalid(format!(
                "flags {flags:#x}: this build defines none, so a set bit means the file \
                 carries something it cannot interpret"
            )));
        }
        let entry_count = read_u64(bytes, 16);
        let token_count = read_u64(bytes, 24);

        let mut matrix = Self::new(token_count);
        let mut at = IMATRIX_HEADER_LEN;
        for index in 0..entry_count {
            let (entry, next) = read_entry(bytes, at, index)?;
            matrix.insert(entry)?;
            at = next;
        }
        if at != bytes.len() {
            return Err(invalid(format!(
                "{} trailing byte(s) after {entry_count} entry(s)",
                bytes.len() - at
            )));
        }
        Ok(matrix)
    }

    /// Read and parse the file at `path`.
    pub fn read_from_path(path: &Path) -> Result<Self> {
        let mut bytes = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut bytes)?;
        Self::from_bytes(&bytes)
    }
}

/// What [`ImportanceMatrix::check_against`] found. A shape disagreement never
/// reaches here — it is an error, not a finding.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ImportanceCheck {
    /// Model weights the file measured, at the matching width.
    pub present: Vec<String>,
    /// Model weights the run never exercised. NOT zero importance.
    pub absent: Vec<String>,
    /// File entries naming a weight this model does not have.
    pub unknown: Vec<String>,
}

/// A sum of squares is finite and non-negative. Anything else means the bytes
/// were interpreted as something they are not.
fn check_values(name: &str, values: &[f32]) -> Result<()> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() || *value < 0.0 {
            return Err(invalid(format!(
                "'{name}' column {index}: {value} is not a finite non-negative sum of squares"
            )));
        }
    }
    Ok(())
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    let mut raw = [0u8; 4];
    raw.copy_from_slice(&bytes[at..at + 4]);
    u32::from_le_bytes(raw)
}

fn read_u64(bytes: &[u8], at: usize) -> u64 {
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(raw)
}

/// Parse one entry starting at `at`, returning it and the offset after it.
fn read_entry(bytes: &[u8], at: usize, index: u64) -> Result<(ImportanceEntry, usize)> {
    let short = |want: usize| {
        invalid(format!(
            "entry {index}: needs {want} more byte(s) than the file holds"
        ))
    };
    if bytes.len() < at + 20 {
        return Err(short(20));
    }
    let name_len = read_u32(bytes, at) as usize;
    let in_features = read_u64(bytes, at + 4) as usize;
    let rows = read_u64(bytes, at + 12);
    if name_len == 0 || name_len > MAX_NAME_LEN {
        return Err(invalid(format!(
            "entry {index}: name length {name_len} is outside 1..={MAX_NAME_LEN}"
        )));
    }
    if in_features == 0 {
        return Err(invalid(format!("entry {index}: zero columns")));
    }
    let name_at = at + 20;
    let values_at = name_at + name_len;
    let end = values_at
        .checked_add(in_features * 4)
        .ok_or_else(|| invalid(format!("entry {index}: column count overflows the file")))?;
    if bytes.len() < end {
        return Err(short(end - bytes.len()));
    }
    let name = std::str::from_utf8(&bytes[name_at..values_at])
        .map_err(|e| invalid(format!("entry {index}: name is not UTF-8: {e}")))?
        .to_string();
    let sums = bytes[values_at..end]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|raw| f32::from_le_bytes(*raw))
        .collect();
    Ok((ImportanceEntry { name, rows, sums }, end))
}

#[cfg(test)]
mod tests;
