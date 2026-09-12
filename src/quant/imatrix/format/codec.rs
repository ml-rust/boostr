//! Byte layout: the writer, the reader, and every way a file is refused
//! rather than misread.

use std::io::{Read, Write};
use std::path::Path;

use crate::error::Result;

use super::matrix::{
    IMATRIX_HEADER_LEN, IMATRIX_MAGIC, IMATRIX_VERSION, ImportanceEntry, ImportanceMatrix,
    MAX_NAME_LEN, invalid,
};

impl ImportanceMatrix {
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
mod tests {
    use super::super::matrix::tests::{entry, sample};
    use super::*;

    #[test]
    fn round_trip_preserves_every_field() {
        let matrix = sample();
        let parsed = ImportanceMatrix::from_bytes(&matrix.to_bytes()).unwrap();
        assert_eq!(parsed, matrix);
        assert_eq!(parsed.token_count(), 4096);
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn two_matrices_with_the_same_content_produce_identical_bytes() {
        // Insertion order differs; the file must not.
        let mut reversed = ImportanceMatrix::new(4096);
        reversed
            .insert(entry(
                "model.layers.0.self_attn.q_proj.weight",
                512,
                &[3.0, 4.0, 5.0],
            ))
            .unwrap();
        reversed
            .insert(entry(
                "model.layers.0.mlp.down_proj.weight",
                512,
                &[1.0, 2.0],
            ))
            .unwrap();
        assert_eq!(reversed.to_bytes(), sample().to_bytes());
    }

    #[test]
    fn header_is_the_documented_length() {
        assert_eq!(
            sample().to_bytes()[..IMATRIX_MAGIC.len()],
            IMATRIX_MAGIC[..]
        );
        assert_eq!(IMATRIX_HEADER_LEN, 32);
    }

    #[test]
    fn wrong_magic_is_refused() {
        let mut bytes = sample().to_bytes();
        bytes[0] = b'X';
        let err = ImportanceMatrix::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("magic"), "{err}");
    }

    #[test]
    fn unknown_version_is_refused() {
        let mut bytes = sample().to_bytes();
        bytes[8..12].copy_from_slice(&(IMATRIX_VERSION + 1).to_le_bytes());
        assert!(ImportanceMatrix::from_bytes(&bytes).is_err());
    }

    #[test]
    fn set_reserved_flag_is_refused() {
        let mut bytes = sample().to_bytes();
        bytes[12..16].copy_from_slice(&1u32.to_le_bytes());
        let err = ImportanceMatrix::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("flags"), "{err}");
    }

    #[test]
    fn truncated_file_is_refused() {
        let bytes = sample().to_bytes();
        for cut in [
            4,
            IMATRIX_HEADER_LEN,
            IMATRIX_HEADER_LEN + 8,
            bytes.len() - 1,
        ] {
            assert!(
                ImportanceMatrix::from_bytes(&bytes[..cut]).is_err(),
                "accepted a file truncated to {cut} byte(s)"
            );
        }
    }

    #[test]
    fn trailing_bytes_are_refused() {
        let mut bytes = sample().to_bytes();
        bytes.push(0);
        let err = ImportanceMatrix::from_bytes(&bytes).unwrap_err();
        assert!(err.to_string().contains("trailing"), "{err}");
    }

    #[test]
    fn file_round_trips_through_a_path() {
        let dir = std::env::temp_dir().join(format!("boostr-imatrix-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("importance.bstrimtx");
        sample().write_to_path(&path).unwrap();
        assert_eq!(ImportanceMatrix::read_from_path(&path).unwrap(), sample());
        std::fs::remove_dir_all(&dir).ok();
    }
}
