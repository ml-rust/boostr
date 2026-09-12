//! The `(name_off, name_len)` string-table reference (Section 6).

use crate::tcf::error::TcfError;

use super::scalar::RecordField;

/// A `(name_off, name_len)` pair into the string table. Section 6.
///
/// It carries the raw offset and length only. `off` is relative to
/// `Header.string_off`; resolving it to UTF-8 bytes is a later unit's job.
/// Names are provenance, not identity: runtime dispatch uses `tensor_id`,
/// `module_id`, and `role`, so there is no length ceiling on a name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct StringRef {
    /// Byte offset relative to `Header.string_off`.
    pub off: u64,
    /// Byte length of the UTF-8 name.
    pub len: u32,
}

impl StringRef {
    /// A `(off, len)` pair.
    #[must_use]
    pub const fn new(off: u64, len: u32) -> Self {
        Self { off, len }
    }

    /// Byte offset relative to `Header.string_off`.
    #[must_use]
    pub const fn off(self) -> u64 {
        self.off
    }

    /// Byte length of the UTF-8 name.
    #[must_use]
    pub const fn len(self) -> u32 {
        self.len
    }

    /// True when the name is the empty string.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Decode the pair from its two separately-specified offsets.
    ///
    /// # Errors
    /// `TcfError::SectionBounds` when either read runs past `bytes`.
    pub fn read(
        bytes: &[u8],
        off_at: usize,
        len_at: usize,
        record: &'static str,
    ) -> Result<Self, TcfError> {
        Ok(Self {
            off: u64::read(bytes, off_at, record)?,
            len: u32::read(bytes, len_at, record)?,
        })
    }

    /// Encode the pair at its two separately-specified offsets.
    ///
    /// # Errors
    /// `TcfError::SectionBounds` when either write runs past `out`.
    pub fn write(
        &self,
        out: &mut [u8],
        off_at: usize,
        len_at: usize,
        record: &'static str,
    ) -> Result<(), TcfError> {
        self.off.write(out, off_at, record)?;
        self.len.write(out, len_at, record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_ref_roundtrips() {
        let mut buf = [0u8; 16];
        let sref = StringRef::new(0x1234_5678_9abc, 42);
        sref.write(&mut buf, 0, 8, "T").expect("in bounds");
        assert_eq!(StringRef::read(&buf, 0, 8, "T"), Ok(sref));
        assert!(!sref.is_empty());
    }
}
