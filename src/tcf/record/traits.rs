//! The `Record` trait: the shared decode/encode shape of the six TCF v1
//! array records. FORMAT.md Section 4.1 lists them in file order.

use crate::tcf::error::TcfError;

/// A fixed-size TCF v1 array record.
///
/// Implemented by `ModuleRecord` (Section 7), `TensorRecord` (Section 8), `ContractRecord`
/// (Section 9), `CalibrationRecord` (Section 10), `RelationRecord` (Section 11), and
/// `WorkloadProfileRecord` (Section 10.5).
///
/// `Header` (Section 5) deliberately does NOT implement this trait. It is a
/// singleton, not an array element, and its `header_digest` covers bytes
/// `[0,192)` with `[144,160)` treated as zero (Section 5.3) — a self-referential
/// exclusion no array record has. It exposes the same method names as
/// inherent items instead.
pub trait Record: Sized {
    /// Exact on-disk size of one record in bytes.
    const SIZE: usize;

    /// Decode one record from the first `SIZE` bytes of `bytes`.
    ///
    /// Every read is bounds-checked: a slice shorter than `SIZE` returns
    /// `TcfError::SectionBounds`, never a panic. A non-zero reserved byte
    /// returns `TcfError::NonzeroReserved` (Section 4, Section 8.1.5).
    fn decode(bytes: &[u8]) -> Result<Self, TcfError>;

    /// Encode this record into the first `SIZE` bytes of `out`, writing
    /// every reserved range as zero (Section 4).
    fn encode(&self, out: &mut [u8]) -> Result<(), TcfError>;
}
