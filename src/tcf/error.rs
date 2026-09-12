//! `TcfError`: the error type every fallible decode in this crate returns.
//! Variant names match the `E_*` codes in `FORMAT.md` Section 17 verbatim
//! (minus the `E_` prefix, cased to Rust convention). No variant panics its
//! way into existence — every decode path in this crate returns `Result`.
//!
//! A handful of variants are producer-side (payload checks, sink I/O) rather than
//! Section 17 reader codes; each of those is marked as such in its doc
//! comment and its `Display` message omits the `E_` prefix to keep it
//! visibly distinct from a Section 17 code.

use core::fmt;

/// A TCF v1 decode or validation failure. See FORMAT.md Section 17 for the
/// normative name and trigger condition of each variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TcfError {
    /// First 8 bytes are not `54 43 46 00 00 00 00 00`.
    BadMagic,
    /// `major != 1`.
    UnsupportedMajor { major: u16 },
    /// An unknown bit is set in `required_features`, or a bit this version
    /// mandates is clear.
    UnknownRequiredFeature { bit: u32 },
    /// `header_digest` fails.
    HeaderDigestMismatch,
    /// `directory_digest` fails.
    DirectoryDigestMismatch,
    /// A section range overflows, overlaps, falls outside `[192, data_off)`,
    /// or `header_bytes` is not 192.
    SectionBounds { section: &'static str },
    /// A section or `data_offset` is not 64-byte aligned.
    MisalignedSection { section: &'static str },
    /// A reserved field or padding byte is non-zero.
    NonzeroReserved { field: &'static str },
    /// `rank` outside 1..8, a dimension at index `>= rank` is non-zero in a
    /// record's padded `dims`, or a `dims` slice whose length is not `rank`.
    ///
    /// The last case catches a caller passing a record's padded `[u64; 8]`
    /// straight to a shape check that wants exactly `rank` dims.
    InvalidRank { rank: u32 },
    /// Block tensor with `rank < 2` or a last dimension the block length
    /// does not divide. Also raised when a tensor's proof fields
    /// (`proof_count`, `proof_format`, `proof_rel_off`) don't match its
    /// encoding: `64`/`1`/anything for block, `0`/`0`/`0` for raw.
    /// Section 15.3.
    InvalidQuantShape { tensor_id: u32 },
    /// Encoding differs from module preference with `fallback_reason = NONE`.
    MissingFallbackReason { tensor_id: u32 },
    /// `payload_digest` fails.
    PayloadDigestMismatch { tensor_id: u32 },
    /// Recomputed logical stream differs from `semantic_digest`.
    SemanticDigestMismatch { tensor_id: u32 },
    /// A proof value differs from the decoded value.
    ProofMismatch { tensor_id: u32, proof_index: u32 },
    /// No kernel matches `(encoding, contract_digest, role)`.
    ActivationContractMismatch { tensor_id: u32 },
    /// A recomputed `contract_digest` differs from the stored one, or a
    /// `contract_id` resolves to a differing digest.
    ContractDigestMismatch { contract_id: u32 },
    /// A recomputed `policy_digest` differs from the stored one.
    PolicyDigestMismatch { module_id: u32 },
    /// A recomputed `relation_digest` differs from the stored one.
    RelationDigestMismatch { output_tensor_id: u32 },
    /// `resident_bytes` or `transfer_bytes` differs from `physical_span_bytes`.
    ResidentBytesViolation { tensor_id: u32 },
    /// Encoding value not in the v1 registry.
    UnsupportedEncoding { raw: u16 },
    /// `execution_role` value not in the v1 registry.
    UnknownExecutionRole { raw: u16 },
    /// `schema_id` names a layout schema this version does not define.
    BadSchemaId { schema_id: u32 },
    /// Any other enumerated field carries a value not defined in v1.
    UnknownEnumValue { field: &'static str, raw: u32 },
    /// `ACCESS_PROFILE_VALID` set with no resolvable `workload_profile_id`.
    MissingWorkloadProfile { tensor_id: u32 },
    /// Block or proof arithmetic: a byte count, block count, or element
    /// index would overflow `u64`. Producer-side, not a Section 17 reader
    /// code. Section 12.3.
    TileArithmeticOverflow,
    /// A string-table name is not valid UTF-8. Section 6 requires the table
    /// to be UTF-8; Section 17 defines no code for a malformed name, because
    /// names are provenance and never dispatch. Reader-side, not a Section
    /// 17 code. Section 6.
    InvalidUtf8Name { off: u64, len: u32 },
    /// A tensor's payload disagrees with the `TensorRecord` registered for
    /// it, or no payload was supplied for it at all. Producer-side, not a
    /// Section 17 reader code: `tensor_index` is the tensor's position in
    /// the directory, `expected` is what the record determines, and
    /// `supplied` is what the producer handed over. Section 8.0.1,
    /// Section 12.3, Section 15.2.
    ///
    /// Raised by [`crate::tcf::writer::TcfWriter::finish_streaming`] when a
    /// `produce` callback returns the wrong payload kind or the wrong
    /// count, and by [`crate::tcf::writer::TcfWriter::finish_into`] when a
    /// tensor was registered without one.
    PayloadMismatch {
        tensor_index: u32,
        expected: String,
        supplied: String,
    },
    /// A seek or write on the sink a producer streams into failed.
    /// Producer-side, not a Section 17 reader code: a reader borrows an
    /// in-memory slice and performs no I/O.
    ///
    /// `std::io::Error` is neither `Clone`, `Eq`, nor `Hash`, so the two
    /// parts of it this crate needs are carried directly: `kind` is
    /// `Copy + Eq + Hash` and `message` is the original error's `Display`
    /// text. See [`crate::tcf::writer::TcfWriter::finish_into`].
    Io {
        kind: std::io::ErrorKind,
        message: String,
    },
}

impl fmt::Display for TcfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "E_BAD_MAGIC: first 8 bytes are not the TCF magic"),
            Self::UnsupportedMajor { major } => {
                write!(f, "E_UNSUPPORTED_MAJOR: major={major}, expected 1")
            }
            Self::UnknownRequiredFeature { bit } => {
                write!(
                    f,
                    "E_UNKNOWN_REQUIRED_FEATURE: unknown bit {bit} set in required_features"
                )
            }
            Self::HeaderDigestMismatch => {
                write!(
                    f,
                    "E_HEADER_DIGEST_MISMATCH: header_digest does not match bytes [0,192)"
                )
            }
            Self::DirectoryDigestMismatch => write!(
                f,
                "E_DIRECTORY_DIGEST_MISMATCH: directory_digest does not match bytes [192, data_off)"
            ),
            Self::SectionBounds { section } => write!(
                f,
                "E_SECTION_BOUNDS: section '{section}' overflows, overlaps, falls outside [192, data_off), or header_bytes is not 192"
            ),
            Self::MisalignedSection { section } => {
                write!(
                    f,
                    "E_MISALIGNED_SECTION: section '{section}' is not 64-byte aligned"
                )
            }
            Self::NonzeroReserved { field } => {
                write!(
                    f,
                    "E_NONZERO_RESERVED: reserved field '{field}' is non-zero"
                )
            }
            Self::InvalidRank { rank } => {
                write!(
                    f,
                    "E_INVALID_RANK: rank={rank}, must be 1..=8, the dims slice must hold \
                     exactly {rank} entries (a record's padded [u64; 8] must be sliced first), \
                     and a padded dimension at index >= rank must be zero"
                )
            }
            Self::InvalidQuantShape { tensor_id } => write!(
                f,
                "E_INVALID_QUANT_SHAPE: tensor {tensor_id} is block-encoded with rank < 2 or a last dimension its block length does not divide, or its proof fields don't match its encoding"
            ),
            Self::MissingFallbackReason { tensor_id } => write!(
                f,
                "E_MISSING_FALLBACK_REASON: tensor {tensor_id} encoding differs from module preference with fallback_reason = NONE"
            ),
            Self::PayloadDigestMismatch { tensor_id } => {
                write!(
                    f,
                    "E_PAYLOAD_DIGEST_MISMATCH: tensor {tensor_id} payload_digest mismatch"
                )
            }
            Self::SemanticDigestMismatch { tensor_id } => {
                write!(
                    f,
                    "E_SEMANTIC_DIGEST_MISMATCH: tensor {tensor_id} semantic_digest mismatch"
                )
            }
            Self::ProofMismatch {
                tensor_id,
                proof_index,
            } => write!(
                f,
                "E_PROOF_MISMATCH: tensor {tensor_id} proof index {proof_index} does not match the decoded value"
            ),
            Self::ActivationContractMismatch { tensor_id } => write!(
                f,
                "E_ACTIVATION_CONTRACT_MISMATCH: no kernel matches tensor {tensor_id}'s (encoding, contract_digest, execution_role)"
            ),
            Self::ContractDigestMismatch { contract_id } => write!(
                f,
                "E_CONTRACT_DIGEST_MISMATCH: contract {contract_id} contract_digest differs from the recomputed digest, or a contract_id resolves to a differing digest"
            ),
            Self::PolicyDigestMismatch { module_id } => write!(
                f,
                "E_POLICY_DIGEST_MISMATCH: module {module_id} policy_digest differs from the recomputed digest"
            ),
            Self::RelationDigestMismatch { output_tensor_id } => write!(
                f,
                "E_RELATION_DIGEST_MISMATCH: relation on output tensor {output_tensor_id} relation_digest differs from the recomputed digest"
            ),
            Self::ResidentBytesViolation { tensor_id } => write!(
                f,
                "E_RESIDENT_BYTES_VIOLATION: tensor {tensor_id} resident_bytes or transfer_bytes differs from physical_span_bytes"
            ),
            Self::UnsupportedEncoding { raw } => {
                write!(
                    f,
                    "E_UNSUPPORTED_ENCODING: encoding 0x{raw:04x} is not in the v1 registry"
                )
            }
            Self::UnknownExecutionRole { raw } => {
                write!(
                    f,
                    "E_UNKNOWN_EXECUTION_ROLE: execution_role {raw} is not in the v1 registry"
                )
            }
            Self::BadSchemaId { schema_id } => {
                write!(
                    f,
                    "E_BAD_SCHEMA_ID: schema_id={schema_id} names a layout schema this version does not define"
                )
            }
            Self::UnknownEnumValue { field, raw } => {
                write!(
                    f,
                    "E_UNKNOWN_ENUM_VALUE: field '{field}' has value {raw}, undefined in v1"
                )
            }
            Self::MissingWorkloadProfile { tensor_id } => write!(
                f,
                "E_MISSING_WORKLOAD_PROFILE: tensor {tensor_id} has ACCESS_PROFILE_VALID set with no resolvable workload_profile_id"
            ),
            Self::TileArithmeticOverflow => {
                write!(
                    f,
                    "TILE_ARITHMETIC_OVERFLOW: block geometry arithmetic overflows u64"
                )
            }
            Self::InvalidUtf8Name { off, len } => write!(
                f,
                "INVALID_UTF8_NAME: string table name at ({off}, {len}) is not valid UTF-8"
            ),
            Self::PayloadMismatch {
                tensor_index,
                expected,
                supplied,
            } => write!(
                f,
                "PAYLOAD_MISMATCH: tensor index {tensor_index} requires {expected}, the producer supplied {supplied}"
            ),
            Self::Io { kind, message } => {
                write!(f, "IO: sink {kind:?} while writing the file: {message}")
            }
        }
    }
}

impl std::error::Error for TcfError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_names_the_field_and_raw_value() {
        let err = TcfError::UnknownEnumValue {
            field: "module_role",
            raw: 99,
        };
        let msg = err.to_string();
        assert!(msg.contains("module_role"));
        assert!(msg.contains("99"));
    }

    #[test]
    fn display_names_unsupported_encoding_raw() {
        let err = TcfError::UnsupportedEncoding { raw: 0x0199 };
        assert!(err.to_string().contains("0199"));
    }

    #[test]
    fn is_a_std_error() {
        fn assert_error<E: std::error::Error>(_: &E) {}
        assert_error(&TcfError::BadMagic);
    }
}
