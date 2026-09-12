//! Header checks `TcfFile::open` runs before it decodes a record: magic,
//! version, feature bits, flags, and the two header digests. Section 5.

use crate::tcf::consts::{HEADER_BYTES, MAGIC, MAJOR};
use crate::tcf::digest::{Digest128, hash_128};
use crate::tcf::error::TcfError;
use crate::tcf::flags::{HeaderFlags, RequiredFeatures};
use crate::tcf::record::field::RecordField;
use crate::tcf::record::{HEADER_DIGEST_RANGE, Header};

/// Section 5: the first 8 bytes.
pub(super) fn check_magic(head: &[u8]) -> Result<(), TcfError> {
    match head.get(..MAGIC.len()) {
        Some(m) if m == MAGIC.as_slice() => Ok(()),
        _ => Err(TcfError::BadMagic),
    }
}

/// Section 5: `major = 1`.
pub(super) fn check_major(head: &[u8]) -> Result<(), TcfError> {
    let major = <u16 as RecordField>::read(head, 8, "Header")?;
    if major != MAJOR {
        return Err(TcfError::UnsupportedMajor { major });
    }
    Ok(())
}

/// Section 5.2, Section 8.1.5: an unknown `required_features` bit is a
/// capability claim the reader cannot satisfy, and is rejected before any
/// other flag field.
pub(super) fn check_required_feature_bits(head: &[u8]) -> Result<(), TcfError> {
    let raw = <u64 as RecordField>::read(head, 48, "Header")?;
    let unknown = RequiredFeatures::from_bits_retain(raw).unknown_bits();
    if unknown != 0 {
        return Err(TcfError::UnknownRequiredFeature {
            bit: unknown.trailing_zeros(),
        });
    }
    Ok(())
}

/// Section 5.2: bits 0 through 3 MUST be set; bit 4 is set exactly when
/// `relation_count > 0`; bit 5 exactly when `workload_count > 0`.
///
/// A mandated bit whose state contradicts Section 5.2 is reported as
/// `E_UNKNOWN_REQUIRED_FEATURE` naming that bit. Section 17 defines no
/// separate code for a feature claim that is absent rather than unknown, and
/// the reader's action is the same in both cases: reject, naming the bit.
pub(super) fn check_required_feature_counts(header: &Header) -> Result<(), TcfError> {
    let features = header.required_features;
    let mandatory = [
        (0u32, RequiredFeatures::ACTIVATION_CONTRACTS),
        (1, RequiredFeatures::PLACEMENT_METADATA),
        (2, RequiredFeatures::SEMANTIC_DIGESTS),
        (3, RequiredFeatures::SOURCE_PROOFS),
    ];
    for (bit, flag) in mandatory {
        if !features.contains(flag) {
            return Err(TcfError::UnknownRequiredFeature { bit });
        }
    }
    let conditional = [
        (4u32, RequiredFeatures::RELATIONS, header.relation_count > 0),
        (
            5,
            RequiredFeatures::WORKLOAD_PROFILES,
            header.workload_count > 0,
        ),
    ];
    for (bit, flag, expected) in conditional {
        if features.contains(flag) != expected {
            return Err(TcfError::UnknownRequiredFeature { bit });
        }
    }
    Ok(())
}

/// Section 5.1: bit 0 `LITTLE_ENDIAN` set, every other bit zero.
pub(super) fn check_header_flags(head: &[u8]) -> Result<(), TcfError> {
    let raw = <u32 as RecordField>::read(head, 20, "Header")?;
    let flags = HeaderFlags::from_bits_retain(raw);
    if flags.unknown_bits() != 0 || !flags.contains(HeaderFlags::LITTLE_ENDIAN) {
        return Err(TcfError::NonzeroReserved {
            field: "Header.flags",
        });
    }
    Ok(())
}

/// Section 5.3: BLAKE3-128 over `[0,192)` with `[144,160)` treated as zero.
pub(super) fn check_header_digest(head: &[u8], header: &Header) -> Result<(), TcfError> {
    let mut image = [0u8; HEADER_BYTES as usize];
    let source = head
        .get(..HEADER_BYTES as usize)
        .ok_or(TcfError::SectionBounds { section: "Header" })?;
    image.copy_from_slice(source);
    let zeroed = image
        .get_mut(HEADER_DIGEST_RANGE)
        .ok_or(TcfError::SectionBounds { section: "Header" })?;
    zeroed.fill(0);
    if hash_128(&image) != Digest128::from_bytes(header.header_digest) {
        return Err(TcfError::HeaderDigestMismatch);
    }
    Ok(())
}

/// Section 5.3: BLAKE3-128 over the exact range `[192, data_off)`, section
/// padding included.
pub(super) fn check_directory_digest(directory: &[u8], header: &Header) -> Result<(), TcfError> {
    let body = directory
        .get(HEADER_BYTES as usize..)
        .ok_or(TcfError::SectionBounds {
            section: "data_off",
        })?;
    if hash_128(body) != Digest128::from_bytes(header.directory_digest) {
        return Err(TcfError::DirectoryDigestMismatch);
    }
    Ok(())
}

/// Section 5.2: `required_features` bit 6 named the two-level tile
/// encodings, which were retired. A file claiming it is refused, as an
/// older reader refuses a capability it lacks.
pub(super) fn check_two_level_feature_bit(header: &Header) -> Result<(), TcfError> {
    if header
        .required_features
        .contains(RequiredFeatures::TWO_LEVEL_SCALES)
    {
        return Err(TcfError::UnknownRequiredFeature { bit: 6 });
    }
    Ok(())
}
