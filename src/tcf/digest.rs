//! TCF v1 digests: the two truncated BLAKE3 widths the format stores, and
//! the two content digests this crate can compute from logical values.
//! FORMAT.md Section 5.3, Section 7, Section 9, Section 10,
//! Section 10.5, Section 11, Section 15.1, Section 15.2.
//!
//! Record-level digests are BLAKE3-128 over a byte range each record owns.
//! `header_digest` and `directory_digest` cover ranges only the writer's
//! layout knows, so they are computed there over [`hash_128`]. The three
//! digests defined over a single encoded record — [`policy_digest`]
//! (Section 7), [`contract_digest`] (Section 9), [`relation_digest`]
//! (Section 11) — live here, because Section 9 requires a producer to
//! compute all three and a reader to verify all three, and both sides must
//! call the same code. This module also supplies [`payload_digest`], the
//! digest over a tensor's stored bytes. A block-encoded tensor's semantic
//! digest IS its payload digest: the GGML block stream is the logical value.

use crate::tcf::error::TcfError;

/// Bytes of a `ModuleRecord` covered by `policy_digest`, before the name is
/// concatenated. Section 7.
const POLICY_DIGEST_BYTES: usize = 64;

/// Bytes of a `ContractRecord` covered by `contract_digest`. Section 9.
const CONTRACT_DIGEST_BYTES: usize = 40;

/// Width of the `contract_id` field zeroed inside that range. Section 9.
const CONTRACT_ID_BYTES: usize = 4;

/// Bytes of a `RelationRecord` covered by `relation_digest`. Section 11.
const RELATION_DIGEST_BYTES: usize = 32;

/// A BLAKE3 digest truncated to its leading 16 bytes. Section 5.3.
///
/// This is the width every digest field in the format uses:
/// `header_digest`, `directory_digest`, `policy_digest` (Section 7),
/// `contract_digest` (Section 9), `relation_digest` (Section 11),
/// `semantic_digest` and `payload_digest` (Section 15).
///
/// **Not a signature.** Section 5.3 and Section 15.4 are explicit: these
/// digests detect accidental corruption and packing or layout divergence.
/// They are unkeyed and unauthenticated, so they cannot stop a deliberately
/// malicious producer from fabricating both bad data and a matching digest.
/// Do not treat a match as proof of origin or of authorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Digest128([u8; 16]);

impl Digest128 {
    /// Wrap 16 stored bytes, e.g. as read from a record's digest field.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// The 16 digest bytes, in the order they are stored.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

/// A full-width, 32-byte BLAKE3 digest. Section 10, Section 10.5.
///
/// `CalibrationRecord` stores a dataset content BLAKE3-256 and
/// `WorkloadProfileRecord` stores a workload content BLAKE3-256, both 32
/// bytes. Carries the same caveat as [`Digest128`]: a content identity check,
/// never an adversarial signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Digest256([u8; 32]);

impl Digest256 {
    /// Wrap 32 stored bytes, e.g. as read from a record's digest field.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The 32 digest bytes, in the order they are stored.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// BLAKE3 over `input`, truncated to its leading 16 bytes. Section 5.3.
///
/// Every BLAKE3-128 field in the format is this function over that field's
/// governing byte range: bytes `[0,192)` with `[144,160)` zeroed for
/// `header_digest`, `[192, data_off)` for `directory_digest`, `[0,64)` plus
/// the UTF-8 name for `policy_digest` (Section 7), `[0,40)` with
/// `contract_id` zeroed for `contract_digest` (Section 9), `[0,32)` for
/// `relation_digest` (Section 11).
#[must_use]
pub fn hash_128(input: &[u8]) -> Digest128 {
    truncate_128(blake3::hash(input).as_bytes())
}

/// BLAKE3 over `input`, full 32-byte output. Section 10, Section 10.5.
#[must_use]
pub fn hash_256(input: &[u8]) -> Digest256 {
    let full = blake3::hash(input);
    let mut out = [0u8; 32];
    for (dst, src) in out.iter_mut().zip(full.as_bytes().iter()) {
        *dst = *src;
    }
    Digest256(out)
}

/// `policy_digest`: BLAKE3-128 over bytes `[0,64)` of an encoded
/// `ModuleRecord`, concatenated with the module's exact UTF-8 name bytes.
/// Section 7.
///
/// The digest field itself lies at `[64,80)`, outside the hashed range, so
/// the record's own stored digest never feeds the computation: Section 7
/// hashes the first 64 bytes "with the digest field absent".
///
/// `name` is the string-table slice `(name_off, name_len)` names, byte for
/// byte. A module with no name contributes no bytes. The name is hashed
/// as-is and is never validated as UTF-8 here — Section 6 makes a name
/// provenance, and a malformed one is a separate error the caller raises.
///
/// Section 9: **a producer MUST compute it, and a reader MUST verify it.**
/// A mismatch is `E_POLICY_DIGEST_MISMATCH`.
///
/// # Errors
/// [`TcfError::SectionBounds`] if `record` is shorter than 64 bytes.
pub fn policy_digest(record: &[u8], name: &[u8]) -> Result<Digest128, TcfError> {
    let head = record
        .get(..POLICY_DIGEST_BYTES)
        .ok_or(TcfError::SectionBounds {
            section: "ModuleRecord",
        })?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(head);
    hasher.update(name);
    Ok(truncate_128(hasher.finalize().as_bytes()))
}

/// `contract_digest`: BLAKE3-128 over bytes `[0,40)` of an encoded
/// `ContractRecord` with `contract_id` at `[0,4)` treated as zero.
/// Section 9.
///
/// The identity kernel dispatch uses. `contract_id` is excluded so the same
/// contract carries the same digest whatever local number a file gives it,
/// which is what lets a runtime refuse two files that agree on the id and
/// disagree on the contract.
///
/// Section 9: **a producer MUST compute it, and a reader MUST verify it.**
/// A mismatch is `E_CONTRACT_DIGEST_MISMATCH`.
///
/// # Errors
/// [`TcfError::SectionBounds`] if `record` is shorter than 40 bytes.
pub fn contract_digest(record: &[u8]) -> Result<Digest128, TcfError> {
    let head = record
        .get(..CONTRACT_DIGEST_BYTES)
        .ok_or(TcfError::SectionBounds {
            section: "ContractRecord",
        })?;
    let mut image = [0u8; CONTRACT_DIGEST_BYTES];
    image.copy_from_slice(head);
    image
        .get_mut(..CONTRACT_ID_BYTES)
        .ok_or(TcfError::SectionBounds {
            section: "ContractRecord",
        })?
        .fill(0);
    Ok(hash_128(&image))
}

/// `relation_digest`: BLAKE3-128 over bytes `[0,32)` of an encoded
/// `RelationRecord` — every field up to but excluding the digest itself.
/// Section 11.
///
/// No name is concatenated, because a relation has none. It exists so a
/// runtime can detect a relation rewired to different operands without
/// re-reading the tensors it names.
///
/// Section 9: **a producer MUST compute it, and a reader MUST verify it.**
/// A mismatch is `E_RELATION_DIGEST_MISMATCH`.
///
/// # Errors
/// [`TcfError::SectionBounds`] if `record` is shorter than 32 bytes.
pub fn relation_digest(record: &[u8]) -> Result<Digest128, TcfError> {
    let head = record
        .get(..RELATION_DIGEST_BYTES)
        .ok_or(TcfError::SectionBounds {
            section: "RelationRecord",
        })?;
    Ok(hash_128(head))
}

/// `payload_digest`: BLAKE3-128 over exactly `logical_payload_bytes`.
/// Section 15.1.
///
/// The caller passes the tensor's logical payload with trailing 64-byte
/// alignment padding **excluded** (Section 14.4). Padding is covered by no
/// digest, so including it here would make the digest disagree with every
/// other conforming implementation.
///
/// Question answered: did these exact stored bytes change?
#[must_use]
pub fn payload_digest(payload: &[u8]) -> Digest128 {
    hash_128(payload)
}

/// Leading 16 bytes of a 32-byte BLAKE3 output. Section 5.3.
fn truncate_128(full: &[u8; 32]) -> Digest128 {
    let mut out = [0u8; 16];
    for (dst, src) in out.iter_mut().zip(full.iter()) {
        *dst = *src;
    }
    Digest128(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The official BLAKE3 test-vector input: byte `i` is `i % 251`.
    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    fn to_hex(bytes: &[u8]) -> String {
        let mut out = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }

    /// Official BLAKE3 test vectors (unkeyed hash, full 32-byte output),
    /// asserted **before** truncation so the primitive itself is pinned to
    /// an independent oracle rather than to this crate's own round trip.
    /// Covers the empty input, a short input, and inputs on both sides of
    /// the 1024-byte chunk boundary.
    #[test]
    fn official_blake3_vectors() {
        let cases: [(usize, &str); 6] = [
            (
                0,
                "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262",
            ),
            (
                1,
                "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213",
            ),
            (
                3,
                "e1be4d7a8ab5560aa4199eea339849ba8e293d55ca0a81006726d184519e647f",
            ),
            (
                1023,
                "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11",
            ),
            (
                1024,
                "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7",
            ),
            (
                1025,
                "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444",
            ),
        ];
        for (len, expected) in cases {
            let digest = hash_256(&pattern(len));
            assert_eq!(to_hex(digest.as_bytes()), expected, "input length {len}");
        }
    }

    /// A `Digest128` is the leading half of the same BLAKE3 output.
    #[test]
    fn digest128_is_the_leading_16_bytes_of_the_full_hash() {
        let input = pattern(1025);
        let full = hash_256(&input);
        let short = hash_128(&input);
        assert_eq!(short.as_bytes(), &full.as_bytes()[..16]);
        assert_eq!(to_hex(short.as_bytes()), "d00278ae47eb27b34faecf67b4fe263f");
    }

    #[test]
    fn digest_newtypes_round_trip_their_bytes() {
        let short = Digest128::from_bytes([7u8; 16]);
        assert_eq!(short.as_bytes(), &[7u8; 16]);
        let long = Digest256::from_bytes([9u8; 32]);
        assert_eq!(long.as_bytes(), &[9u8; 32]);
        assert_eq!(short, Digest128::from_bytes([7u8; 16]));
    }

    /// Section 15.1: the payload digest is BLAKE3-128 over exactly the
    /// logical payload bytes, so appending alignment padding must change it.
    #[test]
    fn payload_digest_excludes_alignment_padding() {
        let payload = pattern(100);
        let mut padded = payload.clone();
        padded.resize(128, 0);
        assert_eq!(payload_digest(&payload), hash_128(&payload));
        assert_ne!(payload_digest(&payload), payload_digest(&padded));
    }

    /// Section 7: bytes `[0,64)` then the name, and nothing else. Bytes at
    /// or past 64 — the digest field itself included — cannot change it.
    #[test]
    fn policy_digest_covers_bytes_0_to_64_and_the_name() {
        let record = pattern(128);
        let base = policy_digest(&record, b"block.0").expect("digest");
        assert_eq!(
            base,
            hash_128(&[&record[..64], b"block.0".as_slice()].concat())
        );

        let mut later = record.clone();
        later[64] ^= 0xff;
        later[127] ^= 0xff;
        assert_eq!(policy_digest(&later, b"block.0"), Ok(base));

        let mut covered = record.clone();
        covered[63] ^= 0xff;
        assert_ne!(policy_digest(&covered, b"block.0"), Ok(base));

        assert_ne!(policy_digest(&record, b"block.1"), Ok(base));
        assert_ne!(policy_digest(&record, b""), Ok(base));
    }

    /// Section 9: bytes `[0,40)` with `contract_id` zeroed. Renumbering a
    /// contract leaves the digest alone; changing a covered field does not.
    #[test]
    fn contract_digest_covers_bytes_0_to_40_without_contract_id() {
        let record = pattern(64);
        let base = contract_digest(&record).expect("digest");

        let mut renumbered = record.clone();
        renumbered[..4].copy_from_slice(&0xdead_beefu32.to_le_bytes());
        assert_eq!(contract_digest(&renumbered), Ok(base));

        let mut later = record.clone();
        later[40] ^= 0xff;
        assert_eq!(contract_digest(&later), Ok(base));

        let mut covered = record.clone();
        covered[39] ^= 0xff;
        assert_ne!(contract_digest(&covered), Ok(base));
    }

    /// Section 11: bytes `[0,32)`, the digest field at `[32,48)` excluded.
    #[test]
    fn relation_digest_covers_bytes_0_to_32() {
        let record = pattern(64);
        let base = relation_digest(&record).expect("digest");
        assert_eq!(base, hash_128(&record[..32]));

        let mut later = record.clone();
        later[32] ^= 0xff;
        assert_eq!(relation_digest(&later), Ok(base));

        let mut covered = record.clone();
        covered[31] ^= 0xff;
        assert_ne!(relation_digest(&covered), Ok(base));
    }

    /// A short slice errors rather than panicking, like every other decode
    /// in this crate.
    #[test]
    fn a_short_record_errors_rather_than_panics() {
        assert_eq!(
            policy_digest(&pattern(63), b""),
            Err(TcfError::SectionBounds {
                section: "ModuleRecord"
            })
        );
        assert_eq!(
            contract_digest(&pattern(39)),
            Err(TcfError::SectionBounds {
                section: "ContractRecord"
            })
        );
        assert_eq!(
            relation_digest(&pattern(31)),
            Err(TcfError::SectionBounds {
                section: "RelationRecord"
            })
        );
    }
}
