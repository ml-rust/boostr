//! `Header`: exactly 192 bytes at offset 0. FORMAT.md Section 5.

use crate::tcf::consts::{HEADER_BYTES, MAGIC, MAJOR};
use crate::tcf::error::TcfError;
use crate::tcf::flags::{HeaderFlags, RequiredFeatures};

/// Post-decode check on the two fields that decide whether the file is a TCF
/// v1 file at all. Section 5, Section 17.
fn validate_identity(header: &Header) -> Result<(), TcfError> {
    if header.magic != MAGIC {
        return Err(TcfError::BadMagic);
    }
    if header.major != MAJOR {
        return Err(TcfError::UnsupportedMajor {
            major: header.major,
        });
    }
    Ok(())
}

crate::define_record! {
    /// The file header. Section 5.
    ///
    /// `Header` is a singleton, not an array record, so it does not
    /// implement [`crate::tcf::record::traits::Record`]; it exposes the same
    /// method names as inherent items. Its `header_digest` covers bytes
    /// `[0,192)` with `[144,160)` treated as zero (Section 5.3) — a
    /// self-referential exclusion no array record has. No digest is computed
    /// or checked in this unit, and section bounds are a later unit's job.
    pub struct Header;
    size = HEADER_BYTES as usize;
    mode = singleton;
    validate = validate_identity;
    rows {
        /// `54 43 46 00 00 00 00 00` (`TCF\0\0\0\0\0`). Section 5.
        field magic: [u8; 8] => 0, 8;
        /// `1` in v1. Section 5.
        field major: u16 => 8, 2;
        field minor: u16 => 10, 2;
        /// `192`. Section 5.
        field header_bytes: u32 => 12, 4;
        /// `1`. Section 5.
        field schema_id: u32 => 16, 4;
        /// Bit 0 `LITTLE_ENDIAN` must be 1 in v1. Section 5.1.
        flags flags: HeaderFlags => 20, 4;
        field tensor_count: u32 => 24, 4;
        field module_count: u32 => 28, 4;
        field contract_count: u32 => 32, 4;
        field calibration_count: u32 => 36, 4;
        field relation_count: u32 => 40, 4;
        field workload_count: u32 => 44, 4;
        /// An unknown bit here is a capability claim the reader cannot
        /// satisfy, so it rejects as `E_UNKNOWN_REQUIRED_FEATURE`, never as
        /// `E_NONZERO_RESERVED`. Section 5.2, Section 8.1.5.
        flags required_features: RequiredFeatures => 48, 8;
        field module_off: u64 => 56, 8;
        field tensor_off: u64 => 64, 8;
        field contract_off: u64 => 72, 8;
        field calibration_off: u64 => 80, 8;
        field relation_off: u64 => 88, 8;
        field string_off: u64 => 96, 8;
        field string_len: u64 => 104, 8;
        field proof_off: u64 => 112, 8;
        field proof_len: u64 => 120, 8;
        field data_off: u64 => 128, 8;
        field file_len: u64 => 136, 8;
        /// BLAKE3-128 over `[0,192)` with `[144,160)` treated as zero. Section 5.3.
        field header_digest: [u8; 16] => 144, 16;
        /// BLAKE3-128 over `[192, data_off)`, padding included. Section 5.3.
        field directory_digest: [u8; 16] => 160, 16;
        field workload_off: u64 => 176, 8;
        reserved => 184, 8;
    }
}

/// Byte range of `Header.header_digest`, treated as zero when computing it.
/// Section 5.3. The digest itself is a later unit's job.
pub const HEADER_DIGEST_RANGE: core::ops::Range<usize> = 144..160;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcf::record::testkit::put;

    fn sample() -> [u8; HEADER_BYTES as usize] {
        let mut b = [0u8; HEADER_BYTES as usize];
        put(&mut b, 0, &MAGIC);
        put(&mut b, 8, &MAJOR.to_le_bytes());
        put(&mut b, 10, &0u16.to_le_bytes());
        put(&mut b, 12, &HEADER_BYTES.to_le_bytes());
        put(&mut b, 16, &crate::tcf::consts::SCHEMA_ID.to_le_bytes());
        put(&mut b, 20, &HeaderFlags::LITTLE_ENDIAN.bits().to_le_bytes());
        put(&mut b, 24, &435u32.to_le_bytes());
        put(&mut b, 28, &200u32.to_le_bytes());
        put(&mut b, 32, &8u32.to_le_bytes());
        put(&mut b, 36, &2u32.to_le_bytes());
        put(&mut b, 40, &0u32.to_le_bytes());
        put(&mut b, 44, &1u32.to_le_bytes());
        put(&mut b, 48, &0x2fu64.to_le_bytes());
        put(&mut b, 56, &192u64.to_le_bytes());
        put(&mut b, 64, &25792u64.to_le_bytes());
        put(&mut b, 72, &137152u64.to_le_bytes());
        put(&mut b, 80, &137664u64.to_le_bytes());
        put(&mut b, 88, &137920u64.to_le_bytes());
        put(&mut b, 96, &138048u64.to_le_bytes());
        put(&mut b, 104, &17408u64.to_le_bytes());
        put(&mut b, 112, &155456u64.to_le_bytes());
        put(&mut b, 120, &55680u64.to_le_bytes());
        put(&mut b, 128, &211200u64.to_le_bytes());
        put(&mut b, 136, &4_255_211_200u64.to_le_bytes());
        put(&mut b, 144, &[0x99u8; 16]);
        put(&mut b, 160, &[0xaau8; 16]);
        put(&mut b, 176, &137792u64.to_le_bytes());
        b
    }

    #[test]
    fn roundtrips_byte_for_byte() {
        let bytes = sample();
        let header = Header::decode(&bytes).expect("valid header");
        let mut out = [0xffu8; HEADER_BYTES as usize];
        header.encode(&mut out).expect("encodes");
        assert_eq!(out, bytes);
    }

    #[test]
    fn decodes_counts_offsets_and_digests() {
        let h = Header::decode(&sample()).expect("valid header");
        assert_eq!(h.magic, MAGIC);
        assert_eq!(h.major, MAJOR);
        assert_eq!(h.header_bytes, HEADER_BYTES);
        assert_eq!(h.tensor_count, 435);
        assert_eq!(h.workload_off, 137792);
        assert_eq!(h.file_len, 4_255_211_200);
        assert_eq!(h.header_digest, [0x99u8; 16]);
        assert_eq!(h.directory_digest, [0xaau8; 16]);
        assert!(h.flags.contains(HeaderFlags::LITTLE_ENDIAN));
        assert!(
            h.required_features
                .contains(RequiredFeatures::WORKLOAD_PROFILES)
        );
        assert_eq!(HEADER_DIGEST_RANGE, 144..160);
    }

    #[test]
    fn bad_magic_and_major_are_rejected() {
        let mut bytes = sample();
        bytes[2] = b'X';
        assert_eq!(Header::decode(&bytes), Err(TcfError::BadMagic));

        let mut bytes = sample();
        put(&mut bytes, 8, &2u16.to_le_bytes());
        assert_eq!(
            Header::decode(&bytes),
            Err(TcfError::UnsupportedMajor { major: 2 })
        );
    }

    #[test]
    fn unknown_required_feature_bit_has_its_own_error() {
        let mut bytes = sample();
        put(&mut bytes, 48, &(1u64 << 7).to_le_bytes());
        assert_eq!(
            Header::decode(&bytes),
            Err(TcfError::UnknownRequiredFeature { bit: 7 })
        );
    }

    #[test]
    fn unknown_header_flag_bit_is_reserved() {
        let mut bytes = sample();
        put(&mut bytes, 20, &0b11u32.to_le_bytes());
        assert_eq!(
            Header::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "Header.flags"
            })
        );
    }

    #[test]
    fn nonzero_reserved_is_rejected() {
        let mut bytes = sample();
        bytes[191] = 1;
        assert_eq!(
            Header::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "Header.reserved@184"
            })
        );
    }

    #[test]
    fn short_slice_errors_rather_than_panics() {
        assert_eq!(
            Header::decode(&[0u8; 191]),
            Err(TcfError::SectionBounds { section: "Header" })
        );
    }
}
