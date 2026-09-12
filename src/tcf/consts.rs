//! Wire-level constants for TCF v1. Every value is taken verbatim from
//! `FORMAT.md`; none is derived or guessed.

/// File magic at offset 0: `54 43 46 00 00 00 00 00` (`TCF\0\0\0\0\0`). Section 5.
pub const MAGIC: [u8; 8] = [0x54, 0x43, 0x46, 0x00, 0x00, 0x00, 0x00, 0x00];

/// Required `major` version field value. Section 5.
pub const MAJOR: u16 = 1;

/// Required `header_bytes` field value: exact header size in bytes, and the
/// header's own record size. Section 5.
pub const HEADER_BYTES: u32 = 192;

/// Required `schema_id` field value. Section 5.
pub const SCHEMA_ID: u32 = 1;

/// Every top-level section starts at a multiple of this. Section 4.1.
pub const SECTION_ALIGN: u64 = 64;

/// `ModuleRecord` size in bytes. Section 7.
pub const MODULE_RECORD_BYTES: usize = 128;

/// `TensorRecord` size in bytes. Section 8.
pub const TENSOR_RECORD_BYTES: usize = 256;

/// `ContractRecord` size in bytes. Section 9.
pub const CONTRACT_RECORD_BYTES: usize = 64;

/// `CalibrationRecord` size in bytes. Section 10.
pub const CALIBRATION_RECORD_BYTES: usize = 128;

/// `RelationRecord` size in bytes. Section 11.
pub const RELATION_RECORD_BYTES: usize = 64;

/// `WorkloadProfileRecord` size in bytes. Section 10.5.
pub const WORKLOAD_PROFILE_RECORD_BYTES: usize = 128;

/// Mandatory `proof_count` value for every quantized tensor. Section 15.3.
pub const PROOF_COUNT: u32 = 64;

/// `ModuleRecord.parent_id` sentinel meaning "root module". Section 7.
pub const ROOT_PARENT_ID: u32 = 0xffff_ffff;

/// `RelationRecord.input_tensor_id[i]` sentinel meaning "unused slot". Section 11.
pub const UNUSED_INPUT_ID: u32 = 0xffff_ffff;

/// Maximum valid `TensorRecord.rank`; minimum is 1. Section 8.
pub const MAX_RANK: u32 = 8;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_matches_spec_bytes() {
        assert_eq!(MAGIC, [0x54, 0x43, 0x46, 0x00, 0x00, 0x00, 0x00, 0x00]);
        assert_eq!(&MAGIC[0..3], b"TCF");
    }

    #[test]
    fn sentinels_are_all_ones() {
        assert_eq!(ROOT_PARENT_ID, u32::MAX);
        assert_eq!(UNUSED_INPUT_ID, u32::MAX);
    }

    #[test]
    fn record_sizes_are_64_byte_multiples() {
        for size in [
            HEADER_BYTES as usize,
            MODULE_RECORD_BYTES,
            TENSOR_RECORD_BYTES,
            CONTRACT_RECORD_BYTES,
            CALIBRATION_RECORD_BYTES,
            RELATION_RECORD_BYTES,
            WORKLOAD_PROFILE_RECORD_BYTES,
        ] {
            assert_eq!(size % 64, 0);
        }
    }
}
