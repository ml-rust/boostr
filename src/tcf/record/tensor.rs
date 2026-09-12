//! `TensorRecord`: 256 bytes. FORMAT.md Section 8.

use crate::tcf::consts::{MAX_RANK, TENSOR_RECORD_BYTES};
use crate::tcf::encoding::Encoding;
use crate::tcf::enums::{
    ExecutionRole, FallbackReason, LayoutId, ProofFormat, ResidencyClass, Role,
};
use crate::tcf::error::TcfError;
use crate::tcf::flags::TensorFlags;

/// Post-decode cross-field check: `rank` is 1 through 8 and dimensions at
/// index `>= rank` are zero. A block-encoded tensor's proof fields are
/// `proof_count = 64`, `proof_format = 1`; a raw tensor's are
/// `proof_count = 0`, `proof_format = 0`, `proof_rel_off = 0` — there is
/// nothing to prove.
fn validate_shape(record: &TensorRecord) -> Result<(), TcfError> {
    if record.rank < 1 || record.rank > MAX_RANK {
        return Err(TcfError::InvalidRank { rank: record.rank });
    }
    for (i, dim) in record.dims.iter().enumerate() {
        let trailing = u32::try_from(i).unwrap_or(u32::MAX) >= record.rank;
        if trailing && *dim != 0 {
            return Err(TcfError::InvalidRank { rank: record.rank });
        }
    }
    match record.encoding {
        Encoding::Block(_) => {
            if record.proof_count != crate::tcf::consts::PROOF_COUNT
                || record.proof_format != ProofFormat::DequantF16
            {
                return Err(TcfError::InvalidQuantShape {
                    tensor_id: record.tensor_id,
                });
            }
        }
        Encoding::Raw(_) => {
            if record.proof_count != 0
                || record.proof_format != ProofFormat::None
                || record.proof_rel_off != 0
            {
                return Err(TcfError::InvalidQuantShape {
                    tensor_id: record.tensor_id,
                });
            }
        }
    }
    Ok(())
}

crate::define_record! {
    /// One tensor's directory entry: intrinsic truth, producer decision, and
    /// workload observation in one flat fixed-size record. Section 8, Section 10.5.1,
    /// Section 15.3.
    ///
    /// The f32 telemetry fields decode structurally only; finiteness policy
    /// is deferred to the verifier unit. Digest fields are opaque: nothing
    /// here computes or checks a digest.
    pub struct TensorRecord;
    size = TENSOR_RECORD_BYTES;
    mode = record;
    validate = validate_shape;
    rows {
        field tensor_id: u32 => 0, 4;
        field module_id: u32 => 4, 4;
        /// `(name_off, name_len)` into the string table. Section 6.
        strref name: (name_off => 8, name_len => 16);
        /// Semantic metadata, never a dispatch key. Section 8.7.
        field role: Role => 20, 2;
        field encoding: Encoding => 22, 2;
        /// Mandatory whenever `encoding` differs from the module's
        /// highest-ranked `preferred_encoding`. Section 8.6.
        field fallback_reason: FallbackReason => 24, 2;
        field residency_class: ResidencyClass => 26, 2;
        flags flags: TensorFlags => 28, 4;
        /// 1 through 8. Section 8.
        field rank: u32 => 32, 4;
        field calibration_id: u32 => 36, 4;
        /// Eight u64. Every entry at index `>= rank` is zero. Section 8.
        field dims: [u64; 8] => 40, 64;
        field activation_contract_id: u32 => 104, 4;
        field layout_id: LayoutId => 108, 4;
        /// Absolute, and a multiple of 64. Section 8.
        field data_offset: u64 => 112, 8;
        field logical_payload_bytes: u64 => 120, 8;
        field physical_span_bytes: u64 => 128, 8;
        field resident_bytes: u64 => 136, 8;
        field transfer_bytes: u64 => 144, 8;
        /// Task delta, valid only when `SENSITIVITY_VALID` is set. Section 8.2.
        field sensitivity_delta: f32 => 152, 4;
        field sensitivity_ci95: f32 => 156, 4;
        /// Valid only when `ACCESS_PROFILE_VALID` is set, and read against
        /// `workload_profile_id`. Section 10.5.1.
        field accesses_per_generation: f32 => 160, 4;
        field bytes_read_per_generation: f32 => 164, 4;
        field sensitivity_samples: u32 => 168, 4;
        field sensitivity_seed_count: u32 => 172, 4;
        field access_profile_samples: u32 => 176, 4;
        // v1 defines no per-tensor backend restriction (Section 8.3): a mask
        // enumerating CUDA/METAL/WGSL grows with every new API, and the two
        // real cases already have homes (`residency_class = HOST_ONLY`,
        // `E_ACTIVATION_CONTRACT_MISMATCH`). Reserved as an extension point
        // a future capability-based restriction can claim with its own
        // `required_features` bit.
        reserved => 180, 4;
        /// The dispatch key. Section 8.6.1.
        field execution_role: ExecutionRole => 184, 2;
        reserved => 186, 2;
        field workload_profile_id: u32 => 188, 4;
        /// BLAKE3-128, carried opaquely. Section 15.2.
        field semantic_digest: [u8; 16] => 192, 16;
        /// BLAKE3-128, carried opaquely. Section 15.1.
        field payload_digest: [u8; 16] => 208, 16;
        /// Relative to `Header.proof_off`. Section 15.3.
        field proof_rel_off: u64 => 224, 8;
        field proof_count: u32 => 232, 4;
        field proof_format: ProofFormat => 236, 4;
        reserved => 240, 16;
    }
}

impl TensorRecord {
    /// The `rank` significant dimensions, trailing zeros excluded. Section 8.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        let rank = usize::try_from(self.rank).unwrap_or(0).min(self.dims.len());
        match self.dims.get(..rank) {
            Some(shape) => shape,
            None => &[],
        }
    }

    /// `logical_payload_bytes` as the shape and the encoding determine it.
    /// Section 8.0.1.
    ///
    /// The stored field is never the authority: for a block encoding the
    /// length is the block byte count the shape implies (Section 12.3), and
    /// for a raw encoding it is `product(dims) * width`
    /// (Section 8.0.1). A producer computes this rather than accepting one,
    /// and a reader rejects a stored value that disagrees — otherwise a
    /// truncated or padded raw payload round-trips unnoticed, because
    /// nothing else in the file constrains a raw tensor's length.
    ///
    /// # Errors
    /// - [`TcfError::InvalidQuantShape`] if a block-encoded tensor has
    ///   `rank < 2`, or a row width that is not a whole number of blocks.
    /// - [`TcfError::InvalidRank`] if `rank` is outside `1..=8`.
    /// - [`TcfError::TileArithmeticOverflow`] if the length overflows `u64`.
    pub fn determined_payload_bytes(&self) -> Result<u64, TcfError> {
        match self.encoding {
            // A block stream is whole blocks per row, so the row width must
            // divide by the block width.
            Encoding::Block(block) => block.payload_bytes(self.shape(), self.rank, self.tensor_id),
            Encoding::Raw(raw) => {
                let shape = self.shape();
                if shape.is_empty() {
                    return Err(TcfError::InvalidRank { rank: self.rank });
                }
                let mut elements: u64 = 1;
                for dim in shape {
                    elements = elements
                        .checked_mul(*dim)
                        .ok_or(TcfError::TileArithmeticOverflow)?;
                }
                elements
                    .checked_mul(raw.width_bytes())
                    .ok_or(TcfError::TileArithmeticOverflow)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcf::encoding::{BlockEncoding, RawEncoding};
    use crate::tcf::record::field::StringRef;
    use crate::tcf::record::testkit::{put, roundtrips, short_slice_errors};
    use crate::tcf::record::traits::Record;

    fn sample() -> [u8; TENSOR_RECORD_BYTES] {
        let mut b = [0u8; TENSOR_RECORD_BYTES];
        put(&mut b, 0, &42u32.to_le_bytes());
        put(&mut b, 4, &7u32.to_le_bytes());
        put(&mut b, 8, &128u64.to_le_bytes());
        put(&mut b, 16, &24u32.to_le_bytes());
        put(&mut b, 20, &Role::LinearWeight.to_u16().to_le_bytes());
        put(
            &mut b,
            22,
            &Encoding::Block(BlockEncoding::Q6K).to_u16().to_le_bytes(),
        );
        put(&mut b, 24, &FallbackReason::None.to_u16().to_le_bytes());
        put(&mut b, 26, &ResidencyClass::Hot.to_u16().to_le_bytes());
        put(
            &mut b,
            28,
            &TensorFlags::SENSITIVITY_VALID
                .union(TensorFlags::TASK_CRITICAL)
                .bits()
                .to_le_bytes(),
        );
        put(&mut b, 32, &2u32.to_le_bytes());
        put(&mut b, 36, &1u32.to_le_bytes());
        put(&mut b, 40, &4096u64.to_le_bytes());
        put(&mut b, 48, &11008u64.to_le_bytes());
        put(&mut b, 104, &1u32.to_le_bytes());
        put(&mut b, 108, &LayoutId::RowMajorDense.to_u32().to_le_bytes());
        put(&mut b, 112, &4096u64.to_le_bytes());
        put(&mut b, 120, &36_634_624u64.to_le_bytes());
        put(&mut b, 128, &36_634_624u64.to_le_bytes());
        put(&mut b, 136, &36_634_624u64.to_le_bytes());
        put(&mut b, 144, &36_634_624u64.to_le_bytes());
        put(&mut b, 152, &0.0125f32.to_le_bytes());
        put(&mut b, 156, &0.0031f32.to_le_bytes());
        put(&mut b, 160, &1.0f32.to_le_bytes());
        put(&mut b, 164, &36_634_624.0f32.to_le_bytes());
        put(&mut b, 168, &256u32.to_le_bytes());
        put(&mut b, 172, &3u32.to_le_bytes());
        put(&mut b, 176, &512u32.to_le_bytes());
        put(&mut b, 184, &ExecutionRole::Matmul.to_u16().to_le_bytes());
        // 180: reserved (formerly `backend_forbid_mask`), left zero.
        put(&mut b, 188, &1u32.to_le_bytes());
        put(&mut b, 192, &[0x11u8; 16]);
        put(&mut b, 208, &[0x22u8; 16]);
        put(&mut b, 224, &8192u64.to_le_bytes());
        put(&mut b, 232, &crate::tcf::consts::PROOF_COUNT.to_le_bytes());
        put(&mut b, 236, &ProofFormat::DequantF16.to_u32().to_le_bytes());
        b
    }

    #[test]
    fn roundtrips_byte_for_byte() {
        roundtrips::<TensorRecord>(&sample());
    }

    /// Whole 210-byte Q6_K super-blocks per row: 4096 x 11008 is 4096 rows
    /// of 43 super-blocks.
    #[test]
    fn determined_payload_bytes_for_a_block_encoding() {
        let rec = TensorRecord::decode(&sample()).expect("valid record");
        assert_eq!(rec.determined_payload_bytes(), Ok(4096 * 43 * 210));
    }

    /// Section 8.0.1: `product(dims) * width` for a raw encoding, one case
    /// per width class.
    #[test]
    fn determined_payload_bytes_for_a_raw_encoding() {
        let mut bytes = sample();
        // A raw tensor has no proof vector. Section 15.3.
        put(&mut bytes, 224, &0u64.to_le_bytes());
        put(&mut bytes, 232, &0u32.to_le_bytes());
        put(&mut bytes, 236, &ProofFormat::None.to_u32().to_le_bytes());
        for (encoding, width) in [
            (RawEncoding::F32, 4u64),
            (RawEncoding::Bf16, 2),
            (RawEncoding::I8, 1),
        ] {
            put(
                &mut bytes,
                22,
                &Encoding::Raw(encoding).to_u16().to_le_bytes(),
            );
            let rec = TensorRecord::decode(&bytes).expect("valid record");
            assert_eq!(
                rec.determined_payload_bytes(),
                Ok(4096 * 11008 * width),
                "{encoding:?}"
            );
        }
    }

    #[test]
    fn decodes_shape_and_telemetry() {
        let rec = TensorRecord::decode(&sample()).expect("valid record");
        assert_eq!(rec.tensor_id, 42);
        assert_eq!(rec.name, StringRef::new(128, 24));
        assert_eq!(rec.rank, 2);
        assert_eq!(rec.shape(), &[4096, 11008]);
        assert_eq!(rec.execution_role, ExecutionRole::Matmul);
        assert!(rec.flags.contains(TensorFlags::TASK_CRITICAL));
        assert!((rec.sensitivity_delta - 0.0125).abs() < 1e-9);
        assert_eq!(rec.semantic_digest, [0x11u8; 16]);
        assert_eq!(rec.payload_digest, [0x22u8; 16]);
    }

    #[test]
    fn rank_out_of_range_is_rejected() {
        for bad in [0u32, 9, u32::MAX] {
            let mut bytes = sample();
            put(&mut bytes, 32, &bad.to_le_bytes());
            put(&mut bytes, 40, &[0u8; 64]);
            assert_eq!(
                TensorRecord::decode(&bytes),
                Err(TcfError::InvalidRank { rank: bad })
            );
        }
    }

    #[test]
    fn nonzero_trailing_dim_is_rejected() {
        let mut bytes = sample();
        put(&mut bytes, 40 + 2 * 8, &1u64.to_le_bytes());
        assert_eq!(
            TensorRecord::decode(&bytes),
            Err(TcfError::InvalidRank { rank: 2 })
        );
    }

    #[test]
    fn block_quantized_proof_fields_must_match() {
        // Every combination that decodes as a valid ProofFormat but is wrong
        // for a quantized tensor, which requires exactly (64, DequantF16).
        for (count, format) in [(0u32, 0u32), (64, 0), (0, 1), (63, 1)] {
            let mut bytes = sample();
            put(&mut bytes, 232, &count.to_le_bytes());
            put(&mut bytes, 236, &format.to_le_bytes());
            assert_eq!(
                TensorRecord::decode(&bytes),
                Err(TcfError::InvalidQuantShape { tensor_id: 42 })
            );
        }
    }

    #[test]
    fn undefined_proof_format_is_rejected_before_shape_validation() {
        // Section 15.3 defines only 0 and 1. An undefined value is an unknown enum
        // value, caught at field decode, not a shape inconsistency.
        let mut bytes = sample();
        put(&mut bytes, 236, &2u32.to_le_bytes());
        assert_eq!(
            TensorRecord::decode(&bytes),
            Err(TcfError::UnknownEnumValue {
                field: "proof_format",
                raw: 2
            })
        );
    }

    #[test]
    fn raw_encoding_requires_zeroed_proof_fields() {
        let mut bytes = sample();
        put(
            &mut bytes,
            22,
            &Encoding::Raw(crate::tcf::encoding::RawEncoding::F16)
                .to_u16()
                .to_le_bytes(),
        );
        for (rel_off, count, format) in [
            (0u64, 1u32, 0u32),
            (0, 0, 1),
            (1, 0, 0),
            (
                8192,
                crate::tcf::consts::PROOF_COUNT,
                ProofFormat::DequantF16.to_u32(),
            ),
        ] {
            let mut b = bytes;
            put(&mut b, 224, &rel_off.to_le_bytes());
            put(&mut b, 232, &count.to_le_bytes());
            put(&mut b, 236, &format.to_le_bytes());
            assert_eq!(
                TensorRecord::decode(&b),
                Err(TcfError::InvalidQuantShape { tensor_id: 42 })
            );
        }

        put(&mut bytes, 224, &0u64.to_le_bytes());
        put(&mut bytes, 232, &0u32.to_le_bytes());
        put(&mut bytes, 236, &0u32.to_le_bytes());
        assert!(TensorRecord::decode(&bytes).is_ok());
    }

    #[test]
    fn nonzero_reserved_is_rejected() {
        let mut bytes = sample();
        bytes[180] = 1;
        assert_eq!(
            TensorRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "TensorRecord.reserved@180"
            })
        );

        let mut bytes = sample();
        bytes[186] = 1;
        assert_eq!(
            TensorRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "TensorRecord.reserved@186"
            })
        );

        let mut bytes = sample();
        bytes[255] = 1;
        assert_eq!(
            TensorRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "TensorRecord.reserved@240"
            })
        );
    }

    #[test]
    fn unknown_flag_bits_are_rejected() {
        let mut bytes = sample();
        put(&mut bytes, 28, &(1u32 << 5).to_le_bytes());
        assert_eq!(
            TensorRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "TensorRecord.flags"
            })
        );
    }

    #[test]
    fn short_slice_errors_rather_than_panics() {
        short_slice_errors::<TensorRecord>("TensorRecord");
    }
}
