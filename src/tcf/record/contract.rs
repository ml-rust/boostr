//! `ContractRecord`: 64 bytes. FORMAT.md Section 9.

use crate::tcf::consts::CONTRACT_RECORD_BYTES;
use crate::tcf::enums::{
    DotAccumulator, InputRepresentation, MathMode, OutputDtype, QuantAxis, RoundingMode,
    ScaleComputeDtype,
};
use crate::tcf::flags::ContractFlags;

crate::define_record! {
    /// The activation contract a kernel must satisfy. Section 9.
    ///
    /// `contract_digest` is BLAKE3-128 over bytes `[0,40)` with
    /// `contract_id` treated as zero. It is an integrity identity: it
    /// detects a corrupted or tampered record and distinguishes two
    /// records that share a `contract_id`. It plays no part in kernel
    /// dispatch, which resolves on the typed semantic fields (Section 8.6).
    /// This unit carries the digest opaquely: it is computed by
    /// [`crate::tcf::digest::contract_digest`], filled in by the writer, and
    /// verified by the reader at `open` (Section 9).
    ///
    /// `qmin` and `qmax` are the only signed integers in the v1 schema.
    pub struct ContractRecord;
    size = CONTRACT_RECORD_BYTES;
    mode = record;
    rows {
        field contract_id: u32 => 0, 4;
        field input_representation: InputRepresentation => 4, 2;
        /// Groups are 32 values along K for `A8S32_DYNAMIC`. Section 9.2.
        field quant_group: u16 => 6, 2;
        /// v1 defines only `LAST` (the K axis). Section 9.1.
        field quant_axis: QuantAxis => 8, 2;
        /// v1 defines only `RN_EVEN`. Section 9.1.
        field rounding_mode: RoundingMode => 10, 2;
        /// Signed. Section 9.
        field qmin: i16 => 12, 2;
        /// Signed. Section 9.
        field qmax: i16 => 14, 2;
        /// v1 defines only `F32`. Section 9.1.
        field scale_compute_dtype: ScaleComputeDtype => 16, 2;
        field dot_accumulator: DotAccumulator => 18, 2;
        field output_dtype: OutputDtype => 20, 2;
        field math_mode: MathMode => 22, 2;
        reserved => 24, 4;
        field calibration_id: u32 => 28, 4;
        flags flags: ContractFlags => 32, 4;
        reserved => 36, 4;
        /// BLAKE3-128, carried opaquely. Section 9.
        field contract_digest: [u8; 16] => 40, 16;
        reserved => 56, 8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcf::error::TcfError;
    use crate::tcf::record::testkit::{put, roundtrips, short_slice_errors};
    use crate::tcf::record::traits::Record;

    fn sample() -> [u8; CONTRACT_RECORD_BYTES] {
        let mut b = [0u8; CONTRACT_RECORD_BYTES];
        put(&mut b, 0, &1u32.to_le_bytes());
        put(
            &mut b,
            4,
            &InputRepresentation::A8S32Dynamic.to_u16().to_le_bytes(),
        );
        put(&mut b, 6, &32u16.to_le_bytes());
        put(&mut b, 8, &QuantAxis::Last.to_u16().to_le_bytes());
        put(&mut b, 10, &RoundingMode::RnEven.to_u16().to_le_bytes());
        put(&mut b, 12, &(-127i16).to_le_bytes());
        put(&mut b, 14, &127i16.to_le_bytes());
        put(&mut b, 16, &ScaleComputeDtype::F32.to_u16().to_le_bytes());
        put(
            &mut b,
            18,
            &DotAccumulator::I32ThenF32Scale.to_u16().to_le_bytes(),
        );
        put(&mut b, 20, &OutputDtype::F32.to_u16().to_le_bytes());
        put(
            &mut b,
            22,
            &MathMode::ReassociationAllowed.to_u16().to_le_bytes(),
        );
        put(&mut b, 28, &2u32.to_le_bytes());
        put(&mut b, 40, &[0x33u8; 16]);
        b
    }

    #[test]
    fn roundtrips_byte_for_byte() {
        roundtrips::<ContractRecord>(&sample());
    }

    #[test]
    fn qmin_qmax_roundtrip_negative_values() {
        let rec = ContractRecord::decode(&sample()).expect("valid record");
        assert_eq!(rec.qmin, -127);
        assert_eq!(rec.qmax, 127);

        let mut bytes = sample();
        put(&mut bytes, 12, &i16::MIN.to_le_bytes());
        put(&mut bytes, 14, &(-1i16).to_le_bytes());
        let rec = ContractRecord::decode(&bytes).expect("valid record");
        assert_eq!(rec.qmin, i16::MIN);
        assert_eq!(rec.qmax, -1);
        let mut out = [0u8; CONTRACT_RECORD_BYTES];
        rec.encode(&mut out).expect("encodes");
        assert_eq!(out, bytes);
    }

    #[test]
    fn nonzero_reserved_is_rejected() {
        let mut bytes = sample();
        bytes[24] = 1;
        assert_eq!(
            ContractRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "ContractRecord.reserved@24"
            })
        );

        let mut bytes = sample();
        bytes[39] = 1;
        assert_eq!(
            ContractRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "ContractRecord.reserved@36"
            })
        );

        let mut bytes = sample();
        bytes[63] = 1;
        assert_eq!(
            ContractRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "ContractRecord.reserved@56"
            })
        );
    }

    #[test]
    fn any_flag_bit_is_rejected() {
        let mut bytes = sample();
        put(&mut bytes, 32, &1u32.to_le_bytes());
        assert_eq!(
            ContractRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "ContractRecord.flags"
            })
        );
    }

    #[test]
    fn short_slice_errors_rather_than_panics() {
        short_slice_errors::<ContractRecord>("ContractRecord");
    }
}
