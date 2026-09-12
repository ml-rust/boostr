//! `RelationRecord`: 64 bytes. FORMAT.md Section 11.

use crate::tcf::consts::{RELATION_RECORD_BYTES, UNUSED_INPUT_ID};
use crate::tcf::enums::RelationType;
use crate::tcf::flags::RelationFlags;

crate::define_record! {
    /// A model relation between tensors. Section 11.
    ///
    /// For `LOW_RANK_RESIDUAL` (Section 11.1) the inputs are `[0]` the quantized
    /// base tensor, `[1]` U, `[2]` V, and the output is
    /// `dequant(base) + U*V`. Low-rank stays a model relation, never a
    /// monolithic packed tensor each backend decodes specially.
    ///
    /// `relation_digest` is carried opaquely here: it is computed by
    /// [`crate::tcf::digest::relation_digest`], filled in by the writer, and
    /// verified by the reader at `open` (Section 9).
    pub struct RelationRecord;
    size = RELATION_RECORD_BYTES;
    mode = record;
    rows {
        field relation_type: RelationType => 0, 2;
        flags flags: RelationFlags => 2, 2;
        field output_tensor_id: u32 => 4, 4;
        /// Four u32. `consts::UNUSED_INPUT_ID` (`0xffffffff`) marks an
        /// unused slot; the raw id is carried through. Section 11.
        field input_tensor_id: [u32; 4] => 8, 16;
        field rank_or_parameter: u32 => 24, 4;
        field activation_contract_id: u32 => 28, 4;
        /// BLAKE3-128, carried opaquely. Section 11.
        field relation_digest: [u8; 16] => 32, 16;
        reserved => 48, 16;
    }
}

impl RelationRecord {
    /// The used input ids, in order, stopping at the first unused slot. Section 11.
    ///
    /// `impl Iterator` already carries `#[must_use]`; a second one is redundant.
    pub fn inputs(&self) -> impl Iterator<Item = u32> + '_ {
        self.input_tensor_id
            .iter()
            .copied()
            .take_while(|id| *id != UNUSED_INPUT_ID)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcf::error::TcfError;
    use crate::tcf::record::testkit::{put, roundtrips, short_slice_errors};
    use crate::tcf::record::traits::Record;

    fn sample() -> [u8; RELATION_RECORD_BYTES] {
        let mut b = [0u8; RELATION_RECORD_BYTES];
        put(
            &mut b,
            0,
            &RelationType::LowRankResidual.to_u16().to_le_bytes(),
        );
        put(&mut b, 4, &10u32.to_le_bytes());
        put(&mut b, 8, &11u32.to_le_bytes());
        put(&mut b, 12, &12u32.to_le_bytes());
        put(&mut b, 16, &13u32.to_le_bytes());
        put(&mut b, 20, &UNUSED_INPUT_ID.to_le_bytes());
        put(&mut b, 24, &16u32.to_le_bytes());
        put(&mut b, 28, &1u32.to_le_bytes());
        put(&mut b, 32, &[0x88u8; 16]);
        b
    }

    #[test]
    fn roundtrips_byte_for_byte() {
        roundtrips::<RelationRecord>(&sample());
    }

    #[test]
    fn unused_input_sentinel_is_carried_raw() {
        let rec = RelationRecord::decode(&sample()).expect("valid record");
        assert_eq!(rec.input_tensor_id, [11, 12, 13, UNUSED_INPUT_ID]);
        assert_eq!(rec.inputs().collect::<Vec<_>>(), vec![11, 12, 13]);
        assert_eq!(rec.relation_type, RelationType::LowRankResidual);
        assert_eq!(rec.rank_or_parameter, 16);
    }

    #[test]
    fn nonzero_reserved_is_rejected() {
        let mut bytes = sample();
        bytes[63] = 1;
        assert_eq!(
            RelationRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "RelationRecord.reserved@48"
            })
        );
    }

    #[test]
    fn any_flag_bit_is_rejected() {
        let mut bytes = sample();
        put(&mut bytes, 2, &1u16.to_le_bytes());
        assert_eq!(
            RelationRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "RelationRecord.flags"
            })
        );
    }

    #[test]
    fn short_slice_errors_rather_than_panics() {
        short_slice_errors::<RelationRecord>("RelationRecord");
    }
}
