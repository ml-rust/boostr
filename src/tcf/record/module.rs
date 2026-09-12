//! `ModuleRecord`: 128 bytes. FORMAT.md Section 7.

use crate::tcf::consts::MODULE_RECORD_BYTES;
use crate::tcf::encoding::Encoding;
use crate::tcf::enums::{ModuleRole, ResidencyClass, StateDtype};
use crate::tcf::flags::{PolicyFlags, StateFlags};

crate::define_record! {
    /// A module in the model tree. Section 7.
    ///
    /// `policy_digest` is BLAKE3-128 over bytes `[0,64)` of the record with
    /// the digest field absent, concatenated with the module's exact UTF-8
    /// name bytes. This unit carries it opaquely: it is computed by
    /// [`crate::tcf::digest::policy_digest`], filled in by the writer, and
    /// verified by the reader at `open` (Section 9).
    pub struct ModuleRecord;
    size = MODULE_RECORD_BYTES;
    mode = record;
    rows {
        field module_id: u32 => 0, 4;
        /// `consts::ROOT_PARENT_ID` (`0xffffffff`) marks a root module.
        field parent_id: u32 => 4, 4;
        /// `(name_off, name_len)` into the string table. Section 6.
        strref name: (name_off => 8, name_len => 16);
        field module_role: ModuleRole => 20, 2;
        /// `0` means the module declares no fallback. Section 7.
        field fallback_encoding: Option<Encoding> => 22, 2;
        /// Four u16, ordered by producer preference, highest first. `0`
        /// means unused and decodes to `None`. Section 7.
        field preferred_encoding: [Option<Encoding>; 4] => 24, 8;
        field activation_contract_id: u32 => 32, 4;
        flags policy_flags: PolicyFlags => 36, 4;
        field min_quant_k: u32 => 40, 4;
        // Formerly `quant_group` (44) and `quant_tile` (46). Both duplicated
        // geometry the encoding identifier already names (`Q4S32_T64` names
        // group 32, tile 64), and a module holding two different encodings
        // could not give either field a truthful value. Reserved as an
        // extension point a future capability-based restriction can claim
        // with its own `required_features` bit. Section 7.
        reserved => 44, 2;
        reserved => 46, 2;
        field default_residency: ResidencyClass => 48, 2;
        field state_dtype: StateDtype => 50, 2;
        flags state_flags: StateFlags => 52, 4;
        reserved => 56, 8;
        /// BLAKE3-128, carried opaquely. Section 7.
        field policy_digest: [u8; 16] => 64, 16;
        reserved => 80, 48;
    }
}

impl ModuleRecord {
    /// True when this module is a tree root (`parent_id == 0xffffffff`). Section 7.
    #[must_use]
    pub const fn is_root(&self) -> bool {
        self.parent_id == crate::tcf::consts::ROOT_PARENT_ID
    }

    /// The producer's highest-ranked preferred encoding, if any. Section 7.
    #[must_use]
    pub const fn top_preferred_encoding(&self) -> Option<Encoding> {
        self.preferred_encoding[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcf::encoding::{BlockEncoding, RawEncoding};
    use crate::tcf::error::TcfError;
    use crate::tcf::record::field::StringRef;
    use crate::tcf::record::testkit::{put, roundtrips, short_slice_errors};
    use crate::tcf::record::traits::Record;

    fn sample() -> [u8; MODULE_RECORD_BYTES] {
        let mut b = [0u8; MODULE_RECORD_BYTES];
        put(&mut b, 0, &7u32.to_le_bytes());
        put(&mut b, 4, &crate::tcf::consts::ROOT_PARENT_ID.to_le_bytes());
        put(&mut b, 8, &64u64.to_le_bytes());
        put(&mut b, 16, &11u32.to_le_bytes());
        put(&mut b, 20, &ModuleRole::Ffn.to_u16().to_le_bytes());
        put(
            &mut b,
            22,
            &Encoding::Raw(RawEncoding::F16).to_u16().to_le_bytes(),
        );
        put(
            &mut b,
            24,
            &Encoding::Block(BlockEncoding::Q6K).to_u16().to_le_bytes(),
        );
        put(&mut b, 32, &3u32.to_le_bytes());
        put(
            &mut b,
            36,
            &PolicyFlags::FORBID_REQUANT.bits().to_le_bytes(),
        );
        put(&mut b, 40, &64u32.to_le_bytes());
        put(&mut b, 48, &ResidencyClass::Warm.to_u16().to_le_bytes());
        put(&mut b, 50, &StateDtype::F32.to_u16().to_le_bytes());
        put(&mut b, 64, &[0xabu8; 16]);
        b
    }

    #[test]
    fn roundtrips_byte_for_byte() {
        let bytes = sample();
        roundtrips::<ModuleRecord>(&bytes);
    }

    #[test]
    fn decodes_every_field() {
        let rec = ModuleRecord::decode(&sample()).expect("valid record");
        assert_eq!(rec.module_id, 7);
        assert!(rec.is_root());
        assert_eq!(rec.name, StringRef::new(64, 11));
        assert_eq!(rec.module_role, ModuleRole::Ffn);
        assert_eq!(rec.fallback_encoding, Some(Encoding::Raw(RawEncoding::F16)));
        assert_eq!(rec.state_dtype, StateDtype::F32);
        assert_eq!(rec.policy_digest, [0xabu8; 16]);
    }

    #[test]
    fn fallback_encoding_zero_sentinel_is_none() {
        let mut bytes = sample();
        put(&mut bytes, 22, &0u16.to_le_bytes());
        let rec = ModuleRecord::decode(&bytes).expect("valid record");
        assert_eq!(rec.fallback_encoding, None);

        let mut out = [0u8; MODULE_RECORD_BYTES];
        rec.encode(&mut out).expect("encodes");
        assert_eq!(&out[22..24], &0u16.to_le_bytes());
    }

    #[test]
    fn preferred_encoding_zero_sentinel_is_none() {
        let rec = ModuleRecord::decode(&sample()).expect("valid record");
        assert_eq!(
            rec.top_preferred_encoding(),
            Some(Encoding::Block(BlockEncoding::Q6K))
        );
        assert_eq!(rec.preferred_encoding[1], None);
        assert_eq!(rec.preferred_encoding[2], None);
        assert_eq!(rec.preferred_encoding[3], None);
    }

    #[test]
    fn nonzero_reserved_is_rejected() {
        let mut bytes = sample();
        bytes[44] = 1;
        assert_eq!(
            ModuleRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "ModuleRecord.reserved@44"
            })
        );

        let mut bytes = sample();
        bytes[46] = 1;
        assert_eq!(
            ModuleRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "ModuleRecord.reserved@46"
            })
        );

        let mut bytes = sample();
        bytes[60] = 1;
        assert_eq!(
            ModuleRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "ModuleRecord.reserved@56"
            })
        );

        let mut bytes = sample();
        bytes[127] = 1;
        assert_eq!(
            ModuleRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "ModuleRecord.reserved@80"
            })
        );
    }

    #[test]
    fn unknown_state_flag_bit_is_rejected() {
        let mut bytes = sample();
        put(&mut bytes, 52, &1u32.to_le_bytes());
        assert_eq!(
            ModuleRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "ModuleRecord.state_flags"
            })
        );
    }

    #[test]
    fn short_slice_errors_rather_than_panics() {
        short_slice_errors::<ModuleRecord>("ModuleRecord");
    }
}
