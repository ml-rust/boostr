//! `WorkloadProfileRecord`: 128 bytes. FORMAT.md Section 10.5.

use crate::tcf::consts::WORKLOAD_PROFILE_RECORD_BYTES;
use crate::tcf::enums::WorkloadKind;
use crate::tcf::flags::WorkloadProfileFlags;

crate::define_record! {
    /// The provenance of a tensor's access telemetry. Section 10.5, Section 10.5.1.
    ///
    /// `accesses_per_generation` is a workload observation, not a property
    /// of the tensor: an expert measured at `0.2` on a general-text set can
    /// run at `48` under different routing. `TensorRecord.workload_profile_id`
    /// names the record that measurement came from.
    ///
    /// `avg_generated_tokens` and `avg_prompt_tokens` decode structurally
    /// only; finiteness policy is deferred to the verifier unit. Digest
    /// fields are carried opaquely.
    pub struct WorkloadProfileRecord;
    size = WORKLOAD_PROFILE_RECORD_BYTES;
    mode = record;
    rows {
        field workload_id: u32 => 0, 4;
        field workload_kind: WorkloadKind => 4, 2;
        flags flags: WorkloadProfileFlags => 6, 2;
        field generation_count: u32 => 8, 4;
        field avg_generated_tokens: f32 => 12, 4;
        field avg_prompt_tokens: f32 => 16, 4;
        reserved => 20, 4;
        /// `(dataset_name_off, dataset_name_len)` into the string table. Section 6.
        strref dataset_name: (dataset_name_off => 24, dataset_name_len => 32);
        reserved => 36, 4;
        /// `(runtime_name_off, runtime_name_len)` into the string table. Section 6.
        strref runtime_name: (runtime_name_off => 40, runtime_name_len => 48);
        reserved => 52, 4;
        /// Workload content BLAKE3-256, carried opaquely. Section 10.5.
        field workload_digest: [u8; 32] => 56, 32;
        /// Runtime and config digest, carried opaquely. Section 10.5.
        field runtime_config_digest: [u8; 16] => 88, 16;
        /// Unix seconds. Section 10.5.
        field producer_timestamp: u64 => 104, 8;
        reserved => 112, 16;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcf::error::TcfError;
    use crate::tcf::record::field::StringRef;
    use crate::tcf::record::testkit::{put, roundtrips, short_slice_errors};
    use crate::tcf::record::traits::Record;

    fn sample() -> [u8; WORKLOAD_PROFILE_RECORD_BYTES] {
        let mut b = [0u8; WORKLOAD_PROFILE_RECORD_BYTES];
        put(&mut b, 0, &1u32.to_le_bytes());
        put(&mut b, 4, &WorkloadKind::Chat.to_u16().to_le_bytes());
        put(&mut b, 8, &1024u32.to_le_bytes());
        put(&mut b, 12, &256.5f32.to_le_bytes());
        put(&mut b, 16, &48.25f32.to_le_bytes());
        put(&mut b, 24, &20u64.to_le_bytes());
        put(&mut b, 32, &9u32.to_le_bytes());
        put(&mut b, 40, &29u64.to_le_bytes());
        put(&mut b, 48, &6u32.to_le_bytes());
        put(&mut b, 56, &[0x66u8; 32]);
        put(&mut b, 88, &[0x77u8; 16]);
        put(&mut b, 104, &1_700_000_001u64.to_le_bytes());
        b
    }

    #[test]
    fn roundtrips_byte_for_byte() {
        roundtrips::<WorkloadProfileRecord>(&sample());
    }

    #[test]
    fn decodes_both_string_refs() {
        let rec = WorkloadProfileRecord::decode(&sample()).expect("valid record");
        assert_eq!(rec.dataset_name, StringRef::new(20, 9));
        assert_eq!(rec.runtime_name, StringRef::new(29, 6));
        assert_eq!(rec.workload_kind, WorkloadKind::Chat);
        assert!((rec.avg_generated_tokens - 256.5).abs() < 1e-9);
    }

    #[test]
    fn nonzero_reserved_is_rejected() {
        for (byte, field) in [
            (20usize, "WorkloadProfileRecord.reserved@20"),
            (36, "WorkloadProfileRecord.reserved@36"),
            (52, "WorkloadProfileRecord.reserved@52"),
            (127, "WorkloadProfileRecord.reserved@112"),
        ] {
            let mut bytes = sample();
            bytes[byte] = 1;
            assert_eq!(
                WorkloadProfileRecord::decode(&bytes),
                Err(TcfError::NonzeroReserved { field })
            );
        }
    }

    #[test]
    fn any_flag_bit_is_rejected() {
        let mut bytes = sample();
        put(&mut bytes, 6, &1u16.to_le_bytes());
        assert_eq!(
            WorkloadProfileRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "WorkloadProfileRecord.flags"
            })
        );
    }

    #[test]
    fn short_slice_errors_rather_than_panics() {
        short_slice_errors::<WorkloadProfileRecord>("WorkloadProfileRecord");
    }
}
