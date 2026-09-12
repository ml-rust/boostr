//! `CalibrationRecord`: 128 bytes. FORMAT.md Section 10.

use crate::tcf::consts::CALIBRATION_RECORD_BYTES;
use crate::tcf::enums::PrimaryMetric;
use crate::tcf::flags::CalibrationFlags;

crate::define_record! {
    /// The measurement `sensitivity_delta` is reported against. Section 10.
    ///
    /// The evaluator name and digest freeze the ASR or LM model version,
    /// normalization policy, sampling settings, and prompt preprocessing: a
    /// metric name alone does not identify an experiment.
    ///
    /// `baseline_metric` and `acceptance_margin` decode structurally only;
    /// finiteness policy is deferred to the verifier unit. Digest fields are
    /// carried opaquely.
    pub struct CalibrationRecord;
    size = CALIBRATION_RECORD_BYTES;
    mode = record;
    rows {
        field calibration_id: u32 => 0, 4;
        /// Tensor RMS and waveform correlation MUST NOT appear here. Section 10.
        field primary_metric: PrimaryMetric => 4, 2;
        flags flags: CalibrationFlags => 6, 2;
        field sample_count: u32 => 8, 4;
        field seed_count: u32 => 12, 4;
        field baseline_metric: f32 => 16, 4;
        field acceptance_margin: f32 => 20, 4;
        /// `(dataset_name_off, dataset_name_len)` into the string table. Section 6.
        strref dataset_name: (dataset_name_off => 24, dataset_name_len => 32);
        reserved => 36, 4;
        /// `(evaluator_name_off, evaluator_name_len)` into the string table. Section 6.
        strref evaluator_name: (evaluator_name_off => 40, evaluator_name_len => 48);
        reserved => 52, 4;
        /// Dataset content BLAKE3-256, carried opaquely. Section 10.
        field dataset_digest: [u8; 32] => 56, 32;
        /// Evaluator config digest, carried opaquely. Section 10.
        field evaluator_config_digest: [u8; 16] => 88, 16;
        /// Unix seconds. Section 10.
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

    fn sample() -> [u8; CALIBRATION_RECORD_BYTES] {
        let mut b = [0u8; CALIBRATION_RECORD_BYTES];
        put(&mut b, 0, &1u32.to_le_bytes());
        put(&mut b, 4, &PrimaryMetric::Perplexity.to_u16().to_le_bytes());
        put(&mut b, 8, &256u32.to_le_bytes());
        put(&mut b, 12, &3u32.to_le_bytes());
        put(&mut b, 16, &5.6789f32.to_le_bytes());
        put(&mut b, 20, &0.01f32.to_le_bytes());
        put(&mut b, 24, &0u64.to_le_bytes());
        put(&mut b, 32, &8u32.to_le_bytes());
        put(&mut b, 40, &8u64.to_le_bytes());
        put(&mut b, 48, &12u32.to_le_bytes());
        put(&mut b, 56, &[0x44u8; 32]);
        put(&mut b, 88, &[0x55u8; 16]);
        put(&mut b, 104, &1_700_000_000u64.to_le_bytes());
        b
    }

    #[test]
    fn roundtrips_byte_for_byte() {
        roundtrips::<CalibrationRecord>(&sample());
    }

    #[test]
    fn decodes_both_string_refs() {
        let rec = CalibrationRecord::decode(&sample()).expect("valid record");
        assert_eq!(rec.dataset_name, StringRef::new(0, 8));
        assert_eq!(rec.evaluator_name, StringRef::new(8, 12));
        assert_eq!(rec.primary_metric, PrimaryMetric::Perplexity);
        assert_eq!(rec.producer_timestamp, 1_700_000_000);
        assert_eq!(rec.dataset_digest, [0x44u8; 32]);
    }

    #[test]
    fn nonzero_reserved_is_rejected() {
        for (byte, field) in [
            (36usize, "CalibrationRecord.reserved@36"),
            (52, "CalibrationRecord.reserved@52"),
            (127, "CalibrationRecord.reserved@112"),
        ] {
            let mut bytes = sample();
            bytes[byte] = 1;
            assert_eq!(
                CalibrationRecord::decode(&bytes),
                Err(TcfError::NonzeroReserved { field })
            );
        }
    }

    #[test]
    fn any_flag_bit_is_rejected() {
        let mut bytes = sample();
        put(&mut bytes, 6, &1u16.to_le_bytes());
        assert_eq!(
            CalibrationRecord::decode(&bytes),
            Err(TcfError::NonzeroReserved {
                field: "CalibrationRecord.flags"
            })
        );
    }

    #[test]
    fn short_slice_errors_rather_than_panics() {
        short_slice_errors::<CalibrationRecord>("CalibrationRecord");
    }
}
