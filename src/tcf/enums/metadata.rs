//! Enumerations decoded from `CalibrationRecord`, `WorkloadProfileRecord`,
//! and `RelationRecord` fields. FORMAT.md Section 10, Section 10.5, Section 11, Section 15.3.

crate::define_enum_u16! {
    /// `CalibrationRecord.primary_metric`. Section 10. `sensitivity_delta` is a
    /// task delta, never an RMS score; tensor RMS and waveform correlation
    /// MUST NOT appear here.
    pub enum PrimaryMetric as "primary_metric" {
        Wer = 1,
        Cer = 2,
        TaskSpecific = 3,
        Perplexity = 4,
    }
}

crate::define_enum_u16! {
    /// `WorkloadProfileRecord.workload_kind`. Section 10.5.
    pub enum WorkloadKind as "workload_kind" {
        General = 1,
        Chat = 2,
        Code = 3,
        Summarization = 4,
        Speech = 5,
        TaskSpecific = 6,
    }
}

crate::define_enum_u16! {
    /// `RelationRecord.relation_type`. Section 11. v1 defines only
    /// `LOW_RANK_RESIDUAL`.
    pub enum RelationType as "relation_type" {
        LowRankResidual = 1,
    }
}

crate::define_enum_u32! {
    /// `TensorRecord.proof_format`. Section 15.3.
    ///
    /// `None` is the stored value on a tensor in a raw encoding: the bytes are
    /// the values, so there is nothing to prove and `payload_digest` already
    /// covers them. `DequantF16` carries one LE binary16 expected dequantized
    /// value per proof index, and is mandatory on every quantized tensor.
    pub enum ProofFormat as "proof_format" {
        None = 0,
        DequantF16 = 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_metric_roundtrips() {
        for (raw, variant) in [
            (1u16, PrimaryMetric::Wer),
            (2, PrimaryMetric::Cer),
            (3, PrimaryMetric::TaskSpecific),
            (4, PrimaryMetric::Perplexity),
        ] {
            assert_eq!(PrimaryMetric::try_from(raw), Ok(variant));
            assert_eq!(variant.to_u16(), raw);
        }
        assert!(PrimaryMetric::try_from(0).is_err());
        assert!(PrimaryMetric::try_from(5).is_err());
    }

    #[test]
    fn workload_kind_roundtrips() {
        for raw in 1u16..=6 {
            let kind = WorkloadKind::try_from(raw).expect("defined value");
            assert_eq!(kind.to_u16(), raw);
        }
        assert!(WorkloadKind::try_from(0).is_err());
        assert!(WorkloadKind::try_from(7).is_err());
    }

    #[test]
    fn relation_type_only_low_rank_residual() {
        assert_eq!(RelationType::try_from(1), Ok(RelationType::LowRankResidual));
        assert!(RelationType::try_from(0).is_err());
        assert!(RelationType::try_from(2).is_err());
    }

    #[test]
    fn proof_format_only_one() {
        assert_eq!(ProofFormat::try_from(0u32), Ok(ProofFormat::None));
        assert_eq!(ProofFormat::try_from(1u32), Ok(ProofFormat::DequantF16));
        assert!(ProofFormat::try_from(2u32).is_err());
    }
}
