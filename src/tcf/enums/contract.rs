//! Enumerations decoded from `ContractRecord` fields. FORMAT.md Section 9.
//!
//! v1 defines exactly one valid value for `QuantAxis`, `RoundingMode`, and
//! `ScaleComputeDtype` (Section 9.1); each is still modeled as a fallible enum so a
//! future minor version can add variants without changing decode behavior
//! for value `1`/`0`.

crate::define_enum_u16! {
    /// `ContractRecord.input_representation`. Section 9.1.
    ///
    /// `GgmlReference` is the value a `Block` tensor declares. A GGML block
    /// type fixes its weight bytes and not its activation path: `ggml-quants.c`
    /// runs K-quants against Q8_K activations on the CPU, `ggml-cuda` runs the
    /// same weights against Q8_1 activations, and simpler backends dequantize
    /// and use f32. A file cannot pin one of those without refusing the
    /// others, so TCF pins the kernel family instead. The record's group,
    /// range, and accumulator fields are then nominal (`0`, `0..0`,
    /// `GgmlReference`). A future encoding whose kernels agree on one
    /// activation representation declares that representation instead.
    pub enum InputRepresentation as "input_representation" {
        F32 = 1,
        F16 = 2,
        Bf16 = 3,
        A8S32Dynamic = 4,
        GgmlReference = 5,
    }
}

crate::define_enum_u16! {
    /// `ContractRecord.quant_axis`. Section 9.1. v1 defines only `LAST` (the K axis).
    pub enum QuantAxis as "quant_axis" {
        Last = 0,
    }
}

crate::define_enum_u16! {
    /// `ContractRecord.rounding_mode`. Section 9.1. v1 defines only `RN_EVEN`.
    pub enum RoundingMode as "rounding_mode" {
        RnEven = 1,
    }
}

crate::define_enum_u16! {
    /// `ContractRecord.scale_compute_dtype`. Section 9.1. v1 defines only `F32`.
    pub enum ScaleComputeDtype as "scale_compute_dtype" {
        F32 = 1,
    }
}

crate::define_enum_u16! {
    /// `ContractRecord.dot_accumulator`. Section 9.1.
    ///
    /// `GgmlReference` pairs with `InputRepresentation::GgmlReference`: the
    /// accumulator is the one `ggml-quants.c` uses for the block type on the
    /// executing backend.
    pub enum DotAccumulator as "dot_accumulator" {
        F32 = 1,
        I32ThenF32Scale = 2,
        GgmlReference = 3,
    }
}

crate::define_enum_u16! {
    /// `ContractRecord.output_dtype`. Section 9.1.
    pub enum OutputDtype as "output_dtype" {
        F32 = 1,
        F16 = 2,
        Bf16 = 3,
    }
}

crate::define_enum_u16! {
    /// `ContractRecord.math_mode`. Section 9.1.
    pub enum MathMode as "math_mode" {
        ReassociationAllowed = 1,
        ReassociationForbidden = 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_representation_roundtrips() {
        for (raw, variant) in [
            (1u16, InputRepresentation::F32),
            (2, InputRepresentation::F16),
            (3, InputRepresentation::Bf16),
            (4, InputRepresentation::A8S32Dynamic),
            (5, InputRepresentation::GgmlReference),
        ] {
            assert_eq!(InputRepresentation::try_from(raw), Ok(variant));
            assert_eq!(variant.to_u16(), raw);
        }
        assert!(InputRepresentation::try_from(0).is_err());
        assert!(InputRepresentation::try_from(6).is_err());
    }

    #[test]
    fn quant_axis_only_last() {
        assert_eq!(QuantAxis::try_from(0), Ok(QuantAxis::Last));
        assert!(QuantAxis::try_from(1).is_err());
    }

    #[test]
    fn rounding_mode_only_rn_even() {
        assert_eq!(RoundingMode::try_from(1), Ok(RoundingMode::RnEven));
        assert!(RoundingMode::try_from(0).is_err());
    }

    #[test]
    fn scale_compute_dtype_only_f32() {
        assert_eq!(ScaleComputeDtype::try_from(1), Ok(ScaleComputeDtype::F32));
        assert!(ScaleComputeDtype::try_from(2).is_err());
    }

    #[test]
    fn dot_accumulator_roundtrips() {
        assert_eq!(DotAccumulator::try_from(1), Ok(DotAccumulator::F32));
        assert_eq!(
            DotAccumulator::try_from(2),
            Ok(DotAccumulator::I32ThenF32Scale)
        );
        assert_eq!(
            DotAccumulator::try_from(3),
            Ok(DotAccumulator::GgmlReference)
        );
        assert!(DotAccumulator::try_from(4).is_err());
    }

    #[test]
    fn output_dtype_roundtrips() {
        for (raw, variant) in [
            (1u16, OutputDtype::F32),
            (2, OutputDtype::F16),
            (3, OutputDtype::Bf16),
        ] {
            assert_eq!(OutputDtype::try_from(raw), Ok(variant));
        }
        assert!(OutputDtype::try_from(4).is_err());
    }

    #[test]
    fn math_mode_roundtrips() {
        assert_eq!(MathMode::try_from(1), Ok(MathMode::ReassociationAllowed));
        assert_eq!(MathMode::try_from(2), Ok(MathMode::ReassociationForbidden));
        assert!(MathMode::try_from(3).is_err());
    }
}
