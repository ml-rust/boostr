//! Enumerations decoded from `TensorRecord` fields. FORMAT.md Section 8.

use crate::tcf::error::TcfError;

crate::define_enum_u16! {
    /// `TensorRecord.role`: semantic metadata, not a dispatch key. Section 8.7.
    pub enum Role as "role" {
        Other = 0,
        LinearWeight = 1,
        Embedding = 2,
        Conv1dWeight = 3,
        Bias = 4,
        NormScale = 5,
        SnakeAlpha = 6,
        SsmProjection = 7,
        SsmDynamics = 8,
        MoeExpert = 9,
        MoeRouter = 10,
        DitWeight = 11,
        FsqProjection = 12,
        FsqLevels = 13,
        VaeWeight = 14,
        IndexedTable = 15,
    }
}

/// `TensorRecord.execution_role`: the dispatch key. Section 8.6.1. Uses its own
/// `E_UNKNOWN_EXECUTION_ROLE` error, named separately from
/// `E_UNKNOWN_ENUM_VALUE` per Section 17: an unknown execution role names a
/// dispatch a reader cannot resolve, a distinct failure mode from any other
/// unrecognized enumerated field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionRole {
    Matmul,
    Lookup,
    Conv1d,
    Indexed,
    StateUpdate,
    Elementwise,
}

impl ExecutionRole {
    /// Encode as the wire `u16` representation.
    #[must_use]
    pub const fn to_u16(self) -> u16 {
        match self {
            Self::Matmul => 0,
            Self::Lookup => 1,
            Self::Conv1d => 2,
            Self::Indexed => 3,
            Self::StateUpdate => 4,
            Self::Elementwise => 5,
        }
    }
}

impl TryFrom<u16> for ExecutionRole {
    type Error = TcfError;

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Matmul),
            1 => Ok(Self::Lookup),
            2 => Ok(Self::Conv1d),
            3 => Ok(Self::Indexed),
            4 => Ok(Self::StateUpdate),
            5 => Ok(Self::Elementwise),
            other => Err(TcfError::UnknownExecutionRole { raw: other }),
        }
    }
}

crate::define_enum_u16! {
    /// `TensorRecord.fallback_reason`. Section 8.6. Mandatory whenever `encoding`
    /// differs from the module's highest-ranked `preferred_encoding`.
    pub enum FallbackReason as "fallback_reason" {
        None = 0,
        RankLt2 = 1,
        /// The tensor's shape cannot satisfy the preferred encoding's
        /// geometry. Execution geometry (group size, tile width) is not
        /// frozen by this format, so this variant never hardcodes a number.
        ShapeIncompatibleWithEncoding = 2,
        RoleForbidsQuant = 3,
        TaskSensitivity = 4,
        PhysicalSizeNotBeneficial = 5,
        NumericRange = 6,
        SourceNonfinite = 7,
        UnsupportedEncoding = 8,
        UserPinnedPrecision = 9,
        ProducerPolicy = 10,
    }
}

crate::define_enum_u16! {
    /// `TensorRecord.residency_class`. Section 8.5. Names intent, never a device API.
    pub enum ResidencyClass as "residency_class" {
        Hot = 0,
        Warm = 1,
        Cold = 2,
        HostOnly = 3,
        /// Initial placement is runtime-selected, but residency MUST NOT
        /// change afterward. Section 8.5 deliberately never names the pinned
        /// target.
        NoMigrate = 4,
    }
}

crate::define_enum_u32! {
    /// `TensorRecord.layout_id`. Section 8.4. Names the layout as stored; v1
    /// defines exactly one value.
    pub enum LayoutId as "layout_id" {
        RowMajorDense = 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! roundtrip_u16 {
        ($ty:ty, $($variant:expr),+ $(,)?) => {
            $(
                let raw = <$ty>::to_u16($variant);
                assert_eq!(<$ty>::try_from(raw), Ok($variant));
            )+
        };
    }

    #[test]
    fn role_roundtrips() {
        roundtrip_u16!(
            Role,
            Role::Other,
            Role::LinearWeight,
            Role::Embedding,
            Role::Conv1dWeight,
            Role::Bias,
            Role::NormScale,
            Role::SnakeAlpha,
            Role::SsmProjection,
            Role::SsmDynamics,
            Role::MoeExpert,
            Role::MoeRouter,
            Role::DitWeight,
            Role::FsqProjection,
            Role::FsqLevels,
            Role::VaeWeight,
            Role::IndexedTable,
        );
        assert_eq!(Role::IndexedTable.to_u16(), 15);
    }

    #[test]
    fn role_rejects_out_of_range() {
        let err = Role::try_from(16).unwrap_err();
        assert_eq!(
            err,
            TcfError::UnknownEnumValue {
                field: "role",
                raw: 16
            }
        );
    }

    #[test]
    fn execution_role_roundtrips_and_rejects() {
        for r in [
            ExecutionRole::Matmul,
            ExecutionRole::Lookup,
            ExecutionRole::Conv1d,
            ExecutionRole::Indexed,
            ExecutionRole::StateUpdate,
            ExecutionRole::Elementwise,
        ] {
            assert_eq!(ExecutionRole::try_from(r.to_u16()), Ok(r));
        }
        assert_eq!(
            ExecutionRole::try_from(6),
            Err(TcfError::UnknownExecutionRole { raw: 6 })
        );
    }

    #[test]
    fn fallback_reason_roundtrips() {
        roundtrip_u16!(
            FallbackReason,
            FallbackReason::None,
            FallbackReason::RankLt2,
            FallbackReason::ShapeIncompatibleWithEncoding,
            FallbackReason::RoleForbidsQuant,
            FallbackReason::TaskSensitivity,
            FallbackReason::PhysicalSizeNotBeneficial,
            FallbackReason::NumericRange,
            FallbackReason::SourceNonfinite,
            FallbackReason::UnsupportedEncoding,
            FallbackReason::UserPinnedPrecision,
            FallbackReason::ProducerPolicy,
        );
        assert!(FallbackReason::try_from(11).is_err());
    }

    #[test]
    fn residency_class_roundtrips() {
        roundtrip_u16!(
            ResidencyClass,
            ResidencyClass::Hot,
            ResidencyClass::Warm,
            ResidencyClass::Cold,
            ResidencyClass::HostOnly,
            ResidencyClass::NoMigrate,
        );
        assert!(ResidencyClass::try_from(5).is_err());
    }

    #[test]
    fn layout_id_only_defines_row_major_dense() {
        assert_eq!(LayoutId::RowMajorDense.to_u32(), 0);
        assert_eq!(LayoutId::try_from(0u32), Ok(LayoutId::RowMajorDense));
        assert_eq!(
            LayoutId::try_from(1u32),
            Err(TcfError::UnknownEnumValue {
                field: "layout_id",
                raw: 1
            })
        );
    }
}
