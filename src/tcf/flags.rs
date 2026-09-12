//! Hand-rolled bitflag newtypes for TCF v1's flag fields. No external
//! bitflags crate: each type is a newtype over its wire-width unsigned
//! integer, with `const` flag constructors, `contains`, `bits`, `union`, and
//! `unknown_bits`.

macro_rules! define_flags {
    (
        $(#[$meta:meta])*
        pub struct $Name:ident($repr:ty);
        known = $known:expr;
        {
            $( $(#[$fmeta:meta])* const $flag:ident = $bit:expr; )*
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
        pub struct $Name($repr);

        impl $Name {
            $(
                $(#[$fmeta])*
                pub const $flag: Self = Self($bit);
            )*

            /// Mask of every bit this field defines in v1. Bits outside this
            /// mask are reserved and MUST be zero (FORMAT.md Section 8.1.5,
            /// Section 4).
            pub const KNOWN_BITS: $repr = $known;

            /// The empty flag set.
            pub const NONE: Self = Self(0);

            /// Build from a raw wire value, keeping every bit as-is
            /// (including any unknown bit).
            #[must_use]
            pub const fn from_bits_retain(bits: $repr) -> Self {
                Self(bits)
            }

            /// The raw wire value.
            #[must_use]
            pub const fn bits(self) -> $repr {
                self.0
            }

            /// True if every bit set in `other` is also set in `self`.
            #[must_use]
            pub const fn contains(self, other: Self) -> bool {
                self.0 & other.0 == other.0
            }

            /// The union of two flag sets.
            #[must_use]
            pub const fn union(self, other: Self) -> Self {
                Self(self.0 | other.0)
            }

            /// Bits set in `self` outside `KNOWN_BITS`. A reader MUST reject
            /// a file where this is non-zero (a reserved bit is set).
            #[must_use]
            pub const fn unknown_bits(self) -> $repr {
                self.0 & !Self::KNOWN_BITS
            }
        }
    };
}

define_flags! {
    /// `Header.flags`. Section 5.1.
    pub struct HeaderFlags(u32);
    known = 0x1;
    {
        /// bit 0: must be 1 in v1.
        const LITTLE_ENDIAN = 1 << 0;
    }
}

define_flags! {
    /// `Header.required_features`. Section 5.2. A reader MUST reject any file with
    /// an unknown bit set. Bits 0-3 MUST be set in every conforming v1 file.
    pub struct RequiredFeatures(u64);
    known = 0x7f;
    {
        const ACTIVATION_CONTRACTS = 1 << 0;
        const PLACEMENT_METADATA = 1 << 1;
        const SEMANTIC_DIGESTS = 1 << 2;
        const SOURCE_PROOFS = 1 << 3;
        /// Set only when `relation_count > 0`.
        const RELATIONS = 1 << 4;
        /// Set only when `workload_count > 0`.
        const WORKLOAD_PROFILES = 1 << 5;
        /// Named the retired two-level tile encodings (Section 5.2). No
        /// current encoding sets it, and the reader refuses a file that
        /// does: such a file needs a decoder this crate no longer has.
        const TWO_LEVEL_SCALES = 1 << 6;
    }
}

define_flags! {
    /// `TensorRecord.flags`. Section 8.2.
    pub struct TensorFlags(u32);
    known = 0x1f;
    {
        const SENSITIVITY_VALID = 1 << 0;
        const ACCESS_PROFILE_VALID = 1 << 1;
        const ALLOW_OFFLOAD = 1 << 2;
        const MUST_VERIFY_BEFORE_USE = 1 << 3;
        const TASK_CRITICAL = 1 << 4;
    }
}

define_flags! {
    /// `ModuleRecord.policy_flags`. Section 7.2.
    pub struct PolicyFlags(u32);
    known = 0x3;
    {
        /// bit 0: downstream tools must not re-encode this module.
        const FORBID_REQUANT = 1 << 0;
        /// bit 1: host-only by producer policy.
        const FORBID_DEVICE_PLACEMENT = 1 << 1;
    }
}

define_flags! {
    /// `ContractRecord.flags`. Section 9, Section 8.1.5. v1 defines no bit: the field is an
    /// extension point sized ahead of need, and every bit MUST be zero.
    pub struct ContractFlags(u32);
    known = 0;
    {}
}

define_flags! {
    /// `CalibrationRecord.flags`. Section 10, Section 8.1.5. v1 defines no bit; every bit
    /// MUST be zero.
    pub struct CalibrationFlags(u16);
    known = 0;
    {}
}

define_flags! {
    /// `WorkloadProfileRecord.flags`. Section 10.5, Section 8.1.5. v1 defines no bit; every
    /// bit MUST be zero.
    pub struct WorkloadProfileFlags(u16);
    known = 0;
    {}
}

define_flags! {
    /// `RelationRecord.flags`. Section 11, Section 8.1.5. v1 defines no bit; every bit MUST
    /// be zero.
    pub struct RelationFlags(u16);
    known = 0;
    {}
}

define_flags! {
    /// `ModuleRecord.state_flags`. Section 7, Section 8.1.5. v1 defines no bit; every bit
    /// MUST be zero.
    pub struct StateFlags(u32);
    known = 0;
    {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_flags_little_endian_bit() {
        let f = HeaderFlags::LITTLE_ENDIAN;
        assert_eq!(f.bits(), 1);
        assert!(f.contains(HeaderFlags::LITTLE_ENDIAN));
        assert_eq!(f.unknown_bits(), 0);
    }

    #[test]
    fn header_flags_unknown_bit_detected() {
        let f = HeaderFlags::from_bits_retain(0b10);
        assert_ne!(f.unknown_bits(), 0);
    }

    #[test]
    fn required_features_union_and_contains() {
        let mandatory = RequiredFeatures::ACTIVATION_CONTRACTS
            .union(RequiredFeatures::PLACEMENT_METADATA)
            .union(RequiredFeatures::SEMANTIC_DIGESTS)
            .union(RequiredFeatures::SOURCE_PROOFS);
        assert_eq!(mandatory.bits(), 0b1111);
        assert!(mandatory.contains(RequiredFeatures::SOURCE_PROOFS));
        assert!(!mandatory.contains(RequiredFeatures::RELATIONS));
        assert_eq!(mandatory.unknown_bits(), 0);
    }

    #[test]
    fn required_features_unknown_bit_detected() {
        let f = RequiredFeatures::from_bits_retain(1 << 7);
        assert_eq!(f.unknown_bits(), 1 << 7);
        assert_eq!(RequiredFeatures::TWO_LEVEL_SCALES.unknown_bits(), 0);
    }

    #[test]
    fn tensor_flags_roundtrip() {
        let f = TensorFlags::SENSITIVITY_VALID.union(TensorFlags::TASK_CRITICAL);
        assert!(f.contains(TensorFlags::SENSITIVITY_VALID));
        assert!(f.contains(TensorFlags::TASK_CRITICAL));
        assert!(!f.contains(TensorFlags::ALLOW_OFFLOAD));
        assert_eq!(f.unknown_bits(), 0);
    }

    #[test]
    fn policy_flags_bits() {
        assert_eq!(PolicyFlags::FORBID_REQUANT.bits(), 1);
        assert_eq!(PolicyFlags::FORBID_DEVICE_PLACEMENT.bits(), 2);
        assert_eq!(PolicyFlags::NONE.bits(), 0);
    }

    #[test]
    fn extension_point_flags_define_no_bits() {
        assert_eq!(ContractFlags::KNOWN_BITS, 0);
        assert_eq!(CalibrationFlags::KNOWN_BITS, 0);
        assert_eq!(WorkloadProfileFlags::KNOWN_BITS, 0);
        assert_eq!(RelationFlags::KNOWN_BITS, 0);
        assert_eq!(StateFlags::KNOWN_BITS, 0);
        assert_eq!(ContractFlags::NONE.unknown_bits(), 0);
        assert_eq!(StateFlags::from_bits_retain(1).unknown_bits(), 1);
        assert_eq!(RelationFlags::from_bits_retain(1).unknown_bits(), 1);
        assert_eq!(
            CalibrationFlags::from_bits_retain(0x8000).unknown_bits(),
            0x8000
        );
        assert_eq!(WorkloadProfileFlags::from_bits_retain(2).unknown_bits(), 2);
    }
}
