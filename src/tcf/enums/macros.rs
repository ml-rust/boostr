//! The crate-wide macros that generate the fallible-decode shape every TCF
//! enum shares: an EXHAUSTIVE enum, a `to_u16`/`to_u32` encoder, and a
//! `TryFrom` decoder that rejects every value not listed.
//!
//! # Exhaustive on purpose
//!
//! These enums are NOT `#[non_exhaustive]`. tcf-core has exactly two
//! consumers, boostr and compressr, and both are ours. A new variant must
//! break their `match` arms at compile time, so the reader that cannot honour
//! it is forced to say so before a file carrying it exists. Wire safety does
//! not depend on this: `TryFrom` matches the raw integer and rejects every
//! value not listed.

/// Generates a fallible u16-backed enum: exhaustive, a `to_u16` encoder, and
/// a `TryFrom<u16>` decoder that rejects every value not listed
/// (`E_UNKNOWN_ENUM_VALUE`, naming `$field` — the wire field name, not the
/// Rust type name). No catch-all variant: an unrecognized value is always a
/// decode error, never a value the type can represent.
#[macro_export]
macro_rules! define_enum_u16 {
    (
        $(#[$meta:meta])*
        pub enum $Name:ident as $field:literal {
            $( $(#[$vmeta:meta])* $Variant:ident = $val:expr ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $Name {
            $( $(#[$vmeta])* $Variant, )*
        }

        impl $Name {
            /// Encode as the wire `u16` representation.
            #[must_use]
            pub const fn to_u16(self) -> u16 {
                match self {
                    $( Self::$Variant => $val, )*
                }
            }
        }

        impl core::convert::TryFrom<u16> for $Name {
            type Error = $crate::tcf::error::TcfError;

            fn try_from(value: u16) -> Result<Self, Self::Error> {
                match value {
                    $( $val => Ok(Self::$Variant), )*
                    other => Err($crate::tcf::error::TcfError::UnknownEnumValue {
                        field: $field,
                        raw: other as u32,
                    }),
                }
            }
        }
    };
}

/// Generates a fallible u32-backed enum. See [`define_enum_u16!`].
#[macro_export]
macro_rules! define_enum_u32 {
    (
        $(#[$meta:meta])*
        pub enum $Name:ident as $field:literal {
            $( $(#[$vmeta:meta])* $Variant:ident = $val:expr ),* $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $Name {
            $( $(#[$vmeta])* $Variant, )*
        }

        impl $Name {
            /// Encode as the wire `u32` representation.
            #[must_use]
            pub const fn to_u32(self) -> u32 {
                match self {
                    $( Self::$Variant => $val, )*
                }
            }
        }

        impl core::convert::TryFrom<u32> for $Name {
            type Error = $crate::tcf::error::TcfError;

            fn try_from(value: u32) -> Result<Self, Self::Error> {
                match value {
                    $( $val => Ok(Self::$Variant), )*
                    other => Err($crate::tcf::error::TcfError::UnknownEnumValue {
                        field: $field,
                        raw: other,
                    }),
                }
            }
        }
    };
}
