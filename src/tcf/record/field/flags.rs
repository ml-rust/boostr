//! Flag-field codecs with unknown-bit rejection (Section 8.1.5), and the
//! `required_features` exception (Section 5.2).

use crate::tcf::error::TcfError;
use crate::tcf::flags::{
    CalibrationFlags, ContractFlags, HeaderFlags, PolicyFlags, RelationFlags, RequiredFeatures,
    StateFlags, TensorFlags, WorkloadProfileFlags,
};

use super::scalar::RecordField;

/// One flag field, decoded with unknown-bit rejection.
///
/// Section 8.1.5: "Every bit not defined by this specification is reserved and MUST
/// be zero, in every flag field. A reader encountering a set bit it does not
/// recognize MUST reject the file."
pub trait RecordFlags: Sized {
    /// Exact wire width of this flag field in bytes.
    const LEN: usize;

    /// Decode the flag field at `off`, rejecting any unknown bit. `field`
    /// names the record and field for error reporting.
    fn read_checked(
        bytes: &[u8],
        off: usize,
        record: &'static str,
        field: &'static str,
    ) -> Result<Self, TcfError>;

    /// Encode the flag field at `off`.
    fn write(&self, out: &mut [u8], off: usize, record: &'static str) -> Result<(), TcfError>;
}

macro_rules! impl_record_flags {
    ($( $ty:ty : $repr:ty => $len:expr ),* $(,)?) => { $(
        impl RecordFlags for $ty {
            const LEN: usize = $len;

            fn read_checked(
                bytes: &[u8],
                off: usize,
                record: &'static str,
                field: &'static str,
            ) -> Result<Self, TcfError> {
                let value = Self::from_bits_retain(<$repr as RecordField>::read(bytes, off, record)?);
                if value.unknown_bits() != 0 {
                    return Err(TcfError::NonzeroReserved { field });
                }
                Ok(value)
            }

            fn write(&self, out: &mut [u8], off: usize, record: &'static str) -> Result<(), TcfError> {
                <$repr as RecordField>::write(&self.bits(), out, off, record)
            }
        }
    )* };
}

// Section 8.1.5 table: every one of these rejects an unknown bit as
// `E_NONZERO_RESERVED`. The five with no defined bits are extension points
// sized ahead of need; a field with no defined bits and no zero requirement
// is a field a producer fills with anything.
impl_record_flags!(
    HeaderFlags: u32 => 4,
    TensorFlags: u32 => 4,
    PolicyFlags: u32 => 4,
    ContractFlags: u32 => 4,
    CalibrationFlags: u16 => 2,
    WorkloadProfileFlags: u16 => 2,
    RelationFlags: u16 => 2,
    StateFlags: u32 => 4,
);

/// `Header.required_features`. Section 5.2, Section 8.1.5.
///
/// The single exception to the `E_NONZERO_RESERVED` rule: a set bit here is
/// a **capability claim** the reader cannot satisfy, not a malformed byte,
/// so it rejects as `E_UNKNOWN_REQUIRED_FEATURE` naming the offending bit.
impl RecordFlags for RequiredFeatures {
    const LEN: usize = 8;

    fn read_checked(
        bytes: &[u8],
        off: usize,
        record: &'static str,
        _field: &'static str,
    ) -> Result<Self, TcfError> {
        let value = Self::from_bits_retain(u64::read(bytes, off, record)?);
        let unknown = value.unknown_bits();
        if unknown != 0 {
            return Err(TcfError::UnknownRequiredFeature {
                bit: unknown.trailing_zeros(),
            });
        }
        Ok(value)
    }

    fn write(&self, out: &mut [u8], off: usize, record: &'static str) -> Result<(), TcfError> {
        self.bits().write(out, off, record)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_features_unknown_bit_is_its_own_error() {
        let bytes = 0x80u64.to_le_bytes();
        assert_eq!(
            RequiredFeatures::read_checked(&bytes, 0, "Header", "Header.required_features"),
            Err(TcfError::UnknownRequiredFeature { bit: 7 })
        );
    }

    #[test]
    fn other_flag_fields_reject_unknown_bits_as_reserved() {
        let bytes = 0x20u32.to_le_bytes();
        assert_eq!(
            TensorFlags::read_checked(&bytes, 0, "TensorRecord", "TensorRecord.flags"),
            Err(TcfError::NonzeroReserved {
                field: "TensorRecord.flags"
            })
        );
        let bytes = 1u16.to_le_bytes();
        assert_eq!(
            RelationFlags::read_checked(&bytes, 0, "RelationRecord", "RelationRecord.flags"),
            Err(TcfError::NonzeroReserved {
                field: "RelationRecord.flags"
            })
        );
    }
}
