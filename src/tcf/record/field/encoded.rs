//! `RecordField` codecs for the enumerated wire types: `Encoding` and its
//! zero-means-absent slots (Section 7, Section 12), and every u16 / u32 enum
//! in the v1 schema.

use crate::tcf::encoding::Encoding;
use crate::tcf::enums::{
    DotAccumulator, ExecutionRole, FallbackReason, InputRepresentation, LayoutId, MathMode,
    ModuleRole, OutputDtype, PrimaryMetric, ProofFormat, QuantAxis, RelationType, ResidencyClass,
    Role, RoundingMode, ScaleComputeDtype, StateDtype, WorkloadKind,
};
use crate::tcf::error::TcfError;

use super::scalar::{RecordField, element_off};

/// `TensorRecord.encoding`. Section 12.
impl RecordField for Encoding {
    const LEN: usize = 2;

    fn read(bytes: &[u8], off: usize, record: &'static str) -> Result<Self, TcfError> {
        Self::try_from(u16::read(bytes, off, record)?)
    }

    fn write(&self, out: &mut [u8], off: usize, record: &'static str) -> Result<(), TcfError> {
        self.to_u16().write(out, off, record)
    }
}

/// Zero-means-absent decode shared by every optional-encoding slot: `0` is
/// NOT an encoding identifier, so it never reaches `Encoding::try_from`,
/// which correctly rejects `0` as unassigned. Used by `Option<Encoding>`
/// (`ModuleRecord.fallback_encoding`, Section 7) and `[Option<Encoding>; 4]`
/// (`ModuleRecord.preferred_encoding`, Section 7).
fn optional_encoding_from_u16(raw: u16) -> Result<Option<Encoding>, TcfError> {
    if raw == 0 {
        Ok(None)
    } else {
        Ok(Some(Encoding::try_from(raw)?))
    }
}

/// Inverse of [`optional_encoding_from_u16`]: `None` writes back as `0`.
fn optional_encoding_to_u16(slot: Option<Encoding>) -> u16 {
    match slot {
        None => 0,
        Some(e) => e.to_u16(),
    }
}

/// `ModuleRecord.fallback_encoding`. Section 7. `0` means the module declares no
/// fallback, symmetric with `preferred_encoding` slots.
impl RecordField for Option<Encoding> {
    const LEN: usize = 2;

    fn read(bytes: &[u8], off: usize, record: &'static str) -> Result<Self, TcfError> {
        optional_encoding_from_u16(u16::read(bytes, off, record)?)
    }

    fn write(&self, out: &mut [u8], off: usize, record: &'static str) -> Result<(), TcfError> {
        optional_encoding_to_u16(*self).write(out, off, record)
    }
}

/// `ModuleRecord.preferred_encoding[4]`, four u16, ordered by producer
/// preference, highest first. Section 7. `0` means "unused" and is NOT an encoding
/// identifier: it decodes to `None` and never reaches `Encoding::try_from`,
/// which correctly rejects `0` as unassigned.
impl RecordField for [Option<Encoding>; 4] {
    const LEN: usize = 8;

    fn read(bytes: &[u8], off: usize, record: &'static str) -> Result<Self, TcfError> {
        let mut slots = [None; 4];
        for (i, slot) in slots.iter_mut().enumerate() {
            let raw = u16::read(bytes, element_off(off, i, 2, record)?, record)?;
            *slot = optional_encoding_from_u16(raw)?;
        }
        Ok(slots)
    }

    fn write(&self, out: &mut [u8], off: usize, record: &'static str) -> Result<(), TcfError> {
        for (i, slot) in self.iter().enumerate() {
            optional_encoding_to_u16(*slot).write(out, element_off(off, i, 2, record)?, record)?;
        }
        Ok(())
    }
}

macro_rules! impl_field_enum_u16 {
    ($( $ty:ty ),* $(,)?) => { $(
        impl RecordField for $ty {
            const LEN: usize = 2;

            fn read(bytes: &[u8], off: usize, record: &'static str) -> Result<Self, TcfError> {
                Self::try_from(u16::read(bytes, off, record)?)
            }

            fn write(&self, out: &mut [u8], off: usize, record: &'static str) -> Result<(), TcfError> {
                self.to_u16().write(out, off, record)
            }
        }
    )* };
}

macro_rules! impl_field_enum_u32 {
    ($( $ty:ty ),* $(,)?) => { $(
        impl RecordField for $ty {
            const LEN: usize = 4;

            fn read(bytes: &[u8], off: usize, record: &'static str) -> Result<Self, TcfError> {
                Self::try_from(u32::read(bytes, off, record)?)
            }

            fn write(&self, out: &mut [u8], off: usize, record: &'static str) -> Result<(), TcfError> {
                self.to_u32().write(out, off, record)
            }
        }
    )* };
}

impl_field_enum_u16!(
    ModuleRole,
    StateDtype,
    ResidencyClass,
    Role,
    FallbackReason,
    ExecutionRole,
    InputRepresentation,
    QuantAxis,
    RoundingMode,
    ScaleComputeDtype,
    DotAccumulator,
    OutputDtype,
    MathMode,
    PrimaryMetric,
    WorkloadKind,
    RelationType,
);

impl_field_enum_u32!(LayoutId, ProofFormat);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preferred_encoding_zero_is_none() {
        let bytes = [0u8; 8];
        let slots = <[Option<Encoding>; 4]>::read(&bytes, 0, "T").expect("all unused");
        assert_eq!(slots, [None, None, None, None]);
        let mut out = [0xffu8; 8];
        slots.write(&mut out, 0, "T").expect("in bounds");
        assert_eq!(out, [0u8; 8]);
    }
}
