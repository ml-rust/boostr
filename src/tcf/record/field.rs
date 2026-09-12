//! Field-level codecs shared by every TCF v1 record: bounds-checked
//! little-endian reads at a fixed offset (Section 4), reserved-range zero checking
//! (Section 4, Section 8.1.5), and the `StringRef` name pair (Section 6).
//!
//! Nothing here indexes a slice blindly. Every accessor takes a `&[u8]` of
//! unknown length and returns `TcfError::SectionBounds` rather than panicking.

use crate::tcf::encoding::Encoding;
use crate::tcf::enums::{
    DotAccumulator, ExecutionRole, FallbackReason, InputRepresentation, LayoutId, MathMode,
    ModuleRole, OutputDtype, PrimaryMetric, ProofFormat, QuantAxis, RelationType, ResidencyClass,
    Role, RoundingMode, ScaleComputeDtype, StateDtype, WorkloadKind,
};
use crate::tcf::error::TcfError;
use crate::tcf::flags::{
    CalibrationFlags, ContractFlags, HeaderFlags, PolicyFlags, RelationFlags, RequiredFeatures,
    StateFlags, TensorFlags, WorkloadProfileFlags,
};

/// Bounds-checked immutable subslice `[off, off+len)`.
fn field_slice<'a>(
    bytes: &'a [u8],
    off: usize,
    len: usize,
    record: &'static str,
) -> Result<&'a [u8], TcfError> {
    let end = off
        .checked_add(len)
        .ok_or(TcfError::SectionBounds { section: record })?;
    bytes
        .get(off..end)
        .ok_or(TcfError::SectionBounds { section: record })
}

/// Bounds-checked mutable subslice `[off, off+len)`.
fn field_slice_mut<'a>(
    out: &'a mut [u8],
    off: usize,
    len: usize,
    record: &'static str,
) -> Result<&'a mut [u8], TcfError> {
    let end = off
        .checked_add(len)
        .ok_or(TcfError::SectionBounds { section: record })?;
    out.get_mut(off..end)
        .ok_or(TcfError::SectionBounds { section: record })
}

/// Reject a non-zero reserved range. Section 4: "Unused and padding bytes MUST be
/// zero. A reader MUST reject a non-zero reserved field in major version 1
/// unless a negotiated feature bit defines it."
///
/// `field` is built with `concat!` at the macro site, so it names record,
/// field, and offset together: `"TensorRecord.reserved@186"`.
pub fn expect_zero(
    bytes: &[u8],
    off: usize,
    len: usize,
    field: &'static str,
) -> Result<(), TcfError> {
    let s = field_slice(bytes, off, len, field)?;
    if s.iter().any(|b| *b != 0) {
        return Err(TcfError::NonzeroReserved { field });
    }
    Ok(())
}

/// Write a reserved range as zero. Section 4.
pub fn zero_range(
    out: &mut [u8],
    off: usize,
    len: usize,
    record: &'static str,
) -> Result<(), TcfError> {
    let s = field_slice_mut(out, off, len, record)?;
    s.fill(0);
    Ok(())
}

/// The default post-decode hook: a record with no cross-field invariant.
///
/// # Errors
/// Never. It exists so the layout macro always has a hook to call.
pub fn no_validate<T>(_record: &T) -> Result<(), TcfError> {
    Ok(())
}

/// One decodable record field, read at a fixed offset from a byte slice.
///
/// `LEN` is asserted against the offset table's declared size at compile
/// time by `define_record!`, so a type/size mismatch is a build error.
pub trait RecordField: Sized {
    /// Exact wire width of this field in bytes.
    const LEN: usize;

    /// Decode the field at `off`. `record` names the enclosing record for
    /// bounds-error reporting.
    fn read(bytes: &[u8], off: usize, record: &'static str) -> Result<Self, TcfError>;

    /// Encode the field at `off`.
    fn write(&self, out: &mut [u8], off: usize, record: &'static str) -> Result<(), TcfError>;
}

macro_rules! impl_field_scalar {
    ($( $ty:ty => $len:expr ),* $(,)?) => { $(
        impl RecordField for $ty {
            const LEN: usize = $len;

            fn read(bytes: &[u8], off: usize, record: &'static str) -> Result<Self, TcfError> {
                let s = field_slice(bytes, off, $len, record)?;
                let arr = <[u8; $len]>::try_from(s)
                    .map_err(|_| TcfError::SectionBounds { section: record })?;
                Ok(<$ty>::from_le_bytes(arr))
            }

            fn write(&self, out: &mut [u8], off: usize, record: &'static str) -> Result<(), TcfError> {
                let s = field_slice_mut(out, off, $len, record)?;
                s.copy_from_slice(&self.to_le_bytes());
                Ok(())
            }
        }
    )* };
}

// Section 4: all integer and floating-point storage is little-endian. `i16` is the
// only signed integer in the v1 schema (`ContractRecord.qmin`/`qmax`, Section 9).
// f32 fields decode structurally here; finiteness policy deferred to the
// verifier unit.
impl_field_scalar!(u16 => 2, u32 => 4, u64 => 8, i16 => 2, f32 => 4);

macro_rules! impl_field_bytes {
    ($( $len:expr ),* $(,)?) => { $(
        impl RecordField for [u8; $len] {
            const LEN: usize = $len;

            fn read(bytes: &[u8], off: usize, record: &'static str) -> Result<Self, TcfError> {
                let s = field_slice(bytes, off, $len, record)?;
                <[u8; $len]>::try_from(s).map_err(|_| TcfError::SectionBounds { section: record })
            }

            fn write(&self, out: &mut [u8], off: usize, record: &'static str) -> Result<(), TcfError> {
                let s = field_slice_mut(out, off, $len, record)?;
                s.copy_from_slice(self);
                Ok(())
            }
        }
    )* };
}

// Opaque digest and magic byte ranges. Section 5, Section 7, Section 9, Section 10, Section 10.5, Section 11, Section 15:
// carried verbatim; no digest is computed or checked in this unit.
impl_field_bytes!(8, 16, 32);

/// Offset of array element `i` of width `w`, without panicking arithmetic.
fn element_off(off: usize, i: usize, w: usize, record: &'static str) -> Result<usize, TcfError> {
    i.checked_mul(w)
        .and_then(|d| off.checked_add(d))
        .ok_or(TcfError::SectionBounds { section: record })
}

/// `TensorRecord.dims[8]`, eight u64. Section 8.
impl RecordField for [u64; 8] {
    const LEN: usize = 64;

    fn read(bytes: &[u8], off: usize, record: &'static str) -> Result<Self, TcfError> {
        let mut dims = [0u64; 8];
        for (i, slot) in dims.iter_mut().enumerate() {
            *slot = u64::read(bytes, element_off(off, i, 8, record)?, record)?;
        }
        Ok(dims)
    }

    fn write(&self, out: &mut [u8], off: usize, record: &'static str) -> Result<(), TcfError> {
        for (i, v) in self.iter().enumerate() {
            v.write(out, element_off(off, i, 8, record)?, record)?;
        }
        Ok(())
    }
}

/// `RelationRecord.input_tensor_id[4]`, four u32. Section 11. `0xffffffff`
/// (`consts::UNUSED_INPUT_ID`) marks an unused slot; the raw value is
/// carried through, with no enum lookup.
impl RecordField for [u32; 4] {
    const LEN: usize = 16;

    fn read(bytes: &[u8], off: usize, record: &'static str) -> Result<Self, TcfError> {
        let mut ids = [0u32; 4];
        for (i, slot) in ids.iter_mut().enumerate() {
            *slot = u32::read(bytes, element_off(off, i, 4, record)?, record)?;
        }
        Ok(ids)
    }

    fn write(&self, out: &mut [u8], off: usize, record: &'static str) -> Result<(), TcfError> {
        for (i, v) in self.iter().enumerate() {
            v.write(out, element_off(off, i, 4, record)?, record)?;
        }
        Ok(())
    }
}

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

/// A `(name_off, name_len)` pair into the string table. Section 6.
///
/// It carries the raw offset and length only. `off` is relative to
/// `Header.string_off`; resolving it to UTF-8 bytes is a later unit's job.
/// Names are provenance, not identity: runtime dispatch uses `tensor_id`,
/// `module_id`, and `role`, so there is no length ceiling on a name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct StringRef {
    /// Byte offset relative to `Header.string_off`.
    pub off: u64,
    /// Byte length of the UTF-8 name.
    pub len: u32,
}

impl StringRef {
    /// A `(off, len)` pair.
    #[must_use]
    pub const fn new(off: u64, len: u32) -> Self {
        Self { off, len }
    }

    /// Byte offset relative to `Header.string_off`.
    #[must_use]
    pub const fn off(self) -> u64 {
        self.off
    }

    /// Byte length of the UTF-8 name.
    #[must_use]
    pub const fn len(self) -> u32 {
        self.len
    }

    /// True when the name is the empty string.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    /// Decode the pair from its two separately-specified offsets.
    ///
    /// # Errors
    /// `TcfError::SectionBounds` when either read runs past `bytes`.
    pub fn read(
        bytes: &[u8],
        off_at: usize,
        len_at: usize,
        record: &'static str,
    ) -> Result<Self, TcfError> {
        Ok(Self {
            off: u64::read(bytes, off_at, record)?,
            len: u32::read(bytes, len_at, record)?,
        })
    }

    /// Encode the pair at its two separately-specified offsets.
    ///
    /// # Errors
    /// `TcfError::SectionBounds` when either write runs past `out`.
    pub fn write(
        &self,
        out: &mut [u8],
        off_at: usize,
        len_at: usize,
        record: &'static str,
    ) -> Result<(), TcfError> {
        self.off.write(out, off_at, record)?;
        self.len.write(out, len_at, record)
    }
}

#[cfg(test)]
mod tests;
