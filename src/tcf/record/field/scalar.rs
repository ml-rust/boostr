//! The `RecordField` trait, bounds-checked slicing, reserved-range helpers,
//! and the scalar / byte-array / fixed-array field codecs (Section 4).

use crate::tcf::error::TcfError;

/// Bounds-checked immutable subslice `[off, off+len)`.
pub(super) fn field_slice<'a>(
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
pub(super) fn field_slice_mut<'a>(
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
pub(super) fn element_off(
    off: usize,
    i: usize,
    w: usize,
    record: &'static str,
) -> Result<usize, TcfError> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_reads_are_little_endian() {
        let bytes = [0x01, 0x02, 0x03, 0x04];
        assert_eq!(u16::read(&bytes, 0, "T"), Ok(0x0201));
        assert_eq!(u32::read(&bytes, 0, "T"), Ok(0x0403_0201));
    }

    #[test]
    fn short_slice_errors_instead_of_panicking() {
        let bytes = [0u8; 3];
        assert_eq!(
            u32::read(&bytes, 0, "T"),
            Err(TcfError::SectionBounds { section: "T" })
        );
        assert_eq!(
            u64::read(&bytes, usize::MAX, "T"),
            Err(TcfError::SectionBounds { section: "T" })
        );
    }

    #[test]
    fn signed_i16_roundtrips_negative() {
        let mut buf = [0u8; 2];
        (-128i16).write(&mut buf, 0, "T").expect("in bounds");
        assert_eq!(i16::read(&buf, 0, "T"), Ok(-128));
    }

    #[test]
    fn expect_zero_rejects_a_set_byte() {
        let mut bytes = [0u8; 8];
        assert_eq!(expect_zero(&bytes, 0, 8, "R.reserved@0"), Ok(()));
        bytes[5] = 1;
        assert_eq!(
            expect_zero(&bytes, 0, 8, "R.reserved@0"),
            Err(TcfError::NonzeroReserved {
                field: "R.reserved@0"
            })
        );
    }
}
