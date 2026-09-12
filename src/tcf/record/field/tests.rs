//! Tests for the record field codecs.

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

#[test]
fn preferred_encoding_zero_is_none() {
    let bytes = [0u8; 8];
    let slots = <[Option<Encoding>; 4]>::read(&bytes, 0, "T").expect("all unused");
    assert_eq!(slots, [None, None, None, None]);
    let mut out = [0xffu8; 8];
    slots.write(&mut out, 0, "T").expect("in bounds");
    assert_eq!(out, [0u8; 8]);
}

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

#[test]
fn string_ref_roundtrips() {
    let mut buf = [0u8; 16];
    let sref = StringRef::new(0x1234_5678_9abc, 42);
    sref.write(&mut buf, 0, 8, "T").expect("in bounds");
    assert_eq!(StringRef::read(&buf, 0, 8, "T"), Ok(sref));
    assert!(!sref.is_empty());
}
