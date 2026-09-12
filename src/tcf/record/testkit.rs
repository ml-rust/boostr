//! Test-only helpers shared by the record unit tests. Not compiled into the
//! library.

use crate::tcf::error::TcfError;
use crate::tcf::record::traits::Record;

/// Write `value` at `off`, panicking (test-only) if it does not fit.
pub fn put(buf: &mut [u8], off: usize, value: &[u8]) {
    buf[off..off + value.len()].copy_from_slice(value);
}

/// Assert `encode(decode(bytes)) == bytes`, byte for byte, padding included.
pub fn roundtrips<R: Record + core::fmt::Debug>(bytes: &[u8]) {
    let record = R::decode(bytes).expect("sample record decodes");
    let mut out = vec![0xffu8; R::SIZE];
    record.encode(&mut out).expect("sample record encodes");
    assert_eq!(&out[..], &bytes[..R::SIZE], "byte-for-byte round trip");
}

/// Assert `R::decode` rejects a slice one byte short of `R::SIZE` with
/// `TcfError::SectionBounds`, never a panic. `section` is `R`'s name, as
/// reported in the error.
pub fn short_slice_errors<R: Record + core::fmt::Debug + PartialEq>(section: &'static str) {
    let bytes = vec![0u8; R::SIZE - 1];
    assert_eq!(R::decode(&bytes), Err(TcfError::SectionBounds { section }));
}
