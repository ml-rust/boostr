//! Field-level codecs shared by every TCF v1 record: bounds-checked
//! little-endian reads at a fixed offset (Section 4), reserved-range zero checking
//! (Section 4, Section 8.1.5), and the `StringRef` name pair (Section 6).
//!
//! Nothing here indexes a slice blindly. Every accessor takes a `&[u8]` of
//! unknown length and returns `TcfError::SectionBounds` rather than panicking.

mod encoded;
mod flags;
mod scalar;
mod string_ref;

pub use flags::RecordFlags;
pub use scalar::{RecordField, expect_zero, no_validate, zero_range};
pub use string_ref::StringRef;
