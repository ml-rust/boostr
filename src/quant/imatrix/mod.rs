//! Importance matrices: what a calibration run measures about a weight's
//! input columns, and the file a quantizer reads it back from.
//!
//! `collect` owns the capture and the device-side accumulation.
//! `format` owns the file layout, its writer, its reader, and the check a
//! consumer runs before trusting one.

pub mod collect;
pub mod format;

pub use collect::{arm, disarm, finish, is_armed, observe, register_name, registered_names};
pub use format::{
    IMATRIX_HEADER_LEN, IMATRIX_MAGIC, IMATRIX_VERSION, ImportanceCheck, ImportanceEntry,
    ImportanceMatrix, MAX_NAME_LEN,
};
