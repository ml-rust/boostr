//! TCF: GGUF's block-quantized payloads in a strict, self-describing
//! container. See `FORMAT.md` beside this module for the binary layout.
//!
//! What the container adds over a GGUF file holding the same bytes:
//!
//! - an activation contract every tensor names, refused at open when it
//!   dangles and checked against the kernel at dispatch;
//! - a proof vector per quantized tensor that the READER's block decoder
//!   must reproduce, plus payload and directory digests;
//! - provenance: module records, a fallback reason on every tensor not at
//!   its module's preferred encoding, calibration and per-tensor
//!   sensitivity records when the allocation was measured;
//! - a directory computable from headers alone, so a placement planner
//!   never touches a payload page.
//!
//! The payload encodings are `Encoding::Raw` (literal element types) and
//! `Encoding::Block` (GGML block streams, identified by `ggml_type`). TCF's
//! own tile encodings were removed after measurement found no advantage
//! over the GGML blocks at any bit width; their identifier range stays
//! unassigned so an old file is refused by name.
//!
//! A block tensor declares the `GgmlReference` contract: the kernel family
//! `ggml-quants.c` defines for its block type, whose activation path is
//! per backend. The container therefore pins bytes, provenance, and the
//! family today. Pinning one activation representation across backends,
//! or a payload measurably better than the GGML blocks, is future work and
//! arrives with its own encoding or contract value.

pub mod binary16;
pub mod consts;
pub mod digest;
pub mod encoding;
pub mod enums;
pub mod error;
pub mod flags;
pub mod proof;
pub mod reader;
pub mod record;
pub mod streaming;
#[cfg(test)]
pub(crate) mod test_blocks;
pub mod writer;

pub use binary16::{bits_to_f32, f32_to_bits};
pub use consts::*;
pub use digest::{
    Digest128, Digest256, contract_digest, hash_128, hash_256, payload_digest, policy_digest,
    relation_digest,
};
pub use encoding::*;
pub use enums::*;
pub use error::TcfError;
pub use flags::*;
pub use proof::{
    BlockDecoder, PROOF_BYTES, PROOF_FORMAT_F16, PROOF_FORMAT_NONE, block_proof_values,
    proof_indices,
};
pub use reader::TcfFile;
pub use record::{
    CalibrationRecord, ContractRecord, HEADER_DIGEST_RANGE, Header, ModuleRecord, Record,
    RelationRecord, StringRef, TensorRecord, WorkloadProfileRecord,
};
pub use writer::{Payload, TcfWriter};
