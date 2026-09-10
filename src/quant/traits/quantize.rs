//! Quantization operations trait — float tensor → block-packed quantized bytes
//!
//! This is the WRITER side of [`DequantOps`](crate::quant::DequantOps). The two
//! MUST agree byte for byte: every writer in `quant/cpu/kernels/quantize/` is
//! built as the exact inverse of the matching reader in
//! `quant/cpu/kernels/dequant_simple.rs` / `dequant_k_quants/`, and the readers
//! are the authority whenever the two disagree.
//!
//! # Why a writer belongs in boostr
//!
//! Quantization is a kernel, not a pipeline step: picking a block scale is the
//! same arithmetic on every backend and the packing is dictated by the GGUF
//! block layout that boostr's readers already encode. Keeping the writer next
//! to the reader is what makes "round-trip through our own dequant kernel" a
//! usable test — a writer that ships with its own private reader can agree with
//! itself while disagreeing with the format.
//!
//! # Accuracy
//!
//! Only Q4_1 picks its scale by plain absmax.
//!
//! # Importance-weighted (imatrix) K-quants
//!
//! `ggml-quants.c` carries a SECOND writer per K-quant, `quantize_row_*_K_impl`,
//! which `quantize_*_K` takes the moment an importance matrix is supplied. It
//! derives its per-sub-block weights from the importance as well as the data,
//! and nearly every GGUF quant the ecosystem ships comes out of it, so a
//! comparison against the no-imatrix writer is a comparison against something
//! the ecosystem does not use. [`QuantizeOps::quantize_with_importance`] is that
//! path — `quant/cpu/kernels/quantize/search_imatrix.rs` and the `*_imatrix`
//! writers beside it. Passing no importance leaves the output bit-identical to
//! [`QuantizeOps::quantize`].
//!
//! - Q2_K, Q3_K, Q4_K, Q5_K and Q6_K run llama.cpp's iterative per-sub-block
//!   scale search — `quant/cpu/kernels/quantize/search.rs`.
//! - Q4_0 and Q8_0 sweep their single block scale against an unweighted
//!   squared-error objective, scoring the binary16 value the reader loads —
//!   `quant/cpu/kernels/quantize/block_scale.rs`.
//! - Q4_1 takes a direct min/max fit. Neither search models a signed offset
//!   that is ADDED.
//!
//! # Q4_0 and Q8_0 deliberately diverge from llama.cpp
//!
//! llama.cpp's `quantize_row_q4_0` and `quantize_row_q8_0` do no search, so
//! boostr's bytes for those two formats are NOT identical to llama.cpp's on the
//! same input. The output is a valid block of the same format and size,
//! llama.cpp reads it correctly, and it reconstructs the source more closely.
//! Anything needing byte-for-byte reproduction of llama.cpp's encoder must not
//! use these two writers.

use crate::error::Result;
use crate::quant::{QuantFormat, QuantTensor};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Quantize a float tensor into GGUF block-packed storage
pub trait QuantizeOps<R: Runtime> {
    /// Quantize `input` into `format`, returning tightly-packed blocks
    ///
    /// # Contract
    ///
    /// - `input` dtype must be F32, F16 or BF16 (converted to F32 internally)
    /// - The last dimension of `input.shape()` must be a multiple of
    ///   `format.block_size()` — quantization runs along the LAST axis, so a
    ///   `[out_features, in_features]` weight blocks along `in_features`
    /// - Output storage is exactly `format.storage_bytes(numel)` bytes
    /// - Output shape is the LOGICAL element shape, unchanged from `input`
    ///
    /// # Errors
    ///
    /// - [`Error::UnsupportedQuantFormat`](crate::error::Error::UnsupportedQuantFormat)
    ///   if this backend has no writer for `format`
    /// - [`Error::QuantError`](crate::error::Error::QuantError) on a
    ///   non-float dtype or a block-size mismatch
    fn quantize(&self, input: &Tensor<R>, format: QuantFormat) -> Result<QuantTensor<R>>;

    /// Quantize `input` into `format`, weighting each column by `importance`
    ///
    /// `importance` is one non-negative entry per COLUMN of the weight matrix —
    /// per element of the LAST axis, the axis quantization runs along — so a
    /// `[out_features, in_features]` weight takes `in_features` entries and
    /// every row indexes the same vector. `None` is the plain
    /// [`quantize`](QuantizeOps::quantize) path, bit for bit.
    ///
    /// Loading the importance from a file is the caller's job. This takes the
    /// values.
    ///
    /// # Errors
    ///
    /// - [`Error::QuantError`](crate::error::Error::QuantError) if `importance`
    ///   is not exactly as long as the last axis, or holds a non-finite or
    ///   negative entry. A wrong-length vector is NEVER quietly ignored: the
    ///   file it would produce is indistinguishable from an unweighted one, so
    ///   the mistake would survive every check downstream.
    /// - [`Error::UnsupportedQuantFormat`](crate::error::Error::UnsupportedQuantFormat)
    ///   if `format` has no importance-weighted writer on this backend
    /// - everything [`quantize`](QuantizeOps::quantize) returns
    fn quantize_with_importance(
        &self,
        input: &Tensor<R>,
        format: QuantFormat,
        importance: Option<&[f32]>,
    ) -> Result<QuantTensor<R>> {
        match importance {
            None => self.quantize(input, format),
            Some(_) => Err(crate::error::Error::UnsupportedQuantFormat {
                format: format!(
                    "{} has no importance-weighted quantize kernel on this backend",
                    format.name()
                ),
            }),
        }
    }
}
