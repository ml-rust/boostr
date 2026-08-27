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
//! The K-quant writers (Q4_K, Q5_K, Q6_K) do NOT pick their scale by absmax.
//! They run llama.cpp's iterative per-sub-block scale search, which is worth
//! roughly 9% of the quantization error at identical file size. See
//! `quant/cpu/kernels/quantize/search.rs`.

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
}
