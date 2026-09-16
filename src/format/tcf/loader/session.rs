//! `TcfSession`: a validated view of a [`TcfLoader`]'s directory reused across
//! loads, plus the digest/proof verification and encoding-to-format mapping
//! both `TcfLoader` and `TcfSession` share.

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::directory::TcfLoader;
use crate::error::{Error, Result};
use crate::format::tcf::block::BoostrBlockDecoder;
use crate::format::tcf::error::tcf_tensor_error;
use crate::format::tcf::metadata::encoding_name;
use crate::quant::{QuantFormat, QuantTensor};
use crate::tcf::{Encoding, TcfFile};

/// One validated view of a [`TcfLoader`]'s directory, reused across loads.
///
/// Bound by [`TcfLoader::session`]. What a session skips is the repeated
/// DIRECTORY parse, never a payload check: every load below still verifies
/// the tensor's digests and proof vector first (Section 15).
///
/// The borrow is what makes this sound — the validated view points into the
/// loader's mapping, so it cannot outlive the loader that owns it.
pub struct TcfSession<'a> {
    loader: &'a TcfLoader,
    file: TcfFile<'a>,
}

impl<'a> TcfSession<'a> {
    /// Build a session bound to `loader`, reopening and validating the file
    /// once up front.
    pub(super) fn new(loader: &'a TcfLoader) -> Result<Self> {
        Ok(Self {
            loader,
            file: loader.file()?,
        })
    }

    /// The loader this session reads through.
    pub fn loader(&self) -> &'a TcfLoader {
        self.loader
    }

    /// Verify and decode `name` into host f32 values, row-major.
    ///
    /// # Errors
    /// Every error [`TcfLoader::load_tensor_f32`] raises.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let index = self.loader.index_of(name)?;
        self.loader.decode_at(&self.file, index)
    }

    /// Verify and decode `name` onto `device` as a dense f32 tensor.
    ///
    /// # Errors
    /// Every error [`TcfLoader::load_tensor`] raises.
    pub fn tensor<R: Runtime<DType = DType>>(
        &self,
        name: &str,
        device: &R::Device,
    ) -> Result<Tensor<R>> {
        let index = self.loader.index_of(name)?;
        let values = self.loader.decode_at(&self.file, index)?;
        let shape = self.loader.shape_at(index)?;
        Tensor::<R>::from_slice(&values, &shape, device).map_err(Error::Numr)
    }

    /// Verify `name` and place it on `device` STILL QUANTIZED.
    ///
    /// # Errors
    /// Every error [`TcfLoader::load_quant_tensor`] raises, a raw encoding
    /// included.
    pub fn quant_tensor<R: Runtime<DType = DType>>(
        &self,
        name: &str,
        device: &R::Device,
    ) -> Result<QuantTensor<R>> {
        let index = self.loader.index_of(name)?;
        self.loader.quant_at(&self.file, index, device)
    }
}

/// Verify one tensor's digests and proof vector. Section 15.
///
/// A block tensor's proof is checked with boostr's own decoder, so a stream
/// these kernels would read differently from the producer fails here rather
/// than at first use.
pub(super) fn verify_at(
    file: &TcfFile<'_>,
    record: &crate::tcf::TensorRecord,
    name: &str,
) -> Result<()> {
    file.verify_tensor_with(record, Some(&BoostrBlockDecoder))
        .map_err(|e| tcf_tensor_error(name, "verify", e))
}

/// The runtime format for a quantized encoding.
///
/// # Errors
/// [`Error::ModelError`] naming the encoding and the tensor, when the
/// encoding is raw or a block layout this build has no kernel for.
pub(super) fn quant_format(encoding: Encoding, name: &str) -> Result<QuantFormat> {
    match encoding {
        Encoding::Block(block) => crate::format::tcf::block::block_format(block, name),
        raw @ Encoding::Raw(_) => Err(Error::ModelError {
            reason: format!(
                "TCF tensor '{name}': encoding {} is not quantized, so it has no packed form; load it as a dense tensor",
                encoding_name(raw),
            ),
        }),
    }
}
