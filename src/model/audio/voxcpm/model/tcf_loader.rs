//! [`VoxCpm2Model::from_tcf`]: load the whole model from a single TCF file.
//!
//! # A TCF carries weights and provenance, never a model config
//!
//! `from_gguf` takes an optional `config.json` path because a GGUF *could*
//! embed one in its metadata map and today's compressr does not. TCF is not
//! the same case: the format has no metadata map to embed one in. Its
//! directory is a fixed set of record arrays — header, modules, tensors,
//! contracts, calibrations, relations, workload profiles — plus a string
//! table that exists to name those records (SPECIFICATION.md Sections 5-11).
//! There is no free-form key/value section anywhere in v1, so no writer can
//! put a `config.json` in one.
//!
//! `config_json` is therefore a REQUIRED path here rather than an
//! `Option`. An option that must always be `Some` turns a fact known at
//! compile time into an error at load time.
//!
//! # What a TCF does NOT hold, same as a GGUF
//!
//! The AudioVAE and `tokenizer.json`. compressr converts the transformer
//! stack only, so `audiovae_path` stays a separate argument exactly as it is
//! for [`from_checkpoint`](VoxCpm2Model::from_checkpoint) and
//! [`from_gguf`](VoxCpm2Model::from_gguf).
//!
//! # What stays quantized in memory
//!
//! Every tensor at a native encoding, which is most of the matmul weights.
//! Section 12's raw encodings — the BF16 the writer falls back to for a
//! rank-1 tensor it cannot tile, plus anything pinned — arrive dense. The
//! rule and its counts live on
//! [`TcfSource`](crate::model::audio::voxcpm::loader::support::TcfSource);
//! nothing in this file special-cases a tensor by name.

use crate::error::Result;
use crate::format::tcf::TcfLoader;
use crate::model::audio::voxcpm::loader::support::TcfSource;
use crate::model::audio::voxcpm::model::loader::{StackConfigs, VoxCpm2Model};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, TensorOps, TypeConversionOps, UnaryOps};
use numr::runtime::Runtime;
use std::path::Path;

impl<R: Runtime<DType = DType>> VoxCpm2Model<R>
where
    R::Client: TypeConversionOps<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    /// Load the whole model from a TCF plus the separate AudioVAE file.
    ///
    /// `config_json` is the checkpoint's own `config.json` — see the module
    /// docs for why the file cannot supply it.
    ///
    /// `dtype` casts every transformer-stack tensor that arrives dense, same
    /// as [`from_gguf`](Self::from_gguf). Decoded tensors arrive as F32, so
    /// `None` means F32 rather than the BF16 a safetensors checkpoint gives.
    ///
    /// `Some(BF16)`/`Some(F16)` is REJECTED once a natively encoded
    /// projection is reached: `quant_matmul` requires F32 activations, so
    /// honouring the request would mean dequantizing the very weights this
    /// path keeps packed. The error names the tensor.
    ///
    /// # Errors
    /// [`crate::error::Error::ModelError`] for a file that fails any
    /// Section 15 check, one that repeats a tensor name (see [`TcfSource`]),
    /// an unreadable `config_json`, or a missing or misshapen weight.
    pub fn from_tcf<P: AsRef<Path>, Q: AsRef<Path>>(
        tcf_path: P,
        config_json: &Path,
        audiovae_path: Q,
        device: &R::Device,
        dtype: Option<DType>,
    ) -> Result<Self> {
        let cfgs = StackConfigs::from_config_json(config_json)?;
        // Opened ONCE for all five transformer-stack sub-models, and its
        // directory bound ONCE by `TcfSource` on top of that — a per-tensor
        // reparse would be quadratic over 577 names.
        let loader = TcfLoader::open(tcf_path.as_ref())?;
        let mut source = TcfSource::new(&loader)?;
        Self::from_source(&mut source, cfgs, audiovae_path.as_ref(), device, dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn rejects_missing_tcf() {
        let device = <CpuRuntime as Runtime>::default_device();
        assert!(
            VoxCpm2Model::<CpuRuntime>::from_tcf(
                "/nonexistent/voxcpm2.tcf",
                Path::new("/nonexistent/config.json"),
                "/nonexistent/audiovae.safetensors",
                &device,
                Some(DType::F32),
            )
            .is_err()
        );
    }
}
