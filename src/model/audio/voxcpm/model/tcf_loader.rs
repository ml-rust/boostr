//! [`VoxCpm2Model::from_tcf`]: load the whole model from a single TCF file.
//!
//! # A TCF carries weights and provenance, never a model config
//!
//! `from_gguf` takes an optional `config.json` path because a GGUF embeds
//! one in its metadata map when compressr writes it. TCF is not the same
//! case: the format has no metadata map to embed one in. Its
//! directory is a fixed set of record arrays — header, modules, tensors,
//! contracts, calibrations, relations, workload profiles — plus a string
//! table that exists to name those records (`src/tcf/FORMAT.md` Sections 5-11).
//! There is no free-form key/value section anywhere in v1, so no writer can
//! put a `config.json` in one.
//!
//! `config_json` is therefore a REQUIRED path here rather than an
//! `Option`. An option that must always be `Some` turns a fact known at
//! compile time into an error at load time.
//!
//! # The AudioVAE, same rule as a GGUF
//!
//! compressr embeds the VAE (folded, dense, under `vae.`) when the input
//! directory holds `audiovae.pth`; the TCF shares the GGUF's plan pipeline,
//! so a TCF written from such a directory carries it too. `from_tcf` probes
//! for it and reads it from the file when present; `audiovae_path` is then
//! optional, and ignored when given. An older file still needs the path. A
//! block-encoded `vae.*` tensor is refused by name — the VAE loaders are
//! verified against dense fixtures only. No TCF holds a `tokenizer.json`.
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
use crate::format::tcf::{TcfLoader, encoding_name};
use crate::model::audio::voxcpm::loader::support::{DenseWeightSource, TcfSource};
use crate::model::audio::voxcpm::model::loader::{StackConfigs, VoxCpm2Model, packed_vae_tensor};
use crate::model::audio::voxcpm::vae::{VAE_GGUF_PROBE_TENSOR, VAE_GGUF_ROOT};
use crate::quant::traits::DequantOps;
use crate::tcf::Encoding;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, TensorOps, TypeConversionOps, UnaryOps};
use numr::runtime::Runtime;
use std::path::Path;

impl<R: Runtime<DType = DType>> VoxCpm2Model<R>
where
    R::Client: TypeConversionOps<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    /// Load the whole model from a TCF, and the separate AudioVAE file when
    /// the TCF does not embed one.
    ///
    /// `config_json` is the checkpoint's own `config.json` — see the module
    /// docs for why the file cannot supply it. The AudioVAE is the file's
    /// own `vae.*` tensors when present, else `audiovae_path`, else an error
    /// naming both.
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
    /// `vae_decoder_dtype` casts every AudioVAE DECODER tensor independently
    /// of `dtype` — see [`super::loader`]'s module docs. The encoder always
    /// loads at F32.
    ///
    /// # Errors
    /// [`crate::error::Error::ModelError`] for a file that fails any
    /// Section 15 check, one that repeats a tensor name (see [`TcfSource`]),
    /// an unreadable `config_json`, or a missing or misshapen weight.
    pub fn from_tcf<P: AsRef<Path>>(
        tcf_path: P,
        config_json: &Path,
        audiovae_path: Option<&Path>,
        device: &R::Device,
        dtype: Option<DType>,
        vae_decoder_dtype: Option<DType>,
    ) -> Result<Self> {
        let cfgs = StackConfigs::from_config_json(config_json)?;
        // Opened ONCE for all five transformer-stack sub-models, and its
        // directory bound ONCE by `TcfSource` on top of that — a per-tensor
        // reparse would be quadratic over 577 names.
        let loader = TcfLoader::open(tcf_path.as_ref())?;
        check_embedded_vae_dense(&loader)?;
        let mut source = TcfSource::new(&loader)?;
        Self::from_source(
            &mut source,
            cfgs,
            audiovae_path,
            device,
            dtype,
            vae_decoder_dtype,
        )
    }

    /// Load the whole model from a TCF, materializing EVERY natively encoded
    /// weight to dense F32 instead of keeping it packed.
    ///
    /// The TCF half of the weight-encoding-only measurement mode — see
    /// [`from_gguf_dense`](Self::from_gguf_dense) for why it exists and
    /// [`DenseWeightSource`] for the activation-contract argument behind it.
    /// A cross-format quality comparison is valid only when both artifacts
    /// run the same activation contract, so both halves of a comparison load
    /// through the dense entry point or neither does.
    ///
    /// There is no `dtype` argument: the mode fixes F32.
    /// `vae_decoder_dtype` still casts the AudioVAE decoder independently —
    /// that codec is not part of the encoding-only measurement this mode
    /// exists for. The encoder always loads at F32. The AudioVAE itself is
    /// resolved exactly as [`from_tcf`](Self::from_tcf) resolves it.
    ///
    /// # Errors
    /// Every error [`from_tcf`](Self::from_tcf) raises.
    pub fn from_tcf_dense<P: AsRef<Path>, C: DequantOps<R>>(
        tcf_path: P,
        config_json: &Path,
        audiovae_path: Option<&Path>,
        device: &R::Device,
        client: &C,
        vae_decoder_dtype: Option<DType>,
    ) -> Result<Self> {
        let cfgs = StackConfigs::from_config_json(config_json)?;
        let loader = TcfLoader::open(tcf_path.as_ref())?;
        check_embedded_vae_dense(&loader)?;
        let mut source = TcfSource::new(&loader)?;
        Self::from_source(
            &mut DenseWeightSource::new(&mut source, client),
            cfgs,
            audiovae_path,
            device,
            Some(DType::F32),
            vae_decoder_dtype,
        )
    }
}

/// Refuse a TCF whose embedded AudioVAE holds a block-encoded tensor.
///
/// The TCF half of `gguf_loader`'s guard, on the same terms: only a file
/// carrying our probe tensor is checked, the directory alone decides, and
/// a raw encoding of any element type passes (the `sr_bin_boundaries` I32
/// vector is stored verbatim and never read).
fn check_embedded_vae_dense(loader: &TcfLoader) -> Result<()> {
    if loader.tensor_info(VAE_GGUF_PROBE_TENSOR).is_err() {
        return Ok(());
    }
    for info in loader
        .tensors()
        .iter()
        .filter(|t| t.name.starts_with(VAE_GGUF_ROOT))
    {
        if let Encoding::Block(_) = info.encoding() {
            return Err(packed_vae_tensor(
                &info.name,
                &encoding_name(info.encoding()),
            ));
        }
    }
    Ok(())
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
                Some(Path::new("/nonexistent/audiovae.safetensors")),
                &device,
                Some(DType::F32),
                None,
            )
            .is_err()
        );
    }

    /// The fixture file carries no `vae.*` tensor, so there is nothing to
    /// guard and the probe answers "not embedded".
    #[test]
    fn a_tcf_without_a_vae_passes_the_guard() {
        use crate::format::tcf::fixtures;
        let file = fixtures::write_temp(&fixtures::good_file());
        let loader = TcfLoader::open(file.path()).expect("opens");
        check_embedded_vae_dense(&loader).expect("no VAE to check");
        assert!(loader.tensor_info(VAE_GGUF_PROBE_TENSOR).is_err());
    }
}
