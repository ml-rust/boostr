//! [`GgmlNamedGguf`]: a [`WeightSource`] that rewrites our HuggingFace tensor
//! names into cstr's ggml-conventional spelling before reading.

use super::names::{hf_to_ggml_name, restore_leading_unit_dims};
use crate::error::{Error, Result};
use crate::format::gguf::Gguf;
use crate::model::audio::voxcpm::loader::support::WeightSource;
use crate::nn::Weight;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A VoxCPM2 GGUF whose tensors are named the llama.cpp way
/// (`tslm.blk.0.attn_q.weight`) rather than the checkpoint's own way
/// (`base_lm.layers.0.self_attn.q_proj.weight`).
///
/// It WRAPS a [`Gguf`] instead of changing it: the bare `Gguf` impl of
/// [`WeightSource`] stays the verbatim-name path that `compressr convert
/// --format gguf` output needs, and this type is layered on only when
/// [`probe_naming`](super::names::probe_naming) says the file needs it.
/// Neither path can drift into the other's naming that way.
///
/// # cstr's AudioVAE is NOT covered
///
/// cstr's file also carries 312 `vae.*` tensors, and this type maps none of
/// them. Their VAE uses a different scheme again (`vae.enc.conv0.*`,
/// `vae.enc.blk.0.res.0.0.alpha`) AND keeps `weight_norm` unfolded as
/// `weight_g`/`weight_v` pairs, which is where the extra 76 tensors over our
/// 236 come from. Reading it needs a second name map — the `weight_norm`
/// fold itself is [`crate::nn::fuse_weight_norm`], already used by the
/// `.pth` reader and callable from here. The whole-model loader probes for
/// OUR embedded spelling (`vae.decoder.model.0.weight`), finds nothing in
/// cstr's file, and falls back to the `audiovae_path` argument, which stays
/// required for this convention.
///
/// cstr's file also embeds no `config.json`, so the `config_json` path
/// argument is still required for it. Its `voxcpm2.*` metadata keys do carry
/// every config value, but reading config out of GGUF metadata is its own
/// unit.
pub(crate) struct GgmlNamedGguf {
    inner: Gguf,
}

impl GgmlNamedGguf {
    pub(crate) fn new(inner: Gguf) -> Self {
        Self { inner }
    }

    /// Rewrite `name`, or fail naming it.
    ///
    /// An unmapped name is an ERROR rather than a pass-through, with one
    /// exception: a name the file holds VERBATIM is read as spelled. That
    /// is what keeps [`has_named`](WeightSource::has_named) and
    /// `load_named` in agreement for a tensor outside the map — an embedded
    /// `vae.*` tensor written under our own names, for instance. Passing
    /// every unmapped name through would instead look up a HuggingFace key
    /// in a file that has none, and the resulting "tensor not found" would
    /// blame the file instead of this map, which is the thing that is
    /// actually incomplete.
    fn translate(&self, name: &str) -> Result<String> {
        if let Some(mapped) = hf_to_ggml_name(name) {
            return Ok(mapped);
        }
        if self.inner.tensor_info(name).is_ok() {
            return Ok(name.to_string());
        }
        Err(Error::ModelError {
            reason: format!(
                "{name}: no ggml-conventional counterpart is mapped for this VoxCPM2 \
                 tensor; this GGUF uses llama.cpp-style names and the name map does \
                 not cover this key"
            ),
        })
    }

    /// The key `name` reads from: its mapped spelling when the map covers
    /// it, else the name itself.
    fn key_for(&self, name: &str) -> String {
        hf_to_ggml_name(name).unwrap_or_else(|| name.to_string())
    }
}

impl<R: Runtime<DType = DType>> WeightSource<R> for GgmlNamedGguf {
    /// Dense read, translated. Dequantizes exactly like the bare [`Gguf`]
    /// impl does, then restores any leading unit dims their writer squeezed
    /// off — see [`restore_leading_unit_dims`] for why that is one named
    /// tensor and not a general same-element-count fallback.
    fn load_named(&mut self, name: &str, device: &R::Device) -> Result<Tensor<R>> {
        let mapped = self.translate(name)?;
        let tensor = self.inner.load_tensor_f32::<R>(&mapped, device)?;
        match restore_leading_unit_dims(name, tensor.shape()) {
            Some(shape) => Ok(tensor.reshape(&shape)?),
            None => Ok(tensor),
        }
    }

    /// Packed-if-available read, translated.
    ///
    /// The squeeze fix is applied to the dense variant here too, for
    /// consistency with `load_named`. It never fires on the packed variant:
    /// a K-quant block tensor is a matmul weight, and no matmul weight is in
    /// the squeeze table.
    fn load_named_weight(&mut self, name: &str, device: &R::Device) -> Result<Weight<R>> {
        let mapped = self.translate(name)?;
        let quantized = self
            .inner
            .tensor_info(&mapped)?
            .ggml_type
            .to_quant_format()
            .is_some();
        if quantized {
            return Ok(Weight::Quantized(
                self.inner.load_tensor_quantized::<R>(&mapped, device)?,
            ));
        }
        Ok(Weight::Standard(WeightSource::<R>::load_named(
            self, name, device,
        )?))
    }

    /// Answers for the key `load_named` would read: the mapped spelling for
    /// a transformer-stack name, the verbatim one for anything the map does
    /// not cover. cstr's own `vae.*` tensors use a third scheme that is
    /// neither, so a probe for an embedded AudioVAE under OUR names comes
    /// back `false` on that file and the loader falls back to the separate
    /// checkpoint.
    fn has_named(&self, name: &str) -> bool {
        self.inner.tensor_info(&self.key_for(name)).is_ok()
    }
}
