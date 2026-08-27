//! Shared tensor-fetch and sub-module-assembly helpers for the VoxCPM2
//! encoder/decoder loaders.
//!
//! Same idiom as `neucodec/loader/support.rs` (not reused directly: that
//! module's helper is private to its own `loader` submodule).

use crate::error::{Error, Result};
use crate::format::gguf::Gguf;
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::voxcpm::vae::causal_conv1d::CausalConv1d;
use crate::model::audio::voxcpm::vae::res_unit::ResUnit;
use crate::model::audio::voxcpm::vae::snake::Snake;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// A checkpoint a VoxCPM2 sub-loader can read a named tensor out of.
///
/// The file format is an I/O detail: every sub-loader below knows the
/// *layout* (which key holds which weight, and what shape it must have),
/// which is identical whether the bytes arrive from safetensors or from a
/// GGUF file. Keeping the read behind this trait is what stops that layout
/// knowledge from being duplicated once per format.
///
/// Public because the sub-model `from_source` constructors take it as a
/// bound, and those are the API a caller uses to load several sub-models
/// out of ONE open checkpoint instead of reopening it per sub-model.
pub trait WeightSource<R: Runtime<DType = DType>> {
    /// Load the tensor stored under exactly `name`.
    fn load_named(&mut self, name: &str, device: &R::Device) -> Result<Tensor<R>>;
}

impl<R: Runtime<DType = DType>> WeightSource<R> for SafeTensorsLoader {
    fn load_named(&mut self, name: &str, device: &R::Device) -> Result<Tensor<R>> {
        self.load_tensor::<R>(name, device)
    }
}

impl<R: Runtime<DType = DType>> WeightSource<R> for Gguf {
    /// DEQUANTIZES. `load_tensor_f32` expands a K-quant block tensor to a
    /// dense F32 tensor, so a 1.2 GB Q4_K VoxCPM2 file becomes the same
    /// full-size resident weight set a BF16 safetensors checkpoint produces
    /// (larger, in fact, until the loader's `dtype` cast narrows it). What a
    /// GGUF buys here is a smaller file and single-file distribution — NOT a
    /// smaller memory footprint.
    ///
    /// Keeping the weights quantized in memory needs `QuantTensor` plus a
    /// `MaybeQuantLinear` that dispatches to `QuantMatmulOps`, which is a
    /// LATER unit. This impl is the file-format path, not the finished
    /// quantized-inference path.
    ///
    /// The tensor names are the checkpoint's ORIGINAL HuggingFace names:
    /// `compressr convert --format gguf` writes them verbatim, so
    /// `gguf_to_hf_name` is deliberately not applied.
    fn load_named(&mut self, name: &str, device: &R::Device) -> Result<Tensor<R>> {
        self.load_tensor_f32::<R>(name, device)
    }
}

/// Load `{prefix}.{name}` and verify its shape matches `expected`.
///
/// A trailing `.` on `prefix` is absorbed, and an empty `prefix` reads
/// `name` at the checkpoint root, so callers can pass either spelling.
pub(crate) fn checked_tensor<R: Runtime<DType = DType>, S: WeightSource<R>>(
    loader: &mut S,
    device: &R::Device,
    prefix: &str,
    name: &str,
    expected: &[usize],
) -> Result<Tensor<R>> {
    let prefix = prefix.trim_end_matches('.');
    let full = if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    };
    let t = loader.load_named(&full, device)?;
    if t.shape() != expected {
        return Err(Error::ModelError {
            reason: format!(
                "{full}: expected shape {expected:?}, checkpoint has {:?}",
                t.shape()
            ),
        });
    }
    Ok(t)
}

/// Checkpoint-tensor reader shared by the encoder and decoder loaders: both
/// walk the same `Snake -> depthwise CausalConv1d -> Snake -> pointwise
/// CausalConv1d` `ResUnit` layout, just under different key prefixes and
/// kernel-size constants, so that walk lives here once. `encoder.rs` and
/// `decoder.rs` each add their own inherent `impl` (block/front/head
/// assembly) on this same type for their block-specific layout.
///
/// `S` is the checkpoint the weights come from — see [`WeightSource`].
pub(crate) struct TensorLoader<'a, R: Runtime<DType = DType>, S: WeightSource<R>> {
    pub(crate) loader: &'a mut S,
    pub(crate) device: &'a R::Device,
    pub(crate) prefix: String,
    /// Cast every tensor this loader reads to this dtype. `None` keeps the
    /// checkpoint's own (the AudioVAE encoder/decoder construction sites
    /// pass `None`: that model is F32-native, verified to 5e-07 / 2.4e-05
    /// against PyTorch fixtures, and must not be cast).
    pub(crate) dtype: Option<DType>,
}

impl<R: Runtime<DType = DType>, S: WeightSource<R>> TensorLoader<'_, R, S>
where
    R::Client: TypeConversionOps<R>,
{
    pub(crate) fn tensor(&mut self, name: &str, expected: &[usize]) -> Result<Tensor<R>> {
        let t = checked_tensor::<R, S>(self.loader, self.device, &self.prefix, name, expected)?;
        // VoxCPM2 ships BF16 weights; the AudioVAE ships F32. A forward pass
        // mixing the two errors rather than promoting, so the caller states
        // which dtype it wants and the cast happens once, here.
        match self.dtype {
            // `to_dtype` is a no-op clone when the dtypes already agree and
            // makes a strided safetensors view contiguous itself, so neither
            // needs handling here.
            Some(want) => Ok(t.to_dtype(want)?),
            None => Ok(t),
        }
    }

    pub(crate) fn snake(&mut self, name: &str, channels: usize) -> Result<Snake<R>> {
        let alpha = self.tensor(&format!("{name}.alpha"), &[1, channels, 1])?;
        Snake::new(alpha)
    }

    /// Depthwise causal conv: `[channels, 1, kernel]`.
    pub(crate) fn depthwise_conv(
        &mut self,
        name: &str,
        channels: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<CausalConv1d<R>> {
        let weight = self.tensor(&format!("{name}.weight"), &[channels, 1, kernel])?;
        let bias = self.tensor(&format!("{name}.bias"), &[channels])?;
        CausalConv1d::new(weight, Some(bias), kernel, dilation, channels)
    }

    /// Pointwise (`k=1`, `groups=1`) causal conv: `[out, in, 1]`.
    pub(crate) fn pointwise_conv(
        &mut self,
        name: &str,
        in_c: usize,
        out_c: usize,
    ) -> Result<CausalConv1d<R>> {
        let weight = self.tensor(&format!("{name}.weight"), &[out_c, in_c, 1])?;
        let bias = self.tensor(&format!("{name}.bias"), &[out_c])?;
        CausalConv1d::new(weight, Some(bias), 1, 1, 1)
    }

    pub(crate) fn res_unit(
        &mut self,
        name: &str,
        dim: usize,
        kernel: usize,
        dilation: usize,
    ) -> Result<ResUnit<R>> {
        let snake1 = self.snake(&format!("{name}.block.0"), dim)?;
        let dilated_conv =
            self.depthwise_conv(&format!("{name}.block.1"), dim, kernel, dilation)?;
        let snake2 = self.snake(&format!("{name}.block.2"), dim)?;
        let pointwise_conv = self.pointwise_conv(&format!("{name}.block.3"), dim, dim)?;
        Ok(ResUnit::new(snake1, dilated_conv, snake2, pointwise_conv))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// One `weight` tensor of shape `[2, 3]`, values `1.0 ..= 6.0`.
    fn one_tensor_file() -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("temp file");
        let header = serde_json::json!({
            "weight": { "dtype": "F32", "shape": [2, 3], "data_offsets": [0, 24] }
        })
        .to_string();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("header len");
        file.write_all(header.as_bytes()).expect("header");
        for f in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            file.write_all(&f.to_le_bytes()).expect("data");
        }
        file.flush().expect("flush");
        file
    }

    /// `SafeTensorsLoader` reaches the same bytes through the trait as it
    /// does through its own `load_tensor`, and the shape gate still fires.
    #[test]
    fn safetensors_source_loads_named_tensor() {
        let (_, device) = cpu_setup();
        let file = one_tensor_file();
        let mut loader = SafeTensorsLoader::open(file.path()).expect("open");

        let t: Tensor<CpuRuntime> = loader.load_named("weight", &device).expect("load_named");
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec::<f32>(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let ok = checked_tensor::<CpuRuntime, _>(&mut loader, &device, "", "weight", &[2, 3]);
        assert!(ok.is_ok());
        let wrong_shape = checked_tensor::<CpuRuntime, _>(&mut loader, &device, "", "weight", &[6]);
        assert!(wrong_shape.is_err());
    }
}
