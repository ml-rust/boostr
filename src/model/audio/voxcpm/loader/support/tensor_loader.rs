//! [`TensorLoader`]: the shape-checked, prefix-aware reader that walks a
//! checkpoint's key layout on behalf of the encoder/decoder loaders.

use super::WeightSource;
use crate::error::{Error, Result};
use crate::model::audio::voxcpm::vae::causal_conv1d::CausalConv1d;
use crate::model::audio::voxcpm::vae::res_unit::ResUnit;
use crate::model::audio::voxcpm::vae::snake::Snake;
use crate::nn::{MaybeLoraLinear, MaybeQuantEmbedding, MaybeQuantLinear, Weight};
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Spell the checkpoint key for `name` under `prefix`.
///
/// A trailing `.` on `prefix` is absorbed, and an empty `prefix` reads
/// `name` at the checkpoint root, so callers can pass either spelling.
fn full_name(prefix: &str, name: &str) -> String {
    let prefix = prefix.trim_end_matches('.');
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Load `{prefix}.{name}` as a DENSE tensor and verify its shape matches
/// `expected`.
pub(crate) fn checked_tensor<R: Runtime<DType = DType>, S: WeightSource<R>>(
    loader: &mut S,
    device: &R::Device,
    prefix: &str,
    name: &str,
    expected: &[usize],
) -> Result<Tensor<R>> {
    let full = full_name(prefix, name);
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

/// Load `{prefix}.{name}` in its most compact form and verify its shape
/// matches `expected`.
///
/// [`Weight::shape`] is the LOGICAL element shape for every variant, so a
/// quantized weight is gated by exactly the same check a dense one is — a
/// packed weight must never be the one that skips validation.
///
/// Used only by [`TensorLoader::linear`] and [`TensorLoader::embedding`]
/// below, so it stays private here.
fn checked_weight<R: Runtime<DType = DType>, S: WeightSource<R>>(
    loader: &mut S,
    device: &R::Device,
    prefix: &str,
    name: &str,
    expected: &[usize],
) -> Result<Weight<R>> {
    let full = full_name(prefix, name);
    let w = loader.load_named_weight(&full, device)?;
    if w.shape() != expected {
        return Err(Error::ModelError {
            reason: format!(
                "{full}: expected shape {expected:?}, checkpoint has {:?}",
                w.shape()
            ),
        });
    }
    Ok(w)
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

    /// Read `{name}.weight` (`[out_features, in_features]`) and, when
    /// `with_bias`, `{name}.bias` (`[out_features]`), as a linear layer that
    /// keeps a block-quantized weight PACKED.
    ///
    /// `[out, in]` is the order both the safetensors checkpoint and
    /// `quant_matmul`'s `[N, K]` contract use, so nothing is transposed on
    /// either path.
    ///
    /// Returns [`MaybeLoraLinear`], not a bare [`MaybeQuantLinear`], so every
    /// projection loaded through this single funnel can later carry a LoRA
    /// adapter; a freshly loaded projection is always the unadapted `Plain`
    /// variant, wrapped at the very end via `.into()`.
    pub(crate) fn linear(
        &mut self,
        name: &str,
        out_features: usize,
        in_features: usize,
        with_bias: bool,
    ) -> Result<MaybeLoraLinear<R>> {
        let weight_key = format!("{name}.weight");
        let weight = checked_weight::<R, S>(
            self.loader,
            self.device,
            &self.prefix,
            &weight_key,
            &[out_features, in_features],
        )?;

        // A quantized weight fixes the arithmetic dtype: `quant_matmul`
        // requires F32 activations and emits F32, so a BF16/F16 request
        // cannot be honoured. Dequantizing to obey it would silently undo
        // the whole point of the quantized path, and ignoring it would
        // silently run a different dtype than the caller asked for — so it
        // is an error, named, at load time rather than a kernel-level
        // surprise mid-forward.
        let weight = match weight {
            Weight::Standard(t) => Weight::Standard(match self.dtype {
                Some(want) => t.to_dtype(want)?,
                None => t,
            }),
            packed => {
                if let Some(want) = self.dtype
                    && want != DType::F32
                {
                    return Err(Error::ModelError {
                        reason: format!(
                            "{}: requested dtype {want:?}, but the checkpoint stores this \
                             weight quantized and quant_matmul requires F32 activations; \
                             load this model with dtype F32 or None",
                            full_name(&self.prefix, &weight_key)
                        ),
                    });
                }
                packed
            }
        };

        let bias = if with_bias {
            let b = checked_tensor::<R, S>(
                self.loader,
                self.device,
                &self.prefix,
                &format!("{name}.bias"),
                &[out_features],
            )?;
            // `QuantLinear::forward` adds the bias straight onto
            // `quant_matmul`'s F32 output, which errors on a dtype mismatch
            // rather than promoting — so a quantized weight forces the bias
            // to F32 regardless of `self.dtype`.
            Some(if weight.is_quantized() {
                b.to_dtype(DType::F32)?
            } else {
                match self.dtype {
                    Some(want) => b.to_dtype(want)?,
                    None => b,
                }
            })
        } else {
            None
        };

        Ok(MaybeQuantLinear::from_weight(weight, bias).into())
    }

    /// Read `{name}.weight` (`[vocab_size, hidden_size]`) as an embedding
    /// table that keeps a block-quantized weight PACKED.
    ///
    /// Same rule as [`Self::linear`]: a quantized weight fixes the output
    /// dtype at F32 (`QuantEmbedding::forward` dequantizes gathered rows to
    /// F32), so a non-F32 `self.dtype` request is an error, named, at load
    /// time rather than a silent downgrade of what the caller asked for.
    ///
    /// A GGML block tensor is row-blocked, so `QuantTensor::gather_rows`
    /// slices the bytes a row occupies and the table never needs a dense
    /// copy.
    pub(crate) fn embedding(
        &mut self,
        name: &str,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<MaybeQuantEmbedding<R>> {
        let weight_key = format!("{name}.weight");
        let shape = [vocab_size, hidden_size];
        let weight =
            checked_weight::<R, S>(self.loader, self.device, &self.prefix, &weight_key, &shape)?;

        let weight = match weight {
            Weight::Standard(t) => Weight::Standard(match self.dtype {
                Some(want) => t.to_dtype(want)?,
                None => t,
            }),
            packed => {
                if let Some(want) = self.dtype
                    && want != DType::F32
                {
                    return Err(Error::ModelError {
                        reason: format!(
                            "{}: requested dtype {want:?}, but the checkpoint stores this \
                             embedding table quantized and QuantEmbedding::forward \
                             dequantizes gathered rows to F32; load this model with \
                             dtype F32 or None",
                            full_name(&self.prefix, &weight_key)
                        ),
                    });
                }
                packed
            }
        };

        MaybeQuantEmbedding::from_weight(weight, false)
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
    use crate::format::safetensors_loader::SafeTensorsLoader;
    use crate::quant::{QuantFormat, QuantTensor};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
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

    /// A source whose every weight is block-quantized, standing in for a
    /// GGUF. Written by hand rather than through a real GGUF file because
    /// the gates under test are `TensorLoader::linear`'s, not the reader's.
    struct QuantSource {
        shape: Vec<usize>,
    }

    impl WeightSource<CpuRuntime> for QuantSource {
        fn load_named(&mut self, name: &str, _device: &CpuDevice) -> Result<Tensor<CpuRuntime>> {
            Err(Error::ModelError {
                reason: format!("{name}: dense read not expected in this test"),
            })
        }

        fn load_named_weight(
            &mut self,
            _name: &str,
            device: &CpuDevice,
        ) -> Result<Weight<CpuRuntime>> {
            let numel: usize = self.shape.iter().product();
            // Q4_0: 32 elements per 18-byte block. Values are irrelevant —
            // nothing here multiplies.
            let bytes = vec![0u8; numel / 32 * 18];
            Ok(Weight::Quantized(QuantTensor::<CpuRuntime>::from_bytes(
                &bytes,
                QuantFormat::Q4_0,
                &self.shape,
                device,
            )?))
        }
    }

    /// The shape gate is not something a packed weight gets to skip:
    /// `Weight::shape` is the logical element shape for both variants.
    #[test]
    fn quantized_weight_is_shape_checked() {
        let (_, device) = cpu_setup();
        let mut source = QuantSource { shape: vec![2, 32] };
        let mut tl = TensorLoader::<CpuRuntime, _> {
            loader: &mut source,
            device: &device,
            prefix: "base_lm.layers.0.self_attn".to_string(),
            dtype: None,
        };

        let ok = tl.linear("q_proj", 2, 32, false);
        assert!(ok.is_ok(), "matching shape rejected: {:?}", ok.err());

        // `MaybeLoraLinear` is not `Debug`, so `expect_err` cannot be used.
        let Err(err) = tl.linear("q_proj", 4, 32, false) else {
            panic!("wrong out_features accepted");
        };
        let msg = err.to_string();
        assert!(msg.contains("expected shape [4, 32]"), "got {msg}");
        assert!(msg.contains("[2, 32]"), "got {msg}");
    }

    /// BF16/F16 plus a quantized weight cannot run: `quant_matmul` requires
    /// F32 activations. The load errors and names the dtype rather than
    /// dequantizing behind the caller's back or dropping the request.
    #[test]
    fn quantized_weight_rejects_a_narrow_dtype() {
        let (_, device) = cpu_setup();
        for want in [DType::BF16, DType::F16] {
            let mut source = QuantSource { shape: vec![2, 32] };
            let mut tl = TensorLoader::<CpuRuntime, _> {
                loader: &mut source,
                device: &device,
                prefix: "base_lm.layers.0.self_attn".to_string(),
                dtype: Some(want),
            };

            let Err(err) = tl.linear("q_proj", 2, 32, false) else {
                panic!("a narrow dtype was accepted alongside a quantized weight");
            };
            let msg = err.to_string();
            assert!(msg.contains(&format!("{want:?}")), "got {msg}");
            assert!(msg.contains("F32"), "got {msg}");
            assert!(msg.contains("q_proj.weight"), "got {msg}");
        }
    }

    /// F32 (and `None`) keep the weight packed — the whole point of the path.
    #[test]
    fn quantized_weight_survives_an_f32_request() {
        let (_, device) = cpu_setup();
        for dtype in [None, Some(DType::F32)] {
            let mut source = QuantSource { shape: vec![2, 32] };
            let mut tl = TensorLoader::<CpuRuntime, _> {
                loader: &mut source,
                device: &device,
                prefix: "base_lm".to_string(),
                dtype,
            };

            let layer = tl.linear("q_proj", 2, 32, false).expect("linear");
            assert!(
                matches!(
                    layer,
                    MaybeLoraLinear::Plain(MaybeQuantLinear::Quantized(_))
                ),
                "the weight was dequantized instead of staying packed"
            );
        }
    }
}
