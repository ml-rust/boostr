//! [`TensorLoader`]: the shape-checked, prefix-aware reader that walks a
//! checkpoint's key layout on behalf of the encoder/decoder loaders.

use super::weight_source::WeightSource;
use crate::error::{Error, Result};
use crate::model::audio::voxcpm::vae::causal_conv1d::CausalConv1d;
use crate::model::audio::voxcpm::vae::res_unit::ResUnit;
use crate::model::audio::voxcpm::vae::snake::Snake;
use crate::nn::{MaybeQuantEmbedding, MaybeQuantLinear, Weight};
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
    pub(crate) fn linear(
        &mut self,
        name: &str,
        out_features: usize,
        in_features: usize,
        with_bias: bool,
    ) -> Result<MaybeQuantLinear<R>> {
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

        Ok(MaybeQuantLinear::from_weight(weight, bias))
    }

    /// Read `{name}.weight` (`[vocab_size, hidden_size]`) as an embedding
    /// table that keeps a block-quantized weight PACKED.
    ///
    /// Same rule as [`Self::linear`]: a quantized weight fixes the output
    /// dtype at F32 (`QuantEmbedding::forward` dequantizes gathered rows to
    /// F32), so a non-F32 `self.dtype` request is an error, named, at load
    /// time rather than a silent downgrade of what the caller asked for.
    pub(crate) fn embedding(
        &mut self,
        name: &str,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Result<MaybeQuantEmbedding<R>> {
        let weight_key = format!("{name}.weight");
        let weight = checked_weight::<R, S>(
            self.loader,
            self.device,
            &self.prefix,
            &weight_key,
            &[vocab_size, hidden_size],
        )?;

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
