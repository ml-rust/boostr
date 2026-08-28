//! Timestep-embedding primitives shared by every diffusion / flow-matching
//! model in this crate (currently VoxCPM2's local DiT `feat_decoder`).
//!
//! Two pieces, mirroring the reference (`voxcpm/modules/locdit/local_dit_v2.py:7-47`):
//! [`SinusoidalPosEmb`] turns a scalar-per-sample timestep into a log-spaced
//! sinusoidal embedding, and [`TimestepEmbedding`] is the small biased MLP
//! that projects that embedding through the model's hidden width.

use crate::error::{Error, Result};
use crate::nn::lora_targets::{
    LoraTargets, adapt_if_targeted, load_lora_child, push_projection_name,
};
use crate::nn::maybe_lora::MaybeLoraLinear;
use crate::nn::module::{Module, child_params, extend_named};
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{
    Var, var_cat, var_cos, var_mul, var_mul_scalar, var_reshape, var_silu, var_sin,
};
use numr::dtype::DType;
use numr::ops::{ActivationOps, BinaryOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Log-spaced sinusoidal position/timestep embedding.
///
/// `dim -> dim`: input `x: [batch]` (a scalar timestep per sample) produces
/// `[batch, dim]`. Reference (`local_dit_v2.py:7-22`):
///
/// ```text
/// half_dim = dim // 2
/// emb = log(10000) / (half_dim - 1)
/// emb = exp(arange(half_dim) * -emb)
/// emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
/// emb = cat((emb.sin(), emb.cos()), dim=-1)
/// ```
///
/// Three details are load-bearing:
/// - the divisor is `half_dim - 1`, NOT `half_dim` (511 for `dim=1024`, not 512)
/// - `scale` multiplies the input and defaults to 1000, not 1
/// - the halves concatenate as `(sin, cos)`, NOT `(cos, sin)`
///
/// The `[half_dim]` frequency table has no learned parameters; it is
/// precomputed once at construction (like `RoPE`'s cos/sin caches) rather
/// than recomputed every forward call.
pub struct SinusoidalPosEmb<R: Runtime> {
    /// `[half_dim]`, `freq[i] = exp(-i * log(10000) / (half_dim - 1))`.
    freq: Var<R>,
    dim: usize,
    scale: f32,
}

impl<R: Runtime<DType = DType>> SinusoidalPosEmb<R> {
    /// The reference's default `forward(x, scale=1000)` argument.
    pub const DEFAULT_SCALE: f32 = 1000.0;

    /// `dim` must be even and at least 4 (so `half_dim - 1 >= 1` and the
    /// divisor above is never zero). Uses [`Self::DEFAULT_SCALE`].
    pub fn new(dim: usize, device: &R::Device) -> Result<Self> {
        Self::with_scale(dim, Self::DEFAULT_SCALE, device)
    }

    /// As [`Self::new`], with an explicit `scale`.
    pub fn with_scale(dim: usize, scale: f32, device: &R::Device) -> Result<Self> {
        if dim == 0 || !dim.is_multiple_of(2) {
            return Err(Error::InvalidArgument {
                arg: "dim",
                reason: format!("SinusoidalPosEmb requires a nonzero, even dim, got {dim}"),
            });
        }
        let half_dim = dim / 2;
        if half_dim < 2 {
            return Err(Error::InvalidArgument {
                arg: "dim",
                reason: format!(
                    "SinusoidalPosEmb requires dim >= 4 (half_dim - 1 must be nonzero), got {dim}"
                ),
            });
        }
        let divisor = (half_dim - 1) as f32;
        let log_10000 = 10000f32.ln();
        let freq_data: Vec<f32> = (0..half_dim)
            .map(|i| (-(i as f32) * log_10000 / divisor).exp())
            .collect();
        let freq = Tensor::<R>::from_slice(&freq_data, &[half_dim], device)?;
        Ok(Self {
            freq: Var::new(freq, false),
            dim,
            scale,
        })
    }

    /// `x: [batch]` -> `[batch, dim]`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + ShapeOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ShapeOps<R>,
    {
        let shape = x.shape().to_vec();
        if shape.len() != 1 {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "SinusoidalPosEmb expects a rank-1 [batch] input, got {}D",
                    shape.len()
                ),
            });
        }
        let batch = shape[0];
        let half_dim = self.dim / 2;

        let x_scaled = var_mul_scalar(x, self.scale as f64, client).map_err(Error::Numr)?;
        let x_col = var_reshape(&x_scaled, &[batch, 1]).map_err(Error::Numr)?;
        let freq_row = var_reshape(&self.freq, &[1, half_dim]).map_err(Error::Numr)?;
        // Broadcast [batch, 1] * [1, half_dim] -> [batch, half_dim].
        let emb = var_mul(&x_col, &freq_row, client).map_err(Error::Numr)?;

        let sin_half = var_sin(&emb, client).map_err(Error::Numr)?;
        let cos_half = var_cos(&emb, client).map_err(Error::Numr)?;
        // (sin, cos) order — NOT (cos, sin).
        var_cat(&[&sin_half, &cos_half], -1, client).map_err(Error::Numr)
    }
}

/// `linear_2(silu(linear_1(x)))`, `dim -> dim -> dim`, both linears biased.
///
/// Reference: `TimestepEmbedding` (`local_dit_v2.py:25-47`).
///
/// Built on [`MaybeLoraLinear`] rather than plain `Linear`: its only caller
/// (VoxCPM2's local DiT) also ships as a GGUF, where these two weights are
/// block-quantized and multiply PACKED through `quant_matmul` instead of
/// being expanded to dense F32 at load. A safetensors checkpoint yields the
/// `Standard` variant and the dense path is unchanged. `MaybeLoraLinear`
/// additionally lets either linear carry a LoRA adapter.
pub struct TimestepEmbedding<R: Runtime> {
    linear_1: MaybeLoraLinear<R>,
    linear_2: MaybeLoraLinear<R>,
}

impl<R: Runtime<DType = DType>> TimestepEmbedding<R> {
    /// Build from loaded `linear_1`/`linear_2` weights. Both MUST carry a
    /// bias — the reference constructs both with `bias=True`.
    pub fn new(linear_1: MaybeLoraLinear<R>, linear_2: MaybeLoraLinear<R>) -> Self {
        Self { linear_1, linear_2 }
    }

    /// `x: [..., dim]` -> `[..., dim]`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        // `QuantMatmulOps` + `BinaryOps` + `TypeConversionOps` are what
        // `MaybeLoraLinear::forward` needs over a dense `Linear::forward`:
        // the packed multiply, its bias add, and the decomposed-quant arm's
        // cast of activations to F32.
        C: RuntimeClient<R>
            + TensorOps<R>
            + ActivationOps<R>
            + ScalarOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + ActivationOps<R> + ScalarOps<R> + BinaryOps<R> + DequantOps<R>,
    {
        let hidden = self.linear_1.forward(client, x)?;
        let hidden = var_silu(&hidden, client).map_err(Error::Numr)?;
        self.linear_2.forward(client, &hidden)
    }

    /// Wrap `linear_1`/`linear_2` that `targets` names with a fresh LoRA
    /// adapter each, returning how many were adapted. `prefix` is the
    /// dotted path the owning `local_dit` `time_mlp`/`delta_time_mlp` field
    /// would pass to `extend_named` for this MLP, so each linear's path
    /// (via [`LoraTargets::join`]) matches `named_parameters()`'s path
    /// exactly. A leaf step: no zero-match check here — see
    /// [`crate::model::audio::voxcpm::minicpm4::MiniCpm4Attention::apply_lora`]'s
    /// doc comment for why.
    pub fn apply_lora(
        &mut self,
        targets: &LoraTargets,
        rank: usize,
        alpha: f32,
        device: &R::Device,
        prefix: &str,
    ) -> Result<usize> {
        let mut adapted = adapt_if_targeted(
            &mut self.linear_1,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "linear_1",
        )?;
        adapted += adapt_if_targeted(
            &mut self.linear_2,
            targets,
            rank,
            alpha,
            device,
            prefix,
            "linear_2",
        )?;
        Ok(adapted)
    }

    /// Every dotted projection path [`Self::apply_lora`] would adapt under
    /// `prefix` — `linear_1`, `linear_2` — INDEPENDENT of whether either
    /// linear is dense, block-quantized, or decomposed-quantized. Unlike
    /// `named_parameters()`, this never enumerates empty for a quantized
    /// linear: which projections exist is a STRUCTURAL property of this
    /// type, not a function of whether its weights happen to carry a
    /// `Var<R>`. Built with the same [`crate::nn::push_projection_name`]
    /// helper `apply_lora`'s [`adapt_if_targeted`] calls use, so a path here
    /// is never hand-written separately from the one `apply_lora` matches.
    pub fn lora_projection_names(&self, prefix: &str) -> Vec<String> {
        let mut names = Vec::new();
        push_projection_name(&mut names, prefix, "linear_1");
        push_projection_name(&mut names, prefix, "linear_2");
        names
    }

    /// Write back updated `linear_1`/`linear_2` adapter values from an
    /// optimizer's `params` map, keeping their [`TensorId`]s. See
    /// [`crate::nn::MaybeLoraLinear::load_lora_parameters`] for the
    /// per-projection semantics. No prefix needed — unlike
    /// [`Self::apply_lora`], lookup is by ID.
    pub fn load_lora_parameters(
        &mut self,
        params: &std::collections::HashMap<TensorId, Tensor<R>>,
    ) -> Result<usize> {
        let mut written = load_lora_child(&mut self.linear_1, params, "linear_1")?;
        written += load_lora_child(&mut self.linear_2, params, "linear_2")?;
        Ok(written)
    }
}

/// Names ARE the field names (`linear_1`, `linear_2`) — the owning
/// `time_mlp`/`delta_time_mlp` checkpoint segment is added by the caller
/// (VoxCPM2's local DiT).
impl<R: Runtime<DType = DType>> Module<R> for TimestepEmbedding<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = child_params(&self.linear_1);
        params.extend(child_params(&self.linear_2));
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(&mut params, "linear_1", self.linear_1.named_parameters());
        extend_named(&mut params, "linear_2", self.linear_2.named_parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{MaybeQuantLinear, Weight};
    use numr::runtime::cpu::CpuRuntime;

    /// `dim = 8` -> `half_dim = 4`, divisor = `half_dim - 1 = 3`.
    /// `freq[i] = exp(-i * ln(10000) / 3)`:
    ///   freq[0] = exp(0)                     = 1.0
    ///   freq[1] = exp(-ln(10000)/3)          = 10000^(-1/3)  ≈ 0.0464158883
    ///   freq[2] = exp(-2*ln(10000)/3)        = 10000^(-2/3)  ≈ 0.0021544347
    ///   freq[3] = exp(-3*ln(10000)/3)        = 10000^(-1)    = 0.0001
    /// With `scale = 1000` (default) and `x = [1.0]`:
    ///   emb = 1000 * 1.0 * freq = [1000, 46.4158883, 2.1544347, 0.1]
    /// Output = [sin(emb), cos(emb)] (8 values), computed here from the same
    /// closed-form `f64::sin`/`cos` and compared bit-for-bit-close.
    #[test]
    fn matches_hand_computed_values_dim8() {
        let device = <CpuRuntime as Runtime>::default_device();
        let client = <CpuRuntime as Runtime>::default_client(&device);

        let pos_emb = SinusoidalPosEmb::<CpuRuntime>::new(8, &device).unwrap();
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap();
        let x = Var::new(x, false);

        let out = pos_emb.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[1, 8]);

        let out_data: Vec<f32> = out.tensor().contiguous().unwrap().to_vec();

        let divisor = 3.0f64;
        let log_10000 = 10000f64.ln();
        let freqs: Vec<f64> = (0..4)
            .map(|i| (-(i as f64) * log_10000 / divisor).exp())
            .collect();
        let scale = 1000.0f64;
        let emb: Vec<f64> = freqs.iter().map(|f| scale * 1.0 * f).collect();
        let mut expected = Vec::with_capacity(8);
        expected.extend(emb.iter().map(|e| e.sin() as f32));
        expected.extend(emb.iter().map(|e| e.cos() as f32));

        for (got, want) in out_data.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-4,
                "got {got}, want {want} (out={out_data:?}, expected={expected:?})"
            );
        }
    }

    /// The 511-vs-512 divisor trap: `dim = 1024` -> `half_dim = 512`, and the
    /// divisor MUST be `half_dim - 1 = 511`, not `half_dim = 512`. A test
    /// that passes with either divisor is worthless, so this pins `freq[1]`
    /// (the first frequency where the two divisors diverge) against the
    /// `511` closed form and asserts it does NOT match the `512` closed form.
    #[test]
    fn uses_half_dim_minus_one_divisor_not_half_dim() {
        let device = <CpuRuntime as Runtime>::default_device();
        let client = <CpuRuntime as Runtime>::default_client(&device);

        let pos_emb = SinusoidalPosEmb::<CpuRuntime>::new(1024, &device).unwrap();
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap();
        let x = Var::new(x, false);
        let out = pos_emb.forward(&client, &x).unwrap();
        let out_data: Vec<f32> = out.tensor().contiguous().unwrap().to_vec();

        let log_10000 = 10000f64.ln();
        let scale = 1000.0f64;

        // freq[1] under the correct divisor (511) and the wrong one (512).
        let freq1_correct = (-log_10000 / 511.0).exp();
        let freq1_wrong = (-log_10000 / 512.0).exp();
        let sin1_correct = (scale * freq1_correct).sin() as f32;
        let sin1_wrong = (scale * freq1_wrong).sin() as f32;

        // sin half occupies indices [0, half_dim) = [0, 512); index 1 is freq[1].
        let got = out_data[1];
        assert!(
            (got - sin1_correct).abs() < 1e-4,
            "expected the half_dim-1=511 divisor: got {got}, want {sin1_correct}"
        );
        assert!(
            (got - sin1_wrong).abs() > 1e-3,
            "divisor-511 and divisor-512 results should differ measurably, but got {got} \
             matches the wrong (512) divisor {sin1_wrong}"
        );
    }

    /// Concatenation order is `(sin, cos)`: the first `half_dim` outputs are
    /// `sin(emb)` and the second `half_dim` are `cos(emb)`, not the reverse.
    #[test]
    fn concatenates_sin_before_cos() {
        let device = <CpuRuntime as Runtime>::default_device();
        let client = <CpuRuntime as Runtime>::default_client(&device);

        let pos_emb = SinusoidalPosEmb::<CpuRuntime>::new(8, &device).unwrap();
        // x = 0 makes emb = 0 everywhere, so sin(0) = 0 and cos(0) = 1 —
        // an unambiguous fingerprint for which half is which.
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device).unwrap();
        let x = Var::new(x, false);
        let out = pos_emb.forward(&client, &x).unwrap();
        let out_data: Vec<f32> = out.tensor().contiguous().unwrap().to_vec();

        assert_eq!(&out_data[0..4], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&out_data[4..8], &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn rejects_odd_dim() {
        let device = <CpuRuntime as Runtime>::default_device();
        assert!(SinusoidalPosEmb::<CpuRuntime>::new(7, &device).is_err());
    }

    #[test]
    fn rejects_dim_below_four() {
        let device = <CpuRuntime as Runtime>::default_device();
        assert!(SinusoidalPosEmb::<CpuRuntime>::new(2, &device).is_err());
    }

    #[test]
    fn timestep_embedding_forward_shape() {
        let device = <CpuRuntime as Runtime>::default_device();
        let client = <CpuRuntime as Runtime>::default_client(&device);

        let w1 = Tensor::<CpuRuntime>::from_slice(&[0.01f32; 16], &[4, 4], &device).unwrap();
        let b1 = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device).unwrap();
        let w2 = Tensor::<CpuRuntime>::from_slice(&[0.02f32; 16], &[4, 4], &device).unwrap();
        let b2 = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device).unwrap();
        let mlp = TimestepEmbedding::<CpuRuntime>::new(
            MaybeQuantLinear::from_weight(Weight::Standard(w1), Some(b1)).into(),
            MaybeQuantLinear::from_weight(Weight::Standard(w2), Some(b2)).into(),
        );

        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 8], &[2, 4], &device).unwrap();
        let x = Var::new(x, false);
        let out = mlp.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
    }
}
