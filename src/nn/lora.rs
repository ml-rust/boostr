//! LoRA (Low-Rank Adaptation) layer.
//!
//! Adds a low-rank A*B decomposition to an existing linear layer:
//! output = base_linear(x) + (x @ A^T) @ B^T * scaling
//!
//! where A: [rank, in_features], B: [out_features, rank], scaling = alpha / rank.

use crate::error::{Error, Result};
use crate::nn::module::Module;
use crate::quant::traits::QuantMatmulOps;
use numr::autograd::{Var, var_add, var_matmul, var_mul_scalar, var_transpose};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

use super::{Linear, MaybeQuantLinear};

/// LoRA adapter wrapping a frozen base linear layer.
///
/// The base is [`MaybeQuantLinear`] rather than [`Linear`] because LoRA's
/// base is frozen by construction — only `lora_a`/`lora_b` train. A frozen
/// weight does not need to be dense: it can just as well be a block-quantized
/// GGUF weight (`MaybeQuantLinear::Quantized`) or a decomposed AWQ/GPTQ one.
/// A dense trainable adapter riding on a frozen quantized base is exactly
/// QLoRA, and it means fine-tuning can run directly on a quantized checkpoint
/// without ever dequantizing the base weights. `MaybeQuantLinear::Standard`
/// covers the plain dense case exactly, so this is a strict generalization —
/// no behavior change for existing dense-base callers.
pub struct LoraLinear<R: Runtime> {
    /// Frozen base linear layer — dense, block-quantized, or decomposed.
    base: MaybeQuantLinear<R>,
    /// Low-rank down-projection: [rank, in_features]
    lora_a: Var<R>,
    /// Low-rank up-projection: [out_features, rank]
    lora_b: Var<R>,
    /// Scaling factor: alpha / rank
    scaling: f32,
}

impl<R: Runtime<DType = DType>> LoraLinear<R> {
    /// Create a LoRA adapter around an existing linear layer.
    ///
    /// - `base`: The frozen base linear layer — a plain `Linear` converts in
    ///   automatically (dense case), or pass a `MaybeQuantLinear` directly
    ///   for a quantized base (QLoRA)
    /// - `rank`: Low-rank dimension (typical: 4, 8, 16, 32)
    /// - `alpha`: Scaling factor (typical: rank or 2*rank)
    /// - `device`: Device to allocate LoRA weights on
    pub fn new(
        base: impl Into<MaybeQuantLinear<R>>,
        rank: usize,
        alpha: f32,
        device: &R::Device,
    ) -> Result<Self> {
        let base = base.into();
        let shape = base.shape();
        let out_features = *shape.first().ok_or_else(|| Error::ModelError {
            reason: "LoRA base weight has no dimensions to derive out_features from".into(),
        })?;
        let in_features = *shape.get(1).ok_or_else(|| Error::ModelError {
            reason: "LoRA base weight must have at least 2 dimensions [out_features, in_features]"
                .into(),
        })?;

        // Initialize A with Kaiming uniform (simple LCG PRNG), B with zeros (standard LoRA init)
        let a_data = {
            let bound = (1.0 / in_features as f64).sqrt() as f32;
            let mut state: u64 = 42;
            let data: Vec<f32> = (0..rank * in_features)
                .map(|_| {
                    // Simple LCG for deterministic init
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = (state >> 33) as f32 / (1u64 << 31) as f32; // [0, 1)
                    (u * 2.0 - 1.0) * bound
                })
                .collect();
            data
        };

        let lora_a = Var::new(
            Tensor::from_slice(&a_data, &[rank, in_features], device)?,
            true,
        );
        let lora_b = Var::new(
            Tensor::zeros(&[out_features, rank], DType::F32, device)?,
            true,
        );

        Ok(Self {
            base,
            lora_a,
            lora_b,
            scaling: alpha / rank as f32,
        })
    }

    /// Create from pre-loaded LoRA weights.
    ///
    /// `trainable` controls gradient tracking on `lora_a`/`lora_b`: `false` for
    /// inference or merge-only use (the adapter is fixed), `true` to resume
    /// training this adapter further (e.g. continued fine-tuning after a
    /// checkpoint load). The base layer's own trainability is unaffected —
    /// pass it separately when constructing `base`.
    pub fn from_weights(
        base: impl Into<MaybeQuantLinear<R>>,
        lora_a: Tensor<R>,
        lora_b: Tensor<R>,
        alpha: f32,
        trainable: bool,
    ) -> Self {
        let rank = lora_a.shape()[0];
        Self {
            base: base.into(),
            lora_a: Var::new(lora_a, trainable),
            lora_b: Var::new(lora_b, trainable),
            scaling: alpha / rank as f32,
        }
    }

    /// Create from adapter tensors while preserving stable autograd IDs.
    ///
    /// The stable-id counterpart of [`Self::from_weights`], mirroring
    /// [`Linear::with_ids`]. `Tensor::clone` mints a fresh tensor id, so a
    /// rebuild that routes optimizer-updated tensors through `from_weights`
    /// would hand every adapter a NEW `TensorId` each step — detaching the
    /// optimizer state keyed by that id and turning a resumed LoRA run into a
    /// fresh, never-converging one. Callers that rebuild a model from a
    /// `TensorId`-keyed parameter map must use this.
    ///
    /// `trainable` applies to `lora_a`/`lora_b` only; the base carries its own
    /// flag from however `base` was constructed.
    pub fn with_ids(
        base: impl Into<MaybeQuantLinear<R>>,
        lora_a: Tensor<R>,
        lora_a_id: TensorId,
        lora_b: Tensor<R>,
        lora_b_id: TensorId,
        alpha: f32,
        trainable: bool,
    ) -> Self {
        let rank = lora_a.shape()[0];
        Self {
            base: base.into(),
            lora_a: Var::with_id(lora_a, lora_a_id, trainable),
            lora_b: Var::with_id(lora_b, lora_b_id, trainable),
            scaling: alpha / rank as f32,
        }
    }

    /// The low-rank down-projection factor `[rank, in_features]`.
    pub fn lora_a(&self) -> &Var<R> {
        &self.lora_a
    }

    /// The low-rank up-projection factor `[out_features, rank]`.
    pub fn lora_b(&self) -> &Var<R> {
        &self.lora_b
    }

    /// Forward: base(x) + (x @ A^T @ B^T) * scaling
    ///
    /// `base(x)` goes through [`MaybeQuantLinear::forward`], so this works
    /// identically whether the frozen base is dense, block-quantized, or
    /// decomposed — only the adapter path below ever needs a gradient.
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + QuantMatmulOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
    {
        let base_out = self.base.forward(client, input)?;

        // LoRA path: input @ A^T @ B^T * scaling
        let a_t = var_transpose(&self.lora_a).map_err(crate::error::Error::Numr)?;
        let lora_mid = var_matmul(input, &a_t, client).map_err(crate::error::Error::Numr)?;
        let b_t = var_transpose(&self.lora_b).map_err(crate::error::Error::Numr)?;
        let lora_out = var_matmul(&lora_mid, &b_t, client).map_err(crate::error::Error::Numr)?;

        // Scale and add — TRACKED.
        //
        // These must be `var_*` ops. Computing them on `.tensor()` and re-wrapping
        // with `Var::new` produces a LEAF with grad_fn = None, which severs the
        // graph: backward would reach neither `lora_a`/`lora_b` nor the base, so
        // every LoRA adapter would silently never train.
        let scaled = var_mul_scalar(&lora_out, self.scaling as f64, client)
            .map_err(crate::error::Error::Numr)?;
        let result = var_add(&base_out, &scaled, client).map_err(crate::error::Error::Numr)?;

        Ok(result)
    }

    /// Get reference to the base linear layer.
    pub fn base(&self) -> &MaybeQuantLinear<R> {
        &self.base
    }

    /// The base weight, if it is `Var`-wrapped — i.e. only when the base is
    /// dense (`MaybeQuantLinear::Standard`). A quantized base has no
    /// `Var<R>` weight: block-quantized storage carries nothing trainable,
    /// so `None` here signals "quantized base", not an error.
    pub fn weight(&self) -> Option<&Var<R>> {
        self.base.weight()
    }

    /// Get LoRA rank.
    pub fn rank(&self) -> usize {
        self.lora_a.tensor().shape()[0]
    }

    /// Get scaling factor.
    pub fn scaling(&self) -> f32 {
        self.scaling
    }

    /// Merge the adapter into the base weight, producing a plain `Linear`.
    ///
    /// Computes `W + scaling * (B @ A)`, matching the base weight layout
    /// `[out_features, in_features]` (`lora_b` is `[out, rank]`, `lora_a` is
    /// `[rank, in]`, so `B @ A` is `[out, in]`). The result carries no adapter
    /// and is not part of any gradient path — for export and inference after
    /// training. The base's bias, if any, is carried over unchanged.
    ///
    /// # Errors
    ///
    /// Only the dense (`Standard`) base can be merged. A quantized base
    /// (`Quantized` or `DecomposedQuant`) has no `Var<R>` weight to add the
    /// low-rank delta into — folding the adapter in would require
    /// requantizing the merged result, which this does not do. Keep the
    /// adapter separate (train and serve it alongside the quantized base)
    /// instead of merging.
    pub fn merge_into_base<C>(&self, client: &C) -> Result<Linear<R>>
    where
        C: RuntimeClient<R> + TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
        R::Client: TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
    {
        let base = match &self.base {
            MaybeQuantLinear::Standard(linear) => linear,
            MaybeQuantLinear::Quantized(_) | MaybeQuantLinear::DecomposedQuant(_) => {
                return Err(Error::ModelError {
                    reason: "cannot merge a LoRA adapter into a quantized base — merging would \
                             require requantizing the result; keep the adapter separate instead \
                             of merging"
                        .into(),
                });
            }
        };

        let ba = client
            .matmul(self.lora_b.tensor(), self.lora_a.tensor())
            .map_err(crate::error::Error::Numr)?;
        let scaled = client
            .mul_scalar(&ba, self.scaling as f64)
            .map_err(crate::error::Error::Numr)?;
        let merged_weight = client
            .add(base.weight().tensor(), &scaled)
            .map_err(crate::error::Error::Numr)?;

        let bias = base.bias().map(|b| b.tensor().clone());
        Ok(Linear::new(merged_weight, bias, false))
    }
}

impl<R: Runtime> Module<R> for LoraLinear<R> {
    // Enumerate the base as well as the adapters, and let the caller's
    // `requires_grad` filter decide what actually trains. Returning only the
    // adapters would be wrong in both directions: a QUANTIZED base already
    // contributes nothing here (`MaybeQuantLinear::parameters` is empty for
    // the quantized variants), so nothing needs suppressing for QLoRA; while
    // a DENSE base that a caller deliberately left trainable — oxidizr's
    // `lora.train_modules` opts a named projection's base back in even under
    // `freeze_base: true` — would silently vanish from the optimizer's
    // parameter set while still being checkpointed, so the weight would be
    // saved every step and never once updated.
    fn parameters(&self) -> Vec<&Var<R>> {
        // `Linear` also has an INHERENT `parameters()` returning `(TensorId, &Var)`
        // pairs, which shadows the trait method — disambiguate explicitly.
        let mut params = Module::parameters(&self.base);
        params.push(&self.lora_a);
        params.push(&self.lora_b);
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params: Vec<(String, &Var<R>)> = self
            .base
            .named_parameters()
            .into_iter()
            .map(|(name, var)| (format!("base.{name}"), var))
            .collect();
        params.push(("lora_a".to_string(), &self.lora_a));
        params.push(("lora_b".to_string(), &self.lora_b));
        params
    }
}

#[cfg(test)]
mod tests;
