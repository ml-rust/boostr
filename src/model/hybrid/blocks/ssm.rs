//! The hybrid model's SSM block: pre-norm, Mamba2, residual.

use crate::error::{Error, Result};
use crate::inference::SsmState;
use crate::model::mamba::mamba2::{Mamba2, Mamba2Config};
use crate::model::traits::ModelClient;
use crate::nn::{RmsNorm, VarBuilder};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, IndexingOps, NormalizationOps, ReduceOps, ScalarOps,
    TensorOps, UnaryOps,
};
use numr::runtime::Runtime;

/// SSM block: pre-norm → Mamba2 → residual
pub(in crate::model::hybrid) struct SsmBlock<R: Runtime> {
    pub(in crate::model::hybrid) norm: RmsNorm<R>,
    pub(in crate::model::hybrid) mamba: Mamba2<R>,
}

/// Epsilon of the `norm` pre-norm oxidizr writes before each Mamba2 mixer.
/// The Hugging Face layout's `input_layernorm` takes `rms_norm_eps` instead.
pub const SSM_PRENORM_EPS: f32 = 1e-6;

impl<R: Runtime<DType = DType>> SsmBlock<R> {
    /// The SSM block of one layer, in either checkpoint layout:
    ///
    /// - oxidizr: `norm.weight` (eps [`SSM_PRENORM_EPS`]) and `mamba2.*`
    /// - Hugging Face: `input_layernorm.weight` (eps `rms_norm_eps`) and `mixer.*`
    ///
    /// # Errors
    ///
    /// A tensor the block reads is missing.
    pub(in crate::model::hybrid) fn load(
        layer_vb: &mut VarBuilder<R>,
        mamba_config: &Mamba2Config,
        rms_norm_eps: f32,
    ) -> Result<Self> {
        let (norm_name, eps, mixer) = if layer_vb.contains("mamba2.in_proj.weight") {
            ("norm.weight", SSM_PRENORM_EPS, "mamba2")
        } else {
            ("input_layernorm.weight", rms_norm_eps, "mixer")
        };
        let norm = RmsNorm::new(layer_vb.take_tensor(norm_name)?, eps, false);
        let mamba = Mamba2::from_varbuilder(mamba_config, &mut layer_vb.pp(mixer), false)?;
        Ok(Self { norm, mamba })
    }

    pub(in crate::model::hybrid) fn forward_inference<C>(
        &self,
        client: &C,
        x: &Var<R>,
        state: &mut SsmState<R>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + IndexingOps<R>,
    {
        self.forward_with_mixer(client, x, |normed| {
            let out = self
                .mamba
                .forward_inference(client, normed.tensor(), state)?;
            Ok(Var::new(out, false))
        })
    }

    /// The block around `mixer`: pre-norm, `mixer`, residual.
    ///
    /// `mixer` maps the normed `[B, S, hidden]` input to the Mamba2 output
    /// of the same shape. A batched decode step passes a mixer that runs
    /// each row over its own state.
    pub(in crate::model::hybrid) fn forward_with_mixer<C>(
        &self,
        client: &C,
        x: &Var<R>,
        mixer: impl FnOnce(&Var<R>) -> Result<Var<R>>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + IndexingOps<R>,
    {
        let normed = self.norm.forward(client, x)?;
        let out = mixer(&normed)?;
        numr::autograd::var_add(x, &out, client).map_err(Error::Numr)
    }
}
