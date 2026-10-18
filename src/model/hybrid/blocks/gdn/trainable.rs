//! The trainable switch of a [`GdnBlock`] and the decay `ssm_a` each
//! forward reads.
//!
//! A frozen block multiplies the loaded `ssm_a = -exp(A_log)` in as is, so
//! inference stays bit-identical to the checkpoint. Training moves the
//! decay to `a_log = ln(-ssm_a)`, and every forward then reads
//! `-exp(a_log)`. An optimizer step on `a_log` therefore reaches inference
//! too, and no cached copy goes stale.

use super::layer::GdnBlock;
use crate::error::{Error, Result};
use crate::nn::GatedRmsNorm;
use numr::autograd::{Var, var_exp, var_neg};
use numr::dtype::DType;
use numr::ops::{TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> GdnBlock<R> {
    /// Set `requires_grad` on the conv kernel, the decay, `ssm_dt_bias` and
    /// the gated norm weight. Every `TensorId` stays.
    ///
    /// The first call with `trainable = true` creates `a_log = ln(-ssm_a)`
    /// in the stored dtype. From then on the decay is `-exp(a_log)`, which
    /// equals the loaded `ssm_a` up to float rounding. `trainable = false`
    /// on a block without `a_log` changes nothing.
    ///
    /// The projections keep the `requires_grad` they were built with. A
    /// LoRA adapter trains when attached with `trainable = true`; see
    /// [`Self::attach_adapter_trainable`] and
    /// [`Self::attach_gate_adapters_trainable`].
    ///
    /// # Errors
    ///
    /// numr errors from the `ln(-ssm_a)` ops propagate.
    pub fn with_trainable<C: UnaryOps<R>>(mut self, client: &C, trainable: bool) -> Result<Self> {
        if trainable && self.a_log.is_none() {
            let a_log = a_log_from_ssm_a(client, self.ssm_a.tensor())?;
            self.a_log = Some(Var::new(a_log, true));
        }
        if let Some(a_log) = self.a_log.as_mut() {
            a_log.set_requires_grad(trainable);
        }
        self.conv_weight.set_requires_grad(trainable);
        self.ssm_dt_bias.set_requires_grad(trainable);
        let weight = self.norm.weight();
        self.norm = GatedRmsNorm::with_id(
            weight.tensor().clone(),
            weight.id(),
            self.norm.eps(),
            self.norm.order(),
            trainable,
        );
        Ok(self)
    }

    /// `true` when the decay, conv kernel, `ssm_dt_bias` and norm weight
    /// require grad.
    pub fn is_trainable(&self) -> bool {
        self.conv_weight.requires_grad()
    }

    /// The decay `[value_heads]` for the detached forward: the loaded
    /// `ssm_a`, or `-exp(a_log)` once `a_log` exists.
    pub(super) fn decay<C: UnaryOps<R>>(&self, client: &C) -> Result<Tensor<R>> {
        match &self.a_log {
            None => Ok(self.ssm_a.tensor().clone()),
            Some(a_log) => {
                let a = client.exp(a_log.tensor()).map_err(Error::Numr)?;
                client.neg(&a).map_err(Error::Numr)
            }
        }
    }

    /// The decay as a tracked `Var`: `-exp(a_log)` with its gradient to
    /// `a_log`, or a frozen alias of `ssm_a` when `a_log` is absent.
    pub(super) fn decay_var<C>(&self, client: &C) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + TensorOps<R>,
        R::Client: TensorOps<R>,
    {
        match &self.a_log {
            None => Ok(self.ssm_a.alias()),
            Some(a_log) => {
                let a = var_exp(a_log, client).map_err(Error::Numr)?;
                var_neg(&a, client).map_err(Error::Numr)
            }
        }
    }
}

/// `ln(-ssm_a)`, in `ssm_a`'s dtype. `ssm_a` holds `-exp(A_log)`, so the
/// result is `A_log` up to float rounding.
fn a_log_from_ssm_a<R, C>(client: &C, ssm_a: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: UnaryOps<R>,
{
    let positive = client.neg(ssm_a).map_err(Error::Numr)?;
    client.log(&positive).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::super::forward::tests::block;
    use crate::test_utils::cpu_setup;
    use numr::autograd::{backward, var_sum};

    /// A frozen block reads the loaded bits; `with_trainable` keeps them
    /// up to rounding and routes the gradient to `a_log`.
    #[test]
    fn decay_moves_to_a_log_when_trainable() {
        let (client, device) = cpu_setup();
        let frozen = block(&device, 0x6d4e_0010, false);
        assert!(!frozen.is_trainable());
        let loaded = frozen.ssm_a.tensor().to_vec::<f32>();
        assert_eq!(frozen.decay(&client).unwrap().to_vec::<f32>(), loaded);
        assert!(frozen.a_log.is_none());

        let trained = frozen.with_trainable(&client, true).unwrap();
        assert!(trained.is_trainable());
        assert!(trained.norm.weight().requires_grad());
        assert!(trained.ssm_dt_bias.requires_grad());
        let a_log = trained.a_log.as_ref().unwrap();
        assert!(a_log.requires_grad());
        for (got, want) in a_log.tensor().to_vec::<f32>().iter().zip(&loaded) {
            assert!((got - (-want).ln()).abs() < 1e-5, "{got} vs ln(-{want})");
        }
        let decay = trained.decay(&client).unwrap().to_vec::<f32>();
        for (got, want) in decay.iter().zip(&loaded) {
            assert!((got - want).abs() < 1e-5 * want.abs(), "{got} vs {want}");
        }

        let decay = trained.decay_var(&client).unwrap();
        let loss = var_sum(&decay, &[0], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        // d/d(a_log) of -exp(a_log) is the decay itself.
        let grad = grads.get(a_log.id()).unwrap().to_vec::<f32>();
        for (g, d) in grad.iter().zip(decay.tensor().to_vec::<f32>()) {
            assert!((g - d).abs() < 1e-6, "{g} vs {d}");
        }
    }

    /// Turning training off again keeps `a_log` and every id.
    #[test]
    fn with_trainable_false_keeps_ids() {
        let (client, device) = cpu_setup();
        let trained = block(&device, 0x6d4e_0011, false)
            .with_trainable(&client, true)
            .unwrap();
        let ids = (
            trained.conv_weight.id(),
            trained.a_log.as_ref().unwrap().id(),
            trained.norm.weight().id(),
        );
        let frozen = trained.with_trainable(&client, false).unwrap();
        assert!(!frozen.is_trainable());
        let a_log = frozen.a_log.as_ref().unwrap();
        assert!(!a_log.requires_grad());
        assert_eq!(
            ids,
            (
                frozen.conv_weight.id(),
                a_log.id(),
                frozen.norm.weight().id()
            )
        );
    }
}
