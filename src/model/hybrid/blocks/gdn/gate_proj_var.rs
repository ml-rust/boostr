//! Tracked forward of the [`GateProjections`] pair, for training.
//!
//! The shapes equal those of the detached `forward`. A fused pair splits
//! its `[batch, seq, 2 * heads]` output with two tracked narrows, so the
//! gradient of each half reaches `x` and any adapter factors. The fused
//! base weight is frozen; see `params`. The importance-matrix tap runs on
//! the detached forward only.

use super::gate_proj::{GateProjections, SeparateGates};
use crate::error::{Error, Result};
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_narrow};
use numr::dtype::DType;
use numr::ops::{BinaryOps, MatmulOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime<DType = DType>> GateProjections<R> {
    /// `(alpha_raw, beta_raw)`, each `[batch, seq, heads]` and tracked, for
    /// `x` `[batch, seq, hidden_size]`. On the fused path both narrow one
    /// `[batch, seq, 2 * heads]` result; `beta_raw` starts at column
    /// `heads`.
    ///
    /// # Errors
    ///
    /// numr errors from the projections and narrows propagate.
    pub(super) fn forward_var<C>(&self, client: &C, x: &Var<R>) -> Result<(Var<R>, Var<R>)>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + DequantOps<R> + MatmulOps<R> + ShapeOps<R>,
    {
        match self {
            Self::Fused { linear, heads, .. } => {
                let both = linear.forward(client, x)?;
                split_heads_var(&both, *heads)
            }
            Self::FusedAdapted { lora, heads, .. } => {
                let both = lora.forward(client, x)?;
                split_heads_var(&both, *heads)
            }
            Self::Separate(gates) => {
                let SeparateGates { alpha, beta } = gates.as_ref();
                Ok((alpha.forward(client, x)?, beta.forward(client, x)?))
            }
        }
    }
}

/// `(alpha_raw, beta_raw)`: tracked narrows of the first and second
/// `heads` columns of the last dim.
fn split_heads_var<R>(both: &Var<R>, heads: usize) -> Result<(Var<R>, Var<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + ShapeOps<R>,
{
    let alpha = var_narrow(both, -1, 0, heads).map_err(Error::Numr)?;
    let beta = var_narrow(both, -1, heads, heads).map_err(Error::Numr)?;
    Ok((alpha, beta))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{Linear, LoraFactors, MaybeQuantLinear};
    use crate::test_utils::cpu_setup;
    use numr::autograd::{backward, var_add, var_mul, var_sum};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    const HEADS: usize = 4;
    const HIDDEN: usize = 8;
    const RANK: usize = 2;

    fn tensor(seed: u64, shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let mut state = seed;
        let data: Vec<f32> = (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
            })
            .collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
    }

    fn dense(w: &Tensor<CpuRuntime>, trainable: bool) -> MaybeQuantLinear<CpuRuntime> {
        MaybeQuantLinear::Standard(Linear::new(w.clone(), None, trainable))
    }

    fn separate(
        wa: &Tensor<CpuRuntime>,
        wb: &Tensor<CpuRuntime>,
        trainable: bool,
    ) -> GateProjections<CpuRuntime> {
        GateProjections::Separate(Box::new(SeparateGates {
            alpha: dense(wa, trainable).into(),
            beta: dense(wb, trainable).into(),
        }))
    }

    /// `sum(alpha * c_a) + sum(beta * c_b)`, so the two halves get
    /// different upstream gradients.
    fn loss(
        client: &CpuClient,
        alpha: &Var<CpuRuntime>,
        beta: &Var<CpuRuntime>,
        device: &CpuDevice,
    ) -> Var<CpuRuntime> {
        let shape = alpha.shape().to_vec();
        let ca = Var::new(tensor(50, &shape, device), false);
        let cb = Var::new(tensor(51, &shape, device), false);
        let a = var_sum(
            &var_mul(alpha, &ca, client).unwrap(),
            &[0, 1, 2],
            false,
            client,
        )
        .unwrap();
        let b = var_sum(
            &var_mul(beta, &cb, client).unwrap(),
            &[0, 1, 2],
            false,
            client,
        )
        .unwrap();
        var_add(&a, &b, client).unwrap()
    }

    fn assert_close(got: &Tensor<CpuRuntime>, want: &Tensor<CpuRuntime>) {
        let got = got.contiguous().unwrap().to_vec::<f32>();
        let want = want.contiguous().unwrap().to_vec::<f32>();
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-5, "element {i}: {g} vs {w}");
        }
    }

    /// Fused and separate pairs over the same weights give the detached
    /// values and the same gradient to `x`. Separate trainable weights get
    /// a gradient; the fused base weight does not.
    #[test]
    fn fused_and_separate_give_gradients_to_the_input() {
        let (client, device) = cpu_setup();
        let wa = tensor(1, &[HEADS, HIDDEN], &device);
        let wb = tensor(2, &[HEADS, HIDDEN], &device);
        let x_data = tensor(3, &[2, 3, HIDDEN], &device);

        let fused = GateProjections::new(dense(&wa, false), dense(&wb, false)).unwrap();
        assert!(matches!(fused, GateProjections::Fused { .. }));
        let x_fused = Var::new(x_data.clone(), true);
        let (alpha_f, beta_f) = fused.forward_var(&client, &x_fused).unwrap();
        assert_eq!(alpha_f.shape(), &[2, 3, HEADS]);
        assert_eq!(beta_f.shape(), &[2, 3, HEADS]);
        let (alpha_d, beta_d) = fused.forward(&client, &x_fused).unwrap();
        assert_close(alpha_f.tensor(), &alpha_d);
        assert_close(beta_f.tensor(), &beta_d);
        let grads_f = backward(&loss(&client, &alpha_f, &beta_f, &device), &client).unwrap();
        let dx_fused = grads_f.get(x_fused.id()).unwrap();
        let GateProjections::Fused { linear, .. } = &fused else {
            panic!("dense pair must fuse");
        };
        assert!(grads_f.get(linear.weight().id()).is_none());

        let sep = separate(&wa, &wb, true);
        let x_sep = Var::new(x_data, true);
        let (alpha_s, beta_s) = sep.forward_var(&client, &x_sep).unwrap();
        assert_eq!(alpha_s.shape(), &[2, 3, HEADS]);
        assert_close(alpha_s.tensor(), alpha_f.tensor());
        assert_close(beta_s.tensor(), beta_f.tensor());
        let grads_s = backward(&loss(&client, &alpha_s, &beta_s, &device), &client).unwrap();
        assert_close(grads_s.get(x_sep.id()).unwrap(), dx_fused);
        let GateProjections::Separate(gates) = &sep else {
            panic!("built separate");
        };
        for gate in [&gates.alpha, &gates.beta] {
            let (id, _) = gate.trainable_parameters()[0];
            assert!(grads_s.get(id).is_some(), "a trainable gate weight trains");
        }
    }

    /// A trainable block-diagonal adapter on the fused pair gets gradients
    /// on both factors, and `x` still gets one.
    #[test]
    fn fused_adapter_factors_get_gradients() {
        let (client, device) = cpu_setup();
        let wa = tensor(4, &[HEADS, HIDDEN], &device);
        let wb = tensor(5, &[HEADS, HIDDEN], &device);
        let factors = LoraFactors {
            lora_a: tensor(6, &[RANK, HIDDEN], &device),
            lora_b: tensor(7, &[HEADS, RANK], &device),
            scaling: 1.5,
        };
        let adapted = GateProjections::new(dense(&wa, false), dense(&wb, false))
            .unwrap()
            .with_adapters(&client, Some(factors), None, DType::F32, true)
            .unwrap();
        let GateProjections::FusedAdapted { lora, .. } = &adapted else {
            panic!("dense pair must fuse");
        };
        let x = Var::new(tensor(8, &[1, 5, HIDDEN], &device), true);
        let (alpha, beta) = adapted.forward_var(&client, &x).unwrap();
        let grads = backward(&loss(&client, &alpha, &beta, &device), &client).unwrap();
        assert!(grads.get(x.id()).is_some());
        assert!(grads.get(lora.lora_a().id()).is_some());
        assert!(grads.get(lora.lora_b().id()).is_some());
    }
}
