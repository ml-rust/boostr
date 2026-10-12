//! Unmerged LoRA adapters on the `ssm_alpha` and `ssm_beta` projections
//! of a [`GdnBlock`](super::GdnBlock): one block-diagonal adapter over the
//! fused dense pair, or one adapter per projection kept apart.

use super::gate_proj::{GateProjections, SeparateGates};
use crate::error::{Error, Result};
use crate::nn::{FusedLoraPart, LoraFactors, LoraLinear, MaybeLoraLinear, MaybeQuantLinear};
use numr::dtype::DType;
use numr::ops::{ScalarOps, ShapeOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime<DType = DType>> GateProjections<R> {
    /// These projections with unmerged adapters: `alpha` on `ssm_alpha`,
    /// `beta` on `ssm_beta`, each `[rank, hidden_size]` / `[heads, rank]`
    /// in the block's value-head order, or `None`. The factors are cast
    /// with [`LoraFactors::cast_for`]; `activation_dtype` is the dtype `x`
    /// arrives in. A fused pair becomes `FusedAdapted` through
    /// [`LoraLinear::fused_block_diagonal`]; a separate pair adapts each
    /// projection on its own. `self` is unchanged; the caller swaps the
    /// result in.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when both are `None`, a targeted projection
    /// already carries an adapter, or a factor shape is wrong; numr errors
    /// propagate.
    pub(super) fn with_adapters<C>(
        &self,
        client: &C,
        alpha: Option<LoraFactors<R>>,
        beta: Option<LoraFactors<R>>,
        activation_dtype: DType,
    ) -> Result<Self>
    where
        C: RuntimeClient<R> + ScalarOps<R> + ShapeOps<R> + TypeConversionOps<R>,
    {
        if alpha.is_none() && beta.is_none() {
            return Err(Error::ModelError {
                reason: "gdn gates: no adapter for ssm_alpha or ssm_beta".to_string(),
            });
        }
        match self {
            Self::Fused {
                linear,
                heads,
                alpha_id,
                beta_id,
            } => {
                let base = MaybeQuantLinear::Standard(linear.alias());
                let mut parts = Vec::with_capacity(2);
                for factors in [alpha, beta] {
                    let factors = factors
                        .map(|f| f.cast_for(client, &base, activation_dtype))
                        .transpose()?;
                    parts.push(FusedLoraPart {
                        out_features: *heads,
                        factors,
                    });
                }
                let lora = LoraLinear::fused_block_diagonal(client, base, parts, false)?;
                Ok(Self::FusedAdapted {
                    lora: Box::new(lora),
                    heads: *heads,
                    alpha_id: *alpha_id,
                    beta_id: *beta_id,
                })
            }
            Self::FusedAdapted { .. } => Err(Error::ModelError {
                reason: "gdn gates: ssm_alpha/ssm_beta already carry an adapter".to_string(),
            }),
            Self::Separate(gates) => Ok(Self::Separate(Box::new(SeparateGates {
                alpha: adapt_one(client, &gates.alpha, alpha, activation_dtype, "ssm_alpha")?,
                beta: adapt_one(client, &gates.beta, beta, activation_dtype, "ssm_beta")?,
            }))),
        }
    }
}

/// `gate` with `factors` attached, or an alias of it for `None`.
fn adapt_one<R, C>(
    client: &C,
    gate: &MaybeLoraLinear<R>,
    factors: Option<LoraFactors<R>>,
    activation_dtype: DType,
    name: &str,
) -> Result<MaybeLoraLinear<R>>
where
    R: Runtime<DType = DType>,
    C: TypeConversionOps<R>,
{
    let Some(factors) = factors else {
        return Ok(gate.alias());
    };
    let MaybeLoraLinear::Plain(base) = gate else {
        return Err(Error::ModelError {
            reason: format!("gdn gates: {name} already carries an adapter"),
        });
    };
    let factors = factors.cast_for(client, base, activation_dtype)?;
    let lora = LoraLinear::from_scaled_weights(
        base.alias(),
        factors.lora_a,
        factors.lora_b,
        factors.scaling,
        false,
    )?;
    Ok(MaybeLoraLinear::Lora(Box::new(lora)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Linear;
    use crate::test_utils::cpu_setup;
    use numr::autograd::Var;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    const HEADS: usize = 4;
    const HIDDEN: usize = 8;

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

    fn dense(w: Tensor<CpuRuntime>) -> MaybeQuantLinear<CpuRuntime> {
        MaybeQuantLinear::Standard(Linear::new(w, None, false))
    }

    const RANK: usize = 2;

    fn factors(seed: u64, scaling: f32, device: &CpuDevice) -> LoraFactors<CpuRuntime> {
        LoraFactors {
            lora_a: tensor(seed, &[RANK, HIDDEN], device),
            lora_b: tensor(seed + 1, &[HEADS, RANK], device),
            scaling,
        }
    }

    /// `x (W + s B A)^T` with plain numr ops, `[.., HEADS]`.
    fn merged_out(
        client: &CpuClient,
        x: &Tensor<CpuRuntime>,
        w: &Tensor<CpuRuntime>,
        f: Option<&LoraFactors<CpuRuntime>>,
    ) -> Vec<f32> {
        use numr::ops::{BinaryOps, MatmulOps};
        let w = match f {
            Some(f) => {
                let ba = client.matmul(&f.lora_b, &f.lora_a).unwrap();
                let delta = client.mul_scalar(&ba, f64::from(f.scaling)).unwrap();
                client.add(w, &delta).unwrap()
            }
            None => w.clone(),
        };
        let w_t = w.t().unwrap().contiguous().unwrap();
        client.matmul(x, &w_t).unwrap().to_vec()
    }

    fn assert_close(got: &Tensor<CpuRuntime>, want: &[f32]) {
        let got = got.contiguous().unwrap().to_vec::<f32>();
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < 1e-5, "element {i}: {g} vs {w}");
        }
    }

    /// Fused and separate pairs with an adapter on one or both
    /// projections equal the merged weights.
    #[test]
    fn adapters_equal_merged_weights_fused_and_separate() {
        use crate::quant::QuantFormat;
        use crate::quant::traits::QuantizeOps;
        let (client, device) = cpu_setup();
        let wa = tensor(11, &[HEADS, HIDDEN], &device);
        let wb = tensor(12, &[HEADS, HIDDEN], &device);
        let x2 = tensor(13, &[5, HIDDEN], &device);
        let x = Var::new(x2.reshape(&[1, 5, HIDDEN]).unwrap(), false);
        for (alpha, beta) in [(true, false), (false, true), (true, true)] {
            let fa = alpha.then(|| factors(20, 1.5, &device));
            let fb = beta.then(|| factors(30, 0.75, &device));
            let want_a = merged_out(&client, &x2, &wa, fa.as_ref());
            let want_b = merged_out(&client, &x2, &wb, fb.as_ref());

            let fused = GateProjections::new(dense(wa.clone()), dense(wb.clone()))
                .unwrap()
                .with_adapters(&client, fa, fb, DType::F32)
                .unwrap();
            assert!(matches!(fused, GateProjections::FusedAdapted { .. }));
            let (a, b) = fused.forward(&client, &x).unwrap();
            assert_close(&a, &want_a);
            assert_close(&b, &want_b);
            assert!(
                fused
                    .with_adapters(&client, Some(factors(40, 1.0, &device)), None, DType::F32)
                    .is_err(),
                "a second adapter is refused"
            );
        }

        // Block-quantized alpha keeps the pair apart; the adapter adds its
        // delta to the quantized base's output.
        let wq = tensor(14, &[HEADS, 32], &device);
        let quant = client.quantize(&wq, QuantFormat::Q8_0).unwrap();
        let qa = MaybeQuantLinear::Quantized(crate::nn::QuantLinear::new(quant, None));
        let wb32 = tensor(15, &[HEADS, 32], &device);
        let separate = GateProjections::new(qa, dense(wb32.clone())).unwrap();
        assert!(matches!(separate, GateProjections::Separate(_)));
        let x32 = Var::new(tensor(16, &[1, 3, 32], &device), false);
        let (base_a, _) = separate.forward(&client, &x32).unwrap();
        let fb = LoraFactors {
            lora_a: tensor(17, &[RANK, 32], &device),
            lora_b: tensor(18, &[HEADS, RANK], &device),
            scaling: 2.0,
        };
        let want_b = merged_out(
            &client,
            &x32.tensor().reshape(&[3, 32]).unwrap(),
            &wb32,
            Some(&fb),
        );
        let adapted = separate
            .with_adapters(&client, None, Some(fb), DType::F32)
            .unwrap();
        let (a, b) = adapted.forward(&client, &x32).unwrap();
        assert_eq!(
            a.contiguous().unwrap().to_vec::<f32>(),
            base_a.contiguous().unwrap().to_vec::<f32>(),
            "the unadapted projection is unchanged"
        );
        assert_close(&b, &want_b);
        assert!(
            separate
                .with_adapters(&client, None, None, DType::F32)
                .is_err()
        );
    }
}
