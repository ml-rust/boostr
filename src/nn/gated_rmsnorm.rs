//! Gated RMS normalization module
//!
//! RMSNorm fused with a SiLU gate. Two architectures order the two steps
//! differently:
//!
//! - Mamba2: `norm(x * silu(z))` — gate first, then normalize.
//! - Gated DeltaNet (Qwen3-Next): `silu(z) * norm(x)` — normalize first,
//!   then gate. Matches `build_norm_gated` in llama.cpp.
//!
//! Mamba2 still gates inline in `model/mamba/mamba2/forward.rs` (step 11)
//! and `model/mamba/mamba2/inference.rs` (step 11); a later unit switches
//! those to `GateOrder::MulThenNorm`.

use crate::error::{Error, Result};
use crate::nn::module::Module;
use numr::autograd::{Var, var_mul, var_rms_norm, var_silu, var_silu_mul};
use numr::dtype::DType;
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Where the SiLU gate is applied relative to the norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateOrder {
    /// `norm(x * silu(z))`. Mamba2.
    MulThenNorm,
    /// `silu(z) * norm(x)`. Gated DeltaNet.
    NormThenMul,
}

/// RMSNorm with a SiLU gate.
///
/// `x` and `z` are `[.., H, D]`; `weight` is `[D]` and broadcasts over
/// every leading dim, so each head is normalized on its own.
pub struct GatedRmsNorm<R: Runtime> {
    weight: Var<R>,
    eps: f32,
    order: GateOrder,
}

impl<R: Runtime> GatedRmsNorm<R> {
    /// Create a new gated norm.
    pub fn new(weight: Tensor<R>, eps: f32, order: GateOrder, trainable: bool) -> Self {
        Self {
            weight: Var::new(weight, trainable),
            eps,
            order,
        }
    }

    /// Create from a tensor while preserving its stable autograd ID.
    pub fn with_id(
        weight: Tensor<R>,
        weight_id: TensorId,
        eps: f32,
        order: GateOrder,
        trainable: bool,
    ) -> Self {
        Self {
            weight: Var::with_id(weight, weight_id, trainable),
            eps,
            order,
        }
    }

    /// Forward over `[.., D]` input `x` and gate `z` of the same shape.
    ///
    /// Both orders run one fused kernel for the gate and one for the norm.
    pub fn forward<C>(&self, client: &C, x: &Var<R>, z: &Var<R>) -> Result<Var<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + CompareOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
    {
        if x.shape() != z.shape() {
            return Err(Error::InvalidArgument {
                arg: "z",
                reason: format!("gate shape {:?} != input shape {:?}", z.shape(), x.shape()),
            });
        }
        match self.order {
            GateOrder::MulThenNorm => {
                let gate = var_silu(z, client).map_err(Error::Numr)?;
                let gated = var_mul(x, &gate, client).map_err(Error::Numr)?;
                var_rms_norm(&gated, &self.weight, self.eps, client).map_err(Error::Numr)
            }
            GateOrder::NormThenMul => {
                let normed =
                    var_rms_norm(x, &self.weight, self.eps, client).map_err(Error::Numr)?;
                var_silu_mul(z, &normed, client).map_err(Error::Numr)
            }
        }
    }

    /// Gate order.
    pub fn order(&self) -> GateOrder {
        self.order
    }

    /// Epsilon under the root.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Get the weight parameter
    pub fn weight(&self) -> &Var<R> {
        &self.weight
    }

    /// All parameters with their stable autograd IDs.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        vec![(self.weight.id(), &self.weight)]
    }

    /// Cheap duplicate that preserves `weight`'s `TensorId`. Uses
    /// [`Var::alias`], not [`Clone`]: a `clone` would mint a fresh id and
    /// orphan `weight`'s gradient.
    pub fn alias(&self) -> Self {
        Self {
            weight: self.weight.alias(),
            eps: self.eps,
            order: self.order,
        }
    }
}

impl<R: Runtime> Module<R> for GatedRmsNorm<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        vec![self.weight()]
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        vec![("weight".to_string(), self.weight())]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    const EPS: f32 = 1e-6;

    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    fn rms_norm(row: &[f32], w: &[f32]) -> Vec<f32> {
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let inv = 1.0 / (mean_sq + EPS).sqrt();
        row.iter().zip(w).map(|(v, w)| v * inv * w).collect()
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            assert!((a - b).abs() < 1e-5, "idx={i}: got {a}, want {b}");
        }
    }

    /// `[1, 2, 4]`: two heads of width 4, weight `[4]` shared per head.
    fn inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let x = vec![1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0];
        let z = vec![0.5f32, -1.0, 2.0, 0.0, 1.5, -0.5, -2.0, 3.0];
        let w = vec![1.0f32, 0.5, 2.0, -1.0];
        (x, z, w)
    }

    #[test]
    fn norm_then_mul_matches_hand() {
        let (client, device) = cpu_setup();
        let (x, z, w) = inputs();
        let norm = GatedRmsNorm::new(
            Tensor::<CpuRuntime>::from_slice(&w, &[4], &device).unwrap(),
            EPS,
            GateOrder::NormThenMul,
            false,
        );
        let xv = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x, &[1, 2, 4], &device).unwrap(),
            false,
        );
        let zv = Var::new(
            Tensor::<CpuRuntime>::from_slice(&z, &[1, 2, 4], &device).unwrap(),
            false,
        );
        let out = norm.forward(&client, &xv, &zv).unwrap();
        assert_eq!(out.shape(), &[1, 2, 4]);

        let mut want = Vec::new();
        for h in 0..2 {
            let normed = rms_norm(&x[h * 4..h * 4 + 4], &w);
            for d in 0..4 {
                want.push(silu(z[h * 4 + d]) * normed[d]);
            }
        }
        assert_close(&out.tensor().to_vec::<f32>(), &want);
    }

    #[test]
    fn mul_then_norm_matches_hand() {
        let (client, device) = cpu_setup();
        let (x, z, w) = inputs();
        let norm = GatedRmsNorm::new(
            Tensor::<CpuRuntime>::from_slice(&w, &[4], &device).unwrap(),
            EPS,
            GateOrder::MulThenNorm,
            false,
        );
        let xv = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x, &[1, 2, 4], &device).unwrap(),
            false,
        );
        let zv = Var::new(
            Tensor::<CpuRuntime>::from_slice(&z, &[1, 2, 4], &device).unwrap(),
            false,
        );
        let out = norm.forward(&client, &xv, &zv).unwrap();

        let mut want = Vec::new();
        for h in 0..2 {
            let gated: Vec<f32> = (0..4).map(|d| x[h * 4 + d] * silu(z[h * 4 + d])).collect();
            want.extend(rms_norm(&gated, &w));
        }
        assert_close(&out.tensor().to_vec::<f32>(), &want);
    }

    #[test]
    fn rejects_shape_mismatch() {
        let (client, device) = cpu_setup();
        let (x, _, w) = inputs();
        let norm = GatedRmsNorm::new(
            Tensor::<CpuRuntime>::from_slice(&w, &[4], &device).unwrap(),
            EPS,
            GateOrder::NormThenMul,
            false,
        );
        let xv = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x, &[1, 2, 4], &device).unwrap(),
            false,
        );
        let zv = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x[..4], &[1, 1, 4], &device).unwrap(),
            false,
        );
        assert!(norm.forward(&client, &xv, &zv).is_err());
    }
}
