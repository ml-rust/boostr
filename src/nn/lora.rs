//! LoRA (Low-Rank Adaptation) layer.
//!
//! Adds a low-rank A*B decomposition to an existing linear layer:
//! output = base_linear(x) + (x @ A^T) @ B^T * scaling
//!
//! where A: [rank, in_features], B: [out_features, rank], scaling = alpha / rank.

use crate::error::Result;
use numr::autograd::{Var, var_add, var_matmul, var_mul_scalar, var_transpose};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use super::Linear;

/// LoRA adapter wrapping a frozen base Linear layer.
pub struct LoraLinear<R: Runtime> {
    /// Frozen base linear layer
    base: Linear<R>,
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
    /// - `base`: The frozen base linear layer
    /// - `rank`: Low-rank dimension (typical: 4, 8, 16, 32)
    /// - `alpha`: Scaling factor (typical: rank or 2*rank)
    /// - `device`: Device to allocate LoRA weights on
    pub fn new(base: Linear<R>, rank: usize, alpha: f32, device: &R::Device) -> Self {
        let in_features = base.weight().tensor().shape()[1];
        let out_features = base.weight().tensor().shape()[0];

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
            Tensor::from_slice(&a_data, &[rank, in_features], device),
            true,
        );
        let lora_b = Var::new(
            Tensor::zeros(&[out_features, rank], DType::F32, device),
            true,
        );

        Self {
            base,
            lora_a,
            lora_b,
            scaling: alpha / rank as f32,
        }
    }

    /// Create from pre-loaded LoRA weights.
    pub fn from_weights(base: Linear<R>, lora_a: Tensor<R>, lora_b: Tensor<R>, alpha: f32) -> Self {
        let rank = lora_a.shape()[0];
        Self {
            base,
            lora_a: Var::new(lora_a, false),
            lora_b: Var::new(lora_b, false),
            scaling: alpha / rank as f32,
        }
    }

    /// Forward: base(x) + (x @ A^T @ B^T) * scaling
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R> + TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
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
    pub fn base(&self) -> &Linear<R> {
        &self.base
    }

    /// Get LoRA rank.
    pub fn rank(&self) -> usize {
        self.lora_a.tensor().shape()[0]
    }

    /// Get scaling factor.
    pub fn scaling(&self) -> f32 {
        self.scaling
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_lora_linear_creation() {
        let device = <CpuRuntime as Runtime>::default_device();
        let weight: Tensor<CpuRuntime> = Tensor::zeros(&[64, 32], DType::F32, &device);
        let base = Linear::new(weight, None, false);
        let lora = LoraLinear::new(base, 8, 16.0, &device);
        assert_eq!(lora.rank(), 8);
        assert!((lora.scaling() - 2.0).abs() < 1e-6); // alpha/rank = 16/8 = 2
    }

    /// Gradients must reach the LoRA factors.
    ///
    /// Regression: the scale-and-add tail was built with `Var::new(...)` on raw
    /// tensors, producing a LEAF with no grad_fn. Backward then reached neither
    /// `lora_a`/`lora_b` nor the base, so EVERY LoRA adapter silently never
    /// trained — no error, no NaN, and the loss still falls because the rest of
    /// the network learns.
    #[test]
    fn test_lora_forward_propagates_gradient_to_factors() {
        use crate::test_utils::cpu_setup;
        use numr::autograd::{backward, var_sum};

        let (client, device) = cpu_setup();
        let (in_features, out_features, rank) = (4usize, 3usize, 2usize);

        // Asymmetric weights so a genuine zero gradient cannot pass by accident.
        let base_w: Vec<f32> = (0..out_features * in_features)
            .map(|i| (i as f32) * 0.1 - 0.5)
            .collect();
        let base = Linear::new(
            Tensor::<CpuRuntime>::from_slice(&base_w, &[out_features, in_features], &device),
            None,
            false,
        );
        let lora = LoraLinear::new(base, rank, 16.0, &device);

        let x_vals: Vec<f32> = (0..2 * in_features)
            .map(|i| (i as f32) * 0.25 - 0.75)
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_vals, &[2, in_features], &device),
            false,
        );

        let out = lora.forward(&client, &x).expect("lora forward");
        let loss = var_sum(&out, &[0, 1], false, &client).expect("reduce");
        let grads = backward(&loss, &client).expect("backward");

        // lora_b is zero-initialised, so d(loss)/d(lora_a) is zero at step 0 by
        // construction; lora_b is the factor that must receive signal immediately.
        let b_grad = grads
            .get(lora.lora_b.id())
            .expect("lora_b must receive a gradient");
        let b_vals: Vec<f32> = b_grad.contiguous().expect("contig").to_vec();
        let magnitude: f32 = b_vals.iter().map(|v| v.abs()).sum();
        assert!(
            magnitude > 1e-8,
            "lora_b gradient is all zeros ({magnitude}) — the LoRA graph is severed"
        );

        // And lora_a must at least be reachable in the graph.
        assert!(
            grads.get(lora.lora_a.id()).is_some(),
            "lora_a must be reachable from the loss"
        );
    }
}
