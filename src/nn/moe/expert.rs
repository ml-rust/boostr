//! MoE Expert — individual SwiGLU MLP

use crate::error::{Error, Result};
use crate::nn::linear::Linear;
use crate::nn::maybe_lora::MaybeLoraLinear;
use crate::nn::module::Module;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_mul, var_silu};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::{Tensor, TensorId};

/// Single expert MLP (SwiGLU architecture).
///
/// Architecture: `down_proj(silu(gate_proj(x)) * up_proj(x))`
///
/// Each projection is a [`MaybeLoraLinear`], so an expert can be LoRA-adapted
/// per projection without a separate expert type.
pub struct Expert<R: Runtime> {
    gate_proj: MaybeLoraLinear<R>,
    up_proj: MaybeLoraLinear<R>,
    down_proj: MaybeLoraLinear<R>,
}

impl<R: Runtime<DType = DType>> Expert<R> {
    /// Create from plain linear projections.
    pub fn new(gate_proj: Linear<R>, up_proj: Linear<R>, down_proj: Linear<R>) -> Self {
        Self {
            gate_proj: gate_proj.into(),
            up_proj: up_proj.into(),
            down_proj: down_proj.into(),
        }
    }

    /// Create from projections that may each carry a LoRA adapter.
    pub fn new_adapted(
        gate_proj: MaybeLoraLinear<R>,
        up_proj: MaybeLoraLinear<R>,
        down_proj: MaybeLoraLinear<R>,
    ) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Create from tensors. Expects:
    /// - gate_proj: `[intermediate, hidden]`
    /// - up_proj: `[intermediate, hidden]`
    /// - down_proj: `[hidden, intermediate]`
    pub fn from_tensors(
        gate_proj: Tensor<R>,
        up_proj: Tensor<R>,
        down_proj: Tensor<R>,
        trainable: bool,
    ) -> Self {
        Self::new(
            Linear::new(gate_proj, None, trainable),
            Linear::new(up_proj, None, trainable),
            Linear::new(down_proj, None, trainable),
        )
    }

    /// SwiGLU forward: `down_proj(silu(gate_proj(x)) * up_proj(x))`
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        R: Runtime<DType = numr::dtype::DType>,
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + QuantMatmulOps<R>
            + TypeConversionOps<R>,
        R::Client: TensorOps<R> + ActivationOps<R> + ScalarOps<R> + BinaryOps<R> + DequantOps<R>,
    {
        let gate = self.gate_proj.forward(client, x)?;
        let up = self.up_proj.forward(client, x)?;

        let gate_silu = var_silu(&gate, client).map_err(Error::Numr)?;
        let hidden = var_mul(&gate_silu, &up, client).map_err(Error::Numr)?;
        self.down_proj.forward(client, &hidden)
    }

    /// Fold every adapter into its base weight, producing a plain expert.
    ///
    /// Mirrors [`LoraLinear::merge_into_base`](crate::nn::LoraLinear::merge_into_base)
    /// at expert granularity — for export and inference after training.
    pub fn merge_adapters<C>(&self, client: &C) -> Result<Self>
    where
        C: RuntimeClient<R> + TensorOps<R> + BinaryOps<R> + ScalarOps<R>,
        R::Client: TensorOps<R> + BinaryOps<R> + ScalarOps<R> + DequantOps<R>,
    {
        Ok(Self::new(
            self.gate_proj.merge_into_base(client)?,
            self.up_proj.merge_into_base(client)?,
            self.down_proj.merge_into_base(client)?,
        ))
    }

    pub fn gate_proj(&self) -> &MaybeLoraLinear<R> {
        &self.gate_proj
    }

    pub fn up_proj(&self) -> &MaybeLoraLinear<R> {
        &self.up_proj
    }

    pub fn down_proj(&self) -> &MaybeLoraLinear<R> {
        &self.down_proj
    }

    /// All parameters with their stable autograd IDs, adapters included.
    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        let mut params = self.gate_proj.parameters();
        params.extend(self.up_proj.parameters());
        params.extend(self.down_proj.parameters());
        params
    }
}

impl<R: Runtime<DType = DType>> Module<R> for Expert<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        Expert::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        for (prefix, proj) in [
            ("gate_proj", &self.gate_proj),
            ("up_proj", &self.up_proj),
            ("down_proj", &self.down_proj),
        ] {
            params.extend(
                proj.named_parameters()
                    .into_iter()
                    .map(|(name, var)| (format!("{prefix}.{name}"), var)),
            );
        }
        params
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        Expert::parameters(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::lora::LoraLinear;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    const HIDDEN: usize = 4;
    const INTER: usize = 6;
    const RANK: usize = 2;
    const ALPHA: f32 = 4.0;
    const TOKENS: usize = 2;

    /// Asymmetric, non-zero values so an accidental zero cannot make a test pass.
    fn vals(count: usize, offset: f32) -> Vec<f32> {
        (0..count)
            .map(|i| (i as f32 * 0.13 + offset).sin() * 0.5)
            .collect()
    }

    fn tensor(data: &[f32], shape: &[usize], device: &CpuDevice) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(data, shape, device).unwrap()
    }

    /// The three plain SwiGLU projections, built from fixed data.
    fn projections(
        device: &CpuDevice,
        trainable: bool,
    ) -> (Linear<CpuRuntime>, Linear<CpuRuntime>, Linear<CpuRuntime>) {
        let gate = tensor(&vals(INTER * HIDDEN, 0.0), &[INTER, HIDDEN], device);
        let up = tensor(&vals(INTER * HIDDEN, 1.0), &[INTER, HIDDEN], device);
        let down = tensor(&vals(HIDDEN * INTER, 2.0), &[HIDDEN, INTER], device);
        (
            Linear::new(gate, None, trainable),
            Linear::new(up, None, trainable),
            Linear::new(down, None, trainable),
        )
    }

    fn input(device: &CpuDevice) -> Var<CpuRuntime> {
        Var::new(
            tensor(&vals(TOKENS * HIDDEN, 3.0), &[TOKENS, HIDDEN], device),
            false,
        )
    }

    /// Wrap a projection in a LoRA adapter with a NON-ZERO `lora_b`.
    ///
    /// `LoraLinear::new` zero-initializes `lora_b`, which makes the adapted forward
    /// identical to the plain one — a test built on it proves nothing.
    fn adapt(
        base: Linear<CpuRuntime>,
        device: &CpuDevice,
        offset: f32,
    ) -> MaybeLoraLinear<CpuRuntime> {
        let in_features = base.weight().tensor().shape()[1];
        let out_features = base.weight().tensor().shape()[0];
        let a = tensor(
            &vals(RANK * in_features, offset),
            &[RANK, in_features],
            device,
        );
        let b = tensor(
            &vals(out_features * RANK, offset + 0.7),
            &[out_features, RANK],
            device,
        );
        LoraLinear::from_weights(base, a, b, ALPHA, true).into()
    }

    #[test]
    fn test_expert_forward_shape() {
        let (client, device) = cpu_setup();
        let gate_w = tensor(&[0.1f32; INTER * HIDDEN], &[INTER, HIDDEN], &device);
        let up_w = tensor(&[0.1f32; INTER * HIDDEN], &[INTER, HIDDEN], &device);
        let down_w = tensor(&[0.1f32; HIDDEN * INTER], &[HIDDEN, INTER], &device);

        let expert = Expert::from_tensors(gate_w, up_w, down_w, false);

        let x = Var::new(
            tensor(&[1.0f32; TOKENS * HIDDEN], &[TOKENS, HIDDEN], &device),
            false,
        );
        let out = expert.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[TOKENS, HIDDEN]);
    }

    /// `Expert::new` with plain `Linear`s must behave exactly as before the
    /// projections became `MaybeLoraLinear` — same SwiGLU composition, bit for bit.
    #[test]
    fn test_plain_expert_matches_direct_linear_path() {
        let (client, device) = cpu_setup();
        let (gate, up, down) = projections(&device, false);
        let x = input(&device);

        // Reference: the pre-change body, computed on standalone `Linear`s.
        let (ref_gate, ref_up, ref_down) = projections(&device, false);
        let g = ref_gate.forward(&client, &x).unwrap();
        let u = ref_up.forward(&client, &x).unwrap();
        let g_silu = var_silu(&g, &client).unwrap();
        let hidden = var_mul(&g_silu, &u, &client).unwrap();
        let expected: Vec<f32> = ref_down
            .forward(&client, &hidden)
            .unwrap()
            .tensor()
            .to_vec();

        let expert = Expert::new(gate, up, down);
        let actual: Vec<f32> = expert.forward(&client, &x).unwrap().tensor().to_vec();

        assert_eq!(actual, expected);
    }

    /// An adapted expert exposes its base weights AND both adapter factors.
    ///
    /// `Module::parameters` is unfiltered — whether a base TRAINS is decided by
    /// its `requires_grad`, which `trainable_parameters` filters on (see
    /// `test_frozen_bases_leave_only_adapters_trainable`). Enumeration and
    /// trainability are deliberately separate: dropping a base here would also
    /// drop one a caller had deliberately left trainable.
    #[test]
    fn test_adapted_expert_parameters_include_bases_and_adapters() {
        let (_client, device) = cpu_setup();
        let (gate, up, down) = projections(&device, true);
        let expert = Expert::new_adapted(
            adapt(gate, &device, 0.4),
            adapt(up, &device, 0.9),
            down.into(),
        );

        let params = Module::parameters_with_ids(&expert);
        // 3 base weights (no biases) + 2 adapted projections x 2 factors.
        assert_eq!(params.len(), 7);

        for proj in [expert.gate_proj(), expert.up_proj()] {
            let (a, b) = proj.adapters().expect("projection is adapted");
            assert!(params.iter().any(|(id, _)| *id == a.id()));
            assert!(params.iter().any(|(id, _)| *id == b.id()));
            let base_weight_id = proj.weight().expect("dense base has a Var weight").id();
            assert!(
                params.iter().any(|(id, _)| *id == base_weight_id),
                "an adapted projection's dense base weight is still enumerated"
            );
        }
        assert!(expert.down_proj().adapters().is_none());
        let down_weight_id = expert
            .down_proj()
            .weight()
            .expect("dense base has a Var weight")
            .id();
        assert!(params.iter().any(|(id, _)| *id == down_weight_id));
    }

    /// With frozen bases, only the adapter factors are trainable.
    #[test]
    fn test_frozen_bases_leave_only_adapters_trainable() {
        let (_client, device) = cpu_setup();
        let (gate, up, down) = projections(&device, false);
        let expert = Expert::new_adapted(
            adapt(gate, &device, 0.4),
            adapt(up, &device, 0.9),
            down.into(),
        );

        let trainable = Module::trainable_parameters(&expert);
        assert_eq!(trainable.len(), 4);

        let mut expected_ids = Vec::new();
        for proj in [expert.gate_proj(), expert.up_proj()] {
            let (a, b) = proj.adapters().expect("projection is adapted");
            expected_ids.push(a.id());
            expected_ids.push(b.id());
        }
        for (id, var) in &trainable {
            assert!(expected_ids.contains(id));
            assert!(var.requires_grad());
        }
    }

    /// A non-zero `lora_b` must change the output; otherwise the adapter is inert.
    #[test]
    fn test_adapted_forward_differs_from_plain() {
        let (client, device) = cpu_setup();
        let x = input(&device);

        let (gate, up, down) = projections(&device, false);
        let plain: Vec<f32> = Expert::new(gate, up, down)
            .forward(&client, &x)
            .unwrap()
            .tensor()
            .to_vec();

        let (gate, up, down) = projections(&device, false);
        let adapted: Vec<f32> = Expert::new_adapted(
            adapt(gate, &device, 0.4),
            adapt(up, &device, 0.9),
            adapt(down, &device, 1.5),
        )
        .forward(&client, &x)
        .unwrap()
        .tensor()
        .to_vec();

        assert_eq!(plain.len(), adapted.len());
        let max_diff = plain
            .iter()
            .zip(&adapted)
            .map(|(p, a)| (p - a).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-3,
            "adapter had no effect (max diff {max_diff})"
        );
    }

    /// Merging folds the adapters into the base weights without changing the output.
    #[test]
    fn test_merge_adapters_preserves_forward() {
        let (client, device) = cpu_setup();
        let x = input(&device);
        let (gate, up, down) = projections(&device, false);
        let expert = Expert::new_adapted(
            adapt(gate, &device, 0.4),
            adapt(up, &device, 0.9),
            adapt(down, &device, 1.5),
        );

        let adapted: Vec<f32> = expert.forward(&client, &x).unwrap().tensor().to_vec();

        let merged = expert.merge_adapters(&client).unwrap();
        assert!(merged.gate_proj().adapters().is_none());
        assert!(merged.up_proj().adapters().is_none());
        assert!(merged.down_proj().adapters().is_none());

        let merged_out: Vec<f32> = merged.forward(&client, &x).unwrap().tensor().to_vec();
        assert_eq!(adapted.len(), merged_out.len());
        for (a, m) in adapted.iter().zip(&merged_out) {
            assert!((a - m).abs() < 1e-6, "merged output diverged: {a} vs {m}");
        }
    }
}
