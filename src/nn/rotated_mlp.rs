//! SwiGLU MLP over [`MaybeRotatedLinear`] projections: the `qwen35` FFN.
//!
//! `forward(x) = down(silu(gate(x)) * up(x))`, the same shape as
//! [`crate::model::llama::model::blocks::mlp::LlamaMlp`].
//!
//! The fused `quant_swiglu` kernel `LlamaMlp` reaches for still is not used
//! here: `gate` and `up` may be Hadamard-rotated, and that kernel reads the
//! raw activation, not a rotated one. `gate` and `up` instead go through one
//! [`MaybeRotatedLinear::forward_batch`] call, which runs any rotation they
//! share ONCE and each fused `quant_matmul_batch` (rotated and plain
//! members split into one call each) instead of one `forward` per
//! projection, then numr's fused `var_silu_mul`, then `down`.

use crate::error::{Error, Result};
use crate::nn::linear::MaybeRotatedLinear;
use crate::nn::module::Module;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_silu_mul};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, FwhtOps, MatmulOps, ScalarOps, TensorOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::TensorId;

/// SwiGLU MLP whose three projections may each be Hadamard-rotated.
pub struct RotatedMlp<R: Runtime> {
    gate: MaybeRotatedLinear<R>,
    up: MaybeRotatedLinear<R>,
    down: MaybeRotatedLinear<R>,
}

impl<R: Runtime<DType = DType>> RotatedMlp<R> {
    /// Build from built projections.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `gate` and `up` disagree on shape, or
    /// `down` is not `[gate.in_features, gate.out_features]`.
    pub fn new(
        gate: MaybeRotatedLinear<R>,
        up: MaybeRotatedLinear<R>,
        down: MaybeRotatedLinear<R>,
    ) -> Result<Self> {
        let gate_shape = gate.base().shape();
        let up_shape = up.base().shape();
        let down_shape = down.base().shape();
        if gate_shape.len() != 2 || gate_shape != up_shape {
            return Err(Error::ModelError {
                reason: format!(
                    "rotated_mlp: gate {gate_shape:?} and up {up_shape:?} must share a 2-D shape"
                ),
            });
        }
        let want_down = [gate_shape[1], gate_shape[0]];
        if down_shape != want_down {
            return Err(Error::ModelError {
                reason: format!("rotated_mlp: down must be {want_down:?}, got {down_shape:?}"),
            });
        }
        Ok(Self { gate, up, down })
    }

    /// `down(silu(gate(x)) * up(x))`.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + QuantMatmulOps<R>
            + TypeConversionOps<R>
            + FwhtOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>
            + MatmulOps<R>,
    {
        let mut projected =
            MaybeRotatedLinear::forward_batch(&[&self.gate, &self.up], client, x)?.into_iter();
        let gate = projected.next().ok_or_else(|| Error::ModelError {
            reason: "rotated_mlp: forward_batch returned no gate output".to_string(),
        })?;
        let up = projected.next().ok_or_else(|| Error::ModelError {
            reason: "rotated_mlp: forward_batch returned no up output".to_string(),
        })?;
        let hidden = var_silu_mul(&gate, &up, client).map_err(Error::Numr)?;
        self.down.forward(client, &hidden)
    }

    /// Model width: `gate`'s input width.
    pub fn hidden_size(&self) -> usize {
        self.gate.base().shape()[1]
    }

    /// FFN width: `gate`'s output width.
    pub fn intermediate_size(&self) -> usize {
        self.gate.base().shape()[0]
    }

    pub fn gate(&self) -> &MaybeRotatedLinear<R> {
        &self.gate
    }

    pub fn up(&self) -> &MaybeRotatedLinear<R> {
        &self.up
    }

    pub fn down(&self) -> &MaybeRotatedLinear<R> {
        &self.down
    }

    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        let mut params = self.gate.parameters();
        params.extend(self.up.parameters());
        params.extend(self.down.parameters());
        params
    }

    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.parameters()
            .into_iter()
            .filter(|param| param.1.requires_grad())
            .collect()
    }

    pub fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        for (prefix, layer) in [("gate", &self.gate), ("up", &self.up), ("down", &self.down)] {
            for (name, var) in layer.named_parameters() {
                params.push((format!("{prefix}.{name}"), var));
            }
        }
        params
    }
}

impl<R: Runtime<DType = DType>> Module<R> for RotatedMlp<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        RotatedMlp::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        RotatedMlp::named_parameters(self)
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        RotatedMlp::parameters(self)
    }

    fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        RotatedMlp::trainable_parameters(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::llama::model::blocks::mlp::LlamaMlp;
    use crate::nn::hadamard::HadamardRotation;
    use crate::nn::linear::RotatedLinear;
    use crate::nn::{Linear, MaybeQuantLinear};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    const HIDDEN: usize = 8;
    const INTER: usize = 12;

    fn tensor(device: &CpuDevice, shape: &[usize], seed: u32) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let h = (i as u32).wrapping_mul(2_654_435_761u32).wrapping_add(seed);
                (h % 1000) as f32 / 1000.0 - 0.5
            })
            .collect();
        Tensor::<CpuRuntime>::from_slice(&data, shape, device).unwrap()
    }

    fn plain(w: Tensor<CpuRuntime>) -> MaybeQuantLinear<CpuRuntime> {
        MaybeQuantLinear::Standard(Linear::new(w, None, false))
    }

    #[test]
    fn plain_matches_llama_mlp() {
        let (client, device) = cpu_setup();
        let gate = tensor(&device, &[INTER, HIDDEN], 1);
        let up = tensor(&device, &[INTER, HIDDEN], 2);
        let down = tensor(&device, &[HIDDEN, INTER], 3);

        let rotated = RotatedMlp::new(
            MaybeRotatedLinear::Plain(plain(gate.clone())),
            MaybeRotatedLinear::Plain(plain(up.clone())),
            MaybeRotatedLinear::Plain(plain(down.clone())),
        )
        .unwrap();
        let llama = LlamaMlp {
            gate_proj: plain(gate),
            up_proj: plain(up),
            down_proj: plain(down),
        };
        assert_eq!(rotated.hidden_size(), HIDDEN);
        assert_eq!(rotated.intermediate_size(), INTER);

        let x = Var::new(tensor(&device, &[1, 5, HIDDEN], 9), false);
        let a: Vec<f32> = rotated.forward(&client, &x).unwrap().tensor().to_vec();
        let b: Vec<f32> = llama.forward(&client, &x).unwrap().tensor().to_vec();
        assert_eq!(a.len(), b.len());
        for (p, q) in a.iter().zip(&b) {
            assert!((p - q).abs() < 1e-6, "{p} vs {q}");
        }
    }

    /// `gate` and `up` both `Rotated`, sharing one `HadamardRotation`, must
    /// match rotating `x` once by hand and running the same weights through
    /// [`LlamaMlp`] — proving `forward_batch`'s one-rotation, two-fused-call
    /// path computes the same thing as rotate-then-per-projection-forward.
    #[test]
    fn gate_and_up_both_rotated_sharing_one_rotation_matches_manual_rotate_then_forward() {
        let (client, device) = cpu_setup();
        let gate_w = tensor(&device, &[INTER, HIDDEN], 1);
        let up_w = tensor(&device, &[INTER, HIDDEN], 2);
        let down_w = tensor(&device, &[HIDDEN, INTER], 3);

        let signs: Vec<i8> = (0..HIDDEN)
            .map(|i| if i % 2 == 0 { -1 } else { 1 })
            .collect();
        let rotation =
            HadamardRotation::<CpuRuntime>::new(HIDDEN, Some(&signs), DType::F32, &device).unwrap();

        let mlp = RotatedMlp::new(
            RotatedLinear::new(plain(gate_w.clone()), rotation.clone())
                .unwrap()
                .into(),
            RotatedLinear::new(plain(up_w.clone()), rotation.clone())
                .unwrap()
                .into(),
            MaybeRotatedLinear::Plain(plain(down_w.clone())),
        )
        .unwrap();

        let x = Var::new(tensor(&device, &[1, 5, HIDDEN], 9), false);
        let out = mlp.forward(&client, &x).unwrap();

        let x_rotated = Var::new(rotation.forward(&client, x.tensor()).unwrap(), false);
        let llama = LlamaMlp {
            gate_proj: plain(gate_w),
            up_proj: plain(up_w),
            down_proj: plain(down_w),
        };
        let manual = llama.forward(&client, &x_rotated).unwrap();

        let a: Vec<f32> = out.tensor().to_vec();
        let b: Vec<f32> = manual.tensor().to_vec();
        assert_eq!(
            a, b,
            "batched rotate-once must match manual rotate-then-forward bit-for-bit"
        );
    }

    #[test]
    fn rejects_shape_mismatch() {
        let (_client, device) = cpu_setup();
        let gate = tensor(&device, &[INTER, HIDDEN], 1);
        let up = tensor(&device, &[INTER + 1, HIDDEN], 2);
        let down = tensor(&device, &[HIDDEN, INTER], 3);
        assert!(
            RotatedMlp::new(
                MaybeRotatedLinear::Plain(plain(gate.clone())),
                MaybeRotatedLinear::Plain(plain(up)),
                MaybeRotatedLinear::Plain(plain(down)),
            )
            .is_err()
        );
        let bad_down = tensor(&device, &[HIDDEN, INTER + 1], 3);
        assert!(
            RotatedMlp::new(
                MaybeRotatedLinear::Plain(plain(gate.clone())),
                MaybeRotatedLinear::Plain(plain(gate)),
                MaybeRotatedLinear::Plain(plain(bad_down)),
            )
            .is_err()
        );
    }
}
