//! `Plain`/`Rotated` dispatch over [`MaybeQuantLinear`], mirroring
//! [`crate::nn::maybe_lora::MaybeLoraLinear`]'s `Plain`/adapted dispatch.
//! [`MaybeRotatedLinear::forward_batch`] fuses several projections that
//! share one input, running any shared Hadamard rotation once.

use super::maybe_quant_linear::MaybeQuantLinear;
use super::rotated_linear::RotatedLinear;
use crate::error::{Error, Result};
use crate::nn::hadamard::HadamardRotation;
use crate::nn::module::Module;
use crate::quant::QuantTensor;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{BinaryOps, FwhtOps, MatmulOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::TensorId;

/// A linear projection that is either plain or Hadamard-rotated.
// `Rotated` is boxed for the same reason `MaybeLoraLinear::Lora` is: `Plain`
// is the common variant and stays inline, so the size gap is accepted.
#[allow(clippy::large_enum_variant)]
pub enum MaybeRotatedLinear<R: Runtime> {
    Plain(MaybeQuantLinear<R>),
    Rotated(Box<RotatedLinear<R>>),
}

impl<R: Runtime<DType = DType>> MaybeRotatedLinear<R> {
    /// Forward pass: plain base, or rotate-then-base.
    pub fn forward<C>(&self, client: &C, input: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + FwhtOps<R>,
        R::Client: TensorOps<R> + DequantOps<R> + MatmulOps<R>,
    {
        match self {
            Self::Plain(base) => base.forward(client, input),
            Self::Rotated(rotated) => rotated.forward(client, input),
        }
    }

    /// Forward several projections that share one input, running any shared
    /// Hadamard rotation ONCE.
    ///
    /// `layers` is partitioned into its `Plain` and `Rotated` members,
    /// order preserved for the returned `Vec`. The `Plain` subset runs
    /// through one [`MaybeQuantLinear::forward_batch`] call over `input`.
    /// Every `Rotated` member must carry the same rotation as the first one
    /// ([`HadamardRotation::same_rotation_as`]) — the rotation then runs
    /// ONCE. When every `Rotated` base is block-quantized without a bias
    /// the rotation goes to `quant_matmul_batch_rotated`, which folds it
    /// into the activation quantization where the backend has that kernel;
    /// otherwise it runs as [`HadamardRotation::forward`] and the bases run
    /// through a second [`MaybeQuantLinear::forward_batch`] call over the
    /// rotated activation. So a batch mixing `Plain` and `Rotated` members
    /// costs one rotation plus two fused batched matmuls, not one `forward`
    /// per layer. An all-`Plain` batch behaves exactly as before (one fused
    /// call, no rotation); a single-member batch works either way.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when two `Rotated` members disagree on
    /// rotation (different `block_size`, or different sign storage).
    pub fn forward_batch<C>(layers: &[&Self], client: &C, input: &Var<R>) -> Result<Vec<Var<R>>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + QuantMatmulOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + FwhtOps<R>,
        R::Client: TensorOps<R> + DequantOps<R> + MatmulOps<R>,
    {
        let mut plain_idx = Vec::new();
        // `(original index, the Rotated member)` — captured directly while
        // partitioning, so the rotation-agreement check below never needs
        // to re-match a `Plain` layer against `Rotated`.
        let mut rotated: Vec<(usize, &RotatedLinear<R>)> = Vec::new();
        for (i, layer) in layers.iter().enumerate() {
            match layer {
                Self::Plain(_) => plain_idx.push(i),
                Self::Rotated(rotated_layer) => rotated.push((i, rotated_layer.as_ref())),
            }
        }

        let mut plain_outs = if plain_idx.is_empty() {
            Vec::new()
        } else {
            let bases: Vec<&MaybeQuantLinear<R>> =
                plain_idx.iter().map(|&i| layers[i].base()).collect();
            MaybeQuantLinear::forward_batch(&bases, client, input)?
        }
        .into_iter();

        let mut rotated_outs = if let [(first_idx, first), rest @ ..] = rotated.as_slice() {
            let first_rotation = first.rotation();
            for (i, member) in rest {
                let rotation = member.rotation();
                if !rotation.same_rotation_as(first_rotation) {
                    return Err(Error::ModelError {
                        reason: format!(
                            "forward_batch: Rotated layer at index {i} does not share layer \
                             {first_idx}'s rotation (block_size {} vs {}, signs storage {})",
                            rotation.block_size(),
                            first_rotation.block_size(),
                            if rotation.signs_ptr() == first_rotation.signs_ptr() {
                                "matched"
                            } else {
                                "did not match"
                            }
                        ),
                    });
                }
            }
            let quant_weights: Vec<&QuantTensor<R>> = rotated
                .iter()
                .filter_map(|(_, member)| member.quant_weight_without_bias())
                .collect();
            if quant_weights.len() == rotated.len() {
                // Every base is block-quantized without a bias: the rotation
                // goes with the batch, folded into the activation
                // quantization where the backend has that kernel.
                let rotation = first_rotation.rotation_for(input.tensor())?;
                client
                    .quant_matmul_batch_rotated(input.tensor(), &rotation, &quant_weights)?
                    .into_iter()
                    .map(|t| Var::new(t, false))
                    .collect::<Vec<Var<R>>>()
            } else {
                let rotated_input = first_rotation.forward(client, input.tensor())?;
                let rotated_input = Var::new(rotated_input, false);
                let bases: Vec<&MaybeQuantLinear<R>> =
                    rotated.iter().map(|(_, member)| member.base()).collect();
                MaybeQuantLinear::forward_batch(&bases, client, &rotated_input)?
            }
        } else {
            Vec::new()
        }
        .into_iter();

        // Both `forward_batch` calls preserve the order of their `bases`
        // slice, which was built by iterating `layers` in order — so the
        // n-th `Plain`/`Rotated` output lines up with the n-th `Plain`/
        // `Rotated` member in one pass over `layers`. A positional walk
        // restores the original order in O(n), no index bookkeeping or sort.
        layers
            .iter()
            .map(|layer| match layer {
                Self::Plain(_) => plain_outs.next().ok_or_else(|| Error::ModelError {
                    reason: "forward_batch: plain output count did not match plain input count"
                        .to_string(),
                }),
                Self::Rotated(_) => rotated_outs.next().ok_or_else(|| Error::ModelError {
                    reason: "forward_batch: rotated output count did not match rotated input \
                             count"
                        .to_string(),
                }),
            })
            .collect()
    }

    /// The underlying base linear layer, rotation aside.
    pub fn base(&self) -> &MaybeQuantLinear<R> {
        match self {
            Self::Plain(base) => base,
            Self::Rotated(rotated) => rotated.base(),
        }
    }

    /// The Hadamard rotation attached to this layer, if any.
    pub fn rotation(&self) -> Option<&HadamardRotation<R>> {
        match self {
            Self::Plain(_) => None,
            Self::Rotated(rotated) => Some(rotated.rotation()),
        }
    }

    pub fn weight(&self) -> Option<&Var<R>> {
        self.base().weight()
    }

    pub fn bias(&self) -> Option<&Var<R>> {
        self.base().bias()
    }

    /// `true` when a Hadamard rotation is attached.
    pub fn is_rotated(&self) -> bool {
        matches!(self, Self::Rotated(_))
    }

    pub fn parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.base().parameters()
    }

    pub fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        self.base().trainable_parameters()
    }

    pub fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        self.base().named_parameters()
    }
}

impl<R: Runtime<DType = DType>> From<MaybeQuantLinear<R>> for MaybeRotatedLinear<R> {
    fn from(base: MaybeQuantLinear<R>) -> Self {
        Self::Plain(base)
    }
}

impl<R: Runtime<DType = DType>> From<RotatedLinear<R>> for MaybeRotatedLinear<R> {
    fn from(rotated: RotatedLinear<R>) -> Self {
        Self::Rotated(Box::new(rotated))
    }
}

impl<R: Runtime<DType = DType>> Module<R> for MaybeRotatedLinear<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        MaybeRotatedLinear::parameters(self)
            .into_iter()
            .map(|param| param.1)
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        MaybeRotatedLinear::named_parameters(self)
    }

    fn parameters_with_ids(&self) -> Vec<(TensorId, &Var<R>)> {
        MaybeRotatedLinear::parameters(self)
    }

    fn trainable_parameters(&self) -> Vec<(TensorId, &Var<R>)> {
        MaybeRotatedLinear::trainable_parameters(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::linear::dense::Linear;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    const IN: usize = 16;
    const BLOCK: usize = 8;

    fn plain_base(
        out: usize,
        seed: f32,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> MaybeQuantLinear<CpuRuntime> {
        let data: Vec<f32> = (0..out * IN).map(|i| ((i as f32) * seed).sin()).collect();
        let weight = Tensor::<CpuRuntime>::from_slice(&data, &[out, IN], device).unwrap();
        MaybeQuantLinear::Standard(Linear::new(weight, None, false))
    }

    fn rotation(
        signs: &[i8],
        device: &<CpuRuntime as Runtime>::Device,
    ) -> HadamardRotation<CpuRuntime> {
        HadamardRotation::<CpuRuntime>::new(BLOCK, Some(signs), DType::F32, device).unwrap()
    }

    fn rotated(
        out: usize,
        seed: f32,
        rot: HadamardRotation<CpuRuntime>,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> MaybeRotatedLinear<CpuRuntime> {
        RotatedLinear::new(plain_base(out, seed, device), rot)
            .unwrap()
            .into()
    }

    fn rotated_plain(
        out: usize,
        seed: f32,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> MaybeRotatedLinear<CpuRuntime> {
        MaybeRotatedLinear::Plain(plain_base(out, seed, device))
    }

    fn input(device: &<CpuRuntime as Runtime>::Device) -> Var<CpuRuntime> {
        let x_data: Vec<f32> = (0..3 * IN).map(|i| (i as f32) * 0.05 - 0.3).collect();
        Var::new(
            Tensor::<CpuRuntime>::from_slice(&x_data, &[3, IN], device).unwrap(),
            false,
        )
    }

    #[test]
    fn forward_batch_two_rotated_layers_sharing_one_rotation_matches_per_layer_forward() {
        let (client, device) = cpu_setup();
        let signs: Vec<i8> = (0..IN).map(|i| if i % 3 == 0 { -1 } else { 1 }).collect();
        let rot = rotation(&signs, &device);

        let a = rotated(4, 0.01, rot.clone(), &device);
        let b = rotated(5, 0.02, rot, &device);
        let x = input(&device);

        let batched = MaybeRotatedLinear::forward_batch(&[&a, &b], &client, &x).unwrap();
        let solo_a = a.forward(&client, &x).unwrap();
        let solo_b = b.forward(&client, &x).unwrap();

        assert_eq!(
            batched[0].tensor().to_vec::<f32>(),
            solo_a.tensor().to_vec::<f32>()
        );
        assert_eq!(
            batched[1].tensor().to_vec::<f32>(),
            solo_b.tensor().to_vec::<f32>()
        );
    }

    #[test]
    fn forward_batch_mixed_rotated_and_plain_matches_per_layer_forward_order_preserved() {
        let (client, device) = cpu_setup();
        let signs: Vec<i8> = (0..IN).map(|i| if i % 2 == 0 { -1 } else { 1 }).collect();
        let rot = rotation(&signs, &device);

        let a = rotated(3, 0.01, rot.clone(), &device);
        let p = rotated_plain(4, 0.02, &device);
        let b = rotated(5, 0.03, rot, &device);
        let x = input(&device);

        let batched = MaybeRotatedLinear::forward_batch(&[&a, &p, &b], &client, &x).unwrap();
        let solo_a = a.forward(&client, &x).unwrap();
        let solo_p = p.forward(&client, &x).unwrap();
        let solo_b = b.forward(&client, &x).unwrap();

        assert_eq!(
            batched[0].tensor().to_vec::<f32>(),
            solo_a.tensor().to_vec::<f32>()
        );
        assert_eq!(
            batched[1].tensor().to_vec::<f32>(),
            solo_p.tensor().to_vec::<f32>()
        );
        assert_eq!(
            batched[2].tensor().to_vec::<f32>(),
            solo_b.tensor().to_vec::<f32>()
        );
    }

    #[test]
    fn forward_batch_rejects_rotated_layers_with_different_signs() {
        let (client, device) = cpu_setup();
        let signs_a: Vec<i8> = (0..IN).map(|i| if i % 2 == 0 { -1 } else { 1 }).collect();
        let signs_b: Vec<i8> = (0..IN).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();

        let a = rotated(3, 0.01, rotation(&signs_a, &device), &device);
        let b = rotated(4, 0.02, rotation(&signs_b, &device), &device);
        let x = input(&device);

        assert!(MaybeRotatedLinear::forward_batch(&[&a, &b], &client, &x).is_err());
    }

    #[test]
    fn maybe_rotated_plain_forwards_identically_to_bare_maybe_quant_linear() {
        let (client, device) = cpu_setup();
        let weight_data: Vec<f32> = (0..4 * IN).map(|i| (i as f32) * 0.02).collect();
        let weight = Tensor::<CpuRuntime>::from_slice(&weight_data, &[4, IN], &device).unwrap();
        let bare = MaybeQuantLinear::Standard(Linear::new(
            Tensor::<CpuRuntime>::from_slice(&weight_data, &[4, IN], &device).unwrap(),
            None,
            false,
        ));
        let plain: MaybeRotatedLinear<CpuRuntime> =
            MaybeQuantLinear::Standard(Linear::new(weight, None, false)).into();

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32; IN], &[1, IN], &device).unwrap(),
            false,
        );

        let bare_out = bare.forward(&client, &x).unwrap();
        let plain_out = plain.forward(&client, &x).unwrap();
        assert_eq!(
            bare_out.tensor().to_vec::<f32>(),
            plain_out.tensor().to_vec::<f32>()
        );
        assert!(!plain.is_rotated());
    }
}
