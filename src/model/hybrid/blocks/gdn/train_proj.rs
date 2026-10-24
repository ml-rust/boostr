//! Projections and input checks of the tracked training forward of a
//! [`GdnBlock`].
//!
//! A Hadamard-rotated projection detaches its input, and an AWQ/GPTQ
//! decomposed base gives no input gradient. The forward refuses both
//! before it computes anything.

use super::gate_proj::GateProjections;
use super::layer::GdnBlock;
use crate::error::{Error, Result};
use crate::nn::segment_masks::SegmentMasks;
use crate::nn::{MaybeQuantLinear, MaybeRotatedLinear, check_segments, var_contiguous};
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::ops::{BinaryOps, MatmulOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime<DType = DType>> GdnBlock<R> {
    /// The checks [`Self::forward_train`] runs before any compute: `x` is
    /// `[batch, seq, hidden_size]`, `segments` covers `x` and the conv
    /// kernel, and every projection has a tracked path.
    ///
    /// # Errors
    ///
    /// As [`Self::forward_train`]: [`Error::ModelError`] for the shape,
    /// [`Error::InvalidArgument`] for `segments` or a projection.
    pub(super) fn check_train_input(
        &self,
        x: &Var<R>,
        segments: Option<&SegmentMasks<R>>,
    ) -> Result<()> {
        let shape = x.shape();
        if shape.len() != 3 || shape[2] != self.cfg.hidden_size {
            return Err(Error::ModelError {
                reason: format!(
                    "gdn: expected [batch, seq, {}], got {shape:?}",
                    self.cfg.hidden_size
                ),
            });
        }
        if let Some(masks) = segments {
            check_segments(
                masks,
                shape[0],
                shape[1],
                self.cfg.conv_kernel,
                x.tensor().dtype(),
            )?;
        }
        self.check_tracked()
    }

    /// Refuse every projection the tracked forward cannot carry a gradient
    /// through. Runs before any compute.
    fn check_tracked(&self) -> Result<()> {
        for (linear, name) in [
            (&self.attn_qkv, "attn_qkv"),
            (&self.attn_gate, "attn_gate"),
            (&self.ssm_out, "ssm_out"),
        ] {
            if linear.is_rotated() {
                return Err(rotated_error(name));
            }
            check_base(linear.base(), name)?;
        }
        if let GateProjections::Separate(gates) = &self.gates {
            check_base(gates.alpha.base(), "ssm_alpha")?;
            check_base(gates.beta.base(), "ssm_beta")?;
        }
        Ok(())
    }
}

/// The tracked forward of one projection: plain base, or base plus adapter.
pub(super) fn tracked_forward<R, C>(
    linear: &MaybeRotatedLinear<R>,
    name: &'static str,
    client: &C,
    x: &Var<R>,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + QuantMatmulOps<R>
        + TypeConversionOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R> + BinaryOps<R> + MatmulOps<R> + DequantOps<R>,
{
    match linear {
        MaybeRotatedLinear::Plain(base) => base.forward(client, x),
        MaybeRotatedLinear::Lora(lora) => lora.forward(client, x),
        MaybeRotatedLinear::Rotated(_) => Err(rotated_error(name)),
    }
}

/// `qkv[.., start .. start + len]` reshaped to `shape`, tracked.
pub(super) fn narrow_heads<R>(
    qkv: &Var<R>,
    start: usize,
    len: usize,
    shape: &[usize],
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + ShapeOps<R>,
{
    let part = var_narrow(qkv, 2, start, len).map_err(Error::Numr)?;
    let part = var_contiguous(&part)?;
    var_reshape(&part, shape).map_err(Error::Numr)
}

fn check_base<R: Runtime>(base: &MaybeQuantLinear<R>, name: &'static str) -> Result<()> {
    match base {
        MaybeQuantLinear::DecomposedQuant(_) => Err(Error::InvalidArgument {
            arg: name,
            reason: format!(
                "gdn.{name} has a decomposed AWQ/GPTQ base, which gives no input gradient. \
                 Load it as a dense or GGUF block-quantized weight to train the block"
            ),
        }),
        MaybeQuantLinear::Standard(_) | MaybeQuantLinear::Quantized(_) => Ok(()),
    }
}

fn rotated_error(name: &'static str) -> Error {
    Error::InvalidArgument {
        arg: name,
        reason: format!(
            "gdn.{name} is Hadamard-rotated, and the rotation detaches its input. \
             Load the block without the Hadamard attach to train it"
        ),
    }
}

#[cfg(test)]
mod tests {
    //! The refusal of projections without a tracked path, and of segment
    //! masks the block cannot use.

    use super::super::forward::tests::block;
    use super::super::gate_proj::{GateProjections, SeparateGates};
    use super::super::layer::GdnBlock;
    use crate::error::Error;
    use crate::nn::hadamard::HadamardRotation;
    use crate::nn::segment_masks::SegmentMasks;
    use crate::nn::{Linear, MaybeQuantLinear, RotatedLinear};
    use crate::quant::decomposed::{
        DecomposedQuantLinear, DecomposedQuantMethod, DecomposedQuantTensor,
    };
    use crate::test_utils::{cpu_setup, ids_for};
    use numr::autograd::Var;
    use numr::dtype::DType;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    const HIDDEN: usize = 8;
    const H_V: usize = 4;
    const SEQ: usize = 9;

    type T = Tensor<CpuRuntime>;

    fn expect_refusal(block: &GdnBlock<CpuRuntime>, want: &str) {
        let (client, device) = cpu_setup();
        let x = T::zeros(&[1, SEQ, HIDDEN], DType::F32, &device).unwrap();
        match block.forward_train(&client, &Var::new(x, false), None) {
            Err(Error::InvalidArgument { arg, reason }) => {
                assert_eq!(arg, want);
                assert!(reason.contains(want), "{reason}");
            }
            Err(other) => panic!("{want}: expected InvalidArgument, got {other}"),
            Ok(_) => panic!("{want}: forward_train must refuse"),
        }
    }

    fn decomposed(device: &CpuDevice, out: usize, inp: usize) -> MaybeQuantLinear<CpuRuntime> {
        let z = |shape: &[usize]| T::zeros(shape, DType::F32, device).unwrap();
        let weight = DecomposedQuantTensor::new(
            z(&[inp, 1]),
            z(&[1, out]),
            z(&[1, out]),
            None,
            DecomposedQuantMethod::Awq { group_size: 128 },
            vec![out, inp],
        );
        MaybeQuantLinear::DecomposedQuant(Box::new(DecomposedQuantLinear::new(weight, None)))
    }

    #[test]
    fn refuses_rotated_projection() {
        let (_, device) = cpu_setup();
        let mut block = block(&device, 0x6d4e_0035, false);
        let signs: Vec<i8> = (0..HIDDEN)
            .map(|i| if i % 2 == 0 { -1 } else { 1 })
            .collect();
        let rotation =
            HadamardRotation::<CpuRuntime>::new(HIDDEN, Some(&signs), DType::F32, &device).unwrap();
        let w = block.attn_qkv.weight().unwrap().tensor().clone();
        let base = MaybeQuantLinear::Standard(Linear::new(w, None, false));
        block.attn_qkv = RotatedLinear::new(base, rotation).unwrap().into();
        expect_refusal(&block, "attn_qkv");
    }

    #[test]
    fn refuses_decomposed_base() {
        let (_, device) = cpu_setup();
        let mut out_block = block(&device, 0x6d4e_0036, false);
        let value_dim = out_block.config().value_dim();
        out_block.ssm_out = decomposed(&device, HIDDEN, value_dim).into();
        expect_refusal(&out_block, "ssm_out");

        let mut gate_block = block(&device, 0x6d4e_0037, false);
        let beta = T::zeros(&[H_V, HIDDEN], DType::F32, &device).unwrap();
        gate_block.gates = GateProjections::Separate(Box::new(SeparateGates {
            alpha: decomposed(&device, H_V, HIDDEN).into(),
            beta: MaybeQuantLinear::Standard(Linear::new(beta, None, false)).into(),
        }));
        expect_refusal(&gate_block, "ssm_alpha");
    }

    /// Masks of the wrong shape, dtype or kernel coverage are refused before
    /// any compute, and the error names `segments`.
    #[test]
    fn refuses_segments_the_block_cannot_use() {
        let (client, device) = cpu_setup();
        let block = block(&device, 0x6d4e_0038, false);
        let kernel = block.config().conv_kernel;
        let x = Var::new(
            T::zeros(&[1, SEQ, HIDDEN], DType::F32, &device).unwrap(),
            false,
        );
        let masks = |seq: usize, kernel: usize, dtype: DType| {
            let ids = ids_for(&[seq]);
            let ids = T::from_slice(&ids, &[1, seq], &device).unwrap();
            SegmentMasks::from_ids(&client, &ids, kernel, dtype).unwrap()
        };
        for (bad, why) in [
            (masks(SEQ - 1, kernel, DType::F32), "seq"),
            (masks(SEQ, kernel, DType::F64), "dtype"),
            (masks(SEQ, kernel - 1, DType::F32), "kernel"),
        ] {
            match block.forward_train(&client, &x, Some(&bad)) {
                Err(Error::InvalidArgument { arg, .. }) => assert_eq!(arg, "segments", "{why}"),
                Err(other) => panic!("{why}: expected InvalidArgument, got {other}"),
                Ok(_) => panic!("{why}: forward_train must refuse"),
            }
        }
    }
}
