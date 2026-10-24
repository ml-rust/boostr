//! [`GdnBlock::forward_train_checkpointed`]: the training forward under
//! activation checkpointing.
//!
//! The segment keeps the block input and drops every activation inside the
//! block. Backward runs the forward once more to rebuild them.

use super::layer::GdnBlock;
use crate::error::{Error, Result};
use crate::nn::segment_masks::SegmentMasks;
use crate::ops::traits::GatedDeltaNetOps;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, checkpoint_with_client};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, MatmulOps, ScalarOps, ShapeOps, TensorOps,
    TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime<DType = DType>> GdnBlock<R> {
    /// [`Self::forward_train`] under activation checkpointing.
    ///
    /// Values equal `forward_train` bit for bit: the same ops run in the
    /// same order on `client`. The recompute runs on `client` too.
    ///
    /// `segments` packs documents as in `forward_train`. The segment closure
    /// owns a clone of the masks, which copies handles only, so the
    /// recompute sees the same document starts.
    ///
    /// The segment closure must be `'static`, so it captures
    /// [`Self::alias`], which keeps every `TensorId`. The segment inputs are
    /// `x` followed by every parameter that requires grad: trainable
    /// projection weights, attached LoRA factors, the conv kernel, `A_log`,
    /// `dt_bias` and the norm weight. The recompute differentiates the ids
    /// it was given, so each of them gets the gradient `forward_train`
    /// gives it. A frozen adapter is no input and gets no gradient.
    ///
    /// # Errors
    ///
    /// As [`Self::forward_train`]. The shape and projection checks run
    /// before the segment, so they keep their error variants. An error
    /// inside the segment arrives as [`Error::Numr`].
    pub fn forward_train_checkpointed<C>(
        &self,
        client: &C,
        x: &Var<R>,
        segments: Option<&SegmentMasks<R>>,
    ) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + CompareOps<R>
            + MatmulOps<R>
            + QuantMatmulOps<R>
            + TypeConversionOps<R>
            + GatedDeltaNetOps<R>
            + 'static,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + CompareOps<R>
            + MatmulOps<R>
            + DequantOps<R>
            + GatedDeltaNetOps<R>,
    {
        self.check_train_input(x, segments)?;
        let block = self.alias();
        let masks = segments.cloned();
        let trainable = self.trainable_parameters();
        let mut inputs: Vec<&Var<R>> = Vec::with_capacity(trainable.len() + 1);
        inputs.push(x);
        inputs.extend(trainable.iter().map(|(_, param)| *param));
        checkpoint_with_client(
            move |segment_inputs, client: &C| {
                let input = segment_inputs.first().ok_or_else(|| {
                    numr::error::Error::Internal(
                        "checkpointed GdnBlock segment received no input".to_string(),
                    )
                })?;
                block
                    .forward_train(client, input, masks.as_ref())
                    .map_err(|e| {
                        numr::error::Error::Backend(format!("checkpointed GdnBlock forward: {e}"))
                    })
            },
            &inputs,
            client,
        )
        .map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::super::GdnProjection;
    use super::super::init::tests::{
        BATCH, SEQ, cfg, f64_bits, init_block, input, seeded_weighted_loss, uniform,
    };
    use super::super::train_fixtures::{HIDDEN, KERNEL, build64, cfg64, init_params};
    use crate::nn::LoraFactors;
    use crate::nn::segment_masks::SegmentMasks;
    use crate::test_utils::{check_packed_vs_separate_tol, cpu_setup, ids_for};
    use numr::autograd::Var;
    use numr::autograd::backward;
    use numr::dtype::DType;
    use numr::runtime::RuntimeClient;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;
    use numr::tensor::TensorId;

    type Block = super::GdnBlock<CpuRuntime>;
    const RANK: usize = 2;

    /// Output bits and the gradient bits of `x` and every trainable id,
    /// for one forward and backward.
    fn run(
        client: &CpuClient,
        block: &Block,
        dtype: DType,
        checkpointed: bool,
        lens: Option<&[usize]>,
    ) -> (Vec<u64>, Vec<(TensorId, Vec<u64>)>) {
        let device = client.device().clone();
        let x = input(client, &device, dtype, 0x6d4e_0300, true);
        let masks = lens.map(|lens| {
            let row = ids_for(lens);
            let ids: Vec<i32> = row.iter().chain(&row).copied().collect();
            let ids = Tensor::<CpuRuntime>::from_slice(&ids, &[BATCH, row.len()], &device).unwrap();
            SegmentMasks::from_ids(client, &ids, cfg().conv_kernel, dtype).unwrap()
        });
        let y = if checkpointed {
            block
                .forward_train_checkpointed(client, &x, masks.as_ref())
                .unwrap()
        } else {
            block.forward_train(client, &x, masks.as_ref()).unwrap()
        };
        let grads = backward(&seeded_weighted_loss(client, &y, 0x6d4e_0301), client).unwrap();
        let mut ids = vec![x.id()];
        ids.extend(block.trainable_parameters().into_iter().map(|(id, _)| id));
        let grad_bits = ids
            .into_iter()
            .map(|id| {
                let grad = grads
                    .get(id)
                    .unwrap_or_else(|| panic!("checkpointed={checkpointed}: no gradient for {id}"));
                (id, f64_bits(grad))
            })
            .collect();
        (f64_bits(y.tensor()), grad_bits)
    }

    fn assert_same(client: &CpuClient, block: &Block, dtype: DType) {
        assert_same_packed(client, block, dtype, None);
    }

    fn assert_same_packed(client: &CpuClient, block: &Block, dtype: DType, lens: Option<&[usize]>) {
        let plain = run(client, block, dtype, false, lens);
        let checkpointed = run(client, block, dtype, true, lens);
        assert_eq!(checkpointed.0, plain.0, "{dtype:?}: output bits");
        assert_eq!(checkpointed.1.len(), plain.1.len());
        for ((id, got), (_, want)) in checkpointed.1.iter().zip(&plain.1) {
            assert_eq!(got, want, "{dtype:?}: gradient bits of {id}");
        }
    }

    fn factors(
        client: &CpuClient,
        device: &CpuDevice,
        dtype: DType,
        out: usize,
        inp: usize,
        seed: u64,
    ) -> LoraFactors<CpuRuntime> {
        LoraFactors {
            lora_a: uniform(client, device, dtype, &[RANK, inp], seed),
            lora_b: uniform(client, device, dtype, &[out, RANK], seed + 1),
            scaling: 0.5,
        }
    }

    /// Trainable LoRA factors on `attn_qkv`, `ssm_out` and both gates.
    fn attach_adapters(client: &CpuClient, block: &mut Block, dtype: DType) {
        let device = client.device().clone();
        let cfg = cfg();
        let (hidden, heads) = (cfg.hidden_size, cfg.value_heads);
        let qkv = factors(client, &device, dtype, cfg.qkv_dim(), hidden, 0x6d4e_0310);
        let out = factors(client, &device, dtype, hidden, cfg.value_dim(), 0x6d4e_0312);
        let alpha = factors(client, &device, dtype, heads, hidden, 0x6d4e_0314);
        let beta = factors(client, &device, dtype, heads, hidden, 0x6d4e_0316);
        block
            .attach_adapter_trainable(client, GdnProjection::AttnQkv, qkv, dtype, true)
            .unwrap();
        block
            .attach_adapter_trainable(client, GdnProjection::SsmOut, out, dtype, true)
            .unwrap();
        block
            .attach_gate_adapters_trainable(client, Some(alpha), Some(beta), dtype, true)
            .unwrap();
    }

    #[test]
    fn checkpointed_equals_forward_train_f64() {
        let (client, device) = cpu_setup();
        let block = init_block(&client, &device, DType::F64, 0x6d4e_0320, true);
        assert_same(&client, &block, DType::F64);
    }

    #[test]
    fn checkpointed_equals_forward_train_f32() {
        let (client, device) = cpu_setup();
        let block = init_block(&client, &device, DType::F32, 0x6d4e_0321, true);
        assert_same(&client, &block, DType::F32);
    }

    /// A trainable block with trainable adapters: every weight and every
    /// LoRA factor gets the `forward_train` gradient.
    #[test]
    fn checkpointed_equals_forward_train_with_adapters() {
        let (client, device) = cpu_setup();
        let mut block = init_block(&client, &device, DType::F64, 0x6d4e_0322, true);
        attach_adapters(&client, &mut block, DType::F64);
        let lora: Vec<String> = block
            .named_parameters()
            .into_iter()
            .filter(|(name, _)| name.contains("lora_"))
            .map(|(name, _)| name)
            .collect();
        assert_eq!(
            lora.len(),
            8,
            "two factors on each of four adapters: {lora:?}"
        );
        assert_same(&client, &block, DType::F64);
    }

    /// A frozen block with trainable adapters trains the LoRA factors only,
    /// with the `forward_train` gradients.
    #[test]
    fn frozen_block_trains_only_adapters() {
        let (client, device) = cpu_setup();
        let mut block = init_block(&client, &device, DType::F64, 0x6d4e_0323, false);
        assert!(block.trainable_parameters().is_empty());
        attach_adapters(&client, &mut block, DType::F64);
        let trainable: Vec<String> = block
            .named_parameters()
            .into_iter()
            .filter(|(_, var)| var.requires_grad())
            .map(|(name, _)| name)
            .collect();
        assert_eq!(
            trainable,
            [
                "in_proj_qkv.lora_a",
                "in_proj_qkv.lora_b",
                "alpha_proj.lora_a",
                "alpha_proj.lora_b",
                "beta_proj.lora_a",
                "beta_proj.lora_b",
                "out_proj.lora_a",
                "out_proj.lora_b",
            ]
        );
        assert_same(&client, &block, DType::F64);

        let x = input(&client, &device, DType::F64, 0x6d4e_0324, false);
        let y = block.forward_train_checkpointed(&client, &x, None).unwrap();
        let grads = backward(&seeded_weighted_loss(&client, &y, 0x6d4e_0325), &client).unwrap();
        for (name, var) in block.named_parameters() {
            if name.contains("proj") && !var.requires_grad() {
                assert!(grads.get(var.id()).is_none(), "frozen {name} gets none");
            }
        }
    }

    /// Packed rows: the recompute sees the same document starts, so values
    /// and gradients equal `forward_train` bit for bit. Starts at 3 and 4
    /// fall inside and on the edge of a chunk of four.
    #[test]
    fn checkpointed_packed_equals_forward_train() {
        let (client, device) = cpu_setup();
        let block = init_block(&client, &device, DType::F64, 0x6d4e_0326, true);
        let lens = [3, 1, SEQ - 4];
        assert_same_packed(&client, &block, DType::F64, Some(&lens));
    }

    /// The checkpointed packed forward equals the documents run alone, in
    /// output, input gradient and every parameter gradient.
    #[test]
    fn checkpointed_packed_equals_separate_documents() {
        let (client, device) = cpu_setup();
        let cfg = cfg64(2, false);
        let block = build64(&client, &device, &cfg, &init_params(&cfg, 0x6d4e_0327));
        let params = block.trainable_parameters();
        let fwd = |c: &CpuClient, x: &Var<CpuRuntime>, s: Option<&SegmentMasks<CpuRuntime>>| {
            block.forward_train_checkpointed(c, x, s)
        };
        check_packed_vs_separate_tol(&fwd, &params, HIDDEN, KERNEL, &[6, 1, 3], 1e-9);
    }
}
