//! [`Qwen35AttentionBlock::forward_train_checkpointed`]: the tracked
//! training forward under activation checkpointing.

use super::layer::Qwen35AttentionBlock;
use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use crate::nn::RoPE;
use numr::autograd::{Var, checkpoint_with_client};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Qwen35AttentionBlock<R> {
    /// Same result as [`Self::forward_train`], computed with activation
    /// checkpointing: the block's intermediates are dropped during the
    /// forward pass and recomputed during backward.
    ///
    /// `numr::autograd::checkpoint_with_client` runs the segment on
    /// `client`, both on the forward pass and on the recompute. The stored
    /// closure is `Send + Sync + 'static`, so it holds [`Self::alias`], an
    /// aliased `rope` and a handle on `positions`. An alias keeps every
    /// `TensorId`, so each gradient lands on the parameter it belongs to.
    /// `x` and every trainable parameter are the segment's inputs.
    ///
    /// # Errors
    ///
    /// As [`Self::forward_train`]. The projection check runs before the
    /// segment is built.
    pub fn forward_train_checkpointed<C>(
        &self,
        client: &C,
        x: &Var<R>,
        rope: &RoPE<R>,
        positions: &Tensor<R>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + 'static,
        R::Client: ModelClient<R>,
    {
        self.check_tracked()?;
        let block = self.alias();
        let rope = rope.alias();
        let positions = positions.clone();
        let trainable = self.trainable_parameters();
        let mut inputs: Vec<&Var<R>> = Vec::with_capacity(trainable.len() + 1);
        inputs.push(x);
        inputs.extend(trainable.iter().map(|(_, param)| *param));
        checkpoint_with_client(
            move |segment_inputs, client: &C| {
                let input = segment_inputs.first().ok_or_else(|| {
                    numr::error::Error::Internal(
                        "checkpointed qwen35_attention segment got no input".to_string(),
                    )
                })?;
                block
                    .forward_train(client, input, &rope, &positions)
                    .map_err(|e| {
                        numr::error::Error::Backend(format!(
                            "checkpointed Qwen35AttentionBlock forward: {e}"
                        ))
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
    use super::super::Qwen35AttentionProjection;
    use super::super::train_fixture::{
        Fixture, HIDDEN, Lcg, SEQ, cfg32, positions, rope, to_f64, weighted_loss,
    };
    use crate::nn::LoraFactors;
    use crate::test_utils::cpu_setup;
    use numr::autograd::{Var, backward};
    use numr::dtype::DType;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::{Tensor, TensorId};

    /// Checkpointed and plain forwards give the same output and the same
    /// gradient on `x`, every trainable weight, the norms and a LoRA pair.
    #[test]
    fn checkpointed_gradients_equal_plain_gradients() {
        let (client, device) = cpu_setup();
        let fixture = Fixture::f64(&cfg32(), 0x3a7e_0050);
        let mut block = fixture.block(&device, true);
        let mut rng = Lcg(0x3a7e_0051);
        let q_dim = fixture.cfg.q_dim();
        let mut t = |shape: &[usize]| {
            let data = rng.vec(shape.iter().product(), 0.3);
            Tensor::<CpuRuntime>::from_slice(&data, shape, &device).unwrap()
        };
        let factors = LoraFactors {
            lora_a: t(&[2, q_dim]),
            lora_b: t(&[HIDDEN, 2]),
            scaling: 0.5,
        };
        block
            .attach_adapter_trainable(
                &client,
                Qwen35AttentionProjection::Output,
                factors,
                DType::F64,
                true,
            )
            .unwrap();
        let x = fixture.x(&device, true);
        let (rope, pos) = (rope(&device), positions(SEQ, &device));
        let w_data = rng.vec(x.tensor().numel(), 1.0);
        let w = Var::new(
            Tensor::from_slice(&w_data, x.shape(), &device).unwrap(),
            false,
        );

        let mut ids: Vec<TensorId> = vec![x.id()];
        ids.extend(block.trainable_parameters().into_iter().map(|(id, _)| id));
        assert_eq!(
            ids.len(),
            1 + 3 + 1 + 2 + 2,
            "x, q/k/v, base o, lora pair, norms"
        );

        let plain = block.forward_train(&client, &x, &rope, &pos).unwrap();
        let ckpt = block
            .forward_train_checkpointed(&client, &x, &rope, &pos)
            .unwrap();
        assert_eq!(to_f64(plain.tensor()), to_f64(ckpt.tensor()));
        let g_plain = backward(&weighted_loss(&client, &plain, &w), &client).unwrap();
        let g_ckpt = backward(&weighted_loss(&client, &ckpt, &w), &client).unwrap();
        for id in ids {
            let a = to_f64(g_plain.get(id).expect("plain gradient"));
            let b = to_f64(g_ckpt.get(id).expect("checkpointed gradient"));
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() <= 1e-12 * (1.0 + x.abs()), "{x} vs {y}");
            }
        }
    }
}
