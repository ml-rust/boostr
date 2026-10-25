//! [`HybridModel`] hidden-state forward for embedding extraction.

use super::build::HybridModel;
use crate::error::Result;
use crate::inference::kv_cache::layered::LayeredKvCacheConfig;
use crate::inference::{LayeredKvCache, LayeredSsmState};
use crate::model::traits::ModelClient;
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ConvOps, IndexingOps, NormalizationOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> HybridModel<R>
where
    R::Client: IndexingOps<R>,
{
    /// Contextualized hidden states for embedding extraction.
    ///
    /// Runs embed + all attention/SSM blocks + final norm (no `lm_head`) with
    /// fresh throwaway KV cache and SSM state. Returns shape `[B, S, hidden]`.
    pub fn forward_hidden<C>(&self, client: &C, input_ids: &Tensor<R>) -> Result<Var<R>>
    where
        C: ModelClient<R> + ConvOps<R> + NormalizationOps<R> + UnaryOps<R> + ActivationOps<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ActivationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + DequantOps<R>,
    {
        let shape = input_ids.shape();
        let (batch, seq_len) = (shape[0], shape[1]);
        let device = input_ids.device();
        let dtype = self.embed_tokens.weight().tensor().dtype();
        let (num_kv_heads, head_dim) = self.kv_cache_dims().unwrap_or((1, 1));
        let kv_config = LayeredKvCacheConfig {
            batch_size: batch,
            num_kv_heads,
            initial_capacity: seq_len,
            max_seq_len: self.config.max_seq_len,
            head_dim,
            dtype,
        };
        let mut kv_cache =
            LayeredKvCache::<R>::new(self.num_attention_layers(), &kv_config, device)?;
        let mut ssm_state = match &self.mamba_config {
            Some(mamba) => Some(LayeredSsmState::<R>::new(
                self.num_ssm_layers(),
                batch,
                mamba,
                dtype,
                device,
            )?),
            None => None,
        };
        self.run_layers(client, input_ids, &mut kv_cache, ssm_state.as_mut(), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::super::forward::tests::{checkpoint, config, ids, model};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    #[test]
    fn forward_hidden_runs_an_mla_model() {
        let (client, dev) = cpu_setup();
        let config = config(&[true, false], false);
        let tensors = checkpoint(&config, &dev);
        let m = model(&config, &tensors);
        let input = Tensor::<CpuRuntime>::from_slice(&ids(5), &[1, 5], &dev).unwrap();
        let hidden = m.forward_hidden(&client, &input).unwrap();
        assert_eq!(hidden.shape(), &[1, 5, 16]);
    }
}
