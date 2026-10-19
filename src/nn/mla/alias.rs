//! [`Mla::alias`]: a cheap duplicate that keeps every parameter's `TensorId`.

use super::Mla;
use crate::nn::{MaybeQuantLinear, RmsNorm};
use numr::runtime::Runtime;

impl<R: Runtime> Mla<R> {
    /// Cheap duplicate that keeps every parameter's `TensorId`, for capture in a
    /// `'static` activation-checkpointing closure. Each child goes through its
    /// own `alias`, never [`Clone`]: a `clone` mints fresh ids and orphans the gradients.
    pub fn alias(&self) -> Self {
        Self {
            q_down: self.q_down.as_ref().map(MaybeQuantLinear::alias),
            q_up: self.q_up.alias(),
            q_norm: self.q_norm.as_ref().map(RmsNorm::alias),
            kv_compress: self.kv_compress.alias(),
            kv_norm: self.kv_norm.as_ref().map(RmsNorm::alias),
            kv_decompress: self.kv_decompress.alias(),
            o_proj: self.o_proj.alias(),
            rope: self.rope.alias(),
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            head_dim_v: self.head_dim_v,
            rope_head_dim: self.rope_head_dim,
            kv_lora_rank: self.kv_lora_rank,
            scale: self.scale,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::MlaConfig;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::TensorId;

    /// `alias()` keeps each parameter's `TensorId` and its trainability.
    #[test]
    fn alias_preserves_parameter_ids() {
        let device = CpuDevice::new();
        let config = MlaConfig::deepseek_v2(16, 2, 8, 8, 4, 16);
        let mla = Mla::<CpuRuntime>::from_config(&config, &device).unwrap();

        let aliased = mla.alias();

        let ids = |m: &Mla<CpuRuntime>| -> Vec<(TensorId, bool)> {
            m.parameters()
                .into_iter()
                .map(|(id, var)| (id, var.requires_grad()))
                .collect()
        };
        assert_eq!(ids(&mla).len(), 7);
        assert_eq!(ids(&mla), ids(&aliased));
    }
}
