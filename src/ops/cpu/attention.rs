//! CPU implementation of AttentionOps — delegates to impl_generic

use crate::error::Result;
use crate::ops::impl_generic::attention::multi_head_attention_impl;
use crate::ops::traits::AttentionOps;
use numr::autograd::Var;
use numr::runtime::cpu::{CpuClient, CpuRuntime};

impl AttentionOps<CpuRuntime> for CpuClient {
    fn multi_head_attention(
        &self,
        q: &Var<CpuRuntime>,
        k: &Var<CpuRuntime>,
        v: &Var<CpuRuntime>,
        mask: Option<&Var<CpuRuntime>>,
        num_heads: usize,
    ) -> Result<Var<CpuRuntime>> {
        multi_head_attention_impl(self, q, k, v, mask, num_heads)
    }
}
