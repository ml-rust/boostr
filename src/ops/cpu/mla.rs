//! CPU implementation of MlaOps — delegates to impl_generic

use crate::error::Result;
use crate::ops::impl_generic::mla::scaled_dot_product_attention_impl;
use crate::ops::traits::MlaOps;
use numr::autograd::Var;
use numr::runtime::cpu::{CpuClient, CpuRuntime};

impl MlaOps<CpuRuntime> for CpuClient {
    fn scaled_dot_product_attention(
        &self,
        q: &Var<CpuRuntime>,
        k: &Var<CpuRuntime>,
        v: &Var<CpuRuntime>,
        scale: f64,
        causal: bool,
    ) -> Result<Var<CpuRuntime>> {
        scaled_dot_product_attention_impl(self, q, k, v, scale, causal)
    }
}
