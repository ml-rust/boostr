//! WebGPU stub for `FlashAlibiOps` — no WGSL shader exists for the fused
//! ALiBi Flash Attention kernel, mirroring how `FlashAttentionOps::flash_attention_fwd_fp8_kv`
//! is stubbed on this backend in `flash.rs`.

use crate::error::{Error, Result};
use crate::ops::traits::FlashAlibiOps;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl FlashAlibiOps<WgpuRuntime> for WgpuClient {
    fn flash_attention_fwd_alibi(
        &self,
        _q: &Tensor<WgpuRuntime>,
        _k: &Tensor<WgpuRuntime>,
        _v: &Tensor<WgpuRuntime>,
        _num_heads: usize,
        _head_dim: usize,
        _causal: bool,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        Err(Error::InvalidArgument {
            arg: "op",
            reason: "flash_attention_fwd_alibi not implemented on WebGPU: no WGSL shader for it"
                .into(),
        })
    }
}
