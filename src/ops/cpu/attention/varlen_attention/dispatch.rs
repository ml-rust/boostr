//! `VarLenAttentionOps` for `CpuClient`: checks inputs, stages them as F32,
//! runs the kernels, and casts the results back to the input dtype.

use crate::error::{Error, Result};
use crate::ops::traits::VarLenAttentionOps;
use crate::ops::traits::attention::packed_segments::validate_host_cu_seqlens;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use super::backward::varlen_attention_bwd_cpu;
use super::check::{float_dtype, head_counts, offsets_i32, packed_shape};
use super::forward::varlen_attention_fwd_cpu;

/// Cast to F32 with contiguous storage. An F32 contiguous input is a refcount clone.
fn stage_f32(client: &CpuClient, t: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    Ok(client.cast(t, DType::F32)?.contiguous()?)
}

/// Check q/k/v layout and return `(total_tokens_q, total_tokens_k)`.
fn check_qkv(
    op: &'static str,
    q: &Tensor<CpuRuntime>,
    k: &Tensor<CpuRuntime>,
    v: &Tensor<CpuRuntime>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(usize, usize)> {
    head_counts(op, num_heads, num_kv_heads)?;
    let tq = packed_shape(op, "q", q, num_heads, head_dim)?;
    let tk = packed_shape(op, "k", k, num_kv_heads, head_dim)?;
    let tv = packed_shape(op, "v", v, num_kv_heads, head_dim)?;
    if tv != tk {
        return Err(Error::InvalidArgument {
            arg: "v",
            reason: format!("{op} (CPU): v holds {tv} tokens, k holds {tk}; they must match"),
        });
    }
    Ok((tq, tk))
}

impl VarLenAttentionOps<CpuRuntime> for CpuClient {
    fn varlen_attention_fwd(
        &self,
        q: &Tensor<CpuRuntime>,
        k: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        cu_seqlens_q: &Tensor<CpuRuntime>,
        cu_seqlens_k: &Tensor<CpuRuntime>,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        _max_seqlen_q: usize,
        _max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
        window_size: usize,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        const OP: &str = "varlen_attention_fwd";
        let dtype = float_dtype(OP, &[("q", q), ("k", k), ("v", v)])?;
        let (tq, tk) = check_qkv(OP, q, k, v, num_heads, num_kv_heads, head_dim)?;
        offsets_i32(OP, "cu_seqlens_q", cu_seqlens_q)?;
        offsets_i32(OP, "cu_seqlens_k", cu_seqlens_k)?;
        // CPU tensors live in host memory, so this read is no device transfer.
        let cu_q = cu_seqlens_q.contiguous()?.try_to_vec::<i32>()?;
        let cu_k = cu_seqlens_k.contiguous()?.try_to_vec::<i32>()?;
        validate_host_cu_seqlens("cu_seqlens_q", &cu_q, batch_size, tq)?;
        validate_host_cu_seqlens("cu_seqlens_k", &cu_k, batch_size, tk)?;

        let (out, lse) = varlen_attention_fwd_cpu(
            &stage_f32(self, q)?,
            &stage_f32(self, k)?,
            &stage_f32(self, v)?,
            &cu_q,
            &cu_k,
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
            window_size,
        )?;
        Ok((self.cast(&out, dtype)?, lse))
    }

    fn varlen_attention_bwd(
        &self,
        dout: &Tensor<CpuRuntime>,
        q: &Tensor<CpuRuntime>,
        k: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        output: &Tensor<CpuRuntime>,
        lse: &Tensor<CpuRuntime>,
        _cu_seqlens_q: &Tensor<CpuRuntime>,
        _cu_seqlens_k: &Tensor<CpuRuntime>,
        host_cu_seqlens_q: &[i32],
        host_cu_seqlens_k: &[i32],
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        _max_seqlen_q: usize,
        _max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
        window_size: usize,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        const OP: &str = "varlen_attention_bwd";
        let dtype = float_dtype(
            OP,
            &[
                ("q", q),
                ("k", k),
                ("v", v),
                ("dout", dout),
                ("output", output),
            ],
        )?;
        let (tq, tk) = check_qkv(OP, q, k, v, num_heads, num_kv_heads, head_dim)?;
        packed_shape(OP, "dout", dout, num_heads, head_dim)?;
        packed_shape(OP, "output", output, num_heads, head_dim)?;
        if lse.dtype() != DType::F32 || lse.shape() != [tq, num_heads] {
            return Err(Error::InvalidArgument {
                arg: "lse",
                reason: format!(
                    "{OP} (CPU): lse must be F32 [{tq}, {num_heads}], got {:?} {:?}",
                    lse.dtype(),
                    lse.shape()
                ),
            });
        }
        validate_host_cu_seqlens("host_cu_seqlens_q", host_cu_seqlens_q, batch_size, tq)?;
        validate_host_cu_seqlens("host_cu_seqlens_k", host_cu_seqlens_k, batch_size, tk)?;

        let (dq, dk, dv) = varlen_attention_bwd_cpu(
            &stage_f32(self, dout)?,
            &stage_f32(self, q)?,
            &stage_f32(self, k)?,
            &stage_f32(self, v)?,
            &stage_f32(self, output)?,
            &lse.contiguous()?,
            host_cu_seqlens_q,
            host_cu_seqlens_k,
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
            window_size,
        )?;
        Ok((
            self.cast(&dq, dtype)?,
            self.cast(&dk, dtype)?,
            self.cast(&dv, dtype)?,
        ))
    }
}
