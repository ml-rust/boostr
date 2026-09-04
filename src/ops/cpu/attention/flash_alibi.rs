//! CPU reference for `FlashAlibiOps`: `softmax(Q@K^T * scale + alibi_bias) @ V`.
//!
//! Reuses the existing ALiBi bias ops (`AlibiOps::alibi_add_bias` /
//! `alibi_add_bias_causal`, `ops/cpu/position/alibi.rs`) rather than a third
//! copy of the slope formula — its slope (`2^(-8h/H)`) and sign convention
//! (`-slope * |qi - ki|`) already match `flash_attention_alibi_fp32_impl` in
//! `alibi.cu`. Not `standard_attention_fwd`-based: that helper has no bias
//! input, so this composes the same matmul/softmax/logsumexp primitives by
//! hand with the ALiBi bias spliced in between the score scale and the
//! softmax, matching the fused kernel's order of operations.

use crate::error::{Error, Result};
use crate::ops::traits::{AlibiOps, FlashAlibiOps};
use numr::dtype::DType;
use numr::ops::{ActivationOps, CumulativeOps, MatmulOps, ScalarOps, TypeConversionOps};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl FlashAlibiOps<CpuRuntime> for CpuClient {
    fn flash_attention_fwd_alibi(
        &self,
        q: &Tensor<CpuRuntime>,
        k: &Tensor<CpuRuntime>,
        v: &Tensor<CpuRuntime>,
        num_heads: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        for (name, t) in [("q", q), ("k", k), ("v", v)] {
            if t.dtype() != DType::F32 {
                return Err(Error::InvalidArgument {
                    arg: name,
                    reason: format!(
                        "flash_attention_fwd_alibi requires {name} in F32, got {:?}",
                        t.dtype()
                    ),
                });
            }
        }

        let q_shape = q.shape();
        if q_shape.len() != 4 {
            return Err(Error::InvalidArgument {
                arg: "q",
                reason: format!("expected 4D [B, H, S, D], got {}D", q_shape.len()),
            });
        }
        let (batch_size, q_heads, seq_len_q, q_head_dim) =
            (q_shape[0], q_shape[1], q_shape[2], q_shape[3]);
        if q_heads != num_heads || q_head_dim != head_dim {
            return Err(Error::InvalidArgument {
                arg: "q",
                reason: format!(
                    "q shape [{batch_size}, {q_heads}, {seq_len_q}, {q_head_dim}] does not match \
                     num_heads={num_heads} head_dim={head_dim}"
                ),
            });
        }
        if !matches!(head_dim, 64 | 128) {
            return Err(Error::InvalidArgument {
                arg: "head_dim",
                reason: format!(
                    "flash_attention_fwd_alibi supports head_dim 64 or 128, got {head_dim}"
                ),
            });
        }

        let k_shape = k.shape().to_vec();
        if k_shape.len() != 4
            || k_shape[0] != batch_size
            || k_shape[1] != num_heads
            || k_shape[3] != head_dim
        {
            return Err(Error::InvalidArgument {
                arg: "k",
                reason: format!(
                    "expected [{batch_size}, {num_heads}, seq_len_k, {head_dim}], got {k_shape:?}"
                ),
            });
        }
        let seq_len_k = k_shape[2];
        if v.shape() != k_shape.as_slice() {
            return Err(Error::InvalidArgument {
                arg: "v",
                reason: format!("v shape {:?} must match k shape {:?}", v.shape(), k_shape),
            });
        }

        let scale = (head_dim as f32).sqrt().recip();

        // Q @ K^T * scale -> [B, H, S_q, S_k]
        let k_t = k.transpose(-2, -1).map_err(Error::Numr)?.contiguous()?;
        let scores = self.matmul(q, &k_t).map_err(Error::Numr)?;
        let scores = self
            .mul_scalar(&scores, scale as f64)
            .map_err(Error::Numr)?;

        // Splice in the ALiBi bias in-place, before softmax — same order as
        // the fused kernel (score = Q@K^T*scale, then += alibi_bias).
        //
        // Bottom-right causal masking: `position` shifts the query position so a
        // `seq_len_q`-query decode step lines up with the LAST `seq_len_q`
        // positions of the key sequence, matching `key_offset` in
        // `flash_attention_alibi_fp32_impl` (`alibi.cu`). `seq_len_q == seq_len_k`
        // gives `position == 0` and leaves this unchanged.
        if causal {
            let position = seq_len_k.saturating_sub(seq_len_q);
            self.alibi_add_bias_causal(
                &scores, batch_size, num_heads, seq_len_q, seq_len_k, position,
            )?;
        } else {
            self.alibi_add_bias(&scores, batch_size, num_heads, seq_len_q, seq_len_k)?;
        }

        // Log-sum-exp for the backward pass / kernel-parity `L`: F32,
        // [B, H, S_q]. Matches `L_base[...] = m_local + logf(l_local)` in
        // the kernel — max plus log-sum-of-exp, not the bare denominator.
        let lse = self.logsumexp(&scores, &[3], false).map_err(Error::Numr)?;
        let lse = if lse.dtype() != DType::F32 {
            self.cast(&lse, DType::F32).map_err(Error::Numr)?
        } else {
            lse
        };

        let weights = self.softmax(&scores, -1).map_err(Error::Numr)?;
        let output = self.matmul(&weights, v).map_err(Error::Numr)?;

        Ok((output, lse))
    }
}
