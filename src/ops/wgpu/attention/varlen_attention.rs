//! WebGPU implementation of VarLenAttentionOps
//!
//! Variable-length (packed) Flash Attention with cu_seqlens indexing.
//! F32 only (WebGPU limitation).

use crate::error::{Error, Result};
use crate::ops::impl_generic::attention::{StandardAttnConfig, standard_attention_bwd};
use crate::ops::traits::VarLenAttentionOps;
use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime, get_buffer};
use numr::tensor::Tensor;
use wgpu::BufferUsages;

/// Pack a packed-sequence slice `[s, H, D]` into the dense attention layout
/// `[1, H, s, D]`.
fn pack_seq(
    slice: &Tensor<WgpuRuntime>,
    heads: usize,
    s: usize,
    d: usize,
) -> Result<Tensor<WgpuRuntime>> {
    let hsd = slice.transpose(0, 1).map_err(Error::Numr)?.contiguous()?;
    hsd.reshape(&[1, heads, s, d]).map_err(Error::Numr)
}

/// Unpack a dense gradient `[1, H, s, D]` back to packed layout `[s, H, D]`.
fn unpack_seq(
    dense: &Tensor<WgpuRuntime>,
    heads: usize,
    s: usize,
    d: usize,
) -> Result<Tensor<WgpuRuntime>> {
    let hsd = dense.reshape(&[heads, s, d]).map_err(Error::Numr)?;
    hsd.transpose(0, 1)
        .map_err(Error::Numr)?
        .contiguous()
        .map_err(Error::Numr)
}

/// Concatenate per-sequence gradient pieces along the token axis, or return a
/// zero tensor of `full_shape` when there are no pieces (all sequences empty).
fn cat_token_parts(
    client: &WgpuClient,
    parts: &[Tensor<WgpuRuntime>],
    full_shape: &[usize],
    device: &<WgpuRuntime as numr::runtime::Runtime>::Device,
) -> Result<Tensor<WgpuRuntime>> {
    if parts.is_empty() {
        return Ok(Tensor::<WgpuRuntime>::zeros(full_shape, DType::F32, device));
    }
    let refs: Vec<&Tensor<WgpuRuntime>> = parts.iter().collect();
    client.cat(&refs, 0).map_err(Error::Numr)
}

const VARLEN_SHADER_SOURCE: &str = include_str!("../shaders/attention/varlen_attention.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct VarlenParams {
    total_tokens_q: u32,
    total_tokens_k: u32,
    num_heads: u32,
    head_dim: u32,
    batch_size: u32,
    causal: u32,
    scale: f32,
    _pad: u32,
}

fn validate_f32(t: &Tensor<WgpuRuntime>, op: &str) -> Result<()> {
    if t.dtype() != DType::F32 {
        return Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("{}: WebGPU requires F32, got {:?}", op, t.dtype()),
        });
    }
    Ok(())
}

impl VarLenAttentionOps<WgpuRuntime> for WgpuClient {
    fn varlen_attention_fwd(
        &self,
        q: &Tensor<WgpuRuntime>,
        k: &Tensor<WgpuRuntime>,
        v: &Tensor<WgpuRuntime>,
        cu_seqlens_q: &Tensor<WgpuRuntime>,
        cu_seqlens_k: &Tensor<WgpuRuntime>,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        _max_seqlen_q: usize,
        _max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        // WebGPU WGSL shader implements MHA only.  GQA requires non-trivial shader
        // changes; guard until a shader update is provided.
        if num_kv_heads != num_heads {
            return Err(Error::InvalidArgument {
                arg: "num_kv_heads",
                reason: format!(
                    "varlen_attention_fwd (WebGPU): GQA (num_kv_heads={num_kv_heads} != \
                     num_heads={num_heads}) not yet supported on WebGPU backend"
                ),
            });
        }
        validate_f32(q, "varlen_attention_fwd")?;
        validate_f32(k, "varlen_attention_fwd")?;
        validate_f32(v, "varlen_attention_fwd")?;
        // cu_seqlens are I32 — do NOT validate as F32

        let total_tokens_q = q.shape()[0];
        let total_tokens_k = k.shape()[0];

        // Create output tensors
        let output = Tensor::<WgpuRuntime>::zeros(q.shape(), DType::F32, q.device());
        let lse_shape = vec![total_tokens_q, num_heads];
        let lse = Tensor::<WgpuRuntime>::zeros(&lse_shape, DType::F32, q.device());

        // Get buffers
        let q_buf = get_buffer(q.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "q buffer not found".into(),
        })?;
        let k_buf = get_buffer(k.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "k buffer not found".into(),
        })?;
        let v_buf = get_buffer(v.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "v buffer not found".into(),
        })?;
        let cu_q_buf =
            get_buffer(cu_seqlens_q.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "cu_seqlens_q buffer not found".into(),
            })?;
        let cu_k_buf =
            get_buffer(cu_seqlens_k.storage().ptr()).ok_or_else(|| Error::KernelError {
                reason: "cu_seqlens_k buffer not found".into(),
            })?;
        let out_buf = get_buffer(output.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "output buffer not found".into(),
        })?;
        let lse_buf = get_buffer(lse.storage().ptr()).ok_or_else(|| Error::KernelError {
            reason: "lse buffer not found".into(),
        })?;

        let params = VarlenParams {
            total_tokens_q: total_tokens_q as u32,
            total_tokens_k: total_tokens_k as u32,
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            batch_size: batch_size as u32,
            causal: if causal { 1 } else { 0 },
            scale: 1.0f32 / (head_dim as f32).sqrt(),
            _pad: 0,
        };

        let params_buf = self.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("varlen_params"),
            size: std::mem::size_of::<VarlenParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.wgpu_queue()
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

        let cache = self.pipeline_cache();
        let module = cache.get_or_create_module("varlen_attention_fwd_f32", VARLEN_SHADER_SOURCE);

        let layout = cache.get_or_create_layout(numr::runtime::wgpu::shaders::LayoutKey {
            num_storage_buffers: 7,
            num_uniform_buffers: 1,
            num_readonly_storage: 5,
        });
        let pipeline = cache.get_or_create_pipeline(
            "varlen_attention_fwd_f32",
            "varlen_attention_fwd_f32",
            &module,
            &layout,
        );

        let bind_group = cache.create_bind_group(
            &layout,
            &[
                &q_buf,
                &k_buf,
                &v_buf,
                &cu_q_buf,
                &cu_k_buf,
                &out_buf,
                &lse_buf,
                &params_buf,
            ],
        );

        let workgroups = (total_tokens_q as u32).div_ceil(256);

        let mut encoder =
            self.wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("varlen_attention_fwd"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("varlen_attention_fwd"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.wgpu_queue().submit(std::iter::once(encoder.finish()));

        Ok((output, lse))
    }

    fn varlen_attention_bwd(
        &self,
        dout: &Tensor<WgpuRuntime>,
        q: &Tensor<WgpuRuntime>,
        k: &Tensor<WgpuRuntime>,
        v: &Tensor<WgpuRuntime>,
        output: &Tensor<WgpuRuntime>,
        _lse: &Tensor<WgpuRuntime>,
        cu_seqlens_q: &Tensor<WgpuRuntime>,
        cu_seqlens_k: &Tensor<WgpuRuntime>,
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        _max_seqlen_q: usize,
        _max_seqlen_k: usize,
        head_dim: usize,
        causal: bool,
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        validate_f32(dout, "varlen_attention_bwd")?;
        validate_f32(q, "varlen_attention_bwd")?;
        validate_f32(k, "varlen_attention_bwd")?;
        validate_f32(v, "varlen_attention_bwd")?;

        // Each packed sequence is sliced out, reshaped to the dense
        // `[1, H, s, D]` layout, and run through the shared standard-attention
        // backward (same algorithm as every other backend). The cu_seqlens are
        // small i32 offset tables — reading them host-side is metadata access,
        // not a GPU↔CPU transfer of the attention payload.
        let cu_q = cu_seqlens_q.to_vec::<i32>();
        let cu_k = cu_seqlens_k.to_vec::<i32>();
        let d = head_dim;

        let mut dq_parts = Vec::with_capacity(batch_size);
        let mut dk_parts = Vec::with_capacity(batch_size);
        let mut dv_parts = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let sq_start = cu_q[b] as usize;
            let s_q = cu_q[b + 1] as usize - sq_start;
            let sk_start = cu_k[b] as usize;
            let s_k = cu_k[b + 1] as usize - sk_start;
            if s_q == 0 || s_k == 0 {
                continue;
            }

            let q_seq = pack_seq(&q.narrow(0, sq_start, s_q)?, num_heads, s_q, d)?;
            let k_seq = pack_seq(&k.narrow(0, sk_start, s_k)?, num_kv_heads, s_k, d)?;
            let v_seq = pack_seq(&v.narrow(0, sk_start, s_k)?, num_kv_heads, s_k, d)?;
            let o_seq = pack_seq(&output.narrow(0, sq_start, s_q)?, num_heads, s_q, d)?;
            let do_seq = pack_seq(&dout.narrow(0, sq_start, s_q)?, num_heads, s_q, d)?;

            let cfg = StandardAttnConfig {
                num_heads,
                num_kv_heads,
                causal,
                window_size: 0,
            };
            let (dq_s, dk_s, dv_s) =
                standard_attention_bwd(self, &do_seq, &q_seq, &k_seq, &v_seq, &o_seq, cfg)?;

            dq_parts.push(unpack_seq(&dq_s, num_heads, s_q, d)?);
            dk_parts.push(unpack_seq(&dk_s, num_kv_heads, s_k, d)?);
            dv_parts.push(unpack_seq(&dv_s, num_kv_heads, s_k, d)?);
        }

        let device = q.device();
        let dq = cat_token_parts(self, &dq_parts, q.shape(), device)?;
        let dk = cat_token_parts(self, &dk_parts, k.shape(), device)?;
        let dv = cat_token_parts(self, &dv_parts, v.shape(), device)?;
        Ok((dq, dk, dv))
    }
}
