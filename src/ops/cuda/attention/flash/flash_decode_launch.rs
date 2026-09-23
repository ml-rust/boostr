//! Kernel launch plumbing for the S_q=1 decode path: kernel-name lookup,
//! module load, grid sizing and the two launch builders (whole-sequence and
//! split-KV). Called by the entry points in `flash_decode.rs`.

use crate::error::{Error, Result};
use crate::ops::cuda::kernels;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::super::decode_split::{
    DECODE_HEAD_DIMS, decode_dtype_suffix, decode_kv_span, decode_slices, decode_supports_head_dim,
};
use super::flash_utils::AttentionParams;

/// Kernel name stem for a supported decode `(head_dim, dtype)`.
pub(super) fn decode_kernel_stem(head_dim: usize, dtype: DType) -> Result<String> {
    if !decode_supports_head_dim(head_dim) {
        return Err(Error::InvalidArgument {
            arg: "head_dim",
            reason: format!(
                "decode attention supports head_dim {DECODE_HEAD_DIMS:?}, got {head_dim}"
            ),
        });
    }
    Ok(format!(
        "decode_attention_{head_dim}_{}",
        decode_dtype_suffix(dtype)?
    ))
}

/// Launches the decode kernels for `p.num_heads` rows per batch and returns
/// the output as `[B, num_heads, 1, D]` plus the LSE `[B, num_heads, 1]`.
///
/// `out_seq_fold` is the kernel's output-row rule: `1` stores row `bh` at
/// `bh * D`; `S_q > 1` means `num_heads` is a fold `H * S_q` and the kernel
/// stores row `h * S_q + s` at token-major `o[b, s, h, :]`. The returned
/// tensor's shape describes the bytes only in the `1` case; the folded
/// caller reshapes.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_decode(
    client: &CudaClient,
    q: &Tensor<CudaRuntime>,
    k: &Tensor<CudaRuntime>,
    v: &Tensor<CudaRuntime>,
    p: &AttentionParams,
    kv_seq_stride: usize,
    window_size: usize,
    kv_start: u64,
    kv_end: u64,
    out_seq_fold: usize,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let device = q.device();
    let device_index = device.id();
    let stem = decode_kernel_stem(p.head_dim, q.dtype())?;

    let module = kernels::get_or_load_module(
        client.context(),
        device_index,
        kernels::DECODE_ATTENTION_MODULE,
    )?;

    let output = Tensor::<CudaRuntime>::empty(
        &[p.batch_size, p.num_heads, 1, p.head_dim],
        q.dtype(),
        device,
    )?;
    let lse = Tensor::<CudaRuntime>::empty(&[p.batch_size, p.num_heads, 1], DType::F32, device)?;

    let base_blocks = p.batch_size * p.num_heads;
    // kv_end takes priority: when set, size against the buffer's full
    // capacity (`kv_seq_stride`), since a per-row end can extend up to it
    // regardless of `p.seq_len_k`. Null leaves every row ending at
    // `p.seq_len_k`, which is then the true span.
    let span_source = if kv_end == 0 {
        p.seq_len_k
    } else {
        kv_seq_stride
    };
    let slices = decode_slices(
        device_index,
        p.num_heads,
        decode_kv_span(span_source, window_size),
        p.head_dim,
    );
    let splits = slices.splits;

    let q_ptr = q.ptr();
    let k_ptr = k.ptr();
    let v_ptr = v.ptr();
    let o_ptr = output.ptr();
    let lse_ptr = lse.ptr();
    let nh_i32 = p.num_heads as i32;
    let nkv_i32 = p.num_kv_heads as i32;
    let sk_i32 = p.seq_len_k as i32;
    let stride_i32 = kv_seq_stride as i32;
    let window_i32 = window_size as i32;
    let fold_i32 = out_seq_fold as i32;
    let scale = (p.head_dim as f32).sqrt().recip();

    if splits > 1 {
        // Unnormalized per-slice accumulators plus their (m, l) statistics.
        let partial_o =
            Tensor::<CudaRuntime>::empty(&[base_blocks, splits, p.head_dim], DType::F32, device)?;
        let partial_ml =
            Tensor::<CudaRuntime>::empty(&[base_blocks, splits, 2], DType::F32, device)?;
        let po_ptr = partial_o.ptr();
        let pml_ptr = partial_ml.ptr();
        let splits_i32 = splits as i32;
        let fill_i32 = slices.fill as i32;

        let split_func = kernels::get_kernel_function(&module, &format!("{stem}_split"))?;
        let split_cfg = LaunchConfig {
            grid_dim: (base_blocks as u32, splits as u32, 1),
            block_dim: (p.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = client.stream().launch_builder(&split_func);
            builder.arg(&q_ptr);
            builder.arg(&k_ptr);
            builder.arg(&v_ptr);
            builder.arg(&po_ptr);
            builder.arg(&pml_ptr);
            builder.arg(&nh_i32);
            builder.arg(&nkv_i32);
            builder.arg(&sk_i32);
            builder.arg(&stride_i32);
            builder.arg(&scale);
            builder.arg(&window_i32);
            builder.arg(&kv_start);
            builder.arg(&kv_end);
            builder.arg(&fill_i32);
            builder.launch(split_cfg).map_err(|e| Error::KernelError {
                reason: format!("decode_attention split kernel launch failed: {:?}", e),
            })?;
        }

        let combine_func = kernels::get_kernel_function(&module, &format!("{stem}_combine"))?;
        let combine_cfg = LaunchConfig {
            grid_dim: (base_blocks as u32, 1, 1),
            block_dim: (p.head_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = client.stream().launch_builder(&combine_func);
            builder.arg(&po_ptr);
            builder.arg(&pml_ptr);
            builder.arg(&o_ptr);
            builder.arg(&lse_ptr);
            builder.arg(&splits_i32);
            builder.arg(&nh_i32);
            builder.arg(&fold_i32);
            builder
                .launch(combine_cfg)
                .map_err(|e| Error::KernelError {
                    reason: format!("decode_attention combine kernel launch failed: {:?}", e),
                })?;
        }

        return Ok((output, lse));
    }

    let func = kernels::get_kernel_function(&module, &stem)?;
    let cfg = LaunchConfig {
        grid_dim: (base_blocks as u32, 1, 1),
        block_dim: (p.head_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&q_ptr);
        builder.arg(&k_ptr);
        builder.arg(&v_ptr);
        builder.arg(&o_ptr);
        builder.arg(&lse_ptr);
        builder.arg(&nh_i32);
        builder.arg(&nkv_i32);
        builder.arg(&sk_i32);
        builder.arg(&stride_i32);
        builder.arg(&scale);
        builder.arg(&window_i32);
        builder.arg(&kv_start);
        builder.arg(&kv_end);
        builder.arg(&fold_i32);
        builder.launch(cfg).map_err(|e| Error::KernelError {
            reason: format!("decode_attention kernel launch failed: {:?}", e),
        })?;
    }

    Ok((output, lse))
}
