//! `quant_matmul_batch_rotated` on CUDA: the Hadamard rotation folded into
//! the feature-major MMQ activation producer at one token.
//!
//! At decode a rotated linear ran `fwht` and then the activation
//! quantization as two launches, with the rotated f32 row stored and
//! reloaded between them. `fwht_quantize_f32_q8_1_mmq`
//! (`kernels/fwht_quant_act.cu`) does both in one launch from shared
//! memory and forms the same record bytes; the kernel header states how.
//!
//! The fused path takes one token, an F32 activation, a `block_size` that
//! is a multiple of 128 (so every 128-value k-group of the record lies in
//! one segment) within the shared-memory budget, and weights that all
//! have a feature-major kernel at this `k`. Anything else runs the two
//! launches, which is what the trait's default does.

use crate::error::{Error, Result};
use crate::quant::QuantTensor;
use crate::quant::traits::Rotation;
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::LaunchConfig;
use numr::dtype::DType;
use numr::ops::FwhtOps;
use numr::runtime::Device;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use super::super::kernels::{self, FWHT_QUANT_ACT_MODULE};
use super::batched_gemv::{mmq_plan, mmq_run, quant_matmul_batch_impl};
use super::mmq_feat_major;

/// Widest segment the fused kernel holds in shared memory: the 48 KB every
/// device grants a block without opt-in, in f32.
const FWHT_MAX_BLOCK_SIZE: usize = 48 * 1024 / 4;

/// Threads per launch block, one segment per block.
const FWHT_THREADS: usize = 256;

/// Values per record k-group; a segment must hold whole k-groups.
const RECORD_KGROUP: usize = 128;

/// One-token rotated batch: the fused producer where it applies, else the
/// rotation and the batch as two steps.
pub(super) fn quant_matmul_batch_rotated_impl(
    client: &CudaClient,
    activation: &Tensor<CudaRuntime>,
    rotation: &Rotation<'_, CudaRuntime>,
    weights: &[&QuantTensor<CudaRuntime>],
) -> Result<Vec<Tensor<CudaRuntime>>> {
    if let Some(outputs) = fused(client, activation, rotation, weights)? {
        return Ok(outputs);
    }
    let rotated = client
        .fwht(activation, rotation.block_size, rotation.signs)
        .map_err(Error::Numr)?;
    quant_matmul_batch_impl(client, &rotated, weights)
}

/// `Ok(None)` when the fused kernel does not cover this call.
fn fused(
    client: &CudaClient,
    activation: &Tensor<CudaRuntime>,
    rotation: &Rotation<'_, CudaRuntime>,
    weights: &[&QuantTensor<CudaRuntime>],
) -> Result<Option<Vec<Tensor<CudaRuntime>>>> {
    let a_shape = activation.shape();
    let Some(&k) = a_shape.last() else {
        return Ok(None);
    };
    if weights.is_empty() || activation.dtype() != DType::F32 || k == 0 {
        return Ok(None);
    }
    let m = activation.numel() / k;
    let block_size = rotation.block_size;
    let segment_ok = m == 1
        && block_size.is_power_of_two()
        && block_size.is_multiple_of(RECORD_KGROUP)
        && block_size <= FWHT_MAX_BLOCK_SIZE
        && k.is_multiple_of(block_size);
    if !segment_ok {
        return Ok(None);
    }
    let signs = match rotation.signs {
        None => None,
        Some(s) => {
            if s.dtype() != DType::F32 || s.shape() != [k] {
                return Ok(None);
            }
            Some(s.contiguous()?)
        }
    };
    let device = activation.device();
    let device_index = device.id();
    let Some(formats) = mmq_plan(weights, m, k, device_index)? else {
        return Ok(None);
    };

    let act_contig = activation.contiguous()?;
    let slots = mmq_feat_major::shared_record_slots(&formats, m, device_index);
    let (q8_buf, ntok) = fwht_quantize_activation_q8_1_mmq(
        client,
        &act_contig,
        signs.as_ref(),
        k,
        block_size,
        slots as usize,
    )?;
    mmq_run(
        client,
        activation,
        weights,
        &formats,
        q8_buf.ptr(),
        ntok,
        m,
        k,
    )
    .map(Some)
}

/// Rotate one contiguous F32 row of `k` values by `signs` (contiguous,
/// `[k]`, or none) and the transform per `block_size`, and quantize it into
/// the feature-major MMQ record with `slots` token slots, in one launch.
/// Byte for byte the record `helpers::quantize_activation_q8_1_mmq` forms
/// from `fwht(row, block_size, signs)` at one token. Returns the buffer and
/// its token stride.
///
/// # Errors
///
/// [`Error::QuantError`] when `block_size` is not a power of two, a
/// multiple of 128 and at most 12288, or does not divide `k`.
pub fn fwht_quantize_activation_q8_1_mmq(
    client: &CudaClient,
    act_contig: &Tensor<CudaRuntime>,
    signs: Option<&Tensor<CudaRuntime>>,
    k: usize,
    block_size: usize,
    slots: usize,
) -> Result<(Tensor<CudaRuntime>, u32)> {
    let segment_ok = block_size.is_power_of_two()
        && block_size.is_multiple_of(RECORD_KGROUP)
        && block_size <= FWHT_MAX_BLOCK_SIZE
        && k.is_multiple_of(block_size);
    if !segment_ok {
        return Err(Error::QuantError {
            reason: format!(
                "fwht_quantize_f32_q8_1_mmq takes a power-of-two block_size that is a multiple \
                 of {RECORD_KGROUP}, at most {FWHT_MAX_BLOCK_SIZE} and dividing K; got \
                 block_size={block_size} K={k}"
            ),
        });
    }
    let device = act_contig.device();
    let device_index = device.id();
    let ntok = slots.max(1);
    let kgroups = k / RECORD_KGROUP;
    let bytes = kgroups * ntok * 144;
    let buf = Tensor::<CudaRuntime>::empty(&[bytes], DType::U8, device)?;

    let module =
        kernels::get_or_load_module(client.context(), device_index, FWHT_QUANT_ACT_MODULE)?;
    let func = kernels::get_kernel_function(&module, "fwht_quantize_f32_q8_1_mmq")?;

    let act_ptr = act_contig.ptr();
    let signs_ptr = signs.map(Tensor::ptr).unwrap_or(0);
    let out_ptr = buf.ptr();
    let k_u32 = k as u32;
    let block_size_u32 = block_size as u32;
    let ntok_u32 = ntok as u32;
    let n_segments = (k / block_size) as u32;

    let cfg = LaunchConfig {
        grid_dim: (n_segments, ntok_u32, 1),
        block_dim: (block_size.min(FWHT_THREADS) as u32, 1, 1),
        shared_mem_bytes: (block_size * 4) as u32,
    };

    unsafe {
        let mut builder = client.stream().launch_builder(&func);
        builder.arg(&act_ptr);
        builder.arg(&signs_ptr);
        builder.arg(&out_ptr);
        builder.arg(&k_u32);
        builder.arg(&block_size_u32);
        builder.arg(&ntok_u32);
        builder.launch(cfg).map_err(|e| Error::QuantError {
            reason: format!(
                "CUDA fwht_quantize_f32_q8_1_mmq launch failed at K={k} block_size={block_size} \
                 ntok={ntok}: {e:?}"
            ),
        })?;
    }

    Ok((buf, ntok_u32))
}
