//! Widening helpers for optimizer tensors narrower than F32 (BF16/F16).
//!
//! The CPU fused optimizers compute every narrow-dtype step in f32 and store
//! the result back narrow, matching what the CUDA `_bf16`/`_f16` kernels do.
//!
//! Requires the `f16` feature: `half::bf16` and `half::f16` only implement
//! numr's `Element` trait when `numr/f16` is on, and without it numr cannot
//! allocate, cast, or compute on BF16/F16 tensors either.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Read any float tensor into an f32 buffer.
///
/// Accepts mixed precision across a single optimizer group: a BF16 parameter
/// paired with F32 moments reads correctly without the caller matching dtypes.
pub(super) fn read_f32(t: &Tensor<CpuRuntime>, op: &str) -> Result<Vec<f32>> {
    match t.dtype() {
        DType::F32 => Ok(t.to_vec::<f32>()),
        DType::F64 => Ok(t.to_vec::<f64>().into_iter().map(|v| v as f32).collect()),
        DType::F16 => Ok(t
            .to_vec::<half::f16>()
            .into_iter()
            .map(|v| v.to_f32())
            .collect()),
        DType::BF16 => Ok(t
            .to_vec::<half::bf16>()
            .into_iter()
            .map(|v| v.to_f32())
            .collect()),
        dt => Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("{}: unsupported dtype {:?}", op, dt),
        }),
    }
}

/// Store an f32 buffer back as a tensor of `dtype`.
pub(super) fn write_narrow(
    data: &[f32],
    dtype: DType,
    shape: &[usize],
    device: &CpuDevice,
    op: &str,
) -> Result<Tensor<CpuRuntime>> {
    match dtype {
        DType::F16 => {
            let narrow: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
            Ok(Tensor::<CpuRuntime>::from_slice(&narrow, shape, device)?)
        }
        DType::BF16 => {
            let narrow: Vec<half::bf16> = data.iter().map(|&v| half::bf16::from_f32(v)).collect();
            Ok(Tensor::<CpuRuntime>::from_slice(&narrow, shape, device)?)
        }
        DType::F32 => Ok(Tensor::<CpuRuntime>::from_slice(data, shape, device)?),
        dt => Err(Error::InvalidArgument {
            arg: "dtype",
            reason: format!("{}: unsupported dtype {:?}", op, dt),
        }),
    }
}
