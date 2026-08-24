//! Shared F32 master-weight plumbing for the optimizers.
//!
//! Every optimizer that supports parameters narrower than F32 repeats the same
//! four moves: build the master copy, widen the gradient, run its own kernel,
//! then write the stepped value back. Only the kernel call differs, so the
//! other three live here. See [`crate::optimizer::precision`] for WHY the
//! master copy exists at all.

use crate::error::Result;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Build the F32 master copy for a narrow parameter, or `None` when the
/// parameter is already at the optimizer's state dtype.
///
/// The update runs against the master and a cast of the master is written back
/// into the caller's `params` map, so the model keeps computing in its own
/// dtype while the update arithmetic stays F32. For an F32 or F64 parameter
/// this is `None`: no copy, no extra allocation, and the numbers are
/// bit-identical to a build without master weights.
///
/// The master is also what must be handed to the kernels: they mutate the
/// buffers they are given in place (CUDA and WGPU write through a
/// storage-sharing clone of the parameter), so the narrow parameter must never
/// go in. Even a kernel that does not write the parameter — LAMB — allocates
/// its update vector at the parameter's dtype, which would narrow the whole
/// update.
pub(crate) fn init_master<R, C>(
    client: &C,
    param: &Tensor<R>,
    state_dtype: DType,
) -> Result<Option<Tensor<R>>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TypeConversionOps<R>,
{
    if state_dtype == param.dtype() {
        Ok(None)
    } else {
        Ok(Some(client.cast(param, state_dtype)?))
    }
}

/// Widen a gradient to the optimizer state dtype, or `None` when it already
/// matches.
///
/// CPU dispatch keys on the parameter dtype alone and reads the gradient at
/// that width, so a narrow gradient against an F32 master must be widened
/// first. The caller keeps the returned tensor alive only for the current
/// parameter, so only one widened gradient is resident at a time — batching
/// them would hold an F32 copy of EVERY gradient at once.
pub(crate) fn widen_grad<R, C>(
    client: &C,
    grad: &Tensor<R>,
    state_dtype: DType,
) -> Result<Option<Tensor<R>>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TypeConversionOps<R>,
{
    if grad.dtype() == state_dtype {
        Ok(None)
    } else {
        Ok(Some(client.cast(grad, state_dtype)?))
    }
}

/// Write a stepped value back: update the master in place and return the tensor
/// to store in the caller's `params` map.
///
/// On the narrow path the caller's map receives a FRESH cast of the master,
/// never the handle the kernel returned: on CUDA and WGPU that handle aliases
/// the master's storage. On the F32 path there is no master and the stepped
/// tensor is returned unchanged, so that path stays bit-identical to a build
/// without master weights.
pub(crate) fn write_back<R, C>(
    client: &C,
    master: Option<&mut Tensor<R>>,
    stepped: Tensor<R>,
    param_dtype: DType,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TypeConversionOps<R>,
{
    match master {
        Some(master) => {
            *master = stepped;
            Ok(client.cast(master, param_dtype)?)
        }
        None => Ok(stepped),
    }
}
