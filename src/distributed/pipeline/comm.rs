//! Activation send/recv helpers for pipeline parallelism.
//!
//! Wraps `comm_utils::send_tensor_with_metadata` / `recv_tensor_with_metadata`
//! with a tag scheme that distinguishes forward vs backward and micro-batch IDs.
//!
//! Tag scheme: `mb_id * 2` for forward activations, `mb_id * 2 + 1` for backward gradients.

use crate::distributed::comm_utils::{recv_tensor_with_metadata, send_tensor_with_metadata};
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::{Communicator, Runtime};
use numr::tensor::Tensor;

/// Compute the communication tag for a micro-batch activation transfer.
///
/// Forward: `base + mb_id * 2`, Backward: `base + mb_id * 2 + 1`.
/// `base` allows interleaved schedules with virtual stages to avoid tag collisions.
fn activation_tag(mb_id: usize, is_backward: bool, base: u32) -> Result<u32> {
    let offset = mb_id
        .checked_mul(2)
        .and_then(|v| {
            if is_backward {
                v.checked_add(1)
            } else {
                Some(v)
            }
        })
        .ok_or_else(|| Error::DistributedError {
            reason: format!("micro-batch id {mb_id} overflows tag range"),
        })?;
    let tag = base.checked_add(u32::try_from(offset).map_err(|_| Error::DistributedError {
        reason: format!("tag overflow for mb_id={mb_id}"),
    })?);
    tag.ok_or_else(|| Error::DistributedError {
        reason: format!("tag overflow for mb_id={mb_id} base={base}"),
    })
}

/// Send an activation tensor to a destination pipeline stage.
///
/// Uses the metadata-tagged protocol from `comm_utils` (sends shape header + data).
pub fn send_activation<R: Runtime<DType = DType>>(
    comm: &dyn Communicator,
    tensor: &Tensor<R>,
    dest: usize,
    mb_id: usize,
    is_backward: bool,
) -> Result<()> {
    let tag = activation_tag(mb_id, is_backward, 0)?;
    send_tensor_with_metadata(comm, tensor, dest, tag)
}

/// Receive an activation tensor from a source pipeline stage.
///
/// Allocates a new tensor with the received shape and dtype on `device`.
pub fn recv_activation<R: Runtime<DType = DType>>(
    comm: &dyn Communicator,
    src: usize,
    mb_id: usize,
    is_backward: bool,
    device: &R::Device,
) -> Result<Tensor<R>> {
    let tag = activation_tag(mb_id, is_backward, 0)?;
    recv_tensor_with_metadata::<R>(comm, src, tag, device)
}

/// Send with a base tag offset (for interleaved schedules with virtual stages).
pub fn send_activation_tagged<R: Runtime<DType = DType>>(
    comm: &dyn Communicator,
    tensor: &Tensor<R>,
    dest: usize,
    mb_id: usize,
    is_backward: bool,
    base_tag: u32,
) -> Result<()> {
    let tag = activation_tag(mb_id, is_backward, base_tag)?;
    send_tensor_with_metadata(comm, tensor, dest, tag)
}

/// Receive with a base tag offset (for interleaved schedules with virtual stages).
pub fn recv_activation_tagged<R: Runtime<DType = DType>>(
    comm: &dyn Communicator,
    src: usize,
    mb_id: usize,
    is_backward: bool,
    base_tag: u32,
    device: &R::Device,
) -> Result<Tensor<R>> {
    let tag = activation_tag(mb_id, is_backward, base_tag)?;
    recv_tensor_with_metadata::<R>(comm, src, tag, device)
}

/// Compute loss on the last pipeline stage and return the output gradient (ones tensor).
///
/// Extracts the saved forward output for `mb_id`, applies `loss_fn`, records the
/// scalar loss value, and returns a ones tensor as the initial gradient.
pub(super) fn compute_loss_grad<R: Runtime<DType = DType>>(
    forward_output: &mut Option<numr::autograd::Var<R>>,
    mb_id: usize,
    loss_fn: Option<&super::schedule_1f1b::LossFn<'_, R>>,
    losses: &mut Vec<f64>,
    device: &R::Device,
) -> Result<Tensor<R>> {
    let output = forward_output
        .take()
        .ok_or_else(|| Error::DistributedError {
            reason: format!("no saved output for micro-batch {mb_id}"),
        })?;

    let loss_fn = loss_fn.ok_or_else(|| Error::DistributedError {
        reason: "last stage requires loss_fn".to_string(),
    })?;

    let loss_var = loss_fn(&output)?;
    if let Ok(v) = loss_var.tensor().item::<f32>() {
        losses.push(v as f64);
    }

    Ok(Tensor::<R>::ones(
        loss_var.tensor().shape(),
        DType::F32,
        device,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_tag_forward() {
        assert_eq!(activation_tag(0, false, 0).unwrap(), 0);
        assert_eq!(activation_tag(1, false, 0).unwrap(), 2);
        assert_eq!(activation_tag(3, false, 0).unwrap(), 6);
    }

    #[test]
    fn test_activation_tag_backward() {
        assert_eq!(activation_tag(0, true, 0).unwrap(), 1);
        assert_eq!(activation_tag(1, true, 0).unwrap(), 3);
        assert_eq!(activation_tag(3, true, 0).unwrap(), 7);
    }

    #[test]
    fn test_activation_tag_with_base() {
        assert_eq!(activation_tag(0, false, 100).unwrap(), 100);
        assert_eq!(activation_tag(1, true, 100).unwrap(), 103);
    }
}
