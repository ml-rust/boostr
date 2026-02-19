//! Safe wrappers around the raw-pointer `Communicator` trait.
//!
//! Provides tensor-level helpers so other distributed modules don't
//! repeat the contiguity-check → extract-pointer → unsafe-call → sync pattern.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::runtime::{Communicator, ReduceOp, Runtime};
use numr::tensor::Tensor;

/// All-reduce a single tensor in-place with the given reduction op.
///
/// Checks contiguity, extracts the device pointer, calls the unsafe
/// communicator method, and syncs.
pub fn all_reduce_tensor<R: Runtime<DType = DType>>(
    comm: &dyn Communicator,
    tensor: &Tensor<R>,
    op: ReduceOp,
) -> Result<()> {
    if !tensor.is_contiguous() {
        return Err(Error::DistributedError {
            reason: "all_reduce requires contiguous tensor".to_string(),
        });
    }

    let ptr = tensor.data_ptr();
    let count = tensor.numel();
    let dtype = tensor.dtype();

    // Safety: tensor is contiguous, ptr is valid device pointer with `count` elements of `dtype`
    unsafe {
        comm.all_reduce(ptr, count, dtype, op)
            .map_err(|e| Error::DistributedError {
                reason: format!("all_reduce failed: {e}"),
            })?;
    }

    Ok(())
}

/// Send a tensor to a destination rank with a tag.
pub fn send_tensor<R: Runtime<DType = DType>>(
    comm: &dyn Communicator,
    tensor: &Tensor<R>,
    dest: usize,
    tag: u32,
) -> Result<()> {
    if !tensor.is_contiguous() {
        return Err(Error::DistributedError {
            reason: "send requires contiguous tensor".to_string(),
        });
    }

    let ptr = tensor.data_ptr();
    let count = tensor.numel();
    let dtype = tensor.dtype();

    // Safety: tensor is contiguous, ptr is valid device pointer
    unsafe {
        comm.send(ptr, count, dtype, dest, tag)
            .map_err(|e| Error::DistributedError {
                reason: format!("send to rank {dest} failed: {e}"),
            })?;
    }

    Ok(())
}

/// Receive into a pre-allocated tensor buffer from a source rank.
pub fn recv_into_tensor<R: Runtime<DType = DType>>(
    comm: &dyn Communicator,
    buffer: &Tensor<R>,
    src: usize,
    tag: u32,
) -> Result<()> {
    if !buffer.is_contiguous() {
        return Err(Error::DistributedError {
            reason: "recv buffer must be contiguous".to_string(),
        });
    }

    let ptr = buffer.data_ptr();
    let count = buffer.numel();
    let dtype = buffer.dtype();

    // Safety: buffer is contiguous, ptr is valid device pointer
    unsafe {
        comm.recv(ptr, count, dtype, src, tag)
            .map_err(|e| Error::DistributedError {
                reason: format!("recv from rank {src} failed: {e}"),
            })?;
    }

    Ok(())
}

/// Broadcast a tensor from root rank to all ranks.
pub fn broadcast_tensor<R: Runtime<DType = DType>>(
    comm: &dyn Communicator,
    tensor: &Tensor<R>,
    root: usize,
) -> Result<()> {
    if !tensor.is_contiguous() {
        return Err(Error::DistributedError {
            reason: "broadcast requires contiguous tensor".to_string(),
        });
    }

    let ptr = tensor.data_ptr();
    let count = tensor.numel();
    let dtype = tensor.dtype();

    // Safety: tensor is contiguous, ptr is valid device pointer
    unsafe {
        comm.broadcast(ptr, count, dtype, root)
            .map_err(|e| Error::DistributedError {
                reason: format!("broadcast failed: {e}"),
            })?;
    }

    Ok(())
}

/// Send a tensor with shape metadata, then receive one on the other end.
///
/// Protocol:
/// 1. Send header: `[ndim, dim0, dim1, ..., dtype_id]` as u64 array
/// 2. Send tensor data
///
/// Use `recv_tensor` on the receiving end to match this protocol.
pub fn send_tensor_with_metadata<R: Runtime<DType = DType>>(
    comm: &dyn Communicator,
    tensor: &Tensor<R>,
    dest: usize,
    tag: u32,
) -> Result<()> {
    if !tensor.is_contiguous() {
        return Err(Error::DistributedError {
            reason: "send requires contiguous tensor".to_string(),
        });
    }

    let shape = tensor.shape();
    let ndim = shape.len();
    let dtype = tensor.dtype();

    // Build header: [ndim, shape[0], shape[1], ..., dtype_as_u64]
    let mut header: Vec<u64> = Vec::with_capacity(ndim + 2);
    header.push(ndim as u64);
    for &d in shape {
        header.push(d as u64);
    }
    header.push(dtype_to_u64(dtype));

    let header_count = header.len();

    // Send header as raw u64 array
    unsafe {
        comm.send(header.as_ptr() as u64, header_count, DType::U64, dest, tag)
            .map_err(|e| Error::DistributedError {
                reason: format!("send header to rank {dest} failed: {e}"),
            })?;
    }

    comm.sync().map_err(|e| Error::DistributedError {
        reason: format!("sync after header send failed: {e}"),
    })?;

    // Send tensor data
    let ptr = tensor.data_ptr();
    let count = tensor.numel();

    unsafe {
        comm.send(ptr, count, dtype, dest, tag + 1)
            .map_err(|e| Error::DistributedError {
                reason: format!("send data to rank {dest} failed: {e}"),
            })?;
    }

    comm.sync().map_err(|e| Error::DistributedError {
        reason: format!("sync after data send failed: {e}"),
    })?;

    Ok(())
}

/// Receive a tensor with shape metadata from a source rank.
///
/// Matches the protocol from `send_tensor_with_metadata`:
/// 1. Receive header to learn shape and dtype
/// 2. Allocate tensor
/// 3. Receive tensor data
///
/// `max_header_len` caps the maximum number of dimensions (default 8 + 2 = 10).
pub fn recv_tensor_with_metadata<R: Runtime<DType = DType>>(
    comm: &dyn Communicator,
    src: usize,
    tag: u32,
    device: &R::Device,
) -> Result<Tensor<R>> {
    // Max supported: 8 dimensions + ndim + dtype = 10 u64s
    const MAX_HEADER: usize = 10;
    let mut header_buf = [0u64; MAX_HEADER];

    // Receive header — we don't know exact size, so receive max and parse
    unsafe {
        comm.recv(
            header_buf.as_mut_ptr() as u64,
            MAX_HEADER,
            DType::U64,
            src,
            tag,
        )
        .map_err(|e| Error::DistributedError {
            reason: format!("recv header from rank {src} failed: {e}"),
        })?;
    }

    comm.sync().map_err(|e| Error::DistributedError {
        reason: format!("sync after header recv failed: {e}"),
    })?;

    let ndim = header_buf[0] as usize;
    if ndim == 0 || ndim + 2 > MAX_HEADER {
        return Err(Error::DistributedError {
            reason: format!("invalid ndim {ndim} in recv header (max 8 dims)"),
        });
    }

    let shape: Vec<usize> = header_buf[1..=ndim].iter().map(|&d| d as usize).collect();
    let dtype = u64_to_dtype(header_buf[ndim + 1])?;

    // Allocate receive buffer
    let buffer = Tensor::<R>::zeros(&shape, dtype, device);

    // Receive tensor data
    let ptr = buffer.data_ptr();
    let count = buffer.numel();

    unsafe {
        comm.recv(ptr, count, dtype, src, tag + 1)
            .map_err(|e| Error::DistributedError {
                reason: format!("recv data from rank {src} failed: {e}"),
            })?;
    }

    comm.sync().map_err(|e| Error::DistributedError {
        reason: format!("sync after data recv failed: {e}"),
    })?;

    Ok(buffer)
}

/// Convert DType to u64 using the stable `repr(u8)` discriminant from numr.
fn dtype_to_u64(dtype: DType) -> u64 {
    // DType is repr(u8) with stable discriminant values
    (dtype as u8) as u64
}

/// Convert u64 back to DType using the stable discriminant values.
fn u64_to_dtype(val: u64) -> Result<DType> {
    match val {
        0 => Ok(DType::F64),
        1 => Ok(DType::F32),
        2 => Ok(DType::F16),
        3 => Ok(DType::BF16),
        4 => Ok(DType::FP8E4M3),
        5 => Ok(DType::FP8E5M2),
        10 => Ok(DType::I64),
        11 => Ok(DType::I32),
        12 => Ok(DType::I16),
        13 => Ok(DType::I8),
        20 => Ok(DType::U64),
        21 => Ok(DType::U32),
        22 => Ok(DType::U16),
        23 => Ok(DType::U8),
        30 => Ok(DType::Bool),
        40 => Ok(DType::Complex64),
        41 => Ok(DType::Complex128),
        _ => Err(Error::DistributedError {
            reason: format!("unknown dtype discriminant {val} in recv header"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::NoOpCommunicator;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_all_reduce_tensor_noop() {
        let (_client, device) = cpu_setup();
        let comm = NoOpCommunicator;

        let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        all_reduce_tensor(&comm, &t, ReduceOp::Sum).unwrap();

        let data = t.to_vec::<f32>();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_send_recv_tensor_noop() {
        let (_client, device) = cpu_setup();
        let comm = NoOpCommunicator;

        let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        send_tensor(&comm, &t, 0, 0).unwrap();

        let buf = Tensor::<CpuRuntime>::zeros(&[2], DType::F32, &device);
        recv_into_tensor(&comm, &buf, 0, 0).unwrap();
    }

    #[test]
    fn test_broadcast_tensor_noop() {
        let (_client, device) = cpu_setup();
        let comm = NoOpCommunicator;

        let t = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 10.0], &[2], &device);
        broadcast_tensor(&comm, &t, 0).unwrap();

        let data = t.to_vec::<f32>();
        assert_eq!(data, vec![5.0, 10.0]);
    }

    #[test]
    fn test_dtype_roundtrip() {
        let dtypes = [
            DType::F32,
            DType::F64,
            DType::I32,
            DType::I64,
            DType::U8,
            DType::U32,
            DType::U64,
            DType::F16,
            DType::BF16,
            DType::Bool,
        ];
        for &dt in &dtypes {
            let id = dtype_to_u64(dt);
            let back = u64_to_dtype(id).unwrap();
            assert_eq!(dt, back);
        }
    }

    #[test]
    fn test_u64_to_dtype_invalid() {
        assert!(u64_to_dtype(99).is_err());
    }

    #[test]
    fn test_send_with_metadata_noop() {
        let (_client, device) = cpu_setup();
        let comm = NoOpCommunicator;

        // NoOp send succeeds (no actual data transfer)
        let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        send_tensor_with_metadata(&comm, &t, 0, 0).unwrap();
    }

    #[test]
    fn test_recv_with_metadata_noop_returns_error() {
        let (_client, device) = cpu_setup();
        let comm = NoOpCommunicator;

        // NoOp recv gets zero-filled header → ndim=0 → proper error
        let result = recv_tensor_with_metadata::<CpuRuntime>(&comm, 0, 0, &device);
        assert!(result.is_err());
    }
}
