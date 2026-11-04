//! [`SrClient`]: the client bound every super-resolution network composes.

use numr::ops::{BinaryOps, FusedConvOps, ImageOps, ScalarOps, ShapeOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};

/// The numr op traits a super-resolution forward pass and load use.
///
/// `CpuClient`, `CudaClient` and `WgpuClient` implement every one of them.
/// The blanket impl below makes any such client an `SrClient`. The trait
/// exists to keep the network signatures short.
///
/// F16 and BF16 are unavailable on WebGPU, because numr's WebGPU backend
/// computes in F32 only. Load with `SrExecution::new(DType::F32, ConvLayout::Nchw)`
/// there. NHWC also runs there, and it ran slower on the GPU measured.
pub trait SrClient<R: Runtime>:
    RuntimeClient<R>
    + BinaryOps<R>
    + FusedConvOps<R>
    + ImageOps<R>
    + ScalarOps<R>
    + ShapeOps<R>
    + TypeConversionOps<R>
{
}

impl<R, C> SrClient<R> for C
where
    R: Runtime,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + FusedConvOps<R>
        + ImageOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TypeConversionOps<R>,
{
}
