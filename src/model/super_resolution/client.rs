//! [`SrClient`]: the client bound every super-resolution network composes.

use numr::ops::{BinaryOps, FusedConvOps, ImageOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};

/// The numr op traits a super-resolution forward pass and load use.
///
/// `CpuClient` and `CudaClient` implement every one of them. The blanket
/// impl below makes any such client an `SrClient`. The trait exists to keep
/// the network signatures short.
pub trait SrClient<R: Runtime>:
    RuntimeClient<R> + BinaryOps<R> + FusedConvOps<R> + ImageOps<R> + TypeConversionOps<R>
{
}

impl<R, C> SrClient<R> for C
where
    R: Runtime,
    C: RuntimeClient<R> + BinaryOps<R> + FusedConvOps<R> + ImageOps<R> + TypeConversionOps<R>,
{
}
