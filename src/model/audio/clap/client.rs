//! [`ClapClient`]: the client bound every CLAP forward needs.

use crate::model::encoder::EncoderClient;
use numr::ops::UtilityOps;
use numr::runtime::Runtime;

/// A backend client (e.g. `CpuClient`, `CudaClient`) that runs both CLAP
/// towers: the encoder ops of the text tower, plus `clamp`.
pub trait ClapClient<R: Runtime>: EncoderClient<R> + UtilityOps<R> {}

impl<R: Runtime, C: EncoderClient<R> + UtilityOps<R>> ClapClient<R> for C {}
