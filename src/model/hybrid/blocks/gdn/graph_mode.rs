//! CUDA graph-mode decode step for [`GdnBlock`]: the `seq == 1` math of
//! [`GdnBlock::forward`], with the state carried in place.

#[cfg(feature = "cuda")]
use super::layer::GdnBlock;
#[cfg(feature = "cuda")]
use crate::error::{Error, Result};
#[cfg(feature = "cuda")]
use crate::inference::GdnState;
#[cfg(feature = "cuda")]
use numr::autograd::Var;

#[cfg(feature = "cuda")]
impl GdnBlock<numr::runtime::cuda::CudaRuntime> {
    /// One decode step for graph capture and replay.
    ///
    /// Same math as [`forward`](Self::forward) with `seq == 1`. It reads
    /// `state.conv()` and `state.ssm()`, whose addresses are stable, and
    /// writes the new window and delta-rule state back into those same
    /// buffers through [`GdnState::copy_from_captured`], so the next replay
    /// reads this step's state. Every intermediate is graph-managed. Nothing
    /// is read on the host.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `x` is not `[batch, 1, hidden_size]` or
    /// `state` was built for another batch size; op errors propagate.
    pub fn forward_graph_mode(
        &self,
        client: &numr::runtime::cuda::CudaClient,
        x: &Var<numr::runtime::cuda::CudaRuntime>,
        state: &GdnState<numr::runtime::cuda::CudaRuntime>,
    ) -> Result<Var<numr::runtime::cuda::CudaRuntime>> {
        let shape = x.shape();
        if shape.len() != 3 || shape[1] != 1 {
            return Err(Error::ModelError {
                reason: format!("gdn graph mode: expected [batch, 1, hidden], got {shape:?}"),
            });
        }
        let (out, window, ssm) = self.forward_core(client, x, state.conv(), state.ssm())?;
        state.copy_from_captured(client, &window, &ssm)?;
        Ok(out)
    }
}
