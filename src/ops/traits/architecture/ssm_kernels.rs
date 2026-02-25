//! Structured State Space Duality (SSD) kernel operations trait
//!
//! Mamba2 SSD chunk-parallel forward pass: splits a sequence into chunks,
//! computes intra-chunk outputs via tiled matmul, propagates states across
//! chunks, then combines. These four operations compose into the full
//! chunk-parallel SSM forward used by Mamba2.
//!
//! All operations work on `Tensor<R>` (no autograd) — inference-path
//! optimizations. The existing `model/mamba/ssm.rs` uses autograd-composed
//! ops for training.

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// SSD chunk-parallel kernel operations for Mamba2.
///
/// The four methods correspond to the four stages of the SSD block
/// decomposition algorithm:
///
/// 1. **cumsum** — compute decay schedule `dA_cumsum = cumsum(dt * A)`
/// 2. **chunk_state** — compute per-chunk final states from inputs + decay
/// 3. **state_passing** — propagate states across chunks sequentially
/// 4. **chunk_scan** — compute output: `y = C @ h + D * x`
///
/// Typical forward call order: cumsum → chunk_state → state_passing → chunk_scan.
#[allow(non_snake_case)]
pub trait SsmKernelOps<R: Runtime> {
    /// Compute discretised decay cumulative sum.
    ///
    /// Applies optional softplus to `dt`, multiplies by `A`, and computes
    /// an inclusive prefix sum within each chunk.
    ///
    /// # Layout
    ///
    /// - `dt`: `[batch, seqlen, nheads]` — timestep durations
    /// - `A`: `[nheads]` — diagonal state decay (negative values)
    /// - `dt_bias`: optional `[nheads]` — bias added to dt before softplus
    /// - `chunk_size`: number of positions per chunk
    /// - `dt_softplus`: whether to apply softplus to dt
    ///
    /// # Returns `(dt_out, dA_cumsum)`
    ///
    /// - `dt_out`: `[batch, nheads, nchunks, chunk_size]` — processed dt values
    /// - `dA_cumsum`: `[batch, nheads, nchunks, chunk_size]` — inclusive cumsum of dt*A
    fn ssd_chunk_cumsum(
        &self,
        dt: &Tensor<R>,
        a: &Tensor<R>,
        dt_bias: Option<&Tensor<R>>,
        chunk_size: usize,
        dt_softplus: bool,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Compute per-chunk final hidden states.
    ///
    /// For each chunk, accumulates the weighted outer product of `B` and `x`
    /// using the decay schedule, yielding the state at the end of the chunk
    /// (before inter-chunk propagation).
    ///
    /// # Layout
    ///
    /// - `x`: `[batch, seqlen, nheads, headdim]` — input activations
    /// - `b`: `[batch, seqlen, ngroups, dstate]` — input projection
    /// - `dt`: `[batch, nheads, nchunks, chunk_size]` — processed dt from cumsum
    /// - `dA_cumsum`: `[batch, nheads, nchunks, chunk_size]` — from cumsum
    ///
    /// # Returns
    ///
    /// - `states`: `[batch, nchunks, nheads, headdim, dstate]` — per-chunk states
    fn ssd_chunk_state(
        &self,
        x: &Tensor<R>,
        b: &Tensor<R>,
        dt: &Tensor<R>,
        dA_cumsum: &Tensor<R>,
    ) -> Result<Tensor<R>>;

    /// Propagate hidden states across chunks (sequential scan).
    ///
    /// Applies the recurrence `h[c] = exp(dA_last[c]) * h[c-1] + h[c]`
    /// across all chunks, where `dA_last[c]` is the decay at the last
    /// position of chunk `c`.
    ///
    /// # Layout
    ///
    /// - `states`: `[batch, nchunks, nheads, headdim, dstate]` — per-chunk states (modified in-place or copied)
    /// - `dA_cumsum`: `[batch, nheads, nchunks, chunk_size]` — from cumsum
    ///
    /// # Returns
    ///
    /// - `states_out`: `[batch, nchunks, nheads, headdim, dstate]` — propagated states
    fn ssd_state_passing(&self, states: &Tensor<R>, dA_cumsum: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compute chunk output via state projection.
    ///
    /// For each position, projects the hidden state through `C` and adds
    /// a skip connection through `D`:
    ///   `y[t] = C[t] @ h[chunk_of(t)] + D * x[t]`
    ///
    /// # Layout
    ///
    /// - `x`: `[batch, seqlen, nheads, headdim]` — input activations
    /// - `states`: `[batch, nchunks, nheads, headdim, dstate]` — propagated states
    /// - `c`: `[batch, seqlen, ngroups, dstate]` — output projection
    /// - `dA_cumsum`: `[batch, nheads, nchunks, chunk_size]` — for intra-chunk decay
    /// - `d`: optional `[nheads]` — skip connection weights
    ///
    /// # Returns
    ///
    /// - `output`: `[batch, seqlen, nheads, headdim]`
    fn ssd_chunk_scan(
        &self,
        x: &Tensor<R>,
        states: &Tensor<R>,
        c: &Tensor<R>,
        dA_cumsum: &Tensor<R>,
        d: Option<&Tensor<R>>,
    ) -> Result<Tensor<R>>;
}
