//! Gated DeltaNet recurrence operations trait
//!
//! The linear-attention recurrence used by Qwen3-Next style hybrid models:
//! a per-head `[S_k, S_v]` state updated by a gated delta rule. Two entry
//! points: a single-token step for decode and a chunked prefill (chunk 64,
//! UT transform) for prompts. Both compose numr primitives and run on every
//! backend without device transfers.
//!
//! The math ports `build_delta_net_autoregressive` and
//! `build_delta_net_chunking` from the PrismML llama.cpp fork
//! (`src/models/delta-net-base.cpp`) one to one.
//!
//! # State orientation
//!
//! State here is `[batch, H, S_k, S_v]`: row index is the key dimension,
//! column index is the value dimension, so `o = q @ state`, `state += k ⊗ d`
//! with `k` down the rows and `d` along the columns.
//!
//! The fork stores the transpose. In ggml `ne[0]` is the fastest axis and
//! its state is `[S_v, S_v, H_v, n_seqs]`. The autoregressive path contracts
//! `k` against `ne[0]`:
//!
//! ```text
//! sk = ggml_mul     (ctx0, s, k);      // k is [S_k, 1, H, B]: broadcast over ne[1]
//! sk = ggml_sum_rows(ctx0, sk);        // sums ne[0]  -> ne[0] is the k index
//! d  = ggml_sub(ctx0, v, ggml_transpose(ctx0, sk));  // [S_v, 1, H, B]
//! kd = ggml_mul   (ctx0, ggml_repeat(ctx0, k, s), ggml_transpose(ctx0, d));
//! o  = ggml_sum_rows(ctx0, ggml_mul(ctx0, s, q));   // contracts ne[0] with q
//! ```
//!
//! So the fork's state element `[b][h][ne1 = v][ne0 = k]` is this op's
//! `[b][h][k][v]`. Shapes agree because `S_k == S_v`. A state copied from
//! the fork's recurrent cache must be transposed on its last two axes.
//!
//! # Scale
//!
//! `q` is multiplied by `1 / sqrt(S_k)` at the top of both paths, before
//! any use (`q = ggml_scale(ctx0, q, 1.0f / sqrtf(S_k))`). Nothing else is
//! scaled.

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Gated DeltaNet recurrence.
///
/// # Layout
///
/// - `q`, `k`: `[batch, seq, H, S_k]`. `H` is the value head count. The
///   caller repeats key heads to match before the call. `k` is L2-normalized
///   by the caller.
/// - `v`: `[batch, seq, H, S_v]`
/// - `g`: `[batch, seq, H]` log decay, `<= 0`
/// - `beta`: `[batch, seq, H]` write strength in `(0, 1)`
/// - `state`: `[batch, H, S_k, S_v]`
///
/// # Returns `(o, state)`
///
/// - `o`: `[batch, seq, H, S_v]`
/// - `state`: `[batch, H, S_k, S_v]` after the last token
///
/// # Per-token math
///
/// ```text
/// S  = S * exp(g)
/// d  = beta * (v - k @ S)
/// S  = S + k^T ⊗ d
/// o  = (q / sqrt(S_k)) @ S
/// ```
pub trait GatedDeltaNetOps<R: Runtime> {
    /// Single-token recurrence. `seq` must be 1.
    fn gdn_step(
        &self,
        q: &Tensor<R>,
        k: &Tensor<R>,
        v: &Tensor<R>,
        g: &Tensor<R>,
        beta: &Tensor<R>,
        state: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>)>;

    /// Chunked prefill over the whole sequence.
    ///
    /// Splits `seq` into chunks of `chunk_size` (the fork uses 64), solves
    /// the intra-chunk delta rule with the UT transform `(I + A)^-1`, and
    /// carries the state across chunks. The sequence is zero-padded to a
    /// chunk multiple; padded tokens have `beta = 0` and `g = 0`, so they
    /// contribute nothing to the state. Output is sliced back to `seq`.
    ///
    /// Same result as `seq` calls to [`gdn_step`](Self::gdn_step) up to
    /// float rounding.
    #[allow(clippy::too_many_arguments)]
    fn gdn_chunk_prefill(
        &self,
        q: &Tensor<R>,
        k: &Tensor<R>,
        v: &Tensor<R>,
        g: &Tensor<R>,
        beta: &Tensor<R>,
        state: &Tensor<R>,
        chunk_size: usize,
    ) -> Result<(Tensor<R>, Tensor<R>)>;
}
