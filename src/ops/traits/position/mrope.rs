//! Interleaved multi-section RoPE (ggml `GGML_ROPE_TYPE_IMROPE`) trait.

use crate::error::Result;
use numr::autograd::Var;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Interleaved multi-section rotary embedding, the `qwen35` / Qwen3-VL
/// position encoding. Ported from `ggml_compute_forward_rope_flt` and
/// `ggml_mrope_cache_init` in `ggml/src/ggml-cpu/ops.cpp`.
///
/// # Layout contract
///
/// - `x`: `[batch, seq, heads, head_dim]`
/// - `cos_cache`, `sin_cache`: `[max_pos, n_rot / 2]`, the table
///   `RoPE::precompute_freqs(max_pos, n_rot, freq_base, ..)` builds
/// - `positions`: `[4, seq]`, I32 or I64, one row per stream `t, h, w, e`,
///   shared across the batch. ggml reads the same layout:
///   `p_t = pos[i2]; p_h = pos[i2 + ne2]; p_w = pos[i2 + ne2*2]; p_e = pos[i2 + ne2*3]`
/// - `sections`: pair counts per stream `[t, h, w, e]`
/// - `n_rot`: rotated width. Dims `n_rot..head_dim` pass through.
/// - Output: `[batch, seq, heads, head_dim]`
///
/// # Rule, derived from ggml
///
/// (a) Pairs are NeoX split-half over the first `n_rot` dims:
/// ```text
/// case GGML_ROPE_TYPE_IMROPE:
///     rotate_pairs<T>(n_dims, n_dims/2, cache, src + n_offs, dst_data + n_offs);
/// // rotate_pairs: ic = i0/2; x0 = src[ic]; x1 = src[ic + n_offset];
/// ```
/// so pair `i` is `(x[i], x[i + n_rot/2])` for `i < n_rot/2`.
///
/// (b) Stream per pair, `sector = i % sum(sections)`:
/// ```text
/// int sector = (i0 / 2) % sect_dims;
/// if (is_imrope) {
///     if (sector % 3 == 1 && sector < 3 * sections[1]) { theta = theta_h; }
///     else if (sector % 3 == 2 && sector < 3 * sections[2]) { theta = theta_w; }
///     else if (sector % 3 == 0 && sector < 3 * sections[0]) { theta = theta_t; }
///     else { theta = theta_e; }
/// }
/// ```
/// The interleave is by `sector % 3`, not contiguous blocks. A sector that
/// fails its stream's `< 3 * sections[s]` bound falls through to `e`.
///
/// (c) Frequency per pair depends on `i` only:
/// ```text
/// const float theta_scale = powf(freq_base, -2.0f/n_dims);
/// theta_t *= theta_scale; theta_w *= theta_scale; theta_h *= theta_scale; theta_e *= theta_scale;
/// ```
/// so `angle(i) = pos[stream(i)] * freq_base^(-2i / n_rot)`.
///
/// (d) `theta_base_*` is the stream's integer position. Text tokens carry
/// `t = h = w = pos`, `e = 0` (`llm_graph_input_pos::set_input`).
///
/// # Text-only equivalence
///
/// Because (c) is independent of the stream, equal streams reduce the op to
/// plain partial NeoX RoPE over `n_rot` dims. Text sets `e = 0`, so the
/// reduction holds exactly when no pair falls through to `e`. For the
/// `qwen35` sections `[11, 11, 10, 0]` (`sum = 32 = n_rot/2`) every sector
/// resolves to `t`, `h` or `w`, and text IMROPE equals `RoPEOps::apply_rope`
/// on `x[.., ..n_rot]`. Sections such as `[1, 0, 1, 0]` send sector 1 to `e`,
/// so the general path is the one implemented.
pub trait MRopeOps<R: Runtime> {
    /// Apply interleaved multi-section RoPE.
    ///
    /// `selector` is the one-hot stream selector from
    /// [`mrope_stream_selector`](crate::ops::impl_generic::position::mrope_stream_selector),
    /// `[4, 1, n_rot / 2]`. It is a pure function of `sections` and `n_rot` —
    /// build it once per layer at construction time and reuse it across
    /// every call, instead of rebuilding it per forward pass.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` when a shape disagrees with the layout contract, when
    /// `n_rot` is odd, zero or above `head_dim`, or when `selector`'s shape
    /// is not `[4, 1, n_rot / 2]`.
    fn apply_mrope_interleaved(
        &self,
        x: &Var<R>,
        cos_cache: &Var<R>,
        sin_cache: &Var<R>,
        positions: &Tensor<R>,
        selector: &Tensor<R>,
        n_rot: usize,
    ) -> Result<Var<R>>;

    /// [`Self::apply_mrope_interleaved`] as one kernel launch where a
    /// backend has one; otherwise the composed op.
    ///
    /// Same arguments and layout contract. A backend that fuses must return
    /// the composed op's values bit for bit, so callers can pick this
    /// method without a numeric change. The default forwards to
    /// [`Self::apply_mrope_interleaved`].
    ///
    /// # Errors
    ///
    /// As [`Self::apply_mrope_interleaved`].
    fn mrope_interleaved_fused(
        &self,
        x: &Var<R>,
        cos_cache: &Var<R>,
        sin_cache: &Var<R>,
        positions: &Tensor<R>,
        selector: &Tensor<R>,
        n_rot: usize,
    ) -> Result<Var<R>> {
        self.apply_mrope_interleaved(x, cos_cache, sin_cache, positions, selector, n_rot)
    }
}
