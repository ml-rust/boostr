//! Output layout and KV-range types for `FlashAttentionOps`.

use crate::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Memory layout of the attention output.
///
/// Both layouts hold the same values in the same dtype: the kernels change
/// only the address each element is stored to, so `TokenMajor` is bitwise
/// `HeadMajor.permute([0, 2, 1, 3])`. `TokenMajor` reshapes to
/// `[B, S_q, H * D]` for free, which is what an output projection reads,
/// so a consumer picks it to drop the `permute + contiguous` copy after
/// attention. At `S_q == 1` the two layouts are the same bytes.
///
/// The logsumexp is `[B, H, S_q]` under either layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttnOutLayout {
    /// `[B, H, S_q, D]`: the layout of `q`.
    #[default]
    HeadMajor,
    /// `[B, S_q, H, D]`.
    TokenMajor,
}

impl AttnOutLayout {
    /// Output shape for `batch`, `num_heads`, `seq_len_q`, `head_dim`.
    pub fn shape(
        self,
        batch: usize,
        num_heads: usize,
        seq_len_q: usize,
        head_dim: usize,
    ) -> [usize; 4] {
        match self {
            Self::HeadMajor => [batch, num_heads, seq_len_q, head_dim],
            Self::TokenMajor => [batch, seq_len_q, num_heads, head_dim],
        }
    }

    /// A head-major tensor re-laid into `self`. Copies only for `TokenMajor`
    /// at `S_q > 1`; the head-major bytes already are the token-major bytes
    /// at `S_q == 1`, so that case is a reshape.
    pub fn from_head_major<R: Runtime>(self, out: Tensor<R>) -> Result<Tensor<R>> {
        match self {
            Self::HeadMajor => Ok(out),
            Self::TokenMajor => {
                let shape = out.shape();
                let [b, h, s, d] = [shape[0], shape[1], shape[2], shape[3]];
                if s == 1 {
                    return Ok(out.reshape(&[b, 1, h, d])?);
                }
                Ok(out.permute(&[0, 2, 1, 3])?.contiguous()?)
            }
        }
    }
}

/// How many keys each batch row has in the cache buffer.
///
/// The key tensor's dim-2 stays the memory stride under either variant, so a
/// full-capacity KV cache buffer needs no copy or narrowing.
pub enum KvSeqLen<'a, R: Runtime> {
    /// Every row ends at the same key; the tensor's dim-2 stays the stride.
    Shared(usize),
    /// Row `b` ends at `ends[b]`: a `[B]` I32 tensor on the key's device.
    PerRow(&'a Tensor<R>),
}

// Manual impls: a derive would add `R: Clone + Copy` bounds on the runtime,
// but a `KvSeqLen` is `Copy` regardless of the runtime type.
impl<R: Runtime> Clone for KvSeqLen<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Runtime> Copy for KvSeqLen<'_, R> {}
