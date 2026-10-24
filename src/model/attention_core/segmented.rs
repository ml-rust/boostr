//! Document-masked flash attention: packed varlen attention over the
//! flattened `B * S` tokens of the attention core.

use super::spec::AttentionCoreSpec;
use crate::error::{Error, Result};
use crate::nn::var_ops::var_contiguous;
use crate::ops::traits::{PackedSegments, VarLenAttentionOps};
use crate::ops::var_varlen_attention;
use numr::autograd::{Var, var_permute, var_reshape};
use numr::ops::ScalarOps;
use numr::runtime::Runtime;

/// Pack `[B, H, S, D]` into the token-major `[B * S, H, D]` varlen layout.
fn pack_rows<R: Runtime>(x: &Var<R>, heads: usize, head_dim: usize) -> Result<Var<R>> {
    let (batch, seq) = (x.shape()[0], x.shape()[2]);
    let x = var_permute(x, &[0, 2, 1, 3]).map_err(Error::Numr)?;
    let x = var_contiguous(&x)?;
    var_reshape(&x, &[batch * seq, heads, head_dim]).map_err(Error::Numr)
}

/// Check that `segments` cover all `batch * seq` tokens and start a segment at every row.
fn check_row_starts<R: Runtime>(
    segments: &PackedSegments<R>,
    batch: usize,
    seq: usize,
) -> Result<()> {
    let invalid = |reason: String| Error::InvalidArgument {
        arg: "segments",
        reason,
    };
    let total = batch * seq;
    if segments.total_tokens() != total {
        return Err(invalid(format!(
            "attention_core_flash: segments cover {} tokens, the batch holds {batch} x {seq} = \
             {total}; build the segments over the flattened B * S tokens",
            segments.total_tokens()
        )));
    }
    let offsets = segments.host_cu_seqlens();
    for row in 1..batch {
        // `row * seq < total`, and the constructor fits `total` in i32.
        let start = (row * seq) as i32;
        if offsets.binary_search(&start).is_err() {
            return Err(invalid(format!(
                "attention_core_flash: no segment starts at row {row} (token {start}); \
                 end every document at its row end and start a new segment there"
            )));
        }
    }
    Ok(())
}

/// Causal attention inside each segment of the flattened `B * S` tokens.
///
/// `q` is `[B, H, S, D]`, `k` and `v` are `[B, H_kv, S, D]`: the prologue output.
/// Returns the token-major `[B, S, H, D]` output, ready for the token-major epilogue.
///
/// The scale is `1 / sqrt(D)`, the same as the unsegmented flash path.
/// A sliding window returns an error: varlen attention has no window.
pub(super) fn segmented_flash_attention<R>(
    q: &Var<R>,
    k: &Var<R>,
    v: &Var<R>,
    segments: &PackedSegments<R>,
    spec: &AttentionCoreSpec<'_, R>,
) -> Result<Var<R>>
where
    R: Runtime,
    R::Client: ScalarOps<R> + VarLenAttentionOps<R>,
{
    if spec.sliding_window != 0 {
        return Err(Error::InvalidArgument {
            arg: "spec.sliding_window",
            reason: format!(
                "attention_core_flash: sliding_window {} cannot run with segments, varlen \
                 attention has no window; set spec.sliding_window to 0 or pass segments None",
                spec.sliding_window
            ),
        });
    }
    let (batch, seq_q, seq_k) = (q.shape()[0], q.shape()[2], k.shape()[2]);
    if seq_q != seq_k {
        return Err(Error::InvalidArgument {
            arg: "segments",
            reason: format!(
                "attention_core_flash: segments need equal query and key lengths, got q {seq_q} \
                 and k {seq_k}; pass segments None for a KV-cached block"
            ),
        });
    }
    check_row_starts(segments, batch, seq_q)?;

    let q_packed = pack_rows(q, spec.num_heads, spec.head_dim)?;
    let k_packed = pack_rows(k, spec.num_kv_heads, spec.head_dim)?;
    let v_packed = pack_rows(v, spec.num_kv_heads, spec.head_dim)?;
    let client = R::default_client(q.tensor().device());
    let scale = (spec.head_dim as f64).sqrt().recip();
    let out = var_varlen_attention(
        &client, &q_packed, &k_packed, &v_packed, segments, scale, true, 0,
    )?;
    var_reshape(&out, &[batch, seq_q, spec.num_heads, spec.head_dim]).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn row_start_inside_a_segment_is_rejected() {
        let (_, dev) = cpu_setup();
        let segs = PackedSegments::<CpuRuntime>::from_lengths(&[3, 5], &dev).unwrap();
        let e = check_row_starts(&segs, 2, 4).err().unwrap();
        assert!(e.to_string().contains("no segment starts at row 1"), "{e}");
    }

    #[test]
    fn row_starts_on_boundaries_pass() {
        let (_, dev) = cpu_setup();
        let segs = PackedSegments::<CpuRuntime>::from_lengths(&[1, 3, 0, 4], &dev).unwrap();
        assert!(check_row_starts(&segs, 2, 4).is_ok());
    }

    #[test]
    fn token_count_mismatch_is_rejected() {
        let (_, dev) = cpu_setup();
        let segs = PackedSegments::<CpuRuntime>::from_lengths(&[4, 3], &dev).unwrap();
        let e = check_row_starts(&segs, 2, 4).err().unwrap();
        assert!(e.to_string().contains("segments cover 7 tokens"), "{e}");
    }
}
