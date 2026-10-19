//! Block-diagonal document mask for attention over packed documents.

use super::flash_standard::build_attention_mask;
use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Additive `[B, 1, S, S]` F32 bias that keeps attention inside each document.
///
/// `segment_ids` is a `[B, S]` I32 tensor. Equal ids in one row mark one document.
/// Query `i` sees key `j` when `segment_ids[b, i] == segment_ids[b, j]`.
/// `causal` also requires `j <= i`.
/// Allowed entries hold `0`. Blocked entries hold `-inf`, like the causal mask.
///
/// The diagonal is always allowed, so no softmax row is all `-inf`.
/// The compare and the select run on the device. No id is read back to the host.
pub fn document_mask_bias<R, C>(
    client: &C,
    segment_ids: &Tensor<R>,
    batch: usize,
    seq_len: usize,
    causal: bool,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: CompareOps<R> + ConditionalOps<R>,
{
    if segment_ids.dtype() != DType::I32 {
        return Err(Error::InvalidArgument {
            arg: "segment_ids",
            reason: format!(
                "document_mask_bias: segment_ids must be I32, got {:?}; cast the ids to I32",
                segment_ids.dtype()
            ),
        });
    }
    if segment_ids.shape() != [batch, seq_len] {
        return Err(Error::InvalidArgument {
            arg: "segment_ids",
            reason: format!(
                "document_mask_bias: segment_ids must be [{batch}, {seq_len}] to match the \
                 attention input, got {:?}",
                segment_ids.shape()
            ),
        });
    }
    let device = segment_ids.device();
    let ids = segment_ids.contiguous()?;
    let query_ids = ids.reshape(&[batch, 1, seq_len, 1])?;
    let key_ids = ids.reshape(&[batch, 1, 1, seq_len])?;
    let same_document = client.eq(&query_ids, &key_ids)?;

    let allowed = if causal {
        build_attention_mask::<R>(seq_len, seq_len, true, 0, device)?
    } else {
        Tensor::<R>::zeros(&[1, 1, 1, 1], DType::F32, device)?
    };
    let blocked = Tensor::<R>::full_scalar(&[1, 1, 1, 1], DType::F32, f64::NEG_INFINITY, device)?;
    Ok(client.where_cond(&same_document, &allowed, &blocked)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    const NEG: f32 = f32::NEG_INFINITY;

    fn bias(ids: &[i32], batch: usize, seq: usize, causal: bool) -> Vec<f32> {
        let (client, dev) = cpu_setup();
        let ids = Tensor::<CpuRuntime>::from_slice(ids, &[batch, seq], &dev).unwrap();
        let out = document_mask_bias(&client, &ids, batch, seq, causal).unwrap();
        assert_eq!(out.shape(), &[batch, 1, seq, seq]);
        assert_eq!(out.dtype(), DType::F32);
        out.to_vec::<f32>()
    }

    #[test]
    fn causal_bias_is_block_diagonal_lower_triangle() {
        let got = bias(&[0, 0, 1, 1], 1, 4, true);
        #[rustfmt::skip]
        let want = [
            0.0, NEG, NEG, NEG,
            0.0, 0.0, NEG, NEG,
            NEG, NEG, 0.0, NEG,
            NEG, NEG, 0.0, 0.0,
        ];
        assert_eq!(got, want);
    }

    #[test]
    fn non_causal_bias_is_block_diagonal() {
        let got = bias(&[5, 5, 9], 1, 3, false);
        #[rustfmt::skip]
        let want = [
            0.0, 0.0, NEG,
            0.0, 0.0, NEG,
            NEG, NEG, 0.0,
        ];
        assert_eq!(got, want);
    }

    #[test]
    fn rows_use_their_own_ids() {
        let got = bias(&[0, 1, 0, 0], 2, 2, false);
        assert_eq!(&got[..4], &[0.0, NEG, NEG, 0.0]);
        assert_eq!(&got[4..], &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn rejects_non_i32_ids() {
        let (client, dev) = cpu_setup();
        let ids = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0], &[1, 2], &dev).unwrap();
        let e = document_mask_bias(&client, &ids, 1, 2, true).err().unwrap();
        assert!(e.to_string().contains("must be I32"), "{e}");
    }

    #[test]
    fn rejects_wrong_shape() {
        let (client, dev) = cpu_setup();
        let ids = Tensor::<CpuRuntime>::from_slice(&[0i32, 0, 1], &[1, 3], &dev).unwrap();
        let e = document_mask_bias(&client, &ids, 1, 4, true).err().unwrap();
        assert!(e.to_string().contains("must be [1, 4]"), "{e}");
    }
}
