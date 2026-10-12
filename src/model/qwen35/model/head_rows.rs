//! [`Qwen35Model::candidate_logits`]: F32 logits of a few candidate tokens
//! per hidden row, read from the matching `lm_head` rows only.

use super::build::Qwen35Model;
use crate::error::{Error, Result};
use crate::model::traits::ModelClient;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Qwen35Model<R>
where
    R::Client: IndexingOps<R>,
{
    /// Logits `[Q, K]` F32: `out[q, k]` is the logit of token
    /// `candidates[q, k]` at hidden row `q`, equal to that entry of the
    /// full `lm_head` output.
    ///
    /// - `hidden`: `[Q, hidden_size]` rows after `output_norm`, as
    ///   [`forward_qwen35_gather`](Self::forward_qwen35_gather) returns them
    /// - `candidates`: `[Q, K]` i64 token ids, `K >= 1` for every row; an id
    ///   may repeat within and across rows
    ///
    /// Only the `Q * K` chosen `lm_head` rows are read, and the dot
    /// products run in F32 whatever the model dtype. A tied head reads the
    /// embedding storage it shares. A Hadamard-rotated head rotates
    /// `hidden` first, as its full forward does. A block-quantized head
    /// dequantizes only the chosen rows. See
    /// [`MaybeRotatedLinear::row_logits_f32`](crate::nn::MaybeRotatedLinear::row_logits_f32).
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `lm_head` carries a LoRA adapter.
    /// [`Error::InvalidArgument`] when `hidden` is not `[Q >= 1,
    /// hidden_size]` float, when `candidates` is not `[Q, K >= 1]` i64 on
    /// `hidden`'s device, or when a candidate id lies outside
    /// `[0, vocab_size)`; the last names the first such id.
    pub fn candidate_logits<C>(
        &self,
        client: &C,
        hidden: &Tensor<R>,
        candidates: &Tensor<R>,
    ) -> Result<Tensor<R>>
    where
        C: ModelClient<R>,
    {
        if self.lm_head.is_adapted() {
            return Err(Error::ModelError {
                reason: "qwen35: lm_head carries a LoRA adapter; adapters never target \
                         lm_head, so candidate_logits has no adapted-head path"
                    .into(),
            });
        }
        self.lm_head.row_logits_f32(client, hidden, candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::super::build::tests::{HIDDEN, Head, VOCAB, chunked_model};
    use super::super::forward_gather::tests::{CHUNK, assert_close, caches, ids, index};
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    /// `[2, 9]` tokens, both rows crossing two chunk boundaries.
    const TOKENS: [i64; 18] = [3, 8, 13, 2, 7, 12, 1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8];
    /// Flat positions `b * 9 + t` of the questions.
    const GATHER: [i64; 3] = [4, 8, 17];
    const K: usize = 5;
    /// Candidates per question; ids repeat across questions.
    const CANDIDATES: [i64; 3 * K] = [0, 15, 7, 3, 9, 7, 2, 15, 11, 0, 4, 9, 1, 14, 7];

    /// `candidate_logits` over the gathered rows against the full forward's
    /// logits at the same positions and ids.
    fn check(client: &CpuClient, device: &CpuDevice, model: &Qwen35Model<CpuRuntime>) {
        let input_ids = ids(device, &TOKENS, 2);
        let (mut kv, mut gdn) = caches(device, model, 2);
        let full = model
            .forward_qwen35(client, &input_ids, &mut kv, &mut gdn, 0)
            .unwrap()
            .to_vec::<f32>();
        let want: Vec<f32> = (0..GATHER.len() * K)
            .map(|i| full[GATHER[i / K] as usize * VOCAB + CANDIDATES[i] as usize])
            .collect();

        let (mut kv, mut gdn) = caches(device, model, 2);
        let gather = index(device, &GATHER);
        let hidden = model
            .forward_qwen35_gather(client, &input_ids, &mut kv, &mut gdn, 0, &gather)
            .unwrap();
        assert_eq!(hidden.shape(), &[GATHER.len(), HIDDEN]);
        let candidates =
            Tensor::<CpuRuntime>::from_slice(&CANDIDATES, &[GATHER.len(), K], device).unwrap();
        let got = model
            .candidate_logits(client, &hidden, &candidates)
            .unwrap();
        assert_eq!(got.shape(), &[GATHER.len(), K]);
        assert_eq!(got.dtype(), DType::F32);
        assert_close(&got.to_vec::<f32>(), &want, 1e-4);
    }

    #[test]
    fn own_head_matches_full_logits() {
        let (client, device) = cpu_setup();
        let model = chunked_model(&device, 0x3535_0050, CHUNK, false, Head::Own);
        check(&client, &device, &model);
    }

    #[test]
    fn tied_head_matches_full_logits() {
        let (client, device) = cpu_setup();
        let model = chunked_model(&device, 0x3535_0051, CHUNK, false, Head::Tied);
        check(&client, &device, &model);
    }

    /// Every projection, `lm_head` included, wrapped `Rotated`.
    #[test]
    fn rotated_head_matches_full_logits() {
        let (client, device) = cpu_setup();
        let model = chunked_model(&device, 0x3535_0052, CHUNK, true, Head::Own);
        assert!(model.lm_head.is_rotated());
        check(&client, &device, &model);
    }

    #[test]
    fn bad_candidates_and_adapted_head_are_refused() {
        let (client, device) = cpu_setup();
        let mut model = chunked_model(&device, 0x3535_0053, CHUNK, false, Head::Own);
        let hidden = Tensor::<CpuRuntime>::ones(&[2, HIDDEN], DType::F32, &device).unwrap();
        let past_end =
            Tensor::<CpuRuntime>::from_slice(&[1i64, 2, VOCAB as i64, 3], &[2, 2], &device)
                .unwrap();
        let err = model
            .candidate_logits(&client, &hidden, &past_end)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("row id 16 at flat position 2 is outside [0, 16)"),
            "{err}"
        );
        let one_row = Tensor::<CpuRuntime>::from_slice(&[1i64, 2], &[1, 2], &device).unwrap();
        assert!(model.candidate_logits(&client, &hidden, &one_row).is_err());

        let a = Tensor::<CpuRuntime>::ones(&[2, HIDDEN], DType::F32, &device).unwrap();
        let b = Tensor::<CpuRuntime>::ones(&[VOCAB, 2], DType::F32, &device).unwrap();
        model.lm_head.attach_adapter(a, b, 1.0).unwrap();
        let ok = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4], &[2, 2], &device).unwrap();
        let err = model
            .candidate_logits(&client, &hidden, &ok)
            .unwrap_err()
            .to_string();
        assert!(err.contains("LoRA"), "{err}");
    }
}
