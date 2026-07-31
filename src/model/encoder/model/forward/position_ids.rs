//! Position-id generation, which differs by architecture family.

use crate::model::encoder::config::ArchFamily;
use crate::model::encoder::model::Encoder;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Encoder<R> {
    /// Compute position IDs from input token IDs on the host.
    ///
    /// Returns a flat `Vec<i64>` of length `batch * seq_len`. Called before CUDA
    /// graph capture so the D2H read for XLM-RoBERTa runs outside the captured
    /// region.
    pub(crate) fn compute_position_ids_host(
        &self,
        flat_input_ids: &[i64],
        batch: usize,
        seq_len: usize,
    ) -> Vec<i64> {
        if self.config.arch_family != ArchFamily::XlmRoberta {
            return (0..seq_len as i64).cycle().take(batch * seq_len).collect();
        }

        // XLM-RoBERTa reserves position `pad_id` for padding and numbers real
        // tokens from `pad_id + 1` upward, so the ids depend on token values.
        let pad_id = self.config.padding_token_id;
        let mut pos_flat: Vec<i64> = Vec::with_capacity(batch * seq_len);
        for b in 0..batch {
            let mut count: i64 = 0;
            for s in 0..seq_len {
                let tok = flat_input_ids[b * seq_len + s];
                if tok == pad_id {
                    pos_flat.push(pad_id);
                } else {
                    count += 1;
                    pos_flat.push(pad_id + count);
                }
            }
        }
        pos_flat
    }

    /// Build the position-ID tensor for a forward pass from `input_ids`.
    ///
    /// BERT-style families use `[0, 1, ..., S-1]` shaped `[S]` and broadcast
    /// across the batch; XLM-RoBERTa needs per-row ids and is shaped `[B, S]`.
    pub(in crate::model::encoder) fn position_ids_tensor(
        &self,
        input_ids: &Tensor<R>,
        shape: &[usize],
        seq_len: usize,
    ) -> Tensor<R> {
        let device = input_ids.device();
        if self.config.arch_family != ArchFamily::XlmRoberta {
            let pos_ids: Vec<i64> = (0..seq_len as i64).collect();
            return Tensor::<R>::from_slice(&pos_ids, &[seq_len], device);
        }

        let batch = if shape.len() == 2 { shape[0] } else { 1 };
        let flat_ids: Vec<i64> = input_ids.to_vec();
        let pos_flat = self.compute_position_ids_host(&flat_ids, batch, seq_len);
        if shape.len() == 2 {
            Tensor::<R>::from_slice(&pos_flat, &[batch, seq_len], device)
        } else {
            Tensor::<R>::from_slice(&pos_flat, &[seq_len], device)
        }
    }
}
