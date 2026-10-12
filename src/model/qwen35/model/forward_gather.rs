//! [`Qwen35Model::forward_qwen35_gather`]: the token forward that returns
//! normed hidden rows at chosen positions instead of logits everywhere.
//!
//! A right-padded batch reads the same rows at each sequence's real
//! positions as each sequence run alone: attention is causal, the GDN conv
//! is causal, and the chunked GDN recurrence masks every later token out of
//! an earlier row. Padding after a sequence's last real token therefore
//! changes only the rows at padded positions and the caches.

use super::build::Qwen35Model;
use super::forward_layers::AttentionStep;
use crate::error::{Error, Result};
use crate::inference::{LayeredGdnState, LayeredKvCache};
use crate::model::traits::ModelClient;
use crate::nn::linear::maybe_rotated::row_checks::name_out_of_range;
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ConvOps, IndexingOps, MatmulOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Device, Runtime};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Qwen35Model<R>
where
    R::Client: IndexingOps<R>,
{
    /// The token forward of [`forward_qwen35`](Self::forward_qwen35),
    /// returning hidden rows `[G, hidden_size]` after `output_norm` instead
    /// of logits. `lm_head` is not applied.
    ///
    /// - `input_ids`: `[batch, seq]` token ids
    /// - `kv_cache`, `gdn_state`: advanced over every token, as
    ///   `forward_qwen35` advances them
    /// - `start_pos`: the IMROPE position of `input_ids[.., 0]`; text-only
    ///   callers pass `kv_cache.seq_len()`
    /// - `gather`: `[G]` i64 flat row indices `b * seq + t`, in
    ///   `[0, batch * seq)`; an index may repeat
    ///
    /// `output_norm` is per row, so normalizing the `G` gathered rows equals
    /// gathering the normalized `batch * seq` rows; it runs on the `G` rows.
    ///
    /// # Errors
    ///
    /// [`Error::ModelError`] when `input_ids` is not `[batch, seq]` with
    /// `batch * seq >= 1`, when `start_pos + seq` exceeds the rope table, or
    /// when a cache holds fewer layers than the model has of that kind.
    /// [`Error::InvalidArgument`] when `gather` is empty, not 1-D, not i64,
    /// not on `input_ids`' device, or holds an index outside
    /// `[0, batch * seq)`; the last names the first such index. Every
    /// `gather` and rope error returns before a cache is written. Block
    /// errors propagate.
    pub fn forward_qwen35_gather<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        kv_cache: &mut LayeredKvCache<R>,
        gdn_state: &mut LayeredGdnState<R>,
        start_pos: usize,
        gather: &Tensor<R>,
    ) -> Result<Tensor<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + ConvOps<R>
            + DequantOps<R>
            + MatmulOps<R>,
    {
        let shape = input_ids.shape();
        if shape.len() != 2 || shape[0] * shape[1] == 0 {
            return Err(Error::ModelError {
                reason: format!("qwen35: input_ids must be [batch, seq], non-empty, got {shape:?}"),
            });
        }
        let (batch, seq) = (shape[0], shape[1]);
        let rows = batch * seq;
        let hidden_size = self.config.hidden_size;
        check_gather(input_ids, gather)?;
        let positions = self.text_positions(seq, start_pos, input_ids.device())?;
        let embeds = self.embed_tokens(client, input_ids)?;
        // `index_select` over the flat embeddings bounds every gather index
        // against `batch * seq` on the device, before any cache is written.
        let flat_embeds = embeds.contiguous()?.reshape(&[rows, hidden_size])?;
        client
            .index_select(&flat_embeds, 0, gather)
            .map_err(|e| name_out_of_range(e.into(), gather, rows, "gather", "gather index"))?;

        let hidden = self.forward_blocks(
            client,
            Var::new(embeds, false),
            &positions,
            kv_cache,
            gdn_state,
            &AttentionStep::Shared,
        )?;
        let flat = hidden
            .tensor()
            .contiguous()?
            .reshape(&[rows, hidden_size])?;
        let picked = client.index_select(&flat, 0, gather)?;
        let normed = self.norm.forward(client, &Var::new(picked, false))?;
        Ok(normed.tensor().clone())
    }
}

/// `gather` is a non-empty 1-D i64 tensor on `input_ids`' device.
fn check_gather<R: Runtime<DType = DType>>(
    input_ids: &Tensor<R>,
    gather: &Tensor<R>,
) -> Result<()> {
    let bad = |reason: String| {
        Err(Error::InvalidArgument {
            arg: "gather",
            reason,
        })
    };
    if gather.shape().len() != 1 || gather.numel() == 0 {
        return bad(format!("must be [G >= 1], got {:?}", gather.shape()));
    }
    if gather.dtype() != DType::I64 {
        return bad(format!("must be I64, got {:?}", gather.dtype()));
    }
    if !gather.device().is_same(input_ids.device()) {
        return bad(format!(
            "is on {}, input_ids on {}",
            gather.device().name(),
            input_ids.device().name()
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::build::tests::{H_KV, HD, HIDDEN, Head, MAX_POS, VOCAB, chunked_model};
    use super::*;
    use crate::inference::kv_cache::layered::LayeredKvCacheConfig;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    /// GDN chunk width of every model here, so short prompts cross chunks.
    pub(crate) const CHUNK: usize = 4;

    pub(crate) fn caches(
        device: &CpuDevice,
        model: &Qwen35Model<CpuRuntime>,
        batch: usize,
    ) -> (LayeredKvCache<CpuRuntime>, LayeredGdnState<CpuRuntime>) {
        let kv_config = LayeredKvCacheConfig {
            batch_size: batch,
            num_kv_heads: H_KV,
            initial_capacity: MAX_POS,
            max_seq_len: MAX_POS,
            head_dim: HD,
            dtype: DType::F32,
        };
        let kv =
            LayeredKvCache::<CpuRuntime>::new(model.num_attention_layers(), &kv_config, device)
                .unwrap();
        let gdn = LayeredGdnState::<CpuRuntime>::zeros(
            model.num_gdn_layers(),
            model.gdn_config(),
            batch,
            DType::F32,
            device,
        )
        .unwrap();
        (kv, gdn)
    }

    pub(crate) fn ids(device: &CpuDevice, data: &[i64], batch: usize) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(data, &[batch, data.len() / batch], device).unwrap()
    }

    pub(crate) fn index(device: &CpuDevice, data: &[i64]) -> Tensor<CpuRuntime> {
        Tensor::<CpuRuntime>::from_slice(data, &[data.len()], device).unwrap()
    }

    /// Fresh caches, one gather forward.
    pub(crate) fn gather_fresh(
        client: &CpuClient,
        model: &Qwen35Model<CpuRuntime>,
        input_ids: &Tensor<CpuRuntime>,
        gather: &[i64],
    ) -> Vec<f32> {
        let device = input_ids.device().clone();
        let (mut kv, mut gdn) = caches(&device, model, input_ids.shape()[0]);
        model
            .forward_qwen35_gather(
                client,
                input_ids,
                &mut kv,
                &mut gdn,
                0,
                &index(&device, gather),
            )
            .unwrap()
            .to_vec::<f32>()
    }

    pub(crate) fn assert_close(got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() < tol, "element {i}: {g} vs {w}");
        }
    }

    /// `[2, 9]` tokens: two sequences, each crossing two chunk boundaries.
    fn two_rows() -> Vec<i64> {
        (0..18).map(|i| ((i * 5 + 3) % VOCAB) as i64).collect()
    }

    #[test]
    fn gathered_rows_equal_full_hidden_rows() {
        let (client, device) = cpu_setup();
        let model = chunked_model(&device, 0x3535_0040, CHUNK, false, Head::Own);
        let input_ids = ids(&device, &two_rows(), 2);
        let full = model.forward_hidden(&client, &input_ids).unwrap();
        let full = full.tensor().to_vec::<f32>();
        let gather = [0i64, 5, 8, 9, 17, 5, 12];
        let got = gather_fresh(&client, &model, &input_ids, &gather);
        let want: Vec<f32> = gather
            .iter()
            .flat_map(|&r| full[r as usize * HIDDEN..(r as usize + 1) * HIDDEN].to_vec())
            .collect();
        assert_close(&got, &want, 1e-5);
    }

    /// Rows of lengths 1, 4 (one whole chunk), 6 (crosses one boundary) and
    /// 10 (crosses two), right-padded to 10 with a pad id no row uses. Each
    /// row's last real token reads the same hidden row as that row run
    /// alone. A second pad id changes the rows at padded positions, so the
    /// pads are processed, not skipped.
    #[test]
    fn right_padded_batch_matches_rows_run_alone() {
        let (client, device) = cpu_setup();
        let model = chunked_model(&device, 0x3535_0041, CHUNK, false, Head::Own);
        let lens = [1usize, 4, 6, 10];
        let width = 10;
        let real = |r: usize| -> Vec<i64> {
            (0..lens[r])
                .map(|t| ((r * 3 + t * 5 + 1) % (VOCAB - 2)) as i64)
                .collect()
        };
        let padded = |pad: i64| -> Vec<i64> {
            (0..lens.len())
                .flat_map(|r| {
                    let mut row = real(r);
                    row.resize(width, pad);
                    row
                })
                .collect()
        };
        let last: Vec<i64> = (0..lens.len())
            .map(|r| (r * width + lens[r] - 1) as i64)
            .collect();
        let pad_a = (VOCAB - 1) as i64;
        let pad_b = (VOCAB - 2) as i64;
        let batched = gather_fresh(&client, &model, &ids(&device, &padded(pad_a), 4), &last);

        for r in 0..lens.len() {
            let alone = ids(&device, &real(r), 1);
            let want = gather_fresh(&client, &model, &alone, &[lens[r] as i64 - 1]);
            assert_close(&batched[r * HIDDEN..(r + 1) * HIDDEN], &want, 1e-4);
        }

        let mut probe = last.clone();
        probe.push(5); // row 0, position 5: a pad
        let a = gather_fresh(&client, &model, &ids(&device, &padded(pad_a), 4), &probe);
        let b = gather_fresh(&client, &model, &ids(&device, &padded(pad_b), 4), &probe);
        let split = lens.len() * HIDDEN;
        assert_close(&a[..split], &b[..split], 1e-5);
        let pad_diff = a[split..]
            .iter()
            .zip(&b[split..])
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        assert!(
            pad_diff > 1e-3,
            "pad id did not reach a padded row: {pad_diff}"
        );
    }

    #[test]
    fn bad_gather_is_refused_before_caches_move() {
        let (client, device) = cpu_setup();
        let model = chunked_model(&device, 0x3535_0042, CHUNK, false, Head::Own);
        let input_ids = ids(&device, &two_rows(), 2);
        let (mut kv, mut gdn) = caches(&device, &model, 2);
        let mut run = |gather: &Tensor<CpuRuntime>| {
            model
                .forward_qwen35_gather(&client, &input_ids, &mut kv, &mut gdn, 0, gather)
                .unwrap_err()
                .to_string()
        };

        let err = run(&index(&device, &[3, 18, 4]));
        assert!(
            err.contains("gather index 18 at flat position 1 is outside [0, 18)"),
            "{err}"
        );
        let err = run(&index(&device, &[-2]));
        assert!(err.contains("gather index -2"), "{err}");
        let empty = Tensor::<CpuRuntime>::zeros(&[0], DType::I64, &device).unwrap();
        assert!(run(&empty).contains("G >= 1"));
        let as_i32 = Tensor::<CpuRuntime>::from_slice(&[1i32, 2], &[2], &device).unwrap();
        assert!(run(&as_i32).contains("I64"));
        let two_d = Tensor::<CpuRuntime>::from_slice(&[1i64, 2], &[1, 2], &device).unwrap();
        assert!(run(&two_d).contains("G >= 1"));
        assert_eq!(kv.seq_len(), 0);
    }

    /// Prefill of 6 tokens then 3 decode steps, each gathering its last
    /// row, equals one 9-token prefill gathered at positions 5..9.
    #[test]
    fn prefill_then_decode_matches_single_prefill() {
        let (client, device) = cpu_setup();
        let model = chunked_model(&device, 0x3535_0043, CHUNK, false, Head::Own);
        let tokens: Vec<i64> = two_rows()[..9].to_vec();
        let want = gather_fresh(&client, &model, &ids(&device, &tokens, 1), &[5, 6, 7, 8]);

        let (mut kv, mut gdn) = caches(&device, &model, 1);
        let mut got = model
            .forward_qwen35_gather(
                &client,
                &ids(&device, &tokens[..6], 1),
                &mut kv,
                &mut gdn,
                0,
                &index(&device, &[5]),
            )
            .unwrap()
            .to_vec::<f32>();
        for (t, &token) in tokens.iter().enumerate().skip(6) {
            let step = model
                .forward_qwen35_gather(
                    &client,
                    &ids(&device, &[token], 1),
                    &mut kv,
                    &mut gdn,
                    t,
                    &index(&device, &[0]),
                )
                .unwrap();
            got.extend(step.to_vec::<f32>());
        }
        assert_eq!(kv.seq_len(), 9);
        assert_close(&got, &want, 1e-4);
    }
}
