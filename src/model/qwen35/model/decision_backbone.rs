//! [`DecisionBackbone`] for [`Qwen35Model`]: the prefix runs as one
//! batch-1 sequence, and row lanes attend to its KV cache through
//! [`AttentionStep::SharedPrefix`] after taking its GDN state.

use super::build::Qwen35Model;
use super::forward_layers::AttentionStep;
use crate::error::{Error, Result};
use crate::inference::decision::{DecisionBackbone, PrefixState};
use crate::inference::kv_cache::layered::LayeredKvCacheConfig;
use crate::inference::{LayeredGdnState, LayeredKvCache};
use crate::model::traits::ModelClient;
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ConvOps, IndexingOps, MatmulOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Qwen35Model<R>
where
    R::Client: IndexingOps<R>,
{
    /// Empty caches for `batch` lanes of `capacity` positions, in the
    /// model's dtype.
    fn decision_caches(
        &self,
        batch: usize,
        capacity: usize,
        device: &R::Device,
    ) -> Result<(LayeredKvCache<R>, LayeredGdnState<R>)> {
        let dtype = self.dtype();
        let kv_config = LayeredKvCacheConfig {
            batch_size: batch,
            num_kv_heads: self.attention_config.num_kv_heads,
            initial_capacity: capacity,
            max_seq_len: capacity,
            head_dim: self.attention_config.head_dim,
            dtype,
        };
        let kv = LayeredKvCache::<R>::new(self.num_attention_layers(), &kv_config, device)?;
        let gdn = LayeredGdnState::<R>::zeros(
            self.num_gdn_layers(),
            &self.gdn_config,
            batch,
            dtype,
            device,
        )?;
        Ok((kv, gdn))
    }
}

impl<R> DecisionBackbone<R> for Qwen35Model<R>
where
    R: Runtime<DType = DType>,
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
    fn max_positions(&self) -> usize {
        self.rope_table_len()
    }

    /// The config's `vocab_size`, the row count of `lm_head` and of the
    /// embedding, as the build checks.
    fn vocab_size(&self) -> usize {
        self.config().vocab_size
    }

    /// Text-only: the suffix's rope positions start at the prefix length.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when a row is `>= tokens.len()`, naming
    /// it. [`Error::ModelError`] when the prefix passes the rope table.
    fn prefill_prefix<C: ModelClient<R>>(
        &self,
        client: &C,
        tokens: &[u32],
        rows: &[usize],
    ) -> Result<PrefixState<R>> {
        let p = tokens.len();
        if let Some(&bad) = rows.iter().find(|&&r| r >= p) {
            return Err(Error::InvalidArgument {
                arg: "rows",
                reason: format!("prefix row {bad} is outside a prefix of {p} positions"),
            });
        }
        let device = client.device();
        let (mut kv, mut gdn) = self.decision_caches(1, p.max(1), device)?;
        let mut hidden = None;
        if p > 0 {
            let ids: Vec<i64> = tokens.iter().map(|&t| i64::from(t)).collect();
            let ids = Tensor::<R>::from_slice(&ids, &[1, p], device)?;
            if rows.is_empty() {
                let positions = self.text_positions(p, 0, device)?;
                let embeds = self.embed_tokens(client, &ids)?;
                self.forward_blocks(
                    client,
                    Var::new(embeds, false),
                    &positions,
                    &mut kv,
                    &mut gdn,
                    &AttentionStep::Shared,
                )?;
            } else {
                let gather: Vec<i64> = rows.iter().map(|&r| r as i64).collect();
                let gather = Tensor::<R>::from_slice(&gather, &[gather.len()], device)?;
                hidden = Some(self.gather_hidden(
                    client,
                    &ids,
                    &mut kv,
                    &mut gdn,
                    0,
                    &gather,
                    &AttentionStep::Shared,
                )?);
            }
        }
        Ok(PrefixState {
            kv,
            gdn,
            len: p,
            rope_end: p,
            rows: rows.to_vec(),
            hidden,
        })
    }

    fn lane_caches(
        &self,
        lanes: usize,
        width: usize,
        device: &R::Device,
    ) -> Result<(LayeredKvCache<R>, LayeredGdnState<R>)> {
        self.decision_caches(lanes, width, device)
    }

    /// # Errors
    ///
    /// [`Error::ModelError`] when `ids` is not `[lanes, width]` or `kv`
    /// does not hold `lanes` lanes; the gather forward's errors propagate,
    /// including a rope error when `prefix.rope_end + width` passes the
    /// table.
    fn forward_rows<C: ModelClient<R>>(
        &self,
        client: &C,
        ids: &Tensor<R>,
        prefix: &PrefixState<R>,
        gdn: &mut LayeredGdnState<R>,
        kv: &mut LayeredKvCache<R>,
        gather: &Tensor<R>,
    ) -> Result<Tensor<R>> {
        let shape = ids.shape();
        let rows = shape.first().copied().unwrap_or(0);
        let lanes = kv.layer(0).map_or(rows, |layer| layer.batch_size());
        if shape.len() != 2 || lanes != rows {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen35: row ids must be [lanes, width] over a cache of {lanes} lanes, \
                     got {shape:?}"
                ),
            });
        }
        self.gather_hidden(
            client,
            ids,
            kv,
            gdn,
            prefix.rope_end,
            gather,
            &AttentionStep::SharedPrefix {
                prefix_kv: &prefix.kv,
            },
        )
    }

    fn candidate_logits<C: ModelClient<R>>(
        &self,
        client: &C,
        hidden: &Tensor<R>,
        ids: &Tensor<R>,
    ) -> Result<Tensor<R>> {
        Qwen35Model::candidate_logits(self, client, hidden, ids)
    }
}

#[cfg(test)]
mod tests {
    use super::super::build::tests::{Head, VOCAB, chunked_model};
    use super::super::forward_gather::tests::{CHUNK, assert_close, index};
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn tokens(n: usize, salt: usize) -> Vec<u32> {
        (0..n)
            .map(|t| ((salt + t * 5 + 2) % VOCAB) as u32)
            .collect()
    }

    /// `[lanes, width]` ids of rows `lens` long, right-padded with id 0,
    /// and the flat index of every real position.
    fn lanes(device: &CpuDevice, lens: &[usize]) -> (Tensor<CpuRuntime>, Vec<i64>) {
        let width = lens.iter().copied().max().unwrap();
        let mut ids = Vec::new();
        let mut real = Vec::new();
        for (l, &n) in lens.iter().enumerate() {
            ids.extend(tokens(n, 3 * l + 1).iter().map(|&t| i64::from(t)));
            ids.extend(std::iter::repeat_n(0, width - n));
            real.extend((0..n).map(|t| (l * width + t) as i64));
        }
        let ids = Tensor::<CpuRuntime>::from_slice(&ids, &[lens.len(), width], device).unwrap();
        (ids, real)
    }

    /// Lanes over the shared prefix against the same lanes with the
    /// prefix KV copied into each one and the ordinary shared step.
    fn check_against_copied_prefix(client: &CpuClient, device: &CpuDevice, prefix_len: usize) {
        let model = chunked_model(device, 0x3535_0060, CHUNK, false, Head::Own);
        let prefix = tokens(prefix_len, 0);
        let state = model.prefill_prefix(client, &prefix, &[]).unwrap();
        let lens = [5usize, 3, 1];
        let (ids, real) = lanes(device, &lens);
        let gather = index(device, &real);

        let (mut kv, mut gdn) = model.lane_caches(lens.len(), 5, device).unwrap();
        for l in 0..lens.len() {
            gdn.install_lane(client, l, &state.gdn).unwrap();
        }
        let got = model
            .forward_rows(client, &ids, &state, &mut gdn, &mut kv, &gather)
            .unwrap()
            .to_vec::<f32>();

        let (mut kv, mut gdn) = model
            .decision_caches(lens.len(), prefix_len + 5, device)
            .unwrap();
        for l in 0..lens.len() {
            if prefix_len > 0 {
                kv.install_lane(&state.kv, l, client).unwrap();
            }
            gdn.install_lane(client, l, &state.gdn).unwrap();
        }
        let want = model
            .forward_qwen35_gather(client, &ids, &mut kv, &mut gdn, prefix_len, &gather)
            .unwrap()
            .to_vec::<f32>();
        assert_close(&got, &want, 1e-5);
    }

    /// A prefix of 6 crosses one GDN chunk boundary; the model has two
    /// query heads over one KV head.
    #[test]
    fn shared_prefix_matches_copied_prefix() {
        let (client, device) = cpu_setup();
        check_against_copied_prefix(&client, &device, 6);
    }

    #[test]
    fn empty_prefix_takes_the_shared_step() {
        let (client, device) = cpu_setup();
        check_against_copied_prefix(&client, &device, 0);
    }

    /// Lanes holding 6 and 4 keys cannot take a shared-cursor append.
    #[test]
    fn shared_step_refuses_unequal_lanes() {
        let (client, device) = cpu_setup();
        let model = chunked_model(&device, 0x3535_0061, CHUNK, false, Head::Own);
        let long = model.prefill_prefix(&client, &tokens(6, 0), &[]).unwrap();
        let short = model.prefill_prefix(&client, &tokens(4, 0), &[]).unwrap();
        let (mut kv, mut gdn) = model.decision_caches(2, 16, &device).unwrap();
        for (lane, state) in [&long, &short].into_iter().enumerate() {
            kv.install_lane(&state.kv, lane, &client).unwrap();
            gdn.install_lane(&client, lane, &state.gdn).unwrap();
        }
        let ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2], &[2, 1], &device).unwrap();
        let err = model
            .forward_qwen35_gather(
                &client,
                &ids,
                &mut kv,
                &mut gdn,
                6,
                &index(&device, &[0, 1]),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("the lanes hold [6, 4] keys"), "{err}");

        // The shared-prefix step appends at the lane cursor the same way.
        let err = model
            .forward_rows(
                &client,
                &ids,
                &long,
                &mut gdn,
                &mut kv,
                &index(&device, &[0]),
            )
            .unwrap_err()
            .to_string();
        assert!(err.contains("the lanes hold [6, 4] keys"), "{err}");
    }

    #[test]
    fn prefix_rows_are_checked_and_kept() {
        let (client, device) = cpu_setup();
        let model = chunked_model(&device, 0x3535_0062, CHUNK, false, Head::Own);
        let prefix = tokens(6, 0);
        let Err(err) = model.prefill_prefix(&client, &prefix, &[2, 6]) else {
            panic!("a prefix row past the prefix must be refused");
        };
        let err = err.to_string();
        assert!(
            err.contains("prefix row 6 is outside a prefix of 6"),
            "{err}"
        );
        let state = model.prefill_prefix(&client, &prefix, &[5, 1]).unwrap();
        assert_eq!((state.len, state.rope_end), (6, 6));
        assert_eq!(state.hidden_row(1), Some(1));
        let ids: Vec<i64> = prefix.iter().map(|&t| i64::from(t)).collect();
        let ids = Tensor::<CpuRuntime>::from_slice(&ids, &[1, 6], &device).unwrap();
        let (mut kv, mut gdn) = model.lane_caches(1, 6, &device).unwrap();
        let want = model
            .forward_qwen35_gather(
                &client,
                &ids,
                &mut kv,
                &mut gdn,
                0,
                &index(&device, &[5, 1]),
            )
            .unwrap()
            .to_vec::<f32>();
        let got = state.hidden.as_ref().unwrap().to_vec::<f32>();
        assert_close(&got, &want, 1e-6);
    }
}
