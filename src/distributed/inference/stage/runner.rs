//! [`StageRunner`]: one pipeline stage over a [`LoadedModel`].
//!
//! The model is a [`LlamaStage`](crate::model::llama::LlamaStage) that holds
//! the weights of the stage's layer range only, or a whole model. The
//! runner executes only its range, and its KV caches cover only that range.

use anyhow::{Result, anyhow, bail};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use std::sync::Arc;

use super::super::transport::{Activation, GenRequestHeader, LayerAssignment, StagePayload};
use super::book::RequestBook;
use super::config::{StageConfig, check_assignment, check_held, refusal};
use super::contract::{Stage, StageOutput};
use super::tensors::{host_f32, input_vars};
use crate::inference::LayeredKvCache;
use crate::model::LoadedModel;
use crate::model::traits::ModelClient;

/// One pipeline stage: layers `start_layer..end_layer` of a model, one KV
/// cache per request id.
///
/// The stage with the embedding takes token ids. Every other stage takes
/// the previous stage's [`Activation`]. The stage with the LM head returns
/// the logits of the last position. Every other stage returns its
/// activation. The header's position is the RoPE offset, and it must equal
/// the position the request reached on this stage.
pub struct StageRunner<R: Runtime> {
    model: Arc<LoadedModel<R>>,
    device: R::Device,
    assignment: LayerAssignment,
    config: StageConfig,
    kv_heads: usize,
    head_dim: usize,
    book: RequestBook<LayeredKvCache<R>>,
}

impl<R> StageRunner<R>
where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R> + TypeConversionOps<R>,
{
    /// A stage for `assignment` of `model` on `device`.
    ///
    /// # Errors
    ///
    /// - a model with SSM or Gated DeltaNet state, or a tensor-parallel shard
    /// - a model with no KV head count or head dimension
    /// - an assignment `check_assignment` refuses for this model
    /// - an assignment outside the layers a `LlamaStage` model holds
    /// - a config [`StageConfig::check`] refuses
    pub fn new(
        model: Arc<LoadedModel<R>>,
        device: R::Device,
        assignment: LayerAssignment,
        config: StageConfig,
    ) -> Result<Self> {
        if let Some(reason) = refusal(&*model) {
            bail!(
                "stage layers {}..{}: {reason}",
                assignment.start_layer,
                assignment.end_layer
            );
        }
        check_assignment(&assignment, model.num_layers())?;
        check_held(&assignment, &*model)?;
        config.check(model.max_seq_len())?;
        let (Some(kv_heads), Some(head_dim)) = (model.num_kv_heads(), model.head_dim()) else {
            bail!(
                "a {} model reports no KV head count or head dimension, so a stage cannot \
                 size its KV cache",
                model.model_type()
            );
        };
        Ok(Self {
            book: RequestBook::new(config.max_requests, config.max_seq_len),
            model,
            device,
            assignment,
            config,
            kv_heads,
            head_dim,
        })
    }

    pub fn assignment(&self) -> &LayerAssignment {
        &self.assignment
    }

    /// Requests this stage holds a KV cache for.
    pub fn in_flight(&self) -> usize {
        self.book.len()
    }

    /// Runs one prefill or decode frame. See [`Stage::forward`].
    ///
    /// # Errors
    ///
    /// The request's frame or payload does not fit this stage, or the model
    /// fails. Each error frees the request on this stage.
    pub fn forward(
        &mut self,
        header: &GenRequestHeader,
        payload: StagePayload,
    ) -> Result<StageOutput> {
        let layers = (self.assignment.end_layer - self.assignment.start_layer) as usize;
        let (config, device) = (&self.config, &self.device);
        let (kv_heads, head_dim) = (self.kv_heads, self.head_dim);
        let mut cache = self.book.checkout(header, || {
            LayeredKvCache::new_positional(
                layers,
                1,
                kv_heads,
                config.initial_capacity,
                config.max_seq_len,
                head_dim,
                config.kv_dtype,
                device,
            )
            .map_err(|e| anyhow!("request {}: no KV cache: {e}", header.request_id))
        })?;
        let output = self
            .run(header, payload, &mut cache)
            .map_err(|e| e.context(format!("request {}", header.request_id)))?;
        self.book.checkin(header, cache);
        Ok(output)
    }

    /// Frees `request_id`. Returns whether this stage held it.
    pub fn release(&mut self, request_id: u64) -> bool {
        self.book.release(request_id)
    }

    fn run(
        &self,
        header: &GenRequestHeader,
        payload: StagePayload,
        cache: &mut LayeredKvCache<R>,
    ) -> Result<StageOutput> {
        let seq_len = header.seq_len as usize;
        let (hidden, mlp) = input_vars(
            &*self.model,
            &self.device,
            &self.assignment,
            seq_len,
            payload,
        )?;
        let (hidden, mlp) = self.model.forward_layers_range(
            hidden,
            mlp,
            cache,
            self.assignment.start_layer as usize,
            self.assignment.end_layer as usize,
            header.position as usize,
        )?;
        if !self.assignment.has_lm_head {
            return Ok(StageOutput::Activation(Activation {
                hidden: host_f32(hidden.tensor())?,
                mlp: mlp.map(|m| host_f32(m.tensor())).transpose()?,
            }));
        }
        // Only the last position samples, so only its row goes through the head.
        let last = |v: Var<R>| -> Result<Var<R>> {
            Ok(Var::new(v.tensor().narrow(1, seq_len - 1, 1)?, false))
        };
        let logits = self
            .model
            .forward_head(last(hidden)?, mlp.map(last).transpose()?)?;
        let row = host_f32(&logits)?;
        let vocab = self.model.vocab_size();
        if row.len() != vocab {
            bail!("the LM head returned {} logits, not {vocab}", row.len());
        }
        Ok(StageOutput::Logits(row))
    }
}

impl<R> Stage for StageRunner<R>
where
    R: Runtime<DType = DType>,
    R::Client: ModelClient<R> + TypeConversionOps<R>,
    Self: Send,
{
    fn forward(&mut self, header: &GenRequestHeader, payload: StagePayload) -> Result<StageOutput> {
        StageRunner::forward(self, header, payload)
    }

    fn release(&mut self, request_id: u64) {
        StageRunner::release(self, request_id);
    }

    fn release_all(&mut self) {
        self.book.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::transport::StageOp;
    use super::*;
    use crate::test_utils::{TINY_HIDDEN, TINY_MAX_SEQ_LEN, TINY_VOCAB, cpu_setup, random_llama};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    const LAYERS: usize = 3;

    fn config() -> StageConfig {
        StageConfig {
            max_requests: 4,
            initial_capacity: 4,
            max_seq_len: TINY_MAX_SEQ_LEN,
            kv_dtype: DType::F32,
        }
    }

    fn range(start: u32, end: u32) -> LayerAssignment {
        LayerAssignment {
            start_layer: start,
            end_layer: end,
            has_embedding: start == 0,
            has_lm_head: end as usize == LAYERS,
        }
    }

    struct Rig {
        model: Arc<LoadedModel<CpuRuntime>>,
        device: CpuDevice,
    }

    impl Rig {
        fn new() -> Self {
            let (client, device) = cpu_setup();
            let model = Arc::new(random_llama(&client, &device, LAYERS));
            Self { model, device }
        }

        fn stage(&self, start: u32, end: u32) -> StageRunner<CpuRuntime> {
            StageRunner::new(
                Arc::clone(&self.model),
                self.device.clone(),
                range(start, end),
                config(),
            )
            .unwrap()
        }

        /// The last row of a one-shot full-model prefill of `tokens`.
        fn reference(&self, tokens: &[i64]) -> Vec<f32> {
            let attn_heads = self.model.num_kv_heads().unwrap();
            let mut cache = LayeredKvCache::new_positional(
                LAYERS,
                1,
                attn_heads,
                4,
                TINY_MAX_SEQ_LEN,
                self.model.head_dim().unwrap(),
                DType::F32,
                &self.device,
            )
            .unwrap();
            let ids =
                Tensor::<CpuRuntime>::from_slice(tokens, &[1, tokens.len()], &self.device).unwrap();
            let logits = self
                .model
                .forward_with_kv_cache(&ids, &mut cache, 0)
                .unwrap();
            let all: Vec<f32> = logits.contiguous().unwrap().to_vec();
            all[all.len() - TINY_VOCAB..].to_vec()
        }
    }

    fn header(request_id: u64, op: StageOp, position: usize, seq_len: usize) -> GenRequestHeader {
        GenRequestHeader {
            request_id,
            op,
            position: position as u32,
            seq_len: seq_len as u32,
        }
    }

    /// Runs `tokens` through `stages` in order, the activation crossing the wire format.
    fn pipeline(
        stages: &mut [StageRunner<CpuRuntime>],
        h: GenRequestHeader,
        tokens: &[i64],
    ) -> Result<Vec<f32>> {
        let mut payload = StagePayload::Tokens(tokens.to_vec());
        for stage in stages.iter_mut() {
            match stage.forward(&h, payload)? {
                StageOutput::Activation(a) => {
                    assert_eq!(a.hidden.len(), tokens.len() * TINY_HIDDEN);
                    let bytes = StagePayload::Activation(a).encode()?;
                    payload = StagePayload::decode(&bytes, h.seq_len, false)?;
                }
                StageOutput::Logits(row) => return Ok(row),
            }
        }
        bail!("no stage returned logits")
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(1.0f32, |m, x| m.max(x.abs()));
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                g.is_finite() && (g - w).abs() <= 1e-4 * scale,
                "logit {i}: {g}, the full model gives {w}"
            );
        }
    }

    #[test]
    fn test_one_stage_and_split_stages_match_the_full_model() {
        let rig = Rig::new();
        let prompt = [3, 1, 4, 1, 5];
        let want = rig.reference(&prompt);
        let mut whole = [rig.stage(0, 3)];
        let mut split = [rig.stage(0, 1), rig.stage(1, 2), rig.stage(2, 3)];
        let h = header(1, StageOp::Prefill, 0, prompt.len());
        assert_close(&pipeline(&mut whole, h, &prompt).unwrap(), &want);
        assert_close(&pipeline(&mut split, h, &prompt).unwrap(), &want);
    }

    /// Stages that each loaded only their own layers' tensors give the
    /// logits of the whole model, and a stage refuses a range it lacks.
    #[test]
    fn test_range_loaded_stages_match_the_full_model() {
        use crate::model::llama::LayerSpan;
        use crate::nn::VarBuilder;
        use crate::test_utils::{random_llama_checkpoint, span_map};
        let (client, device) = cpu_setup();
        let (map, model_config) = random_llama_checkpoint(&client, &device, LAYERS, false);
        let all = LayerSpan::new(0..LAYERS, LAYERS).unwrap();
        let mut whole = span_map(&map, &model_config, &all);
        let mut vb = VarBuilder::new(&mut whole, &device);
        let rig = Rig {
            model: Arc::new(LoadedModel::load(&model_config, &mut vb).unwrap()),
            device: device.clone(),
        };
        let load = |start: u32, end: u32| {
            let span = LayerSpan::new(start as usize..end as usize, LAYERS).unwrap();
            let mut part = span_map(&map, &model_config, &span);
            let mut vb = VarBuilder::new(&mut part, &device);
            Arc::new(LoadedModel::load_layers(&model_config, &mut vb, span).unwrap())
        };
        let prompt = [2i64, 7, 1, 8];
        let want = rig.reference(&prompt);
        let mut stages: Vec<StageRunner<CpuRuntime>> = (0..LAYERS as u32)
            .map(|l| StageRunner::new(load(l, l + 1), device.clone(), range(l, l + 1), config()))
            .collect::<Result<_>>()
            .unwrap();
        let h = header(4, StageOp::Prefill, 0, prompt.len());
        assert_close(&pipeline(&mut stages, h, &prompt).unwrap(), &want);

        let err = StageRunner::new(load(1, 2), device, range(1, 3), config())
            .err()
            .unwrap();
        assert!(
            format!("{err:#}").contains("holds layers 1..2 of 3"),
            "{err:#}"
        );
    }

    #[test]
    fn test_positions_advance_across_decode_frames() {
        let rig = Rig::new();
        let tokens = [2, 7, 1, 8, 2, 8];
        let mut stages = [rig.stage(0, 2), rig.stage(2, 3)];
        pipeline(&mut stages, header(9, StageOp::Prefill, 0, 3), &tokens[..3]).unwrap();
        let mut row = Vec::new();
        for p in 3..tokens.len() {
            let h = header(9, StageOp::Decode, p, 1);
            row = pipeline(&mut stages, h, &tokens[p..=p]).unwrap();
        }
        assert_close(&row, &rig.reference(&tokens));
    }

    #[test]
    fn test_request_ids_isolate_their_caches() {
        let rig = Rig::new();
        let (a, b) = ([5i64, 3, 9, 4], [11i64, 2, 6, 6, 1]);
        let mut stages = [rig.stage(0, 1), rig.stage(1, 3)];
        pipeline(&mut stages, header(1, StageOp::Prefill, 0, 2), &a[..2]).unwrap();
        pipeline(&mut stages, header(2, StageOp::Prefill, 0, 3), &b[..3]).unwrap();
        pipeline(&mut stages, header(1, StageOp::Decode, 2, 1), &a[2..3]).unwrap();
        let row_b = pipeline(&mut stages, header(2, StageOp::Decode, 3, 2), &b[3..]).unwrap();
        let row_a = pipeline(&mut stages, header(1, StageOp::Decode, 3, 1), &a[3..]).unwrap();
        assert_close(&row_a, &rig.reference(&a));
        assert_close(&row_b, &rig.reference(&b));
        assert_eq!(stages[0].in_flight(), 2);
    }

    #[test]
    fn test_release_and_errors_free_the_request() {
        let rig = Rig::new();
        let mut stage = rig.stage(0, 3);
        let tokens = StagePayload::Tokens(vec![1, 2]);
        stage
            .forward(&header(1, StageOp::Prefill, 0, 2), tokens.clone())
            .unwrap();
        stage
            .forward(&header(2, StageOp::Prefill, 0, 2), tokens.clone())
            .unwrap();
        assert_eq!(stage.in_flight(), 2);
        assert!(stage.release(1));
        assert_eq!(stage.in_flight(), 1);
        let err = stage
            .forward(
                &header(1, StageOp::Decode, 2, 1),
                StagePayload::Tokens(vec![3]),
            )
            .unwrap_err();
        assert!(format!("{err:#}").contains("not open"), "{err:#}");
        let err = stage
            .forward(
                &header(2, StageOp::Decode, 5, 1),
                StagePayload::Tokens(vec![3]),
            )
            .unwrap_err();
        assert!(format!("{err:#}").contains("is at position 2"), "{err:#}");
        assert_eq!(stage.in_flight(), 0);
    }

    #[test]
    fn test_a_payload_of_the_wrong_kind_or_size_fails() {
        let rig = Rig::new();
        let mut first = rig.stage(0, 1);
        let mut last = rig.stage(1, 3);
        let act = StagePayload::Activation(Activation {
            hidden: vec![0.0; TINY_HIDDEN],
            mlp: None,
        });
        let h = header(1, StageOp::Prefill, 0, 1);
        assert!(first.forward(&h, act.clone()).is_err());
        let err = last.forward(&h, StagePayload::Tokens(vec![1])).unwrap_err();
        assert!(format!("{err:#}").contains("not token ids"), "{err:#}");
        let short = StagePayload::Activation(Activation {
            hidden: vec![0.0; TINY_HIDDEN - 1],
            mlp: None,
        });
        assert!(last.forward(&h, short).is_err());
        assert!(matches!(
            last.forward(&h, act).unwrap(),
            StageOutput::Logits(_)
        ));
        assert_eq!((first.in_flight(), last.in_flight()), (0, 1));
    }

    #[test]
    fn test_a_bad_assignment_or_model_is_refused() {
        let rig = Rig::new();
        let new = |a: LayerAssignment| {
            StageRunner::new(Arc::clone(&rig.model), rig.device.clone(), a, config())
        };
        assert!(new(range(1, 4)).is_err());
        assert!(
            new(LayerAssignment {
                has_lm_head: false,
                ..range(1, 3)
            })
            .is_err()
        );
        let mamba = crate::model::mamba::model::tiny_mamba2(&rig.device);
        let err = StageRunner::new(
            Arc::new(LoadedModel::Mamba2(Box::new(mamba))),
            rig.device.clone(),
            LayerAssignment {
                start_layer: 0,
                end_layer: 1,
                has_embedding: true,
                has_lm_head: true,
            },
            StageConfig {
                max_seq_len: 16,
                ..config()
            },
        )
        .err()
        .unwrap();
        assert!(err.to_string().contains("SSM state"), "{err}");
    }
}
