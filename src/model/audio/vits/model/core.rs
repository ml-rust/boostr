//! [`VitsModel`]: struct definition, checkpoint loading, and the speaker
//! conditioning shared by both duration-predictor variants, the flow, and
//! the decoder. [`super::synthesize`] holds the forward pass.

use crate::error::{Error, Result};
use crate::model::audio::vits::coupling::VitsResidualCouplingBlock;
use crate::model::audio::vits::duration_predictor::{
    VitsDurationPredictor, VitsStochasticDurationPredictor,
};
use crate::model::audio::vits::hifigan::VitsHifiGan;
use crate::model::audio::vits::model::client::VitsModelClient;
use crate::model::audio::vits::model::options::VitsSynthOptions;
use crate::model::audio::vits::text_encoder::VitsTextEncoder;
use crate::model::config::VitsConfig;
use crate::nn::{Embedding, VarBuilder};
use numr::dtype::DType;
use numr::ops::{IndexingOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// The two duration-predictor architectures HF's `VitsModel.__init__`
/// chooses between, per [`VitsConfig::use_stochastic_duration_prediction`].
/// Every published MMS-TTS checkpoint sets this `true` (stochastic); the
/// deterministic variant is reachable only from a config that says so.
/// Both are boxed: the two predictors differ widely in size.
pub(super) enum VitsDurationPredictorKind<R: Runtime> {
    Stochastic(Box<VitsStochasticDurationPredictor<R>>),
    Deterministic(Box<VitsDurationPredictor<R>>),
}

/// The full VITS (MMS-TTS) acoustic model + vocoder, inference path only.
pub struct VitsModel<R: Runtime> {
    pub(super) text_encoder: VitsTextEncoder<R>,
    pub(super) duration_predictor: VitsDurationPredictorKind<R>,
    pub(super) flow: VitsResidualCouplingBlock<R>,
    pub(super) decoder: VitsHifiGan<R>,
    /// Present iff `config.num_speakers > 1` (HF only constructs
    /// `self.embed_speaker` under that condition — never on
    /// `speaker_embedding_size`, which instead gates the `cond` layers
    /// inside the duration predictor / flow / decoder).
    embed_speaker: Option<Embedding<R>>,
    pub(super) config: VitsConfig,
}

impl<R: Runtime<DType = DType>> VitsModel<R> {
    /// Loads `text_encoder.*`, `duration_predictor.*`, `flow.*`,
    /// `decoder.*` and, iff `config.num_speakers > 1`,
    /// `embed_speaker.weight`, under `vb`'s current prefix (the checkpoint's
    /// top level for the real MMS-TTS weights). Does NOT load
    /// `posterior_encoder.*` — that submodule backs only HF's training-time
    /// forward branch, which this crate never runs.
    pub fn from_varbuilder<C>(
        client: &C,
        vb: &mut VarBuilder<'_, R>,
        config: &VitsConfig,
    ) -> Result<Self>
    where
        C: VitsModelClient<R>,
    {
        let mut te_vb = vb.pp("text_encoder");
        let text_encoder = VitsTextEncoder::from_varbuilder(&mut te_vb, config)?;
        drop(te_vb);

        let duration_predictor = {
            let mut dp_vb = vb.pp("duration_predictor");
            if config.use_stochastic_duration_prediction {
                VitsDurationPredictorKind::Stochastic(Box::new(
                    VitsStochasticDurationPredictor::from_varbuilder(&mut dp_vb, config)?,
                ))
            } else {
                VitsDurationPredictorKind::Deterministic(Box::new(
                    VitsDurationPredictor::from_varbuilder(&mut dp_vb, config)?,
                ))
            }
        };

        let mut flow_vb = vb.pp("flow");
        let flow = VitsResidualCouplingBlock::from_varbuilder(client, &mut flow_vb, config)?;
        drop(flow_vb);

        let mut dec_vb = vb.pp("decoder");
        let decoder = VitsHifiGan::from_varbuilder(client, &mut dec_vb, config)?;
        drop(dec_vb);

        let embed_speaker = if config.num_speakers > 1 {
            let mut emb_vb = vb.pp("embed_speaker");
            let weight = emb_vb.take_tensor("weight")?;
            drop(emb_vb);
            Some(Embedding::new(weight, false))
        } else {
            None
        };

        Ok(Self {
            text_encoder,
            duration_predictor,
            flow,
            decoder,
            embed_speaker,
            config: config.clone(),
        })
    }

    pub fn config(&self) -> &VitsConfig {
        &self.config
    }

    /// [`VitsSynthOptions::from_config`] for this model's own config — the
    /// same defaults HF's `VitsModel.forward` uses when a caller changes
    /// nothing.
    pub fn default_synth_options(&self) -> VitsSynthOptions {
        VitsSynthOptions::from_config(&self.config)
    }

    /// HF's `speaker_embeddings = self.embed_speaker(speaker_id).unsqueeze(-1)`,
    /// gated exactly as HF gates it: only when `config.num_speakers > 1` AND
    /// a `speaker_id` is given. `speaker_id` on a single-speaker model, or
    /// no `speaker_id` on a multi-speaker one, both yield `None` — HF does
    /// not error on either.
    pub(super) fn speaker_embedding<C>(
        &self,
        client: &C,
        speaker_id: Option<usize>,
    ) -> Result<Option<Tensor<R>>>
    where
        C: VitsModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let (Some(embed_speaker), true, Some(id)) = (
            &self.embed_speaker,
            self.config.num_speakers > 1,
            speaker_id,
        ) else {
            return Ok(None);
        };
        if id >= self.config.num_speakers {
            return Err(Error::InvalidArgument {
                arg: "speaker_id",
                reason: format!("expected 0..{}, got {id}", self.config.num_speakers),
            });
        }
        let device = client.device();
        let idx = Tensor::<R>::from_slice(&[id as i64], &[1], device).map_err(Error::Numr)?;
        let embedding = embed_speaker.forward(client, &idx)?;
        let embedding = embedding
            .tensor()
            .unsqueeze(2)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?;
        Ok(Some(embedding))
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{build_map, tiny_config};
    use super::*;
    use crate::test_utils::cpu_setup;

    #[test]
    fn from_varbuilder_loads_a_single_speaker_checkpoint_with_no_embed_speaker() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config(1, 0);
        let mut map = build_map(&cfg);
        let mut vb = crate::nn::VarBuilder::new(&mut map, &device);
        let model = VitsModel::from_varbuilder(&client, &mut vb, &cfg).expect("load model");
        assert!(model.embed_speaker.is_none());
        assert!(matches!(
            model.duration_predictor,
            VitsDurationPredictorKind::Deterministic(_)
        ));
    }

    #[test]
    fn from_varbuilder_loads_embed_speaker_when_multi_speaker() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config(3, 2);
        let mut map = build_map(&cfg);
        let mut vb = crate::nn::VarBuilder::new(&mut map, &device);
        let model = VitsModel::from_varbuilder(&client, &mut vb, &cfg).expect("load model");
        assert!(model.embed_speaker.is_some());
    }

    #[test]
    fn speaker_embedding_is_none_without_a_speaker_id_or_single_speaker() {
        let (client, device) = cpu_setup();

        let single_cfg = tiny_config(1, 0);
        let mut single_map = build_map(&single_cfg);
        let mut single_vb = crate::nn::VarBuilder::new(&mut single_map, &device);
        let single_model =
            VitsModel::from_varbuilder(&client, &mut single_vb, &single_cfg).expect("load model");
        assert!(
            single_model
                .speaker_embedding(&client, Some(0))
                .unwrap()
                .is_none(),
            "single-speaker model never conditions, even with a speaker_id given"
        );

        let multi_cfg = tiny_config(3, 2);
        let mut multi_map = build_map(&multi_cfg);
        let mut multi_vb = crate::nn::VarBuilder::new(&mut multi_map, &device);
        let multi_model =
            VitsModel::from_varbuilder(&client, &mut multi_vb, &multi_cfg).expect("load model");
        assert!(
            multi_model
                .speaker_embedding(&client, None)
                .unwrap()
                .is_none(),
            "no speaker_id given means no conditioning, even on a multi-speaker model"
        );
    }

    #[test]
    fn speaker_embedding_produces_the_expected_shape_and_rejects_out_of_range_ids() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config(3, 2);
        let mut map = build_map(&cfg);
        let mut vb = crate::nn::VarBuilder::new(&mut map, &device);
        let model = VitsModel::from_varbuilder(&client, &mut vb, &cfg).expect("load model");

        let embedding = model
            .speaker_embedding(&client, Some(1))
            .expect("valid speaker id")
            .expect("multi-speaker model with a speaker_id conditions");
        assert_eq!(embedding.shape(), &[1, cfg.speaker_embedding_size, 1]);

        let err = model
            .speaker_embedding(&client, Some(cfg.num_speakers))
            .expect_err("speaker id at num_speakers is out of range");
        assert!(matches!(
            err,
            Error::InvalidArgument {
                arg: "speaker_id",
                ..
            }
        ));
    }
}
