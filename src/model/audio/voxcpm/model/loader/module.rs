//! The [`Module`] impl for [`VoxCpm2Model`]: whole-model parameter
//! enumeration.

use super::model::VoxCpm2Model;
use crate::model::audio::voxcpm::fsq::loader::FSQ_LAYER_PREFIX;
use crate::model::audio::voxcpm::local_dit::DEFAULT_LOCAL_DIT_PREFIX;
use crate::model::audio::voxcpm::local_encoder::DEFAULT_LOCAL_ENCODER_PREFIX;
use crate::model::audio::voxcpm::minicpm4::{DEFAULT_MINICPM4_PREFIX, DEFAULT_RESIDUAL_LM_PREFIX};
use crate::nn::{Module, extend_named};
use numr::autograd::Var;
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Whole-model parameter enumeration for fine-tuning (e.g. LoRA target
/// matching, [`SimpleTrainer`](crate::trainer::simple::SimpleTrainer)'s
/// `HashMap<TensorId, Tensor<R>>` build). Names are checkpoint keys
/// verbatim, reusing each sub-model's own default checkpoint prefix
/// constant rather than a hand-copied literal:
///
/// - `feat_encoder.*`, `base_lm.*`, `residual_lm.*`, `feat_decoder.*` —
///   [`DEFAULT_LOCAL_ENCODER_PREFIX`], [`DEFAULT_MINICPM4_PREFIX`],
///   [`DEFAULT_RESIDUAL_LM_PREFIX`], [`DEFAULT_LOCAL_DIT_PREFIX`].
/// - `fsq_layer.*` — [`FSQ_LAYER_PREFIX`], for `fsq` (`ScalarQuantization`)
///   only.
/// - `aux`'s six projections (`enc_to_lm_proj`, `lm_to_dit_proj`,
///   `res_to_dit_proj`, `fusion_concat_proj`, `stop_proj`, `stop_head`) live
///   at the checkpoint ROOT with no prefix at all — see
///   [`AuxProjections`](crate::model::audio::voxcpm::fsq::AuxProjections)'s own `Module` impl — so they are appended
///   unprefixed here, unlike every other sub-model.
///
/// `vae_encoder`/`vae_decoder` are DELIBERATELY EXCLUDED: the AudioVAE is a
/// frozen audio codec loaded from a SEPARATE checkpoint
/// (`audiovae.pth`, see the module docs) that is never a
/// fine-tuning target, and neither `AudioVaeEncoder` nor `AudioVaeDecoder`
/// implements `Module<R>` — their `CausalConv1d`/`EncoderBlock` internals
/// were not audited for this unit. A caller needing to enumerate them
/// would need that follow-up unit first.
impl<R: Runtime<DType = DType>> Module<R> for VoxCpm2Model<R> {
    fn parameters(&self) -> Vec<&Var<R>> {
        let mut params = self.feat_encoder.parameters();
        params.extend(self.base_lm.parameters());
        params.extend(self.residual_lm.parameters());
        params.extend(self.feat_decoder.parameters());
        params.extend(self.fsq.parameters());
        params.extend(self.aux.parameters());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Var<R>)> {
        let mut params = Vec::new();
        extend_named(
            &mut params,
            DEFAULT_LOCAL_ENCODER_PREFIX,
            self.feat_encoder.named_parameters(),
        );
        extend_named(
            &mut params,
            DEFAULT_MINICPM4_PREFIX,
            self.base_lm.named_parameters(),
        );
        extend_named(
            &mut params,
            DEFAULT_RESIDUAL_LM_PREFIX,
            self.residual_lm.named_parameters(),
        );
        extend_named(
            &mut params,
            DEFAULT_LOCAL_DIT_PREFIX,
            self.feat_decoder.named_parameters(),
        );
        extend_named(&mut params, FSQ_LAYER_PREFIX, self.fsq.named_parameters());
        // Root-level, no prefix — see the impl doc above.
        params.extend(self.aux.named_parameters());
        params
    }
}
