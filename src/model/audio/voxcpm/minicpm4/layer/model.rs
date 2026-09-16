use crate::model::audio::voxcpm::minicpm4::attention::MiniCpm4Attention;
use crate::model::audio::voxcpm::minicpm4::mlp::MiniCpm4Mlp;
use crate::nn::RmsNorm;
use numr::runtime::Runtime;

pub struct MiniCpm4Layer<R: Runtime> {
    pub(crate) input_layernorm: RmsNorm<R>,
    pub(crate) self_attn: MiniCpm4Attention<R>,
    pub(crate) post_attention_layernorm: RmsNorm<R>,
    pub(crate) mlp: MiniCpm4Mlp<R>,
}
