use crate::model::audio::voxcpm::bidirectional::attention::BidirectionalAttention;
use crate::model::audio::voxcpm::bidirectional::mlp::BidirectionalMlp;
use crate::nn::RmsNorm;
use numr::runtime::Runtime;

pub struct BidirectionalLayer<R: Runtime> {
    pub(crate) input_layernorm: RmsNorm<R>,
    pub(crate) self_attn: BidirectionalAttention<R>,
    pub(crate) post_attention_layernorm: RmsNorm<R>,
    pub(crate) mlp: BidirectionalMlp<R>,
}
