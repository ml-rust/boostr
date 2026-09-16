use crate::model::audio::neucodec::config::NeuCodecDecoderConfig;
use crate::model::audio::neucodec::istft_head::IstftHead;
use crate::model::audio::neucodec::resnet_block::ResnetBlock;
use crate::model::audio::neucodec::transformer_block::TransformerBlock;
use crate::nn::{Conv1d, LayerNorm, Linear};
use numr::runtime::Runtime;

/// Bundled, already-built weights for the full acoustic decoder.
pub struct NeuCodecDecoderWeights<R: Runtime> {
    pub fc: Linear<R>,
    pub embed: Conv1d<R>,
    pub prior_net: Vec<ResnetBlock<R>>,
    pub layers: Vec<TransformerBlock<R>>,
    pub norm: LayerNorm<R>,
    pub post_net: Vec<ResnetBlock<R>>,
    pub head: IstftHead<R>,
}

/// NeuCodec acoustic decoder: FSQ features -> waveform.
pub struct NeuCodecDecoder<R: Runtime> {
    pub(super) config: NeuCodecDecoderConfig,
    pub(super) fc: Linear<R>,
    pub(super) embed: Conv1d<R>,
    pub(super) prior_net: Vec<ResnetBlock<R>>,
    pub(super) layers: Vec<TransformerBlock<R>>,
    pub(super) norm: LayerNorm<R>,
    pub(super) post_net: Vec<ResnetBlock<R>>,
    pub(super) head: IstftHead<R>,
}
