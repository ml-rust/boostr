//! Multimodal model wrapper combining vision/audio encoders with an LLM backbone.

use crate::error::{Error, Result};
use crate::model::audio::WhisperEncoder;
use crate::model::config::{AudioConfig, UniversalConfig, VisionConfig};
use crate::model::registry::LoadedModel;
use crate::model::vision::{ClipEncoder, MultimodalProjector, SigLipEncoder};
use crate::nn::VarBuilder;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, IndexingOps, MatmulOps, NormalizationOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Vision encoder variant.
pub enum VisionEncoderVariant<R: Runtime> {
    /// OpenAI CLIP ViT encoder.
    Clip(Box<ClipEncoder<R>>),
    /// Google SigLIP encoder.
    SigLip(Box<SigLipEncoder<R>>),
}

impl<R: Runtime> VisionEncoderVariant<R> {
    /// Forward pass through the vision encoder.
    ///
    /// `pixel_values`: `[B, 3, H, W]` normalized image tensor.
    /// Returns: `[B, num_patches, vision_hidden]`
    pub fn forward_inference<C>(&self, client: &C, pixel_values: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + ShapeOps<R>
            + ConvOps<R>
            + UnaryOps<R>,
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        match self {
            Self::Clip(enc) => enc.forward_inference(client, pixel_values),
            Self::SigLip(enc) => enc.forward_inference(client, pixel_values),
        }
    }
}

/// Multimodal model: vision/audio encoders + projectors + LLM backbone.
///
/// Wraps a `LoadedModel` (the text LLM) with optional vision and audio
/// encoder pipelines. Each encoder pipeline consists of an encoder
/// (CLIP/SigLIP/Whisper) followed by a projector that maps encoder
/// hidden states to the LLM's embedding dimension.
pub struct MultimodalModel<R: Runtime> {
    vision_encoder: Option<VisionEncoderVariant<R>>,
    vision_projector: Option<MultimodalProjector<R>>,
    audio_encoder: Option<WhisperEncoder<R>>,
    audio_projector: Option<MultimodalProjector<R>>,
    llm: LoadedModel<R>,
    config: UniversalConfig,
}

impl<R: Runtime<DType = DType>> MultimodalModel<R>
where
    R::Client: IndexingOps<R> + crate::quant::DequantOps<R> + numr::ops::TypeConversionOps<R>,
{
    /// Load a multimodal model from a VarBuilder and config.
    ///
    /// Loads vision/audio encoders from their respective weight prefixes,
    /// then loads the LLM backbone using a config copy with vision/audio
    /// set to `None` (so `LoadedModel::load` dispatches to the text model).
    pub fn from_varbuilder(vb: &mut VarBuilder<R>, config: &UniversalConfig) -> Result<Self> {
        // Load vision encoder + projector if configured
        let (vision_encoder, vision_projector) = if let Some(ref vision_config) = config.vision {
            let (enc, proj) = Self::load_vision(vb, vision_config, config.hidden_size)?;
            (Some(enc), Some(proj))
        } else {
            (None, None)
        };

        // Load audio encoder + projector if configured
        let (audio_encoder, audio_projector) = if let Some(ref audio_config) = config.audio {
            let (enc, proj) = Self::load_audio(vb, audio_config, config.hidden_size)?;
            (Some(enc), Some(proj))
        } else {
            (None, None)
        };

        // Load LLM backbone with vision/audio stripped from config
        let mut llm_config = config.clone();
        llm_config.vision = None;
        llm_config.audio = None;
        let llm = LoadedModel::load(&llm_config, vb)?;

        Ok(Self {
            vision_encoder,
            vision_projector,
            audio_encoder,
            audio_projector,
            llm,
            config: config.clone(),
        })
    }

    /// Load vision encoder and projector from the VarBuilder.
    fn load_vision(
        vb: &mut VarBuilder<R>,
        vision_config: &VisionConfig,
        llm_hidden: usize,
    ) -> Result<(VisionEncoderVariant<R>, MultimodalProjector<R>)> {
        let mut vision_vb = vb.pp("vision_model");
        let encoder = match vision_config.encoder_type.as_str() {
            "clip" => {
                let enc = ClipEncoder::from_varbuilder(&mut vision_vb, vision_config)?;
                VisionEncoderVariant::Clip(Box::new(enc))
            }
            "siglip" => {
                let enc = SigLipEncoder::from_varbuilder(&mut vision_vb, vision_config)?;
                VisionEncoderVariant::SigLip(Box::new(enc))
            }
            other => {
                return Err(Error::ModelError {
                    reason: format!(
                        "unknown vision encoder type: '{other}', expected 'clip' or 'siglip'"
                    ),
                });
            }
        };
        drop(vision_vb);

        let mut proj_vb = vb.pp("multi_modal_projector");
        let projector = MultimodalProjector::from_varbuilder(
            &mut proj_vb,
            vision_config.hidden_size,
            llm_hidden,
            vision_config,
        )?;

        Ok((encoder, projector))
    }

    /// Load audio encoder and projector from the VarBuilder.
    fn load_audio(
        vb: &mut VarBuilder<R>,
        audio_config: &AudioConfig,
        llm_hidden: usize,
    ) -> Result<(WhisperEncoder<R>, MultimodalProjector<R>)> {
        let mut audio_vb = vb.pp("audio_encoder");
        let encoder = WhisperEncoder::from_varbuilder(&mut audio_vb, audio_config)?;
        drop(audio_vb);

        // Build a VisionConfig for the projector (it accepts VisionConfig for type/hidden params)
        let proj_vision_config = VisionConfig {
            encoder_type: String::new(),
            hidden_size: audio_config.hidden_size,
            num_layers: 0,
            num_heads: 0,
            patch_size: 1,
            image_size: 1,
            intermediate_size: 0,
            projector_type: audio_config.projector_type.clone(),
            projector_depth: 2,
            select_layer: None,
        };

        let mut proj_vb = vb.pp("audio_projector");
        let projector = MultimodalProjector::from_varbuilder(
            &mut proj_vb,
            audio_config.hidden_size,
            llm_hidden,
            &proj_vision_config,
        )?;

        Ok((encoder, projector))
    }
}

impl<R: Runtime<DType = DType>> MultimodalModel<R> {
    /// Encode images through the vision encoder and projector.
    ///
    /// `pixel_values`: `[B, 3, H, W]` normalized image tensor.
    /// Returns: `[B, num_image_tokens, llm_hidden]`
    pub fn encode_images<C>(&self, client: &C, pixel_values: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + ShapeOps<R>
            + ConvOps<R>
            + UnaryOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let encoder = self
            .vision_encoder
            .as_ref()
            .ok_or_else(|| Error::ModelError {
                reason: "encode_images called but no vision encoder is loaded".into(),
            })?;
        let projector = self
            .vision_projector
            .as_ref()
            .ok_or_else(|| Error::ModelError {
                reason: "encode_images called but no vision projector is loaded".into(),
            })?;

        let vision_features = encoder.forward_inference(client, pixel_values)?;
        projector.forward_inference(client, &vision_features)
    }

    /// Encode audio through the Whisper encoder and projector.
    ///
    /// `mel`: `[B, num_mel_bins, audio_len]` log-mel spectrogram.
    /// Returns: `[B, num_audio_tokens, llm_hidden]`
    pub fn encode_audio<C>(&self, client: &C, mel: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + ActivationOps<R>
            + NormalizationOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + ShapeOps<R>
            + UnaryOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + ConvOps<R> + ReduceOps<R> + BinaryOps<R>,
    {
        let encoder = self
            .audio_encoder
            .as_ref()
            .ok_or_else(|| Error::ModelError {
                reason: "encode_audio called but no audio encoder is loaded".into(),
            })?;
        let projector = self
            .audio_projector
            .as_ref()
            .ok_or_else(|| Error::ModelError {
                reason: "encode_audio called but no audio projector is loaded".into(),
            })?;

        let audio_features = encoder.forward_inference(client, mel)?;
        projector.forward_inference(client, &audio_features)
    }

    /// Get the universal config.
    pub fn config(&self) -> &UniversalConfig {
        &self.config
    }

    /// Get a reference to the inner LLM backbone.
    pub fn llm(&self) -> &LoadedModel<R> {
        &self.llm
    }
}
