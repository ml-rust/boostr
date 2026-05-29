//! Standalone image embedder wrapping a vision encoder (SigLIP/CLIP).
//!
//! Takes raw image bytes, preprocesses to normalized tensor, runs the encoder,
//! mean-pools patch tokens to a single vector. Used by the `/v1/embeddings`
//! endpoint when the input contains images.

use std::path::Path;

use crate::error::{Error, Result};
use crate::model::config::VisionConfig;
use crate::model::vision::preprocess::{preprocess_image, preprocess_image_custom};
use crate::model::vision::{ClipEncoder, SigLipEncoder};
use crate::nn::{VarBuilder, VarMap};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps, TensorOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::model::traits::ModelClient;

/// Vision encoder variant backing the embedder.
pub enum VisionEncoderKind<R: Runtime> {
    Clip(Box<ClipEncoder<R>>),
    SigLip(Box<SigLipEncoder<R>>),
}

/// Standalone image embedder: bytes -> preprocess -> encode -> mean pool -> Vec<f32>.
pub struct ImageEmbedder<R: Runtime> {
    encoder: VisionEncoderKind<R>,
    image_size: usize,
    hidden_size: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

impl<R: Runtime<DType = DType>> ImageEmbedder<R> {
    /// Construct from an already-loaded encoder and its config.
    ///
    /// `mean`/`std` are the RGB normalization constants for the training
    /// distribution (ImageNet for CLIP, SigLIP uses `[0.5; 3]`/`[0.5; 3]`).
    pub fn new(
        encoder: VisionEncoderKind<R>,
        image_size: usize,
        hidden_size: usize,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Self {
        Self {
            encoder,
            image_size,
            hidden_size,
            mean,
            std,
        }
    }

    /// Load a standalone SigLIP or CLIP encoder from a single safetensors file.
    ///
    /// The checkpoint is expected to contain the vision tower at the top level
    /// (no `vision_model.` prefix). If your checkpoint uses a prefix, strip it
    /// before calling or load into a `VarMap` manually and use [`Self::new`].
    pub fn from_safetensors<P: AsRef<Path>>(
        path: P,
        config: &VisionConfig,
        device: &R::Device,
    ) -> Result<Self> {
        let mut varmap = VarMap::<R>::from_safetensors(path, device)?;
        let mut vb = VarBuilder::new(&mut varmap, device);
        let encoder = match config.encoder_type.as_str() {
            "clip" => {
                VisionEncoderKind::Clip(Box::new(ClipEncoder::from_varbuilder(&mut vb, config)?))
            }
            "siglip" => VisionEncoderKind::SigLip(Box::new(SigLipEncoder::from_varbuilder(
                &mut vb, config,
            )?)),
            other => {
                return Err(Error::ModelError {
                    reason: format!(
                        "unknown vision encoder type '{other}' (expected 'clip' or 'siglip')"
                    ),
                });
            }
        };

        let (mean, std) = default_norm(&config.encoder_type);
        Ok(Self::new(
            encoder,
            config.image_size,
            config.hidden_size,
            mean,
            std,
        ))
    }

    /// Hidden size of the produced embedding vector.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Embed raw image bytes (PNG/JPEG/...) into a single `[hidden_size]` f32 vector.
    ///
    /// Pipeline: decode → resize to `image_size` → normalize → encoder → mean pool patches.
    /// Caller applies L2-normalize if requested.
    pub fn embed_bytes<C>(&self, client: &C, bytes: &[u8], device: &R::Device) -> Result<Vec<f32>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + BinaryOps<R>
            + ScalarOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + ShapeOps<R>
            + ConvOps<R>
            + ReduceOps<R>
            + UnaryOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        // Preprocess: bytes -> [3, H, W] flat f32
        let pixels = if self.mean == IMAGENET_MEAN && self.std == IMAGENET_STD {
            preprocess_image(bytes, self.image_size)?
        } else {
            preprocess_image_custom(bytes, self.image_size, self.mean, self.std)?
        };

        // [3, H, W] -> [1, 3, H, W]
        let pixel_tensor =
            Tensor::<R>::from_slice(&pixels, &[1, 3, self.image_size, self.image_size], device);

        // Encoder forward: [1, 3, H, W] -> [1, num_patches, hidden]
        let patches = match &self.encoder {
            VisionEncoderKind::Clip(enc) => enc.forward_inference(client, &pixel_tensor)?,
            VisionEncoderKind::SigLip(enc) => enc.forward_inference(client, &pixel_tensor)?,
        };

        // Mean pool over patch dimension: [1, N, H] -> [1, H]
        let pooled = client.mean(&patches, &[1], false).map_err(Error::Numr)?;

        let data: Vec<f32> = pooled.to_vec();
        if data.len() != self.hidden_size {
            return Err(Error::ModelError {
                reason: format!(
                    "unexpected pooled embedding size: got {}, expected {}",
                    data.len(),
                    self.hidden_size
                ),
            });
        }
        Ok(data)
    }

    /// Convenience: embed bytes using the runtime's default client for `device`.
    pub fn embed_bytes_default(&self, bytes: &[u8], device: &R::Device) -> Result<Vec<f32>>
    where
        R::Client: ModelClient<R> + ConvOps<R>,
    {
        let client = R::default_client(device);
        self.embed_bytes(&client, bytes, device)
    }
}

const IMAGENET_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const IMAGENET_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];
const SIGLIP_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
const SIGLIP_STD: [f32; 3] = [0.5, 0.5, 0.5];

fn default_norm(encoder_type: &str) -> ([f32; 3], [f32; 3]) {
    match encoder_type {
        "siglip" => (SIGLIP_MEAN, SIGLIP_STD),
        _ => (IMAGENET_MEAN, IMAGENET_STD),
    }
}
