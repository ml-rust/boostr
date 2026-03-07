//! Image preprocessing for vision encoders.
//!
//! NOTE: Requires the `image` crate as a dependency in boostr's Cargo.toml.
//! If not yet added, add: `image = { version = "0.25", default-features = false, features = ["png", "jpeg"] }`

use crate::error::{Error, Result};

/// ImageNet normalization constants used by CLIP and SigLIP.
const IMAGENET_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const IMAGENET_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

/// Preprocess raw image bytes into a normalized float buffer in [C, H, W] layout.
///
/// Steps:
/// 1. Decode image from bytes (PNG, JPEG, etc.)
/// 2. Resize to `image_size x image_size` using bilinear interpolation
/// 3. Convert to float [0, 1]
/// 4. Normalize with ImageNet mean/std
/// 5. Return as `Vec<f32>` in channels-first [C, H, W] layout
///
/// The caller constructs a `Tensor` from this buffer on the desired device.
///
/// Returns `3 * image_size * image_size` floats.
pub fn preprocess_image(bytes: &[u8], image_size: usize) -> Result<Vec<f32>> {
    // Decode image
    let img = image::load_from_memory(bytes).map_err(|e| Error::ModelError {
        reason: format!("failed to decode image: {e}"),
    })?;

    // Resize to target size using bilinear interpolation
    let resized = img.resize_exact(
        image_size as u32,
        image_size as u32,
        image::imageops::FilterType::Triangle,
    );

    // Convert to RGB8
    let rgb = resized.to_rgb8();

    // Build channels-first normalized output: [C, H, W]
    let total = 3 * image_size * image_size;
    let mut output = Vec::with_capacity(total);

    for c in 0..3 {
        for y in 0..image_size {
            for x in 0..image_size {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                let val = pixel[c] as f32 / 255.0;
                let normalized = (val - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                output.push(normalized);
            }
        }
    }

    Ok(output)
}

/// Preprocess with custom normalization constants.
pub fn preprocess_image_custom(
    bytes: &[u8],
    image_size: usize,
    mean: [f32; 3],
    std: [f32; 3],
) -> Result<Vec<f32>> {
    let img = image::load_from_memory(bytes).map_err(|e| Error::ModelError {
        reason: format!("failed to decode image: {e}"),
    })?;

    let resized = img.resize_exact(
        image_size as u32,
        image_size as u32,
        image::imageops::FilterType::Triangle,
    );

    let rgb = resized.to_rgb8();

    let total = 3 * image_size * image_size;
    let mut output = Vec::with_capacity(total);

    for c in 0..3 {
        for y in 0..image_size {
            for x in 0..image_size {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                let val = pixel[c] as f32 / 255.0;
                let normalized = (val - mean[c]) / std[c];
                output.push(normalized);
            }
        }
    }

    Ok(output)
}
