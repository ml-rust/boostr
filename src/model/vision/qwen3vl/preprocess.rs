//! Image geometry and normalization for the Qwen3-VL tower.
//!
//! Pipeline: decode -> [`smart_resize`] picks a target size on a
//! `patch * merge` grid -> [`resize_pad_ceil`] scales the image to fit and
//! centers it on black -> [`normalize_chw`] maps bytes to planar floats.
//! The float arithmetic in [`smart_resize`] and [`resize_pad_ceil`] runs in
//! `f32`, in the same order as `mtmd-image.cpp`, so the chosen sizes agree
//! with the reference for every input.

use super::bicubic::{RgbImage, resize_bicubic};
use super::config::Qwen3VlVisionConfig;
use crate::error::{Error, Result};

/// One image ready for [`super::Qwen3VlVision::encode`].
#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessedImage {
    /// `3 * height * width` normalized floats, planar CHW.
    pub pixels: Vec<f32>,
    /// Width in pixels after resize, a multiple of `patch * merge`.
    pub width: usize,
    /// Height in pixels after resize, a multiple of `patch * merge`.
    pub height: usize,
    /// Output tokens per row: `width / (patch * merge)`.
    pub nx: usize,
    /// Output token rows: `height / (patch * merge)`.
    pub ny: usize,
}

impl PreprocessedImage {
    /// Number of output tokens.
    pub fn n_tokens(&self) -> usize {
        self.nx * self.ny
    }
}

/// Decode PNG, JPEG or any other format the `image` crate reads into RGB8.
pub fn decode_image(bytes: &[u8]) -> Result<RgbImage> {
    let img = image::load_from_memory(bytes).map_err(|e| Error::ModelError {
        reason: format!("qwen3vl vision: image decode failed: {e}"),
    })?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    RgbImage::new(w, h, rgb.into_raw()).ok_or_else(|| Error::ModelError {
        reason: format!("qwen3vl vision: decoded {w}x{h} image has a short buffer"),
    })
}

/// Target size preserving aspect ratio on an `align` grid
/// (`calc_size_preserved_ratio` in `mtmd-image.cpp`).
///
/// Each side rounds to the nearest multiple of `align` (at least `align`).
/// An area above `max_pixels` shrinks by `sqrt(area / max)` and floors to
/// the grid. An area below `min_pixels` grows by `sqrt(min / area)` and
/// ceils to the grid. Returns `(width, height)`; a zero side returns
/// `(0, 0)`.
pub fn smart_resize(
    width: usize,
    height: usize,
    align: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    if width == 0 || height == 0 || align == 0 {
        return (0, 0);
    }
    let f = align as f32;
    let round_by = |x: f32| (x / f).round() as i64 * align as i64;
    let ceil_by = |x: f32| (x / f).ceil() as i64 * align as i64;
    let floor_by = |x: f32| (x / f).floor() as i64 * align as i64;
    let (w, h) = (width as f32, height as f32);
    let a = align as i64;

    let mut w_bar = round_by(w).max(a);
    let mut h_bar = round_by(h).max(a);
    if max_pixels > 0 && h_bar * w_bar > max_pixels as i64 {
        let beta = (h * w / max_pixels as f32).sqrt();
        h_bar = floor_by(h / beta).max(a);
        w_bar = floor_by(w / beta).max(a);
    } else if min_pixels > 0 && h_bar * w_bar < min_pixels as i64 {
        let beta = (min_pixels as f32 / (h * w)).sqrt();
        h_bar = ceil_by(h * beta);
        w_bar = ceil_by(w * beta);
    }
    (w_bar as usize, h_bar as usize)
}

/// Scaled size and top-left offset of the image inside a `tw x th` canvas
/// under `PAD_CEIL`: one scale fits both sides, each side ceils, and the
/// remainder splits with the floor half on the top and left.
pub fn pad_ceil_geometry(
    width: usize,
    height: usize,
    tw: usize,
    th: usize,
) -> (usize, usize, usize, usize) {
    let scale_w = tw as f32 / width as f32;
    let scale_h = th as f32 / height as f32;
    let scale = scale_w.min(scale_h);
    let new_w = ((width as f32 * scale).ceil() as usize).min(tw);
    let new_h = ((height as f32 * scale).ceil() as usize).min(th);
    let off_x = (tw - new_w) / 2;
    let off_y = (th - new_h) / 2;
    (new_w, new_h, off_x, off_y)
}

/// Resize `src` onto a `tw x th` black canvas with `PAD_CEIL` placement
/// (`img_tool::resize` in `mtmd-image.cpp`). A source already at the
/// target size is copied unchanged.
pub fn resize_pad_ceil(src: &RgbImage, tw: usize, th: usize) -> RgbImage {
    if src.width == tw && src.height == th {
        return src.clone();
    }
    let (new_w, new_h, off_x, off_y) = pad_ceil_geometry(src.width, src.height, tw, th);
    let scaled = resize_bicubic(src, new_w, new_h);
    let mut canvas = RgbImage::filled(tw, th, [0, 0, 0]);
    canvas.composite(&scaled, off_x, off_y);
    canvas
}

/// Map bytes to `(byte / 255 - mean) / std`, planar CHW.
pub fn normalize_chw(img: &RgbImage, mean: &[f32; 3], std: &[f32; 3]) -> Vec<f32> {
    let n = img.width * img.height;
    let mut out = vec![0f32; n * 3];
    for p in 0..n {
        for c in 0..3 {
            let v = img.data[p * 3 + c] as f32 / 255.0;
            out[c * n + p] = (v - mean[c]) / std[c];
        }
    }
    out
}

/// Full pipeline on a decoded image.
pub fn preprocess(src: &RgbImage, cfg: &Qwen3VlVisionConfig) -> Result<PreprocessedImage> {
    let align = cfg.align();
    let (tw, th) = smart_resize(
        src.width,
        src.height,
        align,
        cfg.min_pixels(),
        cfg.max_pixels(),
    );
    if tw == 0 || th == 0 {
        return Err(Error::ModelError {
            reason: format!(
                "qwen3vl vision: image {}x{} has an empty side",
                src.width, src.height
            ),
        });
    }
    let resized = resize_pad_ceil(src, tw, th);
    let pixels = normalize_chw(&resized, &cfg.image_mean, &cfg.image_std);
    Ok(PreprocessedImage {
        pixels,
        width: tw,
        height: th,
        nx: tw / align,
        ny: th / align,
    })
}

/// Full pipeline on encoded image bytes.
pub fn preprocess_bytes(bytes: &[u8], cfg: &Qwen3VlVisionConfig) -> Result<PreprocessedImage> {
    let img = decode_image(bytes)?;
    preprocess(&img, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::vision::qwen3vl::config::bonsai2_test_config;

    #[test]
    fn smart_resize_rounds_to_grid() {
        assert_eq!(
            smart_resize(400, 300, 32, 8 * 1024, 4096 * 1024),
            (416, 288)
        );
        assert_eq!(
            smart_resize(256, 192, 32, 8 * 1024, 4096 * 1024),
            (256, 192)
        );
    }

    #[test]
    fn smart_resize_grows_below_min_area() {
        // 96x64 -> 6144 px < 8192: beta = sqrt(8192 / 6144), ceil to 32.
        assert_eq!(smart_resize(96, 64, 32, 8 * 1024, 4096 * 1024), (128, 96));
    }

    #[test]
    fn smart_resize_shrinks_above_max_area() {
        // 4000x3000 -> 12M px > 4194304: beta = 1.6915, 4000 / beta = 2364.8
        // and 3000 / beta = 1773.6, both floored to the 32 grid.
        let (w, h) = smart_resize(4000, 3000, 32, 8 * 1024, 4096 * 1024);
        assert_eq!((w, h), (2336, 1760));
        assert!(w * h <= 4096 * 1024);
    }

    #[test]
    fn smart_resize_never_below_align() {
        let (w, h) = smart_resize(3, 3, 32, 0, 0);
        assert_eq!((w, h), (32, 32));
    }

    #[test]
    fn pad_ceil_centers_with_floor_offset() {
        // 96x64 into 128x96: scale 4/3, 64 * 4/3 = 85.33 -> 86, offset (96-86)/2 = 5.
        assert_eq!(pad_ceil_geometry(96, 64, 128, 96), (128, 86, 0, 5));
        // 400x300 into 416x288: scale 0.96, 400 * 0.96 = 384, offset 16.
        assert_eq!(pad_ceil_geometry(400, 300, 416, 288), (384, 288, 16, 0));
    }

    #[test]
    fn resize_pad_ceil_pads_black() {
        let src = RgbImage::filled(96, 64, [255, 255, 255]);
        let out = resize_pad_ceil(&src, 128, 96);
        assert_eq!((out.width, out.height), (128, 96));
        // padded rows 0..5 and 91..96 stay black
        assert!(out.data[..5 * 128 * 3].iter().all(|&b| b == 0));
        assert!(out.data[91 * 128 * 3..].iter().all(|&b| b == 0));
        // interior is white
        assert!(
            out.data[5 * 128 * 3..91 * 128 * 3]
                .iter()
                .all(|&b| b == 255)
        );
    }

    #[test]
    fn normalize_maps_zero_to_minus_one() {
        let img = RgbImage::new(2, 1, vec![0, 255, 128, 255, 0, 128]).unwrap();
        let out = normalize_chw(&img, &[0.5; 3], &[0.5; 3]);
        // planar: R = [0, 255], G = [255, 0], B = [128, 128]
        assert_eq!(out[0], -1.0);
        assert_eq!(out[1], 1.0);
        assert_eq!(out[2], 1.0);
        assert_eq!(out[3], -1.0);
        assert!((out[4] - (128.0 / 255.0 - 0.5) / 0.5).abs() < 1e-7);
    }

    #[test]
    fn preprocess_reports_token_grid() {
        let cfg = bonsai2_test_config();
        let src = RgbImage::filled(400, 300, [10, 20, 30]);
        let out = preprocess(&src, &cfg).unwrap();
        assert_eq!((out.width, out.height), (416, 288));
        assert_eq!((out.nx, out.ny), (13, 9));
        assert_eq!(out.pixels.len(), 3 * 416 * 288);
    }
}
