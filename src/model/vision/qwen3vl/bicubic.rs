//! Separable bicubic resampler in 22-bit fixed point.
//!
//! Matches the fixed-point bicubic in `mtmd-image.cpp` (`resize_pillow`,
//! `RESIZE_ALGO_BICUBIC`) byte for byte: the same filter (`a = -0.5`),
//! the same f64 weight normalization, the same rounding into `i32`
//! weights, and the same accumulate-shift-clamp per output sample.

/// Fractional bits of one fixed-point weight.
const PRECISION_BITS: u32 = 22;
/// Filter support radius for bicubic, in source samples at scale 1.
const FILTER_SUPPORT: f64 = 2.0;

/// Interleaved 8-bit RGB image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbImage {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `height * width * 3` bytes, row-major, RGB per pixel.
    pub data: Vec<u8>,
}

impl RgbImage {
    /// Build an image from interleaved RGB bytes. Panics never: a wrong
    /// byte count returns `None`.
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> Option<Self> {
        (data.len() == width * height * 3).then_some(Self {
            width,
            height,
            data,
        })
    }

    /// Solid-color image.
    pub fn filled(width: usize, height: usize, color: [u8; 3]) -> Self {
        let mut data = Vec::with_capacity(width * height * 3);
        for _ in 0..width * height {
            data.extend_from_slice(&color);
        }
        Self {
            width,
            height,
            data,
        }
    }

    /// Copy `src` into `self` at `(offset_x, offset_y)`. Pixels that fall
    /// outside `self` are skipped.
    pub fn composite(&mut self, src: &RgbImage, offset_x: usize, offset_y: usize) {
        for y in 0..src.height {
            let dy = y + offset_y;
            if dy >= self.height {
                continue;
            }
            for x in 0..src.width {
                let dx = x + offset_x;
                if dx >= self.width {
                    continue;
                }
                let s = (y * src.width + x) * 3;
                let d = (dy * self.width + dx) * 3;
                self.data[d..d + 3].copy_from_slice(&src.data[s..s + 3]);
            }
        }
    }

    /// Planar CHW copy of the bytes.
    pub fn to_chw(&self) -> Vec<u8> {
        let n = self.width * self.height;
        let mut out = vec![0u8; n * 3];
        for p in 0..n {
            for c in 0..3 {
                out[c * n + p] = self.data[p * 3 + c];
            }
        }
        out
    }
}

/// Bicubic kernel with `a = -0.5`, zero outside `[-2, 2]`.
fn bicubic_filter(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

/// Per-output-sample kernel: first source index, count, and fixed-point
/// weights (`ksize` per output sample, zero-padded).
struct Kernel {
    ksize: usize,
    bounds: Vec<(usize, usize)>,
    weights: Vec<i32>,
}

fn precompute_weights(in_size: usize, out_size: usize) -> Kernel {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = FILTER_SUPPORT * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut pre = vec![0f64; out_size * ksize];
    let mut bounds = Vec::with_capacity(out_size);
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        // Truncation toward zero, then clamp, as `static_cast<int>` does.
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).max(0) as usize;
        let xmax = xmax.min(in_size);
        let count = xmax.saturating_sub(xmin);
        let row = &mut pre[xx * ksize..(xx + 1) * ksize];
        let mut ww = 0.0;
        for (x, w) in row.iter_mut().enumerate().take(count) {
            *w = bicubic_filter(((x + xmin) as f64 - center + 0.5) * ss);
            ww += *w;
        }
        if ww != 0.0 {
            for w in row.iter_mut().take(count) {
                *w /= ww;
            }
        }
        bounds.push((xmin, count));
    }
    let fxp_scale = (1u64 << PRECISION_BITS) as f64;
    let weights = pre
        .iter()
        .map(|&w| {
            let rounded = w * fxp_scale + if w < 0.0 { -0.5 } else { 0.5 };
            rounded as i32
        })
        .collect();
    Kernel {
        ksize,
        bounds,
        weights,
    }
}

fn clip8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

fn resample_horizontal(src: &[u8], in_w: usize, in_h: usize, out_w: usize, k: &Kernel) -> Vec<u8> {
    let mut out = vec![0u8; out_w * in_h * 3];
    let bias = 1i32 << (PRECISION_BITS - 1);
    for yy in 0..in_h {
        let src_row = &src[yy * in_w * 3..(yy + 1) * in_w * 3];
        let dst_row = &mut out[yy * out_w * 3..(yy + 1) * out_w * 3];
        for xx in 0..out_w {
            let (xmin, count) = k.bounds[xx];
            let kw = &k.weights[xx * k.ksize..xx * k.ksize + count];
            let mut acc = [bias; 3];
            for (x, &w) in kw.iter().enumerate() {
                let p = &src_row[(xmin + x) * 3..(xmin + x) * 3 + 3];
                for c in 0..3 {
                    acc[c] = acc[c].wrapping_add((p[c] as i32).wrapping_mul(w));
                }
            }
            for c in 0..3 {
                dst_row[xx * 3 + c] = clip8(acc[c] >> PRECISION_BITS);
            }
        }
    }
    out
}

fn resample_vertical(src: &[u8], in_w: usize, out_h: usize, k: &Kernel) -> Vec<u8> {
    let row_elems = in_w * 3;
    let mut out = vec![0u8; row_elems * out_h];
    let bias = 1i32 << (PRECISION_BITS - 1);
    let mut acc = vec![0i32; row_elems];
    for yy in 0..out_h {
        let (ymin, count) = k.bounds[yy];
        let kw = &k.weights[yy * k.ksize..yy * k.ksize + count];
        acc.fill(bias);
        for (y, &w) in kw.iter().enumerate() {
            let src_row = &src[(ymin + y) * row_elems..(ymin + y + 1) * row_elems];
            for (a, &s) in acc.iter_mut().zip(src_row) {
                *a = a.wrapping_add((s as i32).wrapping_mul(w));
            }
        }
        let dst_row = &mut out[yy * row_elems..(yy + 1) * row_elems];
        for (d, &a) in dst_row.iter_mut().zip(&acc) {
            *d = clip8(a >> PRECISION_BITS);
        }
    }
    out
}

/// Resize `src` to `target_width x target_height`. Horizontal pass first,
/// then vertical; a pass whose size does not change is skipped.
pub fn resize_bicubic(src: &RgbImage, target_width: usize, target_height: usize) -> RgbImage {
    let need_h = target_width != src.width;
    let need_v = target_height != src.height;
    let mut data = src.data.clone();
    let mut width = src.width;
    if need_h {
        let k = precompute_weights(src.width, target_width);
        data = resample_horizontal(&data, src.width, src.height, target_width, &k);
        width = target_width;
    }
    if need_v {
        let k = precompute_weights(src.height, target_height);
        data = resample_vertical(&data, width, target_height, &k);
    }
    RgbImage {
        width: target_width,
        height: target_height,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_size_copies() {
        let img = RgbImage::filled(3, 2, [1, 2, 3]);
        let out = resize_bicubic(&img, 3, 2);
        assert_eq!(out, img);
    }

    #[test]
    fn constant_image_stays_constant() {
        let img = RgbImage::filled(7, 5, [200, 100, 50]);
        let out = resize_bicubic(&img, 13, 4);
        assert_eq!(out.width, 13);
        assert_eq!(out.height, 4);
        for p in out.data.chunks(3) {
            assert_eq!(p, &[200, 100, 50]);
        }
    }

    /// Two source pixels to four. Output centers sit at 0.25, 0.75, 1.25
    /// and 1.75 in source space; support 2 covers both sources for every
    /// output sample. Hand-computed from the kernel (`a = -0.5`):
    /// sample 0 has weights f(0.25) = 0.8672 and f(1.25) = -0.0703, which
    /// normalize to 1.0882 and -0.0882, so 255 * -0.0882 clamps to 0;
    /// sample 1 has f(0.75) = 0.2266 and f(0.25) = 0.8672 -> 0.2071 and
    /// 0.7929, so 255 * 0.7929 = 202.2 -> 202 after the 22-bit rounding
    /// bias. Samples 2 and 3 mirror them.
    #[test]
    fn two_to_four_hand_values() {
        let img = RgbImage::new(2, 1, vec![0, 0, 0, 255, 255, 255]).unwrap();
        let out = resize_bicubic(&img, 4, 1);
        assert_eq!(
            out.data,
            vec![0, 0, 0, 53, 53, 53, 202, 202, 202, 255, 255, 255]
        );
    }

    /// Fixed-point weights for the 2 -> 4 case: 1.0882 * 2^22 = 4564390 and
    /// -0.0882 * 2^22 = -370086, truncated after a half-unit bias.
    #[test]
    fn two_to_four_fixed_point_weights() {
        let k = precompute_weights(2, 4);
        assert_eq!(k.ksize, 5);
        assert_eq!(k.bounds, vec![(0, 2); 4]);
        assert_eq!(&k.weights[0..2], &[4564390, -370086]);
        assert_eq!(&k.weights[5..7], &[3325484, 868820]);
    }

    /// Three sources to five keeps the center sample and the bright peak
    /// overshoots past its neighbors before clamping.
    #[test]
    fn three_to_five_symmetric() {
        let img = RgbImage::new(3, 1, vec![0, 0, 0, 255, 255, 255, 0, 0, 0]).unwrap();
        let out = resize_bicubic(&img, 5, 1);
        assert_eq!(
            out.data,
            vec![
                0, 0, 0, 101, 101, 101, 255, 255, 255, 101, 101, 101, 0, 0, 0
            ]
        );
    }

    /// Downscale 6 -> 2 widens the support to 6 source samples per output.
    #[test]
    fn six_to_two_downscale() {
        let img = RgbImage::new(
            6,
            1,
            vec![
                10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
            ],
        )
        .unwrap();
        let out = resize_bicubic(&img, 2, 1);
        assert_eq!(out.data, vec![44, 54, 64, 126, 136, 146]);
    }

    #[test]
    fn composite_clips_at_edges() {
        let mut dst = RgbImage::filled(4, 4, [0, 0, 0]);
        let src = RgbImage::filled(3, 3, [9, 9, 9]);
        dst.composite(&src, 2, 2);
        assert_eq!(&dst.data[(2 * 4 + 2) * 3..(2 * 4 + 2) * 3 + 3], &[9, 9, 9]);
        assert_eq!(&dst.data[(3 * 4 + 3) * 3..(3 * 4 + 3) * 3 + 3], &[9, 9, 9]);
        assert_eq!(&dst.data[0..3], &[0, 0, 0]);
    }
}
