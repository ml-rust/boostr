//! Patch embedding, learned position table and merge-block token order.

use crate::error::{Error, Result};
use crate::nn::{Conv2d, VarBuilder};
use crate::quant::traits::DequantOps;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConvOps, MatmulOps, PaddingMode};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Two stride-`patch` convolutions over the same image, summed, plus the
/// learned position table resized to the patch grid.
pub(super) struct PatchEmbed<R: Runtime> {
    conv0: Conv2d<R>,
    conv1: Conv2d<R>,
    bias: Tensor<R>,
    /// `[grid * grid, hidden]`, raster order over the training grid.
    pos: Tensor<R>,
    grid: usize,
    hidden: usize,
    merge: usize,
}

impl<R: Runtime<DType = DType>> PatchEmbed<R> {
    /// Take `v.patch_embd.*` and `v.position_embd.weight` from `vb`.
    pub(super) fn from_varbuilder(
        vb: &mut VarBuilder<R>,
        patch: usize,
        hidden: usize,
        grid: usize,
        merge: usize,
    ) -> Result<Self>
    where
        R::Client: DequantOps<R>,
    {
        let w0 = vb.take_tensor_dequant("v.patch_embd.weight", DType::F32)?;
        let w1 = vb.take_tensor_dequant("v.patch_embd.weight.1", DType::F32)?;
        let bias = vb.take_tensor_dequant("v.patch_embd.bias", DType::F32)?;
        let pos = vb.take_tensor_dequant("v.position_embd.weight", DType::F32)?;
        let want_w = [hidden, 3, patch, patch];
        for (name, w) in [("v.patch_embd.weight", &w0), ("v.patch_embd.weight.1", &w1)] {
            if w.shape() != want_w.as_slice() {
                return Err(Error::ModelError {
                    reason: format!(
                        "qwen3vl vision: {name} has shape {:?}, want {want_w:?}",
                        w.shape()
                    ),
                });
            }
        }
        if bias.shape() != [hidden].as_slice() {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen3vl vision: v.patch_embd.bias has shape {:?}, want [{hidden}]",
                    bias.shape()
                ),
            });
        }
        if pos.shape() != [grid * grid, hidden].as_slice() {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen3vl vision: v.position_embd.weight has shape {:?}, want [{}, {hidden}]",
                    pos.shape(),
                    grid * grid
                ),
            });
        }
        let conv = |w: Tensor<R>| {
            Conv2d::new(
                w,
                None,
                (patch, patch),
                PaddingMode::Valid,
                (1, 1),
                1,
                false,
            )
        };
        Ok(Self {
            conv0: conv(w0),
            conv1: conv(w1),
            bias,
            pos,
            grid,
            hidden,
            merge,
        })
    }

    /// `pixels`: `[1, 3, H, W]`. Returns `[ph * pw, hidden]` in merge-block
    /// order with patch bias and position embedding added.
    pub(super) fn forward<C>(
        &self,
        client: &C,
        pixels: &Tensor<R>,
        ph: usize,
        pw: usize,
    ) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + ConvOps<R> + BinaryOps<R> + MatmulOps<R>,
    {
        let p0 = self.conv0.forward_inference(client, pixels)?;
        let p1 = self.conv1.forward_inference(client, pixels)?;
        let patches = client.add(&p0, &p1)?;
        let got = patches.shape();
        if got != [1, self.hidden, ph, pw].as_slice() {
            return Err(Error::ModelError {
                reason: format!(
                    "qwen3vl vision: patch conv produced {got:?}, want [1, {}, {ph}, {pw}]",
                    self.hidden
                ),
            });
        }
        let tokens = patches
            .reshape(&[self.hidden, ph * pw])?
            .transpose(0, 1)?
            .contiguous()?;
        let tokens = to_block_order(&tokens, ph, pw, self.merge)?;
        let tokens = client.add(&tokens, &self.bias)?;

        let pos = self.resized_positions(client, ph, pw)?;
        let pos = to_block_order(&pos, ph, pw, self.merge)?;
        Ok(client.add(&tokens, &pos)?)
    }

    /// Position table for a `ph x pw` grid, raster order `[ph * pw, hidden]`.
    ///
    /// The training grid is used as is when the sizes match. Otherwise the
    /// table is bilinearly resampled with aligned corners: two small host
    /// weight matrices, one per axis, multiply the table on the device.
    fn resized_positions<C>(&self, client: &C, ph: usize, pw: usize) -> Result<Tensor<R>>
    where
        C: RuntimeClient<R> + MatmulOps<R>,
    {
        let g = self.grid;
        if ph == g && pw == g {
            return Ok(self.pos.clone());
        }
        let device = self.pos.device();
        let wy = Tensor::<R>::from_slice(&align_corners_weights(g, ph), &[ph, g], device)?;
        let wx = Tensor::<R>::from_slice(&align_corners_weights(g, pw), &[pw, g], device)?;
        let c = self.hidden;
        // rows: [G, G*C] -> [ph, G*C] -> [ph, G, C] -> [G, ph, C] -> [G, ph*C]
        let by_row = client.matmul(&wy, &self.pos.reshape(&[g, g * c])?)?;
        let by_row = by_row
            .reshape(&[ph, g, c])?
            .permute(&[1, 0, 2])?
            .contiguous()?
            .reshape(&[g, ph * c])?;
        // cols: [pw, ph*C] -> [pw, ph, C] -> [ph, pw, C] -> [ph*pw, C]
        let by_col = client.matmul(&wx, &by_row)?;
        Ok(by_col
            .reshape(&[pw, ph, c])?
            .permute(&[1, 0, 2])?
            .contiguous()?
            .reshape(&[ph * pw, c])?)
    }
}

/// Bilinear resampling weights `[dst, src]` with aligned corners, the
/// `GGML_SCALE_FLAG_ALIGN_CORNERS` mapping: `s = d * (src - 1) / (dst - 1)`,
/// split between `floor(s)` and `floor(s) + 1`, both clamped to the grid.
pub(super) fn align_corners_weights(src: usize, dst: usize) -> Vec<f32> {
    let mut w = vec![0f32; dst * src];
    let sf = if dst > 1 && src > 1 {
        (dst - 1) as f32 / (src - 1) as f32
    } else {
        dst as f32 / src as f32
    };
    for d in 0..dst {
        let s = d as f32 / sf;
        let s0f = s.floor();
        let s0 = (s0f as i64).clamp(0, src as i64 - 1) as usize;
        let s1 = (s0f as i64 + 1).clamp(0, src as i64 - 1) as usize;
        let frac = (s - s0f).clamp(0.0, 1.0);
        w[d * src + s0] += 1.0 - frac;
        w[d * src + s1] += frac;
    }
    w
}

/// Reorder raster `[ph * pw, C]` rows into merge-block order
/// `(by, bx, dy, dx)`, so the `merge * merge` patches of one output token
/// are consecutive rows.
pub(super) fn to_block_order<R: Runtime>(
    t: &Tensor<R>,
    ph: usize,
    pw: usize,
    merge: usize,
) -> Result<Tensor<R>> {
    let c = t.shape()[1];
    Ok(t.reshape(&[ph / merge, merge, pw / merge, merge, c])?
        .permute(&[0, 2, 1, 3, 4])?
        .contiguous()?
        .reshape(&[ph * pw, c])?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn align_corners_identity_when_sizes_match() {
        let w = align_corners_weights(4, 4);
        for d in 0..4 {
            for s in 0..4 {
                assert_eq!(w[d * 4 + s], if d == s { 1.0 } else { 0.0 });
            }
        }
    }

    #[test]
    fn align_corners_endpoints_map_to_endpoints() {
        // 3 -> 5: s = d * 2 / 4 = d / 2
        let w = align_corners_weights(3, 5);
        assert_eq!(&w[0..3], &[1.0, 0.0, 0.0]);
        assert_eq!(&w[3..6], &[0.5, 0.5, 0.0]);
        assert_eq!(&w[6..9], &[0.0, 1.0, 0.0]);
        assert_eq!(&w[12..15], &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn align_corners_single_destination_row() {
        let w = align_corners_weights(4, 1);
        assert_eq!(w, vec![1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn block_order_permutes_rows() {
        let (_, device) = cpu_setup();
        // 4x4 grid, C = 1, row value = raster index
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let t = Tensor::<CpuRuntime>::from_slice(&data, &[16, 1], &device).unwrap();
        let out: Vec<f32> = to_block_order(&t, 4, 4, 2).unwrap().to_vec();
        let expect: Vec<f32> = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
            .iter()
            .map(|&i| i as f32)
            .collect();
        assert_eq!(out, expect);
    }
}
