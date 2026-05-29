//! 1D reflection padding.
//!
//! Matches PyTorch's `nn.ReflectionPad1d` — pads each end of the last axis
//! by reflecting the interior (excluding the boundary sample):
//!
//! ```text
//! input  = [a, b, c, d, e]         pad=2
//! output = [c, b, a, b, c, d, e, d, c]
//!           └──┘   └─ body ─┘ └──┘
//!          mirror            mirror
//! ```
//!
//! Kokoro's ISTFTNet generator applies this before the final conv_post /
//! addition on the last upsample stage. CPU-only for now (GPU would need a
//! gather primitive). Input `[B, C, T]` → output `[B, C, T + pad_left + pad_right]`.

use crate::error::{Error, Result};
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Apply 1D reflection padding on the last dim.
///
/// `pad_left` and `pad_right` must each be strictly less than the input
/// length on the last dim — PyTorch's constraint. Reflecting by `>= T` would
/// require reflecting the reflection, which is rarely what the caller wants
/// and makes `[B, C, T=1]` inputs ill-defined.
pub fn reflection_pad_1d(
    x: &Tensor<CpuRuntime>,
    pad_left: usize,
    pad_right: usize,
) -> Result<Tensor<CpuRuntime>> {
    let shape = x.shape();
    if shape.len() != 3 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!("expected [B, C, T], got {shape:?}"),
        });
    }
    let (b, c, t) = (shape[0], shape[1], shape[2]);
    if pad_left >= t || pad_right >= t {
        return Err(Error::InvalidArgument {
            arg: "pad",
            reason: format!(
                "pad sizes ({pad_left}, {pad_right}) must each be < last-axis length ({t})"
            ),
        });
    }
    if pad_left == 0 && pad_right == 0 {
        // No-op; return a contiguous clone so downstream ops that assume
        // contiguous memory aren't surprised by input strides.
        return x.contiguous().map_err(Error::Numr);
    }

    let flat: Vec<f32> = x.contiguous().map_err(Error::Numr)?.to_vec();
    let t_out = t + pad_left + pad_right;
    let mut out = vec![0.0f32; b * c * t_out];

    for bi in 0..b {
        for ci in 0..c {
            let src_base = (bi * c + ci) * t;
            let dst_base = (bi * c + ci) * t_out;

            // Left reflection: dst[k] = src[pad_left - k] for k in 0..pad_left.
            for k in 0..pad_left {
                out[dst_base + k] = flat[src_base + pad_left - k];
            }
            // Body: copy src directly.
            for n in 0..t {
                out[dst_base + pad_left + n] = flat[src_base + n];
            }
            // Right reflection: dst[pad_left + t + k] = src[t - 2 - k].
            for k in 0..pad_right {
                out[dst_base + pad_left + t + k] = flat[src_base + t - 2 - k];
            }
        }
    }

    Ok(Tensor::<CpuRuntime>::from_slice(
        &out,
        &[b, c, t_out],
        x.device(),
    ))
}

#[cfg(test)]
#[allow(clippy::useless_vec)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;

    #[test]
    fn reflects_both_sides() {
        let (_client, device) = cpu_setup();
        let x =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &device);
        let y = reflection_pad_1d(&x, 2, 2).unwrap();
        let v: Vec<f32> = y.to_vec();
        // Input [1,2,3,4,5], pad=2 each side → [3,2,1,2,3,4,5,4,3].
        assert_eq!(v, vec![3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0]);
    }

    #[test]
    fn zero_padding_is_identity() {
        let (_client, device) = cpu_setup();
        let data = vec![1.0f32, 2.0, 3.0];
        let x = Tensor::<CpuRuntime>::from_slice(&data, &[1, 1, 3], &device);
        let y = reflection_pad_1d(&x, 0, 0).unwrap();
        let v: Vec<f32> = y.to_vec();
        assert_eq!(v, data);
    }

    #[test]
    fn handles_multichannel_batched_input() {
        let (_client, device) = cpu_setup();
        // B=2, C=2, T=3. Each (b, c) row should pad independently.
        let x = Tensor::<CpuRuntime>::from_slice(
            &[
                1.0f32, 2.0, 3.0, // (0, 0)
                4.0, 5.0, 6.0, // (0, 1)
                7.0, 8.0, 9.0, // (1, 0)
                10.0, 11.0, 12.0, // (1, 1)
            ],
            &[2, 2, 3],
            &device,
        );
        let y = reflection_pad_1d(&x, 1, 1).unwrap();
        assert_eq!(y.shape(), &[2, 2, 5]);
        let v: Vec<f32> = y.to_vec();
        // Row (0, 0) = [1, 2, 3], pad=1 → [2, 1, 2, 3, 2].
        assert_eq!(&v[0..5], &[2.0, 1.0, 2.0, 3.0, 2.0]);
        // Row (1, 1) = [10, 11, 12], pad=1 → [11, 10, 11, 12, 11].
        assert_eq!(&v[15..20], &[11.0, 10.0, 11.0, 12.0, 11.0]);
    }

    #[test]
    fn asymmetric_padding() {
        let (_client, device) = cpu_setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 4], &device);
        let y = reflection_pad_1d(&x, 1, 3).unwrap();
        let v: Vec<f32> = y.to_vec();
        // Input [1,2,3,4], left=1 → [2], right=3 → [3, 2, 1].
        // Expected: [2, 1, 2, 3, 4, 3, 2, 1].
        assert_eq!(v, vec![2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn rejects_too_large_padding() {
        let (_client, device) = cpu_setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 1, 2], &device);
        assert!(reflection_pad_1d(&x, 2, 0).is_err());
        assert!(reflection_pad_1d(&x, 0, 2).is_err());
    }

    #[test]
    fn rejects_wrong_rank() {
        let (_client, device) = cpu_setup();
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[2, 2], &device);
        assert!(reflection_pad_1d(&x, 1, 1).is_err());
    }
}
