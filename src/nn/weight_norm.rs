//! Weight-norm reparameterization loader helper.
//!
//! StyleTTS2-family decoders (Kokoro etc.) store Conv1d / Linear weights as the
//! pair `(weight_v, weight_g)` rather than a fused `weight`. At load time we
//! reconstruct:
//!
//! ```text
//!     weight = weight_g * weight_v / ||weight_v||
//! ```
//!
//! where the norm is taken over every axis except the output-channel axis. This
//! is the inverse of `torch.nn.utils.weight_norm`. We do the reconstruction once
//! at load time so that downstream modules (`Conv1d`, `ConvTranspose1d`, `Linear`)
//! see a regular weight tensor — no runtime overhead, no special-casing.

use crate::error::{Error, Result};
use numr::ops::{BinaryOps, ReduceOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Fuse a weight-norm pair `(v, g)` into a single weight tensor.
///
/// * `v` — direction, shape `[C_out, ...]`.
/// * `g` — magnitude per output channel. Accepted shapes: `[C_out]`, `[C_out, 1]`,
///   `[C_out, 1, 1]`, etc. (any rank whose leading dim is `C_out` and whose
///   remaining dims are all 1). Everything gets reshaped to match `v`'s rank.
/// * `dim` — the output-channel axis. For Conv1d / Linear weights this is `0`
///   (PyTorch convention). Set to `1` for transposed-conv weight layouts
///   `[C_in, C_out, K]` where `C_out` lives on axis 1.
///
/// Returns the fused weight tensor shaped like `v`.
pub fn fuse_weight_norm<R, C>(
    client: &C,
    v: &Tensor<R>,
    g: &Tensor<R>,
    dim: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    let v_shape = v.shape();
    if dim >= v_shape.len() {
        return Err(Error::InvalidArgument {
            arg: "dim",
            reason: format!(
                "weight-norm axis {dim} out of range for weight of rank {}",
                v_shape.len()
            ),
        });
    }
    let c_out = v_shape[dim];

    let g_total: usize = g.shape().iter().product();
    if g_total != c_out {
        return Err(Error::InvalidArgument {
            arg: "g",
            reason: format!(
                "weight_g must have {c_out} elements (one per output channel), got shape {:?}",
                g.shape()
            ),
        });
    }

    // Broadcast-shape for g: 1 everywhere except `c_out` on `dim`.
    let mut broadcast_shape = vec![1usize; v_shape.len()];
    broadcast_shape[dim] = c_out;
    let g_broadcast = g.reshape(&broadcast_shape).map_err(Error::Numr)?;

    // ||v|| computed over every axis except `dim`, keepdim so broadcasting lines up.
    let reduce_dims: Vec<usize> = (0..v_shape.len()).filter(|&d| d != dim).collect();
    let v_sq = client.mul(v, v).map_err(Error::Numr)?;
    let norm_sq = client.sum(&v_sq, &reduce_dims, true).map_err(Error::Numr)?;
    let norm = client.sqrt(&norm_sq).map_err(Error::Numr)?;

    // weight = v * (g / ||v||)
    let scale = client.div(&g_broadcast, &norm).map_err(Error::Numr)?;
    client.mul(v, &scale).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn identity_when_v_is_unit_and_g_is_one() {
        let (client, device) = cpu_setup();
        // v: two output channels, each a unit vector of length 3 (norm = 1).
        // g = [1, 1] → fused weight == v.
        let v = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[2, 1, 3],
            &device,
        );
        let g = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2, 1, 1], &device);
        let w = fuse_weight_norm(&client, &v, &g, 0).unwrap();
        assert_eq!(w.shape(), &[2, 1, 3]);
        let flat: Vec<f32> = w.to_vec();
        let v_flat: Vec<f32> = v.to_vec();
        for (a, b) in flat.iter().zip(v_flat.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn scales_to_requested_per_channel_magnitude() {
        let (client, device) = cpu_setup();
        // v channel 0 has norm 2, channel 1 has norm 5. g = [4, 10].
        // Expected per-channel norm of fused weight: [4, 10].
        let v = Tensor::<CpuRuntime>::from_slice(
            &[2.0f32, 0.0, 0.0, 3.0, 4.0, 0.0],
            &[2, 1, 3],
            &device,
        );
        let g = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 10.0], &[2], &device);
        let w = fuse_weight_norm(&client, &v, &g, 0).unwrap();
        let flat: Vec<f32> = w.to_vec();
        let c0_norm = (flat[0].powi(2) + flat[1].powi(2) + flat[2].powi(2)).sqrt();
        let c1_norm = (flat[3].powi(2) + flat[4].powi(2) + flat[5].powi(2)).sqrt();
        assert!((c0_norm - 4.0).abs() < 1e-4, "c0 norm {c0_norm}");
        assert!((c1_norm - 10.0).abs() < 1e-4, "c1 norm {c1_norm}");
    }

    #[test]
    fn accepts_flat_g() {
        let (client, device) = cpu_setup();
        let v = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device);
        let g = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device);
        assert!(fuse_weight_norm(&client, &v, &g, 0).is_ok());
    }

    #[test]
    fn rejects_wrong_g_size() {
        let (client, device) = cpu_setup();
        let v = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[2, 1, 3], &device);
        let g = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 3], &[3], &device);
        assert!(fuse_weight_norm(&client, &v, &g, 0).is_err());
    }

    #[test]
    fn rejects_dim_out_of_range() {
        let (client, device) = cpu_setup();
        let v = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[2, 2], &device);
        let g = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device);
        assert!(fuse_weight_norm(&client, &v, &g, 5).is_err());
    }

    #[test]
    fn axis_1_works_for_transposed_conv_layout() {
        // Transposed-conv weight shape: [C_in, C_out, K]. Output channel axis = 1.
        let (client, device) = cpu_setup();
        let v = Tensor::<CpuRuntime>::from_slice(
            &[
                1.0f32, 0.0, 0.0, // c_in=0, c_out=0
                0.0, 2.0, 0.0, // c_in=0, c_out=1
                0.0, 0.0, 0.0, // c_in=1, c_out=0
                0.0, 0.0, 0.0, // c_in=1, c_out=1
            ],
            &[2, 2, 3],
            &device,
        );
        let g = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 6.0], &[2], &device);
        let w = fuse_weight_norm(&client, &v, &g, 1).unwrap();
        assert_eq!(w.shape(), &[2, 2, 3]);
        // ||v[:, 0, :]|| = 1, ||v[:, 1, :]|| = 2 → scales = 3, 3 respectively.
        let flat: Vec<f32> = w.to_vec();
        assert!((flat[0] - 3.0).abs() < 1e-4);
        assert!((flat[4] - 6.0).abs() < 1e-4);
    }
}
