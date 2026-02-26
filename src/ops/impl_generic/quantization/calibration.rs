//! Generic calibration implementations
//!
//! THE algorithm — same for all backends.
//! Composes numr primitives: abs, max, mul, mean, matmul, square, narrow.

use crate::error::{Error, Result};
use numr::dtype::DType;
use numr::ops::{BinaryOps, MatmulOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// AWQ channel importance: score[j] = mean_i(max_batch(|act[:,j]|) * |W[i,j]|)
///
/// activations: [N, K], weights: [M, K] → output: [K]
pub fn awq_channel_scores_impl<R, C>(
    client: &C,
    activations: &Tensor<R>,
    weights: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + UnaryOps<R> + ReduceOps<R> + BinaryOps<R> + ScalarOps<R>,
{
    let act_shape = activations.shape();
    let w_shape = weights.shape();

    if act_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "activations",
            reason: format!("expected 2D [N, K], got {:?}", act_shape),
        });
    }
    if w_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "weights",
            reason: format!("expected 2D [M, K], got {:?}", w_shape),
        });
    }
    if act_shape[1] != w_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "weights",
            reason: format!(
                "channel dim mismatch: activations K={}, weights K={}",
                act_shape[1], w_shape[1]
            ),
        });
    }

    // act_scale[j] = max_n(|act[n, j]|) → [K]
    let abs_act = client.abs(activations).map_err(Error::Numr)?;
    let act_scale = client.max(&abs_act, &[0], false).map_err(Error::Numr)?;

    // |W| → [M, K]
    let abs_w = client.abs(weights).map_err(Error::Numr)?;

    // scaled[i, j] = act_scale[j] * |W[i, j]| → [M, K] (broadcast act_scale [K] over M)
    let scaled = client.mul(&abs_w, &act_scale).map_err(Error::Numr)?;

    // score[j] = mean_i(scaled[i, j]) → [K]
    client.mean(&scaled, &[0], false).map_err(Error::Numr)
}

/// Diagonal Fisher Information: fisher[i] = mean_n(grad[n, i]^2)
///
/// gradients: [N, P] → output: [P]
pub fn fisher_information_impl<R, C>(
    client: &C,
    gradients: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + UnaryOps<R> + ReduceOps<R>,
{
    let shape = gradients.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "gradients",
            reason: format!("expected 2D [N, P], got {:?}", shape),
        });
    }

    let squared = client.square(gradients).map_err(Error::Numr)?;
    client.mean(&squared, &[0], false).map_err(Error::Numr)
}

/// GPTQ Hessian update: H_new = H + (2/batch) * X^T @ X
///
/// hessian: [K, K], x_block: [B, K] → output: [K, K]
pub fn gptq_hessian_update_impl<R, C>(
    client: &C,
    hessian: &Tensor<R>,
    x_block: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + MatmulOps<R> + ScalarOps<R> + BinaryOps<R>,
{
    let h_shape = hessian.shape();
    let x_shape = x_block.shape();

    if h_shape.len() != 2 || h_shape[0] != h_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "hessian",
            reason: format!("expected square 2D [K, K], got {:?}", h_shape),
        });
    }
    if x_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "x_block",
            reason: format!("expected 2D [B, K], got {:?}", x_shape),
        });
    }
    if x_shape[1] != h_shape[0] {
        return Err(Error::InvalidArgument {
            arg: "x_block",
            reason: format!("K mismatch: x_block K={}, hessian K={}", x_shape[1], h_shape[0]),
        });
    }

    let batch = x_shape[0];
    let x_t = x_block.transpose(-2, -1).map_err(Error::Numr)?;
    let xt_x = client.matmul(&x_t, x_block).map_err(Error::Numr)?;
    let scale = 2.0 / batch as f64;
    let scaled = client.mul_scalar(&xt_x, scale).map_err(Error::Numr)?;
    client.add(hessian, &scaled).map_err(Error::Numr)
}

/// GPTQ column-wise quantization with error compensation.
///
/// weight: [M, K], h_inv: [K, K] → (quantized [M, K], scales [M, K/G], zeros [M, K/G])
pub fn gptq_quantize_column_impl<R, C>(
    _client: &C,
    weight: &Tensor<R>,
    h_inv: &Tensor<R>,
    num_bits: u32,
    group_size: u32,
    symmetric: bool,
) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + UnaryOps<R>
        + BinaryOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + MatmulOps<R>
        + ShapeOps<R>,
{
    let w_shape = weight.shape();
    let h_shape = h_inv.shape();

    if w_shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "weight",
            reason: format!("expected 2D [M, K], got {:?}", w_shape),
        });
    }
    if h_shape.len() != 2 || h_shape[0] != h_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "h_inv",
            reason: format!("expected square 2D [K, K], got {:?}", h_shape),
        });
    }

    let m = w_shape[0];
    let k = w_shape[1];

    if h_shape[0] != k {
        return Err(Error::InvalidArgument {
            arg: "h_inv",
            reason: format!("K mismatch: weight K={}, h_inv K={}", k, h_shape[0]),
        });
    }
    if group_size == 0 || k % group_size as usize != 0 {
        return Err(Error::InvalidArgument {
            arg: "group_size",
            reason: format!("K={} must be divisible by group_size={}", k, group_size),
        });
    }

    let device = weight.device().clone();
    let num_groups = k / group_size as usize;
    let qmax = (1u64 << num_bits) - 1;
    let qmax_f = qmax as f64;

    // Work with a mutable copy of weights — we'll update it column by column
    // to propagate quantization error.
    // Since we can't mutate tensors in-place with numr ops, we accumulate
    // the error correction as we go.
    //
    // The GPTQ algorithm processes columns left to right:
    //   For each column j:
    //     1. Quantize W[:, j] using group scale/zero
    //     2. Compute error: err = (W[:, j] - Q[:, j]) / h_inv[j, j]
    //     3. Update remaining columns: W[:, j+1:] -= err * h_inv[j, j+1:]

    // We'll collect the quantized weight data, scales and zeros via CPU
    // since the column loop is inherently sequential and touches individual columns.
    // This is the standard GPTQ approach — the loop is O(K) kernel launches.

    let w_data: Vec<f32> = weight.contiguous().to_vec();
    let h_data: Vec<f32> = h_inv.contiguous().to_vec();

    let mut w_work = w_data.clone();
    let mut q_out = vec![0.0f32; m * k];
    let mut scales_out = vec![0.0f32; m * num_groups];
    let mut zeros_out = vec![0.0f32; m * num_groups];

    for col in 0..k {
        let group_idx = col / group_size as usize;

        // At the start of each group, compute scale and zero for this group
        if col % group_size as usize == 0 {
            for row in 0..m {
                let start = row * k + col;
                let end = start + group_size as usize;
                let group_vals = &w_work[start..end];

                let (scale, zero) = if symmetric {
                    let amax = group_vals
                        .iter()
                        .fold(0.0f32, |acc, &v| acc.max(v.abs()));
                    let s = if amax == 0.0 { 1.0 } else { amax / (qmax_f as f32 / 2.0) };
                    (s, 0.0f32)
                } else {
                    let vmin = group_vals.iter().cloned().fold(f32::INFINITY, f32::min);
                    let vmax = group_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let s = if (vmax - vmin).abs() < 1e-10 {
                        1.0
                    } else {
                        (vmax - vmin) / qmax_f as f32
                    };
                    let z = (-vmin / s).round().clamp(0.0, qmax_f as f32);
                    (s, z)
                };

                scales_out[row * num_groups + group_idx] = scale;
                zeros_out[row * num_groups + group_idx] = zero;
            }
        }

        let h_inv_jj = h_data[col * k + col];

        for row in 0..m {
            let idx = row * k + col;
            let w_val = w_work[idx];
            let scale = scales_out[row * num_groups + group_idx];
            let zero = zeros_out[row * num_groups + group_idx];

            // Quantize
            let q = if symmetric {
                let half = qmax_f as f32 / 2.0;
                ((w_val / scale).round().clamp(-half, half - 1.0) + half) * scale - half * scale
            } else {
                let q_int = (w_val / scale + zero).round().clamp(0.0, qmax_f as f32);
                (q_int - zero) * scale
            };

            q_out[idx] = q;

            // Error compensation: propagate error to remaining columns
            let err = (w_val - q) / if h_inv_jj.abs() < 1e-10 { 1.0 } else { h_inv_jj };
            for j2 in (col + 1)..k {
                let h_val = h_data[col * k + j2];
                w_work[row * k + j2] -= err * h_val;
            }
        }
    }

    let q_tensor = Tensor::<R>::from_slice(&q_out, &[m, k], &device);
    let s_tensor = Tensor::<R>::from_slice(&scales_out, &[m, num_groups], &device);
    let z_tensor = Tensor::<R>::from_slice(&zeros_out, &[m, num_groups], &device);

    Ok((q_tensor, s_tensor, z_tensor))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn test_awq_channel_scores_shape() {
        let (client, device) = cpu_setup();
        let act = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32; 4 * 8],
            &[4, 8],
            &device,
        );
        let w = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32; 6 * 8],
            &[6, 8],
            &device,
        );
        let result = awq_channel_scores_impl(&client, &act, &w).unwrap();
        assert_eq!(result.shape(), &[8]);
    }

    #[test]
    fn test_awq_channel_scores_values() {
        let (client, device) = cpu_setup();
        // act = [[1, -2], [3, -1]], weights = [[2, 1], [1, 3]]
        // act_scale = max_abs over rows = [3, 2]
        // |W| * act_scale = [[2*3, 1*2], [1*3, 3*2]] = [[6, 2], [3, 6]]
        // score = mean over rows = [4.5, 4.0]
        let act = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, -2.0, 3.0, -1.0],
            &[2, 2],
            &device,
        );
        let w = Tensor::<CpuRuntime>::from_slice(
            &[2.0f32, 1.0, 1.0, 3.0],
            &[2, 2],
            &device,
        );
        let result = awq_channel_scores_impl(&client, &act, &w).unwrap();
        let data = result.to_vec::<f32>();
        assert!((data[0] - 4.5).abs() < 1e-5);
        assert!((data[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_fisher_information_shape() {
        let (client, device) = cpu_setup();
        let grads = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32; 16 * 32],
            &[16, 32],
            &device,
        );
        let result = fisher_information_impl(&client, &grads).unwrap();
        assert_eq!(result.shape(), &[32]);
    }

    #[test]
    fn test_fisher_information_values() {
        let (client, device) = cpu_setup();
        // grads = [[1, 2], [3, 4]]
        // squared = [[1, 4], [9, 16]]
        // mean over rows = [5.0, 10.0]
        let grads = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0],
            &[2, 2],
            &device,
        );
        let result = fisher_information_impl(&client, &grads).unwrap();
        let data = result.to_vec::<f32>();
        assert!((data[0] - 5.0).abs() < 1e-5);
        assert!((data[1] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_gptq_hessian_update_shape() {
        let (client, device) = cpu_setup();
        let h = Tensor::<CpuRuntime>::zeros(&[8, 8], DType::F32, &device);
        let x = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32; 4 * 8],
            &[4, 8],
            &device,
        );
        let result = gptq_hessian_update_impl(&client, &h, &x).unwrap();
        assert_eq!(result.shape(), &[8, 8]);
    }

    #[test]
    fn test_gptq_hessian_update_symmetry() {
        let (client, device) = cpu_setup();
        let h = Tensor::<CpuRuntime>::zeros(&[4, 4], DType::F32, &device);
        let x_data: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let x = Tensor::<CpuRuntime>::from_slice(&x_data, &[2, 4], &device);
        let result = gptq_hessian_update_impl(&client, &h, &x).unwrap();
        let data = result.to_vec::<f32>();
        // H = (2/B) * X^T X should be symmetric
        for i in 0..4 {
            for j in 0..4 {
                let diff = (data[i * 4 + j] - data[j * 4 + i]).abs();
                assert!(diff < 1e-5, "not symmetric at [{},{}]: {} vs {}", i, j, data[i * 4 + j], data[j * 4 + i]);
            }
        }
    }

    #[test]
    fn test_gptq_quantize_column_basic() {
        let (client, device) = cpu_setup();
        let w_data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let w = Tensor::<CpuRuntime>::from_slice(&w_data, &[4, 8], &device);
        // Identity-like h_inv
        let mut h_inv_data = vec![0.0f32; 64];
        for i in 0..8 {
            h_inv_data[i * 8 + i] = 1.0;
        }
        let h_inv = Tensor::<CpuRuntime>::from_slice(&h_inv_data, &[8, 8], &device);

        let (q, scales, zeros) = gptq_quantize_column_impl(
            &client, &w, &h_inv, 4, 4, false,
        ).unwrap();

        assert_eq!(q.shape(), &[4, 8]);
        assert_eq!(scales.shape(), &[4, 2]);
        assert_eq!(zeros.shape(), &[4, 2]);

        // Verify shapes, scales positive, values finite
        let q_data = q.to_vec::<f32>();
        let s_data = scales.to_vec::<f32>();
        for &s in &s_data {
            assert!(s > 0.0, "scale should be positive, got {}", s);
        }
        for (i, &v) in q_data.iter().enumerate() {
            assert!(v.is_finite(), "non-finite quantized value at {}: {}", i, v);
        }
    }
}
