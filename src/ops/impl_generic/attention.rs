//! Generic multi-head attention implementation
//!
//! THE algorithm — same for all backends.
//! Composes numr autograd primitives: matmul, softmax, add, mul_scalar, transpose.

use crate::error::{Error, Result};
use numr::autograd::{Var, var_add, var_matmul, var_mul_scalar, var_softmax, var_transpose};
use numr::ops::ScalarOps;
use numr::runtime::{Runtime, RuntimeClient};

/// Multi-head attention: softmax((Q @ K^T) / sqrt(d)) @ V
///
/// All inputs/outputs are `Var<R>` for autograd support.
pub fn multi_head_attention_impl<R, C>(
    client: &C,
    q: &Var<R>,
    k: &Var<R>,
    v: &Var<R>,
    mask: Option<&Var<R>>,
    num_heads: usize,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + ScalarOps<R>,
    R::Client: ScalarOps<R>,
{
    // Validate shapes
    let q_shape = q.tensor().shape().to_vec();
    let k_shape = k.tensor().shape().to_vec();
    let v_shape = v.tensor().shape().to_vec();

    if q_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("expected 4D [B, H, S, D], got {}D", q_shape.len()),
        });
    }
    if k_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "k",
            reason: format!("expected 4D [B, H, S_kv, D], got {}D", k_shape.len()),
        });
    }
    if v_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "v",
            reason: format!("expected 4D [B, H, S_kv, D], got {}D", v_shape.len()),
        });
    }
    if q_shape[1] != num_heads {
        return Err(Error::InvalidArgument {
            arg: "num_heads",
            reason: format!("num_heads={} but q has H={}", num_heads, q_shape[1]),
        });
    }
    // B, H, D must match across q, k, v
    if q_shape[0] != k_shape[0] || q_shape[1] != k_shape[1] || q_shape[3] != k_shape[3] {
        return Err(Error::InvalidArgument {
            arg: "k",
            reason: format!(
                "q is {:?} but k is {:?} (B, H, D must match)",
                q_shape, k_shape
            ),
        });
    }
    if k_shape[0] != v_shape[0]
        || k_shape[1] != v_shape[1]
        || k_shape[2] != v_shape[2]
        || k_shape[3] != v_shape[3]
    {
        return Err(Error::InvalidArgument {
            arg: "v",
            reason: format!(
                "k is {:?} but v is {:?} (must match exactly)",
                k_shape, v_shape
            ),
        });
    }

    let head_dim = q_shape[3];
    let scale = (head_dim as f64).sqrt().recip();

    // Q @ K^T → [B, H, S, S_kv]
    let k_t = var_transpose(k).map_err(Error::Numr)?;
    let scores = var_matmul(q, &k_t, client).map_err(Error::Numr)?;

    // Scale by 1/sqrt(d)
    let scores = var_mul_scalar(&scores, scale, client).map_err(Error::Numr)?;

    // Add mask if provided
    let scores = match mask {
        Some(m) => var_add(&scores, m, client).map_err(Error::Numr)?,
        None => scores,
    };

    // Softmax over last dim
    let weights = var_softmax(&scores, -1, client).map_err(Error::Numr)?;

    // Weights @ V → [B, H, S, D]
    var_matmul(&weights, v, client).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn setup() -> (numr::runtime::cpu::CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (client, device)
    }

    #[test]
    fn test_attention_output_shape() {
        let (client, device) = setup();
        let b = 2;
        let h = 4;
        let s = 8;
        let d = 16;

        let q = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![0.1f32; b * h * s * d], &[b, h, s, d], &device),
            false,
        );
        let k = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![0.1f32; b * h * s * d], &[b, h, s, d], &device),
            false,
        );
        let v = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![0.1f32; b * h * s * d], &[b, h, s, d], &device),
            false,
        );

        let out = multi_head_attention_impl(&client, &q, &k, &v, None, h).unwrap();
        assert_eq!(out.tensor().shape(), &[b, h, s, d]);
    }

    #[test]
    fn test_attention_with_mask() {
        let (client, device) = setup();
        let b = 1;
        let h = 1;
        let s = 4;
        let d = 8;

        let q = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![0.1f32; b * h * s * d], &[b, h, s, d], &device),
            false,
        );
        let k = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![0.1f32; b * h * s * d], &[b, h, s, d], &device),
            false,
        );
        let v = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![0.1f32; b * h * s * d], &[b, h, s, d], &device),
            false,
        );

        // Causal mask: -inf for future positions
        let mut mask_data = vec![0.0f32; s * s];
        for i in 0..s {
            for j in (i + 1)..s {
                mask_data[i * s + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Var::new(
            Tensor::<CpuRuntime>::from_slice(&mask_data, &[1, 1, s, s], &device),
            false,
        );

        let out = multi_head_attention_impl(&client, &q, &k, &v, Some(&mask), h).unwrap();
        assert_eq!(out.tensor().shape(), &[b, h, s, d]);
    }

    #[test]
    fn test_attention_kv_different_seqlen() {
        let (client, device) = setup();
        let b = 1;
        let h = 2;
        let s_q = 4;
        let s_kv = 8;
        let d = 16;

        let q = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.1f32; b * h * s_q * d],
                &[b, h, s_q, d],
                &device,
            ),
            false,
        );
        let k = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.1f32; b * h * s_kv * d],
                &[b, h, s_kv, d],
                &device,
            ),
            false,
        );
        let v = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.1f32; b * h * s_kv * d],
                &[b, h, s_kv, d],
                &device,
            ),
            false,
        );

        let out = multi_head_attention_impl(&client, &q, &k, &v, None, h).unwrap();
        assert_eq!(out.tensor().shape(), &[b, h, s_q, d]);
    }

    #[test]
    fn test_attention_invalid_rank() {
        let (client, device) = setup();
        let q = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 8], &[2, 4], &device),
            false,
        );
        let k = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 8], &[2, 4], &device),
            false,
        );
        let v = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 8], &[2, 4], &device),
            false,
        );

        let result = multi_head_attention_impl(&client, &q, &k, &v, None, 1);
        assert!(result.is_err());
    }
}
