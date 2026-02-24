//! Generic scaled dot-product attention for MLA
//!
//! Allows different K and V last dimensions (needed for MLA's decoupled RoPE).
//! Q@K^T uses D_k, output has D_v from V.

use crate::error::{Error, Result};
use numr::autograd::{Var, var_add, var_matmul, var_mul_scalar, var_softmax, var_transpose};
use numr::dtype::DType;
use numr::ops::ScalarOps;
use numr::runtime::{Runtime, RuntimeClient};

/// Scaled dot-product attention: softmax(Q @ K^T * scale + mask) @ V
///
/// Unlike multi_head_attention_impl, this allows K and V to have different
/// last dimensions. K dim = head_dim + rope_head_dim, V dim = head_dim_v.
///
/// - `q`: `[B, H, S, D_k]`
/// - `k`: `[B, H, S_kv, D_k]`
/// - `v`: `[B, H, S_kv, D_v]`
/// - Output: `[B, H, S, D_v]`
pub fn scaled_dot_product_attention_impl<R, C>(
    client: &C,
    q: &Var<R>,
    k: &Var<R>,
    v: &Var<R>,
    scale: f64,
    causal: bool,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ScalarOps<R>,
    R::Client: ScalarOps<R>,
{
    let q_shape = q.tensor().shape().to_vec();
    let k_shape = k.tensor().shape().to_vec();
    let v_shape = v.tensor().shape().to_vec();

    if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
        return Err(Error::InvalidArgument {
            arg: "q/k/v",
            reason: "expected 4D tensors [B, H, S, D]".into(),
        });
    }

    // B, H must match
    if q_shape[0] != k_shape[0] || q_shape[1] != k_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "k",
            reason: format!("B,H mismatch: q={:?}, k={:?}", q_shape, k_shape),
        });
    }
    // K and V must share B, H, S_kv
    if k_shape[0] != v_shape[0] || k_shape[1] != v_shape[1] || k_shape[2] != v_shape[2] {
        return Err(Error::InvalidArgument {
            arg: "v",
            reason: format!("B,H,S_kv mismatch: k={:?}, v={:?}", k_shape, v_shape),
        });
    }
    // Q and K must share D_k
    if q_shape[3] != k_shape[3] {
        return Err(Error::InvalidArgument {
            arg: "k",
            reason: format!("D_k mismatch: q D={}, k D={}", q_shape[3], k_shape[3]),
        });
    }

    // Q @ K^T → [B, H, S, S_kv]
    let k_t = var_transpose(k).map_err(Error::Numr)?;
    let scores = var_matmul(q, &k_t, client).map_err(Error::Numr)?;

    // Scale
    let scores = var_mul_scalar(&scores, scale, client).map_err(Error::Numr)?;

    // Causal mask
    let scores = if causal {
        let s_q = q_shape[2];
        let s_kv = k_shape[2];
        let mut mask_data = vec![0.0f32; s_q * s_kv];
        for i in 0..s_q {
            for j in (i + 1)..s_kv {
                mask_data[i * s_kv + j] = f32::NEG_INFINITY;
            }
        }
        let mask_tensor = numr::tensor::Tensor::<R>::from_slice(
            &mask_data,
            &[1, 1, s_q, s_kv],
            q.tensor().device(),
        );
        let mask = Var::new(mask_tensor, false);
        var_add(&scores, &mask, client).map_err(Error::Numr)?
    } else {
        scores
    };

    // Softmax over last dim
    let weights = var_softmax(&scores, -1, client).map_err(Error::Numr)?;

    // Weights @ V → [B, H, S, D_v]
    var_matmul(&weights, v, client).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    #[test]
    fn test_sdpa_different_kv_dims() {
        let (client, device) = cpu_setup();
        let b = 1;
        let h = 2;
        let s = 4;
        let d_k = 8; // Q and K dim
        let d_v = 6; // V dim (different!)

        let q = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.1f32; b * h * s * d_k],
                &[b, h, s, d_k],
                &device,
            ),
            false,
        );
        let k = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.1f32; b * h * s * d_k],
                &[b, h, s, d_k],
                &device,
            ),
            false,
        );
        let v = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.1f32; b * h * s * d_v],
                &[b, h, s, d_v],
                &device,
            ),
            false,
        );

        let scale = 1.0 / (d_k as f64).sqrt();
        let out = scaled_dot_product_attention_impl(&client, &q, &k, &v, scale, true).unwrap();
        assert_eq!(out.tensor().shape(), &[b, h, s, d_v]);
    }

    #[test]
    fn test_sdpa_same_dims() {
        let (client, device) = cpu_setup();
        let b = 1;
        let h = 1;
        let s = 3;
        let d = 4;

        let q = Var::new(
            Tensor::<CpuRuntime>::from_slice(&vec![0.1f32; b * h * s * d], &[b, h, s, d], &device),
            false,
        );
        let k = q.clone();
        let v = q.clone();

        let scale = 1.0 / (d as f64).sqrt();
        let out = scaled_dot_product_attention_impl(&client, &q, &k, &v, scale, false).unwrap();
        assert_eq!(out.tensor().shape(), &[b, h, s, d]);
    }
}
