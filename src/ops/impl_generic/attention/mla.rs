//! Generic scaled dot-product attention for MLA
//!
//! Allows different K and V last dimensions (needed for MLA's decoupled RoPE).
//! Q@K^T uses D_k, output has D_v from V.

use super::document_mask::document_mask_bias;
use crate::error::{Error, Result};
use numr::autograd::{Var, var_add, var_matmul, var_mul_scalar, var_softmax, var_transpose};
use numr::dtype::DType;
use numr::ops::{CompareOps, ConditionalOps, ScalarOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

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
    C: RuntimeClient<R> + ScalarOps<R> + TypeConversionOps<R>,
    R::Client: ScalarOps<R>,
{
    check_shapes(q, k, v)?;
    let mask = if causal {
        Some(top_left_causal_mask(client, q, k)?)
    } else {
        None
    };
    scaled_dot_product_attention_masked_impl(client, q, k, v, scale, mask.as_ref())
}

/// [`scaled_dot_product_attention_impl`] with an optional document mask.
///
/// `segment_ids` is `[B, S]` I32 with `S == S_kv`. Equal ids in one row mark one document.
/// No query attends to a key of another document. `causal` applies inside each document.
/// `None` runs [`scaled_dot_product_attention_impl`] unchanged.
///
/// RoPE needs no position reset per document. Rotated scores depend only on
/// the offset `i - j`.
///
/// The mask builds on the device from `segment_ids` in the dtype of the scores.
pub fn segmented_scaled_dot_product_attention_impl<R, C>(
    client: &C,
    q: &Var<R>,
    k: &Var<R>,
    v: &Var<R>,
    scale: f64,
    causal: bool,
    segment_ids: Option<&Tensor<R>>,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ScalarOps<R> + CompareOps<R> + ConditionalOps<R> + TypeConversionOps<R>,
    R::Client: ScalarOps<R>,
{
    let Some(segment_ids) = segment_ids else {
        return scaled_dot_product_attention_impl(client, q, k, v, scale, causal);
    };
    check_shapes(q, k, v)?;
    let (batch, seq_q, seq_kv) = (q.shape()[0], q.shape()[2], k.shape()[2]);
    if seq_q != seq_kv {
        return Err(Error::InvalidArgument {
            arg: "segment_ids",
            reason: format!(
                "segment_ids need equal query and key lengths, got q {seq_q} and k {seq_kv}; \
                 pass segment_ids None for a KV-cached block"
            ),
        });
    }
    let bias = document_mask_bias(
        client,
        segment_ids,
        batch,
        seq_q,
        causal,
        q.tensor().dtype(),
    )?;
    scaled_dot_product_attention_masked_impl(client, q, k, v, scale, Some(&bias))
}

/// softmax(Q @ K^T * scale + mask) @ V with a caller-built additive mask.
///
/// `mask` broadcasts against the `[B, H, S, S_kv]` scores and must share their dtype.
/// Softmax accumulates in F32 for F16 and BF16 scores, so the scores need no cast.
/// `None` adds no mask. The mask takes no gradient.
pub fn scaled_dot_product_attention_masked_impl<R, C>(
    client: &C,
    q: &Var<R>,
    k: &Var<R>,
    v: &Var<R>,
    scale: f64,
    mask: Option<&Tensor<R>>,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ScalarOps<R>,
    R::Client: ScalarOps<R>,
{
    check_shapes(q, k, v)?;

    // Q @ K^T → [B, H, S, S_kv]
    let k_t = var_transpose(k).map_err(Error::Numr)?;
    let scores = var_matmul(q, &k_t, client).map_err(Error::Numr)?;

    // Scale
    let scores = var_mul_scalar(&scores, scale, client).map_err(Error::Numr)?;

    let scores = match mask {
        Some(mask) => {
            if mask.dtype() != scores.tensor().dtype() {
                return Err(Error::InvalidArgument {
                    arg: "mask",
                    reason: format!(
                        "mask dtype {:?} must match the scores dtype {:?}",
                        mask.dtype(),
                        scores.tensor().dtype()
                    ),
                });
            }
            let mask = Var::new(mask.clone(), false);
            var_add(&scores, &mask, client).map_err(Error::Numr)?
        }
        None => scores,
    };

    // Softmax over last dim
    let weights = var_softmax(&scores, -1, client).map_err(Error::Numr)?;

    // Weights @ V → [B, H, S, D_v]
    var_matmul(&weights, v, client).map_err(Error::Numr)
}

/// `[1, 1, S, S_kv]` causal mask in the dtype of `q`: key `j > i` holds `-inf`.
///
/// Query `i` aligns with key `i`, the same rule as the CUDA SDPA kernel.
fn top_left_causal_mask<R, C>(client: &C, q: &Var<R>, k: &Var<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TypeConversionOps<R>,
{
    let s_q = q.shape()[2];
    let s_kv = k.shape()[2];
    let mut mask_data = vec![0.0f32; s_q * s_kv];
    for i in 0..s_q {
        for j in (i + 1)..s_kv {
            mask_data[i * s_kv + j] = f32::NEG_INFINITY;
        }
    }
    let mask = Tensor::<R>::from_slice(&mask_data, &[1, 1, s_q, s_kv], q.tensor().device())?;
    let dtype = q.tensor().dtype();
    if dtype == DType::F32 {
        Ok(mask)
    } else {
        Ok(client.cast(&mask, dtype)?)
    }
}

/// Check the `[B, H, S, D_k]`, `[B, H, S_kv, D_k]`, `[B, H, S_kv, D_v]` layout.
fn check_shapes<R: Runtime>(q: &Var<R>, k: &Var<R>, v: &Var<R>) -> Result<()> {
    let q_shape = q.tensor().shape();
    let k_shape = k.tensor().shape();
    let v_shape = v.tensor().shape();

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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn segment_ids_reject_unequal_lengths() {
        let (client, device) = cpu_setup();
        let var = |s: usize| {
            let t = Tensor::<CpuRuntime>::from_slice(&vec![0.1f32; s * 4], &[1, 1, s, 4], &device)
                .unwrap();
            Var::new(t, false)
        };
        let ids = Tensor::<CpuRuntime>::from_slice(&[0i32, 0], &[1, 2], &device).unwrap();
        let e = segmented_scaled_dot_product_attention_impl(
            &client,
            &var(2),
            &var(3),
            &var(3),
            0.5,
            true,
            Some(&ids),
        )
        .err()
        .unwrap();
        assert!(e.to_string().contains("equal query and key lengths"), "{e}");
    }

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
            )
            .unwrap(),
            false,
        );
        let k = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.1f32; b * h * s * d_k],
                &[b, h, s, d_k],
                &device,
            )
            .unwrap(),
            false,
        );
        let v = Var::new(
            Tensor::<CpuRuntime>::from_slice(
                &vec![0.1f32; b * h * s * d_v],
                &[b, h, s, d_v],
                &device,
            )
            .unwrap(),
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
            Tensor::<CpuRuntime>::from_slice(&vec![0.1f32; b * h * s * d], &[b, h, s, d], &device)
                .unwrap(),
            false,
        );
        let k = q.clone();
        let v = q.clone();

        let scale = 1.0 / (d as f64).sqrt();
        let out = scaled_dot_product_attention_impl(&client, &q, &k, &v, scale, false).unwrap();
        assert_eq!(out.tensor().shape(), &[b, h, s, d]);
    }
}
