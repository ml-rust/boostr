//! [`latent_attention`]: causal attention of many query heads over one
//! shared cache head, the attention step of [`AbsorbedMla`](super::AbsorbedMla).

use crate::error::{Error, Result};
use crate::model::attention_mask::causal_window_mask;
use crate::nn::var_ops::var_contiguous;
use numr::autograd::{
    Var, var_add, var_matmul, var_mul_scalar, var_reshape, var_softmax, var_transpose,
};
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

/// `softmax(q · keysᵀ * scale + causal) · values`, every query head over the
/// one cache head.
///
/// - `q`: `[B, H, S, D]`. Query `i` sits at absolute position `S_k - S + i`.
/// - `keys`: `[B, 1, S_k, D]`
/// - `values`: `[B, 1, S_k, D_v]`
/// - Output: `[B, H, S, D_v]`
///
/// The heads stack as rows of one matmul against the shared head, so no
/// head is repeated.
///
/// # Errors
///
/// Shapes outside the layout above, or a backend error.
pub(super) fn latent_attention<R, C>(
    client: &C,
    q: &Var<R>,
    keys: &Var<R>,
    values: &Var<R>,
    scale: f64,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let (q_shape, k_shape, v_shape) = (q.shape().to_vec(), keys.shape(), values.shape());
    let (&[batch, heads, seq_q, width], &[kb, 1, seq_k, kw], &[vb, 1, vk, width_v]) =
        (q_shape.as_slice(), k_shape, v_shape)
    else {
        return Err(shape_error(&q_shape, k_shape, v_shape));
    };
    if kb != batch || vb != batch || kw != width || vk != seq_k || seq_q > seq_k {
        return Err(shape_error(&q_shape, k_shape, v_shape));
    }

    let rows =
        var_reshape(&var_contiguous(q)?, &[batch, 1, heads * seq_q, width]).map_err(Error::Numr)?;
    let keys_t = var_transpose(keys).map_err(Error::Numr)?;
    let scores = var_matmul(&rows, &keys_t, client).map_err(Error::Numr)?;
    let scores = var_mul_scalar(&scores, scale, client).map_err(Error::Numr)?;
    let scores = var_reshape(&scores, &[batch, heads, seq_q, seq_k]).map_err(Error::Numr)?;
    // A single query sees every key, so only a block of queries is masked.
    let scores = if seq_q > 1 {
        let mask = causal_window_mask(
            client,
            seq_q,
            seq_k,
            0,
            scores.tensor().dtype(),
            scores.tensor().device(),
        )?;
        var_add(&scores, &Var::new(mask, false), client).map_err(Error::Numr)?
    } else {
        scores
    };
    let weights = var_softmax(&scores, -1, client).map_err(Error::Numr)?;
    let weights = var_reshape(&weights, &[batch, 1, heads * seq_q, seq_k]).map_err(Error::Numr)?;
    let out = var_matmul(&weights, values, client).map_err(Error::Numr)?;
    var_reshape(&out, &[batch, heads, seq_q, width_v]).map_err(Error::Numr)
}

fn shape_error(q: &[usize], k: &[usize], v: &[usize]) -> Error {
    Error::InvalidArgument {
        arg: "q",
        reason: format!(
            "latent attention needs q [B, H, S, D], keys [B, 1, S_k, D] and values \
             [B, 1, S_k, D_v] with S <= S_k; got q {q:?}, keys {k:?}, values {v:?}"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::impl_generic::attention::mla::scaled_dot_product_attention_masked_impl;
    use crate::test_utils::{Lcg, cpu_setup, max_abs_diff};

    /// The shared head broadcast to every query head through the generic
    /// attention gives the same output.
    #[test]
    fn matches_attention_with_the_head_repeated() {
        let (client, dev) = cpu_setup();
        let (b, h, d, dv, seq_k) = (2, 3, 6, 4, 7);
        for seq_q in [1, 3, 7] {
            let mut rng = Lcg(40 + seq_q as u64);
            let q = Var::new(rng.tensor(&dev, &[b, h, seq_q, d], 1.0), false);
            let k = rng.tensor(&dev, &[b, 1, seq_k, d], 1.0);
            let v = rng.tensor(&dev, &[b, 1, seq_k, dv], 1.0);
            let got = latent_attention(
                &client,
                &q,
                &Var::new(k.clone(), false),
                &Var::new(v.clone(), false),
                0.3,
            )
            .unwrap();
            assert_eq!(got.shape(), &[b, h, seq_q, dv]);

            let repeat = |t: &numr::tensor::Tensor<_>, w| {
                Var::new(
                    t.broadcast_to(&[b, h, seq_k, w])
                        .unwrap()
                        .contiguous()
                        .unwrap(),
                    false,
                )
            };
            let mask = causal_window_mask(&client, seq_q, seq_k, 0, DType::F32, &dev).unwrap();
            let want = scaled_dot_product_attention_masked_impl(
                &client,
                &q,
                &repeat(&k, d),
                &repeat(&v, dv),
                0.3,
                Some(&mask),
            )
            .unwrap();
            let diff = max_abs_diff(
                &got.tensor().contiguous().unwrap().to_vec::<f32>(),
                &want.tensor().contiguous().unwrap().to_vec::<f32>(),
            );
            assert!(diff <= 1e-5, "seq_q {seq_q}: max diff {diff}");
        }
    }

    #[test]
    fn a_wrong_shape_is_refused() {
        let (client, dev) = cpu_setup();
        let q = Var::new(Lcg(1).tensor(&dev, &[1, 2, 3, 4], 1.0), false);
        let two_heads = Var::new(Lcg(2).tensor(&dev, &[1, 2, 3, 4], 1.0), false);
        let err = latent_attention(&client, &q, &two_heads, &two_heads, 1.0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("keys [B, 1, S_k, D]"), "{err}");
    }
}
