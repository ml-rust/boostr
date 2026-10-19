//! Autograd integration for variable-length (packed) attention.
//!
//! Wraps `VarLenAttentionOps` into a `Var`-level operation over a packed
//! `[total_tokens, H, D]` buffer split by [`PackedSegments`].

use crate::error::{Error, Result};
use crate::ops::traits::{PackedSegments, VarLenAttentionOps};
use numr::autograd::{GradFn, TensorId, Var, var_mul_scalar};
use numr::ops::ScalarOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::sync::Arc;

/// Head layout saved from the forward pass for the backward.
struct VarLenAttentionConfig {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    causal: bool,
}

/// Backward function for packed varlen attention.
///
/// Saved state: Q, K, V, output, LSE from the forward pass, plus the segments.
/// Computes dQ, dK, dV via `VarLenAttentionOps::varlen_attention_bwd`.
/// GQA: dK and dV come back with `num_kv_heads` heads, the shape of K and V.
struct VarLenAttentionBackward<R: Runtime> {
    input_ids: [TensorId; 3],      // q, k, v
    saved_tensors: Vec<Tensor<R>>, // [q, k, v, output, lse]
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 3],
    segments: PackedSegments<R>,
    config: VarLenAttentionConfig,
}

impl<R: Runtime> GradFn<R> for VarLenAttentionBackward<R>
where
    R::Client: VarLenAttentionOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> numr::error::Result<Vec<Option<Tensor<R>>>> {
        // One kernel returns dQ, dK and dV together, so no slot is skipped.
        let client = R::default_client(grad_output.device());
        // The kernels index every input through implied row-major strides.
        // The engine can hand down a strided view, so this node makes each
        // input contiguous. An already contiguous tensor is a refcount clone.
        let grad_output = grad_output.contiguous()?;
        let q = self.saved_tensors[0].contiguous()?;
        let k = self.saved_tensors[1].contiguous()?;
        let v = self.saved_tensors[2].contiguous()?;
        let output = self.saved_tensors[3].contiguous()?;
        let lse = self.saved_tensors[4].contiguous()?;
        let cfg = &self.config;
        let segs = &self.segments;

        let (dq, dk, dv) = client
            .varlen_attention_bwd(
                &grad_output,
                &q,
                &k,
                &v,
                &output,
                &lse,
                segs.cu_seqlens(),
                segs.cu_seqlens(),
                segs.host_cu_seqlens(),
                segs.host_cu_seqlens(),
                segs.num_segments(),
                cfg.num_heads,
                cfg.num_kv_heads,
                segs.max_seqlen(),
                segs.max_seqlen(),
                cfg.head_dim,
                cfg.causal,
            )
            .map_err(|e| {
                numr::error::Error::Internal(format!("varlen_attention_bwd failed: {e}"))
            })?;

        Ok(vec![Some(dq), Some(dk), Some(dv)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> numr::error::Result<Vec<Option<Var<R>>>> {
        // First-order only: wrap the Tensor results as detached Vars.
        let grads = self.backward_all(grad_output.tensor())?;
        Ok(grads
            .into_iter()
            .map(|g| g.map(|t| Var::new(t, false)))
            .collect())
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        &self.saved_tensors
    }

    fn name(&self) -> &'static str {
        "VarLenAttentionBackward"
    }
}

/// Check the packed layout and return `(num_heads, num_kv_heads, head_dim)`.
fn packed_heads<R: Runtime>(
    q: &Tensor<R>,
    k: &Tensor<R>,
    v: &Tensor<R>,
    segments: &PackedSegments<R>,
) -> Result<(usize, usize, usize)> {
    let invalid = |arg: &'static str, reason: String| Error::InvalidArgument { arg, reason };
    let (qs, ks, vs) = (q.shape(), k.shape(), v.shape());
    if qs.len() != 3 || ks.len() != 3 {
        return Err(invalid(
            "q",
            format!("var_varlen_attention: q and k must be [T, H, D], got {qs:?} and {ks:?}"),
        ));
    }
    if vs != ks {
        return Err(invalid(
            "v",
            format!("var_varlen_attention: v shape {vs:?} must equal k shape {ks:?}"),
        ));
    }
    let total = segments.total_tokens();
    if qs[0] != total || ks[0] != total {
        return Err(invalid(
            "segments",
            format!(
                "var_varlen_attention: segments cover {total} tokens, q holds {}, k holds {}; \
                 build the segments from the same pack",
                qs[0], ks[0]
            ),
        ));
    }
    let (num_heads, num_kv_heads, head_dim) = (qs[1], ks[1], qs[2]);
    if ks[2] != head_dim || head_dim == 0 {
        return Err(invalid(
            "k",
            format!(
                "var_varlen_attention: head_dim of q {head_dim} and k {} must match",
                ks[2]
            ),
        ));
    }
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        return Err(invalid(
            "k",
            format!(
                "var_varlen_attention: q heads {num_heads} must be a non-zero multiple of \
                 k heads {num_kv_heads}"
            ),
        ));
    }
    Ok((num_heads, num_kv_heads, head_dim))
}

/// Packed variable-length attention with autograd tracking.
///
/// Inputs are `q: [T, H, D]`, `k` and `v: [T, H_kv, D]`, with
/// `H % H_kv == 0` (GQA when `H_kv < H`). `segments` splits the `T` tokens.
/// No token attends across a segment boundary. `causal` masks keys after
/// the query inside each segment.
///
/// The kernels apply `1 / sqrt(D)`. A different `scale` multiplies Q by
/// `scale * sqrt(D)` first, on the gradient path.
///
/// Returns the `[T, H, D]` output. The log-sum-exp stays inside the backward node.
pub fn var_varlen_attention<R, C>(
    client: &C,
    q: &Var<R>,
    k: &Var<R>,
    v: &Var<R>,
    segments: &PackedSegments<R>,
    scale: f64,
    causal: bool,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + ScalarOps<R> + VarLenAttentionOps<R>,
    R::Client: VarLenAttentionOps<R> + ScalarOps<R>,
{
    let (num_heads, num_kv_heads, head_dim) =
        packed_heads(q.tensor(), k.tensor(), v.tensor(), segments)?;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::InvalidArgument {
            arg: "scale",
            reason: format!("var_varlen_attention: scale must be finite and positive, got {scale}"),
        });
    }

    let q_ratio = scale * (head_dim as f64).sqrt();
    let q_scaled = if (q_ratio - 1.0).abs() <= 1e-12 {
        q.alias()
    } else {
        var_mul_scalar(q, q_ratio, client)?
    };

    let q_t = q_scaled.tensor().contiguous()?;
    let k_t = k.tensor().contiguous()?;
    let v_t = v.tensor().contiguous()?;
    let (output, lse) = client.varlen_attention_fwd(
        &q_t,
        &k_t,
        &v_t,
        segments.cu_seqlens(),
        segments.cu_seqlens(),
        segments.num_segments(),
        num_heads,
        num_kv_heads,
        segments.max_seqlen(),
        segments.max_seqlen(),
        head_dim,
        causal,
    )?;

    let needs_grad = q_scaled.requires_grad() || k.requires_grad() || v.requires_grad();
    if !needs_grad {
        return Ok(Var::new(output, false));
    }
    let grad_fn = VarLenAttentionBackward {
        input_ids: [q_scaled.id(), k.id(), v.id()],
        saved_tensors: vec![q_t, k_t, v_t, output.clone(), lse],
        input_grad_fns: [
            q_scaled.grad_fn().cloned(),
            k.grad_fn().cloned(),
            v.grad_fn().cloned(),
        ],
        segments: segments.clone(),
        config: VarLenAttentionConfig {
            num_heads,
            num_kv_heads,
            head_dim,
            causal,
        },
    };
    Ok(Var::from_op(output, Arc::new(grad_fn)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn var(data_len: usize, shape: &[usize], grad: bool) -> Var<CpuRuntime> {
        let (_, dev) = cpu_setup();
        let data: Vec<f32> = (0..data_len).map(|i| (i as f32 * 0.3).sin()).collect();
        Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, shape, &dev).unwrap(),
            grad,
        )
    }

    #[test]
    fn rejects_segments_from_another_pack() {
        let (client, dev) = cpu_setup();
        let q = var(5 * 2 * 4, &[5, 2, 4], true);
        let segs = PackedSegments::<CpuRuntime>::from_lengths(&[2, 2], &dev).unwrap();
        let e = var_varlen_attention(&client, &q, &q, &q, &segs, 0.5, true)
            .err()
            .unwrap();
        assert!(e.to_string().contains("segments cover 4 tokens"), "{e}");
    }

    #[test]
    fn rejects_non_multiple_kv_heads() {
        let (client, dev) = cpu_setup();
        let q = var(4 * 3 * 4, &[4, 3, 4], true);
        let k = var(4 * 2 * 4, &[4, 2, 4], true);
        let segs = PackedSegments::<CpuRuntime>::from_lengths(&[4], &dev).unwrap();
        let e = var_varlen_attention(&client, &q, &k, &k, &segs, 0.5, false)
            .err()
            .unwrap();
        assert!(e.to_string().contains("non-zero multiple"), "{e}");
    }

    #[test]
    fn no_grad_inputs_give_detached_output() {
        let (client, dev) = cpu_setup();
        let q = var(4 * 2 * 4, &[4, 2, 4], false);
        let segs = PackedSegments::<CpuRuntime>::from_lengths(&[1, 3], &dev).unwrap();
        let out = var_varlen_attention(&client, &q, &q, &q, &segs, 0.5, true).unwrap();
        assert!(!out.requires_grad());
        assert_eq!(out.tensor().shape(), &[4, 2, 4]);
    }

    #[test]
    fn custom_scale_matches_prescaled_q() {
        let (client, dev) = cpu_setup();
        let q = var(16, &[4, 1, 4], false);
        let k = var(16, &[4, 1, 4], false);
        let segs = PackedSegments::<CpuRuntime>::from_lengths(&[4], &dev).unwrap();
        // scale 1.0 with D = 4 equals the native scale 0.5 on 2 * q.
        let a = var_varlen_attention(&client, &q, &k, &k, &segs, 1.0, true).unwrap();
        let q2 = Var::new(client.mul_scalar(q.tensor(), 2.0).unwrap(), false);
        let b = var_varlen_attention(&client, &q2, &k, &k, &segs, 0.5, true).unwrap();
        assert_eq!(a.tensor().to_vec::<f32>(), b.tensor().to_vec::<f32>());
    }
}
