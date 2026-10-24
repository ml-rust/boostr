//! [`Mla`] forward pass, with an optional document mask.

use super::model::Mla;
use crate::error::{Error, Result};
use crate::nn::var_ops::var_contiguous;
use crate::ops::RoPEOps;
use crate::ops::impl_generic::attention::document_mask::document_mask_bias;
use crate::ops::impl_generic::attention::mla::{
    scaled_dot_product_attention_impl, scaled_dot_product_attention_masked_impl,
};
use crate::ops::impl_generic::attention::rope::apply_rope_impl;
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_broadcast_to, var_cat, var_narrow, var_permute, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NormalizationOps, ReduceOps, ScalarOps, ShapeOps,
    TensorOps, TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> Mla<R> {
    /// Forward pass: [B, S, hidden] → [B, S, hidden]
    pub fn forward<C>(&self, client: &C, hidden: &Var<R>) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + NormalizationOps<R>
            + ShapeOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + QuantMatmulOps<R>
            + RoPEOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + DequantOps<R>,
    {
        self.forward_masked(client, hidden, None)
    }

    /// [`Mla::forward`] with an optional document mask over packed documents.
    ///
    /// `segment_ids` is `[B, S]` I32. Equal ids in one row mark one document.
    /// No token attends to a token of another document. Attention stays causal inside each document.
    /// `None` runs [`Mla::forward`] unchanged.
    ///
    /// RoPE needs no position reset per document. Rotated scores depend only on
    /// the offset `i - j`. The mask builds on the device in the dtype of `hidden`.
    pub fn forward_with_segments<C>(
        &self,
        client: &C,
        hidden: &Var<R>,
        segment_ids: Option<&Tensor<R>>,
    ) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + NormalizationOps<R>
            + ShapeOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + QuantMatmulOps<R>
            + RoPEOps<R>
            + CompareOps<R>
            + ConditionalOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + DequantOps<R>,
    {
        let Some(segment_ids) = segment_ids else {
            return self.forward(client, hidden);
        };
        let shape = hidden.shape();
        if shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "hidden",
                reason: format!("Mla: hidden must be [B, S, hidden], got {shape:?}"),
            });
        }
        // Scores take the dtype of the activations, so the bias takes it too.
        let bias = document_mask_bias(
            client,
            segment_ids,
            shape[0],
            shape[1],
            true,
            hidden.tensor().dtype(),
        )?;
        self.forward_masked(client, hidden, Some(&bias))
    }

    /// Shared forward body. `bias` replaces the causal mask. `None` keeps the causal mask.
    fn forward_masked<C>(
        &self,
        client: &C,
        hidden: &Var<R>,
        bias: Option<&Tensor<R>>,
    ) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + NormalizationOps<R>
            + ShapeOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + QuantMatmulOps<R>
            + RoPEOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + DequantOps<R>,
    {
        let shape = hidden.shape().to_vec();
        let batch = shape[0];
        let seq_len = shape[1];
        let qk_dim = self.head_dim + self.rope_head_dim;

        // === Q path ===
        let q = if let Some(q_down) = &self.q_down {
            let q_latent = q_down.forward(client, hidden)?;
            let q_latent = if let Some(norm) = &self.q_norm {
                norm.forward(client, &q_latent)?
            } else {
                q_latent
            };
            self.q_up.forward(client, &q_latent)?
        } else {
            self.q_up.forward(client, hidden)?
        };

        // [B, S, num_heads * qk_dim] → [B, S, H, qk_dim] → [B, H, S, qk_dim]
        let q = var_reshape(&q, &[batch, seq_len, self.num_heads, qk_dim]).map_err(Error::Numr)?;
        let q = var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let q = var_contiguous(&q)?;

        // Split Q into nope and pe
        let q_nope = var_narrow(&q, 3, 0, self.head_dim).map_err(Error::Numr)?;
        let q_nope = var_contiguous(&q_nope)?;
        let q_pe = var_narrow(&q, 3, self.head_dim, self.rope_head_dim).map_err(Error::Numr)?;
        let q_pe = var_contiguous(&q_pe)?;

        // === KV path ===
        // Compress: [B, S, hidden] → [B, S, kv_lora_rank + rope_head_dim]
        let kv_compressed = self.kv_compress.forward(client, hidden)?;

        // Split: c_kv [B, S, kv_lora_rank], k_pe_raw [B, S, rope_head_dim]
        let c_kv = var_narrow(&kv_compressed, 2, 0, self.kv_lora_rank).map_err(Error::Numr)?;
        let c_kv = var_contiguous(&c_kv)?;
        let k_pe_raw = var_narrow(&kv_compressed, 2, self.kv_lora_rank, self.rope_head_dim)
            .map_err(Error::Numr)?;
        let k_pe_raw = var_contiguous(&k_pe_raw)?;

        // Normalize c_kv
        let c_kv = if let Some(norm) = &self.kv_norm {
            norm.forward(client, &c_kv)?
        } else {
            c_kv
        };

        // Decompress: [B, S, kv_lora_rank] → [B, S, num_heads * (head_dim + head_dim_v)]
        let kv = self.kv_decompress.forward(client, &c_kv)?;
        let kv = var_reshape(
            &kv,
            &[
                batch,
                seq_len,
                self.num_heads,
                self.head_dim + self.head_dim_v,
            ],
        )
        .map_err(Error::Numr)?;
        // → [B, H, S, head_dim + head_dim_v]
        let kv = var_permute(&kv, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let kv = var_contiguous(&kv)?;

        // Split K_nope and V
        let k_nope = var_narrow(&kv, 3, 0, self.head_dim).map_err(Error::Numr)?;
        let k_nope = var_contiguous(&k_nope)?;
        let v = var_narrow(&kv, 3, self.head_dim, self.head_dim_v).map_err(Error::Numr)?;
        let v = var_contiguous(&v)?;

        // K_pe: [B, S, rope_head_dim] → [B, 1, S, rope_head_dim] → [B, H, S, rope_head_dim]
        let k_pe = var_reshape(&k_pe_raw, &[batch, 1, seq_len, self.rope_head_dim])
            .map_err(Error::Numr)?;
        let k_pe = var_broadcast_to(&k_pe, &[batch, self.num_heads, seq_len, self.rope_head_dim])
            .map_err(Error::Numr)?;
        let k_pe = var_contiguous(&k_pe)?;

        // Apply RoPE to q_pe and k_pe (decoupled)
        let q_pe = apply_rope_impl(client, &q_pe, self.rope.cos_cache(), self.rope.sin_cache())?;
        let k_pe = apply_rope_impl(client, &k_pe, self.rope.cos_cache(), self.rope.sin_cache())?;

        // Concatenate: Q = [q_nope, q_pe], K = [k_nope, k_pe]
        let q = var_cat(&[&q_nope, &q_pe], 3, client).map_err(Error::Numr)?;
        let k = var_cat(&[&k_nope, &k_pe], 3, client).map_err(Error::Numr)?;

        // Attention: Q,K [B, H, S, qk_dim], V [B, H, S, head_dim_v]
        let attn_out = match bias {
            None => scaled_dot_product_attention_impl(client, &q, &k, &v, self.scale, true)?,
            Some(bias) => scaled_dot_product_attention_masked_impl(
                client,
                &q,
                &k,
                &v,
                self.scale,
                Some(bias),
            )?,
        };

        // [B, H, S, head_dim_v] → [B, S, H, head_dim_v] → [B, S, H*head_dim_v]
        let attn_out = var_permute(&attn_out, &[0, 2, 1, 3]).map_err(Error::Numr)?;
        let attn_out = var_contiguous(&attn_out)?;
        let attn_out = var_reshape(
            &attn_out,
            &[batch, seq_len, self.num_heads * self.head_dim_v],
        )
        .map_err(Error::Numr)?;

        // Output projection
        self.o_proj.forward(client, &attn_out)
    }
}

#[cfg(all(test, feature = "f16"))]
mod tests {
    use super::*;
    use crate::nn::mla::MlaWeights;
    use crate::nn::{Linear, MaybeQuantLinear, MlaConfig, RmsNorm};
    use crate::test_utils::{Lcg, cpu_setup, ids_for, max_abs_diff};
    use numr::autograd::{backward, var_cast, var_mul, var_sum};
    use numr::runtime::cpu::CpuRuntime;

    const HIDDEN: usize = 16;

    type T = Tensor<CpuRuntime>;

    fn f32s(t: &T) -> Vec<f32> {
        t.contiguous()
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec::<f32>()
    }

    /// `data` rounded to `round`, then stored as `store`.
    fn rounded(data: &[f32], shape: &[usize], round: DType, store: DType) -> T {
        let (_, dev) = cpu_setup();
        let t = T::from_slice(data, shape, &dev).unwrap();
        t.to_dtype(round).unwrap().to_dtype(store).unwrap()
    }

    /// An MLA layer whose weights hold `dtype`-rounded values, stored as `store`.
    /// The F32 reference and the low-precision layer share every weight value.
    fn build(round: DType, store: DType) -> Mla<CpuRuntime> {
        let (_, dev) = cpu_setup();
        let config = MlaConfig::deepseek_v2(HIDDEN, 2, 8, 8, 4, 16);
        let mut rng = Lcg(5);
        let mut lin = |out: usize, inp: usize| {
            let data = rng.data(out * inp, 0.4);
            MaybeQuantLinear::Standard(Linear::new(
                rounded(&data, &[out, inp], round, store),
                None,
                true,
            ))
        };
        let (q_down, q_up) = (lin(8, HIDDEN), lin(2 * 12, 8));
        let (kv_compress, kv_decompress) = (lin(8 + 4, HIDDEN), lin(2 * 16, 8));
        let o_proj = lin(HIDDEN, 2 * 8);
        let norm = |n: usize| RmsNorm::new(rounded(&vec![1.0; n], &[n], round, store), 1e-6, true);
        let weights = MlaWeights {
            q_down: Some(q_down),
            q_up,
            q_norm: Some(norm(8)),
            kv_compress,
            kv_norm: Some(norm(8)),
            kv_decompress,
            o_proj,
        };
        Mla::with_ids(&config, weights, &dev).unwrap()
    }

    struct Run {
        y: Vec<f32>,
        gx: Vec<f32>,
        gp: Vec<Vec<f32>>,
    }

    /// Forward and backward on `x` of `[1, seq, HIDDEN]`. The loss sums the
    /// squared outputs in F32.
    fn run(mla: &Mla<CpuRuntime>, x: &[f32], dtype: DType, ids: Option<&[i32]>) -> Run {
        let (client, dev) = cpu_setup();
        let seq = x.len() / HIDDEN;
        let xv = Var::new(rounded(x, &[1, seq, HIDDEN], dtype, dtype), true);
        let ids = ids.map(|i| T::from_slice(i, &[1, seq], &dev).unwrap());
        let y = mla
            .forward_with_segments(&client, &xv, ids.as_ref())
            .unwrap();
        assert_eq!(y.tensor().dtype(), dtype);
        let y32 = var_cast(&y, DType::F32, &client).unwrap();
        let sq = var_mul(&y32, &y32, &client).unwrap();
        let loss = var_sum(&sq, &[0, 1, 2], false, &client).unwrap();
        let g = backward(&loss, &client).unwrap();
        let grad = |id, n: usize| g.get(id).map_or(vec![0.0; n], f32s);
        Run {
            y: f32s(y.tensor()),
            gx: grad(xv.id(), x.len()),
            gp: mla
                .parameters()
                .into_iter()
                .map(|(id, p)| grad(id, p.tensor().numel()))
                .collect(),
        }
    }

    /// `got` is within `rel * max|want|` of `want`. The bound scales with the
    /// signal, because a low-precision value carries a relative rounding error.
    fn assert_close(got: &[f32], want: &[f32], rel: f32, what: &str) {
        let peak = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak > 0.0, "{what}: reference is all zero");
        let diff = max_abs_diff(got, want);
        assert!(diff <= rel * peak, "{what}: diff {diff} > {rel} * {peak}");
    }

    /// Output and gradient bounds relative to the peak. BF16 keeps 8 significant
    /// bits and F16 keeps 11. Gradients cross more rounded operations than the
    /// output, so their bound is wider.
    fn bounds(dtype: DType) -> (f32, f32) {
        match dtype {
            DType::BF16 => (0.06, 0.2),
            _ => (0.01, 0.04),
        }
    }

    fn compare_to_f32(dtype: DType, lens: &[usize], masked: bool) {
        let seq: usize = lens.iter().sum();
        let x = Lcg(7).data(seq * HIDDEN, 1.0);
        // Both runs read the same `dtype`-rounded input.
        let x = f32s(&rounded(&x, &[1, seq, HIDDEN], dtype, DType::F32));
        let ids = ids_for(lens);
        let ids = masked.then_some(ids.as_slice());
        let want = run(&build(dtype, DType::F32), &x, DType::F32, ids);
        let got = run(&build(dtype, dtype), &x, dtype, ids);
        let (out_rel, grad_rel) = bounds(dtype);
        let tag = format!("{dtype:?} lens {lens:?}");
        assert_close(&got.y, &want.y, out_rel, &format!("y {tag}"));
        assert_close(&got.gx, &want.gx, grad_rel, &format!("gx {tag}"));
        for (i, (g, w)) in got.gp.iter().zip(&want.gp).enumerate() {
            assert_close(g, w, grad_rel, &format!("param {i} {tag}"));
        }
    }

    #[test]
    fn half_precision_document_mask_matches_f32() {
        for dtype in [DType::BF16, DType::F16] {
            compare_to_f32(dtype, &[5, 4, 3], true);
        }
    }

    #[test]
    fn half_precision_causal_matches_f32() {
        for dtype in [DType::BF16, DType::F16] {
            compare_to_f32(dtype, &[12], false);
        }
    }

    #[test]
    fn bf16_single_document_equals_causal_bit_for_bit() {
        let x = Lcg(9).data(8 * HIDDEN, 1.0);
        let mla = build(DType::BF16, DType::BF16);
        let plain = run(&mla, &x, DType::BF16, None);
        let ids = ids_for(&[8]);
        let single = run(&mla, &x, DType::BF16, Some(&ids));
        assert_eq!(plain.y, single.y);
    }

    #[test]
    fn bf16_perturbing_one_document_leaves_the_other_bit_identical() {
        let (first, rest) = (5, 7);
        let x = Lcg(21).data((first + rest) * HIDDEN, 1.0);
        let mut x2 = x.clone();
        let noise = Lcg(22).data(first * HIDDEN, 3.0);
        x2.iter_mut().zip(&noise).for_each(|(a, n)| *a += n);
        let mla = build(DType::BF16, DType::BF16);
        let ids = ids_for(&[first, rest]);
        let a = run(&mla, &x, DType::BF16, Some(&ids));
        let b = run(&mla, &x2, DType::BF16, Some(&ids));
        assert_ne!(a.y[..first * HIDDEN], b.y[..first * HIDDEN]);
        assert_eq!(a.y[first * HIDDEN..], b.y[first * HIDDEN..]);
    }
}
