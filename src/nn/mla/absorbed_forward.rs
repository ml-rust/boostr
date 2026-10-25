//! [`AbsorbedMla::forward_inference`]: the KV-cached absorbed MLA forward,
//! for a prefill of any length and a single-token decode alike.

use super::absorbed::AbsorbedMla;
use super::latent_attention::latent_attention;
use crate::error::{Error, Result};
use crate::inference::KvCache;
use crate::nn::var_ops::var_contiguous;
use crate::ops::impl_generic::attention::rope::apply_rope_impl;
use crate::ops::traits::{KvCacheOps, RoPEOps};
use crate::quant::traits::{DequantOps, QuantMatmulOps};
use numr::autograd::{Var, var_cat, var_matmul, var_narrow, var_permute, var_reshape};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps, ShapeOps, TensorOps, TypeConversionOps};
use numr::runtime::{Runtime, RuntimeClient};

impl<R: Runtime<DType = DType>> AbsorbedMla<R> {
    /// One KV-cached step: `hidden` `[B, S, hidden]` → `[B, S, hidden]`.
    ///
    /// `position` is the absolute position of the first token of `hidden`.
    /// It must equal `cache.seq_len()`. The step rotates the decoupled RoPE
    /// part at positions `position..position + S`, appends one
    /// `[c_kv | k_pe]` entry per token to `cache` (K and V alike), and
    /// attends over every cached entry. Query `i` sees keys up to
    /// `position + i`.
    ///
    /// # Errors
    ///
    /// `hidden` not 3-D, a `position` other than `cache.seq_len()`, a cache
    /// whose head count or width differs from
    /// [`cache_kv_heads`](Self::cache_kv_heads) and
    /// [`cache_head_dim`](Self::cache_head_dim), or a position past the RoPE
    /// tables.
    pub fn forward_inference<C>(
        &self,
        client: &C,
        hidden: &Var<R>,
        cache: &mut KvCache<R>,
        position: usize,
    ) -> Result<Var<R>>
    where
        C: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + ShapeOps<R>
            + BinaryOps<R>
            + TypeConversionOps<R>
            + QuantMatmulOps<R>
            + RoPEOps<R>
            + KvCacheOps<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + DequantOps<R>,
    {
        let shape = hidden.shape().to_vec();
        let &[batch, seq_len, _] = shape.as_slice() else {
            return Err(Error::InvalidArgument {
                arg: "hidden",
                reason: format!("AbsorbedMla: hidden must be [B, S, hidden], got {shape:?}"),
            });
        };
        self.check_cache(cache, position)?;
        let (h, d, rope_d, r) = (
            self.num_heads,
            self.head_dim,
            self.rope_head_dim,
            self.kv_lora_rank,
        );

        // Rotary tables for positions `position..position + S`.
        let cos = var_narrow(self.rope.cos_cache(), 0, position, seq_len).map_err(Error::Numr)?;
        let sin = var_narrow(self.rope.sin_cache(), 0, position, seq_len).map_err(Error::Numr)?;

        // === Query: [q_nope · W_UK | rotated q_pe], [B, H, S, r + rope] ===
        let q = match &self.q_down {
            Some(q_down) => {
                let latent = q_down.forward(client, hidden)?;
                let latent = match &self.q_norm {
                    Some(norm) => norm.forward(client, &latent)?,
                    None => latent,
                };
                self.q_up.forward(client, &latent)?
            }
            None => self.q_up.forward(client, hidden)?,
        };
        let q = var_reshape(&q, &[batch, seq_len, h, d + rope_d]).map_err(Error::Numr)?;
        let q = var_contiguous(&var_permute(&q, &[0, 2, 1, 3]).map_err(Error::Numr)?)?;
        let q_nope = var_contiguous(&var_narrow(&q, 3, 0, d).map_err(Error::Numr)?)?;
        let q_pe = var_contiguous(&var_narrow(&q, 3, d, rope_d).map_err(Error::Numr)?)?;
        let q_latent = var_matmul(&q_nope, &self.w_uk, client).map_err(Error::Numr)?;
        let q_pe = apply_rope_impl(client, &q_pe, &cos, &sin)?;
        let q = var_cat(&[&q_latent, &q_pe], 3, client).map_err(Error::Numr)?;
        let q = var_contiguous(&q)?;

        // === Cache entry: [norm(c_kv) | rotated k_pe], [B, 1, S, r + rope] ===
        let kv = self.kv_compress.forward(client, hidden)?;
        let c_kv = var_contiguous(&var_narrow(&kv, 2, 0, r).map_err(Error::Numr)?)?;
        let c_kv = match &self.kv_norm {
            Some(norm) => norm.forward(client, &c_kv)?,
            None => c_kv,
        };
        let c_kv = var_reshape(&c_kv, &[batch, 1, seq_len, r]).map_err(Error::Numr)?;
        let k_pe = var_contiguous(&var_narrow(&kv, 2, r, rope_d).map_err(Error::Numr)?)?;
        let k_pe = var_reshape(&k_pe, &[batch, 1, seq_len, rope_d]).map_err(Error::Numr)?;
        let k_pe = apply_rope_impl(client, &k_pe, &cos, &sin)?;
        let entry = var_cat(&[&c_kv, &k_pe], 3, client).map_err(Error::Numr)?;
        let entry = entry.tensor().contiguous()?;
        cache.update_fused(&entry, &entry, client)?;

        let (keys, values) = cache.get_kv()?;
        let keys = Var::new(keys.contiguous()?, false);
        let latents = Var::new(values.narrow(3, 0, r)?.contiguous()?, false);

        // === Attention over the cache, then latent → per-head V → o_proj ===
        let out = latent_attention(client, &q, &keys, &latents, self.scale)?;
        let out = var_matmul(&out, &self.w_uv_t, client).map_err(Error::Numr)?;
        let out = var_contiguous(&var_permute(&out, &[0, 2, 1, 3]).map_err(Error::Numr)?)?;
        let out = var_reshape(&out, &[batch, seq_len, h * self.head_dim_v]).map_err(Error::Numr)?;
        self.o_proj.forward(client, &out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::config::RopeScalingConfig;
    use crate::nn::mla::{Mla, MlaConfig, MlaWeights};
    use crate::nn::{Linear, MaybeQuantLinear, RmsNorm};
    use crate::test_utils::{Lcg, cpu_setup, max_abs_diff};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    const HIDDEN: usize = 16;
    const MAX_SEQ: usize = 16;
    /// The absorbed form reorders sums, so F32 results agree to rounding.
    const TOL: f32 = 2e-5;

    type T = Tensor<CpuRuntime>;

    fn config(scaling: Option<RopeScalingConfig>) -> MlaConfig {
        MlaConfig {
            rope_scaling: scaling,
            ..MlaConfig::deepseek_v2(HIDDEN, 2, 8, 6, 4, MAX_SEQ)
        }
    }

    /// An MLA layer with seeded weights. The same seed builds the same layer.
    fn build(config: &MlaConfig, seed: u64) -> Mla<CpuRuntime> {
        let (_, dev) = cpu_setup();
        let mut rng = Lcg(seed);
        let (h, d, rope, r, qr) = (
            config.num_heads,
            config.head_dim,
            config.rope_head_dim,
            config.kv_lora_rank,
            config.q_lora_rank,
        );
        let mut lin = |out: usize, inp: usize| {
            MaybeQuantLinear::Standard(Linear::new(rng.tensor(&dev, &[out, inp], 0.4), None, false))
        };
        let (q_down, q_up) = (lin(qr, HIDDEN), lin(h * (d + rope), qr));
        let kv_compress = lin(r + rope, HIDDEN);
        let kv_decompress = lin(h * (d + config.head_dim_v), r);
        let o_proj = lin(HIDDEN, h * config.head_dim_v);
        let mut norm_rng = Lcg(seed ^ 0x55);
        let mut norm = |n: usize| {
            let w: Vec<f32> = norm_rng.data(n, 0.3).iter().map(|v| 1.0 + v).collect();
            RmsNorm::new(
                T::from_slice(&w, &[n], &dev).unwrap(),
                config.norm_eps,
                false,
            )
        };
        let weights = MlaWeights {
            q_down: Some(q_down),
            q_up,
            q_norm: Some(norm(qr)),
            kv_compress,
            kv_norm: Some(norm(r)),
            kv_decompress,
            o_proj,
        };
        Mla::with_ids(config, weights, &dev).unwrap()
    }

    fn cache(dev: &CpuDevice, batch: usize, config: &MlaConfig) -> KvCache<CpuRuntime> {
        let width = config.latent_cache_dim();
        KvCache::new(batch, 1, 4, MAX_SEQ, width, DType::F32, dev).unwrap()
    }

    /// The training forward over the whole of `x`.
    fn reference(mla: &Mla<CpuRuntime>, x: &T) -> Vec<f32> {
        let (client, _) = cpu_setup();
        let y = mla.forward(&client, &Var::new(x.clone(), false)).unwrap();
        y.tensor().contiguous().unwrap().to_vec::<f32>()
    }

    /// The absorbed forward over `x`, fed as consecutive chunks of `chunks`
    /// tokens, with the outputs joined in token order.
    fn absorbed(mla: AbsorbedMla<CpuRuntime>, x: &T, chunks: &[usize], cfg: &MlaConfig) -> T {
        let (client, dev) = cpu_setup();
        let batch = x.shape()[0];
        let mut kv = cache(&dev, batch, cfg);
        let mut outs = Vec::new();
        let mut at = 0;
        for &n in chunks {
            let piece = Var::new(x.narrow(1, at, n).unwrap().contiguous().unwrap(), false);
            let y = mla.forward_inference(&client, &piece, &mut kv, at).unwrap();
            outs.push(y.tensor().contiguous().unwrap());
            at += n;
        }
        assert_eq!(at, x.shape()[1], "chunks must cover x");
        let refs: Vec<&T> = outs.iter().collect();
        T::cat(&refs, 1).unwrap()
    }

    fn input(batch: usize, seq: usize, seed: u64) -> T {
        let (_, dev) = cpu_setup();
        Lcg(seed).tensor(&dev, &[batch, seq, HIDDEN], 1.0)
    }

    fn assert_matches(got: &T, want: &[f32], what: &str) {
        let got = got.to_vec::<f32>();
        let diff = max_abs_diff(&got, want);
        assert!(diff <= TOL, "{what}: max diff {diff}");
    }

    #[test]
    fn decode_token_by_token_equals_the_training_forward() {
        let cfg = config(None);
        let x = input(1, 9, 11);
        let want = reference(&build(&cfg, 3), &x);
        let got = absorbed(
            build(&cfg, 3).absorb(DType::F32).unwrap(),
            &x,
            &[1; 9],
            &cfg,
        );
        assert_matches(&got, &want, "token by token");
    }

    #[test]
    fn prefill_then_decode_equals_one_shot() {
        let cfg = config(None);
        let x = input(1, 10, 12);
        let want = reference(&build(&cfg, 4), &x);
        let mla = || build(&cfg, 4).absorb(DType::F32).unwrap();
        assert_matches(&absorbed(mla(), &x, &[10], &cfg), &want, "one shot");
        assert_matches(
            &absorbed(mla(), &x, &[6, 1, 1, 1, 1], &cfg),
            &want,
            "prefill then decode",
        );
        assert_matches(
            &absorbed(mla(), &x, &[3, 4, 1, 2], &cfg),
            &want,
            "chunked prefill",
        );
    }

    #[test]
    fn batched_rows_and_scaled_rope_match_the_training_forward() {
        let scaling = RopeScalingConfig {
            scaling_type: "linear".into(),
            factor: 2.0,
            original_max_position_embeddings: None,
            low_freq_factor: None,
            high_freq_factor: None,
            attention_factor: None,
            beta_fast: None,
            beta_slow: None,
            short_factor: None,
            long_factor: None,
        };
        let cfg = config(Some(scaling));
        let x = input(2, 7, 13);
        let want = reference(&build(&cfg, 5), &x);
        let mla = || build(&cfg, 5).absorb(DType::F32).unwrap();
        assert_matches(&absorbed(mla(), &x, &[7], &cfg), &want, "batch one shot");
        assert_matches(
            &absorbed(mla(), &x, &[4, 1, 1, 1], &cfg),
            &want,
            "batch decode",
        );
        let unscaled = reference(&build(&config(None), 5), &x);
        assert!(
            max_abs_diff(&want, &unscaled) > 1e-3,
            "the scaled tables must move the output"
        );
    }

    #[test]
    fn absorb_refuses_a_kv_b_proj_with_a_bias() {
        let (_, dev) = cpu_setup();
        let cfg = config(None);
        let mut mla = build(&cfg, 7);
        let w = mla.kv_decompress.weight().unwrap().tensor().clone();
        let bias = T::zeros(&[w.shape()[0]], DType::F32, &dev).unwrap();
        mla.kv_decompress = MaybeQuantLinear::Standard(Linear::new(w, Some(bias), false));
        let err = mla.absorb(DType::F32).err().expect("refused").to_string();
        assert!(err.contains("bias"), "{err}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_decode_matches_cpu() {
        use numr::runtime::cuda::CudaRuntime;
        let Some(gpu) = crate::test_utils::cuda_setup() else {
            return;
        };
        let cfg = config(None);
        let x = input(1, 6, 15);
        let want = absorbed(
            build(&cfg, 8).absorb(DType::F32).unwrap(),
            &x,
            &[3, 1, 1, 1],
            &cfg,
        )
        .to_vec::<f32>();

        let cpu = build(&cfg, 8);
        let to_gpu = |t: &T| {
            Tensor::<CudaRuntime>::from_slice(&t.to_vec::<f32>(), t.shape(), &gpu.device).unwrap()
        };
        let lin = |l: &MaybeQuantLinear<CpuRuntime>| {
            MaybeQuantLinear::Standard(Linear::new(
                to_gpu(l.weight().unwrap().tensor()),
                None,
                false,
            ))
        };
        let norm = |n: &Option<RmsNorm<CpuRuntime>>| {
            n.as_ref()
                .map(|n| RmsNorm::new(to_gpu(n.weight().tensor()), cfg.norm_eps, false))
        };
        let weights = MlaWeights {
            q_down: cpu.q_down.as_ref().map(lin),
            q_up: lin(&cpu.q_up),
            q_norm: norm(&cpu.q_norm),
            kv_compress: lin(&cpu.kv_compress),
            kv_norm: norm(&cpu.kv_norm),
            kv_decompress: lin(&cpu.kv_decompress),
            o_proj: lin(&cpu.o_proj),
        };
        let mla = Mla::with_ids(&cfg, weights, &gpu.device)
            .unwrap()
            .absorb(DType::F32)
            .unwrap();
        let mut kv = KvCache::<CudaRuntime>::new(
            1,
            1,
            4,
            MAX_SEQ,
            cfg.latent_cache_dim(),
            DType::F32,
            &gpu.device,
        )
        .unwrap();
        let x_gpu = to_gpu(&x);
        let mut got = Vec::new();
        let mut at = 0;
        for n in [3, 1, 1, 1] {
            let piece = Var::new(x_gpu.narrow(1, at, n).unwrap().contiguous().unwrap(), false);
            let y = mla
                .forward_inference(&gpu.client, &piece, &mut kv, at)
                .unwrap();
            got.extend(y.tensor().contiguous().unwrap().to_vec::<f32>());
            at += n;
        }
        let diff = max_abs_diff(&got, &want);
        assert!(diff <= 1e-4, "CUDA vs CPU max diff {diff}");
    }
}
