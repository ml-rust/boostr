//! The attention block's parameters, its `AttentionCoreSpec`, and the
//! cache-free (training / full-prefill) forward.

use crate::error::Result;
use crate::model::attention_core::{AttentionCoreSpec, AttentionKernel, attention_core_masked};
use crate::model::traits::ModelClient;
use crate::nn::{MaybeQuantLinear, RoPE};
use crate::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, UnaryOps,
};
use numr::runtime::Runtime;

/// GQA attention with Q/K/V projections
pub struct LlamaAttention<R: Runtime> {
    pub(crate) q_proj: MaybeQuantLinear<R>,
    pub(crate) k_proj: MaybeQuantLinear<R>,
    pub(crate) v_proj: MaybeQuantLinear<R>,
    pub(crate) o_proj: MaybeQuantLinear<R>,
    pub(crate) num_heads: usize,
    pub(crate) num_kv_heads: usize,
    pub(crate) head_dim: usize,
    /// Optional Q/K layer norms (Command-R, Cohere)
    pub(crate) q_norm: Option<crate::nn::RmsNorm<R>>,
    pub(crate) k_norm: Option<crate::nn::RmsNorm<R>>,
    /// Use ALiBi instead of RoPE (Falcon v1, BLOOM, MPT)
    pub(crate) use_alibi: bool,
    /// Sliding-window attention span. `0` disables windowing (unlimited context).
    ///
    /// The window is INCLUSIVE of the current token: query `i` may attend keys
    /// `j` with `i - sliding_window < j <= i`, i.e. exactly `sliding_window`
    /// keys. This matches the flash-attention kernel contract in
    /// `ops/impl_generic/attention/flash_standard.rs`.
    ///
    /// IGNORED when `use_alibi` is set. ALiBi's bias kernel writes the causal
    /// structure together with the distance bias; the two mechanisms do not
    /// compose here, so ALiBi models always attend the full context.
    pub(crate) sliding_window: usize,
}

impl<R: Runtime<DType = DType>> LlamaAttention<R> {
    /// Borrowed view of this block's attention parameters, for
    /// [`attention_core_masked`].
    pub(super) fn core_spec(&self) -> AttentionCoreSpec<'_, R> {
        AttentionCoreSpec {
            num_heads: self.num_heads,
            num_kv_heads: self.num_kv_heads,
            head_dim: self.head_dim,
            q_norm: self.q_norm.as_ref(),
            k_norm: self.k_norm.as_ref(),
            use_alibi: self.use_alibi,
            // LLaMA-lineage blocks always rotate (or use ALiBi); NoPE is a
            // VoxCPM2 `residual_lm` property, never one of theirs.
            skip_rope: false,
            sliding_window: self.sliding_window,
            // This block runs the materialized-mask kernel. Its ALiBi variants
            // require it, and `tests/qwen3_parity.rs` pins its numbers.
            kernel: AttentionKernel::Masked,
        }
    }

    /// Apply optional Q/K layer norms (Command-R, Cohere).
    /// Input shape: [B, H, S, D] — norm is applied over the last dimension (head_dim).
    pub(super) fn apply_qk_norms<C>(
        &self,
        client: &C,
        q: &Var<R>,
        k: &Var<R>,
    ) -> Result<(Var<R>, Var<R>)>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R>,
    {
        let q = match &self.q_norm {
            Some(norm) => norm.forward(client, q)?,
            None => q.clone(),
        };
        let k = match &self.k_norm {
            Some(norm) => norm.forward(client, k)?,
            None => k.clone(),
        };
        Ok((q, k))
    }

    /// Apply RoPE to Q/K or skip for ALiBi models.
    pub(super) fn apply_rotary_if_needed<C>(
        &self,
        client: &C,
        q: Var<R>,
        k: Var<R>,
        cos: &Var<R>,
        sin: &Var<R>,
    ) -> Result<(Var<R>, Var<R>)>
    where
        C: ModelClient<R>,
    {
        if self.use_alibi {
            Ok((q, k))
        } else {
            let q = client.apply_rope(&q, cos, sin)?;
            let k = client.apply_rope(&k, cos, sin)?;
            Ok((q, k))
        }
    }

    pub fn forward<C>(&self, client: &C, x: &Var<R>, rope: &RoPE<R>) -> Result<Var<R>>
    where
        C: ModelClient<R>,
        R::Client: TensorOps<R>
            + ScalarOps<R>
            + ReduceOps<R>
            + IndexingOps<R>
            + ShapeOps<R>
            + ActivationOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + DequantOps<R>,
    {
        // Q/K/V projections (batched: quantize activation once for all 3)
        let qkv = MaybeQuantLinear::forward_batch(
            &[&self.q_proj, &self.k_proj, &self.v_proj],
            client,
            x,
        )?;
        let (q, k, v) = (&qkv[0], &qkv[1], &qkv[2]);

        // Everything between the projections and `o_proj` — reshape/permute,
        // Q/K norm, RoPE, GQA, causal(+window)/ALiBi mask, attention — lives in
        // `attention_core_masked` so this block and a trainer's block cannot
        // drift on the step order (notably: norm BEFORE rope). It is the same
        // sequence and the same math `attention_core`'s `Masked` arm runs; the
        // separate entry point only drops the flash kernel's `R::Client` bound,
        // which the `Model` trait's fixed bound list cannot carry.
        let attn_out = attention_core_masked(
            client,
            q,
            k,
            v,
            Some(rope.cos_cache()),
            Some(rope.sin_cache()),
            &self.core_spec(),
        )?;

        // Output projection
        self.o_proj.forward(client, &attn_out)
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the sliding-window causal mask and the sliding-window config wiring.

    use super::super::super::builders::{build_block_from_config, build_block_from_varbuilder};
    use crate::model::config::ModelConfig;
    use crate::nn::{VarBuilder, VarMap};
    use crate::test_utils::cpu_setup;
    use numr::dtype::DType;
    use numr::ops::{LinalgOps, ScalarOps};
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    const NEG: f32 = f32::MIN;

    /// Reference mask: `0.0` where query `i` may attend key `j`, `f32::MIN` else.
    ///
    /// `window_size == 0` means unlimited, matching the kernel contract.
    fn expected_mask(sq: usize, sk: usize, window_size: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(sq * sk);
        for i in 0..sq {
            for j in 0..sk {
                let future = j > i;
                let too_old = window_size > 0 && j + window_size <= i;
                out.push(if future || too_old { NEG } else { 0.0 });
            }
        }
        out
    }

    fn mask_values(sq: usize, sk: usize, window_size: usize) -> Vec<f32> {
        let (client, device) = cpu_setup();
        let mask = crate::model::attention_mask::causal_window_mask::<CpuRuntime, _>(
            &client,
            sq,
            sk,
            window_size,
            DType::F32,
            &device,
        )
        .unwrap();
        assert_eq!(mask.shape(), &[1, 1, sq, sk]);
        mask.to_vec::<f32>()
    }

    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn windowed_mask_has_the_exact_expected_pattern() {
        let actual = mask_values(6, 6, 3);
        assert_eq!(actual, expected_mask(6, 6, 3));

        // Spelled out for the record: row `i` keeps exactly `j ∈ (i - 3, i]`.
        #[rustfmt::skip]
        let literal = vec![
            0.0, NEG, NEG, NEG, NEG, NEG,
            0.0, 0.0, NEG, NEG, NEG, NEG,
            0.0, 0.0, 0.0, NEG, NEG, NEG,
            NEG, 0.0, 0.0, 0.0, NEG, NEG,
            NEG, NEG, 0.0, 0.0, 0.0, NEG,
            NEG, NEG, NEG, 0.0, 0.0, 0.0,
        ];
        assert_eq!(actual, literal);
    }

    #[test]
    fn window_of_one_keeps_only_the_diagonal() {
        assert_eq!(mask_values(4, 4, 1), expected_mask(4, 4, 1));
    }

    #[test]
    fn disjoint_regions_never_sum_to_negative_infinity() {
        // `triu(1)` and `tril(-w)` must not overlap: an overlap would sum
        // `f32::MIN + f32::MIN` and overflow to `-inf`.
        for window_size in 1..=5 {
            for value in mask_values(6, 6, window_size) {
                assert!(value.is_finite(), "window {window_size} produced {value}");
            }
        }
    }

    #[test]
    fn zero_window_is_bit_identical_to_pure_causal() {
        let (client, device) = cpu_setup();
        // The pre-window formulation: `triu(full(f32::MIN), 1)`.
        let zeros = Tensor::<CpuRuntime>::zeros(&[5, 5], DType::F32, &device).unwrap();
        let filled = client.add_scalar(&zeros, f32::MIN as f64).unwrap();
        let pure_causal = client.triu(&filled, 1).unwrap().to_vec::<f32>();

        assert_eq!(bits(&mask_values(5, 5, 0)), bits(&pure_causal));
    }

    #[test]
    fn window_wider_than_the_sequence_is_pure_causal() {
        assert_eq!(mask_values(4, 4, 10), expected_mask(4, 4, 0));
        assert_eq!(mask_values(4, 4, 4), expected_mask(4, 4, 0));
    }

    // ── Builder wiring ──────────────────────────────────────────────────

    fn window_config(sliding_window: Option<usize>) -> ModelConfig {
        let window_line = match sliding_window {
            Some(w) => format!("\n  sliding_window: {w}"),
            None => String::new(),
        };
        let yaml = format!(
            r#"
model_type: llama
vocab_size: 32
hidden_size: 8
num_layers: 1
max_seq_len: 32
intermediate_size: 16
rms_norm_eps: 1.0e-5
attention:
  num_heads: 2
  rope_theta: 10000.0{window_line}
"#
        );
        serde_saphyr::from_str(&yaml).unwrap()
    }

    fn block_from_config(sliding_window: Option<usize>) -> usize {
        let (_, device) = cpu_setup();
        let config = window_config(sliding_window);
        let block =
            build_block_from_config::<CpuRuntime>(&config, &device, 2, 2, 4, 16, DType::F32)
                .expect("block build must succeed on CPU");
        block.self_attn.sliding_window
    }

    fn block_from_varbuilder(sliding_window: Option<usize>) -> usize {
        let (_, device) = cpu_setup();
        let config = window_config(sliding_window);
        let mut varmap = VarMap::<CpuRuntime>::new();
        for (name, shape) in [
            ("self_attn.q_proj.weight", vec![8usize, 8]),
            ("self_attn.k_proj.weight", vec![8, 8]),
            ("self_attn.v_proj.weight", vec![8, 8]),
            ("self_attn.o_proj.weight", vec![8, 8]),
            ("input_layernorm.weight", vec![8]),
            ("post_attention_layernorm.weight", vec![8]),
            ("mlp.gate_proj.weight", vec![16, 8]),
            ("mlp.up_proj.weight", vec![16, 8]),
            ("mlp.down_proj.weight", vec![8, 16]),
        ] {
            varmap.insert(
                name.into(),
                Tensor::<CpuRuntime>::zeros(&shape, DType::F32, &device).unwrap(),
            );
        }
        let mut vb = VarBuilder::new(&mut varmap, &device);
        let block = build_block_from_varbuilder::<CpuRuntime>(&mut vb, &config, 2, 2, 4).unwrap();
        block.self_attn.sliding_window
    }

    #[test]
    fn builders_read_sliding_window_from_config() {
        assert_eq!(block_from_config(Some(64)), 64);
        assert_eq!(block_from_varbuilder(Some(64)), 64);
    }

    #[test]
    fn absent_sliding_window_is_disabled() {
        assert_eq!(block_from_config(None), 0);
        assert_eq!(block_from_varbuilder(None), 0);
    }

    #[test]
    fn explicit_zero_sliding_window_is_disabled() {
        // `Some(0)` is not a zero-width window — it maps to the disabled sentinel.
        assert_eq!(block_from_config(Some(0)), 0);
        assert_eq!(block_from_varbuilder(Some(0)), 0);
    }

    /// ALiBi's prefill mask MUST mask the future.
    ///
    /// Regression guard for a real bug: this path built its bias with
    /// `alibi_add_bias`, which writes a SYMMETRIC `-slope * |qi - ki|` over the
    /// whole rectangle and masks nothing — so every prefill position attended to
    /// future tokens. The comment above it even claimed the opposite. Only
    /// `alibi_add_bias_causal` sets `ki > qi` to -inf.
    #[test]
    fn alibi_prefill_mask_is_causal() {
        let (client, device) = cpu_setup();
        let yaml = r#"
model_type: llama
vocab_size: 32
hidden_size: 8
num_layers: 1
max_seq_len: 32
intermediate_size: 16
rms_norm_eps: 1.0e-5
attention:
  num_heads: 2
  rope_theta: 10000.0
  use_alibi: true
"#;
        let config: ModelConfig = serde_saphyr::from_str(yaml).unwrap();
        let block =
            build_block_from_config::<CpuRuntime>(&config, &device, 2, 2, 4, 16, DType::F32)
                .expect("block build must succeed on CPU");
        assert!(block.self_attn.use_alibi, "fixture must exercise ALiBi");

        let sq = 4;
        // Go through the shared rule directly. `attention_core` is what the forward
        // path actually calls, so testing anything else would leave the real mask
        // unguarded — and a non-causal mask is invisible to shape checks while still
        // producing fluent text.
        let mask = crate::model::prefill_attention_mask(
            &client,
            1,
            sq,
            sq,
            &block.self_attn.core_spec(),
            DType::F32,
            &device,
        )
        .expect("mask builds");
        let values: Vec<f32> = mask.tensor().contiguous().unwrap().to_vec();

        // Head 0 occupies the first sq*sq entries.
        for i in 0..sq {
            for j in 0..sq {
                let got = values[i * sq + j];
                if j > i {
                    assert_eq!(
                        got,
                        f32::NEG_INFINITY,
                        "future key ({i},{j}) must be masked, got {got}"
                    );
                } else {
                    assert!(
                        got.is_finite() && got <= 0.0,
                        "past/self key ({i},{j}) must carry a finite ALiBi bias, got {got}"
                    );
                }
            }
        }
    }
}
