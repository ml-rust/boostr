//! `prefill` and `decode_step`, over the cached layer stack in `super::stack`.

use crate::error::{Error, Result};
use crate::inference::LayeredKvCache;
use crate::model::audio::voxcpm::minicpm4::model::MiniCpm4Model;
use crate::model::traits::ModelClient;
use crate::quant::traits::DequantOps;
use numr::autograd::{Var, var_reshape};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> MiniCpm4Model<R> {
    /// Run the full prefix and populate `kv_cache` with its K/V.
    ///
    /// `inputs_embeds: [batch, seq, hidden_size]` -> `[batch, seq,
    /// hidden_size]`, the same value and shape
    /// [`forward`](MiniCpm4Model::forward) returns for the same input.
    ///
    /// The cache is RESET first, mirroring the reference's `fill_caches`, which
    /// zeroes the buffers before copying the primed length in. Call this ONCE
    /// per sequence, then [`decode_step`](Self::decode_step) from
    /// `position == seq`.
    ///
    /// `kv_start` is the per-row left-padding start, `[batch]` I32: row `b`
    /// attends no key below `kv_start[b]`, on every layer, here and on every
    /// later [`decode_step`](Self::decode_step) with the same tensor. `None`
    /// is the unpadded call.
    pub fn prefill<C>(
        &self,
        client: &C,
        inputs_embeds: &Var<R>,
        kv_cache: &mut LayeredKvCache<R>,
        kv_start: Option<&Tensor<R>>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
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
        let shape = inputs_embeds.shape().to_vec();
        if shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "inputs_embeds",
                reason: format!(
                    "expected 3D [batch, seq, hidden_size], got {}D",
                    shape.len()
                ),
            });
        }
        self.check_cache(kv_cache, shape[0], shape[2], shape[1], 0)?;

        kv_cache.reset();
        self.forward_cached(client, inputs_embeds, kv_cache, 0, kv_start)
    }

    /// Advance one position.
    ///
    /// `embed: [batch, hidden_size]` -> `[batch, hidden_size]`. The step entry
    /// point takes and returns the SQUEEZED 2D shape because that is what the
    /// reference's `forward_step` is handed (`curr_embed[:, 0, :]`); a caller
    /// porting a generation loop passes the same tensor it already has, with no
    /// unsqueeze on either side. The `[batch, 1, hidden]` view is rebuilt
    /// internally for the layer stack.
    ///
    /// `position` is the ABSOLUTE index of this embedding and must equal
    /// `kv_cache.seq_len()` — the slot the cache will write. Accepting any
    /// other index would rotate the query at one position while filing its key
    /// at another, which stays shape-valid and silently computes a different
    /// model.
    ///
    /// `kv_start` is the same per-row left-padding start the cache was
    /// prefilled with (see [`prefill`](Self::prefill)); `None` when it was
    /// prefilled unpadded.
    ///
    /// Errors (never panics, never writes out of range) when `position`
    /// reaches the cache's `max_length`, when it disagrees with the cache
    /// length, or when the shape or batch does not match.
    pub fn decode_step<C>(
        &self,
        client: &C,
        embed: &Var<R>,
        kv_cache: &mut LayeredKvCache<R>,
        position: usize,
        kv_start: Option<&Tensor<R>>,
    ) -> Result<Var<R>>
    where
        C: ModelClient<R> + TypeConversionOps<R>,
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
        let shape = embed.shape().to_vec();
        if shape.len() != 2 {
            return Err(Error::InvalidArgument {
                arg: "embed",
                reason: format!(
                    "expected 2D [batch, hidden_size] (one position), got {}D",
                    shape.len()
                ),
            });
        }
        let (batch, hidden) = (shape[0], shape[1]);
        self.check_cache(kv_cache, batch, hidden, 1, position)?;

        // The write-order rule that makes this cache equivalent to the
        // reference's preallocated one: `position` must be the slot the cache
        // will actually write. `prefill` is exempt because it resets first.
        // `check_cache` above already required layer 0 to exist, so this
        // cannot be `None` in practice; propagate rather than default to 0,
        // which would silently mask that invariant if it were ever broken.
        let filled = kv_cache
            .layer(0)
            .ok_or_else(|| Error::ModelError {
                reason: "KV cache missing layer 0 after validation".to_string(),
            })?
            .seq_len();
        if filled != position {
            return Err(Error::InvalidArgument {
                arg: "position",
                reason: format!(
                    "expected {filled} (the next free cache slot), got {position}; \
                     positions must be written in order from 0"
                ),
            });
        }

        let x = var_reshape(embed, &[batch, 1, hidden]).map_err(Error::Numr)?;
        let out = self.forward_cached(client, &x, kv_cache, position, kv_start)?;
        var_reshape(&out, &[batch, hidden]).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the MiniCPM4 incremental-decode path.

    use super::*;
    use crate::model::audio::voxcpm::minicpm4::model::tests::{
        HIDDEN, filled, tiny_model, tiny_nope_model,
    };
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn values(v: &Var<CpuRuntime>) -> Vec<f32> {
        v.tensor().contiguous().expect("contiguous").to_vec::<f32>()
    }

    /// Non-degenerate, deterministic embeddings for `seq` positions, batch 1.
    fn embeds(seq: usize, device: &numr::runtime::cpu::CpuDevice) -> Var<CpuRuntime> {
        let data: Vec<f32> = (0..seq * HIDDEN)
            .map(|i| ((i % 11) as f32 - 5.0) / 8.0)
            .collect();
        Var::new(
            Tensor::<CpuRuntime>::from_slice(&data, &[1, seq, HIDDEN], device).expect("embeds"),
            false,
        )
    }

    /// One position of `[1, seq, HIDDEN]` as the `[1, HIDDEN]` shape
    /// `decode_step` takes.
    fn row_at(x: &Var<CpuRuntime>, position: usize) -> Var<CpuRuntime> {
        Var::new(
            x.tensor()
                .narrow(1, position, 1)
                .expect("narrow")
                .contiguous()
                .expect("contiguous")
                .reshape(&[1, HIDDEN])
                .expect("reshape"),
            false,
        )
    }

    /// The load-bearing test: stepping one position at a time must reproduce
    /// the full-sequence forward. Different key-axis reduction order, so a
    /// tight tolerance rather than bit-equality.
    #[test]
    fn step_wise_matches_full_sequence() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);
        let seq = 5;

        let x = embeds(seq, &device);
        let full = values(&model.forward(&client, &x).expect("forward"));

        let mut cache = model.new_kv_cache(1, 8).expect("cache");
        for position in 0..seq {
            let row = row_at(&x, position);
            let step = values(
                &model
                    .decode_step(&client, &row, &mut cache, position, None)
                    .expect("decode_step"),
            );
            assert_eq!(step.len(), HIDDEN);
            let expected = &full[position * HIDDEN..(position + 1) * HIDDEN];
            for (got, want) in step.iter().zip(expected) {
                assert!(
                    (got - want).abs() < 1e-5,
                    "position {position}: step {got} vs full {want}"
                );
            }
            // A model whose output is constant across positions would pass the
            // comparison above vacuously.
            assert!(step.iter().any(|v| v.abs() > 1e-6), "degenerate output");
        }
        assert_eq!(cache.seq_len(), seq);
    }

    /// Prefill a prefix, then continue stepping — the real generate sequence.
    #[test]
    fn prefill_then_step_matches_full_sequence() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);
        let seq = 5;
        let prefix = 3;

        let x = embeds(seq, &device);
        let full = values(&model.forward(&client, &x).expect("forward"));

        let mut cache = model.new_kv_cache(1, seq).expect("cache");
        let prefix_x = Var::new(
            x.tensor()
                .narrow(1, 0, prefix)
                .expect("narrow")
                .contiguous()
                .expect("contiguous"),
            false,
        );
        let primed = values(
            &model
                .prefill(&client, &prefix_x, &mut cache, None)
                .expect("prefill"),
        );
        assert_eq!(cache.seq_len(), prefix);
        for (got, want) in primed.iter().zip(&full[..prefix * HIDDEN]) {
            assert!((got - want).abs() < 1e-5, "prefill {got} vs full {want}");
        }

        for position in prefix..seq {
            let row = row_at(&x, position);
            let step = values(
                &model
                    .decode_step(&client, &row, &mut cache, position, None)
                    .expect("decode_step"),
            );
            let expected = &full[position * HIDDEN..(position + 1) * HIDDEN];
            for (got, want) in step.iter().zip(expected) {
                assert!(
                    (got - want).abs() < 1e-5,
                    "position {position}: step {got} vs full {want}"
                );
            }
        }
    }

    /// The reference raises at `current_length >= max_length`. So do we — and
    /// without panicking or writing past the cache.
    #[test]
    fn stepping_past_max_length_errors() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);
        let max_length = 2;
        let mut cache = model.new_kv_cache(1, max_length).expect("cache");

        let row = Var::new(filled(&[1, HIDDEN], 3, &device), false);
        for position in 0..max_length {
            model
                .decode_step(&client, &row, &mut cache, position, None)
                .expect("in-range step");
        }
        let err = model
            .decode_step(&client, &row, &mut cache, max_length, None)
            .unwrap_err();
        assert!(err.to_string().contains("max_length"), "got {err}");
        assert_eq!(
            cache.seq_len(),
            max_length,
            "cache advanced past max_length"
        );
    }

    #[test]
    fn decode_step_rejects_out_of_order_position() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);
        let mut cache = model.new_kv_cache(1, 8).expect("cache");
        let row = Var::new(filled(&[1, HIDDEN], 3, &device), false);

        // Cache is empty, so only position 0 is writable.
        let err = model
            .decode_step(&client, &row, &mut cache, 3, None)
            .unwrap_err();
        assert!(
            err.to_string().contains("next free cache slot"),
            "got {err}"
        );
    }

    #[test]
    fn decode_step_rejects_3d_input() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);
        let mut cache = model.new_kv_cache(1, 8).expect("cache");
        let x = Var::new(filled(&[1, 1, HIDDEN], 3, &device), false);
        let err = model
            .decode_step(&client, &x, &mut cache, 0, None)
            .unwrap_err();
        assert!(err.to_string().contains("2D"), "got {err}");
    }

    #[test]
    fn prefill_rejects_batch_mismatch() {
        let (client, device) = cpu_setup();
        let model = tiny_model(&device);
        let mut cache = model.new_kv_cache(2, 8).expect("cache");
        let x = embeds(3, &device);
        let err = model.prefill(&client, &x, &mut cache, None).unwrap_err();
        assert!(err.to_string().contains("batch"), "got {err}");
    }

    /// The NoPE (`residual_lm`) stack has TWO RoPE call sites to skip — the
    /// full-sequence one inside `attention_core_masked` and the direct one in
    /// `forward_cached`. Honouring only one leaves the paths computing different
    /// models, which is exactly what this comparison catches.
    #[test]
    fn nope_step_wise_matches_full_sequence() {
        let (client, device) = cpu_setup();
        let model = tiny_nope_model(&device);
        assert!(!model.uses_rope());
        let seq = 5;

        let x = embeds(seq, &device);
        let full = values(&model.forward(&client, &x).expect("forward"));

        let mut cache = model.new_kv_cache(1, 8).expect("cache");
        for position in 0..seq {
            let row = row_at(&x, position);
            let step = values(
                &model
                    .decode_step(&client, &row, &mut cache, position, None)
                    .expect("decode_step"),
            );
            let expected = &full[position * HIDDEN..(position + 1) * HIDDEN];
            for (got, want) in step.iter().zip(expected) {
                assert!(
                    (got - want).abs() < 1e-5,
                    "position {position}: step {got} vs full {want}"
                );
            }
            assert!(step.iter().any(|v| v.abs() > 1e-6), "degenerate output");
        }
        assert_eq!(cache.seq_len(), seq);
    }

    /// Left padding under `kv_start`: a short row padded in front and stacked
    /// under a longer one, prefilled then stepped, reproduces its own
    /// unpadded batch-1 run at every real position. Without the start the
    /// zero pad rows are attended and the values move, which the final
    /// assertion checks so the test cannot pass vacuously.
    #[test]
    fn left_padded_batch_matches_unpadded_rows() {
        let (client, device) = cpu_setup();
        for model in [tiny_model(&device), tiny_nope_model(&device)] {
            let (long, short) = (5usize, 3usize);
            let pad = long - short;
            let x_long = embeds(long, &device);
            let x_short = embeds(short, &device);
            let zeros =
                Tensor::<CpuRuntime>::zeros(&[1, pad, HIDDEN], DType::F32, &device).expect("zeros");
            let padded = Tensor::cat(&[&zeros, x_short.tensor()], 1).expect("pad");
            let batch = Var::new(
                Tensor::cat(&[x_long.tensor(), &padded], 0).expect("stack"),
                false,
            );
            let kv_start = Tensor::<CpuRuntime>::from_slice(&[0i32, pad as i32], &[2], &device)
                .expect("start");

            // The same step row for every batch row, so row 1 of the batch
            // and the single-row run see identical inputs.
            let step_row = filled(&[1, HIDDEN], 3, &device);
            let step_pair = Var::new(
                Tensor::cat(&[&step_row, &step_row], 0).expect("stack"),
                false,
            );
            let run = |start: Option<&Tensor<CpuRuntime>>| {
                let mut cache = model.new_kv_cache(2, 8).expect("cache");
                let out = model
                    .prefill(&client, &batch, &mut cache, start)
                    .expect("prefill");
                let mut rows = values(&out);
                let stepped = model
                    .decode_step(&client, &step_pair, &mut cache, long, start)
                    .expect("step");
                rows.extend(values(&stepped));
                rows
            };
            let masked = run(Some(&kv_start));

            let mut cache = model.new_kv_cache(1, 8).expect("cache");
            let mut want = values(
                &model
                    .prefill(&client, &x_short, &mut cache, None)
                    .expect("prefill"),
            );
            let step_in = Var::new(step_row, false);
            want.extend(values(
                &model
                    .decode_step(&client, &step_in, &mut cache, short, None)
                    .expect("step"),
            ));

            // Row 1 of the batch: prefill rows `pad..long`, then its step row.
            let row1_prefill = &masked[(long + pad) * HIDDEN..2 * long * HIDDEN];
            let row1_step = &masked[(2 * long + 1) * HIDDEN..(2 * long + 2) * HIDDEN];
            let got: Vec<f32> = row1_prefill.iter().chain(row1_step).copied().collect();
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() < 1e-5, "padded {g} vs unpadded {w}");
            }

            let unmasked = run(None);
            assert!(
                unmasked
                    .iter()
                    .zip(&masked)
                    .any(|(a, b)| (a - b).abs() > 1e-4),
                "kv_start changed nothing, so the pad rows were never attended"
            );
        }
    }
}
