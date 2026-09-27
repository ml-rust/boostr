//! [`VitsModel::synthesize`] / [`VitsModel::synthesize_with_noise`]: the
//! forward pass, exactly HF's `VitsModel.forward` (inference path).

use crate::error::{Error, Result};
use crate::model::audio::vits::coupling::{sample_prior_latents, sample_prior_latents_with_noise};
use crate::model::audio::vits::length_regulator::regulate_length;
use crate::model::audio::vits::model::client::VitsModelClient;
use crate::model::audio::vits::model::core::{VitsDurationPredictorKind, VitsModel};
use crate::model::audio::vits::model::options::{VitsSynthOptions, VitsSynthesis};
use numr::dtype::DType;
use numr::ops::{IndexingOps, ScalarOps, TensorOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> VitsModel<R> {
    /// Runs the normal inference path: HF's `VitsModel.forward` with
    /// `speaker_id=opts.speaker_id` and no explicit noise, drawing both
    /// Gaussian noises via [`numr::ops::RandomOps::randn_seeded`].
    ///
    /// `input_ids` / `attention_mask`: `[batch, seq]` (attention_mask 1.0
    /// valid, 0.0 padded). The duration noise (`[batch, 2, seq]`) and the
    /// prior-latent noise (`[batch, flow_size, output_len]`) need distinct
    /// draws — reusing one seed for both would correlate two Gaussians HF
    /// draws independently — so this derives the prior seed as
    /// `opts.seed.wrapping_add(1)` from the duration seed `opts.seed`. Only
    /// the two draws' independence matters for perceptual output quality;
    /// neither draw is meant to bit-match `torch.randn`'s stream (a parity
    /// test replays a captured draw instead, via
    /// [`VitsModel::synthesize_with_noise`]).
    pub fn synthesize<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        attention_mask: &Tensor<R>,
        opts: &VitsSynthOptions,
    ) -> Result<VitsSynthesis<R>>
    where
        C: VitsModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        self.synthesize_impl(client, input_ids, attention_mask, opts, None, None)
    }

    /// Same pipeline as [`VitsModel::synthesize`], but `duration_noise`
    /// (`[batch, 2, seq]`) and `prior_noise` (`[batch, flow_size,
    /// output_len]`) replace the two internal `randn_seeded` draws. Exists
    /// so a parity test can replay HF's captured `torch.randn` /
    /// `torch.randn_like` draws bit-for-bit; [`VitsModel::synthesize`] is
    /// the normal entry point.
    pub fn synthesize_with_noise<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        attention_mask: &Tensor<R>,
        opts: &VitsSynthOptions,
        duration_noise: &Tensor<R>,
        prior_noise: &Tensor<R>,
    ) -> Result<VitsSynthesis<R>>
    where
        C: VitsModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        self.synthesize_impl(
            client,
            input_ids,
            attention_mask,
            opts,
            Some(duration_noise),
            Some(prior_noise),
        )
    }

    fn synthesize_impl<C>(
        &self,
        client: &C,
        input_ids: &Tensor<R>,
        attention_mask: &Tensor<R>,
        opts: &VitsSynthOptions,
        duration_noise: Option<&Tensor<R>>,
        prior_noise: Option<&Tensor<R>>,
    ) -> Result<VitsSynthesis<R>>
    where
        C: VitsModelClient<R>,
        R::Client: TensorOps<R> + ScalarOps<R> + IndexingOps<R>,
    {
        let batch = attention_mask.shape()[0];
        let seq_len = attention_mask.shape()[1];
        let padding_mask = attention_mask
            .reshape(&[batch, seq_len, 1])
            .map_err(Error::Numr)?;

        let speaker_embeddings = self.speaker_embedding(client, opts.speaker_id)?;

        let text_out =
            self.text_encoder
                .forward(client, input_ids, &padding_mask, Some(attention_mask))?;

        let hidden_states = text_out
            .last_hidden_state
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?;
        let input_padding_mask = padding_mask
            .transpose(1, 2)
            .and_then(|t| t.contiguous())
            .map_err(Error::Numr)?;

        let log_duration = match (&self.duration_predictor, duration_noise) {
            (VitsDurationPredictorKind::Stochastic(dp), Some(noise)) => dp
                .forward_reverse_with_noise(
                    client,
                    &hidden_states,
                    &input_padding_mask,
                    speaker_embeddings.as_ref(),
                    opts.noise_scale_duration,
                    noise,
                )?,
            (VitsDurationPredictorKind::Stochastic(dp), None) => dp.forward_reverse(
                client,
                &hidden_states,
                &input_padding_mask,
                speaker_embeddings.as_ref(),
                opts.noise_scale_duration,
                opts.seed,
            )?,
            (VitsDurationPredictorKind::Deterministic(dp), _) => dp.forward(
                client,
                &hidden_states,
                &input_padding_mask,
                speaker_embeddings.as_ref(),
            )?,
        };

        let regulated = regulate_length(
            client,
            &log_duration,
            &input_padding_mask,
            &text_out.prior_means,
            &text_out.prior_log_variances,
            opts.speaking_rate,
        )?;

        // See `synthesize`'s docs: the prior-latent draw must differ from
        // the duration draw, so it derives its own seed from `opts.seed`.
        let prior_seed = opts.seed.wrapping_add(1);
        let prior_latents = match prior_noise {
            Some(noise) => sample_prior_latents_with_noise(
                client,
                &regulated.expanded_prior_means,
                &regulated.expanded_prior_log_variances,
                opts.noise_scale,
                noise,
            )?,
            None => sample_prior_latents(
                client,
                &regulated.expanded_prior_means,
                &regulated.expanded_prior_log_variances,
                opts.noise_scale,
                prior_seed,
            )?,
        };

        let latents = self.flow.reverse(
            client,
            &prior_latents,
            &regulated.output_padding_mask,
            speaker_embeddings.as_ref(),
        )?;
        let spectrogram = client
            .mul(&latents, &regulated.output_padding_mask)
            .map_err(Error::Numr)?;
        let waveform = self
            .decoder
            .forward(client, &spectrogram, speaker_embeddings.as_ref())?;
        let waveform = waveform
            .squeeze(Some(1))
            .contiguous()
            .map_err(Error::Numr)?;

        let upsample_product: usize = self.config.upsample_rates.iter().product();
        let sequence_lengths = client
            .mul_scalar(&regulated.predicted_lengths, upsample_product as f64)
            .map_err(Error::Numr)?;

        Ok(VitsSynthesis {
            waveform,
            sequence_lengths,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{build_map as build_tiny_map, tiny_config};
    use super::*;
    use crate::nn::VarBuilder;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn synthesize_produces_a_finite_waveform_matching_predicted_lengths() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config(1, 0);
        let mut map = build_tiny_map(&cfg);
        let mut vb = VarBuilder::new(&mut map, &device);
        let model = VitsModel::<CpuRuntime>::from_varbuilder(&client, &mut vb, &cfg)
            .expect("load synthetic model");

        let batch = 1;
        let seq = 5;
        let input_ids: Vec<i64> = (1..=seq as i64).collect();
        let input_ids =
            Tensor::<CpuRuntime>::from_slice(&input_ids, &[batch, seq], &device).unwrap();
        let attention_mask =
            Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; batch * seq], &[batch, seq], &device)
                .unwrap();

        let opts = model.default_synth_options();
        let out = model
            .synthesize(&client, &input_ids, &attention_mask, &opts)
            .expect("synthesize");

        let upsample_product: usize = cfg.upsample_rates.iter().product();
        let lengths: Vec<i64> = out.sequence_lengths.contiguous().unwrap().to_vec();
        assert_eq!(lengths.len(), batch);
        let expected_samples = lengths[0] as usize;
        assert_eq!(out.waveform.shape(), &[batch, expected_samples]);
        // sequence_lengths is predicted_lengths * upsample_product, so it is
        // always a multiple of upsample_product.
        assert_eq!(lengths[0] as usize % upsample_product, 0);

        let samples: Vec<f32> = out.waveform.contiguous().unwrap().to_vec();
        assert!(samples.iter().all(|s| s.is_finite() && s.abs() <= 1.0));
    }

    #[test]
    fn synthesize_with_noise_is_deterministic_given_the_same_draws() {
        let (client, device) = cpu_setup();
        let cfg = tiny_config(1, 0);
        let mut map = build_tiny_map(&cfg);
        let mut vb = VarBuilder::new(&mut map, &device);
        let model = VitsModel::<CpuRuntime>::from_varbuilder(&client, &mut vb, &cfg)
            .expect("load synthetic model");

        let batch = 1;
        let seq = 4;
        let input_ids: Vec<i64> = (1..=seq as i64).collect();
        let input_ids =
            Tensor::<CpuRuntime>::from_slice(&input_ids, &[batch, seq], &device).unwrap();
        let attention_mask =
            Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; batch * seq], &[batch, seq], &device)
                .unwrap();
        let opts = model.default_synth_options();

        // The deterministic duration predictor draws no noise, so
        // `duration_noise` is unused; only `prior_noise` needs a fixed
        // shape, matching `expanded_prior_means`'s shape once durations are
        // known. Run once un-noised to learn that shape.
        let baseline = model
            .synthesize(&client, &input_ids, &attention_mask, &opts)
            .expect("baseline synthesize");
        let output_len =
            baseline.waveform.shape()[1] / cfg.upsample_rates.iter().product::<usize>();

        let duration_noise = Tensor::<CpuRuntime>::from_slice(
            &vec![0.0f32; batch * 2 * seq],
            &[batch, 2, seq],
            &device,
        )
        .unwrap();
        let prior_noise = Tensor::<CpuRuntime>::from_slice(
            &vec![0.25f32; batch * cfg.flow_size * output_len],
            &[batch, cfg.flow_size, output_len],
            &device,
        )
        .unwrap();

        let first = model
            .synthesize_with_noise(
                &client,
                &input_ids,
                &attention_mask,
                &opts,
                &duration_noise,
                &prior_noise,
            )
            .expect("first noised synthesize");
        let second = model
            .synthesize_with_noise(
                &client,
                &input_ids,
                &attention_mask,
                &opts,
                &duration_noise,
                &prior_noise,
            )
            .expect("second noised synthesize");

        assert_eq!(
            first.waveform.contiguous().unwrap().to_vec::<f32>(),
            second.waveform.contiguous().unwrap().to_vec::<f32>(),
            "the same noise tensors must reproduce the same waveform bit-for-bit"
        );
    }
}
