use super::core::IStftNetGenerator;
use crate::error::{Error, Result};
use numr::ops::{ActivationOps, BinaryOps, ScalarOps, ShapeOps};

impl IStftNetGenerator<numr::runtime::cpu::CpuRuntime> {
    /// Whether this generator has the per-stage noise-conditioning modules
    /// populated. When `false`, [`Self::forward_cpu_full`] behaves exactly
    /// like the generic `forward`.
    pub fn has_noise_path(&self) -> bool {
        !self.noise_convs.is_empty() && !self.noise_res.is_empty()
    }

    /// Compute the harmonic excitation spectrogram `[B, n_fft+2, T]` from a
    /// frame-rate F0 contour. Concatenates `m_source` output's magnitude and
    /// phase along the channel axis — the shape the reference Kokoro
    /// implementation's `noise_convs[i]` modules expect. Exposed as a building block for callers that want to
    /// inspect or reuse the excitation without running the full forward.
    pub fn harmonic_excitation_spec_cpu(
        &self,
        client: &numr::runtime::cpu::CpuClient,
        f0: &numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
        n_fft: usize,
        hop_length: usize,
    ) -> Result<numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>> {
        // Upsample f0 from frame rate to audio rate before the source module.
        // Upstream `nn.Upsample(scale_factor=60)` operates on `[B, 1, T_f0]`
        // with default nearest-neighbor interpolation; we do the same via a
        // reshape-broadcast trick matching `nearest_upsample_1d` in
        // `adain_resblk1d.rs`.
        let f0_shape = f0.shape();
        if f0_shape.len() != 3 || f0_shape[2] != 1 {
            return Err(Error::InvalidArgument {
                arg: "f0",
                reason: format!("expected [B, T, 1], got {f0_shape:?}"),
            });
        }
        let scale = self.f0_upsample_factor.max(1);
        let (b, t) = (f0_shape[0], f0_shape[1]);
        let f0_audio = if scale == 1 {
            f0.clone()
        } else {
            // [B, T, 1] -> [B, T, scale] (broadcast) -> [B, T*scale, 1].
            f0.broadcast_to(&[b, t, scale])
                .map_err(Error::Numr)?
                .contiguous()?
                .reshape(&[b, t * scale, 1])
                .map_err(Error::Numr)?
        };

        let excitation = self.m_source.forward(client, &f0_audio)?;
        let exc_shape = excitation.shape();
        let (bb, t_audio) = (exc_shape[0], exc_shape[1]);
        let waveform = excitation.reshape(&[bb, t_audio]).map_err(Error::Numr)?;
        let hann = crate::model::audio::stft::hann_window(n_fft, f0.device())?;
        let (mag, phase) = crate::model::audio::stft::stft(
            client,
            &waveform,
            &hann,
            crate::model::audio::stft::StftOptions {
                n_fft,
                hop_length,
                center: true,
            },
        )?;
        client.cat(&[&mag, &phase], 1).map_err(Error::Numr)
    }

    /// CPU forward with the full noise-conditioning path enabled when the
    /// `noise_convs` / `noise_res` fields are populated. Falls back to the
    /// same behavior as generic `forward` when they're empty, so existing
    /// callers are unaffected until they attach noise weights.
    ///
    /// Flow (noise path active):
    ///
    /// ```text
    /// har = harmonic_excitation_spec_cpu(f0)        [B, n_fft+2, T_har]
    /// for stage i:
    ///     x = leaky_relu(x)
    ///     x_source = noise_res[i](noise_convs[i](har), style)
    ///     x = ups[i](x) + x_source
    ///     x = mean(resblocks[i*K .. (i+1)*K](x, style))
    /// x = leaky_relu(x)
    /// (mag, phase) = conv_post(x)
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn forward_cpu_full(
        &self,
        client: &numr::runtime::cpu::CpuClient,
        x: &numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
        style: &numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
        f0: &numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
    ) -> Result<(
        numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
        numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
    )> {
        if !self.has_noise_path() {
            return self.forward(client, x, style, f0);
        }

        let har =
            self.harmonic_excitation_spec_cpu(client, f0, self.stft.n_fft, self.stft.hop_length)?;

        let mut x = x.clone();
        for stage in 0..self.num_upsamples() {
            x = client
                .leaky_relu(&x, self.leaky_slope)
                .map_err(Error::Numr)?;

            // Noise conditioning: har → Conv1d(noise_convs[stage]) →
            // AdaINResBlock1(noise_res[stage]) → residual to add to x.
            let noise_c = self.noise_convs[stage].forward_inference(client, &har)?;
            let x_source = self.noise_res[stage].forward(client, &noise_c, style)?;

            x = self.ups[stage].forward(client, &x)?;

            // Crop x_source to trunk length. STFT `center=True` yields one
            // extra spec frame vs the T_latent * total_upsample trunk rate
            // (e.g. T_spec = 1321 vs trunk = 1320 at the final stage for
            // Kokoro defaults). Trim from the right — the trailing frame
            // corresponds to the reflected-padding tail and carries no real
            // information.
            let trunk_t = x.shape()[2];
            let source_t = x_source.shape()[2];
            let x_source = if source_t > trunk_t {
                x_source
                    .narrow(2, 0, trunk_t)
                    .map_err(Error::Numr)?
                    .contiguous()?
            } else if source_t < trunk_t {
                return Err(Error::InvalidArgument {
                    arg: "x_source",
                    reason: format!(
                        "noise residual is shorter ({source_t}) than trunk ({trunk_t}); \
                         check f0_upsample_factor vs upsample_ratios config"
                    ),
                });
            } else {
                x_source
            };

            x = client.add(&x, &x_source).map_err(Error::Numr)?;
            // Reflection-pad on the last stage AFTER the excitation add so
            // noise_convs output shape alignment matches, and the extended
            // context feeds the final `leaky_relu → conv_post` head that
            // reduces back to trunk length via same-padding.
            let is_last = stage == self.num_upsamples() - 1;
            if is_last && self.last_stage_reflect_pad > 0 {
                x = crate::model::audio::reflection_pad::reflection_pad_1d(
                    &x,
                    self.last_stage_reflect_pad,
                    self.last_stage_reflect_pad,
                )?;
            }

            // Average the K parallel resblocks for this stage.
            let mut xs: Option<numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>> = None;
            for k in 0..self.num_kernels {
                let idx = stage * self.num_kernels + k;
                let out = self.resblocks[idx].forward(client, &x, style)?;
                xs = Some(match xs {
                    None => out,
                    Some(prev) => client.add(&prev, &out).map_err(Error::Numr)?,
                });
            }
            let summed = xs.expect("at least one resblock per stage — validated in new()");
            x = client
                .mul_scalar(&summed, 1.0 / self.num_kernels as f64)
                .map_err(Error::Numr)?;
        }

        let x = client
            .leaky_relu(&x, self.leaky_slope)
            .map_err(Error::Numr)?;
        self.conv_post.forward(client, &x)
    }
}
