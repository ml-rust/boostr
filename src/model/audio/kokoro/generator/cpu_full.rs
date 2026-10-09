use super::core::{IStftNetGenerator, POST_LEAKY_SLOPE};
use crate::error::{Error, Result};
use numr::ops::{ActivationOps, BinaryOps, PadMode, ScalarOps, ShapeOps};

impl IStftNetGenerator<numr::runtime::cpu::CpuRuntime> {
    /// Whether this generator has the per-stage noise-conditioning modules
    /// populated. When `false`, [`Self::forward_cpu_full`] behaves exactly
    /// like the generic `forward`.
    pub fn has_noise_path(&self) -> bool {
        !self.noise_convs.is_empty() && !self.noise_res.is_empty()
    }

    /// The source module's excitation `[B, T_audio]` for a frame-rate F0
    /// contour `[B, T, 1]`: the contour nearest-upsampled by
    /// `f0_upsample_factor`, as upstream's `nn.Upsample` does, then
    /// `m_source`.
    pub fn excitation_cpu(
        &self,
        client: &numr::runtime::cpu::CpuClient,
        f0: &numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
    ) -> Result<numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>> {
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
        excitation
            .reshape(&[exc_shape[0], exc_shape[1]])
            .map_err(Error::Numr)
    }

    /// Compute the harmonic excitation spectrogram `[B, n_fft+2, T]` from a
    /// frame-rate F0 contour: [`excitation_cpu`](Self::excitation_cpu), then
    /// its magnitude and phase concatenated along the channel axis, the shape
    /// upstream's `noise_convs[i]` modules expect.
    pub fn harmonic_excitation_spec_cpu(
        &self,
        client: &numr::runtime::cpu::CpuClient,
        f0: &numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
        n_fft: usize,
        hop_length: usize,
    ) -> Result<numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>> {
        let waveform = self.excitation_cpu(client, f0)?;
        source_spec(client, &waveform, n_fft, hop_length)
    }

    /// The spectrum `[B, n_fft+2, T]` the noise convolutions read from an
    /// excitation `waveform` `[B, T_audio]`: its magnitude and phase at this
    /// generator's STFT, `torch.stft` with `pad_mode='reflect'`.
    pub fn source_spectrum_cpu(
        &self,
        client: &numr::runtime::cpu::CpuClient,
        waveform: &numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
    ) -> Result<numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>> {
        source_spec(client, waveform, self.stft.n_fft, self.stft.hop_length)
    }

    /// CPU forward with the full noise-conditioning path enabled when the
    /// `noise_convs` / `noise_res` fields are populated: the excitation from
    /// `f0` ([`excitation_cpu`](Self::excitation_cpu)), then
    /// [`forward_cpu_with_spectrum`](Self::forward_cpu_with_spectrum). Falls
    /// back to the generic `forward` when they're empty.
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
        let source = self.excitation_cpu(client, f0)?;
        let har = self.source_spectrum_cpu(client, &source)?;
        self.forward_cpu_with_spectrum(client, x, style, &har)
    }

    /// The noise-conditioned generator over a given excitation spectrum
    /// `har` `[B, n_fft+2, T_har]`: everything
    /// [`forward_cpu_full`](Self::forward_cpu_full) does after the source
    /// module and its STFT. The source's draws are random, so a caller that
    /// supplies the spectrum gets a deterministic result. Errors when the
    /// noise path is not loaded.
    ///
    /// Flow:
    ///
    /// ```text
    /// for stage i:
    ///     x = leaky_relu(x)
    ///     x_source = noise_res[i](noise_convs[i](har), style)
    ///     x = ups[i](x)              # last stage: reflect-pad 1 left
    ///     x = x + x_source
    ///     x = mean(resblocks[i*K .. (i+1)*K](x, style))
    /// x = leaky_relu(x, 0.01)
    /// (mag, phase) = conv_post(x)
    /// ```
    #[allow(clippy::type_complexity)]
    pub fn forward_cpu_with_spectrum(
        &self,
        client: &numr::runtime::cpu::CpuClient,
        x: &numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
        style: &numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
        har: &numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
    ) -> Result<(
        numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
        numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
    )> {
        if !self.has_noise_path() {
            return Err(Error::InvalidArgument {
                arg: "har",
                reason: "this generator has no noise path to condition on an excitation; load a checkpoint with noise_convs and noise_res".into(),
            });
        }

        let mut x = x.clone();
        for stage in 0..self.num_upsamples() {
            x = client
                .leaky_relu(&x, self.leaky_slope)
                .map_err(Error::Numr)?;

            // Noise conditioning: har → Conv1d(noise_convs[stage]) →
            // AdaINResBlock1(noise_res[stage]) → residual to add to x.
            let noise_c = self.noise_convs[stage].forward_inference(client, har)?;
            let x_source = self.noise_res[stage].forward(client, &noise_c, style)?;

            x = self.ups[stage].forward(client, &x)?;
            // Reference Kokoro's `ReflectionPad1d((p, 0))` on the last stage,
            // before the excitation add: the trunk gains `p` frames on the
            // left and meets the source's `center=True` frame count.
            let is_last = stage == self.num_upsamples() - 1;
            if is_last && self.last_stage_reflect_pad > 0 {
                x = client
                    .pad_mode(&x, &[self.last_stage_reflect_pad, 0], PadMode::Reflect)
                    .map_err(Error::Numr)?;
            }

            let trunk_t = x.shape()[2];
            let source_t = x_source.shape()[2];
            if source_t != trunk_t {
                return Err(Error::InvalidArgument {
                    arg: "x_source",
                    reason: format!(
                        "noise residual has {source_t} frames and the trunk {trunk_t} at \
                         upsample stage {stage}; check f0_upsample_factor and \
                         last_stage_reflect_pad against the upsample config"
                    ),
                });
            }
            x = client.add(&x, &x_source).map_err(Error::Numr)?;

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
            let summed = xs.ok_or_else(|| Error::InvalidArgument {
                arg: "num_kernels",
                reason: "each upsample stage needs at least one resblock".into(),
            })?;
            x = client
                .mul_scalar(&summed, 1.0 / self.num_kernels as f64)
                .map_err(Error::Numr)?;
        }

        // Upstream's last `F.leaky_relu(x)` takes torch's default slope,
        // not the per-stage one.
        let x = client
            .leaky_relu(&x, POST_LEAKY_SLOPE)
            .map_err(Error::Numr)?;
        self.conv_post.forward(client, &x)
    }
}

/// `waveform`'s magnitude and phase at `n_fft` and `hop_length`,
/// concatenated along the channel axis: upstream `TorchSTFT.transform`,
/// `torch.stft` at its default `pad_mode='reflect'`.
fn source_spec(
    client: &numr::runtime::cpu::CpuClient,
    waveform: &numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
    n_fft: usize,
    hop_length: usize,
) -> Result<numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>> {
    let hann = crate::model::audio::stft::hann_window(n_fft, waveform.device())?;
    let (mag, phase) = crate::model::audio::stft::stft(
        client,
        waveform,
        &hann,
        crate::model::audio::stft::StftOptions {
            n_fft,
            hop_length,
            center: true,
            pad: crate::model::audio::stft::CenterPad::Reflect,
        },
    )?;
    client.cat(&[&mag, &phase], 1).map_err(Error::Numr)
}
