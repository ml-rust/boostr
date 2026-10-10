//! The generator's forward passes on any runtime: the source excitation,
//! its spectrum, and the noise-conditioned upsampling stages.

use super::core::{IStftNetGenerator, POST_LEAKY_SLOPE};
use crate::error::{Error, Result};
use crate::model::audio::kokoro::KokoroClient;
use crate::model::audio::stft::{CenterPad, StftOptions, hann_window, stft};
use numr::dtype::DType;
use numr::ops::PadMode;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

impl<R: Runtime<DType = DType>> IStftNetGenerator<R> {
    /// The source module's excitation `[B, T_audio]` for a frame-rate F0
    /// contour `[B, T, 1]`: the contour nearest-upsampled by
    /// `f0_upsample_factor`, as upstream's `nn.Upsample` does, then
    /// `m_source`. `seed` reads as [`crate::model::audio::kokoro::SineGen::forward`]
    /// reads it.
    pub fn excitation<C>(&self, client: &C, f0: &Tensor<R>, seed: Option<u64>) -> Result<Tensor<R>>
    where
        C: KokoroClient<R>,
    {
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
        let excitation = self.m_source.forward(client, &f0_audio, seed)?;
        let exc_shape = excitation.shape();
        excitation
            .reshape(&[exc_shape[0], exc_shape[1]])
            .map_err(Error::Numr)
    }

    /// The harmonic excitation spectrogram `[B, n_fft+2, T]` of a
    /// frame-rate F0 contour: [`excitation`](Self::excitation), then its
    /// magnitude and phase concatenated along the channel axis, the shape
    /// upstream's `noise_convs[i]` modules expect.
    pub fn harmonic_excitation_spec<C>(
        &self,
        client: &C,
        f0: &Tensor<R>,
        n_fft: usize,
        hop_length: usize,
        seed: Option<u64>,
    ) -> Result<Tensor<R>>
    where
        C: KokoroClient<R>,
    {
        let waveform = self.excitation(client, f0, seed)?;
        source_spec(client, &waveform, n_fft, hop_length)
    }

    /// The spectrum `[B, n_fft+2, T]` the noise convolutions read from an
    /// excitation `waveform` `[B, T_audio]`: its magnitude and phase at this
    /// generator's STFT, `torch.stft` with `pad_mode='reflect'`.
    pub fn source_spectrum<C>(&self, client: &C, waveform: &Tensor<R>) -> Result<Tensor<R>>
    where
        C: KokoroClient<R>,
    {
        source_spec(client, waveform, self.stft.n_fft, self.stft.hop_length)
    }

    /// The full vocoder: trunk `x [B, C, T]` + style `[B, style_dim]` + F0
    /// contour `[B, T_f0, 1]` → `(mag, phase)`, each `[B, F, T_out]`.
    ///
    /// The excitation from `f0` ([`excitation`](Self::excitation)), its
    /// spectrum, then [`forward_with_spectrum`](Self::forward_with_spectrum).
    /// `seed` picks the excitation's PRNG stream. A generator loaded without
    /// `noise_convs` / `noise_res` has no noise path. It runs the upsampling
    /// stages without a source and ignores `f0` and `seed`.
    #[allow(clippy::type_complexity)]
    pub fn forward<C>(
        &self,
        client: &C,
        x: &Tensor<R>,
        style: &Tensor<R>,
        f0: &Tensor<R>,
        seed: Option<u64>,
    ) -> Result<(Tensor<R>, Tensor<R>)>
    where
        C: KokoroClient<R>,
    {
        if !self.has_noise_path() {
            return self.stages(client, x, style, None);
        }
        let source = self.excitation(client, f0, seed)?;
        let har = self.source_spectrum(client, &source)?;
        self.stages(client, x, style, Some(&har))
    }

    /// The noise-conditioned generator over a given excitation spectrum
    /// `har` `[B, n_fft+2, T_har]`: everything [`forward`](Self::forward)
    /// does after the source module and its STFT. It draws no random
    /// number, so a caller that supplies the spectrum gets a deterministic
    /// result. Errors when the noise path is not loaded.
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
    pub fn forward_with_spectrum<C>(
        &self,
        client: &C,
        x: &Tensor<R>,
        style: &Tensor<R>,
        har: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>)>
    where
        C: KokoroClient<R>,
    {
        if !self.has_noise_path() {
            return Err(Error::InvalidArgument {
                arg: "har",
                reason: "this generator has no noise path to condition on an excitation; load a checkpoint with noise_convs and noise_res".into(),
            });
        }
        self.stages(client, x, style, Some(har))
    }

    /// The upsampling stages and `conv_post`. With `har`, each stage adds
    /// its noise residual, and the last stage first reflect-pads the trunk
    /// to the source's frame count. Without `har`, the stages run on the
    /// trunk alone.
    #[allow(clippy::type_complexity)]
    fn stages<C>(
        &self,
        client: &C,
        x: &Tensor<R>,
        style: &Tensor<R>,
        har: Option<&Tensor<R>>,
    ) -> Result<(Tensor<R>, Tensor<R>)>
    where
        C: KokoroClient<R>,
    {
        let mut x = x.clone();
        for stage in 0..self.num_upsamples() {
            x = client
                .leaky_relu(&x, self.leaky_slope)
                .map_err(Error::Numr)?;

            // Noise conditioning: har → Conv1d(noise_convs[stage]) →
            // AdaINResBlock1(noise_res[stage]) → residual to add to x.
            let x_source = match har {
                Some(har) => {
                    let noise_c = self.noise_convs[stage].forward_inference(client, har)?;
                    Some(self.noise_res[stage].forward(client, &noise_c, style)?)
                }
                None => None,
            };

            x = self.ups[stage].forward(client, &x)?;

            if let Some(x_source) = x_source {
                // Reference Kokoro's `ReflectionPad1d((p, 0))` on the last
                // stage, before the excitation add: the trunk gains `p`
                // frames on the left and meets the source's `center=True`
                // frame count.
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
            }

            // Average the K parallel resblocks for this stage.
            let mut xs: Option<Tensor<R>> = None;
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
fn source_spec<R, C>(
    client: &C,
    waveform: &Tensor<R>,
    n_fft: usize,
    hop_length: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: KokoroClient<R>,
{
    let hann = hann_window::<R>(n_fft, waveform.device())?;
    let (mag, phase) = stft(
        client,
        waveform,
        &hann,
        StftOptions {
            n_fft,
            hop_length,
            center: true,
            pad: CenterPad::Reflect,
        },
    )?;
    client.cat(&[&mag, &phase], 1).map_err(Error::Numr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::kokoro::generator::test_support::{resblock, zeros};
    use crate::model::audio::kokoro::{
        GeneratorStftParams, IStftNetGeneratorOpts, MagPhaseHead, SineGen, SourceModuleHnNSF,
        UpsampleBlock,
    };
    use crate::nn::Conv1d;
    use crate::test_utils::cpu_setup;
    use numr::ops::PaddingMode;
    use numr::runtime::cpu::CpuRuntime;

    const STYLE_DIM: usize = 4;

    fn filled(
        shape: &[usize],
        value: f32,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![value; n], shape, device).unwrap()
    }

    fn conv(
        weight: Tensor<CpuRuntime>,
        c_out: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Conv1d<CpuRuntime> {
        Conv1d::new(
            weight,
            Some(zeros(&[c_out], device)),
            1,
            PaddingMode::Same,
            1,
            1,
            false,
        )
    }

    /// One stride-1 upsample stage over 4 channels. The STFT hop equals the
    /// F0 upsample factor, so the source spectrum has one frame more than
    /// the trunk, and the last stage's reflect pad closes the gap. The zero
    /// trunk weights leave the output a function of the source alone.
    fn tiny_generator(
        with_noise: bool,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> IStftNetGenerator<CpuRuntime> {
        let n_fft = 4;
        let ups = vec![UpsampleBlock::new(
            zeros(&[4, 4, 1], device),
            None,
            1,
            PaddingMode::Valid,
            0,
            1,
            1,
            None,
        )];
        let source = SourceModuleHnNSF::new(
            SineGen::new(24_000.0, 1),
            filled(&[1, 2], 1.0, device),
            zeros(&[1], device),
        )
        .unwrap();
        let (noise_convs, noise_res) = if with_noise {
            (
                vec![conv(filled(&[4, n_fft + 2, 1], 1.0, device), 4, device)],
                vec![resblock(4, STYLE_DIM, device)],
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let head = conv(filled(&[n_fft + 2, 4, 3], 0.01, device), n_fft + 2, device);
        IStftNetGenerator::new(
            source,
            ups,
            vec![resblock(4, STYLE_DIM, device)],
            noise_convs,
            noise_res,
            MagPhaseHead::new(head, n_fft).unwrap(),
            IStftNetGeneratorOpts {
                num_kernels: 1,
                leaky_slope: 0.1,
                stft: GeneratorStftParams {
                    n_fft,
                    hop_length: 2,
                },
                last_stage_reflect_pad: 1,
                f0_upsample_factor: 2,
            },
        )
        .unwrap()
    }

    fn run(
        g: &IStftNetGenerator<CpuRuntime>,
        seed: Option<u64>,
    ) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
        let (client, device) = cpu_setup();
        let x = zeros(&[1, 4, 3], &device);
        let style = zeros(&[1, STYLE_DIM], &device);
        let f0 = filled(&[1, 3, 1], 200.0, &device);
        let (mag, phase) = g.forward(&client, &x, &style, &f0, seed).unwrap();
        (mag.to_vec(), phase.to_vec(), mag.shape().to_vec())
    }

    #[test]
    fn the_noise_path_meets_the_padded_trunk() {
        let (_client, device) = cpu_setup();
        let g = tiny_generator(true, &device);
        let (mag, _, shape) = run(&g, Some(1));
        // Three trunk frames plus the one reflected frame.
        assert_eq!(shape, vec![1, 3, 4]);
        assert!(mag.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn one_seed_reproduces_the_output_and_another_changes_it() {
        let (_client, device) = cpu_setup();
        let g = tiny_generator(true, &device);
        assert_eq!(run(&g, Some(3)), run(&g, Some(3)));
        assert_ne!(run(&g, Some(3)).0, run(&g, Some(4)).0);
    }

    #[test]
    fn without_the_noise_path_the_seed_changes_nothing() {
        let (_client, device) = cpu_setup();
        let g = tiny_generator(false, &device);
        assert!(!g.has_noise_path());
        assert_eq!(run(&g, Some(3)), run(&g, Some(4)));
        // No source, no reflect pad: the trunk keeps its three frames.
        assert_eq!(run(&g, None).2, vec![1, 3, 3]);
    }

    #[test]
    fn forward_with_spectrum_needs_the_noise_path() {
        let (client, device) = cpu_setup();
        let g = tiny_generator(false, &device);
        let x = zeros(&[1, 4, 3], &device);
        let style = zeros(&[1, STYLE_DIM], &device);
        let har = zeros(&[1, 6, 4], &device);
        assert!(g.forward_with_spectrum(&client, &x, &style, &har).is_err());
    }
}
