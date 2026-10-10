use crate::error::{Error, Result};
use crate::model::audio::kokoro::{AdaINResBlock1, MagPhaseHead, SourceModuleHnNSF, UpsampleBlock};
use numr::runtime::Runtime;

/// Parameters controlling the STFT used for noise-path conditioning.
#[derive(Debug, Clone, Copy)]
pub struct GeneratorStftParams {
    pub n_fft: usize,
    pub hop_length: usize,
}

pub struct IStftNetGenerator<R: Runtime> {
    pub m_source: SourceModuleHnNSF<R>,
    /// One `UpsampleBlock` per stage. Length = `num_upsamples`.
    pub ups: Vec<UpsampleBlock<R>>,
    /// Flat-indexed resblocks: `resblocks[stage * num_kernels + k]`. Length
    /// must equal `num_upsamples * num_kernels`.
    pub resblocks: Vec<AdaINResBlock1<R>>,
    /// Plain `Conv1d` (no weight-norm) that maps the concatenated
    /// `[mag | phase]` excitation spectrogram `[B, n_fft+2, T_spec]` into
    /// each stage's channel width. Optional — when empty the noise path is
    /// skipped. When populated, length must equal `num_upsamples`.
    pub noise_convs: Vec<crate::nn::Conv1d<R>>,
    /// Per-stage `AdaINResBlock1` applied to `noise_convs[i]`'s output.
    /// Same optionality + length requirement as `noise_convs`.
    pub noise_res: Vec<AdaINResBlock1<R>>,
    pub conv_post: MagPhaseHead<R>,
    pub num_kernels: usize,
    pub leaky_slope: f64,
    /// STFT parameters of the excitation spectrum the noise path reads.
    pub stft: GeneratorStftParams,
    /// Frames of reflection padding added on the LEFT of `x` on the last
    /// upsample stage, just before `+ x_source`: reference Kokoro's
    /// `ReflectionPad1d((p, 0))`, `p = 1`. Set to 0 to skip.
    pub last_stage_reflect_pad: usize,
    /// Total audio-rate upsample factor applied to f0 before the noise-path
    /// STFT. Equals `prod(upsample_rates)` — for Kokoro-82M that's
    /// `10 * 6 = 60`. Only used when the noise path is active.
    pub f0_upsample_factor: usize,
}

/// The slope of the leaky ReLU before `conv_post`: `F.leaky_relu`'s
/// default, which upstream's generator calls without one.
pub(super) const POST_LEAKY_SLOPE: f64 = 0.01;

/// Options controlling how an `IStftNetGenerator` is constructed — gathers
/// all the scalar knobs (kernel count, activation slope, STFT geometry,
/// reflection-pad size) so the constructor signature stays narrow.
#[derive(Debug, Clone, Copy)]
pub struct IStftNetGeneratorOpts {
    pub num_kernels: usize,
    pub leaky_slope: f64,
    pub stft: GeneratorStftParams,
    /// Left reflection-pad size for the last upsample stage. Reference
    /// Kokoro uses `ReflectionPad1d((1, 0))`. Set to 0 to skip.
    pub last_stage_reflect_pad: usize,
    /// Audio-rate upsample factor for the noise path (see struct field).
    pub f0_upsample_factor: usize,
}

impl Default for IStftNetGeneratorOpts {
    fn default() -> Self {
        Self {
            num_kernels: 3,
            leaky_slope: 0.1,
            stft: GeneratorStftParams {
                n_fft: 20,
                hop_length: 5,
            },
            last_stage_reflect_pad: 1,
            f0_upsample_factor: 60,
        }
    }
}

impl<R: Runtime> IStftNetGenerator<R> {
    pub fn new(
        m_source: SourceModuleHnNSF<R>,
        ups: Vec<UpsampleBlock<R>>,
        resblocks: Vec<AdaINResBlock1<R>>,
        noise_convs: Vec<crate::nn::Conv1d<R>>,
        noise_res: Vec<AdaINResBlock1<R>>,
        conv_post: MagPhaseHead<R>,
        opts: IStftNetGeneratorOpts,
    ) -> Result<Self> {
        if ups.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "ups",
                reason: "must have at least one upsample stage".into(),
            });
        }
        if opts.num_kernels == 0 {
            return Err(Error::InvalidArgument {
                arg: "opts.num_kernels",
                reason: "must be > 0".into(),
            });
        }
        if resblocks.len() != ups.len() * opts.num_kernels {
            return Err(Error::InvalidArgument {
                arg: "resblocks",
                reason: format!(
                    "expected {} resblocks (num_upsamples {} * num_kernels {}), got {}",
                    ups.len() * opts.num_kernels,
                    ups.len(),
                    opts.num_kernels,
                    resblocks.len()
                ),
            });
        }
        // Noise fields are both-or-neither; partial population is a bug.
        match (noise_convs.len(), noise_res.len()) {
            (0, 0) => {}
            (a, b) if a == ups.len() && b == ups.len() => {}
            (a, b) => {
                return Err(Error::InvalidArgument {
                    arg: "noise_convs / noise_res",
                    reason: format!(
                        "must both be empty OR both match num_upsamples ({}); got ({a}, {b})",
                        ups.len()
                    ),
                });
            }
        }
        if opts.stft.n_fft == 0 || opts.stft.hop_length == 0 {
            return Err(Error::InvalidArgument {
                arg: "opts.stft",
                reason: "n_fft and hop_length must be > 0".into(),
            });
        }
        Ok(Self {
            m_source,
            ups,
            resblocks,
            noise_convs,
            noise_res,
            conv_post,
            num_kernels: opts.num_kernels,
            leaky_slope: opts.leaky_slope,
            stft: opts.stft,
            last_stage_reflect_pad: opts.last_stage_reflect_pad,
            f0_upsample_factor: opts.f0_upsample_factor,
        })
    }

    pub fn num_upsamples(&self) -> usize {
        self.ups.len()
    }

    /// Whether this generator has the per-stage noise-conditioning modules
    /// populated. When `false`, [`Self::forward`] runs the upsampling
    /// stages without a source.
    pub fn has_noise_path(&self) -> bool {
        !self.noise_convs.is_empty() && !self.noise_res.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::kokoro::generator::test_support::{conv, resblock, zeros};
    use crate::model::audio::kokoro::{PoolParams, SineGen};
    use crate::test_utils::cpu_setup;
    use numr::ops::PaddingMode;
    use numr::runtime::cpu::CpuRuntime;

    fn build_tiny_generator(
        device: &<CpuRuntime as Runtime>::Device,
    ) -> IStftNetGenerator<CpuRuntime> {
        let style_dim = 4;
        let n_fft = 4;
        // Two upsample stages: C_in=8 → C_mid=4 (stride 2), then C_mid=4 → C_out=4 (stride 1).
        let ups = vec![
            UpsampleBlock::new(
                zeros(&[8, 4, 2], device), // ConvTranspose1d weight [C_in, C_out, K]
                None,
                2,
                PaddingMode::Valid,
                0,
                1,
                1,
                None,
            ),
            UpsampleBlock::new(
                zeros(&[4, 4, 1], device),
                None,
                1,
                PaddingMode::Valid,
                0,
                1,
                1,
                None,
            ),
        ];
        // 2 stages × 2 kernels = 4 resblocks. Stage 0 ch=4, stage 1 ch=4.
        let resblocks = vec![
            resblock(4, style_dim, device),
            resblock(4, style_dim, device),
            resblock(4, style_dim, device),
            resblock(4, style_dim, device),
        ];
        let source = SourceModuleHnNSF::new(
            SineGen::new(24_000.0, 1),
            zeros(&[1, 2], device),
            zeros(&[1], device),
        )
        .unwrap();
        let mag_phase = MagPhaseHead::new(conv(2 * (n_fft / 2 + 1), 4, 3, device), n_fft).unwrap();

        IStftNetGenerator::new(
            source,
            ups,
            resblocks,
            Vec::new(),
            Vec::new(),
            mag_phase,
            IStftNetGeneratorOpts {
                num_kernels: 2,
                last_stage_reflect_pad: 0,
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[test]
    fn forward_returns_mag_phase_shapes() {
        let (client, device) = cpu_setup();
        let g = build_tiny_generator(&device);
        let x = zeros(&[1, 8, 3], &device);
        let style = zeros(&[1, 4], &device);
        let f0 = zeros(&[1, 3, 1], &device);
        let (mag, phase) = g.forward(&client, &x, &style, &f0, None).unwrap();
        // First ups stride=2, kernel=2 on T=3: L_out = (3-1)*2 + 2 = 6.
        // Second ups stride=1, kernel=1 on T=6: L_out = 6.
        assert_eq!(mag.shape(), &[1, 3, 6]); // n_fft/2+1 = 3
        assert_eq!(phase.shape(), &[1, 3, 6]);
    }

    #[test]
    fn new_rejects_wrong_resblock_count() {
        let (_client, device) = cpu_setup();
        let ups = vec![UpsampleBlock::new(
            zeros(&[4, 4, 1], &device),
            None,
            1,
            PaddingMode::Valid,
            0,
            1,
            1,
            None,
        )];
        // 1 stage × 3 kernels = 3 expected, give 2.
        let resblocks = vec![resblock(4, 2, &device), resblock(4, 2, &device)];
        let source = SourceModuleHnNSF::new(
            SineGen::new(24_000.0, 1),
            zeros(&[1, 2], &device),
            zeros(&[1], &device),
        )
        .unwrap();
        let mag_phase = MagPhaseHead::new(conv(6, 4, 3, &device), 4).unwrap();
        let bad = IStftNetGenerator::new(
            source,
            ups,
            resblocks,
            Vec::new(),
            Vec::new(),
            mag_phase,
            IStftNetGeneratorOpts {
                num_kernels: 3,
                last_stage_reflect_pad: 0,
                ..Default::default()
            },
        );
        assert!(bad.is_err());
    }

    #[test]
    fn new_rejects_empty_ups() {
        let (_client, device) = cpu_setup();
        let source = SourceModuleHnNSF::new(
            SineGen::new(24_000.0, 1),
            zeros(&[1, 2], &device),
            zeros(&[1], &device),
        )
        .unwrap();
        let mag_phase = MagPhaseHead::new(conv(6, 4, 3, &device), 4).unwrap();
        let bad = IStftNetGenerator::new(
            source,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
            mag_phase,
            IStftNetGeneratorOpts {
                num_kernels: 1,
                last_stage_reflect_pad: 0,
                ..Default::default()
            },
        );
        assert!(bad.is_err());
    }

    // Silence the unused-import warning for types that are useful to pub-use
    // but not touched in this file's tests.
    #[test]
    fn _pool_params_type_is_in_scope() {
        let _: Option<PoolParams<CpuRuntime>> = None;
    }
}
