//! `IStftNetGenerator` — Kokoro's spectrogram decoder.
//!
//! Upstream `decoder.generator` composes:
//!
//! ```text
//! m_source  : SourceModuleHnNSF                           (f0 → excitation)
//! f0_upsamp : nn.Upsample                                 (f0 rate-matching)
//! stft      : TorchSTFT(n_fft, hop, win)                  (noise-path spectral analysis)
//! ups[k]    : weight-normed ConvTranspose1d               (main-path upsampling)
//! noise_convs[k] : plain Conv1d                           (harmonic-spectrum conditioning)
//! noise_res[k]   : AdaINResBlock1                         (per-stage noise residuals)
//! resblocks[k * num_kernels + j] : AdaINResBlock1         (per-stage main residuals)
//! conv_post : weight-normed Conv1d → (exp|sin) split      (mag/phase head)
//! ```
//!
//! Forward (simplified to the main path — see `KNOWN LIMITATIONS` below):
//!
//! ```text
//! for each upsample stage i:
//!     x = leaky_relu(x)
//!     x = ups[i](x)
//!     if i == last_stage: reflection_pad(x)
//!     x = x + x_source_i        # see limitation (1)
//!     # average across parallel resblocks
//!     x = mean(resblocks[i*K .. (i+1)*K](x, style))
//! x = leaky_relu(x)
//! (mag, phase) = MagPhaseHead(x)
//! ```
//!
//! # Known limitations (documented, not hidden)
//!
//! 1. **Noise path not yet wired into `forward`.** Upstream passes
//!    `har_source` through a forward STFT, concatenates magnitude + phase
//!    into `[B, n_fft+2, T]`, and feeds that to `noise_convs[i]` followed
//!    by `noise_res[i]`. The STFT primitive itself is now available — see
//!    [`crate::model::audio::stft`] — but wiring it into the generator
//!    requires adding `noise_convs`/`noise_res` fields to the struct,
//!    extending `load_kokoro_v2` for those tensors, and threading the
//!    residual through `Decoder`. That belongs in a focused follow-up
//!    session. For CPU callers wanting the noise path today, call
//!    [`IStftNetGenerator::harmonic_excitation_spec_cpu`] to compute the
//!    excitation spectrogram and add the residual manually.
//! 2. **Reflection padding skipped.** Before the final `+ x_source` add on
//!    the last upsample stage, upstream applies `ReflectionPad1d(3)`.
//!    Deferred until a reflection-pad primitive is ported; the current
//!    build effectively uses zero padding for the last stage.

use crate::error::{Error, Result};
use crate::model::audio::kokoro::{AdaINResBlock1, MagPhaseHead, SourceModuleHnNSF, UpsampleBlock};
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConvOps, MatmulOps, NormalizationOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

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
    /// STFT parameters used by the CPU-specialized noise-path forward.
    pub stft: GeneratorStftParams,
    /// Reflection-pad size applied to `x` on the last upsample stage, just
    /// before `+ x_source`. Matches upstream's `ReflectionPad1d(p)` where
    /// `p = (conv_post_kernel - 1) / 2`. Set to 0 to skip (legacy callers).
    pub last_stage_reflect_pad: usize,
    /// Total audio-rate upsample factor applied to f0 before the noise-path
    /// STFT. Equals `prod(upsample_rates)` — for Kokoro-82M that's
    /// `10 * 6 = 60`. Only used when the noise path is active.
    pub f0_upsample_factor: usize,
}

/// Options controlling how an `IStftNetGenerator` is constructed — gathers
/// all the scalar knobs (kernel count, activation slope, STFT geometry,
/// reflection-pad size) so the constructor signature stays narrow.
#[derive(Debug, Clone, Copy)]
pub struct IStftNetGeneratorOpts {
    pub num_kernels: usize,
    pub leaky_slope: f64,
    pub stft: GeneratorStftParams,
    /// Reflection-pad size for the last upsample stage. Kokoro's upstream
    /// uses `(conv_post_kernel - 1) / 2 = 3` (kernel 7). Set to 0 to skip.
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
            last_stage_reflect_pad: 3,
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

    /// Forward: main trunk `x [B, C, T]` + style `s [B, style_dim]` + frame-
    /// rate F0 `[B, T, 1]` → `(mag [B, F, T_out], phase [B, F, T_out])`.
    ///
    /// Excitation is computed via `m_source(f0)` and returned in the debug
    /// tuple for introspection, but NOT yet injected into the trunk — see the
    /// "Known limitations" section at module level.
    #[allow(clippy::type_complexity)]
    pub fn forward<C>(
        &self,
        client: &C,
        x: &Tensor<R>,
        style: &Tensor<R>,
        f0: &Tensor<R>,
    ) -> Result<(Tensor<R>, Tensor<R>)>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + ConvOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + TensorOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ReduceOps<R>
            + ScalarOps<R>
            + ShapeOps<R>
            + CompareOps<R>
            + TypeConversionOps<R>
            + UtilityOps<R>,
    {
        // Compute excitation (currently unused — see limitation (1)).
        let _excitation = self.m_source.forward(client, f0)?;

        let mut x = x.clone();
        for stage in 0..self.num_upsamples() {
            x = client
                .leaky_relu(&x, self.leaky_slope)
                .map_err(Error::Numr)?;
            x = self.ups[stage].forward(client, &x)?;

            // TODO: add `x_source_i` from noise_convs[stage](har_spec) +
            // noise_res[stage](...) once a forward-STFT primitive lands.

            // Average K parallel resblocks for this stage.
            let mut xs: Option<Tensor<R>> = None;
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

impl IStftNetGenerator<numr::runtime::cpu::CpuRuntime> {
    /// Whether this generator has the per-stage noise-conditioning modules
    /// populated. When `false`, [`Self::forward_cpu_full`] behaves exactly
    /// like the generic `forward`.
    pub fn has_noise_path(&self) -> bool {
        !self.noise_convs.is_empty() && !self.noise_res.is_empty()
    }

    /// Compute the harmonic excitation spectrogram `[B, n_fft+2, T]` from a
    /// frame-rate F0 contour. Concatenates `m_source` output's magnitude and
    /// phase along the channel axis — the shape upstream `noise_convs[i]`
    /// modules expect. Exposed as a building block for callers that want to
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
        let hann = crate::model::audio::kokoro::hann_window(n_fft, f0.device());
        let (mag, phase) = crate::model::audio::stft::stft(
            &waveform,
            &hann,
            crate::model::audio::stft::StftOptions {
                n_fft,
                hop_length,
                center: true,
            },
        )?;
        use numr::ops::ShapeOps;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::kokoro::{KokoroAdaIn1d, PoolParams, SineGen};
    use crate::nn::Conv1d;
    use crate::test_utils::cpu_setup;
    use numr::ops::PaddingMode;
    use numr::runtime::cpu::CpuRuntime;

    fn zeros(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device)
    }
    fn ones(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; n], shape, device)
    }
    fn conv(
        c_out: usize,
        c_in: usize,
        k: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Conv1d<CpuRuntime> {
        Conv1d::new(
            zeros(&[c_out, c_in, k], device),
            Some(zeros(&[c_out], device)),
            1,
            PaddingMode::Same,
            1,
            1,
            false,
        )
    }
    fn adain(
        c: usize,
        s: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> KokoroAdaIn1d<CpuRuntime> {
        KokoroAdaIn1d::new(
            zeros(&[2 * c, s], device),
            zeros(&[2 * c], device),
            ones(&[c], device),
            zeros(&[c], device),
            1e-5,
        )
        .unwrap()
    }

    fn resblock(
        c: usize,
        s: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> AdaINResBlock1<CpuRuntime> {
        AdaINResBlock1::new(
            [
                conv(c, c, 3, device),
                conv(c, c, 3, device),
                conv(c, c, 3, device),
            ],
            [
                conv(c, c, 3, device),
                conv(c, c, 3, device),
                conv(c, c, 3, device),
            ],
            [
                adain(c, s, device),
                adain(c, s, device),
                adain(c, s, device),
            ],
            [
                adain(c, s, device),
                adain(c, s, device),
                adain(c, s, device),
            ],
            [
                ones(&[1, c, 1], device),
                ones(&[1, c, 1], device),
                ones(&[1, c, 1], device),
            ],
            [
                ones(&[1, c, 1], device),
                ones(&[1, c, 1], device),
                ones(&[1, c, 1], device),
            ],
            1e-9,
        )
        .unwrap()
    }

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
                0.1,
            ),
            UpsampleBlock::new(
                zeros(&[4, 4, 1], device),
                None,
                1,
                PaddingMode::Valid,
                0,
                1,
                1,
                0.1,
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
        let (mag, phase) = g.forward(&client, &x, &style, &f0).unwrap();
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
            0.1,
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
