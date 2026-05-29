//! Kokoro top-level decoder.
//!
//! Wires the ASR features + pitch curve + energy curve + style into the
//! ISTFTNet generator. Upstream checkpoint layout (from `hexgrad/kokoro`):
//!
//! ```text
//! decoder
//! ├── asr_res   : Sequential[Conv1d(512, 64, k=1)]                    # asr projection
//! ├── F0_conv   : weight-normed Conv1d(1, 1, k=3, s=2)                # pitch conditioning
//! ├── N_conv    : weight-normed Conv1d(1, 1, k=3, s=2)                # energy conditioning
//! ├── encode    : AdainResBlk1d(66  -> 1024, learned_sc, no pool)      # dim_in+2 → 1024
//! ├── decode.0  : AdainResBlk1d(1090 -> 1024, learned_sc, no pool)    # 1024 + 2 + 64
//! ├── decode.1  : AdainResBlk1d(1090 -> 1024, learned_sc, no pool)
//! ├── decode.2  : AdainResBlk1d(1090 -> 1024, learned_sc, no pool)
//! ├── decode.3  : AdainResBlk1d(1090 -> 512,  learned_sc, pool)       # last stage upsamples
//! └── generator : IStftNetGenerator                                    # → (mag, phase)
//! ```
//!
//! Forward (composition):
//!
//! ```text
//! 1. f0  = F0_conv(F0_curve.unsqueeze(1))    [B, 1, T/2]
//!    n   = N_conv(N_curve.unsqueeze(1))      [B, 1, T/2]
//!    asr = asr_res(asr_feats)                [B, 64, T]
//!
//! 2. x = cat([asr, f0, n], dim=1)            [B, 66, T']  (align T' via conditioning down-rate)
//!    x = encode(x, style)                     [B, 1024, T']
//!
//! 3. for block in decode[0..3]:
//!        x = cat([x, f0, n, asr], dim=1)     [B, 1090, T']
//!        x = block(x, style)                  [B, 1024, T'] (block 0..2) or [B, 512, 2·T'] (block 3)
//!
//! 4. return generator(x, style, F0_curve) → (mag, phase)
//! ```
//!
//! The exact time-axis alignment (how T/2, T, T' relate) depends on the real
//! checkpoint's upsample recipe; this assembly validates shapes at forward
//! time and surfaces precise error messages when the caller's inputs don't
//! line up.

use crate::error::{Error, Result};
use crate::model::audio::kokoro::{AdainResBlk1d, IStftNetGenerator};
use crate::nn::Conv1d;
use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConvOps, MatmulOps, NormalizationOps, ReduceOps,
    ScalarOps, ShapeOps, TensorOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub struct Decoder<R: Runtime> {
    /// `Conv1d(asr_c_in, asr_proj_c_out, k=1)` — projects ASR channels down.
    pub asr_res: Conv1d<R>,
    /// Stride-2 Conv1d for pitch curve.
    pub f0_conv: Conv1d<R>,
    /// Stride-2 Conv1d for energy curve.
    pub n_conv: Conv1d<R>,
    /// Initial encoder block: `(asr_proj + 2) → 1024`.
    pub encode: AdainResBlk1d<R>,
    /// 4 decode blocks; first 3 are 1024→1024, last upsamples to 512.
    pub decode: Vec<AdainResBlk1d<R>>,
    /// ISTFTNet generator (returns mag + phase).
    pub generator: IStftNetGenerator<R>,
}

impl<R: Runtime> Decoder<R> {
    pub fn new(
        asr_res: Conv1d<R>,
        f0_conv: Conv1d<R>,
        n_conv: Conv1d<R>,
        encode: AdainResBlk1d<R>,
        decode: Vec<AdainResBlk1d<R>>,
        generator: IStftNetGenerator<R>,
    ) -> Result<Self> {
        if decode.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "decode",
                reason: "must have at least one decode block".into(),
            });
        }
        Ok(Self {
            asr_res,
            f0_conv,
            n_conv,
            encode,
            decode,
            generator,
        })
    }

    /// Forward: full phonemes-to-(mag, phase) path.
    ///
    /// * `asr_feats [B, C_asr, T]` — ASR-path features (raw, pre-projection).
    /// * `f0_curve [B, T_f0]` — predicted pitch, frame-rate (matches duration-
    ///   expanded phoneme count).
    /// * `n_curve [B, T_f0]` — predicted energy, same time axis as `f0_curve`.
    /// * `style [B, style_dim]` — decoder-half style vector (128-d for Kokoro).
    ///
    /// Returns `(mag, phase)` ready for `istft`.
    #[allow(clippy::type_complexity)]
    pub fn forward<C>(
        &self,
        client: &C,
        asr_feats: &Tensor<R>,
        f0_curve: &Tensor<R>,
        n_curve: &Tensor<R>,
        style: &Tensor<R>,
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
        let f0_shape = f0_curve.shape();
        let n_shape = n_curve.shape();
        if f0_shape != n_shape || f0_shape.len() != 2 {
            return Err(Error::InvalidArgument {
                arg: "f0_curve/n_curve",
                reason: format!(
                    "both must be [B, T_f0] with matching shapes, got {f0_shape:?} vs {n_shape:?}"
                ),
            });
        }
        let (b, t_f0) = (f0_shape[0], f0_shape[1]);

        // Promote F0/N to [B, 1, T_f0] before the stride-2 conv.
        let f0 = f0_curve.reshape(&[b, 1, t_f0]).map_err(Error::Numr)?;
        let n = n_curve.reshape(&[b, 1, t_f0]).map_err(Error::Numr)?;
        let f0 = self.f0_conv.forward_inference(client, &f0)?;
        let n = self.n_conv.forward_inference(client, &n)?;

        // Encode takes the RAW asr features (512 ch) + F0 + N → 514 channel input.
        // asr_res is a SEPARATE projection (512 → 64) used only in the decode
        // stages, not in encode. This matches upstream's dataflow.
        let x = client.cat(&[asr_feats, &f0, &n], 1).map_err(Error::Numr)?;
        let mut x = self.encode.forward(client, &x, style)?;

        let asr_res_proj = self.asr_res.forward_inference(client, asr_feats)?;

        // Each decode block re-concatenates `[x, F0, N, asr_res]`.
        for block in &self.decode {
            let cat = client
                .cat(&[&x, &f0, &n, &asr_res_proj], 1)
                .map_err(Error::Numr)?;
            x = block.forward(client, &cat, style)?;
        }

        // Generator takes the original frame-rate f0_curve for source-filter
        // excitation (not the strided f0 tensor). Promote to [B, T_f0, 1].
        let f0_for_gen = f0_curve.reshape(&[b, t_f0, 1]).map_err(Error::Numr)?;
        self.generator.forward(client, &x, style, &f0_for_gen)
    }
}

impl Decoder<numr::runtime::cpu::CpuRuntime> {
    /// CPU forward that uses the generator's full noise-conditioned path
    /// (`forward_cpu_full`) when the checkpoint supplied `noise_convs` /
    /// `noise_res`. Falls back to the noise-free `forward` otherwise.
    #[allow(clippy::type_complexity)]
    pub fn forward_cpu_full(
        &self,
        client: &numr::runtime::cpu::CpuClient,
        asr_feats: &numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
        f0_curve: &numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
        n_curve: &numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
        style: &numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
    ) -> Result<(
        numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
        numr::tensor::Tensor<numr::runtime::cpu::CpuRuntime>,
    )> {
        let f0_shape = f0_curve.shape();
        let n_shape = n_curve.shape();
        if f0_shape != n_shape || f0_shape.len() != 2 {
            return Err(Error::InvalidArgument {
                arg: "f0_curve/n_curve",
                reason: format!(
                    "both must be [B, T_f0] with matching shapes, got {f0_shape:?} vs {n_shape:?}"
                ),
            });
        }
        let (b, t_f0) = (f0_shape[0], f0_shape[1]);

        let f0 = f0_curve.reshape(&[b, 1, t_f0]).map_err(Error::Numr)?;
        let n = n_curve.reshape(&[b, 1, t_f0]).map_err(Error::Numr)?;
        let f0 = self.f0_conv.forward_inference(client, &f0)?;
        let n = self.n_conv.forward_inference(client, &n)?;
        let x = client.cat(&[asr_feats, &f0, &n], 1).map_err(Error::Numr)?;
        let mut x = self.encode.forward(client, &x, style)?;
        let asr_res_proj = self.asr_res.forward_inference(client, asr_feats)?;
        for block in &self.decode {
            let cat = client
                .cat(&[&x, &f0, &n, &asr_res_proj], 1)
                .map_err(Error::Numr)?;
            x = block.forward(client, &cat, style)?;
        }
        let f0_for_gen = f0_curve.reshape(&[b, t_f0, 1]).map_err(Error::Numr)?;
        self.generator
            .forward_cpu_full(client, &x, style, &f0_for_gen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::kokoro::AdaINResBlock1;
    use crate::model::audio::kokoro::{
        KokoroAdaIn1d, MagPhaseHead, SineGen, SourceModuleHnNSF, UpsampleBlock,
    };
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
        stride: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Conv1d<CpuRuntime> {
        Conv1d::new(
            zeros(&[c_out, c_in, k], device),
            Some(zeros(&[c_out], device)),
            stride,
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
    fn resblk1d(
        c_in: usize,
        c_out: usize,
        s: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> AdainResBlk1d<CpuRuntime> {
        let needs_shortcut = c_in != c_out;
        AdainResBlk1d::new(
            adain(c_in, s, device),
            adain(c_out, s, device),
            conv(c_out, c_in, 3, 1, device),
            conv(c_out, c_out, 3, 1, device),
            if needs_shortcut {
                Some(conv(c_out, c_in, 1, 1, device))
            } else {
                None
            },
            None,
            0.2,
        )
    }
    fn resblock1(
        c: usize,
        s: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> AdaINResBlock1<CpuRuntime> {
        AdaINResBlock1::new(
            [
                conv(c, c, 3, 1, device),
                conv(c, c, 3, 1, device),
                conv(c, c, 3, 1, device),
            ],
            [
                conv(c, c, 3, 1, device),
                conv(c, c, 3, 1, device),
                conv(c, c, 3, 1, device),
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

    #[test]
    fn forward_returns_mag_phase() {
        let (client, device) = cpu_setup();
        let style_dim = 4;
        // Channel plan (matches upstream Kokoro dataflow):
        //   asr_c_in = 8, F0/N each 1 channel.
        //   Encode input: asr(8) + F0(1) + N(1) = 10 → 8.
        //   asr_res: 8 → 4 (used only in decode).
        //   Decode input: x(8) + F0(1) + N(1) + asr_res(4) = 14 → 8 (first), → 4 (last).
        let asr_res = conv(4, 8, 1, 1, &device);
        let f0_conv = conv(1, 1, 3, 1, &device); // stride=1 for test simplicity
        let n_conv = conv(1, 1, 3, 1, &device);
        let encode = resblk1d(10, 8, style_dim, &device);
        let decode = vec![
            resblk1d(14, 8, style_dim, &device),
            resblk1d(14, 4, style_dim, &device), // last stage projects to 4 for generator input
        ];
        let n_fft = 4;
        let source = SourceModuleHnNSF::new(
            SineGen::new(24_000.0, 1),
            zeros(&[1, 2], &device),
            zeros(&[1], &device),
        )
        .unwrap();
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
        let resblocks = vec![
            resblock1(4, style_dim, &device),
            resblock1(4, style_dim, &device),
        ];
        let mag_phase = MagPhaseHead::new(conv(6, 4, 3, 1, &device), n_fft).unwrap();
        let generator = IStftNetGenerator::new(
            source,
            ups,
            resblocks,
            Vec::new(),
            Vec::new(),
            mag_phase,
            crate::model::audio::kokoro::IStftNetGeneratorOpts {
                num_kernels: 2,
                last_stage_reflect_pad: 0,
                ..Default::default()
            },
        )
        .unwrap();

        let decoder = Decoder::new(asr_res, f0_conv, n_conv, encode, decode, generator).unwrap();

        let t = 5;
        let asr = zeros(&[1, 8, t], &device);
        let f0 = zeros(&[1, t], &device);
        let ne = zeros(&[1, t], &device);
        let style = zeros(&[1, style_dim], &device);
        let (mag, phase) = decoder.forward(&client, &asr, &f0, &ne, &style).unwrap();
        // n_fft/2 + 1 = 3 magnitude/phase bins. Time axis matches the single
        // ups stage's output (stride=1 → same as input = 5).
        assert_eq!(mag.shape(), &[1, 3, 5]);
        assert_eq!(phase.shape(), &[1, 3, 5]);
    }

    #[test]
    fn new_rejects_empty_decode() {
        let (_client, device) = cpu_setup();
        let asr_res = conv(4, 8, 1, 1, &device);
        let f0_conv = conv(1, 1, 3, 1, &device);
        let n_conv = conv(1, 1, 3, 1, &device);
        let encode = resblk1d(6, 8, 2, &device);
        let source = SourceModuleHnNSF::new(
            SineGen::new(24_000.0, 1),
            zeros(&[1, 2], &device),
            zeros(&[1], &device),
        )
        .unwrap();
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
        let resblocks = vec![resblock1(4, 2, &device)];
        let mag_phase = MagPhaseHead::new(conv(6, 4, 3, 1, &device), 4).unwrap();
        let generator = IStftNetGenerator::new(
            source,
            ups,
            resblocks,
            Vec::new(),
            Vec::new(),
            mag_phase,
            crate::model::audio::kokoro::IStftNetGeneratorOpts {
                num_kernels: 1,
                last_stage_reflect_pad: 0,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(Decoder::new(asr_res, f0_conv, n_conv, encode, Vec::new(), generator).is_err());
    }

    #[test]
    fn forward_rejects_mismatched_f0_n_shapes() {
        let (client, device) = cpu_setup();
        let asr_res = conv(4, 8, 1, 1, &device);
        let f0_conv = conv(1, 1, 3, 1, &device);
        let n_conv = conv(1, 1, 3, 1, &device);
        let encode = resblk1d(6, 8, 2, &device);
        let decode = vec![resblk1d(14, 4, 2, &device)];
        let source = SourceModuleHnNSF::new(
            SineGen::new(24_000.0, 1),
            zeros(&[1, 2], &device),
            zeros(&[1], &device),
        )
        .unwrap();
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
        let resblocks = vec![resblock1(4, 2, &device)];
        let mag_phase = MagPhaseHead::new(conv(6, 4, 3, 1, &device), 4).unwrap();
        let generator = IStftNetGenerator::new(
            source,
            ups,
            resblocks,
            Vec::new(),
            Vec::new(),
            mag_phase,
            crate::model::audio::kokoro::IStftNetGeneratorOpts {
                num_kernels: 1,
                last_stage_reflect_pad: 0,
                ..Default::default()
            },
        )
        .unwrap();
        let decoder = Decoder::new(asr_res, f0_conv, n_conv, encode, decode, generator).unwrap();

        let asr = zeros(&[1, 8, 5], &device);
        let f0 = zeros(&[1, 5], &device);
        let n = zeros(&[1, 7], &device); // mismatched
        let style = zeros(&[1, 2], &device);
        assert!(decoder.forward(&client, &asr, &f0, &n, &style).is_err());
    }
}
