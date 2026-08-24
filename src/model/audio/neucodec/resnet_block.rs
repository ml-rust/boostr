//! `ResnetBlock` — the pre/post-net residual block used by NeuCodec's
//! acoustic decoder (`prior_net.{0,1}`, `post_net.{0,1}`).
//!
//! Checkpoint shape per block (all `[1024]` for `norm1`/`norm2`):
//! `norm1 -> swish -> conv1(k=3) -> norm2 -> swish -> dropout(0.1) ->
//! conv2(k=3)`, residual add around the whole block. Both norms carry weight
//! AND bias over the full 1024-channel width. The dropout carries no weights,
//! so the checkpoint cannot reveal it — it comes from the upstream
//! constructor (`ResnetBlock(..., dropout=0.1)`) and is inactive in eval mode.
//!
//! ## Norm choice: GroupNorm(num_groups=32, eps=1e-6), channels-first
//!
//! The checkpoint only records the affine shape (`[1024]`), which cannot on
//! its own distinguish GroupNorm from a per-timestep LayerNorm over the
//! channel axis. Upstream `neucodec/codec_decoder_vocos.py` settles it:
//!
//! ```python
//! def Normalize(in_channels, num_groups=32):
//!     return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels,
//!                               eps=1e-6, affine=True)
//! ```
//!
//! applied directly to the `[B, C, T]` tensor with no channels-last permute,
//! and with swish (`x * sigmoid(x)`, i.e. SiLU) as the activation. This is
//! numerically distinct from a channel-wise LayerNorm: GroupNorm(32)
//! normalizes within each group of `C/32` channels jointly across time, while
//! a LayerNorm would normalize across all channels of a single frame.
//!
//! Note the decoder's bare `acoustic_decoder.norm.{weight,bias}` (between the
//! transformer stack and `post_net`) IS a per-token LayerNorm — the two
//! conventions genuinely coexist in this model, so consistency arguments do
//! not apply here.

use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::nn::{Conv1d, Dropout, GroupNorm, TrainMode};
use numr::autograd::{Var, var_add, var_silu};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Number of GroupNorm groups used by every NeuCodec `ResnetBlock`
/// (`Normalize(in_channels, num_groups=32)` upstream).
pub const RESNET_NORM_GROUPS: usize = 32;

/// GroupNorm epsilon used by every NeuCodec `ResnetBlock` (upstream `eps=1e-6`,
/// NOT PyTorch's `1e-5` default).
pub const RESNET_NORM_EPS: f32 = 1e-6;

/// Bundled, already-built weights for one `ResnetBlock`.
pub struct ResnetBlockWeights<R: Runtime> {
    pub norm1: GroupNorm<R>,
    pub conv1: Conv1d<R>,
    pub norm2: GroupNorm<R>,
    pub conv2: Conv1d<R>,
}

/// Dropout probability upstream constructs every NeuCodec `ResnetBlock` with
/// (`ResnetBlock(..., dropout=0.1)`). Inactive in eval mode, which is the
/// default here.
pub const RESNET_DROPOUT_P: f64 = 0.1;

/// One `ResnetBlock`:
/// `norm1 -> swish -> conv1 -> norm2 -> swish -> dropout -> conv2`, with a
/// residual add around the whole block. Operates on channels-first
/// `[B, C, T]` tensors (matches `Conv1d`'s layout) in and out.
///
/// Starts in EVAL mode (dropout is identity), which is what a loaded
/// pretrained decoder wants; call [`TrainMode::set_training`] before
/// finetuning.
pub struct ResnetBlock<R: Runtime> {
    norm1: GroupNorm<R>,
    conv1: Conv1d<R>,
    norm2: GroupNorm<R>,
    dropout: Dropout,
    conv2: Conv1d<R>,
}

impl<R: Runtime> ResnetBlock<R> {
    pub fn new(weights: ResnetBlockWeights<R>) -> Self {
        let mut dropout = Dropout::new(RESNET_DROPOUT_P);
        dropout.set_training(false);
        Self {
            norm1: weights.norm1,
            conv1: weights.conv1,
            norm2: weights.norm2,
            dropout,
            conv2: weights.conv2,
        }
    }
}

impl<R: Runtime> TrainMode for ResnetBlock<R> {
    fn set_training(&mut self, training: bool) {
        self.dropout.set_training(training);
    }

    fn is_training(&self) -> bool {
        self.dropout.is_training()
    }
}

impl<R: Runtime<DType = DType>> ResnetBlock<R> {
    /// Forward: `x [B, C, T] -> [B, C, T]` (shape-preserving; both convs use
    /// same-padding so the time axis is untouched).
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!("expected [B, C, T], got {shape:?}"),
            });
        }

        let h = self.norm1.forward(client, x)?;
        let h = var_silu(&h, client).map_err(Error::Numr)?;
        let h = self.conv1.forward(client, &h)?;

        let h = self.norm2.forward(client, &h)?;
        let h = var_silu(&h, client).map_err(Error::Numr)?;
        let h = self.dropout.forward(client, &h)?;
        let h = self.conv2.forward(client, &h)?;

        var_add(x, &h, client).map_err(Error::Numr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::ops::PaddingMode;
    use numr::runtime::cpu::CpuRuntime;

    fn zeros(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Var<CpuRuntime> {
        let n: usize = shape.iter().product();
        Var::new(
            numr::tensor::Tensor::<CpuRuntime>::try_from_slice(&vec![0.0f32; n], shape, device)
                .unwrap(),
            false,
        )
    }

    /// Test-scale norm: same eps as production, but a group count that divides
    /// the small channel widths used in these tests.
    fn group_norm(
        c: usize,
        groups: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> GroupNorm<CpuRuntime> {
        GroupNorm::new(
            numr::tensor::Tensor::<CpuRuntime>::try_from_slice(&vec![1.0f32; c], &[c], device)
                .unwrap(),
            numr::tensor::Tensor::<CpuRuntime>::try_from_slice(&vec![0.0f32; c], &[c], device)
                .unwrap(),
            groups,
            RESNET_NORM_EPS,
            false,
        )
    }

    fn conv_with(
        c: usize,
        k: usize,
        w: f32,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Conv1d<CpuRuntime> {
        let n = c * c * k;
        Conv1d::new(
            numr::tensor::Tensor::<CpuRuntime>::try_from_slice(&vec![w; n], &[c, c, k], device)
                .unwrap(),
            Some(
                numr::tensor::Tensor::<CpuRuntime>::try_from_slice(&vec![0.0f32; c], &[c], device)
                    .unwrap(),
            ),
            1,
            PaddingMode::Same,
            1,
            1,
            false,
        )
    }

    fn conv(c: usize, k: usize, device: &<CpuRuntime as Runtime>::Device) -> Conv1d<CpuRuntime> {
        conv_with(c, k, 0.01, device)
    }

    fn block(
        c: usize,
        k: usize,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> ResnetBlock<CpuRuntime> {
        ResnetBlock::new(ResnetBlockWeights {
            norm1: group_norm(c, 2, device),
            conv1: conv(c, k, device),
            norm2: group_norm(c, 2, device),
            conv2: conv(c, k, device),
        })
    }

    #[test]
    fn forward_preserves_shape() {
        let (client, device) = cpu_setup();
        let b = ResnetBlock::new(ResnetBlockWeights {
            norm1: group_norm(8, 2, &device),
            conv1: conv(8, 3, &device),
            norm2: group_norm(8, 2, &device),
            conv2: conv(8, 3, &device),
        });
        let x = zeros(&[2, 8, 10], &device);
        let out = b.forward(&client, &x).unwrap();
        assert_eq!(out.shape(), &[2, 8, 10]);
    }

    /// With genuinely zero conv weights AND biases, `conv2` emits zero
    /// everywhere, so the block degenerates to the residual passthrough.
    ///
    /// The conv weights must actually be zero: an earlier version of this test
    /// used the shared `conv()` helper (weight 0.01) and passed only by
    /// accident, because channel-wise LayerNorm made each frame zero-mean and a
    /// uniform-weight conv then summed that to ~0. Under the correct GroupNorm
    /// — which centers within a group ACROSS TIME, not across channels within a
    /// frame — that cancellation does not occur, which is exactly what exposed
    /// the norm bug.
    #[test]
    fn zero_conv_leaves_residual_unchanged() {
        let (client, device) = cpu_setup();
        let b = ResnetBlock::new(ResnetBlockWeights {
            norm1: group_norm(4, 2, &device),
            conv1: conv_with(4, 3, 0.0, &device),
            norm2: group_norm(4, 2, &device),
            conv2: conv_with(4, 3, 0.0, &device),
        });
        let x_data = vec![
            1.0f32, -2.0, 3.0, 0.5, 2.0, -1.0, 0.0, 4.0, 1.5, -0.5, 2.5, 3.5,
        ];
        let x = Var::new(
            numr::tensor::Tensor::<CpuRuntime>::try_from_slice(&x_data, &[1, 4, 3], &device)
                .unwrap(),
            false,
        );
        let out = b.forward(&client, &x).unwrap();
        let out_data: Vec<f32> = out.tensor().contiguous().unwrap().to_vec();
        for (a, b) in out_data.iter().zip(x_data.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "expected identity passthrough: {a} vs {b}"
            );
        }
    }

    /// GroupNorm normalizes *within a group across time*, so a per-frame
    /// LayerNorm over all channels is NOT an acceptable substitute. Pin the
    /// distinction: with `groups = C` each channel is normalized over its own
    /// time axis alone, which zeroes any per-channel offset — something a
    /// channel-wise LayerNorm would preserve.
    #[test]
    fn norm_is_group_over_time_not_layernorm_over_channels() {
        let (client, device) = cpu_setup();
        // 2 channels, 4 timesteps. Channel 0 is centered at 10, channel 1 at
        // -10; both have identical shape around their mean.
        let x_data = vec![9.0f32, 11.0, 9.0, 11.0, -11.0, -9.0, -11.0, -9.0];
        let x = Var::new(
            numr::tensor::Tensor::<CpuRuntime>::try_from_slice(&x_data, &[1, 2, 4], &device)
                .unwrap(),
            false,
        );
        // groups == channels: normalize each channel over time independently.
        let norm = group_norm(2, 2, &device);
        let out = norm.forward(&client, &x).unwrap();
        let vals: Vec<f32> = out.tensor().contiguous().unwrap().to_vec();
        // Both channels collapse to the same +/-1 pattern — the large opposite
        // per-channel offsets are removed, which only per-channel-over-time
        // normalization does.
        let expected = [-1.0f32, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        for (got, want) in vals.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-3,
                "GroupNorm over time expected {want}, got {got}"
            );
        }
    }

    #[test]
    fn output_is_finite_for_nonzero_weights() {
        let (client, device) = cpu_setup();
        let c = 6;
        let k = 3;
        let n = c * c * k;
        let block = ResnetBlock::new(ResnetBlockWeights {
            norm1: GroupNorm::new(
                numr::tensor::Tensor::<CpuRuntime>::try_from_slice(&vec![1.2f32; c], &[c], &device)
                    .unwrap(),
                numr::tensor::Tensor::<CpuRuntime>::try_from_slice(&vec![0.1f32; c], &[c], &device)
                    .unwrap(),
                3,
                RESNET_NORM_EPS,
                false,
            ),
            conv1: Conv1d::new(
                numr::tensor::Tensor::<CpuRuntime>::try_from_slice(
                    &vec![0.05f32; n],
                    &[c, c, k],
                    &device,
                )
                .unwrap(),
                Some(
                    numr::tensor::Tensor::<CpuRuntime>::try_from_slice(
                        &vec![0.02f32; c],
                        &[c],
                        &device,
                    )
                    .unwrap(),
                ),
                1,
                PaddingMode::Same,
                1,
                1,
                false,
            ),
            norm2: GroupNorm::new(
                numr::tensor::Tensor::<CpuRuntime>::try_from_slice(&vec![0.9f32; c], &[c], &device)
                    .unwrap(),
                numr::tensor::Tensor::<CpuRuntime>::try_from_slice(
                    &vec![-0.1f32; c],
                    &[c],
                    &device,
                )
                .unwrap(),
                3,
                RESNET_NORM_EPS,
                false,
            ),
            conv2: Conv1d::new(
                numr::tensor::Tensor::<CpuRuntime>::try_from_slice(
                    &vec![-0.03f32; n],
                    &[c, c, k],
                    &device,
                )
                .unwrap(),
                Some(
                    numr::tensor::Tensor::<CpuRuntime>::try_from_slice(
                        &vec![0.0f32; c],
                        &[c],
                        &device,
                    )
                    .unwrap(),
                ),
                1,
                PaddingMode::Same,
                1,
                1,
                false,
            ),
        });
        let x_data: Vec<f32> = (0..(2 * c * 5)).map(|i| (i as f32 * 0.13).sin()).collect();
        let x = Var::new(
            numr::tensor::Tensor::<CpuRuntime>::try_from_slice(&x_data, &[2, c, 5], &device)
                .unwrap(),
            false,
        );
        let out = block.forward(&client, &x).unwrap();
        for v in out.tensor().contiguous().unwrap().to_vec::<f32>() {
            assert!(v.is_finite());
        }
    }

    /// Blocks default to EVAL, so a loaded pretrained decoder is deterministic;
    /// `set_training(true)` must actually arm the upstream `dropout=0.1`.
    #[test]
    fn defaults_to_eval_and_dropout_arms_on_train() {
        let (client, device) = cpu_setup();
        let mut b = ResnetBlock::new(ResnetBlockWeights {
            norm1: group_norm(4, 2, &device),
            // Nonzero convs so dropout downstream of norm2 can actually move
            // the output.
            conv1: conv(4, 3, &device),
            norm2: group_norm(4, 2, &device),
            conv2: conv(4, 3, &device),
        });
        assert!(!b.is_training(), "must start in eval mode");

        let x_data: Vec<f32> = (0..(4 * 8)).map(|i| (i as f32 * 0.37).sin()).collect();
        let x = Var::new(
            numr::tensor::Tensor::<CpuRuntime>::try_from_slice(&x_data, &[1, 4, 8], &device)
                .unwrap(),
            false,
        );

        let a: Vec<f32> = b
            .forward(&client, &x)
            .unwrap()
            .tensor()
            .contiguous()
            .unwrap()
            .to_vec();
        let c: Vec<f32> = b
            .forward(&client, &x)
            .unwrap()
            .tensor()
            .contiguous()
            .unwrap()
            .to_vec();
        assert_eq!(a, c, "eval mode must be deterministic (dropout = identity)");

        b.set_training(true);
        assert!(b.is_training());
        // With p=0.1 over 32 elements, at least one differing run is
        // overwhelmingly likely; try a handful to make the test robust.
        let differs = (0..8).any(|_| {
            let t: Vec<f32> = b
                .forward(&client, &x)
                .unwrap()
                .tensor()
                .contiguous()
                .unwrap()
                .to_vec();
            t.iter().zip(a.iter()).any(|(p, q)| (p - q).abs() > 1e-6)
        });
        assert!(differs, "training mode did not arm dropout");
    }

    #[test]
    fn rejects_wrong_rank_input() {
        let (client, device) = cpu_setup();
        let b = block(4, 3, &device);
        let x = zeros(&[4, 10], &device);
        assert!(b.forward(&client, &x).is_err());
    }
}
