//! NeuCodec's semantic adapter (upstream `neucodec.module.SemanticEncoder`):
//! projects the Wav2Vec2-BERT semantic branch into the shared 1024-dim latent
//! space, channels-first, length-preserving throughout.
//!
//! Verified against the upstream source AND the checkpoint's
//! `semantic_adapter.*` tensors (6 total):
//!
//! ```text
//! x [B, 1024, T]
//!   -> conv1   Conv1d(1024 -> 1024, k=3, pad=1, no bias)   upstream `initial_conv`
//!   -> h    = conv1(x)
//!   -> skip = relu(h)                                       (see the note below)
//!   -> r    = conv3(relu(conv2(skip))) + skip               upstream `residual_blocks` + residual
//!   -> conv4(r)                                             upstream `final_conv`, no bias
//! [B, 1024, T]
//! ```
//!
//! HF checkpoint names `conv1..conv4`; upstream names `initial_conv`,
//! `residual_blocks.1` (first conv), `residual_blocks.3` (second conv), and
//! `final_conv`. `residual_blocks.0`/`.2` are the `ReLU`s in between, which
//! carry no weights. The bias pattern pins the mapping: `conv1`/`conv4` are
//! weight-only (matching `bias=False` on `initial_conv`/`final_conv`), while
//! `conv2`/`conv3` carry bias (matching `bias=True` on the inner convs).
//!
//! ## The skip adds `relu(h)`, not `h` — an in-place-ReLU side effect
//!
//! Upstream reads `x = self.residual_blocks(x) + x`, where `x` has already
//! been reassigned to `initial_conv(x)`. That looks like "add `h`". It is not:
//! `residual_blocks[0]` is `nn.ReLU(inplace=True)`, so evaluating
//! `residual_blocks(x)` REWRITES `x` in place before the `+ x` is applied. The
//! tensor actually added is therefore `relu(h)`.
//!
//! This is invisible in the checkpoint and easy to read past in the source.
//! Reconstructing the obvious `+ h` reading gives `max|d| = 2.10` against an
//! output of rms 1.36 — completely wrong audio, not a rounding difference.
//! Verified by evaluating six candidate wirings against upstream's own module
//! on the real weights; only the `relu(h)` skip reproduces it exactly (0.0e0).

use crate::error::{Error, Result};
use crate::model::audio::neucodec::client::NeuCodecClient;
use crate::nn::Conv1d;
use numr::autograd::{Var, var_add, var_relu};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Channel width of the semantic adapter (checkpoint: 1024 throughout).
pub const SEMANTIC_ADAPTER_CHANNELS: usize = 1024;
/// Kernel size of every conv in the adapter.
pub const SEMANTIC_ADAPTER_KERNEL_SIZE: usize = 3;

/// `SemanticEncoder` port: four same-padded k=3 convs with a residual
/// connection around the middle two, channels-first throughout.
pub struct SemanticAdapter<R: Runtime> {
    conv1: Conv1d<R>,
    conv2: Conv1d<R>,
    conv3: Conv1d<R>,
    conv4: Conv1d<R>,
}

/// Already-built weights for [`SemanticAdapter`].
pub struct SemanticAdapterWeights<R: Runtime> {
    pub conv1: Conv1d<R>,
    pub conv2: Conv1d<R>,
    pub conv3: Conv1d<R>,
    pub conv4: Conv1d<R>,
}

impl<R: Runtime<DType = DType>> SemanticAdapter<R> {
    /// Assemble the adapter from already-loaded conv weights.
    pub fn new(weights: SemanticAdapterWeights<R>) -> Self {
        Self {
            conv1: weights.conv1,
            conv2: weights.conv2,
            conv3: weights.conv3,
            conv4: weights.conv4,
        }
    }

    /// Forward: `x [B, 1024, T] -> [B, 1024, T]`, channels-first.
    pub fn forward<C>(&self, client: &C, x: &Var<R>) -> Result<Var<R>>
    where
        C: NeuCodecClient<R>,
        R::Client: NeuCodecClient<R>,
    {
        let shape = x.shape().to_vec();
        if shape.len() != 3 || shape[1] != SEMANTIC_ADAPTER_CHANNELS {
            return Err(Error::InvalidArgument {
                arg: "x",
                reason: format!(
                    "expected semantic latent [B, {SEMANTIC_ADAPTER_CHANNELS}, T], got {shape:?}"
                ),
            });
        }

        let h = self.conv1.forward(client, x)?;

        // The skip adds relu(h), NOT h — see the module doc: upstream's first
        // residual-block layer is `nn.ReLU(inplace=True)`, which rewrites the
        // very tensor that `residual_blocks(x) + x` then adds.
        let skip = var_relu(&h, client).map_err(Error::Numr)?;

        let r = self.conv2.forward(client, &skip)?;
        let r = var_relu(&r, client).map_err(Error::Numr)?;
        let r = self.conv3.forward(client, &r)?;
        let r = var_add(&r, &skip, client).map_err(Error::Numr)?;

        self.conv4.forward(client, &r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::ops::PaddingMode;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    fn conv(device: &<CpuRuntime as Runtime>::Device, bias: bool) -> Conv1d<CpuRuntime> {
        let c = SEMANTIC_ADAPTER_CHANNELS;
        let k = SEMANTIC_ADAPTER_KERNEL_SIZE;
        let weight_data = vec![0.01f32; c * c * k];
        let weight =
            Tensor::<CpuRuntime>::try_from_slice(&weight_data, &[c, c, k], device).unwrap();
        let bias = if bias {
            Some(Tensor::<CpuRuntime>::try_from_slice(&vec![0.0f32; c], &[c], device).unwrap())
        } else {
            None
        };
        Conv1d::new(
            weight,
            bias,
            1,
            PaddingMode::Custom(1, 1, 0, 0),
            1,
            1,
            false,
        )
    }

    fn adapter(device: &<CpuRuntime as Runtime>::Device) -> SemanticAdapter<CpuRuntime> {
        SemanticAdapter::new(SemanticAdapterWeights {
            conv1: conv(device, false),
            conv2: conv(device, true),
            conv3: conv(device, true),
            conv4: conv(device, false),
        })
    }

    #[test]
    fn forward_preserves_shape() {
        let (client, device) = cpu_setup();
        let adapter = adapter(&device);

        let t = 8;
        let x_data: Vec<f32> = (0..(SEMANTIC_ADAPTER_CHANNELS * t))
            .map(|i| (i as f32) * 0.001)
            .collect();
        let x = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(
                &x_data,
                &[1, SEMANTIC_ADAPTER_CHANNELS, t],
                &device,
            )
            .unwrap(),
            false,
        );

        let y = adapter.forward(&client, &x).expect("forward");
        assert_eq!(y.shape(), &[1, SEMANTIC_ADAPTER_CHANNELS, t]);
        for v in y.tensor().contiguous().unwrap().to_vec::<f32>() {
            assert!(v.is_finite(), "adapter output is not finite: {v}");
        }
    }

    #[test]
    fn rejects_wrong_rank() {
        let (client, device) = cpu_setup();
        let adapter = adapter(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(
                &vec![0.0f32; SEMANTIC_ADAPTER_CHANNELS * 8],
                &[SEMANTIC_ADAPTER_CHANNELS, 8],
                &device,
            )
            .unwrap(),
            false,
        );
        assert!(adapter.forward(&client, &x).is_err());
    }

    #[test]
    fn rejects_wrong_channels() {
        let (client, device) = cpu_setup();
        let adapter = adapter(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&vec![0.0f32; 8 * 8], &[1, 8, 8], &device)
                .unwrap(),
            false,
        );
        assert!(adapter.forward(&client, &x).is_err());
    }
}
