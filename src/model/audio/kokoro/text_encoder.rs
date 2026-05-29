//! Kokoro text encoder.
//!
//! Architecture (matches StyleTTS2 `models.TextEncoder`):
//!
//! ```text
//!     phoneme_ids [B, T]
//!       → Embedding(n_symbols, channels)           → [B, T, C]
//!       → transpose to [B, C, T]
//!       → ConvBlock × depth:
//!           Conv1d(C, C, kernel_size, padding=same) + channel-wise LayerNorm
//!           + LeakyReLU(0.2)
//!       → transpose to [B, T, C]
//!       → BiLSTM(C, C/2) batch_first → [B, T, C]
//! ```
//!
//! Returns `[B, T, C]`.
//!
//! "Channel-wise LayerNorm" normalizes across the channel axis independently at
//! each time step, which is `F.layer_norm` applied after transposing `[B,C,T]`
//! → `[B,T,C]`.

use crate::error::{Error, Result};
use crate::nn::{BiLstm, Conv1d, Embedding};
use numr::dtype::DType;
#[allow(unused_imports)]
use numr::ops::{
    ActivationOps, BinaryOps, ConvOps, IndexingOps, MatmulOps, NormalizationOps, PaddingMode,
    ReduceOps, ScalarOps, TensorOps, UnaryOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Per-conv-block parameters.
pub struct ConvBlock<R: Runtime> {
    conv: Conv1d<R>,
    ln_weight: Tensor<R>, // [channels]
    ln_bias: Tensor<R>,   // [channels]
    eps: f32,
    leaky_slope: f64,
}

impl<R: Runtime> ConvBlock<R> {
    pub fn new(
        conv: Conv1d<R>,
        ln_weight: Tensor<R>,
        ln_bias: Tensor<R>,
        eps: f32,
        leaky_slope: f64,
    ) -> Self {
        Self {
            conv,
            ln_weight,
            ln_bias,
            eps,
            leaky_slope,
        }
    }

    /// Forward: `[B, C, T]` → `[B, C, T]`.
    pub fn forward<C>(&self, client: &C, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R> + ConvOps<R> + NormalizationOps<R> + ActivationOps<R> + TensorOps<R>,
    {
        let y = self.conv.forward_inference(client, x)?;
        // Channel-wise LayerNorm: [B,C,T] -> [B,T,C] -> LN over last dim -> [B,C,T]
        let y_bt_c = y.transpose(1, 2).map_err(Error::Numr)?.contiguous();
        let y_ln = client
            .layer_norm(&y_bt_c, &self.ln_weight, &self.ln_bias, self.eps)
            .map_err(Error::Numr)?;
        let y_bct = y_ln.transpose(1, 2).map_err(Error::Numr)?.contiguous();
        client
            .leaky_relu(&y_bct, self.leaky_slope)
            .map_err(Error::Numr)
    }
}

/// Configuration for `TextEncoder::build`.
#[derive(Debug, Clone, Copy)]
pub struct TextEncoderConfig {
    pub n_symbols: usize,
    pub channels: usize,
    pub kernel_size: usize,
    pub depth: usize,
    pub eps: f32,
    pub leaky_slope: f64,
}

impl Default for TextEncoderConfig {
    fn default() -> Self {
        // Kokoro-82M defaults from hexgrad/kokoro config.json.
        Self {
            n_symbols: 178,
            channels: 512,
            kernel_size: 5,
            depth: 3,
            eps: 1e-5,
            leaky_slope: 0.2,
        }
    }
}

pub struct TextEncoder<R: Runtime> {
    embedding: Embedding<R>,
    conv_blocks: Vec<ConvBlock<R>>,
    lstm: BiLstm<R>,
    channels: usize,
}

impl<R: Runtime> TextEncoder<R> {
    /// Construct from already-loaded submodules. Callers are responsible for
    /// reading tensors out of the VarBuilder in the right order; a dedicated
    /// loader helper will land with the Kokoro safetensors loader (M7).
    pub fn new(
        embedding: Embedding<R>,
        conv_blocks: Vec<ConvBlock<R>>,
        lstm: BiLstm<R>,
        channels: usize,
    ) -> Result<Self> {
        if 2 * lstm.hidden_size() != channels {
            return Err(Error::InvalidArgument {
                arg: "lstm",
                reason: format!(
                    "BiLSTM total output width must equal text encoder channels ({channels}), \
                     got 2 * {}",
                    lstm.hidden_size()
                ),
            });
        }
        Ok(Self {
            embedding,
            conv_blocks,
            lstm,
            channels,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Forward: phoneme ids `[B, T]` → hidden states `[B, T, C]`.
    pub fn forward<C>(&self, client: &C, phoneme_ids: &Tensor<R>) -> Result<Tensor<R>>
    where
        R: Runtime<DType = DType>,
        C: RuntimeClient<R>
            + IndexingOps<R>
            + ConvOps<R>
            + NormalizationOps<R>
            + ActivationOps<R>
            + TensorOps<R>
            + MatmulOps<R>
            + BinaryOps<R>
            + UnaryOps<R>
            + ReduceOps<R>
            + ScalarOps<R>
            + UtilityOps<R>,
        R::Client: IndexingOps<R>,
    {
        let shape = phoneme_ids.shape();
        if shape.len() != 2 {
            return Err(Error::InvalidArgument {
                arg: "phoneme_ids",
                reason: format!("expected [B, T], got {shape:?}"),
            });
        }

        // [B, T] -> [B, T, C]
        let embedded = self.embedding.forward(client, phoneme_ids)?;
        // [B, T, C] -> [B, C, T]
        let mut h = embedded
            .tensor()
            .transpose(1, 2)
            .map_err(Error::Numr)?
            .contiguous();

        for block in &self.conv_blocks {
            h = block.forward(client, &h)?;
        }

        // [B, C, T] -> [B, T, C] for LSTM
        let h_btc = h.transpose(1, 2).map_err(Error::Numr)?.contiguous();
        self.lstm.forward(client, &h_btc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::{BiLstm, Conv1d, Embedding, Lstm};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

    fn zeros(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![0.0f32; n], shape, device)
    }

    fn ones(shape: &[usize], device: &<CpuRuntime as Runtime>::Device) -> Tensor<CpuRuntime> {
        let n: usize = shape.iter().product();
        Tensor::<CpuRuntime>::from_slice(&vec![1.0f32; n], shape, device)
    }

    fn build_tiny_encoder(device: &<CpuRuntime as Runtime>::Device) -> TextEncoder<CpuRuntime> {
        // channels=4, kernel=3, depth=2, n_symbols=8, hidden=2 (so 2*hidden=4=channels).
        let channels = 4;
        let kernel_size = 3;

        let emb = Embedding::new(zeros(&[8, channels], device), false);

        let mut blocks = Vec::new();
        for _ in 0..2 {
            let conv = Conv1d::new(
                zeros(&[channels, channels, kernel_size], device),
                Some(zeros(&[channels], device)),
                1,
                PaddingMode::Same,
                1,
                1,
                false,
            );
            blocks.push(ConvBlock::new(
                conv,
                ones(&[channels], device),
                zeros(&[channels], device),
                1e-5,
                0.2,
            ));
        }

        let hidden = channels / 2;
        let lstm_f = Lstm::new(
            zeros(&[4 * hidden, channels], device),
            zeros(&[4 * hidden, hidden], device),
            zeros(&[4 * hidden], device),
            zeros(&[4 * hidden], device),
        )
        .unwrap();
        let lstm_b = Lstm::new(
            zeros(&[4 * hidden, channels], device),
            zeros(&[4 * hidden, hidden], device),
            zeros(&[4 * hidden], device),
            zeros(&[4 * hidden], device),
        )
        .unwrap();
        let bi = BiLstm::new(lstm_f, lstm_b).unwrap();

        TextEncoder::new(emb, blocks, bi, channels).unwrap()
    }

    #[test]
    fn forward_shape_is_b_t_c() {
        let (client, device) = cpu_setup();
        let enc = build_tiny_encoder(&device);

        let ids = Tensor::<CpuRuntime>::from_slice(&[1i64, 2, 3, 4, 5, 6], &[2, 3], &device);
        let out = enc.forward(&client, &ids).unwrap();
        assert_eq!(out.shape(), &[2, 3, 4]);
    }

    #[test]
    fn zero_weights_yield_finite_output() {
        let (client, device) = cpu_setup();
        let enc = build_tiny_encoder(&device);

        let ids = Tensor::<CpuRuntime>::from_slice(&[0i64; 5], &[1, 5], &device);
        let out = enc.forward(&client, &ids).unwrap();
        for v in out.to_vec::<f32>() {
            assert!(v.is_finite(), "got non-finite value {v}");
        }
    }

    #[test]
    fn rejects_rank_other_than_2() {
        let (client, device) = cpu_setup();
        let enc = build_tiny_encoder(&device);
        let ids = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2], &[3], &device);
        assert!(enc.forward(&client, &ids).is_err());
    }

    #[test]
    fn new_rejects_lstm_width_mismatch() {
        let (_client, device) = cpu_setup();
        // channels = 4 but BiLSTM width = 2 * 3 = 6 — should reject.
        let emb = Embedding::new(zeros(&[8, 4], &device), false);
        let hidden = 3;
        let lstm_f = Lstm::new(
            zeros(&[4 * hidden, 4], &device),
            zeros(&[4 * hidden, hidden], &device),
            zeros(&[4 * hidden], &device),
            zeros(&[4 * hidden], &device),
        )
        .unwrap();
        let lstm_b = Lstm::new(
            zeros(&[4 * hidden, 4], &device),
            zeros(&[4 * hidden, hidden], &device),
            zeros(&[4 * hidden], &device),
            zeros(&[4 * hidden], &device),
        )
        .unwrap();
        let bi = BiLstm::new(lstm_f, lstm_b).unwrap();
        assert!(TextEncoder::new(emb, Vec::new(), bi, 4).is_err());
    }
}
