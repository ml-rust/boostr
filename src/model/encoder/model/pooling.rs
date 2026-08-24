//! Pooling strategies and the one padded-path implementation of each.
//!
//! The padded pooling logic lives here rather than being repeated in the
//! inference, training and CUDA-graph forwards, so that adding a strategy
//! cannot leave one of those paths behind.

use crate::error::{Error, Result};
use crate::model::encoder::config::{ArchFamily, EncoderConfig};
use numr::autograd::{Var, var_narrow, var_reshape};
use numr::dtype::DType;
use numr::ops::{BinaryOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Pooling strategy for producing a single vector from encoder outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Average all token hidden states (most common for sentence embeddings).
    /// GGUF `pooling_type = 1`.
    Mean,
    /// Use the `[CLS]` token's hidden state (position 0). GGUF `pooling_type = 2`.
    Cls,
    /// Use the last real (non-padding) token's hidden state. GGUF
    /// `pooling_type = 3`.
    ///
    /// Used by causal embedding backbones such as Qwen3-Embedding, where only
    /// the final position has attended to the whole input. With a padding mask
    /// this selects the last unmasked position per row, not `seq_len - 1`.
    Last,
}

impl Pooling {
    /// The strategy an architecture family was trained to be read out with.
    ///
    /// Callers that build an `Encoder` directly should use this rather than
    /// assuming `Mean`: a causal backbone read with mean pooling averages in
    /// prefix states that never saw the whole input, which produces plausible
    /// but materially worse vectors instead of an error.
    pub fn for_arch(arch: ArchFamily) -> Self {
        match arch {
            ArchFamily::Qwen3 => Self::Last,
            _ => Self::Mean,
        }
    }

    /// The strategy to read this model out with: what the file declares, else
    /// the architecture default.
    ///
    /// Prefer this over [`Self::for_arch`] whenever a config came from a model
    /// file. The `bert` GGUF namespace covers both mean-pooled and CLS-pooled
    /// encoders — `bge-m3` declares `pooling_type = 2` — so the architecture
    /// alone does not determine the answer, and guessing wrong yields a
    /// perfectly well-formed vector from the wrong read-out position.
    ///
    /// An unrecognised code falls back to the architecture default rather than
    /// failing: the namespaces that support only one strategy already reject
    /// anything else at parse time, so a stray value can only reach here from a
    /// namespace that does not constrain it.
    pub fn from_config(config: &EncoderConfig) -> Self {
        match config.declared_pooling_type {
            Some(1) => Self::Mean,
            Some(2) => Self::Cls,
            Some(3) => Self::Last,
            _ => Self::for_arch(config.arch_family),
        }
    }
}

/// Pool `[B, S, H]` hidden states to `[B, H]`.
///
/// `ones_scalar` must be a `[1]` f32 tensor holding 1.0. The CUDA graph path
/// passes one allocated outside the capture region; other callers pass `None`
/// and one is allocated on demand. An inline allocation inside a capture region
/// bakes a host stack pointer into the graph and faults on replay.
pub(in crate::model::encoder) fn pool_padded<R, C>(
    client: &C,
    hidden: &Tensor<R>,
    mask: Option<&Tensor<R>>,
    pooling: Pooling,
    ones_scalar: Option<&Tensor<R>>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + BinaryOps<R>
        + ReduceOps<R>
        + ShapeOps<R>
        + ScalarOps<R>
        + IndexingOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let hidden_var = Var::new(hidden.clone(), false);

    match pooling {
        Pooling::Mean => match mask {
            Some(m) => {
                let shape = m.shape().to_vec();
                let (batch, seq_len) = (shape[0], shape[1]);

                let mask_3d = m.reshape(&[batch, seq_len, 1]).map_err(Error::Numr)?;
                let masked = client
                    .mul(hidden_var.tensor(), &mask_3d)
                    .map_err(Error::Numr)?;
                let summed = client.sum(&masked, &[1], false).map_err(Error::Numr)?;
                let counts = client.sum(m, &[1], true).map_err(Error::Numr)?;

                let owned_ones;
                let ones = match ones_scalar {
                    Some(o) => o,
                    None => {
                        owned_ones = Tensor::<R>::try_from_slice(&[1.0f32], &[1], m.device())?;
                        &owned_ones
                    }
                };
                let counts = client.maximum(&counts, ones).map_err(Error::Numr)?;
                client.div(&summed, &counts).map_err(Error::Numr)
            }
            None => client
                .mean(hidden_var.tensor(), &[1], false)
                .map_err(Error::Numr),
        },

        Pooling::Cls => select_position(&hidden_var, 0),

        Pooling::Last => {
            let shape = hidden_var.shape().to_vec();
            let (batch, seq_len) = (shape[0], shape[1]);

            match mask {
                // Gather the last unmasked position of each row. Padding is
                // usually on the right, so `seq_len - 1` would pick a pad
                // position and return the embedding of nothing.
                Some(m) => {
                    let mask_host: Vec<f32> = m.to_vec();
                    let mut indices = Vec::with_capacity(batch);
                    for b in 0..batch {
                        let row = &mask_host[b * seq_len..(b + 1) * seq_len];
                        let last = row.iter().rposition(|&v| v != 0.0).ok_or_else(|| {
                            Error::ModelError {
                                reason: format!(
                                    "sequence {b} has no unmasked token, so last-token \
                                     pooling has nothing to select"
                                ),
                            }
                        })?;
                        indices.push((b * seq_len + last) as i64);
                    }
                    let flat = hidden_var
                        .tensor()
                        .contiguous()?
                        .reshape(&[batch * seq_len, shape[2]])
                        .map_err(Error::Numr)?;
                    let idx = Tensor::<R>::try_from_slice(&indices, &[batch], m.device())?;
                    client.embedding_lookup(&flat, &idx).map_err(Error::Numr)
                }
                None => select_position(&hidden_var, seq_len - 1),
            }
        }
    }
}

/// Slice one position out of `[B, S, H]` into `[B, H]`.
fn select_position<R: Runtime<DType = DType>>(hidden: &Var<R>, pos: usize) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let picked = var_narrow(hidden, 1, pos, 1).map_err(Error::Numr)?;
    let picked = Var::new(picked.tensor().contiguous()?, false);
    let shape = picked.shape().to_vec();
    Ok(var_reshape(&picked, &[shape[0], shape[2]])
        .map_err(Error::Numr)?
        .tensor()
        .clone())
}
