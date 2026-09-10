//! The calibration pass: run the model over the selected windows with the
//! importance collector armed, and hand back what it measured.
//!
//! Forward only. No loss, no backward, no KV cache, no sampling — the model
//! never consumes its own output, so nothing here draws a random number. The
//! windows come from `shared::windows::select_windows`, the SAME selection
//! `token_ce` scores, so an importance matrix and an evaluation of its effect
//! are computed over the same tokens.

use std::error::Error;

use boostr::model::llama::Llama;
use boostr::model::traits::{Model, ModelClient};
use boostr::quant::imatrix::{self, ImportanceMatrix};
use boostr::quant::traits::DequantOps;
use numr::autograd::Var;
use numr::dtype::DType;
use numr::ops::TypeConversionOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// What one calibration run produced.
pub struct Collected {
    /// The measured statistics, ready to write.
    pub matrix: ImportanceMatrix,
    /// Tokens fed to the model, summed over every window.
    pub tokens: u64,
    /// Dense weights a loader bound a checkpoint name to, but that no window
    /// ever exercised. Recorded here and ABSENT from the file — a zero
    /// importance vector and "never measured" mean opposite things.
    pub unexercised: Vec<String>,
}

/// Run every window through the model with the collector armed.
///
/// `windows` holds `seq_len + 1` tokens each, the same slices `token_ce`
/// scores; this pass feeds the first `seq_len` of each, which is exactly the
/// set of positions whose activations `token_ce`'s loss depends on.
///
/// The collector must already be armed by the caller BEFORE the model was
/// built: the checkpoint names are bound during loading, not during the
/// forward pass.
pub fn collect_windows<R, C>(
    model: &Llama<R>,
    client: &C,
    device: &R::Device,
    windows: &[&[i64]],
) -> Result<Collected, Box<dyn Error>>
where
    R: Runtime<DType = DType>,
    C: ModelClient<R> + DequantOps<R> + TypeConversionOps<R>,
    R::Client: ModelClient<R> + DequantOps<R> + TypeConversionOps<R>,
{
    let loaded = imatrix::registered_names();
    let mut tokens = 0u64;

    for (index, window) in windows.iter().enumerate() {
        let seq_len = window.len() - 1;
        let inputs = Tensor::<R>::from_slice(&window[..seq_len], &[1, seq_len], device)?;
        // The logits are dropped: this run wants the ACTIVATIONS that fed
        // every linear layer, and the collector already summed those on the
        // device while the forward pass ran.
        let _ = model.forward(client, &Var::new(inputs, false))?;
        tokens += seq_len as u64;
        eprintln!(
            "window {}/{}: {seq_len} token(s) accumulated, {tokens} so far",
            index + 1,
            windows.len()
        );
    }

    let matrix = imatrix::finish::<R>(tokens)?;
    let unexercised = loaded
        .into_iter()
        .filter(|name| matrix.get(name).is_none())
        .collect();
    Ok(Collected {
        matrix,
        tokens,
        unexercised,
    })
}
