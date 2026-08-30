//! [`TorchPthSource`]: reads a `torch.save`d state dict as a [`WeightSource`],
//! folding `torch.nn.utils.weight_norm` reparameterization on the way out.
//!
//! # Why the reader takes the `.pth` directly
//!
//! VoxCPM2's `AudioVAE` ships as `audiovae.pth`, a torch pickle whose conv
//! weights are stored as `weight_g`/`weight_v` pairs. That pair exists to
//! condition gradients during TRAINING; at inference it is a fixed rescale
//!
//! ```text
//!     weight = (weight_g / ||weight_v||) * weight_v
//! ```
//!
//! with the norm over every axis but the output-channel axis. The reference
//! repo folds it in a Python conversion script that writes an intermediate
//! `audiovae.safetensors`. Doing the same fold here removes that build step,
//! the intermediate artifact, and the stale-conversion bug it invites — the
//! runtime reads the checkpoint the model actually ships.
//!
//! The arithmetic itself is [`crate::nn::fuse_weight_norm`], shared with the
//! Kokoro loaders (and available to the cstr GGUF path, which stores the same
//! pairs unfolded).

use super::weight_source::WeightSource;
use crate::error::{Error, Result};
use crate::format::torch_pt::TorchStateDict;
use crate::nn::fuse_weight_norm;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, TensorOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::path::Path;

/// Suffixes `torch.nn.utils.weight_norm` splits a `weight` into.
const G_SUFFIX: &str = ".weight_g";
const V_SUFFIX: &str = ".weight_v";

/// The nesting `torch.save({"metadata": …, "state_dict": {…}})` adds. The
/// flattened keys carry it as a leading segment; the reference script drops
/// it with `blob["state_dict"] if "state_dict" in blob else blob`.
const STATE_DICT_ROOT: &str = "state_dict.";

/// A `.pt` / `.pth` state dict presented as a [`WeightSource`], with
/// `weight_norm` pairs folded into the single `weight` a forward pass uses.
///
/// Reads are lazy: only the pickle metadata is parsed up front, and each
/// requested tensor is materialized on demand by [`TorchStateDict`].
pub struct TorchPthSource {
    state: TorchStateDict,
    /// Prefix every lookup is spelled under — [`STATE_DICT_ROOT`] when the
    /// checkpoint nests its tensors under `state_dict`, else empty.
    root: String,
}

impl TorchPthSource {
    /// Open `path` and check every `weight_norm` pair is complete.
    ///
    /// The completeness check runs here, over the whole key set, rather than
    /// lazily per read: the reference script raises `KeyError` for a `_g`
    /// with no `_v` whether or not anything asks for that weight, and a
    /// half-written checkpoint must fail the load, not one later lookup.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let state = TorchStateDict::open(path)?;
        let root = if state.keys().any(|k| k.starts_with(STATE_DICT_ROOT)) {
            STATE_DICT_ROOT.to_string()
        } else {
            String::new()
        };

        let orphans: Vec<String> = state
            .keys()
            .filter_map(|k| k.strip_suffix(G_SUFFIX))
            .map(|stem| format!("{stem}{V_SUFFIX}"))
            .filter(|v| !state.has(v))
            .collect();
        if let Some(missing) = orphans.first() {
            return Err(Error::ModelError {
                reason: format!(
                    "weight_norm pair is incomplete: {missing} is absent but its \
                     weight_g partner is present ({} such key(s))",
                    orphans.len()
                ),
            });
        }

        Ok(Self { state, root })
    }
}

impl<R: Runtime<DType = DType>> WeightSource<R> for TorchPthSource
where
    R::Client: ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    fn load_named(&mut self, name: &str, device: &R::Device) -> Result<Tensor<R>> {
        let key = format!("{}{name}", self.root);
        if self.state.has(&key) {
            return self.state.load_tensor::<R>(&key, device);
        }

        // Not stored directly: the only other way this name exists is as a
        // folded `weight_norm` pair.
        let Some(stem) = key.strip_suffix(".weight") else {
            return self.state.load_tensor::<R>(&key, device);
        };
        let g_key = format!("{stem}{G_SUFFIX}");
        if !self.state.has(&g_key) {
            // Let the state dict phrase the not-found error, listing what it
            // does hold.
            return self.state.load_tensor::<R>(&key, device);
        }
        let v_key = format!("{stem}{V_SUFFIX}");
        if !self.state.has(&v_key) {
            return Err(Error::ModelError {
                reason: format!("{g_key} has no matching {v_key}"),
            });
        }

        let g = self.state.load_tensor::<R>(&g_key, device)?;
        let v = self.state.load_tensor::<R>(&v_key, device)?;
        // Axis 0 unconditionally: `weight_norm`'s default `dim=0` is what
        // wrote these pairs, for the transposed convolutions too — there the
        // normalized axis is the INPUT-channel axis, and reinterpreting it as
        // the output one would rescale the wrong slices.
        let client = R::default_client(device);
        fuse_weight_norm(&client, &v, &g, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_file() {
        assert!(TorchPthSource::open("/nonexistent/audiovae.pth").is_err());
    }
}
