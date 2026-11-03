//! [`TorchPthSource`]: a `torch.save`d state dict read as a [`WeightSource`],
//! with every lookup spelled under one root prefix.
//!
//! Training frameworks nest the state dict one level down. BasicSR saves
//! `{"params_ema": …}` or `{"params": …}`, Lightning-style scripts save
//! `{"state_dict": …}`, and settings sit beside it, such as `metadata.in_nc`.
//! The flattened keys carry that wrapper as a leading segment. The source
//! strips it, so a model asks for `conv_first.weight`, not
//! `params_ema.conv_first.weight`.
//!
//! The source stores tensors as saved. A model that needs a fold on read,
//! such as VoxCPM2's `weight_norm` pairs, wraps this type.

use super::api::TorchStateDict;
use crate::error::Result;
use crate::format::weight_source::WeightSource;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::path::Path;

/// Root prefixes [`TorchPthSource::open`] detects, in priority order.
///
/// The order matches `checkpoint.get("params_ema", checkpoint.get("params",
/// checkpoint))` in BasicSR-style loaders: a file with both `params_ema`
/// and `params` loads the EMA weights.
pub const ROOT_PREFIXES: [&str; 3] = ["params_ema.", "params.", "state_dict."];

/// A `.pt` / `.pth` state dict presented as a [`WeightSource`], read under
/// one root prefix.
///
/// Reads are lazy: only the pickle metadata parses up front, and each
/// requested tensor loads on demand through [`TorchStateDict`]. Keys
/// outside the root, such as `metadata.in_nc`, stay readable through
/// [`TorchPthSource::state_dict`].
pub struct TorchPthSource {
    state: TorchStateDict,
    /// Prefix every lookup is spelled under, with its trailing `.`; empty
    /// for tensors at the root.
    root: String,
}

impl TorchPthSource {
    /// Open `path` and detect its root prefix with [`detect_root`].
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let state = TorchStateDict::open(path)?;
        let root = detect_root(&state).to_string();
        Ok(Self { state, root })
    }

    /// Open `path` and read every tensor under `root`.
    ///
    /// `root` is the full prefix, trailing `.` included, such as
    /// `"params."`. Pass `""` for tensors at the root.
    pub fn open_with_root(path: impl AsRef<Path>, root: impl Into<String>) -> Result<Self> {
        Ok(Self::from_state_dict(TorchStateDict::open(path)?, root))
    }

    /// Wrap an open state dict and read every tensor under `root`.
    pub fn from_state_dict(state: TorchStateDict, root: impl Into<String>) -> Self {
        Self {
            state,
            root: root.into(),
        }
    }

    /// The root prefix in use, trailing `.` included. Empty for tensors at
    /// the root.
    pub fn root(&self) -> &str {
        &self.root
    }

    /// The underlying state dict, for keys outside the root such as
    /// `metadata.in_nc`.
    pub fn state_dict(&self) -> &TorchStateDict {
        &self.state
    }

    /// The stored key for model name `name`: the root prefix plus `name`.
    pub fn stored_key(&self, name: &str) -> String {
        format!("{}{name}", self.root)
    }

    /// Every model name the source holds: the tensor keys under the root,
    /// with the root stripped.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        let root = self.root.as_str();
        self.state.keys().filter_map(move |k| k.strip_prefix(root))
    }
}

/// The first of [`ROOT_PREFIXES`] that some tensor key starts with, else
/// `""` for tensors at the root.
pub fn detect_root(state: &TorchStateDict) -> &'static str {
    ROOT_PREFIXES
        .into_iter()
        .find(|prefix| state.keys().any(|k| k.starts_with(*prefix)))
        .unwrap_or("")
}

impl<R: Runtime<DType = DType>> WeightSource<R> for TorchPthSource {
    fn load_named(&mut self, name: &str, device: &R::Device) -> Result<Tensor<R>> {
        self.state.load_tensor::<R>(&self.stored_key(name), device)
    }

    fn has_named(&self, name: &str) -> bool {
        self.state.has(&self.stored_key(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_file() {
        let err = TorchPthSource::open("/nonexistent/model.pth")
            .err()
            .map(|e| e.to_string());
        assert!(
            err.as_deref()
                .is_some_and(|e| e.contains("/nonexistent/model.pth")),
            "{err:?}"
        );
    }
}
