//! Unified weight-source adapter for the Kokoro loader.
//!
//! The tier-3 loader (`load_kokoro_v2`) needs to read tensors by name from
//! either:
//!
//! * a safetensors directory (if the user pre-converted the upstream
//!   checkpoint), or
//! * the upstream `kokoro-v1_0.pth` directly (via [`TorchStateDict`]).
//!
//! Both source types expose the same two operations the tier-1 helpers need:
//! "load this tensor by name" and "does this tensor exist". This enum wraps
//! them so the helpers don't have to be generic / trait-objectified.

use crate::error::Result;
use crate::format::TorchStateDict;
use crate::format::safetensors_loader::SafeTensorsLoader;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::path::Path;

pub enum KokoroWeightSource {
    SafeTensors(SafeTensorsLoader),
    Pickle(TorchStateDict),
}

impl KokoroWeightSource {
    /// Auto-detect from a model directory. Prefers a `.safetensors` file if
    /// one exists (faster loads, zero pickle surface); falls back to the
    /// upstream `.pth` otherwise.
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = model_dir.as_ref();
        // Safetensors first: either `model.safetensors` or any `.safetensors`
        // that `SafeTensorsLoader::open` picks up.
        let has_st = std::fs::read_dir(dir)
            .map(|entries| {
                entries
                    .flatten()
                    .any(|e| e.path().extension().is_some_and(|x| x == "safetensors"))
            })
            .unwrap_or(false);
        if has_st {
            let st = SafeTensorsLoader::open(dir)?;
            return Ok(Self::SafeTensors(st));
        }
        // Otherwise look for a single `.pth` / `.pt` file in the directory.
        for name in ["kokoro-v1_0.pth", "kokoro.pth", "model.pth", "model.pt"] {
            let candidate = dir.join(name);
            if candidate.is_file() {
                return Ok(Self::Pickle(TorchStateDict::open(&candidate)?));
            }
        }
        // Last resort: any `.pth` / `.pt` file.
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.extension().is_some_and(|x| x == "pth" || x == "pt") {
                    return Ok(Self::Pickle(TorchStateDict::open(&p)?));
                }
            }
        }
        Err(crate::error::Error::ModelError {
            reason: format!("no .safetensors or .pth weights found in {}", dir.display()),
        })
    }

    /// Load a tensor by flattened name.
    pub fn load_tensor<R: Runtime<DType = DType>>(
        &mut self,
        name: &str,
        device: &R::Device,
    ) -> Result<Tensor<R>> {
        match self {
            Self::SafeTensors(s) => s.load_tensor::<R>(name, device),
            Self::Pickle(p) => p.load_tensor::<R>(name, device),
        }
    }

    /// Whether a tensor with this name is present.
    pub fn has_tensor(&self, name: &str) -> bool {
        match self {
            Self::SafeTensors(s) => s.tensor_info(name).is_ok(),
            Self::Pickle(p) => p.has(name),
        }
    }
}
