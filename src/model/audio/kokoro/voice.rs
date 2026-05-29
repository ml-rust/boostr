//! Voice-spec resolver.
//!
//! Mirrors how blazr's `load_model` resolves an LLM argument (built-in ID vs
//! direct path vs HF cache): callers hand the resolver a `--voice SPEC` value,
//! it figures out where the file lives, delegates format dispatch to
//! `load_voice_style`, and returns a ready tensor.
//!
//! SPEC forms accepted:
//!
//! * `af_alloy` — bare voice ID. Resolved to
//!   `{voice_dir}/af_alloy.safetensors` (then `.pt`, `.pth`). `voice_dir`
//!   comes from (in order): explicit `asset_dir`, `$BLAZR_VOICE_DIR`, the
//!   bundled `blazr/assets/kokoro_voices/` shipped with the binary.
//! * `./path/to/file.safetensors` / `.pt` / `.pth` — direct path. File must
//!   exist; extension selects the format.
//! * `name:/path/to/dir` — future-proofing for when we want to co-locate
//!   multiple voices under a single custom directory. Rejected for now; use
//!   `$BLAZR_VOICE_DIR` instead.

use crate::error::{Error, Result};
use crate::model::audio::kokoro::loader::load_voice_pack;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::path::{Path, PathBuf};

/// Resolver configuration.
#[derive(Debug, Clone, Default)]
pub struct VoiceResolver {
    /// Primary asset directory. `None` → use `$BLAZR_VOICE_DIR` or the
    /// compiled-in bundled path. Callers typically pass the blazr binary's
    /// own `assets/kokoro_voices/` path.
    pub asset_dir: Option<PathBuf>,
}

impl VoiceResolver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_asset_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.asset_dir = Some(dir.into());
        self
    }

    /// Resolve the SPEC into a concrete filesystem path without reading it.
    /// Useful for error reporting and CLI validation.
    pub fn resolve_path(&self, spec: &str) -> Result<PathBuf> {
        if spec.is_empty() {
            return Err(Error::ModelError {
                reason: "voice SPEC is empty".into(),
            });
        }

        // Path form: contains a path separator or explicit file extension.
        let looks_like_path = spec.contains('/')
            || spec.contains('\\')
            || spec.ends_with(".safetensors")
            || spec.ends_with(".pt")
            || spec.ends_with(".pth");
        if looks_like_path {
            let p = PathBuf::from(spec);
            if !p.exists() {
                return Err(Error::ModelError {
                    reason: format!("voice file not found: {}", p.display()),
                });
            }
            return Ok(p);
        }

        // Bare ID form: search candidate directories in priority order for
        // `{id}.{safetensors,pt,pth}`.
        let dirs = self.candidate_dirs();
        for dir in &dirs {
            for ext in ["safetensors", "pt", "pth"] {
                let candidate = dir.join(format!("{spec}.{ext}"));
                if candidate.is_file() {
                    return Ok(candidate);
                }
            }
        }

        let dirs_pretty: Vec<String> = dirs.iter().map(|d| d.display().to_string()).collect();
        Err(Error::ModelError {
            reason: format!(
                "voice id {spec:?} not found in any of: {dirs_pretty:?} (looked for \
                 .safetensors, .pt, .pth)"
            ),
        })
    }

    /// Resolve + load. Returns the full voice pack tensor `[T, 1, 256]` (or
    /// `[T, 256]`). Use [`select_voice_style`] to pick the row for a given
    /// phoneme count and [`split_voice_style`] to separate decoder vs
    /// predictor halves.
    pub fn load<R: Runtime<DType = DType>>(
        &self,
        spec: &str,
        device: &R::Device,
    ) -> Result<Tensor<R>> {
        let path = self.resolve_path(spec)?;
        load_voice_pack::<R>(&path, device)
    }

    /// Candidate directory list in priority order.
    fn candidate_dirs(&self) -> Vec<PathBuf> {
        let mut out = Vec::new();
        if let Some(dir) = &self.asset_dir {
            out.push(dir.clone());
        }
        if let Ok(env) = std::env::var("BLAZR_VOICE_DIR") {
            let p = PathBuf::from(env);
            if !out.contains(&p) {
                out.push(p);
            }
        }
        // Last-resort bundled path relative to the binary (dev builds).
        // Production builds should always pass `asset_dir` explicitly.
        for rel in ["./assets/kokoro_voices", "../blazr/assets/kokoro_voices"] {
            let p = PathBuf::from(rel);
            if !out.contains(&p) {
                out.push(p);
            }
        }
        out
    }
}

/// Free-function convenience equivalent to `VoiceResolver::default().load(spec)`.
/// Matches blazr's `load_model` top-level function signature style.
pub fn resolve_and_load<R: Runtime<DType = DType>>(
    spec: &str,
    asset_dir: Option<&Path>,
    device: &R::Device,
) -> Result<Tensor<R>> {
    let mut resolver = VoiceResolver::new();
    if let Some(d) = asset_dir {
        resolver = resolver.with_asset_dir(d);
    }
    resolver.load::<R>(spec, device)
}

/// Pick the style row matching a phoneme-count budget.
///
/// Kokoro voice packs are `[T, 1, 2*D]` or `[T, 2*D]`. Upstream indexes by
/// `len(phonemes) - 1`, clamped to `[0, T-1]`. Returns a `[1, 2*D]` row that
/// can then be split into decoder and predictor halves via
/// [`split_voice_style`].
pub fn select_voice_style<R: Runtime<DType = DType>>(
    voice_pack: &Tensor<R>,
    phoneme_count: usize,
) -> Result<Tensor<R>> {
    let shape = voice_pack.shape();
    let (rows, style_width) = match shape.len() {
        3 => {
            if shape[1] != 1 {
                return Err(Error::ModelError {
                    reason: format!("voice pack middle dim must be 1, got shape {shape:?}"),
                });
            }
            (shape[0], shape[2])
        }
        2 => (shape[0], shape[1]),
        _ => {
            return Err(Error::ModelError {
                reason: format!("voice pack rank must be 2 or 3, got shape {shape:?}"),
            });
        }
    };
    if rows == 0 {
        return Err(Error::ModelError {
            reason: "voice pack is empty".into(),
        });
    }
    let idx = phoneme_count.saturating_sub(1).min(rows - 1);
    let flat = match shape.len() {
        3 => voice_pack
            .reshape(&[rows, style_width])
            .map_err(|e| Error::ModelError {
                reason: format!("reshape voice pack: {e}"),
            })?,
        _ => voice_pack.clone(),
    };
    flat.narrow(0, idx, 1).map_err(|e| Error::ModelError {
        reason: format!("narrow voice pack: {e}"),
    })
}

/// Split a row-selected voice style `[1, 2*D]` into `(decoder_style [1, D],
/// predictor_style [1, D])`. Decoder half is the first `D` channels,
/// predictor half is the last `D` — matching `ref_s[:, :128]` and
/// `ref_s[:, 128:]` in the upstream Python source.
pub fn split_voice_style<R: Runtime<DType = DType>>(
    style_row: &Tensor<R>,
    style_dim: usize,
) -> Result<(Tensor<R>, Tensor<R>)> {
    let shape = style_row.shape();
    if shape.len() != 2 || shape[1] != 2 * style_dim {
        return Err(Error::ModelError {
            reason: format!(
                "style row shape must be [B, {}], got {shape:?}",
                2 * style_dim
            ),
        });
    }
    let decoder = style_row
        .narrow(1, 0, style_dim)
        .map_err(|e| Error::ModelError {
            reason: format!("narrow decoder style: {e}"),
        })?
        .contiguous();
    let predictor = style_row
        .narrow(1, style_dim, style_dim)
        .map_err(|e| Error::ModelError {
            reason: format!("narrow predictor style: {e}"),
        })?
        .contiguous();
    Ok((decoder, predictor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_spec() {
        let r = VoiceResolver::new();
        assert!(r.resolve_path("").is_err());
    }

    #[test]
    fn path_form_requires_existing_file() {
        let r = VoiceResolver::new();
        assert!(
            r.resolve_path("/nonexistent-voice-xyz.safetensors")
                .is_err()
        );
    }

    #[test]
    fn id_form_searches_asset_dir() {
        let tmp = std::env::temp_dir().join("boostr_voice_resolver_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("af_alloy.safetensors"), b"").unwrap();

        let r = VoiceResolver::new().with_asset_dir(&tmp);
        let resolved = r.resolve_path("af_alloy").unwrap();
        assert_eq!(resolved, tmp.join("af_alloy.safetensors"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn id_form_prefers_safetensors_over_pt() {
        let tmp = std::env::temp_dir().join("boostr_voice_resolver_pref_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("af.safetensors"), b"").unwrap();
        std::fs::write(tmp.join("af.pt"), b"").unwrap();

        let r = VoiceResolver::new().with_asset_dir(&tmp);
        let resolved = r.resolve_path("af").unwrap();
        assert!(resolved.to_string_lossy().ends_with("af.safetensors"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn id_form_falls_back_to_pt() {
        let tmp = std::env::temp_dir().join("boostr_voice_resolver_pt_fallback");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::write(tmp.join("af.pt"), b"").unwrap();

        let r = VoiceResolver::new().with_asset_dir(&tmp);
        let resolved = r.resolve_path("af").unwrap();
        assert!(resolved.to_string_lossy().ends_with("af.pt"));

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn unknown_id_reports_searched_dirs() {
        let r = VoiceResolver::new();
        let err = r.resolve_path("af_nope").unwrap_err();
        match err {
            Error::ModelError { reason } => assert!(reason.contains("af_nope")),
            _ => panic!("wrong error variant"),
        }
    }

    #[test]
    fn select_voice_style_clamps_to_last_row() {
        use numr::runtime::cpu::{CpuDevice, CpuRuntime};
        let device = CpuDevice::new();
        // pack [3, 1, 4]: row 0 = 1s, row 1 = 2s, row 2 = 3s
        let data: Vec<f32> = (0..3).flat_map(|r| vec![(r + 1) as f32; 4]).collect();
        let pack = Tensor::<CpuRuntime>::from_slice(&data, &[3, 1, 4], &device);
        let picked = select_voice_style(&pack, 2).unwrap();
        assert_eq!(picked.shape(), &[1, 4]);
        let v: Vec<f32> = picked.to_vec();
        assert_eq!(v, vec![2.0, 2.0, 2.0, 2.0]);
        // Out-of-range phoneme count clamps to the last row.
        let last = select_voice_style(&pack, 100).unwrap();
        let lv: Vec<f32> = last.to_vec();
        assert_eq!(lv, vec![3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn split_voice_style_halves_match() {
        use numr::runtime::cpu::{CpuDevice, CpuRuntime};
        let device = CpuDevice::new();
        let row = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1, 8],
            &device,
        );
        let (dec, pred) = split_voice_style(&row, 4).unwrap();
        assert_eq!(dec.shape(), &[1, 4]);
        assert_eq!(pred.shape(), &[1, 4]);
        let d: Vec<f32> = dec.to_vec();
        let p: Vec<f32> = pred.to_vec();
        assert_eq!(d, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(p, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn select_voice_style_rejects_bad_rank() {
        use numr::runtime::cpu::{CpuDevice, CpuRuntime};
        let device = CpuDevice::new();
        let bad = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 4], &[4], &device);
        assert!(select_voice_style(&bad, 0).is_err());
    }

    #[test]
    fn split_voice_style_rejects_wrong_width() {
        use numr::runtime::cpu::{CpuDevice, CpuRuntime};
        let device = CpuDevice::new();
        let row = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[1, 6], &device);
        // style_dim = 4 → expected [1, 8] but we have [1, 6]
        assert!(split_voice_style(&row, 4).is_err());
    }

    #[test]
    fn path_with_slash_is_treated_as_path() {
        let tmp = std::env::temp_dir().join("boostr_voice_resolver_path_form");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        let f = tmp.join("custom.safetensors");
        std::fs::write(&f, b"").unwrap();

        let r = VoiceResolver::new();
        let resolved = r.resolve_path(f.to_str().unwrap()).unwrap();
        assert_eq!(resolved, f);

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
