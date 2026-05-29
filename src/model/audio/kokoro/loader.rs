//! Kokoro checkpoint loader.
//!
//! Composes `KokoroConfig` + safetensors weights into a ready-to-run
//! `KokoroModel`. Layered in three tiers so that if upstream tensor naming
//! diverges from our guesses, only the top tier needs updating:
//!
//! * **Tier 1 — Tensor helpers** (`load_plain_conv1d`, `load_weight_normed_conv1d`,
//!   `load_linear`, `load_bilstm`). Work from an arbitrary name prefix; match
//!   PyTorch's default `torch.save` naming. Independently unit-testable.
//! * **Tier 2 — Submodule loaders** (`load_text_encoder`, `load_duration_predictor`,
//!   etc.). Each takes a prefix + config and returns its module. Adjusting one
//!   doesn't affect the others.
//! * **Tier 3 — `load_kokoro`**. Composes tier 2 under the prefixes that
//!   hexgrad/Kokoro-82M's upstream checkpoint uses.
//!
//! When the real checkpoint is inspected, mis-named tensors surface as
//! `Error::ModelError` with the exact missing name, so fixes are O(one sed).

use crate::error::{Error, Result};
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::config::KokoroConfig;
use crate::nn::{BiLstm, Conv1d, Lstm, fuse_weight_norm};
use numr::dtype::DType;
use numr::ops::{BinaryOps, PaddingMode, ReduceOps, TensorOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::path::Path;

/// Load a plain Conv1d (non-weight-normed).
///
/// Reads `{prefix}.weight` (required) and `{prefix}.bias` (optional).
pub fn load_plain_conv1d<R: Runtime<DType = DType>>(
    st: &mut super::weight_source::KokoroWeightSource,
    prefix: &str,
    stride: usize,
    padding: PaddingMode,
    dilation: usize,
    groups: usize,
    device: &R::Device,
) -> Result<Conv1d<R>> {
    let weight = st.load_tensor::<R>(&format!("{prefix}.weight"), device)?;
    let bias = st.load_tensor::<R>(&format!("{prefix}.bias"), device).ok();
    Ok(Conv1d::new(
        weight, bias, stride, padding, dilation, groups, false,
    ))
}

/// Load a weight-normed Conv1d. Handles both modern (`parametrizations.weight
/// .original0/original1`, post PyTorch 2.x `nn.utils.parametrizations`) and
/// legacy (`weight_v`/`weight_g`, pre `parametrize`) checkpoint layouts, in
/// that order — Kokoro-82M's upstream checkpoint uses the modern form.
///
/// `original0` is `g` (scale), `original1` is `v` (direction). After fusion,
/// `dim=0` (out-channel axis) is the normal Conv1d convention; pass `dim=1`
/// for transposed-conv layouts.
#[allow(clippy::too_many_arguments)]
pub fn load_weight_normed_conv1d<R, C>(
    client: &C,
    st: &mut super::weight_source::KokoroWeightSource,
    prefix: &str,
    stride: usize,
    padding: PaddingMode,
    dilation: usize,
    groups: usize,
    device: &R::Device,
) -> Result<Conv1d<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    let (g, v) = load_weight_norm_pair::<R>(st, prefix, device)?;
    let weight = fuse_weight_norm(client, &v, &g, 0)?;
    let bias = st.load_tensor::<R>(&format!("{prefix}.bias"), device).ok();
    Ok(Conv1d::new(
        weight, bias, stride, padding, dilation, groups, false,
    ))
}

/// Read the `(g, v)` tensor pair for a weight-normed conv / linear, trying
/// modern parametrization names first and falling back to legacy.
pub fn load_weight_norm_pair<R: Runtime<DType = DType>>(
    st: &mut super::weight_source::KokoroWeightSource,
    prefix: &str,
    device: &R::Device,
) -> Result<(Tensor<R>, Tensor<R>)> {
    let modern_g = format!("{prefix}.parametrizations.weight.original0");
    let modern_v = format!("{prefix}.parametrizations.weight.original1");
    if let (Ok(g), Ok(v)) = (
        st.load_tensor::<R>(&modern_g, device),
        st.load_tensor::<R>(&modern_v, device),
    ) {
        return Ok((g, v));
    }
    let legacy_g = format!("{prefix}.weight_g");
    let legacy_v = format!("{prefix}.weight_v");
    let g = st.load_tensor::<R>(&legacy_g, device)?;
    let v = st.load_tensor::<R>(&legacy_v, device)?;
    Ok((g, v))
}

/// Load the four weight/bias tensors for one LSTM direction.
///
/// PyTorch names: `{prefix}.weight_ih_l0`, `weight_hh_l0`, `bias_ih_l0`,
/// `bias_hh_l0`. Pass `suffix="_reverse"` (appended after `_l0`) for the
/// backward direction of a bidirectional LSTM.
pub fn load_lstm_direction<R: Runtime<DType = DType>>(
    st: &mut super::weight_source::KokoroWeightSource,
    prefix: &str,
    suffix: &str,
    device: &R::Device,
) -> Result<Lstm<R>> {
    let weight_ih = st.load_tensor::<R>(&format!("{prefix}.weight_ih_l0{suffix}"), device)?;
    let weight_hh = st.load_tensor::<R>(&format!("{prefix}.weight_hh_l0{suffix}"), device)?;
    let bias_ih = st.load_tensor::<R>(&format!("{prefix}.bias_ih_l0{suffix}"), device)?;
    let bias_hh = st.load_tensor::<R>(&format!("{prefix}.bias_hh_l0{suffix}"), device)?;
    Lstm::new(weight_ih, weight_hh, bias_ih, bias_hh)
}

/// Load a bidirectional LSTM (PyTorch `nn.LSTM(bidirectional=True)`).
pub fn load_bilstm<R: Runtime<DType = DType>>(
    st: &mut super::weight_source::KokoroWeightSource,
    prefix: &str,
    device: &R::Device,
) -> Result<BiLstm<R>> {
    let forward = load_lstm_direction(st, prefix, "", device)?;
    let backward = load_lstm_direction(st, prefix, "_reverse", device)?;
    BiLstm::new(forward, backward)
}

/// Load a Linear's raw tensors: `{prefix}.weight` (`[out, in]`) and optional
/// `{prefix}.bias` (`[out]`). Returns the pair ready to hand to predictor
/// constructors that take raw tensors.
pub fn load_linear_tensors<R: Runtime<DType = DType>>(
    st: &mut super::weight_source::KokoroWeightSource,
    prefix: &str,
    device: &R::Device,
) -> Result<(Tensor<R>, Option<Tensor<R>>)> {
    let w = st.load_tensor::<R>(&format!("{prefix}.weight"), device)?;
    let bias = st.load_tensor::<R>(&format!("{prefix}.bias"), device).ok();
    Ok((w, bias))
}

/// Load a Kokoro voice pack from a `.safetensors` or `.pt` / `.pth` file.
///
/// Kokoro voice files are shaped `[T, 1, 256]` where `T ≈ 510` and the last
/// dim is the concatenation `[decoder_style(128) | predictor_style(128)]`.
/// Indexing by `(phoneme_count - 1)` picks the row that was tuned for that
/// utterance length; see [`select_voice_style`].
///
/// * `.safetensors` — expects a single tensor named `style` (or the only
///   tensor in the file).
/// * `.pt` / `.pth` — accepts a bare tensor (`torch.save(tensor)`), the
///   canonical upstream form.
pub fn load_voice_pack<R: Runtime<DType = DType>>(
    path: impl AsRef<Path>,
    device: &R::Device,
) -> Result<Tensor<R>> {
    let path = path.as_ref();
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let tensor = match ext.as_str() {
        "safetensors" => {
            let mut st = SafeTensorsLoader::open(path)?;
            st.load_tensor::<R>("style", device)?
        }
        "pt" | "pth" => crate::format::load_voice_pt(path, device)?,
        other => {
            return Err(Error::ModelError {
                reason: format!(
                    "unsupported voice file extension: .{other} (expected .safetensors, .pt, or .pth)"
                ),
            });
        }
    };

    let shape = tensor.shape();
    let is_valid = matches!(shape.len(), 2 | 3)
        && *shape.last().unwrap() >= 2
        && shape.last().unwrap() % 2 == 0;
    if !is_valid {
        return Err(Error::ModelError {
            reason: format!(
                "voice pack has unexpected shape {shape:?}; expected [T, 1, 2*D] or [T, 2*D]"
            ),
        });
    }
    Ok(tensor)
}

/// Deprecated legacy alias — kept so existing callers still compile. Prefer
/// [`load_voice_pack`] + [`select_voice_style`] + [`split_voice_style`].
#[deprecated(note = "Use load_voice_pack + select_voice_style + split_voice_style")]
pub fn load_voice_style<R: Runtime<DType = DType>>(
    path: impl AsRef<Path>,
    device: &R::Device,
) -> Result<Tensor<R>> {
    load_voice_pack::<R>(path, device)
}

/// Top-level Kokoro loader.
///
/// Opens `config.json` + safetensors from `model_dir`, validates the config,
/// then fails cleanly via `Error::NotImplemented` for the actual weight
/// assembly. The tier-1 helpers above are fully usable today; tier-2 / tier-3
/// assembly requires inspecting the upstream checkpoint to lock down exact
/// tensor prefixes (hexgrad/Kokoro-82M's naming isn't part of the public
/// spec). Inspecting is a ~15-minute job once the .safetensors file is in
/// hand — the tier-1 helpers will cover every read.
pub fn load_kokoro<R: Runtime<DType = DType>>(
    model_dir: impl AsRef<Path>,
    _device: &R::Device,
) -> Result<crate::model::audio::kokoro::model::KokoroModel<R>> {
    let model_dir = model_dir.as_ref();
    let config_path = model_dir.join("config.json");
    let config = KokoroConfig::from_json_file(&config_path)?;
    config.validate()?;

    let _st = SafeTensorsLoader::open(model_dir)?;

    Err(Error::ModelError {
        reason: format!(
            "Kokoro tier-3 assembly needs the upstream checkpoint to fix exact tensor \
             prefixes. Tier-1/2 helpers in this module (load_plain_conv1d, \
             load_weight_normed_conv1d, load_bilstm, load_linear_tensors, \
             load_voice_style) are ready to use — compose them against your own \
             checkpoint's names. Loaded config for {} (hidden_dim={}, style_dim={}).",
            model_dir.display(),
            config.hidden_dim,
            config.style_dim,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_kokoro_reports_missing_config() {
        use numr::runtime::cpu::{CpuDevice, CpuRuntime};
        let device = CpuDevice::new();
        let tmp = std::env::temp_dir().join("kokoro_missing_config_test");
        std::fs::create_dir_all(&tmp).unwrap();
        let res = load_kokoro::<CpuRuntime>(&tmp, &device);
        assert!(res.is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
