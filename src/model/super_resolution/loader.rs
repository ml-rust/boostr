//! Super-resolution checkpoint loading from a `torch.save`d state dict.
//!
//! The load is strict, as PyTorch's `load_state_dict(strict=True)`: a
//! missing key, an unexpected key and a shape mismatch are each an error.

use crate::error::{Error, Result};
use crate::format::TorchPthSource;
use crate::model::super_resolution::client::SrClient;
use crate::model::super_resolution::compact::{SrVggCompact, SrVggCompactConfig};
use crate::model::super_resolution::layout::SrExecution;
use crate::model::super_resolution::rrdb::{
    RrdbConfig, RrdbNet, check_esrgan_metadata, is_legacy_esrgan, legacy_esrgan_renames,
};
use crate::nn::{VarBuilder, VarMap};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::collections::{BTreeSet, HashMap};
use std::path::Path;

/// Keys listed per category in a key-set error before the count of the rest.
const LISTED_KEYS: usize = 8;

/// Load an SRVGGNetCompact `.pth`, such as `realesr-general-x4v3.pth`.
///
/// [`TorchPthSource::open`] detects the `params_ema.`, `params.` or
/// `state_dict.` root. The tensor keys under that root must equal
/// [`SrVggCompactConfig::expected_keys`]. Keys outside the root, such as
/// another wrapper's tensors, are not read. Weights load onto `device`, cast
/// to `exec.dtype`, and permute to `exec.layout` once.
///
/// # Errors
///
/// - The file does not open or parse. The error names the path.
/// - A key is missing or unexpected. The error names the path, the root and
///   the keys.
/// - A weight has the wrong shape. The error names the path, the key and
///   both shapes.
pub fn load_compact<R>(
    path: impl AsRef<Path>,
    device: &R::Device,
    config: &SrVggCompactConfig,
    exec: SrExecution,
) -> Result<SrVggCompact<R>>
where
    R: Runtime<DType = DType>,
    R::Client: SrClient<R>,
{
    let path = path.as_ref();
    config.validate()?;
    let mut source = TorchPthSource::open(path)?;
    let expected = config.expected_keys();
    check_key_set(path, source.root(), &expected, source.names())?;

    let mut map = read_weights::<R>(path, &mut source, &expected, &HashMap::new(), device)?;
    let client = R::default_client(device);
    let mut vb = VarBuilder::new(&mut map, device);
    SrVggCompact::from_varbuilder(&client, &mut vb, config, exec).map_err(|e| in_file(path, e))
}

/// Load an RRDBNet x2 `.pth`, such as `RealESRGAN_x2plus.pth` or
/// `2x_PureVision.pth`.
///
/// [`TorchPthSource::open`] detects the `params_ema.`, `params.` or
/// `state_dict.` root. A state dict that holds
/// [`crate::model::super_resolution::rrdb::LEGACY_ESRGAN_MARKER`] reads
/// through the legacy ESRGAN key remap of
/// [`crate::model::super_resolution::rrdb::keys`]. The tensor keys under the
/// root, after any remap, must equal [`RrdbConfig::expected_keys`]. The
/// `metadata.*` scalars, when present, must agree with `config`. Weights
/// load onto `device`, cast to `exec.dtype`, and permute to `exec.layout`
/// once.
///
/// # Errors
///
/// - The config is invalid, or its scale is not 2.
/// - The file does not open or parse. The error names the path.
/// - A `metadata.*` field disagrees with `config`. The error names the path
///   and the field.
/// - A legacy key is unknown, or two legacy keys map to one name. The error
///   names the path and the keys.
/// - A key is missing or unexpected. The error names the path, the root and
///   the keys.
/// - A weight has the wrong shape. The error names the path, the key and
///   both shapes.
pub fn load_rrdb<R>(
    path: impl AsRef<Path>,
    device: &R::Device,
    config: &RrdbConfig,
    exec: SrExecution,
) -> Result<RrdbNet<R>>
where
    R: Runtime<DType = DType>,
    R::Client: SrClient<R>,
{
    let path = path.as_ref();
    config.validate()?;
    let mut source = TorchPthSource::open(path)?;
    check_esrgan_metadata(source.state_dict(), config).map_err(|e| in_file(path, e))?;
    let expected = config.expected_keys();

    let (stored, renames) = if is_legacy_esrgan(source.names()) {
        let renames = legacy_esrgan_renames(source.names(), config.num_block)
            .map_err(|e| in_file(path, e))?;
        let root = format!("{} (legacy ESRGAN keys, remapped)", source.root());
        check_key_set(path, &root, &expected, renames.values().map(String::as_str))?;
        let mut stored: Vec<String> = renames.keys().cloned().collect();
        stored.sort_unstable();
        (stored, renames)
    } else {
        check_key_set(path, source.root(), &expected, source.names())?;
        (expected, HashMap::new())
    };

    let mut map = read_weights::<R>(path, &mut source, &stored, &renames, device)?;
    let client = R::default_client(device);
    let mut vb = VarBuilder::new(&mut map, device);
    RrdbNet::from_varbuilder(&client, &mut vb, config, exec).map_err(|e| in_file(path, e))
}

/// Read every key of `stored` from `source` into a [`VarMap`], under its
/// name in `renames`, else under its own name.
fn read_weights<R>(
    path: &Path,
    source: &mut TorchPthSource,
    stored: &[String],
    renames: &HashMap<String, String>,
    device: &R::Device,
) -> Result<VarMap<R>>
where
    R: Runtime<DType = DType>,
    R::Client: SrClient<R>,
{
    let rename = |name: &str| {
        renames
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.to_string())
    };
    VarMap::<R>::from_weight_source(source, stored, rename, device).map_err(|e| Error::ModelError {
        reason: format!("{}: reading weights: {e}", path.display()),
    })
}

/// `e` with the checkpoint path in front.
fn in_file(path: &Path, e: Error) -> Error {
    Error::ModelError {
        reason: format!("{}: {e}", path.display()),
    }
}

/// Fail unless `found` holds exactly the `expected` keys.
fn check_key_set<'a>(
    path: &Path,
    root: &str,
    expected: &[String],
    found: impl Iterator<Item = &'a str>,
) -> Result<()> {
    let found: BTreeSet<&str> = found.collect();
    let wanted: BTreeSet<&str> = expected.iter().map(String::as_str).collect();
    let missing: Vec<&str> = wanted.difference(&found).copied().collect();
    let unexpected: Vec<&str> = found.difference(&wanted).copied().collect();
    if missing.is_empty() && unexpected.is_empty() {
        return Ok(());
    }
    Err(Error::ModelError {
        reason: format!(
            "{}: state dict under root '{root}' does not match the network config: \
             missing {}, unexpected {}; pass the config the checkpoint was trained with",
            path.display(),
            list_keys(&missing),
            list_keys(&unexpected)
        ),
    })
}

fn list_keys(keys: &[&str]) -> String {
    if keys.is_empty() {
        return "none".to_string();
    }
    let shown = keys
        .iter()
        .take(LISTED_KEYS)
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    match keys.len().saturating_sub(LISTED_KEYS) {
        0 => format!("[{shown}]"),
        rest => format!("[{shown}, +{rest} more]"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn exact_key_set_passes() {
        let expected = keys(&["body.0.weight", "body.0.bias"]);
        let found = ["body.0.bias", "body.0.weight"];
        assert!(check_key_set(Path::new("m.pth"), "params.", &expected, found.into_iter()).is_ok());
    }

    #[test]
    fn missing_and_unexpected_keys_are_named() {
        let expected = keys(&["body.0.weight", "body.0.bias"]);
        let found = ["body.0.weight", "body.9.weight"];
        let err = check_key_set(Path::new("m.pth"), "params.", &expected, found.into_iter())
            .err()
            .map(|e| e.to_string());
        let err = err.unwrap_or_default();
        assert!(err.contains("m.pth"), "{err}");
        assert!(err.contains("missing [body.0.bias]"), "{err}");
        assert!(err.contains("unexpected [body.9.weight]"), "{err}");
    }

    #[test]
    fn long_key_lists_are_truncated() {
        let many: Vec<String> = (0..12).map(|i| format!("k{i:02}")).collect();
        let refs: Vec<&str> = many.iter().map(String::as_str).collect();
        let listed = list_keys(&refs);
        assert!(listed.ends_with("k07, +4 more]"), "{listed}");
    }
}
