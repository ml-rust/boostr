//! Load a super-resolution network from a model directory.
//!
//! The directory holds exactly one `.pth`. The key set of its state dict
//! picks the network, and the shapes of a few tensors give the config. The
//! detection reads key names and tensor shapes from the pickle metadata. It
//! reads no weights.
//!
//! | Key set | Network |
//! | --- | --- |
//! | `body.{2i}` convs with `body.{2i+1}` PReLUs | SRVGGNetCompact |
//! | `conv_first`, `body.{b}.rdb{d}.conv{c}`, `conv_body`, `conv_up1`, `conv_up2`, `conv_hr`, `conv_last` | RRDBNet |
//! | `model.0`, `model.1.sub.{b}.RDB{d}.conv{c}.0`, `model.3`, `model.6`, `model.8`, `model.10` | RRDBNet, legacy ESRGAN keys |

use crate::error::{Error, Result};
use crate::format::TorchPthSource;
use crate::model::super_resolution::client::SrClient;
use crate::model::super_resolution::compact::SrVggCompactConfig;
use crate::model::super_resolution::layout::SrExecution;
use crate::model::super_resolution::loader::{load_compact, load_rrdb};
use crate::model::super_resolution::model::SrModel;
use crate::model::super_resolution::rrdb::{LEGACY_ESRGAN_MARKER, RrdbConfig};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::path::{Path, PathBuf};

/// Keys listed in an error before the count of the rest.
const SAMPLE_KEYS: usize = 8;

/// Extension of a checkpoint file.
const CHECKPOINT_EXTENSION: &str = "pth";

/// Marker key of an RRDBNet state dict with current keys.
const RRDB_MARKER: &str = "conv_first.weight";

/// Marker key of an SRVGGNetCompact state dict.
const COMPACT_MARKER: &str = "body.0.weight";

/// Kernel side of every conv in both networks.
const KERNEL: usize = 3;

/// The network and config a state dict holds.
enum Detected {
    Compact(SrVggCompactConfig),
    Rrdb(RrdbConfig),
}

/// Load the one `.pth` in `dir` as the network its keys name.
///
/// The key set picks SRVGGNetCompact, RRDBNet with current keys, or RRDBNet
/// with legacy ESRGAN `model.N` keys. The config comes from tensor shapes:
///
/// - Compact: `num_conv` counts the `body.N` convs. `num_out_ch` equals
///   `num_in_ch`, and `upscale` is the square root of the last conv's
///   channels over `num_out_ch`.
/// - RRDBNet: `num_in_ch` is the first conv's input channels over 4.
///   `num_block` counts the RRDB blocks. `scale` is 2.
///
/// [`load_compact`] or [`load_rrdb`] then loads strictly, so a key or shape
/// the config does not explain is an error. Weights load onto `device`,
/// cast to `exec.dtype`, and permute to `exec.layout` once.
///
/// # Errors
///
/// - `dir` does not list, or holds no `.pth` or several. The error names
///   `dir` and the `.pth` files found.
/// - The key set matches no network. The error names `dir` and sample keys.
/// - A shape is not a valid network shape. The error names the file, the
///   key and the shape.
/// - The errors of [`load_compact`] and [`load_rrdb`].
pub fn load_sr_dir<R>(dir: &Path, device: &R::Device, exec: SrExecution) -> Result<SrModel<R>>
where
    R: Runtime<DType = DType>,
    R::Client: SrClient<R>,
{
    let path = single_checkpoint(dir)?;
    match detect(dir, &path)? {
        Detected::Compact(config) => load_compact::<R>(&path, device, &config, exec)
            .map(|net| SrModel::Compact(Box::new(net))),
        Detected::Rrdb(config) => {
            load_rrdb::<R>(&path, device, &config, exec).map(|net| SrModel::Rrdb(Box::new(net)))
        }
    }
}

/// The only `.pth` file in `dir`.
fn single_checkpoint(dir: &Path) -> Result<PathBuf> {
    let list_error = |e: std::io::Error| Error::ModelError {
        reason: format!(
            "{}: cannot list the model directory: {e}. Pass a readable directory",
            dir.display()
        ),
    };
    let mut found: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(list_error)? {
        let path = entry.map_err(list_error)?.path();
        let is_checkpoint = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case(CHECKPOINT_EXTENSION));
        if is_checkpoint && path.is_file() {
            found.push(path);
        }
    }
    found.sort_unstable();
    if found.len() == 1 {
        return Ok(found.remove(0));
    }
    let names: Vec<String> = found
        .iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect();
    let listed = if names.is_empty() {
        "none".to_string()
    } else {
        names.join(", ")
    };
    Err(Error::ModelError {
        reason: format!(
            "{}: expected exactly one .pth file, found {}: {listed}. \
             Keep one checkpoint in the directory",
            dir.display(),
            names.len()
        ),
    })
}

/// Pick the network of `path` from its key set and read its config from
/// tensor shapes.
fn detect(dir: &Path, path: &Path) -> Result<Detected> {
    let source = TorchPthSource::open(path)?;
    let names: Vec<&str> = source.names().collect();
    let has = |key: &str| names.contains(&key);
    if has(LEGACY_ESRGAN_MARKER) {
        let shapes = Shapes::new(path, &source);
        let blocks = count_blocks(&names, "model.1.sub.", "RDB1.conv1.0.weight");
        let first = shapes.conv(LEGACY_ESRGAN_MARKER)?;
        let last = shapes.conv("model.10.weight")?;
        let grow = shapes.conv("model.1.sub.0.RDB1.conv1.0.weight")?;
        return rrdb_config(path, &first, &last, &grow, blocks).map(Detected::Rrdb);
    }
    if has(RRDB_MARKER) {
        let shapes = Shapes::new(path, &source);
        let blocks = count_blocks(&names, "body.", "rdb1.conv1.weight");
        let first = shapes.conv(RRDB_MARKER)?;
        let last = shapes.conv("conv_last.weight")?;
        let grow = shapes.conv("body.0.rdb1.conv1.weight")?;
        return rrdb_config(path, &first, &last, &grow, blocks).map(Detected::Rrdb);
    }
    if has(COMPACT_MARKER) {
        let shapes = Shapes::new(path, &source);
        let convs = count_compact_convs(&names);
        let first = shapes.conv(COMPACT_MARKER)?;
        let last = shapes.conv(&format!("body.{}.weight", 2 * convs.saturating_sub(1)))?;
        return compact_config(path, &first, &last, convs).map(Detected::Compact);
    }
    Err(Error::ModelError {
        reason: format!(
            "{}: the keys of {} match no super-resolution network. Sample keys: {}. \
             Use an SRVGGNetCompact or RRDBNet x2 checkpoint",
            dir.display(),
            path.display(),
            sample_keys(&names)
        ),
    })
}

/// Output channels, input channels of one 3x3 conv, as `[out, in]`.
type ConvChannels = [usize; 2];

/// Tensor shapes of one state dict, read without the weights.
struct Shapes<'a> {
    path: &'a Path,
    source: &'a TorchPthSource,
}

impl<'a> Shapes<'a> {
    fn new(path: &'a Path, source: &'a TorchPthSource) -> Self {
        Self { path, source }
    }

    /// The channels of conv weight `name`, which must be `[out, in, 3, 3]`.
    fn conv(&self, name: &str) -> Result<ConvChannels> {
        let stored = self.source.stored_key(name);
        let shape = self.source.state_dict().shape(&stored);
        match shape {
            Some(&[out, inp, KERNEL, KERNEL]) => Ok([out, inp]),
            _ => Err(Error::ModelError {
                reason: format!(
                    "{}: key '{stored}' has shape {shape:?}, expected [out, in, {KERNEL}, {KERNEL}]. \
                     Use an SRVGGNetCompact or RRDBNet x2 checkpoint",
                    self.path.display()
                ),
            }),
        }
    }
}

/// The compact config for a first conv, a last conv and `convs` convs.
fn compact_config(
    path: &Path,
    first: &ConvChannels,
    last: &ConvChannels,
    convs: usize,
) -> Result<SrVggCompactConfig> {
    let shape_error = |why: String| Error::ModelError {
        reason: format!(
            "{}: cannot read an SRVGGNetCompact config: {why}. \
             Use a checkpoint with the same input and output channels",
            path.display()
        ),
    };
    if convs < 2 {
        return Err(shape_error(format!(
            "found {convs} body convs, need at least 2"
        )));
    }
    let [num_feat, num_in_ch] = *first;
    let [last_out, _] = *last;
    let num_out_ch = num_in_ch;
    let per_channel = last_out / num_out_ch.max(1);
    let upscale = per_channel.isqrt();
    if upscale == 0 || upscale * upscale * num_out_ch != last_out {
        return Err(shape_error(format!(
            "the last conv has {last_out} output channels, which is not {num_out_ch} x upscale^2"
        )));
    }
    Ok(SrVggCompactConfig {
        num_in_ch,
        num_out_ch,
        num_feat,
        num_conv: convs - 2,
        upscale,
    })
}

/// The RRDBNet x2 config for a first conv, a last conv, a dense block's
/// first conv and `blocks` RRDB blocks.
fn rrdb_config(
    path: &Path,
    first: &ConvChannels,
    last: &ConvChannels,
    grow: &ConvChannels,
    blocks: usize,
) -> Result<RrdbConfig> {
    let [num_feat, unshuffled] = *first;
    let config = RrdbConfig {
        num_in_ch: unshuffled / 4,
        num_out_ch: last[0],
        num_feat,
        num_block: blocks,
        num_grow_ch: grow[0],
        scale: 2,
    };
    if unshuffled % 4 != 0 || blocks == 0 {
        return Err(Error::ModelError {
            reason: format!(
                "{}: cannot read an RRDBNet config: the first conv has {unshuffled} input channels \
                 and the state dict has {blocks} blocks. \
                 Use an RRDBNet x2 checkpoint with a pixel-unshuffle head",
                path.display()
            ),
        });
    }
    Ok(config)
}

/// Count the `body.N.weight` names with an even `N`.
fn count_compact_convs(names: &[&str]) -> usize {
    names
        .iter()
        .filter_map(|n| n.strip_prefix("body.")?.strip_suffix(".weight"))
        .filter_map(|n| n.parse::<usize>().ok())
        .filter(|n| n % 2 == 0)
        .count()
}

/// Count the names `{prefix}{block}.{suffix}` with a decimal `block`.
fn count_blocks(names: &[&str], prefix: &str, suffix: &str) -> usize {
    names
        .iter()
        .filter_map(|n| n.strip_prefix(prefix)?.split_once('.'))
        .filter(|(block, rest)| {
            *rest == suffix && !block.is_empty() && block.bytes().all(|b| b.is_ascii_digit())
        })
        .count()
}

/// The first sorted `names`, with the count of the rest.
fn sample_keys(names: &[&str]) -> String {
    let mut sorted = names.to_vec();
    sorted.sort_unstable();
    if sorted.is_empty() {
        return "none".to_string();
    }
    let shown = sorted
        .iter()
        .take(SAMPLE_KEYS)
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    match sorted.len().saturating_sub(SAMPLE_KEYS) {
        0 => format!("[{shown}]"),
        rest => format!("[{shown}, +{rest} more]"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_compact_convs_not_prelus() {
        let names = [
            "body.0.weight",
            "body.0.bias",
            "body.1.weight",
            "body.2.weight",
            "body.2.bias",
        ];
        assert_eq!(count_compact_convs(&names), 2);
    }

    #[test]
    fn counts_blocks_by_suffix() {
        let names = [
            "body.0.rdb1.conv1.weight",
            "body.1.rdb1.conv1.weight",
            "body.1.rdb1.conv2.weight",
            "conv_first.weight",
        ];
        assert_eq!(count_blocks(&names, "body.", "rdb1.conv1.weight"), 2);
    }

    #[test]
    fn compact_config_reads_upscale_from_last_conv() {
        let config = compact_config(Path::new("m.pth"), &[64, 3], &[48, 64], 34)
            .map(|c| (c.num_conv, c.upscale, c.num_feat));
        assert_eq!(config.ok(), Some((32, 4, 64)));
    }

    #[test]
    fn compact_config_rejects_non_square_channels() {
        let err = compact_config(Path::new("m.pth"), &[64, 3], &[40, 64], 34)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("m.pth") && err.contains("40"), "{err}");
    }

    #[test]
    fn rrdb_config_divides_unshuffled_channels() {
        let config = rrdb_config(Path::new("m.pth"), &[64, 12], &[3, 64], &[32, 64], 23)
            .map(|c| (c.num_in_ch, c.num_block, c.num_grow_ch));
        assert_eq!(config.ok(), Some((3, 23, 32)));
        assert!(rrdb_config(Path::new("m.pth"), &[64, 13], &[3, 64], &[32, 64], 23).is_err());
    }

    #[test]
    fn sample_keys_truncate() {
        let many: Vec<String> = (0..10).map(|i| format!("k{i}")).collect();
        let refs: Vec<&str> = many.iter().map(String::as_str).collect();
        assert!(sample_keys(&refs).ends_with("k7, +2 more]"));
    }
}
