//! [`VaeCheckpoint`]: the `AudioVAE`'s weight source, whichever of the two
//! containers it ships in.
//!
//! VoxCPM2 publishes the `AudioVAE` as `audiovae.pth` — a torch pickle whose
//! convolutions are still `weight_norm`-reparameterized. Earlier revisions of
//! this port required that file be converted to `audiovae.safetensors` by an
//! external Python script first; the `.pth` is now read directly (see
//! [`TorchPthSource`]), and the converted file keeps working unchanged for
//! anyone who already has one.
//!
//! Which container a path holds is decided by its BYTES, not its extension: a
//! name is a hint and gets renamed, while the magic is what the file is.

use crate::error::{Error, Result};
use crate::format::safetensors_loader::SafeTensorsLoader;
use crate::model::audio::voxcpm::loader::support::{TorchPthSource, WeightSource};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ReduceOps, TensorOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::io::Read;
use std::path::{Path, PathBuf};

/// File name the reference repo publishes the unconverted `AudioVAE` under.
const AUDIOVAE_PTH: &str = "audiovae.pth";

/// Local ZIP header magic. Every `torch.save` file since PyTorch 1.6 is a ZIP
/// archive, so this is what separates a `.pth` from a safetensors file, whose
/// first eight bytes are instead a little-endian JSON header length.
const ZIP_MAGIC: [u8; 4] = [b'P', b'K', 0x03, 0x04];

/// An open `AudioVAE` checkpoint, in whichever container it was shipped.
pub enum VaeCheckpoint {
    /// An already weight-norm-folded `audiovae.safetensors`.
    SafeTensors(SafeTensorsLoader),
    /// The published `audiovae.pth`, folded as it is read.
    TorchPth(TorchPthSource),
}

impl VaeCheckpoint {
    /// Open the `AudioVAE` weights at `path`.
    ///
    /// `path` may be the checkpoint file itself or a directory holding it. A
    /// directory keeps preferring safetensors, so a tree that already holds a
    /// converted file loads exactly the tensors it did before; `audiovae.pth`
    /// is used only when the directory has no safetensors at all.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            let pth = path.join(AUDIOVAE_PTH);
            if pth.is_file() && !dir_has_safetensors(path)? {
                return Ok(Self::TorchPth(TorchPthSource::open(pth)?));
            }
            return Ok(Self::SafeTensors(SafeTensorsLoader::open(path)?));
        }

        if is_zip(path)? {
            Ok(Self::TorchPth(TorchPthSource::open(path)?))
        } else {
            Ok(Self::SafeTensors(SafeTensorsLoader::open(path)?))
        }
    }
}

impl<R: Runtime<DType = DType>> WeightSource<R> for VaeCheckpoint
where
    R::Client: ReduceOps<R> + UnaryOps<R> + BinaryOps<R> + TensorOps<R>,
{
    fn load_named(&mut self, name: &str, device: &R::Device) -> Result<Tensor<R>> {
        match self {
            Self::SafeTensors(loader) => loader.load_named(name, device),
            Self::TorchPth(source) => source.load_named(name, device),
        }
    }
}

/// Does `dir` hold any `.safetensors` file? Mirrors the extension filter
/// [`SafeTensorsLoader::open`] itself applies to a directory, so a `true`
/// here means that loader has something to open.
fn dir_has_safetensors(dir: &Path) -> Result<bool> {
    let entries = std::fs::read_dir(dir).map_err(|e| Error::ModelError {
        reason: format!("reading directory {}: {e}", dir.display()),
    })?;
    for entry in entries {
        let path: PathBuf = entry
            .map_err(|e| Error::ModelError {
                reason: format!("reading directory {}: {e}", dir.display()),
            })?
            .path();
        if path.extension().is_some_and(|e| e == "safetensors") {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Do the first bytes of `path` spell a ZIP local-file header?
fn is_zip(path: &Path) -> Result<bool> {
    let mut file = std::fs::File::open(path).map_err(|e| Error::ModelError {
        reason: format!("opening {}: {e}", path.display()),
    })?;
    let mut magic = [0u8; 4];
    let mut filled = 0;
    while filled < magic.len() {
        let read = file
            .read(&mut magic[filled..])
            .map_err(|e| Error::ModelError {
                reason: format!("reading {}: {e}", path.display()),
            })?;
        if read == 0 {
            // Shorter than a magic number: not a ZIP. Let the safetensors
            // reader phrase the real complaint about the file.
            return Ok(false);
        }
        filled += read;
    }
    Ok(magic == ZIP_MAGIC)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::audio::voxcpm::vae::{AudioVaeDecoder, AudioVaeEncoder};
    use numr::runtime::cpu::CpuRuntime;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn a_short_file_is_not_a_zip() {
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(b"PK").expect("write");
        file.flush().expect("flush");
        assert!(!is_zip(file.path()).expect("sniff"));
    }

    #[test]
    fn zip_magic_is_recognised() {
        let mut file = NamedTempFile::new().expect("temp file");
        file.write_all(b"PK\x03\x04rest of an archive")
            .expect("write");
        file.flush().expect("flush");
        assert!(is_zip(file.path()).expect("sniff"));
    }

    /// A safetensors file opens as safetensors, extension or not: its first
    /// eight bytes are a header length, which never spells `PK\x03\x04`.
    #[test]
    fn safetensors_bytes_are_not_a_zip() {
        let mut file = NamedTempFile::new().expect("temp file");
        let header = r#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        file.write_all(&(header.len() as u64).to_le_bytes())
            .expect("header len");
        file.write_all(header.as_bytes()).expect("header");
        file.write_all(&1.0f32.to_le_bytes()).expect("data");
        file.flush().expect("flush");
        assert!(!is_zip(file.path()).expect("sniff"));
        assert!(matches!(
            VaeCheckpoint::open(file.path()).expect("open"),
            VaeCheckpoint::SafeTensors(_)
        ));
    }

    /// THE equivalence gate: reading the published `audiovae.pth` must give
    /// the same tensors the reference `convert_audiovae.py` wrote, for EVERY
    /// key that script emitted — folded pairs included.
    ///
    /// Both paths are supplied by the environment because neither file is in
    /// the repo; the test skips when either is absent rather than failing on
    /// a machine that has no checkpoint.
    #[test]
    fn pth_matches_the_converted_safetensors() {
        let (Ok(pth), Ok(st)) = (
            std::env::var("VOXCPM2_AUDIOVAE_PTH"),
            std::env::var("VOXCPM2_AUDIOVAE_SAFETENSORS"),
        ) else {
            return;
        };
        if !Path::new(&pth).is_file() || !Path::new(&st).is_file() {
            return;
        }

        let device = <CpuRuntime as Runtime>::default_device();
        let reference = SafeTensorsLoader::open(&st).expect("open safetensors");
        let names = reference.tensor_names();
        assert!(!names.is_empty(), "reference checkpoint is empty");
        let mut folded = VaeCheckpoint::open(&pth).expect("open pth");
        assert!(
            matches!(folded, VaeCheckpoint::TorchPth(_)),
            "sniffed wrong"
        );
        let mut plain = VaeCheckpoint::open(&st).expect("open safetensors source");

        let mut worst = 0.0f32;
        for name in &names {
            let want: numr::tensor::Tensor<CpuRuntime> =
                plain.load_named(name, &device).expect("reference tensor");
            if want.dtype() != DType::F32 {
                // `sr_bin_boundaries` is I32 and is copied through unchanged;
                // an elementwise float comparison would misread its bytes.
                continue;
            }
            let got: numr::tensor::Tensor<CpuRuntime> =
                folded.load_named(name, &device).expect(name);
            assert_eq!(got.shape(), want.shape(), "{name}: shape");

            let got: Vec<f32> = got.contiguous().expect("contiguous").to_vec();
            let want: Vec<f32> = want.contiguous().expect("contiguous").to_vec();
            for (a, b) in got.iter().zip(want.iter()) {
                let scale = b.abs().max(1e-3);
                let rel = (a - b).abs() / scale;
                assert!(rel < 4e-6, "{name}: {a} vs {b} (relative {rel})");
                worst = worst.max(rel);
            }
        }
        println!(
            "compared {} tensors, worst relative error {worst:e}",
            names.len()
        );

        // And the assemblies themselves come up off the `.pth`, so the shape
        // gates in the encoder/decoder loaders see the folded weights too.
        AudioVaeEncoder::<CpuRuntime>::from_checkpoint(&pth, &device).expect("encoder");
        AudioVaeDecoder::<CpuRuntime>::from_checkpoint(&pth, &device).expect("decoder");
    }
}
