//! `TcfLoader`'s struct definition, `open`, and directory accessors.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use crate::tcf::{CalibrationRecord, TcfFile};
use memmap2::Mmap;

use crate::error::{Error, Result};
use crate::format::tcf::error::tcf_error;
use crate::format::tcf::metadata::{TcfHeaderInfo, TcfModuleInfo, TcfTensorInfo};

/// A memory-mapped TCF file with its directory decoded.
///
/// `Debug` prints the path and the decoded directory, never the mapped bytes.
#[derive(Debug)]
pub struct TcfLoader {
    pub(super) mmap: Mmap,
    pub(super) path: PathBuf,
    header: TcfHeaderInfo,
    modules: Vec<TcfModuleInfo>,
    tensors: Vec<TcfTensorInfo>,
    calibrations: Vec<CalibrationRecord>,
    by_name: HashMap<String, usize>,
}

impl TcfLoader {
    /// Open and validate a `.tcf` file.
    ///
    /// `TcfFile::open` checks the header digest, every section range, the
    /// directory digest, every record, and the derived record digests. No
    /// payload byte is read here.
    ///
    /// # Errors
    /// [`Error::Io`] if the file cannot be opened or mapped.
    /// [`Error::ModelError`] carrying the spec's `E_*` code, for any
    /// structural or digest failure.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        // SAFETY: the mapping is read-only and never mutated here. The caller
        // must not truncate or replace the file while this loader is live.
        let mmap = unsafe { Mmap::map(&file) }?;

        let context = format!("open {}", path.display());
        let tcf = TcfFile::open(&mmap).map_err(|e| tcf_error(&context, e))?;

        let header = TcfHeaderInfo::from(tcf.header());

        let mut modules = Vec::with_capacity(tcf.modules().len());
        for record in tcf.modules() {
            let name = tcf
                .string(record.name)
                .map_err(|e| tcf_error(&context, e))?
                .to_string();
            modules.push(TcfModuleInfo::new(record, name));
        }

        let mut tensors = Vec::with_capacity(tcf.tensors().len());
        let mut by_name = HashMap::with_capacity(tcf.tensors().len());
        for (index, record) in tcf.tensors().iter().enumerate() {
            let name = tcf
                .string(record.name)
                .map_err(|e| tcf_error(&context, e))?
                .to_string();
            by_name.entry(name.clone()).or_insert(index);
            tensors.push(TcfTensorInfo::new(*record, name));
        }

        let calibrations = tcf.calibrations().to_vec();
        drop(tcf);
        Ok(Self {
            mmap,
            path,
            header,
            modules,
            tensors,
            calibrations,
            by_name,
        })
    }

    /// The file this loader mapped.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Header counts, offsets, and version. Section 5.
    pub fn header(&self) -> &TcfHeaderInfo {
        &self.header
    }

    /// Every module, in file order. Section 7.
    pub fn modules(&self) -> &[TcfModuleInfo] {
        &self.modules
    }

    /// The module with `module_id`, if the file declares one.
    pub fn module(&self, module_id: u32) -> Option<&TcfModuleInfo> {
        self.modules.iter().find(|m| m.module_id == module_id)
    }

    /// Every calibration record, in file order. Section 10. A measured
    /// tensor's `calibration_id` names one of these; an unmeasured file has
    /// none.
    pub fn calibrations(&self) -> &[CalibrationRecord] {
        &self.calibrations
    }

    /// Every tensor, in file order. Section 8.
    pub fn tensors(&self) -> &[TcfTensorInfo] {
        &self.tensors
    }

    /// Tensor names, in file order.
    ///
    /// Names are provenance, never identity (Section 6), so a file can repeat
    /// one. [`TcfLoader::tensor_info`] resolves a repeated name to the first
    /// occurrence; this iterator yields every entry.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|t| t.name.as_str())
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// True when the file declares no tensor.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// The directory entry for `name`.
    ///
    /// # Errors
    /// [`Error::ModelError`] when the file declares no tensor of that name.
    pub fn tensor_info(&self, name: &str) -> Result<&TcfTensorInfo> {
        let index = self.index_of(name)?;
        self.tensors.get(index).ok_or_else(|| Error::ModelError {
            reason: format!("TCF tensor index {index} is out of range"),
        })
    }

    /// The module owning `name`, if the file declares it.
    ///
    /// # Errors
    /// [`Error::ModelError`] when the file declares no tensor of that name.
    pub fn owning_module(&self, name: &str) -> Result<Option<&TcfModuleInfo>> {
        Ok(self.module(self.tensor_info(name)?.module_id()))
    }

    /// The row-major shape of tensor `index`.
    pub(super) fn shape_at(&self, index: usize) -> Result<Vec<usize>> {
        self.tensors
            .get(index)
            .map(TcfTensorInfo::shape)
            .ok_or_else(|| Error::ModelError {
                reason: format!("TCF tensor index {index} is out of range"),
            })
    }

    /// The name of tensor `index`, or `"<unknown>"` when the index is out of
    /// range. Used only to label an error.
    pub(super) fn name_at(&self, index: usize) -> &str {
        self.tensors
            .get(index)
            .map_or("<unknown>", |t| t.name.as_str())
    }

    /// The file-order index of `name`.
    pub(super) fn index_of(&self, name: &str) -> Result<usize> {
        self.by_name
            .get(name)
            .copied()
            .ok_or_else(|| Error::ModelError {
                reason: format!("TCF tensor not found: {name}"),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::open_fixture;
    use super::*;
    use crate::format::tcf::fixtures;
    use crate::tcf::{BlockEncoding, Encoding, FallbackReason, RawEncoding};

    #[test]
    fn directory_metadata_survives_the_round_trip() {
        let (_file, loader) = open_fixture(&fixtures::good_file()).expect("opens");
        assert_eq!(loader.header().major, 1);
        assert_eq!(loader.header().tensor_count, 4);
        assert_eq!(loader.len(), 4);
        let names: Vec<&str> = loader.tensor_names().collect();
        assert_eq!(
            names,
            vec!["layer.w", "layer.bias", "layer.scale", "layer.pinned"]
        );
    }

    /// Section 8.6: the reason a tensor sits below its module's preference is
    /// the metadata that makes TCF worth reading.
    #[test]
    fn encoding_and_fallback_reason_are_both_reachable() {
        let (_file, loader) = open_fixture(&fixtures::good_file()).expect("opens");

        let weight = loader.tensor_info("layer.w").expect("known name");
        assert_eq!(weight.encoding(), Encoding::Block(BlockEncoding::Q8_0));
        assert_eq!(weight.fallback_reason(), FallbackReason::None);
        assert!(!weight.is_fallback());
        assert_eq!(weight.bits_per_weight(), Some(8.5));
        assert_eq!(weight.shape(), vec![2, 64]);

        assert_eq!(loader.tensors()[fixtures::T_FALLBACK].name, "layer.pinned");
        let pinned = loader.tensor_info("layer.pinned").expect("known name");
        assert_eq!(pinned.encoding(), Encoding::Raw(RawEncoding::F16));
        assert_eq!(
            pinned.fallback_reason(),
            FallbackReason::UserPinnedPrecision
        );
        assert!(pinned.is_fallback());
        assert_eq!(pinned.bits_per_weight(), None);

        let module = loader
            .owning_module("layer.pinned")
            .expect("known name")
            .expect("module resolves");
        assert_eq!(
            module.top_preferred_encoding(),
            Some(Encoding::Block(BlockEncoding::Q8_0))
        );
    }

    /// Section 5.3: a mutated directory byte fails `directory_digest`, so the
    /// file never opens.
    #[test]
    fn a_corrupted_directory_is_rejected_on_open() {
        let mut bytes = fixtures::good_file();
        let off = crate::tcf::HEADER_BYTES as usize;
        bytes[off] ^= 0x01;

        let file = fixtures::write_temp(&bytes);
        let err = TcfLoader::open(file.path()).expect_err("a corrupted directory is rejected");
        assert!(
            err.to_string().contains("E_DIRECTORY_DIGEST_MISMATCH"),
            "{err}"
        );
    }

    /// Section 12: an identifier outside the v1 registry is rejected by name,
    /// never decoded as something else.
    #[test]
    fn an_unassigned_encoding_is_rejected_by_identifier() {
        let mut bytes = fixtures::good_file();
        // 0x0109 sits in the retired tile-encoding range.
        fixtures::set_encoding(&mut bytes, fixtures::T_Q8, 0x0109);

        let file = fixtures::write_temp(&bytes);
        let err = TcfLoader::open(file.path()).expect_err("an unknown encoding is rejected");
        let text = err.to_string();
        assert!(text.contains("E_UNSUPPORTED_ENCODING"), "{text}");
        assert!(text.contains("0109"), "{text}");
    }

    #[test]
    fn an_unknown_name_is_named_in_the_error() {
        let (_file, loader) = open_fixture(&fixtures::good_file()).expect("opens");
        let err = loader.load_tensor_f32("nope").expect_err("unknown name");
        assert!(err.to_string().contains("nope"), "{err}");
    }

    #[test]
    fn a_non_tcf_file_is_rejected_by_magic() {
        let file = fixtures::write_temp(&[0u8; 512]);
        let err = TcfLoader::open(file.path()).expect_err("not a TCF file");
        assert!(err.to_string().contains("E_BAD_MAGIC"), "{err}");
    }
}
