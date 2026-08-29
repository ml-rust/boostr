//! `TcfLoader`: open a `.tcf` file and load its tensors as dense `Tensor<R>`.
//!
//! The file is memory-mapped and the directory is decoded once, so the
//! metadata a placement planner needs is available without touching a
//! payload page (Section 16). Payload pages are read only when a tensor is
//! loaded or verified.
//!
//! Every load verifies the tensor first: `payload_digest`, then the
//! recomputed logical stream against `semantic_digest`, then the proof
//! vector (Section 15). A reader that skips this cannot tell a correct file
//! from a corrupted one, which is the failure the format exists to prevent.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use tcf_core::TcfFile;

use super::decode::decode_tensor_f32;
use super::error::{tcf_error, tcf_tensor_error};
use super::metadata::{TcfHeaderInfo, TcfModuleInfo, TcfTensorInfo};
use crate::error::{Error, Result};

/// A memory-mapped TCF file with its directory decoded.
///
/// `Debug` prints the path and the decoded directory, never the mapped bytes.
#[derive(Debug)]
pub struct TcfLoader {
    mmap: Mmap,
    path: PathBuf,
    header: TcfHeaderInfo,
    modules: Vec<TcfModuleInfo>,
    tensors: Vec<TcfTensorInfo>,
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

        drop(tcf);
        Ok(Self {
            mmap,
            path,
            header,
            modules,
            tensors,
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

    /// Verify and decode `name` into host f32 values, row-major.
    ///
    /// # Errors
    /// [`Error::ModelError`] for an unknown name, a failed digest or proof
    /// check, or an encoding with no decode path.
    pub fn load_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let index = self.index_of(name)?;
        let file = self.file()?;
        self.decode_at(&file, index)
    }

    /// Verify and decode `name` onto `device` as a dense f32 tensor.
    ///
    /// # Errors
    /// Every error [`TcfLoader::load_tensor_f32`] raises, plus a numr
    /// allocation or upload failure.
    pub fn load_tensor<R: Runtime<DType = DType>>(
        &self,
        name: &str,
        device: &R::Device,
    ) -> Result<Tensor<R>> {
        let index = self.index_of(name)?;
        let file = self.file()?;
        let values = self.decode_at(&file, index)?;
        let shape = self.shape_at(index)?;
        Tensor::<R>::from_slice(&values, &shape, device).map_err(Error::Numr)
    }

    /// Verify and decode several tensors, opening the file once.
    ///
    /// Prefer this over repeated [`TcfLoader::load_tensor`] calls: each call
    /// revalidates the whole directory, which is O(directory) work per
    /// tensor.
    ///
    /// # Errors
    /// Every error [`TcfLoader::load_tensor`] raises, for the first name that
    /// fails.
    pub fn load_tensors<R: Runtime<DType = DType>>(
        &self,
        names: &[&str],
        device: &R::Device,
    ) -> Result<Vec<Tensor<R>>> {
        let file = self.file()?;
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            let index = self.index_of(name)?;
            let values = self.decode_at(&file, index)?;
            let shape = self.shape_at(index)?;
            out.push(Tensor::<R>::from_slice(&values, &shape, device).map_err(Error::Numr)?);
        }
        Ok(out)
    }

    /// Verify every tensor's digests and proof vector without decoding.
    /// Section 15.
    ///
    /// # Errors
    /// [`Error::ModelError`] naming the first tensor that fails.
    pub fn verify_all(&self) -> Result<()> {
        let file = self.file()?;
        for (index, record) in file.tensors().iter().enumerate() {
            let name = self.name_at(index);
            file.verify_tensor(record)
                .map_err(|e| tcf_tensor_error(name, "verify", e))?;
        }
        Ok(())
    }

    /// Reopen the mapped bytes as a validated file.
    ///
    /// `TcfFile` borrows the bytes it validates, so it cannot be stored
    /// beside the `Mmap` that owns them without a self-referential struct.
    /// Reparsing costs one directory pass and keeps every payload read behind
    /// the same validation the first open performed.
    fn file(&self) -> Result<TcfFile<'_>> {
        let context = format!("reopen {}", self.path.display());
        TcfFile::open(&self.mmap).map_err(|e| tcf_error(&context, e))
    }

    /// Verify tensor `index`, then decode it to host f32 values.
    fn decode_at(&self, file: &TcfFile<'_>, index: usize) -> Result<Vec<f32>> {
        let name = self.name_at(index);
        let record = file.tensors().get(index).ok_or_else(|| Error::ModelError {
            reason: format!("TCF tensor index {index} is out of range"),
        })?;
        file.verify_tensor(record)
            .map_err(|e| tcf_tensor_error(name, "verify", e))?;
        let payload = file
            .payload(record)
            .map_err(|e| tcf_tensor_error(name, "payload", e))?;
        decode_tensor_f32(record, payload, name)
    }

    /// The row-major shape of tensor `index`.
    fn shape_at(&self, index: usize) -> Result<Vec<usize>> {
        self.tensors
            .get(index)
            .map(TcfTensorInfo::shape)
            .ok_or_else(|| Error::ModelError {
                reason: format!("TCF tensor index {index} is out of range"),
            })
    }

    /// The name of tensor `index`, or `"<unknown>"` when the index is out of
    /// range. Used only to label an error.
    fn name_at(&self, index: usize) -> &str {
        self.tensors
            .get(index)
            .map_or("<unknown>", |t| t.name.as_str())
    }

    /// The file-order index of `name`.
    fn index_of(&self, name: &str) -> Result<usize> {
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
    use super::super::fixtures;
    use super::*;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;
    use tcf_core::{Encoding, FallbackReason, NativeEncoding, RawEncoding};

    fn open_fixture(bytes: &[u8]) -> Result<(tempfile::NamedTempFile, TcfLoader)> {
        let file = fixtures::write_temp(bytes);
        let loader = TcfLoader::open(file.path())?;
        Ok((file, loader))
    }

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
        assert_eq!(
            weight.encoding(),
            Encoding::Native(NativeEncoding::Q4S32T64)
        );
        assert_eq!(weight.fallback_reason(), FallbackReason::None);
        assert!(!weight.is_fallback());
        assert_eq!(weight.bits_per_weight(), Some(4.5));
        assert_eq!(weight.shape(), vec![1, 64]);

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
            Some(Encoding::Native(NativeEncoding::Q4S32T64))
        );
    }

    #[test]
    fn a_quantized_tensor_loads_onto_a_device() {
        let (_file, loader) = open_fixture(&fixtures::good_file()).expect("opens");
        let (_client, device) = cpu_setup();
        let tensor = loader
            .load_tensor::<CpuRuntime>("layer.w", &device)
            .expect("loads");
        assert_eq!(tensor.shape(), &[1, 64]);
        assert_eq!(tensor.to_vec::<f32>(), fixtures::expected_q4_values());
    }

    #[test]
    fn a_raw_tensor_loads_onto_a_device() {
        let (_file, loader) = open_fixture(&fixtures::good_file()).expect("opens");
        let (_client, device) = cpu_setup();
        let tensor = loader
            .load_tensor::<CpuRuntime>("layer.bias", &device)
            .expect("loads");
        assert_eq!(tensor.shape(), &[4]);
        assert_eq!(tensor.to_vec::<f32>(), fixtures::RAW_F32_VALUES.to_vec());
    }

    #[test]
    fn a_batch_load_returns_each_tensor_in_order() {
        let (_file, loader) = open_fixture(&fixtures::good_file()).expect("opens");
        let (_client, device) = cpu_setup();
        let loaded = loader
            .load_tensors::<CpuRuntime>(&["layer.bias", "layer.w"], &device)
            .expect("loads");
        assert_eq!(loaded[0].shape(), &[4]);
        assert_eq!(loaded[1].shape(), &[1, 64]);
    }

    #[test]
    fn every_tensor_verifies() {
        let (_file, loader) = open_fixture(&fixtures::good_file()).expect("opens");
        loader.verify_all().expect("a known-good file verifies");
    }

    /// Section 15.1: a single flipped payload byte is caught by
    /// `payload_digest`, and the error names the tensor.
    #[test]
    fn a_corrupted_payload_is_rejected_on_load() {
        let mut bytes = fixtures::good_file();
        fixtures::corrupt_payload(&mut bytes, fixtures::T_Q4);
        let (_file, loader) = open_fixture(&bytes).expect("the directory is untouched");

        let err = loader
            .load_tensor_f32("layer.w")
            .expect_err("a corrupted payload is rejected");
        let text = err.to_string();
        assert!(text.contains("E_PAYLOAD_DIGEST_MISMATCH"), "{text}");
        assert!(text.contains("layer.w"), "{text}");

        let err = loader.verify_all().expect_err("verify_all rejects it too");
        assert!(err.to_string().contains("E_PAYLOAD_DIGEST_MISMATCH"));
    }

    /// Section 5.3: a mutated directory byte fails `directory_digest`, so the
    /// file never opens.
    #[test]
    fn a_corrupted_directory_is_rejected_on_open() {
        let mut bytes = fixtures::good_file();
        let off = tcf_core::HEADER_BYTES as usize;
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
        // 0x0109 is a deliberate gap in the native quantized range.
        fixtures::set_encoding(&mut bytes, fixtures::T_Q4, 0x0109);

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
