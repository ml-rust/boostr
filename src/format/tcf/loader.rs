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

use crate::tcf::{Encoding, TcfFile};
use memmap2::Mmap;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::block::{BoostrBlockDecoder, block_format};
use super::decode::decode_tensor_f32;
use super::error::{tcf_error, tcf_tensor_error};
use super::metadata::{TcfHeaderInfo, TcfModuleInfo, TcfTensorInfo, encoding_name};
use crate::error::{Error, Result};
use crate::quant::contract::ActivationContract;
use crate::quant::{QuantFormat, QuantTensor};

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
        let session = self.session()?;
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            out.push(session.tensor::<R>(name, device)?);
        }
        Ok(out)
    }

    /// Bind the directory ONCE for a run of loads whose names are not all
    /// known up front.
    ///
    /// [`TcfLoader::load_tensors`] answers that for a fixed list; a model
    /// loader walking hundreds of names one at a time cannot use it, and the
    /// one-shot loads reparse and revalidate the whole directory per call
    /// (see `TcfLoader::file`). Over a 577-tensor file that is quadratic.
    /// A session pays it once and keeps every per-tensor check intact.
    ///
    /// # Errors
    /// [`Error::ModelError`] carrying the spec's `E_*` code, if the mapped
    /// bytes no longer validate.
    pub fn session(&self) -> Result<TcfSession<'_>> {
        Ok(TcfSession {
            loader: self,
            file: self.file()?,
        })
    }

    /// Verify `name` and place it on `device` STILL QUANTIZED.
    ///
    /// The packed payload goes to the device verbatim, so a TCF model is held
    /// at its on-disk size rather than at 8x that as f32. Dequantization
    /// happens per use, through `DequantOps`, or not at all once a fused
    /// quantized matmul consumes the [`QuantTensor`] directly.
    ///
    /// Block encodings only: the tensor is the same `QuantTensor` a GGUF
    /// file yields, decoded by the same kernels. A raw encoding stores
    /// literal values with no scale (Section 12), so it has no quantized
    /// form to hold — load it with [`TcfLoader::load_tensor`].
    ///
    /// # Errors
    /// [`Error::ModelError`] for an unknown name, a failed digest or proof
    /// check, or a raw encoding.
    /// [`Error::QuantError`] when the shape and the payload length disagree.
    pub fn load_quant_tensor<R: Runtime<DType = DType>>(
        &self,
        name: &str,
        device: &R::Device,
    ) -> Result<QuantTensor<R>> {
        let index = self.index_of(name)?;
        let file = self.file()?;
        self.quant_at(&file, index, device)
    }

    /// Verify and place several tensors quantized, opening the file once.
    ///
    /// Prefer this over repeated [`TcfLoader::load_quant_tensor`] calls: each
    /// call revalidates the whole directory, which is O(directory) work per
    /// tensor.
    ///
    /// # Errors
    /// Every error [`TcfLoader::load_quant_tensor`] raises, for the first name
    /// that fails.
    pub fn load_quant_tensors<R: Runtime<DType = DType>>(
        &self,
        names: &[&str],
        device: &R::Device,
    ) -> Result<Vec<QuantTensor<R>>> {
        let session = self.session()?;
        let mut out = Vec::with_capacity(names.len());
        for name in names {
            out.push(session.quant_tensor::<R>(name, device)?);
        }
        Ok(out)
    }

    /// The block format of `name`, or an error naming the encoding when it
    /// is raw.
    ///
    /// A placement planner calls this to size a tensor before deciding where
    /// it lives, without reading a payload page.
    ///
    /// # Errors
    /// [`Error::ModelError`] for an unknown name or a raw encoding.
    pub fn quant_format(&self, name: &str) -> Result<QuantFormat> {
        quant_format(self.tensor_info(name)?.encoding(), name)
    }

    /// Verify every tensor's digests and proof vector without decoding.
    /// Section 15.
    ///
    /// # Errors
    /// [`Error::ModelError`] naming the first tensor that fails.
    pub fn verify_all(&self) -> Result<()> {
        let file = self.file()?;
        for (index, record) in file.tensors().iter().enumerate() {
            verify_at(&file, record, self.name_at(index))?;
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
        verify_at(file, record, name)?;
        let payload = file
            .payload(record)
            .map_err(|e| tcf_tensor_error(name, "payload", e))?;
        decode_tensor_f32(record, payload, name)
    }

    /// Verify tensor `index`, then place its packed payload on `device`.
    fn quant_at<R: Runtime<DType = DType>>(
        &self,
        file: &TcfFile<'_>,
        index: usize,
        device: &R::Device,
    ) -> Result<QuantTensor<R>> {
        let name = self.name_at(index);
        let record = file.tensors().get(index).ok_or_else(|| Error::ModelError {
            reason: format!("TCF tensor index {index} is out of range"),
        })?;
        verify_at(file, record, name)?;
        let format = quant_format(record.encoding, name)?;
        let payload = file
            .payload(record)
            .map_err(|e| tcf_tensor_error(name, "payload", e))?;
        let shape = self.shape_at(index)?;

        // Section 3: every tensor names a `ContractRecord` in its own file,
        // and `TcfFile::open` has already rejected an id that resolves to
        // nothing, so this lookup fails only for a record from elsewhere.
        // The contract rides on the weight from here: a kernel router sees a
        // `QuantTensor`, never the file it came from.
        let contract = file
            .contract(record)
            .map_err(|e| tcf_tensor_error(name, "activation contract", e))?;
        let contract = ActivationContract::from_record(name, record.execution_role, contract);

        Ok(
            QuantTensor::<R>::from_bytes(payload, format, &shape, device)?
                .with_activation_contract(contract),
        )
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

/// One validated view of a [`TcfLoader`]'s directory, reused across loads.
///
/// Bound by [`TcfLoader::session`]. What a session skips is the repeated
/// DIRECTORY parse, never a payload check: every load below still verifies
/// the tensor's digests and proof vector first (Section 15).
///
/// The borrow is what makes this sound — the validated view points into the
/// loader's mapping, so it cannot outlive the loader that owns it.
pub struct TcfSession<'a> {
    loader: &'a TcfLoader,
    file: TcfFile<'a>,
}

impl<'a> TcfSession<'a> {
    /// The loader this session reads through.
    pub fn loader(&self) -> &'a TcfLoader {
        self.loader
    }

    /// Verify and decode `name` into host f32 values, row-major.
    ///
    /// # Errors
    /// Every error [`TcfLoader::load_tensor_f32`] raises.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let index = self.loader.index_of(name)?;
        self.loader.decode_at(&self.file, index)
    }

    /// Verify and decode `name` onto `device` as a dense f32 tensor.
    ///
    /// # Errors
    /// Every error [`TcfLoader::load_tensor`] raises.
    pub fn tensor<R: Runtime<DType = DType>>(
        &self,
        name: &str,
        device: &R::Device,
    ) -> Result<Tensor<R>> {
        let index = self.loader.index_of(name)?;
        let values = self.loader.decode_at(&self.file, index)?;
        let shape = self.loader.shape_at(index)?;
        Tensor::<R>::from_slice(&values, &shape, device).map_err(Error::Numr)
    }

    /// Verify `name` and place it on `device` STILL QUANTIZED.
    ///
    /// # Errors
    /// Every error [`TcfLoader::load_quant_tensor`] raises, a raw encoding
    /// included.
    pub fn quant_tensor<R: Runtime<DType = DType>>(
        &self,
        name: &str,
        device: &R::Device,
    ) -> Result<QuantTensor<R>> {
        let index = self.loader.index_of(name)?;
        self.loader.quant_at(&self.file, index, device)
    }
}

/// Verify one tensor's digests and proof vector. Section 15.
///
/// A block tensor's proof is checked with boostr's own decoder, so a stream
/// these kernels would read differently from the producer fails here rather
/// than at first use.
fn verify_at(file: &TcfFile<'_>, record: &crate::tcf::TensorRecord, name: &str) -> Result<()> {
    file.verify_tensor_with(record, Some(&BoostrBlockDecoder))
        .map_err(|e| tcf_tensor_error(name, "verify", e))
}

/// The runtime format for a quantized encoding.
///
/// # Errors
/// [`Error::ModelError`] naming the encoding and the tensor, when the
/// encoding is raw or a block layout this build has no kernel for.
fn quant_format(encoding: Encoding, name: &str) -> Result<QuantFormat> {
    match encoding {
        Encoding::Block(block) => block_format(block, name),
        raw @ Encoding::Raw(_) => Err(Error::ModelError {
            reason: format!(
                "TCF tensor '{name}': encoding {} is not quantized, so it has no packed form; load it as a dense tensor",
                encoding_name(raw),
            ),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures;
    use super::*;
    use crate::tcf::{BlockEncoding, Encoding, FallbackReason, RawEncoding};
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

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

    #[test]
    fn a_quantized_tensor_loads_onto_a_device() {
        let (_file, loader) = open_fixture(&fixtures::good_file()).expect("opens");
        let (_client, device) = cpu_setup();
        let tensor = loader
            .load_tensor::<CpuRuntime>("layer.w", &device)
            .expect("loads");
        assert_eq!(tensor.shape(), &[2, 64]);
        assert_eq!(tensor.to_vec::<f32>(), fixtures::expected_q8_0_values());
    }

    /// The point of the packed path: the device holds the payload at its
    /// on-disk size, and dequantizing it later gives the same values the
    /// dense path gives.
    #[test]
    fn a_quantized_tensor_loads_still_packed_and_dequantizes_identically() {
        use crate::quant::DequantOps;

        let (_file, loader) = open_fixture(&fixtures::good_file()).expect("opens");
        let (client, device) = cpu_setup();
        let qt = loader
            .load_quant_tensor::<CpuRuntime>("layer.w", &device)
            .expect("loads packed");

        assert_eq!(qt.shape(), &[2, 64]);
        // Four Q8_0 blocks of 34 bytes.
        assert_eq!(qt.storage_bytes(), 4 * 34);
        assert_eq!(qt.format(), QuantFormat::Q8_0);

        let dense = client
            .dequantize(&qt, numr::dtype::DType::F32)
            .expect("dequantizes");
        assert_eq!(dense.shape(), &[2, 64]);
        assert_eq!(dense.to_vec::<f32>(), fixtures::expected_q8_0_values());
    }

    /// Section 9: the contract rides out of the file on the weight, because
    /// a kernel router sees a `QuantTensor` and never the file behind it.
    /// The fixture declares the ggml kernel family, so a kernel promising
    /// exact f32 activations must refuse it rather than run arithmetic the
    /// file never asked for, while the block kernels run it.
    #[test]
    fn a_quantized_tensor_carries_its_activation_contract_and_gates_dispatch() {
        use crate::quant::KernelContract;
        use crate::quant::traits::QuantMatmulOps;
        use crate::tcf::{DotAccumulator, ExecutionRole, InputRepresentation};

        const F32_KERNEL: KernelContract = KernelContract::f32_activation("test f32 matmul");

        let (_file, loader) = open_fixture(&fixtures::good_file()).expect("opens");
        let (client, device) = cpu_setup();
        let qt = loader
            .load_quant_tensor::<CpuRuntime>("layer.w", &device)
            .expect("loads packed");

        let declared = qt.activation_contract().expect("a TCF weight declares one");
        assert_eq!(declared.tensor, "layer.w");
        assert_eq!(declared.role, ExecutionRole::Matmul);
        assert_eq!(
            declared.input_representation,
            InputRepresentation::GgmlReference
        );
        assert_eq!(declared.dot_accumulator, DotAccumulator::GgmlReference);
        assert_eq!(declared.quant_group, 0);

        let err = qt
            .check_activation_contract(&F32_KERNEL)
            .expect_err("an f32 kernel does not satisfy the ggml family contract");
        let text = err.to_string();
        assert!(text.contains("E_ACTIVATION_CONTRACT_MISMATCH"), "{text}");
        assert!(text.contains("layer.w"), "{text}");

        // The CPU block kernel is in the family, so the weight runs.
        let activation =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32; 64], &[1, 64], &device).expect("activation");
        let out = client
            .quant_matmul(&activation, &qt)
            .expect("the block kernel satisfies the declared contract");
        assert_eq!(out.shape(), &[1, 2]);
    }

    /// A raw encoding has no packed form, and the error says so rather than
    /// reinterpreting its literal values as codes.
    #[test]
    fn a_raw_tensor_has_no_packed_form() {
        let (_file, loader) = open_fixture(&fixtures::good_file()).expect("opens");
        let (_client, device) = cpu_setup();
        let err = loader
            .load_quant_tensor::<CpuRuntime>("layer.bias", &device)
            .expect_err("rejects");
        assert!(err.to_string().contains("not quantized"), "{err}");
        assert!(loader.quant_format("layer.bias").is_err());
    }

    #[test]
    fn the_encoding_descriptor_is_reachable_without_reading_a_payload() {
        let (_file, loader) = open_fixture(&fixtures::good_file()).expect("opens");
        let format = loader.quant_format("layer.w").expect("block");
        assert_eq!(format, QuantFormat::Q8_0);
        assert_eq!(format.storage_bytes(2 * 64).expect("bytes"), 4 * 34);
    }

    /// A GGML block tensor loads as the `QuantTensor` a GGUF gives — the same
    /// kernels a GGUF file reaches — carrying the file's activation contract,
    /// and decodes to the hand-computed values.
    #[test]
    fn a_block_tensor_loads_as_a_gguf_quant_tensor() {
        let (_file, loader) = open_fixture(&fixtures::block_file(0.0)).expect("opens");
        let (client, device) = cpu_setup();
        assert_eq!(
            loader.quant_format("layer.q8").expect("format"),
            QuantFormat::Q8_0
        );
        loader
            .verify_all()
            .expect("proofs check with boostr's decoder");

        let qt = loader
            .load_quant_tensor::<CpuRuntime>("layer.q8", &device)
            .expect("loads");
        assert_eq!(qt.shape(), &[2, 64]);
        assert_eq!(qt.format(), QuantFormat::Q8_0);
        assert!(qt.activation_contract().is_some());
        let dense =
            crate::quant::DequantOps::dequantize(&client, &qt, DType::F32).expect("dequantizes");
        assert_eq!(dense.to_vec::<f32>(), fixtures::expected_q8_0_values());

        let tensor = loader
            .load_tensor::<CpuRuntime>("layer.q8", &device)
            .expect("loads dense");
        assert_eq!(tensor.to_vec::<f32>(), fixtures::expected_q8_0_values());
    }

    /// A block tensor whose proof disagrees with its bytes is refused: the
    /// digests pass (the bytes are what the producer wrote), the decoder
    /// catches it.
    #[test]
    fn a_block_tensor_with_a_wrong_proof_is_refused() {
        let (_file, loader) = open_fixture(&fixtures::block_file(1.0)).expect("opens");
        let (_client, device) = cpu_setup();
        let err = loader
            .load_quant_tensor::<CpuRuntime>("layer.q8", &device)
            .expect_err("refuses");
        assert!(err.to_string().contains("E_PROOF_MISMATCH"), "{err}");
        assert!(loader.verify_all().is_err());
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
        assert_eq!(loaded[1].shape(), &[2, 64]);
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
        fixtures::corrupt_payload(&mut bytes, fixtures::T_Q8);
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
