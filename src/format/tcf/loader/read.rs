//! Verified tensor loading: dense f32 and quantized, on `TcfLoader` itself.

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::directory::TcfLoader;
use super::session::{TcfSession, verify_at};
use crate::error::{Error, Result};
use crate::format::tcf::error::{tcf_error, tcf_tensor_error};
use crate::quant::contract::ActivationContract;
use crate::quant::{QuantFormat, QuantTensor};
use crate::tcf::TcfFile;

impl TcfLoader {
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
        TcfSession::new(self)
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
        super::session::quant_format(self.tensor_info(name)?.encoding(), name)
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
    pub(super) fn file(&self) -> Result<TcfFile<'_>> {
        let context = format!("reopen {}", self.path.display());
        TcfFile::open(&self.mmap).map_err(|e| tcf_error(&context, e))
    }

    /// Verify tensor `index`, then decode it to host f32 values.
    pub(super) fn decode_at(&self, file: &TcfFile<'_>, index: usize) -> Result<Vec<f32>> {
        let name = self.name_at(index);
        let record = file.tensors().get(index).ok_or_else(|| Error::ModelError {
            reason: format!("TCF tensor index {index} is out of range"),
        })?;
        verify_at(file, record, name)?;
        let payload = file
            .payload(record)
            .map_err(|e| tcf_tensor_error(name, "payload", e))?;
        crate::format::tcf::decode::decode_tensor_f32(record, payload, name)
    }

    /// Verify tensor `index`, then place its packed payload on `device`.
    pub(super) fn quant_at<R: Runtime<DType = DType>>(
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
        let format = super::session::quant_format(record.encoding, name)?;
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
}

#[cfg(test)]
mod tests {
    use super::super::test_support::open_fixture;
    use super::*;
    use crate::format::tcf::fixtures;
    use crate::test_utils::cpu_setup;
    use numr::runtime::cpu::CpuRuntime;

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
        let dense = crate::quant::DequantOps::dequantize(&client, &qt, numr::dtype::DType::F32)
            .expect("dequantizes");
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
}
