use super::*;
use crate::tcf::binary16::f32_to_bits;
use crate::tcf::consts::{HEADER_BYTES, PROOF_COUNT, SECTION_ALIGN, TENSOR_RECORD_BYTES};
use crate::tcf::encoding::block::BlockEncoding;
use crate::tcf::encoding::raw::RawEncoding;
use crate::tcf::encoding::registry::Encoding;
use crate::tcf::enums::{
    DotAccumulator, ExecutionRole, FallbackReason, InputRepresentation, LayoutId, MathMode,
    ModuleRole, OutputDtype, PrimaryMetric, ProofFormat, QuantAxis, RelationType, ResidencyClass,
    Role, RoundingMode, ScaleComputeDtype, StateDtype, WorkloadKind,
};
use crate::tcf::flags::{
    CalibrationFlags, ContractFlags, PolicyFlags, RelationFlags, RequiredFeatures, StateFlags,
    TensorFlags, WorkloadProfileFlags,
};
use crate::tcf::reader::TcfFile;
use crate::tcf::test_blocks::{Q8_0Decoder, q8_0_proof, q8_0_stream};
use std::io::SeekFrom;

fn module(module_id: u32, name: StringRef) -> ModuleRecord {
    ModuleRecord {
        module_id,
        parent_id: crate::tcf::consts::ROOT_PARENT_ID,
        name,
        module_role: ModuleRole::Ffn,
        fallback_encoding: None,
        // No ranked preference, so Section 8.6 mandates no
        // `fallback_reason` on any tensor in this module.
        preferred_encoding: [None; 4],
        activation_contract_id: 1,
        policy_flags: PolicyFlags::NONE,
        min_quant_k: 64,
        default_residency: ResidencyClass::Warm,
        state_dtype: StateDtype::F32,
        state_flags: StateFlags::NONE,
        policy_digest: [0u8; 16],
    }
}

fn contract(contract_id: u32) -> ContractRecord {
    ContractRecord {
        contract_id,
        input_representation: InputRepresentation::A8S32Dynamic,
        quant_group: 32,
        quant_axis: QuantAxis::Last,
        rounding_mode: RoundingMode::RnEven,
        qmin: -127,
        qmax: 127,
        scale_compute_dtype: ScaleComputeDtype::F32,
        dot_accumulator: DotAccumulator::I32ThenF32Scale,
        output_dtype: OutputDtype::F32,
        math_mode: MathMode::ReassociationAllowed,
        calibration_id: 0,
        flags: ContractFlags::NONE,
        // Writer-owned: Section 9 makes the producer compute it.
        contract_digest: [0u8; 16],
    }
}

fn calibration(calibration_id: u32, dataset: StringRef, evaluator: StringRef) -> CalibrationRecord {
    CalibrationRecord {
        calibration_id,
        primary_metric: PrimaryMetric::Perplexity,
        flags: CalibrationFlags::NONE,
        sample_count: 256,
        seed_count: 3,
        baseline_metric: 5.6789,
        acceptance_margin: 0.01,
        dataset_name: dataset,
        evaluator_name: evaluator,
        dataset_digest: [0x44u8; 32],
        evaluator_config_digest: [0x55u8; 16],
        producer_timestamp: 1_700_000_000,
    }
}

fn tensor(tensor_id: u32, module_id: u32, name: StringRef, encoding: Encoding) -> TensorRecord {
    TensorRecord {
        tensor_id,
        module_id,
        name,
        role: Role::LinearWeight,
        encoding,
        fallback_reason: FallbackReason::None,
        residency_class: ResidencyClass::Hot,
        flags: TensorFlags::NONE,
        rank: 2,
        calibration_id: 0,
        dims: [1, 64, 0, 0, 0, 0, 0, 0],
        activation_contract_id: 1,
        layout_id: LayoutId::RowMajorDense,
        data_offset: 0,
        logical_payload_bytes: 0,
        physical_span_bytes: 0,
        resident_bytes: 0,
        transfer_bytes: 0,
        sensitivity_delta: 0.0,
        sensitivity_ci95: 0.0,
        accesses_per_generation: 0.0,
        bytes_read_per_generation: 0.0,
        sensitivity_samples: 0,
        sensitivity_seed_count: 0,
        access_profile_samples: 0,
        execution_role: ExecutionRole::Matmul,
        workload_profile_id: 0,
        semantic_digest: [0u8; 16],
        payload_digest: [0u8; 16],
        proof_rel_off: 0,
        proof_count: 0,
        proof_format: ProofFormat::None,
    }
}

/// Every block encoding the fixtures cover, each with a shape that is one
/// row of whole blocks: two 32-element formats and two 256-element ones,
/// plus a 32-element one whose block length is not a multiple of 4.
const EVERY_BLOCK: [(BlockEncoding, [u64; 2]); 5] = [
    (BlockEncoding::Q4_0, [1, 64]),
    (BlockEncoding::Q8_0, [1, 64]),
    (BlockEncoding::Q4K, [1, 256]),
    (BlockEncoding::Q6K, [1, 256]),
    (BlockEncoding::Q5_0, [1, 64]),
];

/// A deterministic byte stream of the length `block` needs for `dims`. The
/// writer stores a block stream verbatim, so its contents are arbitrary
/// here; `seed` keeps two tensors' payloads distinct.
fn block_bytes_for(block: BlockEncoding, dims: [u64; 2], seed: u32) -> Vec<u8> {
    let len = block.payload_bytes(&dims, 2, 0).expect("whole blocks") as usize;
    (0..len)
        .map(|i| (i as u32).wrapping_mul(31).wrapping_add(seed * 7) as u8)
        .collect()
}

/// Proof values for a fixture whose bytes no decoder reads: any 64 values.
fn fixture_proof(seed: u32) -> Vec<f32> {
    (0..PROOF_COUNT).map(|i| (i + seed) as f32 * 0.25).collect()
}

/// `tensor` for a block encoding, with the shape `EVERY_BLOCK` gives it.
fn block_tensor(
    tensor_id: u32,
    module_id: u32,
    name: StringRef,
    block: BlockEncoding,
    dims: [u64; 2],
) -> TensorRecord {
    let mut record = tensor(tensor_id, module_id, name, Encoding::Block(block));
    record.dims = [dims[0], dims[1], 0, 0, 0, 0, 0, 0];
    record
}

/// A writer holding two modules, one contract, one tensor per block
/// encoding in `EVERY_BLOCK`, and one raw tensor.
fn populated() -> TcfWriter {
    populated_with(true)
}

/// The same writer, with every tensor either carrying its payload
/// (`payloads`) or registered from its record alone.
///
/// One builder for both, so the two writers differ in nothing but where
/// the payloads come from. That is what makes
/// `finish_streaming_is_byte_identical` a test of the write path rather
/// than of two hand-kept fixtures.
fn populated_with(payloads: bool) -> TcfWriter {
    let mut w = TcfWriter::new();
    let ffn = w.intern("model.layers.0.ffn").expect("interns");
    let attn = w.intern("model.layers.0.attn").expect("interns");
    w.add_module(module(0, ffn)).expect("adds");
    w.add_module(module(1, attn)).expect("adds");
    w.add_contract(contract(1)).expect("adds");
    let dataset = w.intern("wikitext2").expect("interns");
    let evaluator = w.intern("perplexity-v1").expect("interns");
    w.add_calibration(calibration(1, dataset, evaluator))
        .expect("adds");

    for (i, (block, dims)) in EVERY_BLOCK.into_iter().enumerate() {
        let id = u32::try_from(i).unwrap_or(0);
        let name = w.intern("w").expect("interns");
        let record = block_tensor(id, id % 2, name, block, dims);
        if payloads {
            w.add_block_tensor(record, block_bytes_for(block, dims, id), &fixture_proof(id))
                .expect("adds");
        } else {
            w.register_block_tensor(record).expect("registers");
        }
    }

    let name = w.intern("bias").expect("interns");
    let mut raw = tensor(5, 0, name, Encoding::Raw(RawEncoding::F16));
    raw.role = Role::Bias;
    raw.rank = 1;
    raw.dims = [8, 0, 0, 0, 0, 0, 0, 0];
    raw.execution_role = ExecutionRole::Elementwise;
    if payloads {
        w.add_raw_tensor(raw, (0u8..16).collect()).expect("adds");
    } else {
        w.register_raw_tensor(raw).expect("registers");
    }
    w
}

fn block_record(w: &mut TcfWriter) -> TensorRecord {
    let name = w.intern("blk.0.ffn_down.weight").expect("interns");
    let mut record = tensor(
        0,
        0,
        name,
        Encoding::Block(crate::tcf::encoding::BlockEncoding::Q8_0),
    );
    record.dims = [2, 64, 0, 0, 0, 0, 0, 0];
    record
}

fn block_writer() -> (TcfWriter, Vec<f32>) {
    let mut w = TcfWriter::new();
    let name = w.intern("m").expect("interns");
    w.add_module(module(0, name)).expect("adds");
    w.add_contract(contract(1)).expect("adds");
    let record = block_record(&mut w);
    let bytes = q8_0_stream(2, 64, 0);
    let proof = q8_0_proof(&bytes, &[2, 64], 0);
    w.add_block_tensor(record, bytes, &proof)
        .expect("adds block");
    (w, proof)
}

/// A block tensor round-trips: its bytes come back verbatim, its
/// semantic digest equals its payload digest, and its proof vector
/// verifies against a decoder that agrees with the producer and fails
/// against one that does not. Without a decoder, verification stops at
/// the digests.
#[test]
fn block_tensor_round_trips_and_proves_with_a_decoder() {
    let (w, _) = block_writer();
    let file_bytes = w.finish().expect("writes");
    let file = TcfFile::open(&file_bytes).expect("reader accepts");
    let t = file.tensors().first().expect("tensor 0");
    assert_eq!(
        t.encoding,
        Encoding::Block(crate::tcf::encoding::BlockEncoding::Q8_0)
    );
    assert_eq!(t.logical_payload_bytes, 4 * 34);
    assert_eq!(t.proof_count, PROOF_COUNT);
    assert_eq!(t.semantic_digest, t.payload_digest);
    assert_eq!(
        file.payload(t).expect("payload"),
        q8_0_stream(2, 64, 0).as_slice()
    );

    file.verify_tensor(t).expect("digests verify");
    file.verify_tensor_with(t, Some(&Q8_0Decoder { bias: 0.0 }))
        .expect("proof verifies with the producer's decoder");
    assert_eq!(
        file.verify_tensor_with(t, Some(&Q8_0Decoder { bias: 1.0 })),
        Err(TcfError::ProofMismatch {
            tensor_id: 0,
            proof_index: 0
        })
    );
}

/// The writer rejects a block stream of the wrong length, a proof vector
/// of the wrong count, and a row width that is not whole blocks.
#[test]
fn block_tensor_lengths_are_checked() {
    let mut w = TcfWriter::new();
    let record = block_record(&mut w);
    let proof = vec![0.0f32; PROOF_COUNT as usize];
    assert_eq!(
        w.add_block_tensor(record, vec![0u8; 4 * 34 - 1], &proof),
        Err(TcfError::InvalidQuantShape { tensor_id: 0 })
    );
    let record = block_record(&mut w);
    assert_eq!(
        w.add_block_tensor(record, q8_0_stream(2, 64, 0), &proof[..63]),
        Err(TcfError::InvalidQuantShape { tensor_id: 0 })
    );
    let mut record = block_record(&mut w);
    record.dims = [2, 48, 0, 0, 0, 0, 0, 0];
    assert_eq!(
        w.add_block_tensor(record, q8_0_stream(2, 64, 0), &proof),
        Err(TcfError::InvalidQuantShape { tensor_id: 0 })
    );
    // A block record through the raw entry points is refused.
    let record = block_record(&mut w);
    assert_eq!(
        w.add_raw_tensor(record, q8_0_stream(2, 64, 0)),
        Err(TcfError::UnsupportedEncoding { raw: 0x0208 })
    );
    let record = block_record(&mut w);
    assert_eq!(
        w.register_raw_tensor(record),
        Err(TcfError::UnsupportedEncoding { raw: 0x0208 })
    );
}

#[test]
fn round_trips_through_the_reader() {
    let bytes = populated().finish().expect("writes");
    let file = TcfFile::open(&bytes).expect("reader accepts");

    assert_eq!(file.modules().len(), 2);
    assert_eq!(file.contracts().len(), 1);
    assert_eq!(file.tensors().len(), 6);
    assert_eq!(file.header().tensor_count, 6);
    assert_eq!(file.header().module_count, 2);
    assert_eq!(file.header().contract_count, 1);
    assert_eq!(file.header().calibration_count, 1);
    let cal = file.calibrations().first().expect("calibration 0");
    assert_eq!(cal.calibration_id, 1);
    assert_eq!(cal.primary_metric, PrimaryMetric::Perplexity);
    assert_eq!(file.string(cal.dataset_name).expect("utf8"), "wikitext2");
    assert_eq!(
        file.string(cal.evaluator_name).expect("utf8"),
        "perplexity-v1"
    );

    let ffn = file.modules().first().expect("module 0");
    assert_eq!(ffn.module_id, 0);
    assert_eq!(ffn.module_role, ModuleRole::Ffn);
    assert!(ffn.is_root());
    assert_eq!(file.string(ffn.name).expect("utf8"), "model.layers.0.ffn");
    let attn = file.modules().get(1).expect("module 1");
    assert_eq!(file.string(attn.name).expect("utf8"), "model.layers.0.attn");

    let c = file.contracts().first().expect("contract 0");
    // Section 9: the writer computed the digest; every other field is
    // the caller's, unchanged.
    assert_ne!(c.contract_digest, [0u8; 16]);
    let mut expected = contract(1);
    expected.contract_digest = c.contract_digest;
    assert_eq!(*c, expected);

    for (i, (block, dims)) in EVERY_BLOCK.into_iter().enumerate() {
        let t = file.tensors().get(i).expect("tensor");
        assert_eq!(t.encoding, Encoding::Block(block));
        assert_eq!(t.role, Role::LinearWeight);
        assert_eq!(t.shape(), &dims[..]);
        assert_eq!(t.proof_count, PROOF_COUNT);
        assert_eq!(t.proof_format, ProofFormat::DequantF16);
        assert_eq!(t.data_offset % SECTION_ALIGN, 0);
        assert_eq!(t.resident_bytes, t.physical_span_bytes);
        assert_eq!(t.transfer_bytes, t.physical_span_bytes);
        assert_eq!(
            t.logical_payload_bytes,
            block.payload_bytes(&dims, 2, 0).expect("whole blocks")
        );
        assert_eq!(t.semantic_digest, t.payload_digest);
        assert_eq!(
            file.payload(t).expect("payload"),
            block_bytes_for(block, dims, i as u32).as_slice()
        );
        assert_eq!(file.string(t.name).expect("utf8"), "w");
        file.verify_tensor(t).expect("verifies");
    }

    let raw = file.tensors().get(5).expect("raw tensor");
    assert_eq!(raw.encoding, Encoding::Raw(RawEncoding::F16));
    assert_eq!(raw.rank, 1);
    assert_eq!(raw.logical_payload_bytes, 16);
    assert_eq!(raw.physical_span_bytes, 64);
    assert_eq!(raw.proof_count, 0);
    assert_eq!(raw.proof_format, ProofFormat::None);
    assert_eq!(raw.proof_rel_off, 0);
    let expected: Vec<u8> = (0u8..16).collect();
    assert_eq!(file.payload(raw).expect("payload"), expected.as_slice());
    file.verify_tensor(raw).expect("verifies");
}

/// A block stream lands byte for byte at the tensor's `data_offset`, and
/// its span is padded to the section alignment.
#[test]
fn a_block_stream_lands_at_its_data_offset() {
    let stream = q8_0_stream(1, 64, 9);
    let mut w = TcfWriter::new();
    let name = w.intern("stream").expect("interns");
    w.add_module(module(0, name)).expect("adds");
    // Section 3: every tensor names a contract the file carries.
    w.add_contract(contract(1)).expect("adds");
    let record = block_tensor(7, 0, name, BlockEncoding::Q8_0, [1, 64]);
    w.add_block_tensor(record, stream.clone(), &q8_0_proof(&stream, &[1, 64], 7))
        .expect("adds");
    let bytes = w.finish().expect("writes");

    let file = TcfFile::open(&bytes).expect("reader accepts");
    let t = file.tensors().first().expect("tensor 0");
    assert_eq!(t.logical_payload_bytes, 68);
    assert_eq!(t.physical_span_bytes, 128);

    let at = usize::try_from(t.data_offset).expect("fits");
    let stored = bytes.get(at..at + stream.len()).expect("in bounds");
    assert_eq!(stored, stream.as_slice());
    assert_eq!(file.payload(t).expect("payload"), stream.as_slice());
    file.verify_tensor_with(t, Some(&Q8_0Decoder { bias: 0.0 }))
        .expect("verifies with the producer's decoder");
}

#[test]
fn padding_and_reserved_bytes_are_zero() {
    let bytes = populated().finish().expect("writes");
    let file = TcfFile::open(&bytes).expect("reader accepts");

    for t in file.tensors() {
        let start = usize::try_from(t.data_offset + t.logical_payload_bytes).expect("fits");
        let end = usize::try_from(t.data_offset + t.physical_span_bytes).expect("fits");
        assert!(end > start, "every tensor here needs padding");
        let padding = bytes.get(start..end).expect("in bounds");
        assert!(padding.iter().all(|b| *b == 0));
    }

    // Header reserved tail, Section 5.
    assert_eq!(bytes.get(184..192), Some([0u8; 8].as_slice()));

    // TensorRecord reserved ranges 180..184, 186..188, 240..256, Section 8.
    let base = usize::try_from(file.header().tensor_off).expect("fits");
    for i in 0..file.tensors().len() {
        let at = base + i * TENSOR_RECORD_BYTES;
        assert_eq!(bytes.get(at + 180..at + 184), Some([0u8; 4].as_slice()));
        assert_eq!(bytes.get(at + 186..at + 188), Some([0u8; 2].as_slice()));
        assert_eq!(bytes.get(at + 240..at + 256), Some([0u8; 16].as_slice()));
    }
}

#[test]
fn every_section_is_64_aligned_and_inside_the_directory() {
    let bytes = populated().finish().expect("writes");
    let file = TcfFile::open(&bytes).expect("reader accepts");
    let h = file.header();
    let spans = [
        (h.module_off, u64::from(h.module_count)),
        (h.tensor_off, u64::from(h.tensor_count)),
        (h.contract_off, u64::from(h.contract_count)),
        (h.calibration_off, u64::from(h.calibration_count)),
        (h.string_off, h.string_len),
        (h.proof_off, h.proof_len),
    ];
    for (off, len) in spans {
        if len == 0 {
            assert_eq!(off, 0);
            continue;
        }
        assert_eq!(off % SECTION_ALIGN, 0);
        assert!(off >= u64::from(HEADER_BYTES));
        assert!(off < h.data_off);
    }
    assert_eq!(h.data_off % SECTION_ALIGN, 0);
    assert_eq!(h.file_len, u64::try_from(bytes.len()).expect("fits"));
    assert_eq!(h.relation_off, 0);
    assert_eq!(h.workload_off, 0);
}

#[test]
fn a_caller_set_data_offset_is_rejected_by_name() {
    let mut w = TcfWriter::new();
    let name = w.intern("w").expect("interns");
    w.add_module(module(0, name)).expect("adds");
    let mut record = block_tensor(0, 0, name, BlockEncoding::Q8_0, [1, 64]);
    record.data_offset = 4096;
    assert_eq!(
        w.add_block_tensor(record, q8_0_stream(1, 64, 0), &fixture_proof(0)),
        Err(TcfError::NonzeroReserved {
            field: "TensorRecord.data_offset"
        })
    );
}

#[test]
fn every_writer_owned_field_is_rejected_by_name() {
    let mut w = TcfWriter::new();
    let name = w.intern("w").expect("interns");
    let base = block_tensor(0, 0, name, BlockEncoding::Q8_0, [1, 64]);

    /// A mutation of one writer-owned field, paired with the field name
    /// the writer must name when rejecting it.
    type FieldConflict = (fn(&mut TensorRecord), &'static str);

    let cases: [FieldConflict; 10] = [
        (|t| t.data_offset = 64, "TensorRecord.data_offset"),
        (
            |t| t.logical_payload_bytes = 36,
            "TensorRecord.logical_payload_bytes",
        ),
        (
            |t| t.physical_span_bytes = 64,
            "TensorRecord.physical_span_bytes",
        ),
        (|t| t.resident_bytes = 64, "TensorRecord.resident_bytes"),
        (|t| t.transfer_bytes = 64, "TensorRecord.transfer_bytes"),
        (|t| t.proof_rel_off = 128, "TensorRecord.proof_rel_off"),
        (|t| t.proof_count = 64, "TensorRecord.proof_count"),
        (
            |t| t.proof_format = ProofFormat::DequantF16,
            "TensorRecord.proof_format",
        ),
        (
            |t| t.semantic_digest = [1u8; 16],
            "TensorRecord.semantic_digest",
        ),
        (
            |t| t.payload_digest = [1u8; 16],
            "TensorRecord.payload_digest",
        ),
    ];

    for (set, field) in cases {
        let mut record = base;
        set(&mut record);
        assert_eq!(
            w.add_block_tensor(record, q8_0_stream(1, 64, 0), &fixture_proof(0)),
            Err(TcfError::NonzeroReserved { field })
        );
    }
}

#[test]
fn an_empty_file_round_trips() {
    let bytes = TcfWriter::new().finish().expect("writes");
    assert_eq!(bytes.len(), HEADER_BYTES as usize);
    let file = TcfFile::open(&bytes).expect("reader accepts");
    assert!(file.tensors().is_empty());
    assert!(file.modules().is_empty());
    assert_eq!(file.header().data_off, u64::from(HEADER_BYTES));
    assert_eq!(file.header().file_len, u64::from(HEADER_BYTES));
    assert_eq!(
        file.header().required_features,
        RequiredFeatures::from_bits_retain(0b1111)
    );
}

fn relation() -> RelationRecord {
    RelationRecord {
        relation_type: RelationType::LowRankResidual,
        flags: RelationFlags::NONE,
        output_tensor_id: 0,
        input_tensor_id: [0, 1, 2, crate::tcf::consts::UNUSED_INPUT_ID],
        rank_or_parameter: 16,
        activation_contract_id: 1,
        // Writer-owned: Section 9 makes the producer compute it.
        relation_digest: [0u8; 16],
    }
}

fn workload() -> WorkloadProfileRecord {
    WorkloadProfileRecord {
        workload_id: 1,
        workload_kind: WorkloadKind::Chat,
        flags: WorkloadProfileFlags::NONE,
        generation_count: 1024,
        avg_generated_tokens: 256.5,
        avg_prompt_tokens: 48.25,
        dataset_name: StringRef::new(0, 0),
        runtime_name: StringRef::new(0, 0),
        workload_digest: [0x44u8; 32],
        runtime_config_digest: [0x55u8; 16],
        producer_timestamp: 1_700_000_000,
    }
}

#[test]
fn required_feature_bits_track_their_counts() {
    let plain = populated().finish().expect("writes");
    let file = TcfFile::open(&plain).expect("reader accepts");
    assert!(
        !file
            .header()
            .required_features
            .contains(RequiredFeatures::RELATIONS)
    );
    assert!(
        !file
            .header()
            .required_features
            .contains(RequiredFeatures::WORKLOAD_PROFILES)
    );

    let mut w = populated();
    w.add_relation(relation()).expect("adds");
    let with_relation = w.finish().expect("writes");
    let file = TcfFile::open(&with_relation).expect("reader accepts");
    assert!(
        file.header()
            .required_features
            .contains(RequiredFeatures::RELATIONS)
    );
    assert!(
        !file
            .header()
            .required_features
            .contains(RequiredFeatures::WORKLOAD_PROFILES)
    );
    assert_eq!(file.relations().len(), 1);
    let r = file.relations().first().expect("relation 0");
    assert_ne!(r.relation_digest, [0u8; 16]);
    let mut expected = relation();
    expected.relation_digest = r.relation_digest;
    assert_eq!(*r, expected);

    let mut w = populated();
    w.add_workload_profile(workload()).expect("adds");
    let with_workload = w.finish().expect("writes");
    let file = TcfFile::open(&with_workload).expect("reader accepts");
    assert!(
        file.header()
            .required_features
            .contains(RequiredFeatures::WORKLOAD_PROFILES)
    );
    assert!(
        !file
            .header()
            .required_features
            .contains(RequiredFeatures::RELATIONS)
    );
    assert_eq!(*file.workload_profiles().first().expect("w0"), workload());
}

#[test]
fn a_corrupted_payload_byte_fails_verification() {
    let mut bytes = populated().finish().expect("writes");
    let data_off = {
        let file = TcfFile::open(&bytes).expect("reader accepts");
        usize::try_from(file.header().data_off).expect("fits")
    };
    let target = bytes.get_mut(data_off).expect("first payload byte");
    *target ^= 0xff;

    let file = TcfFile::open(&bytes).expect("directory is untouched");
    let t = file.tensors().first().expect("tensor 0");
    assert_eq!(
        file.verify_tensor(t),
        Err(TcfError::PayloadDigestMismatch { tensor_id: 0 })
    );
}

#[test]
fn a_corrupted_directory_byte_fails_at_open() {
    let mut bytes = populated().finish().expect("writes");
    let at = HEADER_BYTES as usize;
    let target = bytes.get_mut(at).expect("first directory byte");
    *target ^= 0xff;
    assert_eq!(
        TcfFile::open(&bytes).unwrap_err(),
        TcfError::DirectoryDigestMismatch
    );
}

#[test]
fn a_corrupted_header_byte_fails_at_open() {
    let mut bytes = populated().finish().expect("writes");
    let target = bytes.get_mut(10).expect("Header.minor");
    *target = 9;
    assert_eq!(
        TcfFile::open(&bytes).unwrap_err(),
        TcfError::HeaderDigestMismatch
    );
}

#[test]
fn interning_the_same_name_stores_one_copy() {
    let mut w = TcfWriter::new();
    let a = w.intern("shared").expect("interns");
    let b = w.intern("shared").expect("interns");
    assert_eq!(a, b);
    assert_eq!(w.strings.len(), "shared".len());
    assert_eq!(w.intern("").expect("interns"), StringRef::new(0, 0));
}

#[test]
fn an_encoding_mismatched_add_is_rejected() {
    let mut w = TcfWriter::new();
    let name = w.intern("w").expect("interns");
    let raw = tensor(0, 0, name, Encoding::Raw(RawEncoding::F16));
    assert_eq!(
        w.add_block_tensor(raw, vec![0u8; 128], &fixture_proof(0)),
        Err(TcfError::UnsupportedEncoding { raw: 0x0002 })
    );

    let quantized = block_tensor(0, 0, name, BlockEncoding::Q8_0, [1, 64]);
    assert_eq!(
        w.add_raw_tensor(quantized, vec![0u8; 68]),
        Err(TcfError::UnsupportedEncoding { raw: 0x0208 })
    );
}

#[test]
fn a_block_count_disagreeing_with_the_shape_is_rejected() {
    let mut w = TcfWriter::new();
    let name = w.intern("w").expect("interns");
    let record = block_tensor(3, 0, name, BlockEncoding::Q8_0, [1, 64]);
    assert_eq!(
        w.add_block_tensor(record, q8_0_stream(2, 64, 0), &fixture_proof(0)),
        Err(TcfError::InvalidQuantShape { tensor_id: 3 })
    );

    let mut rank1 = block_tensor(4, 0, name, BlockEncoding::Q8_0, [1, 64]);
    rank1.rank = 1;
    rank1.dims = [64, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(
        w.add_block_tensor(rank1, q8_0_stream(1, 64, 0), &fixture_proof(0)),
        Err(TcfError::InvalidQuantShape { tensor_id: 4 })
    );
}

/// Section 9: the writer computes all three derived record digests, and
/// the reader recomputes all three at `open` — a file that reaches this
/// assertion has already had them verified.
#[test]
fn the_three_record_digests_are_computed_and_verified() {
    let mut w = populated();
    w.add_relation(relation()).expect("adds");
    let bytes = w.finish().expect("writes");
    let file = TcfFile::open(&bytes).expect("reader verifies every digest");

    for m in file.modules() {
        assert_ne!(m.policy_digest, [0u8; 16]);
    }
    for c in file.contracts() {
        assert_ne!(c.contract_digest, [0u8; 16]);
    }
    for r in file.relations() {
        assert_ne!(r.relation_digest, [0u8; 16]);
    }

    // Each digest is the one its own encoded record determines.
    let module = file.modules().first().expect("module 0");
    let mut image = [0u8; MODULE_RECORD_BYTES];
    module.encode(&mut image).expect("encodes");
    image[64..80].fill(0);
    let name = file.string(module.name).expect("utf8");
    assert_eq!(
        policy_digest(&image, name.as_bytes()).expect("policy digest"),
        crate::tcf::digest::Digest128::from_bytes(module.policy_digest)
    );

    // Two modules differing only in their names have different policy
    // digests: Section 7 concatenates the name.
    let attn = file.modules().get(1).expect("module 1");
    assert_ne!(module.policy_digest, attn.policy_digest);
}

/// Each derived digest is writer-owned, so a caller-supplied non-zero
/// value is rejected naming the exact field. Section 9.
#[test]
fn a_caller_set_record_digest_is_rejected_by_name() {
    let mut w = TcfWriter::new();
    let name = w.intern("w").expect("interns");
    let mut m = module(0, name);
    m.policy_digest = [1u8; 16];
    w.add_module(m).expect("adds");
    assert_eq!(
        w.finish(),
        Err(TcfError::NonzeroReserved {
            field: "ModuleRecord.policy_digest"
        })
    );

    let mut w = TcfWriter::new();
    let mut c = contract(1);
    c.contract_digest = [1u8; 16];
    w.add_contract(c).expect("adds");
    assert_eq!(
        w.finish(),
        Err(TcfError::NonzeroReserved {
            field: "ContractRecord.contract_digest"
        })
    );

    let mut w = TcfWriter::new();
    let mut r = relation();
    r.relation_digest = [1u8; 16];
    w.add_relation(r).expect("adds");
    assert_eq!(
        w.finish(),
        Err(TcfError::NonzeroReserved {
            field: "RelationRecord.relation_digest"
        })
    );
}

/// Section 8.0.1: a raw tensor's `logical_payload_bytes` is
/// `product(dims) * width`. The writer computes it, so a byte vector of
/// any other length is rejected rather than stored.
#[test]
fn a_raw_payload_of_the_wrong_length_is_rejected() {
    for supplied in [15usize, 17] {
        let mut w = TcfWriter::new();
        let name = w.intern("bias").expect("interns");
        w.add_module(module(0, name)).expect("adds");
        let mut raw = tensor(9, 0, name, Encoding::Raw(RawEncoding::F16));
        raw.role = Role::Bias;
        raw.rank = 1;
        raw.dims = [8, 0, 0, 0, 0, 0, 0, 0];
        raw.execution_role = ExecutionRole::Elementwise;
        w.add_raw_tensor(raw, vec![0u8; supplied]).expect("adds");
        assert_eq!(
            w.finish(),
            Err(TcfError::InvalidQuantShape { tensor_id: 9 }),
            "{supplied} bytes for 8 F16 elements"
        );
    }
}

/// The width comes from the encoding, never from the byte count: the
/// same 8 elements are 16 bytes in F16 and 32 in F32.
#[test]
fn a_raw_payload_length_follows_the_encoding_width() {
    for (encoding, width) in [
        (RawEncoding::U8, 1u64),
        (RawEncoding::Bf16, 2),
        (RawEncoding::F32, 4),
    ] {
        let mut w = TcfWriter::new();
        let name = w.intern("bias").expect("interns");
        w.add_module(module(0, name)).expect("adds");
        // Section 3: every tensor names a contract the file carries.
        w.add_contract(contract(1)).expect("adds");
        let mut raw = tensor(9, 0, name, Encoding::Raw(encoding));
        raw.role = Role::Bias;
        raw.rank = 1;
        raw.dims = [8, 0, 0, 0, 0, 0, 0, 0];
        raw.execution_role = ExecutionRole::Elementwise;
        let len = usize::try_from(8 * width).expect("fits");
        w.add_raw_tensor(raw, vec![7u8; len]).expect("adds");
        let bytes = w.finish().expect("writes");
        let file = TcfFile::open(&bytes).expect("reader accepts");
        let t = file.tensors().first().expect("tensor 0");
        assert_eq!(t.logical_payload_bytes, 8 * width, "{encoding:?}");
        file.verify_tensor(t).expect("verifies");
    }
}

#[test]
fn multi_block_tensors_verify() {
    let mut w = TcfWriter::new();
    let name = w.intern("wide").expect("interns");
    w.add_module(module(0, name)).expect("adds");
    // Section 3: every tensor names a contract the file carries.
    w.add_contract(contract(1)).expect("adds");
    let record = block_tensor(0, 0, name, BlockEncoding::Q8_0, [4, 128]);
    let stream = q8_0_stream(4, 128, 2);
    w.add_block_tensor(record, stream.clone(), &q8_0_proof(&stream, &[4, 128], 0))
        .expect("adds");
    let bytes = w.finish().expect("writes");

    let file = TcfFile::open(&bytes).expect("reader accepts");
    let t = file.tensors().first().expect("tensor 0");
    assert_eq!(t.logical_payload_bytes, 16 * 34);
    assert_eq!(t.physical_span_bytes, 576);
    file.verify_tensor_with(t, Some(&Q8_0Decoder { bias: 0.0 }))
        .expect("verifies");
}

/// A writer covering both payload shapes the sink path has to get
/// right: several one-row tensors, a multi-row tensor, and a raw
/// tensor whose 16 bytes are padded to a 64-byte span (Section 14.4).
fn streaming_fixture() -> TcfWriter {
    streaming_fixture_with(true)
}

/// The same eight tensors, with payloads or as records only.
fn streaming_fixture_with(payloads: bool) -> TcfWriter {
    let mut w = populated_with(payloads);
    let name = w.intern("wide").expect("interns");
    let record = block_tensor(6, 0, name, BlockEncoding::Q6K, [4, 512]);
    if payloads {
        w.add_block_tensor(
            record,
            block_bytes_for(BlockEncoding::Q6K, [4, 512], 6),
            &fixture_proof(6),
        )
        .expect("adds");
    } else {
        w.register_block_tensor(record).expect("registers");
    }
    // A Q8_0 tensor whose proof values a decoder can check.
    let name = w.intern("blocks").expect("interns");
    let record = block_tensor(7, 0, name, BlockEncoding::Q8_0, [2, 64]);
    if payloads {
        let (bytes, proof) = block_payload();
        w.add_block_tensor(record, bytes, &proof).expect("adds");
    } else {
        w.register_block_tensor(record).expect("registers");
    }
    w
}

/// The Q8_0 stream and its proof values, from the producer's decoder.
fn block_payload() -> (Vec<u8>, Vec<f32>) {
    let bytes = q8_0_stream(2, 64, 7);
    let proof = q8_0_proof(&bytes, &[2, 64], 7);
    (bytes, proof)
}

/// The payload `streaming_fixture_with(false)` expects at each tensor
/// index, built on demand so only one exists at a time.
fn streaming_payload(index: usize) -> Result<Payload, TcfError> {
    match index {
        0..=4 => {
            let (block, dims) = EVERY_BLOCK[index];
            Ok(Payload::Block {
                bytes: block_bytes_for(block, dims, index as u32),
                proof: fixture_proof(index as u32)
                    .iter()
                    .map(|v| f32_to_bits(*v))
                    .collect(),
            })
        }
        5 => Ok(Payload::Raw((0u8..16).collect())),
        6 => Ok(Payload::Block {
            bytes: block_bytes_for(BlockEncoding::Q6K, [4, 512], 6),
            proof: fixture_proof(6).iter().map(|v| f32_to_bits(*v)).collect(),
        }),
        7 => {
            let (bytes, proof) = block_payload();
            Ok(Payload::Block {
                bytes,
                proof: proof.iter().map(|v| f32_to_bits(*v)).collect(),
            })
        }
        other => panic!("the fixture has eight tensors, asked for {other}"),
    }
}

#[test]
fn finish_into_a_cursor_matches_finish() {
    let expected = streaming_fixture().finish().expect("writes");
    let mut sink = std::io::Cursor::new(Vec::<u8>::new());
    streaming_fixture().finish_into(&mut sink).expect("streams");
    let streamed = sink.into_inner();

    assert_eq!(streamed.len(), expected.len());
    assert_eq!(streamed, expected);

    let file = TcfFile::open(&streamed).expect("reader accepts");
    assert_eq!(file.tensors().len(), 8);
    for t in file.tensors() {
        file.verify_tensor(t).expect("verifies");
    }
    let block = file.tensors().get(7).expect("block tensor");
    file.verify_tensor_with(block, Some(&Q8_0Decoder { bias: 0.0 }))
        .expect("block proof verifies");
    // The raw tensor is the one whose span exceeds its payload, so the
    // pad path ran.
    let raw = file.tensors().get(5).expect("raw tensor");
    assert_eq!(raw.logical_payload_bytes, 16);
    assert_eq!(raw.physical_span_bytes, 64);
}

#[test]
fn finish_into_a_file_matches_finish() {
    // A real file is the only sink where pass 2 seeks past the end and
    // leaves a hole for the final directory write to fill.
    let path = std::env::temp_dir().join("tcf_finish_into_a_file_matches_finish.tcf");
    let expected = streaming_fixture().finish().expect("writes");
    {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)
            .expect("creates");
        streaming_fixture().finish_into(&mut file).expect("streams");
    }
    let streamed = std::fs::read(&path).expect("reads back");
    std::fs::remove_file(&path).expect("removes");

    assert_eq!(streamed.len(), expected.len());
    assert_eq!(streamed, expected);
    let file = TcfFile::open(&streamed).expect("reader accepts");
    for t in file.tensors() {
        file.verify_tensor(t).expect("verifies");
    }
}

#[test]
fn an_empty_file_streams_identically() {
    let expected = TcfWriter::new().finish().expect("writes");
    let mut sink = std::io::Cursor::new(Vec::<u8>::new());
    TcfWriter::new().finish_into(&mut sink).expect("streams");
    let streamed = sink.into_inner();

    // No tensors, so `file_len == data_off`: the single directory write
    // is the whole file.
    assert_eq!(streamed, expected);
    let file = TcfFile::open(&streamed).expect("reader accepts");
    assert_eq!(
        u64::try_from(streamed.len()).unwrap(),
        file.header().file_len
    );
    assert_eq!(file.header().file_len, file.header().data_off);
    assert!(file.tensors().is_empty());
}

/// A sink that accepts `budget` bytes in total and then fails every
/// write. Seeks always succeed, so the failure is a write failure.
struct FailingSink {
    budget: usize,
    pos: u64,
}

impl Write for FailingSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.len() > self.budget {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "sink budget exhausted",
            ));
        }
        self.budget -= buf.len();
        self.pos += u64::try_from(buf.len()).unwrap();
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Seek for FailingSink {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        if let SeekFrom::Start(off) = pos {
            self.pos = off;
        }
        Ok(self.pos)
    }
}

#[test]
fn a_failing_sink_surfaces_its_error_kind() {
    // Enough for the first payloads, not for all of them.
    let mut sink = FailingSink {
        budget: 100,
        pos: 0,
    };
    let err = streaming_fixture()
        .finish_into(&mut sink)
        .expect_err("the sink fails partway");
    // The budget is a size-dependent constant: if the fixture's first
    // payload ever grows past it, this would become a first-write
    // failure while still reporting the right kind. Assert the sink
    // took bytes before refusing, so the test keeps testing "partway".
    assert!(
        sink.budget < 100,
        "the sink refused its first write; this no longer tests a partial failure"
    );
    match err {
        TcfError::Io { kind, message } => {
            assert_eq!(kind, std::io::ErrorKind::BrokenPipe);
            assert!(message.contains("budget"));
        }
        other => panic!("expected TcfError::Io, got {other:?}"),
    }
}

// The tests below cover `TcfWriter::finish_streaming`, whose body lives
// in `crate::tcf::streaming`. They are here because they exercise the public
// writer API against the fixtures above, and because byte identity is a
// property of the writer as a whole, not of one module.

/// The property the streaming path exists to keep: for the same tensors
/// in the same order it writes the same bytes as `finish` and
/// `finish_into`, to the byte. Section 4.1 is what makes it possible —
/// the directory is computable from headers alone — and this is the
/// test that proves the code does it.
#[test]
fn finish_streaming_is_byte_identical() {
    let expected = streaming_fixture().finish().expect("writes");

    let mut buffered = std::io::Cursor::new(Vec::<u8>::new());
    streaming_fixture()
        .finish_into(&mut buffered)
        .expect("streams");
    assert_eq!(buffered.into_inner(), expected, "finish_into");

    let mut order = Vec::new();
    let mut sink = std::io::Cursor::new(Vec::<u8>::new());
    streaming_fixture_with(false)
        .finish_streaming(&mut sink, |index| {
            order.push(index);
            streaming_payload(index)
        })
        .expect("streams");
    let streamed = sink.into_inner();

    // One call per tensor, in directory order: tensor n's payload is
    // requested only after tensor n-1's has been written and dropped.
    assert_eq!(order, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(streamed.len(), expected.len());
    assert_eq!(streamed, expected);

    let file = TcfFile::open(&streamed).expect("reader accepts");
    assert_eq!(file.tensors().len(), 8);
    for t in file.tensors() {
        file.verify_tensor(t).expect("verifies");
    }
    let block = file.tensors().get(7).expect("block tensor");
    file.verify_tensor_with(block, Some(&Q8_0Decoder { bias: 0.0 }))
        .expect("block proof verifies");
}

#[test]
fn finish_streaming_into_a_file_is_byte_identical() {
    // A real file is the only sink where pass 2 seeks past the end and
    // leaves a hole for the final directory write to fill.
    let path = std::env::temp_dir().join("tcf_finish_streaming_into_a_file.tcf");
    let expected = streaming_fixture().finish().expect("writes");
    {
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&path)
            .expect("creates");
        streaming_fixture_with(false)
            .finish_streaming(&mut file, streaming_payload)
            .expect("streams");
    }
    let streamed = std::fs::read(&path).expect("reads back");
    std::fs::remove_file(&path).expect("removes");

    assert_eq!(streamed, expected);
    let file = TcfFile::open(&streamed).expect("reader accepts");
    for t in file.tensors() {
        file.verify_tensor(t).expect("verifies");
    }
}

#[test]
fn an_empty_writer_streams_identically() {
    let expected = TcfWriter::new().finish().expect("writes");
    let mut sink = std::io::Cursor::new(Vec::<u8>::new());
    TcfWriter::new()
        .finish_streaming(&mut sink, |index| panic!("no tensors, asked for {index}"))
        .expect("streams");
    assert_eq!(sink.into_inner(), expected);
}

/// The one-row Q8_0 payload the streaming tests hand over.
fn one_row_payload() -> Payload {
    Payload::Block {
        bytes: q8_0_stream(1, 64, 0),
        proof: fixture_proof(0).iter().map(|v| f32_to_bits(*v)).collect(),
    }
}

/// Two Q8_0 tensors. `register_second` decides whether the second one
/// carries its payload or waits for the callback.
fn two_tensor_writer(register_second: bool) -> TcfWriter {
    let mut w = TcfWriter::new();
    let name = w.intern("w").expect("interns");
    w.add_module(module(0, name)).expect("adds");
    let first = block_tensor(0, 0, name, BlockEncoding::Q8_0, [1, 64]);
    w.add_block_tensor(first, q8_0_stream(1, 64, 0), &fixture_proof(0))
        .expect("adds");
    let second = block_tensor(1, 0, name, BlockEncoding::Q8_0, [1, 64]);
    if register_second {
        w.register_block_tensor(second).expect("registers");
    } else {
        w.add_block_tensor(second, q8_0_stream(1, 64, 0), &fixture_proof(0))
            .expect("adds");
    }
    w
}

/// A payload already stored is used as it stands, and the callback is
/// asked only for the tensor that has none.
#[test]
fn a_stored_payload_is_never_requested_from_the_callback() {
    let expected = two_tensor_writer(false).finish().expect("writes");

    let mut asked = Vec::new();
    let mut sink = std::io::Cursor::new(Vec::<u8>::new());
    two_tensor_writer(true)
        .finish_streaming(&mut sink, |index| {
            asked.push(index);
            Ok(one_row_payload())
        })
        .expect("streams");

    assert_eq!(asked, vec![1]);
    assert_eq!(sink.into_inner(), expected);
}

/// A tensor registered without a payload has no bytes to write, so the
/// buffered paths reject it by index instead of leaving a hole.
#[test]
fn finish_rejects_a_tensor_registered_without_a_payload() {
    let err = populated_with(false).finish().expect_err("no payloads");
    match err {
        TcfError::PayloadMismatch {
            tensor_index,
            expected,
            supplied,
        } => {
            assert_eq!(tensor_index, 0);
            assert_eq!(expected, "a payload");
            assert!(supplied.contains("finish_streaming"));
        }
        other => panic!("expected TcfError::PayloadMismatch, got {other:?}"),
    }
}

/// One registered Q8_0 tensor: two blocks, 68 payload bytes.
fn one_registered_block() -> TcfWriter {
    let mut w = TcfWriter::new();
    let name = w.intern("w").expect("interns");
    w.add_module(module(0, name)).expect("adds");
    let record = block_tensor(0, 0, name, BlockEncoding::Q8_0, [1, 64]);
    w.register_block_tensor(record).expect("registers");
    w
}

/// One registered raw tensor: 8 F16 elements, 16 payload bytes.
fn one_registered_raw() -> TcfWriter {
    let mut w = TcfWriter::new();
    let name = w.intern("bias").expect("interns");
    w.add_module(module(0, name)).expect("adds");
    let mut record = tensor(0, 0, name, Encoding::Raw(RawEncoding::F16));
    record.role = Role::Bias;
    record.rank = 1;
    record.dims = [8, 0, 0, 0, 0, 0, 0, 0];
    record.execution_role = ExecutionRole::Elementwise;
    w.register_raw_tensor(record).expect("registers");
    w
}

/// A produced payload that disagrees with its record is an error naming
/// the tensor index and both sides, never a truncated payload or a
/// corrupt file.
#[test]
fn a_produced_payload_that_disagrees_with_its_record_is_rejected() {
    /// A writer, the payload its callback returns, and the two halves
    /// of the message the writer must report.
    type Case = (
        fn() -> TcfWriter,
        fn() -> Payload,
        &'static str,
        &'static str,
    );

    let cases: [Case; 4] = [
        (
            one_registered_block,
            || Payload::Raw(vec![0u8; 68]),
            "a block stream plus proof values for a block encoding",
            "verbatim bytes for a raw encoding",
        ),
        (
            one_registered_raw,
            one_row_payload,
            "verbatim bytes for a raw encoding",
            "a block stream plus proof values for a block encoding",
        ),
        (
            one_registered_block,
            || Payload::Block {
                bytes: q8_0_stream(2, 64, 0),
                proof: fixture_proof(0).iter().map(|v| f32_to_bits(*v)).collect(),
            },
            "68 payload bytes",
            "136 payload bytes",
        ),
        (
            one_registered_raw,
            || Payload::Raw(vec![0u8; 15]),
            "16 payload bytes",
            "15 payload bytes",
        ),
    ];

    for (writer, payload, expected, supplied) in cases {
        let mut sink = std::io::Cursor::new(Vec::<u8>::new());
        let err = writer()
            .finish_streaming(&mut sink, |_| Ok(payload()))
            .expect_err("the payload disagrees");
        assert_eq!(
            err,
            TcfError::PayloadMismatch {
                tensor_index: 0,
                expected: expected.to_owned(),
                supplied: supplied.to_owned(),
            }
        );
        // The message carries both sides, so a producer sees them
        // without matching on the variant.
        let message = err.to_string();
        assert!(message.contains(expected), "{message}");
        assert!(message.contains(supplied), "{message}");
    }
}

/// The callback's own error is the caller's: a producer's read or
/// quantize failure reaches it unchanged.
#[test]
fn a_produce_error_propagates_unchanged() {
    let mut sink = std::io::Cursor::new(Vec::<u8>::new());
    let err = one_registered_block()
        .finish_streaming(&mut sink, |index| {
            Err(TcfError::Io {
                kind: std::io::ErrorKind::NotFound,
                message: format!("tensor {index} vanished"),
            })
        })
        .expect_err("the producer fails");
    assert_eq!(
        err,
        TcfError::Io {
            kind: std::io::ErrorKind::NotFound,
            message: "tensor 0 vanished".to_owned()
        }
    );
}

/// Registration validates the record on the same terms as the
/// payload-carrying call: Section 8 rank, Section 8.0.1 encoding kind,
/// and every writer-owned field.
#[test]
fn registration_rejects_what_adding_rejects() {
    let mut w = TcfWriter::new();
    let name = w.intern("w").expect("interns");

    let raw = tensor(0, 0, name, Encoding::Raw(RawEncoding::F16));
    assert_eq!(
        w.register_block_tensor(raw),
        Err(TcfError::UnsupportedEncoding { raw: 0x0002 })
    );

    let quantized = block_tensor(0, 0, name, BlockEncoding::Q8_0, [1, 64]);
    assert_eq!(
        w.register_raw_tensor(quantized),
        Err(TcfError::UnsupportedEncoding { raw: 0x0208 })
    );

    let mut rank1 = block_tensor(4, 0, name, BlockEncoding::Q8_0, [1, 64]);
    rank1.rank = 1;
    rank1.dims = [64, 0, 0, 0, 0, 0, 0, 0];
    assert_eq!(
        w.register_block_tensor(rank1),
        Err(TcfError::InvalidQuantShape { tensor_id: 4 })
    );

    let mut owned = block_tensor(0, 0, name, BlockEncoding::Q8_0, [1, 64]);
    owned.data_offset = 4096;
    assert_eq!(
        w.register_block_tensor(owned),
        Err(TcfError::NonzeroReserved {
            field: "TensorRecord.data_offset"
        })
    );
}
