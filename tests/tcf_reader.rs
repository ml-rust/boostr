//! Reader tests: fixture files built with the writer and mutated byte by
//! byte, so every Section 17 code is reached by its own trigger.

mod common;

use boostr::tcf::consts::{
    CONTRACT_RECORD_BYTES, HEADER_BYTES, MAGIC, MAJOR, MODULE_RECORD_BYTES, PROOF_COUNT,
    RELATION_RECORD_BYTES, ROOT_PARENT_ID, SCHEMA_ID, TENSOR_RECORD_BYTES, UNUSED_INPUT_ID,
};
use boostr::tcf::digest::{
    contract_digest, hash_128, payload_digest, policy_digest, relation_digest,
};
use boostr::tcf::encoding::block::BlockEncoding;
use boostr::tcf::encoding::registry::Encoding;
use boostr::tcf::enums::RelationType;
use boostr::tcf::enums::{
    DotAccumulator, ExecutionRole, FallbackReason, InputRepresentation, LayoutId, MathMode,
    ModuleRole, OutputDtype, ProofFormat, QuantAxis, ResidencyClass, Role, RoundingMode,
    ScaleComputeDtype, StateDtype,
};
use boostr::tcf::flags::{ContractFlags, PolicyFlags, RelationFlags, StateFlags};
use boostr::tcf::flags::{HeaderFlags, RequiredFeatures, TensorFlags};
use boostr::tcf::proof::PROOF_BYTES;
use boostr::tcf::record::Record;
use boostr::tcf::{
    ContractRecord, Header, ModuleRecord, RelationRecord, StringRef, TcfError, TcfFile,
    TensorRecord, f32_to_bits,
};
use common::{Q8_0Decoder, q8_0_proof, q8_0_stream};

const MODULE_OFF: u64 = 192;
const TENSOR_OFF: u64 = 320;
const CONTRACT_OFF: u64 = 576;
const NAMES: &[u8] = b"block.0w";
const DIMS: [u64; 2] = [2, 128];
/// The contract `good_file`'s tensor is produced under. Section 3 makes
/// every tensor name one, so a file's tensor points at the first
/// contract the builder is given.
const CONTRACT_ID: u32 = 1;
const ENCODING: Encoding = Encoding::Block(BlockEncoding::Q8_0);
/// `[2, 128]` in Q8_0: eight 34-byte blocks.
const LOGICAL_LEN: usize = 8 * 34;
/// `LOGICAL_LEN` rounded up to the 64-byte span.
const PHYSICAL_LEN: usize = 320;

fn align64(v: u64) -> u64 {
    v.div_ceil(64) * 64
}

fn payload() -> Vec<u8> {
    q8_0_stream(2, 128, 3)
}

/// A minimal file: one module, one tensor naming the first of
/// `contracts`, those contract records, and an eight-block Q8_0
/// payload. Valid whenever `contracts` is non-empty — Section 3 makes a
/// tensor's contract mandatory.
fn build_file(contracts: &[ContractRecord]) -> Vec<u8> {
    build_file_with(contracts, &[])
}

/// The same file plus `relations`, which move every later section along
/// by the relation array's length.
fn build_file_with(contracts: &[ContractRecord], relations: &[RelationRecord]) -> Vec<u8> {
    let packed = payload();
    assert_eq!(packed.len(), LOGICAL_LEN);
    let logical_len = packed.len() as u64;
    let physical = align64(logical_len);
    assert_eq!(physical as usize, PHYSICAL_LEN);

    let contract_len = (contracts.len() * CONTRACT_RECORD_BYTES) as u64;
    let relation_len = (relations.len() * RELATION_RECORD_BYTES) as u64;
    let relation_off = if relations.is_empty() {
        0
    } else {
        align64(CONTRACT_OFF + contract_len)
    };
    let string_off = align64(CONTRACT_OFF + contract_len + relation_len);
    let string_len = NAMES.len() as u64;
    let proof_off = align64(string_off + string_len);
    let proof_len = PROOF_BYTES as u64;
    let data_off = align64(proof_off + proof_len);
    let file_len = data_off + physical;

    let mut features = RequiredFeatures::ACTIVATION_CONTRACTS
        .union(RequiredFeatures::PLACEMENT_METADATA)
        .union(RequiredFeatures::SEMANTIC_DIGESTS)
        .union(RequiredFeatures::SOURCE_PROOFS);
    if !relations.is_empty() {
        features = features.union(RequiredFeatures::RELATIONS);
    }

    let header = Header {
        magic: MAGIC,
        major: MAJOR,
        minor: 0,
        header_bytes: HEADER_BYTES,
        schema_id: SCHEMA_ID,
        flags: HeaderFlags::LITTLE_ENDIAN,
        tensor_count: 1,
        module_count: 1,
        contract_count: contracts.len() as u32,
        calibration_count: 0,
        relation_count: relations.len() as u32,
        workload_count: 0,
        required_features: features,
        module_off: MODULE_OFF,
        tensor_off: TENSOR_OFF,
        contract_off: CONTRACT_OFF,
        calibration_off: 0,
        relation_off,
        string_off,
        string_len,
        proof_off,
        proof_len,
        data_off,
        file_len,
        header_digest: [0u8; 16],
        directory_digest: [0u8; 16],
        workload_off: 0,
    };

    let module = ModuleRecord {
        module_id: 1,
        parent_id: ROOT_PARENT_ID,
        name: StringRef::new(0, 7),
        module_role: ModuleRole::Ffn,
        fallback_encoding: None,
        preferred_encoding: [Some(ENCODING), None, None, None],
        activation_contract_id: 0,
        policy_flags: PolicyFlags::NONE,
        min_quant_k: 64,
        default_residency: ResidencyClass::Hot,
        state_dtype: StateDtype::None,
        state_flags: StateFlags::NONE,
        policy_digest: [0u8; 16],
    };

    let tensor = TensorRecord {
        tensor_id: 42,
        module_id: 1,
        name: StringRef::new(7, 1),
        role: Role::LinearWeight,
        encoding: ENCODING,
        fallback_reason: FallbackReason::None,
        residency_class: ResidencyClass::Hot,
        flags: TensorFlags::NONE,
        rank: 2,
        calibration_id: 0,
        dims: [DIMS[0], DIMS[1], 0, 0, 0, 0, 0, 0],
        // Section 3: the tensor names the first contract the caller
        // supplied. A builder given none produces a file whose tensor
        // resolves to nothing, which is what the dangling-reference
        // tests are built from.
        activation_contract_id: contracts.first().map_or(0, |c| c.contract_id),
        layout_id: LayoutId::RowMajorDense,
        data_offset: data_off,
        logical_payload_bytes: logical_len,
        physical_span_bytes: physical,
        resident_bytes: physical,
        transfer_bytes: physical,
        sensitivity_delta: 0.0,
        sensitivity_ci95: 0.0,
        accesses_per_generation: 0.0,
        bytes_read_per_generation: 0.0,
        sensitivity_samples: 0,
        sensitivity_seed_count: 0,
        access_profile_samples: 0,
        execution_role: ExecutionRole::Matmul,
        workload_profile_id: 0,
        // A block stream is its own logical form: the two digests agree.
        semantic_digest: *payload_digest(&packed).as_bytes(),
        payload_digest: *payload_digest(&packed).as_bytes(),
        proof_rel_off: 0,
        proof_count: PROOF_COUNT,
        proof_format: ProofFormat::DequantF16,
    };

    let mut file = vec![0u8; file_len as usize];
    header.encode(&mut file).expect("header encodes");
    module
        .encode(&mut file[MODULE_OFF as usize..])
        .expect("module encodes");
    tensor
        .encode(&mut file[TENSOR_OFF as usize..])
        .expect("tensor encodes");
    for (i, contract) in contracts.iter().enumerate() {
        let at = CONTRACT_OFF as usize + i * CONTRACT_RECORD_BYTES;
        contract.encode(&mut file[at..]).expect("contract encodes");
    }
    for (i, relation) in relations.iter().enumerate() {
        let at = relation_off as usize + i * RELATION_RECORD_BYTES;
        relation.encode(&mut file[at..]).expect("relation encodes");
    }
    let names_at = string_off as usize;
    file[names_at..names_at + NAMES.len()].copy_from_slice(NAMES);

    let proofs: Vec<u8> = q8_0_proof(&packed, &DIMS, 42)
        .iter()
        .flat_map(|v| f32_to_bits(*v).to_le_bytes())
        .collect();
    let proofs_at = proof_off as usize;
    file[proofs_at..proofs_at + proofs.len()].copy_from_slice(&proofs);

    let data_at = data_off as usize;
    file[data_at..data_at + packed.len()].copy_from_slice(&packed);

    // Section 9: a producer MUST compute all three derived digests.
    // Sealing them here is what makes this a conforming file at all.
    seal_policy_digest(&mut file);
    for i in 0..contracts.len() {
        seal_contract_digest(&mut file, CONTRACT_OFF as usize + i * CONTRACT_RECORD_BYTES);
    }
    for i in 0..relations.len() {
        seal_relation_digest(&mut file, relation_off as usize + i * RELATION_RECORD_BYTES);
    }
    reseal(&mut file);
    file
}

/// Write the `policy_digest` the module record and its stored name
/// bytes determine. Section 7. Reads the name out of the file rather
/// than a constant, so a test that edits the name can reseal it.
fn seal_policy_digest(file: &mut [u8]) {
    let at = MODULE_OFF as usize;
    let mut off = [0u8; 8];
    off.copy_from_slice(&file[96..104]);
    let string_off = u64::from_le_bytes(off) as usize;
    let mut name_off = [0u8; 8];
    name_off.copy_from_slice(&file[at + 8..at + 16]);
    let mut name_len = [0u8; 4];
    name_len.copy_from_slice(&file[at + 16..at + 20]);
    let start = string_off + u64::from_le_bytes(name_off) as usize;
    let end = start + u32::from_le_bytes(name_len) as usize;
    let name = file[start..end].to_vec();
    let record = &file[at..at + MODULE_RECORD_BYTES];
    let digest = policy_digest(record, &name).expect("policy digest");
    file[at + 64..at + 80].copy_from_slice(digest.as_bytes());
}

/// Write the `contract_digest` the record at `at` determines. Section 9.
fn seal_contract_digest(file: &mut [u8], at: usize) {
    let record = &file[at..at + CONTRACT_RECORD_BYTES];
    let digest = contract_digest(record).expect("contract digest");
    file[at + 40..at + 56].copy_from_slice(digest.as_bytes());
}

/// Write the `relation_digest` the record at `at` determines. Section 11.
fn seal_relation_digest(file: &mut [u8], at: usize) {
    let record = &file[at..at + RELATION_RECORD_BYTES];
    let digest = relation_digest(record).expect("relation digest");
    file[at + 32..at + 48].copy_from_slice(digest.as_bytes());
}

fn good_file() -> Vec<u8> {
    build_file(&[sample_contract(CONTRACT_ID, MathMode::ReassociationAllowed)])
}

/// `math_mode` is what distinguishes two contracts here: the digest is
/// derived from the record, so two contracts differ in their digests
/// exactly when they differ in a covered field.
fn sample_contract(id: u32, math_mode: MathMode) -> ContractRecord {
    ContractRecord {
        contract_id: id,
        input_representation: InputRepresentation::F16,
        quant_group: 32,
        quant_axis: QuantAxis::Last,
        rounding_mode: RoundingMode::RnEven,
        qmin: -127,
        qmax: 127,
        scale_compute_dtype: ScaleComputeDtype::F32,
        dot_accumulator: DotAccumulator::F32,
        output_dtype: OutputDtype::F16,
        math_mode,
        calibration_id: 0,
        flags: ContractFlags::NONE,
        // Sealed by `build_file_with`, which owns the computation.
        contract_digest: [0u8; 16],
    }
}

fn sample_relation(output_tensor_id: u32) -> RelationRecord {
    RelationRecord {
        relation_type: RelationType::LowRankResidual,
        flags: RelationFlags::NONE,
        output_tensor_id,
        input_tensor_id: [42, 1, 2, UNUSED_INPUT_ID],
        rank_or_parameter: 16,
        activation_contract_id: 0,
        relation_digest: [0u8; 16],
    }
}

/// Recompute `directory_digest` then `header_digest`, in that order:
/// the header digest covers the directory digest field.
fn reseal(file: &mut [u8]) {
    let mut off = [0u8; 8];
    off.copy_from_slice(&file[128..136]);
    let data_off = u64::from_le_bytes(off) as usize;
    let directory = hash_128(&file[192..data_off]);
    file[160..176].copy_from_slice(directory.as_bytes());

    let mut image = [0u8; 192];
    image.copy_from_slice(&file[..192]);
    image[144..160].fill(0);
    let header = hash_128(&image);
    file[144..160].copy_from_slice(header.as_bytes());
}

fn data_off_of(file: &[u8]) -> usize {
    let mut off = [0u8; 8];
    off.copy_from_slice(&file[128..136]);
    u64::from_le_bytes(off) as usize
}

fn put_u64(file: &mut [u8], at: usize, value: u64) {
    file[at..at + 8].copy_from_slice(&value.to_le_bytes());
}

fn patch_tensor(file: &mut [u8], edit: impl FnOnce(&mut TensorRecord)) {
    let at = TENSOR_OFF as usize;
    let mut tensor = TensorRecord::decode(&file[at..at + TENSOR_RECORD_BYTES]).expect("decodes");
    edit(&mut tensor);
    tensor.encode(&mut file[at..]).expect("encodes");
    reseal(file);
}

fn patch_module(file: &mut [u8], edit: impl FnOnce(&mut ModuleRecord)) {
    let at = MODULE_OFF as usize;
    let mut module = ModuleRecord::decode(&file[at..at + MODULE_RECORD_BYTES]).expect("decodes");
    edit(&mut module);
    module.encode(&mut file[at..]).expect("encodes");
    // The edited bytes determine a new `policy_digest`; a producer
    // recomputes it, so a test that edits a module does too.
    seal_policy_digest(file);
    reseal(file);
}

fn open_err(file: &[u8]) -> TcfError {
    match TcfFile::open(file) {
        Ok(_) => panic!("expected the file to be rejected"),
        Err(e) => e,
    }
}

#[test]
fn opens_and_exposes_every_record() {
    let bytes = good_file();
    let file = TcfFile::open(&bytes).expect("valid file");

    assert_eq!(file.header().tensor_count, 1);
    assert_eq!(file.modules().len(), 1);
    assert_eq!(file.tensors().len(), 1);
    assert_eq!(file.contracts().len(), 1);
    assert!(file.calibrations().is_empty());
    assert!(file.relations().is_empty());
    assert!(file.workload_profiles().is_empty());

    let module = file.modules().first().expect("one module");
    assert_eq!(module.module_id, 1);
    assert_eq!(file.string(module.name), Ok("block.0"));

    let tensor = file.tensors().first().expect("one tensor");
    assert_eq!(tensor.tensor_id, 42);
    assert_eq!(tensor.encoding, ENCODING);
    assert_eq!(tensor.shape(), &DIMS);
    assert_eq!(file.string(tensor.name), Ok("w"));

    let payload = file.payload(tensor).expect("payload");
    assert_eq!(payload.len() as u64, tensor.logical_payload_bytes);
    assert_eq!(payload.len(), LOGICAL_LEN);
}

#[test]
fn open_never_reads_a_payload_page() {
    // Section 16: every byte at or past `data_off` is replaced. A reader
    // that touched one could not still accept the file.
    let mut bytes = good_file();
    let data_off = data_off_of(&bytes);
    bytes[data_off..].fill(0xaa);

    let file = TcfFile::open(&bytes).expect("directory alone is enough");
    let tensor = file.tensors().first().expect("one tensor");
    assert_eq!(tensor.physical_span_bytes, PHYSICAL_LEN as u64);

    // The payload really was replaced: `open` did not read it, and the
    // first read of it now sees the poison bytes.
    let payload = file.payload(tensor).expect("payload");
    assert!(payload.iter().all(|b| *b == 0xaa));
    assert_eq!(
        file.verify_tensor(tensor),
        Err(TcfError::PayloadDigestMismatch { tensor_id: 42 })
    );
}

#[test]
fn short_file_is_rejected() {
    let bytes = good_file();
    assert_eq!(
        open_err(&bytes[..100]),
        TcfError::SectionBounds { section: "Header" }
    );
    // Long enough for a header, shorter than `file_len`.
    assert_eq!(
        open_err(&bytes[..512]),
        TcfError::SectionBounds {
            section: "file_len"
        }
    );
}

#[test]
fn bad_magic_is_rejected_first() {
    let mut bytes = good_file();
    bytes[2] = b'X';
    assert_eq!(open_err(&bytes), TcfError::BadMagic);
}

#[test]
fn unsupported_major_is_rejected() {
    let mut bytes = good_file();
    bytes[8..10].copy_from_slice(&2u16.to_le_bytes());
    assert_eq!(open_err(&bytes), TcfError::UnsupportedMajor { major: 2 });
}

#[test]
fn unknown_required_feature_bit_is_rejected() {
    let mut bytes = good_file();
    put_u64(&mut bytes, 48, 0x0f | (1 << 9));
    assert_eq!(
        open_err(&bytes),
        TcfError::UnknownRequiredFeature { bit: 9 }
    );
}

#[test]
fn mandatory_required_feature_bit_must_be_set() {
    let mut bytes = good_file();
    put_u64(&mut bytes, 48, 0x0e);
    reseal(&mut bytes);
    assert_eq!(
        open_err(&bytes),
        TcfError::UnknownRequiredFeature { bit: 0 }
    );
}

#[test]
fn conditional_required_feature_bit_must_match_its_count() {
    // Bit 5 claims workload profiles while `workload_count` is 0.
    let mut bytes = good_file();
    put_u64(&mut bytes, 48, 0x2f);
    reseal(&mut bytes);
    assert_eq!(
        open_err(&bytes),
        TcfError::UnknownRequiredFeature { bit: 5 }
    );
}

#[test]
fn header_flags_must_declare_little_endian_and_nothing_else() {
    let mut bytes = good_file();
    bytes[20..24].copy_from_slice(&0u32.to_le_bytes());
    assert_eq!(
        open_err(&bytes),
        TcfError::NonzeroReserved {
            field: "Header.flags"
        }
    );

    let mut bytes = good_file();
    bytes[20..24].copy_from_slice(&0b11u32.to_le_bytes());
    assert_eq!(
        open_err(&bytes),
        TcfError::NonzeroReserved {
            field: "Header.flags"
        }
    );
}

#[test]
fn nonzero_reserved_header_byte_is_rejected() {
    let mut bytes = good_file();
    bytes[191] = 1;
    assert_eq!(
        open_err(&bytes),
        TcfError::NonzeroReserved {
            field: "Header.reserved@184"
        }
    );
}

#[test]
fn header_digest_mismatch_is_rejected() {
    let mut bytes = good_file();
    // `minor` is covered by `header_digest` and by nothing else.
    bytes[10..12].copy_from_slice(&7u16.to_le_bytes());
    assert_eq!(open_err(&bytes), TcfError::HeaderDigestMismatch);
}

#[test]
fn directory_digest_mismatch_is_rejected() {
    let mut bytes = good_file();
    let mut off = [0u8; 8];
    off.copy_from_slice(&bytes[96..104]);
    let string_off = u64::from_le_bytes(off) as usize;
    bytes[string_off] = b'B';
    assert_eq!(open_err(&bytes), TcfError::DirectoryDigestMismatch);
}

#[test]
fn misaligned_section_is_rejected() {
    let mut bytes = good_file();
    put_u64(&mut bytes, 96, 577);
    reseal(&mut bytes);
    assert_eq!(
        open_err(&bytes),
        TcfError::MisalignedSection {
            section: "String table"
        }
    );

    let mut bytes = good_file();
    let data_off = data_off_of(&bytes) as u64;
    put_u64(&mut bytes, 128, data_off + 1);
    reseal(&mut bytes);
    assert_eq!(
        open_err(&bytes),
        TcfError::MisalignedSection {
            section: "data_off"
        }
    );
}

#[test]
fn overlapping_sections_are_rejected() {
    // The string table is moved on top of the tensor array.
    let mut bytes = good_file();
    put_u64(&mut bytes, 96, TENSOR_OFF);
    reseal(&mut bytes);
    assert_eq!(
        open_err(&bytes),
        TcfError::SectionBounds {
            section: "String table"
        }
    );
}

#[test]
fn oversized_count_is_rejected() {
    let mut bytes = good_file();
    bytes[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
    reseal(&mut bytes);
    assert_eq!(
        open_err(&bytes),
        TcfError::SectionBounds {
            section: "TensorRecord[]"
        }
    );
}

#[test]
fn section_reaching_into_the_payload_is_rejected() {
    let mut bytes = good_file();
    let data_off = data_off_of(&bytes) as u64;
    put_u64(&mut bytes, 96, data_off);
    reseal(&mut bytes);
    assert_eq!(
        open_err(&bytes),
        TcfError::SectionBounds {
            section: "String table"
        }
    );
}

#[test]
fn inflated_resident_bytes_is_rejected() {
    let mut bytes = good_file();
    patch_tensor(&mut bytes, |t| t.resident_bytes = t.physical_span_bytes * 8);
    assert_eq!(
        open_err(&bytes),
        TcfError::ResidentBytesViolation { tensor_id: 42 }
    );

    let mut bytes = good_file();
    patch_tensor(&mut bytes, |t| t.transfer_bytes = 0);
    assert_eq!(
        open_err(&bytes),
        TcfError::ResidentBytesViolation { tensor_id: 42 }
    );
}

#[test]
fn missing_fallback_reason_is_rejected() {
    let mut bytes = good_file();
    patch_module(&mut bytes, |m| {
        m.preferred_encoding[0] = Some(Encoding::Block(BlockEncoding::Q6K));
    });
    assert_eq!(
        open_err(&bytes),
        TcfError::MissingFallbackReason { tensor_id: 42 }
    );
}

#[test]
fn stated_fallback_reason_is_accepted() {
    let mut bytes = good_file();
    patch_module(&mut bytes, |m| {
        m.preferred_encoding[0] = Some(Encoding::Block(BlockEncoding::Q6K));
    });
    patch_tensor(&mut bytes, |t| {
        t.fallback_reason = FallbackReason::TaskSensitivity;
    });
    assert!(TcfFile::open(&bytes).is_ok());
}

#[test]
fn access_profile_without_a_workload_profile_is_rejected() {
    let mut bytes = good_file();
    patch_tensor(&mut bytes, |t| {
        t.flags = TensorFlags::ACCESS_PROFILE_VALID;
        t.workload_profile_id = 3;
    });
    assert_eq!(
        open_err(&bytes),
        TcfError::MissingWorkloadProfile { tensor_id: 42 }
    );
}

#[test]
fn misaligned_tensor_data_offset_is_rejected() {
    let mut bytes = good_file();
    patch_tensor(&mut bytes, |t| t.data_offset += 8);
    assert_eq!(
        open_err(&bytes),
        TcfError::MisalignedSection {
            section: "TensorRecord.data_offset"
        }
    );
}

#[test]
fn payload_span_past_the_file_is_rejected() {
    let mut bytes = good_file();
    patch_tensor(&mut bytes, |t| {
        t.physical_span_bytes += 64;
        t.resident_bytes = t.physical_span_bytes;
        t.transfer_bytes = t.physical_span_bytes;
    });
    assert_eq!(
        open_err(&bytes),
        TcfError::SectionBounds {
            section: "tensor payload"
        }
    );
}

#[test]
fn duplicate_contract_id_with_a_different_digest_is_rejected() {
    let bytes = build_file(&[
        sample_contract(1, MathMode::ReassociationAllowed),
        sample_contract(1, MathMode::ReassociationForbidden),
    ]);
    assert_eq!(
        open_err(&bytes),
        TcfError::ContractDigestMismatch { contract_id: 1 }
    );
}

#[test]
fn duplicate_contract_id_with_the_same_digest_is_accepted() {
    let bytes = build_file(&[
        sample_contract(1, MathMode::ReassociationAllowed),
        sample_contract(1, MathMode::ReassociationAllowed),
    ]);
    let file = TcfFile::open(&bytes).expect("identical digests agree");
    assert_eq!(file.contracts().len(), 2);
}

/// Section 9: a reader MUST verify `policy_digest`. A corrupted stored
/// digest is `E_POLICY_DIGEST_MISMATCH`, naming the module.
#[test]
fn corrupted_policy_digest_is_rejected() {
    let mut bytes = good_file();
    let at = MODULE_OFF as usize + 64;
    bytes[at] ^= 0x01;
    reseal(&mut bytes);
    assert_eq!(
        open_err(&bytes),
        TcfError::PolicyDigestMismatch { module_id: 1 }
    );
}

/// The name is concatenated into `policy_digest`, so renaming a module
/// without recomputing the digest is caught. Section 7.
#[test]
fn policy_digest_covers_the_module_name() {
    let mut bytes = good_file();
    let mut off = [0u8; 8];
    off.copy_from_slice(&bytes[96..104]);
    let string_off = u64::from_le_bytes(off) as usize;
    bytes[string_off] = b'B';
    reseal(&mut bytes);
    assert_eq!(
        open_err(&bytes),
        TcfError::PolicyDigestMismatch { module_id: 1 }
    );
}

/// Section 9: a reader MUST verify `contract_digest` — it is the
/// integrity identity, so a wrong one hides a corrupted record.
#[test]
fn corrupted_contract_digest_is_rejected() {
    let mut bytes = build_file(&[sample_contract(1, MathMode::ReassociationAllowed)]);
    let at = CONTRACT_OFF as usize + 40;
    bytes[at] ^= 0x01;
    reseal(&mut bytes);
    assert_eq!(
        open_err(&bytes),
        TcfError::ContractDigestMismatch { contract_id: 1 }
    );
}

/// A zero digest is the value a producer that computed nothing leaves
/// behind, and is exactly what Section 9 closes.
#[test]
fn zero_contract_digest_is_rejected() {
    let mut bytes = build_file(&[sample_contract(1, MathMode::ReassociationAllowed)]);
    let at = CONTRACT_OFF as usize + 40;
    bytes[at..at + 16].fill(0);
    reseal(&mut bytes);
    assert_eq!(
        open_err(&bytes),
        TcfError::ContractDigestMismatch { contract_id: 1 }
    );
}

/// `contract_id` is excluded from the digest (Section 9), so renumbering
/// a contract leaves its digest valid.
#[test]
fn contract_digest_excludes_contract_id() {
    let mut bytes = build_file(&[sample_contract(1, MathMode::ReassociationAllowed)]);
    bytes[CONTRACT_OFF as usize..CONTRACT_OFF as usize + 4].copy_from_slice(&9u32.to_le_bytes());
    // The tensor follows the renumbering: Section 3 keeps its
    // `activation_contract_id` resolvable, and `patch_tensor` reseals.
    patch_tensor(&mut bytes, |t| t.activation_contract_id = 9);
    let file = TcfFile::open(&bytes).expect("the digest does not cover contract_id");
    assert_eq!(
        file.contracts().first().expect("one contract").contract_id,
        9
    );
}

/// Section 9, Section 11: a reader MUST verify `relation_digest`.
#[test]
fn a_relation_with_a_valid_digest_is_accepted() {
    let bytes = build_file_with(
        &[sample_contract(CONTRACT_ID, MathMode::ReassociationAllowed)],
        &[sample_relation(42)],
    );
    let file = TcfFile::open(&bytes).expect("valid file");
    assert_eq!(file.relations().len(), 1);
}

#[test]
fn corrupted_relation_digest_is_rejected() {
    let mut bytes = build_file_with(
        &[sample_contract(CONTRACT_ID, MathMode::ReassociationAllowed)],
        &[sample_relation(42)],
    );
    let mut off = [0u8; 8];
    off.copy_from_slice(&bytes[88..96]);
    let relation_off = u64::from_le_bytes(off) as usize;
    bytes[relation_off + 32] ^= 0x01;
    reseal(&mut bytes);
    assert_eq!(
        open_err(&bytes),
        TcfError::RelationDigestMismatch {
            output_tensor_id: 42
        }
    );
}

/// A relation rewired to different operands no longer matches its
/// digest, which is the whole reason Section 11 defines one.
#[test]
fn rewired_relation_operands_are_rejected() {
    let mut bytes = build_file_with(
        &[sample_contract(CONTRACT_ID, MathMode::ReassociationAllowed)],
        &[sample_relation(42)],
    );
    let mut off = [0u8; 8];
    off.copy_from_slice(&bytes[88..96]);
    let relation_off = u64::from_le_bytes(off) as usize;
    bytes[relation_off + 8..relation_off + 12].copy_from_slice(&99u32.to_le_bytes());
    reseal(&mut bytes);
    assert_eq!(
        open_err(&bytes),
        TcfError::RelationDigestMismatch {
            output_tensor_id: 42
        }
    );
}

#[test]
fn block_payload_size_must_match_the_shape() {
    let mut bytes = good_file();
    patch_tensor(&mut bytes, |t| t.logical_payload_bytes -= 4);
    assert_eq!(
        open_err(&bytes),
        TcfError::InvalidQuantShape { tensor_id: 42 }
    );
}

#[test]
fn string_ref_out_of_range_is_rejected() {
    let bytes = good_file();
    let file = TcfFile::open(&bytes).expect("valid file");
    assert_eq!(
        file.string(StringRef::new(4, 99)),
        Err(TcfError::SectionBounds {
            section: "string table"
        })
    );
}

#[test]
fn non_utf8_name_is_rejected() {
    let mut bytes = good_file();
    let mut off = [0u8; 8];
    off.copy_from_slice(&bytes[96..104]);
    let string_off = u64::from_le_bytes(off) as usize;
    bytes[string_off] = 0xff;
    // The name feeds `policy_digest` (Section 7), so editing it means
    // recomputing that digest — it is the UTF-8 check, not the digest,
    // this test is about.
    seal_policy_digest(&mut bytes);
    reseal(&mut bytes);

    let file = TcfFile::open(&bytes).expect("names are not validated by open");
    let module = file.modules().first().expect("one module");
    assert_eq!(
        file.string(module.name),
        Err(TcfError::InvalidUtf8Name { off: 0, len: 7 })
    );
}

#[test]
fn verify_tensor_accepts_the_good_file() {
    let bytes = good_file();
    let file = TcfFile::open(&bytes).expect("valid file");
    let tensor = file.tensors().first().expect("one tensor");
    assert_eq!(file.verify_tensor(tensor), Ok(()));
}

#[test]
fn altered_payload_byte_fails_the_payload_digest() {
    let mut bytes = good_file();
    let data_off = data_off_of(&bytes);
    bytes[data_off] ^= 0x11;
    // Digests over the directory are untouched, so `open` still passes.
    let file = TcfFile::open(&bytes).expect("directory is intact");
    let tensor = file.tensors().first().expect("one tensor");
    assert_eq!(
        file.verify_tensor(tensor),
        Err(TcfError::PayloadDigestMismatch { tensor_id: 42 })
    );
}

#[test]
fn altered_code_with_a_resealed_payload_digest_fails_the_semantic_digest() {
    let mut bytes = good_file();
    let data_off = data_off_of(&bytes);
    // Any byte value is a legal Q8_0 code; only the resulting digest
    // mismatch matters here.
    bytes[data_off + 2] ^= 0x11;

    let fresh = *payload_digest(&bytes[data_off..data_off + LOGICAL_LEN]).as_bytes();
    patch_tensor(&mut bytes, |t| t.payload_digest = fresh);

    let file = TcfFile::open(&bytes).expect("directory is intact");
    let tensor = file.tensors().first().expect("one tensor");
    assert_eq!(
        file.verify_tensor(tensor),
        Err(TcfError::SemanticDigestMismatch { tensor_id: 42 })
    );
}

#[test]
fn zero_padding_is_accepted() {
    // `good_file` already carries zero padding bytes (272 logical of a
    // 320-byte physical span): confirm they are exactly what
    // `verify_tensor` walks and accepts.
    let bytes = good_file();
    let data_off = data_off_of(&bytes);
    assert_eq!(
        &bytes[data_off + LOGICAL_LEN..data_off + PHYSICAL_LEN],
        &[0u8; PHYSICAL_LEN - LOGICAL_LEN][..]
    );
    let file = TcfFile::open(&bytes).expect("valid file");
    let tensor = file.tensors().first().expect("one tensor");
    assert_eq!(file.verify_tensor(tensor), Ok(()));
}

#[test]
fn nonzero_padding_byte_is_rejected() {
    // Section 14.4, Section 15.2: a non-zero trailing alignment byte is
    // caught by `verify_tensor`, never by `open` (Section 16).
    let mut bytes = good_file();
    let data_off = data_off_of(&bytes);
    bytes[data_off + PHYSICAL_LEN - 1] = 1;

    let file = TcfFile::open(&bytes).expect("directory is intact, padding is payload");
    let tensor = file.tensors().first().expect("one tensor");
    assert_eq!(
        file.verify_tensor(tensor),
        Err(TcfError::NonzeroReserved {
            field: "TensorRecord.padding"
        })
    );
}

#[test]
fn dangling_module_id_is_rejected() {
    // Section 7: every `TensorRecord.module_id` MUST resolve to a
    // `ModuleRecord` in this file.
    let mut bytes = good_file();
    patch_tensor(&mut bytes, |t| t.module_id = 99);
    assert_eq!(
        open_err(&bytes),
        TcfError::SectionBounds {
            section: "TensorRecord.module_id"
        }
    );
}

#[test]
fn a_tensor_resolves_its_activation_contract() {
    // Section 3: the record the tensor was produced under, whose
    // semantic fields a kernel dispatcher matches against.
    let bytes = good_file();
    let file = TcfFile::open(&bytes).expect("valid file");
    let tensor = file.tensors().first().expect("one tensor");

    let contract = file.contract(tensor).expect("the contract resolves");
    assert_eq!(contract.contract_id, CONTRACT_ID);
    assert_eq!(contract.math_mode, MathMode::ReassociationAllowed);
    assert_eq!(
        contract.contract_digest,
        file.contracts()
            .first()
            .expect("one contract")
            .contract_digest
    );
}

/// The accessor resolves against this file's records, so a record from
/// elsewhere reports the same unresolvable reference `open` would.
#[test]
fn resolving_a_foreign_tensor_record_is_rejected() {
    let bytes = good_file();
    let file = TcfFile::open(&bytes).expect("valid file");
    let mut foreign = *file.tensors().first().expect("one tensor");
    foreign.activation_contract_id = 99;

    assert_eq!(
        file.contract(&foreign),
        Err(TcfError::SectionBounds {
            section: "TensorRecord.activation_contract_id"
        })
    );
}

#[test]
fn dangling_activation_contract_id_is_rejected() {
    // Section 3: a tensor naming a contract the file does not carry
    // leaves its activation representation and accumulation unstated.
    let mut bytes = good_file();
    patch_tensor(&mut bytes, |t| t.activation_contract_id = 99);
    assert_eq!(
        open_err(&bytes),
        TcfError::SectionBounds {
            section: "TensorRecord.activation_contract_id"
        }
    );
}

/// Section 3 and Section 9 define no "no contract" value for
/// `activation_contract_id`, unlike the zero values Section 7 states for
/// `fallback_encoding` and `min_quant_k`. `0` is an ordinary id: it is
/// rejected when no record carries it, and resolves when one does.
#[test]
fn a_zero_activation_contract_id_resolves_or_is_rejected() {
    let unresolvable = build_file(&[]);
    assert_eq!(
        open_err(&unresolvable),
        TcfError::SectionBounds {
            section: "TensorRecord.activation_contract_id"
        }
    );

    let bytes = build_file(&[sample_contract(0, MathMode::ReassociationAllowed)]);
    let file = TcfFile::open(&bytes).expect("contract 0 is a contract like any other");
    let tensor = file.tensors().first().expect("one tensor");
    assert_eq!(tensor.activation_contract_id, 0);
    assert_eq!(file.contract(tensor).expect("resolves").contract_id, 0);
}

#[test]
fn altered_proof_value_fails_the_proof_check() {
    let mut bytes = good_file();
    let mut off = [0u8; 8];
    off.copy_from_slice(&bytes[112..120]);
    let proof_off = u64::from_le_bytes(off) as usize;
    // Proof entry 3 occupies bytes 6 and 7 of the vector.
    bytes[proof_off + 6] ^= 0x3c;
    reseal(&mut bytes);

    let file = TcfFile::open(&bytes).expect("directory digest is resealed");
    let tensor = file.tensors().first().expect("one tensor");
    // Without a decoder the proofs are not checked; with one they are.
    assert_eq!(file.verify_tensor(tensor), Ok(()));
    assert_eq!(
        file.verify_tensor_with(tensor, Some(&Q8_0Decoder { bias: 0.0 })),
        Err(TcfError::ProofMismatch {
            tensor_id: 42,
            proof_index: 3
        })
    );
}

#[test]
fn a_decoder_that_disagrees_with_the_producer_fails_the_proof_check() {
    let bytes = good_file();
    let file = TcfFile::open(&bytes).expect("valid file");
    let tensor = file.tensors().first().expect("one tensor");
    assert_eq!(
        file.verify_tensor_with(tensor, Some(&Q8_0Decoder { bias: 0.0 })),
        Ok(())
    );
    assert_eq!(
        file.verify_tensor_with(tensor, Some(&Q8_0Decoder { bias: 1.0 })),
        Err(TcfError::ProofMismatch {
            tensor_id: 42,
            proof_index: 0
        })
    );
}
